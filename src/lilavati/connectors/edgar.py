"""
SEC EDGAR XBRL Company Facts Connector.

Connects to the SEC EDGAR API (compliant with SEC Fair Access policies),
fetches standardized structured XBRL filings, and extracts canonical Three-Statement
models where every single line item is cryptographically grounded with a CitationRef.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from lilavati.connectors.sec_filing import (
    CANONICAL_ITEMS,
    DURATION_DAYS,
    FilingRef,
    IdentityCheck,
    Kind,
    check_identities,
    extract_filing,
    fact_hash,
    filing_for_fiscal_year,
)
from lilavati.core.provenance import CitationRef, SourcedValue
from lilavati.schemas.statements import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    ThreeStatementModel,
)

# Common institutional tickers to 10-digit CIK map
DEFAULT_TICKER_CIK_MAP: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "META": "0001326801",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "V": "0001403161",
}


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


class SourcedThreeStatementModel(BaseModel):
    """One 10-K mapped into the canonical three-statement model, with every input cited.

    Each statement field is reported (in ``sourced_facts``, with its XBRL tag in ``tags_used``),
    derived from reported items by a stated formula (``derived``), or, for the fields the schema
    defaults to zero, not reported by the filing (``not_reported``; 0.0 in ``statement_model``).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: str
    fiscal_year: int
    form: str
    filing: FilingRef
    statement_model: ThreeStatementModel
    sourced_facts: dict[str, SourcedValue] = Field(
        ..., description="Map of canonical metric name to its SourcedValue and CitationRef"
    )
    tags_used: dict[str, str] = Field(default_factory=dict, description="canonical item -> US-GAAP tag")
    derived: dict[str, str] = Field(
        default_factory=dict, description="statement field -> formula over reported items")
    not_reported: list[str] = Field(
        default_factory=list,
        description="statement fields the filing does not report; 0.0 in statement_model, not a reported zero")
    checks: list[IdentityCheck] = Field(default_factory=list, description="identities on the reported numbers")
    is_formally_verified: bool = Field(
        ..., description="True if every check in `checks` holds (Z3 for the balance sheet equation)"
    )


class EdgarClient:
    """SEC EDGAR client for structured XBRL disclosures."""

    BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    USER_AGENT_ENV = "LILAVATI_SEC_USER_AGENT"

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        SEC fair access requires a real contact in the User-Agent, e.g. ``"Your Name you@example.org"``.
        Pass it here or set ``LILAVATI_SEC_USER_AGENT``; it is checked when a request is made, so
        clients used only with pre-fetched facts need neither.
        """
        self._user_agent = user_agent
        self.timeout = timeout
        self._transport = transport
        self._ticker_map: dict[str, str] | None = None

    @property
    def user_agent(self) -> str:
        ua = self._user_agent or os.environ.get(self.USER_AGENT_ENV, "").strip()
        if not ua:
            raise ValueError(
                f"SEC EDGAR requires a contact in the User-Agent. Set {self.USER_AGENT_ENV} to "
                "something like 'Your Name you@example.org' (SEC fair-access policy)."
            )
        return ua

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

    def resolve_cik(self, ticker_or_cik: str, online: bool = False) -> str:
        """Resolves a ticker symbol or raw CIK into a 10-digit zero-padded CIK string.

        Known tickers come from a small built-in map. With ``online=True``, other tickers are looked
        up in SEC's company list (``company_tickers.json``, one request, cached on the client).
        """
        clean = ticker_or_cik.strip().upper()
        if clean in DEFAULT_TICKER_CIK_MAP:
            return DEFAULT_TICKER_CIK_MAP[clean]
        # If numeric string provided, pad with leading zeros
        if clean.isdigit():
            return clean.zfill(10)
        if online:
            if self._ticker_map is None:
                rows = self.get_json(self.TICKERS_URL)
                self._ticker_map = {str(r["ticker"]).upper(): str(r["cik_str"]).zfill(10) for r in rows.values()}
            if clean in self._ticker_map:
                return self._ticker_map[clean]
            raise ValueError(f"Ticker '{ticker_or_cik}' is not in SEC's company list; provide a CIK.")
        raise ValueError(
            f"Unknown or unmapped ticker symbol: '{ticker_or_cik}'. Provide a 10-digit CIK, register it in "
            "the map, or call resolve_cik(..., online=True) to use SEC's company list."
        )

    def get_json(self, url: str) -> Any:
        """GET a JSON document from sec.gov or data.sec.gov with the configured contact."""
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=self.timeout, transport=self._transport, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
        if resp.status_code == 404:
            raise FileNotFoundError("SEC EDGAR document not found (HTTP 404).")
        if resp.status_code != 200:
            raise RuntimeError(
                f"SEC EDGAR error: HTTP {resp.status_code}. Check that {self.USER_AGENT_ENV} holds a real contact."
            )
        return resp.json()

    def fetch_submissions(self, ticker_or_cik: str) -> dict[str, Any]:
        """The company's SEC submissions index (recent filings: accession, form, dates, documents)."""
        cik = self.resolve_cik(ticker_or_cik, online=True)
        return self.get_json(f"{self.SUBMISSIONS_URL}/CIK{cik}.json")  # type: ignore[no-any-return]

    def fetch_company_facts(self, ticker_or_cik: str) -> dict[str, Any]:
        """Fetches raw structured XBRL company facts JSON from SEC EDGAR."""
        cik = self.resolve_cik(ticker_or_cik, online=True)
        url = f"{self.BASE_URL}/CIK{cik}.json"

        headers = self.headers  # raises a clear error if no contact is configured
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 404:
                raise FileNotFoundError(f"SEC EDGAR facts not found for CIK {cik} (HTTP 404).")
            if resp.status_code != 200:
                raise RuntimeError(
                    f"SEC EDGAR API error: HTTP {resp.status_code} ({resp.text[:200]}). "
                    "Ensure User-Agent header complies with SEC requirements."
                )
            return resp.json()  # type: ignore[no-any-return]

    @staticmethod
    def extract_line_item(
        facts: dict[str, Any],
        candidate_tags: list[str],
        fiscal_year: int,
        period: str = "FY",
        form: str = "10-K",
    ) -> SourcedValue | None:
        """
        Extracts a specific financial line item from SEC company facts JSON, searching
        through candidate US-GAAP taxonomy tags in priority order.
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        for tag in candidate_tags:
            tag_data = us_gaap.get(tag)
            if not tag_data:
                continue

            units = tag_data.get("units", {})
            usd_entries = units.get("USD", [])
            if not usd_entries:
                continue

            # Filter for the matching fiscal year, period, and form
            matching = [
                e
                for e in usd_entries
                if e.get("fy") == fiscal_year
                and e.get("form") == form
                and (period is None or e.get("fp") == period)
            ]

            if not matching:
                continue

            # A 10-K tags every value it reports with its own fiscal year, including the prior-year
            # comparatives, so fy alone does not identify the period. The filing's own period is the
            # latest end date; for flow items in an annual filing, keep full-year durations only
            # (a 10-K can also report quarters). Ties: the latest filing (restatements).
            if period == "FY":
                annual = [e for e in matching if "start" not in e or 350 <= _days(e["start"], e["end"]) <= 380]
                matching = annual or matching
            latest_end = max(str(e.get("end", "")) for e in matching)
            matching = [e for e in matching if str(e.get("end", "")) == latest_end]
            matching.sort(key=lambda x: str(x.get("filed", "")), reverse=True)
            chosen = matching[0]

            val = float(chosen["val"])
            # SEC fact records always carry these; a record without them is not citable
            filing_date_str = str(chosen["filed"])
            end_date_str = str(chosen["end"])
            accn = str(chosen["accn"])

            f_date = date.fromisoformat(filing_date_str)
            p_end = date.fromisoformat(end_date_str)

            doc_hash = fact_hash(chosen)  # the fact record itself, so the citation can be checked

            citation = CitationRef(
                doc_hash=doc_hash,
                filing_type=form,
                filing_date=f_date,
                period_end=p_end,
                page=None,
                table_id="xbrl",
                xbrl_tag=f"us-gaap:{tag}",
                raw_snippet=f"{val:,.0f} USD (accn: {accn})",
            )

            return SourcedValue(
                value=val,
                unit="USD",
                scale=1,
                citation=citation,
            )

        return None

    def extract_three_statement_model(
        self,
        ticker_or_cik: str,
        fiscal_year: int,
        facts: dict[str, Any] | None = None,
    ) -> SourcedThreeStatementModel:
        """The fiscal year's 10-K mapped into the canonical three-statement model.

        The filing is identified first (``sec_filing.filing_for_fiscal_year``) and only its own XBRL
        facts are used, as in ``get_filing_figures``. See ``statement_model_from_filing`` for how each
        field is filled; it raises ``ValueError`` naming the items a required field needs when the
        filing does not report them.
        """
        cik = self.resolve_cik(ticker_or_cik, online=facts is None)
        if facts is None:
            facts = self.fetch_company_facts(cik)
        filing = filing_for_fiscal_year(facts, fiscal_year)
        return statement_model_from_filing(facts, filing, ticker=ticker_or_cik.upper(), fiscal_year=fiscal_year)


# Items beyond CANONICAL_ITEMS that the statement model needs (kind, US-GAAP tags in priority order).
STATEMENT_ITEMS: dict[str, tuple[Kind, list[str]]] = {
    **CANONICAL_ITEMS,
    "investing_cash_flow": ("duration", ["NetCashProvidedByUsedInInvestingActivities"]),
    "financing_cash_flow": ("duration", ["NetCashProvidedByUsedInFinancingActivities"]),
    "fx_effect_on_cash": ("duration", [
        "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
        "EffectOfExchangeRateOnCashAndCashEquivalents",
    ]),
    "cash_incl_restricted": ("instant", ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
    "accounts_receivable": ("instant", ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"]),
    "inventory": ("instant", ["InventoryNet"]),
    "net_ppe": ("instant", ["PropertyPlantAndEquipmentNet"]),
    "accounts_payable": ("instant", ["AccountsPayableCurrent"]),
    "retained_earnings": ("instant", ["RetainedEarningsAccumulatedDeficit"]),
    "common_stock_and_apic": ("instant", ["CommonStocksIncludingAdditionalPaidInCapital"]),
    "common_stock": ("instant", ["CommonStockValue"]),
    "additional_paid_in_capital": ("instant", ["AdditionalPaidInCapitalCommonStock", "AdditionalPaidInCapital"]),
    "long_term_debt": ("instant", ["LongTermDebt"]),
    "commercial_paper": ("instant", ["CommercialPaper"]),
    "short_term_borrowings": ("instant", ["ShortTermBorrowings"]),
    # Total interest expense only: InterestExpenseDebt excludes lease interest (Walmart).
    "interest_expense": ("duration", ["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense"]),
}


def _opening_balance(facts: dict[str, Any], filing: FilingRef, period_start: str,
                     tags: list[str]) -> tuple[str, SourcedValue] | None:
    """The balance at the start of the filing's period (the prior period-end comparative in the filing)."""
    opening = (date.fromisoformat(period_start) - timedelta(days=1)).isoformat()
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        for e in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
            if e.get("accn") == filing.accession and e.get("end") == opening and "start" not in e:
                return tag, SourcedValue(value=float(e["val"]), unit="USD", scale=1, citation=CitationRef(
                    doc_hash=fact_hash(e), filing_type=filing.form, filing_date=filing.filing_date,
                    period_end=date.fromisoformat(opening), page=None, table_id="xbrl",
                    xbrl_tag=f"us-gaap:{tag}", raw_snippet=f"{float(e['val']):,.0f} USD ({filing.accession})"))
    return None


def statement_model_from_filing(facts: dict[str, Any], filing: FilingRef, ticker: str,
                                fiscal_year: int) -> SourcedThreeStatementModel:
    """Map one 10-K's own XBRL facts into the canonical three-statement model, without placeholders.

    Required fields come from the filing or from an accounting identity on reported items, recorded
    in ``derived``:

    - total_liabilities: Liabilities, else LiabilitiesAndStockholdersEquity minus total equity.
    - total_equity: equity including non-controlling interests when tagged, else StockholdersEquity.
    - cogs / gross_profit: whichever is missing is revenue minus the other.
    - sga_expense: gross profit minus operating income (all operating expenses, including R&D; not
      the reported SG&A line), so that the schema's operating_income = gross_profit - sga holds.
    - common_equity: CommonStocksIncludingAdditionalPaidInCapital, else common stock plus APIC.
    - total_debt: LongTermDebt (else its non-current plus current parts) plus commercial paper and
      short-term borrowings where tagged. Lease liabilities are not included.
    - net_change_in_cash: CFO + CFI + CFF, as the schema defines it.

    Receivables, inventory, payables, interest and tax are 0.0 when the filing does not report them
    and are then listed in ``not_reported``. Revenue, operating income, net income, total assets,
    cash, net PP&E, retained earnings and the three cash flows must be reported: if any is missing,
    or a derivation lacks an input, this raises ``ValueError`` naming them (banks, for example, report
    no operating income; use ``get_filing_figures`` for those).

    Checks, on reported numbers only (a derived total would make its own check hold by construction):
    the balance sheet identities of ``sec_filing.check_identities``, and the cash reconciliation
    opening + CFO + CFI + CFF + exchange-rate effect = closing, on cash including restricted cash
    (the total the cash flow statement reconciles; plain cash when that is not tagged). An untagged
    exchange-rate effect is taken as zero and said so in the check.
    """
    ex = extract_filing(facts, filing, STATEMENT_ITEMS)
    v = {k: sv.value for k, sv in ex.values.items()}
    derived: dict[str, str] = {}
    lacking: list[str] = []

    def need(*names: str) -> bool:
        absent = [n for n in names if n not in v]
        lacking.extend(absent)
        return not absent

    need("revenue", "operating_income", "net_income", "total_assets", "cash", "net_ppe",
         "retained_earnings", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow")

    equity = v.get("total_equity_incl_nci", v.get("total_equity"))
    if equity is None:
        lacking.append("total_equity")
    liabilities = v.get("total_liabilities")
    if liabilities is None and equity is not None and need("liabilities_and_equity"):
        liabilities = v["liabilities_and_equity"] - equity
        derived["total_liabilities"] = "liabilities_and_equity - total_equity"
    elif liabilities is None and equity is not None:
        lacking.append("total_liabilities")

    cogs, gross = v.get("cost_of_revenue"), v.get("gross_profit")
    if "revenue" in v and cogs is None and gross is not None:
        cogs = v["revenue"] - gross
        derived["cogs"] = "revenue - gross_profit"
    elif "revenue" in v and gross is None and cogs is not None:
        gross = v["revenue"] - cogs
        derived["gross_profit"] = "revenue - cost_of_revenue"
    elif cogs is None and gross is None:
        lacking.append("cost_of_revenue or gross_profit")

    common = v.get("common_stock_and_apic")
    if common is None and "common_stock" in v and "additional_paid_in_capital" in v:
        common = v["common_stock"] + v["additional_paid_in_capital"]
        derived["common_equity"] = "common_stock + additional_paid_in_capital"
    elif common is None:
        lacking.append("common_stock_and_apic (or common_stock and additional_paid_in_capital)")

    if "long_term_debt" in v:
        debt_parts = ["long_term_debt"]
    else:
        debt_parts = [k for k in ("long_term_debt_noncurrent", "long_term_debt_current") if k in v]
    debt_parts += [k for k in ("commercial_paper", "short_term_borrowings") if k in v]
    if not debt_parts:
        lacking.append("long_term_debt (or its parts), commercial_paper or short_term_borrowings")

    if lacking:
        raise ValueError(
            f"{ticker} FY{fiscal_year} ({filing.accession}): the filing does not report "
            f"{', '.join(dict.fromkeys(lacking))}, needed for the three-statement model. "
            "get_filing_figures returns the items it does report.")
    assert liabilities is not None and equity is not None and cogs is not None and gross is not None
    assert common is not None

    total_debt = sum(v[k] for k in debt_parts)
    if len(debt_parts) > 1:
        derived["total_debt"] = " + ".join(debt_parts)
    derived["sga_expense"] = "gross_profit - operating_income (all operating expenses, not the SG&A line)"
    derived["net_change_in_cash"] = "operating + investing + financing cash flow"

    not_reported = [field for field, item in (
        ("accounts_receivable", "accounts_receivable"), ("inventory", "inventory"),
        ("accounts_payable", "accounts_payable"), ("interest_expense", "interest_expense"),
        ("tax_expense", "income_tax")) if item not in v]

    # ex.values holds reported items only, so no check runs on a derived total.
    checks = check_identities(ex.values)
    balance_checks = [c for c in checks if "assets" in c.name]
    if not balance_checks:
        raise ValueError(f"{ticker} FY{fiscal_year}: no balance sheet total to check against the reported "
                         "items (Liabilities or LiabilitiesAndStockholdersEquity)")

    sourced = dict(ex.values)
    tags_used = dict(ex.tags_used)
    cash_key = "cash_incl_restricted" if "cash_incl_restricted" in v else "cash"
    period_start = _period_start(facts, filing, tags_used["operating_cash_flow"])
    opening = _opening_balance(facts, filing, period_start, [tags_used[cash_key]])
    if opening is None:
        raise ValueError(f"{ticker} FY{fiscal_year}: the opening {cash_key} balance is not in the filing's facts")
    tags_used[f"{cash_key}_opening"], sourced[f"{cash_key}_opening"] = opening
    fx = v.get("fx_effect_on_cash", 0.0)
    flows = v["operating_cash_flow"] + v["investing_cash_flow"] + v["financing_cash_flow"]
    diff = v[cash_key] - (opening[1].value + flows + fx)
    checks.append(IdentityCheck(
        name="opening cash + cash flows = closing cash", holds=abs(diff) <= 0.5, difference=diff,
        detail=f"{cash_key}; CFO + CFI + CFF + exchange-rate effect"
               + ("" if "fx_effect_on_cash" in v else " (exchange-rate effect not tagged, taken as 0)")))
    cash_check = checks[-1]

    zero = 0.0
    model = ThreeStatementModel(
        income_statement=IncomeStatement(
            revenue=v["revenue"], cogs=cogs, gross_profit=gross,
            sga_expense=gross - v["operating_income"], operating_income=v["operating_income"],
            interest_expense=v.get("interest_expense", zero), tax_expense=v.get("income_tax", zero),
            net_income=v["net_income"]),
        cash_flow_statement=CashFlowStatement(
            cfo=v["operating_cash_flow"], cfi=v["investing_cash_flow"], cff=v["financing_cash_flow"],
            net_change_in_cash=flows),
        balance_sheet=BalanceSheet(
            cash_and_equivalents=v["cash"], accounts_receivable=v.get("accounts_receivable", zero),
            inventory=v.get("inventory", zero), net_ppe=v["net_ppe"], total_assets=v["total_assets"],
            accounts_payable=v.get("accounts_payable", zero), total_debt=total_debt,
            total_liabilities=liabilities, retained_earnings=v["retained_earnings"],
            common_equity=common, total_equity=equity),
        is_balanced=all(c.holds for c in balance_checks),
        cash_reconciled=cash_check.holds,
    )
    return SourcedThreeStatementModel(
        ticker=ticker, cik=filing.cik, fiscal_year=fiscal_year, form=filing.form, filing=filing,
        statement_model=model, sourced_facts=sourced, tags_used=tags_used, derived=derived,
        not_reported=not_reported, checks=checks, is_formally_verified=all(c.holds for c in checks))


def _period_start(facts: dict[str, Any], filing: FilingRef, tag: str) -> str:
    """Start date of the filing's own period for a duration tag (the one extract_filing selected)."""
    for e in facts["facts"]["us-gaap"][tag]["units"]["USD"]:
        if e.get("accn") == filing.accession and e.get("end") == filing.report_date.isoformat() and "start" in e:
            lo, hi = DURATION_DAYS[filing.form]
            if lo <= (filing.report_date - date.fromisoformat(e["start"])).days <= hi:
                return str(e["start"])
    raise ValueError(f"no annual period for {tag} in {filing.accession}")
