"""Filing-anchored XBRL extraction from SEC EDGAR: identify a 10-K or 10-Q, then use only its own facts.

Flow (see docs/validation/connectors.md):

1. ``filings_from_submissions``: list a company's filings from the SEC submissions index
   (accession number, form, filing date, report date, primary document).
2. ``extract_filing``: for each canonical line item, try an ordered list of US-GAAP tags and keep
   only facts that belong to that filing: same accession number, end date equal to the filing's
   report date, and (for flow items) a duration of about a year (10-K) or a quarter (10-Q).
   Selecting by the ``fy`` field alone is wrong: a 10-K tags its prior-year comparatives with its
   own fiscal year.
3. Items not found are listed in ``missing``; nothing is filled with zero. Company-specific
   (extension) tags are not guessed.
4. ``check_identities``: assets equal liabilities plus equity (Z3, on the reported numbers), total
   assets equal total liabilities and equity, revenue minus cost of revenue equals gross profit,
   where the items exist.

Values are in US dollars as reported in XBRL (not scaled). Every value carries a citation with the
accession number, tag and period.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef, SourcedValue, record_hash

if TYPE_CHECKING:  # edgar imports this module; the client is only needed at call time
    from lilavati.connectors.edgar import EdgarClient

Kind = Literal["instant", "duration"]

# canonical item -> (instant or duration, US-GAAP tags in priority order)
CANONICAL_ITEMS: dict[str, tuple[Kind, list[str]]] = {
    # Total revenues first: Walmart tags net sales (706,413) as contract revenue and total revenues
    # including membership income (713,163) as Revenues; most filers tag only one of the two.
    "revenue": ("duration", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "SalesRevenueNet", "RevenuesNetOfInterestExpense"]),
    "cost_of_revenue": ("duration", ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"]),
    "gross_profit": ("duration", ["GrossProfit"]),
    "operating_income": ("duration", ["OperatingIncomeLoss"]),
    "pretax_income": ("duration", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ]),
    "income_tax": ("duration", ["IncomeTaxExpenseBenefit"]),
    "net_income": ("duration", ["NetIncomeLoss", "ProfitLoss"]),
    "operating_cash_flow": ("duration", ["NetCashProvidedByUsedInOperatingActivities"]),
    "capital_expenditure": ("duration", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "total_assets": ("instant", ["Assets"]),
    "total_liabilities": ("instant", ["Liabilities"]),
    "total_equity": ("instant", ["StockholdersEquity"]),
    "total_equity_incl_nci": ("instant", ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "liabilities_and_equity": ("instant", ["LiabilitiesAndStockholdersEquity"]),
    "cash": ("instant", ["CashAndCashEquivalentsAtCarryingValue"]),
    "long_term_debt_noncurrent": ("instant", ["LongTermDebtNoncurrent"]),
    "long_term_debt_current": ("instant", ["LongTermDebtCurrent"]),
}

# Accepted durations in days for flow items: annual for 10-K, quarterly for 10-Q.
DURATION_DAYS = {"10-K": (350, 380), "10-Q": (80, 100)}


class FilingRef(BaseModel):
    """One filing from the SEC submissions index."""

    model_config = ConfigDict(frozen=True)

    cik: str
    form: str
    accession: str
    filing_date: date
    report_date: date
    primary_document: str = Field(default="", description="empty when only the company facts were read")

    @property
    def url(self) -> str:
        """The primary document, or the filing's folder when the document name is not known."""
        return (f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
                f"{self.accession.replace('-', '')}/{self.primary_document}")


class IdentityCheck(BaseModel):
    name: str
    holds: bool
    difference: float = Field(..., description="left minus right, US dollars")
    detail: str


class FilingExtraction(BaseModel):
    company: str
    filing: FilingRef
    values: dict[str, SourcedValue]
    tags_used: dict[str, str]
    missing: list[str] = Field(..., description="canonical items not found for this filing")
    checks: list[IdentityCheck]


def fact_hash(entry: dict[str, Any]) -> str:
    """SHA-256 of one XBRL fact record as SEC serves it (canonical JSON: sorted keys, no spaces).

    Re-fetch the company facts, find the entry with the same accession, tag, period and value, and
    hash it the same way to check a citation. The record does not contain the concept name: two
    concepts reported with the same value, period and accession share a hash (Apple's cash with and
    without restricted cash), so the citation's ``xbrl_tag`` is part of the reference.
    """
    return record_hash(entry)


def filings_from_submissions(submissions: dict[str, Any], forms: tuple[str, ...] = ("10-K",)) -> list[FilingRef]:
    """Filings of the given forms from an SEC submissions document, newest first (amendments excluded)."""
    recent = submissions["filings"]["recent"]
    cik = str(submissions["cik"]).zfill(10)
    out = []
    for i, form in enumerate(recent["form"]):
        if form in forms and recent["reportDate"][i]:
            out.append(FilingRef(cik=cik, form=form, accession=recent["accessionNumber"][i],
                                 filing_date=date.fromisoformat(recent["filingDate"][i]),
                                 report_date=date.fromisoformat(recent["reportDate"][i]),
                                 primary_document=recent["primaryDocument"][i]))
    return sorted(out, key=lambda f: f.filing_date, reverse=True)


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def filing_for_fiscal_year(facts: dict[str, Any], fiscal_year: int, form: str = "10-K") -> FilingRef:
    """The ``form`` filing for ``fiscal_year``, identified from the company facts alone.

    Every fact a 10-K reports carries that filing's fiscal year (``fy``), including its prior-year
    comparatives, so the facts with ``fy == fiscal_year``, ``fp == "FY"`` and an annual duration all
    come from the one filing; its period end is their latest end date. If several accessions match
    (a re-filed 10-K), the one with the latest period end is taken, then the earliest filed.
    Amendments (10-K/A) are excluded. The primary document name is not in the company facts, so
    ``url`` is the filing's folder.
    """
    if form not in DURATION_DAYS:
        raise ValueError(f"form must be one of {sorted(DURATION_DAYS)}, got {form!r}")
    lo, hi = DURATION_DAYS[form]
    found: dict[str, tuple[str, str]] = {}  # accession -> (latest end, earliest filed)
    for concept in facts.get("facts", {}).get("us-gaap", {}).values():
        for entries in concept.get("units", {}).values():
            for e in entries:
                if (e.get("form") != form or e.get("fy") != fiscal_year or e.get("fp") != "FY"
                        or "start" not in e or not lo <= _days(e["start"], e["end"]) <= hi):
                    continue
                end, filed = found.get(e["accn"], (e["end"], e["filed"]))
                found[e["accn"]] = (max(end, e["end"]), min(filed, e["filed"]))
    if not found:
        raise ValueError(f"no {form} with annual facts for fiscal year {fiscal_year} in the company facts")
    latest_end = max(end for end, _ in found.values())
    accession = min((a for a, (end, _) in found.items() if end == latest_end), key=lambda a: found[a][1])
    end, filed = found[accession]
    return FilingRef(cik=str(facts["cik"]).zfill(10), form=form, accession=accession,
                     filing_date=date.fromisoformat(filed), report_date=date.fromisoformat(end))


def fiscal_year_of(facts: dict[str, Any], filing: FilingRef) -> int:
    """The fiscal year (``fy``) the filing's own facts carry."""
    for concept in facts.get("facts", {}).get("us-gaap", {}).values():
        for entries in concept.get("units", {}).values():
            for e in entries:
                if e.get("accn") == filing.accession and e.get("fp") == "FY" and "fy" in e:
                    return int(e["fy"])
    raise ValueError(f"no annual facts for {filing.accession} in the company facts")


def prior_period_view(facts: dict[str, Any], filing: FilingRef) -> FilingRef:
    """The filing's prior-year comparatives as a ``FilingRef`` for ``extract_filing``.

    A 10-K reports last year's figures next to this year's under the same accession. Their period end
    is the day before the current annual period starts (read from the filing's own duration facts).
    These are the comparatives as published in this filing (restated if the company restated them),
    so they are point-in-time correct for the filing date.
    """
    lo, hi = DURATION_DAYS.get(filing.form, (350, 380))
    end = filing.report_date.isoformat()
    for concept in facts.get("facts", {}).get("us-gaap", {}).values():
        for entries in concept.get("units", {}).values():
            for e in entries:
                if (e.get("accn") == filing.accession and e.get("end") == end and "start" in e
                        and lo <= _days(e["start"], e["end"]) <= hi):
                    prior_end = date.fromisoformat(e["start"]) - timedelta(days=1)
                    return filing.model_copy(update={"report_date": prior_end})
    raise ValueError(f"no {filing.form} period found for {filing.accession}")


def _select(entries: list[dict[str, Any]], filing: FilingRef, kind: Kind) -> dict[str, Any] | None:
    end = filing.report_date.isoformat()
    lo, hi = DURATION_DAYS.get(filing.form, (350, 380))
    for e in entries:
        if e.get("accn") != filing.accession or e.get("end") != end:
            continue
        if kind == "instant" and "start" not in e:
            return e
        if kind == "duration" and "start" in e and lo <= _days(e["start"], e["end"]) <= hi:
            return e
    return None


def extract_filing(facts: dict[str, Any], filing: FilingRef,
                   items: dict[str, tuple[Kind, list[str]]] = CANONICAL_ITEMS) -> FilingExtraction:
    """Line items reported in ``filing``, from its own XBRL facts (see the module docstring)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    values: dict[str, SourcedValue] = {}
    tags_used: dict[str, str] = {}
    missing: list[str] = []
    for name, (kind, tags) in items.items():
        chosen = None
        for tag in tags:
            entries = us_gaap.get(tag, {}).get("units", {}).get("USD", [])
            chosen = _select(entries, filing, kind)
            if chosen is not None:
                tags_used[name] = tag
                break
        if chosen is None:
            missing.append(name)
            continue
        doc_hash = fact_hash(chosen)
        values[name] = SourcedValue(value=float(chosen["val"]), unit="USD", scale=1, citation=CitationRef(
            doc_hash=doc_hash, filing_type=filing.form, filing_date=filing.filing_date,
            period_end=filing.report_date, page=None, table_id="xbrl", xbrl_tag=f"us-gaap:{tags_used[name]}",
            raw_snippet=f"{float(chosen['val']):,.0f} USD ({filing.accession})"))
    return FilingExtraction(company=str(facts.get("entityName", "")), filing=filing, values=values,
                            tags_used=tags_used, missing=missing, checks=check_identities(values))


def check_identities(values: dict[str, SourcedValue]) -> list[IdentityCheck]:
    """Accounting identities on the reported numbers, for the items present."""
    v = {k: sv.value for k, sv in values.items()}
    checks: list[IdentityCheck] = []
    equity = v.get("total_equity_incl_nci", v.get("total_equity"))
    if "total_assets" in v and "total_liabilities" in v and equity is not None:
        res = SuvarnaEngine.verify_balance_sheet_equation(
            total_assets=v["total_assets"], total_liabilities=v["total_liabilities"],
            shareholders_equity=equity, tolerance=0.5)
        checks.append(IdentityCheck(name="assets = liabilities + equity", holds=res.is_valid,
                                    difference=v["total_assets"] - v["total_liabilities"] - equity,
                                    detail="Z3 on the reported numbers; equity includes non-controlling interests when tagged"))
    if "total_assets" in v and "liabilities_and_equity" in v:
        d = v["total_assets"] - v["liabilities_and_equity"]
        checks.append(IdentityCheck(name="assets = total liabilities and equity", holds=abs(d) <= 0.5,
                                    difference=d, detail="both totals as tagged"))
    if all(k in v for k in ("revenue", "cost_of_revenue", "gross_profit")):
        d = v["revenue"] - v["cost_of_revenue"] - v["gross_profit"]
        checks.append(IdentityCheck(name="revenue - cost of revenue = gross profit", holds=abs(d) <= 0.5,
                                    difference=d, detail="income statement subtotal"))
    return checks


# --- Inputs for comparisons with language models --------------------------------------------

INCOME_STATEMENT_TITLES = (
    "consolidated statements of operations", "consolidated statement of operations",
    "consolidated statements of income", "consolidated statement of income",
    "consolidated statements of earnings", "income statements", "consolidated income statements",
)


def document_text(raw_html: str) -> str:
    """Visible text of an (inline XBRL) filing document, whitespace collapsed."""
    import html as _html
    import re as _re

    # Inline tags are removed without a space: a word split across spans ("IN" + "COME") must stay
    # one word. Every other tag (cells, rows, paragraphs, line breaks) separates text.
    text = _re.sub(r"</?(?:span|a|b|i|u|em|strong|font|sup|sub|ix:[a-zA-Z]+)\b[^>]*>", "", raw_html,
                   flags=_re.IGNORECASE)
    text = _re.sub(r"<[^>]+>", " ", text)
    return _re.sub(r"\s+", " ", _html.unescape(text))


BALANCE_SHEET_TITLES = (
    "consolidated balance sheets", "consolidated balance sheet", "balance sheets",
    "consolidated statements of financial position", "consolidated statement of financial position",
)


MIN_FIGURES = 5  # comma-grouped numbers required within 6,000 characters after a statement title


def _best_anchor(text: str, titles: tuple[str, ...]) -> int | None:
    import re as _re

    low = text.lower()
    spans = sorted({(m.start(), m.end()) for t in titles for m in _re.finditer(_re.escape(t), low)},
                   key=lambda sp: (sp[0], -sp[1]))
    # A shorter title inside a longer match ("balance sheets" in "consolidated balance sheets") is the
    # same heading: keep the longer one only.
    distinct = [sp for sp in spans if not any(o != sp and o[0] <= sp[0] and sp[1] <= o[1] for o in spans)]
    best: tuple[int, int] | None = None
    for start, _ in distinct:
        figures = len(_re.findall(r"\d{1,3}(?:,\d{3})+", text[start: start + 6000]))
        if best is None or figures >= best[0]:  # ties: the later one (statements follow the contents)
            best = (figures, start)
    # A heading followed by almost no figures (a contents line, a note) is not a statement.
    return best[1] if best and best[0] >= MIN_FIGURES else None


def statements_window(text: str, chars: int = 40_000) -> str:
    """The financial statements section: ``chars`` characters from the first primary statement.

    For the income statement and the balance sheet, the title occurrence followed by the most figures
    is taken (a table of contents has titles but few numbers; ties go to the later occurrence, since
    the statements follow the contents); the window starts at whichever comes first (J&J puts the
    balance sheet before the income statement). Comprehensive-income titles are not anchors. Callers
    should check that the figures they need are inside the window (see
    ``scripts/run_filing_comparison.py``).
    """
    starts = [a for a in (_best_anchor(text, INCOME_STATEMENT_TITLES), _best_anchor(text, BALANCE_SHEET_TITLES))
              if a is not None]
    if not starts:
        raise ValueError("no income statement or balance sheet heading found in the document")
    start = min(starts)
    return text[start: start + chars]


def filing_ratios(values: dict[str, float]) -> dict[str, float]:
    """Deterministic ratios from line items (any consistent unit). Only ratios whose inputs exist.

    gross_margin = (revenue - cost_of_revenue) / revenue; net_margin = net_income / revenue;
    return_on_assets = net_income / total_assets (period-end assets, not average);
    free_cash_flow = operating_cash_flow - capital_expenditure;
    liabilities_to_assets = total_liabilities / total_assets. Fractions, not percent.
    """
    v = values
    out: dict[str, float] = {}
    if v.get("revenue") and "cost_of_revenue" in v:
        out["gross_margin"] = (v["revenue"] - v["cost_of_revenue"]) / v["revenue"]
    if v.get("revenue") and "net_income" in v:
        out["net_margin"] = v["net_income"] / v["revenue"]
    if v.get("total_assets") and "net_income" in v:
        out["return_on_assets"] = v["net_income"] / v["total_assets"]
    if "operating_cash_flow" in v and "capital_expenditure" in v:
        out["free_cash_flow"] = v["operating_cash_flow"] - v["capital_expenditure"]
    if v.get("total_assets") and "total_liabilities" in v:
        out["liabilities_to_assets"] = v["total_liabilities"] / v["total_assets"]
    return out


def fetch_filing_extraction(ticker_or_cik: str, form: str = "10-K", filings_back: int = 0,
                            client: EdgarClient | None = None) -> FilingExtraction:
    """Live: the company's ``form`` filing (0 = latest, 1 = the one before, ...) and its own line items.

    Two SEC requests (submissions index and company facts). Needs a contact in
    ``LILAVATI_SEC_USER_AGENT`` or a client built with ``user_agent=``.
    """
    if form not in DURATION_DAYS:
        raise ValueError(f"form must be one of {sorted(DURATION_DAYS)}, got {form!r}")
    if filings_back < 0:
        raise ValueError("filings_back must be 0 (latest) or more")
    from lilavati.connectors.edgar import EdgarClient

    client = client or EdgarClient()
    filings = filings_from_submissions(client.fetch_submissions(ticker_or_cik), forms=(form,))
    if filings_back >= len(filings):
        raise ValueError(f"{ticker_or_cik}: only {len(filings)} {form} filings in SEC's recent submissions index")
    filing = filings[filings_back]
    return extract_filing(client.fetch_company_facts(filing.cik), filing)
