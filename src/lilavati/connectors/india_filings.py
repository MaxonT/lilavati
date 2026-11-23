"""
Ind AS disclosure crosswalk: Schedule III line items to the canonical three-statement model.

The caller supplies the disclosure (line items from an annual report filed under Indian Accounting
Standards); this module does not fetch filings. Every value is cited to the disclosure as supplied;
derived fields carry their formula, and items not given are listed, never filled with zero.
"""

from __future__ import annotations

import hashlib
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from lilavati.connectors.sec_filing import IdentityCheck
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef, SourcedValue, SourceStore
from lilavati.schemas.statements import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    ThreeStatementModel,
)


class IndASCompanyDisclosure(BaseModel):
    """Normalized structured disclosure from Indian regulatory filings (BSE/NSE/MCA)."""

    model_config = ConfigDict(frozen=True)

    company_name: str
    symbol: str = Field(..., description="NSE/BSE Scrip Symbol (e.g. 'RELIANCE', 'INFY', 'TCS')")
    cin: str = Field(..., description="Corporate Identification Number (21-character MCA CIN)")
    fiscal_year: int
    reporting_date: date
    filing_date: date
    currency: str = Field(default="INR")
    scale: int = Field(
        default=10_000_000, description="Default in Crores (1 Cr = 10M INR) or units"
    )
    facts: dict[str, float]


class IndASParseResult(BaseModel):
    """An Ind AS disclosure mapped into the canonical three-statement model.

    Each statement field is reported (cited in ``sourced_facts``), derived from reported items by the
    formula in ``derived``, or, for the fields the schema defaults to zero, listed in
    ``not_reported`` (0.0 in ``statement_model``, not a reported zero). Amounts are in the
    disclosure's currency units (facts times ``scale``).
    """

    model_config = ConfigDict(frozen=True)

    statement_model: ThreeStatementModel
    sourced_facts: dict[str, SourcedValue]
    derived: dict[str, str] = Field(default_factory=dict)
    not_reported: list[str] = Field(default_factory=list)
    checks: list[IdentityCheck] = Field(default_factory=list, description="identities on reported numbers")


class IndiaFilingsClient:
    """Crosswalk from Ind AS (Schedule III, Division II) line items to the canonical statements.

    The disclosure's ``facts`` use the keys below (values in the disclosure's scale, e.g. crores).
    Not validated on live filings yet (docs/validation/connectors.md).
    """

    # canonical item -> Ind AS keys in priority order (first present wins)
    TAXONOMY_MAP: dict[str, list[str]] = {
        "revenue": ["RevenueFromOperations"],
        "other_income": ["OtherIncome"],
        "total_expenses": ["TotalExpenses"],
        "employee_expense": ["EmployeeBenefitsExpense"],
        "finance_costs": ["FinanceCosts"],
        "depreciation": ["DepreciationAndAmortisationExpense"],
        "pbt": ["ProfitBeforeTax"],
        "tax": ["TaxExpense"],
        "pat": ["ProfitForThePeriod", "ProfitLossForThePeriod"],
        "cash": ["CashAndCashEquivalents"],
        "trade_receivables": ["TradeReceivables"],
        "inventories": ["Inventories"],
        "ppe": ["PropertyPlantAndEquipment"],
        "total_assets": ["TotalAssets"],
        "trade_payables": ["TradePayables"],
        "equity_capital": ["EquityShareCapital"],
        "securities_premium": ["SecuritiesPremium"],
        "retained_earnings": ["RetainedEarnings"],
        "other_equity": ["OtherEquity"],
        "total_equity": ["TotalEquity"],
        "total_liabilities": ["TotalLiabilities"],
        "total_equity_and_liabilities": ["TotalEquityAndLiabilities"],
        "cfo": ["NetCashFromOperatingActivities"],
        "cfi": ["NetCashFromInvestingActivities"],
        "cff": ["NetCashFromFinancingActivities"],
        "fx_effect_on_cash": ["EffectOfExchangeRateChangesOnCash"],
        "cash_flow_opening": ["CashAndCashEquivalentsAtBeginningOfPeriod"],
        "cash_flow_closing": ["CashAndCashEquivalentsAtEndOfPeriod"],
    }
    # Schedule III presents cost of goods sold as three lines; their sum is the cost (the change in
    # inventories is negative when inventories rise).
    COGS_COMPONENTS = ["CostOfMaterialsConsumed", "PurchasesOfStockInTrade",
                       "ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade"]
    TAX_COMPONENTS = ["CurrentTax", "DeferredTax"]
    BORROWINGS = ["NonCurrentBorrowings", "CurrentBorrowings"]

    @staticmethod
    def parse_ind_as_disclosure(disclosure: IndASCompanyDisclosure, tolerance: float = 0.01) -> IndASParseResult:
        """Map an Ind AS disclosure into the canonical model, without filling missing items.

        Derived fields (recorded in ``derived``):

        - cogs: the sum of the Schedule III cost lines present (materials consumed, purchases of
          stock-in-trade, change in inventories); listed in ``not_reported`` when none is given
          (service companies). gross_profit = revenue - cogs.
        - operating_income: revenue - (total expenses - finance costs). Other income and exceptional
          items are excluded. sga_expense = gross_profit - operating_income (all other operating
          expenses, including employee costs and depreciation).
        - tax_expense: TaxExpense, else CurrentTax + DeferredTax when both are given.
        - total_debt: the borrowings lines present (lease liabilities excluded); listed in
          ``not_reported`` when none is given (debt-free companies omit the line).
        - total_liabilities: TotalLiabilities, else TotalEquityAndLiabilities - TotalEquity.
        - retained_earnings: RetainedEarnings, else OtherEquity (all reserves, recorded as such).
        - common_equity: EquityShareCapital, plus SecuritiesPremium when given.

        Required, else ``ValueError`` naming them: revenue, total expenses, finance costs, profit for
        the period, cash, PP&E, total assets, total equity, equity share capital, a retained earnings
        or other equity figure, the three cash flows, and the cash flow statement's opening and
        closing cash. Checks (on reported numbers, within ``tolerance`` in the disclosure's units,
        for rounding in published statements): assets = liabilities + equity (Z3) when total
        liabilities is reported; assets = total equity and liabilities when that is reported; opening
        cash + CFO + CFI + CFF + exchange-rate effect = closing cash (Ind AS 7). Cash in the cash flow
        statement can differ from the balance sheet (bank overdrafts), so both are kept.
        """
        f = disclosure.facts
        scale = disclosure.scale
        tm = IndiaFilingsClient.TAXONOMY_MAP
        used: dict[str, str] = {}  # canonical item -> Ind AS key

        def get(item: str) -> float | None:
            for k in tm[item]:
                if k in f:
                    used[item] = k
                    return f[k]
            return None

        r = {item: get(item) for item in tm}
        lacking = [item for item in ("revenue", "total_expenses", "finance_costs", "pat", "cash", "ppe",
                                     "total_assets", "total_equity", "equity_capital", "cfo", "cfi", "cff",
                                     "cash_flow_opening", "cash_flow_closing") if r[item] is None]
        if r["retained_earnings"] is None and r["other_equity"] is None:
            lacking.append("retained_earnings or other_equity")
        liabilities = r["total_liabilities"]
        derived: dict[str, str] = {}
        if liabilities is None and r["total_equity_and_liabilities"] is not None and r["total_equity"] is not None:
            liabilities = r["total_equity_and_liabilities"] - r["total_equity"]
            derived["total_liabilities"] = "TotalEquityAndLiabilities - TotalEquity"
        elif liabilities is None:
            lacking.append("total_liabilities or total_equity_and_liabilities")
        if lacking:
            raise ValueError(f"{disclosure.symbol} FY{disclosure.fiscal_year}: the disclosure does not give "
                             f"{', '.join(lacking)}, needed for the three-statement model")
        v = {k: x for k, x in r.items() if x is not None}
        assert liabilities is not None
        not_reported: list[str] = []

        cogs_keys = [k for k in IndiaFilingsClient.COGS_COMPONENTS if k in f]
        cogs = sum(f[k] for k in cogs_keys)
        if cogs_keys:
            derived["cogs"] = " + ".join(cogs_keys)
        else:
            not_reported.append("cogs")
        derived["gross_profit"] = "revenue - cogs"
        gross = v["revenue"] - cogs
        operating = v["revenue"] - (v["total_expenses"] - v["finance_costs"])
        derived["operating_income"] = "RevenueFromOperations - (TotalExpenses - FinanceCosts)"
        derived["sga_expense"] = "gross_profit - operating_income (all other operating expenses)"

        tax = v.get("tax")
        if tax is None and all(k in f for k in IndiaFilingsClient.TAX_COMPONENTS):
            tax = sum(f[k] for k in IndiaFilingsClient.TAX_COMPONENTS)
            derived["tax_expense"] = "CurrentTax + DeferredTax"
        if tax is None:
            not_reported.append("tax_expense")

        debt_keys = [k for k in IndiaFilingsClient.BORROWINGS if k in f]
        if len(debt_keys) > 1:
            derived["total_debt"] = " + ".join(debt_keys)
        if not debt_keys:
            not_reported.append("total_debt")
        retained = v.get("retained_earnings")
        if retained is None:
            retained = v["other_equity"]
            derived["retained_earnings"] = "OtherEquity (all reserves; retained earnings not given separately)"
        common = v["equity_capital"] + v.get("securities_premium", 0.0)
        if "securities_premium" in v:
            derived["common_equity"] = "EquityShareCapital + SecuritiesPremium"
        for field, item in (("accounts_receivable", "trade_receivables"), ("inventory", "inventories"),
                            ("accounts_payable", "trade_payables")):
            if item not in v:
                not_reported.append(field)

        flows = v["cfo"] + v["cfi"] + v["cff"]
        derived["net_change_in_cash"] = "CFO + CFI + CFF"
        checks: list[IdentityCheck] = []
        if "total_liabilities" in v:
            res = SuvarnaEngine.verify_balance_sheet_equation(
                total_assets=v["total_assets"], total_liabilities=v["total_liabilities"],
                shareholders_equity=v["total_equity"], tolerance=tolerance)
            checks.append(IdentityCheck(
                name="assets = liabilities + equity", holds=res.is_valid,
                difference=(v["total_assets"] - v["total_liabilities"] - v["total_equity"]) * scale,
                detail="Z3 on the reported numbers"))
        if "total_equity_and_liabilities" in v:
            d = v["total_assets"] - v["total_equity_and_liabilities"]
            checks.append(IdentityCheck(name="assets = total equity and liabilities", holds=abs(d) <= tolerance,
                                        difference=d * scale, detail="both totals as reported"))
        fx = v.get("fx_effect_on_cash", 0.0)
        d = v["cash_flow_closing"] - (v["cash_flow_opening"] + flows + fx)
        checks.append(IdentityCheck(
            name="opening cash + cash flows = closing cash", holds=abs(d) <= tolerance, difference=d * scale,
            detail="cash flow statement balances; CFO + CFI + CFF + exchange-rate effect"
                   + ("" if "fx_effect_on_cash" in v else " (exchange-rate effect not given, taken as 0)")))
        balance = [c for c in checks if "assets" in c.name]

        def s(x: float | None) -> float:
            return 0.0 if x is None else x * scale

        model = ThreeStatementModel(
            income_statement=IncomeStatement(
                revenue=s(v["revenue"]), cogs=s(cogs), gross_profit=s(gross), sga_expense=s(gross - operating),
                operating_income=s(operating), interest_expense=s(v["finance_costs"]), tax_expense=s(tax),
                net_income=s(v["pat"])),
            cash_flow_statement=CashFlowStatement(cfo=s(v["cfo"]), cfi=s(v["cfi"]), cff=s(v["cff"]),
                                                  net_change_in_cash=s(flows)),
            balance_sheet=BalanceSheet(
                cash_and_equivalents=s(v["cash"]), accounts_receivable=s(v.get("trade_receivables")),
                inventory=s(v.get("inventories")), net_ppe=s(v["ppe"]), total_assets=s(v["total_assets"]),
                accounts_payable=s(v.get("trade_payables")), total_debt=s(sum(f[k] for k in debt_keys)),
                total_liabilities=s(liabilities), retained_earnings=s(retained), common_equity=s(common),
                total_equity=s(v["total_equity"])),
            is_balanced=all(c.holds for c in balance),
            cash_reconciled=checks[-1].holds,
        )

        doc_hash = disclosure_hash(disclosure)  # the disclosure as supplied: the only source this parser sees
        cited = set(used.values()) | set(cogs_keys) | set(debt_keys)
        if "tax_expense" in derived:
            cited |= set(IndiaFilingsClient.TAX_COMPONENTS)
        sourced: dict[str, SourcedValue] = {}
        for key in sorted(cited):
            val = f[key] * scale
            sourced[key] = SourcedValue(value=val, unit=disclosure.currency, scale=1, citation=CitationRef(
                doc_hash=doc_hash, filing_type="Ind-AS-Annual", filing_date=disclosure.filing_date,
                period_end=disclosure.reporting_date, page=None, table_id="mca_financials",
                xbrl_tag=f"mca-ind-as:{key}", raw_snippet=f"{val:,.0f} {disclosure.currency} (CIN: {disclosure.cin})"))
        return IndASParseResult(statement_model=model, sourced_facts=sourced, derived=derived,
                                not_reported=not_reported, checks=checks)


def disclosure_hash(disclosure: IndASCompanyDisclosure) -> str:
    """SHA-256 of the disclosure as supplied (canonical JSON): the ``doc_hash`` of its citations."""
    return hashlib.sha256(disclosure.model_dump_json().encode()).hexdigest()


def register_disclosure(store: SourceStore, disclosure: IndASCompanyDisclosure) -> str:
    """Register the disclosure in ``store`` so that ``ProvenanceGate(store)`` can resolve its citations."""
    doc_hash = disclosure_hash(disclosure)
    store.add_tagged_values(doc_hash, {f"mca-ind-as:{k}": v * disclosure.scale for k, v in disclosure.facts.items()})
    return doc_hash
