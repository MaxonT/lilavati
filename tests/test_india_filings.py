"""Ind AS crosswalk: Schedule III line items to the canonical statements, without zero-fill.

The disclosure is synthetic (a fictional company, amounts in crores) and built to satisfy the
Schedule III and Ind AS 7 arithmetic shown in the comments. Live annual reports are not covered yet.
"""

from __future__ import annotations

from datetime import date

import pytest

from lilavati.connectors.india_filings import IndASCompanyDisclosure, IndiaFilingsClient

CR = 10_000_000  # 1 crore

FACTS = {
    # Statement of profit and loss
    "RevenueFromOperations": 10_000.0,
    "OtherIncome": 200.0,
    "CostOfMaterialsConsumed": 4_000.0,
    "PurchasesOfStockInTrade": 1_000.0,
    "ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade": -200.0,  # inventories rose
    "EmployeeBenefitsExpense": 1_200.0,
    "FinanceCosts": 300.0,
    "DepreciationAndAmortisationExpense": 500.0,
    "OtherExpenses": 1_700.0,
    "TotalExpenses": 8_500.0,  # 4,000 + 1,000 - 200 + 1,200 + 300 + 500 + 1,700
    "ProfitBeforeTax": 1_700.0,  # 10,000 + 200 - 8,500
    "CurrentTax": 400.0,
    "DeferredTax": 25.0,
    "ProfitForThePeriod": 1_275.0,  # 1,700 - 425
    # Balance sheet
    "PropertyPlantAndEquipment": 6_000.0,
    "Inventories": 1_500.0,
    "TradeReceivables": 1_800.0,
    "CashAndCashEquivalents": 700.0,
    "TotalAssets": 12_000.0,
    "EquityShareCapital": 500.0,
    "OtherEquity": 6_500.0,
    "TotalEquity": 7_000.0,
    "NonCurrentBorrowings": 2_000.0,
    "CurrentBorrowings": 500.0,
    "TradePayables": 1_400.0,
    "TotalEquityAndLiabilities": 12_000.0,
    # Cash flow statement (Ind AS 7): 600 + 1,900 - 1,100 - 710 + 10 = 700
    "CashAndCashEquivalentsAtBeginningOfPeriod": 600.0,
    "NetCashFromOperatingActivities": 1_900.0,
    "NetCashFromInvestingActivities": -1_100.0,
    "NetCashFromFinancingActivities": -710.0,
    "EffectOfExchangeRateChangesOnCash": 10.0,
    "CashAndCashEquivalentsAtEndOfPeriod": 700.0,
}


def _disclosure(facts: dict[str, float]) -> IndASCompanyDisclosure:
    return IndASCompanyDisclosure(
        company_name="Example Manufacturing Ltd (fictional)", symbol="EXAMPLE", cin="U00000XX2000PLC000000",
        fiscal_year=2026, reporting_date=date(2026, 3, 31), filing_date=date(2026, 5, 15), scale=CR,
        facts=facts)


def test_schedule_iii_crosswalk() -> None:
    res = IndiaFilingsClient.parse_ind_as_disclosure(_disclosure(FACTS))
    inc, bs, cf = (res.statement_model.income_statement, res.statement_model.balance_sheet,
                   res.statement_model.cash_flow_statement)
    # Cost of goods sold is the three Schedule III lines (the old map took only the first, 4,000).
    assert inc.cogs == 4_800 * CR and inc.gross_profit == 5_200 * CR
    # Operating income = revenue - (total expenses - finance costs); the old formula left out other
    # expenses and returned 6,000 - 1,200 - 500 = 4,300.
    assert inc.operating_income == 1_800 * CR
    assert inc.sga_expense == (1_200 + 500 + 1_700) * CR  # employee + D&A + other expenses
    assert inc.tax_expense == 425 * CR and res.derived["tax_expense"] == "CurrentTax + DeferredTax"
    assert inc.interest_expense == 300 * CR and inc.net_income == 1_275 * CR
    assert bs.total_liabilities == 5_000 * CR  # 12,000 - 7,000, derived and recorded
    assert res.derived["total_liabilities"] == "TotalEquityAndLiabilities - TotalEquity"
    assert bs.total_debt == 2_500 * CR and bs.common_equity == 500 * CR
    assert bs.retained_earnings == 6_500 * CR and "OtherEquity" in res.derived["retained_earnings"]
    assert cf.net_change_in_cash == 90 * CR
    assert res.not_reported == []
    # Only the check on reported totals runs: A = L + E would hold by construction.
    assert [c.name for c in res.checks] == ["assets = total equity and liabilities",
                                            "opening cash + cash flows = closing cash"]
    assert all(c.holds for c in res.checks)
    assert res.statement_model.is_balanced and res.statement_model.cash_reconciled
    # Citations only for items given in the disclosure.
    assert "TotalLiabilities" not in res.sourced_facts
    assert res.sourced_facts["CostOfMaterialsConsumed"].value == 4_000 * CR
    assert all(len(sv.citation.doc_hash) == 64 for sv in res.sourced_facts.values())


def test_cash_reconciliation_is_checked_not_assumed() -> None:
    # The old parser back-solved opening cash from the flows and set cash_reconciled=True.
    res = IndiaFilingsClient.parse_ind_as_disclosure(_disclosure({**FACTS, "NetCashFromOperatingActivities": 1_950.0}))
    cash = res.checks[-1]
    assert not cash.holds and cash.difference == -50 * CR
    assert res.statement_model.cash_reconciled is False


def test_reported_liabilities_get_the_z3_check() -> None:
    facts = {**FACTS, "TotalLiabilities": 4_900.0}  # inconsistent with assets and equity by 100
    res = IndiaFilingsClient.parse_ind_as_disclosure(_disclosure(facts))
    check = next(c for c in res.checks if c.name == "assets = liabilities + equity")
    assert not check.holds and check.difference == 100 * CR
    assert res.statement_model.is_balanced is False


def test_missing_items_are_listed_or_refused_never_zero() -> None:
    # Optional fields not given: 0.0 in the schema, listed in not_reported.
    service = {k: v for k, v in FACTS.items() if k not in (
        "CostOfMaterialsConsumed", "PurchasesOfStockInTrade",
        "ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade", "Inventories",
        "NonCurrentBorrowings", "CurrentBorrowings", "DeferredTax")}
    res = IndiaFilingsClient.parse_ind_as_disclosure(_disclosure(service))
    assert res.not_reported == ["cogs", "tax_expense", "total_debt", "inventory"]
    assert res.statement_model.income_statement.gross_profit == 10_000 * CR

    # Required items: the old parser returned 0.0 for each and cited it.
    for key in ("ProfitForThePeriod", "TotalExpenses", "CashAndCashEquivalentsAtBeginningOfPeriod"):
        with pytest.raises(ValueError, match="does not give"):
            IndiaFilingsClient.parse_ind_as_disclosure(_disclosure({k: v for k, v in FACTS.items() if k != key}))
