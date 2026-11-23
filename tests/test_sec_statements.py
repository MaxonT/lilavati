"""Three-statement model from one 10-K's own XBRL facts: reported, derived or listed, never filled.

Fixtures are verbatim subsets of SEC company facts (one accession each; see their ``_source``).
Expected values are from the filings, in USD millions; every reported value used here also appears
in the 10-K document text (checked with ``scripts/validate_sec_filings.appears``).
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from lilavati.connectors.edgar import EdgarClient
from lilavati.connectors.sec_filing import filing_for_fiscal_year, filings_from_submissions

FIX = Path(__file__).parent / "fixtures"
M = 1e6


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIX / name).read_text())


def test_fiscal_year_filing_matches_the_submissions_index() -> None:
    # The 10-K is identified from company facts alone; it must be the filing SEC's index lists.
    facts = _load("aapl_companyfacts_10k_fy2025_2026.json")
    index = {f.accession: f for f in filings_from_submissions(_load("aapl_submissions_subset.json"))}
    for fy, accession in ((2026, "0000320193-26-000026"), (2025, "0000320193-25-000079")):
        f = filing_for_fiscal_year(facts, fy)
        assert f.accession == accession
        assert (f.report_date, f.filing_date) == (index[accession].report_date, index[accession].filing_date)
    assert filing_for_fiscal_year(facts, 2026).url.endswith("/000032019326000026/")
    with pytest.raises(ValueError, match="fiscal year 2019"):
        filing_for_fiscal_year(facts, 2019)


def test_apple_fy2026_statement_model_is_reported_not_filled() -> None:
    m = EdgarClient().extract_three_statement_model("AAPL", 2026, facts=_load("aapl_companyfacts_statements_fy2026.json"))
    assert m.filing.accession == "0000320193-26-000026" and m.filing.report_date == date(2026, 9, 26)
    inc, bs, cf = (m.statement_model.income_statement, m.statement_model.balance_sheet,
                   m.statement_model.cash_flow_statement)
    assert (inc.revenue, inc.cogs, inc.gross_profit) == (478_834 * M, 245_232 * M, 233_602 * M)
    assert inc.operating_income == 158_063 * M and inc.net_income == 132_175 * M
    assert inc.sga_expense == 75_539 * M  # "Total operating expenses" in the 10-K
    assert inc.tax_expense == 27_062 * M
    # Apple's income statement has no interest expense line: listed, not passed off as reported.
    assert inc.interest_expense == 0.0 and m.not_reported == ["interest_expense"]
    # The old path set retained earnings to net income (132,175) and common equity to the remainder.
    assert bs.retained_earnings == 13_700 * M and bs.common_equity == 104_815 * M
    assert (bs.total_assets, bs.total_liabilities, bs.total_equity) == (404_548 * M, 292_455 * M, 112_093 * M)
    assert (bs.inventory, bs.accounts_receivable, bs.accounts_payable) == (14_541 * M, 43_817 * M, 82_252 * M)
    # Term debt 78,114 plus commercial paper 1,997; the formula is recorded.
    assert bs.total_debt == 80_111 * M and m.derived["total_debt"] == "long_term_debt + commercial_paper"
    assert (cf.cfo, cf.cfi, cf.cff, cf.net_change_in_cash) == (145_097 * M, -17_280 * M, -127_446 * M, 371 * M)

    checks = {c.name: c for c in m.checks}
    assert set(checks) == {"assets = liabilities + equity", "assets = total liabilities and equity",
                           "revenue - cost of revenue = gross profit", "opening cash + cash flows = closing cash"}
    assert all(c.holds and c.difference == 0 for c in m.checks)
    assert m.statement_model.is_balanced and m.statement_model.cash_reconciled and m.is_formally_verified
    opening = m.sourced_facts["cash_incl_restricted_opening"]
    assert opening.value == 35_934 * M and opening.citation.period_end == date(2025, 9, 27)
    assert all(len(sv.citation.doc_hash) == 64 and sv.citation.page is None for sv in m.sourced_facts.values())


def test_cash_reconciliation_is_checked_not_assumed() -> None:
    # The old path back-solved opening cash from the flows and set cash_reconciled=True regardless.
    facts = _load("aapl_companyfacts_statements_fy2026.json")
    bad = copy.deepcopy(facts)
    for e in bad["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"]:
        if e["end"] == "2026-09-26":
            e["val"] += 1_000_000
    m = EdgarClient().extract_three_statement_model("AAPL", 2026, facts=bad)
    check = next(c for c in m.checks if c.name.startswith("opening cash"))
    assert not check.holds and check.difference == -1_000_000
    assert m.statement_model.cash_reconciled is False and m.is_formally_verified is False


def test_walmart_derives_liabilities_and_does_not_check_its_own_derivation() -> None:
    m = EdgarClient().extract_three_statement_model(
        "0000104169", 2026, facts=_load("wmt_companyfacts_statements_fy2026.json"))
    bs, inc = m.statement_model.balance_sheet, m.statement_model.income_statement
    assert inc.revenue == 713_163 * M  # total revenues, not net sales 706,413
    # Walmart tags no total liabilities: 284,668 total liabilities and equity - 105,887 total equity
    # (including non-controlling interests).
    assert bs.total_equity == 105_887 * M and bs.total_liabilities == 178_781 * M
    assert m.derived["total_liabilities"] == "liabilities_and_equity - total_equity"
    # A = L + E would hold by construction, so only the check on two reported totals runs.
    assert [c.name for c in m.checks if "assets" in c.name] == ["assets = total liabilities and equity"]
    assert bs.accounts_receivable == 11_172 * M and m.tags_used["accounts_receivable"] == "ReceivablesNetCurrent"
    assert m.derived["gross_profit"] == "revenue - cost_of_revenue" and inc.gross_profit == 177_768 * M
    assert bs.common_equity == (797 + 6_816) * M
    cash = next(c for c in m.checks if c.name.startswith("opening cash"))
    assert cash.holds and "not tagged" not in cash.detail  # exchange-rate effect 123 is tagged


def test_missing_operating_income_raises_instead_of_using_net_income() -> None:
    # J&J's income statement has no operating income line; the old path used net income for it.
    with pytest.raises(ValueError, match="operating_income"):
        EdgarClient().extract_three_statement_model(
            "0000200406", 2025, facts=_load("jnj_companyfacts_statements_fy2025.json"))
