"""
Unit tests for the SEC 10-K PDF Connector and Document Decomposition Engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilavati.connectors.pdf_10k import PDF10KParser, SectionBoundaries


def test_pdf_10k_parser_local() -> None:
    sample_path = Path("data/sample_filings/aapl_10k_2026.pdf")
    if not sample_path.exists():
        pytest.skip("Sample 10-K PDF not present in offline test runner")

    parser = PDF10KParser(str(sample_path))
    assert len(parser.doc_hash) == 64
    assert parser.page_count > 50

    sections: SectionBoundaries = parser.locate_sections()
    assert sections.business_page is not None
    assert sections.income_statement_page is not None
    assert sections.balance_sheet_page is not None
    assert sections.cash_flows_page is not None

    filing = parser.extract_full_filing()
    assert filing.company_name == "Apple Inc."
    assert filing.model_current_year.income_statement.revenue > 400000.0
    assert filing.model_current_year.balance_sheet.total_assets > 350000.0
    assert filing.model_current_year.cash_flow_statement.cfo > 100000.0

    # Verify formal SMT verification holds
    assert filing.is_smt_verified is True

    # Verify provenance citations
    assert "curr_revenue" in filing.citations
    assert "curr_total_assets" in filing.citations
    assert filing.citations["curr_revenue"].page == sections.income_statement_page
    assert filing.citations["curr_revenue"].doc_hash == parser.doc_hash
    assert len(filing.citations["curr_revenue"].raw_snippet) > 5


# Lines as pypdf extracts them from Apple Inc.'s FY2026 Form 10-K (statements, pages 35 and 37).
APPLE_IS_LINES = [
    "Total net sales  478,834  416,161  391,035 ",
    "Income before provision for income taxes  159,237  132,729  123,485 ",
    "Provision for income taxes  27,062  20,719  29,749 ",
    "Net income $ 132,175 $ 112,010 $ 93,736 ",
]
APPLE_BS_LINES = [
    "Commercial paper  1,997  7,979 ",
    "Term debt  9,906  12,350 ",
    "Total current liabilities  169,915  165,631 ",
    "Non-current liabilities:",
    "Term debt  68,208  78,328 ",
    "Common stock and additional paid-in capital, $0.00001 par value: 50,400,000 shares ",
    "authorized; 14,532,259 and 14,773,260 shares issued and outstanding, respectively  104,815  93,568 ",
    "Retained earnings/(Accumulated deficit)  13,700  (14,264) ",
]


def test_label_at_line_start_wins_over_substring() -> None:
    # "Income before provision for income taxes" contains "provision for income taxes"; the
    # parser used to report pretax income (159,237) as the tax expense.
    from lilavati.connectors.pdf_10k import PDF10KParser

    val, snippet = PDF10KParser._find_in_lines(APPLE_IS_LINES, ["Provision for income taxes"], 0)
    assert val == 27062.0 and snippet.startswith("Provision for income taxes")


def test_wrapped_label_takes_the_row_amounts_not_the_par_value() -> None:
    from lilavati.connectors.pdf_10k import PDF10KParser

    cur, _ = PDF10KParser._find_in_lines(APPLE_BS_LINES, ["Common stock and additional paid-in capital"], 0)
    prior, _ = PDF10KParser._find_in_lines(APPLE_BS_LINES, ["Common stock and additional paid-in capital"], 1)
    assert (cur, prior) == (104815.0, 93568.0)  # not 0.00001 (the par value)


def test_negative_amounts_in_parentheses() -> None:
    from lilavati.connectors.pdf_10k import PDF10KParser

    assert PDF10KParser._find_in_lines(APPLE_BS_LINES, ["Retained earnings"], 1)[0] == -14264.0


def test_term_debt_sums_current_and_non_current_rows() -> None:
    from lilavati.connectors.pdf_10k import PDF10KParser

    assert PDF10KParser._sum_rows(APPLE_BS_LINES, "Term debt", 0) == 9906.0 + 68208.0
    assert PDF10KParser._sum_rows(APPLE_BS_LINES, "Term debt", 1) == 12350.0 + 78328.0


def test_real_apple_10k_if_present() -> None:
    # Opt-in: the filing is not redistributed. Values checked against the filing's statements.
    from pathlib import Path

    import pytest

    from lilavati.connectors.pdf_10k import PDF10KParser

    pdf = Path(__file__).resolve().parents[1] / "data" / "sample_filings" / "aapl_10k_2026.pdf"
    if not pdf.exists():
        pytest.skip("Apple FY2026 10-K PDF not present (not redistributed)")
    f = PDF10KParser(str(pdf)).extract_full_filing()
    inc, bs = f.model_current_year.income_statement, f.model_current_year.balance_sheet
    assert inc.tax_expense == 27062.0
    assert bs.total_debt == 1997.0 + 9906.0 + 68208.0
    assert bs.common_equity == 104815.0
    assert "interest_expense" in f.missing_items  # not a separate line in Apple's statements


def test_section_header_is_not_a_row_and_keywords_keep_priority() -> None:
    # "Cost of sales:" is a header; joining it with the next line returned the Products row
    # (215,874) instead of "Total cost of sales" (245,232), the first keyword in priority order.
    from lilavati.connectors.pdf_10k import PDF10KParser

    lines = [
        "Total net sales  478,834  416,161  391,035 ",
        "Cost of sales:",
        "   Products  215,874  194,116  185,233 ",
        "   Services  29,358  26,844  25,119 ",
        "Total cost of sales  245,232  220,960  210,352 ",
    ]
    val, _ = PDF10KParser._find_in_lines(lines, ["Total cost of sales", "Cost of sales"], 0)
    assert val == 245232.0
    assert PDF10KParser._find_in_lines(["Cost of sales:", "   Products  215,874 "], ["Cost of sales"], 0) == (None, None)


# Lines as pypdf extracts them from Apple Inc.'s FY2026 Form 10-K (balance sheet page 37, cash flows
# page 39); the cash balance labels wrap onto a second line.
APPLE_BS_TOTALS = [
    "Total assets $ 404,548 $ 359,241 ",
    "Total liabilities  292,455  285,508 ",
    "Total shareholders’ equity  112,093  73,733 ",
    "Total liabilities and shareholders’ equity $ 404,548 $ 359,241 ",
]
APPLE_CF_LINES = [
    "Cash, cash equivalents, and restricted cash and cash equivalents, ",
    "beginning balances $ 35,934 $ 29,943 $ 30,737 ",
    "Cash generated by operating activities  145,097  111,482  118,254 ",
    "Payments for acquisition of property, plant and equipment  (9,041)  (12,715)  (9,447) ",
    "Cash generated by/(used in) investing activities  (17,280)  15,195  2,935 ",
    "Cash used in financing activities  (127,446)  (120,686)  (121,983) ",
    "Cash, cash equivalents, and restricted cash and cash equivalents, ending ",
    "balances $ 36,305 $ 35,934 $ 29,943 ",
]


def _stub_parser(bs: list[str], cf: list[str]):
    from lilavati.connectors.pdf_10k import PDF10KParser

    class Stub(PDF10KParser):
        def __init__(self) -> None:  # no PDF: serve the statement pages' lines
            self.doc_hash, self.page_count = "0" * 64, 40
            self.pages = {35: APPLE_IS_LINES, 37: APPLE_BS_LINES + bs, 39: cf}

        def locate_sections(self) -> SectionBoundaries:
            return SectionBoundaries(income_statement_page=35, balance_sheet_page=37, cash_flows_page=39)

        def _page_lines(self, page_num: int) -> list[str]:
            return self.pages.get(page_num, [])

    return Stub()


def test_wrapped_cash_balance_labels() -> None:
    from lilavati.connectors.pdf_10k import PDF10KParser

    begin, end = PDF10KParser.CASH_BEGIN_LABELS, PDF10KParser.CASH_END_LABELS
    assert [PDF10KParser._find_wrapped(APPLE_CF_LINES, begin, c) for c in (0, 1)] == [35_934, 29_943]
    assert [PDF10KParser._find_wrapped(APPLE_CF_LINES, end, c) for c in (0, 1)] == [36_305, 35_934]


def test_cash_and_balance_checks_use_the_statements() -> None:
    # Apple FY2026: 35,934 + 145,097 - 17,280 - 127,446 = 36,305; FY2025: 29,943 + 111,482 + 15,195
    # - 120,686 = 35,934. Assets 404,548 = 292,455 + 112,093.
    f = _stub_parser(APPLE_BS_TOTALS, APPLE_CF_LINES).extract_full_filing()
    for m in (f.model_current_year, f.model_prior_year):
        assert m.is_balanced and m.cash_reconciled
    assert f.is_smt_verified and len(f.checks) == 4 and all(c.difference == 0 for c in f.checks)


def test_cash_reconciliation_is_not_assumed() -> None:
    # The parser used to set cash_reconciled=True whatever the numbers.
    cf = [line.replace("145,097", "145,098") for line in APPLE_CF_LINES]
    f = _stub_parser(APPLE_BS_TOTALS, cf).extract_full_filing()
    assert f.model_current_year.cash_reconciled is False and f.is_smt_verified is False
    assert f.model_prior_year.cash_reconciled is True


def test_missing_items_are_not_substituted() -> None:
    # Equity used to become assets minus liabilities (so the balance check held by construction) and
    # a missing financing total became minus buybacks minus dividends.
    bs = [line for line in APPLE_BS_TOTALS if not line.startswith("Total shareholders")]
    cf = [line for line in APPLE_CF_LINES if "financing" not in line]
    f = _stub_parser(bs, cf).extract_full_filing()
    assert {"total_equity", "cff"} <= set(f.missing_items)
    assert f.model_current_year.is_balanced is False and f.model_current_year.cash_reconciled is False
    assert f.model_current_year.balance_sheet.total_equity == 0.0  # listed as missing, not derived
    assert f.model_current_year.balance_sheet.common_equity == 104_815  # reported line, not total equity
