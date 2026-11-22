"""Filing-anchored XBRL extraction (lilavati.connectors.sec_filing), on verbatim SEC data for Apple.

Expected values: Apple Inc.'s FY2026 Form 10-K statements (USD millions, as printed).
Fixtures: tests/fixtures/aapl_companyfacts_10k_fy2025_2026.json and aapl_submissions_subset.json
(verbatim subsets of SEC responses; source URLs inside).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilavati.connectors.sec_filing import CANONICAL_ITEMS, extract_filing, filings_from_submissions

FIX = Path(__file__).parent / "fixtures"
FACTS = json.loads((FIX / "aapl_companyfacts_10k_fy2025_2026.json").read_text())
SUBS = json.loads((FIX / "aapl_submissions_subset.json").read_text())


def _latest_10k():
    return filings_from_submissions(SUBS, ("10-K",))[0]


def test_latest_10k_is_identified_from_the_submissions_index() -> None:
    f = _latest_10k()
    assert (f.form, f.accession, str(f.report_date)) == ("10-K", "0000320193-26-000026", "2026-09-26")
    assert f.url.endswith("/000032019326000026/aapl-20260926.htm")


def test_values_are_the_filings_own_period() -> None:
    x = extract_filing(FACTS, _latest_10k())
    millions = {k: round(v.value / 1e6) for k, v in x.values.items()}
    assert millions["revenue"] == 478_834
    assert millions["cost_of_revenue"] == 245_232
    assert millions["income_tax"] == 27_062
    assert millions["net_income"] == 132_175
    assert millions["total_assets"] == 404_548
    assert millions["total_liabilities"] == 292_455
    assert millions["long_term_debt_noncurrent"] == 68_208
    for sv in x.values.values():
        assert str(sv.citation.period_end) == "2026-09-26"
        assert "0000320193-26-000026" in sv.citation.raw_snippet


def test_missing_items_are_listed_not_zero_filled() -> None:
    x = extract_filing(FACTS, _latest_10k())
    assert x.missing == ["total_equity_incl_nci"]  # Apple reports no non-controlling interests
    assert set(x.values) | set(x.missing) == set(CANONICAL_ITEMS)


def test_identities_hold_on_the_reported_numbers() -> None:
    x = extract_filing(FACTS, _latest_10k())
    assert [c.name for c in x.checks] == [
        "assets = liabilities + equity", "assets = total liabilities and equity",
        "revenue - cost of revenue = gross profit"]
    assert all(c.holds and c.difference == 0.0 for c in x.checks)


def test_comparatives_in_the_same_filing_are_not_selected() -> None:
    # The FY2026 10-K carries FY2025 balances (end 2025-09-27) under its own accession number;
    # they are in the data, and extraction must still return the filing's own period.
    f = _latest_10k()
    comparatives = [e for e in FACTS["facts"]["us-gaap"]["Assets"]["units"]["USD"]
                    if e["accn"] == f.accession and e["end"] != str(f.report_date)]
    assert comparatives and comparatives[0]["end"] == "2025-09-27"
    assert round(extract_filing(FACTS, f).values["total_assets"].value / 1e6) == 404_548  # not 359,241


def test_a_failed_identity_is_reported() -> None:
    from lilavati.connectors.sec_filing import check_identities

    x = extract_filing(FACTS, _latest_10k())
    broken = dict(x.values)
    broken["gross_profit"] = broken["gross_profit"].model_copy(update={"value": broken["gross_profit"].value + 1e6})
    gp = next(c for c in check_identities(broken) if c.name.startswith("revenue"))
    assert not gp.holds and gp.difference == pytest.approx(-1e6)


def test_ratios_from_the_filing() -> None:
    from lilavati.connectors.sec_filing import filing_ratios

    x = extract_filing(FACTS, _latest_10k())
    r = filing_ratios({k: v.value / 1e6 for k, v in x.values.items()})
    # Apple FY2026, USD millions: (478,834 - 245,232) / 478,834; 132,175 / 478,834; 132,175 / 404,548;
    # 145,097 - 9,041; 292,455 / 404,548
    assert r["gross_margin"] == pytest.approx(233_602 / 478_834)
    assert r["net_margin"] == pytest.approx(132_175 / 478_834)
    assert r["return_on_assets"] == pytest.approx(132_175 / 404_548)
    assert r["free_cash_flow"] == pytest.approx(136_056)
    assert r["liabilities_to_assets"] == pytest.approx(292_455 / 404_548)
    assert filing_ratios({"revenue": 0.0, "net_income": 1.0}) == {}  # no division by zero


def test_statements_window_skips_the_table_of_contents() -> None:
    from lilavati.connectors.sec_filing import statements_window

    # Synthetic text: a contents line, then the statement with figures.
    text = ("Index Consolidated Statements of Operations 35 Balance Sheets 37 " + "x " * 4000 +
            "CONSOLIDATED STATEMENTS OF OPERATIONS Total net sales 478,834 416,161 391,035 "
            "Net income 132,175 112,010 93,736")
    w = statements_window(text, chars=200)
    assert w.startswith("CONSOLIDATED STATEMENTS OF OPERATIONS") and "478,834" in w
    with pytest.raises(ValueError):
        statements_window("no statements here")
    # Synthetic text, ordered as in J&J's 10-K (balance sheet before income statement); figures made up.
    text2 = ("x " * 4000 + "CONSOLIDATED BALANCE SHEETS Cash 1,111 1,000 Total assets 2,222 2,000 "
             "Total liabilities 1,234 1,100 " + "y " * 100 +
             "CONSOLIDATED STATEMENTS OF EARNINGS Sales 3,333 3,000 Net earnings 4,444 4,000 Tax 5,555")
    assert statements_window(text2, chars=400).startswith("CONSOLIDATED BALANCE SHEETS")


def test_inline_tags_do_not_split_words() -> None:
    # Microsoft's 10-K renders "INCOME" as two spans; replacing every tag with a space produced
    # "IN COME STATEMENTS", so the income statement heading was never found.
    from lilavati.connectors.sec_filing import document_text

    raw = ('<div><span style="x">IN</span><span>COME STATEMENTS</span></div><table><tr>'
           '<td>Revenue</td><td>$ <ix:nonFraction name="us-gaap:Revenues">318,034</ix:nonFraction></td>'
           '<td>261,802</td></tr></table>')
    text = document_text(raw)
    assert "INCOME STATEMENTS" in text
    assert "318,034" in text and "318,034 261,802" in text  # table cells stay separate
