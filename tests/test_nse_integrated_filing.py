"""NSE Integrated Filing (Financial), Ind AS: the exchange's rendered page read into the Ind AS crosswalk.

Committed tests use a small synthetic page in the same layout (fictional figures). The Maruti Suzuki test
reads the real page (NSE, filed 2026-04-28) from git-ignored ``data/raw/india_xbrl/``, found by its
SHA-256, and is skipped when the file is absent: NSE's terms do not yet allow keeping it in the repo.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from lilavati.connectors.india_filings import IndiaFilingsClient
from lilavati.connectors.nse_integrated_filing import (
    citation,
    read_nse_integrated_filing,
    register_filing,
    to_ind_as_disclosure,
)
from lilavati.core.provenance import SourceStore

ROOT = Path(__file__).resolve().parents[1]
CIN = "U00000DL2000PLC000000"


def page(rows: list[list[str]], general: dict[str, str] | None = None) -> bytes:
    info = {"Scrip Code": "999999", "NSE Symbol": "DEMO", "ISIN": "INE000A01000", "Name of company": "Demo Limited",
            "Date of start of financial year": "01-04-2025", "Date of end of financial year": "31-03-2026",
            "Date of board meeting when results were approved": "28-04-2026",
            "Description of presentation currency": "INR", "Level of rounding used in financial results": "Lakhs",
            "Nature of report standalone or consolidated": "Consolidated"}
    info.update(general or {})
    trs = [f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info.items()]
    trs += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return ("<html><head><title>IndAS Details</title><script>var x = '<tr><td>Total assets</td></tr>';</script>"
            "</head><body><table>" + "".join(trs) + "</table></body></html>").encode()


PL = [["Particulars", "3 months/ 6 months ended (dd-mm-yyyy)", "Year to date figures for current period ended (dd-mm-yyyy)"],
      ["A", "Date of start of reporting period", "01-01-2026", "01-04-2025"],
      ["B", "Date of end of reporting period", "31-03-2026", "31-03-2026"],
      ["D", "Nature of report standalone or consolidated", "Consolidated", "Consolidated"],
      ["1", "Income"],
      ["", "Revenue from operations", "2,50,000.00", "10,00,000.00"],
      ["", "Other income", "5,000.00", "20,000.00"],
      ["(a)", "Cost of materials consumed", "1,50,000.00", "6,00,000.00"],
      ["(c)", "Changes in inventories of finished goods, work-in-progress and stock-in-trade", "(2,000.00)", "(8,000.00)"],
      ["(d)", "Employee benefit expense", "20,000.00", "80,000.00"],
      ["(e)", "Finance costs", "1,000.00", "4,000.00"],
      ["(f)", "Depreciation, depletion and amortisation expense", "9,000.00", "36,000.00"],
      ["", "Total expenses", "2,00,000.00", "8,00,000.00"],
      ["5", "Total profit before tax", "55,000.00", "2,20,000.00"],
      ["9", "Total tax expenses", "14,000.00", "56,000.00"],
      ["16", "Total profit (loss) for period", "41,000.00", "1,64,000.00"]]
BS = [["Particulars", "Year ended (dd-mm-yyyy)"],
      ["Date of start of reporting period", "01-04-2025"], ["Date of end of reporting period", "31-03-2026"],
      ["Nature of report standalone or consolidated", "Consolidated"],
      ["", "Property, plant and equipment", "3,00,000.00"],
      ["", "Trade receivables, non-current", "1,000.00"],
      ["", "Inventories", "50,000.00"],
      ["", "Trade receivables, current", "40,000.00"],
      ["", "Cash and cash equivalents", "9,000.00"],
      ["", "Total assets", "6,00,000.00"],
      ["", "Equity share capital", "10,000.00"],
      ["", "Other equity", "3,90,000.00"],
      ["", "Total equity", "4,00,000.00"],
      ["", "Trade payables, non-current"], ["", "Total Trade payable", "0.00"],
      ["", "Trade payables, current"],
      ["", "(A) Total outstanding dues of micro enterprises and small enterprises", "5,000.00"],
      ["", "Total Trade payable", "95,000.00"],
      ["", "Total liabilities", "2,00,000.00"],
      ["", "Total equity and liabilites", "6,00,000.00"]]
SEGMENTS = [["Particulars", "3 months", "Year to date"],
            ["Date of start of reporting period", "01-01-2026", "01-04-2025"],
            ["Date of end of reporting period", "31-03-2026", "31-03-2026"],
            ["", "Revenue from operations", "", ""]]  # the segment section repeats this label, empty
CF = [["Particulars", "Year ended (dd-mm-yyyy)"],
      ["Date of start of reporting period", "01-04-2025"], ["Date of end of reporting period", "31-03-2026"],
      ["", "Cash flows from used in operating activities"],
      ["", "Interest paid", "0.00", "", "Income taxes paid (refund)", "50,000.00"],  # two items in one row
      ["", "Net cash flows from (used in) operating activities", "2,40,000.00"],
      ["", "Cash flows from used in investing activities"],
      ["", "Interest paid", "0.00"],
      ["", "Net cash flows from (used in) investing activities", "(1,80,000.00)"],
      ["", "Net cash flows from (used in) financing activities", "(62,000.00)"],
      ["", "Effect of exchange rate changes on cash and cash equivalents"],
      ["", "Effect of exchange rate changes on cash and cash equivalents", "0.00"],
      ["", "Cash and cash equivalents cash flow statement at beginning of period", "11,000.00"],
      ["", "Cash and cash equivalents cash flow statement at end of period", "9,000.00"]]
NAME = "INTEGRATED_FILING_INDAS_100000_28042026194740_iXBRL_WEB.html"


def test_reads_general_information_and_sections() -> None:
    f = read_nse_integrated_filing(page(PL + BS + SEGMENTS + CF), source_name=NAME)
    assert (f.company_name, f.symbol, f.nature, f.rounding, f.scale) == ("Demo Limited", "DEMO", "Consolidated", "Lakhs", 100_000)
    assert (f.financial_year_start, f.financial_year_end) == (date(2025, 4, 1), date(2026, 3, 31))
    assert f.filed_at == datetime(2026, 4, 28, 19, 47, 40)
    assert [s.kind for s in f.sections] == ["profit_and_loss", "balance_sheet", "other", "cash_flow"]
    assert f.doc_hash == hashlib.sha256(page(PL + BS + SEGMENTS + CF)).hexdigest()
    assert "Total assets" not in f.general  # script content ignored


def test_year_disclosure_values_sums_and_checks() -> None:
    f = read_nse_integrated_filing(page(PL + BS + SEGMENTS + CF), source_name=NAME)
    r = to_ind_as_disclosure(f, cin=CIN)
    facts = r.disclosure.facts
    assert facts["RevenueFromOperations"] == 1_000_000  # lakhs, Indian digit grouping
    assert facts["ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade"] == -8_000  # parentheses
    assert facts["NetCashFromInvestingActivities"] == -180_000
    assert facts["TradeReceivables"] == 41_000 and facts["TradePayables"] == 95_000  # current + non-current
    assert facts["TotalEquityAndLiabilities"] == 600_000  # NSE's misspelt label
    assert facts["EffectOfExchangeRateChangesOnCash"] == 0.0  # the row with a value, not the heading
    assert r.disclosure.scale == 100_000 and r.disclosure.filing_date == date(2026, 4, 28)
    assert "PurchasesOfStockInTrade" in r.missing and "PurchasesOfStockInTrade" not in facts  # not filled
    assert set(r.derived) == {"TradeReceivables", "TradePayables"}
    checks = {c.name: c.holds for c in IndiaFilingsClient.parse_ind_as_disclosure(r.disclosure).checks}
    assert checks == {"assets = liabilities + equity": True, "assets = total equity and liabilities": True,
                      "opening cash + cash flows = closing cash": True}


def test_quarter_takes_the_quarter_column_and_no_balance_sheet() -> None:
    f = read_nse_integrated_filing(page(PL + BS + CF), source_name=NAME)
    r = to_ind_as_disclosure(f, cin=CIN, period="quarter")
    assert r.disclosure.facts["RevenueFromOperations"] == 250_000 and r.period_start == date(2026, 1, 1)
    assert "TotalAssets" in r.missing and "NetCashFromOperatingActivities" in r.missing


def test_citations_resolve_against_the_page() -> None:
    content = page(PL + BS + CF)
    f = read_nse_integrated_filing(content, source_name=NAME)
    r = to_ind_as_disclosure(f, cin=CIN)
    store = SourceStore()
    register_filing(store, f, content)
    assert all(store.check(c) == [] for key in r.sources for c in citation(f, r, key))
    with pytest.raises(ValueError, match="not the page"):
        register_filing(SourceStore(), f, content + b" ")


def test_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="unknown rounding"):
        read_nse_integrated_filing(page(PL, {"Level of rounding used in financial results": "Hundreds"}))
    with pytest.raises(ValueError, match="Integrated Filing"):
        read_nse_integrated_filing(b"<html><table><tr><td>Particulars</td><td>x</td></tr></table></html>")
    f = read_nse_integrated_filing(page(PL + BS + CF))
    with pytest.raises(ValueError, match="filing_date"):
        to_ind_as_disclosure(f, cin=CIN)
    dup = PL + [["", "Revenue from operations", "1.00", "2.00"]]
    with pytest.raises(ValueError, match="2 rows"):
        to_ind_as_disclosure(read_nse_integrated_filing(page(dup + BS + CF)), cin=CIN, filing_date=date(2026, 4, 28))
    standalone = [r if r[:2] != ["D", "Nature of report standalone or consolidated"] else
                  ["D", "Nature of report standalone or consolidated", "Standalone", "Standalone"] for r in PL]
    with pytest.raises(ValueError, match="Standalone"):
        to_ind_as_disclosure(read_nse_integrated_filing(page(standalone + BS + CF)), cin=CIN,
                             filing_date=date(2026, 4, 28))


# -------------------------------------------------------------------------------------------------
# Real filing: Maruti Suzuki India, consolidated, year ended 2026-03-31 (local file, git-ignored)
# -------------------------------------------------------------------------------------------------

MARUTI_SHA256 = "cf868659b98ca2a1b8d5d1567c9348b36d36b1d4aca8d743a44031e82596af60"


def _maruti_page() -> bytes | None:
    for p in sorted((ROOT / "data" / "raw" / "india_xbrl").rglob("*.htm*")):
        content = p.read_bytes()
        if hashlib.sha256(content).hexdigest() == MARUTI_SHA256:
            return content
    return None


@pytest.mark.skipif(_maruti_page() is None, reason="NSE page not in data/raw/india_xbrl (git-ignored)")
def test_maruti_exchange_filing_against_printed_statements() -> None:
    content = _maruti_page()
    assert content is not None
    fixture = json.loads((ROOT / "tests" / "fixtures" / "maruti_indas_consolidated_fy2026.json").read_text())
    f = read_nse_integrated_filing(content)
    assert (f.company_name, f.symbol, f.isin, f.nature, f.rounding) == (
        "Maruti Suzuki India Limited", "MARUTI", "INE585B01010", "Consolidated", "Lakhs")
    r = to_ind_as_disclosure(f, cin=fixture["cin"], filing_date=date(2026, 4, 28))
    assert r.missing == []
    # 1. the 26 values read from this page by hand earlier (fixture "exchange_filing_lakhs") match exactly
    assert {k: r.disclosure.facts[k] for k in fixture["exchange_filing_lakhs"]} == fixture["exchange_filing_lakhs"]
    # 2. against the printed statements (INR million, typed from the PDF): equal except the two
    #    definitional differences the fixture notes: profit before tax (printed includes the share of
    #    associates 2,591 and joint ventures 264) and PP&E (the exchange includes right-of-use assets 5,998)
    diff = {k: round(r.disclosure.facts[k] / 10 - v, 1) for k, v in fixture["facts"].items()}
    assert {k: d for k, d in diff.items() if d} == {"ProfitBeforeTax": -2_855.0, "PropertyPlantAndEquipment": 5_998.0}
    checks = IndiaFilingsClient.parse_ind_as_disclosure(r.disclosure).checks
    assert all(c.holds for c in checks) and len(checks) == 3
    store = SourceStore()
    register_filing(store, f, content)
    assert all(store.check(c) == [] for key in r.sources for c in citation(f, r, key))
