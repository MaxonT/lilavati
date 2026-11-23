# Validation: filing connectors (`connectors/pdf_10k.py`, `connectors/edgar.py`)

Started 2026-11-22 (roadmap 7.5, live-data validation). Tests: `tests/test_pdf_10k.py` (row parsing on
excerpts of real statement lines; a full-filing check that runs when the PDF is present) and
`tests/test_edgar_connector.py`.

## 10-K PDF parser on Apple Inc. FY2026 Form 10-K

Filing: 123 pages, SHA-256 `f9a3e5b32cb7…` (not redistributed; see the FAQ). Reference: the values
printed in the filing's own statements (income statement p. 35, balance sheet p. 37, cash flows
p. 39), read line by line. USD millions.

| Line item | Before fix | Filing | After fix | Status |
| :--- | ---: | ---: | ---: | :--- |
| Provision for income taxes (FY2026) | 159,237 | 27,062 | 27,062 | **fixed**: matched "Income before provision for income taxes" |
| Total debt (commercial paper + current and non-current term debt) | 11,903 | 80,111 | 80,111 | **fixed**: non-current term debt was dropped |
| Common stock and additional paid-in capital | 0.00001 | 104,815 | 104,815 | **fixed**: parsed the par value from the wrapped label |
| Interest expense | 0.0 | not a separate line | 0.0, listed in `missing_items` | **fixed**: a missing item no longer looks reported |
| Cost of sales | 245,232 | 245,232 | 245,232 | pass (a regression to 215,874 during the fix was caught by a test) |
| Revenue, gross margin, operating income, net income | | 478,834 / 233,602 / 158,063 / 132,175 | same | pass |
| Total assets, total liabilities, shareholders' equity | | 404,548 / 292,455 / 112,093 | same | pass |
| Cash, retained earnings, cash from operations | | 36,305 / 13,700 / 145,097 | same | pass |
| FY2025 comparatives (same items) | | as printed | match | pass |
| A = L + E and cash roll-forward (Z3, on the extracted numbers) | | | hold | pass |

Open items: one issuer and one filing only (the roadmap asks for at least two of each); other
`or 0.0` fallbacks remain for items not listed in `missing_items`; footnote markers such as "(1)"
after a label would be read as −1.

## SEC EDGAR XBRL connector

- **fixed:** the client sent a made-up default contact (`admin@lilavati.ai`) in its User-Agent; SEC
  requires a real one (HTTP 403 without it). The contact now comes from `LILAVATI_SEC_USER_AGENT`
  or the `user_agent` argument, checked when a request is made.
- **fixed:** `EdgarClient.extract_line_item` returned comparative periods. Company facts tag every
  value in a 10-K with that filing's fiscal year, including the prior years it repeats, and the
  client picked one by filing date. For Apple "FY2026" it returned FY2025 total assets (359,241
  instead of 404,548) and FY2024 revenue (391,035 instead of 478,834) and net income (93,736
  instead of 132,175). It now takes the filing's own period (latest end date; full-year durations
  for annual flow items). Test: `tests/test_edgar_connector.py`, on verbatim SEC data.

## Filing-anchored extraction (`connectors/sec_filing.py`), five-company pilot

The flow: identify the latest 10-K from the SEC submissions index; for each canonical line item
take only facts with that filing's accession number, its report date as the end date and (for flow
items) a duration of about a year; list items not found as missing; check identities. Validation
(`scripts/validate_sec_filings.py`, 2026-11-22): every extracted value is searched for in the
filing document itself at the filing's scale. A large figure appearing in the document is strong
evidence that the right fact was taken; a small one could appear by coincidence.

| Company | Period end | Items extracted | Found in the filing document | Identities | Not reported as a tagged line |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Apple | 2026-09-26 | 16 | 16 | 3 of 3 hold | equity incl. non-controlling interests |
| Microsoft | 2026-06-30 | 16 | 16 | 3 of 3 hold | equity incl. non-controlling interests |
| Walmart | 2026-01-31 | 15 | 15 | 1 of 1 hold | gross profit, total liabilities |
| Johnson & Johnson | 2025-12-28 | 15 | 15 | 3 of 3 hold | operating income, equity excl. non-controlling interests |
| JPMorgan Chase | 2025-12-31 | 9 | 9 | 2 of 2 hold | cost of revenue, gross profit, operating income, capital expenditure, cash, term debt (bank statements differ) |

Revenue is total revenues as reported: the `Revenues` tag first, then contract revenue (Walmart tags net
sales, 706,413, as contract revenue and total revenues including membership income, 713,163, as
`Revenues`; found when all four models in Tutorial 08 "missed" Walmart's revenue the same way).
Raw output: `results/connectors/sec_filings_pilot.json`. LLM comparison on the same filings:
[Tutorial 08](../tutorials/08_real_10k_llm_vs_lilavati.md), raw replies in `results/filings/`. The same flow is
the MCP tool `get_filing_figures` (`lilavati-mcp-corporate-filings`), which also resolves any listed
ticker through SEC's company list.

Open items:

- Banks and insurers need their own templates.
- Extension (company-specific) tags are reported as missing, not mapped.
- 10-Q filings are supported but not validated; their cash-flow items are year-to-date, not
  quarterly, so the quarterly duration rule lists them as missing (seen on Apple's 10-Q for the
  quarter ended 2026-06-27).
- 10-K/A and restated comparatives are not covered.
- The Ind AS crosswalk (`IndiaFilingsClient.parse_ind_as_disclosure`) fills missing keys with 0.0 and
  cites them, and sets `cash_reconciled=True` without a check. Not validated; do not rely on it.
- Five companies is a pilot, not a validation set.

**Three-statement model** (`EdgarClient.extract_three_statement_model`, MCP tool
`get_company_filing_statements`). It used to select facts by fiscal year and fill what it did not
find: inventory 0, missing interest or tax 0, retained earnings set to net income, operating income
set to net income when untagged, and opening cash back-solved from the cash flows so that
`cash_reconciled` was always `True`. On Apple's FY2026 10-K it returned retained earnings 132,175
(net income; the balance sheet says 13,700), common equity −20,082 (104,815), inventory 0 (14,541)
and total debt 68,208 (non-current term debt only; 80,111 with the current portion and commercial
paper); J&J's operating income came back as its net income (26,804).

It now identifies the fiscal year's 10-K from the company facts (`sec_filing.filing_for_fiscal_year`;
the same accession as SEC's submissions index for all 30 fiscal years of the five companies that the
index lists) and uses only that filing's facts. Each field is reported and cited, derived from
reported items by a formula recorded in `derived` (for example Walmart's total liabilities: total
liabilities and equity minus total equity), or, for receivables, inventory, payables, interest and
tax, listed in `not_reported`. When a required item is not reported it raises and names it. The cash
check is real: opening cash (the prior period-end comparative in the same filing) plus operating,
investing and financing cash flow plus the exchange-rate effect must equal closing cash. Balance
checks use reported totals only, so a derived total cannot confirm itself.

| Company | Fiscal year | Reported values found in the 10-K | Derived | Not reported | Checks |
| :--- | :--- | ---: | :--- | :--- | :--- |
| Apple | 2026 | 28 of 28 | total debt, operating expenses, net change in cash | interest expense | 4 of 4 hold |
| Microsoft | 2026 | 29 of 29 | operating expenses, net change in cash | none | 4 of 4 hold |
| Walmart | 2026 | 29 of 29 | total liabilities, gross profit, common equity, total debt, operating expenses, net change in cash | interest expense | 2 of 2 hold |
| Johnson & Johnson | 2025 | refused: no operating income, no paid-in capital line | | | |
| JPMorgan Chase | 2025 | refused: no operating income, cash, net PP&E or cost of revenue (bank) | | | |

"Operating expenses" fills the schema's `sga_expense` (gross profit minus operating income): it
includes R&D and is not the reported SG&A line. Source: `scripts/validate_sec_filings.py`, section
`statement_model` of `results/connectors/sec_filings_pilot.json`.

**Citations.** An XBRL fact's citation carries the SHA-256 of the fact record as SEC serves it
(canonical JSON, `sec_filing.fact_hash`): re-fetch the company facts, find the record and hash it to
check. XBRL facts have no page, so `page` is `None` (it used to be a placeholder 1).
