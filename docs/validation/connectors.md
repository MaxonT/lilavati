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
| A = L + E (Z3) and cash reconciliation, both years | | | hold | pass |

The checks used to be assumed or circular: `cash_reconciled` was set to `True` for both years
whatever the numbers, the Z3 cash check started from the balance sheet's prior cash or 0.0, a
missing equity line became assets minus liabilities (so A = L + E held by construction), a missing
common equity line became total equity, and missing investing or financing totals became minus
capital expenditure or minus buybacks and dividends. Now nothing is substituted: an item not found is
0.0 and listed in `missing_items`, a check with a missing input is not run and its flag is `False`,
and the cash check uses the cash flow statement's own balances (cash including restricted cash):
FY2026 35,934 + 145,097 − 17,280 − 127,446 = 36,305; FY2025 29,943 + 111,482 + 15,195 − 120,686 =
35,934. Each check is in `ExtractedFiling10K.checks`.

Open items: one issuer and one filing only (the roadmap asks for at least two of each); labels are
Apple's (other filers word some lines differently); prior-year items not found are 0.0 but not listed
in `missing_items`; footnote markers such as "(1)" after a label would be read as −1.

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

## Ind AS crosswalk

`IndiaFilingsClient.parse_ind_as_disclosure` maps Schedule III (Division II) line items, supplied by
the caller, into the canonical model. It does not fetch filings.

**Real results: Maruti Suzuki India, consolidated, year ended 31 March 2026** (checked 2026-11-24).
Source: the annual results filed with NSE and BSE on 28 April 2026 (statement of profit and loss,
assets and liabilities, cash flows; pages 6–8, INR million), typed line by line from the printed,
signed pages (the PDF is a scan; its text layer is unusable). The crosswalk needed no change.

| Item | Printed figures | Crosswalk | Check |
| :--- | :--- | ---: | :--- |
| Cost of goods sold | materials 1,116,635 + purchases 231,845 − change in inventories 22,725 | 1,325,755 | pass |
| Operating income | revenue 1,833,160 − (total expenses 1,688,402 − finance costs 2,387) | 147,145 | pass |
| Other operating expenses | employee 90,497 + D&A 67,417 + other 205,233 − vehicles/dies for own use 2,887 | 360,260 | pass (= gross profit − operating income) |
| Assets = liabilities + equity | 1,488,810 = 417,247 + 1,071,563 | holds | pass (Z3) |
| Assets = total equity and liabilities | 1,488,810 | holds | pass |
| Opening cash + flows = closing cash | 1,845 + 190,999 − 147,335 − 44,840 = 669 | holds | pass |
| Borrowings | no line printed | listed in `not_reported` | pass |

Cross-check against the exchange's Ind AS filing of the same results (NSE integrated filing, INR
lakhs): every line agrees except property, plant and equipment, 340,195 in the filing against
334,197 printed. The difference, 5,998, is the right-of-use assets, a separate printed line that the
exchange format folds into PP&E. Its "total profit before tax" (188,330) also excludes the share of
profit of associates and joint ventures (2,855), which the printed "profit before tax" (191,185)
includes. The same label can mean different numbers in two presentations of one set of results: say
which one a figure comes from. Test: `tests/test_india_filings.py::test_real_annual_results_maruti_suzuki_fy2026`;
figures in `tests/fixtures/maruti_indas_consolidated_fy2026.json` (with sources and notes).

The other tests use a fictional company built to satisfy the Schedule III and Ind AS 7 arithmetic.

Fixed (2026-11-23), on that test disclosure: cost of goods sold took only the first cost line found
(4,000 instead of 4,800, the sum of materials consumed, purchases of stock-in-trade and the change in
inventories); operating income left out other expenses (4,300 instead of 1,800); tax took current
tax alone (400 instead of 425); every key not given became 0.0 and was cited as reported (net
income 0.0 when profit was missing); opening cash was back-solved from the cash flows and
`cash_reconciled` set to `True`. It now returns `IndASParseResult` with `derived` (each formula),
`not_reported` (optional fields not given) and `checks` (balance sheet totals as reported; opening
cash + cash flows + exchange-rate effect = closing cash, Ind AS 7), raises naming any required item
not given, and cites only the items in the disclosure.

Open items: one company and one year (consolidated); a standalone statement and a second company;
no retrieval (the caller types or supplies the figures; the exchange filing is HTML rendered from
XBRL and could be read directly); the "other equity" breakdown (retained earnings is "other equity"
unless given separately, and recorded as such) depends on what the caller supplies.
