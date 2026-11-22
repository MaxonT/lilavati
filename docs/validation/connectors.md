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
[Tutorial 08](../tutorials/08_real_10k_llm_vs_lilavati.md), raw replies in `results/filings/`. Open items: banks and insurers need their
own templates; extension (company-specific) tags are reported as missing, not mapped; 10-Q, 10-K/A
and restated comparatives are not yet covered; five companies is a pilot, not a validation set.
