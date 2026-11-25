# Tutorial 10: An Indian Company from Its Exchange Filing

**Who this is for:** analysts, students and developers working with Indian listed companies; a session
on real filings in a course on AI in finance.
**Time:** about 40 minutes. **Cost:** free (the filing is public; no API calls).

## Learning objectives

1. Explain what a listed Indian company files with the exchanges each quarter, and in which format.
2. Read NSE's page for an Integrated Filing (Financial) into the Ind AS crosswalk, with every figure
   cited to the page.
3. Check the figures two ways: the accounting identities, and the printed statements.
4. Explain why two correct sources can disagree (definitions, not errors).

## Where this fits

[Tutorial 08](08_real_10k_llm_vs_lilavati.md) does this for a US 10-K, where SEC serves XBRL facts
through an API. India has no such API: SEBI's Integrated Filing (circular of 31 December 2024) is filed
in XBRL against SEBI's Integrated Filing Finance Taxonomy (`in-capmkt`, version 1.0 of 31 January 2025),
and NSE publishes each filing as a page rendered from it. This tutorial reads that page.

## Step 1: Download the filing by hand

NSE's terms of use prohibit automated collection (scraping, data extraction) without its written
consent, so download the page in your browser, not with a script:

1. Open
   `https://nsearchives.nseindia.com/corporate/ixbrl/INTEGRATED_FILING_INDAS_153810_28042026194740_iXBRL_WEB.html`
   (Maruti Suzuki India, consolidated results for the year ended 31 March 2026, filed 28 April 2026).
2. Save it with **Save Link As** (or "Save Page As, HTML only"; not "Webpage, Complete", which rewrites
   the file and changes its hash) to
   `data/raw/india_xbrl/MARUTI/2026-03-31/`, keeping NSE's file name. `data/raw/` is git-ignored.

The name ends in `_iXBRL_WEB.html`, but the page is a rendering of the XBRL: labelled tables, no XBRL
tags. Keep NSE's name: its `28042026194740` stamp is the submission time (28 April 2026, 19:47:40), the
date from which the figures were public.

## Step 2: Read the page

```python
from pathlib import Path

from lilavati.connectors.nse_integrated_filing import read_nse_integrated_filing

PAGE = Path("data/raw/india_xbrl/MARUTI/2026-03-31/INTEGRATED_FILING_INDAS_153810_28042026194740_iXBRL_WEB.html")
filing = read_nse_integrated_filing(PAGE.read_bytes(), source_name=PAGE.name)
print(filing.company_name, "|", filing.symbol, "|", filing.isin, "|", filing.nature, "| values in", filing.rounding)
print("financial year", filing.financial_year_start, "to", filing.financial_year_end, "| filed", filing.filed_at)
for s in filing.sections:
    print(f"{s.kind:16} {len(s.rows):3} rows; columns {[(str(c.start), str(c.end)) for c in s.columns]}")
```

```text
Maruti Suzuki India Limited | MARUTI | INE585B01010 | Consolidated | values in Lakhs
financial year 2025-04-01 to 2026-03-31 | filed 2026-04-28 19:47:40
profit_and_loss   61 rows; columns [('2026-01-01', '2026-03-31'), ('2025-04-01', '2026-03-31')]
balance_sheet     96 rows; columns [('2025-04-01', '2026-03-31')]
other             22 rows; columns [('2026-01-01', '2026-03-31'), ('2025-04-01', '2026-03-31')]
other             11 rows; columns [('2026-01-01', '2026-03-31'), ('2025-04-01', '2026-03-31')]
cash_flow        103 rows; columns [('2025-04-01', '2026-03-31')]
```

Each section states its own columns: the income statement has the quarter and the year to date, the
statement of assets and liabilities and the cash flow statement the year. The two "other" sections are
segments and other comprehensive income. Values are in lakhs (1 lakh = 100,000 rupees).

## Step 3: Into the Ind AS crosswalk

```python
from pathlib import Path

from lilavati.connectors.india_filings import IndiaFilingsClient
from lilavati.connectors.nse_integrated_filing import read_nse_integrated_filing, to_ind_as_disclosure

PAGE = Path("data/raw/india_xbrl/MARUTI/2026-03-31/INTEGRATED_FILING_INDAS_153810_28042026194740_iXBRL_WEB.html")
filing = read_nse_integrated_filing(PAGE.read_bytes(), source_name=PAGE.name)
result = to_ind_as_disclosure(filing, cin="L34103DL1981PLC011375")  # the CIN is not on the page
f, lakh = result.disclosure.facts, result.disclosure.scale
print(f"revenue {f['RevenueFromOperations']:,.0f} lakh = INR {f['RevenueFromOperations'] * lakh / 1e7:,.0f} crore")
print(f"profit for the year {f['ProfitForThePeriod']:,.0f} lakh; total assets {f['TotalAssets']:,.0f} lakh")
print("missing:", result.missing)
print("derived:", result.derived)
for c in IndiaFilingsClient.parse_ind_as_disclosure(result.disclosure).checks:
    print(f"{c.name}: {'holds' if c.holds else 'FAILS'}")
```

```text
revenue 18,331,600 lakh = INR 183,316 crore
profit for the year 1,467,950 lakh; total assets 14,888,100 lakh
missing: []
derived: {'TradeReceivables': 'Trade receivables, current + Trade receivables, non-current', 'TradePayables': 'Total Trade payable (Trade payables, current) + Total Trade payable (Trade payables, non-current)'}
assets = liabilities + equity: holds
assets = total equity and liabilities: holds
opening cash + cash flows = closing cash: holds
```

The page shows labels, not XBRL element names, so each crosswalk item is matched to its label within
its section (one of them is NSE's own spelling, "Total equity and liabilites"). Nothing is filled: an
item the page does not show is listed under `missing`. The two sums are recorded under `derived`.

## Step 4: Against the printed statements

The repository holds the same company's printed statements, typed from the results PDF
(`tests/fixtures/maruti_indas_consolidated_fy2026.json`, INR million):

```python
import json
from pathlib import Path

from lilavati.connectors.nse_integrated_filing import read_nse_integrated_filing, to_ind_as_disclosure

PAGE = Path("data/raw/india_xbrl/MARUTI/2026-03-31/INTEGRATED_FILING_INDAS_153810_28042026194740_iXBRL_WEB.html")
result = to_ind_as_disclosure(read_nse_integrated_filing(PAGE.read_bytes(), source_name=PAGE.name),
                              cin="L34103DL1981PLC011375")
printed = json.loads(Path("tests/fixtures/maruti_indas_consolidated_fy2026.json").read_text())["facts"]  # INR million
same = [k for k, v in printed.items() if abs(result.disclosure.facts[k] / 10 - v) < 0.05]  # 1 lakh = 0.1 million
print(f"{len(same)} of {len(printed)} printed items equal the exchange filing")
for k, v in printed.items():
    if k not in same:
        print(f"  {k}: printed {v:,.0f}, exchange {result.disclosure.facts[k] / 10:,.0f} (INR million)")
```

```text
26 of 28 printed items equal the exchange filing
  ProfitBeforeTax: printed 191,185, exchange 188,330 (INR million)
  PropertyPlantAndEquipment: printed 334,197, exchange 340,195 (INR million)
```

Both differences are definitions, and both reconcile to the unit. The printed profit before tax
includes the share of profit of associates (2,591) and joint ventures (264), which the exchange format
shows after tax: 191,185 − 2,855 = 188,330. The exchange's property, plant and equipment includes the
right-of-use assets (5,998), a separate line in the printed statement: 334,197 + 5,998 = 340,195.

## Step 5: Cite every figure

```python
from pathlib import Path

from lilavati.connectors.nse_integrated_filing import (
    citation,
    read_nse_integrated_filing,
    register_filing,
    to_ind_as_disclosure,
)
from lilavati.core.provenance import SourceStore

PAGE = Path("data/raw/india_xbrl/MARUTI/2026-03-31/INTEGRATED_FILING_INDAS_153810_28042026194740_iXBRL_WEB.html")
content = PAGE.read_bytes()
filing = read_nse_integrated_filing(content, source_name=PAGE.name)
result = to_ind_as_disclosure(filing, cin="L34103DL1981PLC011375")
store = SourceStore()
register_filing(store, filing, content)
c = citation(filing, result, "NetCashFromInvestingActivities")[0]
print(c.raw_snippet, "|", c.doc_hash[:12], "| resolves:", store.check(c) == [])
wrong = c.model_copy(update={"raw_snippet": "Net cash flows from (used in) investing activities (14,73,450.00)"})
print("a mistyped figure:", store.check(wrong))
```

```text
Net cash flows from (used in) investing activities (14,73,350.00) | cf868659b98c | resolves: True
a mistyped figure: ['cf868659b98c...: snippet not found in the document']
```

The citation is the row as shown and the SHA-256 of the file you saved. Your hash is the same only if
your saved file is byte-for-byte the same as this one.

## Step 6: Without the download

The crosswalk itself runs on the printed figures in the repository, so this step runs anywhere:

```python
import json
from pathlib import Path

from lilavati.connectors.india_filings import IndASCompanyDisclosure, IndiaFilingsClient

raw = json.loads(Path("tests/fixtures/maruti_indas_consolidated_fy2026.json").read_text())
printed = IndASCompanyDisclosure(**{k: v for k, v in raw.items() if not k.startswith(("_", "exchange"))})
m = IndiaFilingsClient.parse_ind_as_disclosure(printed)
inc = m.statement_model.income_statement
print(f"gross profit INR {inc.gross_profit / 1e7:,.0f} crore; derived: {sorted(m.derived)}")
print([(c.name, c.holds) for c in m.checks])
```

```text
gross profit INR 50,740 crore; derived: ['cogs', 'gross_profit', 'net_change_in_cash', 'operating_income', 'retained_earnings', 'sga_expense']
[('assets = liabilities + equity', True), ('assets = total equity and liabilities', True), ('opening cash + cash flows = closing cash', True)]
```

Schedule III has no gross profit line: it is derived (revenue minus materials, purchases of
stock-in-trade and the change in inventories), and the crosswalk says so.

## What can go wrong

- **A saved page that is not the original.** "Webpage, Complete" rewrites the HTML; the figures may
  still read, but the hash, and so every citation, no longer matches NSE's file.
- **A label NSE changes.** The reader matches labels, so a renamed row shows up under `missing`; it is
  never guessed. Add the new label to `LABELS` in `connectors/nse_integrated_filing.py` with a test.
- **Standalone and consolidated mixed up.** Each column states which it is; a column that disagrees
  with the filing raises an error.
- **Definitions that differ between sources** (Step 4). Reconcile them; do not average them.
- **The quarter.** `period="quarter"` gives the quarter's income statement only: this filing shows the
  balance sheet and the cash flow statement for the year, so for the quarter they are listed as missing
  (19 items).

## Using this with Claude Code and MCP

- **In Claude Code:** download the page by hand, then ask Claude Code to read it with
  `read_nse_integrated_filing` and reconcile it with the results PDF; the
  [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
  holds the data rules (manual downloads only, no filled items, reconcile definitions).
- **MCP:** no MCP tool exposes the Indian connectors yet (an agent would have to pass the page's
  content); for SEC filings, `get_filing_figures` on `lilavati-mcp-corporate-filings` does the same job
  ([MCP setup](../mcp-setup.md)).

## Exercises

1. Download Maruti's standalone filing for the same year. Which figures change, and which identities
   still hold?
2. Pick a second company in a different sector. Which crosswalk items are missing, and why?
3. Run `to_ind_as_disclosure(..., period="quarter")`. Which items are missing, and why is that right?
4. Find the cash flow statement's opening cash (18,450 lakh) and the change in cash (−11,760 lakh) on
   the page, and check the closing cash by hand.
5. Add the "Exceptional items" row to `LABELS` with a test on a synthetic page.

## For instructors

- **Discussion:** the exchange format and the printed statement disagree on profit before tax. Which is
  "right"? (Both: they define it differently.) What would a model that read only one of them report?
- **Lab:** each group downloads one company's filing, runs Steps 2 to 4, and presents every difference
  from the printed statements with its reconciliation.
- **Assessment idea:** give students a page with one row relabelled and ask them to find, from the
  `missing` list and the identities, what changed.
