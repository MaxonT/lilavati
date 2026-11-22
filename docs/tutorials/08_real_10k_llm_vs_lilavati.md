# Tutorial 08: A Real 10-K, Three Ways: LLM Alone, LLM + Līlāvatī, Līlāvatī Alone

**Who this is for:** analysts, students and AI engineers who want to read real filings with AI.
**Time:** about 60 minutes. **Cost:** free (SEC data is public; the LLM part uses a free-tier key).

## Learning objectives

1. Get a company's annual figures from SEC EDGAR in a way that works across companies: identify
   the filing, then take only that filing's own XBRL facts.
2. Check the extraction: accounting identities, items the company does not report, and the figures
   appearing in the filing document itself.
3. Compare three ways of producing the same numbers from the same 10-K, and see where each fails.

## Where this fits

Tutorials [05](05_benchmarking_llms_with_open_models.md)–[07](07_tools_versus_mental_math.md) use
generated cases. This one uses real filings, and it was the first real test of Līlāvatī's own filing
connectors: six bugs were found and fixed on the way (see "What can go wrong").

## Step 1: The flow, and the trap it avoids

```text
ticker -> CIK (SEC company list) -> submissions index -> the 10-K: accession number, report date
      -> company facts (XBRL) -> keep facts with that accession, end date = report date,
         and (for flow items) a duration of about a year
      -> canonical line items + citations; items not reported listed as missing
      -> identity checks; every figure looked up in the filing document
```

The trap: SEC company facts tag every value in a 10-K with that filing's fiscal year, including the
two prior years it repeats for comparison. Selecting "fiscal year 2026" by that tag alone returned
Apple's FY2025 total assets and FY2024 revenue. Anchoring on the filing (accession number and report
date) removes the ambiguity.

## Step 2: Identify the filing and extract its figures

This runs offline on verbatim excerpts of SEC's responses for Apple, kept in `tests/fixtures/`:

```python
import json

from lilavati.connectors.sec_filing import extract_filing, filings_from_submissions

submissions = json.load(open("tests/fixtures/aapl_submissions_subset.json"))
facts = json.load(open("tests/fixtures/aapl_companyfacts_10k_fy2025_2026.json"))

filing = filings_from_submissions(submissions)[0]          # the latest 10-K
print(filing.form, filing.accession, filing.report_date)

result = extract_filing(facts, filing)
for name in ("revenue", "cost_of_revenue", "net_income", "total_assets", "total_liabilities"):
    print(f"{name:18} {result.values[name].value / 1e6:>10,.0f}  ({result.tags_used[name]})")
print("not reported:", result.missing)
for check in result.checks:
    print(f"{check.name}: {'holds' if check.holds else 'FAILS'}")
```

```text
10-K 0000320193-26-000026 2026-09-26
revenue               478,834  (RevenueFromContractWithCustomerExcludingAssessedTax)
cost_of_revenue       245,232  (CostOfGoodsAndServicesSold)
net_income            132,175  (NetIncomeLoss)
total_assets          404,548  (Assets)
total_liabilities     292,455  (Liabilities)
not reported: ['total_equity_incl_nci']
assets = liabilities + equity: holds
assets = total liabilities and equity: holds
revenue - cost of revenue = gross profit: holds
```

Every value carries a citation with the accession number, the XBRL tag and the period. An item the
company does not report (Apple has no non-controlling interests) is listed, never filled with zero.

## Step 3: Ratios by code, not by the model

```python
import json

from lilavati.connectors.sec_filing import extract_filing, filing_ratios, filings_from_submissions

filing = filings_from_submissions(json.load(open("tests/fixtures/aapl_submissions_subset.json")))[0]
result = extract_filing(json.load(open("tests/fixtures/aapl_companyfacts_10k_fy2025_2026.json")), filing)

millions = {k: v.value / 1e6 for k, v in result.values.items()}
for name, value in filing_ratios(millions).items():
    print(f"{name:22} {value:,.4f}")
```

```text
gross_margin           0.4879
net_margin             0.2760
return_on_assets       0.3267
free_cash_flow         136,056.0000
liabilities_to_assets  0.7229
```

## Step 4: Any company, live

The SEC asks every client for a real contact. Put yours in your shell profile (never in a file in
the repository), then validate the flow on any companies:

```bash
export LILAVATI_SEC_USER_AGENT="Your Name you@example.org"
uv run python scripts/validate_sec_filings.py AAPL MSFT WMT JNJ JPM
```

For the five companies above, every extracted value appeared in the filing document at the filing's
scale (64 of 64), and every identity that could be checked held. What the companies do not report
as a tagged line is listed per company on the [connectors validation page](../validation/connectors.md):
Walmart has no gross profit line, J&J no operating income line, and JPMorgan, a bank, has
statements of a different shape altogether.

## Step 5: The same 10-K, three ways

`scripts/run_filing_comparison.py` gives each language model the financial statements section of the
filing (the HTML document's text, from the first primary statement) and asks for seven line items:

- **Arm A, LLM alone:** the model reports the items and computes five ratios itself.
- **Arm B, LLM + Līlāvatī:** the model only reports the items; `filing_ratios` computes the ratios.
- **Arm C, Līlāvatī alone:** XBRL extraction as in Step 2, no model.

Before any call, the script checks that each answer appears in the text the model receives, so no
miss is caused by missing input. Line items must match to the million; ratios to within 0.5%.

```bash
export GROQ_API_KEY=...            # free tier
uv run python scripts/run_filing_comparison.py --tickers AAPL MSFT WMT JNJ --samples 3
```

Measured on 2026-11-22/23 (Apple, Microsoft, Walmart and J&J latest 10-Ks; four open-weight models
on Groq; 3 samples each at temperature 0; raw replies in `results/filings/`):

| | Line items correct | Ratios correct |
| :--- | ---: | ---: |
| A: LLM alone | 324 / 324 | 225 / 228 |
| B: LLM extracts, Līlāvatī computes | 324 / 324 | 228 / 228 |
| C: Līlāvatī alone (XBRL) | from the filing's own facts | computed by code |

Every model read every figure correctly from these four large, cleanly presented filings. The
difference was arithmetic: Qwen3.8 27B extracted J&J's revenue (94,193) and cost of sales (30,256)
correctly and then reported a gross margin of 0.5259 instead of 0.6788, in all three samples. In arm
B, the same model's extraction gave the right ratio, because code computed it. gpt-oss-120b,
gpt-oss-20b and Qwen3.6 27B got every ratio right in both arms.

Read this as a pilot: four companies, one filing each, one provider. It does not show that models
read every filing correctly; harder filings (scale in thousands, restatements, banks, notes rather
than statements) are where to look next.

## What can go wrong (all found while building this tutorial)

- **Comparative periods returned as current.** Selecting by fiscal-year tag returned prior years'
  figures (Apple "FY2026" revenue 391,035, the FY2024 figure, instead of 478,834). Fixed: anchor on
  the filing.
- **Text parsing that works for one company only.** Reading the PDF statements, the parser took
  pretax income as tax expense (the label "Income before provision for income taxes" contains
  "provision for income taxes"), dropped non-current debt, and read a par value printed inside a
  label as common equity. Fixed with general rules, but XBRL is the source of record now.
- **A missing item that looked reported.** Interest expense (not a separate line at Apple) came back
  as 0.0. Missing items are now listed.
- **HTML that splits words.** Microsoft's document renders "INCOME" as two spans, so a plain tag
  strip produced "IN COME STATEMENTS" and the heading was never found.
- **Statements in a different order.** J&J puts the balance sheet before the income statement; the
  text window now starts at whichever comes first.
- **A wrong answer key.** All four models answered Walmart's revenue as 713,163 and the harness
  scored them wrong. The harness was wrong: 713,163 is total revenues (with membership income), and
  the key used net sales (706,413). When every model "fails" the same way, check the key first.
  The stored replies were re-scored with the corrected definition (`--rescore`), not re-asked.
- **A fake contact.** The connector sent a made-up contact address to SEC. It now reads yours from
  `LILAVATI_SEC_USER_AGENT`.

## Using this with Claude Code and MCP

- **From an AI assistant:** the `lilavati-mcp-corporate-filings` server exposes the EDGAR connector
  as tools ([MCP setup](../mcp-setup.md)); set `LILAVATI_SEC_USER_AGENT` in the environment the
  server runs in.
- **Contributing:** the [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
  holds the parsing and data rules learned here (anchor on the filing, list missing items, check a
  harness key when every model fails the same way).

## Exercises

1. Run Step 4 for a company in your own sector. Which line items are missing, and why?
2. Add `effective_tax_rate = income_tax / pretax_income` to `filing_ratios` with a test on the Apple
   fixture, then run the comparison again. Do models compute it correctly in arm A?
3. Try a 10-Q: change `filings_from_submissions(..., forms=("10-Q",))`. What changes in the duration
   rule, and why?
4. Choose a bank. Which canonical items make no sense for it, and which would you add instead?

## For instructors

- **Discussion:** every model read every figure correctly here. Does that make deterministic tools
  unnecessary? Use the J&J gross margin, the Walmart answer key and the comparative-period bug in
  your answer.
- **Lab:** each group takes three companies from one sector, runs Steps 4 and 5, and presents its
  misses (extraction, arithmetic, harness).
- **Assessment idea:** give students a deliberately wrong answer key (as with Walmart's revenue) and
  ask them to find it from the pattern of "errors".
