# Agent workflows

How Līlāvatī's parts combine into agent workflows. Each workflow says what the model does, what code
does, which checks gate progress, what happens on failure, and how to evaluate it. One is implemented
and tested; the others are designs, built from parts that exist, with their status stated. The parts
are described in [agents](agents.md).

**Rules every workflow follows.**

- The model reads, plans and writes; code computes and checks. A number in the output comes from a
  tool or from a cited source line, never from the model's arithmetic.
- Checks gate progress. A failed check sends back only the work it implicates, or the workflow
  abstains: it never adjusts a number to make a check pass.
- The model is not shown the answer a check implies (a model told "should be 404,548" can copy it
  instead of re-reading). Checks name what failed; the source decides the value.
- Each workflow has a measurable claim and an ablation that tests it; until it is measured with real
  models, the claim is a hypothesis.

| Workflow | Status | Parts |
| :--- | :--- | :--- |
| 1. Extraction with diagnosis-driven repair | implemented; first real-model runs on five 10-Ks (no misreadings left to repair) | `FilingExtractionAgent`, `SourceStore`, `diagnose_identities` |
| 2. Covenant monitoring against guidance | tools implemented; agent proposed | `get_filing_figures`, workflow 1, `check_covenant_path`, `find_minimum_upfront_repayment_for_path` |
| 3. Point-in-time research dossier | implemented | `FinancialResearchAgent` |
| 4. Answers with evidence | proposed | MCP tools, `ProvenanceGate(store)`, answer check from the integrations |
| 5. Committee ladder | research design (Phase 11) | role prompts, `AdversarialCommittee`, gates |

## 1. Extraction with diagnosis-driven repair (implemented)

**Problem.** Language models read statement figures well (all 324 line items in
[Tutorial 08](tutorials/08_real_10k_llm_vs_lilavati.md)), but when they slip, the slip looks like a
real number: a prior-year column, a typo, a line that is not in the document.

**Workflow** (`lilavati.agents.extraction.FilingExtractionAgent`):

1. The extractor (a model, or any callable) returns each requested figure with the line it was read from.
2. Provenance: the quoted line must be in the document (`SourceStore`), and the value in the line.
3. Consistency: every identity the figures allow is checked; on failure, `diagnose_identities` names the
   smallest sets of figures that could be wrong.
4. Only the unsupported and diagnosed figures are re-read, with the failed identities named and no
   implied values. After `max_repair_rounds`, the agent abstains on what is left (`unresolved`).

The two checks catch different errors. A prior-year column passes provenance (the value is printed on
the quoted line) and fails the identities; a typo or an invented line fails provenance:

```python
from datetime import date

from lilavati.agents.extraction import FilingExtractionAgent

# Verbatim lines from Apple's FY2026 10-K statements: current year first, then prior years.
doc = "\n".join([
    "Total assets $ 404,548 $ 359,241 ",
    "Total liabilities  292,455  285,508 ",
    "Total shareholders’ equity  112,093  73,733 ",
    "Total liabilities and shareholders’ equity $ 404,548 $ 359,241 ",
])
lines = dict(zip(["total_assets", "total_liabilities", "total_equity", "liabilities_and_equity"], doc.splitlines()))
truth = {"total_assets": 404_548, "total_liabilities": 292_455, "total_equity": 112_093,
         "liabilities_and_equity": 404_548}


def extractor(text: str, figures: list[str], instruction: str) -> dict[str, tuple[float, str]]:
    """Stands in for a model: reads total assets from the prior-year column the first time."""
    out = {f: (float(truth[f]), lines[f]) for f in figures}
    if "re-read" not in instruction and "total_assets" in figures:
        out["total_assets"] = (359_241.0, lines["total_assets"])  # the value IS in the quoted line
    return out


res = FilingExtractionAgent(extractor).run(doc, list(truth), filing_type="10-K",
                                           filing_date=date(2026, 11, 3), period_end=date(2026, 9, 26))
for r in res.rounds:
    print(r.round, r.requested, r.failed_identities, r.candidates)
print(res.detail, res.figures["total_assets"].value)
```

```text
1 ['total_assets', 'total_liabilities', 'total_equity', 'liabilities_and_equity'] ['assets = liabilities + equity', 'assets = total liabilities and equity'] [['total_assets']]
2 ['total_assets'] [] []
consistent after 2 round(s) 404548.0
```

When one identity fails, every figure in it is a candidate and all are re-read (re-reading one picked
by plausibility would be a guess). Tests: `tests/test_extraction_agent.py`; example with a real model:
`examples/10_extraction_repair_agent.py --live groq-gpt-oss-120b`.

**Evaluation (first runs, 2026-11-25).** `scripts/run_extraction_repair_eval.py`, on the five pilot
10-Ks (Apple, Microsoft, Walmart, Johnson & Johnson, JPMorgan): 60 figures, 11 to 13 per filing, each
asked only if its value is printed in the statements text the model sees. Truth is each filing's own
XBRL facts. Four open-weight models on Groq (gpt-oss-120b, gpt-oss-20b, qwen3.6-27b, qwen3.8-27b),
temperature 0, one sample. Four arms: one pass (`single`), re-read what the checks name (`targeted`),
re-read everything after any failure (`reread_all`), and targeted with the implied values shown
(`implied`). Within a run every arm starts from the same first reading, so the arms differ only in the
repair. Raw replies are in `results/extraction/`; the tables are `--report x1` and `--report x2`
(files ending `_t1` are the temperature-1.0 probe).

- **x1 found faults in the checker and the prompt, not misreadings.** The quote check required the
  document's spacing and apostrophes: "(139,500)" for the HTML text's "(139,500 )", and
  "shareholders' equity" for "shareholders’ equity". In the one-pass arm, all 17 figures the agent
  abstained on had correct values. The revenue label read "total revenue (net sales)", and Walmart prints
  both net sales (706,413) and total revenues (713,163). gpt-oss-120b and gpt-oss-20b took net sales,
  and the agent accepted it: Walmart reports no gross profit, so no identity involves revenue.
- **x2 (quote check ignores whitespace and reads curly quotes as straight; label asks for total
  revenues):** every model read all 60 figures correctly in one pass (240 of 240), every arm was
  consistent after the first round, and no repair ran. Four figures were checked against the document
  only (`unchecked`): Walmart's revenue, cost of revenue and total equity (no gross profit or total
  liabilities line), and JPMorgan's revenue.
- **Probe at temperature 1.0** (gpt-oss-20b and qwen3.6-27b, three samples each, 360 figures):
  qwen3.6-27b read all 180 correctly. gpt-oss-20b quoted three equity lines that are not in the
  document, two with a third year's number added and one with dollar signs added. The quote check
  caught all three, and one re-read fixed each. Its one misreading got through: JPMorgan's revenue was
  read from "Noninterest revenue 87,004" instead of "Total net revenue 182,447". That line exists and
  holds the value, and no identity involves a bank's revenue, so the agent accepted it and listed it as
  `unchecked`. No identity failed in any x2 run, so the diagnosis never ran on a real model's output.

**What this does not show.** With no misreadings caught by an identity, the runs cannot say whether targeted
re-reading repairs as well as re-reading everything, or whether models copy implied values. That needs
inputs where models do misread: page text from PDFs, 10-Q columns (three months beside year to date),
Ind AS annual reports in lakhs or crores, or many samples at a higher temperature. The runs do show two
things a repair loop depends on. A quote check that rejects formatting differences turns correct
values into abstentions. And identities cannot catch a wrong figure that none of them involves, so such
figures are reported as `unchecked` rather than verified.

## 2. Covenant monitoring against guidance (tools implemented; agent proposed)

**Question.** Will the borrower meet its covenants over the guidance period, and if not, how much must
be repaid now?

1. Figures: `get_filing_figures` (SEC XBRL, cited), or workflow 1 for a document without XBRL.
2. Terms: the model extracts covenant definitions, step-downs, the sweep grid and management's guidance
   ranges from the credit agreement and the earnings call, each with its quoted line (workflow 1's
   provenance check applies; identities do not, so every term is shown to the analyst).
3. Deterministic: `check_covenant_path` over the guidance ranges; on a breach,
   `find_minimum_upfront_repayment_for_path`.
4. Memo: the model writes it from the tool results; every number must be one of them (workflow 4's check).

**Abstain when** a definition is ambiguous (for example EBITDA add-backs the agreement leaves to
judgement) or a term has no quoted source. **Evaluate** on credit agreements with known covenant
outcomes; the corner-versus-range example in [Tutorial 01, Step 4c](tutorials/01_build_your_first_financial_agent.md)
is the kind of case a corner-only check misses.

## 3. Point-in-time research dossier (implemented)

`FinancialResearchAgent` ([Tutorial 08, Step 6](tutorials/08_real_10k_llm_vs_lilavati.md)): the latest
10-K filed by an as-of date, the three-statement model and committee on reported inputs, a scenario
tree, every citation resolved against the SEC data read; it refuses when an input is not reported. No
model is called. **Extension (proposed):** a model writes the narrative from the dossier's fields only,
checked by workflow 4.

## 4. Answers with evidence (proposed)

**Question.** A user asks a quantitative question; the model answers with tools.

1. The model plans and calls MCP tools; each result carries its engine's `validation` status.
2. Gate on the final answer: every number in it must equal a tool output or a cited source figure
   (within the stated rounding), extending `examples/integrations/_task.py::check_answer` from a test
   helper to a gate; cited figures go through `ProvenanceGate(store)`.
3. A number with no source sends the answer back once, naming it; then the answer is returned with that
   number marked unsupported. An answer resting on a `not reference-checked` engine says so.

**Evaluate** on the benchmark's tool arm: unsupported numbers per answer, with and without the gate.

## 5. Committee ladder (research design, Phase 11)

Same model, data, questions and budget, one rung at a time: a single tool-using agent; language-model
roles (the bull, bear and portfolio-manager prompts in `prompts/committee.py`); the rule-based
`AdversarialCommittee`; each with and without the gates. **Measure** accuracy, unsupported claims, cost
per correct answer and agreement between repeated runs; **the open question** is whether several model
roles reduce financial errors once arithmetic is in tools, or only add cost.
