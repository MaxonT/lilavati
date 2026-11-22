# Līlāvatī-Bench: Financial AI Agent Benchmark Suite

**Līlāvatī-Bench** is a quantitative benchmark (harness and reference solver available; first no-tool baselines for Google and Groq-hosted open-weight models, see below; no leaderboard yet) for evaluating financial foundation models and agents across seven tracks.

---

## The Four Core Institutional Metrics

1. **Proof-of-Citation Score (PoCS):**
   $$\text{PoCS} = \frac{\sum \text{Verified Claims with Matching Document SHA-256 and Valid Span}}{\text{Total Extracted Claims}}$$
   Measures whether claims trace back to statutory filings without hallucinations.

2. **Accounting Invariant Compliance Rate (ICR):**
   $$\text{ICR} = \frac{\sum \text{Statements Satisfying Formal Invariants via Z3 SMT Solver}}{\text{Total Evaluated Statements}}$$
   Tests that $Assets = Liabilities + Equity$ and Cash Flow reconciles with zero tolerance.

3. **Temporal Purity Index (TPI):**
   $$\text{TPI} = \frac{\sum \text{Observations where } T_{\text{filing}} \le T_{\text{cutoff}}}{\text{Total Observations}}$$
   Measures resistance to look-ahead bias and restatement leakage (currently from the filing dates an agent reports; see the roadmap).

4. **Tool Math Purity Ratio (TMPR):**
   $$\text{TMPR} = \frac{\text{Deterministic Tool Computations}}{\text{Deterministic Tool Computations} + \text{LLM Mental Math Assertions}}$$
   Enforces that LLMs never execute arithmetic inside text tokens.

5. **Composite Līlāvatī Index (0 - 100):**
   $$\text{Index} = 100 \times \left(0.30 \cdot \text{PoCS} + 0.30 \cdot \text{ICR} + 0.20 \cdot \text{TPI} + 0.20 \cdot \text{TMPR}\right)$$

---

## The Seven Quantitative Evaluation Tracks

| Track | Focus | Key Evaluated Assertions |
| :--- | :--- | :--- |
| **Track 1** | Three-Statement Reconciliation | Dynamic working capital shocks, SMT balance proofs |
| **Track 2** | Fixed Income & Credit | Bond pricing, duration, Merton structural default probability |
| **Track 3** | Econometrics | ADF unit roots, Engle-Granger cointegration, OU half-life |
| **Track 4** | Financial ML | Fractional differentiation order $d^*$, memory retention ($r \ge 0.25$ on the synthetic random-walk case) |
| **Track 5** | Derivatives | BSM Greeks, SABR volatility smile, Z3 butterfly spread arbitrage |
| **Track 6** | Microstructure & Execution | LOB Microprice, OFI, Almgren-Chriss liquidation schedule |
| **Track 7** | Sustainability Finance | Portfolio WACI, carbon tax sensitivity, SLB step-up covenants |

---

## Running the reference solver

```bash
uv run lilavati-bench run --format markdown
```

The `lilavati-bench` CLI always runs the deterministic reference solver (`ReferenceAgentSolver`);
`--model` only sets the label on the report card. It passes by construction. To score a real LLM, use
the direct-prompt baseline below.

---

## Direct-prompt LLM baselines

`scripts/run_llm_baseline.py` poses each case to a model as plain text with its numeric context (no
tools, no retrieval) and scores the returned JSON with the same tolerance rule as the reference solver.

```bash
export GROQ_API_KEY=...        # never put keys in files, scripts or URLs
uv run python scripts/run_llm_baseline.py --provider groq --model openai/gpt-oss-120b \
    --max-tokens 8192 --rpm 20 --out results/baseline_groq_gpt-oss-120b.jsonl          # 7 golden cases
uv run python scripts/run_llm_baseline.py --provider groq --model openai/gpt-oss-120b \
    --generated 4 --seed 2026 --max-tokens 8192 --rpm 20 \
    --out results/generated_groq_gpt-oss-120b.jsonl                                    # 20 generated cases
```

| Option | Meaning |
| :--- | :--- |
| `--provider gemini` / `groq` | Endpoint and key variable (`GEMINI_API_KEY` / `GROQ_API_KEY`). Groq uses the OpenAI-compatible path. |
| `--resume` | Re-run only provider failures (HTTP errors, timeouts, empty replies not caused by the token cap). Unparsable or wrong answers are never re-rolled. |
| `--rescore` | Re-parse and re-score the stored replies in `--out` without calling the model (use after a parser or scoring fix). |

Each reply is classified as one of: **error** (provider failure: 429/5xx, timeout), **model_error**
(the provider rejected the model's own output, e.g. a malformed tool call), **truncated**
(`finish_reason` shows the token cap was hit), **unparsable** (the model stopped but returned no valid
JSON) or **answered**. Non-answers score 0 in the item mean and are
reported separately. Reasoning returned inline as `<think>...</think>` is stripped before parsing.

Measured results, run settings and caveats are in [`results/README.md`](https://github.com/revendrat/lilavati/blob/main/results/README.md)
and section 5.2 of the [benchmark paper draft](papers/02_benchmark_paper.md). In brief: the 7 golden
cases are saturated, and the 20 generated cases separate models (item accuracy 26%–95% across five
Groq-hosted open-weight models, one pass at temperature 0). Groq is not deterministic at temperature
0, so single runs are approximate.

---

## Tool-augmented arm (paired with the no-tool baseline)

`scripts/run_tool_baseline.py` poses the same generated cases with eight deterministic tools
(`lilavati.bench.tool_eval.TOOLS`: bond analytics, Black-Scholes price and Greeks, order book, WACI,
SLB covenant, balance-sheet and cash-flow checks) plus a `submit_answer` tool for the final answer.
Each record adds the tool calls, their outputs and **tool grounding**: the share of numeric answer
fields equal to a number a tool returned (a measured version of TMPR). A provider rejecting a
malformed tool call is classified `model_error`, not a provider failure.

Interpretation: the expected values are computed by the same engines the tools call, so this arm
measures tool selection, parameter extraction and copying, not engine correctness. Engine
correctness is established separately against independent references ([Numerical
validation](validation/index.md)). Compare the two arms only on identical questions, matched by case
content hash (`lilavati.bench.analysis.paired_difference`). Walk-through: [Tutorial 07](tutorials/07_tools_versus_mental_math.md).

## Evidence trail

| File | Produced by | Contents |
| :--- | :--- | :--- |
| `results/*.jsonl` | run scripts | every raw reply, parsed answer, score, `case_hash`, prompt version (tool runs: tool calls) |
| `results/*.meta.json` | run scripts | settings, prompt and tool-prompt versions, git commit, date |
| `results/legacy_metadata.json` | hand-recorded | settings of runs made before meta files existed (`null` when unknown) |
| `results/manifest.json` | `scripts/build_results_manifest.py` | SHA-256, case set, status (current / superseded) and case hashes per file |
| `results/ANALYSIS.md`, `analysis.json` | `scripts/analyse_results.py` | outcome classes, accuracy, bootstrap CIs, paired comparisons, field-level misses |

Prompt versions: v1 (superseded: wrong vega unit), v2 (bond key said "5-year bond" for generated
bonds of 2-10 years), v3 (current). Tool-prompt versions: t1 (pilot: text JSON answer, which
gpt-oss-120b tried to send as a call to a non-existent `json` tool), t2 (current: `submit_answer`).
`tests/test_results_manifest.py` fails if the manifest does not match the files or if a key pattern
appears anywhere in `results/`. Walk-through: [Tutorial 06](tutorials/06_analysing_and_publishing_results.md).
