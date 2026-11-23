---
name: lilavati-dev
description: Coding, testing and documentation rules for Project Līlāvatī (lilavati-finance). Use whenever writing or reviewing code, tests, docs, papers or benchmark material in this repository, especially numeric finance code (pricing, risk, econometrics, accounting checks), MCP servers, LLM clients, or any change that states a number or capability in docs.
---

# Līlāvatī developer skill

Līlāvatī is an alpha toolkit: deterministic finance engines exposed as FastMCP tools, Z3 accounting
checks, provenance and point-in-time data, and a benchmark. Credibility depends on every number being
right and every claim being true. These rules come from real bugs found in this repo; follow them.

## 1. Workflow (every change)

1. Read the code you are changing and its tests before editing. Re-verify any claim in a task
   description yourself (grep, run it); task notes and reports can be wrong.
2. For a bug: write a test that fails on the current code, then fix, then show it passes.
3. Run and paste the tails in your final message:
   ```bash
   uv run ruff check src tests scripts examples
   uv run mypy src
   uv run pytest
   ```
   If you touched docs: `uv run --extra docs --extra dev mkdocs build --strict`.
   If you touched docs, docstrings or printed text: `uv run python scripts/check_claims.py`.
   If you touched a tutorial or an API a tutorial uses: `uv run python scripts/check_tutorials.py`.
   CI (`.github/workflows/ci.yml`) runs all of these plus every example and the manifest check.
   If you touched an engine used by examples: run the affected `examples/*.py`.
4. Grep docs, papers, tutorials and examples for numbers your change affects; refresh them by
   re-running code. Never edit a quoted number by hand-calculation.
5. Do not commit, push, tag, publish or change GitHub settings unless the maintainer asks.
   Conventional Commits; one logical change per commit.
6. When testing old code against a new test, note that `pyproject.toml` sets
   `pythonpath = ["src"]`, which overrides `PYTHONPATH`: run the old copy with its own config
   (`pytest -c`, `--rootdir`) and confirm which module loaded via `module.__file__`.
7. When delegating to another agent, give it an explicit file list it may touch, forbid git writes,
   and independently verify its key claims (run old vs new code) before reporting them as fact.

## 2. Numeric code rules

- **Zero LLM arithmetic.** Every computed number comes from deterministic code (NumPy, SciPy,
  statsmodels, arch, cvxpy, DuckDB, Z3), exposed as an MCP tool where useful. LLMs extract
  parameters and dispatch tools.
- **Test against independent references, never against the implementation.** Use a textbook
  worked example (Hull, Fabozzi, López de Prado, Tsay), the original paper, or a mature library.
  QuantLib 1.44 is a test-only dev dependency; reference tests go in `tests/reference/` with the
  source cited next to each expected value. Add property tests (parity, monotonicity, limiting
  cases, finite-difference Greeks) where references are thin.
- **State units and conventions in the docstring**: annual vs per-period, compounding/frequency,
  day-count, percent vs decimal (prefer decimals), vega per vol point vs per 1.0, theta per day vs
  per year, price units vs fractional vol. Inconsistent units caused the Almgren-Chriss bug
  (fractional vol used where price-unit vol was required).
- **Check solver output.** A root-finder or optimiser result must be validated (residual check)
  before returning it; raise instead of returning a non-solution. The Merton solver silently
  returned points with 10–90% equation error because convergence was never checked.
- **Never return a placeholder that looks computed.** If a search or solver finds nothing, raise.
  Do not fill result fields with defaults: the fractional-d search returned `optimal_d=1.0,
  adf_p_value=1.0` and the undifferenced input as if they were results.
- **A limit or failure must not look like success.** When a loop stops at a limit, report it
  (`AgentOutput.completed = False`). `max_reflection_iterations` once also capped tool rounds,
  so a 3-step tool plan returned an empty answer with `smt_invariants_verified=True`.
- **No fabricated fallbacks in agents or connectors.** `FinancialSearchAgent` returned Apple's
  figures with invented page numbers for any symbol when SEC was unreachable, and the research agent a
  fixed target price; both looked like results. On missing data return nothing and say why (`notes`),
  or label an illustrative baseline and list every assumed input. A citation field with no real
  value is `None`, not a placeholder (`page=1` for XBRL facts, a hash of an identifier string).
- **A check must not use a value derived from what it checks.** The three-statement model back-solved
  opening cash from the cash flows, so its cash reconciliation always held, and set
  `cash_reconciled=True` anyway; deriving liabilities as assets minus equity makes A = L + E hold by
  construction. Take each input from the source (the prior-period comparative in the same filing),
  run identity checks on reported numbers only, and record every derived field with its formula.
  Test that the check fails on a perturbed input. The 10-K PDF parser and the Ind AS crosswalk had
  the same hard-coded `cash_reconciled=True`, plus equity substituted by assets minus liabilities.
- **A subtotal made of several lines is their sum, not the first line found.** The Ind AS map took
  "first key present" for cost of goods sold (Schedule III shows materials consumed, purchases of
  stock-in-trade and change in inventories) and for tax (current only, no deferred). Map each
  component, sum the ones present, and record the formula.
- **A content hash identifies content, not meaning.** An XBRL fact record has no concept name, so
  Apple's cash with and without restricted cash are identical records with one hash; a store keyed by
  hash alone kept only one concept. Keep every entry under a hash and match on the citation's tag.
  Value tolerances for source amounts are float rounding only: a relative 1e-9 let $479 through on
  a revenue of $478.8 billion.
- **No silent clipping or fallbacks** that change the answer (e.g. clipping both numerator and
  denominator of a sinh ratio). Use numerically stable forms (`expm1`, `log1p`, exp-ratio forms).
- **Thresholds and ratios: compare with a tolerance.** A value exactly at a limit must pass:
  `2.1 / 0.7` is `3.0000000000000004`; `(3 - 2.7) / 3` is `0.09999999999999994`. Compare on the
  level (e.g. emissions ≤ baseline·(1 − target)) or with a small relative tolerance, and test the
  exact-boundary case. This includes barrier touches in path labelling: `100 * 1.1` is
  `110.00000000000001`, so an exact +10% move was labelled "not touched".
- **Build search grids from integer multiples** (`k * step`), never `np.arange` with a float step:
  `np.arange(0.3, 1.3, 0.3)` includes 1.2 and skips the documented end point 1.0. Do not round
  results to fewer decimals than the step has (0.5625 was reported as 0.562).
- **Use raw statistics, not a callee's built-in verdict.** Combining `is_stationary` (fixed 5%)
  with the caller's `target_p_value=0.10` silently ignored the caller's threshold.
- **Validate inputs at the boundary**: non-finite (NaN/inf), non-positive where undefined,
  zero denominators, empty arrays, invalid probabilities (e.g. CRR up-probability outside [0, 1]).
  Raise `ValueError` with a clear message; never let Z3 "parser error" or `ZeroDivisionError` leak.
- **Order and shape independence**: sort inputs that the maths assumes sorted (strikes), check
  equal lengths, reject duplicates where they break formulas.
- **Map library argument conventions explicitly** and test an asymmetric case. `arch_model(p, q)`
  uses p for the ARCH order, the opposite of Bollerslev's GARCH(p, q); passing them through and
  reading results with `.get(name, 0.0)` made GARCH(1,2) silently return α₂ = 0. Read results by
  key without defaults so a mismatch raises.
- **Use the exact formula, not a shortcut**, and test derived quantities on a simulated process
  where they differ: OU half-life is −ln2/ln φ, not ln2/(1−φ) (27% high at φ = e^−0.5).
- **Use published coefficients exactly** and cite them (the Beneish TATA/LVGI coefficients were
  wrong, including a sign). A test should pin the constant (e.g. all indices = 1 ⇒ known value).
- **Assert the reason, not just the outcome.** `assert "REJECTED" in result.pm_decision` passed for a
  firm meant to show manipulation, but it was rejected because its cash flow statement did not
  reconcile. Assert the specific verdict (`startswith("REJECT: manipulation")`) and the inputs'
  consistency, so a test cannot pass for the wrong reason.
- **No invented inputs in examples.** Proxies such as retained earnings = 0.5 × equity, a fixed 70/30
  Scope 1/2 split, or unsourced "~350k tCO2e" for a real company produce plausible numbers that mean
  nothing. Use reported figures, or label inputs as illustrative where they are printed and quoted.
  Never put a real company's name on synthetic data.
- **Required inputs over magic defaults** when a default would silently mean the wrong units
  (e.g. `arrival_price` is keyword-only and required).

## 3. Z3 and "verification" language

All Z3 variables here are bound to concrete inputs, so a `SAT` result equals exact rational
arithmetic on the decimal inputs. It verifies the given numbers; it does not prove a model over
ranges of inputs, and there is no unsat-core extraction. Covenant checks are plain arithmetic.
Describe this accurately; never write "proven", "formally guaranteed", "audited" or
"zero hallucination" about this behaviour.

## 4. Docs, papers and claims

- Every number in README, docs, papers or changelog must come from code run in the session (cite
  the script) or a primary source. If you cannot measure it, say so; never estimate or invent
  results, latencies, scores or citations.
- The README capability table marks each feature ✅ implemented / 🟡 partial / 🗺️ planned. Do
  not advertise planned work as done; keep README, `docs/`, `ROADMAP.md` and this skill consistent.
- **Verify every reference against a primary source** (publisher, ACL Anthology, DBLP, NeurIPS/ICLR proceedings, RePEc).
  The papers carried a non-existent "FinBench" (arXiv:2311.12345, a placeholder id), a citation to a fictional
  "Līlāvatī Benchmark Consortium", wrong years (Toolformer, ReAct) and a wrong author list (TAT-QA). Never cite a
  source you have not looked up; if a detail (e.g. page range) cannot be confirmed, give the DOI instead.
- **Totals quoted from a table must be recomputed from the results file**, not carried over from an
  earlier run: the SEC pilot was quoted as "64 of 64" in four places while its table and
  `results/connectors/sec_filings_pilot.json` said 71 of 71. A page that says it is "generated" or
  "checked" needs a test that does so (`tests/test_mcp_registry.py` compares the MCP tool table
  in `docs/mcp-setup.md` with the registered tools).
- Record audit results in `docs/validation/<engine>.md` (check, reference, measured tolerance,
  status pass / fixed / open) and update `docs/validation/index.md`.
- Benchmark results: report provider failures (429/503/timeouts/empty replies) separately from
  wrong answers; include a content-free baseline; publish raw outputs in `results/`.
- Never re-roll model failures. `--resume` once re-ran unparsable answers along with provider errors,
  lifting a score from 0.4417 to 0.4667. Re-run only errors and empty replies; record
  `finish_reason` so truncation is distinguishable from a wrong answer.
- **Results are evidence: publish and never edit them.** `results/**/*.jsonl` are tracked; each run
  script writes `<out>.meta.json` (settings, prompt version, git commit); older runs are described
  in `results/legacy_metadata.json` (`null` when unknown, never a guess);
  `scripts/build_results_manifest.py` writes `results/manifest.json` and
  `tests/test_results_manifest.py` fails if it is stale. Quote numbers from `results/ANALYSIS.md`
  (`scripts/analyse_results.py`). Keep superseded runs, labelled, rather than deleting them.
- **Case ids are labels, content hashes are identity.** `generate_cases(2026, 2)` and `(2026, 4)` both
  produced `GEN-T2-000` with different bonds. Ids now carry seed and size (`GEN-S2026-N4-T2-000`),
  records store `case_hash`, and comparisons pair on the hash (`analysis.paired_difference`).
- **Identities must be platform-independent.** Never hash floats produced by exp/log/erf: their last digits
  differ between math libraries (the v1 case hash differed between macOS and Linux CI). Hash inputs, not
  computed outputs, and pin a known hash in a test so CI on another OS catches drift.
- **Statement parsing: labels, rows, gaps.** Match a label at the start of a line before matching it as a substring
  ("Income before provision for income taxes" contains "provision for income taxes"); take a row's trailing
  amounts across wrapped lines (a par value inside a label became "common equity"); skip section headers ending
  in ":"; sum repeated rows (current and non-current term debt); report items not found
  (`missing_items`) instead of zero-filling them. Check parsers line by line against a real filing.
- **SEC figures: anchor on the filing, not on `fy`.** A 10-K tags its prior-year comparatives with its own fiscal
  year; selecting by `fy` returned Apple's FY2024 revenue as FY2026. Use the submissions index, then the facts
  with that accession, end date and duration (`connectors/sec_filing.py`). Prefer XBRL over text parsing.
- **HTML to text: inline tags join, block tags separate.** Microsoft's 10-K splits "INCOME" across spans; replacing
  every tag with a space hid the heading. Anchor on headings followed by figures, not the contents page.
- **When every model fails the same way, check the answer key.** All four models "missed" Walmart's revenue
  identically; the key used net sales where the prompt asked for total revenue. Fix the key and re-score the
  stored replies; do not re-ask.
- **Fixtures that quote real data must be real.** A test fixture labelled "as extracted from Apple's 10-K" once
  contained an invented line; copy excerpts verbatim from the source or label the data synthetic.
- **External services need real contacts, supplied by the user.** SEC EDGAR requires a contact in the User-Agent;
  never invent one (`admin@lilavati.ai` was) and never use the maintainer's email without asking; read it from
  `LILAVATI_SEC_USER_AGENT`.
- **Read the prompt before blaming the model.** Prompt v2 defined `bond_price` as "price of the 5-year
  bond" for generated 2-10-year bonds; gpt-oss-120b priced a 10-year bond as 5-year and was scored
  wrong. Bump `llm_eval.PROMPT_VERSION` whenever the model-visible text changes, and never mix versions.
- **Tool-arm results measure dispatch, not engines.** Expected values come from the engines the tools
  call; say so wherever tool-arm accuracy is reported. A provider rejecting a malformed tool call
  (Groq HTTP 400 `tool_use_failed`) is `model_error`, not a provider failure.
- Answer parsers must handle inline reasoning: strip `<think>...</think>` before extracting JSON.
  A greedy first-`{`-to-last-`}` match failed on braces inside `qwen3.6-27b` reasoning. After a
  parser fix, re-score stored replies (`--rescore`) instead of calling the model again.
- **Markdown lists need a blank line before them** (Python-Markdown, unlike GitHub): a list right
  after a paragraph line renders as one run-on paragraph. Do not patch with `<br>`; add the blank
  line. Mermaid diagrams render only through the `mermaid` custom fence in `mkdocs.yml`; use
  `<br/>` for line breaks inside node labels, not `\n`.
- No marketing language ("institutional-grade", "first", "definitive"). The project is alpha.
- **Claims are linted.** `scripts/check_claims.py` fails on proof, guarantee, "audited", maturity and
  priority language in docs, docstrings and printed text unless the line negates it or carries a
  reviewed `claims-ok` marker. Never reword published benchmark case text to satisfy it (that changes
  the case hash): mark it `claims-ok` with the reason.
- **Describe what is tested, not what is planned.** README listed purged cross-validation, tranche
  waterfalls and VWAP as benchmark content; none existed. Feature tables need a "today" and a "planned" column.
- **No third-party scripts without need.** The docs loaded polyfill.io, whose domain served malware after a 2024
  ownership change. Load scripts only from pinned, reputable CDNs, and only when required.
- **Tutorial code is executed.** Titled fences (`python title="file.py"`) build files; untitled blocks
  run against them; `tests/test_*.py` titles run under pytest (`scripts/check_tutorials.py`).
- **Every user-facing feature ships with a tutorial** in `docs/tutorials/NN_<topic>.md`, added to the
  `mkdocs.yml` nav and, if relevant, to the course schedule (`docs/course/index.md`). Structure:
  who it is for and time/cost; learning objectives; runnable steps (commands and code you ran);
  "what can go wrong", tied to real failures found in this repo; "Using this with Claude Code"
  (which skill); exercises; notes for instructors. Link measured numbers to `results/ANALYSIS.md`
  or a validation page instead of copying them, so they cannot drift. Run every code block that
  needs no API key before publishing the tutorial.
- **Before rewriting an example, grep `tests/` for importers** (`tests/test_tutorial_agents.py` imports examples 07–09).
- **When you correct a claim, grep tutorials and examples too.** "Z3 proves covenants", an "equity
  cure" that was really debt paydown, and `serve_agent(host="0.0.0.0")` were corrected in the docs and
  code but survived in Tutorial 01 and `examples/06` until 2026-11-22. Search `docs/tutorials/`,
  `examples/` and `docs/index.md` for the retired wording in the same change.
- User-facing workflows also get a skill (`.claude/skills/<name>/SKILL.md`) so AI agents follow
  the same rules as the tutorial (example: `lilavati-bench` for running and analysing benchmarks).

## 5. Secrets, network and LLM clients

- **Packaging: check what ships.** Hatch honours `.gitignore` but not `.git/info/exclude`, so a local
  `uv build` put the private `.agents/tasks/` briefs into the sdist. Exclusions live in
  `[tool.hatch.build.targets.sdist]`; list `tar tzf dist/*.tar.gz` before any release. Releases go
  through `.github/workflows/release.yml` (Trusted Publishing, clean-install smoke test), never a
  token on a laptop.

- API keys only from environment variables. Never put a key in a file, test, fixture, log, commit
  or URL query string; send it in a header (`x-goog-api-key`, `x-api-key`, `Authorization`).
  Before committing: `git diff --cached | grep -E "AIza[0-9A-Za-z_-]{30}|sk-[A-Za-z0-9_-]{20}|gsk_[A-Za-z0-9]{20}"`
  must be empty (Gemini, OpenAI/Anthropic, Groq key shapes). If a key is pasted into chat, pass it
  only as an environment variable on the command (`GROQ_API_KEY=... uv run ...`), never write it to
  `.env` or a script, and tell the maintainer to rotate it.
- Error handling must not echo request URLs or headers.
- Respect provider rate limits (bounded `--max-requests`, resumable runs). The client retries
  429/5xx with backoff (`ModelSpec.max_retries`); scripts with their own retry loop set it to 0.
- **Every provider path needs a request/response round-trip test** against the provider's
  documented format (`tests/test_llm_tool_calling.py`, `httpx.MockTransport` via the client's
  `transport=` argument) **and an opt-in live test** (`tests/integration/`, `LILAVATI_LIVE_TESTS=1`).
  Gemini and Anthropic silently dropped `tools` because only the mock LLM was ever exercised.
- **Framework integrations go through MCP, in the framework's own environment if needed.**
  `langchain-mcp-adapters` requires `mcp<2` but fastmcp 4 needs `mcp>=2.3`, so the LangGraph
  example runs with `uv run --no-project --with ...` and starts the servers as subprocesses. Do
  not pin around such conflicts inside the project; check `uv tree --invert --package <pkg>`.
- **Integration examples must prove tool use**: assert the final answer quotes the tool-returned
  numbers (`examples/integrations/_task.py::check_answer`), and report flaky runs honestly.
- Free-tier Gemini endpoints return 503 "high demand" often: use generous backoff
  (ADK: `GenerateContentConfig(http_options=HttpOptions(retry_options=HttpRetryOptions(...)))`).
- Groq (OpenAI-compatible, `GROQ_BASE_URL`): it validates tool-call arguments server side and returns
  HTTP 400 `tool_use_failed` for a malformed call (`qwen3.8-27b` failed the live tool test 2 of 3 runs);
  that is a model failure, not a client bug. Temp 0 is not deterministic there, so report run-to-run spread.
- Gemini: put system prompts in `systemInstruction`, sanitise JSON Schema to its OpenAPI subset
  (no `title`, `default`, `additionalProperties`), and echo `thoughtSignature` on function-call
  parts. Anthropic: roles must alternate, so merge parallel `tool_result` blocks into one user turn.

## 6. Repository map (verify before relying on it)

| Area | Module | Notes |
| :--- | :--- | :--- |
| Provenance / bi-temporal | `core/provenance.py`, `core/temporal.py` | `CitationRef`, `SourcedValue`, `SourceStore` + `ProvenanceGate(store)` (citation resolution), DuckDB as-of queries |
| Invariants | `core/invariants.py` | Z3 identity checks; covenant arithmetic |
| Fixed income | `core/fixed_income.py` | bonds (price, durations, convexity, DV01), Nelson-Siegel, Merton |
| Derivatives | `core/derivatives.py` | BSM + Greeks, CRR, implied vol, SABR (Hagan 2002), butterfly check |
| Econometrics | `core/econometrics.py` | ADF, Engle-Granger + OU half-life, GARCH(p,q) (audited) |
| Financial ML | `core/financial_ml.py` | fractional differencing, d* search, triple barrier (audited); no purged CV yet |
| Execution / portfolio | `core/execution.py` | LOB snapshot, Almgren-Chriss (audited), Black-Litterman, HRP (not yet audited) |
| Sustainability | `core/sustainability.py` | WACI, carbon-tax stress, SLB step-up |
| Forensics | `agents/forensics.py` | Beneish (1999), Altman (1968) |
| MCP servers | `mcp_servers/*_server.py` | 8 servers, `lilavati-mcp-*` scripts, `docs/mcp-setup.md` |
| Filing connectors | `connectors/sec_filing.py`, `connectors/edgar.py`, `connectors/pdf_10k.py`, `connectors/india_filings.py` | filing-anchored XBRL (`extract_filing`, `fetch_filing_extraction`, MCP `get_filing_figures`); `EdgarClient` (contact from `LILAVATI_SEC_USER_AGENT`, `resolve_cik(online=True)`); three-statement model `statement_model_from_filing` (reported / `derived` / `not_reported`, raises on missing required items); validation `docs/validation/connectors.md` |
| Integrations | `examples/integrations/` | LangGraph, ADK, A2A examples; `--check` verifies answers quote tool outputs; `adk` extra |
| Benchmark | `bench/` | golden cases, seeded generator (`case_content_hash`, `legacy_case_id`), runner, evaluators. `lilavati-bench run` only runs the reference solver (`--model` is a label) |
| LLM evaluation | `bench/llm_eval.py`, `bench/tool_eval.py`, `bench/analysis.py` | prompts (`PROMPT_VERSION`), parsing, outcome classes; tool arm with grounding; summaries, bootstrap CIs, paired differences. Scripts: `run_llm_baseline.py`, `run_tool_baseline.py`, `build_results_manifest.py`, `analyse_results.py` |
| Tutorials and course | `docs/tutorials/`, `docs/course/index.md` | one tutorial per user-facing feature; user skill `.claude/skills/lilavati-bench/` |

The PyPI distribution is `lilavati-finance` (the name `lilavati` belongs to another project); the
import name is `lilavati`.

## 7. Keeping this skill current

Update this file in the same change whenever you:
- fix a bug whose cause is a general pattern (add a one-line rule to section 2 or 4 with the
  concrete example),
- add, remove or audit an engine or MCP server (update section 6 and the README table),
- change a workflow command or dependency (update section 1).
Keep rules short and concrete. Remove rules that no longer apply. Add an entry below.

### Change log
- 2026-11-19: Created from the alpha review and engine audit (Beneish coefficients, Almgren-Chriss
  units/clipping, Merton convergence, SABR near-ATM, float boundaries in covenants/SLB, Z3 claims,
  PyPI name collision, benchmark honesty).
- 2026-11-19: Added rules from the financial ML audit (placeholder results, float-step grids,
  barrier-touch tolerance, raw statistics over callee verdicts).
- 2026-11-19: Added rules from the econometrics audit (library argument mapping, exact formulas,
  testing old code under the repo's pytest pythonpath).
- 2026-11-19: Added LLM-client rules after fixing Gemini/Anthropic tool calling (round-trip and live
  tests, provider format notes, retries, limits must not look like success).
- 2026-11-20: Added integration rules (MCP-first framework examples, mcp version conflict, answer
  checks, Gemini 503 backoff) after LangGraph/ADK/A2A examples.
- 2026-11-20: Added Groq presets and `--provider groq`; Groq key-pattern secret check; no re-rolling
  of unparsable answers on `--resume`; `finish_reason` recorded; Groq tool-call/determinism notes.
- 2026-11-20: Inline-reasoning parsing rule and `--rescore` after the `<think>` brace bug.
- 2026-11-20: Repo map notes that `lilavati-bench run` is reference-solver only (docs had implied otherwise).
- 2026-11-21: Tutorial-per-feature rule and user skills; results-as-evidence rules (manifest, meta files,
  no edits); content-hash case identity after the generated-id collision; prompt versioning after the
  '5-year bond' key bug; tool-arm interpretation; Ollama preset needed `/v1`.
- 2026-11-22: Grep tutorials/examples when correcting a claim (Tutorial 01 retrofit).
- 2026-11-22: Claims linter, tutorial-code checker, CI docs/examples/tutorials jobs, docs and release
  workflows; sdist exclude after `.agents/tasks/` was found in a local sdist.
- 2026-11-22: Tutorials 02–04 retrofit: assert the reason, no invented inputs or real names on synthetic data, grep tests before rewriting examples.
- 2026-11-22: Reference-verification rule after the paper reference audit.
- 2026-11-22: Platform-independent case hashes after Linux CI found the v1 hash differed from macOS.
- 2026-11-22: Onboarding docs; tested-vs-planned and third-party-script rules.
- 2026-11-22: 10-K parser live validation rules; real fixtures; user-supplied service contacts.
- 2026-11-23: SEC filing-anchoring, HTML text, and answer-key rules (Tutorial 08).
- 2026-11-23: Added the Markdown list and Mermaid rendering rule (lists on the docs home and
  architecture pages ran together; the architecture diagram showed as code).
- 2026-11-23: Removed the outdated "generic project contact" SEC rule (section 5 now reads the contact
  from `LILAVATI_SEC_USER_AGENT`); added connectors and MCP-table rules after a docs/code sweep found
  a total quoted wrong in four places (64 instead of 71) and an MCP tool table that had drifted.
- 2026-11-23: Added the no-fabricated-fallbacks rule (search and research agents, XBRL citation page
  and hash).
- 2026-11-23: Rule against checks on derived inputs, after the three-statement model back-solved
  opening cash and hard-coded `cash_reconciled=True`; repo map updated for `statement_model_from_filing`.
- 2026-11-23: Subtotal-components rule (Ind AS cost of goods, tax); PDF parser and Ind AS crosswalk
  examples added to the derived-inputs rule; `india_filings.py` in the repo map.
- 2026-11-23: Content-hash rule after building `SourceStore` (identical XBRL records share a hash;
  tolerance for source amounts); repo map lists `SourceStore`.
