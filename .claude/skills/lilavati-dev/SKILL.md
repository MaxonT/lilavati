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
   uv run ruff check src tests scripts
   uv run mypy src
   uv run pytest
   ```
   If you touched docs: `uv run --extra docs --extra dev mkdocs build --strict`.
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
- Record audit results in `docs/validation/<engine>.md` (check, reference, measured tolerance,
  status pass / fixed / open) and update `docs/validation/index.md`.
- Benchmark results: report provider failures (429/503/timeouts/empty replies) separately from
  wrong answers; include a content-free baseline; publish raw outputs in `results/`.
- No marketing language ("institutional-grade", "first", "definitive"). The project is alpha.

## 5. Secrets, network and LLM clients

- API keys only from environment variables. Never put a key in a file, test, fixture, log, commit
  or URL query string; send it in a header (`x-goog-api-key`, `x-api-key`, `Authorization`).
  Before committing: `git diff --cached | grep -E "AIza[0-9A-Za-z_-]{30}|sk-[A-Za-z0-9]{20}"` must be empty.
- Error handling must not echo request URLs or headers.
- SEC EDGAR requests need a compliant User-Agent; use a generic project contact, never a personal email.
- Respect provider rate limits (bounded `--max-requests`, resumable runs). The client retries
  429/5xx with backoff (`ModelSpec.max_retries`); scripts with their own retry loop set it to 0.
- **Every provider path needs a request/response round-trip test** against the provider's
  documented format (`tests/test_llm_tool_calling.py`, `httpx.MockTransport` via the client's
  `transport=` argument) **and an opt-in live test** (`tests/integration/`, `LILAVATI_LIVE_TESTS=1`).
  Gemini and Anthropic silently dropped `tools` because only the mock LLM was ever exercised.
- Gemini: put system prompts in `systemInstruction`, sanitise JSON Schema to its OpenAPI subset
  (no `title`, `default`, `additionalProperties`), and echo `thoughtSignature` on function-call
  parts. Anthropic: roles must alternate, so merge parallel `tool_result` blocks into one user turn.

## 6. Repository map (verify before relying on it)

| Area | Module | Notes |
| :--- | :--- | :--- |
| Provenance / bi-temporal | `core/provenance.py`, `core/temporal.py` | `CitationRef`, `SourcedValue`, DuckDB as-of queries |
| Invariants | `core/invariants.py` | Z3 identity checks; covenant arithmetic |
| Fixed income | `core/fixed_income.py` | bonds (price, durations, convexity, DV01), Nelson-Siegel, Merton |
| Derivatives | `core/derivatives.py` | BSM + Greeks, CRR, implied vol, SABR (Hagan 2002), butterfly check |
| Econometrics | `core/econometrics.py` | ADF, Engle-Granger + OU half-life, GARCH(p,q) (audited) |
| Financial ML | `core/financial_ml.py` | fractional differencing, d* search, triple barrier (audited); no purged CV yet |
| Execution / portfolio | `core/execution.py` | LOB snapshot, Almgren-Chriss (audited), Black-Litterman, HRP (not yet audited) |
| Sustainability | `core/sustainability.py` | WACI, carbon-tax stress, SLB step-up |
| Forensics | `agents/forensics.py` | Beneish (1999), Altman (1968) |
| MCP servers | `mcp_servers/*_server.py` | 8 servers, `lilavati-mcp-*` scripts, `docs/mcp-setup.md` |
| Benchmark | `bench/` | golden cases, seeded generator, runner, evaluators; `scripts/run_llm_baseline.py` |

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
