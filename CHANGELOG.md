# Changelog

## [Unreleased]

## [0.2.0] - release candidate (not yet tagged; see docs/releases/v0.2.0.md)
### Fixed
- Beneish M-score used wrong coefficients (TATA 4.037 → 4.679; LVGI +0.0327 → −0.327, per Beneish 1999). Scores quoted in docs refreshed.
- Almgren-Chriss schedule: clipping bug made very risk-averse schedules hold all shares to the end; volatility now in price units; discrete-time solution with η̃ = η − γτ/2; adds expected cost and variance (removes a stray 0.5 factor in variance).
- Merton solver could return non-solutions (equation errors up to 90%) for distressed / long-horizon inputs; now a bracketed 1-D solve with a residual check. Invalid inputs raise `ValueError`.
- Derivatives: SABR near-ATM skew term dropped (now matches QuantLib to ~1e-12); `calibrate_sabr` failed for index-level forwards; `is_arbitrage_free` was only an RMSE test (now checks butterfly and call-spread arbitrage of fitted prices); butterfly check depended on strike order; CRR tree silently mispriced when the up-probability left [0, 1]; non-finite / degenerate inputs now raise `ValueError`.
- Covenant check reported a false breach for ratios exactly at the limit (float rounding); non-finite inputs to invariant checks now raise a clear error.
- SLB covenant could report a missed target when met exactly (float rounding); WACI rejects zero-sum / negative weights; `total_carbon_footprint` relabelled (not PCAF financed emissions).
- Econometrics: GARCH p/q were passed to `arch` swapped (GARCH(1,2) returned α₂ = 0); OU half-life used the ln2/(1−φ) shortcut (27% high at φ = e^−0.5), now −ln2/ln φ; unconverged GARCH fits raise instead of returning start values; collinear/constant/non-finite inputs and invalid significance levels raise. Example/tutorial half-lives refreshed (2.74 → 2.38 days).
- Financial ML: fractional-d search returned a placeholder d*=1.0, p=1.0 when nothing qualified (now raises); ignored the caller's target p-value at d=0; float grid could exceed 1.0; d* rounded below step precision; exact barrier touches missed through float rounding; invalid inputs (NaN, zero/negative prices or horizon) now raise.
- LLM client: Gemini and Anthropic paths accepted `tools` but never sent them or parsed tool calls, so agents never used tools on those providers. Now implemented (Gemini functionDeclarations / functionCall / functionResponse incl. thoughtSignature echo and schema sanitising; Anthropic tool_use / tool_result with role merging). Bounded retries with backoff on 429/5xx/timeouts (`ModelSpec.max_retries`).
- Agent runtime: `max_reflection_iterations` also capped tool rounds, so multi-step tool plans ended with an empty answer while reporting success; now a separate `max_tool_rounds` and an `AgentOutput.completed` flag. Tool errors are flagged to the model (`is_error`).
- `run_llm_baseline.py --resume` re-ran unparsable answers as well as provider errors, re-rolling model failures and inflating scores; it now re-runs only errors and empty replies. Each record now stores the model's `finish_reason`, and summaries count `truncated` replies separately.
- `run_llm_baseline.py` scored answers unparsable when a model's inline `<think>` reasoning contained braces (2 `qwen3.6-27b` answers); reasoning blocks are now stripped first. New `--rescore` re-scores stored replies without API calls.
- Benchmark: generated case ids collided across set sizes (`generate_cases(2026, 2)` and `(2026, 4)` both had `GEN-T2-000`, different bonds); ids now carry seed and size (`GEN-S2026-N4-T2-000`), cases have a content hash (`case_content_hash`), and old ids resolve via `legacy_case_id`. Generated content is unchanged.
- Benchmark prompt: the `bond_price` key said "price of the 5-year bond" for generated bonds of 2-10 years (gpt-oss-120b priced a 10-year bond as 5-year); prompt v3 fixes it and every record stores `prompt_version`.
- Ollama preset pointed at the server root instead of its OpenAI-compatible `/v1` endpoint.
- `case_content_hash` included expected answer values, whose last digits depend on the platform's math library, so hashes differed between macOS and Linux CI; it now hashes only the question (scheme v2), and stored hashes were recomputed by `scripts/migrate_case_hashes.py` without changing any other field.
- `paired_difference` ordered pairs by content hash, so the seeded bootstrap CI changed when the hash scheme did; pairs are now ordered by case id (published Qwen3.8 bounds moved from +0.1083/+0.6875 to +0.1125/+0.6917; bootstrap Monte Carlo error is about ±0.01).
### Changed (breaking)
- `ExecutionEngine.compute_almgren_chriss_schedule` and the matching MCP tool now require keyword `arrival_price`; impact parameters use the paper's per-day units.
- Distribution renamed to `lilavati-finance` (the PyPI name `lilavati` belongs to an unrelated project). Import name is still `lilavati`.
- Generated benchmark case ids now carry the seed and set size (`GEN-T2-000` → `GEN-S2026-N4-T2-000`); match results by `case_content_hash`, or resolve old ids with `legacy_case_id`.
- Benchmark prompt helpers moved from `scripts/run_llm_baseline.py` into `lilavati.bench.llm_eval`.
### Added
- `lilavati-mcp-*` launch commands for all 8 MCP servers, `docs/mcp-setup.md`, and a server registry test.
- `CLAUDE.md` / `AGENTS.md` guidance for AI coding agents.
### Added (integrations)
- `examples/integrations/`: LangGraph (`create_react_agent` + `langchain-mcp-adapters`), Google ADK (`McpToolset`) and A2A server/client examples using Līlāvatī's MCP servers, each with a `--check` that the answer quotes tool outputs. New `adk` optional extra. Presets for `gemini-4-neon`, `gemini-4-hydrogen`, `gemma-4-31b-it`.
### Added (providers)
- Groq presets (`groq-gpt-oss-120b`, `groq-gpt-oss-20b`, `groq-qwen3.6-27b`, `groq-qwen3.8-27b`) via the OpenAI-compatible path; baselines also for `allam-2-7b`; key read only from `GROQ_API_KEY`. `run_llm_baseline.py --provider groq`. Mocked header/URL test and opt-in live test (`LILAVATI_LIVE_GROQ_PRESET`). Baselines in `results/README.md`.
### Added (evaluation and teaching)
- `lilavati.bench.llm_eval` (prompts, parsing, outcome classes incl. `model_error`), `lilavati.bench.tool_eval` (tool arm, `submit_answer`, tool grounding), `lilavati.bench.analysis` (summaries, bootstrap CIs, paired differences); scripts `run_tool_baseline.py`, `build_results_manifest.py`, `analyse_results.py`. Run scripts write `<out>.meta.json`.
- Results published: `results/**/*.jsonl` tracked, `manifest.json`, `ANALYSIS.md`, CC BY 4.0 data licence, manifest/secret test. Tool-arm pilot (t1) kept in `results/archive_tools_pilot_t1/`.
- Tutorials 05-07, draft course `docs/course/index.md`, user skill `.claude/skills/lilavati-bench/`; tutorial-per-feature rule in the dev skill and CONTRIBUTING.
### Added (quality and release)
- `scripts/check_claims.py` (claims linter) and `scripts/check_tutorials.py` (runs tutorial code as written); CI now also lints `scripts/` and `examples/` and runs the strict docs build, every example, tutorial code, claims and manifest checks.
- `.github/workflows/release.yml` (PyPI Trusted Publishing after a clean-install smoke test) and `.github/workflows/docs.yml` (GitHub Pages).
### Fixed (packaging and claims)
- A local `uv build` included the private `.agents/tasks/` briefs in the sdist (hatch ignores `.git/info/exclude`); now excluded explicitly, with a test.
- Proof, guarantee, "audited" and maturity language removed from docs, papers, docstrings, prompts, printed example output and the API server's default description; the benchmark paper's comparison table marks Līlāvatī's point-in-time and provenance support as partial.
### Docs
- Paper references verified against primary sources: removed a non-existent "FinBench" entry and a citation to a fictional "Līlāvatī Benchmark Consortium"; corrected TAT-QA authors and title, Garcez et al. venue, Toolformer and ReAct years, Almgren-Chriss year and DOI; added missing Lean and Isabelle entries. README and framework-paper heritage passages reworded without priority or anachronistic claims.
- Tutorials 02–04 rewritten to the tutorial standard and executed in CI; examples 07–09 rewritten to match. Tutorial 02 now validates its estimates on a simulated pair (true half-life 2.41 days, estimated 2.38) and shows a short-sample half-life underestimate; Tutorial 03 checks statement consistency before scoring (its earlier "manipulation" example had inconsistent statements) and uses reported retained earnings; Tutorial 04 uses explicit Scope 1/2 inputs and shows HRP overweighting the most carbon-intensive asset.
- Framework paper: case studies traced to their scripts (6.3 figures replaced by the credit example's output), latency re-measured, illustrative inputs labelled, reference-pipeline metrics labelled as by construction. Benchmark paper: ConvFinQA venue (EMNLP 2022), PIXIU task count and two reference entries corrected.
- Tutorial 01 rewritten to the tutorial standard: checks both covenants (the stressed example also fails interest coverage), reports debt reduction to comply instead of a mislabelled equity cure, no "Z3 proof" language for plain arithmetic, localhost-only serving, real outputs and runnable tests; `examples/06_build_custom_agent_and_api.py` rewritten to match.
- README: "Why Līlāvatī" names the models and uses prompt-v3 results; new "Who it is for" persona map and "Towards enterprise quality" table. `TODO.md` holds the Phase 8–12 plan (summary in ROADMAP).
- `docs/integrations.md`; README repositioned as a finance tool layer for agent frameworks.
- README, `docs/benchmarks.md`, benchmark paper §5.2, architecture, roadmap and CONTRIBUTING updated for the Groq presets and baselines; benchmark docs now describe the error / truncated / unparsable classification, `--resume` and `--rescore`. Corrected `lilavati-bench run` docs: it always runs the reference solver (`--model` is a label) and has no `--output` option.
- Numerical validation pages per engine (`docs/validation/`); QuantLib added as a test-only dev dependency for reference values.
- Corrected claims of Z3 covenant proofs, SMT-derived equity cures and unsat-core feedback; latency table re-measured.
- README capability table (implemented / partial / planned); roadmap, docs index and dev skill corrected to match the code; Phase 7 (verification & validation gate before announcement) added.

## [0.1.0-alpha] - 2026-11-17
### Added
- Deterministic engines (fixed income, econometrics, financial ML, derivatives, execution, sustainability), Z3 invariants, provenance and bi-temporal layers.
- FastMCP servers, agent framework, adversarial committee, Līlāvatī-Bench harness, tutorials and examples.
### Added (benchmark)
- Templated case generator (`bench/generator.py`) with balanced True/False answers; CI guard that a content-free answerer scores <35%.
- `scripts/run_llm_baseline.py` direct-prompt baseline and first measured results for four Google models (`results/`).
### Changed
- Gemini API key now sent via `x-goog-api-key` header (not URL); `ModelSpec.timeout_seconds` added; thought parts are excluded from Gemini output.
- API server binds to 127.0.0.1 by default.
- Draft papers relabelled: third-party model scores are illustrative placeholders, not measured results.
- Track 4 benchmark threshold aligned between dataset, runner and docs (r >= 0.25 on the synthetic case).
