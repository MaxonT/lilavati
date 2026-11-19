# Changelog

## [Unreleased]
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
### Changed (breaking)
- `ExecutionEngine.compute_almgren_chriss_schedule` and the matching MCP tool now require keyword `arrival_price`; impact parameters use the paper's per-day units.
- Distribution renamed to `lilavati-finance` (the PyPI name `lilavati` belongs to an unrelated project). Import name is still `lilavati`.
### Added
- `lilavati-mcp-*` launch commands for all 8 MCP servers, `docs/mcp-setup.md`, and a server registry test.
- `CLAUDE.md` / `AGENTS.md` guidance for AI coding agents.
### Docs
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
