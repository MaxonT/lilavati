# Changelog

## [Unreleased]
### Fixed
- Beneish M-score used wrong coefficients (TATA 4.037 → 4.679; LVGI +0.0327 → −0.327, per Beneish 1999). Scores quoted in docs refreshed.
- Almgren-Chriss schedule: clipping bug made very risk-averse schedules hold all shares to the end; volatility now in price units; discrete-time solution with η̃ = η − γτ/2; adds expected cost and variance (removes a stray 0.5 factor in variance).
### Changed (breaking)
- `ExecutionEngine.compute_almgren_chriss_schedule` and the matching MCP tool now require keyword `arrival_price`; impact parameters use the paper's per-day units.
- Distribution renamed to `lilavati-finance` (the PyPI name `lilavati` belongs to an unrelated project). Import name is still `lilavati`.
### Added
- `lilavati-mcp-*` launch commands for all 8 MCP servers, `docs/mcp-setup.md`, and a server registry test.
- `CLAUDE.md` / `AGENTS.md` guidance for AI coding agents.
### Docs
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
