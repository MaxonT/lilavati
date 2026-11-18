# Changelog

## [0.1.0-alpha] - Unreleased
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
