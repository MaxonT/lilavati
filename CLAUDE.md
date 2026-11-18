# Līlāvatī: guidance for AI coding agents

Read `.agents/skills/lilavati-dev/SKILL.md` first (rules on provenance, SMT invariants, no LLM arithmetic).

## Commands
```bash
uv sync --extra dev
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
```
All three must pass before you call a change done.

## Rules of thumb
- Tests for numeric code must compare against an independent reference (textbook example, paper, QuantLib/statsmodels), not against the implementation itself.
- Never state a number, latency, score or citation in docs/papers that you did not produce by running code or reading a primary source. The project is alpha; avoid marketing language.
- API keys come only from environment variables. Never write them into files, tests, logs, URLs or commits.
- Do not commit or push unless the maintainer asks. Use Conventional Commit messages.
- Import package is `lilavati`; the PyPI distribution name is `lilavati-finance`.
