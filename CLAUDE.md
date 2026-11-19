# Līlāvatī: guidance for AI coding agents

**Before any code, test or doc change, load the `lilavati-dev` skill**
(`.claude/skills/lilavati-dev/SKILL.md`). It holds the repo's coding, testing and documentation rules.

**Keep the skill current.** When a bug, review or audit reveals a general pattern, or you add, remove or
audit an engine or MCP server, update the skill in the same change (rule + concrete example, and a
line in its change log). Outdated rules are worse than none: remove ones that no longer apply.

## Commands
```bash
uv sync --extra dev
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
uv run --extra docs --extra dev mkdocs build --strict   # when docs change
```

## Non-negotiables (details in the skill)
- Numeric code is tested against independent references (papers, textbooks, QuantLib), not itself.
- No number or capability in docs/papers that you did not produce by running code or read in a primary source.
- API keys only via environment variables; never in files, logs, URLs or commits.
- Do not commit or push unless the maintainer asks. Import package `lilavati`; PyPI distribution `lilavati-finance`.
