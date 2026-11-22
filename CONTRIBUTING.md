# Contributing to Project Līlāvatī

Thank you for your interest in contributing to **Project Līlāvatī**! 🪷

Līlāvatī is an open-source, neuro-symbolic agentic operating system and benchmark suite designed for institutional finance. Our goal is to bring mathematical rigor to financial AI: numbers computed by reference-tested code rather than by the model, and accounting identities checked by Z3 on the numbers given.

Whether you are a quantitative researcher, an academic economist, a software engineer, or a financial analyst, your contributions are invaluable. Please review our [Development Roadmap & To-Do List](ROADMAP.md) to see active priorities.

---

## 🏛️ The Non-Negotiable Core Principles

Before writing any code or proposing a pull request, every contributor (human or AI agent) must adhere to these four core design principles:

1. **Zero LLM Arithmetic:**
   * An LLM must **never** perform mathematical calculations within prompt text or output generation.
   * Calculations (ratios, valuations, yield curve fitting, GARCH estimation) must be delegated to deterministic Python/Polars/DuckDB/FastMCP servers.
2. **Cryptographic Citation Provenance:**
   * Any data extracted from corporate disclosures (10-K, 10-Q, Ind AS) must be wrapped in a `CitationRef` containing the document hash, filing date, page, table ID, and XBRL tag.
3. **Bi-Temporal Point-in-Time (PIT) Isolation:**
   * All historical analysis must enforce strict separation between `valid_time` (the period reported) and `as_of_timestamp` (the publication date). Look-ahead bias and restatement leakage will cause test failures.
4. **Formal Invariant Verification (`Suvarna` Engine):**
   * Financial statements and valuation models must respect first-order logic invariants ($Assets = Liabilities + Equity$, non-negative variance, arbitrage-free surfaces) validated via `z3-solver`.

---

## 🛠️ Development Setup

We use [`uv`](https://docs.astral.sh/uv/) for Python package management. Ensure Python `>=3.11` is installed.

```bash
# 1. Clone the repository
git clone https://github.com/revendrat/lilavati.git
cd lilavati

# 2. Install dependencies with uv
uv sync --all-extras

# 3. Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate on Windows

# 4. Verify installation by running tests
uv run pytest
# Optional, uses your API quota: live tool-calling tests against real models
LILAVATI_LIVE_TESTS=1 GEMINI_API_KEY=... uv run pytest tests/integration -v
LILAVATI_LIVE_TESTS=1 GROQ_API_KEY=... uv run pytest tests/integration -v   # LILAVATI_LIVE_GROQ_PRESET to pick a model
```

---

## 📋 Pull Request (PR) Workflow & Guidelines

### 1. Branch Naming Conventions
Use descriptive, prefix-based branch names:
* `feat/<feature-name>`: New capability, FastMCP server, or financial skill.
* `fix/<bug-name>`: Bug fix or calculation correction.
* `bench/<benchmark-name>`: New datasets, tracks, or evaluators for Līlāvatī-Bench.
* `docs/<topic>`: Documentation, academic references, or tutorial additions.
* `mcp/<server-name>`: Additions or updates to FastMCP servers.

*Example:* `feat/johansen-cointegration`, `mcp/fixed-income-curves`, `bench/track2-covenants`.

### 2. Commit Message Standards (Conventional Commits)
Write concise commit messages following the Conventional Commits specification:
* `feat(econometrics): add Johansen eigenvalue cointegration test`
* `fix(schemas): resolve Ind AS lease liability calculation arc`
* `bench(evaluators): add Proof-of-Citation Score (PoCS) metric`
* `docs(readme): add academic pre-print citations`

### 3. Code Standards & Linting
All code must pass static type checking and linting before a PR can be merged:

```bash
# Format code with ruff
uv run ruff format .

# Check linting rules
uv run ruff check . --fix

# Run strict type checking with mypy
uv run mypy src/
```

### 4. Testing Requirements
Every pull request introducing new calculations or financial schemas **must include tests**:
* Mathematical correctness tests comparing output against independent reference values (e.g., standard textbook problems, QuantLib benchmarks, or Federal Reserve FRED data).
* Edge case tests (zero divisions, negative interest rates, missing table cells, extreme volatility regimes).
* Invariant satisfiability tests using Z3.

### 5. Tutorials (every user-facing feature)
A feature is not done until a user can learn it. Add or update a tutorial in `docs/tutorials/`
(and the `mkdocs.yml` nav) with: who it is for, learning objectives, runnable steps, what can go
wrong (real failures, not hypothetical ones), how to use it with Claude Code and the skills,
exercises, and notes for instructors. If it fits the [course](docs/course/index.md), add it to the
schedule. Link measured numbers to `results/ANALYSIS.md` or a validation page instead of copying them.

### 6. Benchmark results
Result files in `results/` are evidence: never edit them by hand. Run scripts write `<file>.meta.json`;
then run `scripts/build_results_manifest.py` and `scripts/analyse_results.py`, and commit the
updated `manifest.json` and `ANALYSIS.md` with the new files. See [Tutorial 06](docs/tutorials/06_analysing_and_publishing_results.md).

```bash
# Run the test suite with coverage
uv run pytest --cov=src/lilavati tests/
```

---

## 📝 Pull Request Template

When opening a Pull Request, please copy and complete the following template:

```markdown
### Summary of Changes
<!-- A clear 2-3 sentence overview of what this PR introduces or resolves. -->

### Pillar & Scope
- [ ] Pillar I: Fixed Income, Debt & Credit Capital Markets
- [ ] Pillar II: Financial Econometrics & Time Series
- [ ] Pillar III: Financial Machine Learning (de Prado Paradigm)
- [ ] Pillar IV: Derivatives, Volatility Surfaces & Risk
- [ ] Pillar V: Trading, Microstructure & Algorithmic Execution
- [ ] Core Runtime / FastMCP Server / Invariants Engine
- [ ] Līlāvatī-Bench Dataset or Evaluator

### Mathematical & Econometric Verification
<!-- Briefly state the theoretical formula, reference paper, or standard (e.g., Nelson-Siegel 1987, de Prado 2018). -->
- **Formula / Reference:** 
- **Deterministic Tooling Used:** (e.g., FastMCP / Polars / Statsmodels / QuantLib)

### Adherence to Core Invariants
- [ ] **Zero LLM Arithmetic:** No LLM-generated calculations in prompt outputs.
- [ ] **Provenance:** Data elements include `CitationRef` hashes where applicable.
- [ ] **Point-in-Time:** Temporal filters prevent look-ahead bias.
- [ ] **Z3 Verified:** Statement or arbitrage invariants checked.
- [ ] **Tutorial:** user-facing change has a tutorial (or an updated one) in `docs/tutorials/`.

### Test Evidence
<!-- Paste the output of `uv run pytest tests/path_to_test.py` -->
```

---

## 🤖 Using AI Coding Tools (Claude Code, Cursor, Codex, Antigravity)

If you are developing this project using an AI coding assistant:
1. Ensure the agent has loaded our repo-specific skill: [`.claude/skills/lilavati-dev/SKILL.md`](.claude/skills/lilavati-dev/SKILL.md).
   For running or analysing benchmarks, also [`.claude/skills/lilavati-bench/SKILL.md`](.claude/skills/lilavati-bench/SKILL.md).
2. The agent must verify that any code it writes uses typed Pydantic models, strictly avoids inline mental math, and includes unit tests.
3. Review what the agent claims: re-run its key commands yourself. Several bugs in this repository were
   found by checking an agent's report against the code (see the change log in the dev skill).

---

## 💬 Community, Academic Collaboration & Discussions

* **GitHub Discussions:** For architectural proposals, skill designs, and academic paper collaborations.
* **Issues:** For bug reports and targeted feature requests.
* **Pre-Prints & Papers:** If you are interested in co-authoring academic papers using Līlāvatī or Līlāvatī-Bench, please open an issue with the label `academic-collaboration` or reach out to the core maintainers.

We look forward to building the future of institutional financial intelligence together! 🪷

