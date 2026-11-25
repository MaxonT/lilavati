# FAQ and troubleshooting

Answers to the questions and errors early users actually hit. Not here? Open an issue with the
command you ran and its full output.

## About the project

**Is Līlāvatī ready for production use?**
No. It is alpha software for research and teaching, not investment advice, and it has not been
independently reviewed. See [Project status and limitations](https://github.com/revendrat/lilavati#project-status-and-limitations).

**Which engines can I rely on?**
Those with a [validation page](validation/index.md): each lists the independent
reference (textbook, paper or QuantLib), the tolerance measured and the bugs fixed. The order book,
Black-Litterman and HRP engines, the statement graph and the data connectors are not yet audited.

**Does the Z3 check prove my financial model is correct?** <!-- claims-ok: the answer is no -->
No. Z3 checks the numbers you give it (for example, assets equal liabilities plus equity, exactly,
in rational arithmetic). It says nothing about whether those numbers are true. Two narrower things go
further: when identities fail, `diagnose_identities` names the smallest sets of figures that could be
wrong; and `check_covenants_over_ranges` decides whether two covenants hold for every input within
given ranges. Neither says your model or your source figures are right.

**Why does `lilavati-bench run` score 100%?**
It runs the deterministic reference solver, which uses the same engines as the checks, so it passes
by construction; `--model` only labels the report. To score a real language model, use
`scripts/run_llm_baseline.py` ([Tutorial 05](tutorials/05_benchmarking_llms_with_open_models.md)).

**Is Līlāvatī an agent framework?**
No. It is a tool layer: use it from LangGraph, Google ADK, Claude or any MCP client
([Integrations](integrations.md)).

**How do I cite it?**
Use `CITATION.cff` in the repository (GitHub shows a "Cite this repository" button). Benchmark
results are CC BY 4.0; code is Apache-2.0.

## Installation

**Which Python versions are supported?**
3.11 and 3.12 (both run in CI).

**Is it on PyPI?**
Not yet. Install from GitHub:
`pip install "lilavati-finance @ git+https://github.com/revendrat/lilavati@main"` (import name
`lilavati`). Pin a commit hash instead of `main` for a course or a paper.

**`ModuleNotFoundError: No module named 'google'` when running the ADK examples.**
Install the optional extra: `uv sync --extra dev --extra adk`.

**The LangGraph example fails with a dependency conflict on `mcp`.**
`langchain-mcp-adapters` requires `mcp<2`, while Līlāvatī's servers (fastmcp 4) need `mcp>=2.3`.
Run the LangGraph client in its own environment with `uv run --no-project --with ...`; the exact
command is in `examples/integrations/README.md`. The servers still run in the project environment.

## API keys and providers

**Where do I put my API key?**
In an environment variable (`GROQ_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`), never in a file,
notebook, commit or URL. If you pasted a key into a chat or an issue, rotate it.

**Gemini free tier returns HTTP 503 ("high demand") or 429.**
Free-tier capacity varies. The client retries 429 and 5xx with backoff (`ModelSpec.max_retries`);
benchmark runs record failures as provider `error` and `--resume` re-runs only those.

**Groq returns HTTP 400 "`tool calling` is not supported with this model".**
Some models cannot call tools on Groq (for example `allam-2-7b`). Use a model that can (the
`groq-*` presets in `lilavati.llm.providers` were tested with tools); the tool-arm runner stops with a
clear message instead of recording errors.

**Groq returns HTTP 400 `tool_use_failed`.**
The model produced a malformed tool call and Groq rejected it. That is the model's failure, and the
benchmark records it as `model_error`; it is never re-run.

**SEC EDGAR: `ValueError` asking for `LILAVATI_SEC_USER_AGENT`, or HTTP 403.**
SEC asks every client for a real contact. Set `LILAVATI_SEC_USER_AGENT="Your Name you@example.org"`
in your shell (for MCP, in the server's environment; see [MCP setup](mcp-setup.md)). It is not an
API key, but keep it out of committed files all the same.

**Ollama: connection refused or 404.**
Start `ollama serve` and pull a model. Ollama's OpenAI-compatible API is under `/v1`:
`base_url="http://localhost:11434/v1"`.

**Two runs at temperature 0 give different answers.**
Not every provider is deterministic at temperature 0 (two Groq runs of the same cases scored 0.4417
and 0.4917). Run several samples and report the spread.

## Examples, results and the API server

**The 10-K example cannot find `data/sample_filings/aapl_10k_2026.pdf`.**
The filing is not redistributed in this repository. The example tries to download it; if that
fails, download Apple Inc.'s FY2026 Form 10-K from SEC EDGAR and save it under that path.

**`build_results_manifest.py` says a case hash does not match.**
A result file is being checked against the wrong case set: check `--generated` and `--seed`, or the
file's `.meta.json`. Answers are matched to questions by content hash, never by id.

**Can I expose the REST API publicly?**
Not as is: it has no built-in authentication and binds to `127.0.0.1` by default. Put it behind your
own authenticating proxy first (`SECURITY.md`).

**Where are the numbers quoted in the README and papers from?**
`results/ANALYSIS.md` (generated from the stored replies) and the validation pages. CI fails if the
results manifest is stale.
