<div align="center">

# 🪷 Līlāvatī (लीलावती)

### An Open-Source Verifiable Agentic Toolkit for Quantitative Finance

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/Protocol-FastMCP-green)](https://github.com/jlowin/fastmcp)
[![M8ven Score](https://m8ven.ai/badge/mcp/revendrat/lilavati)](https://m8ven.ai/mcp/revendrat/lilavati?s=readme)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Status](https://img.shields.io/badge/Status-Alpha%20%2F%20Research-orange)](#academic-research--pre-prints)

*A domain-specific runtime, standardized FastMCP tool bus, and benchmark suite for autonomous financial reasoning, econometrics, credit risk, and algorithmic execution.*

</div>

---

> ### ⚠️ Project status: alpha / research preview
> * **Not production software and not investment advice.** Outputs have not been independently audited. Do not use for trading, credit or compliance decisions without your own validation. Provided "AS IS" under Apache-2.0.
> * **Default LLM is a mock.** Out of the box, agents use a `mock-evaluator` model that returns canned text, so examples exercise the deterministic tools, not a real LLM. Configure a real provider (OpenAI, Anthropic, Gemini, vLLM, Ollama) via API-key environment variables to run real agent loops.
> * **Benchmark results:** the repo ships the Līlāvatī-Bench *harness*, a deterministic reference solver (passes by construction), a templated case generator, and a first no-tool baseline of four Google models (`results/`). Those models answered nearly all single-step numeric items correctly, so the current cases are largely saturated; harder tracks are future work. Earlier placeholder leaderboard numbers have been removed.
> * **Install from source for now** (see below). The PyPI name `lilavati` belongs to an unrelated project; this package will be published as `lilavati-finance` (import name stays `lilavati`). The optional `quantlib` extra is currently unused by the code.
> * **API server** binds to `127.0.0.1` by default and has no built-in authentication. Put it behind your own auth/proxy before exposing it.

---

## 🚀 Quick start

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pytest                # run the test suite
uv run python examples/01_three_statement_reconciliation.py
uv run lilavati-bench list-cases
```

---

## 📖 The Historic Anecdote & Ethos

In **1150 CE**, Indian mathematician and astronomer **Bhāskarācārya (Bhāskara II)** completed the classical mathematical treatise *Līlāvatī* (*Siddhānta Śiromaṇi*). Unlike contemporary academic texts of antiquity, Bhāskara framed mathematics as an engaging, dialogic conversation with his daughter, Līlāvatī, incorporating affectionate feminine vocatives (*"O Līlāvatī, intelligent girl..."*) throughout.

Crucially, substantial portions of the *Līlāvatī* were devoted to practical commercial and financial mathematics:
* **Miśraka-vyavahāra**: Partnerships, capital shares, joint ventures, and interest rate calculations.
* **Kraya-vikraya**: Buying, selling, price elasticity, and barter economies.
* **Suvarṇa-gaṇita**: Alligation and gold purity verification—history’s earliest recorded treatise on **financial auditing, assaying, and ledger balance verification**.

**Project Līlāvatī** channels this 900-year heritage into modern computational finance. We replace the stochastic, hallucination-prone arithmetic of generic Large Language Models with **deterministic execution, formal invariant verification, and bi-temporal precision**.

---

## ⚡ Why Generic Agent Frameworks Can Fall Short in Finance

General-purpose LLM agent frameworks do not by themselves enforce where numbers come from. In finance, this can lead to failures such as:

1. **Mental Math Hallucinations:** LLMs attempt multi-step CAGR, DCF, or bond duration math inside token generation, producing subtly erroneous values.
2. **Missing Cryptographic Provenance:** Summaries lack verbatim hash-linked references (`CitationRef`) to the underlying 10-K, 10-Q, or statutory footnote.
3. **Look-Ahead & Restatement Poisoning:** Static RAG pipelines ingest historical filings alongside future restatements or subsequent events, corrupting backtests and credit assessments.
4. **Broken Accounting Invariants:** LLMs frequently produce three-statement projections where $Assets \neq Liabilities + Equity$ or the cash flow ending balance fails to reconcile with balance sheet cash.
5. **Econometric Naivety:** Naive agents regress un-differenced non-stationary price series, creating spurious alpha signals from pure noise.

---

## 🏛️ What is implemented today

Status legend: ✅ implemented and tested · 🟡 partial · 🗺️ planned (see [ROADMAP.md](ROADMAP.md)). Numerical validation against independent references is in progress ([docs/validation](docs/validation/index.md)); engines not yet audited should be treated as unvalidated.

| Area | Capability | Status | Module |
| :--- | :--- | :---: | :--- |
| **Fixed income & credit** | Bond price, Macaulay/modified duration, convexity, DV01 | ✅ | `core/fixed_income.py` |
| | Nelson-Siegel curve fit | ✅ | `core/fixed_income.py` |
| | Merton structural default model | ✅ | `core/fixed_income.py` |
| | Covenant checks (net leverage, interest coverage) | ✅ | `core/invariants.py` |
| | Nelson-Siegel-Svensson, key-rate durations, CDS hazard rates, tranche waterfalls | 🗺️ | n/a |
| **Econometrics** | ADF stationarity test | ✅ | `core/econometrics.py` |
| | Engle-Granger cointegration + OU half-life | ✅ | `core/econometrics.py` |
| | GARCH(p,q) volatility | ✅ | `core/econometrics.py` |
| | KPSS / Phillips-Perron, Johansen/VECM, EGARCH, DCC-GARCH, regime switching | 🗺️ | n/a |
| **Financial ML** | Fractional differentiation + minimal-d search | ✅ | `core/financial_ml.py` |
| | Triple-barrier labeling | ✅ | `core/financial_ml.py` |
| | Purged/embargoed CV, meta-labeling, information-driven bars | 🗺️ | n/a |
| **Derivatives** | Black-Scholes price and Greeks, CRR American options, implied vol | ✅ | `core/derivatives.py` |
| | SABR smile calibration, butterfly arbitrage check (Z3) | ✅ | `core/derivatives.py` |
| | Heston, local vol, full-surface no-arbitrage | 🗺️ | n/a |
| **Execution & portfolio** | Order-book snapshot (spread, microprice, imbalance) | ✅ | `core/execution.py` |
| | Almgren-Chriss schedule (discrete, Almgren & Chriss 2000) | ✅ | `core/execution.py` |
| | Black-Litterman, hierarchical risk parity | ✅ | `core/execution.py` |
| | VPIN, TWAP/VWAP/POV | 🗺️ | n/a |
| **Sustainable finance** | WACI, carbon-tax stress test, SLB step-up covenant | ✅ | `core/sustainability.py` |
| | Full NGFS scenario pathways | 🟡 enum only | `schemas/sustainability.py` |
| **Verification & data** | Z3 three-statement invariants | ✅ | `core/invariants.py` |
| | Statement DAG with driver shocks | ✅ | `core/graph.py` |
| | Bi-temporal DuckDB store, `CitationRef` provenance | ✅ | `core/temporal.py`, `core/provenance.py` |
| | SEC EDGAR XBRL, 10-K PDF parser, Ind AS crosswalk (live-data validation pending) | ✅ | `connectors/` |
| | Market data: OHLCV normalisation, curve fit from supplied yields | 🟡 no data fetching / FRED yet | `connectors/market_data.py` |
| | US-GAAP ↔ IFRS reconciliation | 🗺️ | n/a |
| **Agents** | Neuro-symbolic runtime, adversarial committee, forensics (Beneish, Altman) | ✅ | `runtime/`, `agents/` |
| | Eight FastMCP tool servers | ✅ | `mcp_servers/` |

---

## 🧩 The Core Runtime Architecture

Līlāvatī separates the cognitive reasoning of LLMs from deterministic mathematical execution:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE AGENT LAYER                           │
│   Adversarial Committee (Macro Strategist ↔ Quant ↔ Credit ↔ Auditor)  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Dispatches Structured Intent
┌────────────────────────────────────▼───────────────────────────────────┐
│                    LĪLĀVATĪ RUNTIME & VERIFICATION                     │
│  ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────┐ │
│  │ Zero-Arithmetic Guard │ │ Bi-Temporal (PIT) │ │ Z3 SMT Invariants │ │
│  └───────────────────────┘ └───────────────────┘ └───────────────────┘ │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Dispatches Tool Execution via MCP
┌────────────────────────────────────▼───────────────────────────────────┐
│                       STANDARDIZED FAST-MCP BUS                        │
│   mcp-filings-xbrl  │  mcp-fixed-income  │  mcp-econometrics-quant     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Normalizes Ingestion
┌────────────────────────────────────▼───────────────────────────────────┐
│                    CANONICAL DATA & ONTOLOGY LAYER                     │
│   DuckDB Bi-temporal Storage │ SEC XBRL & Ind AS ingestion             │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Zero LLM Arithmetic Policy (design goal):** Agents are designed so that calculated numbers originate from deterministic FastMCP tools, and runtime gates flag numbers lacking tool provenance. This reduces, but cannot mathematically eliminate, arithmetic in model prose.
2. **`CitationRef` Provenance:** Every extracted line item carries a cryptographic hash: `CitationRef(doc_hash, page, table_id, xbrl_tag)`.
3. **Bi-Temporal Point-in-Time Isolation:** Using DuckDB bi-temporal semantics, queries specify `valid_time` and `as_of_timestamp` to prevent look-ahead bias and restatement contamination.
4. **Z3 SMT Invariant Verification (`Suvarna` Engine):** Financial statement identities ($A = L + E$) and no-arbitrage bounds are encoded in first-order logic. Infeasible projections trigger automatic agent self-correction.

---

## 🏆 Līlāvatī-Bench: The Financial Agent Benchmark

To measure progress, we are developing **Līlāvatī-Bench** (harness available; third-party model baselines are not yet published), an open benchmark testing autonomous agents and finance-tuned foundation models across seven tracks:

* **Track 1: Three-Statement Dynamic Reconciliation:** Propagation of operational shocks across 3 statements verified via Z3 SMT solver.
* **Track 2: Fixed Income, Credit & Debt Waterfalls:** Recovery of debt indentures, tranche seniority, covenant headroom, and Merton default probabilities.
* **Track 3: Econometric Rigor & Spurious Correlation Resistance:** Automated detection of unit roots and cointegration testing before model fitting.
* **Track 4: Financial ML Methodological Soundness:** Enforcement of fractional differentiation and purged cross-validation.
* **Track 5: Derivatives & Volatility Arbitrage Invariance:** SABR calibration and arbitrage-free implied volatility surface constraints.
* **Track 6: Market Microstructure & Optimal Execution:** Almgren-Chriss liquidation and VWAP slippage optimization on simulated order books.
* **Track 7: Sustainable Finance & Climate Stress Testing:** Grounded extraction of Scope 1/2/3 disclosures, WACI portfolio calculation, carbon tax cash-flow impairments, and Sustainability-Linked Bond (SLB) covenant step-up verification.

---

## 🎓 Academic Research & Pre-Prints

Two working-draft papers live in `docs/papers/` (not peer reviewed). We invite researchers and practitioners to help with real model evaluations. Ideas for further write-ups:

1. **"Līlāvatī: A Neuro-Symbolic Agentic Framework for Institutional Financial Reasoning"**.
2. **"Līlāvatī-Bench: Stress-Testing Financial LLMs and Autonomous Agents on Accounting Invariants and Look-Ahead Bias"**.
3. **"Eliminating Arithmetic Hallucinations in Financial Agents via SMT Solvers and FastMCP"**.

### Citing Project Līlāvatī

If you use Līlāvatī in your research or applications, please cite:

```bibtex
@software{lilavati2026,
  author = {The Lilavati Team},
  title = {Līlāvatī: An Open-Source, Neuro-Symbolic Agentic Operating System for Finance},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/revendrat/lilavati}}
}
```

---

## 🤝 Contributing & Community

We are building a welcoming, mathematically rigorous community of:
* **Quantitative Researchers & Econometricians:** Implement statistical arbitrage, GARCH, and term-structure modules.
* **Credit & Debt Specialists:** Build covenant parsers and debt waterfall solvers.
* **Core Software Engineers:** Scale FastMCP servers, DuckDB bi-temporal pipelines, and Rust bindings.
* **Academic Contributors:** Expand golden datasets for `Līlāvatī-Bench`.

Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started, and explore our [Development Roadmap & To-Do List](ROADMAP.md) to track active and upcoming engineering milestones.

### Coding with AI Tools (Claude Code, Cursor, Antigravity, Codex)

If you are using AI coding agents or assistants to write code for this repository, please direct your agent to read the repository skill:
👉 [`.agents/skills/lilavati-dev/SKILL.md`](.agents/skills/lilavati-dev/SKILL.md)

This skill instructs coding agents on our **Zero LLM Arithmetic** rule, Z3 invariant verification, type annotations, and PR validation standards.

---

## 📄 License

Project Līlāvatī is licensed under the [Apache License 2.0](LICENSE).

