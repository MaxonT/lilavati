# Project Līlāvatī: The Open-Source Operating System for Institutional Finance

> *"Active and graceful maiden with playful eyes, Līlāvatī! If you have learned the method of inversion, tell me what is that number which multiplied by 3, then increased by three-fourths of the product, divided by 7, diminished by one-third of the quotient, multiplied by itself, diminished by 52, the square root extracted, then 8 added, and divided by 10, gives the number 2?"*
> — **Bhāskarācārya II**, *Līlāvatī* (c. 1150 CE)

---

## What is Project Līlāvatī?

**Project Līlāvatī** is an open-source tool layer for financial agents (alpha) and an evaluation benchmark, covering quantitative finance, debt/credit markets, econometrics, financial machine learning, derivatives, market microstructure, and sustainability finance. Under the hood, models read and plan; reference-tested engines compute; Z3 checks accounting identities on the given numbers.

General-purpose LLM agent frameworks do not by themselves check where a number came from: models compute in text, can use information published after the as-of date, and leave no auditable provenance.

Līlāvatī solves this through a **Domain-Specific Runtime & Protocol Layer**:
1. **Zero LLM Arithmetic:** LLMs plan and reason; all calculations are executed deterministically through Python, Polars, DuckDB, and FastMCP.
2. **First-Order Formal Invariants (Z3 SMT Solver):** Core accounting identities ($Assets = Liabilities + Equity$) are checked with the Z3 SMT solver on the given numbers (exact rational arithmetic), and debt covenants are checked arithmetically with an at-limit tolerance. These verify the numbers supplied, not a model over all inputs.
3. **Cryptographic Provenance (`CitationRef`):** Every numerical fact extracted from SEC EDGAR or Ind AS filings carries an immutable SHA-256 document hash, page, and snippet citation.
4. **Bi-Temporal Point-in-Time Isolation:** Disclosures are indexed with Dual-Time semantics (Valid Time $T_v$ vs System Time $T_s$) so that queries return only what was known at the as-of time; detecting look-ahead use by an agent is still partial (TPI).
5. **OpenTelemetry-Compatible Tracing:** Full compliance audit trails capturing agent turns, LLM prompts/tokens, deterministic tool outputs, and solver check results.

---

## Quickstart

### Installation

```bash
git clone https://github.com/revendrat/lilavati.git
cd lilavati
uv sync
```

### Running the Līlāvatī-Bench CLI

Run the deterministic reference solver across all 7 tracks (`--model` only labels the report card;
to score a real LLM use `scripts/run_llm_baseline.py`, see [Benchmarks](benchmarks.md)):

```bash
uv run lilavati-bench run --format rich
```

### Inspecting Golden Test Cases

```bash
uv run lilavati-bench list-cases
```

---

## Flagship Pillars

| Pillar | Capability | Key Technologies |
| :--- | :--- | :--- |
| **Pillar I** | Dynamic Three-Statement Modeling & Valuation | Reactive DAG, Polars, Z3 SMT Solver |
| **Pillar II** | Fixed Income, Credit & Debt Indentures | Bond Analytics (DV01, Convexity), Nelson-Siegel, Merton Structural Model |
| **Pillar III** | Financial Econometrics & Stationarity | ADF Unit Root, Engle-Granger Cointegration, OU Half-Life, GARCH(p,q) |
| **Pillar IV** | Financial ML (de Prado Paradigm) | Fractional Differentiation ($d^*$), Triple-Barrier Labeling (purged CV planned) |
| **Pillar V** | Derivatives, Volatility Surfaces & Greeks | Black-Scholes-Merton, CRR Binomial Trees, SABR Calibration, Arbitrage Invariants |
| **Pillar VI** | Market Microstructure & Optimal Execution | LOB Microprice, OFI, Almgren-Chriss Liquidation, Black-Litterman, HRP |
| **Pillar VII** | Sustainable Finance & Climate Risk | Scope 1/2/3 WACI, Shadow Carbon Tax Stress Testing, SLB Step-Up Covenants |

---

## 📚 End-to-End Tutorials & Guides

Explore complete, runnable walkthroughs demonstrating how to build, test, and serve verifiable agents across key institutional finance domains:

1. [Tutorial 01: Credit Underwriting & SMT Proofs](tutorials/01_build_your_first_financial_agent.md)
   - *Use Case:* Commercial loan underwriting: maintenance covenants (net leverage and interest coverage, exact at the limit), debt reduction needed to comply, and a Merton default probability.
   - *Companion Code:* `examples/06_build_custom_agent_and_api.py`

2. [Tutorial 02: Statistical Arbitrage & Algorithmic Execution](tutorials/02_statistical_arbitrage_and_execution_agent.md)
   - *Use Case:* Quantitative pairs trading, Engle-Granger cointegration, OU mean-reversion half-life, memory-preserving fractional differentiation ($d^*$), LOB microstructure, and Almgren-Chriss liquidation.
   - *Companion Code:* `examples/07_stat_arb_and_execution_agent.py`

3. [Tutorial 03: Forensic Accounting & Adversarial Investment Committee](tutorials/03_forensic_accounting_and_adversarial_committee.md)
   - *Use Case:* Fundamental equity research, 8-variable Beneish M-Score manipulation detection, Altman Z-Score distress, multi-role adversarial committee deliberation, and Z3 SMT balance sheet reconciliation.
   - *Companion Code:* `examples/08_forensic_committee_agent.py`

4. [Tutorial 04: Climate Transition Risk & Sustainable Portfolio Optimization](tutorials/04_climate_risk_and_sustainable_portfolio.md)
   - *Use Case:* TCFD Weighted Average Carbon Intensity (WACI), carbon tax balance sheet shock stress testing, Sustainability-Linked Bond (SLB) covenants, and Hierarchical Risk Parity (HRP) allocation.
   - *Companion Code:* `examples/09_climate_sustainable_portfolio_agent.py`


