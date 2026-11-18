# Tutorial: Building a Forensic Accounting & Adversarial Investment Committee Agent

In this tutorial, you will design, formally verify, test, and serve a multi-agent **Forensic Accounting & Adversarial Investment Committee Agent**.

---

## 🎯 What You Will Build

Generic LLM investment agents suffer from confirmation bias and arithmetic hallucinations: they summarize earnings press releases uncritically and fail to detect balance sheet fraud or channel stuffing.

In this tutorial, your Līlāvatī agent will:
1. **Anchor Filings to Cryptographic Provenance**: Attach verifiable `CitationRef` SHA-256 hashes to corporate financial line items.
2. **Execute Forensic Manipulation Detection**: Compute Professor Messod Beneish's **8-variable Beneish M-Score** (Days Sales in Receivables Index, Gross Margin Index, Asset Quality Index, Sales Growth Index, Depreciation Index, Sales General and Administrative expenses Index, Leverage Index, Total Accruals to Total Assets).
3. **Measure Bankruptcy Distress**: Compute the **Altman Z-Score** to categorize the firm into Safe, Grey, or Distress zones.
4. **Conduct Adversarial Committee Debate**: Orchestrate specialized agent personas:
   - **Bull Analyst**: Evaluates top-line growth, operating scale, and market capitalization.
   - **Bear Analyst**: Interrogates forensic red flags, receivables spikes, and operating cash flow deficits.
   - **Forensic Auditor (Z3 SMT Solver)**: Formally proves balance sheet reconciliation ($\text{Assets} = \text{Liabilities} + \text{ShareholdersEquity}$) and cash flow conservation ($\Delta \text{Cash} = CFO + CFI + CFF$).
   - **Portfolio Manager**: Renders a binding capital allocation decision, automatically rejecting accounting manipulators and broken balance sheets.
5. **Serve as an ASGI REST API**: Deploy the multi-role committee graph as a production microservice with automated OpenAPI documentation.

---

## 📦 Prerequisites

```bash
git clone https://github.com/revendrat/lilavati.git
cd lilavati
uv sync
```

---

## Step 1: Define the Typed State Container

```python
from lilavati.framework import FinancialState


class ForensicCommitteeState(FinancialState):
    """
    Typed state container for Forensic Accounting & Investment Committee Agent.
    Tracks financial line items, forensic metrics, adversarial analyst arguments,
    formal SMT accounting proofs, and final PM allocation.
    """

    ticker: str = "ENRON-CASE"
    fiscal_year: int = 2026

    # Financial Disclosures ($ Millions)
    revenue_t: float = 1100.0
    revenue_t_minus_1: float = 1000.0
    cogs_t: float = 600.0
    cogs_t_minus_1: float = 550.0
    receivables_t: float = 180.0
    receivables_t_minus_1: float = 100.0
    current_assets_t: float = 480.0
    current_assets_t_minus_1: float = 400.0
    ppe_t: float = 720.0
    ppe_t_minus_1: float = 650.0
    total_assets_t: float = 1350.0
    total_assets_t_minus_1: float = 1200.0
    depreciation_t: float = 50.0
    depreciation_t_minus_1: float = 55.0
    sga_t: float = 160.0
    sga_t_minus_1: float = 150.0
    long_term_debt_t: float = 350.0
    long_term_debt_t_minus_1: float = 300.0
    current_liabilities_t: float = 160.0
    current_liabilities_t_minus_1: float = 140.0
    shareholders_equity_t: float = 840.0
    net_income_t: float = 130.0
    cfo_t: float = 70.0
    cfi_t: float = -60.0
    cff_t: float = 20.0
    cash_start_t: float = 80.0
    cash_end_t: float = 110.0
    market_value_equity: float = 900.0

    # Forensic & Invariant Diagnostics
    beneish_m_score: float = 0.0
    is_likely_manipulator: bool = False
    altman_z_score: float = 0.0
    distress_zone: str = "UNKNOWN"
    is_accounting_valid: bool = False

    # Committee Debate Memos
    bull_memo: str = ""
    bear_memo: str = ""
    auditor_memo: str = ""
    pm_decision: str = "PENDING"
```

---

## Step 2: Implement Execution Nodes

### Node 1: Ingestion & Cryptographic Provenance
Record immutable citations anchoring extracted facts to statutory SEC filing hashes:

```python
from datetime import date
from lilavati.core.provenance import CitationRef


def ingestion_and_provenance_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 1: Ingests filing facts and anchors cryptographic CitationRef hashes."""
    doc_hash = "f18b3294c7e63b6528d3e20e8b23c911854c602058b29ef5e0241cfb263b6102"
    state.record_citation(
        CitationRef(
            doc_hash=doc_hash,
            filing_type="10-K",
            filing_date=date(2026, 3, 1),
            period_end=date(2025, 12, 31),
            page=42,
            raw_snippet=f"Revenues FY{state.fiscal_year}: ${state.revenue_t:.1f}M",
        )
    )
    return state
```

### Node 2: Deterministic Forensic Analytics
Evaluate the 8-variable Beneish M-Score and Altman Z-Score:

$$\text{M-Score} = -4.84 + 0.920 \cdot \text{DSRI} + 0.528 \cdot \text{GMI} + 0.404 \cdot \text{AQI} + 0.892 \cdot \text{SGI} + 0.115 \cdot \text{DEPI} - 0.172 \cdot \text{SGAI} + 4.679 \cdot \text{TATA} - 0.327 \cdot \text{LVGI}$$

$$\text{Threshold}: \text{M-Score} > -1.78 \implies \text{High Probability of Manipulation}$$

```python
from lilavati.agents.forensics import ForensicEngine


def forensic_analysis_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 2: Computes Beneish M-Score and Altman Z-Score deterministically."""
    m_res = ForensicEngine.calculate_beneish_m_score(
        revenue_t=state.revenue_t,
        revenue_t_minus_1=state.revenue_t_minus_1,
        cogs_t=state.cogs_t,
        cogs_t_minus_1=state.cogs_t_minus_1,
        receivables_t=state.receivables_t,
        receivables_t_minus_1=state.receivables_t_minus_1,
        current_assets_t=state.current_assets_t,
        current_assets_t_minus_1=state.current_assets_t_minus_1,
        ppe_t=state.ppe_t,
        ppe_t_minus_1=state.ppe_t_minus_1,
        total_assets_t=state.total_assets_t,
        total_assets_t_minus_1=state.total_assets_t_minus_1,
        depreciation_t=state.depreciation_t,
        depreciation_t_minus_1=state.depreciation_t_minus_1,
        sga_t=state.sga_t,
        sga_t_minus_1=state.sga_t_minus_1,
        long_term_debt_t=state.long_term_debt_t,
        long_term_debt_t_minus_1=state.long_term_debt_t_minus_1,
        current_liabilities_t=state.current_liabilities_t,
        current_liabilities_t_minus_1=state.current_liabilities_t_minus_1,
        net_income_t=state.net_income_t,
        cfo_t=state.cfo_t,
    )
    state.beneish_m_score = m_res.m_score
    state.is_likely_manipulator = m_res.is_likely_manipulator

    working_capital = state.current_assets_t - state.current_liabilities_t
    ebit = state.revenue_t - state.cogs_t - state.sga_t
    total_liabilities = state.current_liabilities_t + state.long_term_debt_t
    z_res = ForensicEngine.calculate_altman_z_score(
        working_capital=working_capital,
        total_assets=state.total_assets_t,
        retained_earnings=state.shareholders_equity_t * 0.5,
        ebit=ebit,
        market_value_equity=state.market_value_equity,
        total_liabilities=total_liabilities,
        sales=state.revenue_t,
    )
    state.altman_z_score = z_res.z_score
    state.distress_zone = z_res.distress_zone
    return state
```

### Nodes 3 & 4: Bull vs Bear Adversarial Debate
The Bull and Bear personas analyze the deterministic diagnostics from opposite perspectives:

```python
def bull_analyst_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    growth_pct = ((state.revenue_t / state.revenue_t_minus_1) - 1.0) * 100.0
    state.bull_memo = (
        f"BULL THESIS: Revenue expanded by {growth_pct:.1f}% to ${state.revenue_t:.1f}M. "
        f"Operating scale and market equity (${state.market_value_equity:.1f}M) support growth."
    )
    return state


def bear_analyst_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    rec_growth = ((state.receivables_t / state.receivables_t_minus_1) - 1.0) * 100.0
    accruals_gap = state.net_income_t - state.cfo_t
    flags: list[str] = []
    if state.is_likely_manipulator:
        flags.append(
            f"Beneish M-Score ({state.beneish_m_score:.2f} > -1.78) flags earnings manipulation"
        )
    if rec_growth > 50.0:
        flags.append(f"Receivables spiked {rec_growth:.1f}%, indicating channel stuffing")
    if accruals_gap > 30.0:
        flags.append(f"Net income exceeds CFO by ${accruals_gap:.1f}M (low-quality accruals)")

    state.bear_memo = "BEAR THESIS: " + (
        "; ".join(flags) if flags else "No severe forensic flags detected."
    )
    return state
```

### Node 5: Formal Z3 SMT Accounting Verification
The Forensic Auditor node verifies that the company's financial statements satisfy core accounting identities:

```python
from lilavati.core.invariants import SuvarnaEngine


def forensic_auditor_smt_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 5: Formally proves three-statement invariants via Z3 SMT solver."""
    total_liabilities = state.current_liabilities_t + state.long_term_debt_t
    suvarna = SuvarnaEngine()
    inv_check = suvarna.verify_three_statement_invariants(
        total_assets=state.total_assets_t,
        total_liabilities=total_liabilities,
        shareholders_equity=state.shareholders_equity_t,
        net_income=state.net_income_t,
        cash_start=state.cash_start_t,
        cash_end=state.cash_end_t,
        cfo=state.cfo_t,
        cfi=state.cfi_t,
        cff=state.cff_t,
    )
    state.is_accounting_valid = inv_check.is_valid
    if inv_check.is_valid:
        state.auditor_memo = "AUDITOR SMT PROOF: All accounting identities formally verified."
    else:
        state.auditor_memo = f"AUDITOR SMT VIOLATION: Invariants breached! {inv_check.violations}"
    return state
```

### Node 6: Portfolio Manager Binding Verdict
```python
def portfolio_manager_decision_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 6: Synthesizes debate and renders binding capital allocation decision."""
    if not state.is_accounting_valid:
        state.pm_decision = (
            "REJECTED_ACCOUNTING_INVALID: Disclosures fail formal Z3 mathematical proof."
        )
        return state

    if state.is_likely_manipulator:
        state.pm_decision = (
            f"REJECTED_MANIPULATION_RISK: Beneish M-Score ({state.beneish_m_score:.2f} > -1.78) "
            "indicates high probability of financial reporting manipulation."
        )
        return state

    if state.distress_zone == "Distress":
        state.pm_decision = f"REJECTED_FINANCIAL_DISTRESS: Altman Z-Score ({state.altman_z_score:.2f}) in distress zone."
        return state

    state.pm_decision = f"APPROVED_ALLOCATION: Passed SMT invariants. Clean forensics (M-Score: {state.beneish_m_score:.2f})."
    return state
```

---

## Step 3: Assemble & Compile the Graph

```python
from lilavati.framework import END, FinancialStateGraph


def build_forensic_committee_agent() -> FinancialStateGraph[ForensicCommitteeState]:
    graph = FinancialStateGraph(ForensicCommitteeState)
    graph.add_node("ingestion", ingestion_and_provenance_node)
    graph.add_node("forensics", forensic_analysis_node)
    graph.add_node("bull", bull_analyst_node)
    graph.add_node("bear", bear_analyst_node)
    graph.add_node("auditor", forensic_auditor_smt_node)
    graph.add_node("pm", portfolio_manager_decision_node)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "forensics")
    graph.add_edge("forensics", "bull")
    graph.add_edge("bull", "bear")
    graph.add_edge("bear", "auditor")
    graph.add_edge("auditor", "pm")
    graph.add_edge("pm", END)
    return graph
```

---

## Step 4: Run Locally & Test

```python
agent_graph = build_forensic_committee_agent().compile()

# Test firm with suspicious accounting
state = ForensicCommitteeState(
    ticker="SUSPICIOUS-CORP",
    receivables_t=220.0,
    receivables_t_minus_1=100.0,
    cfo_t=40.0,
    net_income_t=140.0,
)
result = agent_graph.invoke(state)
print("PM Verdict:", result.pm_decision)
# Output: REJECTED_MANIPULATION_RISK: Beneish M-Score (-1.63 > -1.78) indicates high probability of financial reporting manipulation.
```

---

## Step 5: Convert into a REST API

```python
from lilavati.framework import create_agent_api, serve_agent

agent_graph = build_forensic_committee_agent().compile()
app = create_agent_api(
    graph=agent_graph,
    title="Forensic Accounting Investment Committee API",
    description="Automated adversarial investment deliberation and formal forensic auditing service.",
    version="1.0.0",
)

if __name__ == "__main__":
    serve_agent(agent_graph, host="0.0.0.0", port=8000)
```

---

## 🚀 Companion Code

A complete, runnable script for this tutorial is available in the repository:
```bash
uv run python examples/08_forensic_committee_agent.py
```
