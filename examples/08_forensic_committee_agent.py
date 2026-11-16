"""
Tutorial 03: Forensic Accounting & Adversarial Investment Committee Agent.

Demonstrates:
1. Forensic accounting analytics: 8-variable Beneish M-Score and Altman Z-Score.
2. Grounding financial disclosures with cryptographic SHA-256 CitationRef provenance.
3. Multi-role adversarial debate: Bull Analyst, Bear Analyst, Forensic Auditor (Z3 SMT), and PM.
4. Formal Z3 SMT accounting invariant verification (Assets == Liabilities + Equity).
5. Binding capital allocation synthesis with automated rejection of accounting manipulators.
6. Auto-converting into a FastAPI REST microservice.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi.testclient import TestClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lilavati.agents.forensics import ForensicEngine
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef
from lilavati.framework import (
    END,
    FinancialState,
    FinancialStateGraph,
    create_agent_api,
)

console = Console()


# ==============================================================================
# STEP 1: Define Typed State Container
# ==============================================================================
class ForensicCommitteeState(FinancialState):
    """
    Typed state container for Forensic Accounting & Investment Committee Agent.
    Tracks financial line items, forensic metrics, adversarial analyst arguments,
    formal SMT accounting proofs, and final PM allocation.
    """

    ticker: str = "ENRON-CASE"
    fiscal_year: int = 2026

    # Income Statement & Balance Sheet Inputs ($ Millions)
    revenue_t: float = 1100.0
    revenue_t_minus_1: float = 1000.0
    cogs_t: float = 600.0
    cogs_t_minus_1: float = 550.0
    receivables_t: float = 180.0  # Spike in receivables
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
    cfo_t: float = 70.0  # Divergence: CFO significantly below net income
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


# ==============================================================================
# STEP 2: Implement Execution Nodes
# ==============================================================================
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
    state.record_citation(
        CitationRef(
            doc_hash=doc_hash,
            filing_type="10-K",
            filing_date=date(2026, 3, 1),
            period_end=date(2025, 12, 31),
            page=45,
            raw_snippet=f"Cash Provided by Operating Activities: ${state.cfo_t:.1f}M",
        )
    )
    return state


def forensic_analysis_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 2: Computes Beneish M-Score and Altman Z-Score deterministically."""
    # 8-variable Beneish M-Score
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

    # Altman Z-Score for manufacturing/corporate distress
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


def bull_analyst_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 3: Formulates the Bull Investment Thesis."""
    growth_pct = ((state.revenue_t / state.revenue_t_minus_1) - 1.0) * 100.0
    state.bull_memo = (
        f"BULL THESIS: Revenue expanded by {growth_pct:.1f}% to ${state.revenue_t:.1f}M. "
        f"Operating scale and market equity (${state.market_value_equity:.1f}M) support continued expansion."
    )
    return state


def bear_analyst_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 4: Formulates the Bear Forensic Accounting Thesis."""
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
        state.auditor_memo = (
            "AUDITOR SMT PROOF: All accounting identities formally verified "
            "(Assets == Liabilities + Equity; Cash delta holds)."
        )
    else:
        state.auditor_memo = (
            f"AUDITOR SMT VIOLATION: Financial statements violate mathematical invariants! "
            f"Violations: {inv_check.violations}"
        )
    return state


def portfolio_manager_decision_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Node 6: Synthesizes debate and renders binding capital allocation decision."""
    # Gate 1: SMT Verification
    if not state.is_accounting_valid:
        state.pm_decision = (
            "REJECTED_ACCOUNTING_INVALID: Disclosures fail formal Z3 mathematical proof. "
            "Balance sheet or cash flow reconciliation violated."
        )
        return state

    # Gate 2: Forensic Manipulation Threshold (Beneish M-Score > -1.78)
    if state.is_likely_manipulator:
        state.pm_decision = (
            f"REJECTED_MANIPULATION_RISK: Beneish M-Score ({state.beneish_m_score:.2f} > -1.78) "
            "indicates high probability of financial reporting manipulation."
        )
        return state

    # Gate 3: Distress Zone
    if state.distress_zone == "Distress":
        state.pm_decision = (
            f"REJECTED_FINANCIAL_DISTRESS: Altman Z-Score ({state.altman_z_score:.2f}) "
            "falls within distressed bankruptcy zone."
        )
        return state

    state.pm_decision = (
        f"APPROVED_ALLOCATION: Passed SMT invariants. Clean forensics (M-Score: {state.beneish_m_score:.2f}, "
        f"Z-Score: {state.altman_z_score:.2f} [{state.distress_zone}])."
    )
    return state


# ==============================================================================
# STEP 3: Assemble & Compile Graph
# ==============================================================================
def build_forensic_committee_agent() -> FinancialStateGraph[ForensicCommitteeState]:
    """Assembles the adversarial committee graph."""
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


# ==============================================================================
# STEP 4: Interactive Demonstration & REST API
# ==============================================================================
def main() -> None:
    console.print(
        Panel.fit(
            "[bold red]Project Līlāvatī: Forensic Accounting & Adversarial Committee Agent[/bold red]\n"
            "[dim]Beneish M-Score, Altman Z-Score, and Z3 SMT Formal Invariant Verification[/dim]",
            border_style="red",
        )
    )

    agent_graph = build_forensic_committee_agent().compile()

    # Scenario 1: Manipulated firm (spiking receivables, low CFO)
    distressed_state = ForensicCommitteeState(
        ticker="AGGRESSIVE-CORP",
        receivables_t=220.0,  # +120% spike in receivables
        receivables_t_minus_1=100.0,
        cfo_t=40.0,  # Low cash flow quality
        net_income_t=140.0,
    )
    res_1 = agent_graph.invoke(distressed_state)

    table_1 = Table(title="Scenario 1: Forensic Manipulation Rejection", border_style="red")
    table_1.add_column("Committee Angle", style="bold white")
    table_1.add_column("Evaluation & Output", style="yellow")
    table_1.add_row(
        "Beneish M-Score",
        f"{res_1.beneish_m_score:.2f} (Likely Manipulator: {res_1.is_likely_manipulator})",
    )
    table_1.add_row("Altman Z-Score", f"{res_1.altman_z_score:.2f} ({res_1.distress_zone})")
    table_1.add_row("SMT Accounting Proof", "VALID (SAT)" if res_1.is_accounting_valid else "UNSAT")
    table_1.add_row("Bull Memo", res_1.bull_memo)
    table_1.add_row("Bear Memo", res_1.bear_memo)
    table_1.add_row("PM Binding Verdict", res_1.pm_decision)
    console.print(table_1)

    # Scenario 2: Clean firm
    clean_state = ForensicCommitteeState(
        ticker="BLUE-CHIP-CORP",
        receivables_t=110.0,
        receivables_t_minus_1=100.0,
        cfo_t=150.0,
        net_income_t=120.0,
        current_liabilities_t=160.0,
        long_term_debt_t=350.0,
        shareholders_equity_t=840.0,
        total_assets_t=1350.0,  # 160 + 350 + 840 = 1350 (Balanced)
        cash_start_t=80.0,
        cff_t=-60.0,
        cfi_t=-50.0,
        cash_end_t=120.0,  # 80 + 150 - 50 - 60 = 120 (Balanced)
    )
    res_2 = agent_graph.invoke(clean_state)

    table_2 = Table(title="Scenario 2: Clean Solvency & Invariants Approval", border_style="green")
    table_2.add_column("Committee Angle", style="bold white")
    table_2.add_column("Evaluation & Output", style="green")
    table_2.add_row(
        "Beneish M-Score",
        f"{res_2.beneish_m_score:.2f} (Likely Manipulator: {res_2.is_likely_manipulator})",
    )
    table_2.add_row("Altman Z-Score", f"{res_2.altman_z_score:.2f} ({res_2.distress_zone})")
    table_2.add_row("SMT Accounting Proof", "VALID (SAT)" if res_2.is_accounting_valid else "UNSAT")
    table_2.add_row("PM Binding Verdict", res_2.pm_decision)
    console.print(table_2)

    # Test ASGI REST API
    app = create_agent_api(
        graph=agent_graph,
        title="Forensic Accounting Investment Committee API",
        description="Automated adversarial investment deliberation and formal forensic auditing service.",
        version="1.0.0",
    )
    client = TestClient(app)
    response = client.post("/api/v1/run", json={"ticker": "AAPL-TEST"})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    console.print(
        f"\n[bold green]✓ REST API Test Successful![/bold green] Status: {response.status_code} | "
        f"PM Decision: {data.get('pm_decision')}"
    )


if __name__ == "__main__":
    main()
