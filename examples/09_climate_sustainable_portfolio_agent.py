"""
Tutorial 04: Climate Transition Risk & Sustainable Portfolio Optimization Agent.

Demonstrates:
1. TCFD-aligned Weighted Average Carbon Intensity (WACI) calculation.
2. Climate transition stress testing under statutory/shadow carbon tax shocks ($/tCO2e).
3. Sustainability-Linked Bond (SLB) coupon step-up covenant auditing.
4. Hierarchical Risk Parity (HRP) portfolio weight allocation avoiding matrix inversion singularity.
5. Formal ESG mandate verification gates before trade capital allocation.
6. Auto-converting into a FastAPI REST microservice.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lilavati.core.execution import ExecutionEngine
from lilavati.core.sustainability import SustainabilityEngine
from lilavati.framework import (
    END,
    FinancialState,
    FinancialStateGraph,
    create_agent_api,
)
from lilavati.schemas.sustainability import SustainabilityLinkedBond

console = Console()


# ==============================================================================
# STEP 1: Define Typed State Container
# ==============================================================================
class SustainablePortfolioState(FinancialState):
    """
    Typed state container for Climate Risk & Sustainable Portfolio Agent.
    Tracks green asset returns, carbon intensities, transition tax shocks,
    SLB covenant compliance, and HRP portfolio allocations.
    """

    portfolio_name: str = "Global Green Transition Fund"
    assets: list[str] = ["SolarEnergy", "CleanMobility", "GridStorage", "TransitionUtility"]

    # Historical asset return series (8 periods across 4 assets)
    returns_matrix: list[list[float]] = [
        [0.015, 0.022, 0.008, 0.005],
        [0.018, 0.012, 0.015, -0.002],
        [-0.010, -0.018, 0.002, 0.008],
        [0.025, 0.030, 0.018, 0.004],
        [0.005, 0.015, 0.010, -0.005],
        [0.012, 0.005, 0.020, 0.010],
        [-0.015, -0.008, -0.005, 0.012],
        [0.020, 0.025, 0.014, 0.003],
    ]

    # Carbon accounting inputs (Scope 1+2 emissions in tons, Revenue in $ Millions)
    asset_emissions_tons: list[float] = [20000.0, 45000.0, 30000.0, 180000.0]
    asset_revenues_millions: list[float] = [800.0, 1000.0, 600.0, 1200.0]
    paris_aligned_waci_limit: float = 100.0  # Max 100 tCO2e / $M revenue

    # Transition risk stress testing for key corporate holding ($ Millions)
    holding_revenue: float = 1200.0
    holding_ebit: float = 240.0
    holding_interest_expense: float = 45.0
    holding_taxable_emissions_m_tons: float = 0.22  # 220,000 tons = 0.22M tons
    carbon_tax_price_per_ton: float = 75.0  # $75/tCO2e shock
    pass_through_rate: float = 0.50
    covenant_interest_floor: float = 2.50

    # Sustainability-Linked Bond (SLB) holding
    slb_isin: str = "US1234567890"
    slb_notional_usd: float = 250000000.0
    slb_base_coupon_rate: float = 0.0425  # 4.25%
    slb_spt_reduction_target_pct: float = 0.30  # 30% reduction
    slb_baseline_emissions_tons: float = 500000.0
    slb_reported_emissions_tons: float = 380000.0  # Realized: 24% reduction (Misses 30% SPT)
    slb_step_up_bps: float = 25.0  # 25 bps penalty

    # Quantitative diagnostics
    hrp_weights: list[float] = []
    portfolio_waci: float = 0.0
    is_waci_paris_compliant: bool = False
    stressed_interest_coverage: float = 0.0
    is_covenant_at_risk: bool = False
    slb_spt_achieved: bool = False
    slb_annual_interest_penalty_usd: float = 0.0
    allocation_verdict: str = "PENDING"


# ==============================================================================
# STEP 2: Implement Execution Nodes
# ==============================================================================
def hrp_allocation_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 1: Computes Hierarchical Risk Parity (HRP) optimal asset weights."""
    hrp_res = ExecutionEngine.optimize_hierarchical_risk_parity(
        assets=state.assets,
        returns_matrix=state.returns_matrix,
    )
    state.hrp_weights = hrp_res.weights
    return state


def waci_carbon_intensity_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 2: Computes TCFD Weighted Average Carbon Intensity (WACI)."""
    carbon_res = SustainabilityEngine.calculate_waci(
        portfolio_weights=state.hrp_weights,
        emissions=state.asset_emissions_tons,
        revenues_millions=state.asset_revenues_millions,
        paris_aligned_threshold=state.paris_aligned_waci_limit,
    )
    state.portfolio_waci = carbon_res.waci
    state.is_waci_paris_compliant = carbon_res.carbon_intensity_clean
    return state


def carbon_tax_stress_test_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 3: Stress-tests corporate debtor under statutory or shadow carbon price shock."""
    # Convert taxable emissions into consistent units for $M financials
    # 0.22M tons * $75/ton = $16.5M gross carbon tax liability
    tax_res = SustainabilityEngine.stress_test_carbon_tax(
        revenue=state.holding_revenue,
        ebit=state.holding_ebit,
        interest_expense=state.holding_interest_expense,
        scope1_emissions=state.holding_taxable_emissions_m_tons * 0.7,
        scope2_emissions=state.holding_taxable_emissions_m_tons * 0.3,
        carbon_price_per_ton=state.carbon_tax_price_per_ton,
        pass_through_rate=state.pass_through_rate,
        covenant_interest_floor=state.covenant_interest_floor,
    )
    state.stressed_interest_coverage = tax_res.interest_coverage_after
    state.is_covenant_at_risk = tax_res.is_covenant_at_risk
    return state


def slb_covenant_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 4: Evaluates Sustainability-Linked Bond (SLB) performance covenants."""
    bond = SustainabilityLinkedBond(
        isin=state.slb_isin,
        notional=state.slb_notional_usd,
        base_coupon_rate=state.slb_base_coupon_rate,
        target_reduction_pct=state.slb_spt_reduction_target_pct,
        baseline_emissions=state.slb_baseline_emissions_tons,
        step_up_bps=state.slb_step_up_bps,
        target_year=2026,
    )
    slb_res = SustainabilityEngine.evaluate_slb_covenant(
        bond=bond,
        reported_emissions=state.slb_reported_emissions_tons,
    )
    state.slb_spt_achieved = slb_res.spt_achieved
    state.slb_annual_interest_penalty_usd = slb_res.annual_interest_penalty
    return state


def esg_mandate_gate_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 5: Enforces formal ESG mandate gates before approving capital allocation."""
    # Gate 1: Portfolio WACI must satisfy the Paris-aligned threshold
    if not state.is_waci_paris_compliant:
        state.allocation_verdict = (
            f"REJECTED_WACI_EXCEEDED: Portfolio WACI ({state.portfolio_waci:.1f} tCO2e/$M) "
            f"exceeds mandate limit ({state.paris_aligned_waci_limit:.1f} tCO2e/$M)."
        )
        return state

    # Gate 2: Carbon tax stress test must not trigger covenant breach
    if state.is_covenant_at_risk:
        state.allocation_verdict = (
            f"REJECTED_TRANSITION_RISK: Carbon tax shock drops debt interest coverage "
            f"to {state.stressed_interest_coverage:.2f}x (below {state.covenant_interest_floor:.2f}x floor)."
        )
        return state

    state.allocation_verdict = (
        f"APPROVED_SUSTAINABLE_ALLOCATION: Paris-compliant WACI ({state.portfolio_waci:.1f} tCO2e/$M). "
        f"Stressed coverage ({state.stressed_interest_coverage:.2f}x) resilient. "
        f"SLB step-up penalty: ${state.slb_annual_interest_penalty_usd:,.2f}."
    )
    return state


# ==============================================================================
# STEP 3: Assemble & Compile Graph
# ==============================================================================
def build_sustainable_portfolio_agent() -> FinancialStateGraph[SustainablePortfolioState]:
    """Assembles the Climate Risk & Sustainable Portfolio state graph."""
    graph = FinancialStateGraph(SustainablePortfolioState)
    graph.add_node("hrp_allocation", hrp_allocation_node)
    graph.add_node("waci_carbon", waci_carbon_intensity_node)
    graph.add_node("carbon_tax_stress", carbon_tax_stress_test_node)
    graph.add_node("slb_covenant", slb_covenant_node)
    graph.add_node("esg_mandate_gate", esg_mandate_gate_node)

    graph.set_entry_point("hrp_allocation")
    graph.add_edge("hrp_allocation", "waci_carbon")
    graph.add_edge("waci_carbon", "carbon_tax_stress")
    graph.add_edge("carbon_tax_stress", "slb_covenant")
    graph.add_edge("slb_covenant", "esg_mandate_gate")
    graph.add_edge("esg_mandate_gate", END)
    return graph


# ==============================================================================
# STEP 4: Interactive Demonstration & REST API
# ==============================================================================
def main() -> None:
    console.print(
        Panel.fit(
            "[bold green]Project Līlāvatī: Climate Risk & Sustainable Portfolio Optimization[/bold green]\n"
            "[dim]TCFD WACI, Carbon Tax Stress Testing, SLB Covenants, and Hierarchical Risk Parity[/dim]",
            border_style="green",
        )
    )

    agent_graph = build_sustainable_portfolio_agent().compile()

    # Invoke agent
    initial_state = SustainablePortfolioState()
    result = agent_graph.invoke(initial_state)

    # Display results
    table = Table(title="Sustainable Portfolio & Climate Risk Summary", border_style="green")
    table.add_column("Pillar / Metric", style="bold white")
    table.add_column("Deterministic Calculation", style="green")

    table.add_row("Portfolio Name", result.portfolio_name)
    for asset, weight in zip(result.assets, result.hrp_weights, strict=True):
        table.add_row(f"HRP Weight: {asset}", f"{weight * 100.0:.2f}%")

    table.add_row("Portfolio WACI", f"{result.portfolio_waci:.2f} tCO2e / $M rev")
    table.add_row(
        "Paris Alignment (< 100)", "COMPLIANT" if result.is_waci_paris_compliant else "BREACH"
    )
    table.add_row(
        f"Stressed Coverage (${result.carbon_tax_price_per_ton:.0f}/ton shock)",
        f"{result.stressed_interest_coverage:.2f}x (At Risk: {result.is_covenant_at_risk})",
    )
    table.add_row(
        "SLB Covenant Status",
        f"SPT Achieved: {result.slb_spt_achieved} (Annual Penalty: ${result.slb_annual_interest_penalty_usd:,.2f})",
    )
    table.add_row("ESG Mandate Verdict", result.allocation_verdict)
    console.print(table)

    # Test ASGI REST API
    app = create_agent_api(
        graph=agent_graph,
        title="Sustainable Portfolio Climate API",
        description="Autonomous TCFD WACI, carbon tax stress testing, and HRP allocation service.",
        version="1.0.0",
    )
    client = TestClient(app)
    response = client.post("/api/v1/run", json={"carbon_tax_price_per_ton": 100.0})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    console.print(
        f"\n[bold green]✓ REST API Test Successful![/bold green] Status: {response.status_code} | "
        f"Verdict: {data.get('allocation_verdict')}"
    )


if __name__ == "__main__":
    main()
