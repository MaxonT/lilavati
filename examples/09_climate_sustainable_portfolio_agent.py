"""
Companion script for Tutorial 04 (docs/tutorials/04_climate_risk_and_sustainable_portfolio.md):
carbon intensity, carbon-tax stress and a sustainability-linked bond for a synthetic fund.

1. Hierarchical Risk Parity weights (engine not yet audited; 8 periods of returns: illustrative).
2. TCFD WACI against an illustrative fund mandate of 100 tCO2e per USD m revenue.
3. A debtor's interest coverage under a carbon price (units: million tCO2e with USD m financials).
4. Whether a sustainability-linked bond met its target, and the coupon step-up.

No LLM is involved; all entities are synthetic. Not investment advice.

    uv run python examples/09_climate_sustainable_portfolio_agent.py
"""

from __future__ import annotations

from lilavati.core.execution import ExecutionEngine
from lilavati.core.sustainability import SustainabilityEngine
from lilavati.framework import END, FinancialState, FinancialStateGraph
from lilavati.schemas.sustainability import SustainabilityLinkedBond


class ClimatePortfolioState(FinancialState):
    """Synthetic fund. Emissions in tCO2e (holdings) or million tCO2e (debtor); money as labelled."""

    assets: list[str] = ["SolarEnergy", "CleanMobility", "GridStorage", "TransitionUtility"]
    # 8 periods of returns per asset: far too few for a stable covariance estimate (see below)
    returns_matrix: list[list[float]] = [
        [0.015, 0.022, 0.008, 0.005], [0.018, 0.012, 0.015, -0.002],
        [-0.010, -0.018, 0.002, 0.008], [0.025, 0.030, 0.018, 0.004],
        [0.005, 0.015, 0.010, -0.005], [0.012, 0.005, 0.020, 0.010],
        [-0.015, -0.008, -0.005, 0.012], [0.020, 0.025, 0.014, 0.003],
    ]
    emissions_tco2e: list[float] = [20000.0, 45000.0, 30000.0, 180000.0]  # Scope 1 + 2, per company
    revenues_usd_m: list[float] = [800.0, 1000.0, 600.0, 1200.0]
    mandate_waci_limit: float = 100.0  # tCO2e per USD m revenue; illustrative fund mandate

    # A corporate debtor (USD m, and million tCO2e so that tax comes out in USD m)
    debtor_revenue_usd_m: float = 1200.0
    debtor_ebit_usd_m: float = 240.0
    debtor_interest_usd_m: float = 45.0
    debtor_scope1_mtco2e: float = 0.154
    debtor_scope2_mtco2e: float = 0.066
    carbon_price_usd_per_t: float = 75.0
    pass_through_rate: float = 0.50       # share of the tax passed to customers
    min_interest_coverage: float = 2.50

    # A sustainability-linked bond held by the fund
    slb_notional_usd: float = 250_000_000.0
    slb_base_coupon: float = 0.0425
    slb_target_reduction: float = 0.30    # 30% below baseline
    slb_baseline_tco2e: float = 500_000.0
    slb_reported_tco2e: float = 380_000.0
    slb_step_up_bps: float = 25.0

    # Outputs
    hrp_weights: list[float] = []
    portfolio_waci: float = 0.0
    within_waci_mandate: bool = False
    stressed_interest_coverage: float = 0.0
    coverage_breached_under_stress: bool = False
    slb_target_met: bool = False
    slb_coupon: float = 0.0
    slb_extra_interest_usd: float = 0.0
    decision: str = "PENDING"


def allocation_node(state: ClimatePortfolioState) -> ClimatePortfolioState:
    """Hierarchical Risk Parity weights (López de Prado 2016). Engine not yet audited."""
    state.hrp_weights = ExecutionEngine.optimize_hierarchical_risk_parity(
        assets=state.assets, returns_matrix=state.returns_matrix).weights
    return state


def carbon_intensity_node(state: ClimatePortfolioState) -> ClimatePortfolioState:
    """TCFD weighted average carbon intensity: sum of weight x emissions / revenue (USD m)."""
    res = SustainabilityEngine.calculate_waci(portfolio_weights=state.hrp_weights,
                                              emissions=state.emissions_tco2e,
                                              revenues_millions=state.revenues_usd_m,
                                              paris_aligned_threshold=state.mandate_waci_limit)
    state.portfolio_waci = res.waci
    state.within_waci_mandate = res.carbon_intensity_clean
    return state


def carbon_tax_node(state: ClimatePortfolioState) -> ClimatePortfolioState:
    """Debtor's interest coverage after a carbon price, net of pass-through."""
    res = SustainabilityEngine.stress_test_carbon_tax(
        revenue=state.debtor_revenue_usd_m, ebit=state.debtor_ebit_usd_m,
        interest_expense=state.debtor_interest_usd_m, scope1_emissions=state.debtor_scope1_mtco2e,
        scope2_emissions=state.debtor_scope2_mtco2e, carbon_price_per_ton=state.carbon_price_usd_per_t,
        pass_through_rate=state.pass_through_rate, covenant_interest_floor=state.min_interest_coverage)
    state.stressed_interest_coverage = res.interest_coverage_after
    state.coverage_breached_under_stress = res.is_covenant_at_risk
    return state


def slb_node(state: ClimatePortfolioState) -> ClimatePortfolioState:
    """Was the bond's emissions target met? If not, the coupon steps up."""
    bond = SustainabilityLinkedBond(isin="XS0000000000", notional=state.slb_notional_usd,
                                    base_coupon_rate=state.slb_base_coupon,
                                    target_reduction_pct=state.slb_target_reduction,
                                    baseline_emissions=state.slb_baseline_tco2e,
                                    step_up_bps=state.slb_step_up_bps, target_year=2026)
    res = SustainabilityEngine.evaluate_slb_covenant(bond=bond, reported_emissions=state.slb_reported_tco2e)
    state.slb_target_met = res.spt_achieved
    state.slb_coupon = res.active_coupon_rate
    state.slb_extra_interest_usd = res.annual_interest_penalty
    return state


def decision_node(state: ClimatePortfolioState) -> ClimatePortfolioState:
    """Mandate checks in order; the bond step-up is reported, not a reason to reject."""
    slb = (f"SLB target {'met' if state.slb_target_met else 'missed'}: coupon {state.slb_coupon:.2%}"
           f" (+${state.slb_extra_interest_usd:,.0f} a year)")
    if not state.within_waci_mandate:
        state.decision = f"REJECT: WACI {state.portfolio_waci:.1f} above the {state.mandate_waci_limit:.0f} mandate limit. {slb}"
    elif state.coverage_breached_under_stress:
        state.decision = (f"REJECT: debtor coverage falls to {state.stressed_interest_coverage:.2f}x under a "
                          f"${state.carbon_price_usd_per_t:.0f}/t carbon price. {slb}")
    else:
        state.decision = f"PASS: WACI {state.portfolio_waci:.1f}, stressed coverage {state.stressed_interest_coverage:.2f}x. {slb}"
    return state


def build_climate_portfolio() -> FinancialStateGraph[ClimatePortfolioState]:
    graph = FinancialStateGraph(ClimatePortfolioState)
    nodes = (("allocation", allocation_node), ("carbon_intensity", carbon_intensity_node),
             ("carbon_tax", carbon_tax_node), ("slb", slb_node), ("decision", decision_node))
    for name, node in nodes:
        graph.add_node(name, node)
    graph.set_entry_point("allocation")
    for (a, _), (b, _) in zip(nodes, nodes[1:], strict=False):  # consecutive pairs
        graph.add_edge(a, b)
    graph.add_edge("decision", END)
    return graph


def main() -> None:
    from fastapi.testclient import TestClient

    from lilavati.framework import create_agent_api

    fund = build_climate_portfolio().compile()
    r = fund.invoke(ClimatePortfolioState())
    print("HRP weights:", [round(w, 4) for w in r.hrp_weights])
    print(f"WACI {r.portfolio_waci:.2f} tCO2e/USD m; stressed coverage {r.stressed_interest_coverage:.2f}x")
    print(r.decision)
    print(fund.invoke(ClimatePortfolioState(emissions_tco2e=[20000.0, 45000.0, 30000.0, 90000.0])).decision)

    app = create_agent_api(graph=fund, title="Climate portfolio API")
    resp = TestClient(app).post("/api/v1/run", json={"carbon_price_usd_per_t": 150.0})
    print(f"API HTTP {resp.status_code}: {resp.json()['decision']}")
    print("To serve locally: serve_agent(fund, host='127.0.0.1', port=8000)  (no built-in auth)")


if __name__ == "__main__":
    main()
