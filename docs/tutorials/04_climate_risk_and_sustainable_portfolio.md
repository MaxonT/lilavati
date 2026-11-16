# Tutorial: Building a Climate Risk & Sustainable Portfolio Optimization Agent

In this tutorial, you will design, formally verify, test, and serve a production-ready **Climate Transition Risk & Sustainable Portfolio Optimization Agent**.

---

## 🎯 What You Will Build

Institutional asset managers and banks increasingly face statutory ESG and climate mandates (SFDR Article 8/9, TCFD, CSRD, and SEC climate rules). Generic LLM agents fail in this domain because they cannot compute carbon intensity math, evaluate non-linear carbon tax balance sheet shocks, or optimize risk parity weights.

In this tutorial, your Līlāvatī agent will:
1. **Compute TCFD Weighted Average Carbon Intensity (WACI)**: Measure Scope 1 and Scope 2 emissions normalized by revenue across portfolio assets:
   $$\text{WACI} = \sum_{i=1}^N w_i \left( \frac{\text{Emissions}_i}{\text{Revenue}_i} \right)$$
2. **Stress-Test Corporate Debt Under Carbon Price Shocks**: Model shadow or statutory carbon taxes ($/tCO2e) on corporate EBIT, operating margins, and debt interest coverage headroom ($EBIT / \text{Interest}$).
3. **Audit Sustainability-Linked Bond (SLB) Covenants**: Formally determine whether an issuer achieved its Sustainability Performance Target (SPT) or whether coupon step-up penalties (e.g. +25 bps) trigger.
4. **Allocate Capital via Hierarchical Risk Parity (HRP)**: Leverage machine learning clustering (de Prado) to allocate risk-balanced portfolio weights without encountering matrix inversion singularities or unstable mean-variance allocations.
5. **Enforce Formal ESG Mandate Gates**: Programmatically block capital allocations that violate Paris-aligned decarbonization limits or introduce unhedged transition risk defaults.
6. **Serve as an ASGI REST API**: Expose the sustainable portfolio agent as a high-performance microservice.

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

    # Transition risk stress testing for key corporate debtor ($ Millions)
    holding_revenue: float = 1200.0
    holding_ebit: float = 240.0
    holding_interest_expense: float = 45.0
    holding_taxable_emissions_m_tons: float = 0.22  # 0.22M tons
    carbon_tax_price_per_ton: float = 75.0  # $75/tCO2e shock
    pass_through_rate: float = 0.50
    covenant_interest_floor: float = 2.50

    # Sustainability-Linked Bond (SLB) holding
    slb_isin: str = "US1234567890"
    slb_notional_usd: float = 250000000.0
    slb_base_coupon_rate: float = 0.0425
    slb_spt_reduction_target_pct: float = 0.30
    slb_baseline_emissions_tons: float = 500000.0
    slb_reported_emissions_tons: float = 380000.0
    slb_step_up_bps: float = 25.0

    # Output diagnostics
    hrp_weights: list[float] = []
    portfolio_waci: float = 0.0
    is_waci_paris_compliant: bool = False
    stressed_interest_coverage: float = 0.0
    is_covenant_at_risk: bool = False
    slb_spt_achieved: bool = False
    slb_annual_interest_penalty_usd: float = 0.0
    allocation_verdict: str = "PENDING"
```

---

## Step 2: Implement Execution Nodes

### Node 1: Hierarchical Risk Parity (HRP) Allocation
Calculate diversified risk weights using graph tree clustering instead of matrix inversion:

```python
from lilavati.core.execution import ExecutionEngine


def hrp_allocation_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 1: Computes Hierarchical Risk Parity (HRP) optimal asset weights."""
    hrp_res = ExecutionEngine.optimize_hierarchical_risk_parity(
        assets=state.assets,
        returns_matrix=state.returns_matrix,
    )
    state.hrp_weights = hrp_res.weights
    return state
```

### Node 2: TCFD WACI Carbon Intensity
Evaluate portfolio weighted emissions against statutory thresholds:

```python
from lilavati.core.sustainability import SustainabilityEngine


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
```

### Node 3: Carbon Tax Transition Shock Modeling
Stress-test corporate debtors under statutory carbon pricing:

```python
def carbon_tax_stress_test_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 3: Stress-tests corporate debtor under carbon price shock."""
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
```

### Node 4: Sustainability-Linked Bond (SLB) Covenant Verification
Verify whether debtor achieved its Sustainability Performance Target (SPT):

```python
from lilavati.schemas.sustainability import SustainabilityLinkedBond


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
```

### Node 5: ESG Policy Gate
```python
def esg_mandate_gate_node(state: SustainablePortfolioState) -> SustainablePortfolioState:
    """Node 5: Enforces formal ESG mandate gates before approving capital allocation."""
    if not state.is_waci_paris_compliant:
        state.allocation_verdict = (
            f"REJECTED_WACI_EXCEEDED: Portfolio WACI ({state.portfolio_waci:.1f} tCO2e/$M) "
            f"exceeds mandate limit ({state.paris_aligned_waci_limit:.1f} tCO2e/$M)."
        )
        return state

    if state.is_covenant_at_risk:
        state.allocation_verdict = (
            f"REJECTED_TRANSITION_RISK: Carbon tax shock drops interest coverage "
            f"to {state.stressed_interest_coverage:.2f}x (below {state.covenant_interest_floor:.2f}x floor)."
        )
        return state

    state.allocation_verdict = (
        f"APPROVED_SUSTAINABLE_ALLOCATION: Paris-compliant WACI ({state.portfolio_waci:.1f} tCO2e/$M). "
        f"Stressed coverage ({state.stressed_interest_coverage:.2f}x) resilient."
    )
    return state
```

---

## Step 3: Assemble & Compile the Graph

```python
from lilavati.framework import END, FinancialStateGraph


def build_sustainable_portfolio_agent() -> FinancialStateGraph[SustainablePortfolioState]:
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
```

---

## Step 4: Run Locally & Test

```python
agent_graph = build_sustainable_portfolio_agent().compile()
result = agent_graph.invoke(SustainablePortfolioState())

print(f"Portfolio WACI: {result.portfolio_waci:.2f} tCO2e / $M rev")
print(f"Paris Compliant: {result.is_waci_paris_compliant}")
print(f"SLB SPT Achieved: {result.slb_spt_achieved}")
print(f"Verdict: {result.allocation_verdict}")
```

**Output:**
```text
Portfolio WACI: 103.02 tCO2e / $M rev
Paris Compliant: False
SLB SPT Achieved: False
Verdict: REJECTED_WACI_EXCEEDED: Portfolio WACI (103.0 tCO2e/$M) exceeds mandate limit (100.0 tCO2e/$M).
```

---

## Step 5: Convert into a Production REST API

```python
from lilavati.framework import create_agent_api, serve_agent

agent_graph = build_sustainable_portfolio_agent().compile()
app = create_agent_api(
    graph=agent_graph,
    title="Sustainable Portfolio Climate API",
    description="Autonomous TCFD WACI, carbon tax stress testing, and HRP allocation service.",
    version="1.0.0",
)

if __name__ == "__main__":
    serve_agent(agent_graph, host="0.0.0.0", port=8000)
```

---

## 🚀 Companion Code

A complete, runnable script for this tutorial is available in the repository:
```bash
uv run python examples/09_climate_sustainable_portfolio_agent.py
```
