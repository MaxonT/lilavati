# Tutorial 04: Climate Risk in a Portfolio: Carbon Intensity, Carbon-Tax Stress and Sustainability-Linked Bonds

**Who this is for:** sustainable-finance students, ESG and credit analysts, portfolio managers.
**Time:** about 60 minutes. **Cost:** none (no API key, no LLM). Alpha software; not investment advice.

## Learning objectives

1. Compute a portfolio's weighted average carbon intensity (WACI, TCFD) and check it against a
   fund mandate.
2. Stress a corporate debtor's interest coverage under a carbon price, keeping units consistent.
3. Decide whether a sustainability-linked bond (SLB) met its emissions target, including the case
   exactly at the target, and compute the coupon step-up.
4. See how a risk-based allocation (Hierarchical Risk Parity) can conflict with a carbon mandate.

## Where this fits

No LLM here: these are the deterministic tools an agent calls. In this repository's benchmark,
models answering WACI questions without tools gave values as far off as 1,000 times too small, and
judged a missed SLB target as met ([results](https://github.com/revendrat/lilavati/blob/main/results/ANALYSIS.md),
[Tutorial 06](06_analysing_and_publishing_results.md)). Steps 1–3 form one module; save them as
`climate_portfolio.py`. The fund, companies and bond are synthetic.

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync --extra dev
```

## Step 1: Typed state, with units on every field

The carbon-tax engine multiplies emissions by a price and subtracts the result from EBIT, so the
units must match: emissions in million tCO2e give a tax in USD millions, matching financials in USD
millions. An earlier version of this tutorial derived Scope 1 and Scope 2 by splitting a total
70/30; inputs should be reported figures, not invented splits. The WACI limit of 100 is an
illustrative fund mandate, not a regulatory or "Paris-aligned" figure.

```python title="climate_portfolio.py"
"""Climate-risk portfolio workflow from Tutorial 04 (save as climate_portfolio.py)."""

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
```

## Step 2: Allocation, carbon intensity, carbon-tax stress

**HRP** clusters assets by correlation and allocates risk down the tree, without inverting the
covariance matrix (López de Prado 2016). The engine has **not yet been audited** against a reference
([validation status](../validation/index.md)), and 8 observations are far too few for a stable
covariance estimate; treat the weights as illustrative.

**WACI** (TCFD) is Σ wᵢ × emissionsᵢ / revenueᵢ, in tCO2e per USD million of revenue.

**Carbon tax.** Net cost = (Scope 1 + Scope 2) × price × (1 − pass-through), deducted from EBIT.
Including Scope 2 is a conservative choice; many carbon taxes apply to Scope 1 only (pass Scope 2 = 0).

```python title="climate_portfolio.py"
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
```

**SLB.** The target is met when reported emissions ≤ baseline × (1 − target). The engine compares
levels with a tolerance, so a reduction of exactly 30% passes; computing the ratio
`(500,000 − 350,000) / 500,000` in floating point and comparing it with 0.30 can fail by rounding
(a real bug fixed in this engine: [sustainability validation](../validation/sustainability.md)).

## Step 3: Decide

```python title="climate_portfolio.py"
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
```

## Step 4: Run it

```python
from climate_portfolio import ClimatePortfolioState, build_climate_portfolio

fund = build_climate_portfolio().compile()
r = fund.invoke(ClimatePortfolioState())
print("HRP weights:", [round(w, 4) for w in r.hrp_weights])
print(f"WACI {r.portfolio_waci:.2f} tCO2e/USD m; stressed coverage {r.stressed_interest_coverage:.2f}x")
print(r.decision)
r2 = fund.invoke(ClimatePortfolioState(emissions_tco2e=[20000.0, 45000.0, 30000.0, 90000.0]))
print(r2.decision)
```

Output:

```text
HRP weights: [0.1013, 0.0764, 0.263, 0.5594]
WACI 103.02 tCO2e/USD m; stressed coverage 5.15x
REJECT: WACI 103.0 above the 100 mandate limit. SLB target missed: coupon 4.50% (+$625,000 a year)
PASS: WACI 61.1, stressed coverage 5.15x. SLB target missed: coupon 4.50% (+$625,000 a year)
```

HRP puts 56% in TransitionUtility, the asset with the lowest return volatility (0.58% a period
against 0.84–1.66% for the others) and also by far the highest carbon intensity (150 tCO2e per
USD million). An equal-weight portfolio of the same four assets has a WACI of 67.5, inside the
mandate. A risk-based allocator has no reason to care about carbon; the mandate check has to. The
second run halves TransitionUtility's emissions and the portfolio passes.

The bond missed its target: 380,000 t is 24% below the 500,000 t baseline, short of 30%, so the
coupon steps up 25 bp, which costs the issuer USD 625,000 a year on USD 250 million.

## Step 5: Test it

Save as `tests/test_climate_portfolio.py` (with `climate_portfolio.py` in the repository root):

```python title="tests/test_climate_portfolio.py"
import pytest

from climate_portfolio import ClimatePortfolioState, build_climate_portfolio

FUND = build_climate_portfolio().compile()


def test_waci_matches_the_formula_with_the_weights_used() -> None:
    r = FUND.invoke(ClimatePortfolioState())
    expected = sum(w * e / rev for w, e, rev in zip(r.hrp_weights, r.emissions_tco2e, r.revenues_usd_m, strict=True))
    assert r.portfolio_waci == pytest.approx(expected)
    assert sum(r.hrp_weights) == pytest.approx(1.0)


def test_carbon_tax_reduces_coverage_by_the_net_tax() -> None:
    r = FUND.invoke(ClimatePortfolioState())
    net_tax = (0.154 + 0.066) * 75.0 * (1 - 0.50)  # million t x $/t = USD m
    assert r.stressed_interest_coverage == pytest.approx((240.0 - net_tax) / 45.0)


def test_slb_target_exactly_met_does_not_step_up() -> None:
    r = FUND.invoke(ClimatePortfolioState(slb_reported_tco2e=350_000.0))  # exactly 30% below 500,000
    assert r.slb_target_met and r.slb_coupon == pytest.approx(0.0425)


def test_missed_target_steps_up_the_coupon() -> None:
    r = FUND.invoke(ClimatePortfolioState())  # 380,000 is only 24% below baseline
    assert not r.slb_target_met
    assert r.slb_coupon == pytest.approx(0.0450)
    assert r.slb_extra_interest_usd == pytest.approx(250_000_000 * 0.0025)
```

```bash
uv run pytest tests/test_climate_portfolio.py    # 4 passed
```

## Step 6: Serve it locally

```python
from fastapi.testclient import TestClient

from climate_portfolio import build_climate_portfolio
from lilavati.framework import create_agent_api

app = create_agent_api(graph=build_climate_portfolio().compile(), title="Climate portfolio API")
resp = TestClient(app).post("/api/v1/run", json={"carbon_price_usd_per_t": 150.0})
print(resp.status_code, resp.json()["decision"])
```

To run a server: `serve_agent(graph, host="127.0.0.1", port=8000)`. No built-in authentication;
keep it local (`SECURITY.md`). Complete script:
[`examples/09_climate_sustainable_portfolio_agent.py`](https://github.com/revendrat/lilavati/blob/main/examples/09_climate_sustainable_portfolio_agent.py).

## What can go wrong

- **Mismatched units in the carbon-tax stress.** Emissions in tCO2e with financials in USD millions
  produce a tax a million times too large; the engine's docstring now states the rule.
- **A target met exactly, reported as missed.** Fixed in the engine; Step 5 tests it.
- **Invented inputs.** Splitting a total into Scope 1 and 2 by a fixed ratio, or proxying any
  input, changes results without anyone noticing.
- **Optimisers that ignore the mandate.** HRP minimises risk concentration, not carbon; check the
  mandate after allocating, or put it into the optimisation as a constraint.
- **Unaudited engines.** HRP is not yet validated against a reference; do not rely on its weights.
- **Scope of WACI.** WACI here uses company emissions and revenue; it is not PCAF financed emissions.

## Using this with Claude Code and MCP

- **Contributing code:** the [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
  (units in docstrings, tolerance at thresholds; auditing HRP is an open task in the roadmap).
- **From an AI assistant:** `lilavati-mcp-sustainability` exposes WACI, carbon-tax and SLB tools
  ([MCP setup](../mcp-setup.md)). Ask, for example, *"Did this sustainability-linked bond meet its
  target, and what is the new coupon?"*

## Exercises

1. Raise the carbon price to USD 150 and then 300 per tonne. At what price does the debtor's
   coverage fall below 2.5x?
2. Set `pass_through_rate` to 0 and to 1. Explain the two extremes.
3. Add the mandate as a constraint: scale down the highest-intensity holding until WACI is at the
   limit, renormalise, and test that WACI equals the limit within a tolerance.
4. Audit HRP: reproduce the allocation in López de Prado (2016) on its published example, and
   write the validation page (an open roadmap task).

## For instructors

- **Discussion:** the lowest-risk portfolio is the most carbon-intensive. Who should decide the
  trade-off, and where should it be encoded?
- **Lab extension:** replace the 8 periods with two years of simulated daily returns and compare
  the HRP weights. How stable are they?
- **Link to LLMs:** ask a model, without tools, whether the bond met its target (380,000 against a
  500,000 baseline and a 30% target), and compare its reasoning with Step 2.
