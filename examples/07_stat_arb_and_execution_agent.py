"""
Companion script for Tutorial 02 (docs/tutorials/02_statistical_arbitrage_and_execution_agent.md):
pairs trading and optimal execution on a simulated pair whose true parameters are known.

1. Simulate a cointegrated pair (true hedge ratio 1.6, true half-life about 2.41 days).
2. Engle-Granger cointegration, OU half-life, fractional-differencing d*, order-book microprice.
3. A notional risk limit (plain comparison), then an Almgren-Chriss (2000) liquidation schedule.
4. The same graph as a REST API, called in-process.

No LLM is involved. Synthetic data; not for live trading.

    uv run python examples/07_stat_arb_and_execution_agent.py
"""

from __future__ import annotations

import math

import numpy as np

from lilavati.core.econometrics import EconometricEngine
from lilavati.core.execution import ExecutionEngine
from lilavati.core.financial_ml import FinancialMLEngine
from lilavati.framework import END, FinancialState, FinancialStateGraph

TRUE_BETA, TRUE_THETA = 1.6, 0.25  # used to simulate the pair, so we can check the estimates


def simulate_cointegrated_pair(n: int = 80, seed: int = 42) -> tuple[list[float], list[float]]:
    """Synthetic daily prices. B is a random walk; A = 15 + 1.6 * B + spread, where the spread is
    an AR(1): s_t = (1 - theta) * s_{t-1} + noise. True half-life: -ln 2 / ln(1 - theta) days."""
    rng = np.random.default_rng(seed)
    b = [round(float(x), 2) for x in 100.0 + np.cumsum(rng.normal(0, 1.0, n))]
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = (1 - TRUE_THETA) * s[t - 1] + rng.normal(0, 0.4)
    a = [round(15.0 + TRUE_BETA * bt + st, 2) for bt, st in zip(b, s, strict=True)]
    return a, b


SERIES_A, SERIES_B = simulate_cointegrated_pair()
TRUE_HALF_LIFE = -math.log(2) / math.log(1 - TRUE_THETA)  # 2.41 days


class StatArbExecutionState(FinancialState):
    """Typed state. Prices in USD per share; sizes in shares; time in trading days or hours."""

    pair_name: str = "synthetic pair A/B"
    series_asset_a: list[float] = SERIES_A
    series_asset_b: list[float] = SERIES_B

    # Top of book for asset A
    best_bid: float = 175.40
    best_ask: float = 175.50
    bid_size: float = 2500.0
    ask_size: float = 1800.0
    prev_bid_size: float = 2200.0
    prev_ask_size: float = 2000.0

    # Order to work and limits
    shares_to_liquidate: float = 10000.0
    execution_horizon_hours: float = 2.0
    daily_volatility: float = 0.018      # fraction of price per day (1.8%)
    max_order_notional: float = 3_000_000.0  # USD, at the arrival (mid) price

    # Outputs
    is_cointegrated: bool = False
    cointegration_p_value: float = 1.0
    hedge_ratio: float = 0.0
    half_life_days: float = 0.0
    optimal_fractional_d: float = 1.0
    microprice: float = 0.0
    order_flow_imbalance: float = 0.0
    order_notional: float = 0.0
    expected_execution_cost: float = 0.0  # USD, Almgren-Chriss expected shortfall
    execution_trajectory: list[float] = []  # shares still held at each interval
    trade_verdict: str = "PENDING"


def cointegration_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Engle-Granger test of A on B; hedge ratio and OU half-life of the spread."""
    res = EconometricEngine.test_cointegration(series_a=state.series_asset_a,
                                               series_b=state.series_asset_b,
                                               significance_level=0.05)
    state.is_cointegrated = res.is_cointegrated
    state.cointegration_p_value = res.p_value
    state.hedge_ratio = res.hedge_ratio
    state.half_life_days = res.half_life
    return state


def fractional_diff_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Smallest fractional-differencing order d* (grid step 0.1) whose series passes ADF at 5%."""
    res = FinancialMLEngine.find_optimal_fractional_d(series=state.series_asset_a, step=0.1,
                                                      target_p_value=0.05)
    state.optimal_fractional_d = res.optimal_d
    return state


def order_book_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Size-weighted microprice and order-flow imbalance from the top of the book."""
    lob = ExecutionEngine.analyze_lob_snapshot(
        best_bid=state.best_bid, best_ask=state.best_ask, bid_size=state.bid_size,
        ask_size=state.ask_size, prev_bid_size=state.prev_bid_size, prev_ask_size=state.prev_ask_size)
    state.microprice = lob.micro_price
    state.order_flow_imbalance = lob.order_flow_imbalance
    return state


def execution_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Risk limit first (plain comparison), then the Almgren-Chriss (2000) liquidation schedule."""
    arrival = 0.5 * (state.best_bid + state.best_ask)
    state.order_notional = state.shares_to_liquidate * arrival
    if state.order_notional > state.max_order_notional:
        state.trade_verdict = (f"REJECT: order notional ${state.order_notional:,.0f} exceeds the "
                               f"${state.max_order_notional:,.0f} limit")
        return state
    schedule = ExecutionEngine.compute_almgren_chriss_schedule(
        total_shares=state.shares_to_liquidate, time_horizon_hours=state.execution_horizon_hours,
        n_intervals=6, daily_volatility=state.daily_volatility, arrival_price=arrival,
        permanent_impact_gamma=2.5e-7,   # $/share^2
        temporary_impact_eta=2.5e-6,     # ($/share)/(share/day)
        risk_aversion_lambda=1e-6,       # 1/$
    )
    state.expected_execution_cost = schedule.expected_shortfall
    state.execution_trajectory = schedule.trading_trajectory
    if state.is_cointegrated and state.half_life_days < 10.0:
        state.trade_verdict = (f"APPROVE: cointegrated (p={state.cointegration_p_value:.5f}), half-life "
                               f"{state.half_life_days:.2f} days; expected cost ${state.expected_execution_cost:,.2f}")
    else:
        state.trade_verdict = "REJECT: no fast mean reversion (not cointegrated, or half-life >= 10 days)"
    return state


def build_stat_arb_agent() -> FinancialStateGraph[StatArbExecutionState]:
    graph = FinancialStateGraph(StatArbExecutionState)
    graph.add_node("cointegration", cointegration_node)
    graph.add_node("fractional_diff", fractional_diff_node)
    graph.add_node("order_book", order_book_node)
    graph.add_node("execution", execution_node)
    graph.set_entry_point("cointegration")
    graph.add_edge("cointegration", "fractional_diff")
    graph.add_edge("fractional_diff", "order_book")
    graph.add_edge("order_book", "execution")
    graph.add_edge("execution", END)
    return graph


def main() -> None:
    from fastapi.testclient import TestClient

    from lilavati.framework import create_agent_api

    agent = build_stat_arb_agent().compile()
    r = agent.invoke(StatArbExecutionState())
    print(f"Cointegrated: {r.is_cointegrated} (p={r.cointegration_p_value:.5f}), hedge ratio {r.hedge_ratio:.4f} (true {TRUE_BETA})")
    print(f"Half-life: {r.half_life_days:.2f} days (true {TRUE_HALF_LIFE:.2f})")
    print(f"d*: {r.optimal_fractional_d:.1f}; microprice ${r.microprice:.4f}; OFI {r.order_flow_imbalance:,.0f} shares")
    print(f"Holdings by interval: {[round(x) for x in r.execution_trajectory]}")
    print(r.trade_verdict)
    print("Over the limit:", agent.invoke(StatArbExecutionState(shares_to_liquidate=50_000)).trade_verdict)

    app = create_agent_api(graph=agent, title="Stat-arb execution API")
    resp = TestClient(app).post("/api/v1/run", json={"shares_to_liquidate": 5000.0})
    print(f"API HTTP {resp.status_code}: {resp.json()['trade_verdict']}")
    print("To serve locally: serve_agent(agent, host='127.0.0.1', port=8000)  (no built-in auth)")


if __name__ == "__main__":
    main()
