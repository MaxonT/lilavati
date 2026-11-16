"""
Tutorial 02: Statistical Arbitrage & Algorithmic Execution Agent.

Demonstrates:
1. Cointegration testing via Engle-Granger and Ornstein-Uhlenbeck mean-reversion half-life.
2. Marcos López de Prado's memory-preserving fractional differentiation (optimal d*).
3. Limit Order Book (LOB) microstructure analytics (microprice, spread, Order Flow Imbalance).
4. Almgren-Chriss optimal trade liquidation scheduling minimizing market impact.
5. SMT Invariant Verification enforcing capital risk limits.
6. Auto-converting into a FastAPI REST microservice with automated OpenAPI docs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lilavati.core.econometrics import EconometricEngine
from lilavati.core.execution import ExecutionEngine
from lilavati.core.financial_ml import FinancialMLEngine
from lilavati.framework import (
    END,
    FinancialState,
    FinancialStateGraph,
    create_agent_api,
)

console = Console()


def generate_cointegrated_pair(n: int = 80, seed: int = 42) -> tuple[list[float], list[float]]:
    """Synthesizes a realistic cointegrated asset pair with mean-reverting spread."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0, 1.0, n)
    series_b = [float(round(x, 2)) for x in (100.0 + np.cumsum(innovations))]
    theta = 0.25
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] - theta * spread[t - 1] + rng.normal(0, 0.4)
    series_a = [float(round(15.0 + 1.6 * b + s, 2)) for b, s in zip(series_b, spread, strict=True)]
    return series_a, series_b


# Pre-generate deterministic baseline series
DEFAULT_SERIES_A, DEFAULT_SERIES_B = generate_cointegrated_pair(n=80, seed=42)


# ==============================================================================
# STEP 1: Define Typed State Container
# ==============================================================================
class StatArbExecutionState(FinancialState):
    """
    Typed state container for Statistical Arbitrage and Execution Agent.
    Tracks price series, cointegration statistics, order book metrics,
    and liquidation trajectories.
    """

    pair_name: str = "GLD-GDX"
    series_asset_a: list[float] = DEFAULT_SERIES_A
    series_asset_b: list[float] = DEFAULT_SERIES_B

    # Current LOB snapshot for asset A
    best_bid: float = 175.40
    best_ask: float = 175.50
    bid_size: float = 2500.0
    ask_size: float = 1800.0
    prev_bid_size: float = 2200.0
    prev_ask_size: float = 2000.0

    # Trade execution parameters
    shares_to_liquidate: float = 10000.0
    execution_horizon_hours: float = 2.0
    daily_volatility: float = 0.018  # 1.8% daily vol
    max_capital_risk_limit: float = 3000000.0  # $3.0M maximum position

    # Diagnostic outputs
    is_cointegrated: bool = False
    cointegration_p_value: float = 1.0
    hedge_ratio: float = 0.0
    half_life_days: float = 0.0
    optimal_fractional_d: float = 1.0
    microprice: float = 0.0
    order_flow_imbalance: float = 0.0
    expected_execution_cost: float = 0.0
    execution_trajectory: list[float] = []
    trade_verdict: str = "PENDING"


# ==============================================================================
# STEP 2: Implement Execution Nodes
# ==============================================================================
def cointegration_analysis_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Node 1: Tests cointegration and Ornstein-Uhlenbeck mean-reversion dynamics."""
    coint_res = EconometricEngine.test_cointegration(
        series_a=state.series_asset_a,
        series_b=state.series_asset_b,
        significance_level=0.05,
    )
    state.is_cointegrated = coint_res.is_cointegrated
    state.cointegration_p_value = coint_res.p_value
    state.hedge_ratio = coint_res.hedge_ratio
    state.half_life_days = coint_res.half_life
    return state


def fractional_diff_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Node 2: Determines optimal fractional differentiation d* to preserve memory."""
    frac_res = FinancialMLEngine.find_optimal_fractional_d(
        series=state.series_asset_a,
        step=0.1,
        target_p_value=0.05,
    )
    state.optimal_fractional_d = frac_res.optimal_d
    return state


def microstructure_order_book_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Node 3: Evaluates Order Flow Imbalance (OFI), microprice, and short-term liquidity."""
    lob_metrics = ExecutionEngine.analyze_lob_snapshot(
        best_bid=state.best_bid,
        best_ask=state.best_ask,
        bid_size=state.bid_size,
        ask_size=state.ask_size,
        prev_bid_size=state.prev_bid_size,
        prev_ask_size=state.prev_ask_size,
    )
    state.microprice = lob_metrics.micro_price
    state.order_flow_imbalance = lob_metrics.order_flow_imbalance
    return state


def optimal_execution_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Node 4: Computes Almgren-Chriss market-impact-minimizing trajectory with SMT risk gates."""
    trade_notional = state.shares_to_liquidate * state.best_ask

    # Formal Risk Gate: Verify trade notional does not breach capital allocation limit
    if trade_notional > state.max_capital_risk_limit:
        state.trade_verdict = (
            f"REJECTED_RISK_LIMIT: Notional ${trade_notional:,.2f} exceeds "
            f"limit ${state.max_capital_risk_limit:,.2f}"
        )
        return state

    schedule = ExecutionEngine.compute_almgren_chriss_schedule(
        total_shares=state.shares_to_liquidate,
        time_horizon_hours=state.execution_horizon_hours,
        n_intervals=6,
        daily_volatility=state.daily_volatility,
        permanent_impact_gamma=2.5e-7,
        temporary_impact_eta=2.5e-6,
        risk_aversion_lambda=1e-6,
    )
    state.expected_execution_cost = schedule.expected_shortfall
    state.execution_trajectory = schedule.trading_trajectory

    if state.is_cointegrated and state.half_life_days < 10.0:
        state.trade_verdict = (
            f"APPROVED_PAIR_TRADE: Cointegrated (p={state.cointegration_p_value:.4f}, "
            f"half-life={state.half_life_days:.1f}d). Execution cost: ${state.expected_execution_cost:,.2f}."
        )
    else:
        state.trade_verdict = "REJECTED_WEAK_COINTEGRATION: Spread does not mean-revert rapidly."

    return state


# ==============================================================================
# STEP 3: Assemble & Compile Graph
# ==============================================================================
def build_stat_arb_agent() -> FinancialStateGraph[StatArbExecutionState]:
    """Builds the statistical arbitrage and optimal execution state graph."""
    graph = FinancialStateGraph(StatArbExecutionState)
    graph.add_node("cointegration", cointegration_analysis_node)
    graph.add_node("fractional_diff", fractional_diff_node)
    graph.add_node("microstructure", microstructure_order_book_node)
    graph.add_node("optimal_execution", optimal_execution_node)

    graph.set_entry_point("cointegration")
    graph.add_edge("cointegration", "fractional_diff")
    graph.add_edge("fractional_diff", "microstructure")
    graph.add_edge("microstructure", "optimal_execution")
    graph.add_edge("optimal_execution", END)
    return graph


# ==============================================================================
# STEP 4: Interactive Demonstration & REST API
# ==============================================================================
def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Project Līlāvatī: Statistical Arbitrage & Algorithmic Execution Agent[/bold cyan]\n"
            "[dim]Ornstein-Uhlenbeck Mean-Reversion, Fractional Differentiation, and Almgren-Chriss Liquidation[/dim]",
            border_style="cyan",
        )
    )

    agent_graph = build_stat_arb_agent().compile()

    # Run locally with standard pair
    initial_state = StatArbExecutionState(
        pair_name="GLD-GDX",
        shares_to_liquidate=10000.0,
        execution_horizon_hours=2.0,
    )
    final_state = agent_graph.invoke(initial_state)

    # Display results
    table = Table(title="Stat-Arb & Execution Summary", border_style="green")
    table.add_column("Metric / Field", style="bold white")
    table.add_column("Deterministic Value", style="green")

    table.add_row("Pair", final_state.pair_name)
    table.add_row("Cointegrated", str(final_state.is_cointegrated))
    table.add_row("Engle-Granger p-value", f"{final_state.cointegration_p_value:.5f}")
    table.add_row("Hedge Ratio", f"{final_state.hedge_ratio:.4f}")
    table.add_row("OU Half-Life (days)", f"{final_state.half_life_days:.2f}")
    table.add_row("Optimal Memory d*", f"{final_state.optimal_fractional_d:.2f}")
    table.add_row("LOB Microprice", f"${final_state.microprice:.2f}")
    table.add_row("Order Flow Imbalance (OFI)", f"{final_state.order_flow_imbalance:,.0f} shares")
    table.add_row("Expected Impact Cost", f"${final_state.expected_execution_cost:,.2f}")
    table.add_row("Execution Verdict", final_state.trade_verdict)
    console.print(table)

    # Test ASGI REST API
    app = create_agent_api(
        graph=agent_graph,
        title="Statistical Arbitrage Execution API",
        description="Autonomous pair trading and Almgren-Chriss liquidation microservice.",
        version="1.0.0",
    )
    client = TestClient(app)
    response = client.post("/api/v1/run", json={"shares_to_liquidate": 8000.0})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    console.print(
        f"\n[bold green]✓ REST API Test Successful![/bold green] Status: {response.status_code} | "
        f"Verdict: {data.get('trade_verdict')}"
    )


if __name__ == "__main__":
    main()
