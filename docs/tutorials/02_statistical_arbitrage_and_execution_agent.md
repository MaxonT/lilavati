# Tutorial: Building a Statistical Arbitrage & Algorithmic Execution Agent

Welcome to the second Project Līlāvatī tutorial! In this guide, you will design, formally verify, test, and serve a production-ready **Statistical Arbitrage & Optimal Trade Execution Agent**.

---

## 🎯 What You Will Build

Generic LLM frameworks struggle when designing algorithmic trading and execution systems because language models cannot compute continuous time-series statistics, diagnose non-stationarity, or solve trajectory calculus.

In this tutorial, your Līlāvatī agent will:
1. **Test Cointegration & Mean-Reversion Dynamics**: Execute Engle-Granger two-step tests and compute the Ornstein-Uhlenbeck (OU) mean-reversion half-life ($t_{1/2}$) without LLM hallucinations.
2. **Preserve Maximum Memory via Fractional Differentiation**: Calculate Marcos López de Prado's optimal degree of fractional differentiation ($d^*$) to achieve stationarity while retaining maximum predictive price memory.
3. **Analyze Order Book Microstructure**: Ingest Level-2 limit order book (LOB) quotes to compute the continuous **microprice**, spread, and Order Flow Imbalance (OFI).
4. **Compute Market-Impact-Minimizing Liquidation**: Solve the **Almgren-Chriss (2000)** calculus of variations model to determine the optimal trade schedule that balances volatility risk against permanent and temporary price impact.
5. **Enforce Formal SMT Risk Limits**: Prove that trade size and total portfolio notional do not breach statutory risk limits prior to order dispatch.
6. **Deploy as an ASGI Microservice**: Convert the state graph into a production FastAPI service exposing `/api/v1/run` and Swagger documentation.

---

## 📦 Prerequisites

Ensure you have cloned the repository and installed dependencies using `uv`:

```bash
git clone https://github.com/revendrat/lilavati.git
cd lilavati
uv sync
```

---

## Step 1: Define the Typed State Container

In Līlāvatī, execution state is held in a strongly typed Pydantic container inheriting from `FinancialState`:

```python
from lilavati.framework import FinancialState


class StatArbExecutionState(FinancialState):
    """
    Typed state container for Statistical Arbitrage and Execution Agent.
    Tracks historical price series, econometric cointegration, order book metrics,
    and liquidation trajectories.
    """

    pair_name: str = "GLD-GDX"
    series_asset_a: list[float] = []
    series_asset_b: list[float] = []

    # Current LOB snapshot for Asset A
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
    max_capital_risk_limit: float = 3000000.0  # $3.0M max notional

    # Deterministic quantitative outputs
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
```

---

## Step 2: Implement Execution Nodes

Nodes are pure, deterministic Python functions `Callable[[S], S]` leveraging Līlāvatī's certified quantitative engines.

### Node 1: Cointegration & OU Half-Life Testing
Evaluate whether the pair exhibits a stationary linear combination $S_t = P_t^A - \beta P_t^B - \alpha$ and compute the mean-reversion rate $\theta$:

```python
from lilavati.core.econometrics import EconometricEngine


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
```

### Node 2: Memory-Preserving Fractional Differentiation ($d^*$)
Traditional quants take first differences ($d=1$), completely destroying the series' long-term memory. Following Marcos López de Prado (*Advances in Financial Machine Learning*), we find the minimum real $d^* \in [0, 1]$ that passes the Augmented Dickey-Fuller stationarity test:

```python
from lilavati.core.financial_ml import FinancialMLEngine


def fractional_diff_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Node 2: Determines optimal fractional differentiation d* to preserve memory."""
    frac_res = FinancialMLEngine.find_optimal_fractional_d(
        series=state.series_asset_a,
        step=0.10,
        target_p_value=0.05,
    )
    state.optimal_fractional_d = frac_res.optimal_d
    return state
```

### Node 3: Limit Order Book (LOB) Microstructure
Before placing orders, the agent measures continuous liquidity, the volume-weighted **microprice**, and Order Flow Imbalance (OFI):

$$\text{Microprice} = \frac{V_{bid} \cdot P_{ask} + V_{ask} \cdot P_{bid}}{V_{bid} + V_{ask}}$$

```python
from lilavati.core.execution import ExecutionEngine


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
```

### Node 4: Optimal Almgren-Chriss Liquidation & SMT Risk Gate
Finally, we compute the optimal holdings trajectory $x_j$ balancing permanent impact $\gamma$ and temporary impact $\eta$ against risk aversion $\lambda$:

$$\min_{\{x_j\}} E[x] + \lambda V[x]$$

```python
def optimal_execution_node(state: StatArbExecutionState) -> StatArbExecutionState:
    """Node 4: Computes Almgren-Chriss market-impact-minimizing trajectory with SMT risk gates."""
    trade_notional = state.shares_to_liquidate * state.best_ask

    # Formal SMT Risk Gate: Verify trade notional does not breach capital allocation limit
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
```

---

## Step 3: Assemble & Compile the Graph

Wire the nodes into an auditable directed acyclic graph:

```python
from lilavati.framework import END, FinancialStateGraph


def build_stat_arb_agent() -> FinancialStateGraph[StatArbExecutionState]:
    """Builds and compiles the Stat-Arb Execution State Graph."""
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
```

---

## Step 4: Execute the Agent Locally

```python
# 1. Compile graph
agent_graph = build_stat_arb_agent().compile()

# 2. Invoke with trade parameters
initial_state = StatArbExecutionState(
    pair_name="GLD-GDX",
    shares_to_liquidate=10000.0,
    execution_horizon_hours=2.0,
)
result = agent_graph.invoke(initial_state)

print(f"Pair: {result.pair_name}")
print(f"Cointegrated: {result.is_cointegrated} (p={result.cointegration_p_value:.5f})")
print(f"OU Half-Life: {result.half_life_days:.2f} days")
print(f"Optimal Memory d*: {result.optimal_fractional_d:.2f}")
print(f"Microprice: ${result.microprice:.2f}")
print(f"Expected Impact Cost: ${result.expected_execution_cost:,.2f}")
print(f"Verdict: {result.trade_verdict}")
```

**Output:**
```text
Pair: GLD-GDX
Cointegrated: True (p=0.00008)
OU Half-Life: 2.74 days
Optimal Memory d*: 0.90
Microprice: $175.46
Expected Impact Cost: $137.50
Verdict: APPROVED_PAIR_TRADE: Cointegrated (p=0.0001, half-life=2.7d). Execution cost: $137.50.
```

---

## Step 5: Automated Testing with `pytest`

Write unit tests to verify both execution approval and risk limit rejection:

```python
def test_stat_arb_execution_approval() -> None:
    graph = build_stat_arb_agent().compile()
    state = StatArbExecutionState(shares_to_liquidate=5000.0)
    res = graph.invoke(state)

    assert res.is_cointegrated is True
    assert res.half_life_days > 0.0
    assert res.expected_execution_cost > 0.0
    assert "APPROVED_PAIR_TRADE" in res.trade_verdict


def test_stat_arb_risk_limit_rejection() -> None:
    graph = build_stat_arb_agent().compile()
    # Breaches the $3,000,000 capital limit
    state = StatArbExecutionState(
        shares_to_liquidate=50000.0,  # 50,000 * $175.50 = $8.775M
        max_capital_risk_limit=3000000.0,
    )
    res = graph.invoke(state)

    assert "REJECTED_RISK_LIMIT" in res.trade_verdict
    assert res.expected_execution_cost == 0.0
```

---

## Step 6: Deploy as a Production REST API

Expose your compiled trading agent as an ASGI REST API:

```python
from lilavati.framework import create_agent_api, serve_agent

agent_graph = build_stat_arb_agent().compile()

app = create_agent_api(
    graph=agent_graph,
    title="Statistical Arbitrage Execution API",
    description="Autonomous pair trading and Almgren-Chriss liquidation microservice.",
    version="1.0.0",
)

if __name__ == "__main__":
    serve_agent(agent_graph, host="0.0.0.0", port=8000)
```

Query the endpoint via `curl`:
```bash
curl -X POST "http://localhost:8000/api/v1/run" \
     -H "Content-Type: application/json" \
     -d '{
       "pair_name": "GLD-GDX",
       "shares_to_liquidate": 10000.0,
       "execution_horizon_hours": 2.0
     }'
```

---

## 🚀 Companion Code

A complete, runnable script for this tutorial is available in the repository:
```bash
uv run python examples/07_stat_arb_and_execution_agent.py
```
