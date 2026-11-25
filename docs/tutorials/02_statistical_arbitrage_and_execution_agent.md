# Tutorial 02: Statistical Arbitrage and Optimal Execution, Checked Against a Known Answer

**Who this is for:** quant students, traders and developers learning pairs trading and execution.
**Time:** about 60 minutes. **Cost:** none (no API key, no LLM). Alpha software, not for live trading.

## Learning objectives

1. Test a pair for cointegration (Engle-Granger) and estimate the spread's mean-reversion half-life.
2. Validate those estimates by simulating a pair whose true parameters you know, the habit that
   caught real bugs in this repository.
3. Find the smallest fractional-differencing order that makes a price series stationary
   (López de Prado, *Advances in Financial Machine Learning*, ch. 5).
4. Read the top of an order book (microprice, order-flow imbalance) and schedule a liquidation with
   Almgren-Chriss (2000), with the units each parameter needs.
5. Gate the order with a risk limit before scheduling it, test the workflow, and serve it locally.

## Where this fits

No LLM here: these are the deterministic tools an agent calls. An LLM asked for a half-life or an
execution schedule computes it in text; [Tutorial 07](07_tools_versus_mental_math.md) measures how
often that goes wrong. Steps 1–4 form one module; save them as `stat_arb_agent.py`. Before trusting a
rule chosen as the best of many backtests, see [Tutorial 09](09_is_this_backtest_real.md) (Deflated
Sharpe Ratio, probability of backtest overfitting).

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync --extra dev
```

## Step 1: Simulate a pair with a known answer

Real prices never tell you the true half-life, so start with data where you know it. Asset B is a
random walk; asset A is `15 + 1.6 × B + spread`, and the spread mean-reverts by 25% a day
(θ = 0.25), so its true half-life is −ln 2 / ln(1 − θ) ≈ 2.41 days. The names are labels on
synthetic data, not real tickers.

```python title="stat_arb_agent.py"
"""Statistical arbitrage and execution workflow from Tutorial 02 (save as stat_arb_agent.py)."""

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
```

## Step 2: Typed state, with units

```python title="stat_arb_agent.py"
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
```

## Step 3: Cointegration, fractional differencing, order book

**Cointegration.** Engle-Granger regresses A on B, then runs an ADF test on the residual spread
(MacKinnon p-values). The half-life uses the exact AR(1) formula −ln 2 / ln φ, not the common
shortcut ln 2 / (1 − φ), which is 27% too high at φ = e^−0.5 (a real bug fixed in this engine:
[econometrics validation](../validation/econometrics.md)).

**Fractional differencing.** First differences (d = 1) make prices stationary but throw away
memory; the smallest d on a grid that passes ADF keeps as much memory as possible. If no grid point
passes, the engine raises instead of returning a made-up d* = 1.0
([financial-ML validation](../validation/financial_ml.md)).

**Order book.** Microprice = (bid size × ask + ask size × bid) / (bid size + ask size): it leans
towards the side with less size, where the next trade is likelier.

```python title="stat_arb_agent.py"
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
```

## Step 4: Risk limit, then the Almgren-Chriss schedule

The limit is a plain comparison at the arrival (mid) price, done *before* any schedule is computed.
Almgren-Chriss then trades off market impact against price risk. Mind the units (they follow the
paper, with the trading day as the time unit): `daily_volatility` is a fraction of price, `γ` is in
$/share², `η` in ($/share)/(share/day), `λ` in 1/$. Mixing fractional and price-unit volatility was
a real bug in this engine before its audit.

```python title="stat_arb_agent.py"
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
```

```python title="stat_arb_agent.py"
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
```

## Step 5: Run it

```python
from stat_arb_agent import TRUE_HALF_LIFE, StatArbExecutionState, build_stat_arb_agent

agent = build_stat_arb_agent().compile()
r = agent.invoke(StatArbExecutionState())
print(f"Cointegrated: {r.is_cointegrated} (p={r.cointegration_p_value:.5f}), hedge ratio {r.hedge_ratio:.4f} (true 1.6)")
print(f"Half-life: {r.half_life_days:.2f} days (true {TRUE_HALF_LIFE:.2f})")
print(f"d*: {r.optimal_fractional_d:.1f}; microprice ${r.microprice:.4f}; OFI {r.order_flow_imbalance:,.0f} shares")
print(f"Holdings by interval: {[round(x) for x in r.execution_trajectory]}")
print(r.trade_verdict)
```

Output:

```text
Cointegrated: True (p=0.00008), hedge ratio 1.6525 (true 1.6)
Half-life: 2.38 days (true 2.41)
d*: 0.9; microprice $175.4581; OFI 500 shares
Holdings by interval: [10000, 8178, 6441, 4773, 3154, 1569, 0]
APPROVE: cointegrated (p=0.00008), half-life 2.38 days; expected cost $825.23
```

The estimates land close to the values used to simulate the data. That is the point of Step 1:
you have checked the tools, not just run them.

## Step 6: Test it

Save as `tests/test_stat_arb_agent.py` (with `stat_arb_agent.py` in the repository root):

```python title="tests/test_stat_arb_agent.py"
import numpy as np
import pytest

from stat_arb_agent import TRUE_HALF_LIFE, StatArbExecutionState, build_stat_arb_agent

AGENT = build_stat_arb_agent().compile()


def test_estimates_recover_the_simulated_process() -> None:
    r = AGENT.invoke(StatArbExecutionState())
    assert r.is_cointegrated
    assert r.hedge_ratio == pytest.approx(1.6, rel=0.05)
    assert r.half_life_days == pytest.approx(TRUE_HALF_LIFE, rel=0.10)
    assert r.trade_verdict.startswith("APPROVE")


def test_schedule_sells_everything() -> None:
    r = AGENT.invoke(StatArbExecutionState())
    assert r.execution_trajectory[0] == pytest.approx(10_000)
    assert r.execution_trajectory[-1] == pytest.approx(0.0, abs=1e-6)


def test_order_over_the_notional_limit_is_rejected_before_scheduling() -> None:
    r = AGENT.invoke(StatArbExecutionState(shares_to_liquidate=50_000))  # about $8.8m at mid
    assert r.trade_verdict.startswith("REJECT: order notional")
    assert r.expected_execution_cost == 0.0 and r.execution_trajectory == []


def test_independent_random_walks_are_not_traded() -> None:
    rng = np.random.default_rng(7)
    a = list(100 + np.cumsum(rng.normal(0, 1, 80)))
    b = list(100 + np.cumsum(rng.normal(0, 1, 80)))
    r = AGENT.invoke(StatArbExecutionState(series_asset_a=a, series_asset_b=b))
    assert not r.is_cointegrated
    assert r.trade_verdict.startswith("REJECT: no fast mean reversion")
```

```bash
uv run pytest tests/test_stat_arb_agent.py    # 4 passed
```

## Step 7: Serve it locally

```python
from fastapi.testclient import TestClient

from lilavati.framework import create_agent_api
from stat_arb_agent import build_stat_arb_agent

app = create_agent_api(graph=build_stat_arb_agent().compile(), title="Stat-arb execution API")
resp = TestClient(app).post("/api/v1/run", json={"shares_to_liquidate": 5000.0})
print(resp.status_code, resp.json()["trade_verdict"])
```

To run a server instead: `serve_agent(graph, host="127.0.0.1", port=8000)`. There is no built-in
authentication; keep it local or put it behind your own proxy (`SECURITY.md`). The complete script
is [`examples/07_stat_arb_and_execution_agent.py`](https://github.com/revendrat/lilavati/blob/main/examples/07_stat_arb_and_execution_agent.py).

## What can go wrong

- **Short samples underestimate slow mean reversion.** Simulate with θ = 0.05: the true half-life is
  13.5 days, but 80 days of data estimate 4.98 days, the pair still tests as cointegrated
  (p = 0.034) and the workflow approves a trade on a spread that reverts far more slowly than it
  thinks. A rule like "half-life < 10 days" needs enough data and a confidence interval.
- **Using the half-life shortcut.** ln 2 / (1 − φ) instead of −ln 2 / ln φ (fixed in this engine).
- **A search that "succeeds" with a placeholder.** The fractional-d search once returned d* = 1.0
  and p = 1.0 when nothing qualified; it now raises.
- **Unit mismatches in execution models.** Fractional versus price-unit volatility, per-hour versus
  per-day impact: each produces a schedule that looks plausible and is wrong.
- **Engle-Granger is not symmetric.** Regressing A on B and B on A can give different answers;
  decide which leg is the dependent variable and say so.

## Using this with Claude Code and MCP

- **Contributing code:** the [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
  holds the rules used here (exact formulas, simulated-process tests, units, solvers that raise).
- **From an AI assistant:** `lilavati-mcp-econometrics`, `lilavati-mcp-financial-ml` and
  `lilavati-mcp-portfolio-exec` expose these engines as tools ([MCP setup](../mcp-setup.md)). Ask,
  for example, *"Test these two price series for cointegration and give me the half-life."*

## Exercises

1. Set `TRUE_THETA = 0.05` and rerun. Reproduce the half-life underestimate above, then find how
   many days of data it takes before the estimate is within 20% of the truth.
2. Set `risk_aversion_lambda` to `0.0` and to `1e-4`. Describe the schedules (a straight line at 0;
   front-loaded selling at 1e-4) and explain the change in expected cost.
3. Replace asset A with an independent random walk. Which node now rejects the trade, and why?
4. Run the fractional-differencing search on 500 simulated days instead of 80. Does d* change?

## For instructors

- **Discussion:** the workflow approved a pair whose true half-life is 13.5 days. Which statistic
  should have warned you, and what would a sound trading rule look like?
- **Lab extension:** bootstrap the half-life estimate (resample the spread's innovations) to get a
  confidence interval, and gate trades on its upper bound.
- **Link to LLMs:** ask a model, without tools, for the half-life of the Step 1 spread given φ, and
  compare its formula with the exact one ([Tutorial 07](07_tools_versus_mental_math.md)).
