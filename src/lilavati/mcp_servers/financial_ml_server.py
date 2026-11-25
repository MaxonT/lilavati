"""
FastMCP Server for Financial Machine Learning.

Exposes financial ML primitives from López de Prado (2018) and the backtest-overfitting statistics of
Bailey and López de Prado:
1. Fractional differentiation (the smallest d that makes a series stationary)
2. Triple-barrier labelling
3. The Deflated Sharpe Ratio (selection bias, non-normality, sample length)
4. The probability of backtest overfitting (CSCV)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastmcp import FastMCP

from lilavati.core.backtest_overfitting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from lilavati.core.financial_ml import FinancialMLEngine
from lilavati.validation_status import with_validation_status

mcp = FastMCP(
    name="lilavati-financial-ml",
)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def find_optimal_fractional_diff(
    series: list[float],
    step: float = 0.05,
    target_p_value: float = 0.01,
) -> dict[str, Any]:
    """
    Finds the smallest d on the grid {0, step, 2*step, ..., 1} whose fixed-width-window
    fractionally differenced series (López de Prado 2018, ch. 5; weight threshold 1e-4) has
    ADF p-value <= target_p_value (constant, AIC lag order). Apply it to log prices if you
    follow López de Prado. Returns optimal_d, the ADF p-value, the Pearson correlation between
    the differenced series and the original at the same time indices, and the differenced
    series (aligned with the last points of the input). Raises ValueError for non-finite
    input, too-short series, or when no d in [0, 1] reaches the target.
    """
    result = FinancialMLEngine.find_optimal_fractional_d(
        series=series,
        step=step,
        target_p_value=target_p_value,
    )
    return result.model_dump()


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def apply_triple_barrier_labels(
    prices: list[float],
    holding_horizon: int,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> list[dict[str, Any]]:
    """
    Applies the triple-barrier method (López de Prado 2018, ch. 3) to strictly positive prices:
    +1 if the simple return first reaches +take_profit_pct, -1 if it first reaches
    -stop_loss_pct, 0 if neither happens within holding_horizon bars (vertical barrier).
    Barriers are decimals (0.03 = 3%), touches are inclusive, and the entry bar is not tested.
    Entries without holding_horizon bars after them are dropped.
    """
    labels = FinancialMLEngine.apply_triple_barriers(
        prices=prices,
        holding_horizon=holding_horizon,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    return [label.model_dump() for label in labels]


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def compute_deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_obs: int,
    n_trials: float,
    sharpe_ratio_variance: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: float | None = None,
) -> dict[str, Any]:
    """
    Deflated Sharpe Ratio (Bailey and López de Prado 2014): the probability that the selected strategy's
    true Sharpe ratio exceeds the Sharpe ratio expected from the best of n_trials independent unskilled
    trials, given n_obs return observations, the returns' skewness and raw kurtosis (3 for normal, not
    excess). sharpe_ratio and sharpe_ratio_variance (the variance of the Sharpe ratios across all trials)
    are per observation period; or pass periods_per_year (e.g. 252 for daily) to give them annualised.
    Returns the DSR (a probability, 0 to 1), the expected maximum Sharpe ratio SR0 and the tested Sharpe
    ratio (both per period), and the z statistic. Correlated trials count as fewer independent trials.
    """
    return deflated_sharpe_ratio(sharpe_ratio=sharpe_ratio, n_obs=n_obs, n_trials=n_trials,
                                 sharpe_ratio_variance=sharpe_ratio_variance, skewness=skewness,
                                 kurtosis=kurtosis, periods_per_year=periods_per_year).model_dump()


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def estimate_probability_of_backtest_overfitting(
    performance: list[list[float]],
    n_partitions: int = 16,
    metric: str = "sharpe",
) -> dict[str, Any]:
    """
    Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey, Borwein,
    López de Prado and Zhu 2017). performance: one row per period (e.g. daily returns as decimals), one
    column per strategy configuration tried; the number of rows must be a multiple of n_partitions (even).
    Every half of the row blocks is used once as in-sample: the in-sample best configuration's
    out-of-sample rank gives a logit. Returns pbo (share of combinations where the in-sample best is at or
    below the out-of-sample median; 0 to 1), prob_oos_loss (share where it loses out of sample), the
    degradation slope of its out-of-sample on in-sample statistic, and logit quantiles. metric: "sharpe"
    (mean/std per period) or "mean".
    """
    if metric not in ("sharpe", "mean"):
        raise ValueError("metric must be 'sharpe' or 'mean'.")
    r = probability_of_backtest_overfitting(performance, n_partitions=n_partitions,
                                            metric="sharpe" if metric == "sharpe" else "mean")
    q = np.quantile(np.array(r.logits), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    out = r.model_dump(exclude={"logits", "is_statistic", "oos_statistic"})
    out["logit_quantiles"] = dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"],
                                      [float(x) for x in q], strict=True))
    return out


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
