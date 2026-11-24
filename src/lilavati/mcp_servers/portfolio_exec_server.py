"""
FastMCP Server for Trading Microstructure, Execution, and Portfolio Optimization.

Exposes LOB microstructure analytics, Almgren-Chriss optimal liquidation scheduling,
Black-Litterman allocation, and Hierarchical Risk Parity (HRP) via FastMCP.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.core.execution import ExecutionEngine

mcp = FastMCP(
    name="lilavati-portfolio-execution",
)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def analyze_order_book_microstructure(
    best_bid: float,
    best_ask: float,
    bid_size: float,
    ask_size: float,
    prev_bid_size: float | None = None,
    prev_ask_size: float | None = None,
) -> dict[str, Any]:
    """
    Spread, mid price, microprice and a simplified order-flow imbalance from a top-of-book quote.

    Prices in one currency per share; sizes in shares (or contracts). spread = ask - bid (price
    units); microprice = (bid_size * ask + ask_size * bid) / (bid_size + ask_size). order_flow_imbalance
    = (bid_size - prev_bid_size) - (ask_size - prev_ask_size), in shares, assuming unchanged prices
    (a simplification of Cont et al. 2014); 0.0 when previous sizes are not given. Raises if the book
    is crossed or a size is not positive.
    """
    result = ExecutionEngine.analyze_lob_snapshot(
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        prev_bid_size=prev_bid_size,
        prev_ask_size=prev_ask_size,
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
def compute_almgren_chriss_liquidation_schedule(
    total_shares: float,
    time_horizon_hours: float,
    n_intervals: int,
    daily_volatility: float,
    arrival_price: float,
    permanent_impact_gamma: float = 2.5e-7,
    temporary_impact_eta: float = 2.5e-6,
    risk_aversion_lambda: float = 1e-6,
    hours_per_day: float = 6.5,
    fixed_cost_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Computes the discrete Almgren-Chriss (2000) optimal liquidation trajectory.

    daily_volatility is a fraction (0.02 = 2%/day) and is scaled by arrival_price ($/share)
    to price units. Impact parameters use the paper's units with the trading day as time unit:
    gamma in $/share^2, eta in ($/share)/(share/day), epsilon in $/share, lambda in 1/$.
    Returns holdings, trade sizes, expected cost E[x] ($) and variance V[x] ($^2).
    """
    result = ExecutionEngine.compute_almgren_chriss_schedule(
        total_shares=total_shares,
        time_horizon_hours=time_horizon_hours,
        n_intervals=n_intervals,
        daily_volatility=daily_volatility,
        arrival_price=arrival_price,
        permanent_impact_gamma=permanent_impact_gamma,
        temporary_impact_eta=temporary_impact_eta,
        risk_aversion_lambda=risk_aversion_lambda,
        hours_per_day=hours_per_day,
        fixed_cost_epsilon=fixed_cost_epsilon,
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
def optimize_portfolio_black_litterman(
    assets: list[str],
    covariance_matrix: list[list[float]],
    market_prior_weights: list[float],
    views_matrix: list[list[float]],
    views_returns: list[float],
    views_confidence_std: list[float],
    risk_aversion: float = 2.5,
) -> dict[str, Any]:
    """
    Black-Litterman posterior returns and long-only weights.

    covariance_matrix: covariance of asset returns (decimals, e.g. annual). market_prior_weights:
    fractions summing to 1. views_matrix P (one row per view, one column per asset), views_returns Q
    (decimals, same period as the covariance), views_confidence_std (standard deviation of each view;
    Omega is their squares on the diagonal). tau = 0.05. Returns implied_prior_returns (Pi = risk_aversion
    * Sigma * w_mkt), expected_returns (posterior, decimals) and optimal_weights (fractions, long-only,
    summing to 1, from maximising return - 0.5 * risk_aversion * variance, solved with CVXPY; an error is
    returned if the solver does not reach an optimum, never a default allocation). Not yet checked against a
    published example (issue #15).
    """
    result = ExecutionEngine.optimize_black_litterman(
        assets=assets,
        covariance_matrix=covariance_matrix,
        market_prior_weights=market_prior_weights,
        views_matrix=views_matrix,
        views_returns=views_returns,
        views_confidence_std=views_confidence_std,
        risk_aversion=risk_aversion,
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
def optimize_portfolio_hierarchical_risk_parity(
    assets: list[str],
    returns_matrix: list[list[float]],
) -> dict[str, Any]:
    """
    Hierarchical Risk Parity weights (after López de Prado 2016).

    returns_matrix: one row per period, one column per asset (periodic returns as decimals). Single-
    linkage clustering on the distance sqrt(0.5 * (1 - correlation)), quasi-diagonal ordering, then
    recursive bisection; each split is weighted by inverse cluster variance, approximated here by the
    trace of the cluster's covariance. Returns weights (fractions summing to 1) and ordered_indices.
    Not yet checked against the paper (issue #16).
    """
    result = ExecutionEngine.optimize_hierarchical_risk_parity(
        assets=assets,
        returns_matrix=returns_matrix,
    )
    return result.model_dump()


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
