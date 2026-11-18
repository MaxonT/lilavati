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
    """Computes microprice, spread, and Order Flow Imbalance (OFI) from order book quotes."""
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
    """Solves Black-Litterman optimal asset allocation combining market priors with agent views."""
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
    """Solves Hierarchical Risk Parity (HRP) allocation using clustering without matrix inversion."""
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
