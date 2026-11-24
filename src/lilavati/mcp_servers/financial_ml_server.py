"""
FastMCP Server for Financial Machine Learning.

Exposes Marcos López de Prado's institutional financial ML primitives:
1. Fractional Differentiation (finding optimal d* preserving memory)
2. Triple-Barrier Method (path-dependent labeling)
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

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


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
