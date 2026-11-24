"""
FastMCP Server for Fixed Income, Yield Curves, and Credit Risk.

Exposes deterministic bond pricing, Nelson-Siegel yield curve calibration,
and the Merton structural credit risk model via Model Context Protocol (FastMCP).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.validation_status import with_validation_status

mcp = FastMCP(
    name="lilavati-fixed-income",
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
def analyze_bond(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 2,
) -> dict[str, Any]:
    """
    Computes clean price, Macaulay duration, modified duration, convexity, and DV01 for a bond.

    Rates are annual decimals (0.05 = 5%); ytm is compounded `frequency` times per year.
    Valuation is on a coupon date (no accrued interest; maturity rounded to whole periods).
    Price and DV01 are in face_value units; durations in years; convexity annualised (years^2).
    """
    result = FixedIncomeEngine.analyze_bond(
        face_value=face_value,
        coupon_rate=coupon_rate,
        ytm=ytm,
        years_to_maturity=years_to_maturity,
        frequency=frequency,
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
def fit_yield_curve(
    maturities: list[float],
    yields: list[float],
) -> dict[str, Any]:
    """
    Fits the Nelson-Siegel 4-parameter model to observed benchmark yields and maturities.

    Maturities in years (ascending); yields are zero-coupon yields as decimals (0.045 = 4.5%).
    lambda_decay is the time constant tau in years.
    """
    result = FixedIncomeEngine.fit_nelson_siegel(
        maturities=maturities,
        yields=yields,
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
def evaluate_credit_risk_merton(
    equity_value: float,
    equity_volatility: float,
    debt_face_value: float,
    risk_free_rate: float,
    maturity_years: float,
) -> dict[str, Any]:
    """
    Solves the Merton structural credit risk model for implied asset value,
    asset volatility, distance-to-default, default probability, and theoretical credit spread.

    Volatility is annualised decimal; risk_free_rate is continuously compounded; maturity in
    years. distance_to_default is risk-neutral d2 and default_probability = N(-d2). Raises
    ValueError for non-positive inputs or if the solver does not converge.
    """
    result = FixedIncomeEngine.solve_merton_model(
        equity_value=equity_value,
        equity_volatility=equity_volatility,
        debt_face_value=debt_face_value,
        risk_free_rate=risk_free_rate,
        maturity_years=maturity_years,
    )
    return result.model_dump()


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
