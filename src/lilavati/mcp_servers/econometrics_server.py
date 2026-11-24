"""
FastMCP Server for Financial Econometrics.

Exposes deterministic time-series stationarity testing, cointegration analysis,
and GARCH volatility modeling via Model Context Protocol (FastMCP).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.core.econometrics import EconometricEngine
from lilavati.validation_status import with_validation_status

mcp = FastMCP(
    name="lilavati-econometrics",
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
def analyze_stationarity(
    series: list[float],
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Tests a time series for stationarity using the Augmented Dickey-Fuller (ADF) test.
    Returns test statistics, p-value, and an econometric recommendation.

    Constant-only test regression, AIC lag selection (statsmodels ``adfuller``), MacKinnon
    p-value. ``series`` needs at least 20 finite, non-constant values; ``significance_level``
    is a decimal in (0, 1).
    """
    result = EconometricEngine.test_stationarity(
        series=series,
        significance_level=significance_level,
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
def analyze_cointegration(
    series_a: list[float],
    series_b: list[float],
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Tests whether two price series are cointegrated (for statistical arbitrage / pairs trading)
    and computes the Ornstein-Uhlenbeck mean-reversion half-life.

    Engle-Granger: OLS series_a = alpha + hedge_ratio * series_b + spread, then an ADF test on
    the spread (statsmodels ``coint``). ``half_life`` is in sampling periods (days for daily
    data): -ln 2 / ln|phi| for the spread's fitted AR(1) coefficient phi. Needs at least 30
    equal-length, finite, non-constant, not perfectly collinear values.
    """
    result = EconometricEngine.test_cointegration(
        series_a=series_a,
        series_b=series_b,
        significance_level=significance_level,
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
def fit_garch_volatility(
    returns: list[float],
    p: int = 1,
    q: int = 1,
) -> dict[str, Any]:
    """
    Fits a GARCH(p,q) model to asset returns to quantify conditional volatility
    and volatility clustering persistence.

    ``returns`` are daily decimal returns (0.01 = 1%), at least 100 values. ``p`` is the number
    of GARCH (beta, lagged variance) terms and ``q`` the number of ARCH (alpha, lagged squared
    shock) terms (Bollerslev 1986). ``omega`` is in decimal units; ``annualized_volatility`` is
    the last in-sample conditional volatility times sqrt(252), as a decimal.
    """
    result = EconometricEngine.fit_garch(
        returns=returns,
        p=p,
        q=q,
    )
    return result.model_dump()


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
