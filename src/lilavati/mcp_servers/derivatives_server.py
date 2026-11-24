"""
FastMCP Server for Derivatives, Options Pricing, Greeks, and Volatility Surfaces.

Exposes Black-Scholes-Merton, American CRR binomial trees, 1st & 2nd order Greeks,
SABR stochastic volatility calibration, and SMT butterfly arbitrage verification.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.core.derivatives import DerivativesEngine

mcp = FastMCP(
    name="lilavati-derivatives",
)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def price_european_option(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    is_call: bool = True,
) -> dict[str, Any]:
    """Prices a European option via Black-Scholes-Merton and calculates all Greeks.

    Units: time_to_maturity in years; risk_free_rate, dividend_yield (continuous) and volatility as
    decimals (0.05 = 5%). Greeks: vega per 1 vol point, theta per calendar day, rho per 1% rate move,
    vanna = d(delta)/d(sigma), volga = d(vega)/d(sigma).
    """
    result = DerivativesEngine.price_european_option(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        is_call=is_call,
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
def price_american_option(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    is_call: bool = True,
    steps: int = 150,
) -> dict[str, Any]:
    """Prices an American option with early exercise via Cox-Ross-Rubinstein (CRR) binomial tree.

    Units: time_to_maturity in years; risk_free_rate, dividend_yield (continuous) and volatility as
    decimals (0.05 = 5%); price in the currency of spot and strike. The returned greeks are European
    Black-Scholes-Merton Greeks for the same inputs, not tree Greeks of the American option (vega per
    1 vol point, theta per calendar day, rho per 1% rate move).
    """
    result = DerivativesEngine.price_american_option_crr(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        is_call=is_call,
        steps=steps,
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
def calculate_option_greeks(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    is_call: bool = True,
) -> dict[str, Any]:
    """Calculates analytical 1st & 2nd order Greeks (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga).

    Units: time in years, rates and volatility as decimals. vega per 1 vol point (0.01), theta per
    calendar day, rho per 1 percentage point, vanna = d(delta)/d(sigma), volga = d(vega)/d(sigma).
    """
    result = DerivativesEngine.calculate_bsm_greeks(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        is_call=is_call,
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
def calibrate_sabr_volatility_model(
    forward: float,
    strikes: list[float],
    market_vols: list[float],
    time_to_maturity: float,
    beta: float = 0.5,
) -> dict[str, Any]:
    """Calibrates the SABR stochastic volatility model (alpha, rho, nu) to market option smiles.

    market_vols are decimal Black (lognormal) vols at strikes for one expiry; beta is held fixed.
    is_arbitrage_free checks butterfly/call-spread arbitrage of the fitted smile's prices.
    """
    result = DerivativesEngine.calibrate_sabr(
        forward=forward,
        strikes=strikes,
        market_vols=market_vols,
        time_to_maturity=time_to_maturity,
        beta=beta,
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
def check_butterfly_arbitrage(
    strikes: list[float],
    call_prices: list[float],
) -> dict[str, Any]:
    """
    Butterfly (strike-convexity) no-arbitrage check of call prices with the Z3 solver.

    strikes and call_prices are in the same currency, for European calls on one underlying and one
    expiry. Points are sorted by strike; for each consecutive triple K1 < K2 < K3 the slope difference
    (C3 - C2)/(K3 - K2) - (C2 - C1)/(K2 - K1) must be >= -1e-4 (dimensionless). Call-spread bounds
    are not checked. Strikes must be distinct and all values finite.
    """
    result = DerivativesEngine.verify_butterfly_arbitrage(
        strikes=strikes,
        call_prices=call_prices,
    )
    return result.model_dump()


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
