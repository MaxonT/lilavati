"""
FastMCP Server for Fixed Income, Yield Curves, and Credit Risk.

Exposes deterministic bond pricing, Nelson-Siegel yield curve calibration,
and the Merton structural credit risk model via Model Context Protocol (FastMCP).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.core.covenant_paths import check_covenant_path as _path_check
from lilavati.core.covenant_paths import minimum_upfront_repayment_for_path as _path_repay
from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.invariants import check_covenants_over_ranges as _ranges
from lilavati.core.invariants import minimum_debt_repayment
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


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def find_minimum_debt_repayment(
    total_debt: float,
    cash: float,
    ebitda: float,
    interest_expense: float,
    max_net_leverage: float,
    min_interest_coverage: float,
) -> dict[str, Any]:
    """
    Smallest debt repayment meeting both maintenance covenants: net leverage (debt - cash) / EBITDA
    <= max_net_leverage and interest coverage EBITDA / interest >= min_interest_coverage (ratios in
    times; a ratio at its limit complies). Amounts in one currency unit (e.g. USD millions); EBITDA and
    interest for the same period. Assumes the repayment comes from outside the cash balance (new
    equity, asset sales), so cash is unchanged, and that interest falls in proportion to debt. Solved
    exactly with Z3's optimiser. Returns required_repayment (currency units; 0 if compliant), the
    covenants binding after it, and both ratios after it. How a sponsor equity cure counts is set by
    each credit agreement; this answers the repayment question only.
    """
    try:
        return minimum_debt_repayment(total_debt, cash, ebitda, interest_expense, max_net_leverage,
                                      min_interest_coverage).model_dump()
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def check_covenants_over_ranges(
    total_debt_low: float,
    total_debt_high: float,
    cash_low: float,
    cash_high: float,
    ebitda_low: float,
    ebitda_high: float,
    interest_expense_low: float,
    interest_expense_high: float,
    max_net_leverage: float,
    min_interest_coverage: float,
) -> dict[str, Any]:
    """
    Do both maintenance covenants hold for every combination of inputs within the given ranges (e.g. a
    management guidance range for EBITDA)? Net leverage (debt - cash) / EBITDA <= max_net_leverage and
    EBITDA / interest >= min_interest_coverage (times; at the limit complies). Amounts in one currency
    unit; EBITDA must be positive over its range. Z3 searches the whole box for a breach: holds_for_all
    is a statement about every value in the ranges, not a point check; otherwise a counterexample (a
    breaching combination) and the covenants it violates are returned.
    """
    try:
        return _ranges((total_debt_low, total_debt_high), (cash_low, cash_high), (ebitda_low, ebitda_high),
                       (interest_expense_low, interest_expense_high), max_net_leverage,
                       min_interest_coverage).model_dump()
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def check_covenant_path(
    opening_debt: float,
    opening_cash: float,
    ebitda_ranges: list[list[float]],
    interest_rate_low: float,
    interest_rate_high: float,
    max_net_leverage: list[float],
    min_interest_coverage: float,
    min_liquidity: float = 0.0,
    other_outflows: list[float] | None = None,
    amortisation: list[float] | None = None,
    sweep_fraction: float = 0.0,
    sweep_grid: list[list[float]] | None = None,
) -> dict[str, Any]:
    """
    Do all covenants hold in every period for every EBITDA path within the per-period ranges and every
    rate within [interest_rate_low, interest_rate_high]? Z3 searches all of them: status "holds" is a
    statement about every path in the ranges (not only the corners, which a leverage-based sweep can
    make misleading); "breach" returns a counterexample path, replayed in plain arithmetic; "unknown"
    means Z3 did not decide in time (possible with a rate range).
    Model: annual periods; interest = rate x opening debt of the period (linked to the debt path);
    scheduled repayment = min(amortisation, debt); cash before sweep = cash + EBITDA - other outflows
    (capex, tax, working capital) - interest - scheduled repayment; a cash sweep repays debt with
    sweep_fraction of cash above min_liquidity, or with a leverage grid sweep_grid = [[threshold,
    fraction], ...] (the first threshold net leverage exceeds sets the fraction). Covenants each period:
    net leverage (debt - cash) / EBITDA <= max_net_leverage[t] (one value per period, step-downs allowed),
    EBITDA / interest >= min_interest_coverage, cash >= min_liquidity; at the limit complies. Amounts in
    one currency unit (e.g. USD millions); rates as decimals; ebitda_ranges = [[low, high], ...] per
    period (low = high for a point forecast); other_outflows and amortisation one value per period.
    """
    try:
        return _path_check(opening_debt, opening_cash, [tuple(r) for r in ebitda_ranges],  # type: ignore[misc]
                           (interest_rate_low, interest_rate_high), max_net_leverage, min_interest_coverage,
                           min_liquidity, other_outflows if other_outflows is not None else 0.0,
                           amortisation if amortisation is not None else 0.0,
                           [tuple(g) for g in sweep_grid] if sweep_grid else sweep_fraction).model_dump()  # type: ignore[misc]
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def find_minimum_upfront_repayment_for_path(
    opening_debt: float,
    opening_cash: float,
    ebitda_ranges: list[list[float]],
    interest_rate_low: float,
    interest_rate_high: float,
    max_net_leverage: list[float],
    min_interest_coverage: float,
    min_liquidity: float = 0.0,
    other_outflows: list[float] | None = None,
    amortisation: list[float] | None = None,
    sweep_fraction: float = 0.0,
    sweep_grid: list[list[float]] | None = None,
) -> dict[str, Any]:
    """
    Smallest debt repayment now (from outside the cash balance: new equity or asset sales) that keeps
    every covenant met in every period for every EBITDA path and rate within the ranges. status "found":
    exact; "found_within_tolerance": the smallest lies in (lower_bound, required_repayment] (relative
    width 1e-9 of the debt); "infeasible": even repaying all debt fails (e.g. liquidity); "unknown": not
    decided in time. required_repayment is rounded up so it still complies; required_repayment_exact is
    the fraction.
    Model: annual periods; interest = rate x opening debt of the period (linked to the debt path);
    scheduled repayment = min(amortisation, debt); cash before sweep = cash + EBITDA - other outflows
    (capex, tax, working capital) - interest - scheduled repayment; a cash sweep repays debt with
    sweep_fraction of cash above min_liquidity, or with a leverage grid sweep_grid = [[threshold,
    fraction], ...] (the first threshold net leverage exceeds sets the fraction). Covenants each period:
    net leverage (debt - cash) / EBITDA <= max_net_leverage[t] (one value per period, step-downs allowed),
    EBITDA / interest >= min_interest_coverage, cash >= min_liquidity; at the limit complies. Amounts in
    one currency unit (e.g. USD millions); rates as decimals; ebitda_ranges = [[low, high], ...] per
    period (low = high for a point forecast); other_outflows and amortisation one value per period.
    """
    try:
        return _path_repay(opening_debt, opening_cash, [tuple(r) for r in ebitda_ranges],  # type: ignore[misc]
                           (interest_rate_low, interest_rate_high), max_net_leverage, min_interest_coverage,
                           min_liquidity, other_outflows if other_outflows is not None else 0.0,
                           amortisation if amortisation is not None else 0.0,
                           [tuple(g) for g in sweep_grid] if sweep_grid else sweep_fraction).model_dump()  # type: ignore[misc]
    except ValueError as e:
        return {"error": str(e)}


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
