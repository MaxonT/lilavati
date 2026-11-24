"""
FastMCP Server for Sustainable Finance, Climate Risk, and Carbon Accounting.

Exposes deterministic carbon intensity analytics (WACI), climate transition risk stress testing,
and Sustainability-Linked Bond (SLB) covenant evaluation via Model Context Protocol (FastMCP).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.core.sustainability import SustainabilityEngine
from lilavati.schemas.sustainability import SustainabilityLinkedBond

mcp = FastMCP(
    name="lilavati-sustainability",
)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def calculate_portfolio_carbon_intensity(
    weights: list[float],
    emissions: list[float],
    revenues_millions: list[float],
    paris_aligned_threshold: float = 100.0,
) -> dict[str, Any]:
    """
    Weighted Average Carbon Intensity (WACI, TCFD 2017 and 2021 annex).

    weights: non-negative portfolio weights (normalised to sum to 1). emissions: Scope 1 + 2 per company
    in tCO2e. revenues_millions: revenue per company in millions of one currency (e.g. USD millions).
    Returns waci in tCO2e per million of revenue, total_carbon_footprint (weighted tCO2e) and
    carbon_intensity_clean (waci <= paris_aligned_threshold, in the same tCO2e per million units).
    """
    result = SustainabilityEngine.calculate_waci(
        portfolio_weights=weights,
        emissions=emissions,
        revenues_millions=revenues_millions,
        paris_aligned_threshold=paris_aligned_threshold,
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
def stress_test_carbon_tax_impact(
    revenue: float,
    ebit: float,
    interest_expense: float,
    scope1_emissions: float,
    scope2_emissions: float,
    carbon_price_per_ton: float,
    pass_through_rate: float = 0.50,
    covenant_interest_floor: float = 2.50,
) -> dict[str, Any]:
    """
    EBIT, EBIT margin and interest coverage under a carbon price.

    Net cost = (scope1 + scope2 emissions, tCO2e) * carbon_price_per_ton (currency per tCO2e) * (1 -
    pass_through_rate), deducted from EBIT. Emissions times price must be in the same money unit as
    revenue, ebit and interest_expense: with emissions in tCO2e and a price per tonne, pass financials
    in currency units, not millions. pass_through_rate is a fraction in [0, 1]. Returned _pct fields are
    fractions (0.25 = 25%); interest_coverage_after is EBIT after the cost / interest expense (times);
    is_covenant_at_risk compares it with covenant_interest_floor (times). Pass scope2 = 0 when the tax
    covers Scope 1 only.
    """
    result = SustainabilityEngine.stress_test_carbon_tax(
        revenue=revenue,
        ebit=ebit,
        interest_expense=interest_expense,
        scope1_emissions=scope1_emissions,
        scope2_emissions=scope2_emissions,
        carbon_price_per_ton=carbon_price_per_ton,
        pass_through_rate=pass_through_rate,
        covenant_interest_floor=covenant_interest_floor,
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
def evaluate_sustainability_linked_bond_covenant(
    isin: str,
    notional: float,
    base_coupon_rate: float,
    target_reduction_pct: float,
    baseline_emissions: float,
    step_up_bps: float,
    target_year: int,
    reported_emissions: float,
) -> dict[str, Any]:
    """
    Sustainability-linked bond: is the emissions target met, or does the coupon step up?

    target_reduction_pct is a fraction (0.30 = 30% below baseline); baseline_emissions and
    reported_emissions in the same unit (e.g. tCO2e); base_coupon_rate a decimal (0.04 = 4%);
    step_up_bps in basis points (25 = 0.25%); notional in currency units. The target is met when
    reported <= baseline * (1 - target). Returns spt_achieved, realized_reduction_pct (fraction),
    active_coupon_rate (decimal) and annual_interest_penalty (notional * step-up, currency units per
    year; 0 when the target is met). ICMA Sustainability-Linked Bond Principles (2023).
    """
    bond = SustainabilityLinkedBond(
        isin=isin,
        notional=notional,
        base_coupon_rate=base_coupon_rate,
        target_reduction_pct=target_reduction_pct,
        baseline_emissions=baseline_emissions,
        step_up_bps=step_up_bps,
        target_year=target_year,
    )
    result = SustainabilityEngine.evaluate_slb_covenant(
        bond=bond,
        reported_emissions=reported_emissions,
    )
    return result.model_dump()


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
