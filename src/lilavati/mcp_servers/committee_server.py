"""
FastMCP Server for the Adversarial Investment Committee & Forensic Accounting.

Exposes multi-agent deliberation, Beneish M-Score manipulation detection,
and Altman Z-Score distress analysis over FastMCP.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.agents.committee import AdversarialCommittee
from lilavati.agents.forensics import ForensicEngine
from lilavati.validation_status import with_validation_status

mcp = FastMCP(
    name="lilavati-adversarial-committee",
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
def evaluate_investment_committee_memo(
    symbol: str,
    fiscal_year: int,
    bull_revenue_growth_target: float,
    revenue_t: float,
    revenue_t_minus_1: float,
    cogs_t: float,
    cogs_t_minus_1: float,
    receivables_t: float,
    receivables_t_minus_1: float,
    current_assets_t: float,
    current_assets_t_minus_1: float,
    ppe_t: float,
    ppe_t_minus_1: float,
    total_assets_t: float,
    total_assets_t_minus_1: float,
    depreciation_t: float,
    depreciation_t_minus_1: float,
    sga_t: float,
    sga_t_minus_1: float,
    long_term_debt_t: float,
    long_term_debt_t_minus_1: float,
    current_liabilities_t: float,
    current_liabilities_t_minus_1: float,
    net_income_t: float,
    cfo_t: float,
    cfi_t: float,
    cff_t: float,
    market_value_equity: float,
    total_liabilities_t: float,
    shareholders_equity_t: float,
    retained_earnings_t: float,
    operating_income_t: float,
    cash_start_t: float,
    cash_end_t: float,
    fx_effect_on_cash_t: float = 0.0,
) -> dict[str, Any]:
    """
    Rule-based investment committee memo (no language model): revenue growth against a target,
    Beneish M-score, Altman Z-score, and identity checks on the inputs, then a recommendation by
    fixed rules (rejected if an identity fails, Beneish flags manipulation or Altman is in distress).

    All amounts in one currency unit (e.g. USD millions); suffix _t is the current year, _t_minus_1
    the prior year. Pass reported figures: total liabilities and shareholders' equity from the balance
    sheet (checked: assets = liabilities + equity), retained earnings and operating income (Altman X2,
    X3), and cash at the start and end of the year as in the cash flow statement (checked: start + CFO
    + CFI + CFF + exchange-rate effect = end). "checks" lists each identity with its difference.
    """
    memo = AdversarialCommittee.evaluate_company(
        symbol=symbol,
        fiscal_year=fiscal_year,
        bull_revenue_growth_target=bull_revenue_growth_target,
        revenue_t=revenue_t,
        revenue_t_minus_1=revenue_t_minus_1,
        cogs_t=cogs_t,
        cogs_t_minus_1=cogs_t_minus_1,
        receivables_t=receivables_t,
        receivables_t_minus_1=receivables_t_minus_1,
        current_assets_t=current_assets_t,
        current_assets_t_minus_1=current_assets_t_minus_1,
        ppe_t=ppe_t,
        ppe_t_minus_1=ppe_t_minus_1,
        total_assets_t=total_assets_t,
        total_assets_t_minus_1=total_assets_t_minus_1,
        depreciation_t=depreciation_t,
        depreciation_t_minus_1=depreciation_t_minus_1,
        sga_t=sga_t,
        sga_t_minus_1=sga_t_minus_1,
        long_term_debt_t=long_term_debt_t,
        long_term_debt_t_minus_1=long_term_debt_t_minus_1,
        current_liabilities_t=current_liabilities_t,
        current_liabilities_t_minus_1=current_liabilities_t_minus_1,
        net_income_t=net_income_t,
        cfo_t=cfo_t,
        cfi_t=cfi_t,
        cff_t=cff_t,
        market_value_equity=market_value_equity,
        total_liabilities_t=total_liabilities_t,
        shareholders_equity_t=shareholders_equity_t,
        retained_earnings_t=retained_earnings_t,
        operating_income_t=operating_income_t,
        cash_start_t=cash_start_t,
        cash_end_t=cash_end_t,
        fx_effect_on_cash_t=fx_effect_on_cash_t,
    )
    return memo.model_dump()


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def compute_forensic_beneish_m_score(
    revenue_t: float,
    revenue_t_minus_1: float,
    cogs_t: float,
    cogs_t_minus_1: float,
    receivables_t: float,
    receivables_t_minus_1: float,
    current_assets_t: float,
    current_assets_t_minus_1: float,
    ppe_t: float,
    ppe_t_minus_1: float,
    total_assets_t: float,
    total_assets_t_minus_1: float,
    depreciation_t: float,
    depreciation_t_minus_1: float,
    sga_t: float,
    sga_t_minus_1: float,
    long_term_debt_t: float,
    long_term_debt_t_minus_1: float,
    current_liabilities_t: float,
    current_liabilities_t_minus_1: float,
    net_income_t: float,
    cfo_t: float,
) -> dict[str, Any]:
    """
    Beneish (1999) 8-variable M-score for earnings manipulation.

    Amounts in one currency unit for every input (e.g. USD millions); suffix _t is the current year,
    _t_minus_1 the prior year. depreciation is depreciation and amortisation expense; long_term_debt is
    non-current debt. Returns m_score (dimensionless; above -1.78 sets is_likely_manipulator) and the
    eight indices dsri, gmi, aqi, sgi, depi, sgai, lvgi (ratios near 1.0) and tata ((net income - CFO)
    / total assets, a simplification of Beneish's accruals). Fast-growing companies can exceed -1.78
    through the sales growth index alone; read the indices behind a flag.
    """
    res = ForensicEngine.calculate_beneish_m_score(
        revenue_t=revenue_t,
        revenue_t_minus_1=revenue_t_minus_1,
        cogs_t=cogs_t,
        cogs_t_minus_1=cogs_t_minus_1,
        receivables_t=receivables_t,
        receivables_t_minus_1=receivables_t_minus_1,
        current_assets_t=current_assets_t,
        current_assets_t_minus_1=current_assets_t_minus_1,
        ppe_t=ppe_t,
        ppe_t_minus_1=ppe_t_minus_1,
        total_assets_t=total_assets_t,
        total_assets_t_minus_1=total_assets_t_minus_1,
        depreciation_t=depreciation_t,
        depreciation_t_minus_1=depreciation_t_minus_1,
        sga_t=sga_t,
        sga_t_minus_1=sga_t_minus_1,
        long_term_debt_t=long_term_debt_t,
        long_term_debt_t_minus_1=long_term_debt_t_minus_1,
        current_liabilities_t=current_liabilities_t,
        current_liabilities_t_minus_1=current_liabilities_t_minus_1,
        net_income_t=net_income_t,
        cfo_t=cfo_t,
    )
    return res.model_dump()


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
