"""
A rule-based investment committee over deterministic tools (no language model is called).

The roles are labels for fixed rules: the Bull compares revenue growth with a target, the Bear runs
the Beneish M-score and Altman Z-score, and the Risk officer checks the balance sheet identity (Z3)
and the cash reconciliation on the reported inputs. See ``AdversarialCommittee.evaluate_company``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from lilavati.agents.forensics import (
    AltmanZScoreResult,
    BeneishMScoreResult,
    ForensicEngine,
)
from lilavati.connectors.sec_filing import IdentityCheck
from lilavati.core.invariants import SuvarnaEngine


class CommitteeRecommendation(StrEnum):
    """Final investment committee consensus decision."""

    APPROVED_BUY = "APPROVED_BUY"
    APPROVED_HOLD = "APPROVED_HOLD"
    REJECTED_ACCOUNTING_ANOMALY = "REJECTED_ACCOUNTING_ANOMALY"
    REJECTED_COVENANT_BREACH = "REJECTED_COVENANT_BREACH"
    REJECTED_ARBITRAGE_VIOLATION = "REJECTED_ARBITRAGE_VIOLATION"


class InvestmentCommitteeMemo(BaseModel):
    """Investment Committee Memorandum with its tool outputs and identity checks attached."""

    model_config = ConfigDict(frozen=True)

    company_symbol: str
    fiscal_year: int
    recommendation: CommitteeRecommendation
    bull_thesis: str
    bear_concerns: list[str]
    beneish_m_score: BeneishMScoreResult
    altman_z_score: AltmanZScoreResult
    formal_invariants_satisfied: bool = Field(
        ..., description="every check in `checks` holds (balance sheet identity by Z3, cash reconciliation)")
    checks: list[IdentityCheck] = Field(default_factory=list, description="identities on the given inputs")
    summary_verdict: str


class AdversarialCommittee:
    """Deliberative multi-role investment committee."""

    @staticmethod
    def evaluate_company(
        symbol: str,
        fiscal_year: int,
        bull_revenue_growth_target: float,
        # Multi-year financial facts for Beneish M-Score
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
        *,
        total_liabilities_t: float,
        shareholders_equity_t: float,
        retained_earnings_t: float,
        operating_income_t: float,
        cash_start_t: float,
        cash_end_t: float,
        fx_effect_on_cash_t: float = 0.0,
    ) -> InvestmentCommitteeMemo:
        """
        A rule-based committee over deterministic tools; no language model is called.

        1. "Bull": revenue growth against the target.
        2. "Bear": Beneish M-score (Beneish 1999) and Altman Z-score (Altman 1968; X2 uses retained
           earnings, X3 operating income as EBIT, X4 market equity over total liabilities).
        3. "Risk officer": identities on the reported inputs: total assets = total liabilities +
           shareholders' equity (Z3), and cash at the start + CFO + CFI + CFF + exchange-rate effect =
           cash at the end (cash as in the cash flow statement).
        4. Verdict by fixed rules: rejected if an identity fails, if Beneish flags manipulation or if
           Altman is in distress; approved for allocation only when growth meets the target and Altman
           is safe; otherwise hold.

        The balance sheet and cash inputs are required: the committee used to derive equity as
        assets minus (long-term debt + current liabilities) and start cash from the flows, so its
        identity checks could not fail; Altman used net income for retained earnings.
        """
        # 1. Bull Analyst Perspective
        growth_rate = (revenue_t - revenue_t_minus_1) / revenue_t_minus_1
        bull_thesis = (
            f"Bull Case: Company achieved {growth_rate:.1%} YoY revenue growth (target: {bull_revenue_growth_target:.1%}) "
            f"with Net Income of ${net_income_t:,.0f}."
        )

        # 2. Forensic Bear Auditor Perspective
        m_res = ForensicEngine.calculate_beneish_m_score(
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

        z_res = ForensicEngine.calculate_altman_z_score(
            working_capital=current_assets_t - current_liabilities_t,
            total_assets=total_assets_t,
            retained_earnings=retained_earnings_t,
            ebit=operating_income_t,
            market_value_equity=market_value_equity,
            total_liabilities=total_liabilities_t,
            sales=revenue_t,
        )

        bear_concerns: list[str] = []
        if m_res.is_likely_manipulator:
            bear_concerns.append(
                f"Beneish M-Score ({m_res.m_score:.2f} > -1.78) flags high risk of aggressive revenue recognition."
            )
        if m_res.dsri > 1.3:
            bear_concerns.append(
                f"Receivables grew significantly faster than revenues (DSRI = {m_res.dsri:.2f}), suggesting channel stuffing."
            )
        if z_res.distress_zone == "Distress":
            bear_concerns.append(
                f"Altman Z-Score ({z_res.z_score:.2f} < 1.81) indicates near-term credit insolvency risk."
            )

        # 3. Risk officer: identities on the reported inputs (none derived from what it checks)
        bs = SuvarnaEngine.verify_balance_sheet_equation(
            total_assets=total_assets_t, total_liabilities=total_liabilities_t,
            shareholders_equity=shareholders_equity_t, tolerance=0.5)
        cash_diff = cash_end_t - (cash_start_t + cfo_t + cfi_t + cff_t + fx_effect_on_cash_t)
        checks = [
            IdentityCheck(name="assets = liabilities + equity", holds=bs.is_valid,
                          difference=total_assets_t - total_liabilities_t - shareholders_equity_t,
                          detail="Z3 on the given numbers"),
            IdentityCheck(name="opening cash + cash flows = closing cash", holds=abs(cash_diff) <= 0.5,
                          difference=cash_diff, detail="CFO + CFI + CFF + exchange-rate effect"),
        ]
        identities_hold = all(c.holds for c in checks)

        # 4. Verdict by fixed rules
        if not identities_hold:
            rec = CommitteeRecommendation.REJECTED_ACCOUNTING_ANOMALY
            failed = ", ".join(c.name for c in checks if not c.holds)
            verdict = f"REJECTED: the inputs fail {failed}."
        elif m_res.is_likely_manipulator:
            rec = CommitteeRecommendation.REJECTED_ACCOUNTING_ANOMALY
            verdict = "REJECTED: Forensic scrutiny revealed severe earnings manipulation probability (Beneish M-Score breach)."
        elif z_res.distress_zone == "Distress":
            rec = CommitteeRecommendation.REJECTED_COVENANT_BREACH
            verdict = "REJECTED: Credit profile in distressed territory (Altman Z-Score < 1.81)."
        elif growth_rate >= bull_revenue_growth_target and z_res.distress_zone == "Safe":
            rec = CommitteeRecommendation.APPROVED_BUY
            verdict = ("APPROVED FOR ALLOCATION (rule-based): growth meets the target, Beneish and Altman "
                       "raise no flag, and the balance sheet and cash identities hold on the inputs.")
        else:
            rec = CommitteeRecommendation.APPROVED_HOLD
            verdict = "HOLD: Moderate credit health with acceptable reporting quality."

        return InvestmentCommitteeMemo(
            company_symbol=symbol.upper(),
            fiscal_year=fiscal_year,
            recommendation=rec,
            bull_thesis=bull_thesis,
            bear_concerns=bear_concerns,
            beneish_m_score=m_res,
            altman_z_score=z_res,
            formal_invariants_satisfied=identities_hold,
            checks=checks,
            summary_verdict=verdict,
        )
