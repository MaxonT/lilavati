from lilavati.agents.committee import (
    AdversarialCommittee,
    CommitteeRecommendation,
)


def test_committee_healthy_firm_approved() -> None:
    # A financially healthy growth firm (e.g. similar to Apple or Microsoft)
    memo = AdversarialCommittee.evaluate_company(
        symbol="HEALTHY_CORP",
        fiscal_year=2024,
        bull_revenue_growth_target=0.10,
        revenue_t=1200.0,
        revenue_t_minus_1=1000.0,  # 20% growth > 10% target
        cogs_t=600.0,
        cogs_t_minus_1=500.0,
        receivables_t=180.0,
        receivables_t_minus_1=150.0,
        current_assets_t=500.0,
        current_assets_t_minus_1=450.0,
        ppe_t=800.0,
        ppe_t_minus_1=750.0,
        total_assets_t=1600.0,
        total_assets_t_minus_1=1450.0,
        depreciation_t=60.0,
        depreciation_t_minus_1=55.0,
        sga_t=180.0,
        sga_t_minus_1=150.0,
        long_term_debt_t=200.0,
        long_term_debt_t_minus_1=200.0,
        current_liabilities_t=150.0,
        current_liabilities_t_minus_1=140.0,
        net_income_t=250.0,
        cfo_t=280.0,  # Healthy CFO > Net Income
        cfi_t=-100.0,
        cff_t=-80.0,
        market_value_equity=3500.0,  # High market cap
        total_liabilities_t=600.0,  # 1,600 = 600 + 1,000
        shareholders_equity_t=1000.0,
        retained_earnings_t=700.0,
        operating_income_t=360.0,  # 1,200 - 600 - 180 - 60
        cash_start_t=120.0,
        cash_end_t=220.0,  # 120 + 280 - 100 - 80
    )

    assert memo.recommendation == CommitteeRecommendation.APPROVED_BUY
    assert memo.formal_invariants_satisfied is True
    assert memo.beneish_m_score.is_likely_manipulator is False
    assert memo.altman_z_score.distress_zone == "Safe"
    assert len(memo.bear_concerns) == 0


def test_committee_earnings_manipulator_rejected() -> None:
    # A firm with aggressive channel stuffing and ballooning accruals (Enron style)
    # Revenue jumped from 1000 to 1800, but receivables exploded from 100 to 600!
    # Net income reported as 300, but CFO is actually negative (-50)!
    memo = AdversarialCommittee.evaluate_company(
        symbol="SHADY_INC",
        fiscal_year=2024,
        bull_revenue_growth_target=0.15,
        revenue_t=1800.0,
        revenue_t_minus_1=1000.0,
        cogs_t=1100.0,
        cogs_t_minus_1=600.0,
        receivables_t=600.0,  # Huge receivables spike
        receivables_t_minus_1=100.0,
        current_assets_t=800.0,
        current_assets_t_minus_1=300.0,
        ppe_t=500.0,
        ppe_t_minus_1=500.0,
        total_assets_t=1800.0,
        total_assets_t_minus_1=1000.0,
        depreciation_t=20.0,
        depreciation_t_minus_1=40.0,
        sga_t=200.0,
        sga_t_minus_1=150.0,
        long_term_debt_t=600.0,
        long_term_debt_t_minus_1=200.0,
        current_liabilities_t=300.0,
        current_liabilities_t_minus_1=100.0,
        net_income_t=300.0,
        cfo_t=-50.0,  # Severe cash-to-earnings divergence!
        cfi_t=-50.0,
        cff_t=100.0,
        market_value_equity=1200.0,
        total_liabilities_t=1100.0,  # 1,800 = 1,100 + 700
        shareholders_equity_t=700.0,
        retained_earnings_t=400.0,
        operating_income_t=480.0,  # 1,800 - 1,100 - 200 - 20
        cash_start_t=60.0,
        cash_end_t=60.0,  # 60 - 50 - 50 + 100
    )

    # Must be rejected due to forensic accounting anomalies!
    assert memo.recommendation == CommitteeRecommendation.REJECTED_ACCOUNTING_ANOMALY
    assert memo.beneish_m_score.is_likely_manipulator is True
    assert any("Beneish M-Score" in c for c in memo.bear_concerns)
    assert any("channel stuffing" in c for c in memo.bear_concerns)


def _healthy(**override: float) -> dict:
    base = {
        "symbol": "HEALTHY_CORP", "fiscal_year": 2024, "bull_revenue_growth_target": 0.10,
        "revenue_t": 1200.0, "revenue_t_minus_1": 1000.0, "cogs_t": 600.0, "cogs_t_minus_1": 500.0,
        "receivables_t": 180.0, "receivables_t_minus_1": 150.0, "current_assets_t": 500.0, "current_assets_t_minus_1": 450.0,
        "ppe_t": 800.0, "ppe_t_minus_1": 750.0, "total_assets_t": 1600.0, "total_assets_t_minus_1": 1450.0,
        "depreciation_t": 60.0, "depreciation_t_minus_1": 55.0, "sga_t": 180.0, "sga_t_minus_1": 150.0,
        "long_term_debt_t": 200.0, "long_term_debt_t_minus_1": 200.0, "current_liabilities_t": 150.0,
        "current_liabilities_t_minus_1": 140.0, "net_income_t": 250.0, "cfo_t": 280.0, "cfi_t": -100.0, "cff_t": -80.0,
        "market_value_equity": 3500.0, "total_liabilities_t": 600.0, "shareholders_equity_t": 1000.0,
        "retained_earnings_t": 700.0, "operating_income_t": 360.0, "cash_start_t": 120.0, "cash_end_t": 220.0,
    }
    return {**base, **override}


def test_identity_failures_reject_the_memo() -> None:
    # The committee used to derive equity as assets minus liabilities and start cash from the flows,
    # so these checks always held and the accounting-anomaly verdict was unreachable.
    off_bs = AdversarialCommittee.evaluate_company(**_healthy(shareholders_equity_t=950.0))
    assert off_bs.recommendation == CommitteeRecommendation.REJECTED_ACCOUNTING_ANOMALY
    assert off_bs.formal_invariants_satisfied is False
    assert off_bs.checks[0].difference == 50.0 and "assets = liabilities + equity" in off_bs.summary_verdict

    off_cash = AdversarialCommittee.evaluate_company(**_healthy(cash_end_t=230.0))
    assert off_cash.recommendation == CommitteeRecommendation.REJECTED_ACCOUNTING_ANOMALY
    assert off_cash.checks[1].difference == 10.0 and off_cash.checks[0].holds


def test_altman_uses_retained_earnings_and_operating_income() -> None:
    from lilavati.agents.forensics import ForensicEngine

    memo = AdversarialCommittee.evaluate_company(**_healthy(retained_earnings_t=-300.0))
    # Altman (1968): X2 = retained earnings / total assets, X3 = EBIT / total assets,
    # X4 = market equity / total liabilities. The old code used net income (250) for X2.
    expected = ForensicEngine.calculate_altman_z_score(
        working_capital=500.0 - 150.0, total_assets=1600.0, retained_earnings=-300.0, ebit=360.0,
        market_value_equity=3500.0, total_liabilities=600.0, sales=1200.0)
    assert memo.altman_z_score.z_score == expected.z_score
    # 1.2*0.21875 + 1.4*(-0.1875) + 3.3*0.225 + 0.6*(3500/600) + 0.999*0.75
    # = 0.2625 - 0.2625 + 0.7425 + 3.5 + 0.74925 = 4.99175
    assert abs(memo.altman_z_score.z_score - 4.99175) < 1e-9
