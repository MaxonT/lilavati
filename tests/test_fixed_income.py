import pytest

from lilavati.core.fixed_income import FixedIncomeEngine


def test_bond_analytics() -> None:
    # 5-year 5% annual coupon bond yielding 5% should price at 100 par
    result = FixedIncomeEngine.analyze_bond(
        face_value=100.0,
        coupon_rate=0.05,
        ytm=0.05,
        years_to_maturity=5.0,
        frequency=1,
    )

    assert result.clean_price == pytest.approx(100.0, abs=1e-4)
    # Macaulay duration for 5yr 5% par bond is ~4.545 years
    assert 4.0 < result.macaulay_duration < 5.0
    assert result.modified_duration < result.macaulay_duration
    assert result.dv01 > 0.0
    assert result.convexity > 0.0


@pytest.mark.parametrize(
    ("years_to_maturity", "frequency"),
    [(0.2, 2), (0.25, 2), (0.0, 2), (0.04, 12)],
)
def test_bond_analytics_rejects_maturity_below_one_period(
    years_to_maturity: float, frequency: int
) -> None:
    # Fewer than one whole coupon period rounds to n_periods = 0, which used to end in ZeroDivisionError.
    with pytest.raises(ValueError, match="no whole coupon period"):
        FixedIncomeEngine.analyze_bond(
            face_value=100.0,
            coupon_rate=0.05,
            ytm=0.06,
            years_to_maturity=years_to_maturity,
            frequency=frequency,
        )


def test_bond_analytics_accepts_one_period_maturity() -> None:
    # 0.5 years at frequency 2 is exactly one period: price = (100 + 2.5) / (1 + 0.06 / 2).
    result = FixedIncomeEngine.analyze_bond(
        face_value=100.0, coupon_rate=0.05, ytm=0.06, years_to_maturity=0.5, frequency=2
    )
    assert result.clean_price == pytest.approx(102.5 / 1.03, rel=1e-12)


def test_nelson_siegel_curve_fit() -> None:
    # Upward-sloping yield curve points: 1y, 2y, 5y, 10y, 30y
    maturities = [1.0, 2.0, 5.0, 10.0, 30.0]
    yields = [0.035, 0.038, 0.042, 0.046, 0.050]

    result = FixedIncomeEngine.fit_nelson_siegel(maturities, yields)

    assert result.r_squared > 0.95
    # Long-term level beta0 should be close to 30y yield
    assert 0.03 < result.beta0_level < 0.08
    assert result.lambda_decay > 0.0


def test_merton_structural_model() -> None:
    # Firm with $100M equity value, 40% equity vol, $80M debt face value, 5% r, 1yr maturity
    result = FixedIncomeEngine.solve_merton_model(
        equity_value=100.0,
        equity_volatility=0.40,
        debt_face_value=80.0,
        risk_free_rate=0.05,
        maturity_years=1.0,
    )

    # Implied asset value should be approximately Equity + Debt (~$180M)
    assert 150.0 < result.asset_value < 200.0
    # Asset volatility should be lower than equity volatility (delevered)
    assert result.asset_volatility < 0.40
    # Distance to default should be positive for a healthy firm
    assert result.distance_to_default > 1.5
    assert 0.0 <= result.default_probability < 0.10
    assert result.theoretical_credit_spread_bps > 0.0
