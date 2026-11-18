import pytest

from lilavati.agents.forensics import ForensicEngine


def _args(**over: float) -> dict[str, float]:
    # Identical years => every index equals 1; TATA = 0 (net income == CFO).
    base = {
        "revenue_t": 1000.0, "revenue_t_minus_1": 1000.0,
        "cogs_t": 600.0, "cogs_t_minus_1": 600.0,
        "receivables_t": 100.0, "receivables_t_minus_1": 100.0,
        "current_assets_t": 400.0, "current_assets_t_minus_1": 400.0,
        "ppe_t": 300.0, "ppe_t_minus_1": 300.0,
        "total_assets_t": 1000.0, "total_assets_t_minus_1": 1000.0,
        "depreciation_t": 30.0, "depreciation_t_minus_1": 30.0,
        "sga_t": 150.0, "sga_t_minus_1": 150.0,
        "long_term_debt_t": 200.0, "long_term_debt_t_minus_1": 200.0,
        "current_liabilities_t": 150.0, "current_liabilities_t_minus_1": 150.0,
        "net_income_t": 100.0, "cfo_t": 100.0,
    }
    base.update(over)
    return base


def test_all_indices_one_matches_published_coefficients() -> None:
    # Beneish (1999): -4.84 + .920 + .528 + .404 + .892 + .115 - .172 + 4.679*0 - .327 = -2.480
    res = ForensicEngine.calculate_beneish_m_score(**_args())
    assert res.m_score == pytest.approx(-2.480, abs=1e-9)
    assert res.is_likely_manipulator is False


def test_higher_leverage_lowers_score() -> None:
    base = ForensicEngine.calculate_beneish_m_score(**_args()).m_score
    levered = ForensicEngine.calculate_beneish_m_score(
        **_args(long_term_debt_t=400.0)
    ).m_score
    assert levered < base  # LVGI coefficient is negative (-0.327)


def test_accruals_coefficient_is_4_679() -> None:
    base = ForensicEngine.calculate_beneish_m_score(**_args()).m_score
    # net income 100 above CFO on assets of 1000 => TATA = 0.1 => +0.4679
    hi = ForensicEngine.calculate_beneish_m_score(**_args(cfo_t=0.0)).m_score
    assert hi - base == pytest.approx(4.679 * 0.1, abs=1e-9)
