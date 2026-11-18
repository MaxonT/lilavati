"""Reference tests for the discrete Almgren-Chriss (2000) liquidation schedule.

Reference: Almgren, R. and Chriss, N. (2000), "Optimal Execution of Portfolio Transactions",
Journal of Risk 3(2), 5-39. Equation and section numbers refer to the December 2000 version.

The numeric reference case uses the paper's Section 3.4 / Table 1 parameters. Expected values
come from (a) an independent numpy solution of the paper's discrete optimisation problem
(direct least-squares minimisation of E + lambda*V, no sinh/cosh closed form), (b) the paper's
closed-form E and V (eq. 20) and linear-trajectory E and V (eqs. 10-11), and (c) the paper's
statement "kappa ~ 0.6/day, so kappa*T ~ 3" for these parameters (Section 3.4).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lilavati.core.execution import ExecutionEngine

# Almgren & Chriss (2000), Table 1. Time unit is the trading day; hours_per_day only maps the
# API's hour-denominated horizon back to days (T = 5 days, N = 5, tau = 1 day).
HOURS_PER_DAY = 6.5
PAPER = {
    "total_shares": 1_000_000.0,  # X = 10^6 shares
    "time_horizon_hours": 5 * HOURS_PER_DAY,  # T = 5 days
    "n_intervals": 5,  # N = 5
    "daily_volatility": 0.019,  # 0.3 / sqrt(250), so sigma = 0.019 * 50 = 0.95 $/share/day^0.5
    "arrival_price": 50.0,  # S0 = 50 $/share
    "permanent_impact_gamma": 2.5e-7,  # $/share^2
    "temporary_impact_eta": 2.5e-6,  # ($/share)/(share/day)
    "fixed_cost_epsilon": 0.0625,  # $/share
    "risk_aversion_lambda": 1e-6,  # 1/$
    "hours_per_day": HOURS_PER_DAY,
}
SIGMA = 0.95
TAU = 1.0
T_DAYS = 5.0
ETA_TILDE = 2.5e-6 - 0.5 * 2.5e-7 * TAU  # eq. 8


def _schedule(**overrides: float) -> list[float]:
    params = {**PAPER, **overrides}
    return ExecutionEngine.compute_almgren_chriss_schedule(**params).trading_trajectory  # type: ignore[arg-type]


def _numpy_reference_trajectory(
    x0: float, n: int, tau: float, sigma: float, eta_tilde: float, lam: float
) -> np.ndarray:
    """Independent solution: minimise (eta_tilde/tau)*sum n_k^2 + lam*sigma^2*tau*sum x_k^2.

    This is E + lambda*V from eqs. (5) and (8) without the path-independent terms
    (gamma*X^2/2 and epsilon*X). Unknowns are x_1..x_{N-1}; x_0 = X and x_N = 0. Written as a
    linear least-squares problem and solved with numpy, without the closed form (eq. 17).
    """
    m = n - 1
    # n_k = x_{k-1} - x_k for k = 1..N, as an affine map of y = (x_1..x_{N-1}).
    d = np.zeros((n, m))
    c = np.zeros(n)
    c[0] = x0
    for k in range(n):
        if k - 1 >= 0:
            d[k, k - 1] += 1.0  # + x_{k-1}
        if k < m:
            d[k, k] -= 1.0  # - x_k
    a = np.vstack([math.sqrt(eta_tilde / tau) * d, math.sqrt(lam * sigma**2 * tau) * np.eye(m)])
    b = np.concatenate([-math.sqrt(eta_tilde / tau) * c, np.zeros(m)])
    y, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.concatenate([[x0], y, [0.0]])


def test_paper_case_matches_independent_numpy_minimisation() -> None:
    traj = np.asarray(_schedule())
    ref = _numpy_reference_trajectory(1e6, 5, TAU, SIGMA, ETA_TILDE, 1e-6)
    np.testing.assert_allclose(traj, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("lam", [1e-8, 1e-7, 2e-6, 1e-5])
def test_other_lambdas_match_independent_numpy_minimisation(lam: float) -> None:
    traj = np.asarray(_schedule(risk_aversion_lambda=lam))
    ref = _numpy_reference_trajectory(1e6, 5, TAU, SIGMA, ETA_TILDE, lam)
    np.testing.assert_allclose(traj, ref, rtol=1e-9, atol=1e-6)


def test_paper_case_kappa_and_closed_form_cost_variance() -> None:
    res = ExecutionEngine.compute_almgren_chriss_schedule(**PAPER)  # type: ignore[arg-type]
    kappa = HOURS_PER_DAY / res.half_life_hours  # per day

    # Section 3.4: "kappa ~ 0.6/day, so kappa*T ~ 3" (stated to one significant figure).
    assert kappa == pytest.approx(0.6, abs=0.02)
    assert kappa * T_DAYS == pytest.approx(3.0, abs=0.1)
    # eq. 16 holds exactly: (2/tau^2)(cosh(kappa*tau) - 1) = lambda*sigma^2/eta_tilde
    assert (2.0 / TAU**2) * (math.cosh(kappa * TAU) - 1.0) == pytest.approx(
        1e-6 * SIGMA**2 / ETA_TILDE, rel=1e-12
    )

    # eq. 20 closed-form E and V of the optimal strategy.
    x, g, eps, k, t, tau = 1e6, 2.5e-7, 0.0625, kappa, T_DAYS, TAU
    e_ref = (
        0.5 * g * x**2
        + eps * x
        + ETA_TILDE
        * x**2
        * math.tanh(0.5 * k * tau)
        * (tau * math.sinh(2 * k * t) + 2 * t * math.sinh(k * tau))
        / (2 * tau**2 * math.sinh(k * t) ** 2)
    )
    v_ref = (
        0.5
        * SIGMA**2
        * x**2
        * (tau * math.sinh(k * t) * math.cosh(k * (t - tau)) - t * math.sinh(k * tau))
        / (math.sinh(k * t) ** 2 * math.sinh(k * tau))
    )
    assert res.expected_shortfall == pytest.approx(e_ref, rel=1e-10)
    assert res.shortfall_variance == pytest.approx(v_ref, rel=1e-10)


def test_lambda_zero_matches_linear_trajectory_cost_and_variance() -> None:
    res = ExecutionEngine.compute_almgren_chriss_schedule(**{**PAPER, "risk_aversion_lambda": 0.0})  # type: ignore[arg-type]
    x, n = 1e6, 5
    assert res.trading_trajectory == pytest.approx([x * (n - j) / n for j in range(n + 1)])
    # eq. 10: E = gamma X^2/2 + eps X + (eta - gamma tau/2) X^2 / T
    e_ref = 0.5 * 2.5e-7 * x**2 + 0.0625 * x + ETA_TILDE * x**2 / T_DAYS
    # eq. 11: V = sigma^2 X^2 T (1 - 1/N)(1 - 1/(2N)) / 3
    v_ref = SIGMA**2 * x**2 * T_DAYS * (1 - 1 / n) * (1 - 1 / (2 * n)) / 3.0
    assert res.expected_shortfall == pytest.approx(e_ref, rel=1e-12)
    assert res.shortfall_variance == pytest.approx(v_ref, rel=1e-12)
    assert math.isinf(res.half_life_hours)


@pytest.mark.parametrize("lam", [0.0, 1e-14, 1e-12])
def test_small_lambda_converges_to_twap(lam: float) -> None:
    x0, n = 1e6, 5
    traj = _schedule(risk_aversion_lambda=lam)
    for j, xj in enumerate(traj):
        assert xj / x0 == pytest.approx((n - j) / n, abs=1e-6)


def test_larger_lambda_is_more_front_loaded() -> None:
    lambdas = [0.0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5, 1e-4, 1.0]
    first_holdings = [_schedule(risk_aversion_lambda=lam)[1] for lam in lambdas]
    for a, b in zip(first_holdings, first_holdings[1:], strict=False):
        assert b < a


def test_extreme_risk_aversion_sells_almost_everything_immediately() -> None:
    # Regression for the clipped-sinh bug: lambda = 1e6 used to return
    # [100000, 100000, 100000, 100000, 100000, 0] (holding everything to the end).
    x0 = 100_000.0
    res = ExecutionEngine.compute_almgren_chriss_schedule(
        total_shares=x0,
        time_horizon_hours=4.0,
        n_intervals=5,
        daily_volatility=0.02,
        arrival_price=100.0,
        risk_aversion_lambda=1e6,
    )
    traj = res.trading_trajectory
    assert traj[1] / x0 < 0.01
    assert all(math.isfinite(v) for v in traj + res.trade_sizes)
    assert math.isfinite(res.expected_shortfall)
    assert res.shortfall_variance >= 0.0


@pytest.mark.parametrize("lam", [0.0, 1e-6, 1e-3, 1e6])
@pytest.mark.parametrize("n", [1, 2, 13, 390])
def test_holdings_boundary_conditions_and_monotone(lam: float, n: int) -> None:
    x0 = 250_000.0
    res = ExecutionEngine.compute_almgren_chriss_schedule(
        total_shares=x0,
        time_horizon_hours=6.5,
        n_intervals=n,
        daily_volatility=0.02,
        arrival_price=40.0,
        risk_aversion_lambda=lam,
    )
    traj = res.trading_trajectory
    assert len(traj) == n + 1
    assert traj[0] == x0
    assert traj[-1] == 0.0
    assert all(b <= a for a, b in zip(traj, traj[1:], strict=False))
    assert sum(res.trade_sizes) == pytest.approx(x0, rel=1e-12)


def test_non_convex_eta_tilde_raises() -> None:
    # eta_tilde = 2.5e-6 - 0.5 * 1e-5 * 1 day < 0
    with pytest.raises(ValueError, match="eta_tilde"):
        _schedule(permanent_impact_gamma=1e-5)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        _schedule(arrival_price=0.0)
    with pytest.raises(ValueError):
        _schedule(risk_aversion_lambda=-1e-6)
    with pytest.raises(ValueError):
        _schedule(n_intervals=0)
    with pytest.raises(TypeError):
        ExecutionEngine.compute_almgren_chriss_schedule(  # type: ignore[call-arg]
            total_shares=1.0, time_horizon_hours=1.0, n_intervals=1, daily_volatility=0.02
        )
