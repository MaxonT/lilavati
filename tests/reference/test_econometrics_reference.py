"""Reference tests for ``lilavati.core.econometrics`` (task 06 engine audit).

Expected values come from independent sources, never from the engine itself:

* statsmodels ``adfuller`` / ``coint`` and arch ``arch_model`` called directly with the settings
  the engine documents (regression/trend "c", AIC lag choice, constant-mean Gaussian GARCH).
* arch's own unit-root implementation (``arch.unitroot.ADF``, ``engle_granger``; K. Sheppard),
  which shares no code with statsmodels.
* A NumPy re-implementation of the ADF regression and AIC lag search: Said & Dickey (1984),
  "Testing for unit roots in ARMA models of unknown order", Biometrika 71(3); Hamilton,
  *Time Series Analysis* (1994), section 17.7.
* MacKinnon (2010), "Critical Values for Cointegration Tests", Queen's Economics Department
  Working Paper 1227, Table 2 response-surface coefficients (tau_c, N = 1 and N = 2). The values
  below are as tabulated identically in statsmodels ``adfvalues.tau_2010s`` and arch
  ``critical_values.dickey_fuller.tau_2010``; the PDF itself was not reachable in the audit
  session.
* Known-answer simulations (fixed seeds): AR(1) with phi = 0.5, random walks, y = 2x + I(0)
  noise, independent random walks (size of the test at 5%).
* Ornstein-Uhlenbeck exact discretisation x_{t+1} = e^{-theta} x_t + e_t (half-life ln 2 / theta);
  e.g. Tsay, *Analysis of Financial Time Series* (3rd ed.), AR(1) half-life ln(0.5)/ln(phi).
* GARCH(p, q) as defined by Bollerslev (1986), "Generalized autoregressive conditional
  heteroskedasticity", J. Econometrics 31, eq. (2): q ARCH (alpha) lags, p GARCH (beta) lags;
  simulated with a NumPy recursion independent of arch.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from arch import arch_model
from arch.unitroot import ADF
from arch.unitroot.cointegration import engle_granger
from statsmodels.tsa.stattools import adfuller, coint

from lilavati.core.econometrics import EconometricEngine

LN2 = math.log(2.0)

# MacKinnon (2010) Table 2, constant ("c") case: (tau_inf, beta1, beta2, beta3) at 1%, 5%, 10%.
MACKINNON_2010_C = {
    1: {
        "1%": (-3.43035, -6.5393, -16.786, -79.433),
        "5%": (-2.86154, -2.8903, -4.234, -40.040),
        "10%": (-2.56677, -1.5384, -2.809, 0.0),
    },
    2: {
        "1%": (-3.89644, -10.9519, -33.527, 0.0),
        "5%": (-3.33613, -6.1101, -6.823, 0.0),
        "10%": (-3.04445, -4.2412, -2.720, 0.0),
    },
}


def _mackinnon_cv(n_vars: int, level: str, t: int) -> float:
    b0, b1, b2, b3 = MACKINNON_2010_C[n_vars][level]
    return b0 + b1 / t + b2 / t**2 + b3 / t**3


def _sm_adf(x: np.ndarray) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return adfuller(x, maxlag=None, regression="c", autolag="AIC")


def _ar1(phi: float, n: int, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def _series_cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261119)
    return {
        "ar1_phi_0.5": _ar1(0.5, 400, rng),
        "ar1_phi_0.95": _ar1(0.95, 300, rng),
        "random_walk": np.cumsum(rng.normal(size=250)),
        "random_walk_drift": 100.0 + np.cumsum(0.05 + rng.normal(size=500)),
        "white_noise": rng.normal(size=120),
    }


def _numpy_adf(x: np.ndarray) -> tuple[float, int, int]:
    """Independent ADF: Delta x_t = c + g x_{t-1} + sum_k d_k Delta x_{t-k} + e_t.

    AIC lag choice over 0..maxlag on the common sample (n log(SSR/n) + 2 * n_params; the
    constant terms of the Gaussian likelihood cancel), then re-estimation on all usable
    observations. Returns (t-statistic of g, chosen lag, observations used).
    """
    n = len(x)
    maxlag = min(math.ceil(12.0 * (n / 100.0) ** 0.25), n // 2 - 2)
    dx = np.diff(x)

    def design(lag: int, start: int) -> tuple[np.ndarray, np.ndarray]:
        # rows t = start..n-2 index dx[t] = x[t+1] - x[t]
        rows = range(start, n - 1)
        y = np.array([dx[t] for t in rows])
        cols = [np.ones(len(y)), np.array([x[t] for t in rows])]
        cols += [np.array([dx[t - k] for t in rows]) for k in range(1, lag + 1)]
        return y, np.column_stack(cols)

    best_lag, best_ic = 0, math.inf
    for lag in range(maxlag + 1):
        y, z = design(lag, maxlag)
        coef, *_ = np.linalg.lstsq(z, y, rcond=None)
        ssr = float(np.sum((y - z @ coef) ** 2))
        ic = len(y) * math.log(ssr / len(y)) + 2 * z.shape[1]
        if ic < best_ic - 1e-12:
            best_lag, best_ic = lag, ic
    y, z = design(best_lag, best_lag)
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    resid = y - z @ coef
    s2 = float(resid @ resid) / (len(y) - z.shape[1])
    cov = s2 * np.linalg.inv(z.T @ z)
    return float(coef[1] / math.sqrt(cov[1, 1])), best_lag, len(y)


# ---------------------------------------------------------------------------------------------
# test_stationarity (ADF)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_series_cases()))
def test_adf_equals_statsmodels_called_directly(name: str) -> None:
    x = _series_cases()[name]
    res = EconometricEngine.test_stationarity(x.tolist())
    stat, pval, lag, nobs, cvs, _ = _sm_adf(x)
    assert res.test_statistic == stat
    assert res.p_value == pval
    assert (res.lags_used, res.n_observations) == (lag, nobs)
    assert res.critical_values == {k: float(v) for k, v in cvs.items()}
    assert res.is_stationary is (pval < 0.05)


@pytest.mark.parametrize("name", list(_series_cases()))
def test_adf_matches_numpy_reimplementation_and_arch(name: str) -> None:
    x = _series_cases()[name]
    res = EconometricEngine.test_stationarity(x.tolist())
    t_np, lag_np, nobs_np = _numpy_adf(x)
    assert (res.lags_used, res.n_observations) == (lag_np, nobs_np)
    assert res.test_statistic == pytest.approx(t_np, rel=1e-9)
    # arch: independent implementation, same Schwert max lag and AIC rule.
    arch_adf = ADF(x, trend="c", method="aic")
    assert arch_adf.lags == res.lags_used
    assert res.test_statistic == pytest.approx(arch_adf.stat, rel=1e-9)
    # arch uses its own p-value response surface; MacKinnon (1994) differs by < 0.005 here.
    assert res.p_value == pytest.approx(arch_adf.pvalue, abs=0.005)


@pytest.mark.parametrize("name", list(_series_cases()))
def test_adf_critical_values_are_mackinnon_2010(name: str) -> None:
    res = EconometricEngine.test_stationarity(_series_cases()[name].tolist())
    for level in ("1%", "5%", "10%"):
        expected = _mackinnon_cv(1, level, res.n_observations)
        assert res.critical_values[level] == pytest.approx(expected, abs=1e-9)


def test_adf_known_answers_across_seeds() -> None:
    # AR(1) phi = 0.5, n = 500: the unit root is rejected in every one of 30 seeds.
    for seed in range(30):
        x = _ar1(0.5, 500, np.random.default_rng(seed))
        assert EconometricEngine.test_stationarity(x.tolist()).is_stationary
    # Random walk, n = 250: rejection rate at 5% must be consistent with the nominal size.
    # 150 seeds; binomial(150, 0.05) puts 99.9% of outcomes in [1, 18].
    rejections = sum(
        EconometricEngine.test_stationarity(
            np.cumsum(np.random.default_rng(1000 + s).normal(size=250)).tolist()
        ).is_stationary
        for s in range(150)
    )
    assert 1 <= rejections <= 18


@pytest.mark.parametrize(
    "series",
    [
        [1.0] * 50,  # constant
        [0.0, 1.0, float("nan")] + list(range(40)),  # NaN
        [0.0, float("inf")] + list(range(40)),  # inf
        list(range(19)),  # too short
    ],
)
def test_adf_rejects_invalid_series(series: list[float]) -> None:
    with pytest.raises(ValueError):
        EconometricEngine.test_stationarity(series)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.05, 2.0, float("nan")])
def test_adf_rejects_invalid_significance(level: float) -> None:
    x = np.cumsum(np.random.default_rng(3).normal(size=100)).tolist()
    with pytest.raises(ValueError, match="significance_level"):
        EconometricEngine.test_stationarity(x, significance_level=level)


# ---------------------------------------------------------------------------------------------
# test_cointegration (Engle-Granger + half-life)
# ---------------------------------------------------------------------------------------------


def _coint_pair(seed: int, n: int = 300, phi: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = 50.0 + np.cumsum(rng.normal(size=n))
    y = 3.0 + 2.0 * x + _ar1(phi, n, rng)
    return y, x


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engle_granger_equals_statsmodels_and_arch(seed: int) -> None:
    y, x = _coint_pair(seed)
    res = EconometricEngine.test_cointegration(y.tolist(), x.tolist())
    _, p_sm, _ = coint(y, x, trend="c", method="aeg", maxlag=None, autolag="aic")
    assert res.p_value == p_sm
    # Step-1 OLS against a closed-form simple regression.
    beta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    assert res.hedge_ratio == pytest.approx(beta, rel=1e-10)
    assert res.intercept == pytest.approx(y.mean() - beta * x.mean(), rel=1e-9)
    spread = y - res.intercept - res.hedge_ratio * x
    assert res.spread_std == pytest.approx(float(np.std(spread)), rel=1e-12)
    # arch's independent Engle-Granger: same cointegrating vector and conclusion.
    eg = engle_granger(y, x, trend="c", method="aic")
    assert -eg.cointegrating_vector.iloc[1] == pytest.approx(res.hedge_ratio, rel=1e-9)
    assert (eg.pvalue < 0.05) is res.is_cointegrated


def test_engle_granger_statistic_matches_numpy_on_residuals() -> None:
    # coint = ADF without deterministic terms on OLS residuals; MacKinnon (2010) N = 2 values.
    y, x = _coint_pair(5, n=400, phi=0.9)
    stat, p_sm, cvs = coint(y, x, trend="c", method="aeg", autolag="aic")
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    eg = engle_granger(y, x, trend="c", method="aic")
    assert eg.stat == pytest.approx(stat, rel=1e-8)
    for level, cv in zip(("1%", "5%", "10%"), cvs, strict=True):
        assert cv == pytest.approx(_mackinnon_cv(2, level, len(resid) - 1), abs=1e-9)
    res = EconometricEngine.test_cointegration(y.tolist(), x.tolist())
    assert res.p_value == p_sm


def test_cointegration_known_answers_across_seeds() -> None:
    # y = 2x + stationary AR(1) noise (phi = 0.7, n = 300) cointegrates in all 30 seeds. OLS is
    # superconsistent but its finite-sample spread with phi = 0.7 is wide: measured max |b - 2|
    # = 0.13, median 0.029.
    errors = []
    for seed in range(30):
        y, x = _coint_pair(seed)
        res = EconometricEngine.test_cointegration(y.tolist(), x.tolist())
        assert res.is_cointegrated
        errors.append(abs(res.hedge_ratio - 2.0))
    assert max(errors) < 0.2
    assert float(np.median(errors)) < 0.05
    # Independent random walks: false-rejection rate consistent with 5% (150 seeds, [1, 18]).
    rejections = 0
    for seed in range(150):
        rng = np.random.default_rng(5000 + seed)
        a, b = np.cumsum(rng.normal(size=(2, 250)), axis=1)
        rejections += EconometricEngine.test_cointegration(a.tolist(), b.tolist()).is_cointegrated
    assert 1 <= rejections <= 18


def test_half_life_is_exact_ar1_mapping_of_fitted_phi() -> None:
    y, x = _coint_pair(11, n=500, phi=0.8)
    res = EconometricEngine.test_cointegration(y.tolist(), x.tolist())
    spread = y - res.intercept - res.hedge_ratio * x
    # Independent OLS of spread_t on (1, spread_{t-1}): slope = phi.
    phi = float(np.polyfit(spread[:-1], spread[1:], 1)[0])
    assert res.spread_ar1_coefficient == pytest.approx(phi, rel=1e-10)
    assert res.half_life == pytest.approx(-LN2 / math.log(phi), rel=1e-9)


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0])
def test_half_life_recovers_ou_theta(theta: float) -> None:
    """OU exact discretisation phi = e^{-theta}: half-life = ln 2 / theta periods.

    Before the fix the engine reported ln 2 / (1 - phi_hat): 7.07 (theta 0.1), 1.71 (0.5) and
    1.08 (1.0) against true 6.93, 1.39 and 0.69 (after: 6.71, 1.33, 0.67). The theta = 0.1 case
    passed before too: the shortcut's bias is inside sampling error when phi is near 1.
    """
    n = 4000
    rng = np.random.default_rng(int(theta * 1000))
    phi = math.exp(-theta)
    x = 10.0 + np.cumsum(rng.normal(size=n))
    y = 1.0 + 2.0 * x + _ar1(phi, n, rng)
    res = EconometricEngine.test_cointegration(y.tolist(), x.tolist())
    true_hl = LN2 / theta
    # Delta method: se(phi_hat) ~ sqrt((1 - phi^2) / n); dh/dphi = ln 2 / (phi ln(phi)^2).
    se_hl = math.sqrt((1 - phi**2) / n) * LN2 / (phi * math.log(phi) ** 2)
    assert abs(res.half_life - true_hl) < 4.0 * se_hl


def test_half_life_limits() -> None:
    # White-noise spread (phi ~ 0): half-life near 0, not ln 2 as the shortcut ln2/(-b) gives.
    rng = np.random.default_rng(17)
    x = np.cumsum(rng.normal(size=2000))
    y = 2.0 * x + rng.normal(size=2000)
    res = EconometricEngine.test_cointegration(y.tolist(), x.tolist())
    assert abs(res.spread_ar1_coefficient) < 0.1
    assert res.half_life == pytest.approx(-LN2 / math.log(abs(res.spread_ar1_coefficient)))
    assert res.half_life < 0.35
    # Independent random walks: phi close to 1, long half-life, not cointegrated.
    a, b = np.cumsum(np.random.default_rng(4).normal(size=(2, 300)), axis=1)
    res_rw = EconometricEngine.test_cointegration(a.tolist(), b.tolist())
    assert not res_rw.is_cointegrated
    assert res_rw.spread_ar1_coefficient > 0.9
    assert res_rw.half_life > 7.0


def test_cointegration_rejects_invalid_inputs() -> None:
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.normal(size=100))
    y = 2.0 * x + rng.normal(size=100)
    xn = x.copy()
    xn[10] = np.nan
    cases = [
        (2.0 * x + 1.0, x),  # perfectly collinear: constant spread
        ([3.0] * 100, x),  # constant series_a
        (y, [3.0] * 100),  # constant series_b
        (y, xn),  # NaN
        (y[:29], x[:29]),  # too short
        (y, x[:-1]),  # length mismatch
    ]
    for a, b in cases:
        with pytest.raises(ValueError):
            EconometricEngine.test_cointegration(list(a), list(b))
    with pytest.raises(ValueError, match="significance_level"):
        EconometricEngine.test_cointegration(y.tolist(), x.tolist(), significance_level=1.5)


# ---------------------------------------------------------------------------------------------
# fit_garch
# ---------------------------------------------------------------------------------------------


def _simulate_garch(
    omega: float,
    alphas: list[float],
    betas: list[float],
    n: int,
    seed: int,
    mu: float = 0.0,
    burn: int = 1000,
) -> np.ndarray:
    """Bollerslev (1986) GARCH(p, q) in decimal units: q = len(alphas), p = len(betas)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    uncond = omega / (1.0 - sum(alphas) - sum(betas))
    eps = np.zeros(n + burn)
    var = np.full(n + burn, uncond)
    m = max(len(alphas), len(betas))
    for t in range(m, n + burn):
        var[t] = omega
        var[t] += sum(a * eps[t - i - 1] ** 2 for i, a in enumerate(alphas))
        var[t] += sum(b * var[t - j - 1] for j, b in enumerate(betas))
        eps[t] = math.sqrt(var[t]) * z[t]
    return mu + eps[burn:]


def test_garch11_equals_arch_called_directly() -> None:
    r = _simulate_garch(2e-6, [0.08], [0.90], 3000, seed=1, mu=3e-4)
    res = EconometricEngine.fit_garch(r.tolist())
    ref = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    assert res.omega == pytest.approx(ref.params["omega"] / 1e4, rel=1e-6)
    assert res.alpha_coeffs[0] == pytest.approx(ref.params["alpha[1]"], rel=1e-6)
    assert res.beta_coeffs[0] == pytest.approx(ref.params["beta[1]"], rel=1e-6)
    assert res.persistence == pytest.approx(ref.params["alpha[1]"] + ref.params["beta[1]"])
    assert res.aic == pytest.approx(ref.aic, rel=1e-9)
    assert res.bic == pytest.approx(ref.bic, rel=1e-9)
    # Annualised vol: last in-sample sigma_t in decimal units times sqrt(252).
    sigma_t = ref.conditional_volatility[-1] / 100.0
    assert res.annualized_volatility == pytest.approx(sigma_t * math.sqrt(252), rel=1e-6)
    # sigma_T recomputed from the recursion with the fitted parameters (decimal units).
    w, a, b = res.omega, res.alpha_coeffs[0], res.beta_coeffs[0]
    mu = ref.params["mu"] / 100.0
    e = r - mu
    var = np.empty_like(r)
    var[0] = float(np.mean(e**2))  # arch's backcast is an EWMA; its effect dies out as b^t
    for t in range(1, len(r)):
        var[t] = w + a * e[t - 1] ** 2 + b * var[t - 1]
    assert res.annualized_volatility == pytest.approx(math.sqrt(var[-1] * 252), rel=1e-4)


def test_garch11_recovers_true_parameters() -> None:
    """omega = 2e-6, alpha = 0.08, beta = 0.90 (daily vol 1%), n = 20,000."""
    r = _simulate_garch(2e-6, [0.08], [0.90], 20_000, seed=7)
    res = EconometricEngine.fit_garch(r.tolist())
    ref = arch_model(100 * r, vol="GARCH", p=1, q=1).fit(disp="off")
    se = ref.std_err
    assert abs(res.alpha_coeffs[0] - 0.08) < 4 * se["alpha[1]"]
    assert abs(res.beta_coeffs[0] - 0.90) < 4 * se["beta[1]"]
    assert abs(res.omega - 2e-6) < 4 * se["omega"] / 1e4
    # Implied unconditional annual vol ~ 1% * sqrt(252) = 15.9%.
    uncond = math.sqrt(res.omega / (1 - res.persistence) * 252)
    assert uncond == pytest.approx(0.01 * math.sqrt(252), rel=0.1)


def test_garch_order_convention_p_is_beta_q_is_alpha() -> None:
    """Engine GARCH(p=1, q=2): one beta, two alphas (Bollerslev 1986).

    Before the fix arch was called with p and q unswapped: it fitted alpha[1], beta[1],
    beta[2], and the engine returned alpha_coeffs=[alpha_1, 0.0] (alpha[2] missing, filled
    with 0.0) and dropped beta[2].
    """
    true_a, true_b = [0.05, 0.10], [0.80]
    r = _simulate_garch(5e-6, true_a, true_b, 20_000, seed=21)
    res = EconometricEngine.fit_garch(r.tolist(), p=1, q=2)
    ref = arch_model(100 * r, vol="GARCH", p=2, q=1).fit(disp="off")  # arch: p = ARCH order
    assert len(res.alpha_coeffs) == 2 and len(res.beta_coeffs) == 1
    assert res.alpha_coeffs == pytest.approx(
        [ref.params["alpha[1]"], ref.params["alpha[2]"]], rel=1e-6
    )
    assert res.beta_coeffs == pytest.approx([ref.params["beta[1]"]], rel=1e-6)
    se = ref.std_err
    for i, a in enumerate(true_a, start=1):
        assert abs(res.alpha_coeffs[i - 1] - a) < 4 * se[f"alpha[{i}]"]
    assert abs(res.beta_coeffs[0] - true_b[0]) < 4 * se["beta[1]"]
    # The MCP tool exposes the same convention.
    from lilavati.mcp_servers.econometrics_server import fit_garch_volatility

    report = fit_garch_volatility(r.tolist(), p=1, q=2)
    assert report["alpha_coeffs"] == pytest.approx(res.alpha_coeffs)
    assert report["beta_coeffs"] == pytest.approx(res.beta_coeffs)


def test_pure_arch_model_p0() -> None:
    """GARCH(p=0, q=1) = ARCH(1) (Engle 1982); compare with arch's ARCH(1)."""
    r = _simulate_garch(1e-4 * 0.6, [0.4], [], 5000, seed=3)
    res = EconometricEngine.fit_garch(r.tolist(), p=0, q=1)
    ref = arch_model(100 * r, vol="GARCH", p=1, q=0).fit(disp="off")
    assert res.beta_coeffs == []
    assert res.alpha_coeffs[0] == pytest.approx(ref.params["alpha[1]"], rel=1e-6)
    assert abs(res.alpha_coeffs[0] - 0.4) < 4 * ref.std_err["alpha[1]"]


def test_garch_units_scale_with_returns() -> None:
    # Doubling returns: omega x4, annualised vol x2, alpha/beta unchanged (optimiser tolerance).
    r = _simulate_garch(2e-6, [0.08], [0.90], 3000, seed=9)
    base = EconometricEngine.fit_garch(r.tolist())
    dbl = EconometricEngine.fit_garch((2.0 * r).tolist())
    assert dbl.omega == pytest.approx(4.0 * base.omega, rel=1e-3)
    assert dbl.annualized_volatility == pytest.approx(2.0 * base.annualized_volatility, rel=1e-3)
    assert dbl.alpha_coeffs[0] == pytest.approx(base.alpha_coeffs[0], abs=1e-4)
    assert dbl.beta_coeffs[0] == pytest.approx(base.beta_coeffs[0], abs=1e-4)


@pytest.mark.parametrize(
    ("returns", "kwargs"),
    [
        ([0.001] * 200, {}),  # constant: optimiser fails (code 4) and old code returned guesses
        ([0.01, float("nan")] + [0.0, 0.01] * 60, {}),  # NaN
        ([0.01, -0.01] * 49 + [0.02], {}),  # 99 points
        (None, {"q": 0}),  # q = 0: no ARCH term
        (None, {"p": -1}),
        (None, {"p": 1.0}),
    ],
)
def test_garch_rejects_invalid_inputs(returns: list[float] | None, kwargs: dict) -> None:
    if returns is None:
        returns = _simulate_garch(2e-6, [0.08], [0.90], 500, seed=2).tolist()
    with pytest.raises(ValueError):
        EconometricEngine.fit_garch(returns, **kwargs)
