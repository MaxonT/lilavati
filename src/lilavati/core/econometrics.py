"""
Core Financial Econometrics Engine.

Implements time-series stationarity testing (ADF), Engle-Granger cointegration for statistical
arbitrage, the Ornstein-Uhlenbeck mean-reversion half-life of the cointegrating spread, and
GARCH(p, q) volatility estimation.

All statistics are computed by statsmodels (ADF, Engle-Granger) and arch (GARCH); this module
fixes their settings explicitly, validates inputs and converts units. Audit results:
``docs/validation/econometrics.md``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from arch import arch_model
from pydantic import BaseModel, ConfigDict, Field
from statsmodels.tsa.stattools import adfuller, coint

#: Trading periods per year used to annualise GARCH volatility (daily returns assumed).
PERIODS_PER_YEAR = 252

# Same perfect-collinearity threshold statsmodels' ``coint`` uses (R^2 >= 1 - 100 * sqrt(eps)).
_COLLINEAR_R2 = 1.0 - 100.0 * math.sqrt(np.finfo(np.float64).eps)


def _validated_series(series: Sequence[float] | np.ndarray, name: str, min_len: int) -> np.ndarray:
    """Return ``series`` as a finite, non-constant 1-D float array of length >= ``min_len``."""
    data = np.asarray(series, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {data.shape}.")
    if len(data) < min_len:
        raise ValueError(
            f"Insufficient sample size for {name}: {len(data)} (minimum {min_len} required)."
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.ptp(data) == 0.0:
        raise ValueError(f"{name} is constant; the statistic is undefined.")
    return data


def _validated_significance(significance_level: float) -> float:
    level = float(significance_level)
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        raise ValueError(f"significance_level must lie in (0, 1); got {significance_level}.")
    return level


def _ar1_half_life(phi: float) -> float:
    """Half-life in periods of a deviation that decays as ``phi**h``: h = -ln 2 / ln|phi|.

    For 0 < phi < 1 this is the exact half-life of the AR(1) x_t = phi x_{t-1} + e_t and of an
    Ornstein-Uhlenbeck process sampled once per period (phi = exp(-theta), h = ln 2 / theta).
    For -1 < phi <= 0 (no OU embedding; deviations alternate sign) it is the half-life of the
    deviation's magnitude, 0 at phi = 0. |phi| >= 1 (no mean reversion) gives ``inf``.
    """
    a = abs(phi)
    if a >= 1.0:
        return float("inf")
    if a == 0.0:
        return 0.0
    return float(-math.log(2.0) / math.log(a))


class StationarityResult(BaseModel):
    """Result of unit root stationarity diagnostic testing."""

    model_config = ConfigDict(frozen=True)

    is_stationary: bool = Field(
        ..., description="True if the null hypothesis of a unit root is rejected"
    )
    test_statistic: float = Field(..., description="Augmented Dickey-Fuller test statistic")
    p_value: float = Field(..., description="MacKinnon's approximate p-value")
    lags_used: int = Field(..., description="Number of lags used in the test regression")
    n_observations: int = Field(..., description="Number of observations after lag trimming")
    critical_values: dict[str, float] = Field(
        ..., description="Critical values at 1%, 5%, and 10% levels"
    )
    recommendation: str = Field(..., description="Prescriptive guidance for modeling")


class CointegrationResult(BaseModel):
    """Result of cointegration analysis for pairs and basket trading."""

    model_config = ConfigDict(frozen=True)

    is_cointegrated: bool = Field(
        ..., description="True if the two series form a stationary linear combination"
    )
    p_value: float = Field(..., description="P-value of the Engle-Granger cointegration test")
    hedge_ratio: float = Field(
        ...,
        description=(
            "OLS slope beta in series_a = alpha + beta * series_b + spread: units of series_b "
            "per unit of series_a"
        ),
    )
    intercept: float = Field(..., description="OLS intercept (alpha), in series_a units")
    half_life: float = Field(
        ...,
        description=(
            "Mean-reversion half-life of the spread in sampling periods: -ln 2 / ln|phi| with "
            "phi the fitted AR(1) coefficient (equals ln 2 / theta for an OU process sampled "
            "once per period); inf if |phi| >= 1"
        ),
    )
    spread_ar1_coefficient: float = Field(
        ...,
        description=(
            "phi = 1 + b from the OLS regression delta_spread_t = c + b * spread_{t-1} + e_t"
        ),
    )
    spread_std: float = Field(
        ..., description="Population standard deviation (ddof=0) of the spread, series_a units"
    )
    recommendation: str = Field(..., description="Strategy recommendation")


class GARCHResult(BaseModel):
    """Result of GARCH(p,q) volatility modeling."""

    model_config = ConfigDict(frozen=True)

    p: int = Field(..., description="Number of lagged conditional variances (GARCH, beta) terms")
    q: int = Field(..., description="Number of lagged squared shocks (ARCH, alpha) terms")
    omega: float = Field(
        ..., description="Variance intercept in decimal-return units (per period, squared)"
    )
    alpha_coeffs: list[float] = Field(
        ..., description="alpha_1..alpha_q on past squared shocks (ARCH), unitless"
    )
    beta_coeffs: list[float] = Field(
        ..., description="beta_1..beta_p on past conditional variances (GARCH), unitless"
    )
    persistence: float = Field(
        ..., description="Sum of alpha + beta measuring volatility clustering persistence"
    )
    annualized_volatility: float = Field(
        ...,
        description=(
            "sigma_T * sqrt(252): in-sample conditional volatility of the last observation, "
            "decimal (0.20 = 20%), assuming daily returns"
        ),
    )
    aic: float = Field(
        ...,
        description="AIC of the fit on the scaled returns (100 * returns when rescale=True)",
    )
    bic: float = Field(
        ...,
        description="BIC of the fit on the scaled returns (100 * returns when rescale=True)",
    )


class EconometricEngine:
    """Deterministic financial econometrics calculations."""

    @staticmethod
    def test_stationarity(
        series: Sequence[float],
        significance_level: float = 0.05,
    ) -> StationarityResult:
        """
        Runs the Augmented Dickey-Fuller (ADF) test for a unit root.

        H0: the series has a unit root (non-stationary). H1: the series is stationary.

        Settings (passed explicitly to ``statsmodels.tsa.stattools.adfuller``): test regression
        with a constant and no trend (``regression="c"``); maximum lag
        ``min(ceil(12 * (n / 100) ** 0.25), n // 2 - 2)`` (Schwert 1989 rule, statsmodels
        default; 8 lags at n = 20, 16 at n = 300); lag chosen by
        AIC over 0..maxlag on a common sample, then re-estimated on all usable observations.
        ``p_value`` is MacKinnon's (1994) approximate asymptotic p-value; ``critical_values``
        are MacKinnon's (2010) finite-sample response surfaces at T = ``n_observations``.
        ``is_stationary`` is ``p_value < significance_level``.

        Raises ``ValueError`` for fewer than 20 points, NaN/inf values, a constant series, or
        ``significance_level`` outside (0, 1).
        """
        level = _validated_significance(significance_level)
        data = _validated_series(series, "ADF test", 20)

        with warnings.catch_warnings():
            # statsmodels >= 0.15 warns about the tuple return format.
            warnings.simplefilter("ignore", category=FutureWarning)
            adf_stat, p_val, used_lag, nobs, crit_vals, _ = adfuller(
                data, maxlag=None, regression="c", autolag="AIC"
            )

        is_stat = bool(p_val < level)
        if is_stat:
            rec = "Series is stationary (I(0)). Ready for mean-reverting modeling, ARIMA, or regression."
        else:
            rec = (
                f"Series is non-stationary (p={p_val:.4f} >= {level}). "
                "Do NOT run OLS directly. Apply fractional differentiation (d*) or first differencing."
            )

        return StationarityResult(
            is_stationary=is_stat,
            test_statistic=float(adf_stat),
            p_value=float(p_val),
            lags_used=int(used_lag),
            n_observations=int(nobs),
            critical_values={k: float(v) for k, v in crit_vals.items()},
            recommendation=rec,
        )

    @staticmethod
    def test_cointegration(
        series_a: Sequence[float],
        series_b: Sequence[float],
        significance_level: float = 0.05,
    ) -> CointegrationResult:
        """
        Engle-Granger two-step cointegration test of series_a on series_b, plus the
        mean-reversion half-life of the resulting spread.

        Step 1: OLS ``series_a = alpha + beta * series_b + spread``. Step 2:
        ``statsmodels.tsa.stattools.coint(series_a, series_b, trend="c", method="aeg",
        autolag="aic")``: ADF without deterministic terms on the step-1 residuals, AIC lag
        choice, MacKinnon (1994) p-value for N = 2 variables. The test is not symmetric: the
        result can change if the series are swapped.

        Half-life: regress ``delta_spread_t = c + b * spread_{t-1} + e_t`` by OLS, set
        phi = 1 + b, and report -ln 2 / ln|phi| periods (see ``_ar1_half_life``). This is the
        exact half-life of the fitted AR(1) and equals ln 2 / theta for an Ornstein-Uhlenbeck
        process sampled once per period (phi = exp(-theta)). The common shortcut ln 2 / (-b)
        (e.g. Chan, *Algorithmic Trading*, 2013) is its first-order approximation and
        overstates the half-life when b is not small. Units are the sampling period of the
        input (days for daily closes).

        Raises ``ValueError`` for unequal lengths, fewer than 30 points, NaN/inf values,
        a constant series, perfectly collinear series (constant spread; the test is undefined),
        or ``significance_level`` outside (0, 1).
        """
        level = _validated_significance(significance_level)
        y_raw = np.asarray(series_a, dtype=np.float64)
        x_raw = np.asarray(series_b, dtype=np.float64)
        if y_raw.shape != x_raw.shape:
            raise ValueError(f"Series length mismatch: len(a)={len(y_raw)} != len(b)={len(x_raw)}")
        y = _validated_series(y_raw, "series_a (cointegration)", 30)
        x = _validated_series(x_raw, "series_b (cointegration)", 30)

        # Step 1: OLS regression to find hedge ratio: y = alpha + beta * x
        x_with_const = np.column_stack([np.ones_like(x), x])
        coeffs, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
        alpha, beta = float(coeffs[0]), float(coeffs[1])
        spread = y - (alpha + beta * x)

        r_squared = 1.0 - float(np.sum(spread**2)) / float(np.sum((y - y.mean()) ** 2))
        if r_squared >= _COLLINEAR_R2:
            raise ValueError(
                "series_a and series_b are (almost) perfectly collinear "
                f"(R^2 = {r_squared:.12f}); the spread is constant and the Engle-Granger test "
                "and half-life are undefined."
            )

        # Step 2: Engle-Granger (augmented) test
        _, p_val, _ = coint(y, x, trend="c", method="aeg", maxlag=None, autolag="aic")
        p_val = float(p_val)
        is_coint = bool(p_val < level)

        # Step 3: AR(1) / discretely sampled OU fit of the spread
        lagged_spread = spread[:-1]
        delta_spread = np.diff(spread)
        x_ou = np.column_stack([np.ones_like(lagged_spread), lagged_spread])
        ou_coeffs, _, _, _ = np.linalg.lstsq(x_ou, delta_spread, rcond=None)
        phi = 1.0 + float(ou_coeffs[1])
        half_life = _ar1_half_life(phi)

        if is_coint:
            rec = f"Pair is cointegrated (p={p_val:.4f}). Half-life is {half_life:.1f} periods. Stat-Arb pairs trading viable."
        else:
            rec = f"Pair is not cointegrated (p={p_val:.4f} >= {level}). Spread is non-stationary. Trading not advised."

        return CointegrationResult(
            is_cointegrated=is_coint,
            p_value=p_val,
            hedge_ratio=beta,
            intercept=alpha,
            half_life=half_life,
            spread_ar1_coefficient=phi,
            spread_std=float(np.std(spread)),
            recommendation=rec,
        )

    @staticmethod
    def fit_garch(
        returns: Sequence[float],
        p: int = 1,
        q: int = 1,
        rescale: bool = True,
    ) -> GARCHResult:
        """
        Fits a constant-mean GARCH(p, q) model with Gaussian errors by maximum likelihood (arch).

        r_t = mu + e_t,  e_t = sigma_t z_t,  z_t ~ N(0, 1),
        sigma_t^2 = omega + sum_{i=1..q} alpha_i e_{t-i}^2 + sum_{j=1..p} beta_j sigma_{t-j}^2.

        Order convention (Bollerslev 1986): ``p`` = number of GARCH (beta) lags, ``q`` = number
        of ARCH (alpha) lags. The arch package uses the opposite letters (its ``p`` is the ARCH
        order), so this method calls ``arch_model(p=q, q=p)``.

        Units: ``returns`` are per-period decimal returns (0.01 = 1%), assumed daily. With
        ``rescale=True`` the model is fitted to 100 * returns for numerical conditioning;
        omega is divided by 100^2 and volatility by 100, so omega and
        ``annualized_volatility`` are in decimal units either way. alpha and beta are
        scale-free. ``annualized_volatility`` = sigma_T * sqrt(252), where sigma_T is the
        in-sample conditional volatility of the last observation (not a forecast of T+1).
        AIC/BIC come from the likelihood of the scaled data.

        Raises ``ValueError`` for fewer than 100 points, NaN/inf values, constant returns,
        ``p < 0`` or ``q < 1``, or if the optimiser does not report convergence.
        """
        if isinstance(p, bool) or not isinstance(p, int) or p < 0:
            raise ValueError(f"p (GARCH order) must be an integer >= 0; got {p!r}.")
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError(f"q (ARCH order) must be an integer >= 1; got {q!r}.")
        ret = _validated_series(returns, "GARCH", 100)

        scale_factor = 100.0 if rescale else 1.0
        scaled_ret = ret * scale_factor

        # arch's p is the ARCH (alpha) order and its q the GARCH (beta) order.
        am = arch_model(
            scaled_ret, mean="constant", vol="garch", p=q, q=p, dist="normal", rescale=False
        )
        res = am.fit(disp="off", show_warning=False)
        if res.convergence_flag != 0:
            raise ValueError(
                "GARCH maximum likelihood did not converge "
                f"(optimizer code {res.convergence_flag}: {res.optimization_result.message})."
            )

        params = res.params
        omega = float(params["omega"]) / (scale_factor**2)
        alphas = [float(params[f"alpha[{i}]"]) for i in range(1, q + 1)]
        betas = [float(params[f"beta[{j}]"]) for j in range(1, p + 1)]
        persistence = float(sum(alphas) + sum(betas))

        latest_conditional_vol = float(res.conditional_volatility[-1]) / scale_factor
        annualized_vol = latest_conditional_vol * math.sqrt(PERIODS_PER_YEAR)

        return GARCHResult(
            p=p,
            q=q,
            omega=omega,
            alpha_coeffs=alphas,
            beta_coeffs=betas,
            persistence=persistence,
            annualized_volatility=annualized_vol,
            aic=float(res.aic),
            bic=float(res.bic),
        )
