"""
Derivatives, Options, Volatility Surfaces, and Greeks Core Engine.

Implements:
1. Analytical Black-Scholes-Merton (BSM) with continuous dividend yield.
2. American Option Pricing via Cox-Ross-Rubinstein (CRR) Binomial Trees.
3. First- and Second-Order Greeks: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga.
4. Hagan et al. (2002) SABR Stochastic Volatility Model Calibration.
5. SMT check of the butterfly (strike-convexity) no-arbitrage condition.

Conventions shared by every function in this module:

* ``time_to_maturity`` is in years (ACT/365: one calendar day is ``1/365``).
* ``risk_free_rate`` and ``dividend_yield`` are continuously compounded annual rates in decimal
  (``0.05`` means 5%).
* ``volatility`` is the annualised lognormal (Black) volatility in decimal (``0.20`` means 20%).
* Prices are in the currency units of ``spot``/``strike``; option prices are per unit of underlying.

Greek scaling (see :meth:`DerivativesEngine.calculate_bsm_greeks`): vega and rho are reported per
1 percentage point (``0.01``) move, theta per calendar day (per-year theta / 365), and vanna/volga are
derivatives of the *reported* delta and vega with respect to sigma in decimal units.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import z3
from scipy.optimize import curve_fit, root_scalar
from scipy.stats import norm

from lilavati.schemas.derivatives import (
    ArbitrageVerificationResult,
    ExerciseStyle,
    OptionGreeksResult,
    OptionPricingResult,
    OptionType,
    SABRCalibrationResult,
)

# Implied-volatility search bracket, annualised decimal vol (0.01% to 500%).
_IV_LOWER = 1e-4
_IV_UPPER = 5.0

# Below this |z| the SABR factor z / x(z) is evaluated with its Taylor series. At |z| = 1e-4 the series
# truncation error is < 1e-13 while the closed form loses ~1e-12 to cancellation (measured).
_SABR_SMALL_Z = 1e-4

# Tolerance on discrete slope differences of undiscounted Black call prices when checking a fitted
# SABR smile for butterfly / call-spread arbitrage. Slopes dC/dK are dimensionless, so this is
# independent of the price scale; it only absorbs floating-point noise.
_SABR_ARB_TOL = 1e-9
_SABR_ARB_GRID = 101


def _validate_bsm_inputs(spot: float, strike: float, time_to_maturity: float, volatility: float) -> None:
    if not (time_to_maturity > 0 and volatility > 0 and spot > 0 and strike > 0):
        raise ValueError(
            "Spot, strike, time to maturity, and volatility must be strictly positive."
        )


def _black76_call_undiscounted(forward: float, strikes: np.ndarray, t: float, vols: np.ndarray) -> np.ndarray:
    """Undiscounted Black (1976) call prices ``F N(d1) - K N(d2)``."""
    sd = vols * math.sqrt(t)
    d1 = (np.log(forward / strikes) + 0.5 * sd**2) / sd
    return np.asarray(forward * norm.cdf(d1) - strikes * norm.cdf(d1 - sd), dtype=np.float64)


class DerivativesEngine:
    """Deterministic analytics for options, Greeks, and volatility surfaces."""

    @staticmethod
    def calculate_bsm_greeks(
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        is_call: bool = True,
    ) -> OptionGreeksResult:
        """Analytical Black-Scholes-Merton Greeks of a European option with continuous yield ``q``.

        Inputs follow the module conventions (years, continuous decimal rates, decimal vol). With
        ``V`` the option value, ``sigma`` the volatility in decimal and ``T`` the time to maturity in
        years, the returned fields are:

        * ``delta`` = dV/dS (per 1 unit of spot).
        * ``gamma`` = d2V/dS2 (per 1 unit of spot, squared).
        * ``vega``  = 0.01 * dV/dsigma: price change for a **1 vol-point** (0.01) rise in sigma.
          Example: S=K=100, r=5%, sigma=20%, T=1, q=0 gives vega = 0.3752 (QuantLib ``vega()``
          returns the per-1.0 value 37.52).
        * ``theta`` = dV/dt / 365 = -(dV/dT) / 365: price change per **calendar day** that passes
          (QuantLib ``thetaPerDay()``; the per-year theta is ``365 * theta``). Usually negative.
        * ``rho``   = 0.01 * dV/dr: price change for a **1 percentage-point** (0.01) rise in r
          (QuantLib ``rho() / 100``). Sensitivity to ``q`` is not returned.
        * ``vanna`` = d(delta)/dsigma = d2V/(dS dsigma), with sigma in decimal and **no** 0.01 scaling
          (change in delta per 1.0 of vol; divide by 100 for per vol point). Equivalently
          ``vanna = d(vega_per_1.0)/dS = 100 * d(vega)/dS`` for the reported ``vega``.
        * ``volga`` = d(vega)/dsigma for the *reported* (per-vol-point) ``vega``, i.e.
          ``0.01 * d2V/dsigma2`` (change in per-1.0 vega per vol point).

        Raises:
            ValueError: if spot, strike, time to maturity or volatility is not strictly positive.
        """
        s, k, t, r, q, v = (
            spot,
            strike,
            time_to_maturity,
            risk_free_rate,
            dividend_yield,
            volatility,
        )

        _validate_bsm_inputs(s, k, t, v)

        d1 = (math.log(s / k) + (r - q + 0.5 * v**2) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d2 = norm.cdf(-d2)

        exp_qt = math.exp(-q * t)
        exp_rt = math.exp(-r * t)

        # Delta
        delta = exp_qt * cdf_d1 if is_call else -exp_qt * cdf_neg_d1

        # Gamma (identical for Call and Put)
        gamma = (exp_qt * pdf_d1) / (s * v * math.sqrt(t))

        # Vega per 1 vol point (0.01), identical for Call and Put
        vega = (s * exp_qt * math.sqrt(t) * pdf_d1) * 0.01

        # Theta per calendar day: per-year theta / 365
        common_theta = -(s * exp_qt * pdf_d1 * v) / (2.0 * math.sqrt(t))
        if is_call:
            theta = (common_theta - r * k * exp_rt * cdf_d2 + q * s * exp_qt * cdf_d1) / 365.0
        else:
            theta = (
                common_theta + r * k * exp_rt * cdf_neg_d2 - q * s * exp_qt * cdf_neg_d1
            ) / 365.0

        # Rho per 1 percentage point (0.01) of r
        rho = (k * t * exp_rt * cdf_d2 * 0.01) if is_call else (-k * t * exp_rt * cdf_neg_d2 * 0.01)

        # Second-order Greeks
        # Vanna: d(Delta)/d(sigma) = -exp(-qT) * phi(d1) * (d2 / sigma), sigma in decimal, unscaled
        vanna = -exp_qt * pdf_d1 * (d2 / v)

        # Volga: d(vega per vol point)/d(sigma) = 0.01 * S e^{-qT} sqrt(T) phi(d1) * d1 * d2 / sigma
        volga = (s * exp_qt * math.sqrt(t) * pdf_d1) * (d1 * d2 / v) * 0.01

        return OptionGreeksResult(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            vanna=float(vanna),
            volga=float(volga),
        )

    @staticmethod
    def price_european_option(
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        is_call: bool = True,
    ) -> OptionPricingResult:
        """Analytical Black-Scholes-Merton value of a European option with continuous yield ``q``.

        ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`` and ``P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)``
        (Hull, *Options, Futures, and Other Derivatives*, BSM with dividend yield). Inputs follow the
        module conventions; the attached Greeks use the scaling documented in
        :meth:`calculate_bsm_greeks`.

        Raises:
            ValueError: if spot, strike, time to maturity or volatility is not strictly positive.
        """
        s, k, t, r, q, v = (
            spot,
            strike,
            time_to_maturity,
            risk_free_rate,
            dividend_yield,
            volatility,
        )

        _validate_bsm_inputs(s, k, t, v)

        d1 = (math.log(s / k) + (r - q + 0.5 * v**2) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)

        exp_qt = math.exp(-q * t)
        exp_rt = math.exp(-r * t)

        if is_call:
            price = s * exp_qt * norm.cdf(d1) - k * exp_rt * norm.cdf(d2)
        else:
            price = k * exp_rt * norm.cdf(-d2) - s * exp_qt * norm.cdf(-d1)

        greeks = DerivativesEngine.calculate_bsm_greeks(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield, is_call
        )

        return OptionPricingResult(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=OptionType.CALL if is_call else OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN,
            price=float(price),
            greeks=greeks,
        )

    @staticmethod
    def price_american_option_crr(
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        is_call: bool = True,
        steps: int = 200,
    ) -> OptionPricingResult:
        """Cox-Ross-Rubinstein (1979) binomial tree for an American option with early exercise.

        Tree: ``dt = T/steps``, ``u = exp(sigma sqrt(dt))``, ``d = 1/u``, risk-neutral up probability
        ``p = (exp((r - q) dt) - d) / (u - d)``, discount ``exp(-r dt)`` per step. Inputs follow the
        module conventions. Discretisation error is O(1/steps) (e.g. ~ -2/steps for an ATM
        S=K=100, r=5%, sigma=20%, T=1 call); see ``docs/validation/derivatives.md``.

        Note:
            ``greeks`` holds the **European** BSM Greeks for the same inputs (same scaling as
            :meth:`calculate_bsm_greeks`), not tree Greeks of the American option. They coincide for
            a call with ``q = 0`` (never optimal to exercise early) but differ when early exercise
            has value (e.g. American puts).

        Raises:
            ValueError: if inputs are not strictly positive, ``steps < 1``, or the risk-neutral
                probability ``p`` falls outside (0, 1) (too few steps for the given rates and vol).
        """
        _validate_bsm_inputs(spot, strike, time_to_maturity, volatility)
        if steps < 1:
            raise ValueError("steps must be a positive integer.")

        dt = time_to_maturity / steps
        u = math.exp(volatility * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)
        if not 0.0 < p < 1.0:
            raise ValueError(
                f"CRR risk-neutral probability p={p:.6f} is outside (0, 1); increase steps so that "
                "volatility * sqrt(dt) exceeds |r - q| * dt."
            )
        df = math.exp(-risk_free_rate * dt)

        def nodes(i: int) -> np.ndarray:
            j = np.arange(i + 1, dtype=np.float64)
            return np.asarray(spot * u**j * d ** (i - j), dtype=np.float64)

        # Option payoffs at expiration
        asset_prices = nodes(steps)
        if is_call:
            values = np.maximum(0.0, asset_prices - strike)
        else:
            values = np.maximum(0.0, strike - asset_prices)

        # Backward induction with early exercise check
        for i in range(steps - 1, -1, -1):
            asset_prices = nodes(i)
            continuation_values = df * (p * values[1:] + (1.0 - p) * values[:-1])
            if is_call:
                exercise_values = np.maximum(0.0, asset_prices - strike)
            else:
                exercise_values = np.maximum(0.0, strike - asset_prices)
            values = np.maximum(continuation_values, exercise_values)

        greeks = DerivativesEngine.calculate_bsm_greeks(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield, is_call
        )

        return OptionPricingResult(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=OptionType.CALL if is_call else OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN,
            price=float(values[0]),
            greeks=greeks,
        )

    @staticmethod
    def calculate_implied_volatility(
        price: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        is_call: bool = True,
    ) -> float:
        """Inverts the BSM formula (with yield ``q``) for the annualised decimal implied volatility.

        Brent's method on ``sigma`` in ``[1e-4, 5.0]`` (0.01% to 500%) with SciPy's default
        tolerances. Where vega is negligible (deep ITM/OTM, very short expiry) the price carries
        almost no information about sigma: the returned vol then reprices ``price`` to within
        ~1e-10 but need not equal the vol that generated it, and a price indistinguishable from the
        intrinsic/zero bound returns the bracket edge ``1e-4``.

        Raises:
            ValueError: if no sigma in the bracket reproduces ``price`` (e.g. the price is outside
                the no-arbitrage bounds or implies a vol above 500%).
        """

        def obj(v: float) -> float:
            res = DerivativesEngine.price_european_option(
                spot, strike, time_to_maturity, risk_free_rate, v, dividend_yield, is_call
            )
            return res.price - price

        lo, hi = obj(_IV_LOWER), obj(_IV_UPPER)
        if lo * hi > 0:
            raise ValueError(
                f"No implied volatility in [{_IV_LOWER}, {_IV_UPPER}] reproduces price {price}: "
                f"BSM prices over the bracket span [{lo + price:.12g}, {hi + price:.12g}]."
            )
        sol = root_scalar(obj, bracket=[_IV_LOWER, _IV_UPPER], method="brentq")
        return float(sol.root)

    @staticmethod
    def sabr_implied_volatility(
        f: float,
        k: float,
        t: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """Hagan et al. (2002) lognormal (Black) implied vol of the SABR model, eq. (2.17a).

        Model: ``dF = alpha_t F^beta dW1``, ``d alpha_t = nu alpha_t dW2``, ``dW1 dW2 = rho dt``.
        ``f`` is the forward, ``k`` the strike, ``t`` years to expiry; returns the annualised decimal
        Black vol to plug into Black-76 with the same forward. ``alpha`` is in forward units to the
        power ``1 - beta`` (so the ATM vol is roughly ``alpha / f**(1 - beta)``).

        At ``k == f`` the formula reduces to the ATM expression (2.18). The factor ``z / x(z)`` is
        evaluated with its Taylor series ``1 - rho z / 2 - (3 rho^2 - 2) z^2 / 12`` when
        ``|z| < 1e-4`` to avoid cancellation, so the result is continuous through the money.

        Raises:
            ValueError: if ``f``, ``k`` or ``alpha`` is not positive, ``t`` or ``nu`` is negative,
                ``beta`` is outside [0, 1] or ``|rho| >= 1``.
        """
        if not (f > 0 and k > 0 and alpha > 0 and t >= 0 and nu >= 0):
            raise ValueError("SABR requires f, k, alpha > 0 and t, nu >= 0.")
        if not (0.0 <= beta <= 1.0 and -1.0 < rho < 1.0):
            raise ValueError("SABR requires beta in [0, 1] and rho in (-1, 1).")

        fk = f * k
        log_fk = math.log(f / k)
        fk_pow = fk ** ((1.0 - beta) / 2.0)
        z = (nu / alpha) * fk_pow * log_fk

        if abs(z) < _SABR_SMALL_Z:
            factor = 1.0 - 0.5 * rho * z - (3.0 * rho**2 - 2.0) * z**2 / 12.0
        else:
            x_z = math.log((math.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
            factor = z / x_z

        denom = fk_pow * (
            1.0
            + ((1.0 - beta) ** 2 / 24.0) * (log_fk**2)
            + ((1.0 - beta) ** 4 / 1920.0) * (log_fk**4)
        )

        series = (
            1.0
            + (
                ((1.0 - beta) ** 2 / 24.0) * (alpha**2 / (fk ** (1.0 - beta)))
                + (rho * beta * nu * alpha) / (4.0 * fk_pow)
                + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
            )
            * t
        )

        return float((alpha * factor / denom) * series)

    @staticmethod
    def calibrate_sabr(
        forward: float,
        strikes: Sequence[float],
        market_vols: Sequence[float],
        time_to_maturity: float,
        beta: float = 0.5,
    ) -> SABRCalibrationResult:
        """Least-squares fit of SABR ``(alpha, rho, nu)`` to Black vols at one expiry, ``beta`` fixed.

        ``market_vols`` are annualised decimal Black vols at ``strikes``; the objective is the sum of
        squared vol errors using :meth:`sabr_implied_volatility`. Bounds: ``rho`` in [-0.999, 0.999],
        ``nu`` in [1e-4, 5], and ``alpha`` in ``[1e-4, 5] * forward**(1 - beta)`` (an ATM-equivalent
        lognormal vol of 0.01% to 500%, so the bound scales with the forward level). The initial
        ``alpha`` uses the quoted vol nearest the forward.

        ``rmse`` is in vol units (decimal). ``is_arbitrage_free`` is True when undiscounted Black-76
        call prices implied by the *fitted* smile, on the quoted strikes plus evenly spaced points
        in each gap (at least 101 nodes in total), have discrete slopes in [-1, 0] (no call-spread arbitrage) and non-decreasing
        slopes (no butterfly arbitrage, i.e. non-negative risk-neutral density), to a tolerance of
        1e-9. Calendar arbitrage needs several expiries and is not checked.
        """
        k_arr = np.asarray(strikes, dtype=np.float64)
        v_arr = np.asarray(market_vols, dtype=np.float64)
        if k_arr.shape != v_arr.shape or k_arr.size < 3:
            raise ValueError("strikes and market_vols must have equal length >= 3.")

        scale = forward ** (1.0 - beta)
        alpha_lo, alpha_hi = 1e-4 * scale, 5.0 * scale
        atm_vol = float(v_arr[int(np.argmin(np.abs(k_arr - forward)))])
        alpha_init = float(np.clip(atm_vol * scale, alpha_lo * 1.01, alpha_hi * 0.99))

        def model_func(k: np.ndarray, alpha: float, rho: float, nu: float) -> np.ndarray:
            return np.array(
                [
                    DerivativesEngine.sabr_implied_volatility(
                        forward, float(ki), time_to_maturity, alpha, beta, rho, nu
                    )
                    for ki in k
                ]
            )

        popt, _ = curve_fit(
            model_func,
            k_arr,
            v_arr,
            p0=[alpha_init, -0.2, 0.4],
            bounds=([alpha_lo, -0.999, 1e-4], [alpha_hi, 0.999, 5.0]),
            maxfev=5000,
        )

        calibrated_alpha, calibrated_rho, calibrated_nu = popt
        fitted_vols = model_func(k_arr, calibrated_alpha, calibrated_rho, calibrated_nu)
        rmse = float(np.sqrt(np.mean((fitted_vols - v_arr) ** 2)))

        # Quoted strikes plus evenly spaced interior points in each gap (no near-duplicate nodes).
        knots = np.unique(k_arr)
        n_sub = max(1, math.ceil((_SABR_ARB_GRID - 1) / (knots.size - 1)))
        grid = np.concatenate(
            [np.linspace(a, b, n_sub + 1)[:-1] for a, b in zip(knots[:-1], knots[1:], strict=True)]
            + [knots[-1:]]
        )
        grid_vols = model_func(grid, calibrated_alpha, calibrated_rho, calibrated_nu)
        calls = _black76_call_undiscounted(forward, grid, time_to_maturity, grid_vols)
        slopes = np.diff(calls) / np.diff(grid)
        is_arbitrage_free = bool(
            np.all(slopes <= _SABR_ARB_TOL)
            and np.all(slopes >= -1.0 - _SABR_ARB_TOL)
            and np.all(np.diff(slopes) >= -_SABR_ARB_TOL)
        )

        return SABRCalibrationResult(
            alpha=float(calibrated_alpha),
            beta=float(beta),
            rho=float(calibrated_rho),
            nu=float(calibrated_nu),
            rmse=rmse,
            is_arbitrage_free=is_arbitrage_free,
        )

    @staticmethod
    def verify_butterfly_arbitrage(
        strikes: Sequence[float],
        call_prices: Sequence[float],
        tolerance: float = 1e-4,
    ) -> ArbitrageVerificationResult:
        """
        Checks butterfly-spread (strike-convexity) no-arbitrage of call prices with the Z3 solver.

        Points are sorted by strike first. For each consecutive triple K1 < K2 < K3 the condition is
        ``(C3 - C2)/(K3 - K2) - (C2 - C1)/(K2 - K1) >= -tolerance``; with equal spacing ``h`` this
        is ``(C1 - 2 C2 + C3) / h >= -tolerance`` (non-negative risk-neutral density). The
        tolerance therefore applies to a difference of slopes (dimensionless), not to the
        butterfly's price. Call-spread bounds (slopes in [-1, 0]) are not checked here.

        Raises:
            ValueError: if fewer than 3 points, lengths differ, any value is not finite, or two
                strikes are equal.
        """
        violations: list[str] = []
        if len(strikes) != len(call_prices) or len(strikes) < 3:
            raise ValueError(
                "At least 3 strike/price points required for butterfly arbitrage verification."
            )
        points = sorted(zip((float(x) for x in strikes), (float(x) for x in call_prices), strict=True))
        if not all(math.isfinite(ki) and math.isfinite(ci) for ki, ci in points):
            raise ValueError("Strikes and call prices must be finite numbers.")
        k = [ki for ki, _ in points]
        c = [ci for _, ci in points]
        if any(k[i] == k[i + 1] for i in range(len(k) - 1)):
            raise ValueError("Strikes must be distinct.")

        solver = z3.Solver()

        for i in range(1, len(k) - 1):
            k_prev, k_curr, k_next = k[i - 1], k[i], k[i + 1]
            c_prev, c_curr, c_next = c[i - 1], c[i], c[i + 1]

            # Linear butterfly payoff condition for convex curve
            # (C(K3) - C(K2))/(K3 - K2) - (C(K2) - C(K1))/(K2 - K1) >= -tol
            slope_right = (c_next - c_curr) / (k_next - k_curr)
            slope_left = (c_curr - c_prev) / (k_curr - k_prev)
            butterfly_spread = slope_right - slope_left

            bf_sym = z3.Real(f"butterfly_{i}")
            solver.add(bf_sym == z3.RealVal(str(butterfly_spread)))
            solver.add(bf_sym >= z3.RealVal(str(-tolerance)))

            if butterfly_spread < -tolerance:
                violations.append(
                    f"Butterfly arbitrage violated at strike {k_curr}: "
                    f"Convexity slope delta is {butterfly_spread:.6f} < 0. Implies negative risk-neutral density."
                )

        check = solver.check()
        is_valid = (check == z3.sat) and (len(violations) == 0)

        return ArbitrageVerificationResult(
            is_valid=is_valid,
            violations=violations,
        )
