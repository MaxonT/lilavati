"""
Fixed Income, Debt & Credit Risk Core Engine.

Implements:
1. Bond analytics: Macaulay duration, modified duration, convexity, and DV01.
2. Nelson-Siegel parametric yield curve fitting.
3. Merton Structural Credit Risk model (Distance-to-Default and credit spread).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import brentq, curve_fit
from scipy.stats import norm

# Smallest relative tolerance scipy.optimize.brentq accepts (4 * machine epsilon).
_RTOL = 4.0 * float(np.finfo(np.float64).eps)


class BondAnalyticsResult(BaseModel):
    """Result of bond pricing and sensitivity analysis."""

    model_config = ConfigDict(frozen=True)

    clean_price: float = Field(
        ...,
        description=(
            "Price on a coupon date (accrued = 0, so clean = dirty), in the same currency units "
            "as face_value; per 100 par only when face_value = 100"
        ),
    )
    macaulay_duration: float = Field(..., description="Macaulay duration in years")
    modified_duration: float = Field(
        ...,
        description=(
            "Modified duration in years: -(1/P) dP/dy with y the annual nominal yield; "
            "a 1% (0.01) yield rise lowers price by about modified_duration percent"
        ),
    )
    convexity: float = Field(
        ...,
        description=(
            "Annualised convexity (1/P) d2P/dy2 in years^2 (not per period; per-period "
            "convexity = convexity * frequency^2)"
        ),
    )
    dv01: float = Field(
        ...,
        description=(
            "First-order price change for a 1bp move in the annual yield, "
            "modified_duration * price * 0.0001, positive, in face_value units "
            "(per 100 par only when face_value = 100)"
        ),
    )


class NelsonSiegelCurveResult(BaseModel):
    """Result of Nelson-Siegel yield curve parameter estimation."""

    model_config = ConfigDict(frozen=True)

    beta0_level: float = Field(..., description="Long-term asymptotic rate level y(inf), decimal")
    beta1_slope: float = Field(..., description="Slope loading; y(0+) = beta0 + beta1, decimal")
    beta2_curvature: float = Field(..., description="Medium-term curvature loading, decimal")
    lambda_decay: float = Field(
        ...,
        description=(
            "Time constant tau in years (Nelson-Siegel 1987 tau; QuantLib kappa = 1/tau). "
            "The curvature loading peaks at maturity ~1.79 * tau"
        ),
    )
    r_squared: float = Field(..., description="In-sample R^2 of the unweighted yield fit")


class MertonCreditResult(BaseModel):
    """Result of the Merton structural default risk model."""

    model_config = ConfigDict(frozen=True)

    asset_value: float = Field(
        ..., description="Implied market value of firm assets V0, same units as equity_value"
    )
    asset_volatility: float = Field(
        ..., description="Implied annualised asset volatility sigma_V, decimal"
    )
    distance_to_default: float = Field(
        ...,
        description=(
            "Risk-neutral distance to default d2 = [ln(V0/D) + (r - sigma_V^2/2) T] / "
            "(sigma_V sqrt(T)); uses r, not the physical asset drift"
        ),
    )
    default_probability: float = Field(
        ...,
        description="Risk-neutral probability N(-d2) that V_T < D at maturity T",
    )
    theoretical_credit_spread_bps: float = Field(
        ...,
        description="Continuously compounded spread -ln((V0 - E0)/D)/T - r, in bps, floored at 0",
    )


class FixedIncomeEngine:
    """Deterministic analytics for bonds, yield curves, and credit risk."""

    @staticmethod
    def analyze_bond(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        years_to_maturity: float,
        frequency: int = 2,
    ) -> BondAnalyticsResult:
        """
        Calculates bond price, Macaulay duration, modified duration, convexity, and DV01.

        Conventions (checked against QuantLib ``BondFunctions`` with ``Compounded`` yield at
        the same frequency, 30/360, valuation on a coupon date; see docs/validation):

        * ``coupon_rate`` and ``ytm`` are annual decimals (0.05 = 5%). ``ytm`` is a nominal
          annual yield compounded ``frequency`` times per year (bond-equivalent yield for
          ``frequency=2``). Coupon per period = ``face_value * coupon_rate / frequency``.
        * Valuation is on a coupon date: the bond has ``round(years_to_maturity * frequency)``
          whole periods left and no accrued interest, so clean = dirty price. A maturity that is
          not a whole number of periods is rounded; stubs and accrued interest are not modelled.
          A maturity that rounds to zero periods raises ``ValueError``.
        * Price and DV01 are in the units of ``face_value`` (per 100 par only when
          ``face_value=100``).
        * Macaulay and modified duration are in years; modified = Macaulay / (1 + ytm/frequency).
        * Convexity is annualised, (1/P) d2P/dy2 in years^2 (per-period convexity is this
          times ``frequency**2``).
        * DV01 = modified duration * price * 1e-4: the first-order, positive price change for a
          1bp move in ``ytm``. QuantLib's ``basisPointValue`` instead returns the signed
          second-order estimate of P(y+1bp) - P(y), which differs by 0.5 * C * P * 1e-8.
        """
        n_periods = int(round(years_to_maturity * frequency))
        if n_periods < 1:
            raise ValueError(
                f"years_to_maturity={years_to_maturity} with frequency={frequency} leaves no whole "
                "coupon period (the maturity is rounded to whole periods); use a maturity of more than half a "
                "coupon period, or a larger frequency."
            )
        coupon = (face_value * coupon_rate) / frequency
        period_yield = ytm / frequency

        price = 0.0
        weighted_time = 0.0
        convexity_acc = 0.0

        for t in range(1, n_periods + 1):
            cf = coupon + (face_value if t == n_periods else 0.0)
            df = (1.0 + period_yield) ** (-t)
            pv_cf = cf * df

            price += pv_cf
            time_years = t / frequency
            weighted_time += time_years * pv_cf
            convexity_acc += (t * (t + 1) * cf) / ((1.0 + period_yield) ** (t + 2))

        mac_dur = weighted_time / price
        mod_dur = mac_dur / (1.0 + period_yield)
        convexity = convexity_acc / (price * (frequency**2))
        dv01 = mod_dur * price * 0.0001

        return BondAnalyticsResult(
            clean_price=float(price),
            macaulay_duration=float(mac_dur),
            modified_duration=float(mod_dur),
            convexity=float(convexity),
            dv01=float(dv01),
        )

    @staticmethod
    def _nelson_siegel_func(
        t: np.ndarray,
        b0: float,
        b1: float,
        b2: float,
        lam: float,
    ) -> np.ndarray:
        # Avoid division by zero
        t_lam = np.maximum(t / max(lam, 1e-4), 1e-6)
        factor1 = (1.0 - np.exp(-t_lam)) / t_lam
        factor2 = factor1 - np.exp(-t_lam)
        return b0 + b1 * factor1 + b2 * factor2

    @staticmethod
    def fit_nelson_siegel(
        maturities: Sequence[float],
        yields: Sequence[float],
    ) -> NelsonSiegelCurveResult:
        """
        Fits the 4-parameter Nelson-Siegel model to observed term structure yields:
        y(t) = beta0 + beta1 * ((1 - e^(-t/lam))/(t/lam)) + beta2 * (((1 - e^(-t/lam))/(t/lam)) - e^(-t/lam))

        This is Nelson & Siegel (1987) eq. (2), a model of zero-coupon (spot) yields; fitting
        par or coupon-bond yields is an approximation.

        Units and conventions:

        * ``maturities`` in years (> 0); ``yields`` as decimals (0.045 = 4.5%). Percent inputs
          fall outside the parameter bounds and raise ``ValueError``.
        * ``lambda_decay`` is the time constant tau in years. QuantLib's
          ``NelsonSiegelFitting`` uses kappa = 1/tau; Diebold & Li (2006) use lambda = 1/tau
          in their own time unit (months).
        * Unweighted least squares in yield space (``scipy.optimize.curve_fit``) with bounds
          beta0 in [-0.1, 0.5], beta1 and beta2 in [-0.2, 0.2], tau in [0.1, 10] years. The
          starting point uses the first and last observations, so pass maturities in
          ascending order. The fit is non-convex in tau; no global search is done.
        """
        t = np.asarray(maturities, dtype=np.float64)
        y = np.asarray(yields, dtype=np.float64)

        if len(t) < 4:
            raise ValueError(f"Nelson-Siegel requires at least 4 maturity points, got {len(t)}.")

        # Initial parameter estimates
        b0_init = float(y[-1])
        b1_init = float(y[0] - y[-1])
        b2_init = 0.0
        lam_init = 2.0

        popt, _ = curve_fit(
            FixedIncomeEngine._nelson_siegel_func,
            t,
            y,
            p0=[b0_init, b1_init, b2_init, lam_init],
            bounds=([-0.1, -0.2, -0.2, 0.1], [0.5, 0.2, 0.2, 10.0]),
            maxfev=5000,
        )

        b0, b1, b2, lam = popt
        y_pred = FixedIncomeEngine._nelson_siegel_func(t, b0, b1, b2, lam)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return NelsonSiegelCurveResult(
            beta0_level=float(b0),
            beta1_slope=float(b1),
            beta2_curvature=float(b2),
            lambda_decay=float(lam),
            r_squared=float(r2),
        )

    @staticmethod
    def solve_merton_model(
        equity_value: float,
        equity_volatility: float,
        debt_face_value: float,
        risk_free_rate: float,
        maturity_years: float,
    ) -> MertonCreditResult:
        """
        Solves the Merton (1974) structural model for unobserved asset value (V_A)
        and asset volatility (sigma_A) using equity value and equity volatility.

        Equations (Hull, *Options, Futures, and Other Derivatives*, eqs. 24.3-24.4):

            E = V N(d1) - D e^{-rT} N(d2)
            sigma_E E = N(d1) sigma_V V

        Units and conventions:

        * ``equity_value`` (E, market value) and ``debt_face_value`` (D, a single zero-coupon
          debt payment due at T) in the same currency units; ``asset_value`` is returned in
          those units.
        * ``equity_volatility`` is annualised, decimal (0.40 = 40%); ``asset_volatility``
          likewise.
        * ``risk_free_rate`` is an annual, continuously compounded decimal rate.
        * ``maturity_years`` (T) is in years and is both the debt maturity and the default
          horizon.
        * ``distance_to_default`` is d2 under the risk-neutral measure (drift r, not the
          physical asset drift), and ``default_probability`` = N(-d2) = Q(V_T < D). These are
          not KMV/Moody's EDF-style real-world probabilities.
        * ``theoretical_credit_spread_bps`` = (-ln((V - E)/D)/T - r) * 1e4, continuously
          compounded, floored at 0.

        The system is reduced to one equation in sigma_V and solved by bracketed root finding:
        for each sigma_V the equity equation fixes V in (E, E + D e^{-rT}), and the volatility
        equation changes sign on [sigma_E E / (E + D e^{-rT}), sigma_E], so a root is always
        bracketed. Raises ``ValueError`` for non-positive inputs or if the solution fails to
        satisfy both equations to a relative 1e-8.
        """
        e = equity_value
        sig_e = equity_volatility
        d = debt_face_value
        r = risk_free_rate
        t = maturity_years

        if e <= 0 or sig_e <= 0 or d <= 0 or t <= 0:
            raise ValueError(
                "Merton model requires positive equity_value, equity_volatility, "
                "debt_face_value and maturity_years."
            )

        sqrt_t = math.sqrt(t)
        pv_debt = d * math.exp(-r * t)

        def ncdf(x: float) -> float:
            return 0.5 * math.erfc(-x / math.sqrt(2.0))

        def d1_d2(va: float, sig_a: float) -> tuple[float, float]:
            d1 = (math.log(va / d) + (r + 0.5 * sig_a**2) * t) / (sig_a * sqrt_t)
            return d1, d1 - sig_a * sqrt_t

        def equity_gap(va: float, sig_a: float) -> float:
            d1, d2 = d1_d2(va, sig_a)
            return va * ncdf(d1) - pv_debt * ncdf(d2) - e

        def asset_value_for(sig_a: float) -> float:
            # Call value is increasing in V, with V - D e^{-rT} <= C(V) < V, so the asset value
            # that reproduces E lies in (E, E + D e^{-rT}].
            lo, hi = e, e + pv_debt
            if equity_gap(hi, sig_a) <= 0.0:
                return hi
            return float(
                brentq(equity_gap, lo, hi, args=(sig_a,), xtol=1e-15 * hi, rtol=_RTOL, maxiter=500)
            )

        def vol_gap(sig_a: float) -> float:
            va = asset_value_for(sig_a)
            d1, _ = d1_d2(va, sig_a)
            return ncdf(d1) * sig_a * va - sig_e * e

        # vol_gap < 0 at the lower end (N(d1) < 1 and V <= E + D e^{-rT}) and
        # vol_gap = sigma_E D e^{-rT} N(d2) >= 0 at sigma_E.
        sig_lo = sig_e * e / (e + pv_debt)
        sig_hi = sig_e
        g_lo, g_hi = vol_gap(sig_lo), vol_gap(sig_hi)
        if g_lo >= 0.0:
            sig_a_sol = sig_lo
        elif g_hi <= 0.0:
            sig_a_sol = sig_hi
        else:
            sig_a_sol = float(
                brentq(vol_gap, sig_lo, sig_hi, xtol=1e-15 * sig_hi, rtol=_RTOL, maxiter=500)
            )
        va_sol = asset_value_for(sig_a_sol)

        res_equity = abs(equity_gap(va_sol, sig_a_sol)) / e
        res_vol = abs(vol_gap(sig_a_sol)) / (sig_e * e)
        if not (res_equity < 1e-8 and res_vol < 1e-8):
            raise ValueError(
                "Merton solver did not converge: relative residuals "
                f"{res_equity:.2e} (equity) and {res_vol:.2e} (volatility)."
            )

        # Distance to default
        d1 = (math.log(va_sol / d) + (r + 0.5 * sig_a_sol**2) * t) / (sig_a_sol * math.sqrt(t))
        d2 = d1 - sig_a_sol * math.sqrt(t)

        dd = float(d2)
        pd = float(norm.cdf(-dd))

        # Theoretical credit spread: y_D - r = -(1/T) * ln( (V_A/D)*norm.cdf(-d1) + e^(-rT)*norm.cdf(d2) ) - r
        # Debt value = V_A - Equity
        debt_val = va_sol - e
        implied_yield = -(1.0 / t) * math.log(debt_val / d) if debt_val > 0 else 0.0
        spread_bps = max(0.0, (implied_yield - r) * 10000.0)

        return MertonCreditResult(
            asset_value=va_sol,
            asset_volatility=sig_a_sol,
            distance_to_default=dd,
            default_probability=pd,
            theoretical_credit_spread_bps=spread_bps,
        )
