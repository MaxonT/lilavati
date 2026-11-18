"""Reference tests for ``lilavati.core.fixed_income`` (task 06 engine audit).

Expected values come from independent sources, never from the engine itself:

* QuantLib 1.44 (test-only dependency) for bond price, Macaulay/modified duration, convexity and
  basis-point value, and for the Nelson-Siegel zero-rate function (``NelsonSiegelFitting``).
* Hull, *Options, Futures, and Other Derivatives*, Merton-model worked example ("Example 24.3" in
  the 9th/10th editions, credit-risk chapter): E0=3, sigma_E=0.80, D=10, r=5%, T=1.
* Nelson & Siegel (1987), "Parsimonious Modeling of Yield Curves", J. Business 60(4), eq. (2):
  the spot-rate form and its limits y(0+) = beta0 + beta1, y(inf) = beta0.
* Diebold & Li (2006), "Forecasting the term structure of government bond yields", J. Econometrics
  130, section 2.1: lambda = 0.0609 (per month) puts the maximum of the curvature loading at
  ~30 months.
* Defining equations of the Merton (1974) model (Hull eqs. 24.3-24.4), checked as residuals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from lilavati.core.fixed_income import FixedIncomeEngine

ql = pytest.importorskip("QuantLib")

_QL_FREQ = {1: "Annual", 2: "Semiannual", 4: "Quarterly", 12: "Monthly"}


def _ql_bond_metrics(
    coupon_rate: float, ytm: float, years: int, frequency: int
) -> dict[str, float]:
    """Price a bullet bond in QuantLib on a coupon date with matching conventions.

    Conventions chosen so that QuantLib's year fractions equal k / frequency exactly:
    30/360 Bond Basis day count, unadjusted dates, settlement on the issue date (accrued = 0),
    yield compounded ``frequency`` times per year (``ql.Compounded``).
    """
    freq = getattr(ql, _QL_FREQ[frequency])
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    maturity = today + ql.Period(years, ql.Years)
    schedule = ql.Schedule(
        today,
        maturity,
        ql.Period(freq),
        ql.NullCalendar(),
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Backward,
        False,
    )
    dc = ql.Thirty360(ql.Thirty360.BondBasis)
    bond = ql.FixedRateBond(0, 100.0, schedule, [coupon_rate], dc)
    rate = ql.InterestRate(ytm, dc, ql.Compounded, freq)
    bf = ql.BondFunctions
    return {
        "price": bf.cleanPrice(bond, rate, today),
        "accrued": bf.accruedAmount(bond, today),
        "macaulay": bf.duration(bond, rate, ql.Duration.Macaulay, today),
        "modified": bf.duration(bond, rate, ql.Duration.Modified, today),
        "convexity": bf.convexity(bond, rate, today),
        "bpv": bf.basisPointValue(bond, rate, today),
        "price_up": bf.cleanPrice(
            bond, ql.InterestRate(ytm + 1e-4, dc, ql.Compounded, freq), today
        ),
        "price_dn": bf.cleanPrice(
            bond, ql.InterestRate(ytm - 1e-4, dc, ql.Compounded, freq), today
        ),
    }


# --------------------------------------------------------------------------------------------
# analyze_bond vs QuantLib
# --------------------------------------------------------------------------------------------

BOND_CASES = [
    # (coupon_rate, ytm, years, frequency)
    (0.05, 0.05, 5, 1),  # par bond, annual
    (0.06, 0.08, 10, 2),  # discount bond, semiannual
    (0.03, 0.045, 30, 2),  # long discount bond
    (0.00, 0.04, 7, 4),  # zero-coupon, quarterly compounding
    (0.07, 0.02, 3, 12),  # premium bond, monthly
    (0.10, 0.10, 20, 2),  # high-coupon par bond
]


@pytest.mark.parametrize(("coupon", "ytm", "years", "freq"), BOND_CASES)
def test_bond_matches_quantlib(coupon: float, ytm: float, years: int, freq: int) -> None:
    # Source: QuantLib 1.44 BondFunctions (cleanPrice, duration Macaulay/Modified, convexity).
    q = _ql_bond_metrics(coupon, ytm, years, freq)
    assert q["accrued"] == 0.0  # valuation on a coupon date: clean == dirty
    r = FixedIncomeEngine.analyze_bond(100.0, coupon, ytm, float(years), freq)
    assert r.clean_price == pytest.approx(q["price"], rel=1e-12)
    assert r.macaulay_duration == pytest.approx(q["macaulay"], rel=1e-12)
    assert r.modified_duration == pytest.approx(q["modified"], rel=1e-12)
    # Convexity is annualised, (1/P) d2P/dy2 in years^2, same as QuantLib's for Compounded rates.
    assert r.convexity == pytest.approx(q["convexity"], rel=1e-12)


@pytest.mark.parametrize(("coupon", "ytm", "years", "freq"), BOND_CASES)
def test_dv01_vs_quantlib_and_finite_difference(
    coupon: float, ytm: float, years: int, freq: int
) -> None:
    q = _ql_bond_metrics(coupon, ytm, years, freq)
    r = FixedIncomeEngine.analyze_bond(100.0, coupon, ytm, float(years), freq)
    # Central finite difference of QuantLib prices: DV01 = (P(y-1bp) - P(y+1bp)) / 2.
    # The central difference carries an O(h^2) relative error (third-derivative term, h = 1bp);
    # measured worst case over these bonds is 1.1e-6 (30y 3% bond at 4.5%).
    fd = (q["price_dn"] - q["price_up"]) / 2.0
    assert r.dv01 == pytest.approx(fd, rel=2e-6)
    # QuantLib's basisPointValue is the signed second-order Taylor estimate of P(y+1bp) - P(y):
    # bpv = -D_mod*P*h + 0.5*C*P*h^2. Our DV01 is the first-order, sign-positive term only.
    h = 1e-4
    assert r.dv01 == pytest.approx(-q["bpv"] + 0.5 * q["convexity"] * q["price"] * h * h, rel=1e-10)


def test_bond_face_value_scales_linearly() -> None:
    # Price and DV01 are in face_value units (per 100 par only when face_value == 100);
    # durations and convexity are scale-free.
    a = FixedIncomeEngine.analyze_bond(100.0, 0.06, 0.07, 8.0, 2)
    b = FixedIncomeEngine.analyze_bond(1000.0, 0.06, 0.07, 8.0, 2)
    assert b.clean_price == pytest.approx(10 * a.clean_price, rel=1e-13)
    assert b.dv01 == pytest.approx(10 * a.dv01, rel=1e-13)
    assert b.macaulay_duration == pytest.approx(a.macaulay_duration, rel=1e-13)
    assert b.convexity == pytest.approx(a.convexity, rel=1e-13)


# --------------------------------------------------------------------------------------------
# analyze_bond: textbook closed forms and properties
# --------------------------------------------------------------------------------------------


def test_zero_coupon_closed_forms() -> None:
    # Fabozzi, Bond Markets, Analysis and Strategies (duration/convexity chapter): for a zero
    # maturing in n periods, Macaulay duration = maturity, modified = T / (1 + y/f),
    # convexity = n(n+1) / (f^2 (1 + y/f)^2).
    f, y, years = 2, 0.06, 10.0
    n = int(years * f)
    r = FixedIncomeEngine.analyze_bond(100.0, 0.0, y, years, f)
    assert r.clean_price == pytest.approx(100.0 / (1 + y / f) ** n, rel=1e-14)
    assert r.macaulay_duration == pytest.approx(years, rel=1e-14)
    assert r.modified_duration == pytest.approx(years / (1 + y / f), rel=1e-14)
    assert r.convexity == pytest.approx(n * (n + 1) / (f**2 * (1 + y / f) ** 2), rel=1e-13)


def test_par_bond_prices_at_par_and_perpetuity_limit() -> None:
    # Coupon == yield prices at par for every frequency.
    for f in (1, 2, 4, 12):
        assert FixedIncomeEngine.analyze_bond(100.0, 0.045, 0.045, 12.0, f).clean_price == (
            pytest.approx(100.0, rel=1e-12)
        )
    # Long maturity: Macaulay duration of a par bond tends to (1 + y/f) / y (perpetuity limit,
    # Fabozzi). At 300 years the gap is below 1e-6 years.
    r = FixedIncomeEngine.analyze_bond(100.0, 0.05, 0.05, 300.0, 1)
    assert r.macaulay_duration == pytest.approx(1.05 / 0.05, abs=1e-5)


def test_price_yield_monotone_and_convex() -> None:
    ys = np.linspace(0.001, 0.20, 80)
    prices = np.array(
        [FixedIncomeEngine.analyze_bond(100.0, 0.05, y, 15.0, 2).clean_price for y in ys]
    )
    assert np.all(np.diff(prices) < 0)  # strictly decreasing in yield
    assert np.all(np.diff(prices, 2) > 0)  # convex in yield


def test_duration_convexity_second_order_taylor() -> None:
    # dP/P ~= -D_mod dy + 0.5 C dy^2 for a 50bp move; error is O(dy^3).
    base = FixedIncomeEngine.analyze_bond(100.0, 0.04, 0.05, 20.0, 2)
    dy = 0.005
    shocked = FixedIncomeEngine.analyze_bond(100.0, 0.04, 0.05 + dy, 20.0, 2).clean_price
    approx = base.clean_price * (1 - base.modified_duration * dy + 0.5 * base.convexity * dy**2)
    assert abs(shocked - approx) / base.clean_price < 2e-4
    # Second-order term matters: first-order alone is worse.
    first = base.clean_price * (1 - base.modified_duration * dy)
    assert abs(shocked - first) > abs(shocked - approx)


def test_duration_bounds() -> None:
    # Macaulay duration of a coupon bond is positive and below its maturity; modified < Macaulay.
    for c, y, t, f in [(0.02, 0.08, 30.0, 2), (0.12, 0.03, 5.0, 1), (0.05, 0.05, 1.0, 4)]:
        r = FixedIncomeEngine.analyze_bond(100.0, c, y, t, f)
        assert 0.0 < r.modified_duration < r.macaulay_duration < t


# --------------------------------------------------------------------------------------------
# Nelson-Siegel
# --------------------------------------------------------------------------------------------

NS_MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]


def _ql_ns_zero_rates(params: tuple[float, float, float, float], ts: list[float]) -> list[float]:
    """Continuously-compounded zero rates from QuantLib's NelsonSiegelFitting.

    QuantLib parameterises the decay as kappa = 1 / tau, so pass 1 / lambda_decay.
    """
    b0, b1, b2, tau = params
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    curve = ql.FittedBondDiscountCurve(
        today,
        ql.NelsonSiegelFitting(),
        ql.Array([b0, b1, b2, 1.0 / tau]),
        today + ql.Period(40, ql.Years),
        ql.Actual365Fixed(),
    )
    return [curve.zeroRate(t, ql.Continuous).rate() for t in ts]


NS_PARAMS = [
    # (beta0, beta1, beta2, tau_years)
    (0.05, -0.02, 0.01, 2.0),  # upward sloping
    (0.04, 0.015, -0.01, 1.0),  # inverted
    (0.06, -0.03, 0.04, 3.0),  # humped
    (0.03, -0.01, -0.02, 0.7),  # dip
]


@pytest.mark.parametrize("params", NS_PARAMS)
def test_nelson_siegel_recovers_parameters_from_quantlib_curve(
    params: tuple[float, float, float, float],
) -> None:
    # Synthetic zero curve generated by QuantLib's independent Nelson-Siegel implementation,
    # then fitted by the engine. Nelson & Siegel (1987) eq. (2) is the common functional form.
    ys = _ql_ns_zero_rates(params, NS_MATURITIES)
    r = FixedIncomeEngine.fit_nelson_siegel(NS_MATURITIES, ys)
    b0, b1, b2, tau = params
    assert r.beta0_level == pytest.approx(b0, abs=1e-6)
    assert r.beta1_slope == pytest.approx(b1, abs=1e-6)
    assert r.beta2_curvature == pytest.approx(b2, abs=2e-5)
    assert r.lambda_decay == pytest.approx(tau, rel=1e-4)
    assert r.r_squared > 1 - 1e-12


def test_nelson_siegel_functional_form_matches_quantlib() -> None:
    # The engine's model curve, evaluated at the true parameters, equals QuantLib's zero rates.
    for params in NS_PARAMS:
        ts = [0.1, 0.5, 1.0, 2.5, 4.0, 8.0, 15.0, 30.0]
        ours = FixedIncomeEngine._nelson_siegel_func(np.array(ts), *params)
        np.testing.assert_allclose(ours, _ql_ns_zero_rates(params, ts), rtol=0, atol=1e-12)


def test_nelson_siegel_limits() -> None:
    # Nelson & Siegel (1987): y(m -> 0+) = beta0 + beta1 (instantaneous short rate),
    # y(m -> inf) = beta0 (long-run level). The engine floors m/tau at 1e-6 to avoid 0/0, so the
    # short end is exact only to O(1e-6 * |beta|).
    b0, b1, b2, tau = 0.05, -0.02, 0.01, 2.0
    f = FixedIncomeEngine._nelson_siegel_func
    assert float(f(np.array([1e-9]), b0, b1, b2, tau)[0]) == pytest.approx(b0 + b1, abs=1e-7)
    assert float(f(np.array([1e5]), b0, b1, b2, tau)[0]) == pytest.approx(b0, abs=1e-6)


def test_nelson_siegel_curvature_loading_peak_diebold_li() -> None:
    # Diebold & Li (2006): lambda = 0.0609 per month maximises the curvature loading at about
    # 30 months. In this engine's parameterisation tau (years) = 1 / (0.0609 * 12).
    tau = 1.0 / (0.0609 * 12.0)
    t = np.linspace(0.05, 10.0, 200_000)
    loading = FixedIncomeEngine._nelson_siegel_func(t, 0.0, 0.0, 1.0, tau)
    peak_months = float(t[np.argmax(loading)] * 12.0)
    assert peak_months == pytest.approx(30.0, abs=1.0)  # measured: 29.45 months


def test_nelson_siegel_noisy_fit_and_order_invariance() -> None:
    rng = np.random.default_rng(7)
    params = (0.045, -0.015, 0.02, 1.5)
    clean = np.array(_ql_ns_zero_rates(params, NS_MATURITIES))
    noisy = clean + rng.normal(0.0, 2e-4, size=clean.size)  # 2bp noise
    r = FixedIncomeEngine.fit_nelson_siegel(NS_MATURITIES, noisy.tolist())
    assert r.r_squared > 0.98
    assert r.beta0_level == pytest.approx(params[0], abs=2e-3)
    # Same data in reverse order fits the same curve.
    r_rev = FixedIncomeEngine.fit_nelson_siegel(NS_MATURITIES[::-1], noisy[::-1].tolist())
    for a, b in [
        (r.beta0_level, r_rev.beta0_level),
        (r.beta1_slope, r_rev.beta1_slope),
        (r.beta2_curvature, r_rev.beta2_curvature),
    ]:
        assert a == pytest.approx(b, abs=1e-4)


def test_nelson_siegel_rejects_percent_inputs() -> None:
    # Yields must be decimals (0.045 = 4.5%); percent inputs fall outside the parameter bounds.
    with pytest.raises(ValueError):
        FixedIncomeEngine.fit_nelson_siegel(NS_MATURITIES, [4.5] * len(NS_MATURITIES))


# --------------------------------------------------------------------------------------------
# Merton structural model
# --------------------------------------------------------------------------------------------


def _merton_residuals(
    e: float, sig_e: float, d: float, r: float, t: float, va: float, sig_a: float
) -> tuple[float, float]:
    """Relative residuals of the two Merton equations (Hull eqs. 24.3 and 24.4)."""
    d1 = (math.log(va / d) + (r + 0.5 * sig_a**2) * t) / (sig_a * math.sqrt(t))
    d2 = d1 - sig_a * math.sqrt(t)
    eq1 = va * norm.cdf(d1) - d * math.exp(-r * t) * norm.cdf(d2) - e
    eq2 = norm.cdf(d1) * sig_a * va - sig_e * e
    return abs(eq1) / e, abs(eq2) / (sig_e * e)


def test_merton_hull_worked_example() -> None:
    # Hull, Options, Futures, and Other Derivatives (9th/10th ed.), Example 24.3:
    # E0 = 3, sigma_E = 0.80, D = 10, r = 0.05, T = 1  =>  V0 = 12.40, sigma_V = 0.2123,
    # d2 = 1.1408, N(-d2) = 12.7%, market value of debt V0 - E0 = 9.40,
    # PV of promised payment 10 e^{-0.05} = 9.51, expected loss ~1.2%.
    r = FixedIncomeEngine.solve_merton_model(3.0, 0.80, 10.0, 0.05, 1.0)
    assert r.asset_value == pytest.approx(12.40, abs=0.005)
    assert r.asset_volatility == pytest.approx(0.2123, abs=5e-5)
    assert r.distance_to_default == pytest.approx(1.1408, abs=5e-5)
    assert r.default_probability == pytest.approx(0.127, abs=5e-4)
    debt = r.asset_value - 3.0
    assert debt == pytest.approx(9.40, abs=0.005)
    pv_promised = 10.0 * math.exp(-0.05)
    assert (pv_promised - debt) / pv_promised == pytest.approx(0.012, abs=5e-4)
    # Spread implied by Hull's numbers: -ln(9.40/10) - 0.05 ~ 119 bp at Hull's 2-dp rounding;
    # computed from the unrounded debt value it is 123.66 bp.
    assert r.theoretical_credit_spread_bps == pytest.approx(
        (-math.log(debt / 10.0) - 0.05) * 1e4, rel=1e-12
    )


MERTON_STRESS_CASES = [
    # (E, sigma_E, D, r, T): deep-distress / long-horizon inputs. Before the fix the hybr root
    # finder stopped at non-solutions here and the engine returned them silently.
    (0.1, 2.0, 100.0, 0.03, 5.0),
    (0.1, 2.0, 100.0, 0.0, 5.0),
    (0.1, 3.0, 100.0, 0.03, 10.0),
    (0.1, 3.0, 1000.0, 0.03, 5.0),
    (0.1, 0.05, 1000.0, 0.03, 1.0),
]


@pytest.mark.parametrize("case", MERTON_STRESS_CASES)
def test_merton_solution_satisfies_defining_equations(
    case: tuple[float, float, float, float, float],
) -> None:
    # Reference: the two Merton (1974) equations themselves (Hull eqs. 24.3-24.4). Whatever the
    # solver does internally, the returned (V, sigma_V) must satisfy both.
    e, sig_e, d, rf, t = case
    r = FixedIncomeEngine.solve_merton_model(e, sig_e, d, rf, t)
    res1, res2 = _merton_residuals(e, sig_e, d, rf, t, r.asset_value, r.asset_volatility)
    assert res1 < 1e-8
    assert res2 < 1e-8


@pytest.mark.parametrize(
    "case",
    [
        (3.0, 0.80, 10.0, 0.05, 1.0),
        (100.0, 0.40, 80.0, 0.05, 1.0),
        (40.0, 0.60, 100.0, 0.04, 2.0),
        (10.0, 0.50, 100.0, 0.05, 0.25),
        (1.0, 1.50, 100.0, 0.05, 1.0),
    ],
)
def test_merton_consistency_with_quantlib_black_formula(
    case: tuple[float, float, float, float, float],
) -> None:
    # Equity as a call on assets (Merton 1974): re-price the equity with QuantLib's independent
    # BlackCalculator using the engine's implied V and sigma_V; delta gives the volatility link.
    e, sig_e, d, rf, t = case
    r = FixedIncomeEngine.solve_merton_model(e, sig_e, d, rf, t)
    disc = math.exp(-rf * t)
    fwd = r.asset_value / disc
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, d)
    calc = ql.BlackCalculator(payoff, fwd, r.asset_volatility * math.sqrt(t), disc)
    assert calc.value() == pytest.approx(e, rel=1e-8)
    delta = calc.delta(r.asset_value)
    assert delta * r.asset_volatility * r.asset_value == pytest.approx(sig_e * e, rel=1e-8)
    # PD definition: N(-d2) under the risk-neutral measure, d2 = distance_to_default.
    assert r.default_probability == pytest.approx(norm.cdf(-r.distance_to_default), rel=1e-12)
    # Asset volatility is strictly below equity volatility (leverage) and V > E.
    assert r.asset_volatility < sig_e
    assert r.asset_value > e


def test_merton_limits() -> None:
    # Negligible debt: V -> E + D e^{-rT}, sigma_V -> sigma_E * E / V, PD -> 0, spread -> 0.
    r = FixedIncomeEngine.solve_merton_model(1000.0, 0.2, 1.0, 0.05, 1.0)
    assert r.asset_value == pytest.approx(1000.0 + math.exp(-0.05), rel=1e-12)
    assert r.asset_volatility == pytest.approx(0.2 * 1000.0 / r.asset_value, rel=1e-12)
    assert r.default_probability < 1e-100
    assert r.theoretical_credit_spread_bps == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "case",
    [
        (0.0, 0.4, 80.0, 0.05, 1.0),
        (100.0, 0.0, 80.0, 0.05, 1.0),
        (100.0, 0.4, 0.0, 0.05, 1.0),
        (100.0, 0.4, 80.0, 0.05, 0.0),
    ],
)
def test_merton_rejects_non_positive_inputs(
    case: tuple[float, float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        FixedIncomeEngine.solve_merton_model(*case)


def test_merton_pd_monotone_in_leverage_and_volatility() -> None:
    pds_lev = [
        FixedIncomeEngine.solve_merton_model(100.0, 0.5, d, 0.04, 1.0).default_probability
        for d in (20.0, 50.0, 100.0, 200.0, 400.0)
    ]
    assert all(a < b for a, b in zip(pds_lev, pds_lev[1:], strict=False))
    pds_vol = [
        FixedIncomeEngine.solve_merton_model(100.0, s, 150.0, 0.04, 1.0).default_probability
        for s in (0.2, 0.4, 0.6, 0.8)
    ]
    assert all(a < b for a, b in zip(pds_vol, pds_vol[1:], strict=False))
