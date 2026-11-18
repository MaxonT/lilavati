"""Reference-value tests for ``lilavati.core.derivatives`` (task 06).

Expected values come from independent sources, never from the implementation itself:

* QuantLib 1.44 (test-only dependency): ``AnalyticEuropeanEngine`` (price and Greeks with dividend
  yield), ``BinomialVanillaEngine("crr")`` (American options), ``VanillaOption.impliedVolatility``,
  ``ql.sabrVolatility`` (Hagan lognormal SABR) and ``ql.blackFormula``.
* J. C. Hull, *Options, Futures, and Other Derivatives* (10th ed.): worked examples identified by
  their inputs (chapter numbers differ between editions).
* P. Hagan, D. Kumar, A. Lesniewski, D. Woodward, "Managing Smile Risk", Wilmott Magazine (2002),
  pp. 84-108: eq. (2.17a) general lognormal SABR vol and eq. (2.18) at-the-money limit.
* Analytical identities / no-arbitrage properties (put-call parity, finite differences, bounds).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from lilavati.core.derivatives import DerivativesEngine as E

ql = pytest.importorskip("QuantLib")

# --------------------------------------------------------------------------------------------------
# QuantLib helpers. Maturities are whole days on ACT/365F so that T = days / 365 exactly.
# --------------------------------------------------------------------------------------------------
TODAY = ql.Date(15, 1, 2025)
DC = ql.Actual365Fixed()


def _process(s: float, r: float, q: float, v: float) -> tuple[Any, Any, Any]:
    ql.Settings.instance().evaluationDate = TODAY
    spot = ql.SimpleQuote(s)
    vol = ql.SimpleQuote(v)
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DC)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DC)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(TODAY, ql.NullCalendar(), ql.QuoteHandle(vol), DC)
        ),
    )
    return proc, spot, vol


def _ql_european(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool
) -> tuple[Any, Any, Any]:
    proc, spot, vol = _process(s, r, q, v)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, k)
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(TODAY + days))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return opt, proc, vol


def _ql_american_crr(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool, steps: int
) -> Any:
    proc, _, _ = _process(s, r, q, v)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, k)
    opt = ql.VanillaOption(payoff, ql.AmericanExercise(TODAY, TODAY + days))
    opt.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", steps))
    return opt


# (spot, strike, days, r, q, sigma) — ITM/ATM/OTM, short and long, with and without dividend yield.
EURO_CASES = [
    (100.0, 95.0, 365, 0.05, 0.03, 0.25),
    (100.0, 100.0, 365, 0.05, 0.00, 0.20),
    (100.0, 130.0, 90, 0.02, 0.01, 0.35),
    (50.0, 40.0, 730, 0.08, 0.04, 0.15),
    (2500.0, 2600.0, 182, 0.04, 0.015, 0.18),
]


# --------------------------------------------------------------------------------------------------
# BSM price and Greeks vs QuantLib AnalyticEuropeanEngine
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), EURO_CASES)
def test_bsm_price_and_greeks_match_quantlib(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool
) -> None:
    """Source: QuantLib AnalyticEuropeanEngine. Unit mapping (see calculate_bsm_greeks docstring):
    vega = ql.vega()/100, theta = ql.thetaPerDay() = ql.theta()/365, rho = ql.rho()/100."""
    t = days / 365.0
    opt, _, vol = _ql_european(s, k, days, r, q, v, call)
    res = E.price_european_option(s, k, t, r, v, q, call)
    g = res.greeks
    assert res.price == pytest.approx(opt.NPV(), rel=1e-12, abs=1e-12)
    assert g.delta == pytest.approx(opt.delta(), rel=1e-12, abs=1e-14)
    assert g.gamma == pytest.approx(opt.gamma(), rel=1e-12, abs=1e-14)
    assert g.vega == pytest.approx(opt.vega() / 100.0, rel=1e-12, abs=1e-14)
    assert g.theta == pytest.approx(opt.thetaPerDay(), rel=1e-10, abs=1e-14)
    assert g.theta == pytest.approx(opt.theta() / 365.0, rel=1e-10, abs=1e-14)
    assert g.rho == pytest.approx(opt.rho() / 100.0, rel=1e-12, abs=1e-14)

    # Vanna and volga are not provided by QuantLib: central bumps of QuantLib's own delta and vega
    # in the vol quote (h = 1e-5) give an independent reference.
    h = 1e-5
    vol.setValue(v + h)
    delta_up, vega_up = opt.delta(), opt.vega()
    vol.setValue(v - h)
    delta_dn, vega_dn = opt.delta(), opt.vega()
    vol.setValue(v)
    assert g.vanna == pytest.approx((delta_up - delta_dn) / (2 * h), rel=1e-6, abs=1e-8)
    # volga = d(vega per vol point)/dsigma = d(ql.vega()/100)/dsigma
    assert g.volga == pytest.approx((vega_up - vega_dn) / (2 * h) / 100.0, rel=1e-6, abs=1e-8)


def test_benchmark_vega_convention_is_per_vol_point() -> None:
    """Bench case LILAVATI-T5-001 expects vega 0.375 for S=K=100, r=5%, sigma=20%, T=1.
    QuantLib vega() (per 1.0 of vol) is 37.524; the engine reports per vol point (x 0.01)."""
    opt, _, _ = _ql_european(100.0, 100.0, 365, 0.05, 0.0, 0.20, True)
    g = E.calculate_bsm_greeks(100.0, 100.0, 1.0, 0.05, 0.20)
    assert opt.vega() == pytest.approx(37.524, abs=1e-3)
    assert g.vega == pytest.approx(opt.vega() / 100.0, rel=1e-12)
    assert g.vega == pytest.approx(0.375, abs=0.05 * 0.375)  # benchmark tolerance (5%)
    p = E.price_european_option(100.0, 100.0, 1.0, 0.05, 0.20)
    assert p.price == pytest.approx(10.4506, abs=1e-4)
    assert g.delta == pytest.approx(0.6368, abs=1e-4)
    assert g.gamma == pytest.approx(0.01876, abs=1e-5)


# --------------------------------------------------------------------------------------------------
# Hull textbook worked examples
# --------------------------------------------------------------------------------------------------
def test_hull_bsm_example_non_dividend() -> None:
    """Hull, OFOD 10th ed., Example 15.6: S0=42, K=40, r=10%, sigma=20%, T=0.5 -> c=4.76, p=0.81."""
    c = E.price_european_option(42.0, 40.0, 0.5, 0.10, 0.20, is_call=True).price
    p = E.price_european_option(42.0, 40.0, 0.5, 0.10, 0.20, is_call=False).price
    assert c == pytest.approx(4.76, abs=5e-3)
    assert p == pytest.approx(0.81, abs=5e-3)


def test_hull_index_option_with_dividend_yield() -> None:
    """Hull, OFOD 10th ed., Example 17.1: European index call, S0=930, K=900, r=8%, q=3%,
    sigma=20%, T=2/12 -> c = 51.83."""
    c = E.price_european_option(930.0, 900.0, 2.0 / 12.0, 0.08, 0.20, dividend_yield=0.03).price
    assert c == pytest.approx(51.83, abs=5e-3)


def test_hull_greek_letters_example() -> None:
    """Hull, OFOD 10th ed., ch. 19 running example: S0=49, K=50, r=5%, sigma=20%, T=20 weeks
    (0.3846). Hull reports delta=0.522, gamma=0.066, vega=12.1 (per 1.0 of vol, i.e. 0.121 per
    vol point), theta=-4.31 per year (-0.0118 per calendar day), rho=8.91 (per 1.0, i.e. 0.0891
    per 1%)."""
    g = E.calculate_bsm_greeks(49.0, 50.0, 0.3846, 0.05, 0.20)
    assert g.delta == pytest.approx(0.522, abs=5e-4)
    assert g.gamma == pytest.approx(0.066, abs=5e-4)
    assert g.vega * 100.0 == pytest.approx(12.1, abs=0.05)
    assert g.theta * 365.0 == pytest.approx(-4.31, abs=5e-3)
    assert g.theta == pytest.approx(-0.0118, abs=5e-5)
    assert g.rho * 100.0 == pytest.approx(8.91, abs=5e-3)


@pytest.mark.parametrize(("steps", "hull"), [(5, 4.49), (30, 4.263), (50, 4.272), (100, 4.278), (500, 4.283)])
def test_hull_american_put_binomial(steps: int, hull: float) -> None:
    """Hull, OFOD 10th ed., Example 21.1 / Fig. 21.3 and the convergence values quoted in §21.1:
    American put S0=50, K=50, r=10%, sigma=40%, T=5 months. 5 steps -> 4.49; 30, 50, 100, 500
    steps -> 4.263, 4.272, 4.278, 4.283."""
    p = E.price_american_option_crr(50.0, 50.0, 5.0 / 12.0, 0.10, 0.40, is_call=False, steps=steps)
    decimals = 2 if steps == 5 else 3
    assert p.price == pytest.approx(hull, abs=0.5 * 10.0**-decimals)


# --------------------------------------------------------------------------------------------------
# CRR American vs QuantLib BinomialVanillaEngine("crr")
# --------------------------------------------------------------------------------------------------
AMER_CASES = [
    (50.0, 50.0, 152, 0.10, 0.00, 0.40, False),
    (100.0, 110.0, 365, 0.05, 0.00, 0.30, False),
    (100.0, 90.0, 365, 0.03, 0.06, 0.25, True),
    (100.0, 100.0, 730, 0.05, 0.02, 0.20, False),
]


@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v", "call"), AMER_CASES)
def test_crr_american_matches_quantlib(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool
) -> None:
    """Source: QuantLib BinomialVanillaEngine('crr', 1000). QuantLib's CRR tree uses the
    drift-matched probability 1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma) instead of the exact
    (e^{(r-q)dt} - d)/(u - d), so the two trees differ by O(1/N); measured max |diff| 3.6e-5."""
    mine = E.price_american_option_crr(s, k, days / 365.0, r, v, q, call, steps=1000).price
    ref = _ql_american_crr(s, k, days, r, q, v, call, 1000).NPV()
    assert mine == pytest.approx(ref, abs=1e-4)


def test_crr_european_limit_converges_to_bsm() -> None:
    """Merton (1973) / Hull §11.5: an American call on a non-dividend stock is never exercised
    early, so CRR must converge to BSM. Error shrinks ~ 1/N (measured ~ -2/N for this ATM case)."""
    bsm = E.price_european_option(100.0, 100.0, 1.0, 0.05, 0.20).price
    errs = [
        abs(E.price_american_option_crr(100.0, 100.0, 1.0, 0.05, 0.20, steps=n).price - bsm)
        for n in (50, 200, 1000)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 2.5e-3
    assert errs[0] / errs[2] == pytest.approx(20.0, rel=0.05)  # first-order convergence


def _crr_european_tree(s: float, k: float, t: float, r: float, q: float, v: float, call: bool, n: int) -> float:
    """Independent numpy CRR tree *without* early exercise (Cox, Ross, Rubinstein 1979)."""
    dt = t / n
    u = math.exp(v * math.sqrt(dt))
    p = (math.exp((r - q) * dt) - 1 / u) / (u - 1 / u)
    j = np.arange(n + 1)
    st = s * u ** (2.0 * j - n)
    vals = np.maximum(st - k, 0.0) if call else np.maximum(k - st, 0.0)
    for _ in range(n):
        vals = math.exp(-r * dt) * (p * vals[1:] + (1 - p) * vals[:-1])
    return float(vals[0])


@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), EURO_CASES)
@pytest.mark.parametrize("call", [True, False])
def test_american_at_least_european_and_intrinsic(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool
) -> None:
    """No-arbitrage bound: American >= max(European, intrinsic). Compared with the same-N European
    tree (exact inequality) and with BSM up to that tree's own discretisation error."""
    t = days / 365.0
    n = 400
    amer = E.price_american_option_crr(s, k, t, r, v, q, call, steps=n).price
    euro_tree = _crr_european_tree(s, k, t, r, q, v, call, n)
    euro = E.price_european_option(s, k, t, r, v, q, call).price
    intrinsic = max(0.0, (s - k) if call else (k - s))
    assert amer >= intrinsic - 1e-12
    assert amer >= euro_tree - 1e-10
    assert amer >= euro - abs(euro_tree - euro) - 1e-10


def test_crr_rejects_invalid_probability() -> None:
    """Bug fix: with 1 step, sigma=1%, r=5% the CRR up-probability is 3.06 (> 1). Before the fix the
    tree silently returned 2.926 for an option whose BSM/American value is 4.877."""
    with pytest.raises(ValueError, match="probability"):
        E.price_american_option_crr(100.0, 100.0, 1.0, 0.05, 0.01, steps=1)


def test_european_rejects_non_positive_maturity() -> None:
    """Bug fix: T=0 raised ZeroDivisionError before validation; now a ValueError."""
    with pytest.raises(ValueError):
        E.price_european_option(100.0, 100.0, 0.0, 0.05, 0.2)


# --------------------------------------------------------------------------------------------------
# Property tests: put-call parity, finite differences
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), EURO_CASES)
def test_put_call_parity_with_dividends(
    s: float, k: float, days: int, r: float, q: float, v: float
) -> None:
    """Hull (put-call parity with yield q): C - P = S e^{-qT} - K e^{-rT}."""
    t = days / 365.0
    c = E.price_european_option(s, k, t, r, v, q, True)
    p = E.price_european_option(s, k, t, r, v, q, False)
    assert c.price - p.price == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-10 * s)
    # Greek parity: delta_C - delta_P = e^{-qT}, rho_C - rho_P = 0.01 K T e^{-rT}
    assert c.greeks.delta - p.greeks.delta == pytest.approx(math.exp(-q * t), abs=1e-12)
    assert c.greeks.rho - p.greeks.rho == pytest.approx(0.01 * k * t * math.exp(-r * t), abs=1e-12)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), EURO_CASES)
def test_greeks_match_finite_differences(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool
) -> None:
    """Central finite differences of the BSM price (independent of the analytic Greek code)."""
    t = days / 365.0

    def price(**kw: float) -> float:
        a = {"spot": s, "time_to_maturity": t, "risk_free_rate": r, "volatility": v} | kw
        return E.price_european_option(
            a["spot"], k, a["time_to_maturity"], a["risk_free_rate"], a["volatility"], q, call
        ).price

    def delta(**kw: float) -> float:
        hs = 1e-4 * s
        return (price(**({"spot": s + hs} | kw)) - price(**({"spot": s - hs} | kw))) / (2 * hs)

    g = E.calculate_bsm_greeks(s, k, t, r, v, q, call)
    hs, hv, hr, ht = 1e-4 * s, 1e-5, 1e-5, 1e-6

    def cd(f: Callable[[float], float], x: float, h: float) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)

    assert g.delta == pytest.approx(cd(lambda x: price(spot=x), s, hs), rel=1e-6, abs=1e-8)
    gamma_fd = (price(spot=s + hs) - 2 * price() + price(spot=s - hs)) / hs**2
    assert g.gamma == pytest.approx(gamma_fd, rel=1e-4, abs=1e-7)
    assert g.vega == pytest.approx(0.01 * cd(lambda x: price(volatility=x), v, hv), rel=1e-6, abs=1e-9)
    assert g.rho == pytest.approx(0.01 * cd(lambda x: price(risk_free_rate=x), r, hr), rel=1e-6, abs=1e-9)
    # theta per calendar day = -(dV/dT) / 365
    assert g.theta == pytest.approx(-cd(lambda x: price(time_to_maturity=x), t, ht) / 365.0, rel=1e-5, abs=1e-9)
    assert g.vanna == pytest.approx(cd(lambda x: delta(volatility=x), v, 1e-4), rel=1e-4, abs=1e-7)
    volga_fd = 0.01 * (price(volatility=v + 1e-4) - 2 * price() + price(volatility=v - 1e-4)) / 1e-8
    assert g.volga == pytest.approx(volga_fd, rel=1e-4, abs=1e-7)


# --------------------------------------------------------------------------------------------------
# Implied volatility
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), EURO_CASES)
@pytest.mark.parametrize("call", [True, False])
def test_implied_vol_matches_quantlib(
    s: float, k: float, days: int, r: float, q: float, v: float, call: bool
) -> None:
    """Source: QuantLib VanillaOption.impliedVolatility on the QuantLib NPV."""
    opt, proc, _ = _ql_european(s, k, days, r, q, v, call)
    npv = opt.NPV()
    ql_iv = opt.impliedVolatility(npv, proc, 1e-12, 1000, 1e-4, 5.0)
    iv = E.calculate_implied_volatility(npv, s, k, days / 365.0, r, q, call)
    assert iv == pytest.approx(ql_iv, abs=1e-9)
    assert iv == pytest.approx(v, abs=1e-9)


def test_implied_vol_round_trip_grid() -> None:
    """Round trip over 630 (strike, expiry, vol, type) points incl. deep ITM/OTM (K = 20..400 on
    S = 100), T = 0.01..5y, sigma = 1%..400%. Where per-1.0 vega > 1e-3 the vol must be recovered
    (measured worst relative error 5.4e-10); everywhere a root is found it must reprice
    (measured worst |price residual| 1.9e-11)."""
    worst_rel, worst_res, n_well = 0.0, 0.0, 0
    for call in (True, False):
        for k in (20.0, 50.0, 80.0, 95.0, 100.0, 105.0, 120.0, 200.0, 400.0):
            for t in (0.01, 0.1, 0.5, 1.0, 5.0):
                for v in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0):
                    res = E.price_european_option(100.0, k, t, 0.05, v, 0.02, call)
                    try:
                        iv = E.calculate_implied_volatility(res.price, 100.0, k, t, 0.05, 0.02, call)
                    except ValueError:
                        assert res.greeks.vega * 100.0 <= 1e-3  # only ill-posed points may fail
                        continue
                    rep = E.price_european_option(100.0, k, t, 0.05, iv, 0.02, call).price
                    worst_res = max(worst_res, abs(rep - res.price))
                    if res.greeks.vega * 100.0 > 1e-3:
                        n_well += 1
                        worst_rel = max(worst_rel, abs(iv - v) / v)
    assert n_well == 430
    assert worst_rel < 1e-8
    assert worst_res < 1e-9


def test_implied_vol_rejects_price_outside_bounds() -> None:
    """Call price above S e^{-qT} (or below the forward intrinsic) has no BSM implied vol."""
    with pytest.raises(ValueError, match="No implied volatility"):
        E.calculate_implied_volatility(101.0, 100.0, 100.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="No implied volatility"):
        E.calculate_implied_volatility(1.0, 100.0, 50.0, 1.0, 0.05)


# --------------------------------------------------------------------------------------------------
# SABR: Hagan et al. (2002) and QuantLib sabrVolatility
# --------------------------------------------------------------------------------------------------
def _hagan_2002(f: float, k: float, t: float, a: float, b: float, rho: float, nu: float) -> float:
    """Independent transcription of Hagan et al. (2002) eq. (2.17a); eq. (2.18) when f == k."""
    if f == k:
        return float(
            (a / f ** (1 - b))
            * (
                1
                + (
                    (1 - b) ** 2 / 24 * a**2 / f ** (2 - 2 * b)
                    + rho * b * nu * a / (4 * f ** (1 - b))
                    + (2 - 3 * rho**2) / 24 * nu**2
                )
                * t
            )
        )
    lfk = math.log(f / k)
    fkb = (f * k) ** ((1 - b) / 2)
    z = nu / a * fkb * lfk
    xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    return float(
        a
        / (fkb * (1 + (1 - b) ** 2 / 24 * lfk**2 + (1 - b) ** 4 / 1920 * lfk**4))
        * (z / xz)
        * (
            1
            + (
                (1 - b) ** 2 / 24 * a**2 / fkb**2
                + rho * b * nu * a / (4 * fkb)
                + (2 - 3 * rho**2) / 24 * nu**2
            )
            * t
        )
    )


SABR_CASES = [
    # (f, alpha, beta, rho, nu, T): equity-like, rates-like (beta 0, 0.5, 1)
    (100.0, 2.0, 0.5, -0.3, 0.4, 1.0),
    (0.03, 0.02, 0.5, -0.2, 0.5, 2.0),
    (0.05, 0.04, 1.0, 0.3, 0.6, 5.0),
    (0.03, 0.01, 0.0, -0.4, 0.3, 1.0),
]


@pytest.mark.parametrize(("f", "a", "b", "rho", "nu", "t"), SABR_CASES)
def test_sabr_matches_hagan_and_quantlib(f: float, a: float, b: float, rho: float, nu: float, t: float) -> None:
    """Sources: Hagan et al. (2002) eq. (2.17a)/(2.18) and QuantLib ql.sabrVolatility (note QL's
    argument order: strike, forward, T, alpha, beta, nu, rho). Strikes 0.5F..1.5F."""
    for m in np.linspace(0.5, 1.5, 21):
        k = float(f * m)
        mine = E.sabr_implied_volatility(f, k, t, a, b, rho, nu)
        assert mine == pytest.approx(ql.sabrVolatility(k, f, t, a, b, nu, rho), rel=1e-9)
        assert mine == pytest.approx(_hagan_2002(f, k, t, a, b, rho, nu), rel=1e-9)


def test_sabr_lognormal_limit() -> None:
    """Hagan et al. (2002): beta = 1, nu = 0 is Black's model with constant vol alpha."""
    for k in (50.0, 100.0, 150.0):
        assert E.sabr_implied_volatility(100.0, k, 3.0, 0.25, 1.0, 0.0, 0.0) == pytest.approx(0.25, rel=1e-14)


@pytest.mark.parametrize(
    ("f", "a", "b", "rho", "nu", "t"),
    [(0.05, 0.04, 1.0, 0.3, 0.6, 5.0), (0.03, 0.02, 0.5, -0.2, 0.5, 2.0)],
)
def test_sabr_near_atm_is_continuous(f: float, a: float, b: float, rho: float, nu: float, t: float) -> None:
    """Bug fix: an absolute |F-K| < 1e-6 switch to the ATM formula dropped the first-order skew term.
    For F=0.05 (beta=1), K=F(1+1e-5) the engine returned 0.0455500 vs QuantLib 0.0455510
    (rel. err 2.25e-5); for F=0.03, beta=0.5 the error was 6.8e-6. After: < 1e-9."""
    for eps in (1e-5, 1e-4, 1e-3):
        k = f * (1 + eps)
        assert E.sabr_implied_volatility(f, k, t, a, b, rho, nu) == pytest.approx(
            ql.sabrVolatility(k, f, t, a, b, nu, rho), rel=1e-9
        )
    atm = E.sabr_implied_volatility(f, f, t, a, b, rho, nu)
    assert atm == pytest.approx(_hagan_2002(f, f, t, a, b, rho, nu), rel=1e-14)


@pytest.mark.parametrize(
    ("f", "beta", "true"),
    [(100.0, 0.5, (2.0, -0.3, 0.4)), (0.03, 0.5, (0.04, -0.25, 0.45)), (100.0, 1.0, (0.22, -0.5, 0.8))],
)
def test_calibrate_sabr_recovers_parameters(f: float, beta: float, true: tuple[float, float, float]) -> None:
    """Fit to a smile generated by QuantLib's sabrVolatility with known (alpha, rho, nu)."""
    a, rho, nu = true
    strikes = [f * m for m in (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)]
    vols = [ql.sabrVolatility(k, f, 1.0, a, beta, nu, rho) for k in strikes]
    c = E.calibrate_sabr(f, strikes, vols, 1.0, beta=beta)
    assert c.alpha == pytest.approx(a, rel=1e-5)
    assert c.rho == pytest.approx(rho, abs=1e-5)
    assert c.nu == pytest.approx(nu, rel=1e-5)
    assert c.rmse < 1e-7
    assert c.is_arbitrage_free is True


@pytest.mark.parametrize(("forward", "beta"), [(5000.0, 0.5), (100.0, 0.0)])
def test_calibrate_sabr_index_level_forward(forward: float, beta: float) -> None:
    """Bug fix: alpha was bounded to [1e-4, 5] in absolute units although alpha scales with
    F^(1-beta); for F=5000, beta=0.5 (alpha ~ 0.24*70.7 = 17) or F=100, beta=0 (alpha ~ 24) the
    initial guess fell outside the bounds and curve_fit raised ValueError."""
    strikes = [forward * m for m in (0.8, 0.9, 1.0, 1.1, 1.2)]
    vols = [0.30, 0.26, 0.24, 0.25, 0.27]
    c = E.calibrate_sabr(forward, strikes, vols, 1.0, beta=beta)
    assert c.rmse < 2e-3
    fitted = [E.sabr_implied_volatility(forward, k, 1.0, c.alpha, beta, c.rho, c.nu) for k in strikes]
    assert np.allclose(fitted, vols, atol=3e-3)


def test_calibrate_sabr_flags_negative_density() -> None:
    """Bug fix: is_arbitrage_free used to be ``rmse < 0.05``. A smile generated by Hagan's formula
    with F=3%, beta=0.5, alpha=0.25*sqrt(F), rho=-0.4, nu=0.6, T=10 is fitted exactly
    (rmse ~ 1e-13) yet its Black call prices at the quoted strikes violate butterfly convexity
    (independently confirmed below with QuantLib blackFormula). Before: True. After: False."""
    f, t, a, beta, rho, nu = 0.03, 10.0, 0.25 * math.sqrt(0.03), 0.5, -0.4, 0.6
    strikes = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]
    vols = [ql.sabrVolatility(k, f, t, a, beta, nu, rho) for k in strikes]
    calls = [ql.blackFormula(ql.Option.Call, k, f, vol * math.sqrt(t)) for k, vol in zip(strikes, vols, strict=True)]
    assert (calls[2] - calls[1]) / 0.005 - (calls[1] - calls[0]) / 0.005 < 0  # reference: concave
    c = E.calibrate_sabr(f, strikes, vols, t, beta=beta)
    assert c.rmse < 1e-9
    assert c.is_arbitrage_free is False


# --------------------------------------------------------------------------------------------------
# Butterfly arbitrage check
# --------------------------------------------------------------------------------------------------
def test_butterfly_accepts_bsm_prices() -> None:
    """BSM call prices are convex in strike (Breeden-Litzenberger density = discounted lognormal pdf)."""
    strikes = list(np.linspace(50.0, 150.0, 21))
    calls = [E.price_european_option(100.0, k, 1.0, 0.05, 0.25, 0.02).price for k in strikes]
    res = E.verify_butterfly_arbitrage(strikes, calls)
    assert res.is_valid is True and res.violations == []


def test_butterfly_flags_concave_prices() -> None:
    """Deliberately concave set: C(90)=10, C(100)=9, C(110)=1 -> 10 - 18 + 1 = -7 < 0."""
    res = E.verify_butterfly_arbitrage([90.0, 100.0, 110.0], [10.0, 9.0, 1.0])
    assert res.is_valid is False
    assert len(res.violations) == 1 and "100.0" in res.violations[0]


def test_butterfly_bump_one_strike_in_bsm_strip() -> None:
    """Raising one interior BSM price by 1.5 makes the strip concave at that strike only.
    Unbumped (C1 - 2 C2 + C3)/h at K=100 is 0.184 (h=10); the bump subtracts 2*1.5/10 = 0.3."""
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    calls = [E.price_european_option(100.0, k, 1.0, 0.05, 0.2).price for k in strikes]
    calls[2] += 1.5
    res = E.verify_butterfly_arbitrage(strikes, calls)
    assert res.is_valid is False
    assert len(res.violations) == 1 and "strike 100.0" in res.violations[0]


def test_butterfly_order_independent() -> None:
    """Bug fix: strikes given in descending order flipped the sign of the test. Before, the concave
    set [110, 100, 90] / [1, 9, 10] was reported valid and the convex set [110, 100, 90] /
    [3, 8, 15] was reported invalid. Points are now sorted by strike."""
    assert E.verify_butterfly_arbitrage([110.0, 100.0, 90.0], [1.0, 9.0, 10.0]).is_valid is False
    assert E.verify_butterfly_arbitrage([110.0, 100.0, 90.0], [3.0, 8.0, 15.0]).is_valid is True
    assert E.verify_butterfly_arbitrage([100.0, 90.0, 110.0], [8.0, 15.0, 3.0]).is_valid is True


def test_butterfly_rejects_bad_inputs() -> None:
    """Bug fix: NaN raised z3 'parser error' and duplicate strikes raised ZeroDivisionError."""
    with pytest.raises(ValueError, match="finite"):
        E.verify_butterfly_arbitrage([90.0, 100.0, 110.0], [15.0, 8.0, float("nan")])
    with pytest.raises(ValueError, match="distinct"):
        E.verify_butterfly_arbitrage([90.0, 90.0, 110.0], [15.0, 15.0, 3.0])
