"""Reference tests for multi-period covenant paths (core/covenant_paths.py).

References: the module's stated model evaluated by hand (exact arithmetic in the comments) and by
``simulate_covenant_path`` (plain rational arithmetic, no solver); for a fixed sweep fraction, the worst
corner (every covenant improves with EBITDA and worsens with the rate, argued below); for repayments,
bisection on the plain simulation. The Z3 results must agree with these.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lilavati.core.covenant_paths import (
    check_covenant_path,
    minimum_upfront_repayment_for_path,
    simulate_covenant_path,
)

# Teaching example (Tutorial 01, Step 4c): a leverage-based sweep with a cliff at 3.0x.
CLIFF = {"opening_debt": 270.0, "opening_cash": 30.0, "max_net_leverage": 5.5, "min_interest_coverage": 3.5,
         "min_liquidity": 10.0, "other_outflows": [20.0, 10.0], "sweep": [(3.0, 1.0), (1.5, 0.4)]}
CLIFF_RANGES = [(60.0, 100.0), (50.0, 90.0)]


def test_simulation_matches_hand_calculation() -> None:
    # Year 1, EBITDA 70: interest 0.06 x 270 = 16.2; cash before sweep 30 + 70 - 20 - 16.2 = 63.8;
    # net debt 270 - 63.8 = 206.2, leverage 2.9457 <= 3.0, so the 40% band applies:
    # sweep 0.4 x (63.8 - 10) = 21.52, debt 248.48, cash 42.28.
    # Year 2, EBITDA 50: interest 0.06 x 248.48 = 14.9088 (linked to the debt path);
    # coverage 50 / 14.9088 = 3.3537 < 3.5: breach.
    p = simulate_covenant_path(ebitda=[70.0, 50.0], interest_rate=0.06, **CLIFF)
    y1, y2 = p.periods
    assert (y1.interest, y1.sweep, y1.debt, y1.cash) == (16.2, 21.52, 248.48, 42.28)
    assert y1.breaches == [] and y2.interest == pytest.approx(14.9088, abs=1e-12)
    assert y2.interest_coverage == pytest.approx(50 / 14.9088, rel=1e-12) and y2.breaches == ["interest coverage"]
    assert not p.compliant and p.first_breach == 2


def test_breach_inside_the_ranges_that_every_corner_misses() -> None:
    # A better first year (70 instead of 60) lowers leverage below 3.0x, the sweep drops from 100% to
    # 40%, more debt is left, and year-2 coverage breaches. All four corners comply.
    corners = [simulate_covenant_path(ebitda=list(c), interest_rate=0.06, **CLIFF).compliant
               for c in itertools.product(*CLIFF_RANGES)]
    assert corners == [True, True, True, True]
    res = check_covenant_path(ebitda=CLIFF_RANGES, interest_rate=(0.06, 0.06), **CLIFF)
    assert res.status == "breach" and res.path is not None and res.path.first_breach == 2
    assert res.counterexample is not None
    replay = simulate_covenant_path(ebitda=res.counterexample["ebitda"], interest_rate=0.06, **CLIFF)
    assert not replay.compliant  # an independent evaluation confirms the counterexample


@pytest.mark.parametrize("seed", range(30))
def test_fixed_sweep_paths_are_decided_by_the_worst_corner(seed: int) -> None:
    # With a fixed sweep fraction, higher EBITDA in any period means more cash before the sweep, so
    # lower net debt (the sweep moves cash to debt without changing net debt), at least as much debt
    # repaid (so lower later interest) and at least as much cash after the sweep; a higher rate means
    # more interest. Every covenant therefore holds on all paths iff it holds at EBITDA lows and the rate high.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    lows = [float(round(rng.uniform(40, 90))) for _ in range(n)]
    ranges = [(lo, lo + float(round(rng.uniform(0, 30)))) for lo in lows]
    rate = (float(round(rng.uniform(0.03, 0.08), 3)), float(round(rng.uniform(0.08, 0.12), 3)))
    kw = {"opening_debt": float(round(rng.uniform(150, 400))), "opening_cash": float(round(rng.uniform(0, 40))),
          "max_net_leverage": [float(round(rng.uniform(3.5, 6.0), 1)) for _ in range(n)],
          "min_interest_coverage": float(round(rng.uniform(1.5, 4.0), 2)),
          "min_liquidity": float(round(rng.uniform(0, 15))),
          "other_outflows": [float(round(rng.uniform(5, 30))) for _ in range(n)],
          "amortisation": float(round(rng.uniform(0, 15))), "sweep": float(round(rng.uniform(0, 1), 2))}
    res = check_covenant_path(ebitda=ranges, interest_rate=rate, **kw)
    worst = simulate_covenant_path(ebitda=lows, interest_rate=rate[1], **kw)
    assert res.status in ("holds", "breach")
    assert (res.status == "holds") == worst.compliant


@pytest.mark.parametrize("seed", range(10))
def test_holds_survives_random_sampling(seed: int) -> None:
    # Soundness spot check with a sweep grid: when Z3 says "holds", no sampled path breaches.
    rng = np.random.default_rng(100 + seed)
    kw = {**CLIFF, "min_interest_coverage": float(round(rng.uniform(2.0, 3.0), 2))}
    res = check_covenant_path(ebitda=CLIFF_RANGES, interest_rate=(0.05, 0.06), **kw)
    if res.status != "holds":
        assert res.status == "breach" and res.path is not None and not res.path.compliant
        return
    for _ in range(200):
        e = [float(rng.uniform(lo, hi)) for lo, hi in CLIFF_RANGES]
        assert simulate_covenant_path(ebitda=e, interest_rate=float(rng.uniform(0.05, 0.06)), **kw).compliant


def _bisect_repayment(kw: dict, ebitda: list[float], rate: float) -> float:
    lo, hi = 0.0, kw["opening_debt"]
    if simulate_covenant_path(ebitda=ebitda, interest_rate=rate, **kw).compliant:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        ok = simulate_covenant_path(ebitda=ebitda, interest_rate=rate, **{**kw, "opening_debt": kw["opening_debt"] - mid}).compliant
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


@pytest.mark.parametrize("seed", range(15))
def test_repayment_matches_bisection_on_the_simulation(seed: int) -> None:
    # Point forecast, fixed sweep: more repayment means less debt and interest and the same cash, so
    # the covenants improve monotonically and bisection on the plain simulation is a reference.
    rng = np.random.default_rng(200 + seed)
    n = int(rng.integers(1, 4))
    ebitda = [float(round(rng.uniform(40, 90))) for _ in range(n)]
    kw = {"opening_debt": float(round(rng.uniform(200, 500))), "opening_cash": float(round(rng.uniform(5, 40))),
          "max_net_leverage": float(round(rng.uniform(3.0, 5.0), 1)),
          "min_interest_coverage": float(round(rng.uniform(2.0, 4.0), 2)), "min_liquidity": 0.0,
          "other_outflows": float(round(rng.uniform(5, 20))), "sweep": float(round(rng.uniform(0, 1), 2))}
    rate = float(round(rng.uniform(0.04, 0.09), 3))
    res = minimum_upfront_repayment_for_path(ebitda=[(e, e) for e in ebitda], interest_rate=(rate, rate), **kw)
    if res.status == "infeasible":
        assert not simulate_covenant_path(ebitda=ebitda, interest_rate=rate, **{**kw, "opening_debt": 0.0}).compliant
        return
    assert res.status == "found" and res.required_repayment is not None
    assert res.required_repayment == pytest.approx(_bisect_repayment(kw, ebitda, rate), rel=1e-9, abs=1e-9)


def test_repayment_with_a_sweep_cliff() -> None:
    # The counterexample-guided loop does not settle (each repayment moves the cliff), so the answer
    # comes from bisection on the quantified formula; its upper end works, its lower end does not.
    res = minimum_upfront_repayment_for_path(ebitda=CLIFF_RANGES, interest_rate=(0.06, 0.06), rel_tolerance=1e-6,
                                             **CLIFF)
    assert res.status in ("found", "found_within_tolerance")
    assert res.required_repayment is not None and res.lower_bound is not None
    assert res.required_repayment - res.lower_bound <= 1e-6 * CLIFF["opening_debt"] + 1e-12
    assert 11.7 < res.lower_bound < res.required_repayment < 11.8  # about 11.73 (rel 1e-9 run: 11.7263)
    works = check_covenant_path(ebitda=CLIFF_RANGES, interest_rate=(0.06, 0.06),
                                **{**CLIFF, "opening_debt": CLIFF["opening_debt"] - res.required_repayment})
    assert works.status == "holds"
    if res.lower_bound > 0:
        fails = check_covenant_path(ebitda=CLIFF_RANGES, interest_rate=(0.06, 0.06),
                                    **{**CLIFF, "opening_debt": CLIFF["opening_debt"] - res.lower_bound})
        assert fails.status == "breach"


def test_infeasible_when_liquidity_cannot_be_met() -> None:
    # Outflows exceed EBITDA plus opening cash in year 1: even with no debt (no interest), cash ends
    # below the minimum, so no repayment helps.
    res = minimum_upfront_repayment_for_path(opening_debt=100.0, opening_cash=5.0, ebitda=[(50.0, 60.0)],
                                             interest_rate=(0.05, 0.05), max_net_leverage=6.0,
                                             min_interest_coverage=1.0, min_liquidity=10.0, other_outflows=80.0)
    assert res.status == "infeasible"


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="EBITDA must be positive"):
        check_covenant_path(opening_debt=100.0, opening_cash=0.0, ebitda=[(0.0, 10.0)], interest_rate=(0.05, 0.05),
                            max_net_leverage=4.0, min_interest_coverage=2.0)
    with pytest.raises(ValueError, match="one value per period"):
        check_covenant_path(opening_debt=100.0, opening_cash=0.0, ebitda=[(10.0, 20.0)] * 2,
                            interest_rate=(0.05, 0.05), max_net_leverage=[4.0], min_interest_coverage=2.0)
    with pytest.raises(ValueError, match="sweep fractions"):
        simulate_covenant_path(opening_debt=100.0, opening_cash=0.0, ebitda=[10.0], interest_rate=0.05,
                               max_net_leverage=4.0, min_interest_coverage=2.0, sweep=[(3.0, 1.5)])
