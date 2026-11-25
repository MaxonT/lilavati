"""Multi-period covenant paths with linked inputs: simulation, checks over ranges, smallest repayment.

The model (stated so it can be checked; every function here uses the same one):

- Periods are years; covenants are tested at each period end, on that period's EBITDA.
- Interest is linked to the debt path: ``interest_t = rate * debt_{t-1}`` (opening debt of the period).
- ``scheduled_t = min(amortisation_t, debt_{t-1})``.
- ``cash_pre_t = cash_{t-1} + EBITDA_t - other_outflows_t - interest_t - scheduled_t`` (other outflows:
  capex, taxes, working capital, in the same units).
- Excess cash sweep: ``sweep_t = min(f_t * max(0, cash_pre_t - min_liquidity), debt left)``. The fraction
  ``f_t`` is fixed, or set by a leverage grid (as in many credit agreements): the first
  ``(threshold, fraction)`` whose threshold net leverage exceeds, else 0. A sweep moves money from cash
  to debt, so net debt, and so the net leverage that selects ``f_t``, does not depend on the sweep.
- ``debt_t = debt_{t-1} - scheduled_t - sweep_t``; ``cash_t = cash_pre_t - sweep_t``.
- Covenants each period: net leverage ``(debt_t - cash_t) / EBITDA_t <= max_net_leverage_t`` (step-downs
  allowed), interest coverage ``EBITDA_t / interest_t >= min_interest_coverage`` (vacuous when no interest),
  liquidity ``cash_t >= min_liquidity``. A ratio at its limit complies.

``simulate_covenant_path`` evaluates one path in exact rational arithmetic (plain Python, no solver): it is
the reference the Z3 functions are tested against, and every counterexample Z3 returns is replayed
through it before it is reported. ``check_covenant_path`` asks Z3 whether any EBITDA path within the
per-period ranges, at any rate within its range, breaches a covenant in any period.
``minimum_upfront_repayment_for_path`` finds the smallest repayment now (from outside the cash balance)
that keeps every covenant met for every path in the ranges, by a counterexample-guided loop (optimise
over the scenarios found so far; ask Z3 for a breaching scenario; repeat).

With a fixed rate the constraints are linear with case splits; a rate range makes interest a product of
two unknowns (nonlinear real arithmetic), which Z3 decides but may not finish within the time limit: the
result is then ``unknown``, never a guess.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, Literal

import z3
from pydantic import BaseModel, Field

from lilavati.core.invariants import _frac, _require_finite

Sweep = float | Sequence[tuple[float, float]]


class PathPeriod(BaseModel):
    period: int
    ebitda: float
    interest: float
    scheduled_repayment: float
    sweep: float
    debt: float
    cash: float
    net_leverage: float
    interest_coverage: float | None = Field(default=None, description="None when no interest is due")
    breaches: list[str] = Field(default_factory=list)


class CovenantPath(BaseModel):
    periods: list[PathPeriod]
    compliant: bool
    first_breach: int | None = Field(default=None, description="1-based period of the first breach")


class PathCheckResult(BaseModel):
    status: Literal["holds", "breach", "unknown"]
    counterexample: dict[str, Any] | None = Field(
        default=None, description="EBITDA per period and the rate at which a covenant is breached")
    path: CovenantPath | None = Field(default=None, description="the counterexample replayed in plain arithmetic")
    detail: str = ""


class PathRepaymentResult(BaseModel):
    status: Literal["found", "found_within_tolerance", "infeasible", "unknown"]
    required_repayment: float | None = Field(
        default=None, description="a repayment that keeps every covenant met on every path in the ranges")
    required_repayment_exact: str | None = Field(
        default=None, description="the same as an exact fraction (the float above is rounded up so it still complies)")
    lower_bound: float | None = Field(
        default=None, description="no repayment at or below this works (bisection); equals the answer when exact")
    method: str = ""
    scenarios_used: list[dict[str, Any]] = Field(default_factory=list)
    detail: str = ""


def _qx(x: float | Fraction, ctx: z3.Context) -> z3.ArithRef:
    """Exact Z3 numeral in ``ctx`` for a float (its decimal string) or a Fraction."""
    if isinstance(x, Fraction):
        return z3.RealVal(f"{x.numerator}/{x.denominator}", ctx)
    return z3.RealVal(str(x), ctx)


def _float_up(x: Fraction) -> float:
    """The smallest float not below x, so a reported repayment still complies after rounding."""
    f = float(x)
    return math.nextafter(f, math.inf) if Fraction(f) < x else f


def _grid(sweep: Sweep) -> list[tuple[Fraction, Fraction]]:
    """Sweep as a sorted grid [(threshold, fraction)], highest threshold first; a float is one band."""
    if isinstance(sweep, int | float):
        rows = [(float("-inf"), float(sweep))]
    else:
        rows = [(float(t), float(f)) for t, f in sweep]
    out = []
    for t, f in sorted(rows, key=lambda r: r[0], reverse=True):
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"sweep fractions must be in [0, 1], got {f}")
        out.append((Fraction(-10**30) if t == float("-inf") else Fraction(str(t)), Fraction(str(f))))
    return out


def _per_period(x: float | Sequence[float], n: int, name: str) -> list[float]:
    vals = [float(x)] * n if isinstance(x, int | float) else [float(v) for v in x]
    if len(vals) != n:
        raise ValueError(f"{name}: expected one value per period ({n}), got {len(vals)}")
    _require_finite(**{f"{name}_{i + 1}": v for i, v in enumerate(vals)})
    return vals


def _validate(opening_debt: float | Fraction, opening_cash: float, min_liquidity: float, min_interest_coverage: float,
              rate: tuple[float, float], ebitda: Sequence[tuple[float, float]]) -> None:
    _require_finite(opening_debt=float(opening_debt), opening_cash=opening_cash, min_liquidity=min_liquidity,
                    min_interest_coverage=min_interest_coverage, rate_low=rate[0], rate_high=rate[1])
    if opening_debt < 0 or opening_cash < 0 or min_liquidity < 0 or min_interest_coverage < 0:
        raise ValueError("opening debt and cash, minimum liquidity and minimum coverage must be non-negative")
    if not 0 <= rate[0] <= rate[1]:
        raise ValueError(f"interest rate range must satisfy 0 <= low <= high, got {rate}")
    if not ebitda:
        raise ValueError("at least one period is needed")
    for i, (lo, hi) in enumerate(ebitda, 1):
        _require_finite(**{f"ebitda_{i}_low": lo, f"ebitda_{i}_high": hi})
        if lo > hi:
            raise ValueError(f"EBITDA range for period {i}: low {lo} is above high {hi}")
        if lo <= 0:
            raise ValueError(f"EBITDA must be positive in every period (net leverage is undefined); period {i}")


def simulate_covenant_path(
    opening_debt: float | Fraction,
    opening_cash: float,
    ebitda: Sequence[float],
    interest_rate: float,
    max_net_leverage: float | Sequence[float],
    min_interest_coverage: float,
    min_liquidity: float = 0.0,
    other_outflows: float | Sequence[float] = 0.0,
    amortisation: float | Sequence[float] = 0.0,
    sweep: Sweep = 0.0,
) -> CovenantPath:
    """One path in exact rational arithmetic (the module's model); the reference for the Z3 functions."""
    n = len(ebitda)
    _validate(opening_debt, opening_cash, min_liquidity, min_interest_coverage, (interest_rate, interest_rate),
              [(e, e) for e in ebitda])
    lev = [Fraction(str(v)) for v in _per_period(max_net_leverage, n, "max_net_leverage")]
    out = [Fraction(str(v)) for v in _per_period(other_outflows, n, "other_outflows")]
    amo = [Fraction(str(v)) for v in _per_period(amortisation, n, "amortisation")]
    grid = _grid(sweep)
    d = opening_debt if isinstance(opening_debt, Fraction) else Fraction(str(opening_debt))
    c = Fraction(str(opening_cash))
    rate, m, cmin = Fraction(str(interest_rate)), Fraction(str(min_liquidity)), Fraction(str(min_interest_coverage))
    rows: list[PathPeriod] = []
    first: int | None = None
    for t in range(n):
        e = Fraction(str(ebitda[t]))
        interest = rate * d
        sched = min(amo[t], d)
        pre = c + e - out[t] - interest - sched
        net_debt = d - sched - pre
        frac = next((f for thr, f in grid if net_debt > thr * e), Fraction(0))
        sweep_t = min(frac * max(Fraction(0), pre - m), d - sched)
        d, c = d - sched - sweep_t, pre - sweep_t
        breaches = []
        if net_debt > lev[t] * e:
            breaches.append("net leverage")
        if e < cmin * interest:
            breaches.append("interest coverage")
        if c < m:
            breaches.append("liquidity")
        if breaches and first is None:
            first = t + 1
        rows.append(PathPeriod(period=t + 1, ebitda=float(e), interest=float(interest), scheduled_repayment=float(sched),
                               sweep=float(sweep_t), debt=float(d), cash=float(c), net_leverage=float(net_debt / e),
                               interest_coverage=None if interest == 0 else float(e / interest), breaches=breaches))
    return CovenantPath(periods=rows, compliant=first is None, first_breach=first)


def _encode(opening_debt: z3.ArithRef, opening_cash: float, e: Sequence[z3.ArithRef], rate: z3.ArithRef,
            lev: Sequence[float], cmin: float, m: float, out: Sequence[float], amo: Sequence[float],
            grid: list[tuple[Fraction, Fraction]], tag: str, ctx: z3.Context
            ) -> tuple[list[z3.BoolRef], list[z3.BoolRef], list[z3.ArithRef]]:
    """Path definitions, per-period covenant conditions, and the auxiliary variables the definitions use."""
    defs: list[z3.BoolRef] = []
    ok: list[z3.BoolRef] = []
    aux: list[z3.ArithRef] = []
    d_prev: z3.ArithRef = opening_debt
    c_prev: z3.ArithRef = _qx(opening_cash, ctx)
    zero, mq = z3.RealVal(0, ctx), _qx(m, ctx)
    for t in range(len(e)):
        d_t, c_t = z3.Real(f"{tag}debt_{t + 1}", ctx), z3.Real(f"{tag}cash_{t + 1}", ctx)
        i_t = z3.Real(f"{tag}interest_{t + 1}", ctx)
        sched = z3.If(_qx(amo[t], ctx) <= d_prev, _qx(amo[t], ctx), d_prev)
        pre = c_prev + e[t] - _qx(out[t], ctx) - i_t - sched
        net_debt = d_prev - sched - pre
        frac: z3.ArithRef = zero
        for thr, f in reversed(grid):  # build the If chain so the highest threshold is tested first
            frac = z3.If(net_debt > _qx(thr, ctx) * e[t], _qx(f, ctx), frac)
        excess = z3.If(pre > mq, pre - mq, zero)
        raw = frac * excess
        sweep_t = z3.If(raw <= d_prev - sched, raw, d_prev - sched)
        defs += [i_t == rate * d_prev, d_t == d_prev - sched - sweep_t, c_t == pre - sweep_t]
        aux += [d_t, c_t, i_t]
        ok.append(z3.And(net_debt <= _qx(lev[t], ctx) * e[t], e[t] >= _qx(cmin, ctx) * i_t, c_t >= mq))
        d_prev, c_prev = d_t, c_t
    return defs, ok, aux


def check_covenant_path(
    opening_debt: float | Fraction,
    opening_cash: float,
    ebitda: Sequence[tuple[float, float]],
    interest_rate: tuple[float, float],
    max_net_leverage: float | Sequence[float],
    min_interest_coverage: float,
    min_liquidity: float = 0.0,
    other_outflows: float | Sequence[float] = 0.0,
    amortisation: float | Sequence[float] = 0.0,
    sweep: Sweep = 0.0,
    timeout_ms: int = 30_000,
) -> PathCheckResult:
    """Do all covenants hold in every period for every EBITDA path and rate within the ranges?

    ``ebitda``: one (low, high) range per period (low = high for a point forecast); ``interest_rate``:
    (low, high). Z3 searches for a breach anywhere in the ranges. ``holds`` is a statement about every
    path in the ranges; ``breach`` comes with a counterexample replayed in plain arithmetic
    (``simulate_covenant_path``); ``unknown`` means Z3 did not decide within ``timeout_ms`` (possible
    when the rate is a range, which makes interest nonlinear).
    """
    n = len(ebitda)
    _validate(opening_debt, opening_cash, min_liquidity, min_interest_coverage, interest_rate, ebitda)
    lev = _per_period(max_net_leverage, n, "max_net_leverage")
    out = _per_period(other_outflows, n, "other_outflows")
    amo = _per_period(amortisation, n, "amortisation")
    grid = _grid(sweep)
    ctx = z3.Context()  # fresh per call: results and run time do not depend on earlier Z3 use
    e = [z3.Real(f"ebitda_{t + 1}", ctx) for t in range(n)]
    rate = z3.Real("interest_rate", ctx)
    s = z3.Solver(ctx=ctx)
    s.set("timeout", timeout_ms)
    for t, (lo, hi) in enumerate(ebitda):
        s.add(e[t] >= _qx(lo, ctx), e[t] <= _qx(hi, ctx))
    s.add(rate >= _qx(interest_rate[0], ctx), rate <= _qx(interest_rate[1], ctx))
    defs, ok, _ = _encode(_qx(opening_debt, ctx), opening_cash, e, rate, lev, min_interest_coverage, min_liquidity,
                          out, amo, grid, "", ctx)
    s.add(*defs)
    s.add(z3.Not(z3.And(*ok)))
    res = s.check()
    if res == z3.unsat:
        return PathCheckResult(status="holds",
                               detail=f"no EBITDA path and rate within the ranges breaches a covenant in any of {n} periods")
    if res == z3.unknown:
        return PathCheckResult(status="unknown", detail=f"Z3 did not decide within {timeout_ms} ms ({s.reason_unknown()})")
    mdl = s.model()
    ce_e = [float(_frac(mdl.eval(v, model_completion=True))) for v in e]
    ce_r = float(_frac(mdl.eval(rate, model_completion=True)))
    path = simulate_covenant_path(opening_debt, opening_cash, ce_e, ce_r, lev, min_interest_coverage, min_liquidity,
                                  out, amo, sweep)
    if path.compliant:  # the replay must confirm the breach; anything else is a modelling bug
        raise RuntimeError("Z3 counterexample did not breach when replayed in plain arithmetic")
    return PathCheckResult(status="breach", counterexample={"ebitda": ce_e, "interest_rate": ce_r}, path=path,
                           detail=f"breach in period {path.first_breach}: {', '.join(path.periods[path.first_breach - 1].breaches)}"
                           if path.first_breach else "")


def minimum_upfront_repayment_for_path(
    opening_debt: float,
    opening_cash: float,
    ebitda: Sequence[tuple[float, float]],
    interest_rate: tuple[float, float],
    max_net_leverage: float | Sequence[float],
    min_interest_coverage: float,
    min_liquidity: float = 0.0,
    other_outflows: float | Sequence[float] = 0.0,
    amortisation: float | Sequence[float] = 0.0,
    sweep: Sweep = 0.0,
    max_iterations: int = 8,
    rel_tolerance: float = 1e-9,
    timeout_ms: int = 30_000,
) -> PathRepaymentResult:
    """Smallest repayment now (from outside the cash balance) keeping every covenant met on every path.

    Two methods, both certified:

    1. Counterexample-guided (``found``, exact): minimise the repayment over the scenarios found so far
       (``z3.Optimize``; each scenario a point EBITDA path and rate), then ask ``check_covenant_path`` for a
       breaching scenario at that repayment; add it and repeat. When no breach exists, the repayment works
       for every path and is the smallest that works for the scenarios used, hence the smallest overall.
    2. If that does not settle within ``max_iterations``, or the optimiser does not finish (a
       leverage-based sweep grid can move the breaching paths each time), bisection on a bound b: "some repayment r <= b keeps every covenant met
       for all paths in the ranges" is one quantified formula (exists r, for all paths), decided by Z3, and
       it is monotone in b even when the set of working repayments is not an interval
       (``found_within_tolerance``: the answer is in (lower_bound, required_repayment], width at most
       ``rel_tolerance`` times the debt; required_repayment is re-checked with ``check_covenant_path``).

    ``infeasible``: even repaying all debt fails (e.g. liquidity). ``unknown``: a check was not decided
    within ``timeout_ms`` (possible with a rate range, which makes interest nonlinear). All checks use the
    exact repayment; the reported float is rounded up so it still complies (``required_repayment_exact``
    gives the fraction).
    """
    n = len(ebitda)
    _validate(opening_debt, opening_cash, min_liquidity, min_interest_coverage, interest_rate, ebitda)
    lev = _per_period(max_net_leverage, n, "max_net_leverage")
    out = _per_period(other_outflows, n, "other_outflows")
    amo = _per_period(amortisation, n, "amortisation")
    grid = _grid(sweep)
    debt0 = Fraction(str(opening_debt))
    scenarios: list[dict[str, Any]] = [{"ebitda": [lo for lo, _ in ebitda], "interest_rate": interest_rate[1]}]
    # Any repayment that works for every path works for the scenarios found so far, so each scenario
    # minimum that fails the full check is a strict lower bound: the bisection below starts from it.
    lower = Fraction(0)
    lower_is_strict = False
    for _ in range(max_iterations):
        ctx = z3.Context()
        r = z3.Real("repayment", ctx)
        opt = z3.Optimize(ctx=ctx)
        opt.set("timeout", timeout_ms)
        opt.add(r >= 0, r <= _qx(opening_debt, ctx))
        for k, sc in enumerate(scenarios):
            defs, ok, _ = _encode(_qx(opening_debt, ctx) - r, opening_cash, [_qx(v, ctx) for v in sc["ebitda"]],
                                  _qx(sc["interest_rate"], ctx), lev, min_interest_coverage, min_liquidity, out, amo,
                                  grid, f"s{k}_", ctx)
            opt.add(*defs, *ok)
        h = opt.minimize(r)
        res = opt.check()
        if res == z3.unsat:
            return PathRepaymentResult(status="infeasible", method="counterexample-guided", scenarios_used=scenarios,
                                       detail="no repayment up to the whole debt meets the covenants in these scenarios")
        if res == z3.unknown:
            break  # fall through to the bisection
        r_star = _frac(h.value())
        # Exact debt (a Fraction): the optimum often sits exactly at a covenant limit, where a float
        # conversion could push it just over.
        chk = check_covenant_path(debt0 - r_star, opening_cash, ebitda, interest_rate, lev,
                                  min_interest_coverage, min_liquidity, out, amo, sweep, timeout_ms)
        if chk.status == "holds":
            return PathRepaymentResult(status="found", required_repayment=_float_up(r_star),
                                       required_repayment_exact=str(r_star), lower_bound=float(r_star),
                                       method="counterexample-guided", scenarios_used=scenarios,
                                       detail=f"exact: covenants hold on every path after repaying {float(r_star):,.6g}")
        if chk.status == "unknown" or chk.counterexample is None:
            break
        lower, lower_is_strict = r_star, True
        scenarios.append(chk.counterexample)

    # Bisection on a quantified formula: exists r <= b such that every path in the ranges complies.
    def feasible_up_to(b: Fraction) -> tuple[z3.CheckSatResult, Fraction | None]:
        ctx = z3.Context()
        e = [z3.Real(f"ebitda_{t + 1}", ctx) for t in range(n)]
        rate = z3.Real("interest_rate", ctx)
        box = z3.And(*[z3.And(e[t] >= _qx(lo, ctx), e[t] <= _qx(hi, ctx)) for t, (lo, hi) in enumerate(ebitda)],
                     rate >= _qx(interest_rate[0], ctx), rate <= _qx(interest_rate[1], ctx))
        rr = z3.Real("repayment", ctx)
        defs, ok, aux = _encode(_qx(opening_debt, ctx) - rr, opening_cash, e, rate, lev, min_interest_coverage,
                                min_liquidity, out, amo, grid, "q_", ctx)
        s = z3.Solver(ctx=ctx)
        s.set("timeout", timeout_ms)
        s.add(rr >= 0, rr <= _qx(b, ctx),
              z3.ForAll([*e, rate, *aux], z3.Implies(z3.And(box, *defs), z3.And(*ok))))
        res = s.check()
        return res, (_frac(s.model().eval(rr, model_completion=True)) if res == z3.sat else None)

    res_hi, witness = feasible_up_to(debt0)
    if res_hi == z3.unsat:
        return PathRepaymentResult(status="infeasible", method="quantified bisection", scenarios_used=scenarios,
                                   detail="no repayment up to the whole debt keeps every path compliant")
    if res_hi == z3.unknown or witness is None:
        return PathRepaymentResult(status="unknown", scenarios_used=scenarios,
                                   detail=f"quantified check not decided within {timeout_ms} ms")
    lo, hi = lower, witness
    if not lower_is_strict:
        res0, _ = feasible_up_to(Fraction(0))
        if res0 == z3.sat:
            lo = hi = Fraction(0)
    width = Fraction(str(rel_tolerance)) * max(debt0, Fraction(1))
    while hi - lo > width:
        mid = (lo + hi) / 2
        res_mid, w = feasible_up_to(mid)
        if res_mid == z3.unknown:
            return PathRepaymentResult(status="unknown", required_repayment=_float_up(hi), required_repayment_exact=str(hi),
                                       lower_bound=float(lo), scenarios_used=scenarios,
                                       detail="quantified check not decided during bisection; the repayment shown works")
        if res_mid == z3.sat and w is not None:
            hi = w
        else:
            lo = mid
    chk = check_covenant_path(debt0 - hi, opening_cash, ebitda, interest_rate, lev, min_interest_coverage,
                              min_liquidity, out, amo, sweep, timeout_ms)
    if chk.status == "breach":  # the two methods must agree; anything else is a modelling bug
        raise RuntimeError(f"bisection answer {float(hi)} failed the direct check")
    return PathRepaymentResult(status="found_within_tolerance", required_repayment=_float_up(hi),
                               required_repayment_exact=str(hi), lower_bound=float(lo),
                               method="quantified bisection", scenarios_used=scenarios,
                               detail=f"the smallest repayment is in ({float(lo):,.10g}, {float(hi):,.10g}]")
