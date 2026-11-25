"""
Suvarṇa Formal Invariant Engine (named after Bhāskara II's Suvarṇa-gaṇita).

Uses the Z3 SMT solver to check accounting identities in exact decimal arithmetic.

Scope note: in the ``verify_*`` checks every variable is bound to a concrete input value, so a
Z3 check is equivalent to evaluating the identity in exact rational arithmetic on the decimal
representation of the inputs; it verifies the given numbers. ``verify_debt_covenant`` uses plain
arithmetic. Three functions go further: ``diagnose_identities`` finds the smallest sets of figures
that would explain failed identities (MaxSMT), ``minimum_debt_repayment`` solves for the smallest repayment meeting both covenants,
and ``check_covenants_over_ranges`` decides whether covenants hold for every input in given ranges.
"""

from __future__ import annotations

import math
from fractions import Fraction

import z3
from pydantic import BaseModel, Field


class InvariantCheckResult(BaseModel):
    """Result of an SMT invariant verification check."""

    is_valid: bool = Field(
        ..., description="True if invariants are mathematically satisfiable and consistent"
    )
    status: str = Field(..., description="SMT solver status ('SAT', 'UNSAT', 'UNKNOWN')")
    violations: list[str] = Field(
        default_factory=list, description="List of violated constraints or contradictions"
    )
    model_values: dict[str, float] = Field(
        default_factory=dict, description="Solved variable values if satisfiable"
    )


def _require_finite(**values: float) -> None:
    bad = [k for k, v in values.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"Inputs must be finite numbers; got non-finite: {', '.join(bad)}")


class SuvarnaEngine:
    """Formal verification engine for financial statements and quantitative models."""

    @staticmethod
    def verify_balance_sheet_equation(
        total_assets: float,
        total_liabilities: float,
        shareholders_equity: float,
        tolerance: float = 1e-4,
    ) -> InvariantCheckResult:
        """
        Checks balance sheet equilibrium on the given numbers using the Z3 SMT solver:
        Assets == Liabilities + Equity (within absolute tolerance, in input units).
        """
        _require_finite(
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            shareholders_equity=shareholders_equity,
            tolerance=tolerance,
        )
        solver = z3.Solver()
        a = z3.Real("total_assets")
        liab = z3.Real("total_liabilities")
        e = z3.Real("shareholders_equity")

        tol = z3.RealVal(str(tolerance))
        neg_tol = z3.RealVal(str(-tolerance))

        diff = a - (liab + e)
        solver.add(diff >= neg_tol)
        solver.add(diff <= tol)

        solver.add(a == z3.RealVal(str(total_assets)))
        solver.add(liab == z3.RealVal(str(total_liabilities)))
        solver.add(e == z3.RealVal(str(shareholders_equity)))

        check = solver.check()
        if check == z3.sat:
            return InvariantCheckResult(
                is_valid=True,
                status="SAT",
                violations=[],
                model_values={
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "shareholders_equity": shareholders_equity,
                },
            )

        err = abs(total_assets - (total_liabilities + shareholders_equity))
        return InvariantCheckResult(
            is_valid=False,
            status="UNSAT",
            violations=[
                f"Balance sheet mismatch: Assets ({total_assets}) != Liabilities ({total_liabilities}) "
                f"+ Equity ({shareholders_equity}). Discrepancy = {err:.4f}"
            ],
        )

    @staticmethod
    def verify_three_statement_invariants(
        total_assets: float,
        total_liabilities: float,
        shareholders_equity: float,
        net_income: float,
        cash_start: float,
        cash_end: float,
        cfo: float,
        cfi: float,
        cff: float,
        tolerance: float = 1e-4,
    ) -> InvariantCheckResult:
        """
        Checks balance sheet equilibrium and cash flow reconciliation on the given numbers:
        1. Balance Sheet: Assets == Liabilities + Equity
        2. Cash Flow: CashEnd == CashStart + CFO + CFI + CFF

        Tolerance is absolute, in input units. `net_income` is accepted for API stability but
        not yet checked: a retained-earnings roll-forward (RE_end = RE_start + NI - dividends)
        needs inputs this function does not take.
        """
        _require_finite(
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            shareholders_equity=shareholders_equity,
            net_income=net_income,
            cash_start=cash_start,
            cash_end=cash_end,
            cfo=cfo,
            cfi=cfi,
            cff=cff,
            tolerance=tolerance,
        )
        solver = z3.Solver()

        # Symbolic real variables
        a = z3.Real("total_assets")
        liab = z3.Real("total_liabilities")
        e = z3.Real("shareholders_equity")
        c_start = z3.Real("cash_start")
        c_end = z3.Real("cash_end")
        cfo_s = z3.Real("cfo")
        cfi_s = z3.Real("cfi")
        cff_s = z3.Real("cff")

        # Accounting identities with tolerance
        tol = z3.RealVal(str(tolerance))
        neg_tol = z3.RealVal(str(-tolerance))

        # Balance sheet identity: -tol <= (Assets - (Liabilities + Equity)) <= tol
        bs_diff = a - (liab + e)
        solver.add(bs_diff >= neg_tol)
        solver.add(bs_diff <= tol)

        # Cash flow identity: -tol <= (CashEnd - (CashStart + CFO + CFI + CFF)) <= tol
        cf_diff = c_end - (c_start + cfo_s + cfi_s + cff_s)
        solver.add(cf_diff >= neg_tol)
        solver.add(cf_diff <= tol)

        # Ground actual proposed numbers
        solver.add(a == z3.RealVal(str(total_assets)))
        solver.add(liab == z3.RealVal(str(total_liabilities)))
        solver.add(e == z3.RealVal(str(shareholders_equity)))
        solver.add(c_start == z3.RealVal(str(cash_start)))
        solver.add(c_end == z3.RealVal(str(cash_end)))
        solver.add(cfo_s == z3.RealVal(str(cfo)))
        solver.add(cfi_s == z3.RealVal(str(cfi)))
        solver.add(cff_s == z3.RealVal(str(cff)))

        check = solver.check()
        if check == z3.sat:
            return InvariantCheckResult(
                is_valid=True,
                status="SAT",
                violations=[],
                model_values={
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "shareholders_equity": shareholders_equity,
                    "cash_reconciled_delta": cash_end - cash_start,
                },
            )

        # Identify specific violations
        violations: list[str] = []
        bs_error = abs(total_assets - (total_liabilities + shareholders_equity))
        if bs_error > tolerance:
            violations.append(
                f"Balance sheet mismatch: Assets ({total_assets}) != Liabilities ({total_liabilities}) "
                f"+ Equity ({shareholders_equity}). Delta = {bs_error:.4f}"
            )

        cf_error = abs(cash_end - (cash_start + cfo + cfi + cff))
        if cf_error > tolerance:
            violations.append(
                f"Cash flow mismatch: Ending Cash ({cash_end}) != Starting Cash ({cash_start}) "
                f"+ CFO ({cfo}) + CFI ({cfi}) + CFF ({cff}). Delta = {cf_error:.4f}"
            )

        return InvariantCheckResult(
            is_valid=False,
            status="UNSAT",
            violations=violations,
        )

    @staticmethod
    def verify_debt_covenant(
        total_debt: float,
        cash: float,
        ebitda: float,
        max_leverage_covenant: float,
        min_interest_coverage_covenant: float,
        interest_expense: float,
    ) -> InvariantCheckResult:
        """
        Checks credit agreement maintenance covenants (plain arithmetic, not Z3):
        1. Net Leverage = (Total Debt - Cash) / EBITDA <= max_leverage_covenant
        2. Interest Coverage = EBITDA / InterestExpense >= min_interest_coverage_covenant

        A ratio exactly at its limit complies; comparisons allow a 1e-9 relative tolerance so
        floating-point rounding (e.g. 2.1 / 0.7 = 3.0000000000000004) is not reported as a breach.
        """
        _require_finite(
            total_debt=total_debt,
            cash=cash,
            ebitda=ebitda,
            max_leverage_covenant=max_leverage_covenant,
            min_interest_coverage_covenant=min_interest_coverage_covenant,
            interest_expense=interest_expense,
        )
        rel_tol = 1e-9
        violations: list[str] = []

        if ebitda <= 0:
            violations.append(
                f"Incurrence failure: EBITDA ({ebitda}) is non-positive; leverage ratio undefined."
            )
            return InvariantCheckResult(is_valid=False, status="UNSAT", violations=violations)

        net_debt = total_debt - cash
        current_leverage = net_debt / ebitda

        if current_leverage > max_leverage_covenant * (1.0 + rel_tol) + rel_tol:
            violations.append(
                f"Covenant breach: Net Leverage ({current_leverage:.2f}x) exceeds maximum allowable limit "
                f"({max_leverage_covenant:.2f}x)."
            )

        if interest_expense > 0:
            coverage = ebitda / interest_expense
            if coverage < min_interest_coverage_covenant * (1.0 - rel_tol) - rel_tol:
                violations.append(
                    f"Covenant breach: Interest Coverage ({coverage:.2f}x) is below minimum allowable limit "
                    f"({min_interest_coverage_covenant:.2f}x)."
                )

        is_valid = len(violations) == 0
        return InvariantCheckResult(
            is_valid=is_valid,
            status="SAT" if is_valid else "UNSAT",
            violations=violations,
            model_values={
                "net_debt": net_debt,
                "current_leverage": current_leverage,
                "headroom_leverage": max_leverage_covenant - current_leverage,
            },
        )


# --- Beyond point checks: diagnosis, debt repayment, ranges ------------------------------------
#
# Checking one identity on concrete numbers needs only exact rational arithmetic. Z3 earns its place
# on the questions below: which figures to change (MaxSMT, Bjørner, Phan and Fleckenstein 2015),
# the smallest debt repayment meeting several covenants (linear optimisation in exact rationals), and whether a
# property holds for every value in a range (satisfiability of the negated property).

# name -> coefficients; each identity reads sum(coefficient * figure) == 0.
STATEMENT_IDENTITIES: dict[str, dict[str, float]] = {
    "assets = liabilities + equity": {"total_assets": 1.0, "total_liabilities": -1.0, "total_equity": -1.0},
    "assets = total liabilities and equity": {"total_assets": 1.0, "liabilities_and_equity": -1.0},
    "revenue - cost of revenue = gross profit": {"revenue": 1.0, "cost_of_revenue": -1.0, "gross_profit": -1.0},
    "opening cash + cash flows = closing cash": {
        "cash_closing": 1.0, "cash_opening": -1.0, "operating_cash_flow": -1.0, "investing_cash_flow": -1.0,
        "financing_cash_flow": -1.0, "fx_effect_on_cash": -1.0},
}
OPTIONAL_FIGURES = {"fx_effect_on_cash"}  # taken as 0 when absent (no exchange-rate line)


def applicable_identities(values: dict[str, float]) -> dict[str, dict[str, float]]:
    """The statement identities whose figures are all in ``values`` (optional figures may be absent)."""
    out = {}
    for name, coeffs in STATEMENT_IDENTITIES.items():
        needed = {k: c for k, c in coeffs.items() if k in values or k not in OPTIONAL_FIGURES}
        if all(k in values for k in needed):
            out[name] = needed
    return out


def _q(x: float) -> z3.ArithRef:
    return z3.RealVal(str(x))


def _frac(v: z3.ExprRef) -> Fraction:
    """Exact value of a Z3 numeral (integer or rational)."""
    if isinstance(v, z3.IntNumRef):
        return Fraction(v.as_long())
    if isinstance(v, z3.RatNumRef):
        return Fraction(v.numerator_as_long(), v.denominator_as_long())
    raise ValueError(f"not an exact numeral: {v}")


def _float(v: z3.ExprRef) -> float:
    return float(_frac(v))


class DiagnosisCandidate(BaseModel):
    """A smallest set of figures that, if wrong, would explain every failed identity."""

    figures: list[str]
    implied_values: dict[str, float] | None = Field(
        default=None, description="the values that make the failed identities hold exactly; None if not unique")


class IdentityDiagnosis(BaseModel):
    """Which figures could be wrong, when identities on extracted or projected figures fail."""

    consistent: bool
    failed_identities: list[str] = Field(default_factory=list)
    residuals: dict[str, float] = Field(
        default_factory=dict, description="sum(coefficient * figure) per identity, in input units")
    candidates: list[DiagnosisCandidate] = Field(
        default_factory=list, description="every smallest explanation found (one if the identities pin it down)")
    conflict: list[str] = Field(
        default_factory=list, description="an unsatisfiable core: inputs and identities that cannot all hold")
    detail: str = ""


class DebtRepaymentResult(BaseModel):
    """Smallest debt repayment, funded from outside the cash balance, meeting both maintenance covenants."""

    required_repayment: float = Field(..., description="in the units of total_debt; 0 if already compliant")
    binding: list[str] = Field(default_factory=list, description="covenants at their limit after the repayment")
    net_leverage_after: float
    interest_coverage_after: float | None = Field(default=None, description="None when no interest remains")
    detail: str = ""


class CovenantRangeResult(BaseModel):
    """Whether two maintenance covenants hold for every combination of inputs in the given ranges."""

    holds_for_all: bool
    counterexample: dict[str, float] | None = Field(
        default=None, description="inputs within the ranges at which a covenant is breached")
    violated: list[str] = Field(default_factory=list)
    detail: str = ""


def diagnose_identities(
    values: dict[str, float],
    identities: dict[str, dict[str, float]] | None = None,
    tolerance: float = 0.5,
    max_candidates: int = 10,
) -> IdentityDiagnosis:
    """Find the smallest sets of figures that, if wrong, would explain every failed identity.

    ``identities`` maps a name to coefficients (each reads sum(c * figure) == 0 within ``tolerance``,
    in input units); by default the statement identities whose figures are in ``values``
    (``applicable_identities``). A diagnosis is a smallest set of figures whose change makes every
    identity hold (Reiter 1987: a minimal hitting set of the conflicts). It is found with Z3's MaxSMT
    optimiser: each figure keeping its value is a soft constraint, every identity a hard one.

    When a figure appears in two failing identities, a single candidate usually remains (identified);
    when one identity fails, every figure in it explains the failure equally well, and all are listed:
    the arithmetic cannot say which was misread, only the source can. Implied values are those that
    make the failed identities exact with all other figures unchanged, reported only when unique.
    ``conflict`` is an unsatisfiable core from a separate solve (inputs and identities that cannot all
    hold); it explains, the candidates prescribe.
    """
    if identities is None:
        identities = applicable_identities(values)
    if not identities:
        raise ValueError("No identity applies: pass identities, or values for at least one statement identity.")
    figures = sorted({k for coeffs in identities.values() for k in coeffs})
    missing = [k for k in figures if k not in values]
    if missing:
        raise ValueError(f"Values missing for figures used by the identities: {', '.join(missing)}")
    _require_finite(tolerance=tolerance, **{k: values[k] for k in figures})
    for name, coeffs in identities.items():
        _require_finite(**{f"{name}:{k}": c for k, c in coeffs.items()})
    if tolerance < 0 or max_candidates < 1:
        raise ValueError("tolerance must be >= 0 and max_candidates >= 1")

    x = {k: z3.Real(k) for k in figures}
    tol = _q(tolerance)

    def expr(coeffs: dict[str, float]) -> z3.ArithRef:
        return z3.Sum([_q(c) * x[k] for k, c in coeffs.items()])

    def holds(coeffs: dict[str, float]) -> z3.BoolRef:
        e = expr(coeffs)
        return z3.And(e <= tol, e >= -tol)

    given = {k: Fraction(str(values[k])) for k in figures}
    residuals = {name: sum((Fraction(str(c)) * given[k] for k, c in coeffs.items()), Fraction(0))
                 for name, coeffs in identities.items()}
    failed = [n for n, r in residuals.items() if abs(r) > Fraction(str(tolerance))]
    res_out = {n: float(r) for n, r in residuals.items()}
    if not failed:
        return IdentityDiagnosis(consistent=True, residuals=res_out, detail="every identity holds on the given figures")

    # Unsatisfiable core, for explanation.
    s = z3.Solver()
    s.set("core.minimize", True)
    for k in figures:
        s.assert_and_track(x[k] == _q(values[k]), f"input:{k}")
    for name, coeffs in identities.items():
        s.assert_and_track(holds(coeffs), f"identity:{name}")
    conflict = sorted(str(c) for c in s.unsat_core()) if s.check() == z3.unsat else []

    # Smallest diagnoses (MaxSMT), enumerated by blocking each one found.
    opt = z3.Optimize()
    for coeffs in identities.values():
        opt.add(holds(coeffs))
    for k in figures:
        opt.add_soft(x[k] == _q(values[k]), 1)
    candidates: list[list[str]] = []
    best: int | None = None
    while len(candidates) < max_candidates and opt.check() == z3.sat:
        m = opt.model()
        changed = sorted(k for k in figures if not z3.is_true(m.eval(x[k] == _q(values[k]), model_completion=True)))
        if best is None:
            best = len(changed)
        if len(changed) > best or not changed:
            break
        candidates.append(changed)
        opt.add(z3.Or([x[k] == _q(values[k]) for k in changed]))  # block this set and its supersets

    def implied(changed: list[str]) -> dict[str, float] | None:
        sol = z3.Solver()
        for k in figures:
            if k not in changed:
                sol.add(x[k] == _q(values[k]))
        for name, coeffs in identities.items():
            sol.add(expr(coeffs) == 0 if name in failed else holds(coeffs))
        if sol.check() != z3.sat:
            return None
        m = sol.model()
        vals = {k: m.eval(x[k], model_completion=True) for k in changed}
        sol.add(z3.Or([x[k] != v for k, v in vals.items()]))
        return {k: _float(v) for k, v in vals.items()} if sol.check() == z3.unsat else None

    candidates.sort(key=lambda c: (len(c), c))  # deterministic order
    out = [DiagnosisCandidate(figures=c, implied_values=implied(c)) for c in candidates]
    detail = (f"{len(failed)} identit{'y' if len(failed) == 1 else 'ies'} failed; "
              + ("one smallest explanation: " + ", ".join(candidates[0]) if len(out) == 1 else
                 f"{len(out)} equally small explanations; the identities cannot tell them apart"))
    return IdentityDiagnosis(consistent=False, failed_identities=failed, residuals=res_out, candidates=out,
                             conflict=conflict, detail=detail)


def minimum_debt_repayment(
    total_debt: float,
    cash: float,
    ebitda: float,
    interest_expense: float,
    max_net_leverage: float,
    min_interest_coverage: float,
) -> DebtRepaymentResult:
    """Smallest debt repayment that meets net leverage <= max and interest coverage >= min together.

    Model (stated so it can be checked): the repayment r comes from outside the cash balance (new
    equity, asset-sale proceeds), so cash is unchanged; repaying from existing cash would leave net
    debt, and so net leverage, unchanged. Interest falls in proportion to debt,
    interest_after = interest_expense * (D - r) / D. Whether and how a sponsor's equity cure counts
    (added to EBITDA, or applied to repay debt) is set by each credit agreement; this answers the
    repayment question only.
    Net leverage = (D - r - cash) / EBITDA; coverage = EBITDA / interest_after. A ratio at its limit
    complies. Solved exactly in rationals with ``z3.Optimize`` (minimise r subject to both covenants
    and 0 <= r <= D); with EBITDA > 0 a solution always exists (r = D). Closed form, used as the reference
    test: r* = max(0, D - cash - max_lev * EBITDA, D * (1 - EBITDA / (min_cov * interest))).
    Raises ``ValueError`` if EBITDA <= 0 (leverage undefined), debt <= 0, or an input is negative.
    """
    _require_finite(total_debt=total_debt, cash=cash, ebitda=ebitda, interest_expense=interest_expense,
                    max_net_leverage=max_net_leverage, min_interest_coverage=min_interest_coverage)
    if ebitda <= 0:
        raise ValueError("EBITDA must be positive: net leverage is undefined otherwise, and no repayment cures it.")
    if total_debt <= 0 or cash < 0 or interest_expense < 0 or max_net_leverage < 0 or min_interest_coverage < 0:
        raise ValueError("total_debt must be positive; cash, interest, max leverage and min coverage non-negative.")
    d, c, e, i = _q(total_debt), _q(cash), _q(ebitda), _q(interest_expense)
    lmax, cmin = _q(max_net_leverage), _q(min_interest_coverage)
    r = z3.Real("repayment")
    opt = z3.Optimize()
    opt.add(r >= 0, r <= d, d - r - c <= lmax * e, e * d >= cmin * i * (d - r))
    h = opt.minimize(r)
    if opt.check() != z3.sat:  # pragma: no cover - r = D always satisfies both constraints
        raise RuntimeError("no repayment found; this should not happen for EBITDA > 0")
    r_star = _frac(h.value())
    debt_q, cash_q, ebitda_q, int_q = (Fraction(str(v)) for v in (total_debt, cash, ebitda, interest_expense))
    lev_after = (debt_q - r_star - cash_q) / ebitda_q
    int_after = int_q * (debt_q - r_star) / debt_q
    cov_after = ebitda_q / int_after if int_after > 0 else None
    binding = []
    if r_star > 0 and lev_after == Fraction(str(max_net_leverage)):
        binding.append("net leverage")
    if r_star > 0 and cov_after is not None and cov_after == Fraction(str(min_interest_coverage)):
        binding.append("interest coverage")
    return DebtRepaymentResult(
        required_repayment=float(r_star), binding=binding, net_leverage_after=float(lev_after),
        interest_coverage_after=None if cov_after is None else float(cov_after),
        detail="already compliant" if r_star == 0 else
        "repayment from outside the cash balance; interest assumed proportional to debt outstanding")


def check_covenants_over_ranges(
    total_debt: tuple[float, float],
    cash: tuple[float, float],
    ebitda: tuple[float, float],
    interest_expense: tuple[float, float],
    max_net_leverage: float,
    min_interest_coverage: float,
) -> CovenantRangeResult:
    """Do both covenants hold for every combination of inputs within the (low, high) ranges?

    Z3 searches for a breach: (debt - cash) / EBITDA > max_leverage, or EBITDA / interest <
    min_coverage, with each input anywhere in its range (ranges are independent). No breach found
    (unsat) means the covenants hold for all inputs in the ranges, a statement over ranges rather than
    a point check; otherwise a counterexample is returned. Because EBITDA > 0 on the range, the ratios
    are compared in multiplied-out linear form. A ratio at its limit complies. Reference test: both
    ratios are monotone in each input, so the worst corner (highest debt and interest, lowest cash and
    EBITDA) decides. Raises ``ValueError`` for a range with low > high, EBITDA low <= 0 or negative
    interest.
    """
    ranges = {"total_debt": total_debt, "cash": cash, "ebitda": ebitda, "interest_expense": interest_expense}
    for name, (lo, hi) in ranges.items():
        _require_finite(**{f"{name}_low": lo, f"{name}_high": hi})
        if lo > hi:
            raise ValueError(f"{name}: low {lo} is above high {hi}")
    _require_finite(max_net_leverage=max_net_leverage, min_interest_coverage=min_interest_coverage)
    if ebitda[0] <= 0:
        raise ValueError("EBITDA must be positive over the whole range (leverage is undefined otherwise).")
    if interest_expense[0] < 0:
        raise ValueError("interest_expense must be non-negative.")
    v = {k: z3.Real(k) for k in ranges}
    s = z3.Solver()
    for k, (lo, hi) in ranges.items():
        s.add(v[k] >= _q(lo), v[k] <= _q(hi))
    lev_breach = v["total_debt"] - v["cash"] > _q(max_net_leverage) * v["ebitda"]
    cov_breach = v["ebitda"] < _q(min_interest_coverage) * v["interest_expense"]
    s.add(z3.Or(lev_breach, cov_breach))
    if s.check() == z3.unsat:
        return CovenantRangeResult(holds_for_all=True,
                                   detail="no combination of inputs within the ranges breaches either covenant")
    m = s.model()
    ce = {k: _float(m.eval(v[k], model_completion=True)) for k in ranges}
    violated = [name for name, cond in (("net leverage", lev_breach), ("interest coverage", cov_breach))
                if z3.is_true(m.eval(cond, model_completion=True))]
    return CovenantRangeResult(holds_for_all=False, counterexample=ce, violated=violated,
                               detail="a combination of inputs within the ranges breaches: see counterexample")
