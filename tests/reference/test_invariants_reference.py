"""Reference tests for core/invariants.py.

Expected values are hand-computed from the accounting identities
(Assets = Liabilities + Equity; CashEnd = CashStart + CFO + CFI + CFF) and from the
covenant ratio definitions. Apple FY2026 balance-sheet totals are from the 10-K
(total assets $404,548M) and SEC XBRL; liabilities/equity split chosen to sum exactly.
"""

import numpy as np
import pytest

from lilavati.core.invariants import SuvarnaEngine


def test_exact_decimal_semantics() -> None:
    # 0.1 + 0.2 != 0.3 in binary floats, but inputs are taken as decimals: 0.3 = 0.1 + 0.2.
    assert SuvarnaEngine.verify_balance_sheet_equation(0.3, 0.1, 0.2, tolerance=0.0).is_valid


def test_large_magnitudes_balance() -> None:
    r = SuvarnaEngine.verify_balance_sheet_equation(404_548e6, 285_508e6, 119_040e6, 0.0)
    assert r.is_valid and r.status == "SAT"


def test_tolerance_boundary() -> None:
    assert SuvarnaEngine.verify_balance_sheet_equation(100.0, 60.0, 40.00009).is_valid
    assert not SuvarnaEngine.verify_balance_sheet_equation(100.0, 60.0, 40.0002).is_valid


def test_three_statement_reports_each_violation() -> None:
    r = SuvarnaEngine.verify_three_statement_invariants(
        total_assets=100, total_liabilities=60, shareholders_equity=39, net_income=5,
        cash_start=10, cash_end=12, cfo=5, cfi=-2, cff=-2,
    )
    assert not r.is_valid
    assert len(r.violations) == 2  # BS off by 1, cash off by 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_raise_clear_error(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SuvarnaEngine.verify_balance_sheet_equation(bad, 1.0, 1.0)


def test_covenant_exactly_at_limit_complies() -> None:
    # Net debt 2.1 / EBITDA 0.7 = 3.0x exactly; float division gives 3.0000000000000004.
    r = SuvarnaEngine.verify_debt_covenant(
        total_debt=2.1, cash=0.0, ebitda=0.7, max_leverage_covenant=3.0,
        min_interest_coverage_covenant=0.0, interest_expense=0.0,
    )
    assert r.is_valid


def test_covenant_breach_and_headroom() -> None:
    # Net leverage (545 - 0) / 100 = 5.45x vs 4.50x max; coverage 100/40 = 2.5x vs 3.0x min.
    r = SuvarnaEngine.verify_debt_covenant(
        total_debt=545.0, cash=0.0, ebitda=100.0, max_leverage_covenant=4.5,
        min_interest_coverage_covenant=3.0, interest_expense=40.0,
    )
    assert not r.is_valid and len(r.violations) == 2
    assert r.model_values["current_leverage"] == pytest.approx(5.45)
    assert r.model_values["headroom_leverage"] == pytest.approx(-0.95)


def test_nonpositive_ebitda_fails() -> None:
    r = SuvarnaEngine.verify_debt_covenant(100.0, 0.0, 0.0, 4.5, 2.0, 10.0)
    assert not r.is_valid


# --- Diagnosis, debt repayment and range checks -------------------------------------------------
# Apple FY2026 10-K figures (USD millions), checked against the statements in docs/validation/connectors.md.
APPLE = {"total_assets": 404_548, "total_liabilities": 292_455, "total_equity": 112_093,
         "liabilities_and_equity": 404_548, "revenue": 478_834, "cost_of_revenue": 245_232,
         "gross_profit": 233_602, "cash_opening": 35_934, "operating_cash_flow": 145_097,
         "investing_cash_flow": -17_280, "financing_cash_flow": -127_446, "cash_closing": 36_305}


def test_diagnosis_of_consistent_figures() -> None:
    from lilavati.core.invariants import diagnose_identities

    d = diagnose_identities(APPLE)
    assert d.consistent and d.candidates == [] and all(r == 0 for r in d.residuals.values())


def test_diagnosis_identifies_a_figure_used_by_two_failing_identities() -> None:
    # A typo in total assets breaks both A = L + E and A = total liabilities and equity. Only total
    # assets is in both (Reiter 1987: the single minimal hitting set of the two conflicts), and the
    # implied value is the 10-K's.
    from lilavati.core.invariants import diagnose_identities

    d = diagnose_identities({**APPLE, "total_assets": 404_458})
    assert d.failed_identities == ["assets = liabilities + equity", "assets = total liabilities and equity"]
    assert [c.figures for c in d.candidates] == [["total_assets"]]
    assert d.candidates[0].implied_values == {"total_assets": 404_548.0}


def test_diagnosis_lists_every_explanation_when_one_identity_fails() -> None:
    # A typo in closing cash breaks only the cash roll-forward: each of its five figures explains the
    # failure equally well, with the value it would need (e.g. CFO 145,097 + 45).
    from lilavati.core.invariants import diagnose_identities

    d = diagnose_identities({**APPLE, "cash_closing": 36_350})
    assert sorted(c.figures[0] for c in d.candidates) == sorted(
        ["cash_closing", "cash_opening", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow"])
    implied = {c.figures[0]: c.implied_values for c in d.candidates}
    assert implied["cash_closing"] == {"cash_closing": 36_305.0}
    assert implied["operating_cash_flow"] == {"operating_cash_flow": 145_097.0 + 45}


def test_diagnosis_excludes_a_figure_that_would_break_a_holding_identity() -> None:
    # A typo in equity breaks only A = L + E; total assets is in that identity too, but changing it
    # would break A = total liabilities and equity, so it is not a smallest explanation.
    from lilavati.core.invariants import diagnose_identities

    d = diagnose_identities({**APPLE, "total_equity": 112_039})
    assert [c.figures for c in d.candidates] == [["total_equity"], ["total_liabilities"]]
    assert d.candidates[1].implied_values == {"total_liabilities": 292_455.0 + 54}  # 404,548 - 112,039


def test_diagnosis_needs_two_figures_when_two_independent_errors() -> None:
    from lilavati.core.invariants import diagnose_identities

    d = diagnose_identities({**APPLE, "total_assets": 404_458, "gross_profit": 233_000})
    assert all(len(c.figures) == 2 and "total_assets" in c.figures for c in d.candidates)
    assert {c.figures[0] if c.figures[1] == "total_assets" else c.figures[1] for c in d.candidates} == {
        "revenue", "cost_of_revenue", "gross_profit"}


def test_diagnosis_custom_identity_and_tolerance() -> None:
    from lilavati.core.invariants import diagnose_identities

    ident = {"subtotal": {"a": 1.0, "b": 1.0, "total": -1.0}}
    assert diagnose_identities({"a": 1.0, "b": 2.0, "total": 3.4}, ident, tolerance=0.5).consistent
    d = diagnose_identities({"a": 1.0, "b": 2.0, "total": 3.6}, ident, tolerance=0.5)
    assert not d.consistent and len(d.candidates) == 3
    with pytest.raises(ValueError, match="missing"):
        diagnose_identities({"a": 1.0}, ident)


@pytest.mark.parametrize("seed", range(40))
def test_debt_repayment_matches_the_closed_form(seed: int) -> None:
    # r* = max(0, D - cash - Lmax*E, D*(1 - E/(Cmin*I))) under the stated model (repayment from
    # outside the cash balance, interest proportional to debt). Exact rationals on both sides.
    from fractions import Fraction

    from lilavati.core.invariants import minimum_debt_repayment

    rng = np.random.default_rng(seed)
    d, cash, e, i = (float(round(x)) for x in (rng.uniform(100, 1000), rng.uniform(0, 200), rng.uniform(20, 300),
                                               rng.uniform(1, 80)))
    lmax, cmin = float(round(rng.uniform(1.5, 6.0), 2)), float(round(rng.uniform(1.0, 6.0), 2))
    res = minimum_debt_repayment(d, cash, e, i, lmax, cmin)
    dq, cq, eq, iq, lq, kq = (Fraction(str(x)) for x in (d, cash, e, i, lmax, cmin))
    expected = max(Fraction(0), dq - cq - lq * eq, dq * (1 - eq / (kq * iq)))
    assert Fraction(res.required_repayment).limit_denominator(10**9) == expected or abs(
        res.required_repayment - float(expected)) < 1e-9 * max(1.0, float(expected))
    assert res.net_leverage_after <= lmax + 1e-12
    assert res.interest_coverage_after is None or res.interest_coverage_after >= cmin - 1e-12


def test_debt_repayment_exact_case_and_compliant_case() -> None:
    from lilavati.core.invariants import minimum_debt_repayment

    # D 500, cash 50, EBITDA 100, interest 40, max 3.5x, min 3.0x: leverage needs 500-50-350 = 100,
    # coverage needs 500*(1-100/120) = 83.33; the repayment is 100, leverage binds, coverage 100/32.
    r = minimum_debt_repayment(500, 50, 100, 40, 3.5, 3.0)
    assert r.required_repayment == 100.0 and r.binding == ["net leverage"]
    assert r.net_leverage_after == 3.5 and r.interest_coverage_after == 3.125
    assert minimum_debt_repayment(300, 50, 100, 20, 3.5, 3.0).required_repayment == 0.0
    with pytest.raises(ValueError, match="EBITDA must be positive"):
        minimum_debt_repayment(500, 50, 0, 40, 3.5, 3.0)


@pytest.mark.parametrize("seed", range(60))
def test_covenant_ranges_match_the_worst_corner(seed: int) -> None:
    # Both ratios are monotone in each input, so the covenants hold over the box iff they hold at the
    # corner with the highest debt and interest and the lowest cash and EBITDA.
    from fractions import Fraction

    from lilavati.core.invariants import check_covenants_over_ranges

    rng = np.random.default_rng(1000 + seed)

    def box(lo: float, hi: float) -> tuple[float, float]:
        a, b = sorted(float(round(x)) for x in rng.uniform(lo, hi, 2))
        return a, b

    debt, cash, ebitda, interest = box(200, 800), box(0, 150), box(50, 250), box(5, 60)
    lmax, cmin = 4.0, 3.0
    res = check_covenants_over_ranges(debt, cash, ebitda, interest, lmax, cmin)
    worst_ok = (Fraction(str(debt[1])) - Fraction(str(cash[0])) <= Fraction(str(lmax)) * Fraction(str(ebitda[0]))
                and Fraction(str(ebitda[0])) >= Fraction(str(cmin)) * Fraction(str(interest[1])))
    assert res.holds_for_all == worst_ok
    if not res.holds_for_all:
        ce = res.counterexample
        assert ce is not None
        lev_breach = (ce["total_debt"] - ce["cash"]) / ce["ebitda"] > lmax
        cov_breach = ce["interest_expense"] > 0 and ce["ebitda"] / ce["interest_expense"] < cmin
        assert lev_breach or cov_breach


def test_covenant_ranges_at_the_limit_comply() -> None:
    from lilavati.core.invariants import check_covenants_over_ranges

    # Worst corner exactly at both limits: (530 - 50) / 120 = 4.0 and 120 / 40 = 3.0, exact in decimal.
    # A ratio at its limit complies. (Inputs are read as exact decimals: 100/3 entered as a float is
    # slightly above one third, so a limit built from it is not exactly met.)
    assert check_covenants_over_ranges((400, 530), (50, 60), (120, 140), (20, 40), 4.0, 3.0).holds_for_all
    # One unit beyond either limit breaches.
    over_lev = check_covenants_over_ranges((400, 531), (50, 60), (120, 140), (20, 40), 4.0, 3.0)
    assert not over_lev.holds_for_all and over_lev.violated == ["net leverage"]
    over_cov = check_covenants_over_ranges((400, 530), (50, 60), (120, 140), (20, 41), 4.0, 3.0)
    assert not over_cov.holds_for_all and "interest coverage" in over_cov.violated
    with pytest.raises(ValueError, match="low"):
        check_covenants_over_ranges((450, 400), (50, 60), (100, 140), (20, 30), 4.0, 3.0)
