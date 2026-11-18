"""Reference tests for core/invariants.py.

Expected values are hand-computed from the accounting identities
(Assets = Liabilities + Equity; CashEnd = CashStart + CFO + CFI + CFF) and from the
covenant ratio definitions. Apple FY2026 balance-sheet totals are from the 10-K
(total assets $404,548M) and SEC XBRL; liabilities/equity split chosen to sum exactly.
"""

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
