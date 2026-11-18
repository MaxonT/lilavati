"""Reference tests for core/sustainability.py.

Sources: TCFD (2017) "Implementing the Recommendations", Table 2 (WACI definition);
ICMA "Sustainability-Linked Bond Principles" (2023) for the step-up mechanism.
Expected values below are hand-computed from those definitions.
"""

import pytest

from lilavati.core.sustainability import SustainabilityEngine
from lilavati.schemas.sustainability import SustainabilityLinkedBond


def _bond(target: float = 0.20, baseline: float = 100_000.0) -> SustainabilityLinkedBond:
    return SustainabilityLinkedBond(
        isin="XS0000000000", notional=50_000_000.0, base_coupon_rate=0.04,
        target_reduction_pct=target, baseline_emissions=baseline, step_up_bps=25.0,
        target_year=2026,
    )


def test_waci_hand_computed_tcfd() -> None:
    # 0.6 * 50000/500 + 0.4 * 120000/800 = 60 + 60 = 120 tCO2e / $M
    r = SustainabilityEngine.calculate_waci([0.6, 0.4], [50_000, 120_000], [500, 800])
    assert r.waci == pytest.approx(120.0, rel=1e-12)
    assert r.carbon_intensity_clean is False  # threshold 100


def test_waci_weights_are_normalised() -> None:
    a = SustainabilityEngine.calculate_waci([6, 4], [50_000, 120_000], [500, 800]).waci
    assert a == pytest.approx(120.0, rel=1e-12)


@pytest.mark.parametrize("weights", [[0.0, 0.0], [1.0, -0.5]])
def test_waci_rejects_invalid_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        SustainabilityEngine.calculate_waci(weights, [1.0, 1.0], [1.0, 1.0])


def test_carbon_tax_hand_computed() -> None:
    # (8000 + 2000) t * $100 * (1 - 0.5) = $500,000 absorbed; EBIT 2.0M -> 1.5M
    r = SustainabilityEngine.stress_test_carbon_tax(
        revenue=10_000_000, ebit=2_000_000, interest_expense=500_000,
        scope1_emissions=8_000, scope2_emissions=2_000, carbon_price_per_ton=100.0,
        pass_through_rate=0.5, covenant_interest_floor=3.5,
    )
    assert r.net_carbon_expense == pytest.approx(500_000)
    assert r.ebit_margin_before_pct == pytest.approx(0.20)
    assert r.ebit_margin_after_pct == pytest.approx(0.15)
    assert r.ebit_reduction_pct == pytest.approx(0.25)
    assert r.interest_coverage_after == pytest.approx(3.0)
    assert r.is_covenant_at_risk is True


def test_full_pass_through_has_no_ebit_impact() -> None:
    r = SustainabilityEngine.stress_test_carbon_tax(
        revenue=1e6, ebit=1e5, interest_expense=1e4, scope1_emissions=1e3,
        scope2_emissions=0.0, carbon_price_per_ton=200.0, pass_through_rate=1.0,
    )
    assert r.ebit_reduction_pct == 0.0


def test_slb_missed_target_steps_up_coupon() -> None:
    r = SustainabilityEngine.evaluate_slb_covenant(_bond(), reported_emissions=90_000)
    assert r.spt_achieved is False
    assert r.active_coupon_rate == pytest.approx(0.0425)
    assert r.annual_interest_penalty == pytest.approx(125_000.0)  # 50M * 25bp


def test_slb_target_met_exactly_is_not_missed_by_rounding() -> None:
    # 10% cut from 3.0 is exactly 2.7, but (3 - 2.7) / 3 evaluates to 0.09999999999999994 < 0.1
    # in floating point; the old ratio comparison wrongly reported a missed target.
    r = SustainabilityEngine.evaluate_slb_covenant(
        _bond(target=0.10, baseline=3.0), reported_emissions=2.7
    )
    assert r.spt_achieved is True
    assert r.active_coupon_rate == pytest.approx(0.04)


def test_slb_rejects_nonpositive_baseline() -> None:
    with pytest.raises(ValueError):
        _bond(baseline=0.0)
