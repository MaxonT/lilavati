"""Reference tests for agents/forensics.py Altman Z-score.

Source: Altman (1968), "Financial Ratios, Discriminant Analysis and the Prediction of Corporate
Bankruptcy", J. Finance 23(4): Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 0.999 X5 (ratios as
decimals); zones: < 1.81 distress, 1.81-2.99 grey, > 2.99 safe. Beneish tests: tests/test_beneish.py.
"""

import pytest

from lilavati.agents.forensics import ForensicEngine


def _z(**kw: float) -> tuple[float, str]:
    base = {"working_capital": 100.0, "total_assets": 1000.0, "retained_earnings": 200.0, "ebit": 100.0,
                "market_value_equity": 900.0, "total_liabilities": 600.0, "sales": 1200.0}
    base.update(kw)
    r = ForensicEngine.calculate_altman_z_score(**base)
    return r.z_score, r.distress_zone


def test_hand_computed_z() -> None:
    # X = (0.1, 0.2, 0.1, 1.5, 1.2) -> 0.12 + 0.28 + 0.33 + 0.90 + 1.1988 = 2.8288 (grey)
    z, zone = _z()
    assert z == pytest.approx(2.8288, abs=1e-12)
    assert zone == "Grey"


def test_zones() -> None:
    assert _z(market_value_equity=100.0)[1] == "Grey"  # Z = 2.0288
    assert _z(ebit=200.0, market_value_equity=1200.0)[1] == "Safe"


def test_distress_zone_value() -> None:
    z, zone = _z(market_value_equity=0.0, sales=500.0)
    # 0.12 + 0.28 + 0.33 + 0 + 0.4995 = 1.2295
    assert z == pytest.approx(1.2295, abs=1e-12) and zone == "Distress"
