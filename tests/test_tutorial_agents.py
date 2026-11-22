"""Tests for the companion scripts of Tutorials 02-04 (examples 07, 08, 09).

The tutorials' own code is executed by scripts/check_tutorials.py; these tests pin the example
scripts, which must stay identical in behaviour to the tutorials.
"""

import importlib
import math
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lilavati.framework import create_agent_api

mod_07 = importlib.import_module("examples.07_stat_arb_and_execution_agent")
mod_08 = importlib.import_module("examples.08_forensic_committee_agent")
mod_09 = importlib.import_module("examples.09_climate_sustainable_portfolio_agent")


def test_stat_arb_recovers_the_simulated_process_and_approves() -> None:
    r = mod_07.build_stat_arb_agent().compile().invoke(mod_07.StatArbExecutionState())
    assert r.is_cointegrated
    assert r.hedge_ratio == pytest.approx(mod_07.TRUE_BETA, rel=0.05)
    assert r.half_life_days == pytest.approx(-math.log(2) / math.log(1 - mod_07.TRUE_THETA), rel=0.10)
    assert r.expected_execution_cost > 0.0
    assert r.trade_verdict.startswith("APPROVE")


def test_stat_arb_risk_limit_rejects_before_scheduling() -> None:
    r = mod_07.build_stat_arb_agent().compile().invoke(
        mod_07.StatArbExecutionState(shares_to_liquidate=50_000.0, max_order_notional=1_000_000.0))
    assert r.trade_verdict.startswith("REJECT: order notional")
    assert r.expected_execution_cost == 0.0


def test_forensic_default_firm_fails_the_manipulation_screen() -> None:
    r = mod_08.build_forensic_committee().compile().invoke(mod_08.ForensicCommitteeState())
    assert r.accounting_identities_hold and r.beneish_flag
    assert r.decision.startswith("REJECT: manipulation")


def test_forensic_inconsistent_statements_are_rejected_as_such() -> None:
    # The firm this test once called a "manipulator" had a cash flow statement that did not
    # reconcile; it is rejected for inconsistency, before any score is used.
    r = mod_08.build_forensic_committee().compile().invoke(
        mod_08.ForensicCommitteeState(receivables_t=220.0, cfo_t=40.0, net_income_t=140.0))
    assert not r.accounting_identities_hold
    assert r.decision.startswith("REJECT: reported statements are inconsistent")


def test_forensic_clean_firm_passes() -> None:
    r = mod_08.build_forensic_committee().compile().invoke(mod_08.ForensicCommitteeState(**mod_08.CLEAN))
    assert r.accounting_identities_hold and not r.beneish_flag
    assert r.decision.startswith("PASS SCREEN")


def test_climate_portfolio_rejects_on_waci_and_reports_the_slb_step_up() -> None:
    r = mod_09.build_climate_portfolio().compile().invoke(mod_09.ClimatePortfolioState())
    assert len(r.hrp_weights) == 4 and sum(r.hrp_weights) == pytest.approx(1.0)
    assert r.portfolio_waci > r.mandate_waci_limit and not r.within_waci_mandate
    assert r.decision.startswith("REJECT: WACI")
    assert not r.slb_target_met and r.slb_coupon == pytest.approx(0.045)


def test_climate_portfolio_passes_with_lower_emissions() -> None:
    r = mod_09.build_climate_portfolio().compile().invoke(
        mod_09.ClimatePortfolioState(emissions_tco2e=[10000.0, 20000.0, 15000.0, 50000.0]))
    assert r.within_waci_mandate and r.decision.startswith("PASS")


def test_tutorial_agents_rest_api() -> None:
    app = create_agent_api(mod_07.build_stat_arb_agent().compile(), title="StatArb API")
    response = TestClient(app).post("/api/v1/run", json={"shares_to_liquidate": 2000.0})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["trade_verdict"].startswith("APPROVE")
