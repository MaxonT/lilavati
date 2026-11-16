"""Unit tests for Tutorial Agents (07 Stat-Arb, 08 Forensics, 09 Sustainable Portfolio)."""

import importlib
from typing import Any

from fastapi.testclient import TestClient

from lilavati.framework import create_agent_api

# Dynamic import for modules with numeric prefixes
mod_07 = importlib.import_module("examples.07_stat_arb_and_execution_agent")
StatArbExecutionState = mod_07.StatArbExecutionState
build_stat_arb_agent = mod_07.build_stat_arb_agent

mod_08 = importlib.import_module("examples.08_forensic_committee_agent")
ForensicCommitteeState = mod_08.ForensicCommitteeState
build_forensic_committee_agent = mod_08.build_forensic_committee_agent

mod_09 = importlib.import_module("examples.09_climate_sustainable_portfolio_agent")
SustainablePortfolioState = mod_09.SustainablePortfolioState
build_sustainable_portfolio_agent = mod_09.build_sustainable_portfolio_agent


def test_stat_arb_agent_approval() -> None:
    graph = build_stat_arb_agent().compile()
    state = StatArbExecutionState(
        shares_to_liquidate=5000.0,
        execution_horizon_hours=2.0,
    )
    result = graph.invoke(state)

    assert result.is_cointegrated is True
    assert result.half_life_days > 0.0
    assert result.optimal_fractional_d > 0.0
    assert result.microprice > 0.0
    assert result.expected_execution_cost > 0.0
    assert "APPROVED_PAIR_TRADE" in result.trade_verdict


def test_stat_arb_agent_risk_rejection() -> None:
    graph = build_stat_arb_agent().compile()
    state = StatArbExecutionState(
        shares_to_liquidate=50000.0,
        max_capital_risk_limit=1000000.0,
    )
    result = graph.invoke(state)

    assert "REJECTED_RISK_LIMIT" in result.trade_verdict
    assert result.expected_execution_cost == 0.0


def test_forensic_committee_manipulator_rejection() -> None:
    graph = build_forensic_committee_agent().compile()
    state = ForensicCommitteeState(
        ticker="TEST-FRAUD",
        receivables_t=220.0,
        receivables_t_minus_1=100.0,
        cfo_t=40.0,
        net_income_t=140.0,
    )
    result = graph.invoke(state)

    assert result.beneish_m_score > -1.78
    assert result.is_likely_manipulator is True
    assert "REJECTED" in result.pm_decision


def test_forensic_committee_clean_approval() -> None:
    graph = build_forensic_committee_agent().compile()
    state = ForensicCommitteeState(
        ticker="TEST-CLEAN",
        receivables_t=110.0,
        receivables_t_minus_1=100.0,
        cfo_t=150.0,
        net_income_t=120.0,
        current_liabilities_t=160.0,
        long_term_debt_t=350.0,
        shareholders_equity_t=840.0,
        total_assets_t=1350.0,
        cash_start_t=80.0,
        cff_t=-60.0,
        cfi_t=-50.0,
        cash_end_t=120.0,
    )
    result = graph.invoke(state)

    assert result.is_accounting_valid is True
    assert result.is_likely_manipulator is False
    assert "APPROVED_ALLOCATION" in result.pm_decision


def test_sustainable_portfolio_waci_rejection() -> None:
    graph = build_sustainable_portfolio_agent().compile()
    state = SustainablePortfolioState()
    result = graph.invoke(state)

    assert len(result.hrp_weights) == 4
    assert result.portfolio_waci > 100.0
    assert result.is_waci_paris_compliant is False
    assert "REJECTED_WACI_EXCEEDED" in result.allocation_verdict


def test_sustainable_portfolio_clean_approval() -> None:
    graph = build_sustainable_portfolio_agent().compile()
    state = SustainablePortfolioState(
        asset_emissions_tons=[10000.0, 20000.0, 15000.0, 50000.0],
        paris_aligned_waci_limit=100.0,
    )
    result = graph.invoke(state)

    assert result.portfolio_waci <= 100.0
    assert result.is_waci_paris_compliant is True
    assert "APPROVED_SUSTAINABLE_ALLOCATION" in result.allocation_verdict


def test_tutorial_agents_rest_api() -> None:
    stat_arb_graph = build_stat_arb_agent().compile()
    app = create_agent_api(stat_arb_graph, title="StatArb API")
    client = TestClient(app)
    response = client.post("/api/v1/run", json={"shares_to_liquidate": 2000.0})
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert "trade_verdict" in data
