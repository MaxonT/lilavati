"""
Companion script for Tutorial 01 (docs/tutorials/01_build_your_first_financial_agent.md):
a deterministic credit underwriting workflow, its tests' scenarios, and a local REST API.

1. A typed state container (`FinancialState`), money in USD millions.
2. Three nodes: provenance, maintenance covenants (net leverage and interest coverage, plain
   arithmetic with an at-limit tolerance), and a Merton (1974) one-year default probability.
3. A `FinancialStateGraph` that runs them in order.
4. Two borrowers: one passes, one breaches both covenants.
5. The same graph as a FastAPI app, called in-process with FastAPI's test client.

No LLM is involved: this is the deterministic layer an LLM agent would call as tools.

    uv run python examples/06_build_custom_agent_and_api.py
"""

from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef
from lilavati.framework import END, FinancialState, FinancialStateGraph, create_agent_api


class CreditUnderwritingState(FinancialState):
    """Typed state for the credit underwriting workflow. Money in USD millions."""

    borrower_name: str = "Apollo Industrial Corp"
    fiscal_year: int = 2026
    ebitda: float = 80.0  # USD m, last twelve months
    total_debt: float = 280.0  # USD m
    cash: float = 0.0  # USD m (net leverage uses debt minus cash)
    interest_expense: float = 18.0  # USD m, annual
    equity_value: float = 400.0  # USD m, market value of equity
    equity_volatility: float = 0.30  # annualised, decimal

    # Maintenance covenants from the credit agreement
    max_net_leverage: float = 4.00  # (debt - cash) / EBITDA <= 4.0x
    min_interest_coverage: float = 3.50  # EBITDA / interest >= 3.5x

    # Outputs
    net_leverage: float = 0.0
    interest_coverage: float = 0.0
    covenants_met: bool = False
    covenant_violations: list[str] = []
    debt_reduction_to_comply: float = 0.0  # USD m of net debt to repay to meet the leverage test
    merton_default_probability: float = 0.0
    underwriting_verdict: str = "PENDING"


def ingest_and_provenance_node(state: CreditUnderwritingState) -> CreditUnderwritingState:
    """Record where each input came from (illustrative document hash and pages)."""
    doc_hash = "a7c8e9b01234567890abcdef1234567890abcdef1234567890abcdef12345678"
    for page, label, value in (
        (24, "Adjusted Consolidated EBITDA", state.ebitda),
        (28, "Total Outstanding Indebtedness", state.total_debt),
    ):
        state.record_citation(
            CitationRef(
                doc_hash=doc_hash,
                filing_type="Credit-Agreement",
                filing_date=date(2026, 3, 15),
                period_end=date(2025, 12, 31),
                page=page,
                raw_snippet=f"{label}: ${value:.1f}M",
            )
        )
    state.record_fact("ebitda", state.ebitda)
    state.record_fact("total_debt", state.total_debt)
    return state


def covenant_check_node(state: CreditUnderwritingState) -> CreditUnderwritingState:
    """Check both maintenance covenants with the validated engine (plain arithmetic)."""
    result = SuvarnaEngine.verify_debt_covenant(
        total_debt=state.total_debt,
        cash=state.cash,
        ebitda=state.ebitda,
        max_leverage_covenant=state.max_net_leverage,
        min_interest_coverage_covenant=state.min_interest_coverage,
        interest_expense=state.interest_expense,
    )
    state.net_leverage = (state.total_debt - state.cash) / state.ebitda
    state.interest_coverage = state.ebitda / state.interest_expense
    state.covenants_met = result.is_valid
    state.covenant_violations = result.violations
    excess = (state.total_debt - state.cash) - state.max_net_leverage * state.ebitda
    state.debt_reduction_to_comply = max(0.0, excess)
    return state


def structural_credit_risk_node(state: CreditUnderwritingState) -> CreditUnderwritingState:
    """Merton (1974) one-year default probability, then a rule-based verdict."""
    merton = FixedIncomeEngine.solve_merton_model(
        equity_value=state.equity_value,
        equity_volatility=state.equity_volatility,
        debt_face_value=state.total_debt,
        risk_free_rate=0.045,
        maturity_years=1.0,
    )
    state.merton_default_probability = merton.default_probability
    if not state.covenants_met:
        state.underwriting_verdict = "REJECT: covenant breach (" + "; ".join(state.covenant_violations) + ")"
    elif state.merton_default_probability < 0.05:
        state.underwriting_verdict = "APPROVE: covenants met and Merton PD below 5%"
    else:
        state.underwriting_verdict = "REVIEW: covenants met but Merton PD at or above 5%"
    return state


def build_underwriting_agent() -> FinancialStateGraph[CreditUnderwritingState]:
    builder = FinancialStateGraph(CreditUnderwritingState)
    builder.add_node("ingest", ingest_and_provenance_node)
    builder.add_node("covenant_check", covenant_check_node)
    builder.add_node("credit_risk", structural_credit_risk_node)
    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "covenant_check")
    builder.add_edge("covenant_check", "credit_risk")
    builder.add_edge("credit_risk", END)
    return builder


def main() -> None:
    agent = build_underwriting_agent().compile()

    print("Step 4: two borrowers")
    for borrower in (
        CreditUnderwritingState(
            borrower_name="Apex Logistics Inc", ebitda=100.0, total_debt=320.0,
            interest_expense=15.0, equity_value=600.0,
        ),
        CreditUnderwritingState(
            borrower_name="Stressed Retail Corp", ebitda=50.0, total_debt=275.0,
            interest_expense=18.0, equity_value=120.0,
        ),
    ):
        r = agent.invoke(borrower)
        print(
            f"  {r.borrower_name}: net leverage {r.net_leverage:.2f}x, coverage "
            f"{r.interest_coverage:.2f}x, PD {r.merton_default_probability:.4%}, "
            f"debt reduction to comply {r.debt_reduction_to_comply:.1f}"
        )
        print(f"    {r.underwriting_verdict}")

    print("Step 6: the same graph as a REST API (called in-process)")
    app = create_agent_api(graph=agent, title="Credit underwriting API")
    resp = TestClient(app).post(
        "/api/v1/run",
        json={"borrower_name": "Acme Holdings", "ebitda": 120.0, "total_debt": 360.0,
              "interest_expense": 20.0, "equity_value": 750.0},
    )
    body = resp.json()
    print(f"  HTTP {resp.status_code}: net leverage {body['net_leverage']:.2f}x, "
          f"covenants met {body['covenants_met']}, verdict: {body['underwriting_verdict']}")
    print("  To serve it locally: serve_agent(agent, host='127.0.0.1', port=8000)")
    print("  (no built-in authentication: put it behind your own auth before exposing it)")


if __name__ == "__main__":
    main()
