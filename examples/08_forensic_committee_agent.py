"""
Companion script for Tutorial 03 (docs/tutorials/03_forensic_accounting_and_adversarial_committee.md):
forensic screens and a rule-based bull/bear committee on a synthetic firm.

1. Z3 check of the reported statements (A = L + E; cash roll-forward) before any ratio.
2. Beneish (1999) M-score and Altman (1968) Z-score.
3. Rule-based case for and against, then an ordered screening decision.
4. Three firms: flagged by Beneish, clean, and edited without rebalancing (rejected as inconsistent).

No LLM is involved; the firm is synthetic.

    uv run python examples/08_forensic_committee_agent.py
"""

from __future__ import annotations

from datetime import date

from lilavati.agents.forensics import ForensicEngine
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef
from lilavati.framework import END, FinancialState, FinancialStateGraph


class ForensicCommitteeState(FinancialState):
    """Synthetic firm, two fiscal years. Money in USD millions; _t is this year, _t_minus_1 last year."""

    ticker: str = "SYNTHETIC-CO"
    fiscal_year: int = 2026
    revenue_t: float = 1100.0
    revenue_t_minus_1: float = 1000.0
    cogs_t: float = 600.0
    cogs_t_minus_1: float = 550.0
    receivables_t: float = 180.0
    receivables_t_minus_1: float = 100.0
    current_assets_t: float = 480.0
    current_assets_t_minus_1: float = 400.0
    ppe_t: float = 720.0
    ppe_t_minus_1: float = 650.0
    total_assets_t: float = 1350.0
    total_assets_t_minus_1: float = 1200.0
    depreciation_t: float = 50.0
    depreciation_t_minus_1: float = 55.0
    sga_t: float = 160.0
    sga_t_minus_1: float = 150.0
    long_term_debt_t: float = 350.0
    long_term_debt_t_minus_1: float = 300.0
    current_liabilities_t: float = 160.0
    current_liabilities_t_minus_1: float = 140.0
    shareholders_equity_t: float = 840.0
    retained_earnings_t: float = 420.0  # reported figure, needed by Altman (do not proxy it)
    net_income_t: float = 130.0
    cfo_t: float = 70.0
    cfi_t: float = -60.0
    cff_t: float = 20.0
    cash_start_t: float = 80.0
    cash_end_t: float = 110.0
    market_value_equity: float = 900.0

    # Outputs
    accounting_identities_hold: bool = False
    identity_violations: list[str] = []
    beneish_m_score: float = 0.0
    beneish_flag: bool = False      # M > -1.78 (Beneish 1999 cut-off)
    altman_z_score: float = 0.0
    altman_zone: str = "UNKNOWN"    # Distress < 1.81 <= Grey <= 2.99 < Safe (Altman 1968)
    bull_memo: str = ""
    bear_memo: str = ""
    decision: str = "PENDING"


def provenance_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Link the reported revenue to its filing page (illustrative hash)."""
    state.record_citation(CitationRef(
        doc_hash="f18b3294c7e63b6528d3e20e8b23c911854c602058b29ef5e0241cfb263b6102",
        filing_type="10-K", filing_date=date(2026, 3, 1), period_end=date(2025, 12, 31), page=42,
        raw_snippet=f"Revenues FY{state.fiscal_year}: ${state.revenue_t:.1f}M"))
    return state


def identity_check_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Z3 check, on the reported numbers: A = L + E and closing cash = opening + CFO + CFI + CFF."""
    check = SuvarnaEngine.verify_three_statement_invariants(
        total_assets=state.total_assets_t,
        total_liabilities=state.current_liabilities_t + state.long_term_debt_t,
        shareholders_equity=state.shareholders_equity_t, net_income=state.net_income_t,
        cash_start=state.cash_start_t, cash_end=state.cash_end_t,
        cfo=state.cfo_t, cfi=state.cfi_t, cff=state.cff_t)
    state.accounting_identities_hold = check.is_valid
    state.identity_violations = check.violations
    return state


def forensic_scores_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Beneish (1999) 8-variable M-score and Altman (1968) Z-score."""
    m = ForensicEngine.calculate_beneish_m_score(
        revenue_t=state.revenue_t, revenue_t_minus_1=state.revenue_t_minus_1,
        cogs_t=state.cogs_t, cogs_t_minus_1=state.cogs_t_minus_1,
        receivables_t=state.receivables_t, receivables_t_minus_1=state.receivables_t_minus_1,
        current_assets_t=state.current_assets_t, current_assets_t_minus_1=state.current_assets_t_minus_1,
        ppe_t=state.ppe_t, ppe_t_minus_1=state.ppe_t_minus_1,
        total_assets_t=state.total_assets_t, total_assets_t_minus_1=state.total_assets_t_minus_1,
        depreciation_t=state.depreciation_t, depreciation_t_minus_1=state.depreciation_t_minus_1,
        sga_t=state.sga_t, sga_t_minus_1=state.sga_t_minus_1,
        long_term_debt_t=state.long_term_debt_t, long_term_debt_t_minus_1=state.long_term_debt_t_minus_1,
        current_liabilities_t=state.current_liabilities_t,
        current_liabilities_t_minus_1=state.current_liabilities_t_minus_1,
        net_income_t=state.net_income_t, cfo_t=state.cfo_t)
    state.beneish_m_score, state.beneish_flag = m.m_score, m.is_likely_manipulator
    z = ForensicEngine.calculate_altman_z_score(
        working_capital=state.current_assets_t - state.current_liabilities_t,
        total_assets=state.total_assets_t, retained_earnings=state.retained_earnings_t,
        ebit=state.revenue_t - state.cogs_t - state.sga_t,
        market_value_equity=state.market_value_equity,
        total_liabilities=state.current_liabilities_t + state.long_term_debt_t,
        sales=state.revenue_t)
    state.altman_z_score, state.altman_zone = z.z_score, z.distress_zone
    return state


def bull_bear_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Rule-based case for and against, written only from computed numbers (no LLM here)."""
    rev_growth = state.revenue_t / state.revenue_t_minus_1 - 1.0
    rec_growth = state.receivables_t / state.receivables_t_minus_1 - 1.0
    accruals_gap = state.net_income_t - state.cfo_t
    state.bull_memo = (f"FOR: revenue grew {rev_growth:.1%} to ${state.revenue_t:,.0f}m; Altman "
                       f"Z {state.altman_z_score:.2f} ({state.altman_zone}).")
    flags = []
    if state.beneish_flag:
        flags.append(f"Beneish M {state.beneish_m_score:.2f} is above -1.78")
    if rec_growth > 2 * rev_growth:
        flags.append(f"receivables grew {rec_growth:.0%} against revenue {rev_growth:.0%} "
                     "(possible aggressive revenue recognition; check credit terms)")
    if accruals_gap > 0.25 * state.net_income_t:
        flags.append(f"net income exceeds operating cash flow by ${accruals_gap:,.0f}m")
    state.bear_memo = "AGAINST: " + ("; ".join(flags) if flags else "no forensic flags")
    return state


def decision_node(state: ForensicCommitteeState) -> ForensicCommitteeState:
    """Screening decision. Order matters: inconsistent statements make every ratio meaningless."""
    if not state.accounting_identities_hold:
        state.decision = "REJECT: reported statements are inconsistent (" + "; ".join(state.identity_violations) + ")"
    elif state.beneish_flag:
        state.decision = f"REJECT: manipulation screen (Beneish M {state.beneish_m_score:.2f} > -1.78)"
    elif state.altman_zone == "Distress":
        state.decision = f"REJECT: distress screen (Altman Z {state.altman_z_score:.2f} < 1.81)"
    else:
        state.decision = f"PASS SCREEN: identities hold, Beneish M {state.beneish_m_score:.2f}, Altman Z {state.altman_z_score:.2f}"
    return state


def build_forensic_committee() -> FinancialStateGraph[ForensicCommitteeState]:
    graph = FinancialStateGraph(ForensicCommitteeState)
    for name, node in (("provenance", provenance_node), ("identities", identity_check_node),
                       ("forensics", forensic_scores_node), ("bull_bear", bull_bear_node),
                       ("decision", decision_node)):
        graph.add_node(name, node)
    graph.set_entry_point("provenance")
    graph.add_edge("provenance", "identities")
    graph.add_edge("identities", "forensics")
    graph.add_edge("forensics", "bull_bear")
    graph.add_edge("bull_bear", "decision")
    graph.add_edge("decision", END)
    return graph


CLEAN = {"receivables_t": 110.0, "cfo_t": 150.0, "net_income_t": 120.0,
         "cfi_t": -50.0, "cff_t": -60.0, "cash_end_t": 120.0}


def main() -> None:
    from fastapi.testclient import TestClient

    from lilavati.framework import create_agent_api

    committee = build_forensic_committee().compile()
    cases = {
        "default firm": ForensicCommitteeState(),
        "clean firm": ForensicCommitteeState(**CLEAN),
        "edited without rebalancing": ForensicCommitteeState(receivables_t=220.0, cfo_t=40.0, net_income_t=140.0),
    }
    for label, state in cases.items():
        r = committee.invoke(state)
        print(f"{label}: M {r.beneish_m_score:.2f}, Z {r.altman_z_score:.2f} ({r.altman_zone})")
        print("  ", r.bull_memo)
        print("  ", r.bear_memo)
        print("  ", r.decision)

    app = create_agent_api(graph=committee, title="Forensic screening API")
    resp = TestClient(app).post("/api/v1/run", json={"ticker": "SYNTHETIC-CO", **CLEAN})
    print(f"API HTTP {resp.status_code}: {resp.json()['decision']}")
    print("To serve locally: serve_agent(committee, host='127.0.0.1', port=8000)  (no built-in auth)")


if __name__ == "__main__":
    main()
