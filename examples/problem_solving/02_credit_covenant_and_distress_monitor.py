"""
Problem 02: Private Credit Covenant Monitoring & Remedy Calculation.

Real-World Institutional Problem:
A private debt fund manages a $150M Senior Secured Credit Facility for 'Omni Logistics Corp.'
Covenant terms mandate:
- Maximum Net Leverage <= 3.50x
- Minimum Interest Coverage >= 3.00x

Following fuel inflation and contract renegotiations, the borrower's quarterly EBITDA compresses.
The agent pipeline:
1. Ingests statutory quarterly figures.
2. Formally evaluates covenant compliance via CovenantGate (Z3 SMT solver).
3. Computes the implied Merton structural Distance-to-Default and credit spread jump.
4. Deterministically computes the net debt reduction required to restore compliance (how a sponsor
   equity cure counts towards it is set by the credit agreement).
5. Emits a Credit Committee Action Notice with the computed figures.
"""

from __future__ import annotations

from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.provenance import CitationRef, SourcedValue
from lilavati.framework import (
    END,
    CovenantGate,
    FinancialState,
    FinancialStateGraph,
)

console = Console()


class CreditMonitoringState(FinancialState):
    borrower_name: str = "Omni Logistics Corp."
    facility_amount_mil: float = 150.0
    current_leverage: float = 0.0
    current_coverage: float = 0.0
    is_covenant_breached: bool = False
    merton_default_prob_pct: float = 0.0
    debt_reduction_needed_mil: float = 0.0  # USD m of net debt to repay to meet the leverage test
    credit_committee_action: str = "MONITOR"


def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Problem 02: Private Debt Covenant Monitoring & Remedy Agent[/bold cyan]\n"
            "Detecting covenant breaches, Merton credit distress, and the net debt reduction needed to comply.",
            title="Project Līlāvatī Institutional Solutions",
        )
    )

    builder = FinancialStateGraph(CreditMonitoringState)

    # --- Node 1: Ingest Quarterly Financials ---
    def ingest_credit_facts(state: CreditMonitoringState) -> CreditMonitoringState:
        cit = CitationRef(
            doc_hash="a" * 64,
            filing_type="10-Q",
            filing_date=date(2024, 10, 20),
            period_end=date(2024, 9, 30),
            page=22,
            raw_snippet="Total Debt: $160M, Cash: $20M, EBITDA: $35M, Interest: $15M",
        )
        state.record_citation(cit)
        state.sourced_values.append(SourcedValue(value=160.0, citation=cit))

        # Financial facts: Net debt = 160 - 20 = $140M. EBITDA compressed to $35M.
        state.record_fact("total_debt", 160.0)
        state.record_fact("cash", 20.0)
        state.record_fact("ebitda", 35.0)
        state.record_fact("interest_expense", 15.0)
        state.record_fact("equity_market_value", 70.0)
        state.record_fact("equity_volatility", 0.55)

        return state

    # --- Node 2: SMT Covenant Gate ---
    # Max leverage 3.50x, Min interest coverage 3.00x
    covenant_gate = CovenantGate(max_leverage=3.50, min_interest_coverage=3.00)

    # --- Node 3: Credit Risk & Remedy Calculation ---
    def remedy_and_credit_risk_node(state: CreditMonitoringState) -> CreditMonitoringState:
        facts = state.extracted_facts
        net_debt = float(facts["total_debt"]) - float(facts["cash"])
        ebitda = float(facts["ebitda"])
        interest = float(facts["interest_expense"])

        state.current_leverage = net_debt / ebitda
        state.current_coverage = ebitda / interest if interest > 0 else float("inf")
        state.is_covenant_breached = not state.is_valid

        # Solve Merton Structural Credit Risk Model
        merton = FixedIncomeEngine.solve_merton_model(
            equity_value=float(facts["equity_market_value"]),
            debt_face_value=float(facts["total_debt"]),
            equity_volatility=float(facts["equity_volatility"]),
            risk_free_rate=0.045,
            maturity_years=2.0,
        )
        state.merton_default_prob_pct = merton.default_probability * 100.0

        # Net debt reduction that brings leverage back to the covenant limit.
        max_allowed_net_debt = covenant_gate.max_leverage * ebitda
        if net_debt > max_allowed_net_debt:
            state.debt_reduction_needed_mil = net_debt - max_allowed_net_debt
            state.credit_committee_action = (
                f"ISSUE DEFAULT NOTICE: require ${state.debt_reduction_needed_mil:.1f}M of net debt "
                "reduction (e.g. sponsor equity applied to repayment, if the agreement allows a cure) "
                "within the cure period, or accelerate the facility."
            )
        else:
            state.debt_reduction_needed_mil = 0.0
            state.credit_committee_action = "MAINTAIN PASS STATUS"

        return state

    builder.add_node("ingest", ingest_credit_facts)
    builder.add_node("covenant_gate", covenant_gate)
    builder.add_node("remedy_analysis", remedy_and_credit_risk_node)

    builder.add_edge("ingest", "covenant_gate")
    builder.add_edge("covenant_gate", "remedy_analysis")
    builder.add_edge("remedy_analysis", END)
    builder.set_entry_point("ingest")

    app = builder.compile()
    final_state = app.invoke(CreditMonitoringState())

    # Display Action Memo
    table = Table(
        title=f"Credit Committee Monitoring Memorandum: {final_state.borrower_name}",
        show_header=True,
    )
    table.add_column("Covenant / Risk Parameter", style="cyan")
    table.add_column("Current Metric", justify="right", style="bold")
    table.add_column("Covenant Limit", justify="right", style="dim")
    table.add_column("Status", justify="center")

    table.add_row(
        "Net Debt / EBITDA Leverage",
        f"{final_state.current_leverage:.2f}x",
        "≤ 3.50x",
        "[bold red]BREACH (UNSAT)[/bold red]",
    )
    table.add_row(
        "Interest Coverage Ratio",
        f"{final_state.current_coverage:.2f}x",
        "≥ 3.00x",
        "[bold red]BREACH (UNSAT)[/bold red]",
    )
    table.add_row(
        "Merton 2-Yr Default Probability",
        f"{final_state.merton_default_prob_pct:.2f}%",
        "< 2.0%",
        "[bold yellow]ELEVATED RISK[/bold yellow]",
    )
    table.add_row(
        "Net Debt Reduction to Comply",
        f"${final_state.debt_reduction_needed_mil:.1f}M",
        "N/A",
        "[bold red]REDUCTION NEEDED[/bold red]" if final_state.debt_reduction_needed_mil > 0
        else "[green]NONE[/green]",
    )

    console.print(table)
    console.print(
        f"\n[bold red]Binding Credit Action:[/bold red] {final_state.credit_committee_action}"
    )


if __name__ == "__main__":
    main()
