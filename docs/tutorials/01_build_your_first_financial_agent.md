# Tutorial 01: Your First Credit Workflow: Covenant Checks, Merton PD and a Local API

**Who this is for:** finance students, credit analysts and developers new to Līlāvatī.
**Time:** about 45 minutes. **Cost:** none (no API key, no LLM).

## Learning objectives

1. Model a credit decision as typed state passed through deterministic steps (`FinancialState`,
   `FinancialStateGraph`).
2. Check maintenance covenants (net leverage, interest coverage) with a validated engine, including
   the case where a ratio sits exactly at its limit.
3. Compute a Merton (1974) default probability and explain why it can disagree with a covenant test.
4. Record where each input came from (`CitationRef`), test the workflow with `pytest`, and serve it
   as a local REST API.

## Where this fits

There is no language model in this tutorial, on purpose. This is the deterministic layer that an
LLM agent calls as tools: the model reads a credit agreement and passes `total_debt`, `ebitda` and
`interest_expense`; the numbers and the verdict come from tested code. Tutorials
[05](05_benchmarking_llms_with_open_models.md)–[07](07_tools_versus_mental_math.md) measure what
happens when models do this arithmetic themselves instead.

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync --extra dev
```

Steps 1–3 together form a module; save them as `credit_agent.py` in the repository root.

## Step 1: Typed state

All money is in USD millions; ratios are plain numbers ("x"); volatility is an annual decimal.
Writing units next to every field is a habit worth keeping: an inconsistent unit was the cause of a
real bug in this repository's Almgren-Chriss engine.

```python title="credit_agent.py"
"""Credit underwriting workflow from Tutorial 01 (save as credit_agent.py)."""

from datetime import date

from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef
from lilavati.framework import END, FinancialState, FinancialStateGraph


class CreditUnderwritingState(FinancialState):
    """Typed state for the credit underwriting workflow. Money in USD millions."""

    borrower_name: str = "Apollo Industrial Corp"
    fiscal_year: int = 2026
    ebitda: float = 80.0            # USD m, last twelve months
    total_debt: float = 280.0       # USD m
    cash: float = 0.0               # USD m (net leverage uses debt minus cash)
    interest_expense: float = 18.0  # USD m, annual
    equity_value: float = 400.0     # USD m, market value of equity
    equity_volatility: float = 0.30 # annualised, decimal

    # Maintenance covenants from the credit agreement
    max_net_leverage: float = 4.00   # (debt - cash) / EBITDA <= 4.0x
    min_interest_coverage: float = 3.50  # EBITDA / interest >= 3.5x

    # Outputs
    net_leverage: float = 0.0
    interest_coverage: float = 0.0
    covenants_met: bool = False
    covenant_violations: list[str] = []
    debt_reduction_to_comply: float = 0.0  # USD m of net debt to repay to meet the leverage test
    merton_default_probability: float = 0.0
    underwriting_verdict: str = "PENDING"
```

## Step 2: Three deterministic steps

**Provenance.** Each input is linked to the document, page and text it came from. The hash here is
illustrative; with a real filing it is the SHA-256 of the document (the SEC EDGAR and 10-K
connectors in `lilavati.connectors` compute it for you).

```python title="credit_agent.py"
def ingest_and_provenance_node(state: CreditUnderwritingState) -> CreditUnderwritingState:
    """Record where each input came from (illustrative document hash and pages)."""
    doc_hash = "a7c8e9b01234567890abcdef1234567890abcdef1234567890abcdef12345678"
    for page, label, value in ((24, "Adjusted Consolidated EBITDA", state.ebitda),
                               (28, "Total Outstanding Indebtedness", state.total_debt)):
        state.record_citation(CitationRef(
            doc_hash=doc_hash, filing_type="Credit-Agreement", filing_date=date(2026, 3, 15),
            period_end=date(2025, 12, 31), page=page, raw_snippet=f"{label}: ${value:.1f}M"))
    state.record_fact("ebitda", state.ebitda)
    state.record_fact("total_debt", state.total_debt)
    return state
```

**Covenants.** `SuvarnaEngine.verify_debt_covenant` checks both maintenance tests. It is plain
arithmetic with a small relative tolerance, so a ratio exactly at its limit complies (see "What can
go wrong"). It reports every violation, not just the first.

```python title="credit_agent.py"
def covenant_check_node(state: CreditUnderwritingState) -> CreditUnderwritingState:
    """Check both maintenance covenants with the validated engine (plain arithmetic)."""
    result = SuvarnaEngine.verify_debt_covenant(
        total_debt=state.total_debt, cash=state.cash, ebitda=state.ebitda,
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
```

`debt_reduction_to_comply` is the net debt that would have to be repaid to pass the leverage test.
It is not an "equity cure": how a cure counts (added to EBITDA, or applied to repay debt) is set by
each credit agreement, so model it from the agreement's own definition.

**Structural credit risk.** The Merton model treats equity as a call option on the firm's assets
and solves for asset value and volatility; the solver checks its own residuals and raises if it
fails ([validation](../validation/fixed_income.md)).

```python title="credit_agent.py"
def structural_credit_risk_node(state: CreditUnderwritingState) -> CreditUnderwritingState:
    """Merton (1974) one-year default probability, then a rule-based verdict."""
    merton = FixedIncomeEngine.solve_merton_model(
        equity_value=state.equity_value, equity_volatility=state.equity_volatility,
        debt_face_value=state.total_debt, risk_free_rate=0.045, maturity_years=1.0,
    )
    state.merton_default_probability = merton.default_probability
    if not state.covenants_met:
        state.underwriting_verdict = "REJECT: covenant breach (" + "; ".join(state.covenant_violations) + ")"
    elif state.merton_default_probability < 0.05:
        state.underwriting_verdict = "APPROVE: covenants met and Merton PD below 5%"
    else:
        state.underwriting_verdict = "REVIEW: covenants met but Merton PD at or above 5%"
    return state
```

## Step 3: Assemble the graph

```python title="credit_agent.py"
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
```

## Step 4: Run two borrowers

```python
from credit_agent import CreditUnderwritingState, build_underwriting_agent

agent = build_underwriting_agent().compile()

for borrower in (
    CreditUnderwritingState(borrower_name="Apex Logistics Inc", ebitda=100.0, total_debt=320.0,
                            interest_expense=15.0, equity_value=600.0),
    CreditUnderwritingState(borrower_name="Stressed Retail Corp", ebitda=50.0, total_debt=275.0,
                            interest_expense=18.0, equity_value=120.0),
):
    r = agent.invoke(borrower)
    print(f"{r.borrower_name}: net leverage {r.net_leverage:.2f}x, coverage {r.interest_coverage:.2f}x, "
          f"PD {r.merton_default_probability:.4%}, debt reduction {r.debt_reduction_to_comply:.1f}")
    print("  ", r.underwriting_verdict)
```

Output:

```text
Apex Logistics Inc: net leverage 3.20x, coverage 6.67x, PD 0.0000%, debt reduction 0.0
   APPROVE: covenants met and Merton PD below 5%
Stressed Retail Corp: net leverage 5.50x, coverage 2.78x, PD 0.0039%, debt reduction 75.0
   REJECT: covenant breach (Covenant breach: Net Leverage (5.50x) exceeds maximum allowable limit (4.00x).; Covenant breach: Interest Coverage (2.78x) is below minimum allowable limit (3.50x).)
```

The stressed borrower fails **both** tests. Notice also that its one-year Merton PD is tiny
(0.0039%) despite leverage of 5.5x: the model sees a market value of equity well above zero and
modest volatility over a short horizon. A covenant breach and a low model PD can coexist; neither
number alone is a credit decision.

## Step 5: Test it

Save as `tests/test_credit_agent.py` (with `credit_agent.py` in the repository root):

```python title="tests/test_credit_agent.py"
import pytest

from credit_agent import CreditUnderwritingState, build_underwriting_agent

AGENT = build_underwriting_agent().compile()


def test_covenants_met() -> None:
    r = AGENT.invoke(CreditUnderwritingState(ebitda=50.0, total_debt=150.0, interest_expense=10.0))
    assert r.net_leverage == pytest.approx(3.0)
    assert r.covenants_met and r.debt_reduction_to_comply == 0.0
    assert len(r.citations) == 2


def test_leverage_breach_reports_debt_reduction() -> None:
    r = AGENT.invoke(CreditUnderwritingState(ebitda=50.0, total_debt=250.0, interest_expense=10.0))
    assert not r.covenants_met
    assert r.debt_reduction_to_comply == pytest.approx(50.0)  # 250 - 4.0 * 50
    assert r.underwriting_verdict.startswith("REJECT")


def test_ratio_exactly_at_the_limit_complies() -> None:
    # 2.1 / 0.7 is 3.0000000000000004 in floating point; a naive `<=` would report a breach.
    r = AGENT.invoke(CreditUnderwritingState(ebitda=0.7, total_debt=2.1, interest_expense=0.1,
                                             max_net_leverage=3.0, equity_value=10.0))
    assert r.covenants_met


def test_coverage_breach_is_caught_even_when_leverage_is_fine() -> None:
    r = AGENT.invoke(CreditUnderwritingState(ebitda=50.0, total_debt=100.0, interest_expense=20.0))
    assert not r.covenants_met and r.debt_reduction_to_comply == 0.0
```

```bash
uv run pytest tests/test_credit_agent.py    # 4 passed
```

## Step 6: Serve it as a local REST API

`create_agent_api` turns the compiled graph into a FastAPI app with `POST /api/v1/run`,
`GET /api/v1/health`, `GET /api/v1/schema` and interactive docs at `/docs`.

```python
from fastapi.testclient import TestClient

from credit_agent import build_underwriting_agent
from lilavati.framework import create_agent_api, serve_agent

agent = build_underwriting_agent().compile()
app = create_agent_api(graph=agent, title="Credit underwriting API")

# Call it in-process (no server needed):
resp = TestClient(app).post("/api/v1/run", json={
    "borrower_name": "Acme Holdings", "ebitda": 120.0, "total_debt": 360.0,
    "interest_expense": 20.0, "equity_value": 750.0})
body = resp.json()
print(resp.status_code, body["net_leverage"], body["covenants_met"], body["underwriting_verdict"])
# 200 3.0 True APPROVE: covenants met and Merton PD below 5%

# Or run a local server (localhost only):
# serve_agent(agent, host="127.0.0.1", port=8000)
```

The response is the full state, including the two citations, `covenant_violations` and
`merton_default_probability` (1.68e-08 for this borrower). The server has **no built-in
authentication**: bind it to `127.0.0.1` and put it behind your own authenticating proxy before
exposing it (see `SECURITY.md`).

The complete script, with both borrowers and the API call, is
[`examples/06_build_custom_agent_and_api.py`](https://github.com/revendrat/lilavati/blob/main/examples/06_build_custom_agent_and_api.py):

```bash
uv run python examples/06_build_custom_agent_and_api.py
```

## What can go wrong (real bugs, now fixed and tested)

- **A ratio exactly at its limit reported as a breach.** `2.1 / 0.7` is `3.0000000000000004` in
  floating point, so a naive `<=` failed a borrower at exactly 3.0x. The engine compares with a
  1e-9 relative tolerance ([invariants validation](../validation/invariants.md)).
- **A solver returning a non-solution.** The Merton solver once returned points with large
  equation errors for distressed inputs; it now checks its residual and raises instead
  ([fixed-income validation](../validation/fixed_income.md)). A failure must never look like a result.
- **Checking one covenant and stopping.** An earlier version of this tutorial reported only the
  leverage breach for the stressed borrower, which also fails interest coverage.
- **Calling arithmetic "proof".** Covenant checks are arithmetic on the numbers given; they verify
  those numbers, not the borrower. Say "checked", not "proven".
- **Mixed units.** EBITDA for a quarter against annual interest, or debt in INR crore against EBITDA
  in USD millions, produces a valid-looking ratio that means nothing. Keep units in the field names
  or comments, and check them at the boundary.

## Using this with Claude Code and MCP

- **Contributing code:** load the [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md).
  It holds the rules this tutorial follows (units in docstrings, tolerance at thresholds, solvers
  that raise, no "proven" language).
- **Using the engines from an AI assistant:** run `lilavati-mcp-fixed-income` as an MCP server
  ([MCP setup](../mcp-setup.md)) and ask, for example, *"Using the Merton tool, what is the one-year
  default probability for equity value 120, equity volatility 0.30, debt 275, rate 4.5%?"* The
  assistant extracts the parameters; the tool returns the number.

## Exercises

1. Give the stressed borrower USD 60m of cash. Which covenants pass now, and what is the debt
   reduction needed?
2. Change the Merton horizon from 1 to 5 years for both borrowers. How do the PDs change, and why?
3. Add a third covenant (for example, minimum fixed-charge coverage) as its own node, with a test at
   exactly the limit.
4. Replace the illustrative document hash with the SHA-256 of a real PDF (`hashlib.sha256`), and
   cite real pages.

## For instructors

- **Discussion:** the stressed borrower breaches both covenants but has a Merton PD of 0.0039%.
  Which would you trust, and what information is each missing?
- **Lab extension:** compare the Merton PD with the Altman Z-score ([Tutorial 03](03_forensic_accounting_and_adversarial_committee.md))
  for the same borrowers.
- **Link to the LLM part of the course:** give an LLM the same two borrowers without tools and
  compare its ratios and verdicts with the outputs above ([Tutorial 07](07_tools_versus_mental_math.md)).
