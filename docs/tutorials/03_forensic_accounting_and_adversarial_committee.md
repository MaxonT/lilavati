# Tutorial 03: Forensic Accounting Screens and a Bull/Bear Committee

**Who this is for:** accounting and finance students, equity and credit analysts, auditors.
**Time:** about 60 minutes. **Cost:** none (no API key, no LLM).

## Learning objectives

1. Check that reported statements are internally consistent before computing any ratio.
2. Compute the Beneish (1999) M-score and the Altman (1968) Z-score, and say what each does and
   does not detect.
3. Write the case for and against an investment only from computed numbers.
4. Combine the screens into an ordered decision, test it, and serve it locally.

## Where this fits

The "bull" and "bear" here are rule-based text written from computed numbers, so you can see every
step. In an LLM committee the same structure holds: models argue, tools compute, and the decision
rule reads tool outputs (see `lilavati.agents.committee` and the `lilavati-mcp-committee` server).
Steps 1–3 form one module; save them as `forensic_committee.py`. The firm is synthetic.

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync --extra dev
```

## Step 1: Two years of statements

The forensic scores compare this year (`_t`) with last year (`_t_minus_1`). Altman needs retained
earnings as reported; an earlier version of this tutorial used half of equity as a stand-in, which
quietly invents an input.

```python title="forensic_committee.py"
"""Forensic screening and committee workflow from Tutorial 03 (save as forensic_committee.py)."""

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
```

## Step 2: Consistency first, then the scores

**Identities.** Z3 checks, on the reported numbers, that assets equal liabilities plus equity and
that closing cash equals opening cash plus the three cash flows. This verifies the numbers given;
it says nothing about whether they are true.

**Beneish M-score** (eight indices, coefficients from Beneish 1999):

$$M = -4.84 + 0.920\,\text{DSRI} + 0.528\,\text{GMI} + 0.404\,\text{AQI} + 0.892\,\text{SGI} + 0.115\,\text{DEPI} - 0.172\,\text{SGAI} + 4.679\,\text{TATA} - 0.327\,\text{LVGI}$$

M above −1.78 flags a *likely* manipulator. The engine computes TATA as (net income − operating cash
flow) / total assets, a common simplification of Beneish's balance-sheet accruals definition
([forensics validation](../validation/forensics.md)). Two of these coefficients were wrong in this
engine before its audit, one with the wrong sign.

**Altman Z-score** (Altman 1968): Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 0.999 X5, with zones
Distress (< 1.81), Grey (1.81–2.99) and Safe (> 2.99). It was fitted on public manufacturing firms;
applying it to banks or service firms is a known misuse.

```python title="forensic_committee.py"
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
```

## Step 3: For, against, decide

```python title="forensic_committee.py"
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
```

## Step 4: Three firms

```python
from forensic_committee import ForensicCommitteeState, build_forensic_committee

committee = build_forensic_committee().compile()
cases = {
    "default firm": ForensicCommitteeState(),
    "clean firm": ForensicCommitteeState(receivables_t=110.0, cfo_t=150.0, net_income_t=120.0,
                                         cfi_t=-50.0, cff_t=-60.0, cash_end_t=120.0),
    "edited without rebalancing": ForensicCommitteeState(receivables_t=220.0, cfo_t=40.0, net_income_t=140.0),
}
for label, state in cases.items():
    r = committee.invoke(state)
    print(f"{label}: M {r.beneish_m_score:.2f}, Z {r.altman_z_score:.2f} ({r.altman_zone})")
    print("  ", r.bear_memo)
    print("  ", r.decision)
```

Output:

```text
default firm: M -1.63, Z 3.42 (Safe)
   AGAINST: Beneish M -1.63 is above -1.78; receivables grew 80% against revenue 10% (possible aggressive revenue recognition; check credit terms); net income exceeds operating cash flow by $60m
   REJECT: manipulation screen (Beneish M -1.63 > -1.78)
clean firm: M -2.53, Z 3.42 (Safe)
   AGAINST: no forensic flags
   PASS SCREEN: identities hold, Beneish M -2.53, Altman Z 3.42
edited without rebalancing: M -1.16, Z 3.42 (Safe)
   AGAINST: Beneish M -1.16 is above -1.78; receivables grew 120% against revenue 10% (possible aggressive revenue recognition; check credit terms); net income exceeds operating cash flow by $100m
   REJECT: reported statements are inconsistent (Cash flow mismatch: Ending Cash (110.0) != Starting Cash (80.0) + CFO (40.0) + CFI (-60.0) + CFF (20.0). Delta = 30.0000)
```

Three things to notice:

- The third firm changed operating cash flow without changing closing cash. The identity check
  rejects it before any score matters. An earlier version of this tutorial made exactly this edit
  and reported it as a manipulation finding: the scores of inconsistent statements mean nothing.
- Altman says "Safe" (3.42) for all three, including the firm Beneish flags. Distress and earnings
  manipulation are different questions; one screen cannot answer both.
- A Beneish flag is a reason to investigate, not a finding of fraud.

## Step 5: Test it

Save as `tests/test_forensic_committee.py` (with `forensic_committee.py` in the repository root):

```python title="tests/test_forensic_committee.py"
import pytest

from forensic_committee import ForensicCommitteeState, build_forensic_committee

COMMITTEE = build_forensic_committee().compile()
CLEAN = dict(receivables_t=110.0, cfo_t=150.0, net_income_t=120.0, cfi_t=-50.0, cff_t=-60.0, cash_end_t=120.0)


def test_default_firm_fails_the_manipulation_screen() -> None:
    r = COMMITTEE.invoke(ForensicCommitteeState())
    assert r.accounting_identities_hold
    assert r.beneish_m_score > -1.78 and r.decision.startswith("REJECT: manipulation")


def test_clean_firm_passes() -> None:
    r = COMMITTEE.invoke(ForensicCommitteeState(**CLEAN))
    assert r.beneish_m_score < -1.78 and r.decision.startswith("PASS SCREEN")


def test_inconsistent_statements_are_rejected_before_any_ratio() -> None:
    # Changing CFO without changing closing cash breaks cash = opening + CFO + CFI + CFF.
    r = COMMITTEE.invoke(ForensicCommitteeState(cfo_t=40.0))
    assert not r.accounting_identities_hold
    assert r.decision.startswith("REJECT: reported statements are inconsistent")


def test_altman_uses_reported_retained_earnings() -> None:
    low = COMMITTEE.invoke(ForensicCommitteeState(**CLEAN, retained_earnings_t=0.0))
    high = COMMITTEE.invoke(ForensicCommitteeState(**CLEAN))
    # X2 = RE / TA enters with coefficient 1.4
    assert high.altman_z_score - low.altman_z_score == pytest.approx(1.4 * 420.0 / 1350.0)
```

```bash
uv run pytest tests/test_forensic_committee.py    # 4 passed
```

## Step 6: Serve it locally

```python
from fastapi.testclient import TestClient

from forensic_committee import build_forensic_committee
from lilavati.framework import create_agent_api

app = create_agent_api(graph=build_forensic_committee().compile(), title="Forensic screening API")
resp = TestClient(app).post("/api/v1/run", json={"ticker": "SYNTHETIC-CO"})
print(resp.status_code, resp.json()["decision"])
```

To run a server: `serve_agent(graph, host="127.0.0.1", port=8000)`. No built-in authentication;
keep it local (`SECURITY.md`). Complete script:
[`examples/08_forensic_committee_agent.py`](https://github.com/revendrat/lilavati/blob/main/examples/08_forensic_committee_agent.py).

## What can go wrong

- **Wrong published coefficients.** This engine once had TATA 4.037 instead of 4.679 and LVGI
  +0.0327 instead of −0.327 (Beneish 1999). Its test now pins the constant: with every index equal
  to 1, M = −2.480.
- **Scores on inconsistent statements.** See the third firm above.
- **Invented inputs.** Proxying retained earnings (or any input) changes the score without anyone
  noticing. Missing data should stop the calculation, not be filled in.
- **Model scope.** Altman (1968) is for public manufacturers; Beneish was estimated on US public
  companies. Say where a screen does not apply.
- **Real names on synthetic data.** An earlier version labelled this synthetic firm with the name
  of a real company. Do not attach real names to made-up numbers.

## Using this with Claude Code and MCP

- **Contributing code:** the [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
  (use published coefficients exactly, pin them with a test, validate inputs).
- **From an AI assistant:** `lilavati-mcp-committee` exposes the Beneish score and committee memo
  as tools ([MCP setup](../mcp-setup.md)). Ask, for example, *"Compute the Beneish M-score for
  these two years of statements and tell me which index drives it."*

## Exercises

1. For the default firm, compute each of the eight indices. Which one pushes M above −1.78?
2. Make the third firm consistent (adjust closing cash, and the balance sheet with it). What do the
   screens say now?
3. Set `market_value_equity` to 200. How does Altman Z change, and which zone does the firm fall in?
4. Add a Piotroski F-score node (Piotroski 2000) with a test against a worked example.

## For instructors

- **Discussion:** why must the consistency check come first? What would an analyst conclude if they
  scored the third firm without it?
- **Lab extension:** take two real 10-K filings via the EDGAR connector, compute both screens, and
  compare with what later happened to each company.
- **Link to LLMs:** ask a model for the M-score without tools and compare its coefficients with the
  published ones.
