"""
Unit and Integration Tests for Search Agent, Deep Agent, and Research Agent.
"""

from __future__ import annotations

from datetime import date

from lilavati.agents.deep import DeepFinancialAgent
from lilavati.agents.research import FinancialResearchAgent
from lilavati.agents.search import FinancialSearchAgent
from lilavati.core.graph import StatementGraph


def _sec_client():
    import json
    from pathlib import Path

    import httpx

    from lilavati.connectors.edgar import EdgarClient

    fix = Path(__file__).parent / "fixtures"
    subs = json.loads((fix / "aapl_submissions_subset.json").read_text())
    facts = json.loads((fix / "aapl_companyfacts_10k_fy2025_2026.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/submissions/CIK0000320193.json"):
            return httpx.Response(200, json=subs)
        if request.url.path.endswith("/companyfacts/CIK0000320193.json"):
            return httpx.Response(200, json=facts)
        return httpx.Response(404)

    return EdgarClient(user_agent="Example Org contact@example.org", transport=httpx.MockTransport(handler))


def test_financial_search_agent_is_point_in_time_and_cited() -> None:
    # Apple's FY2026 10-K was filed 2026-11-03 (fixtures: verbatim SEC subsets). As of 2026-11-02 the
    # latest known 10-K is FY2025's; from 2026-11-03 it is FY2026's. Values: the 10-Ks, USD millions.
    agent = FinancialSearchAgent(edgar_client=_sec_client())
    before = agent.search_entity("AAPL", [], as_of_date=date(2026, 11, 2))
    after = agent.search_entity("AAPL", [], as_of_date=date(2026, 11, 3))
    rev_before = {f.metric_name: f.value for f in before.discovered_facts}["revenue"]
    rev_after = {f.metric_name: f.value for f in after.discovered_facts}["revenue"]
    assert rev_before == 416_161 and rev_after == 478_834
    assert {c.filing_date for c in after.verified_citations} == {date(2026, 11, 3)}
    assert all(c.page is None and len(c.doc_hash) == 64 for c in after.verified_citations)

    only_bs = agent.search_entity("AAPL", ["balance_sheet"], as_of_date=date(2026, 11, 3))
    assert {f.topic for f in only_bs.discovered_facts} == {"balance_sheet"}


def test_financial_search_agent_never_invents_figures(monkeypatch) -> None:
    # It used to return Apple's FY2024 figures, with made-up page numbers, for any symbol when SEC
    # could not be reached, and a fixed Indian revenue of 890,000 for any company.
    monkeypatch.delenv("LILAVATI_SEC_USER_AGENT", raising=False)
    agent = FinancialSearchAgent()
    rep = agent.search_entity("MSFT", ["revenue"], as_of_date=date(2024, 11, 1))
    assert rep.discovered_facts == [] and rep.verified_citations == []
    assert "LILAVATI_SEC_USER_AGENT" in rep.notes[0]

    rep_in = agent.search_entity("RELIANCE", ["revenue"], as_of_date=date(2024, 3, 31), jurisdiction="IN")
    assert rep_in.discovered_facts == [] and "not implemented" in rep_in.notes[0]

    early = FinancialSearchAgent(edgar_client=_sec_client()).search_entity("AAPL", [], date(2020, 1, 1))
    assert early.discovered_facts == [] and "No 10-K" in early.notes[0]


def test_deep_financial_agent_scenarios() -> None:
    deep_agent = DeepFinancialAgent()

    graph = StatementGraph(
        revenue=1000.0,
        gross_margin=0.40,
        sga_ratio=0.15,
        tax_rate=0.20,
        interest_rate=0.05,
        capex=50.0,
        depreciation=40.0,
        cash_start=100.0,
        initial_debt=200.0,
        initial_common_equity=300.0,
        initial_net_ppe=500.0,
    )

    tree = deep_agent.analyze_deep_scenarios(
        query="Evaluate 3-year cash generation trajectory",
        baseline_graph=graph,
    )

    assert tree.target_query == "Evaluate 3-year cash generation trajectory"
    assert len(tree.scenario_branches) == 3
    assert tree.probability_weighted_net_income > 0.0
    assert len(tree.deep_reasoning_steps) >= 3


def test_financial_research_agent_dossier(monkeypatch) -> None:
    research_agent = FinancialResearchAgent()
    research_agent.search_agent = FinancialSearchAgent(edgar_client=_sec_client())
    dossier = research_agent.conduct_deep_research(symbol="AAPL", as_of_date=date(2026, 11, 3))

    assert dossier.symbol == "AAPL"
    assert dossier.recommendation in ("OVERWEIGHT", "NEUTRAL", "UNDERWEIGHT")
    assert dossier.citations_count == 5 and len(dossier.proof_of_citation_hashes) == 5
    assert "SEC XBRL" in dossier.data_source and dossier.target_price is None
    assert dossier.assumed_inputs  # every assumption is listed
    assert dossier.scenario_analysis.probability_weighted_net_income > 0.0
    assert "beneish_m_score" in dossier.forensic_accounting_summary

    # Without SEC access the dossier says it is illustrative instead of borrowing Apple's figures.
    monkeypatch.delenv("LILAVATI_SEC_USER_AGENT", raising=False)
    offline = FinancialResearchAgent().conduct_deep_research(symbol="MSFT", as_of_date=date(2024, 11, 1))
    assert offline.data_source.startswith("illustrative") and offline.citations_count == 0
    assert "illustrative" in offline.executive_thesis
