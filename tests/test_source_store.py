"""Resolving citations against the sources actually read (SourceStore, ProvenanceGate(store)).

The gate used to check only a citation's form, so a well-formed citation to a source nobody read, or
with the wrong value, passed. Fixtures: verbatim SEC subset (Apple FY2026 10-K facts), verbatim pypdf
lines from Apple's FY2026 10-K, and the fictional Ind AS disclosure of test_india_filings.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from lilavati.connectors.edgar import EdgarClient
from lilavati.connectors.india_filings import IndiaFilingsClient, register_disclosure
from lilavati.core.provenance import CitationRef, SourcedValue, SourceStore
from lilavati.framework import FinancialState, ProvenanceGate

FIX = Path(__file__).parent / "fixtures"


def _apple() -> tuple[SourceStore, dict[str, SourcedValue]]:
    facts = json.loads((FIX / "aapl_companyfacts_statements_fy2026.json").read_text())
    store = SourceStore()
    assert store.add_xbrl_company_facts(facts) == 69  # every fact record in the fixture
    model = EdgarClient().extract_three_statement_model("AAPL", 2026, facts=facts)
    return store, model.sourced_facts


def _with(sv: SourcedValue, value: float | None = None, **citation: object) -> SourcedValue:
    return SourcedValue(value=sv.value if value is None else value, unit=sv.unit, scale=sv.scale,
                        citation=sv.citation.model_copy(update=citation))


def test_xbrl_citations_resolve_and_mismatches_are_named() -> None:
    store, facts = _apple()
    assert all(store.check(sv.citation, sv.normalized_value) == [] for sv in facts.values())

    rev = facts["revenue"]
    assert "differs from the record's" in store.check(rev.citation, rev.value + 1)[0]
    assert "cited tag us-gaap:Assets" in store.check(rev.citation.model_copy(update={"xbrl_tag": "us-gaap:Assets"}))[0]
    assert "period end" in store.check(rev.citation.model_copy(update={"period_end": date(2025, 9, 27)}))[0]
    assert "not in the source store" in store.check(rev.citation.model_copy(update={"doc_hash": "0" * 64}))[0]


def test_gate_with_a_store_rejects_what_the_form_check_passes() -> None:
    store, facts = _apple()
    state = FinancialState()
    state.sourced_values.extend(facts.values())
    assert ProvenanceGate(store)(state).is_valid

    # A revenue figure one dollar off and a citation to a source nobody read: both well formed.
    tampered = _with(facts["revenue"], value=facts["revenue"].value + 1)
    unknown = _with(facts["net_income"], doc_hash="1" * 64)
    for bad in (tampered, unknown):
        form_only = FinancialState(sourced_values=[bad])
        assert ProvenanceGate()(form_only).is_valid  # the old behaviour, still the default
        checked = ProvenanceGate(store)(FinancialState(sourced_values=[bad]))
        assert not checked.is_valid and len(checked.gate_violations) == 1

    cit_state = FinancialState()
    cit_state.record_citation(unknown.citation)
    assert not ProvenanceGate(store)(cit_state).is_valid


def test_document_snippet_must_be_on_the_cited_page() -> None:
    # Two pages as pypdf extracts them (income statement, then balance sheet rows).
    pages = ["Total net sales  478,834  416,161  391,035 \nProvision for income taxes  27,062  20,719  29,749 ",
             "Total assets $ 404,548 $ 359,241 "]
    store = SourceStore()
    h = store.add_document(b"10-K bytes", pages)
    cit = CitationRef(doc_hash=h, filing_type="10-K", filing_date=date(2026, 11, 3), period_end=date(2026, 9, 26),
                      page=1, raw_snippet="Provision for income taxes 27,062 20,719 29,749")
    assert store.check(cit) == []  # whitespace is ignored on both sides
    assert "not found on page 2" in store.check(cit.model_copy(update={"page": 2}))[0]
    assert "beyond the document's 2 pages" in store.check(cit.model_copy(update={"page": 3}))[0]
    assert "not found" in store.check(cit.model_copy(update={"raw_snippet": "Provision for income taxes 27,063"}))[0]


def test_ind_as_citations_resolve_against_the_registered_disclosure() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_india_filings import FACTS, _disclosure

    d = _disclosure(FACTS)
    store = SourceStore()
    register_disclosure(store, d)
    res = IndiaFilingsClient.parse_ind_as_disclosure(d)
    assert all(store.check(sv.citation, sv.normalized_value) == [] for sv in res.sourced_facts.values())
    rev = res.sourced_facts["RevenueFromOperations"]
    assert "differs from the disclosure's" in store.check(rev.citation, rev.value * 1.01)[0]
    # A different disclosure (one fact changed) has a different hash: its citations do not resolve here.
    other = IndiaFilingsClient.parse_ind_as_disclosure(_disclosure({**FACTS, "OtherIncome": 201.0}))
    assert "not in the source store" in store.check(other.sourced_facts["RevenueFromOperations"].citation)[0]


def test_snippet_match_ignores_spacing_from_text_extraction() -> None:
    # Microsoft's FY2026 10-K as HTML-to-text gives "(139,500 )": a space between the table cells. A
    # model quoting "(139,500)" quotes the document correctly; the characters must still all match.
    page = "Net cash used in investing (139,500 ) (72,599 ) (96,970 ) Effect of foreign exchange rates"
    store = SourceStore()
    h = store.add_document(b"10-K html", [page])
    cit = CitationRef(doc_hash=h, filing_type="10-K", filing_date=date(2026, 7, 29), period_end=date(2026, 6, 30),
                      page=None, raw_snippet="Net cash used in investing (139,500) (72,599) (96,970)")
    assert store.check(cit) == []
    assert store.check(cit.model_copy(update={"raw_snippet": "Net cash used in investing (139,600)"}))


def test_snippet_match_treats_typographic_quotes_as_plain() -> None:
    # Apple's 10-K prints "shareholders’ equity" (U+2019); qwen3.6-27b quoted it with an ASCII apostrophe.
    store = SourceStore()
    h = store.add_document(b"10-K", ["Total shareholders’ equity 112,093 73,733"])
    cit = CitationRef(doc_hash=h, filing_type="10-K", filing_date=date(2026, 11, 3), period_end=date(2026, 9, 26),
                      page=None, raw_snippet="Total shareholders' equity 112,093 73,733")
    assert store.check(cit) == []
