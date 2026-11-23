"""
Financial search agent for Project Līlāvatī.

US: the latest SEC 10-K filed on or before the as-of date (point in time), with figures from that
filing's own XBRL facts (`lilavati.connectors.sec_filing`); every fact carries the filing's
citation. Needs a contact in ``LILAVATI_SEC_USER_AGENT``. If SEC cannot be reached, the report has no
facts and says why in ``notes``; nothing is filled in.

India: live Ind AS retrieval is not implemented; parse a disclosure you hold with
``IndiaFilingsClient.parse_ind_as_disclosure``.
"""

from __future__ import annotations

from datetime import date

import httpx
from pydantic import BaseModel, ConfigDict, Field

from lilavati.connectors.edgar import EdgarClient
from lilavati.connectors.india_filings import IndiaFilingsClient
from lilavati.connectors.sec_filing import extract_filing, filings_from_submissions
from lilavati.core.provenance import CitationRef
from lilavati.core.temporal import BiTemporalWarehouse
from lilavati.llm.client import BaseLLMClient, UniversalLLMClient
from lilavati.llm.providers import INSTITUTIONAL_MODEL_PRESETS, ModelSpec

# canonical item in sec_filing -> (report metric name, topic)
_US_METRICS = {
    "revenue": ("revenue", "income_statement"),
    "net_income": ("net_income", "income_statement"),
    "total_assets": ("total_assets", "balance_sheet"),
    "total_liabilities": ("total_liabilities", "balance_sheet"),
    "operating_cash_flow": ("cfo", "cash_flow"),
}


class SearchFact(BaseModel):
    """A factual financial claim discovered by the search agent."""

    model_config = ConfigDict(frozen=True)

    topic: str
    metric_name: str
    value: float
    unit: str = "USD_MILLIONS"
    citation: CitationRef


class SearchReport(BaseModel):
    """Comprehensive discovery artifact produced by the FinancialSearchAgent."""

    model_config = ConfigDict(frozen=True)

    entity_symbol: str
    as_of_date: date
    discovered_facts: list[SearchFact] = Field(default_factory=list)
    verified_citations: list[CitationRef] = Field(default_factory=list)
    searched_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="why facts are missing, if they are")
    summary: str = ""


class FinancialSearchAgent:
    """Filing search: figures from the latest filing known at the as-of date, each with its citation."""

    def __init__(
        self,
        model_spec: ModelSpec | None = None,
        llm_client: BaseLLMClient | None = None,
        warehouse: BiTemporalWarehouse | None = None,
        edgar_client: EdgarClient | None = None,
    ) -> None:
        self.spec = model_spec or INSTITUTIONAL_MODEL_PRESETS["mock-evaluator"]
        self.llm = llm_client or UniversalLLMClient(spec=self.spec)
        self.warehouse = warehouse or BiTemporalWarehouse()
        self.edgar = edgar_client or EdgarClient()
        self.india = IndiaFilingsClient()

    def search_entity(
        self,
        symbol: str,
        topics: list[str],
        as_of_date: date,
        jurisdiction: str = "US",
    ) -> SearchReport:
        """
        Figures for ``symbol`` from the latest annual filing published on or before ``as_of_date``.

        Values are USD millions. ``topics`` filters by topic ("income_statement", "balance_sheet",
        "cash_flow"); an empty list keeps all.
        """
        facts: list[SearchFact] = []
        notes: list[str] = []
        sources: list[str] = []

        if jurisdiction.upper() == "US":
            sources.append("SEC_EDGAR_XBRL")
            try:
                filings = [f for f in filings_from_submissions(self.edgar.fetch_submissions(symbol))
                           if f.filing_date <= as_of_date]
                if not filings:
                    notes.append(f"No 10-K for {symbol} filed on or before {as_of_date} in SEC's recent index.")
                else:
                    filing = filings[0]
                    ex = extract_filing(self.edgar.fetch_company_facts(filing.cik), filing)
                    for item, (metric, topic) in _US_METRICS.items():
                        if topics and topic not in topics:
                            continue
                        if item not in ex.values:
                            notes.append(f"{item} not reported in {filing.form} {filing.accession}.")
                            continue
                        sv = ex.values[item]
                        facts.append(SearchFact(topic=topic, metric_name=metric, value=sv.value / 1e6,
                                                citation=sv.citation))
            except (ValueError, FileNotFoundError, RuntimeError, httpx.HTTPError) as e:
                notes.append(f"SEC EDGAR unavailable, no facts returned: {e}")
        elif jurisdiction.upper() == "IN":
            sources.append("IND_AS")
            notes.append("Live Ind AS retrieval is not implemented; parse a disclosure you hold with "
                         "IndiaFilingsClient.parse_ind_as_disclosure.")
        else:
            notes.append(f"Unsupported jurisdiction {jurisdiction!r}; use 'US' or 'IN'.")

        citations = [f.citation for f in facts]
        summary_text = (f"{len(facts)} facts for {symbol} ({jurisdiction}) as of {as_of_date}, each with a citation "
                        "to the filing's XBRL fact." if facts else
                        f"No facts for {symbol} ({jurisdiction}) as of {as_of_date}: " + " ".join(notes))
        return SearchReport(
            entity_symbol=symbol,
            as_of_date=as_of_date,
            discovered_facts=facts,
            verified_citations=citations,
            searched_sources=sources,
            notes=notes,
            summary=summary_text,
        )
