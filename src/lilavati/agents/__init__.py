"""Agents: filing search, extraction with repair, research dossier, rule-based committee, scenario tree.

What each does and does not do: docs/agents.md.
"""

from lilavati.agents.committee import (
    AdversarialCommittee,
    CommitteeRecommendation,
    InvestmentCommitteeMemo,
)
from lilavati.agents.deep import DeepFinancialAgent, DeepReasoningTree, ScenarioBranch
from lilavati.agents.extraction import ExtractionResult, FilingExtractionAgent, llm_extractor
from lilavati.agents.forensics import AltmanZScoreResult, BeneishMScoreResult, ForensicEngine
from lilavati.agents.research import FinancialResearchAgent, ResearchDossier
from lilavati.agents.search import FinancialSearchAgent, SearchFact, SearchReport

__all__ = [
    "ForensicEngine",
    "BeneishMScoreResult",
    "AltmanZScoreResult",
    "AdversarialCommittee",
    "InvestmentCommitteeMemo",
    "CommitteeRecommendation",
    "FinancialSearchAgent",
    "SearchReport",
    "SearchFact",
    "DeepFinancialAgent",
    "DeepReasoningTree",
    "ScenarioBranch",
    "FinancialResearchAgent",
    "ResearchDossier",
    "FilingExtractionAgent",
    "ExtractionResult",
    "llm_extractor",
]
