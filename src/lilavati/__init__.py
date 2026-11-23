"""
Līlāvatī (लीलावती)
==================
An open-source tool layer (alpha) with checkable numbers, and a benchmark suite,
for institutional quantitative finance, credit, econometrics, and algorithmic execution.
"""

from lilavati.agents.deep import DeepFinancialAgent, DeepReasoningTree
from lilavati.agents.research import FinancialResearchAgent, ResearchDossier
from lilavati.agents.search import FinancialSearchAgent, SearchReport
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef, SourcedValue, SourceStore
from lilavati.framework.agent import FinancialAgent
from lilavati.framework.gates import (
    CovenantGate,
    ProvenanceGate,
    SMTBalanceGate,
    TemporalPurityGate,
)
from lilavati.framework.graph import END, CompiledFinancialGraph, FinancialStateGraph
from lilavati.framework.state import FinancialState
from lilavati.tracing.tracer import Tracer

__version__ = "0.2.0"

__all__ = [
    "FinancialAgent",
    "FinancialSearchAgent",
    "DeepFinancialAgent",
    "FinancialResearchAgent",
    "ResearchDossier",
    "DeepReasoningTree",
    "SearchReport",
    "FinancialStateGraph",
    "CompiledFinancialGraph",
    "FinancialState",
    "END",
    "SMTBalanceGate",
    "CovenantGate",
    "ProvenanceGate",
    "TemporalPurityGate",
    "CitationRef",
    "SourcedValue",
    "SourceStore",
    "SuvarnaEngine",
    "Tracer",
    "__version__",
]
