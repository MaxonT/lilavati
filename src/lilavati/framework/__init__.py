"""
Pipeline primitives: a small state graph with gates, and an agent wrapper.

- FinancialStateGraph: runs node functions in order or by conditional edges, with tracing.
- FinancialState: Typed base state carrying messages, citations, facts, and verification flags.
- FinancialAgent: Developer agent abstraction with plug-and-play LLMs and FastMCP tools.
- Verification Gates: SMTBalanceGate, CovenantGate, ProvenanceGate, TemporalPurityGate.
"""

from lilavati.framework.agent import FinancialAgent
from lilavati.framework.gates import (
    CovenantGate,
    ProvenanceGate,
    SMTBalanceGate,
    TemporalPurityGate,
)
from lilavati.framework.graph import END, CompiledFinancialGraph, FinancialStateGraph
from lilavati.framework.serving import create_agent_api, serve_agent
from lilavati.framework.state import FinancialState

__all__ = [
    "FinancialState",
    "FinancialStateGraph",
    "CompiledFinancialGraph",
    "FinancialAgent",
    "END",
    "SMTBalanceGate",
    "CovenantGate",
    "ProvenanceGate",
    "TemporalPurityGate",
    "create_agent_api",
    "serve_agent",
]
