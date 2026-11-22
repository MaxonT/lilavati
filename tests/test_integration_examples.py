"""Offline tests for the framework-integration examples' answer checker and wiring."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "integrations"))
task = importlib.import_module("_task")

BOND = {"clean_price": 95.04852622680232, "dv01": 0.0565}
OPTION = {"spot": 100.0, "strike": 105.0, "price": 8.363276638981155}


@pytest.mark.parametrize(
    "results",
    [
        [BOND, OPTION],  # plain dicts
        [{"result": BOND}, {"result": OPTION}],  # wrapped
        [{"content": [{"type": "text", "text": json.dumps(BOND)}]},  # ADK MCP result
         {"structuredContent": OPTION}],
        [[{"type": "text", "text": json.dumps(BOND)}], json.dumps(OPTION)],  # LangChain content
    ],
)
def test_tool_values_handles_framework_result_shapes(results: list[object]) -> None:
    v = task.tool_values(results)
    assert v["bond_clean_price"] == pytest.approx(95.0485, abs=1e-4)
    assert v["option_price"] == pytest.approx(8.3633, abs=1e-4)


def test_check_answer_accepts_quoted_tool_values() -> None:
    ans = "1) clean price 95.0485. 2) call price 8.3633."
    assert task.check_answer(ans, [BOND, OPTION]) == []


def test_check_answer_rejects_model_computed_numbers() -> None:
    problems = task.check_answer("bond 95.10, call 8.40", [BOND, OPTION])
    assert len(problems) == 2


def test_check_answer_flags_missing_tool_call() -> None:
    assert any("not called" in p for p in task.check_answer("95.0485", [BOND]))


def test_servers_are_real_console_scripts() -> None:
    import tomllib

    scripts = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["scripts"]
    for cmd in task.SERVERS.values():
        assert cmd[-1] in scripts


def test_adk_agent_builds_when_adk_installed() -> None:
    pytest.importorskip("google.adk")
    adk_agent = importlib.import_module("adk_agent")
    agent = adk_agent.build_agent()
    assert agent.name == "lilavati_analyst" and len(agent.tools) == 2
