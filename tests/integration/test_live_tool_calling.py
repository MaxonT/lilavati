"""Opt-in live tool-calling tests against real model APIs.

Skipped unless LILAVATI_LIVE_TESTS=1 and the provider's API key is set in the environment.
Each test makes about 2-3 API calls. Run with:

    LILAVATI_LIVE_TESTS=1 GEMINI_API_KEY=... uv run pytest tests/integration -v
    LILAVATI_LIVE_TESTS=1 ANTHROPIC_API_KEY=... uv run pytest tests/integration -v
    LILAVATI_LIVE_TESTS=1 GROQ_API_KEY=... uv run pytest tests/integration -v

Model overrides: LILAVATI_LIVE_GEMINI_MODEL, LILAVATI_LIVE_ANTHROPIC_MODEL, LILAVATI_LIVE_GROQ_PRESET.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest

from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.llm.client import UniversalLLMClient
from lilavati.llm.providers import INSTITUTIONAL_MODEL_PRESETS, ModelProvider, ModelSpec
from lilavati.runtime.agent import AgentConfig, NeuroSymbolicAgent

LIVE = os.environ.get("LILAVATI_LIVE_TESTS") == "1"

BOND_SCHEMA = {
    "type": "object",
    "properties": {
        "face_value": {"type": "number", "description": "Face value, e.g. 100"},
        "coupon_rate": {"type": "number", "description": "Annual coupon as a decimal"},
        "ytm": {"type": "number", "description": "Yield to maturity as a decimal"},
        "years_to_maturity": {"type": "number"},
        "frequency": {"type": "integer", "description": "Coupons per year"},
    },
    "required": ["face_value", "coupon_rate", "ytm", "years_to_maturity", "frequency"],
}


def _price_bond(**kw: Any) -> dict[str, Any]:
    return FixedIncomeEngine.analyze_bond(**kw).model_dump()


def _run(spec: ModelSpec) -> tuple[Any, float]:
    agent = NeuroSymbolicAgent(
        AgentConfig(
            agent_id="live",
            role_name="Fixed income",
            model_spec=spec,
            system_prompt=(
                "You must call the analyze_bond tool for any bond calculation and report the "
                "clean_price it returns, rounded to 4 decimals. Do not compute it yourself."
            ),
        ),
        llm_client=UniversalLLMClient(spec),
    )
    agent.register_tool("analyze_bond", "Price a fixed-coupon bond.", BOND_SCHEMA, _price_bond)
    out = agent.run(
        "Price a 7-year bond, face 100, 4.25% annual coupon paid semiannually, yield 5.1%."
    )
    expected = FixedIncomeEngine.analyze_bond(
        face_value=100, coupon_rate=0.0425, ytm=0.051, years_to_maturity=7, frequency=2
    ).clean_price
    return out, expected


def _assert_used_tool(out: Any, expected: float) -> None:
    assert out.completed, out.response_text
    assert out.tool_calls_executed >= 1
    numbers = [float(x) for x in re.findall(r"\d+\.\d+", out.response_text.replace(",", ""))]
    assert any(abs(n - expected) < 5e-4 for n in numbers), (out.response_text, expected)


@pytest.mark.skipif(not (LIVE and os.environ.get("GEMINI_API_KEY")), reason="live Gemini test")
def test_live_gemini_calls_tool() -> None:
    spec = ModelSpec(
        provider=ModelProvider.GEMINI,
        model_name=os.environ.get("LILAVATI_LIVE_GEMINI_MODEL", "gemini-3.8-flash"),
        api_key_env_var="GEMINI_API_KEY",
        max_tokens=4096,
        timeout_seconds=180,
    )
    _assert_used_tool(*_run(spec))


@pytest.mark.skipif(
    not (LIVE and os.environ.get("ANTHROPIC_API_KEY")), reason="live Anthropic test"
)
def test_live_anthropic_calls_tool() -> None:
    spec = ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_name=os.environ.get("LILAVATI_LIVE_ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        api_key_env_var="ANTHROPIC_API_KEY",
        max_tokens=2048,
        timeout_seconds=180,
    )
    _assert_used_tool(*_run(spec))


@pytest.mark.skipif(not (LIVE and os.environ.get("GROQ_API_KEY")), reason="live Groq test")
def test_live_groq_calls_tool() -> None:
    preset = os.environ.get("LILAVATI_LIVE_GROQ_PRESET", "groq-gpt-oss-120b")
    _assert_used_tool(*_run(INSTITUTIONAL_MODEL_PRESETS[preset]))
