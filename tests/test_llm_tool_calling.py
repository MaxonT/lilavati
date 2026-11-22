"""Provider tool-calling round trips through the agent loop, with HTTP mocked.

Request/response shapes follow each provider's public REST docs:
- Gemini generateContent: tools[].functionDeclarations, parts[].functionCall / functionResponse,
  systemInstruction, and thoughtSignature echoed back on function-call parts (Gemini 3).
- Anthropic Messages: tools[] with input_schema, content blocks tool_use / tool_result.
- OpenAI Chat Completions: tools[].function, message.tool_calls, role "tool" replies.
No network access: requests go to an httpx.MockTransport.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lilavati.llm.client import LLMMessage, UniversalLLMClient
from lilavati.llm.providers import (
    GROQ_BASE_URL,
    INSTITUTIONAL_MODEL_PRESETS,
    ModelProvider,
    ModelSpec,
)
from lilavati.runtime.agent import AgentConfig, NeuroSymbolicAgent

SCHEMA = {
    "type": "object",
    "title": "AddBpsArgs",
    "additionalProperties": False,
    "properties": {
        "rate": {"type": "number", "description": "rate as a decimal", "title": "Rate"},
        "bps": {"type": "number", "default": 0},
    },
    "required": ["rate", "bps"],
}


def _gemini_replies() -> list[dict[str, Any]]:
    return [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "add_bps",
                                    "args": {"rate": 0.05, "bps": 25},
                                },
                                "thoughtSignature": "sig-123",
                            }
                        ],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        },
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "New rate is 0.0525."}]}}
            ]
        },
    ]


def _anthropic_replies() -> list[dict[str, Any]]:
    return [
        {
            "content": [
                {"type": "text", "text": "Calling tool."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "add_bps",
                    "input": {"rate": 0.05, "bps": 25},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "content": [{"type": "text", "text": "New rate is 0.0525."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        },
    ]


def _openai_replies() -> list[dict[str, Any]]:
    return [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "add_bps",
                                    "arguments": json.dumps({"rate": 0.05, "bps": 25}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"role": "assistant", "content": "New rate is 0.0525."}}]},
    ]


def _run(
    provider: ModelProvider, replies: list[dict[str, Any]]
) -> tuple[Any, list[dict[str, Any]], list[httpx.Request]]:
    sent: list[httpx.Request] = []
    bodies: list[dict[str, Any]] = []
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    spec = ModelSpec(provider=provider, model_name="test-model", base_url="https://example.test/v1")
    client = UniversalLLMClient(spec, api_key="test-key", transport=httpx.MockTransport(handler))
    agent = NeuroSymbolicAgent(
        AgentConfig(agent_id="t", role_name="Rates", model_spec=spec, system_prompt="Use tools."),
        llm_client=client,
    )
    agent.register_tool(
        "add_bps", "Add basis points to a rate.", SCHEMA, lambda rate, bps: rate + bps / 10_000
    )
    out = agent.run("Add 25bp to 5%.")
    return out, bodies, sent


def test_gemini_tool_round_trip() -> None:
    out, bodies, sent = _run(ModelProvider.GEMINI, _gemini_replies())
    assert out.tool_calls_executed == 1
    assert out.response_text == "New rate is 0.0525."
    first, second = bodies
    decl = first["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "add_bps"
    # Gemini's OpenAPI-subset schema rejects these JSON Schema keywords.
    params = json.dumps(decl["parameters"])
    assert (
        "additionalProperties" not in params
        and '"title"' not in params
        and '"default"' not in params
    )
    assert decl["parameters"]["required"] == ["rate", "bps"]
    assert first["systemInstruction"]["parts"][0]["text"] == "Use tools."
    assert all(c["role"] in ("user", "model") for c in first["contents"])
    model_turn = second["contents"][-2]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["functionCall"]["name"] == "add_bps"
    assert model_turn["parts"][0]["thoughtSignature"] == "sig-123"
    fr = second["contents"][-1]["parts"][0]["functionResponse"]
    assert fr["name"] == "add_bps"
    assert fr["response"]["result"] == pytest.approx(0.0525)
    # Key travels in a header, never in the URL.
    assert sent[0].headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(sent[0].url)


def test_anthropic_tool_round_trip() -> None:
    out, bodies, sent = _run(ModelProvider.ANTHROPIC, _anthropic_replies())
    assert out.tool_calls_executed == 1
    assert out.response_text == "New rate is 0.0525."
    first, second = bodies
    assert first["tools"][0]["name"] == "add_bps"
    assert first["tools"][0]["input_schema"]["required"] == ["rate", "bps"]
    assert first["system"] == "Use tools."
    assistant = second["messages"][-2]
    assert assistant["role"] == "assistant"
    tool_use = [b for b in assistant["content"] if b["type"] == "tool_use"][0]
    assert tool_use == {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "add_bps",
        "input": {"rate": 0.05, "bps": 25},
    }
    result_msg = second["messages"][-1]
    assert result_msg["role"] == "user"
    block = result_msg["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_01"
    assert json.loads(block["content"]) == pytest.approx(0.0525)
    assert sent[0].headers["x-api-key"] == "test-key"


def test_openai_tool_round_trip_still_works() -> None:
    out, bodies, _ = _run(ModelProvider.OPENAI, _openai_replies())
    assert out.tool_calls_executed == 1
    second = bodies[1]
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_call_id"] == "call_1"


def test_anthropic_merges_parallel_tool_results_into_one_user_turn() -> None:
    replies = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "add_bps",
                    "input": {"rate": 0.01, "bps": 1},
                },
                {
                    "type": "tool_use",
                    "id": "b",
                    "name": "add_bps",
                    "input": {"rate": 0.02, "bps": 2},
                },
            ],
            "usage": {},
        },
        {"content": [{"type": "text", "text": "done"}], "usage": {}},
    ]
    out, bodies, _ = _run(ModelProvider.ANTHROPIC, replies)
    assert out.tool_calls_executed == 2
    roles = [m["role"] for m in bodies[1]["messages"]]
    # Anthropic requires alternating roles: both results go in a single user message.
    assert roles == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in bodies[1]["messages"][-1]["content"]] == ["a", "b"]


def test_tool_error_is_reported_to_model_not_raised() -> None:
    replies = _anthropic_replies()
    replies[0]["content"][1]["input"] = {"rate": 0.05}  # missing required arg -> TypeError
    out, bodies, _ = _run(ModelProvider.ANTHROPIC, replies)
    block = bodies[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "Tool Error" in block["content"]


def _tool_turn(i: int) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": f"t{i}",
                "name": "add_bps",
                "input": {"rate": 0.01, "bps": 1},
            }
        ],
        "usage": {},
    }


def test_multiple_tool_rounds_are_not_capped_by_reflection_limit() -> None:
    # Previously max_reflection_iterations (3) also capped tool rounds: a 3-round tool plan
    # ended with an empty answer while still reporting success.
    replies = [_tool_turn(i) for i in range(3)] + [
        {"content": [{"type": "text", "text": "final"}], "usage": {}}
    ]
    out, _, _ = _run(ModelProvider.ANTHROPIC, replies)
    assert out.tool_calls_executed == 3
    assert out.response_text == "final"
    assert out.completed is True


def test_tool_round_limit_reports_incomplete() -> None:
    replies = [_tool_turn(i) for i in range(20)]
    out, _, _ = _run(ModelProvider.ANTHROPIC, replies)
    assert out.completed is False
    assert out.tool_calls_executed == 8  # default max_tool_rounds


def test_transient_errors_are_retried_then_succeed() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json=_anthropic_replies()[1])

    spec = ModelSpec(provider=ModelProvider.ANTHROPIC, model_name="m", retry_backoff_seconds=0.0)
    client = UniversalLLMClient(spec, api_key="k", transport=httpx.MockTransport(handler))
    resp = client.generate([LLMMessage(role="user", content="hi")])
    assert resp.content == "New rate is 0.0525." and calls["n"] == 3


def test_non_retryable_error_raises_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    spec = ModelSpec(provider=ModelProvider.ANTHROPIC, model_name="m", retry_backoff_seconds=0.0)
    client = UniversalLLMClient(spec, api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.generate([LLMMessage(role="user", content="hi")])
    assert calls["n"] == 1


def test_retries_are_bounded() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={})

    spec = ModelSpec(
        provider=ModelProvider.ANTHROPIC, model_name="m", max_retries=2, retry_backoff_seconds=0.0
    )
    client = UniversalLLMClient(spec, api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.generate([LLMMessage(role="user", content="hi")])
    assert calls["n"] == 3


@pytest.mark.parametrize(
    "preset",
    ["groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen3.6-27b", "groq-qwen3.8-27b"],
)
def test_groq_presets_use_env_key_in_header_only(
    preset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "gsk_test_secret_value_0123456789"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_openai_replies()[0])

    spec = INSTITUTIONAL_MODEL_PRESETS[preset]
    client = UniversalLLMClient(spec, transport=httpx.MockTransport(handler))
    resp = client.generate([LLMMessage(role="user", content="Add 25bp to 5%.")])
    assert resp.tool_calls[0].name == "add_bps"
    req = sent[0]
    assert str(req.url) == f"{GROQ_BASE_URL}/chat/completions"
    assert req.headers["Authorization"] == f"Bearer {secret}"
    assert secret not in str(req.url)
    assert secret not in req.content.decode()
    assert json.loads(req.content)["model"] == spec.model_name


def test_ollama_preset_targets_openai_compatible_endpoint() -> None:
    # Ollama serves the OpenAI-compatible API under /v1; the preset once pointed at the root,
    # so requests went to /chat/completions and could not work.
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_openai_replies()[1])

    spec = INSTITUTIONAL_MODEL_PRESETS["llama3-fin-8b"]
    UniversalLLMClient(spec, transport=httpx.MockTransport(handler)).generate(
        [LLMMessage(role="user", content="hi")]
    )
    assert str(sent[0].url) == "http://localhost:11434/v1/chat/completions"
