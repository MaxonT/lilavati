"""
Multi-Provider LLM Client for Financial Agent Operations.

Connects to Gemini, Anthropic, OpenAI, or local OSS financial models (via vLLM / Ollama)
with support for FastMCP tool declarations, function calling, structured outputs, and automated retries.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from lilavati.llm.providers import ModelProvider, ModelSpec

T = TypeVar("T", bound=BaseModel)


class LLMToolCall(BaseModel):
    """Tool invocation request emitted by the language model."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]
    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque provider data to echo back (e.g. Gemini thoughtSignature).",
    )


class LLMMessage(BaseModel):
    """Normalized chat message for LLM interactions."""

    role: str = Field(..., description="system, user, assistant, or tool")
    content: str = Field(default="")
    tool_calls: list[LLMToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = Field(default=None, description="Tool name, for role='tool' messages")
    is_error: bool = Field(default=False, description="True if a role='tool' message reports an error")


class LLMUsage(BaseModel):
    """Token consumption metrics for auditing and cost attribution."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Normalized response from an LLM model."""

    content: str
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient:
    """Abstract interface for LLM model connectivity."""

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[T],
    ) -> T:
        raise NotImplementedError


class UniversalLLMClient(BaseLLMClient):
    """
    Universal HTTP-based LLM client supporting:
    - Gemini (generativelanguage.googleapis.com)
    - OpenAI / vLLM / Ollama (OpenAI-compatible /chat/completions)
    - Anthropic (/v1/messages)
    """

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.spec = spec
        self._transport = transport
        if api_key:
            self.api_key = api_key
        elif spec.api_key_env_var:
            self.api_key = os.environ.get(spec.api_key_env_var, "")
        else:
            self.api_key = ""

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self.spec.provider == ModelProvider.MOCK:
            return LLMResponse(
                content="[Mock LLM Response] Operation completed according to institutional constraints.",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        if self.spec.provider in (ModelProvider.OPENAI, ModelProvider.VLLM, ModelProvider.OLLAMA):
            return self._call_openai_compatible(messages, tools)
        elif self.spec.provider == ModelProvider.GEMINI:
            return self._call_gemini(messages, tools)
        elif self.spec.provider == ModelProvider.ANTHROPIC:
            return self._call_anthropic(messages, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.spec.provider}")

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[T],
    ) -> T:
        """Forces the LLM to output valid JSON conforming to a Pydantic schema."""
        schema_json = json.dumps(response_model.model_json_schema())
        system_instruction = f"You MUST respond ONLY with a valid JSON object strictly matching this schema:\n{schema_json}"
        augmented = [LLMMessage(role="system", content=system_instruction)] + messages
        response = self.generate(augmented)

        raw_text = response.content.strip()
        # Clean markdown code blocks if present
        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.removeprefix("```").removesuffix("```").strip()

        data = json.loads(raw_text)
        return response_model.model_validate(data)

    def _call_openai_compatible(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        base_url = self.spec.base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        openai_messages: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            openai_messages.append(entry)

        payload: dict[str, Any] = {
            "model": self.spec.model_name,
            "messages": openai_messages,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        data = self._post_json(url, headers, payload)

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        parsed_tools: list[LLMToolCall] = []

        if "tool_calls" in choice and choice["tool_calls"]:
            for tc in choice["tool_calls"]:
                fn = tc["function"]
                args = (
                    json.loads(fn["arguments"])
                    if isinstance(fn["arguments"], str)
                    else fn["arguments"]
                )
                parsed_tools.append(LLMToolCall(id=tc["id"], name=fn["name"], arguments=args))

        usage_dict = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_dict.get("prompt_tokens", 0),
            completion_tokens=usage_dict.get("completion_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
        )
        return LLMResponse(content=content, tool_calls=parsed_tools, usage=usage, raw_response=data)

    def _http(self) -> httpx.Client:
        return httpx.Client(timeout=self.spec.timeout_seconds, transport=self._transport)

    _RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

    def _post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST with bounded exponential-backoff retries on 429/5xx and timeouts.

        Honours a numeric Retry-After header (capped at 60 s). Non-retryable errors and the
        final failed attempt raise httpx errors; messages never include request headers.
        """
        attempts = self.spec.max_retries + 1
        for attempt in range(attempts):
            try:
                with self._http() as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code in self._RETRY_STATUS and attempt < attempts - 1:
                    retry_after = resp.headers.get("retry-after", "")
                    delay = (
                        float(retry_after)
                        if retry_after.replace(".", "", 1).isdigit()
                        else self.spec.retry_backoff_seconds * (2**attempt)
                    )
                    time.sleep(min(delay, 60.0))
                    continue
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data
            except httpx.TimeoutException:
                if attempt >= attempts - 1:
                    raise
                time.sleep(min(self.spec.retry_backoff_seconds * (2**attempt), 60.0))
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _tool_names_by_id(messages: list[LLMMessage]) -> dict[str, str]:
        names: dict[str, str] = {}
        for m in messages:
            for tc in m.tool_calls or []:
                names[tc.id] = tc.name
        return names

    @staticmethod
    def _function_specs(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Accept OpenAI-style {"type": "function", "function": {...}} or bare {name, ...}."""
        specs: list[dict[str, Any]] = []
        for t in tools or []:
            fn = t.get("function", t)
            specs.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return specs

    def _call_gemini(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Gemini generateContent with function calling (functionDeclarations)."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.spec.model_name}:generateContent"
        )
        # Key goes in a header (not the URL) so it can never leak via logs/tracebacks.
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        names = self._tool_names_by_id(messages)
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []

        def append(role: str, part: dict[str, Any]) -> None:
            # Gemini expects alternating turns; merge consecutive parts of the same role.
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": role, "parts": [part]})

        for m in messages:
            if m.role == "system":
                system_texts.append(m.content)
            elif m.role == "tool":
                name = m.name or names.get(m.tool_call_id or "", "tool")
                payload_value = _parse_tool_output(m.content)
                key = "error" if m.is_error else "result"
                append(
                    "user",
                    {"functionResponse": {"name": name, "response": {key: payload_value}}},
                )
            elif m.role == "assistant":
                if m.content:
                    append("model", {"text": m.content})
                for tc in m.tool_calls or []:
                    part: dict[str, Any] = {"functionCall": {"name": tc.name, "args": tc.arguments}}
                    if "thoughtSignature" in tc.provider_metadata:
                        part["thoughtSignature"] = tc.provider_metadata["thoughtSignature"]
                    append("model", part)
            else:
                append("user", {"text": m.content})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.spec.temperature,
                "maxOutputTokens": self.spec.max_tokens,
            },
        }
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        specs = self._function_specs(tools)
        if specs:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {**sp, "parameters": _gemini_schema(sp["parameters"])} for sp in specs
                    ]
                }
            ]

        data = self._post_json(url, headers, payload)

        candidates = data.get("candidates", [])
        usage_md = data.get("usageMetadata", {})
        usage = LLMUsage(
            prompt_tokens=usage_md.get("promptTokenCount", 0),
            completion_tokens=usage_md.get("candidatesTokenCount", 0),
            total_tokens=usage_md.get("totalTokenCount", 0),
        )
        if not candidates:
            return LLMResponse(content="", usage=usage, raw_response=data)

        parts = candidates[0].get("content", {}).get("parts", [])
        text_content = "".join(
            p.get("text", "") for p in parts if "text" in p and not p.get("thought")
        )
        calls: list[LLMToolCall] = []
        for i, p in enumerate(parts):
            fc = p.get("functionCall")
            if not fc:
                continue
            meta = {"thoughtSignature": p["thoughtSignature"]} if "thoughtSignature" in p else {}
            calls.append(
                LLMToolCall(
                    id=fc.get("id") or f"gemini_call_{i}",
                    name=fc["name"],
                    arguments=fc.get("args") or {},
                    provider_metadata=meta,
                )
            )
        return LLMResponse(content=text_content, tool_calls=calls, usage=usage, raw_response=data)

    def _call_anthropic(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Anthropic Messages API with tool use (tool_use / tool_result content blocks)."""
        url = (self.spec.base_url or "https://api.anthropic.com").rstrip("/")
        url = url if url.endswith("/v1/messages") else f"{url.removesuffix('/v1')}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        system_prompts = [m.content for m in messages if m.role == "system"]
        chat: list[dict[str, Any]] = []

        def append(role: str, blocks: list[dict[str, Any]]) -> None:
            # Roles must alternate; merge consecutive same-role messages (e.g. parallel results).
            if chat and chat[-1]["role"] == role:
                chat[-1]["content"].extend(blocks)
            else:
                chat.append({"role": role, "content": list(blocks)})

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
                if m.is_error:
                    block["is_error"] = True
                append("user", [block])
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls or []:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                append("assistant", blocks or [{"type": "text", "text": ""}])
            else:
                append("user", [{"type": "text", "text": m.content}])

        payload: dict[str, Any] = {
            "model": self.spec.model_name,
            "messages": chat,
            "max_tokens": self.spec.max_tokens,
            "temperature": self.spec.temperature,
        }
        if system_prompts:
            payload["system"] = "\n\n".join(system_prompts)
        specs = self._function_specs(tools)
        if specs:
            payload["tools"] = [
                {"name": sp["name"], "description": sp["description"], "input_schema": sp["parameters"]}
                for sp in specs
            ]

        data = self._post_json(url, headers, payload)

        content_parts = data.get("content", [])
        text_content = "".join(
            cp.get("text", "") for cp in content_parts if cp.get("type") == "text"
        )
        calls = [
            LLMToolCall(id=cp["id"], name=cp["name"], arguments=cp.get("input") or {})
            for cp in content_parts
            if cp.get("type") == "tool_use"
        ]
        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )
        return LLMResponse(content=text_content, tool_calls=calls, usage=usage, raw_response=data)


# JSON Schema keywords Gemini's OpenAPI-subset `Schema` accepts in function declarations.
_GEMINI_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum", "properties", "required", "items",
    "minItems", "maxItems", "minimum", "maximum", "anyOf", "propertyOrdering",
}


def _gemini_schema(schema: Any) -> Any:
    """Drop JSON Schema keywords (title, default, additionalProperties, $schema, ...) that the
    Gemini function-declaration schema rejects, recursively."""
    if isinstance(schema, list):
        return [_gemini_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k not in _GEMINI_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: _gemini_schema(sub) for name, sub in v.items()}
        elif k in ("items", "anyOf"):
            out[k] = _gemini_schema(v)
        else:
            out[k] = v
    return out


def _parse_tool_output(content: str) -> Any:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
