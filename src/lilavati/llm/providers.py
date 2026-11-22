"""
LLM Provider Configurations and Model Registry for Project Līlāvatī.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ModelProvider(StrEnum):
    """Supported Foundation Model & LLM Providers."""

    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"
    MOCK = "mock"


class ModelSpec(BaseModel):
    """Configuration specification for a foundation model."""

    model_config = ConfigDict(frozen=True)

    provider: ModelProvider
    model_name: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10, description="Retries on 429/5xx/timeouts")
    retry_backoff_seconds: float = Field(default=2.0, ge=0.0, description="Base for exponential backoff")
    api_key_env_var: str | None = None
    base_url: str | None = None


GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Pre-configured institutional model profiles
INSTITUTIONAL_MODEL_PRESETS: dict[str, ModelSpec] = {
    # Gemini Models (Google DeepMind)
    "gemini-2.5-pro": ModelSpec(
        provider=ModelProvider.GEMINI,
        model_name="gemini-2.5-pro",
        api_key_env_var="GEMINI_API_KEY",
    ),
    "gemini-2.5-flash": ModelSpec(
        provider=ModelProvider.GEMINI,
        model_name="gemini-2.5-flash",
        api_key_env_var="GEMINI_API_KEY",
    ),
    # Gemini 4 family and Gemma 4 via the Gemini API (thinking models; tool calling supported).
    # Live tool-calling tests passed on gemma-4-31b-it and gemini-4-hydrogen (2026-11-19);
    # gemini-4-neon not yet live-tested (free-tier quota exhausted).
    "gemini-4-neon": ModelSpec(
        provider=ModelProvider.GEMINI,
        model_name="gemini-4-neon",
        api_key_env_var="GEMINI_API_KEY",
        max_tokens=8192,
        timeout_seconds=180.0,
    ),
    "gemini-4-hydrogen": ModelSpec(
        provider=ModelProvider.GEMINI,
        model_name="gemini-4-hydrogen",
        api_key_env_var="GEMINI_API_KEY",
        max_tokens=8192,
        timeout_seconds=180.0,
    ),
    "gemma-4-31b-it": ModelSpec(
        provider=ModelProvider.GEMINI,
        model_name="gemma-4-31b-it",
        api_key_env_var="GEMINI_API_KEY",
        max_tokens=8192,
        timeout_seconds=240.0,
    ),
    # Anthropic Claude
    "claude-3-7-sonnet": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-7-sonnet-20250219",
        api_key_env_var="ANTHROPIC_API_KEY",
    ),
    # OpenAI
    "gpt-4o": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        api_key_env_var="OPENAI_API_KEY",
    ),
    "o3-mini": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_name="o3-mini",
        api_key_env_var="OPENAI_API_KEY",
    ),
    # Open-weight models on Groq (OpenAI-compatible endpoint). Key read only from GROQ_API_KEY.
    # Model ids from GET /openai/v1/models; each made a correct tool call on 2026-11-20.
    "groq-gpt-oss-120b": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_name="openai/gpt-oss-120b",
        api_key_env_var="GROQ_API_KEY",
        base_url=GROQ_BASE_URL,
        max_tokens=8192,
    ),
    "groq-gpt-oss-20b": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_name="openai/gpt-oss-20b",
        api_key_env_var="GROQ_API_KEY",
        base_url=GROQ_BASE_URL,
        max_tokens=8192,
    ),
    "groq-qwen3.6-27b": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_name="qwen/qwen3.6-27b",
        api_key_env_var="GROQ_API_KEY",
        base_url=GROQ_BASE_URL,
        max_tokens=8192,
    ),
    "groq-qwen3.8-27b": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_name="qwen/qwen3.8-27b",
        api_key_env_var="GROQ_API_KEY",
        base_url=GROQ_BASE_URL,
        max_tokens=8192,
    ),
    # Local Open-Source Financial Foundation Models
    "llama3-fin-8b": ModelSpec(
        provider=ModelProvider.OLLAMA,
        model_name="llama3-fin:8b",
        base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible API
    ),
    "vllm-finance-70b": ModelSpec(
        provider=ModelProvider.VLLM,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        base_url="http://localhost:8000/v1",
    ),
    # Deterministic Mock Provider for offline testing
    "mock-evaluator": ModelSpec(
        provider=ModelProvider.MOCK,
        model_name="lilavati-mock-reference",
    ),
}
