"""Shared pieces for evaluating real LLMs on Līlāvatī-Bench cases.

Used by ``scripts/run_llm_baseline.py`` (no tools), ``scripts/run_tool_baseline.py`` (tools) and
``scripts/analyse_results.py``, and importable from tutorials:

- `build_prompt`: the exact text a model sees, with a plain-language definition of every answer key.
- `parse_json`: extracts the answer object, tolerating inline ``<think>`` reasoning.
- `classify_outcome`: error / truncated / unparsable / model_error / answered, so that provider
  failures, token-budget truncation and wrong answers are never mixed up.
- `run_metadata`: what to store next to every result file so a run can be reproduced.

API keys are read only from environment variables by the LLM client; nothing here touches them.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

from lilavati.bench.datasets import BenchmarkTestCase, case_content_hash
from lilavati.bench.generator import legacy_case_id
from lilavati.llm.providers import GROQ_BASE_URL, ModelProvider

# Bump when the text a model sees changes. Record it with every result.
#   v1: first pass (wrong vega unit; superseded, results/archive_prompt_v1/)
#   v2: vega unit fixed; bond_price key said "5-year bond" although generated maturities vary
#   v3: bond_price key refers to the bond in the input data
PROMPT_VERSION = "v3"

# --provider -> (client provider, base URL, environment variable holding the key)
PROVIDERS: dict[str, tuple[ModelProvider, str | None, str]] = {
    "gemini": (ModelProvider.GEMINI, None, "GEMINI_API_KEY"),
    "groq": (ModelProvider.OPENAI, GROQ_BASE_URL, "GROQ_API_KEY"),
}

# Plain-language definition of every answer key, so scoring is unambiguous.
KEY_DEFS: dict[str, dict[str, str]] = {
    "track1_three_statement": {
        "is_balanced": "bool: Assets == Liabilities + Equity",
        "cash_reconciled": "bool: cash_start + CFO + CFI + CFF == cash_end",
        "assets_equal_liab_equity": "bool: same test as is_balanced",
    },
    "track2_fixed_income": {
        "bond_price": "number: clean price of the bond in the input data (annual coupons, maturity_years as given)",
        "macaulay_duration_in_range": "bool: 4.0 <= Macaulay duration (years) <= 5.0",
        "merton_distance_to_default_positive": "bool: Merton distance-to-default > 0",
        "default_probability_in_range": "bool: 0 <= Merton default probability <= 0.10",
    },
    "track3_econometrics": {
        "series_non_stationary": "bool: an ADF test on a 200-step Gaussian random walk fails to reject a unit root",
        "rejects_spurious_relationship": "bool: you reject treating levels regressions on it as valid (same direction as above)",
    },
    "track4_financial_ml": {
        "is_stationary_after_fracdiff": "bool: ADF p-value < 0.05 at the optimal d* for a 300-step random walk with drift 0.01, sd 1, start 100",
        "memory_preserved": "bool: correlation between the fractionally differenced and original series >= 0.25",
    },
    "track5_derivatives": {
        "call_price": "number: Black-Scholes call price",
        "delta": "number: call delta",
        "gamma": "number: call gamma",
        "vega": "number: call vega per 1 percentage-point (0.01) change in volatility, i.e. dPrice/dSigma divided by 100",
        "butterfly_arbitrage_free": "bool: C(K1) - 2C(K2) + C(K3) >= 0",
    },
    "track6_microstructure": {
        "spread": "number: ask - bid",
        "microprice": "number: (bid*ask_size + ask*bid_size)/(bid_size+ask_size)",
        "initial_liquidation_shares": "number: shares held at the start of the schedule",
        "final_shares_remaining": "number: shares held at the end of the schedule",
    },
    "track7_sustainability": {
        "portfolio_waci": "number: WACI = sum_i w_i * emissions_i / revenue_i(USD m), tCO2e per USD m",
        "slb_spt_achieved": "bool: reduction target met",
        "slb_active_coupon": "number: coupon after any step-up (decimal, e.g. 0.0425)",
        "slb_penalty_triggered": "bool: coupon step-up applies",
    },
}

# Version of the tool-arm instructions (the no-tool text is PROMPT_VERSION).
#   t1: pilot; final answer as a text JSON object. gpt-oss-120b then "called" a non-existent tool
#       named "json" and Groq rejected it (results/archive_tools_pilot_t1/).
#   t2: final answer through an explicit submit_answer tool (text JSON still accepted).
TOOL_PROMPT_VERSION = "t2"

NO_TOOLS_INSTRUCTION = "Answer WITHOUT using tools; compute carefully."
TOOLS_INSTRUCTION = (
    "Use the provided tools for every calculation and copy the numbers they return; "
    "do not compute anything yourself."
)

# Stop reasons meaning the token cap cut the reply off (OpenAI-compatible, Gemini, Anthropic).
TRUNCATED = {"length", "MAX_TOKENS", "max_tokens"}

OUTCOMES = ("answered", "unparsable", "truncated", "model_error", "error")


def build_prompt(case: BenchmarkTestCase, *, tools: bool = False) -> str:
    """The exact text sent to the model for `case` (no-tool or tool-augmented variant)."""
    keys = "\n".join(
        f'  "{k}": {d}' for k, d in KEY_DEFS[case.track].items() if k in case.expected_values
    )
    instruction = TOOLS_INSTRUCTION if tools else NO_TOOLS_INSTRUCTION
    answer = ("When finished, call the submit_answer tool with exactly these keys:\n" if tools else
              "Respond with ONLY a JSON object (no prose, no markdown) with exactly these keys:\n")
    return (
        f"You are a quantitative finance expert. {instruction}\n\n"
        f"TASK: {case.prompt}\n\nINPUT DATA (JSON): {json.dumps(case.context_data)}\n\n"
        f"{answer}{keys}\n"
    )


def parse_json(text: str) -> dict[str, Any] | None:
    """Extract the answer object from a reply, or None if there is no valid one."""
    # Some models (e.g. qwen3.6-27b on Groq) return reasoning inline as <think>...</think>.
    # Drop it first: braces inside the reasoning would otherwise break the match below.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:  # reasoning never closed (truncated): there is no answer
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def stop_reason(raw: dict[str, Any]) -> str | None:
    """Why the model stopped (e.g. "stop", "length"), so truncation is not scored as a wrong answer."""
    if raw.get("choices"):  # OpenAI-compatible (Groq, vLLM, Ollama)
        reason = raw["choices"][0].get("finish_reason")
    elif raw.get("candidates"):  # Gemini
        reason = raw["candidates"][0].get("finishReason")
    else:  # Anthropic
        reason = raw.get("stop_reason")
    return str(reason) if reason is not None else None


def classify_outcome(record: dict[str, Any]) -> str:
    """One of OUTCOMES for a stored result record.

    - error: provider failure (HTTP 429/5xx, timeout), or an empty reply not explained by the
      token cap. Not the model's fault; may be re-run.
    - model_error: the provider rejected the model's own output (e.g. Groq ``tool_use_failed``
      for a malformed tool call). The model's fault; never re-run.
    - truncated: the token cap cut the reply off.
    - unparsable: the model stopped but gave no valid JSON object.
    - answered: a JSON object was parsed (it may still be wrong; see the score).
    """
    raw = record["raw"]
    if raw.startswith("__MODEL_ERROR__"):
        return "model_error"
    if raw.startswith("__ERROR__"):
        return "error"
    if record.get("finish_reason") in TRUNCATED:
        return "truncated" if record.get("parsed") is None else "answered"
    if not raw.strip():
        return "error"
    return "answered" if record.get("parsed") is not None else "unparsable"


def should_rerun(record: dict[str, Any] | None) -> bool:
    """True only for provider failures. Re-rolling model failures would inflate scores."""
    return record is None or classify_outcome(record) == "error"


def case_index(cases: list[BenchmarkTestCase]) -> dict[str, BenchmarkTestCase]:
    """Look up cases by current id and by the pre-2026-11-21 short generated id."""
    index: dict[str, BenchmarkTestCase] = {}
    for c in cases:
        index[c.case_id] = c
        index.setdefault(legacy_case_id(c.case_id), c)
    return index


def make_record(
    *, provider: str, model: str, sample: int, case: BenchmarkTestCase, raw: str,
    finish_reason: str | None, parsed: dict[str, Any] | None, score: float,
    **extra: Any,
) -> dict[str, Any]:
    """A result row. `case_hash` ties the answer to the exact question it answered."""
    return {
        "provider": provider, "model": model, "sample": sample, "case": case.case_id,
        "case_hash": case_content_hash(case), "prompt_version": PROMPT_VERSION,
        "raw": raw, "finish_reason": finish_reason, "parsed": parsed, "score": score, **extra,
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def run_metadata(**settings: Any) -> dict[str, Any]:
    """Metadata to save next to a result file (``<out>.meta.json``). Never include secrets."""
    status = _git("status", "--porcelain")
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "prompt_version": PROMPT_VERSION,
        **settings,
    }
