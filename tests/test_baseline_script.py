"""Answer parsing in scripts/run_llm_baseline.py (scoring harness, so it must not penalise format)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_baseline.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_llm_baseline", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


baseline = _load()


def test_plain_json_answer() -> None:
    assert baseline.parse_json('{"a": 1, "b": true}') == {"a": 1, "b": True}


def test_braces_inside_think_block_do_not_break_parsing() -> None:
    # qwen3.6-27b on Groq returns reasoning inline; braces in it (sets, dicts) made the
    # greedy first-{ to last-} match span reasoning + answer and fail to decode.
    raw = (
        "<think>\nThe set {x : x > 0} and a dict {'d': 0.4} appear here.\n</think>\n\n"
        '{"series_non_stationary": true, "rejects_spurious_relationship": true}'
    )
    assert baseline.parse_json(raw) == {
        "series_non_stationary": True,
        "rejects_spurious_relationship": True,
    }


def test_unclosed_think_block_is_unparsable() -> None:
    # Truncated mid-reasoning: there is no answer, even if the reasoning contains JSON.
    assert baseline.parse_json('<think>\ntry {"call_price": 1.0} then ') is None


def test_truncated_json_is_unparsable() -> None:
    assert baseline.parse_json('{"spread": 0.1, "microprice": 151.63') is None


def test_outcome_classes() -> None:
    from lilavati.bench.llm_eval import classify_outcome, should_rerun

    assert classify_outcome({"raw": "__ERROR__ HTTP 503"}) == "error"
    assert classify_outcome({"raw": "__MODEL_ERROR__ tool_use_failed"}) == "model_error"
    assert classify_outcome({"raw": "", "finish_reason": "length", "parsed": None}) == "truncated"
    assert classify_outcome({"raw": "", "finish_reason": None, "parsed": None}) == "error"
    assert classify_outcome({"raw": "{bad", "finish_reason": "stop", "parsed": None}) == "unparsable"
    assert classify_outcome({"raw": "{}", "finish_reason": "stop", "parsed": {}}) == "answered"
    # Only provider failures are re-run; model failures are never re-rolled.
    assert should_rerun({"raw": "__ERROR__ Timeout"})
    assert not should_rerun({"raw": "{bad", "finish_reason": "stop", "parsed": None})
    assert not should_rerun({"raw": "__MODEL_ERROR__ tool_use_failed"})


def test_bond_key_does_not_assume_a_maturity() -> None:
    # Prompt v2 said "price of the 5-year bond" for generated bonds of 2-10 years; gpt-oss-120b
    # priced a 10-year bond as 5-year (95.03 vs 5-year price 95.0183) and was scored wrong.
    from lilavati.bench.llm_eval import KEY_DEFS, PROMPT_VERSION

    assert "5-year" not in KEY_DEFS["track2_fixed_income"]["bond_price"]
    assert PROMPT_VERSION == "v3"
