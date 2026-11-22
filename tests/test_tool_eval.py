"""Tool-augmented evaluation (lilavati.bench.tool_eval), with the LLM mocked over HTTP.

Replies follow the OpenAI Chat Completions format (tools[].function, message.tool_calls).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from lilavati.bench.generator import generate_cases
from lilavati.bench.llm_eval import classify_outcome
from lilavati.bench.tool_eval import TOOLS, numbers_in, run_case, tool_grounding
from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.llm.providers import ModelProvider, ModelSpec

SPEC = ModelSpec(provider=ModelProvider.OPENAI, model_name="m", base_url="https://example.test/v1",
                 max_retries=0)


def _bond_case() -> Any:
    return next(c for c in generate_cases(2026, 4) if c.track == "track2_fixed_income")


def _transport(replies: list[dict[str, Any]], status: int = 200) -> httpx.MockTransport:
    queue = list(replies)
    return httpx.MockTransport(lambda request: httpx.Response(status, json=queue.pop(0)))


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def _final(text: str) -> dict[str, Any]:
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


def test_answer_copied_from_tool_is_grounded() -> None:
    case = _bond_case()
    d = case.context_data
    args = {"face_value": d["face_value"], "coupon_rate": d["coupon_rate"], "ytm": d["ytm"],
            "years_to_maturity": d["maturity_years"], "frequency": 1}
    price = FixedIncomeEngine.analyze_bond(**args).clean_price
    res = run_case(case, SPEC, transport=_transport([
        _tool_call("price_bond", args), _final(json.dumps({"bond_price": price}))]))
    assert res["n_tool_calls"] == 1 and res["tool_calls"][0]["name"] == "price_bond"
    assert (res["grounded_fields"], res["numeric_fields"]) == (1, 1)
    assert res["parsed"] == {"bond_price": price}
    assert res["finish_reason"] == "stop"


def test_mental_math_answer_is_not_grounded() -> None:
    case = _bond_case()
    res = run_case(case, SPEC, transport=_transport([_final('{"bond_price": 77.0}')]))
    assert res["n_tool_calls"] == 0
    assert (res["grounded_fields"], res["numeric_fields"]) == (0, 1)


def test_rejected_tool_call_is_a_model_error_not_a_provider_error() -> None:
    # Groq validates tool arguments server side and returns 400 tool_use_failed.
    body = {"error": {"message": "tool call validation failed", "type": "invalid_request_error",
                      "code": "tool_use_failed", "failed_generation": '{"name": "json"}'}}
    res = run_case(_bond_case(), SPEC, transport=_transport([body], status=400))
    assert res["raw"] == "__MODEL_ERROR__ tool_use_failed"
    assert res["rejected_output"] == '{"name": "json"}'  # kept for error analysis
    assert classify_outcome({**res}) == "model_error"


def test_tool_round_limit_is_a_model_error() -> None:
    case = _bond_case()
    d = case.context_data
    args = {"face_value": d["face_value"], "coupon_rate": d["coupon_rate"], "ytm": d["ytm"],
            "years_to_maturity": d["maturity_years"], "frequency": 1}
    res = run_case(case, SPEC, max_tool_rounds=1,
                   transport=_transport([_tool_call("price_bond", args)] * 3))
    assert res["raw"] == "__MODEL_ERROR__ tool_round_limit"


def test_numbers_in_ignores_booleans_and_nan() -> None:
    assert numbers_in({"a": 1, "b": True, "c": [2.5, float("nan")], "d": "x"}) == [1.0, 2.5]


def test_grounding_skips_boolean_fields() -> None:
    case = next(c for c in generate_cases(2026, 4) if c.track == "track1_three_statement")
    assert tool_grounding(case, {"is_balanced": True}, []) == (0, 0)


def test_every_tool_runs_on_a_generated_case() -> None:
    # Guards the wrappers against engine signature drift.
    assert set(TOOLS) == {"price_bond", "price_european_call", "call_greeks", "analyze_order_book",
                          "portfolio_waci", "slb_covenant", "verify_balance_sheet", "reconcile_cash_flow"}
    fn = {k: v[2] for k, v in TOOLS.items()}
    assert fn["price_european_call"](spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.05,
                                     volatility=0.2)["price"] > 0
    assert "delta" in fn["call_greeks"](spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.05,
                                        volatility=0.2)
    assert fn["analyze_order_book"](best_bid=10, best_ask=10.1, bid_size=100, ask_size=300)["spread"] > 0
    assert fn["portfolio_waci"](portfolio_weights=[0.5, 0.5], emissions=[1e4, 2e4],
                                revenues_millions=[100, 200])["waci"] == 100.0
    assert fn["slb_covenant"](base_coupon_rate=0.04, target_reduction_pct=0.25, baseline_emissions=1e5,
                              step_up_bps=25, reported_emissions=78000)["spt_achieved"] is False
    assert fn["verify_balance_sheet"](total_assets=10, total_liabilities=4, shareholders_equity=6)["is_valid"]
    assert fn["reconcile_cash_flow"](cash_start=10, cfo=5, cfi=-2, cff=-1, cash_end=12)["reconciles"]


def test_submit_answer_is_the_answer_channel_and_not_grounding() -> None:
    # Prompt t1 asked for text JSON; gpt-oss-120b then "called" a non-existent tool "json".
    # t2 gives an explicit submit_answer tool; its arguments are the answer, and they are not
    # counted as a tool output for grounding.
    case = _bond_case()
    d = case.context_data
    args = {"face_value": d["face_value"], "coupon_rate": d["coupon_rate"], "ytm": d["ytm"],
            "years_to_maturity": d["maturity_years"], "frequency": 1}
    price = FixedIncomeEngine.analyze_bond(**args).clean_price
    res = run_case(case, SPEC, transport=_transport([
        _tool_call("price_bond", args), _tool_call("submit_answer", {"bond_price": price}), _final("DONE")]))
    assert res["submitted"] and res["parsed"] == {"bond_price": price}
    assert res["n_tool_calls"] == 1  # submit_answer is not a calculation
    assert (res["grounded_fields"], res["numeric_fields"]) == (1, 1)


def test_submitted_but_ungrounded_answer() -> None:
    case = _bond_case()
    res = run_case(case, SPEC, transport=_transport([
        _tool_call("submit_answer", {"bond_price": 77.0}), _final("DONE")]))
    assert res["submitted"] and (res["grounded_fields"], res["numeric_fields"]) == (0, 1)


def test_answer_survives_a_later_model_error() -> None:
    body = {"error": {"message": "bad", "type": "invalid_request_error", "code": "tool_use_failed"}}
    queue = [(200, _tool_call("submit_answer", {"bond_price": 77.0})), (400, body)]
    transport = httpx.MockTransport(lambda request: _pop(queue))
    res = run_case(_bond_case(), SPEC, transport=transport)
    assert res["parsed"] == {"bond_price": 77.0}
    assert res["late_error"] == "__MODEL_ERROR__ tool_use_failed"
    assert classify_outcome(res) == "answered"


def _pop(queue: list[tuple[int, dict[str, Any]]]) -> httpx.Response:
    status, body = queue.pop(0)
    return httpx.Response(status, json=body)


def test_tool_prompt_asks_for_submit_answer() -> None:
    from lilavati.bench.llm_eval import TOOL_PROMPT_VERSION, build_prompt

    case = _bond_case()
    assert "submit_answer" in build_prompt(case, tools=True)
    assert "submit_answer" not in build_prompt(case, tools=False)
    assert TOOL_PROMPT_VERSION == "t2"


def test_model_without_tool_support_raises_instead_of_recording_errors() -> None:
    # allam-2-7b on Groq: HTTP 400 "`tool calling` is not supported with this model". Recording 20
    # errors would put a capability gap into the results as if it were a measurement.
    import pytest

    from lilavati.bench.tool_eval import ToolsUnsupportedError

    body = {"error": {"message": "`tool calling` is not supported with this model",
                      "type": "invalid_request_error", "param": "tool calling"}}
    with pytest.raises(ToolsUnsupportedError):
        run_case(_bond_case(), SPEC, transport=_transport([body], status=400))
