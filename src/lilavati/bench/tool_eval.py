"""Tool-augmented evaluation: the same Līlāvatī-Bench cases, answered with deterministic tools.

Every case gets the same eight tools (the model must choose), each a thin wrapper over a validated
engine. The run records every tool call and output, so we can measure **tool grounding**: the
share of numeric answer fields whose value equals (within the case tolerance) a number some tool
returned. That is a measured version of the Tool Math Purity Ratio (TMPR).

Read results with care: the expected values are computed by the same engines the tools call, so
this arm measures tool selection and parameter extraction, not engine correctness. Engine
correctness is established separately, against independent references (docs/validation/).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import httpx

from lilavati.bench.datasets import BenchmarkTestCase
from lilavati.bench.llm_eval import KEY_DEFS, build_prompt, parse_json, stop_reason
from lilavati.core.derivatives import DerivativesEngine
from lilavati.core.execution import ExecutionEngine
from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.sustainability import SustainabilityEngine
from lilavati.llm.client import LLMMessage, LLMResponse, UniversalLLMClient
from lilavati.llm.providers import ModelSpec
from lilavati.runtime.agent import AgentConfig, NeuroSymbolicAgent
from lilavati.schemas.sustainability import SustainabilityLinkedBond

SYSTEM_PROMPT = (
    "You are a quantitative finance analyst with deterministic calculation tools. Call the tools "
    "for every number you report and copy their outputs exactly. Never compute in your head. "
    "When done, call submit_answer once with the requested keys."
)
SUBMIT = "submit_answer"


def submit_schema(case: BenchmarkTestCase) -> dict[str, Any]:
    """JSON schema of the submit_answer tool for `case`: one typed property per answer key."""
    props: dict[str, Any] = {}
    for key, expected in case.expected_values.items():
        kind = "boolean" if isinstance(expected, bool) else "number"
        props[key] = {"type": kind, "description": KEY_DEFS[case.track].get(key, key)}
    return {"type": "object", "properties": props, "required": list(props)}


def _num(desc: str) -> dict[str, str]:
    return {"type": "number", "description": desc}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props)}


def _price_bond(face_value: float, coupon_rate: float, ytm: float, years_to_maturity: float,
                frequency: int) -> dict[str, Any]:
    return FixedIncomeEngine.analyze_bond(face_value=face_value, coupon_rate=coupon_rate, ytm=ytm,
                                          years_to_maturity=years_to_maturity,
                                          frequency=frequency).model_dump()


def _option_args(spot: float, strike: float, time_to_maturity: float, risk_free_rate: float,
                 volatility: float) -> dict[str, float]:
    return {"spot": spot, "strike": strike, "time_to_maturity": time_to_maturity,
            "risk_free_rate": risk_free_rate, "volatility": volatility}


def _price_call(**kw: float) -> dict[str, Any]:
    return DerivativesEngine.price_european_option(**_option_args(**kw), is_call=True).model_dump()


def _call_greeks(**kw: float) -> dict[str, Any]:
    return DerivativesEngine.calculate_bsm_greeks(**_option_args(**kw), is_call=True).model_dump()


def _order_book(best_bid: float, best_ask: float, bid_size: float, ask_size: float) -> dict[str, Any]:
    return ExecutionEngine.analyze_lob_snapshot(best_bid=best_bid, best_ask=best_ask,
                                                bid_size=bid_size, ask_size=ask_size).model_dump()


def _waci(portfolio_weights: list[float], emissions: list[float],
          revenues_millions: list[float]) -> dict[str, Any]:
    return SustainabilityEngine.calculate_waci(portfolio_weights=portfolio_weights, emissions=emissions,
                                               revenues_millions=revenues_millions).model_dump()


def _slb(base_coupon_rate: float, target_reduction_pct: float, baseline_emissions: float,
         step_up_bps: float, reported_emissions: float) -> dict[str, Any]:
    bond = SustainabilityLinkedBond(isin="XS0000000000", notional=5e7, base_coupon_rate=base_coupon_rate,
                                    target_reduction_pct=target_reduction_pct,
                                    baseline_emissions=baseline_emissions, step_up_bps=step_up_bps,
                                    target_year=2026)
    return SustainabilityEngine.evaluate_slb_covenant(bond=bond, reported_emissions=reported_emissions).model_dump()


def _balance_sheet(total_assets: float, total_liabilities: float, shareholders_equity: float) -> dict[str, Any]:
    return SuvarnaEngine.verify_balance_sheet_equation(
        total_assets=total_assets, total_liabilities=total_liabilities,
        shareholders_equity=shareholders_equity).model_dump()


def _cash_flow(cash_start: float, cfo: float, cfi: float, cff: float, cash_end: float) -> dict[str, Any]:
    implied = cash_start + cfo + cfi + cff
    return {"implied_cash_end": implied, "reported_cash_end": cash_end,
            "difference": cash_end - implied, "reconciles": math.isclose(implied, cash_end, abs_tol=1e-4)}


_OPTION_SCHEMA = _obj({"spot": _num("spot price"), "strike": _num("strike price"),
                       "time_to_maturity": _num("years"), "risk_free_rate": _num("annual, decimal"),
                       "volatility": _num("annual, decimal")})

# name -> (description, JSON schema, function)
TOOLS: dict[str, tuple[str, dict[str, Any], Callable[..., Any]]] = {
    "price_bond": ("Fixed-coupon bond: clean price, Macaulay/modified duration, convexity, DV01.",
                   _obj({"face_value": _num("face value, e.g. 100"),
                         "coupon_rate": _num("annual coupon, decimal"),
                         "ytm": _num("yield to maturity, decimal"),
                         "years_to_maturity": _num("years"),
                         "frequency": {"type": "integer", "description": "coupons per year"}}),
                   _price_bond),
    "price_european_call": ("Black-Scholes price of a European call (no dividends).",
                            _OPTION_SCHEMA, _price_call),
    "call_greeks": ("Black-Scholes delta, gamma, vega (per 1 vol point), theta, rho of a European call.",
                    _OPTION_SCHEMA, _call_greeks),
    "analyze_order_book": ("Top-of-book spread, mid, size-weighted microprice, imbalance.",
                           _obj({"best_bid": _num("bid price"), "best_ask": _num("ask price"),
                                 "bid_size": _num("bid size"), "ask_size": _num("ask size")}),
                           _order_book),
    "portfolio_waci": ("TCFD weighted average carbon intensity (tCO2e per USD m revenue).",
                       _obj({"portfolio_weights": {"type": "array", "items": {"type": "number"}},
                             "emissions": {"type": "array", "items": {"type": "number"},
                                           "description": "tCO2e per holding"},
                             "revenues_millions": {"type": "array", "items": {"type": "number"},
                                                   "description": "revenue in USD millions"}}),
                       _waci),
    "slb_covenant": ("Sustainability-linked bond: is the emissions target met, and the coupon after any step-up.",
                     _obj({"base_coupon_rate": _num("decimal"),
                           "target_reduction_pct": _num("required reduction, decimal (0.25 = 25%)"),
                           "baseline_emissions": _num("tCO2e"), "step_up_bps": _num("basis points"),
                           "reported_emissions": _num("tCO2e")}),
                     _slb),
    "verify_balance_sheet": ("Z3 check that assets equal liabilities plus equity.",
                             _obj({"total_assets": _num(""), "total_liabilities": _num(""),
                                   "shareholders_equity": _num("")}),
                             _balance_sheet),
    "reconcile_cash_flow": ("Check that opening cash + CFO + CFI + CFF equals closing cash.",
                            _obj({"cash_start": _num(""), "cfo": _num(""), "cfi": _num(""),
                                  "cff": _num(""), "cash_end": _num("")}),
                            _cash_flow),
}


def numbers_in(value: Any) -> list[float]:
    """All finite numbers (not booleans) anywhere inside a tool output."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(value) else []
    if isinstance(value, dict):
        return [n for v in value.values() for n in numbers_in(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in numbers_in(v)]
    return []


def tool_grounding(case: BenchmarkTestCase, parsed: dict[str, Any] | None,
                   tool_outputs: list[Any]) -> tuple[int, int]:
    """(grounded, total) over numeric answer fields: grounded if equal to some tool-returned number."""
    pool = [n for out in tool_outputs for n in numbers_in(out)]
    grounded = total = 0
    for key, expected in case.expected_values.items():
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            continue
        value = (parsed or {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total += 1
        if any(math.isclose(float(value), n, rel_tol=case.tolerance, abs_tol=1e-12) for n in pool):
            grounded += 1
    return grounded, total


class _RecordingClient(UniversalLLMClient):
    """Remembers the last stop reason so truncation can be reported."""

    last_finish: str | None = None

    def generate(self, messages: list[LLMMessage], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        resp = super().generate(messages, tools)
        self.last_finish = stop_reason(resp.raw_response)
        return resp


def run_case(case: BenchmarkTestCase, spec: ModelSpec, *, max_tool_rounds: int = 8,
             transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Answer one case with tools. Returns the fields of a result record (no scoring)."""
    log: list[dict[str, Any]] = []

    def wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            entry: dict[str, Any] = {"name": name, "arguments": kwargs}
            log.append(entry)
            try:
                entry["output"] = fn(**kwargs)
            except Exception as err:  # reported to the model by the agent loop
                entry["error"] = f"{type(err).__name__}: {err}"
                raise
            return entry["output"]
        return call

    client = _RecordingClient(spec, transport=transport)
    agent = NeuroSymbolicAgent(
        AgentConfig(agent_id="bench-tools", role_name="Quant analyst", model_spec=spec,
                    system_prompt=SYSTEM_PROMPT, max_tool_rounds=max_tool_rounds),
        llm_client=client,
    )
    for name, (desc, schema, fn) in TOOLS.items():
        agent.register_tool(name, desc, schema, wrap(name, fn))
    submitted: list[dict[str, Any]] = []

    def submit(**answer: Any) -> dict[str, str]:
        submitted.append(answer)
        return {"status": "answer recorded; reply with the word DONE"}

    agent.register_tool(SUBMIT, "Submit the final answer (call once, after all calculations).",
                        submit_schema(case), submit)

    raw: str
    rejected_output: str | None = None
    try:
        out = agent.run(build_prompt(case, tools=True))
        raw = out.response_text if out.completed else "__MODEL_ERROR__ tool_round_limit"
    except httpx.HTTPStatusError as e:
        code = _error_code(e.response)
        if code == "tool_use_failed":
            raw = f"__MODEL_ERROR__ {code}"
            rejected_output = _failed_generation(e.response)  # the model's own text, for error analysis
        elif e.response.status_code == 400 and _tools_unsupported(e.response):
            raise ToolsUnsupportedError(f"{spec.model_name} does not support tool calling on this provider") from None
        else:
            raw = f"__ERROR__ HTTP {e.response.status_code}"  # never include URL/headers
    except httpx.TimeoutException:
        raw = "__ERROR__ Timeout"
    except httpx.HTTPError as e:
        raw = f"__ERROR__ {type(e).__name__}"

    late_error: str | None = None
    if submitted and not raw.startswith("__ERROR__"):
        # The submitted answer counts even if a later turn failed (kept as late_error); a
        # provider error is still a provider error.
        parsed: dict[str, Any] | None = submitted[-1]
        if raw.startswith("__MODEL_ERROR__"):
            late_error, raw = raw, json.dumps(parsed)
    else:
        parsed = None if raw.startswith("__") else parse_json(raw)
    outputs = [e["output"] for e in log if "output" in e]
    grounded, numeric = tool_grounding(case, parsed, outputs)
    return {"raw": raw, "finish_reason": client.last_finish, "parsed": parsed,
            "submitted": bool(submitted), "late_error": late_error,
            "rejected_output": rejected_output, "tool_calls": log, "n_tool_calls": len(log),
            "grounded_fields": grounded, "numeric_fields": numeric}


class ToolsUnsupportedError(RuntimeError):
    """The model cannot call tools on this provider (e.g. allam-2-7b on Groq). Not a result."""


def _tools_unsupported(response: httpx.Response) -> bool:
    try:
        message = str(response.json().get("error", {}).get("message", ""))
    except (ValueError, AttributeError):
        return False
    return "tool" in message.lower() and "not supported" in message.lower()


def _failed_generation(response: httpx.Response) -> str | None:
    try:
        err = response.json().get("error", {})
    except (ValueError, AttributeError):
        return None
    value = err.get("failed_generation") if isinstance(err, dict) else None
    return str(value)[:4000] if value is not None else None


def _error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    err = body.get("error") if isinstance(body, dict) else None
    return err.get("code") if isinstance(err, dict) else None
