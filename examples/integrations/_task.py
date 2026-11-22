"""Shared task, servers and answer check for the framework integration examples.

Standard library only, so it can be imported from any environment (the LangGraph example runs
outside the project environment).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = str(Path(__file__).resolve().parents[2])

# Līlāvatī MCP servers, started over stdio from the project environment.
SERVERS = {
    "fixed_income": ["uv", "run", "--directory", REPO, "lilavati-mcp-fixed-income"],
    "derivatives": ["uv", "run", "--directory", REPO, "lilavati-mcp-derivatives"],
}

INSTRUCTION = (
    "You are a fixed-income and derivatives analyst. Use the provided tools for every "
    "calculation and quote the numbers they return (rounded to 4 decimals). Never compute "
    "financial numbers yourself."
)

QUESTION = (
    "1) Price a 7-year bond: face 100, 4.25% annual coupon paid semiannually, yield 5.1%. "
    "Report its clean price. 2) Price a 1-year European call: spot 100, strike 105, "
    "risk-free rate 4%, volatility 22%. Report the option price."
)


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+", text.replace(",", ""))]


def _walk(obj: Any) -> list[dict[str, Any]]:
    """Every dict inside obj, also parsing JSON strings (MCP results arrive wrapped differently
    per framework: plain dicts, {"result": ...}, structuredContent, or text content blocks)."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, str):
        try:
            return _walk(json.loads(obj))
        except json.JSONDecodeError:
            return out
    if isinstance(obj, dict):
        out.append(obj)
        for v in obj.values():
            out.extend(_walk(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk(v))
    return out


def tool_values(results: list[Any]) -> dict[str, float]:
    """Bond clean price and option price found in raw tool results."""
    found: dict[str, float] = {}
    for d in _walk(results):
        if "clean_price" in d:
            found["bond_clean_price"] = float(d["clean_price"])
        elif "price" in d and "strike" in d:
            found["option_price"] = float(d["price"])
    return found


def check_answer(answer: str, results: list[Any]) -> list[str]:
    """Return problems; empty list means the answer quotes both tool-computed values."""
    values = tool_values(results)
    problems = [f"tool for {k} was not called" for k in ("bond_clean_price", "option_price")
                if k not in values]
    nums = _numbers(answer)
    for key, v in values.items():
        if not any(abs(n - v) < 5e-4 for n in nums):
            problems.append(f"answer does not quote tool {key}={v:.4f}")
    return problems
