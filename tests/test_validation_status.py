"""The validation-status manifest covers every MCP tool, drives the docs, and reaches tool results."""

from __future__ import annotations

import asyncio
import glob
import importlib
import os
import sys
from pathlib import Path

from fastmcp import Client, FastMCP

from lilavati.validation_status import ENGINES, ICON, MEANING, status_for_tool

ROOT = Path(__file__).resolve().parents[1]
SERVERS = sorted(os.path.basename(f)[:-3] for f in glob.glob(str(ROOT / "src/lilavati/mcp_servers/*_server.py")))


def _servers() -> list[FastMCP]:
    out = []
    for name in SERVERS:
        mod = importlib.import_module(f"lilavati.mcp_servers.{name}")
        out.append(next(v for v in vars(mod).values() if isinstance(v, FastMCP)))
    return out


def test_manifest_covers_exactly_the_registered_tools() -> None:
    registered = {t.name for s in _servers() for t in asyncio.run(s.list_tools())}
    in_manifest = [t for e in ENGINES for t in e.tools]
    assert len(in_manifest) == len(set(in_manifest))  # each tool belongs to one engine
    assert set(in_manifest) == registered


def test_every_validation_page_exists() -> None:
    for e in ENGINES:
        assert (ROOT / "docs" / e.page).is_file(), e.page
        assert e.status in MEANING and e.status in ICON


def test_generated_docs_are_current() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import render_validation_status

    assert render_validation_status.main(check=True) == 0, "run scripts/render_validation_status.py"


def test_readme_rows_disclose_unchecked_engines() -> None:
    # The README legend's ✅ means "implemented and tested"; a row for an engine that is not checked
    # against a reference (or only piloted) must say so in the row itself.
    rows = [line for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    needed = {"not reference-checked": "not yet checked against a reference", "piloted on real data": "pilot"}
    for e in ENGINES:
        if not e.readme_row:
            continue
        matches = [r for r in rows if e.readme_row in r]
        assert len(matches) == 1, f"{e.key}: README row {e.readme_row!r} found {len(matches)} times"
        if e.status in needed:
            assert needed[e.status] in matches[0], f"{e.key}: README row does not say {needed[e.status]!r}"


def test_tool_results_carry_the_status_over_mcp() -> None:
    # Called through the MCP protocol (in-memory client), not only as a Python function.
    from lilavati.mcp_servers.fixed_income_server import mcp as fixed_income
    from lilavati.mcp_servers.portfolio_exec_server import mcp as portfolio

    async def call(server: FastMCP, tool: str, args: dict) -> dict:
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            return dict(result.structured_content or {})

    bond = asyncio.run(call(fixed_income, "analyze_bond", {
        "face_value": 100.0, "coupon_rate": 0.05, "ytm": 0.06, "years_to_maturity": 5.0, "frequency": 2}))
    assert bond["validation"]["status"] == "reference-checked"
    assert bond["validation"]["page"].endswith("/validation/fixed_income/")

    hrp = asyncio.run(call(portfolio, "optimize_portfolio_hierarchical_risk_parity", {
        "assets": ["A", "B", "C"],
        "returns_matrix": [[0.01, 0.02, -0.01], [0.0, -0.01, 0.02], [0.02, 0.01, 0.0], [-0.01, 0.0, 0.01]]}))
    assert hrp["validation"]["status"] == "not reference-checked"
    assert "#16" in hrp["validation"]["open_items"]
    assert status_for_tool("get_filing_figures")["status"] == "piloted on real data"
