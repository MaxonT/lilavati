import asyncio
import glob
import importlib
import os

import pytest
from fastmcp import FastMCP

SERVERS = sorted(
    os.path.basename(f)[:-3] for f in glob.glob("src/lilavati/mcp_servers/*_server.py")
)


@pytest.mark.parametrize("module", SERVERS)
def test_server_registers_tools_and_has_main(module: str) -> None:
    mod = importlib.import_module(f"lilavati.mcp_servers.{module}")
    assert callable(mod.main)
    servers = [v for v in vars(mod).values() if isinstance(v, FastMCP)]
    assert len(servers) == 1
    tools = asyncio.run(servers[0].list_tools())
    assert len(tools) >= 1


def test_eight_servers() -> None:
    assert len(SERVERS) == 8


def test_mcp_setup_tool_table_matches_the_servers() -> None:
    import re
    from pathlib import Path


    scripts = {}
    pyproject = Path("pyproject.toml").read_text()
    for cmd, target in re.findall(r'^"?(lilavati-mcp-[\w-]+)"?\s*=\s*"lilavati\.mcp_servers\.(\w+):main"', pyproject, re.M):
        scripts[cmd] = target
    assert len(scripts) == 8

    table = {}
    for cmd, tools in re.findall(r"^\| `(lilavati-mcp-[\w-]+)` \| (.+) \|$", Path("docs/mcp-setup.md").read_text(), re.M):
        table[cmd] = sorted(re.findall(r"`(\w+)`", tools))
    assert set(table) == set(scripts)
    for cmd, module in scripts.items():
        mod = importlib.import_module(f"lilavati.mcp_servers.{module}")
        server = next(v for v in vars(mod).values() if isinstance(v, FastMCP))
        registered = sorted(t.name for t in asyncio.run(server.list_tools()))
        assert table[cmd] == registered, cmd


def test_every_tool_declares_all_four_hints_truthfully() -> None:
    # Hosts use these to warn users, and some directories reject tools missing any of them (an
    # external scan of an earlier commit reported every tool without hints). All tools only compute
    # or read; only the SEC tools reach the network.
    network = {"get_company_filing_statements", "extract_single_line_item", "get_filing_figures"}
    seen = 0
    for module in SERVERS:
        mod = importlib.import_module(f"lilavati.mcp_servers.{module}")
        server = next(v for v in vars(mod).values() if isinstance(v, FastMCP))
        for tool in asyncio.run(server.list_tools()):
            a = tool.annotations
            assert a is not None, tool.name
            hints = (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint)
            assert all(isinstance(h, bool) for h in hints), (tool.name, hints)
            assert a.read_only_hint and not a.destructive_hint and a.idempotent_hint, tool.name
            assert a.open_world_hint == (tool.name in network), tool.name
            seen += 1
    assert seen == 25
