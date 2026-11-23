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
