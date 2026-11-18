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
