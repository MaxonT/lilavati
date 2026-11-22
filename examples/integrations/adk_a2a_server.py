"""Expose the Līlāvatī-powered ADK agent as an A2A (Agent2Agent) server.

MCP lets an agent call Līlāvatī's *tools*; A2A lets other *agents* (any framework that speaks
A2A) delegate tasks to this analyst agent. Its agent card is served at
http://localhost:8001/.well-known/agent-card.json.

    uv sync --extra adk
    GEMINI_API_KEY=... uv run python examples/integrations/adk_a2a_server.py

Security: binds to localhost and has no authentication. Add an auth scheme
(`to_a2a(..., security_schemes=...)`) and a reverse proxy before exposing it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a

sys.path.insert(0, str(Path(__file__).parent))
from adk_agent import build_agent  # noqa: E402

HOST, PORT = "127.0.0.1", int(os.environ.get("LILAVATI_A2A_PORT", "8001"))

os.environ.setdefault("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
app = to_a2a(build_agent(), host="localhost", port=PORT)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
