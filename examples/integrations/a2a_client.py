"""Delegate a task to the Līlāvatī analyst agent over A2A (Agent2Agent).

Start the server first (examples/integrations/adk_a2a_server.py), then:

    uv run python examples/integrations/a2a_client.py --check

The client only speaks the A2A protocol: it discovers the agent card, sends a message and reads
the streamed task updates. Any A2A-capable agent framework can do the same.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict

sys.path.insert(0, str(Path(__file__).parent))
from _task import QUESTION, _numbers  # noqa: E402

URL = os.environ.get("LILAVATI_A2A_URL", "http://localhost:8001")


def _texts(obj: Any) -> list[str]:
    """All text parts in a streamed A2A response (agent messages and artifacts)."""
    out: list[str] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            out.append(obj["text"])
        for v in obj.values():
            out.extend(_texts(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_texts(v))
    return out


async def main(check: bool) -> int:
    # Agent turns include model calls and tool runs; allow minutes, not the 5 s httpx default.
    http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    client = await create_client(URL, client_config=ClientConfig(streaming=True, httpx_client=http))
    request = SendMessageRequest(
        message=Message(
            message_id=str(uuid.uuid4()), role=Role.ROLE_USER, parts=[Part(text=QUESTION)]
        )
    )
    final_text = ""
    async for update in client.send_message(request):
        d = MessageToDict(update)
        if os.environ.get("LILAVATI_DEBUG"):
            print("DEBUG", json.dumps(d)[:300])
        texts = [t for t in _texts(d.get("artifactUpdate") or d.get("task") or d.get("message"))
                 if t.strip()]
        if texts:
            final_text = texts[-1]
    await http.aclose()
    print(f"Answer from A2A agent:\n{final_text}")
    if check:
        # Values computed by Līlāvatī's engines for this task (see _task.QUESTION).
        expected = {"bond_clean_price": 95.0485, "option_price": 8.3633}
        nums = _numbers(final_text)
        missing = [k for k, v in expected.items() if not any(abs(n - v) < 5e-4 for n in nums)]
        print("CHECK:", "OK" if not missing else f"missing {missing}")
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    sys.exit(asyncio.run(main(ap.parse_args().check)))
