"""Google ADK agent using Līlāvatī's MCP servers (fixed income + derivatives) as tools.

    uv sync --extra adk
    GEMINI_API_KEY=... uv run python examples/integrations/adk_agent.py --check

Model: LILAVATI_MODEL (default gemma-4-31b-it via the Gemini API). ADK reads GOOGLE_API_KEY;
this script maps GEMINI_API_KEY to it if needed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

sys.path.insert(0, str(Path(__file__).parent))
from _task import INSTRUCTION, QUESTION, SERVERS, check_answer  # noqa: E402


def build_agent() -> LlmAgent:
    toolsets = [
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(command=cmd[0], args=cmd[1:]),
                timeout=60,
            )
        )
        for cmd in SERVERS.values()
    ]
    return LlmAgent(
        name="lilavati_analyst",
        description="Prices bonds and options with Līlāvatī's deterministic engines.",
        model=os.environ.get("LILAVATI_MODEL", "gemma-4-31b-it"),
        instruction=INSTRUCTION,
        tools=list(toolsets),
        # Free-tier Gemini endpoints often return 503 "high demand"; back off patiently.
        generate_content_config=types.GenerateContentConfig(
            temperature=0,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=6, initial_delay=5.0, max_delay=60.0,
                    http_status_codes=[429, 500, 502, 503, 504],
                )
            ),
        ),
    )


async def main(check: bool) -> int:
    os.environ.setdefault("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    agent = build_agent()
    runner = InMemoryRunner(agent=agent, app_name="lilavati_demo")
    session = await runner.session_service.create_session(app_name="lilavati_demo", user_id="u1")

    tool_results: list[Any] = []
    answer = ""
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=QUESTION)]),
    ):
        for part in (event.content.parts if event.content and event.content.parts else []):
            if part.function_response is not None:
                tool_results.append(part.function_response.response)
            elif part.text and not part.thought and event.is_final_response():
                answer += part.text
    await runner.close()

    print(f"Tool calls: {len(tool_results)}\nAnswer:\n{answer}")
    if check:
        problems = check_answer(answer, tool_results)
        print("\nCHECK:", "OK" if not problems else "; ".join(problems))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify answer quotes tool outputs")
    sys.exit(asyncio.run(main(ap.parse_args().check)))
