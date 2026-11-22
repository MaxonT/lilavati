"""LangGraph agent using Līlāvatī's MCP servers (fixed income + derivatives) as tools.

Runs in its own environment because langchain-mcp-adapters requires mcp<2 while Līlāvatī's
servers (fastmcp 4) need mcp>=2.3. The servers run as subprocesses from the project
environment, so the two never share packages:

    GEMINI_API_KEY=... uv run --no-project \
        --with langgraph --with langchain-mcp-adapters --with langchain-google-genai \
        python examples/integrations/langgraph_agent.py --check

Model: LILAVATI_MODEL (default gemma-4-31b-it via the Gemini API).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parent))
from _task import INSTRUCTION, QUESTION, SERVERS, check_answer  # noqa: E402


async def main(check: bool) -> int:
    client = MultiServerMCPClient(
        {
            name: {"command": cmd[0], "args": cmd[1:], "transport": "stdio"}
            for name, cmd in SERVERS.items()
        }
    )
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} Līlāvatī tools: {', '.join(sorted(t.name for t in tools))}")

    model = ChatGoogleGenerativeAI(
        model=os.environ.get("LILAVATI_MODEL", "gemma-4-31b-it"),
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        max_retries=4,
    )
    # LangGraph supplies orchestration features Līlāvatī does not: checkpointing, threads,
    # interrupts, streaming. Here, an in-memory checkpointer keyed by thread_id.
    agent = create_react_agent(model, tools, prompt=INSTRUCTION, checkpointer=InMemorySaver())
    result = await agent.ainvoke(
        {"messages": [("user", QUESTION)]}, config={"configurable": {"thread_id": "demo-1"}}
    )

    if os.environ.get("LILAVATI_DEBUG"):
        for m in result["messages"]:
            print("DEBUG", type(m).__name__, repr(m.content)[:300], getattr(m, "tool_calls", ""))
    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    answer = result["messages"][-1].content
    answer = answer if isinstance(answer, str) else " ".join(
        p.get("text", "") for p in answer if isinstance(p, dict)
    )
    print(f"\nTool calls: {len(tool_results)}\nAnswer:\n{answer}")
    if check:
        problems = check_answer(answer, tool_results)
        print("\nCHECK:", "OK" if not problems else "; ".join(problems))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify answer quotes tool outputs")
    sys.exit(asyncio.run(main(ap.parse_args().check)))
