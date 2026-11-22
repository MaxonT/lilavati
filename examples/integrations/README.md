# Framework integrations

Līlāvatī is a **tool layer**: validated finance engines served as MCP tools. Orchestration,
persistence, streaming and human-in-the-loop come from your agent framework. These examples run
the same task (price a bond and a European call) in each framework, using Līlāvatī's
`fixed-income` and `derivatives` MCP servers over stdio. `--check` confirms that the final answer
quotes the numbers the tools returned (not numbers the model computed).

| Example | Framework | Environment | Verified (2026-11-19, `gemma-4-31b-it` via Gemini API) |
| :--- | :--- | :--- | :--- |
| `langgraph_agent.py` | LangGraph `create_react_agent` + `langchain-mcp-adapters` | separate env (see below) | `--check` OK (1 of 2 runs; the other returned an empty final message after correct tool calls) |
| `adk_agent.py` | Google ADK `LlmAgent` + `McpToolset` | `uv sync --extra adk` | `--check` OK |
| `adk_a2a_server.py` + `a2a_client.py` | ADK agent served over **A2A** (protocol 1.0, JSON-RPC) | `uv sync --extra adk` | `--check` OK |

Free-tier Gemini endpoints frequently return 503 "high demand"; the ADK example backs off and
retries. Model override: `LILAVATI_MODEL` (e.g. `gemini-4-hydrogen`, `gemini-4-neon`).

## LangGraph

`langchain-mcp-adapters` currently requires `mcp<2`, while Līlāvatī's servers (fastmcp 4) need
`mcp>=2.3`. They cannot share one environment, and they do not need to: the servers run as
subprocesses from the project environment. Run the client in its own environment:

```bash
GEMINI_API_KEY=... uv run --no-project \
  --with langgraph --with langchain-mcp-adapters --with langchain-google-genai \
  python examples/integrations/langgraph_agent.py --check
```

## Google ADK

```bash
uv sync --extra adk
GEMINI_API_KEY=... uv run python examples/integrations/adk_agent.py --check
```

## A2A (Agent2Agent)

MCP connects an agent to Līlāvatī's **tools**. A2A lets **other agents** delegate tasks to a
Līlāvatī-powered agent, whatever framework they use.

```bash
GEMINI_API_KEY=... uv run python examples/integrations/adk_a2a_server.py   # terminal 1
uv run python examples/integrations/a2a_client.py --check                   # terminal 2
curl http://localhost:8001/.well-known/agent-card.json                      # agent card
```

The A2A server binds to localhost with no authentication; add `security_schemes` and a proxy
before exposing it.
