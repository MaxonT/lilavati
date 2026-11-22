# Integrations: LangGraph, Google ADK and A2A

Līlāvatī does not try to replace agent frameworks. It provides validated finance tools over MCP
that any framework can call, and the examples below show the same task running in each.

* **LangGraph** loads Līlāvatī's MCP tools with `langchain-mcp-adapters` and runs them in a
  `create_react_agent` graph with a checkpointer.
* **Google ADK** loads them with `McpToolset` inside an `LlmAgent`.
* **A2A**: the ADK agent can be served over the Agent2Agent protocol (`to_a2a`), so agents built
  with other frameworks can delegate finance tasks to it.

All three were run end to end on 2026-11-19 with `gemma-4-31b-it` via the Gemini API. The
model's final answer quoted the exact tool outputs (bond clean price 95.0485, call price 8.3633).
LangGraph produced an empty final message in 1 of 2 runs, after correct tool calls.

See [`examples/integrations/README.md`](https://github.com/revendrat/lilavati/blob/main/examples/integrations/README.md)
for commands, the dependency note (LangGraph's MCP adapter needs `mcp<2`, so its client runs in a
separate environment) and security notes. MCP client configuration for Claude Code and Claude
Desktop is in [Using the MCP servers](mcp-setup.md).
