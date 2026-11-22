# Getting started

Līlāvatī gives AI agents (and people) finance calculations they can check: deterministic engines,
reference-tested, served as MCP tools. Pick the path that fits you; each takes a few minutes.
Alpha software for research and teaching; not investment advice.

## Install

Python 3.11 or 3.12. From source today (the PyPI package `lilavati-finance` is being prepared;
the import name is `lilavati`):

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync --extra dev            # or: pip install -e ".[dev]"
uv run python -c "import lilavati; print(lilavati.__version__)"
```

Optional extras: `--extra docs` (build this site), `--extra adk` (Google ADK examples). Install
[uv](https://docs.astral.sh/uv/) first if you do not have it.

## Path 1: call the engines from Python

```python
from lilavati.core.derivatives import DerivativesEngine
from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.invariants import SuvarnaEngine

# A 10-year 5% semi-annual bond at a 6% yield (rates are decimals)
bond = FixedIncomeEngine.analyze_bond(face_value=100, coupon_rate=0.05, ytm=0.06,
                                      years_to_maturity=10, frequency=2)
print(f"bond price {bond.clean_price:.4f}, modified duration {bond.modified_duration:.4f}")

# A 6-month European call (Black-Scholes, no dividends)
call = DerivativesEngine.price_european_option(spot=100, strike=105, time_to_maturity=0.5,
                                               risk_free_rate=0.04, volatility=0.25, is_call=True)
print(f"call price {call.price:.4f}, delta {call.greeks.delta:.4f}")

# Maintenance covenants: net leverage <= 4.0x and interest coverage >= 3.0x
check = SuvarnaEngine.verify_debt_covenant(total_debt=420, cash=20, ebitda=100, max_leverage_covenant=4.0,
                                           min_interest_coverage_covenant=3.0, interest_expense=25)
print(f"covenants met: {check.is_valid} {check.violations}")
```

```text
bond price 92.5613, modified duration 7.6650
call price 5.7799, delta 0.4703
covenants met: True []
```

Net leverage here is exactly 4.0x, at the limit; the check compares with a small tolerance so a
ratio at the limit complies (a real bug once reported it as a breach). Every function states its
units in its docstring: see the [API reference](reference/index.md). Next:
[Tutorial 01](tutorials/01_build_your_first_financial_agent.md).

## Path 2: give the tools to an AI assistant (MCP)

Eight MCP servers expose the engines. In Claude Code:

```bash
claude mcp add lilavati-fixed-income -- uv run --directory /path/to/lilavati lilavati-mcp-fixed-income
```

Then ask, for example, *"Use the bond tool to price a 10-year 5% semi-annual bond at a 6% yield."*
The assistant extracts the parameters; the tool returns the number. Configuration for Claude
Desktop, Cursor and other clients, and the list of all servers: [MCP setup](mcp-setup.md).

To see what a server offers without an assistant, connect from Python:

```python
import asyncio

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main() -> None:
    server = StdioTransport(command="uv", args=["run", "lilavati-mcp-fixed-income"])
    async with Client(server) as client:
        print([tool.name for tool in await client.list_tools()])
        result = await client.call_tool("analyze_bond", {
            "face_value": 100, "coupon_rate": 0.05, "ytm": 0.05,
            "years_to_maturity": 5, "frequency": 1})
        print(round(result.data["clean_price"], 6))


asyncio.run(main())
```

```text
['analyze_bond', 'fit_yield_curve', 'evaluate_credit_risk_merton']
100.0
```

## Path 3: use it inside an agent framework

LangGraph, Google ADK and A2A examples call the same MCP servers and check that the final answer
quotes the tool outputs: [Integrations](integrations.md). Līlāvatī is a tool layer, not another
orchestration framework.

## Path 4: benchmark a language model

Reproduce the published analysis from the stored replies (no API key needed):

```bash
uv run python scripts/build_results_manifest.py --check
uv run python scripts/analyse_results.py      # writes results/ANALYSIS.md
```

Run a model yourself with a free key: [Tutorial 05](tutorials/05_benchmarking_llms_with_open_models.md);
with and without tools on the same questions: [Tutorial 07](tutorials/07_tools_versus_mental_math.md).

## Where to go next

| You are | Start with |
| :--- | :--- |
| Student or educator | [Tutorials](tutorials/01_build_your_first_financial_agent.md), [course](course/index.md) |
| Analyst | Tutorials [01](tutorials/01_build_your_first_financial_agent.md), [03](tutorials/03_forensic_accounting_and_adversarial_committee.md), [04](tutorials/04_climate_risk_and_sustainable_portfolio.md) |
| AI or agent engineer | [MCP setup](mcp-setup.md), [Integrations](integrations.md), [Tutorial 07](tutorials/07_tools_versus_mental_math.md) |
| Researcher | [Benchmarks](benchmarks.md), [Tutorial 06](tutorials/06_analysing_and_publishing_results.md), [papers](papers/02_benchmark_paper.md) |
| Model validation or audit | [Numerical validation](validation/index.md) |
| Contributor | [Contributing guide](contributing.md) |

Problems? See the [FAQ and troubleshooting](faq.md).
