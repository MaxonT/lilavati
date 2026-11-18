# Using the MCP servers

Līlāvatī exposes its deterministic engines as eight [FastMCP](https://github.com/jlowin/fastmcp) servers
that speak MCP over stdio. Any MCP client (Claude Code, Claude Desktop, Cursor, ...) can use them, so an
LLM calls the tool instead of doing arithmetic itself.

## Install

```bash
git clone https://github.com/revendrat/lilavati.git && cd lilavati
uv sync
uv run lilavati-mcp-derivatives   # starts a stdio server; Ctrl-C to stop
```

Replace `/path/to/lilavati` below with the absolute path of your clone.

## Claude Code

```bash
claude mcp add lilavati-committee -- uv run --directory /path/to/lilavati lilavati-mcp-committee
claude mcp add lilavati-corporate-filings -- uv run --directory /path/to/lilavati lilavati-mcp-corporate-filings
claude mcp add lilavati-derivatives -- uv run --directory /path/to/lilavati lilavati-mcp-derivatives
claude mcp add lilavati-econometrics -- uv run --directory /path/to/lilavati lilavati-mcp-econometrics
claude mcp add lilavati-financial-ml -- uv run --directory /path/to/lilavati lilavati-mcp-financial-ml
claude mcp add lilavati-fixed-income -- uv run --directory /path/to/lilavati lilavati-mcp-fixed-income
claude mcp add lilavati-portfolio-exec -- uv run --directory /path/to/lilavati lilavati-mcp-portfolio-exec
claude mcp add lilavati-sustainability -- uv run --directory /path/to/lilavati lilavati-mcp-sustainability
```

Then run `/mcp` inside Claude Code to see the tools.

## Claude Desktop / other JSON-configured clients

Add to the client's MCP configuration (`mcpServers`), keeping only the servers you want:

```json
{
  "mcpServers": {
    "lilavati-committee": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-committee"] },
    "lilavati-corporate-filings": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-corporate-filings"] },
    "lilavati-derivatives": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-derivatives"] },
    "lilavati-econometrics": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-econometrics"] },
    "lilavati-financial-ml": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-financial-ml"] },
    "lilavati-fixed-income": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-fixed-income"] },
    "lilavati-portfolio-exec": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-portfolio-exec"] },
    "lilavati-sustainability": { "command": "uv", "args": ["run", "--directory", "/path/to/lilavati", "lilavati-mcp-sustainability"] }
  }
}
```

## Tools

| Server command | Tools |
| :--- | :--- |
| `lilavati-mcp-committee` | `compute_forensic_beneish_m_score`, `evaluate_investment_committee_memo` |
| `lilavati-mcp-corporate-filings` | `extract_single_line_item`, `get_company_filing_statements` |
| `lilavati-mcp-derivatives` | `calculate_option_greeks`, `calibrate_sabr_volatility_model`, `check_butterfly_arbitrage`, `price_american_option`, `price_european_option` |
| `lilavati-mcp-econometrics` | `analyze_cointegration`, `analyze_stationarity`, `fit_garch_volatility` |
| `lilavati-mcp-financial-ml` | `apply_triple_barrier_labels`, `find_optimal_fractional_diff` |
| `lilavati-mcp-fixed-income` | `analyze_bond`, `evaluate_credit_risk_merton`, `fit_yield_curve` |
| `lilavati-mcp-portfolio-exec` | `analyze_order_book_microstructure`, `compute_almgren_chriss_liquidation_schedule`, `optimize_portfolio_black_litterman`, `optimize_portfolio_hierarchical_risk_parity` |
| `lilavati-mcp-sustainability` | `calculate_portfolio_carbon_intensity`, `evaluate_sustainability_linked_bond_covenant`, `stress_test_carbon_tax_impact` |

(This table is generated from the code; `tests/test_mcp_registry.py` checks that every server loads.)

## Notes
* Results are deterministic calculations, not advice. Engines are alpha and still being validated against
  reference values; see the [roadmap](https://github.com/revendrat/lilavati/blob/main/ROADMAP.md).
* The servers make no network calls except `corporate-filings`, which can query SEC EDGAR.
