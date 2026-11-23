"""
FastMCP Server for SEC EDGAR Corporate Filings & Structured XBRL Disclosures.

Exposes tools to fetch, parse, and check (Z3, on the reported numbers) company three-statement filings
with cryptographic CitationRefs.

`get_filing_figures` is the filing-anchored path validated in Tutorial 08 and
docs/validation/connectors.md (identify the filing, then use only its own XBRL facts). The two
fiscal-year tools select by the `fy` tag with an annual-duration and latest-period rule.
SEC requires a contact: set LILAVATI_SEC_USER_AGENT="Your Name you@example.org" in the server's
environment.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.connectors.edgar import EdgarClient
from lilavati.connectors.sec_filing import fetch_filing_extraction, filing_ratios

mcp = FastMCP(
    name="lilavati-corporate-filings",
)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
def get_company_filing_statements(
    ticker_or_cik: str,
    fiscal_year: int,
) -> dict[str, Any]:
    """
    Fetches structured XBRL 10-K facts from SEC EDGAR for a company and fiscal year and maps them
    into a three-statement model. The cited facts are in "sourced_facts". The statement model fills
    items the company does not tag with placeholders (inventory 0, missing interest or tax 0,
    retained earnings set to net income, operating income set to net income if untagged), so do
    not read those fields as reported. Prefer get_filing_figures, which lists missing items instead.
    """
    client = EdgarClient()
    sourced_model = client.extract_three_statement_model(
        ticker_or_cik=ticker_or_cik,
        fiscal_year=fiscal_year,
    )
    return sourced_model.model_dump()


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
def extract_single_line_item(
    ticker_or_cik: str,
    metric_name: str,
    fiscal_year: int,
) -> dict[str, Any]:
    """
    Extracts a specific reported metric (e.g. 'Assets', 'Revenues', 'NetIncomeLoss')
    from SEC EDGAR with verbatim citation and accession number.
    """
    client = EdgarClient()
    facts = client.fetch_company_facts(ticker_or_cik)
    item = client.extract_line_item(
        facts=facts,
        candidate_tags=[metric_name],
        fiscal_year=fiscal_year,
    )
    if not item:
        return {"error": f"Metric '{metric_name}' not found for {ticker_or_cik} FY{fiscal_year}."}
    return item.model_dump()


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
def get_filing_figures(
    ticker_or_cik: str,
    form: str = "10-K",
    filings_back: int = 0,
) -> dict[str, Any]:
    """
    Line items of one SEC filing, from that filing's own XBRL facts, with ratios computed by code.

    form: "10-K" or "10-Q". filings_back: 0 for the latest filing, 1 for the one before, and so on.
    Values are US dollars as reported (not scaled); each carries a citation with the accession
    number and XBRL tag. Items the company does not report are listed under "missing", never zero.
    Ratios are fractions (gross_margin, net_margin, return_on_assets on period-end assets,
    liabilities_to_assets) and free_cash_flow in US dollars, computed only from items present.
    """
    try:
        ex = fetch_filing_extraction(ticker_or_cik, form=form, filings_back=filings_back)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return {"error": str(e)}
    out = ex.model_dump(mode="json")
    out["filing"]["url"] = ex.filing.url
    out["ratios"] = filing_ratios({k: v.value for k, v in ex.values.items()})
    return out


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
