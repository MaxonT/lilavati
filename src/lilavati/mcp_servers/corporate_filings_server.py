"""
FastMCP Server for SEC EDGAR Corporate Filings & Structured XBRL Disclosures.

Exposes tools to fetch, parse, and check (Z3, on the reported numbers) company three-statement filings
with cryptographic CitationRefs.

`get_filing_figures` is the filing-anchored path validated in Tutorial 08 and
docs/validation/connectors.md (identify the filing, then use only its own XBRL facts).
`get_company_filing_statements` identifies the fiscal year's 10-K the same way and maps its facts
into the three-statement model. `extract_single_line_item` selects by the `fy` tag with an
annual-duration and latest-period rule.
SEC requires a contact: set LILAVATI_SEC_USER_AGENT="Your Name you@example.org" in the server's
environment.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from lilavati.connectors.edgar import EdgarClient
from lilavati.connectors.sec_filing import fetch_filing_extraction, filing_ratios
from lilavati.core.invariants import diagnose_identities
from lilavati.validation_status import with_validation_status

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
@with_validation_status
def get_company_filing_statements(
    ticker_or_cik: str,
    fiscal_year: int,
) -> dict[str, Any]:
    """
    A company's 10-K for one fiscal year mapped into a three-statement model, from that filing's
    own XBRL facts (US dollars as reported, not scaled).

    Every statement field is either reported (cited in "sourced_facts", tag in "tags_used"), derived
    from reported items by the formula given in "derived" (e.g. total_debt, or total_liabilities when
    the company tags only the liabilities-and-equity total), or listed in "not_reported" (0.0 in the
    model: receivables, inventory, payables, interest, tax). "sga_expense" is all operating expenses
    (gross profit minus operating income), not the reported SG&A line. "checks" holds the balance
    sheet and cash reconciliation checks on the reported numbers. Returns {"error": ...} when the
    filing lacks an item the model needs (banks report no operating income); get_filing_figures
    then returns what the filing does report.
    """
    try:
        sourced_model = EdgarClient().extract_three_statement_model(
            ticker_or_cik=ticker_or_cik,
            fiscal_year=fiscal_year,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return {"error": str(e)}
    out = sourced_model.model_dump(mode="json")
    out["filing"]["url"] = sourced_model.filing.url
    return out


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
@with_validation_status
def extract_single_line_item(
    ticker_or_cik: str,
    metric_name: str,
    fiscal_year: int,
) -> dict[str, Any]:
    """
    One reported US-GAAP value from SEC EDGAR, with its citation.

    metric_name is a US-GAAP concept name (e.g. 'Assets', 'Revenues', 'NetIncomeLoss'). The value is
    in US dollars as reported (not scaled; only USD facts). The fiscal year's 10-K period is selected
    by the fy tag, the annual duration and the latest period end. Returns {"error": ...} if the concept
    is not reported. For a filing's full set of line items, prefer get_filing_figures.
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
@with_validation_status
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


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
@with_validation_status
def diagnose_accounting_identities(
    values: dict[str, float],
    tolerance: float = 0.5,
) -> dict[str, Any]:
    """
    Which extracted figures could be wrong, when accounting identities fail. values: figures in one
    currency unit, keyed by these names (any subset): total_assets, total_liabilities, total_equity,
    liabilities_and_equity, revenue, cost_of_revenue, gross_profit, cash_opening, operating_cash_flow,
    investing_cash_flow, financing_cash_flow, fx_effect_on_cash, cash_closing. Every identity whose
    figures are present is checked (assets = liabilities + equity; assets = total liabilities and
    equity; revenue - cost of revenue = gross profit; opening cash + cash flows = closing cash), with
    tolerance in the same units for rounding. Returns the failed identities and every smallest set of
    figures that, if wrong, explains all failures (with the values they would need), found with Z3's
    optimiser. One candidate means the identities single it out; several mean only the source can tell.
    """
    try:
        return diagnose_identities(values, tolerance=tolerance).model_dump()
    except ValueError as e:
        return {"error": str(e)}


def main() -> None:
    """Run this MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
