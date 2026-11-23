from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lilavati.connectors.edgar import EdgarClient
from lilavati.mcp_servers.corporate_filings_server import (
    extract_single_line_item,
    get_company_filing_statements,
)


@pytest.fixture
def mock_aapl_facts() -> dict[str, Any]:
    return {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 364980000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "Liabilities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 298150000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 66830000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 391035000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 123216000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 93736000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 118254000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "NetCashProvidedByUsedInInvestingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": -2530000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "NetCashProvidedByUsedInFinancingActivities": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": -116400000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
            }
        },
    }


def test_mcp_corporate_filings_get_statements(mock_aapl_facts: dict[str, Any]) -> None:
    client = EdgarClient()
    real_model = client.extract_three_statement_model(
        "AAPL", fiscal_year=2024, facts=mock_aapl_facts
    )

    with patch("lilavati.mcp_servers.corporate_filings_server.EdgarClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.extract_three_statement_model.return_value = real_model
        mock_client_cls.return_value = mock_instance

        res = get_company_filing_statements(ticker_or_cik="AAPL", fiscal_year=2024)
        assert res["ticker"] == "AAPL"
        assert res["fiscal_year"] == 2024
        assert res["is_formally_verified"] is True
        assert res["statement_model"]["income_statement"]["revenue"] == 391035000000.0


def test_mcp_corporate_filings_extract_line_item(mock_aapl_facts: dict[str, Any]) -> None:
    client = EdgarClient()
    real_item = client.extract_line_item(mock_aapl_facts, ["Assets"], fiscal_year=2024)
    assert real_item is not None

    with patch("lilavati.mcp_servers.corporate_filings_server.EdgarClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.fetch_company_facts.return_value = mock_aapl_facts
        mock_instance.extract_line_item.return_value = real_item
        mock_client_cls.return_value = mock_instance

        res = extract_single_line_item(ticker_or_cik="AAPL", metric_name="Assets", fiscal_year=2024)
        assert res["value"] == 364980000000.0
        assert res["citation"]["xbrl_tag"] == "us-gaap:Assets"

        mock_instance.extract_line_item.return_value = None
        err_res = extract_single_line_item(
            ticker_or_cik="AAPL", metric_name="Unknown", fiscal_year=2024
        )
        assert "error" in err_res


def test_mcp_get_filing_figures_lists_missing_items_and_computes_ratios() -> None:
    import json
    from pathlib import Path

    from lilavati.connectors.sec_filing import extract_filing, filings_from_submissions
    from lilavati.mcp_servers.corporate_filings_server import get_filing_figures

    fix = Path(__file__).parent / "fixtures"
    facts = json.loads((fix / "aapl_companyfacts_10k_fy2025_2026.json").read_text())
    filing = filings_from_submissions(json.loads((fix / "aapl_submissions_subset.json").read_text()))[0]
    real = extract_filing(facts, filing)

    with patch("lilavati.mcp_servers.corporate_filings_server.fetch_filing_extraction", return_value=real):
        res = get_filing_figures("AAPL")
    assert res["filing"]["accession"] == "0000320193-26-000026"
    assert res["filing"]["url"].endswith("/000032019326000026/aapl-20260926.htm")
    assert res["values"]["revenue"]["value"] == 478_834_000_000
    assert res["missing"] == ["total_equity_incl_nci"]
    # Apple FY2026: (478,834 - 245,232) / 478,834, as in Tutorial 08
    assert round(res["ratios"]["gross_margin"], 4) == 0.4879

    with patch("lilavati.mcp_servers.corporate_filings_server.fetch_filing_extraction",
               side_effect=ValueError("form must be one of ['10-K', '10-Q'], got '8-K'")):
        assert "error" in get_filing_figures("AAPL", form="8-K")
