from datetime import date
from typing import Any

import pytest

from lilavati.connectors.edgar import EdgarClient


@pytest.fixture
def mock_aapl_facts() -> dict[str, Any]:
    """Synthetic company facts payload mirroring real Apple Inc. (AAPL) SEC EDGAR XBRL structure."""
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
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 29940000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
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
                "CostOfGoodsAndServicesSold": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 210352000000.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "accn": "0000320193-24-000106",
                            }
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "val": 180683000000.0,
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


def test_resolve_cik() -> None:
    client = EdgarClient()
    assert client.resolve_cik("AAPL") == "0000320193"
    assert client.resolve_cik("msft") == "0000789019"
    assert client.resolve_cik("320193") == "0000320193"

    with pytest.raises(ValueError):
        client.resolve_cik("UNKNOWN_NONEXISTENT_TICKER")


def test_extract_line_item(mock_aapl_facts: dict[str, Any]) -> None:
    client = EdgarClient()
    item = client.extract_line_item(mock_aapl_facts, ["Assets"], fiscal_year=2024)

    assert item is not None
    assert item.value == 364980000000.0
    assert item.citation.xbrl_tag == "us-gaap:Assets"
    assert item.citation.filing_date == date(2024, 11, 1)
    assert len(item.citation.doc_hash) == 64


def test_sec_requests_need_a_user_supplied_contact(monkeypatch) -> None:
    # SEC fair access requires a real contact in the User-Agent (HTTP 403 without one). The
    # client used to send a made-up default; it now reads LILAVATI_SEC_USER_AGENT and refuses
    # to send a request without it.
    import httpx
    import pytest

    monkeypatch.delenv("LILAVATI_SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="LILAVATI_SEC_USER_AGENT"):
        EdgarClient().fetch_company_facts("AAPL")

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"cik": 320193, "facts": {}})

    monkeypatch.setenv("LILAVATI_SEC_USER_AGENT", "Example Org contact@example.org")
    client = EdgarClient(transport=httpx.MockTransport(handler))
    assert client.fetch_company_facts("AAPL") == {"cik": 320193, "facts": {}}
    assert seen[0].headers["User-Agent"] == "Example Org contact@example.org"
    assert str(seen[0].url).endswith("/CIK0000320193.json")


def _aapl_facts() -> dict:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "aapl_companyfacts_10k_fy2025_2026.json"
    return json.loads(path.read_text())


def test_fiscal_year_selects_the_current_period_not_a_comparative() -> None:
    # Company facts tag every value in a 10-K with that filing's fiscal year, including the
    # prior-year comparatives. Selecting by fy alone returned FY2025 assets and FY2024 revenue
    # and net income for "FY2026". Expected values: Apple's FY2026 10-K statements (USD millions).
    facts = _aapl_facts()
    assets = EdgarClient.extract_line_item(facts, ["Assets"], 2026)
    revenue = EdgarClient.extract_line_item(
        facts, ["RevenueFromContractWithCustomerExcludingAssessedTax"], 2026)
    net_income = EdgarClient.extract_line_item(facts, ["NetIncomeLoss"], 2026)
    assert assets is not None and revenue is not None and net_income is not None
    assert assets.value == 404_548_000_000 and str(assets.citation.period_end) == "2026-09-26"
    assert revenue.value == 478_834_000_000
    assert net_income.value == 132_175_000_000


def test_three_statement_revenue_prefers_total_revenues() -> None:
    # Same rule as sec_filing.CANONICAL_ITEMS: Walmart FY2026 total revenues 713,163 (Revenues),
    # not net sales 706,413 (contract revenue). Fixture: verbatim SEC subset.
    import json
    from pathlib import Path

    facts = json.loads((Path(__file__).parent / "fixtures" / "wmt_companyfacts_revenue_fy2026.json").read_text())
    item = EdgarClient.extract_line_item(
        facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], 2026)
    assert item is not None and item.value == 713_163_000_000
    # The three-statement model uses the same order (tests/test_sec_statements.py, Walmart).


def test_unknown_ticker_resolves_through_secs_company_list_only_when_online() -> None:
    import httpx
    import pytest

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"0": {"cik_str": 104169, "ticker": "WMT", "title": "Walmart Inc."}})

    client = EdgarClient(user_agent="Example Org contact@example.org", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="online=True"):
        client.resolve_cik("WMT")
    assert calls == []
    assert client.resolve_cik("wmt", online=True) == "0000104169"
    assert client.resolve_cik("WMT", online=True) == "0000104169"
    assert calls == ["/files/company_tickers.json"]  # fetched once, then cached
    with pytest.raises(ValueError, match="not in SEC's company list"):
        client.resolve_cik("NOSUCH", online=True)
