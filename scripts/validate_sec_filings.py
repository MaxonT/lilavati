"""Pilot validation of filing-anchored XBRL extraction across companies (roadmap 7.5).

For each ticker: resolve the CIK (SEC company list), take the latest 10-K from the submissions
index, extract the canonical line items from that filing's own XBRL facts
(`lilavati.connectors.sec_filing`), check the accounting identities, then cross-check every value
against the filing document itself: the number must appear in the 10-K's text at the scale the
filing uses (millions or thousands).

Needs a real contact for SEC (fair-access policy), read only from the environment:

    LILAVATI_SEC_USER_AGENT="Your Name you@example.org" uv run python scripts/validate_sec_filings.py AAPL MSFT

Downloads are cached in data/cache/sec/ (git-ignored); the summary goes to
results/connectors/sec_filings_pilot.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from lilavati.connectors.sec_filing import document_text, extract_filing, filings_from_submissions

CACHE = Path("data/cache/sec")
OUT = Path("results/connectors/sec_filings_pilot.json")


def get(url: str, name: str) -> bytes:
    path = CACHE / name
    if path.exists():
        return path.read_bytes()
    ua = os.environ.get("LILAVATI_SEC_USER_AGENT", "").strip()
    if not ua:
        raise SystemExit("Set LILAVATI_SEC_USER_AGENT to 'Your Name you@example.org' (SEC fair-access policy).")
    resp = httpx.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=120,
                     follow_redirects=True)
    resp.raise_for_status()
    time.sleep(0.2)  # well under SEC's 10 requests per second
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


def appears(value: float, text: str) -> str | None:
    """The scale at which the value appears in the document text ('millions', 'thousands', 'units')."""
    for scale, label in ((1e6, "millions"), (1e3, "thousands"), (1.0, "units")):
        scaled = abs(value) / scale
        if abs(scaled - round(scaled)) > 1e-9:
            continue
        shown = f"{round(scaled):,}"
        if re.search(rf"(?<![\d,.]){re.escape(shown)}(?![\d,])", text):
            return label
    return None


def main(tickers: list[str]) -> None:
    companies = json.loads(get("https://www.sec.gov/files/company_tickers.json", "company_tickers.json"))
    by_ticker = {row["ticker"]: str(row["cik_str"]).zfill(10) for row in companies.values()}
    report: dict[str, Any] = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "companies": []}
    for ticker in tickers:
        cik = by_ticker[ticker]
        subs = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik}.json", f"submissions_{cik}.json"))
        facts = json.loads(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"facts_{cik}.json"))
        filing = filings_from_submissions(subs)[0]
        doc = document_text(get(filing.url, f"{filing.accession}_{filing.primary_document}").decode("utf-8", "ignore"))
        ex = extract_filing(facts, filing)
        items = {}
        for name, sv in ex.values.items():
            items[name] = {"value": sv.value, "tag": ex.tags_used[name], "in_document": appears(sv.value, doc)}
        not_in_doc = [k for k, v in items.items() if v["in_document"] is None]
        report["companies"].append({
            "ticker": ticker, "company": ex.company, "form": filing.form, "accession": filing.accession,
            "period_end": filing.report_date.isoformat(), "filed": filing.filing_date.isoformat(), "url": filing.url,
            "items": items, "missing": ex.missing, "not_found_in_document": not_in_doc,
            "checks": [c.model_dump() for c in ex.checks]})
        checks = " ".join(f"{'ok' if c.holds else 'FAIL'}" for c in ex.checks)
        print(f"{ticker:5} {ex.company[:28]:28} {filing.report_date} extracted {len(items):2}, missing {len(ex.missing):2}, "
              f"in document {len(items) - len(not_in_doc):2}/{len(items)}, identities [{checks}]")
        if not_in_doc:
            print("      not found in document:", not_in_doc)
        print("      missing:", ex.missing)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["AAPL", "MSFT", "WMT", "JNJ", "JPM"])
