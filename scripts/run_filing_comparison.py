"""Real-filing comparison for Tutorial 08: the same 10-K, three ways.

- Arm A, LLM alone: the model reads the filing's financial statements and reports the line items
  and computes five ratios itself.
- Arm B, LLM + Līlāvatī: the model only reads (extracts the line items); the ratios are computed by
  `lilavati.connectors.sec_filing.filing_ratios`.
- Arm C, Līlāvatī alone: line items from the filing's own XBRL facts (`extract_filing`), validated
  against the document by scripts/validate_sec_filings.py; ratios by the same function.

Truth: the XBRL values of the filing, each confirmed to appear in the filing document. An item is
asked and scored only if its value (USD millions, as printed) appears in the text given to the
model, so a wrong answer is never caused by missing input. Raw items must match to the million;
ratios must be within 0.5% (relative).

Needs data/cache/sec/ from scripts/validate_sec_filings.py and GROQ_API_KEY in the environment:

    GROQ_API_KEY=... uv run python scripts/run_filing_comparison.py --tickers AAPL MSFT WMT JNJ --samples 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from lilavati.bench.llm_eval import parse_json, run_metadata, stop_reason
from lilavati.connectors.sec_filing import (
    document_text,
    extract_filing,
    filing_ratios,
    filings_from_submissions,
    statements_window,
)
from lilavati.llm.client import LLMMessage, UniversalLLMClient
from lilavati.llm.providers import INSTITUTIONAL_MODEL_PRESETS

CACHE = Path("data/cache/sec")
OUT = Path("results/filings")
PROMPT_VERSION = "f1"

ITEMS = {
    "revenue": "total revenue / net sales",
    "cost_of_revenue": "total cost of sales / cost of revenue",
    "net_income": "net income (attributable to the company)",
    "total_assets": "total assets at fiscal year end",
    "total_liabilities": "total liabilities at fiscal year end",
    "operating_cash_flow": "net cash provided by operating activities",
    "capital_expenditure": "payments for property, plant and equipment (a positive number)",
}
RATIOS = {
    "gross_margin": "(revenue - cost_of_revenue) / revenue, as a decimal",
    "net_margin": "net_income / revenue, as a decimal",
    "return_on_assets": "net_income / total_assets (year-end assets), as a decimal",
    "free_cash_flow": "operating_cash_flow - capital_expenditure, USD millions",
    "liabilities_to_assets": "total_liabilities / total_assets, as a decimal",
}
RATIO_INPUTS = {"gross_margin": ("revenue", "cost_of_revenue"), "net_margin": ("net_income", "revenue"),
                "return_on_assets": ("net_income", "total_assets"),
                "free_cash_flow": ("operating_cash_flow", "capital_expenditure"),
                "liabilities_to_assets": ("total_liabilities", "total_assets")}
RATIO_TOL = 0.005


def shown(millions: float) -> str:
    return f"{round(abs(millions)):,}"


def prompt(company: str, period_end: str, items: list[str], ratios: list[str], window: str) -> str:
    keys = "\n".join([f'  "{k}": number, {ITEMS[k]}, USD millions' for k in items] +
                     [f'  "{k}": number, {RATIOS[k]}' for k in ratios])
    task = ("Report these line items for that fiscal year, as printed (USD millions)"
            + (", then compute the ratios yourself." if ratios else ". Do not compute anything else."))
    return (f"You are a financial analyst. Below are the financial statements from {company}'s Form 10-K "
            f"for the fiscal year ended {period_end}. {task}\n\n"
            f"Respond with ONLY a JSON object (no prose) with exactly these keys:\n{keys}\n\n"
            f"FINANCIAL STATEMENTS:\n{window}\n")


def ask(client: UniversalLLMClient, text: str) -> tuple[str, str | None]:
    for attempt in range(4):
        try:
            resp = client.generate([LLMMessage(role="user", content=text)])
            return resp.content, stop_reason(resp.raw_response)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 503) and attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            return f"__ERROR__ HTTP {e.response.status_code}", None
        except httpx.HTTPError as e:
            return f"__ERROR__ {type(e).__name__}", None
    return "__ERROR__ retries", None


def score(parsed: dict[str, Any] | None, truth: dict[str, float], truth_ratios: dict[str, float],
          items: list[str], ratios_from: dict[str, Any] | None) -> dict[str, Any]:
    def num(x: Any) -> float | None:
        return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    p = parsed or {}
    item_ok = {k: (num(p.get(k)) is not None and abs(num(p.get(k)) - truth[k]) <= 0.5) for k in items}  # type: ignore[operator]
    ratio_ok = {}
    for k, t in truth_ratios.items():
        got = num((ratios_from or {}).get(k))
        ratio_ok[k] = got is not None and abs(got - t) <= RATIO_TOL * abs(t)
    return {"items_correct": sum(item_ok.values()), "items_total": len(item_ok), "item_ok": item_ok,
            "ratios_correct": sum(ratio_ok.values()), "ratios_total": len(ratio_ok), "ratio_ok": ratio_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "WMT", "JNJ"])
    ap.add_argument("--models", nargs="+",
                    default=["groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen3.6-27b", "groq-qwen3.8-27b"])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--gap", type=float, default=4.0, help="seconds between calls")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score stored replies against the current truth (no model calls)")
    args = ap.parse_args()

    companies = json.loads((CACHE / "company_tickers.json").read_text())
    by_ticker = {row["ticker"]: str(row["cik_str"]).zfill(10) for row in companies.values()}
    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for ticker in args.tickers:
        cik = by_ticker[ticker]
        facts = json.loads((CACHE / f"facts_{cik}.json").read_text())
        filing = filings_from_submissions(json.loads((CACHE / f"submissions_{cik}.json").read_text()))[0]
        window = statements_window(document_text(
            (CACHE / f"{filing.accession}_{filing.primary_document}").read_bytes().decode("utf-8", "ignore")))
        ex = extract_filing(facts, filing)
        truth_all = {k: ex.values[k].value / 1e6 for k in ITEMS if k in ex.values}
        items = [k for k, v in truth_all.items() if re.search(rf"(?<![\d,.]){re.escape(shown(v))}(?![\d,])", window)]
        not_answerable = sorted(set(ITEMS) - set(items))
        truth = {k: truth_all[k] for k in items}
        truth_ratios = filing_ratios(truth)
        ratios = [k for k in RATIOS if k in truth_ratios]
        print(f"{ticker}: {filing.report_date}, asking {len(items)} items and {len(ratios)} ratios; "
              f"not answerable from the window: {not_answerable}", flush=True)
        for model in args.models:
            spec = INSTITUTIONAL_MODEL_PRESETS[model].model_copy(update={"max_retries": 0, "timeout_seconds": 180.0})
            path = OUT / f"{PROMPT_VERSION}_{ticker}_{model}.jsonl"
            if args.rescore:  # stored replies, current truth; the questions asked are those recorded
                meta = json.loads(path.with_suffix(".meta.json").read_text())
                asked = meta["items_asked"]
                t_items = {k: truth_all[k] for k in asked}
                t_ratios = filing_ratios(t_items)
                rescored = []
                for r in (json.loads(line) for line in path.read_text().splitlines()):
                    if r["arm"].startswith("B"):
                        r["ratios_scored"] = filing_ratios({k: float(v) for k, v in (r["parsed"] or {}).items()
                                                            if k in asked and isinstance(v, (int, float))
                                                            and not isinstance(v, bool)})
                    r.update(score(r["parsed"], t_items, t_ratios, asked, r["ratios_scored"]))
                    rescored.append(r)
                path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rescored))
                meta["rescored"] = "2026-11-23: truth revenue = total revenues (Revenues tag first); replies unchanged"
                path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                records = rescored
                for arm in ("A_llm_alone", "B_llm_extracts_lilavati_computes"):
                    rs = [r for r in records if r["arm"] == arm]
                    summary.append({"ticker": ticker, "model": spec.model_name, "arm": arm, "samples": len(rs),
                                    "items": f"{sum(r['items_correct'] for r in rs)}/{sum(r['items_total'] for r in rs)}",
                                    "ratios": f"{sum(r['ratios_correct'] for r in rs)}/{sum(r['ratios_total'] for r in rs)}"})
                continue
            client = UniversalLLMClient(spec)
            records = []
            for s_idx in range(args.samples):
                for arm in ("A_llm_alone", "B_llm_extracts_lilavati_computes"):
                    asks_ratios = arm.startswith("A")
                    raw, finish = ask(client, prompt(ex.company, str(filing.report_date), items,
                                                     ratios if asks_ratios else [], window))
                    parsed = None if raw.startswith("__") else parse_json(raw)
                    if asks_ratios:
                        ratio_src = parsed
                    else:
                        vals = {k: float(v) for k, v in (parsed or {}).items()
                                if k in items and isinstance(v, (int, float)) and not isinstance(v, bool)}
                        ratio_src = filing_ratios(vals)
                    sc = score(parsed, truth, truth_ratios, items, ratio_src)
                    records.append({"ticker": ticker, "accession": filing.accession, "model": spec.model_name,
                                    "arm": arm, "sample": s_idx, "prompt_version": PROMPT_VERSION,
                                    "raw": raw, "finish_reason": finish, "parsed": parsed,
                                    "ratios_scored": ratio_src, **sc})
                    print(f"  {model:18} s{s_idx} {arm[:1]} items {sc['items_correct']}/{sc['items_total']} "
                          f"ratios {sc['ratios_correct']}/{sc['ratios_total']} ({finish})", flush=True)
                    time.sleep(args.gap)
            path.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
            path.with_suffix(".meta.json").write_text(json.dumps(run_metadata(
                script="scripts/run_filing_comparison.py", case_set="filings", provider="groq",
                model=spec.model_name, samples=args.samples, temperature=spec.temperature,
                max_tokens=spec.max_tokens, ticker=ticker, accession=filing.accession,
                period_end=str(filing.report_date), filing_prompt_version=PROMPT_VERSION,
                items_asked=items, not_answerable=not_answerable, n_records=len(records)), indent=2) + "\n")
            for arm in ("A_llm_alone", "B_llm_extracts_lilavati_computes"):
                rs = [r for r in records if r["arm"] == arm]
                summary.append({"ticker": ticker, "model": spec.model_name, "arm": arm, "samples": len(rs),
                                "items": f"{sum(r['items_correct'] for r in rs)}/{sum(r['items_total'] for r in rs)}",
                                "ratios": f"{sum(r['ratios_correct'] for r in rs)}/{sum(r['ratios_total'] for r in rs)}"})
        summary.append({"ticker": ticker, "model": "lilavati-xbrl", "arm": "C_lilavati_alone", "samples": 1,
                        "items": f"{len(items)}/{len(items)}", "ratios": f"{len(ratios)}/{len(ratios)}",
                        "note": "truth by construction; validated against the document by validate_sec_filings.py"})
    (OUT / f"{PROMPT_VERSION}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
