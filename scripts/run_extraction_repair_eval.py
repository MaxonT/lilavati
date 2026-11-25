"""Evaluate extraction with diagnosis-driven repair on real 10-Ks (docs/agent-workflows.md, workflow 1).

Arms, same model, filings and figures:

- ``single``: one extraction pass, no repair (baseline).
- ``targeted``: re-read only the figures the checks name (provenance and identity diagnosis); implied
  values not shown.
- ``reread_all``: after any failure, re-read every figure.
- ``implied``: targeted, and the re-read request shows the values the identities imply.

Paired design: within a sample, every arm starts from the same first reading (one model call, reused),
so the arms differ only in what happens after the checks. ``single`` is that first reading.

Truth: the filing's own XBRL facts (USD millions), the same filing-anchored values validated against the
documents (scripts/validate_sec_filings.py). A figure is asked only if its value appears in the text the
model sees. Correct means within 0.5 of the truth. Provider errors are recorded separately and never
counted as model errors. Every reply is saved.

Needs data/cache/sec/ (scripts/validate_sec_filings.py) and, for real runs, GROQ_API_KEY:

    uv run python scripts/run_extraction_repair_eval.py --dry-run          # scripted extractor, no API
    uv run python scripts/run_extraction_repair_eval.py --report x2        # tables from saved records
    GROQ_API_KEY=... uv run python scripts/run_extraction_repair_eval.py --models groq-gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from lilavati.agents.extraction import FIGURE_LABELS, FilingExtractionAgent, llm_extractor
from lilavati.bench.llm_eval import run_metadata
from lilavati.connectors.edgar import STATEMENT_ITEMS, _opening_balance, _period_start
from lilavati.connectors.sec_filing import (
    document_text,
    extract_filing,
    filings_from_submissions,
    statements_window,
)
from lilavati.llm.client import UniversalLLMClient
from lilavati.llm.providers import INSTITUTIONAL_MODEL_PRESETS

CACHE = Path("data/cache/sec")
OUT = Path("results/extraction")
# x1: quotes had to match the document's spacing and apostrophes ("(139,500)" for the text's
# "(139,500 )", "shareholders'" for "shareholders’" were sent back), and the revenue label read
# "total revenue (net sales)" (Walmart prints both). x2: the quote check ignores whitespace and treats
# curly quotes as straight (SourceStore); the label asks for total revenues.
VERSION = "x2"
ARMS: dict[str, dict[str, Any]] = {
    "single": {"max_repair_rounds": 0},
    "targeted": {"max_repair_rounds": 2},
    "reread_all": {"max_repair_rounds": 2, "reread": "all"},
    "implied": {"max_repair_rounds": 2, "show_implied_values": True},
}


def truth_for(facts: dict[str, Any], filing: Any) -> dict[str, float]:
    """The filing's own XBRL values for the agent's figures, USD millions."""
    ex = extract_filing(facts, filing, STATEMENT_ITEMS)
    v = {k: sv.value / 1e6 for k, sv in ex.values.items()}
    out = {k: v[k] for k in ("revenue", "cost_of_revenue", "gross_profit", "total_assets", "total_liabilities",
                             "liabilities_and_equity", "operating_cash_flow", "investing_cash_flow",
                             "financing_cash_flow", "fx_effect_on_cash") if k in v}
    if "total_equity_incl_nci" in v or "total_equity" in v:
        out["total_equity"] = v.get("total_equity_incl_nci", v.get("total_equity", 0.0))
    cash_key = "cash_incl_restricted" if "cash_incl_restricted" in v else "cash"
    if cash_key in v and "operating_cash_flow" in ex.tags_used:
        out["cash_closing"] = v[cash_key]
        opening = _opening_balance(facts, filing, _period_start(facts, filing, ex.tags_used["operating_cash_flow"]),
                                   [ex.tags_used[cash_key]])
        if opening is not None:
            out["cash_opening"] = opening[1].value / 1e6
    return out


def shown(x: float) -> str:
    return f"{round(abs(x)):,}"


def scripted(truth: dict[str, float], window: str):  # type: ignore[no-untyped-def]
    """Dry run: reads the truth, quoting the excerpt around it; total assets from the wrong number once."""
    calls = {"n": 0}

    def extract(text: str, figures: list[str], instruction: str) -> dict[str, tuple[float, str]]:
        calls["n"] += 1
        out = {}
        for f in figures:
            m = re.search(rf"(?<![\d,.]){re.escape(shown(truth[f]))}(?![\d,])", text)
            excerpt = text[max(0, (m.start() if m else 0) - 40): (m.end() if m else 0) + 5]
            out[f] = (truth[f], excerpt)
        if calls["n"] == 1 and "total_assets" in figures:
            out["total_assets"] = (truth["total_assets"] - 1000, out["total_assets"][1])  # a misread
        return out

    return extract


def report(version: str) -> None:
    """Markdown tables from the saved records of ``version`` (pooled over filings; provider errors apart)."""
    rows = [json.loads(line) for p in sorted(OUT.glob(f"{version}_*.jsonl")) for line in p.open()]
    errors = [r for r in rows if r.get("provider_error")]
    rows = [r for r in rows if not r.get("provider_error")]
    print(f"{version}: {len(rows)} runs, {len(errors)} provider errors\n")
    print("| Model | Temperature | Arm | Figures | Accepted and correct | Accepted but wrong "
          "| Abstained (of which correct) | Consistent runs | Calls | Prompt tokens |")
    print("| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model, temp in sorted({(r["model"], r.get("temperature", 0.0)) for r in rows}):
        for arm in ARMS:
            rs = [r for r in rows if r["model"] == model and r.get("temperature", 0.0) == temp and r["arm"] == arm]
            if not rs:
                continue
            n = sum(r["n_figures"] for r in rs)
            abst = sum(len(r["unresolved"]) for r in rs)
            abst_ok = sum(r["correct"][f] for r in rs for f in r["unresolved"])
            wrong = sum(len(r["wrong_but_accepted"]) for r in rs)
            print(f"| {model} | {temp:g} | {arm} | {n} | {n - abst - wrong} | {wrong} | {abst} ({abst_ok}) "
                  f"| {sum(r['consistent'] for r in rs)}/{len(rs)} | {sum(r['calls'] for r in rs)} "
                  f"| {sum(r['prompt_tokens'] for r in rs):,} |")
    wrong_list = sorted({(r["ticker"], r["model"], r.get("temperature", 0.0), r["sample"], f, r["values"].get(f),
                          r["truth"][f], f in r.get("unchecked", [])) for r in rows for f in r["wrong_but_accepted"]})
    if wrong_list:
        print("\nAccepted but wrong (ticker, model, temperature, sample, figure, read, filing, unchecked):")
        for w in wrong_list:
            print(" ", w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "WMT", "JNJ", "JPM"])
    ap.add_argument("--models", nargs="+", default=["groq-gpt-oss-120b"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--gap", type=float, default=4.0, help="seconds between model calls")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true", help="scripted extractor; no API calls; nothing saved")
    ap.add_argument("--report", metavar="VERSION", help="print tables from saved records (e.g. x2) and exit")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return

    companies = json.loads((CACHE / "company_tickers.json").read_text())
    by_ticker = {row["ticker"]: str(row["cik_str"]).zfill(10) for row in companies.values()}
    OUT.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        cik = by_ticker[ticker]
        facts = json.loads((CACHE / f"facts_{cik}.json").read_text())
        filing = filings_from_submissions(json.loads((CACHE / f"submissions_{cik}.json").read_text()))[0]
        window = statements_window(document_text(
            (CACHE / f"{filing.accession}_{filing.primary_document}").read_bytes().decode("utf-8", "ignore")))
        truth_all = truth_for(facts, filing)
        figures = [f for f, x in truth_all.items() if f in FIGURE_LABELS
                   and re.search(rf"(?<![\d,.]){re.escape(shown(x))}(?![\d,])", window)]
        truth = {f: truth_all[f] for f in figures}
        print(f"{ticker} {filing.report_date}: {len(figures)} figures answerable from the window "
              f"(not: {sorted(set(truth_all) - set(figures))})", flush=True)
        for model in (["dry-run"] if args.dry_run else args.models):
            records = []
            for s in range(args.samples):
                first: dict[str, Any] = {}  # this sample's first reading, shared by every arm
                for arm in args.arms:
                    transcript: list[dict[str, Any]] = []
                    if args.dry_run:
                        extractor = scripted(truth, window)
                    else:
                        spec = INSTITUTIONAL_MODEL_PRESETS[model].model_copy(
                            update={"timeout_seconds": 180.0, "temperature": args.temperature})
                        base = llm_extractor(UniversalLLMClient(spec), transcript)

                        def extractor(text: str, figs: list[str], instr: str, _b=base, _t=transcript,
                                      _first=first) -> dict[str, tuple[float, str]]:
                            if not _t and "answers" in _first:  # the arm's first call: reuse the reading
                                _t.append({**_first["entry"], "shared_first_reading": True})
                                return dict(_first["answers"])
                            time.sleep(args.gap)
                            answers = _b(text, figs, instr)
                            if len(_t) == 1:
                                _first.update(answers=dict(answers), entry=_t[0])
                            return answers
                    rec: dict[str, Any] = {"ticker": ticker, "accession": filing.accession, "model": model, "arm": arm,
                                           "sample": s, "temperature": args.temperature, "version": VERSION, "figures": figures,
                                           "truth": truth}
                    try:
                        # No exchange-rate line in the filing's XBRL: the cash roll-forward applies without it.
                        fx_absent = "fx_effect_on_cash" not in truth_all
                        res = FilingExtractionAgent(extractor, fx_line_absent=fx_absent, **ARMS[arm]).run(
                            window, figures, filing_type=filing.form, filing_date=filing.filing_date,
                            period_end=filing.report_date)
                    except httpx.HTTPError as e:  # provider failure, not a model error
                        rec.update(provider_error=f"{type(e).__name__}: {e}"[:300], transcript=transcript)
                        records.append(rec)
                        print(f"  {model} {arm} s{s}: provider error", flush=True)
                        continue
                    correct = {f: f in res.figures and abs(res.figures[f].value - truth[f]) <= 0.5 for f in figures}
                    rec.update(
                        provider_error=None, consistent=res.consistent, unresolved=res.unresolved,
                        correct=correct, n_correct=sum(correct.values()), n_figures=len(figures),
                        wrong_but_accepted=[f for f in figures if not correct[f] and f not in res.unresolved],
                        unchecked=res.unchecked,
                        rounds=[r.model_dump() for r in res.rounds],
                        unsupported_first_round=res.rounds[0].unsupported if res.rounds else [],
                        values={f: x.value for f, x in res.figures.items()},
                        quotes_ok={f: x.quote_in_document for f, x in res.figures.items()},
                        calls=len(transcript),  # as if the arm ran alone; the first reading is shared
                        calls_made=sum(not t.get("shared_first_reading") for t in transcript),
                        prompt_tokens=sum(t["prompt_tokens"] for t in transcript),
                        completion_tokens=sum(t["completion_tokens"] for t in transcript),
                        transcript=transcript)
                    records.append(rec)
                    print(f"  {model} {arm:10} s{s}: correct {rec['n_correct']}/{len(figures)}, rounds {len(res.rounds)}, "
                          f"unresolved {res.unresolved}, wrong but accepted {rec['wrong_but_accepted']}, "
                          f"first-round unsupported {rec['unsupported_first_round']}", flush=True)
            if args.dry_run:
                continue
            tag = "" if args.temperature == 0 else f"_t{args.temperature:g}"
            path = OUT / f"{VERSION}_{ticker}_{model}{tag}.jsonl"
            path.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
            path.with_suffix(".meta.json").write_text(json.dumps(run_metadata(
                script="scripts/run_extraction_repair_eval.py", case_set="extraction", model=model,
                model_name=INSTITUTIONAL_MODEL_PRESETS[model].model_name,
                base_url=INSTITUTIONAL_MODEL_PRESETS[model].base_url,
                ticker=ticker, accession=filing.accession, period_end=str(filing.report_date), samples=args.samples,
                temperature=args.temperature, paired_first_reading=True,
                arms=args.arms, extraction_version=VERSION, figures=figures, gap_seconds=args.gap,
                run_date=str(date.today()), n_records=len(records)), indent=2) + "\n")
    if not args.dry_run:  # the summary covers every saved run, not only this invocation
        summary = [json.loads(line) for p in sorted(OUT.glob(f"{VERSION}_*.jsonl")) for line in p.open()]
        (OUT / f"{VERSION}_summary.json").write_text(json.dumps(
            [{k: r.get(k) for k in ("ticker", "model", "arm", "sample", "temperature", "provider_error", "n_correct", "n_figures",
                                     "consistent", "unresolved", "wrong_but_accepted", "unchecked", "unsupported_first_round",
                                     "calls", "calls_made", "prompt_tokens", "completion_tokens")} for r in summary], indent=1) + "\n")


if __name__ == "__main__":
    main()
