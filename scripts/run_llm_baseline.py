"""Direct-prompt LLM baseline for Lilavati-Bench (no tools, no retrieval).

Each golden case is posed to a model as plain text plus its numeric context; the model must
return a JSON object with the case's expected keys. Answers are scored with the same
`_score_case_accuracy` rule used for the reference solver (fraction of expected keys matched
within tolerance). A trivial "always True / zeros" baseline is reported for calibration.

The API key is read ONLY from the environment (GEMINI_API_KEY) and is never written to disk.

    GEMINI_API_KEY=... python scripts/run_llm_baseline.py --model gemma-4-31b-it --samples 10 \
        --temperature 0.7 --rpm 25 --out results/baseline_gemma-4-31b-it.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from lilavati.bench.datasets import BenchmarkTestCase, load_all_test_cases
from lilavati.bench.generator import generate_cases
from lilavati.bench.runner import BenchmarkRunner
from lilavati.llm.client import LLMMessage, UniversalLLMClient
from lilavati.llm.providers import ModelProvider, ModelSpec

# Plain-language definition of every answer key, so scoring is unambiguous.
KEY_DEFS: dict[str, dict[str, str]] = {
    "track1_three_statement": {
        "is_balanced": "bool: Assets == Liabilities + Equity",
        "cash_reconciled": "bool: cash_start + CFO + CFI + CFF == cash_end",
        "assets_equal_liab_equity": "bool: same test as is_balanced",
    },
    "track2_fixed_income": {
        "bond_price": "number: price of the 5-year bond",
        "macaulay_duration_in_range": "bool: 4.0 <= Macaulay duration (years) <= 5.0",
        "merton_distance_to_default_positive": "bool: Merton distance-to-default > 0",
        "default_probability_in_range": "bool: 0 <= Merton default probability <= 0.10",
    },
    "track3_econometrics": {
        "series_non_stationary": "bool: an ADF test on a 200-step Gaussian random walk fails to reject a unit root",
        "rejects_spurious_relationship": "bool: you reject treating levels regressions on it as valid (same direction as above)",
    },
    "track4_financial_ml": {
        "is_stationary_after_fracdiff": "bool: ADF p-value < 0.05 at the optimal d* for a 300-step random walk with drift 0.01, sd 1, start 100",
        "memory_preserved": "bool: correlation between the fractionally differenced and original series >= 0.25",
    },
    "track5_derivatives": {
        "call_price": "number: Black-Scholes call price",
        "delta": "number: call delta",
        "gamma": "number: call gamma",
        "vega": "number: call vega per 1 percentage-point (0.01) change in volatility, i.e. dPrice/dSigma divided by 100",
        "butterfly_arbitrage_free": "bool: C(K1) - 2C(K2) + C(K3) >= 0",
    },
    "track6_microstructure": {
        "spread": "number: ask - bid",
        "microprice": "number: (bid*ask_size + ask*bid_size)/(bid_size+ask_size)",
        "initial_liquidation_shares": "number: shares held at the start of the schedule",
        "final_shares_remaining": "number: shares held at the end of the schedule",
    },
    "track7_sustainability": {
        "portfolio_waci": "number: WACI = sum_i w_i * emissions_i / revenue_i(USD m), tCO2e per USD m",
        "slb_spt_achieved": "bool: reduction target met",
        "slb_active_coupon": "number: coupon after any step-up (decimal, e.g. 0.0425)",
        "slb_penalty_triggered": "bool: coupon step-up applies",
    },
}


def build_prompt(case: BenchmarkTestCase) -> str:
    keys = "\n".join(
        f'  "{k}": {d}' for k, d in KEY_DEFS[case.track].items() if k in case.expected_values
    )
    return (
        "You are a quantitative finance expert. Answer WITHOUT using tools; compute carefully.\n\n"
        f"TASK: {case.prompt}\n\nINPUT DATA (JSON): {json.dumps(case.context_data)}\n\n"
        "Respond with ONLY a JSON object (no prose, no markdown) with exactly these keys:\n"
        f"{keys}\n"
    )


def parse_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def call_with_retry(client: UniversalLLMClient, prompt: str, tries: int = 4) -> str:
    for i in range(tries):
        try:
            return client.generate([LLMMessage(role="user", content=prompt)]).content
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (429, 500, 503) and i < tries - 1:
                time.sleep(20 * (i + 1))
                continue
            return f"__ERROR__ HTTP {code}"  # never include the request URL/headers
        except httpx.TimeoutException:
            if i < tries - 1:
                continue
            return "__ERROR__ Timeout"
        except httpx.HTTPError as e:
            return f"__ERROR__ {type(e).__name__}"
    return "__ERROR__ retries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rpm", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true", help="re-run only failed/empty items in --out")
    ap.add_argument("--generated", type=int, default=0, help="use N templated cases per supported track instead of golden")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = ModelSpec(provider=ModelProvider.GEMINI, model_name=args.model,
                     api_key_env_var="GEMINI_API_KEY", temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout_seconds=args.timeout,
                     max_retries=0)  # this script retries itself (call_with_retry)
    client = UniversalLLMClient(spec)
    if not client.api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    cases = generate_cases(args.seed, args.generated) if args.generated else load_all_test_cases()
    runner = BenchmarkRunner(cases)
    gap = 60.0 / args.rpm
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_track: dict[str, list[float]] = {c.track: [] for c in cases}
    by_id = {c.case_id: c for c in cases}
    prior: dict[tuple[int, str], dict[str, Any]] = {}
    if args.resume and out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            prior[(rec["sample"], rec["case"])] = rec
    errors = parse_fail = 0
    rows: list[dict[str, Any]] = []
    for s_idx in range(args.samples):
        for c in cases:
            rec = prior.get((s_idx, c.case_id))
            if rec is None or rec["parsed"] is None:
                t0 = time.time()
                raw = call_with_retry(client, build_prompt(c))
                ans = parse_json(raw)
                rec = {"model": args.model, "sample": s_idx, "case": c.case_id, "raw": raw,
                       "parsed": ans, "score": runner._score_case_accuracy(c, ans or {})}
                time.sleep(max(0.0, gap - (time.time() - t0)))
            rows.append(rec)
            if rec["raw"].startswith("__ERROR__"):
                errors += 1
            elif rec["parsed"] is None:
                parse_fail += 1
            per_track[by_id[rec["case"]].track].append(rec["score"])
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    means = {t: statistics.mean(v) for t, v in per_track.items()}
    overall = statistics.mean(means.values())
    print(json.dumps({"model": args.model, "samples": args.samples, "temperature": args.temperature,
                      "errors": errors, "unparsable": parse_fail,
                      "overall_mean_accuracy": round(overall, 4),
                      "per_track": {k: round(v, 3) for k, v in means.items()}}, indent=2))


if __name__ == "__main__":
    main()
