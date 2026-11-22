"""Direct-prompt LLM baseline for Lilavati-Bench (no tools, no retrieval).

Each golden case is posed to a model as plain text plus its numeric context; the model must
return a JSON object with the case's expected keys. Answers are scored with the same
`_score_case_accuracy` rule used for the reference solver (fraction of expected keys matched
within tolerance). A trivial "always True / zeros" baseline is reported for calibration.

The API key is read ONLY from the environment (GEMINI_API_KEY or GROQ_API_KEY, per --provider)
and is never written to disk, logged or put in a URL.

    GEMINI_API_KEY=... python scripts/run_llm_baseline.py --model gemma-4-31b-it --samples 10 \
        --temperature 0.7 --rpm 25 --out results/baseline_gemma-4-31b-it.jsonl
    GROQ_API_KEY=... python scripts/run_llm_baseline.py --provider groq \
        --model openai/gpt-oss-120b --rpm 20 --out results/baseline_groq_gpt-oss-120b.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from lilavati.bench.datasets import load_all_test_cases
from lilavati.bench.generator import generate_cases
from lilavati.bench.llm_eval import (
    PROVIDERS,
    build_prompt,
    case_index,
    classify_outcome,
    make_record,
    parse_json,
    run_metadata,
    should_rerun,
    stop_reason,
)
from lilavati.bench.runner import BenchmarkRunner
from lilavati.llm.client import LLMMessage, UniversalLLMClient
from lilavati.llm.providers import ModelSpec


def call_with_retry(client: UniversalLLMClient, prompt: str, tries: int = 4) -> tuple[str, str | None]:
    """Return (text, stop reason); errors come back as "__ERROR__ ..." text with no stop reason."""
    for i in range(tries):
        try:
            resp = client.generate([LLMMessage(role="user", content=prompt)])
            return resp.content, stop_reason(resp.raw_response)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (429, 500, 503) and i < tries - 1:
                time.sleep(20 * (i + 1))
                continue
            return f"__ERROR__ HTTP {code}", None  # never include the request URL/headers
        except httpx.TimeoutException:
            if i < tries - 1:
                continue
            return "__ERROR__ Timeout", None
        except httpx.HTTPError as e:
            return f"__ERROR__ {type(e).__name__}", None
    return "__ERROR__ retries", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rpm", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true", help="re-run only provider errors and empty replies in --out")
    ap.add_argument("--rescore", action="store_true",
                    help="re-parse and re-score the raw replies stored in --out; no API calls")
    ap.add_argument("--generated", type=int, default=0, help="use N templated cases per supported track instead of golden")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    provider, base_url, key_env = PROVIDERS[args.provider]
    spec = ModelSpec(provider=provider, model_name=args.model, base_url=base_url,
                     api_key_env_var=key_env, temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout_seconds=args.timeout,
                     max_retries=0)  # this script retries itself (call_with_retry)
    client = UniversalLLMClient(spec)
    if not client.api_key and not args.rescore:
        raise SystemExit(f"{key_env} not set")

    cases = generate_cases(args.seed, args.generated) if args.generated else load_all_test_cases()
    runner = BenchmarkRunner(cases)
    gap = 60.0 / args.rpm
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = case_index(cases)
    prior: dict[tuple[int, str], dict[str, Any]] = {}
    if args.rescore and not out.exists():
        raise SystemExit(f"--rescore needs existing results in {out}")
    if (args.resume or args.rescore) and out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            if rec["case"] not in index:
                raise SystemExit(f"{out}: case {rec['case']} is not in this case set; check --generated/--seed")
            prior[(rec["sample"], index[rec["case"]].case_id)] = rec
    counts = dict.fromkeys(("answered", "unparsable", "truncated", "model_error", "error"), 0)
    per_track: dict[str, list[float]] = {c.track: [] for c in cases}
    rows: list[dict[str, Any]] = []
    for s_idx in range(args.samples):
        for c in cases:
            rec = prior.get((s_idx, c.case_id))
            if args.rescore:
                if rec is None:
                    raise SystemExit(f"--rescore: no stored reply for sample {s_idx}, {c.case_id}")
                ans = parse_json(rec["raw"])
                rec = {**rec, "parsed": ans, "score": runner._score_case_accuracy(c, ans or {})}
            elif should_rerun(rec):  # provider failures only; model failures are never re-rolled
                t0 = time.time()
                raw, finish = call_with_retry(client, build_prompt(c))
                ans = parse_json(raw)
                rec = make_record(provider=args.provider, model=args.model, sample=s_idx, case=c,
                                  raw=raw, finish_reason=finish, parsed=ans,
                                  score=runner._score_case_accuracy(c, ans or {}))
                time.sleep(max(0.0, gap - (time.time() - t0)))
            assert rec is not None
            rows.append(rec)
            counts[classify_outcome(rec)] += 1
            per_track[c.track].append(rec["score"])
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if not args.rescore:  # a rescore keeps the original run's metadata
        meta = run_metadata(script="scripts/run_llm_baseline.py", arm="no_tools", provider=args.provider,
                            model=args.model, samples=args.samples, temperature=args.temperature,
                            max_tokens=args.max_tokens, timeout_seconds=args.timeout, rpm=args.rpm,
                            case_set="generated" if args.generated else "golden",
                            seed=args.seed if args.generated else None,
                            per_track=args.generated or None, n_records=len(rows))
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    means = {t: statistics.mean(v) for t, v in per_track.items()}
    overall = statistics.mean(means.values())
    print(json.dumps({"provider": args.provider, "model": args.model, "samples": args.samples,
                      "temperature": args.temperature, "outcomes": counts,
                      "errors": counts["error"], "truncated": counts["truncated"],
                      "unparsable": counts["unparsable"],
                      "overall_mean_accuracy": round(overall, 4),
                      "per_track": {k: round(v, 3) for k, v in means.items()}}, indent=2))


if __name__ == "__main__":
    main()
