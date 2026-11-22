"""Tool-augmented LLM baseline for Lilavati-Bench (the paired arm of run_llm_baseline.py).

Same cases, same answer keys, same scoring; the model now has eight deterministic tools
(lilavati.bench.tool_eval.TOOLS) and is told to use them. Each record adds the tool calls made,
their outputs, and tool grounding (numeric answer fields equal to a tool-returned number).

Only generated cases are supported: the golden econometrics and financial-ML cases describe
random series in words, which no tool can reproduce exactly.

The API key is read ONLY from the environment (GROQ_API_KEY or GEMINI_API_KEY, per --provider)
and is never written to disk, logged or put in a URL.

    GROQ_API_KEY=... uv run python scripts/run_tool_baseline.py --provider groq \
        --model openai/gpt-oss-120b --generated 4 --seed 2026 \
        --out results/tools_generated_groq_gpt-oss-120b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from lilavati.bench.generator import generate_cases
from lilavati.bench.llm_eval import (
    PROVIDERS,
    TOOL_PROMPT_VERSION,
    case_index,
    classify_outcome,
    make_record,
    run_metadata,
    should_rerun,
)
from lilavati.bench.runner import BenchmarkRunner
from lilavati.bench.tool_eval import ToolsUnsupportedError, run_case
from lilavati.llm.providers import ModelSpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="groq")
    ap.add_argument("--model", required=True)
    ap.add_argument("--generated", type=int, required=True, help="cases per supported track")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tool-rounds", type=int, default=8)
    ap.add_argument("--case-gap", type=float, default=3.0, help="seconds between cases (rate limits)")
    ap.add_argument("--resume", action="store_true", help="re-run only provider errors in --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    provider, base_url, key_env = PROVIDERS[args.provider]
    spec = ModelSpec(provider=provider, model_name=args.model, base_url=base_url,
                     api_key_env_var=key_env, temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout_seconds=args.timeout,
                     max_retries=4, retry_backoff_seconds=5.0)  # client retries 429/5xx itself
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} not set")

    cases = generate_cases(args.seed, args.generated)
    runner = BenchmarkRunner(cases)
    index = case_index(cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prior: dict[tuple[int, str], dict[str, Any]] = {}
    if args.resume and out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            prior[(rec["sample"], index[rec["case"]].case_id)] = rec

    rows: list[dict[str, Any]] = []
    counts = dict.fromkeys(("answered", "unparsable", "truncated", "model_error", "error"), 0)
    per_track: dict[str, list[float]] = {c.track: [] for c in cases}
    for s_idx in range(args.samples):
        for c in cases:
            rec = prior.get((s_idx, c.case_id))
            if should_rerun(rec):
                t0 = time.time()
                try:
                    res = run_case(c, spec, max_tool_rounds=args.max_tool_rounds)
                except ToolsUnsupportedError as err:  # a capability gap, not a result: write nothing
                    raise SystemExit(f"{err}; no results written") from None
                score = runner._score_case_accuracy(c, res["parsed"] or {})
                rec = make_record(provider=args.provider, model=args.model, sample=s_idx, case=c,
                                  raw=res["raw"], finish_reason=res["finish_reason"],
                                  parsed=res["parsed"], score=score, arm="tools",
                                  tool_prompt_version=TOOL_PROMPT_VERSION, submitted=res["submitted"],
                                  late_error=res["late_error"], rejected_output=res["rejected_output"],
                                  tool_calls=res["tool_calls"], n_tool_calls=res["n_tool_calls"],
                                  grounded_fields=res["grounded_fields"],
                                  numeric_fields=res["numeric_fields"])
                time.sleep(max(0.0, args.case_gap - (time.time() - t0)))
            assert rec is not None
            rows.append(rec)
            counts[classify_outcome(rec)] += 1
            per_track[c.track].append(rec["score"])
            print(f"{c.case_id} {classify_outcome(rec):<11} score={rec['score']:.3f} "
                  f"tools={rec['n_tool_calls']} grounded={rec['grounded_fields']}/{rec['numeric_fields']}",
                  flush=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    meta = run_metadata(script="scripts/run_tool_baseline.py", arm="tools", tool_prompt_version=TOOL_PROMPT_VERSION,
                        provider=args.provider,
                        model=args.model, samples=args.samples, temperature=args.temperature,
                        max_tokens=args.max_tokens, timeout_seconds=args.timeout,
                        max_tool_rounds=args.max_tool_rounds, case_set="generated",
                        seed=args.seed, per_track=args.generated, n_records=len(rows))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    means = {t: statistics.mean(v) for t, v in per_track.items()}
    g = sum(r["grounded_fields"] for r in rows)
    n = sum(r["numeric_fields"] for r in rows)
    print(json.dumps({"provider": args.provider, "model": args.model, "arm": "tools", "outcomes": counts,
                      "overall_mean_accuracy": round(statistics.mean(means.values()), 4),
                      "tool_grounding": round(g / n, 4) if n else None,
                      "per_track": {k: round(v, 3) for k, v in means.items()}}, indent=2))


if __name__ == "__main__":
    main()
