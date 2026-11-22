"""Analysis of stored Līlāvatī-Bench result files (``results/*.jsonl``).

Everything here works on stored records, so analyses are reproducible without API calls:

- `summarise`: outcome classes, accuracy on answered items, item mean with a bootstrap 95% CI,
  per-track means, per-field misses with relative errors, and tool grounding for tool runs.
- `paired_difference`: tools minus no-tools on the *same questions* (matched by case hash), with a
  bootstrap CI on the mean paired difference. This is the comparison that supports a claim such as
  "tools improve accuracy"; unpaired comparisons across different case sets do not.

Statistics are deliberately simple (nonparametric bootstrap over items, fixed seed) so that
students can reproduce them by hand.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from lilavati.bench.datasets import BenchmarkTestCase, case_content_hash
from lilavati.bench.llm_eval import case_index, classify_outcome


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bootstrap_ci(values: list[float], *, n_boot: int = 5000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Degenerate (all equal) samples return (v, v)."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if len(set(values)) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    lo = means[int((1 - level) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return lo, hi


def field_results(case: BenchmarkTestCase, parsed: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per answer key: correct?, the value given, the expected value and the relative error."""
    out: dict[str, dict[str, Any]] = {}
    for key, expected in case.expected_values.items():
        value = (parsed or {}).get(key)
        rel_err: float | None = None
        if isinstance(expected, bool):
            ok = value == expected  # same rule as BenchmarkRunner._score_case_accuracy
        elif isinstance(expected, (int, float)) and isinstance(value, (int, float)) and not isinstance(value, bool):
            ok = math.isclose(float(value), float(expected), rel_tol=case.tolerance)
            rel_err = abs(value - expected) / abs(expected) if expected else None
        else:
            ok = value == expected
        out[key] = {"ok": ok, "value": value, "expected": expected, "rel_err": rel_err}
    return out


def summarise(records: list[dict[str, Any]], cases: list[BenchmarkTestCase]) -> dict[str, Any]:
    """Summary statistics for one result file. Raises if a record's case is not in `cases`."""
    if not records:
        raise ValueError("no records")
    index = case_index(cases)
    outcomes: Counter[str] = Counter()
    scores: list[float] = []
    answered: list[float] = []
    per_track: dict[str, list[float]] = {}
    misses: dict[str, list[dict[str, Any]]] = {}
    grounded = numeric = 0
    for r in records:
        case = index.get(r["case"])
        if case is None:
            raise ValueError(f"case {r['case']} not in the case set")
        if "case_hash" in r and r["case_hash"] != case_content_hash(case):
            raise ValueError(f"case {r['case']}: stored hash does not match the case set")
        kind = classify_outcome(r)
        outcomes[kind] += 1
        scores.append(r["score"])
        per_track.setdefault(case.track, []).append(r["score"])
        if kind == "answered":
            answered.append(r["score"])
            for key, fr in field_results(case, r["parsed"]).items():
                if not fr["ok"]:
                    misses.setdefault(key, []).append({"case": r["case"], **fr})
        grounded += r.get("grounded_fields", 0)
        numeric += r.get("numeric_fields", 0)
    lo, hi = bootstrap_ci(scores)
    return {
        "n": len(records),
        "samples": len({r["sample"] for r in records}),
        "outcomes": {k: outcomes.get(k, 0) for k in ("answered", "unparsable", "truncated", "model_error", "error")},
        "accuracy_on_answered": statistics.mean(answered) if answered else None,
        "item_mean": statistics.mean(scores),
        "item_mean_ci95": (lo, hi),
        "per_track": {t: statistics.mean(v) for t, v in sorted(per_track.items())},
        "field_misses": misses,
        "tool_grounding": (grounded / numeric) if numeric else None,
        "mean_tool_calls": (statistics.mean(r["n_tool_calls"] for r in records)
                            if all("n_tool_calls" in r for r in records) else None),
    }


def paired_difference(a: list[dict[str, Any]], b: list[dict[str, Any]], cases: list[BenchmarkTestCase],
                      ) -> dict[str, Any]:
    """Mean of (b - a) item scores over questions present in both, matched by case hash."""
    index = case_index(cases)

    def by_hash(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
        return {(case_content_hash(index[r["case"]]), r["sample"]): r["score"] for r in rows}

    sa, sb = by_hash(a), by_hash(b)
    keys = sorted(set(sa) & set(sb))
    if not keys:
        raise ValueError("no shared questions to pair")
    diffs = [sb[k] - sa[k] for k in keys]
    lo, hi = bootstrap_ci(diffs)
    return {"n_pairs": len(keys), "mean_a": statistics.mean(sa[k] for k in keys),
            "mean_b": statistics.mean(sb[k] for k in keys), "mean_diff": statistics.mean(diffs),
            "diff_ci95": (lo, hi), "improved": sum(d > 0 for d in diffs),
            "worse": sum(d < 0 for d in diffs), "same": sum(d == 0 for d in diffs)}
