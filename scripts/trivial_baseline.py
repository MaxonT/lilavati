"""Calibration: score a content-free answerer (all bools True, all numbers 0).

On the golden cases, and on the generated cases the benchmark results use (seed 2026, 4 per track). The
score per case is the fraction of answer fields correct, as in results/ANALYSIS.md; the mean is over cases.

    uv run python scripts/trivial_baseline.py
"""
import json
import statistics

from lilavati.bench.datasets import load_all_test_cases
from lilavati.bench.generator import generate_cases
from lilavati.bench.runner import BenchmarkRunner

r = BenchmarkRunner()


def content_free(cases):  # type: ignore[no-untyped-def]
    return [r._score_case_accuracy(c, {k: (True if isinstance(v, bool) else 0.0) for k, v in c.expected_values.items()})
            for c in cases]


golden = load_all_test_cases()
gen = generate_cases(2026, 4)
print(json.dumps({
    "golden_mean_accuracy": round(statistics.mean(content_free(golden)), 4),
    "golden_per_track": {c.track: s for c, s in zip(golden, content_free(golden), strict=True)},
    "generated_s2026_4_mean_accuracy": round(statistics.mean(content_free(gen)), 4),
    "generated_cases": len(gen),
}, indent=2))
