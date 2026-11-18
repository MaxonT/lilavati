"""Calibration: score a content-free answerer (all bools True, all numbers 0) on the golden cases."""
import json
import statistics

from lilavati.bench.datasets import load_all_test_cases
from lilavati.bench.runner import BenchmarkRunner

r = BenchmarkRunner()
res = {}
for c in load_all_test_cases():
    ans = {k: (True if isinstance(v, bool) else 0.0) for k, v in c.expected_values.items()}
    res[c.track] = r._score_case_accuracy(c, ans)
print(json.dumps({"overall_mean_accuracy": round(statistics.mean(res.values()), 4), "per_track": res}, indent=2))
