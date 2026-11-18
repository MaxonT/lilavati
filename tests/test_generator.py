import statistics

from lilavati.bench.datasets import load_all_test_cases
from lilavati.bench.generator import GENERATED_TRACKS, generate_cases
from lilavati.bench.runner import BenchmarkRunner


def _trivial_score(cases) -> float:
    runner = BenchmarkRunner(cases)
    by_track: dict[str, list[float]] = {}
    for c in cases:
        ans = {k: (True if isinstance(v, bool) else 0.0) for k, v in c.expected_values.items()}
        by_track.setdefault(c.track, []).append(runner._score_case_accuracy(c, ans))
    return statistics.mean(statistics.mean(v) for v in by_track.values())


def test_generator_is_deterministic_and_varied() -> None:
    a, b = generate_cases(7, 6), generate_cases(7, 6)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
    assert generate_cases(8, 6)[0].context_data != a[0].context_data
    assert {c.track for c in a} == set(GENERATED_TRACKS)


def test_trivial_answerer_scores_low_on_generated_cases() -> None:
    # Guard against benchmark inflation: content-free answers must not score well.
    assert _trivial_score(generate_cases(0, 20)) < 0.35


def test_generated_booleans_are_balanced() -> None:
    vals = [v for c in generate_cases(0, 20) for v in c.expected_values.values() if isinstance(v, bool)]
    assert 0.35 < sum(vals) / len(vals) < 0.65


def test_golden_trivial_baseline_is_documented_weakness() -> None:
    # The 7 golden cases are guessable (~64%); tracked here so the weakness stays visible.
    assert _trivial_score(load_all_test_cases()) > 0.5
