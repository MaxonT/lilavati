import statistics

from lilavati.bench.datasets import load_all_test_cases
from lilavati.bench.generator import GENERATED_TRACKS, generate_cases, legacy_case_id
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


def test_case_ids_are_unique_across_set_sizes() -> None:
    # generate_cases(2026, 2) and (2026, 4) produced different bonds under the same id
    # "GEN-T2-000", so results matched by id compared different questions.
    small = {c.case_id: c for c in generate_cases(2026, 2)}
    large = {c.case_id: c for c in generate_cases(2026, 4)}
    assert not set(small) & set(large)


def test_content_hash_identifies_content() -> None:
    from lilavati.bench.datasets import case_content_hash

    def first_bond(n: int):
        return next(c for c in generate_cases(2026, n) if c.track == "track2_fixed_income")

    a, b = first_bond(2), first_bond(4)
    assert a.context_data != b.context_data
    assert case_content_hash(a) != case_content_hash(b)
    assert case_content_hash(a) == case_content_hash(first_bond(2))
    # The id changes with the set size, but identical content keeps an identical hash.
    t1_small, t1_large = generate_cases(2026, 2)[0], generate_cases(2026, 4)[0]
    assert t1_small.case_id != t1_large.case_id
    assert case_content_hash(t1_small) == case_content_hash(t1_large)


def test_published_generated_content_is_unchanged() -> None:
    # Pins the cases answered in results/ (seed 2026, 4 per track): renaming ids must not
    # change the questions, or stored answers could no longer be scored.
    cases = {legacy_case_id(c.case_id): c for c in generate_cases(2026, 4)}
    assert cases["GEN-T2-000"].context_data == {
        "face_value": 100.0, "coupon_rate": 0.0215, "ytm": 0.0698, "maturity_years": 6.0,
    }
    assert round(cases["GEN-T2-000"].expected_values["bond_price"], 4) == 76.9634


def test_legacy_ids_map_to_new_ids() -> None:
    assert legacy_case_id("GEN-S2026-N4-T2-000") == "GEN-T2-000"
    assert legacy_case_id("LILAVATI-T2-001") == "LILAVATI-T2-001"
