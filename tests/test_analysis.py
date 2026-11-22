"""Result analysis (lilavati.bench.analysis) on small synthetic records with known answers."""

from __future__ import annotations

from typing import Any

import pytest

from lilavati.bench.analysis import bootstrap_ci, field_results, paired_difference, summarise
from lilavati.bench.datasets import case_content_hash
from lilavati.bench.generator import generate_cases, legacy_case_id

CASES = generate_cases(2026, 4)
BOND = next(c for c in CASES if c.track == "track2_fixed_income")


_ANSWER: dict[str, Any] = {}


def _rec(case_id: str, score: float, raw: str = "{}", parsed: dict[str, Any] | None = _ANSWER,
         finish: str | None = "stop", sample: int = 0) -> dict[str, Any]:
    return {"case": case_id, "sample": sample, "raw": raw, "finish_reason": finish,
            "parsed": parsed, "score": score}


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    xs = [0.0, 0.5, 1.0, 1.0, 0.25]
    lo, hi = bootstrap_ci(xs)
    assert (lo, hi) == bootstrap_ci(xs)
    assert lo <= sum(xs) / len(xs) <= hi
    assert bootstrap_ci([1.0, 1.0]) == (1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_field_results_report_relative_error() -> None:
    expected = BOND.expected_values["bond_price"]
    fr = field_results(BOND, {"bond_price": expected * 1.10})["bond_price"]
    assert not fr["ok"] and fr["rel_err"] == pytest.approx(0.10)
    assert field_results(BOND, {"bond_price": expected})["bond_price"]["ok"]


def test_summarise_counts_outcomes_and_resolves_legacy_ids() -> None:
    legacy = legacy_case_id(BOND.case_id)  # "GEN-T2-000": pre-2026-11-21 files use this form
    rows = [
        _rec(legacy, 1.0, raw='{"bond_price": 1}', parsed={"bond_price": BOND.expected_values["bond_price"]}),
        _rec(CASES[0].case_id, 0.0, raw="__ERROR__ HTTP 503", parsed=None, finish=None),
        _rec(CASES[1].case_id, 0.0, raw="", parsed=None, finish="length"),
    ]
    s = summarise(rows, CASES)
    assert s["outcomes"] == {"answered": 1, "unparsable": 0, "truncated": 1, "model_error": 0, "error": 1}
    assert s["accuracy_on_answered"] == 1.0
    assert s["item_mean"] == pytest.approx(1 / 3)


def test_summarise_rejects_a_hash_mismatch() -> None:
    rec = {**_rec(BOND.case_id, 1.0), "case_hash": "0" * 64}
    with pytest.raises(ValueError, match="hash"):
        summarise([rec], CASES)


def test_paired_difference_matches_on_content_not_id() -> None:
    a = [_rec(legacy_case_id(BOND.case_id), 0.0), _rec(legacy_case_id(CASES[0].case_id), 1.0)]
    b = [_rec(BOND.case_id, 1.0), _rec(CASES[0].case_id, 1.0)]
    d = paired_difference(a, b, CASES)
    assert d["n_pairs"] == 2 and d["mean_diff"] == pytest.approx(0.5)
    assert (d["improved"], d["same"], d["worse"]) == (1, 1, 0)
    assert case_content_hash(BOND)  # pairing key exists for every case
