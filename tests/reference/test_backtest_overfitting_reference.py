"""Reference tests for core/backtest_overfitting.py.

Sources:
- [DSR-slides] López de Prado, M. (2015), "Advances in quantitative meta-strategies", slides 11-14
  (Nomura conference): the Deflated Sharpe Ratio of Bailey and López de Prado (2014), Journal of
  Portfolio Management 40(5), with its numerical example: a daily strategy with annualised SR 2.5 after
  N = 100 independent trials, V[SR_n] = 1/2, T = 1250, skewness -3, kurtosis 10: SR0 ~ 0.1132 and
  DSR ~ 0.9004; DSR ~ 0.9505 after N = 46; with skewness 0 and kurtosis 3, DSR = 0.9505 after N = 88.
- [BODT] Bailey, Borwein, López de Prado, Salehipour and Zhu (2015), "Online tools for demonstration of
  backtest overfitting", eq. (2)-(3): DSR = Z[(SR - SR0) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)].
- [FST] López de Prado and Bailey (2021), "The false strategy theorem", American Mathematical Monthly
  128(9), eq. (1); section 3: SR 2.5 over 5 years of Gaussian daily returns falls below 0.95 confidence
  after 89 independent trials.
- [PBO] Bailey, Borwein, López de Prado and Zhu (2017), "The probability of backtest overfitting",
  Journal of Computational Finance 20(4): ranks (section 2.1 example), CSCV (Algorithm 2.3), PBO
  (section 3.1), combination counts (eq. 2.3 and section 4: S = 4 -> 6, S = 12 -> 924,
  S = 24 -> 2,704,156; the paper's "12,780" for S = 16 is C(16, 8) = 12,870).
- Purging and embargo (López de Prado 2018, AFML ch. 7 and 12) are checked against an independent
  per-observation implementation of the definitions below, and against hand-worked cases.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, rankdata

from lilavati.core.backtest_overfitting import (
    combinatorial_purged_splits,
    deflated_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    purged_kfold_splits,
)

# ---------------------------------------------------------------------------------------------
# Deflated and Probabilistic Sharpe ratios
# ---------------------------------------------------------------------------------------------

EXAMPLE = {"sharpe_ratio": 2.5, "n_obs": 1250, "sharpe_ratio_variance": 0.5, "periods_per_year": 250}


def test_dsr_worked_example() -> None:
    r = deflated_sharpe_ratio(n_trials=100, skewness=-3.0, kurtosis=10.0, **EXAMPLE)
    assert abs(r.expected_max_sharpe_ratio - 0.1132) < 5e-5  # [DSR-slides] SR0 ~ 0.1132 (daily)
    assert abs(r.deflated_sharpe_ratio - 0.9004) < 5e-5  # [DSR-slides] 0.9004
    assert abs(r.sharpe_ratio - 2.5 / math.sqrt(250)) < 1e-15


def test_dsr_fewer_trials_and_normal_returns() -> None:
    assert abs(deflated_sharpe_ratio(n_trials=46, skewness=-3.0, kurtosis=10.0, **EXAMPLE).deflated_sharpe_ratio
               - 0.9505) < 5e-5  # [DSR-slides] N = 46
    normal = {f"n{n}": deflated_sharpe_ratio(n_trials=n, **EXAMPLE).deflated_sharpe_ratio for n in (88, 89)}
    assert abs(normal["n88"] - 0.9505) < 5e-5  # [DSR-slides] N = 88, skewness 0, kurtosis 3
    assert normal["n88"] >= 0.95 > normal["n89"]  # [FST] below 0.95 after 89 trials


def test_dsr_is_psr_at_sr0_and_falls_with_trials() -> None:
    r = deflated_sharpe_ratio(n_trials=100, skewness=-3.0, kurtosis=10.0, **EXAMPLE)
    psr = probabilistic_sharpe_ratio(r.sharpe_ratio, r.expected_max_sharpe_ratio, 1250, -3.0, 10.0)
    assert abs(psr - r.deflated_sharpe_ratio) < 1e-15
    values = [deflated_sharpe_ratio(n_trials=n, **EXAMPLE).deflated_sharpe_ratio for n in (2, 10, 100, 1000)]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("n, rel_error", [(10, 0.0233), (100, 0.0092), (1000, 0.0042)])
def test_expected_maximum_against_exact_expectation(n: int, rel_error: float) -> None:
    # Exact E[max of n iid N(0,1)] = integral of x n phi(x) Phi(x)^(n-1); [FST] eq. (1) is for large n.
    exact = integrate.quad(lambda x: x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1), -12, 12, limit=200)[0]
    approx = expected_maximum_sharpe_ratio(n, 1.0)
    assert abs(approx / exact - 1 - rel_error) < 5e-5  # measured: +2.33%, +0.92%, +0.42%
    assert abs(expected_maximum_sharpe_ratio(n, 4.0) - 2 * approx) < 1e-12  # scales with sqrt(V)


def test_psr_standard_error_with_skewed_returns() -> None:
    # The PSR denominator is the asymptotic variance of the estimated SR for non-normal returns. Returns
    # mu - sigma (E - 1), E ~ Exp(1): skewness -2, raw kurtosis 9, SR = mu/sigma = 0.1. The spread of the
    # SR estimate over 20,000 samples of T = 500 matches sqrt(v/(T-1)) with v = 1 + 0.2 + 2 * 0.01 = 1.22;
    # the normal formula (v = 1.005) is 10% too narrow.
    rng = np.random.default_rng(20261126)
    t, reps, sr = 500, 20_000, 0.1
    x = sr - (rng.exponential(1.0, size=(reps, t)) - 1.0)
    est = x.mean(axis=1) / x.std(axis=1, ddof=1)
    se = math.sqrt((1 - (-2) * sr + (9 - 1) / 4 * sr**2) / (t - 1))
    assert abs(est.std() / se - 1) < 0.03
    assert est.std() / math.sqrt((1 + sr**2 / 2) / (t - 1)) > 1.07
    # and the PSR at the true SR is a fair coin
    assert probabilistic_sharpe_ratio(sr, sr, t, -2.0, 9.0) == 0.5


def test_sharpe_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="impossible"):
        probabilistic_sharpe_ratio(0.1, 0.0, 100, skewness=2.0, kurtosis=3.0)  # needs kurtosis >= 5
    with pytest.raises(ValueError, match="n_trials"):
        expected_maximum_sharpe_ratio(1, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        expected_maximum_sharpe_ratio(10, -1.0)
    with pytest.raises(ValueError, match="n_obs"):
        probabilistic_sharpe_ratio(0.1, 0.0, 1)
    with pytest.raises(ValueError, match="finite"):
        deflated_sharpe_ratio(float("nan"), 100, 10, 0.1)
    with pytest.raises(ValueError, match="periods_per_year"):
        deflated_sharpe_ratio(1.0, 100, 10, 0.1, periods_per_year=0)


# ---------------------------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------------------------

def reference_cscv(m: np.ndarray, s: int, metric: str) -> list[float]:
    """Algorithm 2.3 [PBO] written out directly: join blocks, compute the statistic, rank, logit."""
    t, n = m.shape
    blocks = np.split(m, s)
    out = []
    for c in combinations(range(s), s // 2):
        j = np.vstack([blocks[i] for i in c])
        jbar = np.vstack([blocks[i] for i in range(s) if i not in c])

        def stat(a: np.ndarray) -> np.ndarray:
            return a.mean(axis=0) if metric == "mean" else a.mean(axis=0) / a.std(axis=0, ddof=1)

        best = int(np.argmax(stat(j)))
        w = rankdata(stat(jbar))[best] / (n + 1)
        out.append(math.log(w / (1 - w)))
    return out


def test_pbo_ranking_convention_follows_the_paper() -> None:
    # [PBO] section 2.1: R = (0.5, 1.1, 0.7) has ranks (1, 3, 2); R-bar = (0.6, 0.7, 1.3) has (1, 2, 3).
    assert rankdata([0.5, 1.1, 0.7]).tolist() == [1, 3, 2]
    assert rankdata([0.6, 0.7, 1.3]).tolist() == [1, 2, 3]


def test_pbo_hand_example_every_is_winner_loses_oos() -> None:
    # Four one-row blocks, three trials, statistic = mean. For each of the 6 IS pairs the IS best is the
    # OOS worst (worked by hand: e.g. IS {0, 1} sums (4, 3, 5) -> trial 2; OOS {2, 3} sums (3, 4, 2) ->
    # trial 2 ranks 1), so every relative rank is 1/4, every logit ln(1/3), and PBO = 1.
    m = np.array([[3, 1, 2], [1, 2, 3], [2, 3, 1], [1, 1, 1]], dtype=float)
    r = probability_of_backtest_overfitting(m, n_partitions=4, metric="mean")
    assert r.n_combinations == 6
    assert r.pbo == 1.0
    assert np.allclose(r.logits, math.log(1 / 3), atol=1e-15)


def test_pbo_hand_example_persistent_winner() -> None:
    # One trial is best in every block: it is chosen IS and ranks N = 3 OOS, w = 3/4, logit ln 3, PBO 0.
    m = np.array([[0, 1, 5], [1, 0, 6], [2, 1, 7], [0, 2, 8]], dtype=float)
    r = probability_of_backtest_overfitting(m, n_partitions=4, metric="mean")
    assert r.pbo == 0.0 and np.allclose(r.logits, math.log(3), atol=1e-15)
    assert r.prob_oos_loss == 0.0


@pytest.mark.parametrize("seed, metric, s", [(1, "sharpe", 8), (2, "mean", 8), (3, "sharpe", 10), (4, "sharpe", 4)])
def test_pbo_matches_direct_cscv(seed: int, metric: str, s: int) -> None:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((s * 12, 9)) + rng.normal(0, 0.05, size=9)
    r = probability_of_backtest_overfitting(m, n_partitions=s, metric=metric)  # type: ignore[arg-type]
    ref = reference_cscv(m, s, metric)
    assert np.allclose(r.logits, ref, rtol=0, atol=1e-10)
    assert r.pbo == np.mean(np.array(ref) <= 0)
    assert r.n_combinations == math.comb(s, s // 2)


def test_pbo_combination_counts() -> None:
    assert [math.comb(s, s // 2) for s in (4, 12, 24)] == [6, 924, 2_704_156]  # [PBO] eq. (2.3), section 4
    rng = np.random.default_rng(0)
    assert probability_of_backtest_overfitting(rng.standard_normal((48, 5)), 12).n_combinations == 924
    assert probability_of_backtest_overfitting(rng.standard_normal((64, 5)), 16).n_combinations == 12_870


def test_pbo_noise_is_near_one_half_and_skill_is_near_zero() -> None:
    # With no skill the IS winner's OOS rank is uniform, so PBO averages 1/2 (N even); one trial with a
    # real edge is chosen IS and stays above the OOS median.
    pbos = [probability_of_backtest_overfitting(np.random.default_rng(s).standard_normal((160, 20)), 8).pbo
            for s in range(60)]
    assert abs(np.mean(pbos) - 0.5) < 0.08
    m = np.random.default_rng(1).standard_normal((400, 20))
    m[:, 7] += 0.3
    r = probability_of_backtest_overfitting(m, 8)
    assert r.pbo == 0.0 and r.prob_oos_loss == 0.0


def test_pbo_inputs_are_validated() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(rng.standard_normal((30, 4)), 3)
    with pytest.raises(ValueError, match="multiple"):
        probability_of_backtest_overfitting(rng.standard_normal((30, 4)), 4)
    with pytest.raises(ValueError, match="two trials"):
        probability_of_backtest_overfitting(rng.standard_normal((32, 1)), 4)
    with pytest.raises(ValueError, match="finite"):
        probability_of_backtest_overfitting([[1.0, float("nan")], [0.0, 1.0], [1.0, 2.0], [3.0, 1.0]], 2)
    flat = rng.standard_normal((32, 3))
    flat[:, 1] = 0.01
    with pytest.raises(ValueError, match="zero variance"):
        probability_of_backtest_overfitting(flat, 4)


# ---------------------------------------------------------------------------------------------
# Purged and embargoed cross-validation
# ---------------------------------------------------------------------------------------------

def reference_split(test: list[int], start: np.ndarray, end: np.ndarray, embargo: int) -> tuple[set, set, set]:
    """Per-observation definitions: purge j if [start_j, end_j] meets any test label of its block; after a
    block, embargo the `embargo` positions that follow the first position whose label meets none."""
    n = len(start)
    test_set = set(test)
    blocks, cur = [], [test[0]]
    for p in test[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            blocks.append(cur)
            cur = [p]
    blocks.append(cur)

    def meets(j: int, block: list[int]) -> bool:
        return any(start[j] <= end[i] and end[j] >= start[i] for i in block)

    purged, embargoed = set(), set()
    for block in blocks:
        purged |= {j for j in range(n) if j not in test_set and meets(j, block)}
        j = block[-1] + 1
        while j < n and meets(j, block):
            j += 1
        embargoed |= set(range(j, min(n, j + embargo)))
    embargoed -= test_set | purged
    train = set(range(n)) - test_set - purged - embargoed
    return train, purged, embargoed


def test_purged_kfold_hand_example() -> None:
    # Labels [i, i + 2], 10 observations, 5 folds. Fold 2 tests positions 2-3, whose labels span [2, 5]:
    # positions 0, 1 (labels end at 2, 3) and 4, 5 (start at 4, 5) are purged; with embargo 1, position 6
    # is embargoed; training is 7-9.
    s = purged_kfold_splits(list(range(10)), [i + 2 for i in range(10)], 5, embargo=1)[1]
    assert (s.test, s.purged, s.embargoed, s.train) == ([2, 3], [0, 1, 4, 5], [6], [7, 8, 9])


def test_purged_kfold_without_overlap_is_plain_kfold() -> None:
    pos = list(range(23))
    for s, fold in zip(purged_kfold_splits(pos, pos, 4), np.array_split(np.arange(23), 4), strict=True):
        assert s.test == fold.tolist() and s.purged == [] and s.embargoed == []
        assert s.train == sorted(set(pos) - set(s.test))


@pytest.mark.parametrize("seed, embargo", [(0, 0), (1, 3), (2, 7)])
def test_purged_kfold_matches_definition(seed: int, embargo: int) -> None:
    rng = np.random.default_rng(seed)
    start = np.sort(rng.integers(0, 150, size=200))  # ties allowed
    end = start + rng.integers(0, 11, size=200)  # label horizons 0 to 10
    for s in purged_kfold_splits(start, end, 5, embargo=embargo):
        train, purged, embargoed = reference_split(s.test, start, end, embargo)
        assert (set(s.train), set(s.purged), set(s.embargoed)) == (train, purged, embargoed)


def test_purged_kfold_accepts_datetimes() -> None:
    start = np.arange("2026-01-01", "2026-01-21", dtype="datetime64[D]")
    s = purged_kfold_splits(start, start + np.timedelta64(2, "D"), 4, embargo=1)[1]  # test 5-9
    assert s.purged == [3, 4, 10, 11] and s.embargoed == [12]


@pytest.mark.parametrize("n_groups, k, splits, paths", [(6, 2, 15, 5), (10, 8, 45, 36), (9, 7, 36, 28)])
def test_cpcv_counts_and_paths(n_groups: int, k: int, splits: int, paths: int) -> None:
    pos = list(range(n_groups * 5))
    c = combinatorial_purged_splits(pos, pos, n_groups, k)
    assert len(c.splits) == splits == math.comb(n_groups, k)
    assert len(c.paths) == paths == k * math.comb(n_groups, k) // n_groups
    used = set()
    for path in c.paths:
        assert len(path) == n_groups
        for g, split in enumerate(path):
            assert g in c.test_groups[split]  # the split really tests that group
            used.add((split, g))
    assert len(used) == splits * k  # every (split, tested group) pair is used exactly once


def test_cpcv_matches_definition() -> None:
    rng = np.random.default_rng(5)
    start = np.sort(rng.integers(0, 300, size=240))
    end = start + rng.integers(0, 9, size=240)
    c = combinatorial_purged_splits(start, end, 6, 2, embargo=4)
    for s in c.splits:
        train, purged, embargoed = reference_split(s.test, start, end, 4)
        assert (set(s.train), set(s.purged), set(s.embargoed)) == (train, purged, embargoed)


def test_cv_inputs_are_validated() -> None:
    pos = list(range(10))
    with pytest.raises(ValueError, match="sorted"):
        purged_kfold_splits([1, 0, 2], [1, 0, 2], 2)
    with pytest.raises(ValueError, match="end at or after"):
        purged_kfold_splits([0, 1, 2], [0, 0, 2], 2)
    with pytest.raises(ValueError, match="n_splits"):
        purged_kfold_splits(pos, pos, 11)
    with pytest.raises(ValueError, match="embargo"):
        purged_kfold_splits(pos, pos, 2, embargo=-1)
    with pytest.raises(ValueError, match="no training"):
        purged_kfold_splits(pos, [9] * 10, 2)  # every label runs to the end
    with pytest.raises(ValueError, match="smaller"):
        combinatorial_purged_splits(pos, pos, 4, 4)
    with pytest.raises(ValueError, match="finite"):
        purged_kfold_splits([0.0, float("nan")], [0.0, 1.0], 2)
