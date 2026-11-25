"""
Backtest overfitting: leakage-free cross-validation and selection-adjusted performance statistics.

1. Purged and embargoed K-fold cross-validation (López de Prado 2018, *Advances in Financial Machine
   Learning*, ch. 7). Each observation carries a label interval [start, end] (e.g. a triple-barrier
   label runs from its entry bar to its touch bar). A training observation whose label interval
   intersects a test block's labels is purged; the ``embargo`` observations that follow the purged
   region after a test block are dropped as well (serial correlation leaks forward).
2. Combinatorial purged cross-validation (AFML ch. 12): N contiguous groups, every choice of k test
   groups is a split (C(N, k) splits), and the test predictions combine into C(N-1, k-1) backtest paths,
   each covering every group once.
3. The Probabilistic Sharpe Ratio (PSR) and the Deflated Sharpe Ratio (DSR): the probability that the
   true Sharpe ratio exceeds a benchmark given the sample length, skewness and kurtosis, and, for the
   DSR, with the benchmark set to the Sharpe ratio expected from the best of N unskilled trials. Formula
   as printed in Bailey, Borwein, López de Prado, Salehipour and Zhu (2015), "Online tools for
   demonstration of backtest overfitting", eq. (2)-(3); the expected maximum from López de Prado and
   Bailey (2021), "The False Strategy Theorem", *American Mathematical Monthly* 128(9), eq. (1); the
   worked example (N = 100, T = 1250, DSR 0.9004) from López de Prado (2015), "Advances in quantitative
   meta-strategies" (slides), which reproduces Bailey and López de Prado (2014), *Journal of Portfolio
   Management* 40(5).
4. The probability of backtest overfitting (PBO) by combinatorially symmetric cross-validation (CSCV):
   Bailey, Borwein, López de Prado and Zhu (2017), "The probability of backtest overfitting", *Journal
   of Computational Finance* 20(4), Algorithm 2.3 and section 3.

Units: Sharpe ratios are per observation period unless ``periods_per_year`` is given (then annualised
inputs are divided by sqrt(periods_per_year), variances by periods_per_year). Kurtosis is raw (3 for a
normal distribution), not excess.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import norm, rankdata

EULER_GAMMA = 0.5772156649015329
_MAX_COMBINATIONS = 200_000


# --------------------------------------------------------------------------------------------------
# Purged and embargoed cross-validation
# --------------------------------------------------------------------------------------------------

class CVSplit(BaseModel):
    """One split: positions (0-based, in the input order) for training and testing."""

    train: list[int]
    test: list[int]
    purged: list[int] = Field(default_factory=list, description="dropped: label overlaps a test label")
    embargoed: list[int] = Field(default_factory=list, description="dropped: within the embargo after a test block")


class CombinatorialSplits(BaseModel):
    splits: list[CVSplit]
    groups: list[list[int]] = Field(..., description="positions in each of the N contiguous groups")
    test_groups: list[list[int]] = Field(..., description="the k group ids tested in each split")
    paths: list[list[int]] = Field(
        ..., description="paths[p][g] = index of the split whose test predictions fill group g in path p")


def _label_arrays(label_start: Sequence[Any], label_end: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    start, end = np.asarray(label_start), np.asarray(label_end)
    if start.ndim != 1 or end.ndim != 1 or len(start) != len(end):
        raise ValueError("label_start and label_end must be 1-D sequences of equal length.")
    if len(start) == 0:
        raise ValueError("no observations.")
    if start.dtype.kind in "fc" or end.dtype.kind in "fc":
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            raise ValueError("label times must be finite.")
    elif start.dtype.kind == "M" or end.dtype.kind == "M":
        if np.any(np.isnat(start)) or np.any(np.isnat(end)):
            raise ValueError("label times must not be NaT.")
    elif start.dtype.kind not in "iu" or end.dtype.kind not in "iu":
        raise ValueError("label times must be numbers or numpy datetime64 values.")
    if np.any(end < start):
        raise ValueError("every label must end at or after its start (label_end >= label_start).")
    if np.any(start[1:] < start[:-1]):
        raise ValueError("observations must be sorted by label_start.")
    return start, end


def _split(test_positions: np.ndarray, start: np.ndarray, end: np.ndarray, embargo: int) -> CVSplit:
    n = len(start)
    in_test = np.zeros(n, dtype=bool)
    in_test[test_positions] = True
    purged = np.zeros(n, dtype=bool)
    embargoed = np.zeros(n, dtype=bool)
    # contiguous blocks of test positions
    breaks = np.flatnonzero(np.diff(test_positions) > 1)
    blocks = np.split(test_positions, breaks + 1)
    for block in blocks:
        lo, hi = start[block[0]], end[block].max()
        # intervals [start_j, end_j] and [lo, hi] intersect (inclusive); with sorted starts and a contiguous
        # block this equals intersecting at least one test label
        purged |= (~in_test) & (start <= hi) & (end >= lo)
        first_after = int(np.searchsorted(start, hi, side="right"))  # first start > hi
        stop = min(n, first_after + embargo)
        embargoed[first_after:stop] = True
    embargoed &= ~in_test & ~purged
    train = ~in_test & ~purged & ~embargoed
    return CVSplit(train=np.flatnonzero(train).tolist(), test=test_positions.tolist(),
                   purged=np.flatnonzero(purged).tolist(), embargoed=np.flatnonzero(embargoed).tolist())


def _check_int(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def purged_kfold_splits(label_start: Sequence[Any], label_end: Sequence[Any], n_splits: int,
                        embargo: int = 0) -> list[CVSplit]:
    """Purged K-fold with embargo (AFML ch. 7): K contiguous test folds, in order.

    ``label_start``/``label_end``: each observation's label interval, inclusive, sorted by start (numbers
    or numpy datetime64). Training observations whose interval intersects the test fold's labels are
    purged; the ``embargo`` observations after the purged region that follows a test fold are dropped.
    Folds are ``np.array_split`` of the positions (sizes differ by at most one). Raises if a split has no
    training observations left.
    """
    start, end = _label_arrays(label_start, label_end)
    n_splits = _check_int(n_splits, "n_splits", 2)
    embargo = _check_int(embargo, "embargo", 0)
    if n_splits > len(start):
        raise ValueError(f"n_splits ({n_splits}) exceeds the number of observations ({len(start)}).")
    out = [_split(fold, start, end, embargo) for fold in np.array_split(np.arange(len(start)), n_splits)]
    for i, s in enumerate(out):
        if not s.train:
            raise ValueError(f"purging and embargo leave no training observations in split {i}.")
    return out


def combinatorial_purged_splits(label_start: Sequence[Any], label_end: Sequence[Any], n_groups: int,
                                n_test_groups: int, embargo: int = 0) -> CombinatorialSplits:
    """Combinatorial purged cross-validation (AFML ch. 12).

    The positions are cut into ``n_groups`` contiguous groups (``np.array_split``); every combination of
    ``n_test_groups`` groups is a test set (C(N, k) splits, in ``itertools.combinations`` order), purged
    and embargoed after each contiguous run of test groups as in ``purged_kfold_splits``. Each group is
    tested in C(N-1, k-1) splits; path p takes, for every group, the p-th split (in split order) that
    tests it, so there are C(N-1, k-1) = k/N * C(N, k) paths.
    """
    start, end = _label_arrays(label_start, label_end)
    n_groups = _check_int(n_groups, "n_groups", 2)
    n_test_groups = _check_int(n_test_groups, "n_test_groups", 1)
    embargo = _check_int(embargo, "embargo", 0)
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be smaller than n_groups.")
    if n_groups > len(start):
        raise ValueError(f"n_groups ({n_groups}) exceeds the number of observations ({len(start)}).")
    if math.comb(n_groups, n_test_groups) > _MAX_COMBINATIONS:
        raise ValueError(f"C({n_groups}, {n_test_groups}) splits exceed {_MAX_COMBINATIONS:,}.")
    groups = np.array_split(np.arange(len(start)), n_groups)
    test_groups = [list(c) for c in combinations(range(n_groups), n_test_groups)]
    splits = []
    for i, tg in enumerate(test_groups):
        s = _split(np.concatenate([groups[g] for g in tg]), start, end, embargo)
        if not s.train:
            raise ValueError(f"purging and embargo leave no training observations in split {i}.")
        splits.append(s)
    n_paths = math.comb(n_groups - 1, n_test_groups - 1)
    tested_in = [[i for i, tg in enumerate(test_groups) if g in tg] for g in range(n_groups)]
    paths = [[tested_in[g][p] for g in range(n_groups)] for p in range(n_paths)]
    return CombinatorialSplits(splits=splits, groups=[g.tolist() for g in groups], test_groups=test_groups,
                               paths=paths)


# --------------------------------------------------------------------------------------------------
# Probabilistic and Deflated Sharpe ratios
# --------------------------------------------------------------------------------------------------

def _finite(x: float, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float | np.floating | np.integer) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number.")
    return float(x)


def _sr_variance_factor(sr: float, skewness: float, kurtosis: float) -> float:
    if kurtosis < 1.0 + skewness**2 - 1e-12:
        raise ValueError(f"kurtosis {kurtosis} is impossible with skewness {skewness} "
                         "(raw kurtosis >= 1 + skewness^2; a normal distribution has 3).")
    v = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if v <= 0.0:
        raise ValueError("1 - skewness*SR + (kurtosis-1)/4*SR^2 is not positive for these inputs.")
    return v


def probabilistic_sharpe_ratio(sharpe_ratio: float, benchmark_sharpe_ratio: float, n_obs: int,
                               skewness: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR: the probability that the true Sharpe ratio exceeds the benchmark.

    Z[(SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)], with SR and SR* per observation
    period (not annualised), T observations, g3 the skewness and g4 the raw kurtosis of the returns.
    """
    sr = _finite(sharpe_ratio, "sharpe_ratio")
    sr_star = _finite(benchmark_sharpe_ratio, "benchmark_sharpe_ratio")
    n = _check_int(n_obs, "n_obs", 2)
    v = _sr_variance_factor(sr, _finite(skewness, "skewness"), _finite(kurtosis, "kurtosis"))
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(v)))


def expected_maximum_sharpe_ratio(n_trials: float, sharpe_ratio_variance: float) -> float:
    """Expected maximum of N independent Gaussian Sharpe ratios with mean 0 and the given variance.

    sqrt(V) * [(1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e))], gamma the Euler-Mascheroni constant
    (False Strategy Theorem, eq. 1; an approximation for large N: 0.9% above the exact expectation at
    N = 100, 0.4% at N = 1000, 2.3% at N = 10). Units are those of the variance's Sharpe ratios.
    """
    n = _finite(n_trials, "n_trials")
    v = _finite(sharpe_ratio_variance, "sharpe_ratio_variance")
    if n < 2:
        raise ValueError("n_trials must be at least 2 (the approximation is for many trials).")
    if v < 0:
        raise ValueError("sharpe_ratio_variance must be non-negative.")
    return float(math.sqrt(v) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                 + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e))))


class DeflatedSharpeResult(BaseModel):
    deflated_sharpe_ratio: float = Field(..., description="probability the true SR exceeds the expected maximum of N unskilled trials")
    expected_max_sharpe_ratio: float = Field(..., description="SR0, per observation period")
    sharpe_ratio: float = Field(..., description="the tested SR, per observation period")
    z_statistic: float
    n_obs: int
    n_trials: float
    skewness: float
    kurtosis: float


def deflated_sharpe_ratio(sharpe_ratio: float, n_obs: int, n_trials: float, sharpe_ratio_variance: float,
                          skewness: float = 0.0, kurtosis: float = 3.0,
                          periods_per_year: float | None = None) -> DeflatedSharpeResult:
    """DSR = PSR(SR0), SR0 the expected maximum Sharpe ratio of ``n_trials`` independent unskilled trials.

    ``sharpe_ratio`` is the selected strategy's; ``sharpe_ratio_variance`` the variance of the Sharpe
    ratios across all trials. Both are per observation period, or annualised if ``periods_per_year`` is
    given (then divided by sqrt(q) and q). ``n_trials`` is the number of independent trials (correlated
    trials count for fewer). ``n_obs``, ``skewness`` and raw ``kurtosis`` describe the selected
    strategy's returns.
    """
    sr = _finite(sharpe_ratio, "sharpe_ratio")
    v = _finite(sharpe_ratio_variance, "sharpe_ratio_variance")
    if periods_per_year is not None:
        q = _finite(periods_per_year, "periods_per_year")
        if q <= 0:
            raise ValueError("periods_per_year must be positive.")
        sr, v = sr / math.sqrt(q), v / q
    sr0 = expected_maximum_sharpe_ratio(n_trials, v)
    n = _check_int(n_obs, "n_obs", 2)
    g3, g4 = _finite(skewness, "skewness"), _finite(kurtosis, "kurtosis")
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(_sr_variance_factor(sr, g3, g4))
    return DeflatedSharpeResult(deflated_sharpe_ratio=float(norm.cdf(z)), expected_max_sharpe_ratio=sr0,
                                sharpe_ratio=sr, z_statistic=z, n_obs=n, n_trials=float(n_trials),
                                skewness=g3, kurtosis=g4)


# --------------------------------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# --------------------------------------------------------------------------------------------------

class BacktestOverfittingResult(BaseModel):
    pbo: float = Field(..., description="share of combinations whose logit is <= 0: the IS-best strategy at or below the OOS median")
    n_combinations: int
    n_partitions: int
    n_strategies: int
    prob_oos_loss: float = Field(..., description="share of combinations where the IS-best strategy's OOS statistic is < 0")
    degradation_slope: float | None = Field(
        None, description="OLS slope of the IS-best strategy's OOS statistic on its IS statistic (None if IS values do not vary)")
    degradation_intercept: float | None = None
    logits: list[float]
    is_statistic: list[float] = Field(..., description="per combination: the IS-best strategy's IS statistic")
    oos_statistic: list[float] = Field(..., description="per combination: the same strategy's OOS statistic")
    metric: str


def probability_of_backtest_overfitting(performance: Sequence[Sequence[float]], n_partitions: int = 16,
                                        metric: Literal["sharpe", "mean"] = "sharpe") -> BacktestOverfittingResult:
    """PBO by CSCV (Bailey, Borwein, López de Prado and Zhu 2017, Algorithm 2.3 and section 3).

    ``performance``: T rows (synchronous periods) by N columns (the trials), e.g. daily returns of each
    configuration tried. Rows are cut into S = ``n_partitions`` equal blocks (T must be a multiple of S);
    each of the C(S, S/2) combinations of S/2 blocks is the in-sample set and the rest the out-of-sample
    set. The statistic (Sharpe ratio mean/std with ddof 1, per period, or the mean) is computed for every
    column on both; n* is the IS best (lowest column index on a tie); its OOS rank r (1 = worst, N = best;
    ties averaged) gives w = r/(N + 1) and the logit ln(w/(1 - w)). PBO is the share of logits <= 0.
    """
    m = np.asarray(performance, dtype=float)
    if m.ndim != 2:
        raise ValueError("performance must be a T x N matrix (rows: periods, columns: trials).")
    t, n = m.shape
    if n < 2:
        raise ValueError("at least two trials (columns) are needed.")
    if not np.all(np.isfinite(m)):
        raise ValueError("performance must be finite.")
    s = _check_int(n_partitions, "n_partitions", 2)
    if s % 2:
        raise ValueError("n_partitions must be even.")
    if t % s:
        raise ValueError(f"T = {t} rows is not a multiple of n_partitions = {s}; drop rows first.")
    if t // s * (s // 2) < 2:
        raise ValueError("each half must have at least 2 rows.")
    if math.comb(s, s // 2) > _MAX_COMBINATIONS:
        raise ValueError(f"C({s}, {s // 2}) combinations exceed {_MAX_COMBINATIONS:,}.")
    if metric not in ("sharpe", "mean"):
        raise ValueError("metric must be 'sharpe' or 'mean'.")
    rows = t // s
    blocks = m.reshape(s, rows, n)
    # per-block means and centred sums of squares, combined exactly (no sum-of-squares cancellation)
    b_mean = blocks.mean(axis=1)  # S x N
    b_m2 = ((blocks - b_mean[:, None, :]) ** 2).sum(axis=1)
    rows_half = t // 2

    def stat(idx: list[int]) -> np.ndarray:
        mean: np.ndarray = b_mean[idx].mean(axis=0)
        if metric == "mean":
            return mean
        m2 = b_m2[idx].sum(axis=0) + rows * ((b_mean[idx] - mean) ** 2).sum(axis=0)
        sd = np.sqrt(m2 / (rows_half - 1))
        if np.any(sd <= 1e-12 * np.abs(mean)) or np.any(sd == 0):
            raise ValueError("a trial has zero variance in a half-sample; its Sharpe ratio is undefined.")
        ratio: np.ndarray = mean / sd
        return ratio

    logits, is_best, oos_best = [], [], []
    for c in combinations(range(s), s // 2):
        ins = list(c)
        oos = [b for b in range(s) if b not in c]
        r_is, r_oos = stat(ins), stat(oos)
        k = int(np.argmax(r_is))
        w = float(rankdata(r_oos, method="average")[k]) / (n + 1)
        logits.append(math.log(w / (1 - w)))
        is_best.append(float(r_is[k]))
        oos_best.append(float(r_oos[k]))
    lg, xi, yo = np.array(logits), np.array(is_best), np.array(oos_best)
    slope = intercept = None
    if np.ptp(xi) > 0:
        slope_f, intercept_f = np.polyfit(xi, yo, 1)
        slope, intercept = float(slope_f), float(intercept_f)
    return BacktestOverfittingResult(
        pbo=float(np.mean(lg <= 0)), n_combinations=len(lg), n_partitions=s, n_strategies=n,
        prob_oos_loss=float(np.mean(yo < 0)), degradation_slope=slope, degradation_intercept=intercept,
        logits=lg.tolist(), is_statistic=xi.tolist(), oos_statistic=yo.tolist(), metric=metric)
