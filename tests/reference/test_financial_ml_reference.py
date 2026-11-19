"""Reference tests for core/financial_ml.py.

Sources:
- [AFML] López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley.
  Ch. 5: weight recursion w_k = -w_{k-1}(d - k + 1)/k (eq. in section 5.4.2, snippet 5.1),
  fixed-width window fractional differencing with |w_k| < tau truncation (section 5.5,
  snippet 5.3), minimum-d search with correlation to the original series (section 5.6,
  snippet 5.4). Ch. 3: triple-barrier method (section 3.4, snippets 3.2-3.5).
- Hosking, J. R. M. (1981), "Fractional differencing", Biometrika 68(1): the weights of (1 - B)^d
  are (-1)^k * binom(d, k).

The independent implementations below use a different algorithm from the engine (closed-form
binomial weights via scipy + np.convolve for FFD; a pandas transcription of AFML snippet 3.2 for
the barriers; a direct statsmodels ADF grid search for d*). All random series use fixed seeds.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import binom
from statsmodels.tsa.stattools import adfuller

from lilavati.core.financial_ml import FinancialMLEngine as F

# ---------------------------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------------------------


def _ref_weights(d: float, threshold: float, max_size: int) -> np.ndarray:
    """Hosking (1981) closed form w_k = (-1)^k binom(d, k), newest-first, truncated at the first
    |w_k| < threshold (AFML snippet 5.3) or max_size weights."""
    w = []
    for k in range(max_size):
        w_k = (-1.0) ** k * float(binom(d, k))
        if k > 0 and abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.asarray(w)


def _ref_ffd(x: np.ndarray, d: float, threshold: float = 1e-4) -> pd.Series:
    """FFD as a convolution, indexed by the time index of the current observation:
    y_t = sum_k w_k x_{t-k} for every t with a complete window (AFML snippet 5.3)."""
    w = _ref_weights(d, threshold, len(x))
    y = np.convolve(x, w, mode="valid")  # convolve flips w, so y[j] = sum_k w_k x[j+L-1-k]
    return pd.Series(y, index=np.arange(len(w) - 1, len(x)))


def _ref_adf_p(x: np.ndarray) -> float:
    # Same ADF specification as EconometricEngine.test_stationarity: constant, AIC lag order.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)  # statsmodels tuple-return notice
        return float(adfuller(x, autolag="AIC")[1])


def _ref_min_d(x: np.ndarray, grid: list[float], target: float) -> tuple[float, float, float]:
    """AFML snippet 5.4 search: smallest d with ADF p <= target; correlation with the original
    series at the same timestamps (df1.loc[df2.index] in AFML)."""
    if _ref_adf_p(x) <= target:
        return 0.0, _ref_adf_p(x), 1.0
    original = pd.Series(x)
    for d in grid:
        y = _ref_ffd(x, d)
        if len(y) < 25:
            continue
        p = _ref_adf_p(y.to_numpy())
        if p <= target:
            corr = float(np.corrcoef(original.loc[y.index], y)[0, 1])
            return d, p, corr
    raise AssertionError("reference search found no d")


def _ref_triple_barrier(
    prices: np.ndarray, h: int, pt: float, sl: float
) -> list[tuple[int, int, int, float]]:
    """Transcription of AFML snippets 3.2 (applyPtSlOnT1), 3.3 (earliest touch) and 3.4
    (vertical barrier h bars ahead; events without one are dropped, as getBins does) on an
    integer bar index. Label 0 at the vertical barrier (AFML section 3.4)."""
    close = pd.Series(prices)
    t1 = pd.Series(close.index[h:], index=close.index[: len(close) - h])
    out = []
    for loc, t in t1.items():
        df0 = close.loc[loc:t] / close.loc[loc] - 1  # label slice: entry bar through t1 inclusive
        sl_t = df0[df0 < -sl].index.min()
        pt_t = df0[df0 > pt].index.min()
        first = int(pd.Series([t, sl_t, pt_t], dtype=float).dropna().min())
        if not pd.isna(pt_t) and first == pt_t:
            label = 1
        elif not pd.isna(sl_t) and first == sl_t:
            label = -1
        else:
            label = 0
        out.append((int(loc), label, first, float(close.loc[first] / close.loc[loc] - 1)))
    return out


def _random_walk(seed: int, n: int = 300, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(drift, 1.0, n))


# ---------------------------------------------------------------------------------------------
# get_fractional_weights
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9, 0.99, 1.5, 2.5, -0.3])
def test_weights_match_binomial_closed_form(d: float) -> None:
    # Hosking (1981): (1 - B)^d = sum_k (-1)^k binom(d, k) B^k. Engine returns oldest first.
    w = F.get_fractional_weights(d, size=60, threshold=0.0)
    ref = np.array([(-1.0) ** k * binom(d, k) for k in range(60)])
    if float(d).is_integer():
        ref = ref[: int(d) + 1]
    np.testing.assert_allclose(w[::-1], ref[: len(w)], rtol=1e-12, atol=1e-300)
    assert w[-1] == 1.0  # w_0 multiplies the current observation


def test_weights_hand_values_d_half() -> None:
    # AFML eq. 5.4.2 recursion by hand for d = 0.5:
    # w1 = -0.5, w2 = -w1(0.5-1)/2 = -0.125, w3 = -w2(0.5-2)/3 = -0.0625,
    # w4 = -w3(0.5-3)/4 = -0.0390625
    w = F.get_fractional_weights(0.5, size=5, threshold=0.0)
    assert w.tolist() == [-0.0390625, -0.0625, -0.125, -0.5, 1.0]


@pytest.mark.parametrize("d", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("threshold", [1e-2, 1e-3, 1e-4, 1e-5])
def test_weights_threshold_truncation(d: float, threshold: float) -> None:
    # AFML snippet 5.3: keep weights until the first |w_k| < tau, which is dropped.
    w = F.get_fractional_weights(d, size=10**6, threshold=threshold)
    ref = _ref_weights(d, threshold, 10**6)
    assert len(w) == len(ref)
    assert np.all(np.abs(w) >= threshold)
    assert abs((-1.0) ** len(w) * binom(d, len(w))) < threshold
    np.testing.assert_allclose(w[::-1], ref, rtol=1e-10)


def test_weights_integer_orders() -> None:
    # (1 - B)^0 = 1, (1 - B)^1 = 1 - B, (1 - B)^2 = 1 - 2B + B^2
    assert F.get_fractional_weights(0.0, size=100).tolist() == [1.0]
    assert F.get_fractional_weights(1.0, size=100).tolist() == [-1.0, 1.0]
    assert F.get_fractional_weights(2.0, size=100).tolist() == [1.0, -2.0, 1.0]


def test_weights_respect_size() -> None:
    assert len(F.get_fractional_weights(0.5, size=7, threshold=0.0)) == 7
    assert len(F.get_fractional_weights(0.5, size=1, threshold=0.0)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d": math.nan, "size": 10},
        {"d": math.inf, "size": 10},
        {"d": 0.5, "size": 0},
        {"d": 0.5, "size": 10, "threshold": -1e-4},
        {"d": 0.5, "size": 10, "threshold": math.nan},
    ],
)
def test_weights_reject_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        F.get_fractional_weights(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# fractionally_differentiate
# ---------------------------------------------------------------------------------------------


def test_ffd_hand_example_orientation() -> None:
    # d = 0.5, tau = 0.1 keeps w = (1, -0.5, -0.125) (w3 = -0.0625 dropped).
    # y_2 = 4 - 0.5*2 - 0.125*1 = 2.875; y_3 = 8 - 0.5*4 - 0.125*2 = 5.75;
    # y_4 = 16 - 0.5*8 - 0.125*4 = 11.5. Reversed weights would give y_2 = 1 - 1 - 0.5 = -0.5.
    y = F.fractionally_differentiate([1.0, 2.0, 4.0, 8.0, 16.0], d=0.5, threshold=0.1)
    assert y.tolist() == [2.875, 5.75, 11.5]


def test_ffd_integer_orders() -> None:
    x = _random_walk(0, n=200)
    np.testing.assert_array_equal(F.fractionally_differentiate(x, d=0.0), x)  # identity
    np.testing.assert_allclose(F.fractionally_differentiate(x, d=1.0), np.diff(x), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        F.fractionally_differentiate(x, d=2.0), np.diff(x, n=2), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("d", [0.1, 0.35, 0.5, 0.65, 0.8, 1.0, 1.4])
@pytest.mark.parametrize("threshold", [1e-2, 1e-4])
def test_ffd_matches_independent_convolution(seed: int, d: float, threshold: float) -> None:
    x = _random_walk(seed, n=400, drift=0.05)
    y = F.fractionally_differentiate(x, d=d, threshold=threshold)
    ref = _ref_ffd(x, d, threshold)
    assert len(y) == len(ref)
    np.testing.assert_allclose(y, ref.to_numpy(), rtol=1e-11, atol=1e-10)


def test_ffd_window_longer_than_series_gives_single_point() -> None:
    # Weights are capped at len(series), so a long-memory d on a short series uses one window.
    x = _random_walk(3, n=30)
    y = F.fractionally_differentiate(x, d=0.2, threshold=1e-4)
    ref = _ref_ffd(x, 0.2, 1e-4)
    assert len(y) == 1 and len(ref) == 1
    assert y[0] == pytest.approx(ref.iloc[0], rel=1e-12)


@pytest.mark.parametrize("bad", [[1.0, math.nan, 3.0], [1.0, math.inf, 3.0], []])
def test_ffd_rejects_non_finite_or_empty(bad: list[float]) -> None:
    # Before the audit a single NaN silently turned every output point into NaN.
    with pytest.raises(ValueError):
        F.fractionally_differentiate(bad, d=0.5)


def test_ffd_rejects_non_finite_d() -> None:
    with pytest.raises(ValueError):
        F.fractionally_differentiate([1.0, 2.0, 3.0], d=math.nan)


# ---------------------------------------------------------------------------------------------
# find_optimal_fractional_d
# ---------------------------------------------------------------------------------------------


def _grid(step: float) -> list[float]:
    g = [round(k * step, 12) for k in range(1, math.floor(1 / step + 1e-9) + 1)]
    return g if g and g[-1] >= 1 - 1e-12 else [*g, 1.0]


@pytest.mark.parametrize(
    ("seed", "n", "drift", "step", "target"),
    [
        (0, 300, 0.0, 0.05, 0.01),
        (1, 300, 0.01, 0.05, 0.01),
        (2, 500, 0.0, 0.1, 0.05),
        (3, 300, 0.02, 0.05, 0.05),
        (4, 1000, 0.01, 0.05, 0.01),
    ],
)
def test_optimal_d_matches_independent_search(
    seed: int, n: int, drift: float, step: float, target: float
) -> None:
    x = _random_walk(seed, n=n, drift=drift)
    r = F.find_optimal_fractional_d(x.tolist(), step=step, target_p_value=target)
    d_ref, p_ref, corr_ref = _ref_min_d(x, _grid(step), target)
    assert r.optimal_d == d_ref
    assert r.adf_p_value == pytest.approx(p_ref, rel=1e-9)
    assert r.correlation_with_original == pytest.approx(corr_ref, abs=1e-12)
    np.testing.assert_allclose(r.differentiated_series, _ref_ffd(x, d_ref).to_numpy(), atol=1e-10)
    # Minimality on the grid: every smaller grid d fails the target (or is too short to test).
    for d in [g for g in _grid(step) if g < d_ref]:
        y = _ref_ffd(x, d)
        assert len(y) < 25 or _ref_adf_p(y.to_numpy()) > target


def test_optimal_d_correlation_uses_tail_alignment() -> None:
    # The FFD series ends at the last input point; aligning on the head would give a different
    # (wrong) number. Benchmark series of bench/runner.py (np.random.seed(42), drift 0.01).
    np.random.seed(42)
    x = np.cumsum(np.random.normal(0.01, 1.0, 300)) + 100.0
    r = F.find_optimal_fractional_d(x.tolist())
    y = np.asarray(r.differentiated_series)
    tail = float(np.corrcoef(x[-len(y) :], y)[0, 1])
    head = float(np.corrcoef(x[: len(y)], y)[0, 1])
    assert r.correlation_with_original == pytest.approx(tail, abs=1e-12)
    assert abs(head - tail) > 0.1
    # Benchmark track 4 expects memory_preserved (corr >= 0.25) and p < 0.05 on this series.
    assert r.optimal_d == 0.55
    assert r.adf_p_value < 0.05
    assert r.correlation_with_original >= 0.25


def test_optimal_d_reports_exact_grid_value() -> None:
    # Before the audit optimal_d was rounded to 3 decimals: step 0.0625 gave 0.562, not 0.5625.
    np.random.seed(42)
    x = np.cumsum(np.random.normal(0.01, 1.0, 300)) + 100.0
    r = F.find_optimal_fractional_d(x.tolist(), step=0.0625)
    d_ref, _, _ = _ref_min_d(x, _grid(0.0625), 0.01)
    assert d_ref == 0.5625
    assert r.optimal_d == 0.5625


def test_optimal_d_grid_never_exceeds_one() -> None:
    # step 0.3: the grid is {0.3, 0.6, 0.9, 1.0}. Before the audit np.arange gave
    # {0.3, 0.6, 0.9, 1.2} and returned d = 1.2 here. ADF p-values (statsmodels, seed 4):
    # d=0.9 -> 5.6e-17, d=1.0 -> 2.3e-30, so with target 1e-20 the answer on [0, 1] is 1.0.
    x = _random_walk(4)
    assert _ref_adf_p(_ref_ffd(x, 0.9).to_numpy()) > 1e-20
    assert _ref_adf_p(_ref_ffd(x, 1.0).to_numpy()) <= 1e-20
    r = F.find_optimal_fractional_d(x.tolist(), step=0.3, target_p_value=1e-20)
    assert r.optimal_d == 1.0
    assert r.adf_p_value == pytest.approx(_ref_adf_p(np.diff(x)), rel=1e-9)


def test_already_stationary_series_returns_zero() -> None:
    # White noise: d* = 0, the series is returned unchanged with correlation 1.
    x = np.random.default_rng(7).normal(0.0, 1.0, 300)
    r = F.find_optimal_fractional_d(x.tolist())
    assert r.optimal_d == 0.0
    assert r.adf_p_value == pytest.approx(_ref_adf_p(x), rel=1e-12)
    assert r.correlation_with_original == 1.0
    assert r.differentiated_series == x.tolist()


def test_already_stationary_at_loose_target_returns_zero() -> None:
    # AR(1), phi = 0.93, seed 0: ADF p = 0.0597, i.e. meets a 10% target but not 5%. Before the
    # audit d = 0 also required p < 0.05 (test_stationarity's own level), so the engine skipped
    # d = 0 and returned d = 0.55.
    rng = np.random.default_rng(0)
    e = rng.normal(0.0, 1.0, 200)
    x = np.zeros(200)
    for i in range(1, 200):
        x[i] = 0.93 * x[i - 1] + e[i]
    p0 = _ref_adf_p(x)
    assert 0.05 < p0 <= 0.1
    r = F.find_optimal_fractional_d(x.tolist(), target_p_value=0.1)
    assert r.optimal_d == 0.0
    assert r.adf_p_value == pytest.approx(p0, rel=1e-12)


def test_short_series_only_tests_d_with_enough_points() -> None:
    # n = 40: every d < 1 keeps a window wider than 15 points (< 25 outputs), so only d = 1 is
    # tested. The result is the minimum over testable d, as documented.
    x = _random_walk(1, n=40)
    r = F.find_optimal_fractional_d(x.tolist())
    assert r.optimal_d == 1.0
    assert len(r.differentiated_series) == 39
    assert r.adf_p_value == pytest.approx(_ref_adf_p(np.diff(x)), rel=1e-9)


def test_too_short_for_any_d_raises() -> None:
    # n = 20 passes the ADF minimum, but d = 1 leaves 19 < 25 points. Before the audit this
    # returned optimal_d = 1.0 with a made-up adf_p_value = 1.0 and the undifferenced series.
    with pytest.raises(ValueError, match="too short"):
        F.find_optimal_fractional_d(_random_walk(2, n=20).tolist())


def test_no_d_meets_target_raises() -> None:
    # Quadratic trend + noise: no d in [0, 1] reaches p <= 0.01 (d = 1 leaves a linear trend).
    # Before the audit: optimal_d = 1.0, adf_p_value = 1.0 (not a computed p-value),
    # correlation 1.0 and differentiated_series = the original input.
    t = np.arange(300.0)
    x = t**2 / 100 + np.random.default_rng(3).normal(0.0, 0.1, 300)
    assert _ref_adf_p(np.diff(x)) > 0.01
    with pytest.raises(ValueError, match="No d"):
        F.find_optimal_fractional_d(x.tolist())


@pytest.mark.parametrize(
    "kwargs",
    [{"step": 0.0}, {"step": -0.1}, {"step": 2.0}, {"step": math.nan},
     {"target_p_value": 0.0}, {"target_p_value": 1.0}, {"target_p_value": math.nan}],
)
def test_optimal_d_rejects_invalid_search_parameters(kwargs: dict[str, float]) -> None:
    # Before: step 0 raised ZeroDivisionError, step 2.0 returned d = 2.0, and step -0.1 or
    # target 0 / NaN returned d = 1.0 with adf_p_value = 1.0.
    with pytest.raises(ValueError):
        F.find_optimal_fractional_d(_random_walk(0).tolist(), **kwargs)


def test_optimal_d_rejects_nan() -> None:
    # Before: statsmodels MissingDataError (not a ValueError) leaked.
    x = _random_walk(0)
    x[150] = math.nan
    with pytest.raises(ValueError, match="NaN"):
        F.find_optimal_fractional_d(x.tolist())


# ---------------------------------------------------------------------------------------------
# apply_triple_barriers
# ---------------------------------------------------------------------------------------------


def _as_tuples(labels: list) -> list[tuple[int, int, int, float]]:
    return [(lb.index, lb.label, lb.touch_index, lb.return_at_touch) for lb in labels]


def test_triple_barrier_hand_example() -> None:
    # tp = sl = 3%, h = 3 bars. Hand-computed:
    # i=0 (100):   101 +1.0%, 99 -1.0%, 103.5 +3.5%          -> +1 at 3, r = 0.035
    # i=1 (101):   99 -1.98%, 103.5 +2.48%, 96 -4.95%        -> -1 at 4, r = -5/101
    # i=2 (99):    103.5 +4.55%                               -> +1 at 3, r = 4.5/99
    # i=3 (103.5): 96 -7.25%                                  -> -1 at 4, r = -7.5/103.5
    # i=4 (96):    97 +1.04%, 97.5 +1.56%, 98 +2.08%          ->  0 at 7, r = 2/96
    # i=5..7 have no complete vertical window and are dropped (AFML getBins drops them).
    prices = [100.0, 101.0, 99.0, 103.5, 96.0, 97.0, 97.5, 98.0]
    got = _as_tuples(F.apply_triple_barriers(prices, 3, 0.03, 0.03))
    expected = [
        (0, 1, 3, 0.035),
        (1, -1, 4, -5 / 101),
        (2, 1, 3, 4.5 / 99),
        (3, -1, 4, -7.5 / 103.5),
        (4, 0, 7, 2 / 96),
    ]
    assert [g[:3] for g in got] == [e[:3] for e in expected]
    for g, e in zip(got, expected, strict=True):
        assert g[3] == pytest.approx(e[3], rel=1e-12)
    ref = _ref_triple_barrier(np.array(prices), 3, 0.03, 0.03)
    assert [g[:3] for g in got] == [r[:3] for r in ref]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("h", "pt", "sl"), [(1, 0.005, 0.005), (5, 0.01, 0.01), (10, 0.02, 0.01), (20, 0.03, 0.05)]
)
def test_triple_barrier_matches_afml_transcription(seed: int, h: int, pt: float, sl: float) -> None:
    # Geometric random walk, 1% daily vol. On continuous data exact barrier hits have
    # probability ~0, so AFML's strict inequalities and the engine's inclusive ones agree.
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))
    got = _as_tuples(F.apply_triple_barriers(prices.tolist(), h, pt, sl))
    ref = _ref_triple_barrier(prices, h, pt, sl)
    assert len(got) == len(prices) - h
    assert [g[:3] for g in got] == [r[:3] for r in ref]
    np.testing.assert_allclose([g[3] for g in got], [r[3] for r in ref], rtol=1e-12, atol=1e-15)
    if h >= 5:
        assert {g[1] for g in got} == {-1, 0, 1}  # all three barriers exercised


@pytest.mark.parametrize(
    ("prices", "pt", "sl", "label"),
    [
        ([100.0, 110.0], 0.10, 0.10, 1),  # 100*(1+0.1) = 110.00000000000001 > 110
        ([0.7, 0.91], 0.30, 0.10, 1),
        ([3.0, 2.1], 0.10, 0.30, -1),  # 3*(1-0.3) = 2.0999999999999996 < 2.1
        ([100.0, 90.0], 0.10, 0.10, -1),
    ],
)
def test_triple_barrier_touch_exactly_at_barrier(
    prices: list[float], pt: float, sl: float, label: int
) -> None:
    # A move exactly equal to the barrier counts as a touch. Before the audit the first and third
    # cases were labelled 0 (vertical) because the barrier level was compared without tolerance.
    (lb,) = F.apply_triple_barriers(prices, 1, pt, sl)
    assert lb.label == label
    assert lb.touch_index == 1


def test_triple_barrier_touch_on_vertical_bar_is_horizontal() -> None:
    # The vertical barrier bar is inside the window (AFML close[loc:t1] is inclusive): a
    # take-profit on that bar gives +1, not 0.
    (lb,) = F.apply_triple_barriers([100.0, 101.0, 103.0], 2, 0.03, 0.03)
    assert (lb.label, lb.touch_index) == (1, 2)


def test_triple_barrier_end_of_series() -> None:
    # Entries without a full vertical window are dropped, even if a barrier is touched early.
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 200.0]
    labels = F.apply_triple_barriers(prices, 2, 0.05, 0.05)
    assert [lb.index for lb in labels] == [0, 1, 2, 3]
    assert (labels[-1].label, labels[-1].touch_index) == (1, 5)
    assert F.apply_triple_barriers(prices, 6, 0.05, 0.05) == []


def test_triple_barrier_first_touch_wins() -> None:
    # Stop-loss at bar 1, take-profit later at bar 2: the first touch decides.
    (lb,) = F.apply_triple_barriers([100.0, 94.0, 120.0], 2, 0.05, 0.05)
    assert (lb.label, lb.touch_index) == (-1, 1)
    assert lb.return_at_touch == pytest.approx(-0.06, rel=1e-12)


def test_triple_barrier_disabled_barriers() -> None:
    # math.inf disables the upper barrier; stop_loss_pct >= 1 disables the lower one.
    prices = [100.0, 150.0, 30.0, 10.0]
    assert [lb.label for lb in F.apply_triple_barriers(prices, 3, math.inf, 1.0)] == [0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"holding_horizon": 0},
        {"holding_horizon": -1},
        {"holding_horizon": 1.5},
        {"take_profit_pct": 0.0},
        {"take_profit_pct": -0.05},
        {"take_profit_pct": math.nan},
        {"stop_loss_pct": 0.0},
        {"stop_loss_pct": -0.05},
        {"prices": [100.0, math.nan, 102.0, 103.0]},
        {"prices": [100.0, 0.0, 102.0, 103.0]},
        {"prices": [-100.0, -90.0, -120.0, -100.0]},
        {"prices": []},
    ],
)
def test_triple_barrier_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    # Before: h = 0 produced labels touching the entry bar, h = -1 raised IndexError, a NaN
    # price was silently skipped, a zero price gave an infinite return, negative prices gave
    # inverted labels, and tp = 0 labelled flat moves +1 with return 0.
    args: dict[str, object] = {
        "prices": [100.0, 101.0, 102.0, 103.0],
        "holding_horizon": 2,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.05,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        F.apply_triple_barriers(**args)  # type: ignore[arg-type]
