# Validation: financial ML engine (`core/financial_ml.py`)

Audited 2026-11-19. Tests: `tests/reference/test_financial_ml_reference.py` (146 tests, about 5 s).
The tolerances below are the worst errors measured in this session over the test cases. They are
not bounds derived for all inputs. References: López de Prado (2018), *Advances in Financial
Machine Learning* (AFML), ch. 3 and 5; Hosking (1981), "Fractional differencing", *Biometrika*
68(1). The test file has its own implementations: closed-form binomial weights with
`np.convolve` for FFD, a pandas transcription of AFML snippets 3.2-3.4 for the barriers, and a
direct statsmodels ADF grid search for d*. All random series use fixed seeds. Purged
cross-validation and the backtest-overfitting statistics are in `core/backtest_overfitting.py`
([below](#backtest-statistics-corebacktest_overfittingpy), added 2026-11-26).

## `get_fractional_weights`

Conventions: w_0 = 1, w_k = −w_{k−1}(d − k + 1)/k, the coefficients of (1 − B)^d. Generation stops
at the first |w_k| < `threshold` (that weight is dropped) or at `size` weights. The array is
ordered oldest lag first, so `result[-1] = 1` multiplies the current value.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| 60 weights for 11 values of d (−0.3 to 2.5) equal (−1)^k·binom(d, k) | Hosking (1981); AFML eq. in §5.4.2 | 2.1e-13 rel | pass |
| Hand recursion for d = 0.5: 1, −0.5, −0.125, −0.0625, −0.0390625 | AFML §5.4.2 recursion | exact | pass |
| Truncation at the first \|w_k\| < τ, 5 values of d × τ ∈ {1e-2, 1e-3, 1e-4, 1e-5} (windows up to 4,076 weights) | AFML snippet 5.3 (`getWeights_FFD`) | same window length in every case; 1.1e-11 rel on the weights (recursion round-off over thousands of terms) | pass |
| d = 0 → [1]; d = 1 → [−1, 1]; d = 2 → [1, −2, 1] | (1 − B)^d for integer d | exact | pass |
| Weight order matches the dot-product direction | hand FFD example (see below) | exact | pass |
| NaN/inf `d`, `size` < 1, negative or NaN `threshold` | n/a | `ValueError` | **fixed**: previously NaN `d` returned NaN weights, and `size = 0` returned one weight |

## `fractionally_differentiate`

Conventions: fixed-width-window fractional differencing (FFD, AFML §5.5) of the input in its own
units (AFML applies it to log prices; callers take logs themselves). With `width` weights, the
output has n − width + 1 points and `output[j]` is aligned with `series[width − 1 + j]`. The
window is capped at n, so on short series a small d gives a single output point. Default
τ = 1e-4 (AFML snippet 5.3 uses 1e-5 by default and 1e-2 in snippet 5.4). Widths at τ = 1e-4:
503 points for d = 0.1, 200 for d = 0.5, 38 for d = 0.9.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Hand example x = (1, 2, 4, 8, 16), d = 0.5, τ = 0.1 → (2.875, 5.75, 11.5) | hand calculation with w = (1, −0.5, −0.125) | exact | pass |
| d = 0 is the identity, d = 1 equals `np.diff`, d = 2 equals `np.diff(n=2)` (200-point walk) | definition | 0 abs | pass |
| 3 seeded random walks (400 points) × 7 values of d (0.1 to 1.4) × τ ∈ {1e-2, 1e-4} | independent `np.convolve` with closed-form weights (AFML snippet 5.3) | 1.8e-13 abs, 3.2e-11 rel; output lengths equal | pass |
| Window longer than the series (n = 30, d = 0.2) gives one point | same reference | ≤ 1e-12 rel (asserted) | pass |
| NaN/inf values or an empty series | n/a | `ValueError` | **fixed**: previously one NaN made every output point NaN (all 101 of 101 for d = 0.5, n = 300) with no error; AFML forward-fills instead |

## `find_optimal_fractional_d`

Conventions: search grid {0, step, 2·step, …, 1}, with d = 1 added when `step` does not divide 1.
d* is the smallest grid value whose FFD series (τ = 1e-4) has ADF p-value ≤ `target_p_value`
(AFML §5.6, snippet 5.4). ADF is `EconometricEngine.test_stationarity` (statsmodels `adfuller`,
constant only, lag order by AIC). AFML snippet 5.4 uses `maxlag=1` without autolag, so p-values
are not comparable with the book's figures. Candidates whose FFD output has fewer than 25 points
are skipped. `correlation_with_original` is the Pearson correlation between the FFD series and
the input at the same time indices (AFML: `corr(df1.loc[df2.index], df2)`).

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| d*, p-value, correlation and FFD series on 5 seeded random walks (n = 300 to 1,000; step 0.05 / 0.1; target 0.01 / 0.05); every smaller grid d fails the target | independent statsmodels `adfuller(autolag="AIC")` search on the reference FFD | d* identical (0.05 to 0.6); p 1.2e-13 rel; correlation 4.0e-15 abs | pass |
| Correlation alignment: tail (same timestamps), not head | AFML snippet 5.4 | benchmark series: tail 0.3668 (reported); head alignment would give 0.0853 | pass |
| Benchmark series (`bench/runner.py`, `np.random.seed(42)`, drift 0.01, 300 points) | Track 4 thresholds p < 0.05, correlation ≥ 0.25 | d* = 0.55, p = 0.000899, correlation 0.3668 (unchanged by the audit) | pass |
| White noise | definition | d* = 0, series unchanged, correlation 1 | pass |
| d = 0 when the input meets a target above 5% (AR(1) φ = 0.93, ADF p = 0.0597, target 0.10) | definition of the search | before: d* = 0.55; after: d* = 0 | **fixed**: the d = 0 test also required `is_stationary` (p < 0.05) |
| Grid stays inside [0, 1] (step 0.3, seed 4, target 1e-20) | definition of the search | before: tested {0.3, 0.6, 0.9, 1.2} and returned d* = 1.2; after: {0.3, 0.6, 0.9, 1.0}, d* = 1.0 | **fixed** (`np.arange` overshoot) |
| Reported d* is the grid value (step 0.0625) | definition | before: 0.562; after: 0.5625 | **fixed** (rounding to 3 decimals) |
| No grid d meets the target (quadratic trend + noise; d = 1 gives p = 0.94) | solver-output rule | before: returned d* = 1.0 with adf_p_value = 1.0 (never computed), correlation 1.0 and the undifferenced input; after: `ValueError` naming the smallest p-value tested | **fixed** |
| Series too short for any testable d (n = 20) | solver-output rule | before: same made-up d* = 1.0, p = 1.0; after: `ValueError` | **fixed** |
| Short series (n = 40): only d = 1 leaves ≥ 25 points | documented behaviour | d* = 1.0, p matches the reference to 1e-9 rel | pass (documented) |
| NaN input; step ≤ 0, > 1 or NaN; target outside (0, 1) | n/a | `ValueError` | **fixed**: previously NaN leaked statsmodels `MissingDataError`, step 0 raised `ZeroDivisionError`, step 2.0 returned d* = 2.0, and step −0.1 or target 0 / NaN returned the made-up d* = 1.0, p = 1.0 |
| Small d* can come from sample truncation, not differencing (seed 4, n = 1,000: d* = 0.05; the undifferenced last 639 points already give p = 0.0045) | n/a | n/a | **open**: the d > 0 candidates are tested on the tail left after the FFD window, while d = 0 uses the full series; AFML has the same windowing |

## `apply_triple_barriers`

Conventions: fixed barriers in decimal simple returns (0.03 = 3%) from the entry close:
r_j = (P_j − P_i)/P_i ≥ `take_profit_pct` → +1; r_j ≤ −`stop_loss_pct` → −1; neither within
`holding_horizon` bars → 0 at bar i + h (AFML §3.4; snippet 3.5 instead labels a vertical touch
by the sign of its return). Bars i + 1 to i + h are scanned (the entry bar is not tested). Touches
are inclusive with a 1e-12 return tolerance (AFML uses strict inequalities). Both barriers are
positive and tested on one close per bar, so a bar cannot touch both. Entries without h bars after
them are dropped (AFML `getBins` drops events without `t1`). `math.inf` disables the upper
barrier; `stop_loss_pct` ≥ 1 disables the lower one. AFML's volatility-scaled barriers
(multiples of a daily volatility target per event), side and meta-labelling are not implemented;
pass k·σ as the percentage instead.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Hand example, 8 prices, h = 3, ±3%: labels (+1, −1, +1, −1, 0), touch bars, returns; 3 trailing entries dropped | hand calculation; AFML snippets 3.2-3.4 transcription | labels and touches exact; returns ≤ 1e-12 rel (asserted) | pass |
| 4 seeded geometric random walks (400 bars, 1% vol) × 4 settings (h = 1 to 20): 6,256 events (2,499 × −1, 1,576 × 0, 2,181 × +1) | pandas transcription of AFML snippet 3.2 (`applyPtSlOnT1`) with vertical barrier h bars ahead | labels and touch indices identical; returns 1.1e-16 abs | pass |
| Take-profit on the vertical-barrier bar gives +1, not 0 | AFML: `close[loc:t1]` is inclusive | exact | pass |
| First touch wins (stop at bar 1, take-profit at bar 2 → −1) | AFML snippet 3.3 (earliest of the touches) | exact | pass |
| End of series: entries without a full window dropped; h ≥ n gives no labels | AFML snippet 3.4 / `getBins` | exact | pass |
| Move exactly equal to a barrier: 100 → 110 at +10%, 3.0 → 2.1 at −30% | inclusive convention | before: both labelled 0 (`100·1.1 = 110.00000000000001`, `3·0.7 = 2.0999999999999996`), the first with return_at_touch = 0.1; after: +1 and −1 | **fixed** (compare returns with tolerance) |
| h ≤ 0 or non-integer; tp or sl ≤ 0 or NaN; NaN, zero, negative or no prices | n/a | `ValueError` | **fixed**: previously h = 0 returned labels touching the entry bar, h = −1 raised `IndexError`, a NaN price was skipped, a zero price gave an infinite return, negative prices gave inverted labels (+1 with return −10%), and tp = 0 labelled flat moves +1 with return 0 |

**Fixes and quoted numbers.** On every in-repo input (benchmark track 4, `tests/`,
`examples/07_stat_arb_and_execution_agent.py`, `scripts/benchmark_latency.py` series) the engine
returns the same d*, p-value and correlation before and after the audit (benchmark: d* = 0.55,
p = 0.000899, correlation 0.3668; latency series: d* = 0.3; example 07: d* = 0.90), so no quoted
numbers changed. The latency in `docs/papers/01_framework_paper.md` was not re-measured; the
algorithm is unchanged apart from input checks.

**Open.** (1) No volatility-target, side or meta-labelling inputs for the barriers (AFML §3.5-3.6).
(2) The d* search tests d > 0 on a shorter sample than d = 0 (see the table). (3) Purged and
embargoed cross-validation (AFML ch. 7) is a separate task.

## Backtest statistics (`core/backtest_overfitting.py`)

Added 2026-11-26. Tests: `tests/reference/test_backtest_overfitting_reference.py`. Status:
reference-checked. References: the Deflated Sharpe Ratio formula as printed in Bailey, Borwein,
López de Prado, Salehipour and Zhu (2015), "Online tools for demonstration of backtest overfitting",
eq. (2)-(3); the expected maximum Sharpe ratio from López de Prado and Bailey (2021), "The false
strategy theorem", *American Mathematical Monthly* 128(9), eq. (1); the worked example of Bailey and
López de Prado (2014), *Journal of Portfolio Management* 40(5), as presented in López de Prado (2015),
"Advances in quantitative meta-strategies" (slides 11-14); and Bailey, Borwein, López de Prado and Zhu
(2017), "The probability of backtest overfitting", *Journal of Computational Finance* 20(4),
Algorithm 2.3 and section 3. The 2014 paper itself was not read; its example was.

Conventions: Sharpe ratios per observation period, or annualised with `periods_per_year` (divided by
√q, variances by q); kurtosis raw (3 for normal); PSR = Z[(SR − SR\*)√(T − 1)/√(1 − γ₃SR + (γ₄ − 1)/4·SR²)];
DSR = PSR at SR₀ = √V·[(1 − γ)Z⁻¹(1 − 1/N) + γZ⁻¹(1 − 1/(Ne))], γ the Euler-Mascheroni constant.
PBO: T rows cut into S equal blocks (T must be a multiple of S, S even); for each of the C(S, S/2)
in-sample halves, the in-sample best trial n\* (lowest index on a tie) gets its out-of-sample rank r
(1 = worst; ties averaged), ω = r/(N + 1), logit ln(ω/(1 − ω)); PBO is the share of logits ≤ 0 (the
paper's φ = ∫ f(λ) dλ from −∞ to 0). The statistic is the Sharpe ratio (mean/std, ddof 1) or the mean.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Worked example: daily, annualised SR 2.5, N = 100, V = 1/2, T = 1,250, skewness −3, kurtosis 10 → SR₀ ≈ 0.1132, DSR ≈ 0.9004 | slides 13 | SR₀ 0.113172, DSR 0.900397 (printed to four decimals) | pass |
| N = 46 → 0.9505; skewness 0, kurtosis 3, N = 88 → 0.9505 | slides 14 | 0.950502; 0.950491 | pass |
| Normal returns fall below 0.95 confidence after 89 trials | false strategy theorem, section 3 | N = 88: 0.950491; N = 89: 0.949839 | pass |
| Expected maximum against the exact E[max of N standard normals] (numerical integration) | theorem eq. (1), an approximation for large N | +2.33% (N = 10), +0.92% (N = 100), +0.42% (N = 1,000) | pass (approximation error stated in the docstring) |
| PSR denominator is the sampling variance of the estimated SR for skewed returns (skewness −2, kurtosis 9, SR 0.1, T = 500, 20,000 samples) | eq. (2) denominator | Monte Carlo spread / formula = 1.011; the normal formula is 11% too narrow | pass |
| DSR equals the PSR at SR₀; DSR falls as N grows | definition | 1e-15 | pass |
| Kurtosis below 1 + skewness², N < 2, T < 2, negative variance, non-finite input | n/a | `ValueError` | pass |
| Ranking convention: R = (0.5, 1.1, 0.7) ranks (1, 3, 2) | PBO paper, section 2.1 | exact | pass |
| Hand example, 4 one-row blocks, 3 trials: every in-sample winner is the out-of-sample loser → all logits ln(1/3), PBO 1; a persistent winner → ln 3, PBO 0 | Algorithm 2.3 by hand | 1e-15 | pass |
| Logits equal a direct transcription of Algorithm 2.3 (4 random matrices; Sharpe and mean; S = 4, 8, 10) | Algorithm 2.3 | 1e-10 abs | pass |
| Combinations C(S, S/2): S = 4 → 6, 12 → 924, 24 → 2,704,156 | eq. (2.3), section 4 | exact. The paper prints 12,780 for S = 16; C(16, 8) = 12,870 | pass |
| Pure noise: mean PBO over 60 seeds (N = 20, T = 160, S = 8) is near 1/2; one trial with a real edge gives PBO 0 | section 3.1 | 0.474 (asserted within 0.08 of 0.5); 0 | pass |
| A trial with zero variance in a half-sample | n/a | `ValueError` | **fixed during development**: variance from sums of squares left 1e-19 instead of 0 and passed the check; blocks are now combined from centred moments |

Open: the number of independent trials N is the caller's estimate; correlated trials count for
fewer, and a wrong N makes the DSR too strict or too lenient
([Tutorial 09](../tutorials/09_is_this_backtest_real.md), section 6). Not implemented: estimating the
effective number of trials, the minimum track record length, stochastic dominance.

## Purged and embargoed cross-validation (`core/backtest_overfitting.py`)

Added 2026-11-26. Status: **not reference-checked**. The rules follow López de Prado (2018), AFML
ch. 7 (purging, embargo) and ch. 12 (combinatorial purged CV), but the book's code was not available
in this session, so the tests compare with an independent per-observation implementation of the
definitions and with hand-worked cases.

Conventions: observations sorted by label start; label intervals inclusive (numbers or datetime64);
test folds are `np.array_split` of the positions. A training observation is purged if its interval
meets a test block's labels; after each contiguous test block, the `embargo` observations following
the first observation that meets no test label are dropped. No embargo before a test block. A split
with no training observations left raises. Combinatorial splits: every choice of k of N groups, in
`itertools.combinations` order; path p takes, for each group, the p-th split that tests it, giving
C(N − 1, k − 1) paths.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Labels [i, i + 2], 10 observations, 5 folds, embargo 1: fold 2 tests 2-3, purges 0, 1, 4, 5, embargoes 6, trains on 7-9 | hand calculation | exact | pass |
| Labels with no span: plain contiguous K-fold | definition | exact | pass |
| Random labels (200 observations, ties, horizons 0 to 10; embargo 0, 3, 7) and combinatorial splits (240 observations, 6 groups, 2 tested, embargo 4) | independent per-observation implementation of purging and embargo | identical sets | pass |
| Split and path counts: (6, 2) → 15 and 5; (10, 8) → 45 and 36; (9, 7) → 36 and 28; every path covers each group once, every (split, tested group) pair used once | C(N, k) and k/N·C(N, k) | exact | pass |
| datetime64 labels | definition | exact | pass |
| Unsorted starts, a label ending before it starts, too many folds, negative embargo, no training left | n/a | `ValueError` | pass |

Open: compare with AFML snippets 7.1-7.3 and with a reference implementation (skfolio's
`CombinatorialPurgedCV` would need a test-only dependency; check its licence first), then mark reference-checked.
