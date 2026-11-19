# Validation: econometrics engine (`core/econometrics.py`)

Audited 2026-11-19. Tests: `tests/reference/test_econometrics_reference.py` (47 tests, about 6 s under load).
The tolerances below are the worst errors measured in this session over the test cases. They are
not bounds derived for all inputs. Libraries: statsmodels 0.15.0, arch 9.0.0, NumPy 2.4.6. All
simulations use fixed seeds.

## `test_stationarity` (ADF)

Conventions: test regression with a constant and no trend (`regression="c"`); maximum lag
min(⌈12·(n/100)^¼⌉, n/2 − 2) (Schwert 1989 rule, statsmodels default); lag chosen by AIC on a
common sample, then re-estimated on all usable observations; p-value from MacKinnon (1994)
(asymptotic); critical values from MacKinnon (2010) at T = `n_observations`;
`is_stationary` = p-value < `significance_level`.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Statistic, p-value, lag, nobs, critical values; 5 series (AR(1) φ = 0.5 and 0.95, random walk with and without drift, white noise) | statsmodels `adfuller(maxlag=None, regression="c", autolag="AIC")` called directly | exact (same call) | pass |
| Statistic and chosen lag | NumPy re-implementation of the ADF regression and AIC lag search (Said & Dickey 1984; Hamilton 1994 §17.7) | 5.7e-15 rel; lags identical | pass |
| Statistic, chosen lag, p-value | arch `unitroot.ADF(trend="c", method="aic")` (separate implementation) | statistic 5.7e-15 rel; lags identical; p-value 4.7e-3 abs (arch uses its own p-value surface) | pass |
| Critical values at 1%, 5%, 10% | MacKinnon (2010) Table 2, τ_c, N = 1, response surface in 1/T | 4.4e-16 abs | pass (coefficients as tabulated in statsmodels and arch; the paper PDF could not be fetched in this session) |
| AR(1) φ = 0.5, n = 500, 30 seeds: unit root rejected | known answer | 30/30 rejected; largest p-value 5.2e-9 | pass |
| Random walk, n = 250, 150 seeds: rejection rate at 5% | nominal size; binomial(150, 0.05) 99.9% range [1, 18] | 8/150 = 5.3% | pass |
| Constant series, too short (< 20) | n/a | `ValueError` | pass (unchanged) |
| NaN or inf in the series | n/a | `ValueError` | **fixed**: NaN raised statsmodels `MissingDataError`; inf raised `LinAlgError` ("SVD did not converge") |
| `significance_level` outside (0, 1) | n/a | `ValueError` | **fixed**: 2.0 marked a random walk (p = 0.27) as stationary; 0 or 1 were accepted |

## `test_cointegration` (Engle-Granger and half-life)

Conventions: OLS `series_a = alpha + beta·series_b + spread` (`hedge_ratio` = beta, units of
series_b per unit of series_a); Engle-Granger via statsmodels `coint(trend="c", method="aeg",
autolag="aic")`: ADF without deterministic terms on the OLS residuals, MacKinnon p-value with
N = 2. The test is not symmetric in the two series. `spread_std` uses ddof = 0. Half-life: OLS
Δs_t = c + b·s_{t−1} + e_t, φ = 1 + b (returned as `spread_ar1_coefficient`), half-life =
−ln 2 / ln|φ| sampling periods; for an OU process sampled once per period φ = e^{−θ}, so this is
ln 2 / θ. |φ| ≥ 1 gives `inf`; for −1 < φ ≤ 0 it is the half-life of the deviation's magnitude.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| p-value, 3 seeds of y = 3 + 2x + AR(1) | statsmodels `coint` called directly | exact | pass |
| Hedge ratio and intercept | closed-form simple regression cov(x, y)/var(x); arch `engle_granger` cointegrating vector | 1.3e-15 rel | pass |
| Engle-Granger statistic | arch `engle_granger(trend="c", method="aic")` | 0 rel (identical) | pass |
| Critical values at 1%, 5%, 10% | MacKinnon (2010) Table 2, τ_c, N = 2, T = n − 1 | 0 abs | pass (same caveat as above) |
| y = 3 + 2x + AR(1) (φ = 0.7), n = 300, 30 seeds: cointegrated | known answer | 30/30; worst p = 6.9e-4; \|β − 2\| max 0.13, median 0.029 | pass |
| Two independent random walks, n = 250, 150 seeds: false rejection at 5% | nominal size; range [1, 18] | 9/150 = 6.0% | pass |
| Half-life equals −ln 2 / ln φ̂ for φ̂ from an independent OLS of the spread | AR(1) mapping (e.g. Tsay, *Analysis of Financial Time Series*) | 2.2e-15 rel | **fixed** (see below) |
| OU spread with θ = 0.1, 0.5, 1.0 (φ = e^{−θ}), n = 4,000 | true half-life ln 2 / θ = 6.93, 1.39, 0.69 | 6.71, 1.33, 0.67: within 0.96 delta-method standard errors | **fixed** |
| White-noise spread (φ ≈ 0) | limit: half-life → 0 | 0.16 periods (φ̂ = 0.014) | **fixed**: was 0.68 |
| Perfectly collinear pair (y = 2x + 1) | statsmodels warns the test is unreliable | `ValueError` | **fixed**: returned p = 0, "cointegrated", half-life 1.3e33 or `inf`, "trading viable" |
| Constant series_a or series_b | n/a | `ValueError` | **fixed**: returned a meaningless hedge ratio and half-life (e.g. 1.4e32) |
| NaN, too short (< 30), length mismatch, `significance_level` outside (0, 1) | n/a | `ValueError` | **fixed** for NaN (`LinAlgError`) and significance; others unchanged |

**Bug fixed: half-life used the first-order shortcut.** The engine reported ln 2 / (−b) =
ln 2 / (1 − φ). That is the continuous-time approximation (as in Chan, *Algorithmic Trading*,
2013) and equals the AR(1) half-life only as φ → 1. It overstates the half-life for faster
reversion: by 12% at φ = 0.8, 27% at φ = e^{−0.5}, 58% at φ = e^{−1}. On the OU test cases
(n = 4,000) the engine returned 7.07, 1.71 and 1.08 periods against true 6.93, 1.39 and 0.69.
The engine now uses −ln 2 / ln|φ|. Quoted numbers refreshed by re-running:

| Output | before | after | true value of the simulated spread |
| :--- | ---: | ---: | ---: |
| `examples/02_statistical_arbitrage_pairs.py`, OU half-life | 3.3 days | 2.9 days | 3.11 (φ = 0.8) |
| `examples/07_stat_arb_and_execution_agent.py` and tutorial 02, OU half-life | 2.74 days | 2.38 days | 2.41 (φ = 0.75) |

## `fit_garch`

Conventions: constant mean, Gaussian GARCH(p, q) fitted by arch. Order convention of Bollerslev
(1986): `p` = number of GARCH (β) lags, `q` = number of ARCH (α) lags; arch uses the opposite
letters, so the engine calls `arch_model(p=q, q=p)`. Inputs are per-period decimal returns
(0.01 = 1%), assumed daily. `rescale=True` fits 100·returns; ω is divided by 100² and volatility
by 100, so ω and `annualized_volatility` are in decimal units. `annualized_volatility` =
σ_T·√252, where σ_T is the in-sample conditional volatility of the last observation (not a T+1
forecast). AIC/BIC are those of the scaled fit.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| ω, α, β, AIC, BIC, annualised vol; GARCH(1,1), n = 3,000 | arch `arch_model(100·r, mean="Constant", vol="GARCH", p=1, q=1)` called directly | 0 rel | pass |
| Annualised vol = √(252·σ²_T) from the variance recursion with the fitted parameters | Bollerslev (1986) eq. (2), NumPy recursion | 0 rel (start-up value has died out) | pass |
| Recover ω = 2e-6, α = 0.08, β = 0.90 from a NumPy-simulated GARCH(1,1), n = 20,000 | known parameters | ω 1.68e-6, α 0.0740, β 0.9084: within 1.7 standard errors; implied unconditional vol 15.5% vs 15.9% | pass |
| GARCH(p = 1, q = 2) with α = (0.05, 0.10), β = 0.80, n = 20,000; also via the MCP tool | known parameters; arch with `p=2, q=1` | α (0.0504, 0.1036), β 0.7964: within 0.41 standard errors | **fixed** (see below) |
| ARCH(1) as GARCH(p = 0, q = 1), α = 0.4, n = 5,000 | known parameter; arch `p=1, q=0` | α 0.4042 (0.17 standard errors) | **fixed**: raised `ValueError` from arch |
| Doubling the returns: ω × 4, vol × 2, α and β unchanged | units property | ω 9.0e-5 rel, vol 2.1e-6 rel, α/β 2.9e-6 abs (optimiser tolerance) | pass |
| Constant returns | n/a | `ValueError` | **fixed**: optimiser failed (code 4) and the engine returned its starting values α = 0.01, β = 0.49 |
| Optimiser non-convergence | n/a | `ValueError` with arch's code and message | **fixed**: `convergence_flag` was not checked |
| NaN, fewer than 100 points | n/a | `ValueError` | pass (now with the engine's own message) |
| `q = 0`, `p < 0`, non-integer orders | n/a | `ValueError` | **fixed**: `q = 0` fitted an ARCH(1) and returned `alpha_coeffs = []`, `beta_coeffs = [0.0]` |

**Bug fixed: GARCH orders swapped.** The docstring and fields follow Bollerslev (p = β lags, q =
α lags), but the engine passed `p` and `q` to arch unchanged, where `p` is the ARCH order. Missing
parameters were then read with `params.get(name, 0.0)`, so the error was silent. For the GARCH(1,2)
test case (true α = (0.05, 0.10), β = 0.80) the engine returned α = (0.1257, 0.0) and β = (0.8357):
arch had fitted one α and two βs, and the engine dropped β₂ and reported a zero α₂. GARCH(1,1) is
unaffected, so no quoted numbers changed (example 02 still prints persistence 0.000 and 16.5%).

## Open

* Percent inputs (returns of 1.0 meaning 1%) are not detected; with `rescale=True` they give
  annualised volatility 100 times too large (e.g. 15.66, i.e. 1,566%, for a 1% daily series).
* `annualized_volatility` assumes daily data (√252) and is the last in-sample σ_T, not a one-step
  forecast; example 02 previously labelled it a "forecast" (relabelled 2026-11-19).
* AIC/BIC depend on `rescale` (scaled fit), so they compare only fits with the same setting.
* ADF accepts n = 20, where the default maximum lag (8) leaves 11 observations in the regression.
* `spread_ar1_coefficient` is a new output field. The Engle-Granger residual-based φ̂ is biased
  in small samples; the half-life is a point estimate without a confidence interval.
