# Validation: execution and portfolio (`core/execution.py`)

Almgren-Chriss is checked against the original paper. The order book, Black-Litterman and HRP are
not yet checked against a reference: their audits are open to contributors (#14, #15, #16). Status of
every engine: [validation status](status.md).

## Almgren-Chriss liquidation schedule

Reference: Almgren, R. and Chriss, N. (2000), "Optimal Execution of Portfolio Transactions",
*Journal of Risk* 3(2), 5–39 (equation numbers refer to the December 2000 version). Tests:
`tests/test_almgren_chriss.py`. The reference case is the paper's Section 3.4 / Table 1: X = 10⁶
shares, T = 5 days, N = 5, σ = 0.95 $/share/day^½ (0.3 annual on a $50 stock), γ = 2.5×10⁻⁷,
η = 2.5×10⁻⁶, ε = 0.0625, λ = 10⁻⁶.

| Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- |
| Optimal holdings, paper case and λ ∈ {10⁻⁸, 10⁻⁷, 2×10⁻⁶, 10⁻⁵} | independent least-squares solution of the paper's discrete problem (E + λV, eqs. 5 and 8), solved with NumPy without the closed form | rel 1e-9, abs 1e-6 shares | pass |
| κ for the paper case | Section 3.4: "κ ≈ 0.6/day, so κT ≈ 3" | ±0.02/day, ±0.1 | pass |
| Defining equation for κ | eq. 16: (2/τ²)(cosh κτ − 1) = λσ²/η̃ | rel 1e-12 | pass |
| Expected cost and variance of the optimal strategy | eq. 20 | rel 1e-10 | pass |
| λ = 0: linear schedule, its cost and variance | eqs. 10–11 | rel 1e-12 | pass |
| Small λ converges to TWAP | limit of eq. 17 | 1e-6 of the position | pass |
| More risk aversion front-loads the schedule | monotone in λ over eight values from 0 to 1 | strict | pass |
| Holdings start at X, end at 0, never increase; trades sum to X | boundary conditions, N ∈ {1, 2, 13, 390}, four λ | rel 1e-12 | pass |
| Non-convex case (η̃ = η − γτ/2 < 0) and invalid inputs | raise | n/a | pass |

Fixed in the audit (2026-11-19): a clipped sinh ratio made very risk-averse schedules hold every
share until the end (λ = 10⁶ returned the position unsold until the last interval); volatility was
used as a fraction where the paper needs price units; the continuous-time solution was replaced by
the discrete one with η̃ = η − γτ/2; a stray factor 0.5 in the variance was removed. `arrival_price`
became a required keyword so that the price-unit conversion cannot be skipped.

## Order-book snapshot (not reference-checked, #14)

Spread, mid price and microprice follow their definitions (microprice = (bid size × ask + ask size ×
bid) / (bid size + ask size)); a crossed or locked book and non-positive sizes raise. The order-flow
imbalance is simplified: the change in bid size minus the change in ask size, at unchanged prices
(Cont, Kukanov and Stoikov's measure also accounts for price moves). Tests: `tests/test_execution.py`
(definitions and crossed-book rejection only).

## Black-Litterman (not reference-checked, #15)

Posterior returns follow the formula in the docstring; weights maximise w'μ − ½δw'Σw, long-only and
summing to one, with CVXPY and the Clarabel solver ([libraries](libraries.md)). Checked so far: the
weights equal the closed-form budget-constrained optimum when no long-only bound binds (abs 1e-6); a
solver failure raises (it used to return equal weights, fixed 2026-11-25); invalid inputs raise. Not
yet checked: the posterior against a published worked example (He and Litterman 1999; Idzorek 2005).

## Hierarchical risk parity (not reference-checked, #16)

Single-linkage clustering on d = √(½(1 − ρ)), quasi-diagonal order, recursive bisection weighted by
inverse cluster variance, here approximated by the trace of the cluster's covariance. Two differences
from López de Prado (2016) are to be checked: the clustering input and the cluster variance (the paper
uses the inverse-variance portfolio's variance). Tests: `tests/test_execution.py` (weights sum to one,
lower-volatility assets receive more weight).
