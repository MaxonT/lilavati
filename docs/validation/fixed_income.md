# Validation: fixed income engine (`core/fixed_income.py`)

Audited 2026-11-19. Tests: `tests/reference/test_fixed_income_reference.py` (44 tests). The
tolerances below are the worst errors measured in this session over the test cases. They are not
bounds derived for all inputs. QuantLib 1.44 is a test-only dependency.

## `analyze_bond`

Conventions: annual decimal rates; `ytm` compounded `frequency` times a year; valuation on a
coupon date (no accrued interest, so clean = dirty; maturity rounded to whole periods); price and
DV01 in `face_value` units; durations in years; convexity annualised (years²); DV01 =
D_mod · P · 1e-4, positive.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Price, 6 bonds (annual / semiannual / quarterly / monthly; par, premium, discount, zero) | QuantLib `BondFunctions.cleanPrice`, `Compounded`, 30/360, settle on issue date (accrued = 0) | 2.8e-15 rel | pass |
| Macaulay and modified duration, same 6 bonds | QuantLib `BondFunctions.duration` (Macaulay, Modified) | 1.4e-15 rel | pass |
| Convexity, same 6 bonds | QuantLib `BondFunctions.convexity` (annualised) | 2.8e-15 rel | pass |
| DV01 vs central finite difference of QuantLib prices at ±1bp | QuantLib prices | 1.1e-6 rel (O(h²) difference-scheme error, 30y bond) | pass |
| DV01 vs QuantLib `basisPointValue` | QL BPV = −D·P·h + ½·C·P·h², the signed second-order estimate of P(y+1bp) − P(y) | 1.8e-15 rel after removing the ½·C·P·h² term | pass (definitions differ; documented) |
| Zero-coupon closed forms: D_mac = T, D_mod = T/(1+y/f), C = n(n+1)/(f²(1+y/f)²) | Fabozzi, *Bond Markets, Analysis and Strategies* | 1.4e-16 rel | pass |
| Par bond prices at 100 for f = 1, 2, 4, 12; 300-year par bond D_mac → (1+y)/y | textbook identities | 9.0e-15 rel; 9.2e-6 years | pass |
| Price falls and is convex in yield (80-point grid, 0.1% to 20%) | property | strict | pass |
| Second-order Taylor (duration + convexity) for a 50bp shock | property | 8.8e-5 of price | pass |
| 0 < D_mod < D_mac < maturity; price and DV01 scale linearly with `face_value` | property | 1.5e-16 rel | pass |
| Field descriptions said "per 100 par" for price and DV01 | n/a | n/a | **fixed** (docs): values are in `face_value` units |
| Off-cycle maturity (e.g. 5.2y semiannual is priced as 5.0y) | n/a | n/a | **open**: rounded silently; no stub/accrued model |
| Maturity below half a period (`n_periods = 0`) | n/a | n/a | **open**: raises `ZeroDivisionError`, not a clear `ValueError` |

## `fit_nelson_siegel`

Conventions: maturities in years; zero-coupon yields as decimals; `lambda_decay` is τ in years
(QuantLib κ = 1/τ); unweighted least squares with bounds β0 ∈ [−0.1, 0.5], β1, β2 ∈ [−0.2, 0.2],
τ ∈ [0.1, 10].

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Model curve at fixed parameters equals QuantLib zero rates (4 curve shapes, 8 maturities) | QuantLib `NelsonSiegelFitting` via `FittedBondDiscountCurve`, continuous compounding; Nelson & Siegel (1987) eq. (2) | 1.2e-15 abs | pass |
| Recover (β0, β1, β2, τ) from noise-free QuantLib curves (upward, inverted, humped, dip; 10 maturities 3m to 30y) | QuantLib-generated data | β0 7.1e-10, β1 1.2e-9, β2 9.8e-9 abs; τ 4.7e-7 rel; R² ≥ 1 − 2.2e-15 | pass |
| Limits y(0+) = β0 + β1, y(∞) = β0 | Nelson & Siegel (1987) | 1.5e-8 abs at m = 1e-9 (engine floors m/τ at 1e-6); 2.0e-7 abs at m = 1e5 | pass |
| Curvature loading peak with Diebold-Li λ = 0.0609/month | Diebold & Li (2006), §2.1: peak at ~30 months | peak measured at 29.45 months | pass |
| Fit with 2bp Gaussian noise (seed 7); same data in reverse order | property | R² = 0.99943; β0 error 9.8e-5; order changes β by ≤ 4.8e-7 | pass |
| Percent inputs (4.5 for 4.5%) | n/a | raises `ValueError` ("initial guess outside bounds") | pass (documented) |
| β1 start value y[0] − y[−1] outside ±0.2, or long rate above 0.5 | n/a | n/a | **open**: scipy raises an opaque `ValueError` |

## `solve_merton_model`

Conventions: E and D in the same units; σ annualised decimal; r continuously compounded; T in
years (debt maturity = default horizon); distance to default = risk-neutral d2; default
probability = N(−d2) = Q(V_T < D); spread = −ln((V − E)/D)/T − r in bps, continuous, floored
at 0.

| Check | Reference | Tolerance achieved | Status |
| :--- | :--- | :--- | :--- |
| Worked example E=3, σE=80%, D=10, r=5%, T=1 → V=12.40, σV=0.2123, d2=1.1408, PD=12.7%, debt=9.40, expected loss 1.2% | Hull, *Options, Futures, and Other Derivatives*, Example 24.3 | engine: V=12.3954, σV=0.212305, d2=1.140826, PD=0.126971, debt=9.3954, EL=1.229%; all within Hull's rounding | pass |
| Equity re-priced as a call on assets, and Δ·σV·V = σE·E (5 cases) | QuantLib `BlackCalculator` | 5.9e-15 rel | pass |
| PD = N(−distance_to_default) | definition | exact (same formula) | pass |
| Deep-distress / long-horizon inputs satisfy both Merton equations (5 cases) | Merton (1974) equations, Hull eqs. 24.3-24.4 | residuals ≤ 7.3e-13 rel | **fixed** (see below) |
| Grid of 1,728 inputs (E 0.1 to 500, σE 5% to 300%, D 1 to 1000, r 0 to 10%, T 0.25 to 10y) | Merton equations | before: 41 cases with residual > 1e-6, worst 0.90; after: 0 failures, worst 3.6e-12 | **fixed** |
| Negligible debt: V → E + D·e^{−rT}, σV → σE·E/V, PD → 0, spread → 0 | limiting case | exact (0 rel); PD 1.6e-264 | pass |
| PD rises with leverage and with equity volatility | property | strict | pass |
| Non-positive E, σE, D or T | n/a | `ValueError` | **fixed**: previously D=0 or T=0 raised `ZeroDivisionError`, and E=0 or σE=0 returned output that looked valid (e.g. E=0 gave PD=1.0, σV=1.8e-12) |

**Bug fixed: Merton solver returned non-solutions without warning.** The 2-D `scipy.optimize.root`
(`hybr`) call ignored `sol.success`. For distressed or long-horizon inputs it stopped at points that
do not satisfy the equations, and the engine reported them as results. Example: E=0.1, σE=200%,
D=100, r=3%, T=5:

| | V | σV | d2 | PD | spread (bps) | equity-equation residual |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 0.7806 | 1.0800 | −3.1549 | 0.999197 | 9,680.0 | 10.2% |
| after | 0.2149 | 1.6303 | −3.4667 | 0.999737 | 13,238.2 | 0 |

For E=0.1, σE=300%, D=100, r=3%, T=10 the old solver returned V=5.366, σV=0.416 (residual 90%);
the correct root is V=0.10000, σV=2.99994. The engine now reduces the system to one equation in
σV, with V solved from the equity equation inside (E, E + D·e^{−rT}]. That equation always changes
sign on [σE·E/(E + D·e^{−rT}), σE], so Brent's method on that bracket converges. The engine then
checks both residuals (relative 1e-8) and raises `ValueError` if either fails. In-repo inputs
(examples, benchmark case, tests) give the same V, σV, d2, PD and spread to 6 decimal places
before and after, so no quoted numbers changed.

**Open.** The spread uses the model's debt value V − E, which is consistent with the model but is not
a market spread. Multiple roots were not found on the 1,728-case grid (200-point σV scan); this is
not a proof of uniqueness.
