# Validation: `lilavati.core.derivatives`

Reference checks for `DerivativesEngine` (task 06). The tests are in
`tests/reference/test_derivatives_reference.py`. Every number below was measured in this session with
QuantLib 1.44 (test-only dependency), Python 3.11, on macOS arm64. "Tolerance achieved" is the largest
difference seen across the test cases. The test asserts a looser bound, which is given in brackets.

Sources:

- **QL**: QuantLib 1.44, specifically `AnalyticEuropeanEngine`, `BinomialVanillaEngine("crr")`,
  `VanillaOption.impliedVolatility`, `sabrVolatility` and `blackFormula`.
- **Hull**: J. C. Hull, *Options, Futures, and Other Derivatives*, 10th ed. Examples are identified by
  their inputs because chapter numbers change between editions.
- **Hagan**: Hagan, Kumar, Lesniewski, Woodward, "Managing Smile Risk", *Wilmott Magazine* (2002),
  eq. (2.17a) (general strike) and eq. (2.18) (at the money).

## Units and conventions

| Quantity | Convention |
| :--- | :--- |
| `time_to_maturity` | Years. One calendar day is 1/365 (QL ACT/365F with whole-day maturities). |
| `risk_free_rate`, `dividend_yield` | Continuously compounded, annual, decimal (0.05 = 5%). |
| `volatility`, SABR/IV output | Annualised lognormal (Black) vol, decimal. |
| `delta`, `gamma` | Per 1 unit of spot (unscaled). |
| `vega` | Per 1 vol point: `0.01 * dV/dsigma` = QL `vega()/100`. S=K=100, r=5%, sigma=20%, T=1 gives **0.37524**. This is consistent with benchmark case LILAVATI-T5-001, which expects 0.375 at 5% tolerance. |
| `theta` | Per calendar day: `-(dV/dT)/365` = QL `thetaPerDay()` = QL `theta()/365`. |
| `rho` | Per 1 percentage point of r: `0.01 * dV/dr` = QL `rho()/100`. No dividend rho is returned. |
| `vanna` | `d(delta)/dsigma`, sigma in decimal, **no** 0.01 scaling (per 1.0 of vol). |
| `volga` | `d(vega)/dsigma` for the reported per-vol-point vega, i.e. `0.01 * d2V/dsigma2`. |
| American `greeks` | **European BSM Greeks** for the same inputs, not tree Greeks (see the open items). |

No Greek convention was changed. The benchmark values (call 10.4506, delta 0.6368, gamma 0.01876,
vega 0.3752) are unchanged.

## Results per public function

### `calculate_bsm_greeks` / `price_european_option`

| Check | Reference | Tolerance achieved [asserted] | Status |
| :--- | :--- | :--- | :--- |
| Price, delta, gamma, vega, theta, rho, calls and puts with q > 0, over 5 cases (ITM/ATM/OTM, 90 d to 2 y, S up to 2500) | QL AnalyticEuropeanEngine | max rel. 7.7e-15 (theta), all others ≤ 2.8e-15 [1e-10 to 1e-12] | pass |
| Vanna, volga | Central bump (h=1e-5) of QL delta and vega in the vol quote | max rel. 3.9e-9 (vanna), 6.5e-9 (volga) [1e-6] | pass |
| All 7 Greeks vs central finite differences of the engine's price | Analytical identity | max rel. 5.6e-7 (volga), 3.5e-7 (vanna), 1.3e-7 (delta), ≤ 5.2e-8 for the rest [1e-4 to 1e-5] | pass |
| Put-call parity `C - P = S e^{-qT} - K e^{-rT}`; `delta_C - delta_P = e^{-qT}`; `rho_C - rho_P = 0.01 K T e^{-rT}` | Hull (parity with yield) | [1e-10 S; 1e-12] | pass |
| S0=42, K=40, r=10%, sigma=20%, T=0.5 gives c=4.76, p=0.81 | Hull Example 15.6 | 4.7594, 0.8086 [±0.005] | pass |
| Index call S0=930, K=900, r=8%, q=3%, sigma=20%, T=2/12 gives c=51.83 | Hull Example 17.1 | 51.8330 [±0.005] | pass |
| S0=49, K=50, r=5%, sigma=20%, T=0.3846 gives delta 0.522, gamma 0.066, vega 12.1, theta −4.31/yr, rho 8.91 | Hull ch. 19 running example | 0.5216, 0.0655, 12.105, −4.3054/yr (−0.011796/day), 8.9066 | pass |
| Benchmark T5-001 vega convention | QL `vega()` = 37.524 | engine 0.375240 = QL/100 | pass |
| T = 0 raised `ZeroDivisionError` | n/a | now `ValueError` | fixed |

### `price_american_option_crr`

| Check | Reference | Tolerance achieved [asserted] | Status |
| :--- | :--- | :--- | :--- |
| American put S0=K=50, r=10%, sigma=40%, T=5/12 at 5/30/50/100/500 steps: 4.49 / 4.263 / 4.272 / 4.278 / 4.283 | Hull Example 21.1 and §21.1 | 4.4885 / 4.2634 / 4.2720 / 4.2781 / 4.2830 [half a unit in Hull's last digit] | pass |
| 4 cases (puts, call with q=6%), 1000 steps | QL BinomialVanillaEngine("crr", 1000) | max abs. 3.6e-5 [1e-4]. QL's CRR uses a drift-matched probability, so its tree differs from this one by O(1/N) | pass |
| CRR call with q=0 converges to BSM (no early exercise) | Merton (1973); Hull §11.5 | error −0.0399 / −0.00999 / −0.00200 at 50 / 200 / 1000 steps, ratio 50→1000 ≈ 20 (first order) | pass |
| American ≥ same-N European tree, ≥ intrinsic, ≥ BSM up to the tree's discretisation error | No-arbitrage bounds | 20 cases [1e-10] | pass |
| Risk-neutral `p` outside (0, 1) (1 step, sigma=1%, r=5%: p=3.06) | n/a | before: silently returned 2.926 (BSM/American value 4.877). After: `ValueError` | fixed |

### `calculate_implied_volatility`

| Check | Reference | Tolerance achieved [asserted] | Status |
| :--- | :--- | :--- | :--- |
| IV of the QL NPV, 10 cases with q > 0 | QL `impliedVolatility` (accuracy 1e-12) | max abs. 4.7e-14 [1e-9] | pass |
| Round trip over 630 points: K=20..400 on S=100, T=0.01..5 y, sigma=1%..400%, calls and puts | Self-consistency | 430 points with per-1.0 vega > 1e-3: worst rel. vol error 5.4e-10 [1e-8]. All returned roots reprice to ≤ 1.9e-11 [1e-9] | pass |
| Ill-posed deep ITM/OTM points (vega ≈ 0) | n/a | 200 points have vega ≤ 1e-3. 2 of them raise `ValueError` and the rest return a vol that reprices but need not equal the input | documented |
| Price outside the no-arbitrage bounds | n/a | `ValueError` with the BSM price range in the message (it was SciPy's generic "f(a) and f(b) must have different signs") | pass |

### `sabr_implied_volatility`

| Check | Reference | Tolerance achieved [asserted] | Status |
| :--- | :--- | :--- | :--- |
| 4 parameter sets (beta 0, 0.5, 1; F = 100, 0.03, 0.05), 21 strikes 0.5F..1.5F | QL `sabrVolatility`; independent transcription of Hagan (2.17a)/(2.18) | max rel. 4.6e-15 against both [1e-9] | pass |
| beta=1, nu=0 gives constant Black vol alpha | Hagan (2002) | rel. 1e-14 asserted | pass |
| Near ATM, K = F(1 + 1e-5..1e-3) | QL `sabrVolatility` | before: rel. error 2.25e-5 (F=0.05, beta=1: 0.0455500 vs 0.0455510) and 6.8e-6 (F=0.03, beta=0.5), caused by an absolute `abs(F-K) < 1e-6` switch to the ATM formula. After: 7.3e-13 [1e-9] | fixed |

### `calibrate_sabr`

| Check | Reference | Tolerance achieved [asserted] | Status |
| :--- | :--- | :--- | :--- |
| Recover known (alpha, rho, nu) from a 9-strike smile generated by QL `sabrVolatility` | QL | F=100, beta 0.5: alpha 2.5e-12 rel., rho 3.0e-11, nu 4.1e-11. F=0.03, beta 0.5: 4.6e-8, 4.0e-7, 8.4e-7. F=100, beta 1: 1.4e-9, 5.5e-9, 4.9e-9 [1e-5] | pass |
| F=5000, beta=0.5 and F=100, beta=0 | n/a | before: `ValueError: Initial guess is outside of provided bounds`, because alpha was bounded to an absolute [1e-4, 5] although it scales with F^(1-beta). After: fits with rmse 7.6e-4 and 9.4e-4 | fixed |
| `is_arbitrage_free` on an exactly fitted smile with negative density (F=3%, beta 0.5, rho −0.4, nu 0.6, T=10) | QL `blackFormula` prices at the quoted strikes are concave | before: `True`, because the flag was just `rmse < 0.05`. After: `False`, from a butterfly and call-spread check of Black-76 prices on the fitted smile | fixed |

### `verify_butterfly_arbitrage`

| Check | Reference | Tolerance achieved [asserted] | Status |
| :--- | :--- | :--- | :--- |
| 21 BSM call prices K=50..150 are accepted | BSM prices are convex in K (Breeden-Litzenberger) | valid, no violations | pass |
| Deliberately concave set C(90, 100, 110) = (10, 9, 1) | `10 − 18 + 1 = −7 < 0` | flagged at K=100 | pass |
| BSM strip with C(100) raised by 1.5 | Convexity at 100 drops from 0.184 to −0.116 | flagged at K=100 only | pass |
| Strikes in descending order | n/a | before: the concave set was reported **valid** and the convex set invalid. After: points are sorted first | fixed |
| NaN price, duplicate strikes | n/a | before: z3 `parser error` / `ZeroDivisionError`. After: `ValueError` | fixed |

## Open items (not changed)

1. **American Greeks.** `price_american_option_crr(...).greeks` returns European BSM Greeks. Against
   QL CRR (2000 steps), the tree delta is −0.4140 where the engine reports −0.3858 (Hull put), and
   −0.5512 vs −0.5004, 0.6803 vs 0.6264, −0.3973 vs −0.3476 in the other three test cases. This is now
   documented in the docstring and the MCP tool description. Tree Greeks were not implemented.
2. **Vanna and volga scaling.** Vanna is per 1.0 of vol while vega is per vol point. Volga is the
   derivative of the per-vol-point vega with respect to sigma in decimal. This is documented and left
   unchanged.
3. **IV ill-conditioning.** When vega is close to 0, the implied vol carries no information. The
   function does not report a conditioning measure.
4. **CRR performance.** Node prices are now built with NumPy instead of a Python list comprehension.
   One measured run per setting gave 0.17 s / 0.69 s at 1000 / 2000 steps before the change and
   0.023 s / 0.079 s after. The cost is still O(N²).
