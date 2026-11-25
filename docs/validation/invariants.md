# Validation: invariant engine (`core/invariants.py`)

Audited 2026-11-19. Tests: `tests/reference/test_invariants_reference.py`.

**Scope.** In the `verify_*` checks every Z3 variable is bound to a concrete input, so `SAT`/`UNSAT`
equals exact rational arithmetic on the decimal inputs (e.g. `0.1 + 0.2 == 0.3` holds): it verifies
the numbers given. `verify_debt_covenant` does not use Z3. Three functions (added 2026-11-25) go
further: `diagnose_identities`, `minimum_debt_repayment` and `check_covenants_over_ranges` (below).

| Function | Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `verify_balance_sheet_equation` | A = L + E with exact decimal semantics; 10¹¹-scale totals | accounting identity | 0 (exact) | pass |
| `verify_balance_sheet_equation` | tolerance boundary (9e-5 passes, 2e-4 fails at 1e-4) | definition | absolute | pass |
| all three functions | NaN / ±inf inputs | n/a | n/a | **fixed**: raised an opaque Z3 "parser error"; now a clear `ValueError` |
| `verify_three_statement_invariants` | reports each failed identity separately | identities | absolute | pass |
| `verify_three_statement_invariants` | `net_income` use | n/a | n/a | **open**: accepted but unused; no retained-earnings roll-forward yet |
| `verify_debt_covenant` | breach detection and headroom (5.45x vs 4.50x; 2.5x vs 3.0x) | ratio definitions | 1e-9 | pass |
| `verify_debt_covenant` | ratio exactly at limit (2.1 / 0.7 = 3.0x vs 3.0x max) | definition | 1e-9 rel | **fixed**: float rounding (3.0000000000000004) reported a false breach |

Docs corrected (2026-11-19): README, framework paper and architecture diagram stopped claiming Z3
covenant results, an SMT-derived equity cure and unsatisfiable-core feedback that did not exist. The
functions below now provide a diagnosis, a repayment optimum and a range check; the docs describe
exactly those and no more.

## Diagnosis, smallest debt repayment, covenants over ranges (2026-11-25)

Tests: `tests/reference/test_invariants_reference.py`, `tests/test_framework.py`.

| Function | Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `diagnose_identities` | consistent figures (Apple FY2026 10-K) report no failure | identities | 0.5 (input units) | pass |
| `diagnose_identities` | total assets mistyped breaks two identities; the single candidate is total assets, implied 404,548 | minimal hitting set of the conflicts (Reiter 1987) | exact | pass |
| `diagnose_identities` | closing cash mistyped breaks one identity; all five cash figures listed, each with its implied value | same | exact | pass |
| `diagnose_identities` | equity mistyped; total assets excluded because changing it would break the second assets identity | same | exact | pass |
| `diagnose_identities` | two independent errors need two figures; custom identities; missing figures raise | same | exact | pass |
| `minimum_debt_repayment` | 40 random cases | closed form r* = max(0, D − cash − L·EBITDA, D·(1 − EBITDA / (C·interest))) | exact rationals, 1e-9 rel | pass |
| `minimum_debt_repayment` | hand case: D 500, cash 50, EBITDA 100, interest 40, 3.5x, 3.0x gives 100, leverage binding, coverage 3.125x | hand calculation | exact | pass |
| `check_covenants_over_ranges` | 60 random boxes | worst corner (both ratios monotone in each input) | exact | pass |
| `check_covenants_over_ranges` | worst corner exactly at both limits complies; one unit beyond breaches the right covenant | definition | exact | pass |
| `SMTBalanceGate` | the reflection prompt names the misread figure and its implied value; says when candidates cannot be told apart | diagnosis | n/a | pass |

The repayment model is stated in the function: repayment from outside the cash balance (repaying from
existing cash leaves net debt unchanged) and interest proportional to debt. Whether and how a sponsor
equity cure counts is set by each credit agreement. Ranges are independent intervals; interest as a
rate times debt (dependent inputs) and multi-period covenant paths are open (TODO, Phase 9).

References: Reiter, R. (1987), A theory of diagnosis from first principles, *Artificial Intelligence*
32(1), 57–95; Bjørner, N., Phan, A.-D., and Fleckenstein, L. (2015), νZ - An optimizing SMT solver,
*TACAS 2015*, LNCS 9035, 194–199.

## Multi-period covenant paths with linked inputs (`core/covenant_paths.py`, 2026-11-25)

Model (module docstring): annual periods; interest = rate × each period's opening debt (linked to the
debt path); scheduled amortisation; an excess cash sweep with a fixed fraction or a leverage grid;
covenants each period on net leverage (step-downs allowed), interest coverage and minimum liquidity.
`simulate_covenant_path` evaluates one path in exact rational arithmetic without a solver; it is the
reference, and every breach Z3 reports is replayed through it before it is returned. Tests:
`tests/reference/test_covenant_paths_reference.py`.

| Function | Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `simulate_covenant_path` | two-year example (interest 16.2, sweep 21.52, debt 248.48, year-2 coverage 3.354 < 3.5) | hand calculation | exact | pass |
| `check_covenant_path` | 30 random paths, fixed sweep fraction, 1–4 years, rate ranges | worst corner (every covenant improves with EBITDA and worsens with the rate when the sweep fraction is fixed) | exact | pass |
| `check_covenant_path` | leverage-based sweep: all four corners comply, a path between them breaches (year-1 EBITDA 70 in 60–100) | plain simulation of the counterexample | exact | pass |
| `check_covenant_path` | "holds" with a sweep grid and a rate range: 200 random paths each, none breach | sampling (spot check) | exact | pass |
| `minimum_upfront_repayment_for_path` | 15 point forecasts, fixed sweep | bisection on the plain simulation | rel 1e-9 | pass |
| `minimum_upfront_repayment_for_path` | sweep cliff: interval (lower, upper]; upper holds for every path, lower breaches | direct checks of both ends | rel 1e-6 of debt | pass |
| `minimum_upfront_repayment_for_path` | liquidity cannot be met | infeasible | n/a | pass |

**Methods.** A breach search is one satisfiability query over all paths in the ranges ("holds" when
none exists). The smallest repayment is found by a counterexample-guided loop (exact optimum over the
scenarios found so far, then a new breaching scenario), and, when that does not settle (a sweep cliff
moves with the repayment), by bisection on the quantified formula "some repayment up to b keeps every
path compliant", which is monotone in b even when the set of working repayments is not an interval.
Each call uses a fresh Z3 context: the same problem took 9 s alone and 53 s after other Z3 work in one
process, because solver heuristics depend on the order in which terms were created.

**Found while building it.** The optimum often sits exactly at a covenant limit; converting it to a
float before re-checking pushed it over by about 1e-15 and the solver rejected its own answer. Checks
now use the exact repayment, and the reported float is rounded up so it still complies
(`required_repayment_exact` gives the fraction).

**Limits.** Covenant EBITDA is the period's own (not trailing twelve months); no revolver, dividends or
fixed-charge cover; a rate range makes interest nonlinear and Z3 may return `unknown` (reported as
such, never as a result). A single-period check with interest linked to debt gives the same worst case
as independent ranges for these two covenants; linking matters over several periods.

