# Validation: invariant engine (`core/invariants.py`)

Audited 2026-11-19. Tests: `tests/reference/test_invariants_reference.py`.

**Scope.** Every Z3 variable is bound to a concrete input, so `SAT`/`UNSAT` here equals exact
rational arithmetic on the decimal inputs (e.g. `0.1 + 0.2 == 0.3` holds). It verifies the numbers
given; it does not prove properties over ranges of inputs. There is no unsatisfiable-core
extraction; failures are reported as violation messages. Covenant checks do not use Z3.

| Function | Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `verify_balance_sheet_equation` | A = L + E with exact decimal semantics; 10¹¹-scale totals | accounting identity | 0 (exact) | pass |
| `verify_balance_sheet_equation` | tolerance boundary (9e-5 passes, 2e-4 fails at 1e-4) | definition | absolute | pass |
| all three functions | NaN / ±inf inputs | n/a | n/a | **fixed**: raised an opaque Z3 "parser error"; now a clear `ValueError` |
| `verify_three_statement_invariants` | reports each failed identity separately | identities | absolute | pass |
| `verify_three_statement_invariants` | `net_income` use | n/a | n/a | **open**: accepted but unused; no retained-earnings roll-forward yet |
| `verify_debt_covenant` | breach detection and headroom (5.45x vs 4.50x; 2.5x vs 3.0x) | ratio definitions | 1e-9 | pass |
| `verify_debt_covenant` | ratio exactly at limit (2.1 / 0.7 = 3.0x vs 3.0x max) | definition | 1e-9 rel | **fixed**: float rounding (3.0000000000000004) reported a false breach |

Docs corrected: README, framework paper and architecture diagram no longer claim Z3 covenant
proofs, an SMT-derived equity cure, or unsatisfiable-core feedback.
