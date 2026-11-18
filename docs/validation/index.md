# Numerical validation

Status of the reference-value audit (roadmap item 7.3). Each engine page lists, per public function,
the check, the independent reference (textbook example, original paper, or QuantLib 1.44 used as a
test-only dependency), the worst error measured, and whether a bug was fixed. Reference tests live
in `tests/reference/`.

| Engine | Page | Bugs fixed in audit | Open items |
| :--- | :--- | :---: | :--- |
| Fixed income (bonds, Nelson-Siegel, Merton) | [fixed_income](fixed_income.md) | 1 (Merton solver returned non-solutions in distress) | fractional coupon periods silently rounded; NS start-guess bounds |
| Derivatives (BSM, Greeks, CRR, IV, SABR, butterfly) | [derivatives](derivatives.md) | 7 (incl. SABR near-ATM, `is_arbitrage_free`, butterfly ordering, invalid CRR probability) | American Greeks are European BSM Greeks; calendar arbitrage not checked |
| Invariants (Z3 identities, covenants) | [invariants](invariants.md) | 2 (covenant at-limit false breach; non-finite inputs) | `net_income` unused (no retained-earnings roll-forward) |
| Sustainability (WACI, carbon tax, SLB) | [sustainability](sustainability.md) | 3 (SLB at-target rounding; invalid weights; zero baseline) | none |
| Forensics (Beneish, Altman) | [forensics](forensics.md) | 1 (Beneish coefficients, fixed earlier) | Altman zero-denominator error message |
| Almgren-Chriss execution | see CHANGELOG | 1 (fixed earlier) | none |
| Econometrics, financial ML, LOB / Black-Litterman / HRP, statement graph, connectors | pending | n/a | not yet audited |

The audit is not complete. Until every row is audited, treat unaudited engines as unvalidated.
