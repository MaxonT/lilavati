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
| Econometrics (ADF, Engle-Granger + OU half-life, GARCH) | [econometrics](econometrics.md) | 7 (GARCH p/q swapped; half-life used the ln2/(1−φ) shortcut; unconverged GARCH returned start values; collinear/constant/non-finite inputs; unchecked significance level) | percent-return inputs not detected; √252 daily assumption; MacKinnon tables checked via statsmodels/arch, not the paper |
| Financial ML (fractional differencing, d* search, triple barrier) | [financial_ml](financial_ml.md) | 11 (placeholder d*=1.0 when no solution; ignored target p-value; float grid overshoot; d* rounding; exact barrier touch missed; invalid inputs) | d>0 tested on a shorter sample than d=0 (as in AFML); fixed-% barriers only |
| Filing connectors (10-K PDF parser, EDGAR XBRL, filing-anchored extraction, Ind AS crosswalk) | [connectors](connectors.md) | 6 (wrong tax line, dropped non-current debt, par value read as equity, missing item reported as 0.0, made-up SEC contact, comparative periods returned as current; three-statement model filled untagged fields and assumed cash reconciled; PDF parser and Ind AS crosswalk assumed cash reconciled and substituted missing items; Ind AS cost of goods sold and operating income mis-mapped) | 5-company pilot; banks need their own template; 10-Q supported but not validated (cash flows are year-to-date); restatements not covered; the three-statement model refuses filings without operating income (J&J, banks); Ind AS crosswalk tested on a fictional disclosure only |
| LOB / Black-Litterman / HRP, statement graph | pending | n/a | not yet audited |

The audit is not complete. Until every row is audited, treat unaudited engines as unvalidated.
