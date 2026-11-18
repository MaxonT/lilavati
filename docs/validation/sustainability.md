# Validation: sustainability engine (`core/sustainability.py`)

Audited 2026-11-19. Tests: `tests/reference/test_sustainability_reference.py`.

| Function | Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `calculate_waci` | WACI = Σ wᵢ·Eᵢ/Rᵢ on a hand-computed 2-asset portfolio (120.0) | TCFD (2017) WACI definition | rel 1e-12 | pass |
| `calculate_waci` | weights normalised; zero-sum or negative weights rejected | definition (long-only) | exact | **fixed**: zero-sum weights divided by zero; negative weights were accepted |
| `calculate_waci` | `total_carbon_footprint` meaning | PCAF Standard | n/a | **fixed (docs)**: was labelled "financed emissions"; it is Σ wᵢ·Eᵢ, not PCAF financed emissions |
| `stress_test_carbon_tax` | net cost, margins, EBIT reduction, coverage on a hand-computed case | model definition | 1e-9 | pass |
| `stress_test_carbon_tax` | full pass-through ⇒ no EBIT impact | limiting case | exact | pass |
| `stress_test_carbon_tax` | Scope 2 inclusion, "_pct" fields are fractions | n/a | n/a | documented (modelling choice) |
| `evaluate_slb_covenant` | missed target ⇒ coupon 4.00% → 4.25%, penalty = notional × 25 bp | ICMA SLB Principles (2023) | 1e-12 | pass |
| `evaluate_slb_covenant` | target met exactly (baseline 3.0, 10%, reported 2.7) | definition | exact | **fixed**: floating-point ratio (0.09999999999999994) reported a miss |
| `SustainabilityLinkedBond` | non-positive baseline/notional rejected | definition | exact | **fixed**: baseline 0 caused division by zero |

Removed an unimplemented "Climate VaR" claim from the module docstring.
