# Validation: forensic scores (`agents/forensics.py`)

Audited 2026-11-19. Tests: `tests/test_beneish.py`, `tests/reference/test_forensics_reference.py`.

| Function | Check | Reference | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `calculate_beneish_m_score` | 8-variable coefficients; all indices 1 ⇒ M = −2.480 | Beneish (1999), FAJ 55(5) | 1e-9 | **fixed**: TATA 4.037 → 4.679, LVGI +0.0327 → −0.327 |
| `calculate_beneish_m_score` | TATA definition | Beneish (1999) | n/a | documented: uses (NI − CFO)/TA simplification, not the balance-sheet accruals definition |
| `calculate_altman_z_score` | hand-computed Z = 2.8288 (grey), 1.2295 (distress), safe case | Altman (1968), J. Finance 23(4) | 1e-12 | pass |
| `calculate_altman_z_score` | zero total assets/liabilities | n/a | n/a | open: raises `ZeroDivisionError` instead of a clear error |
