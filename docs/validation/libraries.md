# Libraries, solvers and citations

Līlāvatī does not reimplement numerical methods that a mature open-source library already provides.
It wraps them with stated units, validated inputs and a check on every result before it is returned
(the developer rule "check solver output"). This page lists which library does what, how each result
is checked, and how to cite the libraries if you use Līlāvatī in published work.

## Numerical methods and how their output is checked

| Engine (module) | Method | Library | Check before a result is returned |
| :--- | :--- | :--- | :--- |
| Merton credit model (`core/fixed_income.py`) | two Brent root-finds (asset value, asset volatility) | SciPy `optimize.brentq` | relative residuals of both Merton equations; raises if they are not met ([fixed income](fixed_income.md)) |
| Implied volatility (`core/derivatives.py`) | Brent root-find on volatility in [0.01%, 500%] | SciPy `optimize.root_scalar` | the bracket must contain a solution, else `ValueError` naming the price bounds |
| Nelson-Siegel curve (`core/fixed_income.py`) | bounded nonlinear least squares | SciPy `optimize.curve_fit` | SciPy raises if it does not converge; a clearer start-guess message is open (#8) |
| SABR smile (`core/derivatives.py`) | bounded nonlinear least squares | SciPy `optimize.curve_fit` | fit error (`rmse`, vol units) and a butterfly check on the fitted smile are returned |
| GARCH(p, q) (`core/econometrics.py`) | maximum likelihood | arch | the optimiser's convergence flag; raises if not converged ([econometrics](econometrics.md)) |
| ADF, Engle-Granger (`core/econometrics.py`) | unit-root and cointegration tests | statsmodels `adfuller`, `coint` | settings passed explicitly (trend, lag selection) |
| Black-Litterman weights (`core/execution.py`) | convex quadratic program (long-only, weights sum to 1) | CVXPY with the Clarabel solver | solver status must be optimal, else `RuntimeError`; tests compare with the closed-form optimum when no bound binds. Not yet checked against a published example (#15) |
| Hierarchical risk parity (`core/execution.py`) | single-linkage clustering, recursive bisection | SciPy `cluster.hierarchy` | not yet checked against the paper (#16) |
| Accounting identities, butterfly check (`core/invariants.py`, `core/derivatives.py`) | satisfiability of linear constraints on the given numbers | Z3 | exact rational arithmetic on the concrete inputs; it checks those numbers, not a model over ranges of inputs ([invariants](invariants.md)) |
| Point-in-time store (`core/temporal.py`) | as-of queries | DuckDB | look-ahead tests in `tests/` |
| 10-K PDF text (`connectors/pdf_10k.py`) | text extraction | pypdf | identity checks on the parsed statements ([connectors](connectors.md)) |

**Why Clarabel is named.** CVXPY chooses a solver from those installed when none is named; in our
environment it picked OSQP for the Black-Litterman problem, while Clarabel, SCS, SciPy and HiGHS were
also installed. A different environment could pick a different solver and return slightly different
weights. Clarabel is a required dependency of CVXPY, so naming it keeps results the same everywhere;
the result's `solver` field records it.

**Test-only reference.** QuantLib provides independent reference values in `tests/reference/` (bond
analytics, option pricing); the package does not depend on it at run time.

## Libraries, licences and citations

Versions as locked in `uv.lock` on 2026-11-25; licences as declared on PyPI.

| Library | Used for | Version | Licence | Cite |
| :--- | :--- | :--- | :--- | :--- |
| NumPy | arrays, linear algebra | 2.4.6 | BSD-3-Clause (with bundled MIT, Zlib and other notices) | Harris et al. (2020) |
| SciPy | root-finding, least squares, clustering, distributions | 1.17.1 | BSD-3-Clause | Virtanen et al. (2020) |
| statsmodels | ADF and Engle-Granger tests | 0.15.0 | BSD-3-Clause | Seabold and Perktold (2010) |
| arch | GARCH estimation | 9.0.0 | NCSA | Sheppard et al., software |
| CVXPY | convex optimisation (Black-Litterman weights) | 1.9.3 | Apache-2.0 | Diamond and Boyd (2016) |
| Clarabel | interior-point solver used through CVXPY | 0.11.1 | Apache-2.0 | Goulart and Chen (2024) |
| Z3 (`z3-solver`) | identity and arbitrage checks | 5.4.1.0 | MIT | de Moura and Bjørner (2008) |
| DuckDB | bi-temporal store | 2.0.0 | MIT | Raasveldt and Mühleisen (2019) |
| Polars | market-data normalisation | 2.1.0 | MIT | software |
| pypdf | PDF text extraction | 7.0.1 | BSD-3-Clause | software |
| FastMCP | MCP servers | 4.1.8 | Apache-2.0 | software |
| Pydantic | typed schemas | 2.14.3 | MIT | software |
| httpx | HTTP client (SEC EDGAR) | 0.28.1 | BSD-3-Clause | software |
| QuantLib (tests only) | independent reference values | 1.44 | BSD-3-Clause | software |

### References

- de Moura, L., and Bjørner, N. (2008). Z3: An efficient SMT solver. *TACAS 2008*, LNCS 4963,
  337–340. <https://doi.org/10.1007/978-3-540-78800-3_24>
- Diamond, S., and Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex
  optimization. *Journal of Machine Learning Research* 17(83), 1–5.
  <https://jmlr.org/papers/v17/15-408.html>
- Goulart, P. J., and Chen, Y. (2024). Clarabel: An interior-point solver for conic programs with
  quadratic objectives. arXiv:2405.12762. <https://arxiv.org/abs/2405.12762>
- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature* 585, 357–362.
  <https://doi.org/10.1038/s41586-020-2649-2>
- Raasveldt, M., and Mühleisen, H. (2019). DuckDB: an embeddable analytical database. *SIGMOD '19*,
  1981–1984. <https://doi.org/10.1145/3299869.3320212>
- Seabold, S., and Perktold, J. (2010). statsmodels: Econometric and statistical modeling with
  Python. *Proceedings of the 9th Python in Science Conference*, 92–96.
  <https://doi.org/10.25080/Majora-92bf1922-011>
- Sheppard, K., et al. arch (software). <https://github.com/bashtage/arch>; each release is archived
  on Zenodo with its own DOI.
- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python.
  *Nature Methods* 17, 261–272. <https://doi.org/10.1038/s41592-019-0686-2>

Software without a paper (Polars, pypdf, FastMCP, Pydantic, httpx, QuantLib): cite the project and
the version you used.
