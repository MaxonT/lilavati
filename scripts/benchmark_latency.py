"""Reproducible latency micro-benchmarks for core Lilavati engines.

Usage: uv run python scripts/benchmark_latency.py [--runs 30] [--pdf data/sample_filings/aapl_10k_2026.pdf]
Prints a markdown table (median / p95 in ms) plus the environment.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import z3

from lilavati.core.derivatives import DerivativesEngine
from lilavati.core.financial_ml import FinancialMLEngine
from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef, SourcedValue
from lilavati.core.temporal import BiTemporalWarehouse


def timeit(fn: Callable[[], object], runs: int) -> tuple[float, float]:
    fn()  # warm-up
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(0.95 * len(samples)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pdf", default="data/sample_filings/aapl_10k_2026.pdf")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    prices = (np.cumsum(rng.normal(0.01, 1.0, 1000)) + 100.0).tolist()
    mats = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    ylds = [0.040, 0.041, 0.042, 0.043, 0.0435, 0.044, 0.0445, 0.045, 0.046, 0.0465]
    strikes = np.linspace(80, 120, 9).tolist()
    vols = [0.30, 0.27, 0.25, 0.23, 0.22, 0.225, 0.235, 0.25, 0.27]

    wh = BiTemporalWarehouse()
    cit = CitationRef(doc_hash="a" * 64, filing_type="10-K", filing_date=date(2024, 11, 1),
                      period_end=date(2024, 9, 30), page=1, table_id="t", xbrl_tag="x", raw_snippet="s")
    wh.store_fact("ACME", "revenue", SourcedValue(value=100.0, unit="USD", scale=1, citation=cit))

    cases: list[tuple[str, str, Callable[[], object]]] = [
        ("Balance-sheet equation (Z3)", f"z3 {z3.get_version_string()}",
         lambda: SuvarnaEngine.verify_balance_sheet_equation(404548.0, 285508.0, 119040.0)),
        ("Nelson-Siegel fit (10 pts)", "SciPy", lambda: FixedIncomeEngine.fit_nelson_siegel(mats, ylds)),
        ("Optimal frac. diff d* (1,000 obs)", "NumPy/statsmodels",
         lambda: FinancialMLEngine.find_optimal_fractional_d(prices)),
        ("SABR calibration (9 strikes)", "SciPy", lambda: DerivativesEngine.calibrate_sabr(100.0, strikes, vols, 1.0)),
        ("Bi-temporal as-of query", "DuckDB in-memory",
         lambda: wh.query_as_of("ACME", "revenue", date(2024, 9, 30), date(2024, 12, 1))),
    ]
    pdf = Path(args.pdf)
    if pdf.exists():
        from lilavati.connectors.pdf_10k import PDF10KParser

        cases.append(("10-K full extraction (123 pp)", "pypdf",
                      lambda: PDF10KParser(str(pdf)).extract_full_filing()))

    print(f"Environment: Python {platform.python_version()}, {platform.platform()}, {platform.machine()}, runs={args.runs}\n")
    print("| Operation | Engine | Median (ms) | p95 (ms) |\n| :--- | :--- | ---: | ---: |")
    for name, eng, fn in cases:
        try:
            med, p95 = timeit(fn, 3 if "10-K" in name else args.runs)
            print(f"| {name} | {eng} | {med:.2f} | {p95:.2f} |")
        except Exception as exc:  # report, don't hide
            print(f"| {name} | {eng} | ERROR: {type(exc).__name__}: {exc} | |")


if __name__ == "__main__":
    main()
