"""Seeded, templated case generator for Lilavati-Bench.

Produces fresh instances (new numbers every seed) for tracks whose ground truth can be computed
by the deterministic engines. Boolean answers are balanced ~50/50 so that a content-free
"always True" answerer cannot score well. Use a private seed for held-out leaderboard runs.
"""

from __future__ import annotations

import random
import re
from datetime import date

from lilavati.bench.datasets import (
    DOC_HASH_AAPL_2024,
    DOC_HASH_CBOE_SPX_2024,
    DOC_HASH_GHG_PROTOCOL_2024,
    DOC_HASH_NASDAQ_ITCH_2024,
    DOC_HASH_US_TREASURY_2024,
    BenchmarkTestCase,
)
from lilavati.core.derivatives import DerivativesEngine
from lilavati.core.execution import ExecutionEngine
from lilavati.core.fixed_income import FixedIncomeEngine
from lilavati.core.sustainability import SustainabilityEngine
from lilavati.schemas.sustainability import SustainabilityLinkedBond

GENERATED_TRACKS = (
    "track1_three_statement",
    "track2_fixed_income",
    "track5_derivatives",
    "track6_microstructure",
    "track7_sustainability",
)


def _t1(rng: random.Random, i: int) -> BenchmarkTestCase:
    assets = round(rng.uniform(5e4, 5e5), 0)
    liab = round(assets * rng.uniform(0.3, 0.8), 0)
    balanced = i % 2 == 0
    equity = assets - liab + (0.0 if balanced else float(rng.choice([-1, 1]) * rng.randint(50, 3000)))
    cash_start = round(rng.uniform(1e4, 6e4), 0)
    cfo, cfi = round(rng.uniform(5e4, 1.5e5), 0), -round(rng.uniform(1e3, 3e4), 0)
    cff = -round(rng.uniform(5e4, 1.2e5), 0)
    reconciles = (i // 2) % 2 == 0
    cash_end = cash_start + cfo + cfi + cff + (0.0 if reconciles else float(rng.randint(100, 2000)))
    return BenchmarkTestCase(
        case_id=f"GEN-T1-{i:03d}", track="track1_three_statement",
        title="Generated balance-sheet and cash reconciliation",
        prompt="Check whether the balance sheet balances and whether the cash flow statement "
        "reconciles opening to closing cash (exact, no tolerance beyond rounding).",
        as_of_date=date(2024, 11, 1), doc_hash=DOC_HASH_AAPL_2024, source_citation="generated",
        context_data={"total_assets": assets, "total_liabilities": liab, "total_equity": equity,
                      "cash_start": cash_start, "cfo": cfo, "cfi": cfi, "cff": cff,
                      "cash_end": cash_end, "net_income": round(rng.uniform(1e4, 1e5), 0)},
        expected_values={"is_balanced": balanced, "cash_reconciled": reconciles,
                         "assets_equal_liab_equity": balanced},
    )


def _t2(rng: random.Random, i: int) -> BenchmarkTestCase:
    face, mat = 100.0, rng.randint(2, 10)
    coupon, ytm = round(rng.uniform(0.02, 0.08), 4), round(rng.uniform(0.02, 0.09), 4)
    price = FixedIncomeEngine.analyze_bond(
        face_value=face, coupon_rate=coupon, ytm=ytm, years_to_maturity=float(mat), frequency=1
    ).clean_price
    return BenchmarkTestCase(
        case_id=f"GEN-T2-{i:03d}", track="track2_fixed_income", title="Generated bond price",
        prompt="Price the annual-pay bond described in the input data.",
        as_of_date=date(2024, 10, 15), doc_hash=DOC_HASH_US_TREASURY_2024, source_citation="generated",
        context_data={"face_value": face, "coupon_rate": coupon, "ytm": ytm, "maturity_years": float(mat)},
        expected_values={"bond_price": price}, tolerance=0.005,
    )


def _t5(rng: random.Random, i: int) -> BenchmarkTestCase:
    s, k = 100.0, float(rng.choice([80, 90, 100, 110, 120]))
    r, vol, t = round(rng.uniform(0.01, 0.06), 3), round(rng.uniform(0.1, 0.5), 2), rng.choice([0.25, 0.5, 1.0, 2.0])
    px = DerivativesEngine.price_european_option(spot=s, strike=k, time_to_maturity=t,
                                                 risk_free_rate=r, volatility=vol, is_call=True)
    g = DerivativesEngine.calculate_bsm_greeks(spot=s, strike=k, time_to_maturity=t,
                                               risk_free_rate=r, volatility=vol, is_call=True)
    return BenchmarkTestCase(
        case_id=f"GEN-T5-{i:03d}", track="track5_derivatives", title="Generated BSM call and Greeks",
        prompt="Price the European call and report delta, gamma and vega (Black-Scholes, no dividends).",
        as_of_date=date(2024, 10, 1), doc_hash=DOC_HASH_CBOE_SPX_2024, source_citation="generated",
        context_data={"spot": s, "strike": k, "rate": r, "volatility": vol, "maturity": t},
        expected_values={"call_price": px.price, "delta": g.delta, "gamma": g.gamma, "vega": g.vega},
        tolerance=0.02,
    )


def _t6(rng: random.Random, i: int) -> BenchmarkTestCase:
    bid = round(rng.uniform(20, 300), 2)
    ask = round(bid + rng.choice([0.01, 0.02, 0.05, 0.10, 0.25]), 2)
    bs, asz = float(rng.randint(100, 2000)), float(rng.randint(100, 2000))
    lob = ExecutionEngine.analyze_lob_snapshot(best_bid=bid, best_ask=ask, bid_size=bs, ask_size=asz)
    return BenchmarkTestCase(
        case_id=f"GEN-T6-{i:03d}", track="track6_microstructure", title="Generated microprice",
        prompt="Compute the spread and the size-weighted microprice of the book snapshot.",
        as_of_date=date(2024, 11, 8), doc_hash=DOC_HASH_NASDAQ_ITCH_2024, source_citation="generated",
        context_data={"bid_price": bid, "bid_size": bs, "ask_price": ask, "ask_size": asz},
        expected_values={"spread": lob.spread, "microprice": lob.micro_price}, tolerance=0.002,
    )


def _t7(rng: random.Random, i: int) -> BenchmarkTestCase:
    w1 = round(rng.uniform(0.2, 0.8), 2)
    weights, em = [w1, round(1 - w1, 2)], [float(rng.randint(2, 20) * 10000) for _ in range(2)]
    rev = [float(rng.randint(2, 20) * 100) for _ in range(2)]
    base, target = 100000.0, rng.choice([0.15, 0.2, 0.25])
    achieved = i % 2 == 0
    red = target + 0.03 if achieved else target - 0.03
    reported = base * (1 - red)
    bond = SustainabilityLinkedBond(isin="XS0000000000", notional=5e7, base_coupon_rate=0.04,
                                    target_reduction_pct=target, baseline_emissions=base,
                                    step_up_bps=25.0, target_year=2026)
    waci = SustainabilityEngine.calculate_waci(portfolio_weights=weights, emissions=em, revenues_millions=rev)
    slb = SustainabilityEngine.evaluate_slb_covenant(bond=bond, reported_emissions=reported)
    return BenchmarkTestCase(
        case_id=f"GEN-T7-{i:03d}", track="track7_sustainability", title="Generated WACI and SLB step-up",
        prompt="Compute portfolio WACI, then decide whether the SLB sustainability target is met and "
        "the resulting coupon (25 bps step-up if missed; base coupon 4%).",
        as_of_date=date(2024, 10, 31), doc_hash=DOC_HASH_GHG_PROTOCOL_2024, source_citation="generated",
        context_data={"weights": weights, "emissions": em, "revenues_mil": rev,
                      "slb_base_coupon": 0.04, "slb_target_reduction": target,
                      "slb_baseline_emissions": base, "slb_step_up_bps": 25.0,
                      "slb_reported_emissions": reported},
        expected_values={"portfolio_waci": waci.waci, "slb_spt_achieved": slb.spt_achieved,
                         "slb_active_coupon": slb.active_coupon_rate,
                         "slb_penalty_triggered": not slb.spt_achieved},
        tolerance=0.005,
    )


_BUILDERS = {"track1_three_statement": _t1, "track2_fixed_income": _t2, "track5_derivatives": _t5,
             "track6_microstructure": _t6, "track7_sustainability": _t7}


def generate_cases(seed: int = 0, per_track: int = 10) -> list[BenchmarkTestCase]:
    """Generate `per_track` fresh cases for each supported track, deterministically from `seed`.

    One random stream is shared across tracks, so the content of case i depends on `per_track`
    as well as `seed`. Ids therefore carry both (``GEN-S2026-N4-T2-000``); compare results by
    `case_content_hash`, not by id.
    """
    rng = random.Random(seed)
    cases = [_BUILDERS[t](rng, i) for t in GENERATED_TRACKS for i in range(per_track)]
    prefix = f"GEN-S{seed}-N{per_track}-"
    return [c.model_copy(update={"case_id": prefix + c.case_id.removeprefix("GEN-")}) for c in cases]


def legacy_case_id(case_id: str) -> str:
    """Map a current generated id to the pre-2026-11-21 form (``GEN-S2026-N4-T2-000`` -> ``GEN-T2-000``).

    Result files written before the id change use the short form; resolve them together with the
    seed and per-track count recorded in ``results/manifest.json``.
    """
    return re.sub(r"^GEN-S\d+-N\d+-", "GEN-", case_id)
