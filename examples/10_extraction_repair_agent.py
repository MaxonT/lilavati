"""
Example 10: filing extraction with diagnosis-driven repair (docs/agent-workflows.md, workflow 1).

The document is verbatim text from Apple's FY2026 Form 10-K statements (each line: current year, then
prior years). The agent asks an extractor for seven figures with the line each came from, checks every
quote against the document and every applicable accounting identity, and re-reads only what the
diagnosis names; it abstains on figures it cannot settle.

By default the extractor is scripted to make the commonest real error, reading the prior-year column
for total assets. With --live it is a language model (key from the provider's environment variable):

    uv run python examples/10_extraction_repair_agent.py
    GROQ_API_KEY=... uv run python examples/10_extraction_repair_agent.py --live groq-gpt-oss-120b
"""

from __future__ import annotations

import argparse
from datetime import date

from lilavati.agents.extraction import FilingExtractionAgent, llm_extractor
from lilavati.llm.client import UniversalLLMClient
from lilavati.llm.providers import INSTITUTIONAL_MODEL_PRESETS

DOC = "\n".join([
    "Total net sales  478,834  416,161  391,035 ",
    "Total cost of sales  245,232  220,960  210,352 ",
    "Gross margin  233,602  195,201  180,683 ",
    "Total assets $ 404,548 $ 359,241 ",
    "Total liabilities  292,455  285,508 ",
    "Total shareholders’ equity  112,093  73,733 ",
    "Total liabilities and shareholders’ equity $ 404,548 $ 359,241 ",
])
TRUTH = {"revenue": 478_834, "cost_of_revenue": 245_232, "gross_profit": 233_602, "total_assets": 404_548,
         "total_liabilities": 292_455, "total_equity": 112_093, "liabilities_and_equity": 404_548}


def scripted_extractor(text: str, figures: list[str], instruction: str) -> dict[str, tuple[float, str]]:
    """Reads correctly, except total assets from the prior-year column on the first request."""
    lines = dict(zip(TRUTH, text.splitlines(), strict=True))
    out = {f: (float(TRUTH[f]), lines[f]) for f in figures}
    if "re-read" not in instruction and "total_assets" in figures:
        out["total_assets"] = (359_241.0, lines["total_assets"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", metavar="PRESET", help="use a language model preset, e.g. groq-gpt-oss-120b")
    args = ap.parse_args()
    extractor = (llm_extractor(UniversalLLMClient(INSTITUTIONAL_MODEL_PRESETS[args.live])) if args.live
                 else scripted_extractor)
    res = FilingExtractionAgent(extractor, max_repair_rounds=2).run(
        DOC, list(TRUTH), filing_type="10-K", filing_date=date(2026, 11, 3), period_end=date(2026, 9, 26))
    for r in res.rounds:
        print(f"round {r.round}: asked {r.requested}")
        print(f"  unsupported {r.unsupported}; failed {r.failed_identities}; candidates {r.candidates}")
    print(res.detail)
    for name, f in res.figures.items():
        mark = "ok" if f.value == TRUTH[name] else f"WRONG (10-K: {TRUTH[name]:,})"
        print(f"  {name:24} {f.value:>12,.0f}  {mark}")
    if res.unresolved:
        print("abstained on:", res.unresolved)


if __name__ == "__main__":
    main()
