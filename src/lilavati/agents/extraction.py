"""
Filing extraction with diagnosis-driven repair: a model reads figures, code decides which to re-read.

Workflow (docs/agent-workflows.md, workflow 1):

1. The extractor (a language model, or any callable) reads the requested figures from the document text
   and returns, for each, the value and the line it was read from.
2. Provenance: each quoted line must appear in the document (``SourceStore``), and the value must appear
   in the quoted line. A figure that fails is re-read.
3. Consistency: the statement identities the figures allow are checked; if any fails,
   ``diagnose_identities`` names the smallest sets of figures that could be wrong, and only those are
   re-read.
4. Repeat up to ``max_repair_rounds``. Figures still unsupported or still in a diagnosis are reported in
   ``unresolved``: the agent abstains on them rather than adjust a number to fit. Figures no applied
   identity involves are listed in ``unchecked``: only the document check stands behind them (Walmart's
   revenue when no gross profit is reported: net sales and total revenues are both printed).

The re-read request names the figures and the failed identities but, by default, not the values the
identities imply: a model shown "should be 404,548" can copy it instead of re-reading, which would hide
the misreading rather than fix it. The two checks are complementary: reading the prior-year column of a
two-column statement passes the provenance check (the value is in the quoted line) and fails the
identities; a typo passes nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from lilavati.core.invariants import applicable_identities, diagnose_identities
from lilavati.core.provenance import CitationRef, SourceStore
from lilavati.llm.client import BaseLLMClient, LLMMessage

# (document text, figures to read, instruction) -> {figure: (value, quoted line)}
Extractor = Callable[[str, list[str], str], dict[str, tuple[float, str]]]

# What each canonical figure means, for the extractor's prompt.
FIGURE_LABELS: dict[str, str] = {
    "revenue": "total revenues for the current year (where net sales and other revenues are shown, their total)",
    "cost_of_revenue": "total cost of revenue (cost of sales) for the current year",
    "gross_profit": "gross profit (gross margin) for the current year",
    "total_assets": "total assets at the current year end",
    "total_liabilities": "total liabilities at the current year end",
    "total_equity": "total equity at the current year end (including non-controlling interests if the balance sheet shows them)",
    "liabilities_and_equity": "total liabilities and shareholders' equity at the current year end",
    "cash_opening": "cash (including restricted cash) at the beginning of the current year",
    "operating_cash_flow": "net cash from operating activities for the current year",
    "investing_cash_flow": "net cash from investing activities for the current year (negative if used)",
    "financing_cash_flow": "net cash from financing activities for the current year (negative if used)",
    "cash_closing": "cash (including restricted cash) at the end of the current year",
    "fx_effect_on_cash": "effect of exchange rate changes on cash for the current year",
}


class ExtractedFigure(BaseModel):
    name: str
    value: float
    quote: str = Field(..., description="the line the value was read from, as the extractor gave it")
    quote_in_document: bool
    value_in_quote: bool
    citation: CitationRef


class RepairRound(BaseModel):
    round: int
    requested: list[str]
    unsupported: list[str] = Field(default_factory=list, description="quote not in the document, or value not in it")
    failed_identities: list[str] = Field(default_factory=list)
    candidates: list[list[str]] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    figures: dict[str, ExtractedFigure]
    consistent: bool = Field(..., description="every applicable identity holds and every figure is supported")
    unresolved: list[str] = Field(
        default_factory=list, description="figures the agent abstains on: still unsupported or still diagnosed")
    rounds: list[RepairRound]
    unchecked: list[str] = Field(
        default_factory=list,
        description="supported figures that no applied identity involves: checked against the document only")
    detail: str = ""


def value_in_quote(value: float, quote: str) -> bool:
    """Whether ``value`` is printed in ``quote`` (with thousands separators; negatives may be in parentheses)."""
    if value != int(value):
        return f"{abs(value):,}" in quote or str(abs(value)) in quote
    shown = f"{abs(int(value)):,}"
    return re.search(rf"(?<![\d,.]){re.escape(shown)}(?![\d,])", quote) is not None


class FilingExtractionAgent:
    """Extraction with provenance and identity checks, re-reading only what the diagnosis names."""

    def __init__(self, extractor: Extractor, max_repair_rounds: int = 2, tolerance: float = 0.5,
                 show_implied_values: bool = False, reread: Literal["diagnosed", "all"] = "diagnosed",
                 fx_line_absent: bool = False) -> None:
        """``reread="all"`` re-reads every figure after any failure (the comparison for diagnosis targeting).

        ``fx_line_absent``: the cash flow statement has no exchange-rate line, so the cash roll-forward
        applies without it. Otherwise it applies only if ``fx_effect_on_cash`` was read: an unread line
        must not make the identity fail by its amount.
        """
        self.fx_line_absent = fx_line_absent
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds must be >= 0")
        if reread not in ("diagnosed", "all"):
            raise ValueError("reread must be 'diagnosed' or 'all'")
        self.reread = reread
        self.extractor = extractor
        self.max_repair_rounds = max_repair_rounds
        self.tolerance = tolerance
        self.show_implied_values = show_implied_values

    def run(self, document_text: str, figures: Sequence[str], filing_type: str, filing_date: date,
            period_end: date) -> ExtractionResult:
        """Extract ``figures`` from ``document_text`` (one filing; values in the document's units)."""
        unknown = [f for f in figures if f not in FIGURE_LABELS]
        if unknown:
            raise ValueError(f"unknown figures {unknown}; known: {sorted(FIGURE_LABELS)}")
        store = SourceStore()
        doc_hash = store.add_document(document_text.encode("utf-8"), [document_text])
        found: dict[str, ExtractedFigure] = {}
        rounds: list[RepairRound] = []
        request = list(figures)
        instruction = "Read each figure from the document."
        for k in range(self.max_repair_rounds + 1):
            answers = self.extractor(document_text, request, instruction)
            for name in request:
                if name not in answers:
                    continue
                value, quote = float(answers[name][0]), str(answers[name][1])
                cit = CitationRef(doc_hash=doc_hash, filing_type=filing_type, filing_date=filing_date,
                                  period_end=period_end, page=None, raw_snippet=quote or "(none)")
                found[name] = ExtractedFigure(name=name, value=value, quote=quote,
                                              quote_in_document=bool(quote.strip()) and store.check(cit) == [],
                                              value_in_quote=value_in_quote(value, quote), citation=cit)
            missing = [f for f in figures if f not in found]
            unsupported = sorted(f for f, x in found.items() if not (x.quote_in_document and x.value_in_quote))
            supported = {f: x.value for f, x in found.items() if f not in unsupported}
            rnd = RepairRound(round=k + 1, requested=request, unsupported=unsupported + missing)
            diagnosed: set[str] = set()
            implied: dict[str, float] = {}
            idents = applicable_identities(supported, optional_absent_is_zero=self.fx_line_absent)
            unchecked = sorted(set(supported) - {f for c in idents.values() for f in c})
            if idents:
                d = diagnose_identities(supported, identities=idents, tolerance=self.tolerance)
                rnd.failed_identities = d.failed_identities
                rnd.candidates = [c.figures for c in d.candidates]
                diagnosed = {f for c in d.candidates for f in c.figures}
                implied = {f: v for c in d.candidates if c.implied_values for f, v in c.implied_values.items()}
            rounds.append(rnd)
            redo = sorted(set(unsupported) | set(missing) | diagnosed)
            if not redo:
                return ExtractionResult(figures=found, consistent=True, rounds=rounds, unchecked=unchecked,
                                        detail=f"consistent after {k + 1} round(s)")
            request = list(figures) if self.reread == "all" else redo
            parts = []
            if rnd.failed_identities:
                parts.append("These figures failed an accounting identity ("
                             + "; ".join(rnd.failed_identities) + "): re-read them.")
                if self.show_implied_values and implied:
                    parts.append("The other figures imply: "
                                 + ", ".join(f"{f} = {v:,.6g}" for f, v in implied.items() if f in redo) + ".")
            if unsupported or missing:
                parts.append("Quote the exact line each value appears on; some quotes or values were not found.")
            parts.append("Check you are reading the current-year column. Do not compute or adjust values.")
            instruction = " ".join(parts)
        return ExtractionResult(figures=found, consistent=False, unresolved=redo, rounds=rounds, unchecked=unchecked,
                                detail=f"abstaining on {', '.join(redo)} after {len(rounds)} round(s)")


def llm_extractor(client: BaseLLMClient, transcript: list[dict[str, Any]] | None = None) -> Extractor:
    """An extractor backed by a language model: JSON of {figure: {"value": number, "quote": excerpt}}.

    The model is asked to copy values and an exact excerpt (the row label and its numbers), never to
    compute; parsing failures yield no figure (the agent then re-asks or abstains). Provider errors are
    raised, not hidden. ``transcript`` (if given) receives each request's figures, the reply and token use.
    """

    def extract(document_text: str, figures: list[str], instruction: str) -> dict[str, tuple[float, str]]:
        wanted = "\n".join(f"- {f}: {FIGURE_LABELS[f]}" for f in figures)
        prompt = (f"{instruction}\nFrom the document below, give each figure exactly as printed (no arithmetic), "
                  "with an exact excerpt from the document that contains it: the row label and its numbers, "
                  "copied character for character. Values in the units printed. Use negative numbers for "
                  "amounts in parentheses.\n"
                  f"Figures:\n{wanted}\n\nAnswer with JSON only: "
                  '{"figure_name": {"value": number, "quote": "exact excerpt"}, ...}\n\n'
                  f"Document:\n{document_text}")
        resp = client.generate([LLMMessage(role="user", content=prompt)])
        reply = resp.content
        if transcript is not None:
            transcript.append({"figures": list(figures), "instruction": instruction, "reply": reply,
                               "prompt_tokens": resp.usage.prompt_tokens,
                               "completion_tokens": resp.usage.completion_tokens})
        try:
            data = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        except ValueError:
            return {}
        out: dict[str, tuple[float, str]] = {}
        for f in figures:
            item = data.get(f) if isinstance(data, dict) else None
            if isinstance(item, dict) and isinstance(item.get("value"), int | float) and isinstance(item.get("quote"), str):
                out[f] = (float(item["value"]), item["quote"])
        return out

    return extract
