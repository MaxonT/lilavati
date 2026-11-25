"""
Formal Invariant & SMT Reflection Prompts for Neuro-Symbolic Self-Correction.

When an identity check fails (e.g. Assets != Liabilities + Equity), these templates tell the agent
what failed and, given a diagnosis (``core.invariants.diagnose_identities``), which figures could be
wrong and the values that would make the identities hold, so it re-checks those figures against the
source instead of adjusting numbers to fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilavati.core.invariants import IdentityDiagnosis


def generate_smt_reflection_prompt(
    invariant_name: str,
    violations: Sequence[str],
    current_values: dict[str, float | str | bool],
    diagnosis: IdentityDiagnosis | None = None,
) -> str:
    """Constructs a feedback prompt detailing formal invariant violations (and, if given, a diagnosis)."""
    lines = [
        f"### CRITICAL FORMAL INVARIANT FAILURE: {invariant_name}",
        "",
        "The Z3 SMT solver found that the numbers in your extracted/projected financial model violate fundamental identities:",
        "",
    ]
    for idx, v in enumerate(violations, 1):
        lines.append(f"{idx}. {v}")

    lines.append("")
    lines.append("### EVALUATED PARAMETERS:")
    for k, val in current_values.items():
        lines.append(f"- {k}: {val}")

    if diagnosis is not None and diagnosis.candidates:
        lines.append("")
        lines.append("### WHICH FIGURES TO RE-CHECK (smallest explanations of every failure):")
        for c in diagnosis.candidates:
            implied = ("; would need " + ", ".join(f"{k} = {v:,.6g}" for k, v in c.implied_values.items())
                       if c.implied_values else "")
            lines.append(f"- {', '.join(c.figures)}{implied}")
        if len(diagnosis.candidates) > 1:
            lines.append("The identities cannot tell these apart: check each against the source; do not pick one "
                         "because its implied value looks plausible.")

    lines.append("")
    lines.append("### REQUIRED ACTION:")
    lines.append(
        "Re-examine the primary source disclosures, footnotes, and adjustments. "
        "Correct the line items so that the formal mathematical invariant holds strictly (SAT). "
        "Do NOT invent numbers; cite the source snippet or footnote reconciling the difference."
    )
    return "\n".join(lines)


def generate_citation_failure_prompt(
    failed_claims: Sequence[str],
) -> str:
    """Feedback prompt when claims lack cryptographic provenance."""
    lines = [
        "### CITATION PROVENANCE REJECTION",
        "",
        "The following quantitative claims failed cryptographic provenance verification:",
    ]
    for c in failed_claims:
        lines.append(f"- {c}")

    lines.append("")
    lines.append(
        "Every numerical claim must carry a CitationRef with a valid 64-character SHA-256 document hash, "
        "page number, and verbatim snippet. Provide valid citations or retract the unverified claims."
    )
    return "\n".join(lines)
