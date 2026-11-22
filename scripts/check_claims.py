"""Flag over-claiming language in user-facing docs (roadmap 7.8, enterprise-quality track).

Līlāvatī is alpha software: its Z3 checks verify given numbers (they are not proofs over all inputs),
its point-in-time store limits what queries return (it does not guarantee an agent never leaks), and
it has not been independently audited. This script fails if README, CONTRIBUTING or docs use words
that say otherwise.

A flagged line is allowed when it negates the claim (e.g. "not independently audited", "never write
'proven'"), refers to this project's reference-value audit of an engine ("audited engines"), or ends
with the marker `claims-ok` (in an HTML comment) for a reviewed, legitimate use.

    uv run python scripts/check_claims.py            # exit 1 and list lines if anything is flagged
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = ["README.md", "CONTRIBUTING.md", "SECURITY.md"]
DOC_DIRS = ["docs"]
CODE_DIRS = ["src", "examples"]  # docstrings, comments and printed text are user-facing too

PATTERNS = {
    "proof language": r"\b(formally\s+(verif\w*|prov\w*)|theorem[- ]prov\w*|mathematically\s+(proven|guaranteed|audited)|proven|proves?|proved)\b",
    "guarantee": r"\bguarantee[sd]?\b|\bguaranteeing\b",
    "audit claim": r"\baudited\b",
    "maturity claim": r"\b(production[- ](grade|ready)|institutional[- ]grade|enterprise[- ]grade|battle[- ]tested)\b",
    "absolute claim": r"\b(zero\s+hallucinations?|hallucination[- ]free|bullet[- ]?proof|state[- ]of[- ]the[- ]art|world[- ]class|definitive)\b",
    "priority claim": r"\b(the\s+first|first[- ]ever)\s+(systematic|open|benchmark|framework|integration|mathematically|evaluation)\b"
                      r"|\b(earliest\s+recorded|history'?s\s+(first|earliest))\b",
}
ALLOW = re.compile(
    r"\b(not|never|no|without|cannot|isn't|aren't|doesn't|don't|un\w+)\b[^.]{0,60}"
    r"|audited\s+engines?|engines?\s+(?:not\s+yet\s+)?audited|reference-value audit|claims-ok"
    r"|^Audited \d{4}-\d{2}-\d{2}",  # validation pages: date of this project's reference-value audit
    re.IGNORECASE,
)


BIBLIOGRAPHY = re.compile(r"^\* \*\*[^*]+\*\* \((19|20)\d\d")  # "* **Author, A.** (2015). Title ..."


def targets(root: Path) -> list[Path]:
    paths = [root / f for f in FILES if (root / f).exists()]
    for d in DOC_DIRS:
        paths += sorted((root / d).rglob("*.md"))
    for d in CODE_DIRS:
        paths += sorted(p for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts)
    return paths


def scan(path: Path) -> list[tuple[int, str, str]]:
    hits = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "claims-ok" in line or BIBLIOGRAPHY.match(line):
            continue  # reviewed use, or another author's title in a reference list
        for label, pattern in PATTERNS.items():
            for m in re.finditer(pattern, line, re.IGNORECASE):
                window = line[max(0, m.start() - 70): m.end() + 20]
                if not ALLOW.search(window):
                    hits.append((n, label, line.strip()[:160]))
                    break
    return hits


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    total = 0
    for path in targets(root):
        for n, label, text in scan(path):
            total += 1
            print(f"{path.relative_to(root)}:{n}: [{label}] {text}")
    if total:
        print(f"\n{total} flagged line(s). Reword, or add a `claims-ok` HTML comment after review.")
        return 1
    print("no over-claiming language found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
