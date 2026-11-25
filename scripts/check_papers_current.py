"""Which milestones are newer than the papers? (developer skill, section 4c)

Reads each paper's "Version of <day> <Month> <year>" line in docs/papers/, then every completed (`[x]`) or
partly completed (`[~]`) item in ROADMAP.md and TODO.md that carries an ISO date, and lists the items
dated after the paper's version: the milestones the paper does not yet reflect. Items are suggested for
the benchmark paper when they mention the benchmark, its cases, baselines or results, otherwise for the
framework paper; the suggestion is a hint, not a rule.

    uv run python scripts/check_papers_current.py            # report (exit 0)
    uv run python scripts/check_papers_current.py --strict   # exit 1 if a paper is behind
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = {"framework": ROOT / "docs" / "papers" / "01_framework_paper.md",
          "benchmark": ROOT / "docs" / "papers" / "02_benchmark_paper.md"}
SOURCES = [ROOT / "ROADMAP.md", ROOT / "TODO.md"]
VERSION = re.compile(r"Version of (\d{1,2} [A-Z][a-z]+ \d{4})")
ITEM = re.compile(r"^(\s*)- \[(x|~)\]")
ISO = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
BENCHMARK = re.compile(r"bench|baseline|tool arm|tool-augmented|leaderboard|golden case|generated case|results/",
                       re.IGNORECASE)


@dataclass(frozen=True)
class Milestone:
    source: str
    line: int
    done: date
    status: str  # "x" done, "~" partly done
    text: str

    @property
    def paper(self) -> str:
        return "benchmark" if BENCHMARK.search(self.text) else "framework"


def paper_version(path: Path) -> date:
    m = VERSION.search(path.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"{path.name}: no 'Version of <day> <Month> <year>' line")
    return datetime.strptime(m.group(1), "%d %B %Y").date()


def milestones(path: Path) -> list[Milestone]:
    """Completed or partly completed items with an ISO date; an item runs until the next list item."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Milestone] = []
    i = 0
    while i < len(lines):
        m = ITEM.match(lines[i])
        if not m:
            i += 1
            continue
        indent, status, start = len(m.group(1)), m.group(2), i
        block = [lines[i]]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^\s*(- |#)", lines[i]) \
                and len(lines[i]) - len(lines[i].lstrip()) > indent:
            block.append(lines[i])
            i += 1
        text = " ".join(s.strip() for s in block)
        dates = [date.fromisoformat(d) for d in ISO.findall(text)]
        if dates:
            out.append(Milestone(path.name, start + 1, max(dates), status, text))
    return out


def behind() -> dict[str, list[Milestone]]:
    versions = {name: paper_version(p) for name, p in PAPERS.items()}
    items = [m for src in SOURCES for m in milestones(src)]
    return {name: sorted((m for m in items if m.paper == name and m.done > v), key=lambda m: m.done)
            for name, v in versions.items()}


def main() -> int:
    late = behind()
    for name, items in late.items():
        v = paper_version(PAPERS[name])
        if not items:
            print(f"{name} paper (version {v}): no dated milestone after it")
            continue
        print(f"{name} paper (version {v}): {len(items)} milestone(s) after it")
        for m in items:
            mark = "done" if m.status == "x" else "partly done"
            print(f"  {m.done} {mark:11} {m.source}:{m.line}  {m.text[:110]}")
    return 1 if "--strict" in sys.argv[1:] and any(late.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
