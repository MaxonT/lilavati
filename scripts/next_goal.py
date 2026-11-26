"""Candidates for today's goal: the open TODO items that are not blocked, ranked (lilavati-daily skill).

TODO.md is the queue (ROADMAP.md is the plan). An open item is `- [ ]` or `- [~]` with its continuation
lines; a parent whose sub-items are listed is an umbrella and is skipped in favour of its children.
Markers in the item text:

- `*Blocked: <reason>.*`: needs the maintainer, reviewers, budget, a key or a file downloaded by hand;
  excluded.
- `*Due: YYYY-MM-DD ...*`: ranked first, soonest first.

Ranking after due items: correctness (a bug, or a check that holds by construction), then validation
(audits, reference checks), then the order in TODO.md (phase order). Long items are flagged "split":
pick one slice for the day and record it. The script also lists numbered ROADMAP items that are open but
not referenced in TODO.md, so the two stay in step.

    uv run python scripts/next_goal.py           # top 5 candidates
    uv run python scripts/next_goal.py --all     # every unblocked item
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "TODO.md"
ROADMAP = ROOT / "ROADMAP.md"
OPEN = re.compile(r"^(\s*)- \[( |~)\]")
ANY_ITEM = re.compile(r"^(\s*)- ")
BLOCKED = re.compile(r"\*Blocked: [^*]+\*")
DUE = re.compile(r"\*Due: (20\d\d-\d\d-\d\d)")
CORRECTNESS = re.compile(r"\b(bugs?|incorrect|by construction|read as)\b", re.IGNORECASE)
VALIDATION = re.compile(r"\b(audits?|reference|validat\w*|checked against)\b", re.IGNORECASE)
SPLIT_AT = 350  # characters: longer items are several days of work
ROADMAP_ITEM = re.compile(r"^- \[( |~)\] \**(\d{1,2}\.\d{1,2}[a-z]?)\b")
ROADMAP_REF = re.compile(r"\b(\d{1,2})\.(\d{1,2})([a-z]?)(?:\s*[–-]\s*(\d{1,2})\.(\d{1,2}))?\b")


@dataclass(frozen=True)
class Item:
    line: int
    status: str  # " " open, "~" partly done
    text: str
    blocked: str | None
    due: date | None

    @property
    def size(self) -> str:
        """"split" for several days of work: long, or a list of tasks (two or more ";", four or more ",")."""
        many = self.text.count(";") >= 2 or self.text.count(",") >= 4
        return "split" if len(self.text) > SPLIT_AT or many else "day"

    def rank(self) -> tuple[int, str, int]:
        if self.due:
            return (0, self.due.isoformat(), self.line)
        if CORRECTNESS.search(self.text):
            return (1, "", self.line)
        if VALIDATION.search(self.text):
            return (2, "", self.line)
        return (3, "", self.line)

    def reason(self, today: date) -> str:
        tier = self.rank()[0]
        if tier == 0 and self.due:
            days = (self.due - today).days
            return f"due {self.due} ({'overdue' if days < 0 else f'in {days} days'})"
        return {1: "correctness", 2: "validation", 3: "phase order"}[tier]


def items(path: Path = TODO) -> list[Item]:
    """Open items with their continuation lines; umbrellas (with listed sub-items) are skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Item] = []
    i = 0
    while i < len(lines):
        m = OPEN.match(lines[i])
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
        child = ANY_ITEM.match(lines[i]) if i < len(lines) else None
        if child and len(child.group(1)) > indent:
            continue  # an umbrella: its sub-items are the candidates
        text = " ".join(s.strip() for s in block)
        b, d = BLOCKED.search(text), DUE.search(text)
        out.append(Item(start + 1, status, text, b.group(0) if b else None,
                        date.fromisoformat(d.group(1)) if d else None))
    return out


def candidates(path: Path = TODO) -> list[Item]:
    return sorted((it for it in items(path) if not it.blocked), key=Item.rank)


def roadmap_without_todo(roadmap: Path = ROADMAP, todo: Path = TODO) -> list[str]:
    """Open numbered ROADMAP items (e.g. 7.14) that TODO.md does not reference."""
    refs: set[str] = set()
    for m in ROADMAP_REF.finditer(todo.read_text(encoding="utf-8")):
        major, minor, suffix, end_major, end_minor = m.groups()
        refs.add(f"{int(major)}.{int(minor)}{suffix}")
        if end_major and end_major == major:  # a range such as 7.14–7.16
            refs.update(f"{int(major)}.{k}" for k in range(int(minor), int(end_minor) + 1))
    out = []
    for line in roadmap.read_text(encoding="utf-8").splitlines():
        m = ROADMAP_ITEM.match(line)
        if m and m.group(2) not in refs and not BLOCKED.search(line):
            out.append(line.strip()[:110])
    return out


def main() -> int:
    today = date.today()
    all_items = items()
    cands = candidates()
    shown = cands if "--all" in sys.argv[1:] else cands[:5]
    print(f"{len(cands)} unblocked of {len(all_items)} open TODO items ({len(all_items) - len(cands)} blocked)\n")
    for k, it in enumerate(shown, 1):
        mark = "partly done" if it.status == "~" else "open"
        print(f"{k}. TODO.md:{it.line} [{mark}; {it.reason(today)}; {it.size}]")
        print(f"   {it.text.removeprefix('- [ ] ').removeprefix('- [~] ')[:200]}")
    missing = roadmap_without_todo()
    if missing:
        print(f"\nOpen ROADMAP items with no TODO entry ({len(missing)}): add them to TODO.md or mark them blocked")
        for line in missing:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
