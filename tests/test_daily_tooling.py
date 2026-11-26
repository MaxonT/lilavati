"""The daily-goal tooling: scripts/next_goal.py and scripts/check_all.py (lilavati-daily skill)."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need their module registered
    spec.loader.exec_module(mod)
    return mod


ng = _load("next_goal")
ca = _load("check_all")

TODO = """# TODO
- [ ] Phase-order feature, later in the file.
- [ ] Release on PyPI. *Blocked: maintainer action.*
- [ ] Audit the order book against a reference (7.3).
- [~] Umbrella item with sub-items:
  - [ ] Child one, fix the bug where "(1)" is read as -1.
  - [ ] Child two. *Blocked: needs a file downloaded by hand.*
- [ ] Black-Litterman audit. *Due: 2026-12-01 check issues first.*
- [x] Done item (7.4).
- [ ] A long item; with many; parts, a, b, c, d, e (7.14–7.16).
"""
ROADMAP = """- [ ] 7.3 Engine audits.
- [ ] 7.15 Look-ahead track.
- [ ] 7.21b CI job (needs a CI secret). *Blocked: needs a CI secret.*
- [ ] 1.2 Macro connector.
- [x] 7.4 Purged CV.
"""


def test_next_goal_ranks_unblocked_items(tmp_path: Path) -> None:
    todo = tmp_path / "TODO.md"
    todo.write_text(TODO)
    cands = ng.candidates(todo)
    texts = [c.text for c in cands]
    assert len(ng.items(todo)) == 7  # the umbrella is skipped; its two children count
    assert all("Blocked" not in t for t in texts) and len(cands) == 5
    assert cands[0].due == date(2026, 12, 1)  # due first
    assert "read as" in cands[1].text and cands[1].reason(date(2026, 11, 26)) == "correctness"
    assert "order book" in cands[2].text and cands[2].reason(date(2026, 11, 26)) == "validation"
    assert cands[3].text.startswith("- [ ] Phase-order feature")  # then file order
    assert cands[4].text.startswith("- [ ] A long item")
    long_item = next(i for i in ng.items(todo) if i.text.startswith("- [ ] A long item"))
    assert long_item.size == "split"


def test_roadmap_items_missing_from_todo(tmp_path: Path) -> None:
    todo, roadmap = tmp_path / "TODO.md", tmp_path / "ROADMAP.md"
    todo.write_text(TODO)
    roadmap.write_text(ROADMAP)
    missing = ng.roadmap_without_todo(roadmap, todo)
    assert missing == ["- [ ] 1.2 Macro connector."]  # 7.3 cited, 7.15 in a range, 7.21b blocked, 7.4 done


def test_repository_todo_marks_what_is_blocked() -> None:
    # An open item that says it waits on reviewers, budget or the maintainer must carry the marker, so
    # next_goal.py does not offer it as a day's goal.
    waiting = re.compile(r"after reviewer feedback|needs api budget|deferred by the maintainer|maintainer action",
                         re.IGNORECASE)
    unmarked = [i.line for i in ng.items() if waiting.search(i.text) and not i.blocked]
    assert unmarked == [], f"TODO.md items {unmarked} need '*Blocked: <reason>.*'"
    assert ng.candidates(), "no unblocked candidate left in TODO.md"


def test_check_all_counts_failures_and_reports_papers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ok = ca.Check("ok", (sys.executable, "-c", "pass"))
    bad = ca.Check("bad", (sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"))
    slow = ca.Check("slow", (sys.executable, "-c", "import sys; sys.exit(1)"), slow=True)
    note = ca.Check("papers", (sys.executable, "-c", "import sys; print('behind'); sys.exit(1)"), report_only=True)
    assert ca.run([ok, note], cwd=tmp_path) == 0  # a report-only check never fails the gate
    assert ca.run([ok, bad], cwd=tmp_path) == 1
    assert ca.run([ok, slow], fast=True, cwd=tmp_path) == 0  # skipped with --fast
    out = capsys.readouterr().out
    assert "boom" in out and "behind" in out and "1 check(s) failed: bad" in out


def test_check_all_lists_every_gate() -> None:
    names = [c.name for c in ca.CHECKS]
    assert names == ["lint", "types", "tests", "docs", "claims", "tutorials", "analysis", "manifest", "papers"]
