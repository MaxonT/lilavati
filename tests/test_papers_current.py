"""scripts/check_papers_current.py lists milestones newer than each paper (developer skill, section 4c)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_papers_current", ROOT / "scripts" / "check_papers_current.py")
assert spec and spec.loader
cpc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cpc  # dataclasses need their module registered
spec.loader.exec_module(cpc)


def test_reports_milestones_after_the_paper_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("**Status:** Working paper. Version of 20 November 2026.\n")
    bench = tmp_path / "bench.md"
    bench.write_text("Version of 26 November 2026.\n")
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "- [x] Old engine (2026-11-19).\n"
        "- [x] New engine, started 2026-11-19,\n"
        "      finished 2026-11-25 on a continuation line.\n"
        "- [ ] Planned item (2026-12-01).\n"
        "- [~] Benchmark generator, partly done 2026-11-27.\n"
        "- [x] Undated item.\n")
    monkeypatch.setattr(cpc, "PAPERS", {"framework": paper, "benchmark": bench})
    monkeypatch.setattr(cpc, "SOURCES", [roadmap])
    late = cpc.behind()
    assert [(m.done.isoformat(), m.status, m.line) for m in late["framework"]] == [("2026-11-25", "x", 2)]
    assert [m.done.isoformat() for m in late["benchmark"]] == ["2026-11-27"]  # after 26 November
    monkeypatch.setattr("sys.argv", ["check_papers_current.py", "--strict"])
    assert cpc.main() == 1


def test_repository_papers_parse() -> None:
    assert all(cpc.paper_version(p) for p in cpc.PAPERS.values())
    assert any(m.done for src in cpc.SOURCES for m in cpc.milestones(src))
