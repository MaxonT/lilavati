"""results/manifest.json must describe the result files exactly (hashes, case sets, settings)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("build_results_manifest", ROOT / "scripts" / "build_results_manifest.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_is_current(monkeypatch: pytest.MonkeyPatch) -> None:
    if not any((ROOT / "results").glob("*.jsonl")):
        pytest.skip("no result files in this checkout")
    monkeypatch.chdir(ROOT)
    built = _builder().build()
    stored = json.loads((ROOT / "results" / "manifest.json").read_text())
    assert built == stored, "results/manifest.json is stale: run scripts/build_results_manifest.py"


def test_no_secrets_in_results() -> None:
    import re

    pattern = re.compile(r"AIza[0-9A-Za-z_-]{30}|gsk_[A-Za-z0-9]{20}|sk-[A-Za-z0-9_-]{20}|Bearer ")
    for path in (ROOT / "results").rglob("*"):
        if path.is_file():
            assert not pattern.search(path.read_text(errors="ignore")), path
