"""The claims linter (scripts/check_claims.py) flags over-claims and allows negations and citations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_claims", ROOT / "scripts" / "check_claims.py")
assert spec and spec.loader
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)


def _flags(tmp_path: Path, text: str) -> list[str]:
    f = tmp_path / "doc.md"
    f.write_text(text)
    return [label for _, label, _ in claims.scan(f)]


def test_flags_proof_guarantee_and_maturity_language(tmp_path: Path) -> None:
    assert "proof language" in _flags(tmp_path, "The covenant is formally proven by Z3.")
    assert "guarantee" in _flags(tmp_path, "This guarantees zero look-ahead bias.")
    assert "maturity claim" in _flags(tmp_path, "A production-ready agent.")
    assert "audit claim" in _flags(tmp_path, "An audited and verifiable framework.")
    assert "priority claim" in _flags(tmp_path, "History's earliest recorded treatise on auditing.")


def test_allows_negations_markers_and_bibliography(tmp_path: Path) -> None:
    assert _flags(tmp_path, "Outputs are not independently audited.") == []
    assert _flags(tmp_path, "Say checked, never write proven.") == []
    assert _flags(tmp_path, "Formal theorem proving <!-- claims-ok: field name -->") == []
    assert _flags(tmp_path, "* **Paulson, L. C.** (1994). *Isabelle: A Generic Theorem Prover*.") == []
    # APA reference entries (papers): another author's title, not a claim
    assert _flags(tmp_path, "Paulson, L. C. (1994). *Isabelle: A generic theorem prover*. Springer.") == []
    assert _flags(tmp_path, "Colebrooke, H. T. (Trans.). (1817). *Algebra with arithmetic*. John Murray.") == []
    assert _flags(tmp_path, "de Moura, L., & Kong, S. (2015). The Lean theorem prover. In *CADE-25*.") == []
    # prose that cites in passing is still checked
    assert _flags(tmp_path, "Our method is proven correct (Paulson, 1994).") != []


def test_repository_is_clean() -> None:
    assert claims.main() == 0
