"""One-off migration (2026-11-22): recompute `case_hash` in stored result records with scheme v2.

Scheme v1 hashed the expected answer values too; their last digits come from platform math libraries,
so the same question hashed differently on macOS and Linux. Scheme v2
(`lilavati.bench.datasets.case_content_hash`) hashes only the question. This script rewrites only the
`case_hash` field of records that have one; replies, parsed answers, scores and every other field
are left byte-for-byte as they were (checked before writing).

    uv run python scripts/migrate_case_hashes.py          # then rebuild the manifest and analysis
"""

from __future__ import annotations

import json
from pathlib import Path

from lilavati.bench.datasets import case_content_hash, load_all_test_cases
from lilavati.bench.generator import generate_cases
from lilavati.bench.llm_eval import case_index

RESULTS = Path("results")


def main() -> None:
    legacy = json.loads((RESULTS / "legacy_metadata.json").read_text())
    for path in sorted(RESULTS.rglob("*.jsonl")):
        lines = path.read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        if not any("case_hash" in r for r in rows):
            continue
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else legacy[path.name]
        cases = (load_all_test_cases() if meta["case_set"] == "golden"
                 else generate_cases(int(meta["seed"]), int(meta["per_track"])))
        index = case_index(cases)
        out, changed = [], 0
        for r in rows:
            if "case_hash" in r:
                new_hash = case_content_hash(index[r["case"]])
                changed += new_hash != r["case_hash"]
                migrated = {**r, "case_hash": new_hash}
                assert {k: v for k, v in migrated.items() if k != "case_hash"} == \
                       {k: v for k, v in r.items() if k != "case_hash"}
                r = migrated
            out.append(json.dumps(r, default=str))
        path.write_text("\n".join(out) + "\n")
        if meta_path.exists():
            meta["case_hash_scheme"] = "v2 (recomputed 2026-11-22 by scripts/migrate_case_hashes.py)"
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{path}: {changed} of {len(rows)} case hashes updated")


if __name__ == "__main__":
    main()
