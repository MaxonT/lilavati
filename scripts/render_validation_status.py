"""Render the validation-status page and the docs home pillar table from one source.

Source: `src/lilavati/validation_status.py` (`ENGINES`). Outputs:

- `docs/validation/status.md` (whole file);
- the pillar table in `docs/index.md`, between the markers below.

    uv run python scripts/render_validation_status.py          # write
    uv run python scripts/render_validation_status.py --check  # exit 1 if a file is out of date (CI)
"""

from __future__ import annotations

import sys
from pathlib import Path

from lilavati.validation_status import render_pillar_table, render_status_page

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "validation" / "status.md"
INDEX = ROOT / "docs" / "index.md"
BEGIN = "<!-- BEGIN generated: pillar table (scripts/render_validation_status.py) -->"
END = "<!-- END generated: pillar table -->"


def rendered_index(text: str) -> str:
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    return text[:start] + "\n" + render_pillar_table() + text[end:]


def main(check: bool) -> int:
    targets = {STATUS: render_status_page(), INDEX: rendered_index(INDEX.read_text(encoding="utf-8"))}
    stale = [p for p, want in targets.items() if not p.exists() or p.read_text(encoding="utf-8") != want]
    if check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT)} (run scripts/render_validation_status.py)")
        return 1 if stale else 0
    for p in stale:
        p.write_text(targets[p], encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
