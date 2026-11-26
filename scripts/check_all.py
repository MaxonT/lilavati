"""Every check in one command: the gate before a pull request and at the close of a day's goal.

Runs lint, types, tests, the strict docs build, the claims linter, the tutorial code, the results
analysis and manifest checks, in that order, and prints one line per check. The paper check
(`check_papers_current.py`) is reported, not failed on: a milestone newer than a paper means "revise the
paper" (developer skill, section 4c), which is judged at the close of the day.

    uv run python scripts/check_all.py           # everything (a few minutes)
    uv run python scripts/check_all.py --fast    # skip the docs build and the tutorials
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass(frozen=True)
class Check:
    name: str
    command: Sequence[str]
    slow: bool = False
    report_only: bool = False


CHECKS: tuple[Check, ...] = (
    Check("lint", (PY, "-m", "ruff", "check", "src", "tests", "scripts", "examples")),
    Check("types", (PY, "-m", "mypy", "src")),
    Check("tests", (PY, "-m", "pytest", "-q", "-p", "no:cacheprovider")),
    Check("docs", ("uv", "run", "--extra", "docs", "--extra", "dev", "mkdocs", "build", "--strict"), slow=True),
    Check("claims", (PY, "scripts/check_claims.py")),
    Check("tutorials", (PY, "scripts/check_tutorials.py"), slow=True),
    Check("analysis", (PY, "scripts/analyse_results.py", "--check")),
    Check("manifest", (PY, "scripts/build_results_manifest.py", "--check")),
    Check("papers", (PY, "scripts/check_papers_current.py", "--strict"), report_only=True),
)


def run(checks: Sequence[Check], fast: bool = False, cwd: Path = ROOT) -> int:
    """Run ``checks`` in order; print a summary; return 1 if any check that counts failed."""
    failed: list[tuple[Check, str]] = []
    rows = []
    for c in checks:
        if fast and c.slow:
            rows.append((c.name, "skipped", 0.0))
            continue
        t0 = time.perf_counter()
        r = subprocess.run(list(c.command), cwd=cwd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        output = (r.stdout + r.stderr).strip()
        if r.returncode == 0:
            rows.append((c.name, "pass", dt))
        elif c.report_only:
            rows.append((c.name, "attention", dt))
            print(f"--- {c.name} (reported, not failed) ---\n{output[-1500:]}\n")
        else:
            rows.append((c.name, "FAIL", dt))
            failed.append((c, output))
    for c, output in failed:
        print(f"--- {c.name} failed: {' '.join(c.command)} ---\n{output[-3000:]}\n")
    print("check      result     seconds")
    for name, result, dt in rows:
        print(f"{name:10} {result:10} {dt:7.1f}")
    print("all checks passed" if not failed else f"{len(failed)} check(s) failed: {', '.join(c.name for c, _ in failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(CHECKS, fast="--fast" in sys.argv[1:]))
