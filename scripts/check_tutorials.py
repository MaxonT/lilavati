"""Run the Python code in the tutorials exactly as a reader would copy it (CI job `tutorials`).

Conventions in `docs/tutorials/*.md`:

- A fenced block ```` ```python title="credit_agent.py" ```` is part of a file: blocks with the same title
  are concatenated into that file in a scratch directory.
- Untitled ```` ```python ```` blocks are run one by one from the repository root, with the scratch
  directory on `PYTHONPATH` so they can import the titled files.
- Titled files named `tests/test_*.py` are run with pytest.
- Blocks that need an API key (they mention `*_API_KEY`) are skipped and reported, never run.

Tutorials not yet retrofitted to the tutorial standard are listed in PENDING and skipped.

    uv run python scripts/check_tutorials.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PENDING: set[str] = set()  # tutorials not yet at the tutorial standard (all retrofitted 2026-11-22)
FENCE = re.compile(r'```python(?:\s+title="(?P<title>[^"]+)")?\n(?P<code>.*?)```', re.S)
NEEDS_KEY = re.compile(r"\b[A-Z]+_API_KEY\b")


def check(tutorial: Path) -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        runs: list[tuple[int, str]] = []
        for i, m in enumerate(FENCE.finditer(tutorial.read_text(encoding="utf-8"))):
            code, title = m.group("code"), m.group("title")
            if NEEDS_KEY.search(code):
                print(f"  SKIP block {i} (needs an API key)")
                continue
            if title:
                target = work / title
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(code + "\n\n")
            else:
                runs.append((i, code))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(work), os.environ.get("PYTHONPATH", "")])}
        for i, code in runs:
            r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=300)
            status = "ok" if r.returncode == 0 else "FAIL"
            print(f"  {status:4} block {i}")
            if r.returncode:
                failures.append(f"{tutorial.name} block {i}: {r.stderr.strip()[-500:]}")
        for test in sorted(work.glob("tests/test_*.py")):
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                                "--rootdir", str(work), str(test)], cwd=work, env=env,
                               capture_output=True, text=True, timeout=300)
            status = "ok" if r.returncode == 0 else "FAIL"
            print(f"  {status:4} pytest {test.relative_to(work)}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
            if r.returncode:
                failures.append(f"{tutorial.name} {test.name}: {r.stdout.strip()[-500:]}")
    return failures


def main() -> int:
    failures: list[str] = []
    for tutorial in sorted((ROOT / "docs" / "tutorials").glob("*.md")):
        if tutorial.name in PENDING:
            print(f"{tutorial.name}: pending retrofit (roadmap 7.24), not checked")
            continue
        print(tutorial.name)
        failures += check(tutorial)
    if failures:
        print("\n" + "\n\n".join(failures))
        return 1
    print("all tutorial code ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
