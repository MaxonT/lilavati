"""Every tutorial is reachable and complete (developer skill, section 4b).

A tutorial must be in the docs navigation, the README tutorials table, the docs home list and a week of
the course schedule, and must have the standard sections. The README's tutorial count must be right.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = sorted((ROOT / "docs" / "tutorials").glob("[0-9][0-9]_*.md"))
REQUIRED = ["**Who this is for:**", "**Time:**", "## Learning objectives", "## Where this fits",
            "## Exercises", "## For instructors"]
# headings that may carry a step number ("## 6. What this does and does not show")
ONE_OF = [r"^## (?:\d+\. )?(?:What can go wrong|What this does and does not show)", r"^## Using this with Claude Code"]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_tutorials_are_numbered_without_gaps() -> None:
    numbers = [int(t.name[:2]) for t in TUTORIALS]
    assert numbers == list(range(1, len(TUTORIALS) + 1))


@pytest.mark.parametrize("tutorial", TUTORIALS, ids=lambda p: p.name[:2])
def test_tutorial_is_listed_everywhere(tutorial: Path) -> None:
    name = tutorial.name
    places = {"mkdocs.yml": f"tutorials/{name}", "README.md": f"docs/tutorials/{name}",
              "docs/index.md": f"tutorials/{name}", "docs/course/index.md": f"../tutorials/{name}"}
    missing = [where for where, link in places.items() if link not in read(where)]
    assert not missing, f"{name} is not linked from {missing}"


@pytest.mark.parametrize("tutorial", TUTORIALS, ids=lambda p: p.name[:2])
def test_tutorial_has_the_standard_sections(tutorial: Path) -> None:
    text = tutorial.read_text(encoding="utf-8")
    missing = [s for s in REQUIRED if s not in text]
    missing += [pattern for pattern in ONE_OF if not re.search(pattern, text, re.M)]
    assert not missing, f"{tutorial.name} lacks {missing}"


def test_readme_tutorial_count() -> None:
    m = re.search(r"Every feature comes with a tutorial \((\w+) so far\)", read("README.md"))
    words = {"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    assert m and words.get(m.group(1), -1) == len(TUTORIALS), "update the tutorial count in README.md"
