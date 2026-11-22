from importlib.metadata import version

import lilavati


def test_distribution_name_and_version() -> None:
    assert version("lilavati-finance") == "0.2.0"
    assert lilavati.__name__ == "lilavati"


def test_version_is_the_same_everywhere() -> None:
    # pyproject, the package, the citation file and the API default must agree; the release
    # workflow also checks that the git tag matches pyproject.
    import inspect
    import re
    import tomllib
    from pathlib import Path

    from lilavati.framework.serving import create_agent_api

    root = Path(__file__).resolve().parents[1]
    pv = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    cff = re.search(r"^version: (\S+)", (root / "CITATION.cff").read_text(), re.M)
    assert cff and cff.group(1) == pv
    assert lilavati.__version__ == pv
    assert inspect.signature(create_agent_api).parameters["version"].default == pv


def test_sdist_excludes_private_task_briefs() -> None:
    # .agents/tasks/ is excluded from git via .git/info/exclude, which hatch ignores; without an
    # explicit sdist exclude, a PyPI release would publish the maintainer's agent task briefs.
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert ".agents/tasks" in cfg["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
