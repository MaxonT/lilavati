from importlib.metadata import version

import lilavati


def test_distribution_name_and_version() -> None:
    assert version("lilavati-finance") == "0.1.0"
    assert lilavati.__name__ == "lilavati"
