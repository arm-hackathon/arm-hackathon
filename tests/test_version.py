from pathlib import Path
import tomllib

import aeolus


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_version_matches_project_metadata() -> None:
    with (ROOT / "pyproject.toml").open("rb") as source:
        project_version = tomllib.load(source)["project"]["version"]

    assert project_version == aeolus.__version__ == "0.8.0"
