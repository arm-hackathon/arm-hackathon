"""Shared fixtures: the repo's standard scenario file and its parsed dict."""

import copy
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_SCENARIO_PATH = REPO_ROOT / "scenarios" / "standard_habitat.json"


@pytest.fixture
def standard_scenario_path() -> Path:
    return STANDARD_SCENARIO_PATH


@pytest.fixture
def standard_doc() -> dict:
    """Deep copy of the standard habitat JSON as a dict, safe to mutate."""
    return copy.deepcopy(json.loads(STANDARD_SCENARIO_PATH.read_text(encoding="utf-8")))
