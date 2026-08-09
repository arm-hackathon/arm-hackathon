from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .scenario import Scenario, ScenarioValidationError


def _reject_json_constant(value: str) -> None:
    raise ScenarioValidationError(f"non-finite JSON constant: {value}")


def load_scenario_file(path: str | Path) -> Scenario:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        payload: Any = json.loads(text, parse_constant=_reject_json_constant)
    except UnicodeDecodeError as error:
        raise ScenarioValidationError("scenario must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ScenarioValidationError(
            f"invalid scenario JSON at line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ScenarioValidationError("scenario must be a JSON object")
    return Scenario.from_mapping(payload)
