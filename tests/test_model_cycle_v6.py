"""V6 development boundary must exclude retired evidence and deployment actions."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeolus.model_cycle_v4 import PROHIBITED_HISTORICAL_SEEDS
from aeolus.model_cycle_v5 import (
    V5_CALIBRATION_SEEDS,
    V5_FIT_SEEDS,
    V5_PROHIBITED_HISTORICAL_SEEDS,
    V5_VALIDATION_SEEDS,
)
from aeolus.model_cycle_v6 import (
    V6_CALIBRATION_SEEDS,
    V6_FIT_SEEDS,
    V6_PROHIBITED_HISTORICAL_SEEDS,
    V6_VALIDATION_SEEDS,
    V6DevelopmentRequest,
    validate_v6_development_request,
)


def _request(output_dir: Path, **changes: object) -> V6DevelopmentRequest:
    values: dict[str, object] = {
        "schema_version": "aeolus_sweep_v6",
        "suite_role": "development",
        "fit_seeds": V6_FIT_SEEDS,
        "calibration_seeds": V6_CALIBRATION_SEEDS,
        "validation_seeds": V6_VALIDATION_SEEDS,
        "output_dir": output_dir,
    }
    values.update(changes)
    return V6DevelopmentRequest(**values)


def test_v6_seed_sets_are_fresh_and_disjoint_from_retired_protocols(tmp_path: Path):
    request = _request(tmp_path / "out")

    validate_v6_development_request(request)

    all_v6 = set(V6_FIT_SEEDS + V6_CALIBRATION_SEEDS + V6_VALIDATION_SEEDS)
    assert len(all_v6) == len(V6_FIT_SEEDS + V6_CALIBRATION_SEEDS + V6_VALIDATION_SEEDS)
    assert not all_v6 & PROHIBITED_HISTORICAL_SEEDS
    assert not all_v6 & set(V5_FIT_SEEDS + V5_CALIBRATION_SEEDS + V5_VALIDATION_SEEDS)
    assert V6_PROHIBITED_HISTORICAL_SEEDS >= V5_PROHIBITED_HISTORICAL_SEEDS
    assert not all_v6 & V6_PROHIBITED_HISTORICAL_SEEDS


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "aeolus_sweep_v5"}, "schema"),
        ({"suite_role": "final"}, "development"),
        ({"fit_seeds": (1100,)}, "prohibited"),
        ({"calibration_seeds": (2100,)}, "disjoint"),
        ({"validation_seeds": (2300,)}, "predeclared"),
        ({"authorize_final_suite": True}, "final-suite"),
        ({"authorize_response_integration": True}, "response"),
    ],
)
def test_v6_request_rejects_retired_or_unsafe_protocol_inputs(
    tmp_path: Path, changes: dict[str, object], message: str
):
    with pytest.raises(ValueError, match=message):
        validate_v6_development_request(_request(tmp_path / "out", **changes))


def test_v6_request_rejects_nonempty_output_directory(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        validate_v6_development_request(_request(output_dir))
