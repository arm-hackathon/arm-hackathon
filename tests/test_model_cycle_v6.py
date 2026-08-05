"""V6 development runner contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeolus.model_cycle_v6 import V6DevelopmentRequest, validate_v6_development_request


REPOSITORY = Path(__file__).resolve().parents[1]
SWEEP = REPOSITORY / "scenarios" / "sweep-v6-development.json"


def _request(output_dir: Path, **changes: object) -> V6DevelopmentRequest:
    values: dict[str, object] = {
        "sweep_spec_path": SWEEP,
        "output_dir": output_dir,
    }
    values.update(changes)
    return V6DevelopmentRequest(**values)


def test_v6_request_derives_fit_calibration_and_validation_from_source_sweep(tmp_path: Path):
    request = _request(tmp_path / "out")

    spec = validate_v6_development_request(request)

    assert len(spec.sha256) == 64
    assert {family.role for family in spec.room_families} == {
        "fit",
        "calibration",
        "validation",
    }


def test_v6_request_rejects_nonempty_output_directory(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        validate_v6_development_request(_request(output_dir))


def test_v6_request_rejects_deployment_authorization(tmp_path: Path):
    with pytest.raises(ValueError, match="final-suite"):
        validate_v6_development_request(_request(tmp_path / "out", authorize_final_suite=True))
    with pytest.raises(ValueError, match="response"):
        validate_v6_development_request(_request(tmp_path / "out", authorize_response_integration=True))
