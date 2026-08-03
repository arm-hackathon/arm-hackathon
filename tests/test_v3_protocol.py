"""Focused v3 development/final protocol contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.sweep import generate_sweep, parse_sweep_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _v3_document(role: str) -> dict:
    telemetry = {
        "airflow_noise_fraction": 0.01,
        "airflow_bias_fraction": 0.01,
        "airflow_drift_fraction": 0.01,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.01,
        "co2_sensor_bias_fraction": 0.01,
        "co2_sensor_drift_fraction": 0.01,
    }
    split_names = ("train", "validation") if role == "development" else ("final",)
    return {
        "schema_version": "aeolus_sweep_v3",
        "suite_role": role,
        "base_scenario": "standard_habitat.json",
        "targets": ["cabin_a"],
        "splits": {
            split: {
                "seeds": [100 + index],
                "fault_start_ticks": [25],
                "operating_profiles": [
                    {
                        "id": split,
                        "source_multiplier": 1.0,
                        "shared_airflow_capacity": 30.0,
                        "telemetry": telemetry,
                    }
                ],
                "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
                "blocked_effectiveness": [0.65],
            }
            for index, split in enumerate(split_names)
        },
    }


def test_v3_requires_role_specific_exact_split_set():
    source = REPO_ROOT / "scenarios" / "sweep-v3-development.json"
    development = _v3_document("development")
    parsed = parse_sweep_spec(development, source_path=source)
    assert parsed.suite_role == "development"
    assert tuple(parsed.splits) == ("train", "validation")

    mixed = _v3_document("development")
    mixed["splits"]["final"] = mixed["splits"]["validation"]
    with pytest.raises(ValueError, match="unexpected field 'final'"):
        parse_sweep_spec(mixed, source_path=source)

    retired = _v3_document("final")
    retired["splits"]["test"] = retired["splits"]["final"]
    with pytest.raises(ValueError, match="unexpected field 'test'"):
        parse_sweep_spec(retired, source_path=source)


def test_checked_in_v3_specs_have_roles_and_disjoint_fresh_final_seeds(tmp_path: Path):
    development_path = REPO_ROOT / "scenarios" / "sweep-v3-development.json"
    final_path = REPO_ROOT / "scenarios" / "sweep-v3-final.json"
    development = parse_sweep_spec(
        json.loads(development_path.read_text(encoding="utf-8")), source_path=development_path
    )
    final = parse_sweep_spec(
        json.loads(final_path.read_text(encoding="utf-8")), source_path=final_path
    )

    assert development.suite_role == "development"
    assert final.suite_role == "final"
    assert set(final.splits["final"].seeds).isdisjoint(
        {seed for split in development.splits.values() for seed in split.seeds}
    )
    receipt = generate_sweep(final_path, tmp_path / "final-sweep")
    assert receipt["families_by_split"] == {"final": 180}
