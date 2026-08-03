"""Response-evidence harness tests."""

import json

from aeolus.response_evidence import (
    metrics_for_records,
    response_latency_ticks,
    run_response_evidence,
)
from aeolus.scenario import run_scenario

MINI_SWEEP = {
    "schema_version": "aeolus_sweep_v3",
    "base_scenario": "standard_habitat.json",
    "targets": ["cabin_a"],
    "suite_role": "development",
    "splits": {
        "train": {
            "seeds": [100],
            "fault_start_ticks": [15],
            "operating_profiles": [
                {
                    "id": "primary-low",
                    "source_multiplier": 0.8,
                    "shared_airflow_capacity": 24.0,
                    "telemetry": {
                        "airflow_noise_fraction": 0.025,
                        "airflow_bias_fraction": 0.01,
                        "airflow_drift_fraction": 0.01,
                        "actuator_position_noise_fraction": 0.01,
                        "co2_sensor_noise_fraction": 0.02,
                        "co2_sensor_bias_fraction": 0.01,
                        "co2_sensor_drift_fraction": 0.015,
                    },
                }
            ],
            "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
            "blocked_effectiveness": [0.65],
        },
        "validation": {
            "seeds": [500],
            "fault_start_ticks": [15],
            "operating_profiles": [
                {
                    "id": "primary-low",
                    "source_multiplier": 0.8,
                    "shared_airflow_capacity": 24.0,
                    "telemetry": {
                        "airflow_noise_fraction": 0.025,
                        "airflow_bias_fraction": 0.01,
                        "airflow_drift_fraction": 0.01,
                        "actuator_position_noise_fraction": 0.01,
                        "co2_sensor_noise_fraction": 0.02,
                        "co2_sensor_bias_fraction": 0.01,
                        "co2_sensor_drift_fraction": 0.015,
                    },
                }
            ],
            "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
            "blocked_effectiveness": [0.65],
        },
    },
}


def _write_mini_sweep(tmp_path, standard_scenario_path, document=MINI_SWEEP):
    import shutil

    shutil.copy(standard_scenario_path, tmp_path / "standard_habitat.json")
    spec_path = tmp_path / "sweep.json"
    spec_path.write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    return spec_path


def test_metrics_are_bounded_and_deterministic(standard_scenario_path):
    from aeolus.config import load_scenario

    config = load_scenario(standard_scenario_path)
    records = run_scenario(config)
    zone_ids = [zone.id for zone in config.non_processing_zones()]
    first = metrics_for_records(
        records, ceiling=0.30, zone_ids=zone_ids
    )
    second = metrics_for_records(
        records, ceiling=0.30, zone_ids=zone_ids
    )
    assert first == second
    assert 0 <= first["time_above_ceiling"] <= len(records) * len(zone_ids)
    assert first["energy"] > 0.0
    assert first["invariant_violations"] == 0


def test_response_latency_uses_onset_window():
    history = [
        {"cabin_a": {"reason": "nominal", "commanded": 0.1}},
        {"cabin_a": {"reason": "nominal", "commanded": 0.1}},
        {"cabin_a": {"reason": "nominal", "commanded": 0.1}},
        {"cabin_a": {"reason": "degraded_spare_release", "commanded": 0.05}},
    ]
    assert response_latency_ticks(history, onset_tick=3) == 1
    assert response_latency_ticks(history, onset_tick=4) == 0
    assert response_latency_ticks(history, onset_tick=10) is None


def test_end_to_end_receipt_is_canonical(tmp_path, standard_scenario_path):
    spec_path = _write_mini_sweep(tmp_path, standard_scenario_path)
    output = tmp_path / "evidence"
    receipt = run_response_evidence(spec_path, output)
    assert receipt["evidence_version"] == "aeolus_response_evidence_v1"
    assert receipt["families_evaluated"] == 6
    assert len(receipt["per_family"]) == 6
    assert len(receipt["evidence_sha256"]) == 64
    for row in receipt["per_family"]:
        assert row["fault"]["time_above_ceiling"]["delta"] is not None
        assert set(row["governed_action_ticks"]) <= {
            "nominal",
            "frozen_hold",
            "degraded_spare_release",
            "bounded_rate",
        }
    aggregate = receipt["aggregate"]
    assert aggregate["causality_margin"]["margin_ticks"] == 1
    assert aggregate["invariant_violations"]["governed_total"] == 0
    assert (output / "response-evidence.json").exists()
    on_disk = json.loads(
        (output / "response-evidence.json").read_text(encoding="utf-8")
    )
    assert on_disk["evidence_sha256"] == receipt["evidence_sha256"]


def test_deterministic_receipt(tmp_path, standard_scenario_path):
    spec_path = _write_mini_sweep(tmp_path, standard_scenario_path)
    first = run_response_evidence(spec_path, tmp_path / "a")
    second = run_response_evidence(spec_path, tmp_path / "b")
    assert first["evidence_sha256"] == second["evidence_sha256"]


def test_output_dir_must_be_empty(tmp_path, standard_scenario_path):
    spec_path = _write_mini_sweep(tmp_path, standard_scenario_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "leftover.txt").write_text("x", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="not empty"):
        run_response_evidence(spec_path, output)