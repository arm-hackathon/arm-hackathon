"""Response-evidence harness tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

from aeolus.response import BoundedRecoveryGovernor, ResponseSettings
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


def test_response_latency_annotation_allows_missing_onset():
    history = [{"cabin_a": {"reason": "nominal", "commanded": 0.1}}]
    assert get_type_hints(response_latency_ticks)["onset_tick"] == int | None
    assert response_latency_ticks(history, onset_tick=None) is None


def test_response_latency_scoped_to_affected_zones():
    history = [
        {"cabin_a": {"reason": "nominal", "commanded": 0.1},
         "cabin_b": {"reason": "nominal", "commanded": 0.1}},
        {"cabin_a": {"reason": "nominal", "commanded": 0.1},
         "cabin_b": {"reason": "degraded_spare_release", "commanded": 0.05}},
    ]
    assert response_latency_ticks(
        history, onset_tick=1, affected_zone_ids=("cabin_a",)
    ) is None
    assert response_latency_ticks(
        history, onset_tick=1, affected_zone_ids=("cabin_b",)
    ) == 1
    assert response_latency_ticks(
        history, onset_tick=1, affected_zone_ids=("cabin_a", "cabin_b")
    ) == 1


def test_response_latency_ignores_rate_limits_on_healthy_zones():
    history = [
        {"cabin_a": {"reason": "bounded_rate", "commanded": 0.1}},
        {"cabin_a": {"reason": "nominal", "commanded": 0.1}},
    ]
    assert response_latency_ticks(
        history, onset_tick=1, affected_zone_ids=("cabin_a",)
    ) == 0
    assert response_latency_ticks(
        history, onset_tick=1, affected_zone_ids=("cabin_b",)
    ) is None


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


def test_receipt_binds_custom_factory_settings(tmp_path, standard_scenario_path):
    spec_path = _write_mini_sweep(tmp_path, standard_scenario_path)

    def custom_factory(config):
        return BoundedRecoveryGovernor(
            config, settings=ResponseSettings(max_command_delta=0.03)
        )

    receipt = run_response_evidence(
        spec_path, tmp_path / "custom", governor_factory=custom_factory
    )
    assert receipt["response_settings"]["governor_factory"] == "custom_factory"
    assert receipt["response_settings"]["max_command_delta"] == 0.03
    assert receipt["config"]["response_settings"] == receipt["response_settings"]


def test_output_dir_must_be_empty(tmp_path, standard_scenario_path):
    spec_path = _write_mini_sweep(tmp_path, standard_scenario_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "leftover.txt").write_text("x", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="not empty"):
        run_response_evidence(spec_path, output)


def _row(time_delta, reference_delta, baseline_violations=0, governed_violations=0):
    return {
        "family_id": "f",
        "fault_class": "gradual_primary_fan_degradation",
        "split": "train",
        "fault": {
            "time_above_ceiling": {"baseline": 0.0, "governed": float(time_delta), "delta": float(time_delta)},
            "max_excursion": {"delta": 0.0},
            "energy": {"baseline": 1.0, "governed": 1.0, "overhead_fraction": 0.0},
            "invariant_violations": {
                "baseline": baseline_violations,
                "governed": governed_violations,
            },
        },
        "reference": {"time_above_ceiling": {"delta": float(reference_delta)}},
        "governed_action_ticks": {"degraded_spare_release": 1},
        "response_latency_ticks": 64,
    }


def test_conclusion_derives_within_margin_from_aggregate():
    from aeolus.response_evidence import _aggregate, _conclusion

    rows = [_row(0.0, 0.0), _row(0.0, 0.0), _row(2.0, 0.0)]
    text = _conclusion(rows)
    assert "within its 1-tick causality margin on 2/3" in text
    assert "1 beyond margin" in text
    assert "all 3" not in text
    assert _aggregate(rows)["causality_margin"]["fault_families_exceeding_margin"] == 1


def test_conclusion_reports_invariant_violations_from_aggregate():
    from aeolus.response_evidence import _conclusion

    text = _conclusion([_row(0.0, 0.0, baseline_violations=1, governed_violations=2)])
    assert "1 baseline / 2 governed invariant violations" in text


def test_response_latency_handles_missing_onset():
    history = [{"cabin_a": {"reason": "nominal", "commanded": 0.1}}]
    assert response_latency_ticks(history, onset_tick=None) is None


def test_aggregate_tolerates_zero_baseline_energy():
    from aeolus.response_evidence import _aggregate

    row = {
        "fault": {
            "time_above_ceiling": {"delta": 0.0},
            "max_excursion": {"delta": 0.0},
            "energy": {"baseline": 0.0, "governed": 0.0, "delta": 0.0},
            "invariant_violations": {"baseline": 0, "governed": 0},
        },
        "reference": {"time_above_ceiling": {"delta": 0.0}},
        "governed_action_ticks": {"nominal": 1},
        "response_latency_ticks": None,
    }
    aggregate = _aggregate([row])
    assert aggregate["energy"]["mean_overhead_fraction"] == 0.0
    assert aggregate["energy"]["median_overhead_fraction"] == 0.0


def test_receipt_binds_exact_source_config_and_sweep_bytes(
    tmp_path, standard_scenario_path
):
    spec_path = _write_mini_sweep(tmp_path, standard_scenario_path)
    receipt = run_response_evidence(spec_path, tmp_path / "evidence")

    source = receipt["source"]
    source_file_hashes = source["files_sha256"]
    response_evidence_path = Path(__file__).resolve().parents[1] / "src" / "aeolus" / "response_evidence.py"
    assert source_file_hashes["src/aeolus/response_evidence.py"] == hashlib.sha256(
        response_evidence_path.read_bytes()
    ).hexdigest()
    assert source["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            source_file_hashes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    config = receipt["config"]
    assert config["base_scenario_bytes_sha256"] == hashlib.sha256(
        (tmp_path / "standard_habitat.json").read_bytes()
    ).hexdigest()
    assert len(config["run_spec_sha256"]) == 64
    assert len(config["response_settings_sha256"]) == 64

    sweep = receipt["sweep"]
    assert sweep["bytes_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert sweep["canonical_sha256"] == receipt["sweep_spec_sha256"]
    assert len(sweep["generated_scenarios_manifest_sha256"]) == 64

    expected_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert receipt["environment"]["source_worktree_dirty"] == expected_dirty
    assert receipt["environment"]["python_implementation"] == sys.implementation.name


def test_receipt_changes_when_only_raw_sweep_bytes_change(
    tmp_path, standard_scenario_path
):
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first_spec = _write_mini_sweep(first_dir, standard_scenario_path)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    (second_dir / "standard_habitat.json").write_bytes(
        (tmp_path / "first" / "standard_habitat.json").read_bytes()
    )
    second_spec = second_dir / "sweep.json"
    second_spec.write_text(
        json.dumps(MINI_SWEEP, indent=4) + "\n", encoding="utf-8"
    )

    first = run_response_evidence(first_spec, tmp_path / "first-output")
    second = run_response_evidence(second_spec, tmp_path / "second-output")

    assert first["sweep_spec_sha256"] == second["sweep_spec_sha256"]
    assert first["sweep"]["bytes_sha256"] != second["sweep"]["bytes_sha256"]
    assert first["evidence_sha256"] != second["evidence_sha256"]