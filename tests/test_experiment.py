"""One-command sweep/corpus/model experiment orchestration."""

import json
import re
import sys
from pathlib import Path
from typing import cast

import pytest

from aeolus.experiment import main, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reduced_v2_spec(tmp_path: Path) -> Path:
    base_name = "standard_habitat.json"
    (tmp_path / base_name).write_bytes(
        (REPO_ROOT / "scenarios" / base_name).read_bytes()
    )
    telemetry = {
        "airflow_noise_fraction": 0.02,
        "airflow_bias_fraction": 0.01,
        "airflow_drift_fraction": 0.01,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.02,
        "co2_sensor_bias_fraction": 0.01,
        "co2_sensor_drift_fraction": 0.01,
    }
    splits = {}
    for split, seed in (
        ("train", 10),
        ("validation", 20),
        ("test", 30),
        ("stress", 40),
    ):
        splits[split] = {
            "seeds": [seed],
            "fault_start_ticks": [25],
            "operating_profiles": [
                {
                    "id": split,
                    "source_multiplier": 1.0,
                    "shared_airflow_capacity": 30.0,
                    "telemetry": telemetry,
                }
            ],
            "gradual_profiles": [
                {"duration_ticks": 30, "end_effectiveness": 0.75}
            ],
            "blocked_effectiveness": [0.65],
        }
    path = tmp_path / "sweep-v2.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "aeolus_sweep_v2",
                "base_scenario": base_name,
                "targets": ["cabin_a"],
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_one_command_experiment_emits_stress_evidence_and_artifacts(tmp_path):
    output = tmp_path / "experiment"
    artifacts = tmp_path / "artifacts"
    receipt = run_experiment(_reduced_v2_spec(tmp_path), output, artifacts)

    assert receipt["sweep"]["families_by_split"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
        "stress": 3,
    }
    assert set(receipt["artifact_sha256"]) == {
        "aeolus_fault_detector.json",
        "aeolus_fault_detector.onnx",
        "aeolus_fault_metrics.json",
    }
    metrics = json.loads(
        (artifacts / "aeolus_fault_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["stress_model"] is not None
    assert metrics["stress_rule_baseline"] is not None
    assert metrics["candidate_selection"]["selection_split"] == "validation"
    assert metrics["rule_calibration"]["selection_split"] == "validation"


def test_experiment_receipt_records_canonical_evidence_environment(tmp_path):
    output = tmp_path / "experiment"
    receipt = run_experiment(_reduced_v2_spec(tmp_path), output, tmp_path / "artifacts")

    assert receipt["schema_version"] == "aeolus_experiment_v2"
    environment = cast(dict[str, object], receipt["environment"])
    assert environment["python_implementation"] == sys.implementation.name
    assert cast(str, environment["python_version"]).startswith(
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )
    assert re.fullmatch(r"[0-9a-f]{64}", cast(str, environment["uv_lock_sha256"]))
    assert re.fullmatch(r"[0-9a-f]{40}", cast(str, environment["source_commit"]))
    assert isinstance(environment["source_worktree_dirty"], bool)
    assert cast(int, environment["onnx_ir_version"]) > 0
    assert {"domain": "", "version": 17} in cast(
        list[dict[str, object]], environment["onnx_opsets"]
    )

    stored = json.loads((output / "experiment-receipt.json").read_text(encoding="utf-8"))
    assert stored == receipt


def test_experiment_rejects_nonempty_artifact_output_without_creating_output(tmp_path):
    output = tmp_path / "experiment"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact output directory is not empty"):
        run_experiment(_reduced_v2_spec(tmp_path), output, artifacts)

    assert not output.exists()


def test_experiment_cli_rejects_wrong_arity(capsys):
    assert main([]) == 2
    assert "Usage:" in capsys.readouterr().err


def test_run_experiment_rejects_v3_development_split_layout(tmp_path):
    base_name = "standard_habitat.json"
    (tmp_path / base_name).write_bytes(
        (REPO_ROOT / "scenarios" / base_name).read_bytes()
    )
    profile = {
        "id": "primary-low",
        "source_multiplier": 0.8,
        "shared_airflow_capacity": 24.0,
        "telemetry": {
            "airflow_noise_fraction": 0.02,
            "airflow_bias_fraction": 0.01,
            "airflow_drift_fraction": 0.01,
            "actuator_position_noise_fraction": 0.01,
            "co2_sensor_noise_fraction": 0.02,
            "co2_sensor_bias_fraction": 0.01,
            "co2_sensor_drift_fraction": 0.01,
        },
    }
    item = {
        "seeds": [10],
        "fault_start_ticks": [25],
        "operating_profiles": [profile],
        "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
        "blocked_effectiveness": [0.65],
    }
    item_validation = {
        **item,
        "seeds": [500],
    }
    spec = tmp_path / "dev-sweep.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "aeolus_sweep_v3",
                "base_scenario": base_name,
                "targets": ["cabin_a"],
                "suite_role": "development",
                "splits": {"train": item, "validation": item_validation},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="v2 run_experiment path requires"):
        run_experiment(spec, tmp_path / "experiment", tmp_path / "artifacts")
    assert not (tmp_path / "experiment").exists()
