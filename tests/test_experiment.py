"""One-command sweep/corpus/model experiment orchestration."""

import json
from pathlib import Path

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


def test_experiment_cli_rejects_wrong_arity(capsys):
    assert main([]) == 2
    assert "Usage:" in capsys.readouterr().err
