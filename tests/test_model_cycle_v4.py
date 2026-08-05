"""False-alarm-aware causal gate and v4 development acceptance contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import aeolus.model_cycle_v4 as v4
from aeolus.detector import CLASS_NAMES, FEATURE_WIDTH, WINDOW_TICKS
from aeolus.families import FamilyEvidence
from aeolus.model_cycle_v4 import (
    AlertGate,
    AlertGateConfig,
    calibrate_alert_gate,
    development_acceptance,
    evaluate_gated_detector,
    evaluate_raw_detector_rolling,
    evaluate_rule_rolling,
    reject_forensic_inputs,
    rolling_evaluation_rows,
    run_v4_development,
)


@dataclass
class EncodedDetector:
    """Return the probability vector encoded in the first four features."""

    def predict_probabilities(self, windows):
        return np.asarray([window[-1][:4] for window in windows], dtype=np.float64)


@dataclass
class EncodedRule:
    """Return the highest-probability class encoded in each synthetic window."""

    def reset(self) -> None:
        return None

    def label_window(self, window):
        return CLASS_NAMES[int(np.argmax(np.asarray(window)[-1, :4]))]


def _window(probabilities: tuple[float, float, float, float]) -> list[list[float]]:
    window = np.zeros((WINDOW_TICKS, FEATURE_WIDTH), dtype=np.float64)
    window[-1, :4] = probabilities
    return window.tolist()


def _row(
    family: str,
    role: str,
    tick: int,
    label: str,
    probabilities: tuple[float, float, float, float],
) -> dict:
    return {
        "family_id": family,
        "stream_id": f"{role}:{family}",
        "run_cluster_id": f"cluster:{family}",
        "canonical_reference_sha256": "a" * 64,
        "scenario_role": role,
        "end_tick": tick,
        "observable_onset_tick": 10,
        "label": label,
        "features": _window(probabilities),
    }


def _metrics(
    macro_f1: float,
    false_alarm: float,
    recalls=(0.8, 0.8, 0.8),
    *,
    episodes_per_1000: float = 5.0,
) -> dict:
    return {
        "macro_f1": macro_f1,
        "nominal_false_alarm_rate": false_alarm,
        "per_class": {
            name: {"recall": recall}
            for name, recall in zip(CLASS_NAMES[1:], recalls, strict=True)
        },
        "detection_latency_ticks": {"overall_median": 5.0},
        "alert_burden": {
            "episodes_per_1000_eligible_ticks": episodes_per_1000,
        },
        "cluster_metrics": {"mean_macro_f1": macro_f1},
    }


def _artifact_eligibility(passed: bool = True) -> dict[str, bool]:
    return {
        "onnx_parity_passed": passed,
        "operator_allowlist_passed": passed,
        "strict_artifact_passed": passed,
        "independent_reproduction_verified": passed,
    }


def test_alert_gate_falls_back_to_nominal_below_threshold():
    gate = AlertGate(EncodedDetector(), AlertGateConfig(0.8, 1))

    prediction = gate.predict_window(_window((0.2, 0.7, 0.05, 0.05)))

    assert prediction.label == "nominal"
    assert prediction.probabilities["gradual_primary_fan_degradation"] == 0.7


def test_alert_gate_requires_same_fault_persistence_and_resets_stream_state():
    gate = AlertGate(EncodedDetector(), AlertGateConfig(0.7, 2))
    blocked = _window((0.05, 0.05, 0.85, 0.05))

    assert gate.predict_window(blocked).label == "nominal"
    assert gate.predict_window(blocked).label == "blocked_path"
    gate.reset()
    assert gate.predict_window(blocked).label == "nominal"


def test_excluded_transition_updates_gate_state_without_entering_metrics():
    rows = [
        _row("family-a", "fault", 10, "excluded_transition", (0.05, 0.05, 0.85, 0.05)),
        _row("family-a", "fault", 15, "blocked_path", (0.05, 0.05, 0.85, 0.05)),
    ]

    metrics = evaluate_gated_detector(
        EncodedDetector(), rows, AlertGateConfig(0.7, 2)
    )

    assert metrics["samples"] == 1
    assert metrics["per_class"]["blocked_path"]["recall"] == 1.0


def test_transition_detection_counts_from_observable_onset_for_latency():
    rows = [
        _row("family-a", "fault", 10, "excluded_transition", (0.05, 0.05, 0.85, 0.05)),
        _row("family-a", "fault", 15, "blocked_path", (0.05, 0.05, 0.85, 0.05)),
    ]

    gated = evaluate_gated_detector(
        EncodedDetector(), rows, AlertGateConfig(0.7, 1)
    )
    raw = evaluate_raw_detector_rolling(EncodedDetector(), rows)
    rule = evaluate_rule_rolling(EncodedRule(), rows)  # type: ignore[arg-type]

    assert gated["samples"] == 1
    assert raw["samples"] == 1
    assert rule["samples"] == 1
    assert gated["detection_latency_ticks"]["overall_median"] == 0.0
    assert raw["detection_latency_ticks"]["overall_median"] == 0.0
    assert rule["detection_latency_ticks"]["overall_median"] == 0.0


def test_stream_boundaries_reset_persistence():
    rows = [
        _row("family-a", "fault", 10, "blocked_path", (0.05, 0.05, 0.85, 0.05)),
        _row("family-b", "fault", 10, "blocked_path", (0.05, 0.05, 0.85, 0.05)),
    ]

    metrics = evaluate_gated_detector(
        EncodedDetector(), rows, AlertGateConfig(0.7, 2)
    )

    assert metrics["per_class"]["blocked_path"]["recall"] == 0.0


def test_alert_burden_counts_episodes_not_every_fault_window():
    rows = [
        _row("healthy-a", "reference", tick, "nominal", probabilities)
        for tick, probabilities in (
            (10, (0.05, 0.05, 0.85, 0.05)),
            (11, (0.05, 0.05, 0.85, 0.05)),
            (12, (0.95, 0.02, 0.02, 0.01)),
            (13, (0.05, 0.05, 0.85, 0.05)),
        )
    ]

    metrics = evaluate_gated_detector(
        EncodedDetector(), rows, AlertGateConfig(0.7, 1)
    )

    assert metrics["alert_burden"]["false_alert_windows"] == 3
    assert metrics["alert_burden"]["false_alert_episodes"] == 2
    assert metrics["alert_burden"]["episodes_per_1000_eligible_ticks"] == 500.0


def test_rolling_rows_are_stride_one_and_deduplicate_shared_references():
    trace = tuple(tuple(0.0 for _ in range(FEATURE_WIDTH)) for _ in range(11))
    evidence = {
        family_id: FamilyEvidence(
            family_id=family_id,
            split="validation",
            fault_class="blocked_path",
            observable_onset_tick=10,
            reference_scenario_sha256="a" * 64,
            fault_scenario_sha256=fault_hash * 64,
            reference_trace_ticks=11,
            fault_trace_ticks=11,
            reference_model_input_trace=trace,
            fault_model_input_trace=trace,
        )
        for family_id, fault_hash in (
            ("validation-s900-a", "b"),
            ("validation-s900-b", "c"),
        )
    }

    rows = rolling_evaluation_rows(evidence, family_ids=set(evidence))

    reference_rows = [row for row in rows if row["scenario_role"] == "reference"]
    fault_rows = [row for row in rows if row["scenario_role"] == "fault"]
    assert len(reference_rows) == 2
    assert len(fault_rows) == 4
    assert [row["end_tick"] for row in reference_rows] == [10, 11]
    assert {row["run_cluster_id"] for row in rows} == {"seed:900"}


def test_gate_calibration_is_deterministic_and_uses_declared_grid():
    rows = [
        _row("nominal-a", "reference", 10, "nominal", (0.55, 0.45, 0.0, 0.0)),
        _row("fault-a", "fault", 10, "blocked_path", (0.05, 0.05, 0.9, 0.0)),
    ]
    rule = _metrics(0.3, 0.0, recalls=(0.0, 0.5, 0.0))

    first_config, first_receipt = calibrate_alert_gate(EncodedDetector(), rows, rule)
    second_config, second_receipt = calibrate_alert_gate(EncodedDetector(), rows, rule)

    assert first_config == second_config
    assert first_receipt == second_receipt
    assert first_receipt["grid_size"] == 20
    assert first_receipt["selection_split"] == "train_internal_calibration"
    assert first_receipt["selected_eligible"] is True


def test_development_acceptance_requires_macro_f1_win_and_safety_bounds():
    rule = _metrics(0.7, 0.01)

    assert development_acceptance(
        _metrics(0.71, 0.02), rule, _artifact_eligibility()
    )["passed"] is True
    assert development_acceptance(
        _metrics(0.71, 0.02), rule, _artifact_eligibility(False)
    )["passed"] is False
    assert development_acceptance(
        _metrics(0.7, 0.01), rule, _artifact_eligibility()
    )["passed"] is False
    assert development_acceptance(
        _metrics(0.8, 0.021), rule, _artifact_eligibility()
    )["passed"] is False
    assert development_acceptance(
        _metrics(0.8, 0.01, recalls=(0.8, 0.77, 0.8)),
        rule,
        _artifact_eligibility(),
    )["passed"] is False
    assert development_acceptance(
        _metrics(0.8, 0.01, episodes_per_1000=10.1),
        rule,
        _artifact_eligibility(),
    )["passed"] is False


def test_v4_inputs_reject_historical_forensic_reports():
    with pytest.raises(ValueError, match="forensic"):
        reject_forensic_inputs(
            {"evidence_role": "historical_forensic_only", "errors": {}}
        )
    reject_forensic_inputs({"schema_version": "ordinary-development-input"})


def test_v4_runner_accepts_only_fresh_development_and_writes_all_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(v4, "V4_FIT_SEEDS", (1700,))
    monkeypatch.setattr(v4, "V4_CALIBRATION_SEEDS", (1701,))
    monkeypatch.setattr(v4, "V4_VALIDATION_SEEDS", (1900,))
    repository = Path(__file__).resolve().parents[1]
    base_name = "standard_habitat.json"
    (tmp_path / base_name).write_bytes(
        (repository / "scenarios" / base_name).read_bytes()
    )
    telemetry = {
        "airflow_noise_fraction": 0.01,
        "airflow_bias_fraction": 0.01,
        "airflow_drift_fraction": 0.01,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.01,
        "co2_sensor_bias_fraction": 0.01,
        "co2_sensor_drift_fraction": 0.01,
    }
    document = {
        "schema_version": "aeolus_sweep_v4",
        "suite_role": "development",
        "base_scenario": base_name,
        "targets": ["cabin_a"],
        "splits": {
            split: {
                "seeds": seeds,
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
            for split, seeds in (
                ("train", [1700, 1701]),
                ("validation", [1900]),
            )
        },
    }
    document["splits"]["train"]["seeds"] = [1700, 1701]
    spec = tmp_path / "v4.json"
    spec.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        v4,
        "CANONICAL_V4_DEVELOPMENT_SPEC_SHA256",
        v4.load_sweep_spec(spec).sha256,
    )
    output = tmp_path / "run"

    report = run_v4_development(spec, output, mlp_epochs=10, cnn_epochs=10)

    assert report["schema_version"] == "aeolus_v4_development_evidence_v2"
    assert report["evidence_role"] == "development_only"
    assert len(report["source_provenance"]["source_manifest_sha256"]) == 64
    assert isinstance(report["source_provenance"]["worktree_dirty"], bool)
    assert set(report["candidates"]) == {
        "temporal_mlp_balanced_raw",
        "temporal_mlp_balanced_gated",
        "temporal_cnn_balanced_gated",
        "temporal_cnn_sqrt_gated",
    }
    assert report["response_layer_integration_authorized"] is report[
        "development_gate_passed"
    ]
    assert report["retained_method"] == "rule_baseline"
    assert report["selected_candidate"] is None
    assert report["diagnostic_learned_winner"] in report["candidates"]
    assert report["rule_baseline"]["calibration"]["selection_split"] == (
        "train_internal_calibration"
    )
    assert report["optimization_boundary"]["deployment_compatible_onnx"] is True
    assert (output / "v4-development-report.json").is_file()
    assert not (output / "final").exists()
    with pytest.raises(ValueError, match="not empty"):
        run_v4_development(spec, output, mlp_epochs=10, cnn_epochs=10)
