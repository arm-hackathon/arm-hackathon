"""False-alarm-aware causal gating and v4 development acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from aeolus.baseline import RuleBaseline
from aeolus.config import load_scenario
from aeolus.corpus import EXCLUDED_TRANSITION_LABEL, generate_corpus_v2
from aeolus.detector import (
    CLASS_INDEX,
    CLASS_NAMES,
    Prediction,
    WINDOW_TICKS,
    calibrate_rule_baseline,
    classification_metrics,
    enforce_onnx_parity,
    export_onnx,
    load_verified_corpus,
    save_detector,
    train_temporal_mlp_detector,
    validate_onnx_parity,
)
from aeolus.families import (
    FamilyEvidence,
    build_family_evidence,
    family_window_label,
)
from aeolus.sweep import SWEEP_V4_VERSION, generate_sweep, load_sweep_spec
from aeolus.temporal_cnn import (
    CNN_ONNX_OPERATORS,
    export_temporal_cnn_onnx,
    save_temporal_cnn,
    temporal_cnn_parameter_count,
    train_temporal_cnn,
)


FAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
PERSISTENCE_WINDOWS = (1, 2, 3, 5)
V4_FIT_SEEDS = (700, 701, 702, 703)
V4_CALIBRATION_SEEDS = (704, 705)
V4_VALIDATION_SEEDS = (900, 901, 902, 903, 904, 905)
PROHIBITED_HISTORICAL_SEEDS = frozenset(
    {100, 101, 102, 103, 104, 105, 500, 501, 3000, 3001, 3002}
)
CANONICAL_V4_DEVELOPMENT_SPEC_SHA256 = (
    "5a631af5e646535b2d35bebe726fd89b36f99feba4c558b29a7afe98bfe309ea"
)
MAX_FALSE_ALERT_EPISODES_PER_1000_TICKS = 10.0
MAX_FALSE_ALERT_EPISODE_REGRESSION_PER_1000_TICKS = 2.0
MAX_NOMINAL_FALSE_ALARM_REGRESSION = 0.01
MAXIMUM_FAULT_RECALL_REGRESSION = 0.02
ARTIFACT_ELIGIBILITY_KEYS = frozenset(
    {
        "onnx_parity_passed",
        "operator_allowlist_passed",
        "strict_artifact_passed",
        "independent_reproduction_verified",
    }
)
DEPLOYMENT_ONNX_OPERATOR_ALLOWLIST = frozenset(
    {
        "Abs",
        "Concat",
        "Conv",
        "Div",
        "Gather",
        "Gemm",
        "Greater",
        "MatMul",
        "Mul",
        "ReduceMax",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Transpose",
        "Where",
    }
)


class ProbabilityDetector(Protocol):
    """The deployment-neutral probability surface required by the v4 gate."""

    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True, order=True)
class AlertGateConfig:
    """One predeclared high-confidence causal alert policy."""

    fault_threshold: float
    persistence_windows: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.fault_threshold, bool)
            or not isinstance(self.fault_threshold, (int, float))
            or not math.isfinite(float(self.fault_threshold))
            or not 0.0 < float(self.fault_threshold) <= 1.0
        ):
            raise ValueError("alert fault_threshold must be finite in (0, 1]")
        if (
            isinstance(self.persistence_windows, bool)
            or not isinstance(self.persistence_windows, int)
            or self.persistence_windows < 1
        ):
            raise ValueError("alert persistence_windows must be a positive integer")

    def as_dict(self) -> dict[str, float | int]:
        """Return the strict JSON representation used in evidence receipts."""
        return {
            "fault_threshold": float(self.fault_threshold),
            "persistence_windows": self.persistence_windows,
        }


class AlertGate:
    """Apply confidence and same-fault persistence to causal probabilities."""

    def __init__(self, detector: ProbabilityDetector, config: AlertGateConfig) -> None:
        self.detector = detector
        self.config = config
        self._candidate: str | None = None
        self._consecutive = 0

    def reset(self) -> None:
        """Reset state at one family/scenario stream boundary."""
        self._candidate = None
        self._consecutive = 0

    def predict_window(self, features: Sequence[Sequence[float]]) -> Prediction:
        """Return one gated causal decision while preserving raw probabilities."""
        raw = np.asarray(
            self.detector.predict_probabilities([features]), dtype=np.float64
        )
        if raw.shape != (1, len(CLASS_NAMES)):
            raise ValueError("alert gate received malformed model probabilities")
        return self.apply_probabilities(raw[0])

    def apply_probabilities(self, probabilities: Sequence[float]) -> Prediction:
        """Apply gate state to one already-computed probability vector."""
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if probability_array.shape != (len(CLASS_NAMES),) or not np.isfinite(
            probability_array
        ).all():
            raise ValueError("alert gate received malformed model probabilities")
        if np.any(probability_array < 0.0) or not math.isclose(
            float(probability_array.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("alert gate probabilities must be non-negative and sum to one")
        fault_index = int(np.argmax(probability_array[1:])) + 1
        fault_label = CLASS_NAMES[fault_index]
        fault_confidence = float(probability_array[fault_index])
        if fault_confidence < self.config.fault_threshold:
            self.reset()
            selected_index = CLASS_INDEX["nominal"]
        else:
            if self._candidate == fault_label:
                self._consecutive += 1
            else:
                self._candidate = fault_label
                self._consecutive = 1
            selected_index = (
                fault_index
                if self._consecutive >= self.config.persistence_windows
                else CLASS_INDEX["nominal"]
            )
        selected_label = CLASS_NAMES[selected_index]
        return Prediction(
            label=selected_label,
            confidence=float(probability_array[selected_index]),
            probabilities={
                name: float(probability_array[index])
                for index, name in enumerate(CLASS_NAMES)
            },
        )

    def label_window(self, features: Sequence[Sequence[float]]) -> str:
        """Implement the evaluator's stateful window-labeller protocol."""
        return self.predict_window(features).label


def evaluate_gated_detector(
    detector: ProbabilityDetector,
    rows: Sequence[Mapping[str, Any]],
    config: AlertGateConfig,
) -> dict[str, Any]:
    """Evaluate one gate over ordered streams, including transition state updates."""
    if not rows:
        raise ValueError("gated evaluation requires non-empty rows")
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("stream_id", f"{row['family_id']}:{row['scenario_role']}"),
            row["end_tick"],
        ),
    )
    raw_probabilities = detector.predict_probabilities(
        [row["features"] for row in ordered]
    )
    return evaluate_gated_probabilities(ordered, raw_probabilities, config)


def evaluate_gated_probabilities(
    ordered_rows: Sequence[Mapping[str, Any]],
    raw_probabilities: Sequence[Sequence[float]] | NDArray[np.float64],
    config: AlertGateConfig,
) -> dict[str, Any]:
    """Evaluate precomputed probabilities without repeating model inference per gate."""
    probabilities_array = np.asarray(raw_probabilities, dtype=np.float64)
    if probabilities_array.shape != (len(ordered_rows), len(CLASS_NAMES)):
        raise ValueError("gated evaluation probabilities do not align with rows")
    gate = AlertGate(_UnusedDetector(), config)
    current_stream: str | None = None
    labels: list[int] = []
    predictions: list[int] = []
    all_predictions: list[int] = []
    scored_probabilities: list[list[float]] = []
    scored_rows: list[Mapping[str, Any]] = []
    for row, raw in zip(ordered_rows, probabilities_array, strict=True):
        stream = str(
            row.get("stream_id", f"{row['family_id']}:{row['scenario_role']}")
        )
        if stream != current_stream:
            current_stream = stream
            gate.reset()
        prediction = gate.apply_probabilities(raw)
        all_predictions.append(CLASS_INDEX[prediction.label])
        label = row["label"]
        if label == EXCLUDED_TRANSITION_LABEL:
            continue
        if label not in CLASS_INDEX:
            raise ValueError(f"gated evaluation contains unsupported label {label!r}")
        labels.append(CLASS_INDEX[label])
        predictions.append(CLASS_INDEX[prediction.label])
        scored_probabilities.append(
            [prediction.probabilities[name] for name in CLASS_NAMES]
        )
        scored_rows.append(row)
    if not labels:
        raise ValueError("gated evaluation contains no scored rows")
    prediction_array = np.asarray(predictions, dtype=np.int64)
    metrics = classification_metrics(
        labels,
        prediction_array,
        np.asarray(scored_probabilities, dtype=np.float64),
    )
    metrics["detection_latency_ticks"] = _latency_metrics(
        ordered_rows, np.asarray(all_predictions, dtype=np.int64)
    )
    metrics["alert_burden"] = _alert_burden(scored_rows, prediction_array)
    metrics["cluster_metrics"] = _cluster_metrics(
        scored_rows,
        np.asarray(labels, dtype=np.int64),
        prediction_array,
        np.asarray(scored_probabilities, dtype=np.float64),
    )
    metrics["gate"] = config.as_dict()
    return metrics


class _UnusedDetector:
    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:  # pragma: no cover - state-only adapter
        raise RuntimeError("precomputed gate evaluation does not run inference")


def calibrate_alert_gate(
    detector: ProbabilityDetector,
    validation_rows: Sequence[Mapping[str, Any]],
    rule_metrics: Mapping[str, Any],
) -> tuple[AlertGateConfig, dict[str, Any]]:
    """Select one gate using only the predeclared validation grid."""
    ordered = sorted(
        validation_rows,
        key=lambda row: (
            row.get("stream_id", f"{row['family_id']}:{row['scenario_role']}"),
            row["end_tick"],
        ),
    )
    raw_probabilities = detector.predict_probabilities(
        [row["features"] for row in ordered]
    )
    candidates: list[tuple[AlertGateConfig, dict[str, Any], dict[str, Any]]] = []
    for threshold in FAULT_THRESHOLDS:
        for persistence in PERSISTENCE_WINDOWS:
            config = AlertGateConfig(threshold, persistence)
            metrics = evaluate_gated_probabilities(
                ordered, raw_probabilities, config
            )
            eligibility = _safety_eligibility(metrics, rule_metrics)
            candidates.append((config, metrics, eligibility))
    selected_config, selected_metrics, selected_eligibility = min(
        candidates,
        key=lambda item: (
            not bool(item[2]["eligible"]),
            -float(item[1]["cluster_metrics"]["mean_macro_f1"]),
            -float(item[1]["macro_f1"]),
            float(item[1]["alert_burden"]["episodes_per_1000_eligible_ticks"]),
            float(item[1]["detection_latency_ticks"]["overall_median"]),
            -item[0].fault_threshold,
            item[0].persistence_windows,
        ),
    )
    return selected_config, {
        "selection_split": "train_internal_calibration",
        "grid": {
            "fault_thresholds": list(FAULT_THRESHOLDS),
            "persistence_windows": list(PERSISTENCE_WINDOWS),
        },
        "grid_size": len(candidates),
        "selection_order": [
            "safety_eligible_descending",
            "cluster_mean_macro_f1_descending",
            "window_macro_f1_descending",
            "false_alert_episodes_ascending",
            "causal_latency_ascending",
            "fault_threshold_descending",
            "persistence_windows_ascending",
        ],
        "selected_gate": selected_config.as_dict(),
        "selected_metrics": selected_metrics,
        "selected_eligible": bool(selected_eligibility["eligible"]),
        "selected_eligibility": selected_eligibility,
        "candidates": [
            {
                "gate": config.as_dict(),
                "metrics": metrics,
                "eligibility": eligibility,
            }
            for config, metrics, eligibility in candidates
        ],
    }


def development_acceptance(
    model_metrics: Mapping[str, Any],
    rule_metrics: Mapping[str, Any],
    artifact_eligibility: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the predeclared quality, safety, and artifact gate without escape."""
    if set(artifact_eligibility) != ARTIFACT_ELIGIBILITY_KEYS:
        raise ValueError("artifact eligibility must contain the exact declared checks")
    if any(not isinstance(value, bool) for value in artifact_eligibility.values()):
        raise ValueError("artifact eligibility checks must be booleans")
    safety = _safety_eligibility(model_metrics, rule_metrics)
    cluster_macro_gain = float(
        model_metrics["cluster_metrics"]["mean_macro_f1"]
    ) - float(rule_metrics["cluster_metrics"]["mean_macro_f1"])
    macro_f1_win = cluster_macro_gain > 1e-12
    quality_and_safety_passed = bool(macro_f1_win and safety["eligible"])
    artifact_eligibility_passed = all(artifact_eligibility.values())
    return {
        "criterion": (
            "seed-cluster mean macro-F1 strictly greater than calibrated rules; "
            "false-alert episodes <= 10 per 1,000 healthy ticks and <= 2 above "
            "rules; nominal false-alarm regression <= 0.01; every fault recall "
            "delta >= -0.02; ONNX parity, operator allowlist, strict artifact, "
            "and independent reproduction checks all pass"
        ),
        "passed": bool(quality_and_safety_passed and artifact_eligibility_passed),
        "quality_and_safety_passed": quality_and_safety_passed,
        "artifact_eligibility_passed": artifact_eligibility_passed,
        "artifact_eligibility": dict(artifact_eligibility),
        "macro_f1_win": macro_f1_win,
        "cluster_macro_f1_gain": cluster_macro_gain,
        "window_macro_f1_gain": float(model_metrics["macro_f1"])
        - float(rule_metrics["macro_f1"]),
        **safety,
    }


def reject_forensic_inputs(*inputs: object) -> None:
    """Prevent historical v3 forensic reports from entering v4 development APIs."""
    for value in inputs:
        if isinstance(value, Mapping) and value.get("evidence_role") == "historical_forensic_only":
            raise ValueError("v4 development rejects historical forensic evidence as input")


def rolling_evaluation_rows(
    family_evidence: Mapping[str, FamilyEvidence],
    *,
    family_ids: set[str],
) -> list[dict[str, Any]]:
    """Build stride-one windows while deduplicating shared healthy references."""
    rows: list[dict[str, Any]] = []
    seen_references: set[str] = set()
    for family_id in sorted(family_ids):
        evidence = family_evidence.get(family_id)
        if evidence is None:
            raise ValueError("rolling evaluation family is absent from evidence")
        if (
            evidence.reference_model_input_trace is None
            or evidence.fault_model_input_trace is None
        ):
            raise ValueError("stride-one evaluation requires model-input traces")
        cluster_id = _seed_cluster_id(family_id)
        streams = [("fault", evidence.fault_model_input_trace)]
        if evidence.reference_scenario_sha256 not in seen_references:
            seen_references.add(evidence.reference_scenario_sha256)
            streams.insert(0, ("reference", evidence.reference_model_input_trace))
        for scenario_role, trace in streams:
            stream_id = (
                f"reference:{evidence.reference_scenario_sha256}"
                if scenario_role == "reference"
                else f"fault:{family_id}"
            )
            for start_index in range(0, len(trace) - WINDOW_TICKS + 1):
                end_index = start_index + WINDOW_TICKS
                start_tick = start_index + 1
                end_tick = end_index
                rows.append(
                    {
                        "family_id": family_id,
                        "run_cluster_id": cluster_id,
                        "stream_id": stream_id,
                        "canonical_reference_sha256": (
                            evidence.reference_scenario_sha256
                        ),
                        "split": evidence.split,
                        "scenario_role": scenario_role,
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "observable_onset_tick": evidence.observable_onset_tick,
                        "label": family_window_label(
                            scenario_role=scenario_role,
                            start_tick=start_tick,
                            end_tick=end_tick,
                            evidence=evidence,
                        ),
                        "features": [
                            list(vector) for vector in trace[start_index:end_index]
                        ],
                    }
                )
    if not rows:
        raise ValueError("stride-one evaluation requires at least one family")
    return rows


def evaluate_raw_detector_rolling(
    detector: ProbabilityDetector, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Evaluate raw argmax on the same deduplicated stride-one streams."""
    ordered = _ordered_rows(rows)
    probabilities = np.asarray(
        detector.predict_probabilities([row["features"] for row in ordered]),
        dtype=np.float64,
    )
    scored_rows, labels, scored_probabilities = _scored_evaluation_arrays(
        ordered, probabilities
    )
    all_predictions = np.argmax(probabilities, axis=1)
    predictions = np.argmax(scored_probabilities, axis=1)
    return _complete_operational_metrics(
        scored_rows,
        labels,
        predictions,
        scored_probabilities,
        latency_rows=ordered,
        latency_predictions=all_predictions,
    )


def evaluate_rule_rolling(
    rule: RuleBaseline, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Evaluate one stateful rule policy on deduplicated stride-one streams."""
    ordered = _ordered_rows(rows)
    current_stream: str | None = None
    scored_rows: list[Mapping[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    all_predictions: list[int] = []
    for row in ordered:
        stream = str(row["stream_id"])
        if stream != current_stream:
            current_stream = stream
            rule.reset()
        predicted_label = rule.label_window(row["features"])
        all_predictions.append(CLASS_INDEX[predicted_label])
        label = row["label"]
        if label == EXCLUDED_TRANSITION_LABEL:
            continue
        scored_rows.append(row)
        labels.append(CLASS_INDEX[label])
        predictions.append(CLASS_INDEX[predicted_label])
    prediction_array = np.asarray(predictions, dtype=np.int64)
    probabilities = np.eye(len(CLASS_NAMES), dtype=np.float64)[prediction_array]
    return _complete_operational_metrics(
        scored_rows,
        np.asarray(labels, dtype=np.int64),
        prediction_array,
        probabilities,
        latency_rows=ordered,
        latency_predictions=np.asarray(all_predictions, dtype=np.int64),
    )


def _ordered_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if not rows:
        raise ValueError("operational evaluation requires non-empty rows")
    return sorted(rows, key=lambda row: (str(row["stream_id"]), int(row["end_tick"])))


def _scored_evaluation_arrays(
    ordered: Sequence[Mapping[str, Any]],
    probabilities: NDArray[np.float64],
) -> tuple[list[Mapping[str, Any]], NDArray[np.int64], NDArray[np.float64]]:
    if probabilities.shape != (len(ordered), len(CLASS_NAMES)):
        raise ValueError("operational probabilities do not align with rows")
    indices = [
        index
        for index, row in enumerate(ordered)
        if row["label"] != EXCLUDED_TRANSITION_LABEL
    ]
    scored_rows = [ordered[index] for index in indices]
    labels = np.asarray(
        [CLASS_INDEX[str(row["label"])] for row in scored_rows], dtype=np.int64
    )
    return scored_rows, labels, probabilities[np.asarray(indices, dtype=np.int64)]


def _complete_operational_metrics(
    rows: Sequence[Mapping[str, Any]],
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    *,
    latency_rows: Sequence[Mapping[str, Any]] | None = None,
    latency_predictions: NDArray[np.int64] | None = None,
) -> dict[str, Any]:
    if (latency_rows is None) != (latency_predictions is None):
        raise ValueError("latency rows and predictions must be supplied together")
    metrics = classification_metrics(labels, predictions, probabilities)
    metrics["detection_latency_ticks"] = _latency_metrics(
        rows if latency_rows is None else latency_rows,
        predictions if latency_predictions is None else latency_predictions,
    )
    metrics["alert_burden"] = _alert_burden(rows, predictions)
    metrics["cluster_metrics"] = _cluster_metrics(
        rows, labels, predictions, probabilities
    )
    return metrics


def run_v4_development(
    sweep_spec_path: str | Path,
    output_dir: str | Path,
    *,
    mlp_epochs: int = 300,
    cnn_epochs: int = 300,
) -> dict[str, Any]:
    """Run the predeclared train/validation-only v4 model cycle once."""
    spec = load_sweep_spec(sweep_spec_path)
    if spec.schema_version != SWEEP_V4_VERSION or spec.suite_role != "development":
        raise ValueError("v4 model cycle requires an aeolus_sweep_v4 development suite")
    if tuple(spec.splits) != ("train", "validation"):
        raise ValueError("v4 development requires exactly train and validation splits")
    if spec.sha256 != CANONICAL_V4_DEVELOPMENT_SPEC_SHA256:
        raise ValueError("v4 development sweep does not match the frozen canonical spec")
    if spec.splits["train"].seeds != V4_FIT_SEEDS + V4_CALIBRATION_SEEDS:
        raise ValueError("v4 development train seed clusters drifted from the protocol")
    if spec.splits["validation"].seeds != V4_VALIDATION_SEEDS:
        raise ValueError("v4 development validation seed clusters drifted from the protocol")
    all_seeds = set(spec.splits["train"].seeds + spec.splits["validation"].seeds)
    if all_seeds & PROHIBITED_HISTORICAL_SEEDS:
        raise ValueError("v4 development overlaps a prohibited historical seed cluster")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"v4 development output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sweep_dir = output / "sweep"
    corpus_dir = output / "corpus"
    models_dir = output / "models"
    sweep_receipt = generate_sweep(sweep_spec_path, sweep_dir)
    corpus_receipt = generate_corpus_v2(sweep_dir / "families.json", corpus_dir)
    rows, manifest = load_verified_corpus(
        corpus_dir / "corpus.jsonl",
        sweep_dir / "families.json",
        str(sweep_receipt["family_manifest_sha256"]),
    )
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation")
    }
    if not split_rows["train"] or not split_rows["validation"]:
        raise ValueError("v4 development requires non-empty train and validation rows")
    evidence = build_family_evidence(manifest)
    family_ids_by_seed: dict[int, set[str]] = {}
    for family_id in evidence:
        seed = _family_seed(family_id)
        family_ids_by_seed.setdefault(seed, set()).add(family_id)
    fit_family_ids = set().union(
        *(family_ids_by_seed.get(seed, set()) for seed in V4_FIT_SEEDS)
    )
    calibration_family_ids = set().union(
        *(family_ids_by_seed.get(seed, set()) for seed in V4_CALIBRATION_SEEDS)
    )
    validation_family_ids = set().union(
        *(family_ids_by_seed.get(seed, set()) for seed in V4_VALIDATION_SEEDS)
    )
    expected_family_ids = fit_family_ids | calibration_family_ids | validation_family_ids
    if expected_family_ids != set(evidence):
        raise ValueError("v4 family seed clustering is incomplete or contains drift")
    fit_rows = [row for row in split_rows["train"] if row["family_id"] in fit_family_ids]
    calibration_rows = [
        row
        for row in split_rows["train"]
        if row["family_id"] in calibration_family_ids
    ]
    if not fit_rows or not calibration_rows:
        raise ValueError("v4 train-internal fit/calibration partitions are empty")
    calibration_evidence = {
        family_id: evidence[family_id] for family_id in calibration_family_ids
    }
    rolling_calibration = rolling_evaluation_rows(
        evidence, family_ids=calibration_family_ids
    )
    rolling_validation = rolling_evaluation_rows(
        evidence, family_ids=validation_family_ids
    )
    baseline_config = load_scenario(manifest.families[0].reference_path)
    rule_parameters, rule_calibration = calibrate_rule_baseline(
        rolling_calibration, baseline_config, calibration_evidence
    )
    if rule_calibration.get("selection_split") != "validation":
        raise ValueError("unexpected rule-calibration receipt split")
    rule_calibration = {
        **rule_calibration,
        "selection_split": "train_internal_calibration",
    }
    rule_calibration_metrics = evaluate_rule_rolling(
        RuleBaseline(baseline_config, rule_parameters), rolling_calibration
    )
    rule_metrics = evaluate_rule_rolling(
        RuleBaseline(baseline_config, rule_parameters), rolling_validation
    )

    mlp, mlp_training = train_temporal_mlp_detector(
        fit_rows,
        calibration_rows,
        contract_metadata=manifest.contract_metadata,
        epochs=mlp_epochs,
    )
    cnn_balanced, cnn_balanced_training = train_temporal_cnn(
        fit_rows,
        calibration_rows,
        contract_metadata=manifest.contract_metadata,
        epochs=cnn_epochs,
        weighting_mode="balanced",
    )
    cnn_sqrt, cnn_sqrt_training = train_temporal_cnn(
        fit_rows,
        calibration_rows,
        contract_metadata=manifest.contract_metadata,
        epochs=cnn_epochs,
        weighting_mode="sqrt_inverse",
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_receipts: dict[str, dict[str, Any]] = {}
    base_models = {
        "temporal_mlp_balanced": (mlp, mlp_training),
        "temporal_cnn_balanced": (cnn_balanced, cnn_balanced_training),
        "temporal_cnn_sqrt": (cnn_sqrt, cnn_sqrt_training),
    }
    for name, (detector, training) in base_models.items():
        json_path = models_dir / f"{name}.json"
        onnx_path = models_dir / f"{name}.onnx"
        if name == "temporal_mlp_balanced":
            save_detector(detector, json_path)
            export_onnx(detector, onnx_path)
            parameter_count = _mlp_parameter_count(detector)
        else:
            save_temporal_cnn(detector, json_path)
            export_temporal_cnn_onnx(detector, onnx_path)
            parameter_count = temporal_cnn_parameter_count(detector)
        parity = validate_onnx_parity(
            detector, onnx_path, split_rows["validation"]
        )
        enforce_onnx_parity(parity)
        onnx_operators = _onnx_operators(onnx_path)
        artifact_eligibility = {
            "onnx_parity_passed": bool(
                parity["max_absolute_probability_error"]
                <= parity["maximum_acceptable_probability_error"]
            ),
            "operator_allowlist_passed": set(onnx_operators)
            <= DEPLOYMENT_ONNX_OPERATOR_ALLOWLIST,
            "strict_artifact_passed": _strict_json_artifact_passed(json_path)
            and json_path.stat().st_size > 0
            and onnx_path.stat().st_size > 0,
            # Cross-run reproduction is verified by the closeout receipt, never assumed
            # by the development runner that produced the first artifact.
            "independent_reproduction_verified": False,
        }
        model_receipts[name] = {
            "training": training,
            "parameter_count": parameter_count,
            "json_bytes": json_path.stat().st_size,
            "onnx_bytes": onnx_path.stat().st_size,
            "json_sha256": _sha256_file(json_path),
            "onnx_sha256": _sha256_file(onnx_path),
            "onnx_operators": onnx_operators,
            "onnx_parity": parity,
            "artifact_eligibility": artifact_eligibility,
        }

    mlp_gate, mlp_gate_receipt = calibrate_alert_gate(
        mlp, rolling_calibration, rule_calibration_metrics
    )
    cnn_balanced_gate, cnn_balanced_gate_receipt = calibrate_alert_gate(
        cnn_balanced, rolling_calibration, rule_calibration_metrics
    )
    cnn_sqrt_gate, cnn_sqrt_gate_receipt = calibrate_alert_gate(
        cnn_sqrt, rolling_calibration, rule_calibration_metrics
    )
    mlp_raw_metrics = evaluate_raw_detector_rolling(mlp, rolling_validation)
    mlp_gated_metrics = evaluate_gated_detector(
        mlp, rolling_validation, mlp_gate
    )
    cnn_balanced_metrics = evaluate_gated_detector(
        cnn_balanced, rolling_validation, cnn_balanced_gate
    )
    cnn_sqrt_metrics = evaluate_gated_detector(
        cnn_sqrt, rolling_validation, cnn_sqrt_gate
    )
    candidate_inputs = {
        "temporal_mlp_balanced_raw": (
            "temporal_mlp_balanced",
            None,
            mlp_raw_metrics,
            None,
        ),
        "temporal_mlp_balanced_gated": (
            "temporal_mlp_balanced",
            mlp_gate,
            mlp_gated_metrics,
            mlp_gate_receipt,
        ),
        "temporal_cnn_balanced_gated": (
            "temporal_cnn_balanced",
            cnn_balanced_gate,
            cnn_balanced_metrics,
            cnn_balanced_gate_receipt,
        ),
        "temporal_cnn_sqrt_gated": (
            "temporal_cnn_sqrt",
            cnn_sqrt_gate,
            cnn_sqrt_metrics,
            cnn_sqrt_gate_receipt,
        ),
    }
    candidates: dict[str, dict[str, Any]] = {}
    for name, (base_model, gate, metrics, calibration) in candidate_inputs.items():
        candidates[name] = {
            "base_model": base_model,
            "gate": gate.as_dict() if gate is not None else None,
            "validation_metrics": metrics,
            "development_acceptance": development_acceptance(
                metrics,
                rule_metrics,
                model_receipts[base_model]["artifact_eligibility"],
            ),
            "gate_calibration": calibration,
        }
    selected_name, selected = min(
        candidates.items(),
        key=lambda item: (
            not bool(item[1]["development_acceptance"]["passed"]),
            -float(
                item[1]["validation_metrics"]["cluster_metrics"]["mean_macro_f1"]
            ),
            -float(item[1]["validation_metrics"]["macro_f1"]),
            float(
                item[1]["validation_metrics"]["alert_burden"]
                ["episodes_per_1000_eligible_ticks"]
            ),
            float(
                item[1]["validation_metrics"]["detection_latency_ticks"][
                    "overall_median"
                ]
            ),
            model_receipts[item[1]["base_model"]]["parameter_count"],
            item[0],
        ),
    )
    development_gate_passed = bool(
        selected["development_acceptance"]["passed"]
    )
    deployment_compatible_onnx = all(
        receipt["artifact_eligibility"][key]
        for receipt in model_receipts.values()
        for key in (
            "onnx_parity_passed",
            "operator_allowlist_passed",
            "strict_artifact_passed",
        )
    )
    retained_method = selected_name if development_gate_passed else "rule_baseline"
    report: dict[str, Any] = {
        "schema_version": "aeolus_v4_development_evidence_v2",
        "evidence_role": "development_only",
        "source_sweep_spec_sha256": spec.sha256,
        "family_manifest_sha256": manifest.manifest_sha256,
        "corpus_manifest_sha256": corpus_receipt["manifest_sha256"],
        "contract_metadata": dict(manifest.contract_metadata),
        "families_by_split": sweep_receipt["families_by_split"],
        "rows_by_split": {
            split: len(split_rows[split]) for split in ("train", "validation")
        },
        "cluster_protocol": {
            "fit_seeds": list(V4_FIT_SEEDS),
            "train_internal_calibration_seeds": list(V4_CALIBRATION_SEEDS),
            "single_use_validation_seeds": list(V4_VALIDATION_SEEDS),
            "fit_families": len(fit_family_ids),
            "train_internal_calibration_families": len(calibration_family_ids),
            "validation_families": len(validation_family_ids),
            "rolling_calibration_rows": len(rolling_calibration),
            "rolling_validation_rows": len(rolling_validation),
            "healthy_references_deduplicated_by_sha256": True,
        },
        "predeclared_candidates": list(candidate_inputs),
        "rule_baseline": {
            "parameters": rule_parameters.as_dict(),
            "calibration": rule_calibration,
            "train_internal_calibration_metrics": rule_calibration_metrics,
            "validation_metrics": rule_metrics,
        },
        "models": model_receipts,
        "candidates": candidates,
        "diagnostic_learned_winner": selected_name,
        "selected_candidate": selected_name if development_gate_passed else None,
        "retained_method": retained_method,
        "development_gate_passed": development_gate_passed,
        "response_layer_integration_authorized": development_gate_passed,
        "final_suite_generation_authorized": development_gate_passed,
        "ablation_deltas": _ablation_deltas(candidates),
        "optimization_boundary": {
            "deployment_compatible_onnx": deployment_compatible_onnx,
            "declared_cnn_operators": list(CNN_ONNX_OPERATORS),
            "operator_allowlist": sorted(DEPLOYMENT_ONNX_OPERATOR_ALLOWLIST),
            "int8_status": (
                "eligible_for_training_only_calibration"
                if development_gate_passed
                else "deferred_rejected_quality_candidate"
            ),
            "arm_performance_claim": False,
            "reason": "this development run is not an AArch64 target benchmark",
        },
        "environment": {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_provenance": _source_provenance(Path(__file__).resolve().parents[2]),
    }
    report_path = output / "v4-development-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _source_provenance(repository: Path) -> dict[str, Any]:
    """Bind development evidence to exact source bytes and Git state."""
    relative_paths = (
        "uv.lock",
        "scenarios/sweep-v4-development.json",
        "src/aeolus/sweep.py",
        "src/aeolus/error_analysis.py",
        "src/aeolus/temporal_cnn.py",
        "src/aeolus/model_cycle_v4.py",
        "src/aeolus/edge_benchmark.py",
    )
    source_files = {
        relative: _sha256_file(repository / relative) for relative in relative_paths
    }
    canonical = json.dumps(source_files, sort_keys=True, separators=(",", ":"))
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot establish source Git provenance: {exc}") from None
    return {
        "head_commit": head,
        "worktree_dirty": bool(status),
        "source_files_sha256": source_files,
        "source_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _mlp_parameter_count(detector: object) -> int:
    return sum(
        int(np.asarray(getattr(detector, name)).size)
        for name in (
            "input_weights",
            "hidden_biases",
            "output_weights",
            "output_biases",
        )
    )


def _strict_json_artifact_passed(path: Path) -> bool:
    """Return whether an artifact is strict finite JSON, without trusting its writer."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    return isinstance(value, Mapping)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _onnx_operators(path: Path) -> list[str]:
    import onnx

    return [node.op_type for node in onnx.load(path).graph.node]


def _ablation_deltas(candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    raw = candidates["temporal_mlp_balanced_raw"]["validation_metrics"]
    gated = candidates["temporal_mlp_balanced_gated"]["validation_metrics"]
    balanced = candidates["temporal_cnn_balanced_gated"]["validation_metrics"]
    sqrt = candidates["temporal_cnn_sqrt_gated"]["validation_metrics"]
    return {
        "mlp_gate_minus_raw": {
            "macro_f1": float(gated["macro_f1"]) - float(raw["macro_f1"]),
            "nominal_false_alarm_rate": float(gated["nominal_false_alarm_rate"])
            - float(raw["nominal_false_alarm_rate"]),
        },
        "cnn_sqrt_minus_balanced": {
            "macro_f1": float(sqrt["macro_f1"]) - float(balanced["macro_f1"]),
            "nominal_false_alarm_rate": float(sqrt["nominal_false_alarm_rate"])
            - float(balanced["nominal_false_alarm_rate"]),
        },
        "cnn_balanced_minus_gated_mlp": {
            "macro_f1": float(balanced["macro_f1"]) - float(gated["macro_f1"]),
            "nominal_false_alarm_rate": float(balanced["nominal_false_alarm_rate"])
            - float(gated["nominal_false_alarm_rate"]),
        },
    }


def _safety_eligibility(
    model_metrics: Mapping[str, Any], rule_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    false_alarm_regression = float(model_metrics["nominal_false_alarm_rate"]) - float(
        rule_metrics["nominal_false_alarm_rate"]
    )
    model_episodes = float(
        model_metrics["alert_burden"]["episodes_per_1000_eligible_ticks"]
    )
    rule_episodes = float(
        rule_metrics["alert_burden"]["episodes_per_1000_eligible_ticks"]
    )
    episode_regression = model_episodes - rule_episodes
    recall_deltas = {
        name: float(model_metrics["per_class"][name]["recall"])
        - float(rule_metrics["per_class"][name]["recall"])
        for name in CLASS_NAMES[1:]
    }
    eligible = (
        model_episodes <= MAX_FALSE_ALERT_EPISODES_PER_1000_TICKS + 1e-12
        and episode_regression
        <= MAX_FALSE_ALERT_EPISODE_REGRESSION_PER_1000_TICKS + 1e-12
        and false_alarm_regression <= MAX_NOMINAL_FALSE_ALARM_REGRESSION + 1e-12
        and min(recall_deltas.values())
        >= -MAXIMUM_FAULT_RECALL_REGRESSION - 1e-12
    )
    return {
        "eligible": eligible,
        "maximum_false_alert_episodes_per_1000_ticks": (
            MAX_FALSE_ALERT_EPISODES_PER_1000_TICKS
        ),
        "maximum_false_alert_episode_regression_per_1000_ticks": (
            MAX_FALSE_ALERT_EPISODE_REGRESSION_PER_1000_TICKS
        ),
        "maximum_nominal_false_alarm_regression": (
            MAX_NOMINAL_FALSE_ALARM_REGRESSION
        ),
        "maximum_fault_recall_regression": MAXIMUM_FAULT_RECALL_REGRESSION,
        "false_alert_episodes_per_1000_ticks": model_episodes,
        "false_alert_episode_regression_per_1000_ticks": episode_regression,
        "nominal_false_alarm_regression": false_alarm_regression,
        "fault_recall_deltas": recall_deltas,
    }


def _family_seed(family_id: str) -> int:
    match = re.match(r"^(?:train|validation)-s(?P<seed>[0-9]+)-", family_id)
    if match is None:
        raise ValueError(f"family_id does not encode a seed cluster: {family_id}")
    return int(match.group("seed"))


def _seed_cluster_id(family_id: str) -> str:
    try:
        seed = _family_seed(family_id)
    except ValueError:
        return f"unclustered:{family_id}"
    return f"seed:{seed}"


def _alert_burden(
    rows: Sequence[Mapping[str, Any]], predictions: NDArray[np.int64]
) -> dict[str, Any]:
    healthy = [
        (row, int(prediction))
        for row, prediction in zip(rows, predictions, strict=True)
        if row["scenario_role"] == "reference" and row["label"] == "nominal"
    ]
    by_stream: dict[str, list[tuple[Mapping[str, Any], int]]] = {}
    for row, prediction in healthy:
        stream = str(
            row.get(
                "stream_id",
                f"reference:{row.get('canonical_reference_sha256', row['family_id'])}",
            )
        )
        by_stream.setdefault(stream, []).append((row, prediction))
    episodes = 0
    false_alert_windows = 0
    streams_with_alert = 0
    clusters: set[str] = set()
    alerted_clusters: set[str] = set()
    for stream_rows in by_stream.values():
        active = False
        stream_alerted = False
        for row, prediction in sorted(
            stream_rows, key=lambda item: int(item[0]["end_tick"])
        ):
            cluster = str(
                row.get("run_cluster_id", _seed_cluster_id(str(row["family_id"])))
            )
            clusters.add(cluster)
            is_alert = prediction != CLASS_INDEX["nominal"]
            if is_alert:
                false_alert_windows += 1
                stream_alerted = True
                alerted_clusters.add(cluster)
                if not active:
                    episodes += 1
            active = is_alert
        if stream_alerted:
            streams_with_alert += 1
    eligible_ticks = len(healthy)
    return {
        "eligible_healthy_ticks": eligible_ticks,
        "false_alert_windows": false_alert_windows,
        "false_alert_episodes": episodes,
        "episodes_per_1000_eligible_ticks": (
            1000.0 * episodes / eligible_ticks if eligible_ticks else 0.0
        ),
        "unique_healthy_streams": len(by_stream),
        "healthy_streams_with_any_alert": streams_with_alert,
        "fraction_healthy_streams_with_any_alert": (
            streams_with_alert / len(by_stream) if by_stream else 0.0
        ),
        "seed_clusters": len(clusters),
        "seed_clusters_with_any_alert": len(alerted_clusters),
        "fraction_seed_clusters_with_any_alert": (
            len(alerted_clusters) / len(clusters) if clusters else 0.0
        ),
    }


def _cluster_metrics(
    rows: Sequence[Mapping[str, Any]],
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        cluster = str(
            row.get("run_cluster_id", _seed_cluster_id(str(row["family_id"])))
        )
        grouped.setdefault(cluster, []).append(index)
    per_cluster: dict[str, dict[str, float]] = {}
    for cluster, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=np.int64)
        metrics = classification_metrics(
            labels[selected], predictions[selected], probabilities[selected]
        )
        per_cluster[cluster] = {
            "macro_f1": float(metrics["macro_f1"]),
            "nominal_false_alarm_rate": float(
                metrics["nominal_false_alarm_rate"]
            ),
        }
    return {
        "cluster_count": len(per_cluster),
        "mean_macro_f1": float(
            statistics.mean(value["macro_f1"] for value in per_cluster.values())
        ),
        "per_cluster": per_cluster,
    }


def _latency_metrics(
    rows: Sequence[Mapping[str, Any]], predictions: NDArray[np.int64]
) -> dict[str, Any]:
    detections: dict[str, list[int]] = {name: [] for name in CLASS_NAMES[1:]}
    missed: dict[str, int] = {name: 0 for name in CLASS_NAMES[1:]}
    grouped: dict[str, list[tuple[Mapping[str, Any], int]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if row["scenario_role"] != "fault":
            continue
        grouped.setdefault(str(row["family_id"]), []).append((row, int(prediction)))
    for family_rows in grouped.values():
        ordered = sorted(family_rows, key=lambda item: item[0]["end_tick"])
        fault_label = next(
            (
                str(row["label"])
                for row, _ in ordered
                if row["label"] not in ("nominal", EXCLUDED_TRANSITION_LABEL)
            ),
            None,
        )
        if fault_label is None:
            continue
        expected_index = CLASS_INDEX[fault_label]
        latency = next(
            (
                int(row["end_tick"]) - int(row["observable_onset_tick"])
                for row, prediction in ordered
                if int(row["end_tick"]) >= int(row["observable_onset_tick"])
                and prediction == expected_index
            ),
            None,
        )
        if latency is None:
            missed[fault_label] += 1
        else:
            detections[fault_label].append(latency)
    all_latencies = [latency for values in detections.values() for latency in values]
    strides = [
        int(second[0]["end_tick"]) - int(first[0]["end_tick"])
        for family_rows in grouped.values()
        for first, second in zip(
            sorted(family_rows, key=lambda item: item[0]["end_tick"]),
            sorted(family_rows, key=lambda item: item[0]["end_tick"])[1:],
        )
        if int(second[0]["end_tick"]) > int(first[0]["end_tick"])
    ]
    return {
        "causal_stride_ticks": min(strides) if strides else 0,
        "overall_median": (
            float(statistics.median(all_latencies)) if all_latencies else 0.0
        ),
        "per_class": {
            name: {
                "median": (
                    float(statistics.median(detections[name]))
                    if detections[name]
                    else 0.0
                ),
                "maximum": max(detections[name]) if detections[name] else 0,
                "detected_families": len(detections[name]),
                "missed_families": missed[name],
            }
            for name in CLASS_NAMES[1:]
        },
    }
