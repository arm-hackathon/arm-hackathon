"""Deterministic four-class fault prediction and FP32 ONNX export."""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from aeolus.baseline import (
    RuleBaseline,
    RuleParameters,
    rule_parameter_grid,
)
from aeolus.config import load_scenario
from aeolus.corpus import DEFAULT_WINDOW_TICKS, EXCLUDED_TRANSITION_LABEL
from aeolus.evaluate import validate_v2_rows
from aeolus.families import FamilyEvidence, build_family_evidence, load_family_manifest
from aeolus.model_input import (
    MODEL_INPUT_SHAPE,
    build_model_input_contract,
    model_artifact_metadata,
    model_input_v1,
)
from aeolus.scenario import run_scenario

MODEL_FORMAT = "aeolus_softmax_detector_v1"
MLP_MODEL_FORMAT = "aeolus_temporal_mlp_detector_v1"
TEMPORAL_TRANSFORM_VERSION = "temporal_summary_v1"
METRICS_FORMAT = "aeolus_detector_evidence_v2"
WINDOW_TICKS = DEFAULT_WINDOW_TICKS
FEATURE_WIDTH = MODEL_INPUT_SHAPE[0]
FLAT_FEATURES = WINDOW_TICKS * FEATURE_WIDTH
TEMPORAL_FEATURES = 135
HIDDEN_UNITS = 16
CLASS_NAMES = (
    "nominal",
    "gradual_primary_fan_degradation",
    "blocked_path",
    "frozen_sensor",
)
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
_CONTRACT_KEYS = frozenset(
    {"model_input_version", "selector_sha256", "topology_sha256"}
)
_MODEL_KEYS = frozenset(
    {
        "format",
        "window_ticks",
        "feature_width",
        "class_names",
        "contract_metadata",
        "means",
        "scales",
        "weights",
        "biases",
    }
)
_MLP_MODEL_KEYS = frozenset(
    {
        "format",
        "window_ticks",
        "feature_width",
        "class_names",
        "contract_metadata",
        "transform_version",
        "means",
        "scales",
        "input_weights",
        "hidden_biases",
        "output_weights",
        "output_biases",
    }
)
USAGE = (
    "Usage:\n"
    "  PYTHONPATH=src python -m aeolus.detector train <corpus.jsonl> "
    "<families.json> <expected-family-manifest-sha256> <model.json> "
    "<model.onnx> <metrics.json>\n"
    "  PYTHONPATH=src python -m aeolus.detector predict <model.json> "
    "<scenario.json>"
)


@dataclass(frozen=True)
class Prediction:
    """One model decision with calibrated class probabilities."""

    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class SoftmaxDetector:
    """A topology-bound multinomial linear classifier over telemetry windows."""

    window_ticks: int
    feature_width: int
    class_names: tuple[str, ...]
    contract_metadata: dict[str, str]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    biases: tuple[float, ...]

    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
        """Return one probability vector per exact model-input window."""
        _validate_detector(self)
        matrix = _window_matrix(windows)
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        biases = np.asarray(self.biases, dtype=np.float64)
        logits = ((matrix - means) / scales) @ weights + biases
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(np.clip(logits, -60.0, 0.0))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict_window(self, features: Sequence[Sequence[float]]) -> Prediction:
        """Predict the fault class and confidence for one ten-tick window."""
        probabilities = self.predict_probabilities([features])[0]
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self.class_names[index],
            confidence=float(probabilities[index]),
            probabilities={
                name: float(probabilities[class_index])
                for class_index, name in enumerate(self.class_names)
            },
        )

    def label_window(self, features: list[object]) -> str:
        """Implement the evaluator's window-labeller protocol."""
        return self.predict_window(features).label

    def reset(self) -> None:
        """A compatibility no-op: this classifier has no cross-window state."""


@dataclass(frozen=True)
class TemporalMLPDetector:
    """Compact temporal-summary MLP bound to the model-input contract."""

    window_ticks: int
    feature_width: int
    class_names: tuple[str, ...]
    contract_metadata: dict[str, str]
    transform_version: str
    means: tuple[float, ...]
    scales: tuple[float, ...]
    input_weights: tuple[tuple[float, ...], ...]
    hidden_biases: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_biases: tuple[float, ...]

    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
        """Return probabilities after the embedded temporal transform."""
        _validate_detector(self)
        summaries = temporal_summary_v1(windows)
        normalised = (
            summaries - np.asarray(self.means, dtype=np.float64)
        ) / np.asarray(self.scales, dtype=np.float64)
        hidden = np.maximum(
            normalised @ np.asarray(self.input_weights, dtype=np.float64)
            + np.asarray(self.hidden_biases, dtype=np.float64),
            0.0,
        )
        logits = (
            hidden @ np.asarray(self.output_weights, dtype=np.float64)
            + np.asarray(self.output_biases, dtype=np.float64)
        )
        return _softmax(logits)

    def predict_window(self, features: Sequence[Sequence[float]]) -> Prediction:
        """Predict one exact ten-tick telemetry window."""
        probabilities = self.predict_probabilities([features])[0]
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self.class_names[index],
            confidence=float(probabilities[index]),
            probabilities={
                name: float(probabilities[class_index])
                for class_index, name in enumerate(self.class_names)
            },
        )

    def label_window(self, features: list[object]) -> str:
        """Implement the evaluator's window-labeller protocol."""
        return self.predict_window(features).label

    def reset(self) -> None:
        """A compatibility no-op: this classifier has no cross-window state."""


Detector = SoftmaxDetector | TemporalMLPDetector


def temporal_summary_v1(
    windows: Sequence[Sequence[Sequence[float]]],
) -> NDArray[np.float64]:
    """Return 135 deterministic summaries for exact ``float32[10,24]`` windows."""
    tensor = _window_tensor(windows)
    summaries = _five_temporal_summaries(tensor)
    requested = tensor[:, :, (15, 18, 21)]
    residual = tensor[:, :, (17, 20, 23)]
    ratios = np.divide(
        residual,
        requested,
        out=np.zeros_like(residual),
        where=np.abs(requested) > 1e-9,
    )
    result = np.concatenate((*summaries, *_five_temporal_summaries(ratios)), axis=1)
    if result.shape != (len(tensor), TEMPORAL_FEATURES) or not np.isfinite(result).all():
        raise ValueError("temporal summary transform produced invalid features")
    return result


def _five_temporal_summaries(
    tensor: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    ticks = np.arange(WINDOW_TICKS, dtype=np.float64)
    centred_ticks = ticks - ticks.mean()
    slopes = np.tensordot(
        tensor, centred_ticks / np.sum(centred_ticks**2), axes=([1], [0])
    )
    return (
        tensor[:, -1, :],
        tensor.mean(axis=1),
        tensor.std(axis=1),
        slopes,
        np.max(np.abs(np.diff(tensor, axis=1)), axis=1),
    )


def train_softmax_detector(
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    contract_metadata: Mapping[str, str],
    epochs: int = 600,
    learning_rate: float = 0.5,
    l2_penalty: float = 1e-4,
) -> tuple[SoftmaxDetector, dict[str, Any]]:
    """Fit deterministic class-balanced softmax regression with validation selection."""
    if epochs < 1 or epochs % 10:
        raise ValueError("training epochs must be a positive multiple of ten")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("training learning_rate must be positive and finite")
    if not math.isfinite(l2_penalty) or l2_penalty < 0.0:
        raise ValueError("training l2_penalty must be non-negative and finite")
    contract = _validated_contract_metadata(contract_metadata)
    training_features, training_labels, _ = _scored_matrix(training_rows)
    validation_features, validation_labels, _ = _scored_matrix(validation_rows)
    _require_all_classes(training_labels, "training")
    _require_all_classes(validation_labels, "validation")

    means = training_features.mean(axis=0)
    scales = training_features.std(axis=0)
    scales[scales < 1e-6] = 1.0
    normalised = (training_features - means) / scales
    validation_normalised = (validation_features - means) / scales
    class_count = len(CLASS_NAMES)
    weights = np.zeros((FLAT_FEATURES, class_count), dtype=np.float64)
    biases = np.zeros(class_count, dtype=np.float64)
    counts = np.bincount(training_labels, minlength=class_count)
    sample_weights = np.asarray(
        [1.0 / (class_count * counts[label]) for label in training_labels],
        dtype=np.float64,
    )
    best_score = -1.0
    best_loss = math.inf
    best_epoch = 0
    best_weights = weights.copy()
    best_biases = biases.copy()

    for epoch in range(1, epochs + 1):
        probabilities = _softmax(normalised @ weights + biases)
        probabilities[np.arange(len(training_labels)), training_labels] -= 1.0
        errors = probabilities * sample_weights[:, None]
        weights -= learning_rate * (
            normalised.T @ errors + l2_penalty * weights
        )
        biases -= learning_rate * errors.sum(axis=0)

        if epoch % 10 == 0:
            validation_probabilities = _softmax(
                validation_normalised @ weights + biases
            )
            validation_predictions = np.argmax(validation_probabilities, axis=1)
            validation_metrics = classification_metrics(
                validation_labels,
                validation_predictions,
                validation_probabilities,
            )
            validation_loss = _cross_entropy(
                validation_labels, validation_probabilities
            )
            score = float(validation_metrics["macro_f1"])
            if score > best_score + 1e-12 or (
                math.isclose(score, best_score, abs_tol=1e-12)
                and validation_loss < best_loss
            ):
                best_score = score
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = weights.copy()
                best_biases = biases.copy()

    detector = SoftmaxDetector(
        window_ticks=WINDOW_TICKS,
        feature_width=FEATURE_WIDTH,
        class_names=CLASS_NAMES,
        contract_metadata=contract,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        weights=tuple(
            tuple(float(value) for value in row) for row in best_weights
        ),
        biases=tuple(float(value) for value in best_biases),
    )
    _validate_detector(detector)
    return detector, {
        "epochs_requested": epochs,
        "selected_epoch": best_epoch,
        "learning_rate": learning_rate,
        "l2_penalty": l2_penalty,
        "validation_macro_f1": best_score,
        "validation_cross_entropy": best_loss,
        "training_rows": len(training_labels),
        "validation_rows": len(validation_labels),
        "training_class_counts": {
            name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "class_weight_total_per_class": 1.0 / class_count,
    }


def train_temporal_mlp_detector(
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    contract_metadata: Mapping[str, str],
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2_penalty: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    initialization_seed: int = 0,
) -> tuple[TemporalMLPDetector, dict[str, Any]]:
    """Fit the deterministic 135-16-4 temporal MLP with Adam."""
    if epochs < 5 or epochs % 5:
        raise ValueError("MLP epochs must be a positive multiple of five")
    hyperparameters = (learning_rate, l2_penalty, beta1, beta2, epsilon)
    if any(not math.isfinite(value) for value in hyperparameters):
        raise ValueError("MLP hyperparameters must be finite")
    if learning_rate <= 0.0 or l2_penalty < 0.0 or epsilon <= 0.0:
        raise ValueError("MLP learning rate/epsilon must be positive and L2 non-negative")
    if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
        raise ValueError("MLP Adam beta values must be between zero and one")
    contract = _validated_contract_metadata(contract_metadata)
    training_flat, training_labels, _ = _scored_matrix(training_rows)
    validation_flat, validation_labels, _ = _scored_matrix(validation_rows)
    _require_all_classes(training_labels, "training")
    _require_all_classes(validation_labels, "validation")
    training_features = temporal_summary_v1(
        training_flat.reshape((-1, WINDOW_TICKS, FEATURE_WIDTH))
    )
    validation_features = temporal_summary_v1(
        validation_flat.reshape((-1, WINDOW_TICKS, FEATURE_WIDTH))
    )
    means = training_features.mean(axis=0)
    scales = training_features.std(axis=0)
    scales[scales < 1e-6] = 1.0
    training_normalised = (training_features - means) / scales
    validation_normalised = (validation_features - means) / scales

    class_count = len(CLASS_NAMES)
    counts = np.bincount(training_labels, minlength=class_count)
    sample_weights = np.asarray(
        [1.0 / (class_count * counts[label]) for label in training_labels],
        dtype=np.float64,
    )
    generator = np.random.default_rng(initialization_seed)
    parameters = {
        "input_weights": generator.normal(
            0.0, math.sqrt(2.0 / TEMPORAL_FEATURES),
            size=(TEMPORAL_FEATURES, HIDDEN_UNITS),
        ),
        "hidden_biases": np.zeros(HIDDEN_UNITS, dtype=np.float64),
        "output_weights": generator.normal(
            0.0, math.sqrt(2.0 / HIDDEN_UNITS),
            size=(HIDDEN_UNITS, class_count),
        ),
        "output_biases": np.zeros(class_count, dtype=np.float64),
    }
    first_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    best_score = -1.0
    best_loss = math.inf
    best_epoch = 0
    best_parameters = {name: value.copy() for name, value in parameters.items()}

    for epoch in range(1, epochs + 1):
        hidden_logits = (
            training_normalised @ parameters["input_weights"]
            + parameters["hidden_biases"]
        )
        hidden = np.maximum(hidden_logits, 0.0)
        probabilities = _softmax(
            hidden @ parameters["output_weights"] + parameters["output_biases"]
        )
        errors = probabilities.copy()
        errors[np.arange(len(training_labels)), training_labels] -= 1.0
        errors *= sample_weights[:, None]
        hidden_errors = errors @ parameters["output_weights"].T
        hidden_errors *= hidden_logits > 0.0
        gradients = {
            "input_weights": training_normalised.T @ hidden_errors
            + l2_penalty * parameters["input_weights"],
            "hidden_biases": hidden_errors.sum(axis=0),
            "output_weights": hidden.T @ errors
            + l2_penalty * parameters["output_weights"],
            "output_biases": errors.sum(axis=0),
        }
        for name in parameters:
            first_moments[name] = beta1 * first_moments[name] + (1.0 - beta1) * gradients[name]
            second_moments[name] = beta2 * second_moments[name] + (1.0 - beta2) * (
                gradients[name] ** 2
            )
            corrected_first = first_moments[name] / (1.0 - beta1**epoch)
            corrected_second = second_moments[name] / (1.0 - beta2**epoch)
            parameters[name] -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )

        if epoch % 5 == 0:
            validation_hidden = np.maximum(
                validation_normalised @ parameters["input_weights"]
                + parameters["hidden_biases"],
                0.0,
            )
            validation_probabilities = _softmax(
                validation_hidden @ parameters["output_weights"]
                + parameters["output_biases"]
            )
            validation_metrics = classification_metrics(
                validation_labels,
                np.argmax(validation_probabilities, axis=1),
                validation_probabilities,
            )
            score = float(validation_metrics["macro_f1"])
            loss = _cross_entropy(validation_labels, validation_probabilities)
            if score > best_score + 1e-12 or (
                math.isclose(score, best_score, abs_tol=1e-12) and loss < best_loss
            ):
                best_score = score
                best_loss = loss
                best_epoch = epoch
                best_parameters = {
                    name: value.copy() for name, value in parameters.items()
                }

    detector = TemporalMLPDetector(
        window_ticks=WINDOW_TICKS,
        feature_width=FEATURE_WIDTH,
        class_names=CLASS_NAMES,
        contract_metadata=contract,
        transform_version=TEMPORAL_TRANSFORM_VERSION,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        input_weights=tuple(
            tuple(float(value) for value in row)
            for row in best_parameters["input_weights"]
        ),
        hidden_biases=tuple(float(value) for value in best_parameters["hidden_biases"]),
        output_weights=tuple(
            tuple(float(value) for value in row)
            for row in best_parameters["output_weights"]
        ),
        output_biases=tuple(float(value) for value in best_parameters["output_biases"]),
    )
    _validate_detector(detector)
    return detector, {
        "architecture": [TEMPORAL_FEATURES, HIDDEN_UNITS, class_count],
        "transform_version": TEMPORAL_TRANSFORM_VERSION,
        "epochs_requested": epochs,
        "selected_epoch": best_epoch,
        "learning_rate": learning_rate,
        "l2_penalty": l2_penalty,
        "adam_beta1": beta1,
        "adam_beta2": beta2,
        "adam_epsilon": epsilon,
        "initialization_seed": initialization_seed,
        "validation_macro_f1": best_score,
        "validation_cross_entropy": best_loss,
        "training_rows": len(training_labels),
        "validation_rows": len(validation_labels),
        "training_class_counts": {
            name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "class_weight_total_per_class": 1.0 / class_count,
    }


def classification_metrics(
    labels: Sequence[int] | NDArray[np.int64],
    predictions: Sequence[int] | NDArray[np.int64],
    probabilities: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Return stable multiclass quality metrics in the declared class order."""
    truth = np.asarray(labels, dtype=np.int64)
    inferred = np.asarray(predictions, dtype=np.int64)
    if truth.ndim != 1 or inferred.shape != truth.shape or not len(truth):
        raise ValueError("classification metrics require aligned non-empty labels")
    class_count = len(CLASS_NAMES)
    if np.any(truth < 0) or np.any(truth >= class_count):
        raise ValueError("classification metrics contain an unsupported true label")
    if np.any(inferred < 0) or np.any(inferred >= class_count):
        raise ValueError("classification metrics contain an unsupported prediction")
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for expected, actual in zip(truth, inferred):
        confusion[int(expected), int(actual)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_support = int(confusion[:, index].sum())
        precision = true_positive / predicted_support if predicted_support else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    nominal_index = CLASS_INDEX["nominal"]
    nominal_support = int(confusion[nominal_index, :].sum())
    nominal_false_alarms = nominal_support - int(
        confusion[nominal_index, nominal_index]
    )
    metrics: dict[str, Any] = {
        "samples": int(len(truth)),
        "accuracy": float(np.mean(truth == inferred)),
        "macro_f1": float(np.mean(f1_values)),
        "nominal_false_alarm_rate": (
            nominal_false_alarms / nominal_support if nominal_support else 0.0
        ),
        "nominal_false_alarms": nominal_false_alarms,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if probability_array.shape != (len(truth), class_count):
            raise ValueError("classification probabilities have an unexpected shape")
        if not np.isfinite(probability_array).all():
            raise ValueError("classification probabilities must be finite")
        expected = np.eye(class_count, dtype=np.float64)[truth]
        metrics["brier_score"] = float(
            np.mean(np.sum((probability_array - expected) ** 2, axis=1))
        )
        metrics["cross_entropy"] = _cross_entropy(truth, probability_array)
    return metrics


def evaluate_detector(
    detector: Detector,
    rows: Sequence[Mapping[str, Any]],
    *,
    family_evidence: Mapping[str, FamilyEvidence] | None = None,
) -> dict[str, Any]:
    """Evaluate one detector on already verified rows from one split."""
    features, labels, scored_rows = _scored_matrix(rows)
    windows = features.reshape((-1, WINDOW_TICKS, FEATURE_WIDTH))
    probabilities = detector.predict_probabilities(windows)
    predictions = np.argmax(probabilities, axis=1)
    metrics = classification_metrics(labels, predictions, probabilities)
    metrics["detection_latency_ticks"] = (
        _rolling_latency_metrics(detector, family_evidence)
        if family_evidence is not None
        else _latency_metrics(scored_rows, predictions)
    )
    return metrics


def evaluate_rule_baseline(
    rows: Sequence[Mapping[str, Any]],
    baseline: RuleBaseline,
    *,
    family_evidence: Mapping[str, FamilyEvidence] | None = None,
) -> dict[str, Any]:
    """Evaluate the streaming rule on the same ordered split evidence."""
    ordered = sorted(
        rows,
        key=lambda row: (row["family_id"], row["scenario_role"], row["end_tick"]),
    )
    current_stream: tuple[str, str] | None = None
    labels: list[int] = []
    predictions: list[int] = []
    scored_rows: list[Mapping[str, Any]] = []
    for row in ordered:
        stream = (row["family_id"], row["scenario_role"])
        if stream != current_stream:
            current_stream = stream
            baseline.reset()
        predicted = baseline.label_window(row["features"])
        if row["label"] == EXCLUDED_TRANSITION_LABEL:
            continue
        labels.append(CLASS_INDEX[row["label"]])
        predictions.append(CLASS_INDEX[predicted])
        scored_rows.append(row)
    metrics = classification_metrics(labels, predictions)
    metrics["detection_latency_ticks"] = (
        _rolling_latency_metrics(baseline, family_evidence)
        if family_evidence is not None
        else _latency_metrics(scored_rows, np.asarray(predictions, dtype=np.int64))
    )
    return metrics


def save_detector(detector: Detector, path: str | Path) -> Path:
    """Persist a strict, canonical JSON representation of the detector."""
    _validate_detector(detector)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = _detector_document(detector)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return destination


def _detector_document(detector: Detector) -> dict[str, Any]:
    """Return the exact strict JSON document used for persistence and sizing."""
    _validate_detector(detector)
    common = {
        "window_ticks": detector.window_ticks,
        "feature_width": detector.feature_width,
        "class_names": list(detector.class_names),
        "contract_metadata": dict(detector.contract_metadata),
        "means": list(detector.means),
        "scales": list(detector.scales),
    }
    if isinstance(detector, SoftmaxDetector):
        document = {
            "format": MODEL_FORMAT,
            **common,
            "weights": [list(row) for row in detector.weights],
            "biases": list(detector.biases),
        }
    else:
        document = {
            "format": MLP_MODEL_FORMAT,
            **common,
            "transform_version": detector.transform_version,
            "input_weights": [list(row) for row in detector.input_weights],
            "hidden_biases": list(detector.hidden_biases),
            "output_weights": [list(row) for row in detector.output_weights],
            "output_biases": list(detector.output_biases),
        }
    return document


def _detector_serialized_size(detector: Detector) -> int:
    document = _detector_document(detector)
    return len(
        (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )


def load_detector(
    path: str | Path, *, expected_contract: Mapping[str, str] | None = None
) -> Detector:
    """Load a detector and fail closed on schema or contract drift."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"detector artifact not found: {source}") from None
    except OSError as exc:
        raise ValueError(f"cannot read detector artifact {source}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"detector artifact is not valid JSON: {exc}") from None
    if not isinstance(document, dict):
        raise ValueError("detector artifact schema is incompatible")
    model_format = document.get("format")
    expected_keys = (
        _MODEL_KEYS if model_format == MODEL_FORMAT else _MLP_MODEL_KEYS
        if model_format == MLP_MODEL_FORMAT else None
    )
    if expected_keys is None:
        raise ValueError("detector artifact format is incompatible")
    if set(document) != expected_keys:
        raise ValueError("detector artifact schema is incompatible")
    if (
        not isinstance(document["class_names"], list)
        or not isinstance(document["contract_metadata"], dict)
        or not isinstance(document["means"], list)
        or not isinstance(document["scales"], list)
    ):
        raise ValueError("detector artifact fields are malformed")
    try:
        common_arguments = {
            "window_ticks": document["window_ticks"],
            "feature_width": document["feature_width"],
            "class_names": tuple(document["class_names"]),
            "contract_metadata": dict(document["contract_metadata"]),
            "means": tuple(document["means"]),
            "scales": tuple(document["scales"]),
        }
        if model_format == MODEL_FORMAT:
            if (
                not isinstance(document["weights"], list)
                or any(not isinstance(row, list) for row in document["weights"])
                or not isinstance(document["biases"], list)
            ):
                raise ValueError
            detector: Detector = SoftmaxDetector(
                **common_arguments,
                weights=tuple(tuple(row) for row in document["weights"]),
                biases=tuple(document["biases"]),
            )
        else:
            if (
                not isinstance(document["transform_version"], str)
                or not isinstance(document["input_weights"], list)
                or any(not isinstance(row, list) for row in document["input_weights"])
                or not isinstance(document["hidden_biases"], list)
                or not isinstance(document["output_weights"], list)
                or any(not isinstance(row, list) for row in document["output_weights"])
                or not isinstance(document["output_biases"], list)
            ):
                raise ValueError
            detector = TemporalMLPDetector(
                **common_arguments,
                transform_version=document["transform_version"],
                input_weights=tuple(
                    tuple(row) for row in document["input_weights"]
                ),
                hidden_biases=tuple(document["hidden_biases"]),
                output_weights=tuple(
                    tuple(row) for row in document["output_weights"]
                ),
                output_biases=tuple(document["output_biases"]),
            )
    except (TypeError, ValueError) as exc:
        raise ValueError("detector artifact fields are malformed") from exc
    _validate_detector(detector)
    if expected_contract is not None:
        if detector.contract_metadata != _validated_contract_metadata(expected_contract):
            raise ValueError(
                "detector artifact contract does not match this inference setup"
            )
    return detector


def export_onnx(detector: Detector, path: str | Path) -> Path:
    """Export preprocessing, Gemm and Softmax as one FP32 ONNX graph."""
    _validate_detector(detector)
    if isinstance(detector, TemporalMLPDetector):
        return _export_temporal_mlp_onnx(detector, path)
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ONNX export requires the 'ml' project extra") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    means = np.asarray(detector.means, dtype=np.float32)
    scales = np.asarray(detector.scales, dtype=np.float32)
    weights = np.asarray(detector.weights, dtype=np.float32)
    biases = np.asarray(detector.biases, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["window"], ["flat"], axis=1),
            helper.make_node("Sub", ["flat", "means"], ["centred"]),
            helper.make_node("Div", ["centred", "scales"], ["normalised"]),
            helper.make_node(
                "Gemm",
                ["normalised", "weights", "biases"],
                ["logits"],
                transB=0,
            ),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "aeolus_fault_detector",
        [
            helper.make_tensor_value_info(
                "window",
                TensorProto.FLOAT,
                [None, detector.window_ticks, detector.feature_width],
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, len(CLASS_NAMES)]
            )
        ],
        initializer=[
            numpy_helper.from_array(means, name="means"),
            numpy_helper.from_array(scales, name="scales"),
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(biases, name="biases"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="aeolus",
        producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    metadata = {
        "format": MODEL_FORMAT,
        "window_ticks": str(detector.window_ticks),
        "feature_width": str(detector.feature_width),
        "class_names": json.dumps(list(detector.class_names), separators=(",", ":")),
        **detector.contract_metadata,
    }
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return destination


def _export_temporal_mlp_onnx(
    detector: TemporalMLPDetector, path: str | Path
) -> Path:
    """Export the temporal summary, normalisation and MLP in one FP32 graph."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ONNX export requires the 'ml' project extra") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    initializers = []

    def add_array(name: str, value: object, dtype=np.float32) -> None:
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name))

    add_array("last_index", 9, np.int64)
    add_array("slice_starts_later", [1], np.int64)
    add_array("slice_ends_later", [WINDOW_TICKS], np.int64)
    add_array("slice_starts_earlier", [0], np.int64)
    add_array("slice_ends_earlier", [WINDOW_TICKS - 1], np.int64)
    add_array("slice_axes", [1], np.int64)
    add_array("slice_steps", [1], np.int64)
    ticks = np.arange(WINDOW_TICKS, dtype=np.float32)
    centred = ticks - ticks.mean()
    add_array("slope_coefficients", (centred / np.sum(centred**2)).reshape((-1, 1)))
    add_array("raw_summary_shape", [-1, FEATURE_WIDTH], np.int64)
    add_array("ratio_summary_shape", [-1, 3], np.int64)
    add_array("requested_indices", [15, 18, 21], np.int64)
    add_array("residual_indices", [17, 20, 23], np.int64)
    add_array("ratio_epsilon", 1e-9)
    add_array("one", 1.0)
    add_array("zero", 0.0)

    def summary_nodes(input_name: str, prefix: str, width: int) -> list[str]:
        last = f"{prefix}_last"
        mean = f"{prefix}_mean"
        mean_keep = f"{prefix}_mean_keep"
        centred_name = f"{prefix}_centred"
        squared = f"{prefix}_squared"
        variance = f"{prefix}_variance"
        std = f"{prefix}_std"
        transposed = f"{prefix}_transposed"
        slope_rank3 = f"{prefix}_slope_rank3"
        slope = f"{prefix}_slope"
        later = f"{prefix}_later"
        earlier = f"{prefix}_earlier"
        difference = f"{prefix}_difference"
        absolute_difference = f"{prefix}_absolute_difference"
        maximum_difference = f"{prefix}_maximum_difference"
        nodes.extend(
            [
                helper.make_node("Gather", [input_name, "last_index"], [last], axis=1),
                helper.make_node("ReduceMean", [input_name], [mean], axes=[1], keepdims=0),
                helper.make_node("ReduceMean", [input_name], [mean_keep], axes=[1], keepdims=1),
                helper.make_node("Sub", [input_name, mean_keep], [centred_name]),
                helper.make_node("Mul", [centred_name, centred_name], [squared]),
                helper.make_node("ReduceMean", [squared], [variance], axes=[1], keepdims=0),
                helper.make_node("Sqrt", [variance], [std]),
                helper.make_node("Transpose", [input_name], [transposed], perm=[0, 2, 1]),
                helper.make_node("MatMul", [transposed, "slope_coefficients"], [slope_rank3]),
                helper.make_node(
                    "Reshape",
                    [slope_rank3, "raw_summary_shape" if width == FEATURE_WIDTH else "ratio_summary_shape"],
                    [slope],
                ),
                helper.make_node(
                    "Slice",
                    [input_name, "slice_starts_later", "slice_ends_later", "slice_axes", "slice_steps"],
                    [later],
                ),
                helper.make_node(
                    "Slice",
                    [input_name, "slice_starts_earlier", "slice_ends_earlier", "slice_axes", "slice_steps"],
                    [earlier],
                ),
                helper.make_node("Sub", [later, earlier], [difference]),
                helper.make_node("Abs", [difference], [absolute_difference]),
                helper.make_node(
                    "ReduceMax", [absolute_difference], [maximum_difference], axes=[1], keepdims=0
                ),
            ]
        )
        return [last, mean, std, slope, maximum_difference]

    raw_summaries = summary_nodes("window", "raw", FEATURE_WIDTH)
    nodes.extend(
        [
            helper.make_node(
                "Gather", ["window", "requested_indices"], ["requested"], axis=2
            ),
            helper.make_node(
                "Gather", ["window", "residual_indices"], ["residual"], axis=2
            ),
            helper.make_node("Abs", ["requested"], ["absolute_requested"]),
            helper.make_node(
                "Greater", ["absolute_requested", "ratio_epsilon"], ["has_request"]
            ),
            helper.make_node(
                "Where", ["has_request", "requested", "one"], ["safe_requested"]
            ),
            helper.make_node("Div", ["residual", "safe_requested"], ["raw_ratio"]),
            helper.make_node(
                "Where", ["has_request", "raw_ratio", "zero"], ["ratios"]
            ),
        ]
    )
    ratio_summaries = summary_nodes("ratios", "ratio", 3)
    nodes.append(
        helper.make_node(
            "Concat", [*raw_summaries, *ratio_summaries], ["summaries"], axis=1
        )
    )
    add_array("means", detector.means)
    add_array("scales", detector.scales)
    add_array("input_weights", detector.input_weights)
    add_array("hidden_biases", detector.hidden_biases)
    add_array("output_weights", detector.output_weights)
    add_array("output_biases", detector.output_biases)
    nodes.extend(
        [
            helper.make_node("Sub", ["summaries", "means"], ["centred_summaries"]),
            helper.make_node("Div", ["centred_summaries", "scales"], ["normalised"]),
            helper.make_node(
                "Gemm", ["normalised", "input_weights", "hidden_biases"], ["hidden_logits"]
            ),
            helper.make_node("Relu", ["hidden_logits"], ["hidden"]),
            helper.make_node(
                "Gemm", ["hidden", "output_weights", "output_biases"], ["logits"]
            ),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "aeolus_temporal_mlp_fault_detector",
        [
            helper.make_tensor_value_info(
                "window", TensorProto.FLOAT, [None, WINDOW_TICKS, FEATURE_WIDTH]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, len(CLASS_NAMES)]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="aeolus",
        producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    metadata = {
        "format": MLP_MODEL_FORMAT,
        "transform_version": TEMPORAL_TRANSFORM_VERSION,
        "window_ticks": str(detector.window_ticks),
        "feature_width": str(detector.feature_width),
        "class_names": json.dumps(list(detector.class_names), separators=(",", ":")),
        **detector.contract_metadata,
    }
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return destination


def validate_onnx_parity(
    detector: Detector,
    onnx_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_samples: int = 512,
) -> dict[str, Any]:
    """Measure Python-versus-ONNX probability agreement."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ONNX parity requires the 'ml' project extra") from exc
    features, _, _ = _scored_matrix(rows)
    windows = features[:maximum_samples].reshape((-1, WINDOW_TICKS, FEATURE_WIDTH))
    expected = detector.predict_probabilities(windows)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        ["probabilities"], {"window": windows.astype(np.float32)}
    )[0]
    maximum_error = float(np.max(np.abs(expected - actual)))
    return {
        "samples_checked": int(len(windows)),
        "max_absolute_probability_error": maximum_error,
    }


def calibrate_rule_baseline(
    validation_rows: Sequence[Mapping[str, Any]],
    config: Any,
    family_evidence: Mapping[str, FamilyEvidence],
) -> tuple[RuleParameters, dict[str, Any]]:
    """Select robust rule parameters using validation evidence only."""
    candidates: list[tuple[RuleParameters, dict[str, Any]]] = []
    best_macro = -1.0
    best_false_alarm = math.inf
    for parameters in rule_parameter_grid():
        metrics = evaluate_rule_baseline(
            validation_rows, RuleBaseline(config, parameters)
        )
        macro = float(metrics["macro_f1"])
        false_alarm = float(metrics["nominal_false_alarm_rate"])
        if macro > best_macro + 1e-12 or (
            math.isclose(macro, best_macro, abs_tol=1e-12)
            and false_alarm < best_false_alarm - 1e-12
        ):
            best_macro = macro
            best_false_alarm = false_alarm
            candidates = [(parameters, metrics)]
        elif math.isclose(macro, best_macro, abs_tol=1e-12) and math.isclose(
            false_alarm, best_false_alarm, abs_tol=1e-12
        ):
            candidates.append((parameters, metrics))

    ranked: list[tuple[float, RuleParameters, dict[str, Any]]] = []
    for parameters, _ in candidates:
        metrics = evaluate_rule_baseline(
            validation_rows,
            RuleBaseline(config, parameters),
            family_evidence=family_evidence,
        )
        ranked.append(
            (
                float(metrics["detection_latency_ticks"]["overall_median"]),
                parameters,
                metrics,
            )
        )
    latency, selected, selected_metrics = min(
        ranked, key=lambda item: (item[0], item[1])
    )
    return selected, {
        "selection_split": "validation",
        "grid_size": len(rule_parameter_grid()),
        "classification_tied_candidates": len(candidates),
        "selection_order": [
            "macro_f1_descending",
            "nominal_false_alarm_rate_ascending",
            "causal_latency_ascending",
            "parameters_lexicographic",
        ],
        "selected_parameters": selected.as_dict(),
        "selected_validation_metrics": selected_metrics,
        "selected_validation_latency": latency,
    }


def train_and_export(
    corpus_path: str | Path,
    family_manifest_path: str | Path,
    expected_manifest_sha256: str,
    model_json_path: str | Path,
    model_onnx_path: str | Path,
    metrics_path: str | Path,
) -> dict[str, Any]:
    """Validate evidence, select on validation, then evaluate held-out partitions once."""
    rows, manifest = load_verified_corpus(
        corpus_path, family_manifest_path, expected_manifest_sha256
    )
    split_names = tuple(
        split
        for split in ("train", "validation", "test", "stress")
        if any(row["split"] == split for row in rows)
    )
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in split_names
    }
    if any(split not in split_rows or not split_rows[split] for split in ("train", "validation", "test")):
        raise ValueError("detector training requires non-empty train, validation and test splits")
    softmax_detector, softmax_receipt = train_softmax_detector(
        split_rows["train"],
        split_rows["validation"],
        contract_metadata=manifest.contract_metadata,
    )
    mlp_detector, mlp_receipt = train_temporal_mlp_detector(
        split_rows["train"],
        split_rows["validation"],
        contract_metadata=manifest.contract_metadata,
    )
    candidates: list[tuple[str, Detector, dict[str, Any], dict[str, Any], int]] = []
    for name, candidate, receipt in (
        ("softmax_detector", softmax_detector, softmax_receipt),
        ("temporal_mlp_detector", mlp_detector, mlp_receipt),
    ):
        validation_metrics = evaluate_detector(candidate, split_rows["validation"])
        candidates.append(
            (
                name,
                candidate,
                receipt,
                validation_metrics,
                _detector_serialized_size(candidate),
            )
        )
    selected_name, detector, _, selected_validation, _ = min(
        candidates,
        key=lambda item: (
            -float(item[3]["macro_f1"]),
            float(item[3]["cross_entropy"]),
            item[4],
            item[0],
        ),
    )

    evidence = build_family_evidence(manifest)
    evidence_by_split = {
        split: {
            family_id: item
            for family_id, item in evidence.items()
            if item.split == split
        }
        for split in split_names
    }
    baseline_config = load_scenario(manifest.families[0].reference_path)
    rule_parameters, rule_calibration = calibrate_rule_baseline(
        split_rows["validation"],
        baseline_config,
        evidence_by_split["validation"],
    )
    model_json = save_detector(detector, model_json_path)
    model_onnx = export_onnx(detector, model_onnx_path)
    model_metrics = evaluate_detector(
        detector,
        split_rows["test"],
        family_evidence=evidence_by_split["test"],
    )
    rule_metrics = evaluate_rule_baseline(
        split_rows["test"],
        RuleBaseline(baseline_config, rule_parameters),
        family_evidence=evidence_by_split["test"],
    )
    stress_metrics = None
    stress_rule_metrics = None
    if "stress" in split_rows:
        stress_metrics = evaluate_detector(
            detector,
            split_rows["stress"],
            family_evidence=evidence_by_split["stress"],
        )
        stress_rule_metrics = evaluate_rule_baseline(
            split_rows["stress"],
            RuleBaseline(baseline_config, rule_parameters),
            family_evidence=evidence_by_split["stress"],
        )
    family_counts = {split: 0 for split in split_names}
    for family in manifest.families:
        family_counts[family.split] += 1
    conclusion = evidence_conclusion(
        model_metrics, rule_metrics, model_name=selected_name
    )
    metrics: dict[str, Any] = {
        "format": METRICS_FORMAT,
        "observable_features_only": True,
        "window_ticks": WINDOW_TICKS,
        "feature_width": FEATURE_WIDTH,
        "class_names": list(CLASS_NAMES),
        "contract_metadata": dict(manifest.contract_metadata),
        "family_manifest_sha256": manifest.manifest_sha256,
        "families_by_split": family_counts,
        "split_evidence": {
            split: _split_evidence(split_rows[split], family_counts[split])
            for split in split_names
        },
        "latency_reference": "observable_onset_tick_stride_one_causal_windows",
        "training": {
            "softmax_detector": softmax_receipt,
            "temporal_mlp_detector": mlp_receipt,
        },
        "candidate_selection": {
            "selection_split": "validation",
            "selection_order": [
                "macro_f1_descending",
                "cross_entropy_ascending",
                "serialized_json_size_ascending",
                "candidate_name_lexicographic",
            ],
            "selected_candidate": selected_name,
            "selected_validation_metrics": selected_validation,
            "candidates": {
                name: {
                    "validation_metrics": validation_metrics,
                    "serialized_json_size_bytes": size,
                }
                for name, _, _, validation_metrics, size in candidates
            },
        },
        "rule_calibration": rule_calibration,
        "model": model_metrics,
        "rule_baseline": rule_metrics,
        "stress_model": stress_metrics,
        "stress_rule_baseline": stress_rule_metrics,
        "evidence_conclusion": conclusion,
        "onnx_parity": validate_onnx_parity(
            detector, model_onnx, split_rows["test"]
        ),
        "artifact_sizes_bytes": {
            "model_json": model_json.stat().st_size,
            "model_onnx": model_onnx.stat().st_size,
        },
    }
    _write_metrics(metrics, metrics_path)
    return metrics


def load_verified_corpus(
    corpus_path: str | Path,
    family_manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], Any]:
    """Load corpus rows only after binding them to trusted family replay evidence."""
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("expected family manifest hash must be lowercase SHA-256")
    manifest = load_family_manifest(Path(family_manifest_path))
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("family manifest does not match the expected SHA-256")
    rows = _load_jsonl(corpus_path)
    window_ticks = validate_v2_rows(
        rows,
        manifest.contract_metadata,
        build_family_evidence(manifest),
    )
    if window_ticks != WINDOW_TICKS:
        raise ValueError(
            f"detector requires {WINDOW_TICKS}-tick windows, got {window_ticks}"
        )
    return rows, manifest


def predict_scenario(
    detector_path: str | Path, scenario_path: str | Path
) -> list[dict[str, Any]]:
    """Run rolling inference over one validated scenario."""
    config = load_scenario(scenario_path)
    contract = build_model_input_contract(config)
    expected_contract = model_artifact_metadata(contract)
    detector = load_detector(detector_path, expected_contract=expected_contract)
    vectors = [model_input_v1(record, contract).tolist() for record in run_scenario(config)]
    predictions: list[dict[str, Any]] = []
    for end_index in range(detector.window_ticks - 1, len(vectors)):
        start_index = end_index - detector.window_ticks + 1
        prediction = detector.predict_window(vectors[start_index : end_index + 1])
        predictions.append(
            {
                "end_tick": end_index + 1,
                "label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            }
        )
    return predictions


def evidence_conclusion(
    model: Mapping[str, Any],
    rule: Mapping[str, Any],
    *,
    model_name: str = "learned_detector",
) -> dict[str, Any]:
    """Apply the declared quality-or-latency advantage policy honestly."""
    macro_gain = float(model["macro_f1"]) - float(rule["macro_f1"])
    rule_error = 1.0 - float(rule["macro_f1"])
    model_error = 1.0 - float(model["macro_f1"])
    error_reduction = (
        (rule_error - model_error) / rule_error
        if rule_error > 1e-12
        else (1.0 if model_error < rule_error - 1e-12 else 0.0)
    )
    false_alarm_regression = float(model["nominal_false_alarm_rate"]) - float(
        rule["nominal_false_alarm_rate"]
    )
    model_latency = float(model["detection_latency_ticks"]["overall_median"])
    rule_latency = float(rule["detection_latency_ticks"]["overall_median"])
    latency_reduction = (
        (rule_latency - model_latency) / rule_latency if rule_latency > 0.0 else 0.0
    )
    recall_deltas = {
        name: float(model["per_class"][name]["recall"])
        - float(rule["per_class"][name]["recall"])
        for name in CLASS_NAMES[1:]
    }
    quality_win = (
        error_reduction >= 0.25
        and false_alarm_regression <= 0.01
        and min(recall_deltas.values()) >= -0.02
    )
    latency_win = (
        latency_reduction >= 0.20
        and macro_gain >= -0.01
        and min(recall_deltas.values()) >= -0.02
        and false_alarm_regression <= 0.01
    )
    demonstrated = quality_win or latency_win
    return {
        "criterion": (
            "macro-F1 error reduction >= 25% with false-alarm regression <= 0.01 "
            "and every fault recall within 0.02, or latency reduction >= 20% "
            "with macro-F1 within 0.01, every fault recall within 0.02, and "
            "false-alarm regression <= 0.01"
        ),
        "ai_advantage_demonstrated": demonstrated,
        "preferred_method": model_name if demonstrated else "rule_baseline",
        "macro_f1_gain": macro_gain,
        "macro_f1_error_reduction_fraction": error_reduction,
        "nominal_false_alarm_regression": false_alarm_regression,
        "latency_reduction_fraction": latency_reduction,
        "fault_recall_deltas": recall_deltas,
    }


def _scored_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[Mapping[str, Any]]]:
    scored = [row for row in rows if row["label"] != EXCLUDED_TRANSITION_LABEL]
    if not scored:
        raise ValueError("detector dataset contains no scored rows")
    unsupported = sorted({row["label"] for row in scored} - set(CLASS_NAMES))
    if unsupported:
        raise ValueError(f"detector dataset contains unsupported label {unsupported[0]!r}")
    windows = [row["features"] for row in scored]
    matrix = _window_matrix(windows)
    labels = np.asarray([CLASS_INDEX[row["label"]] for row in scored], dtype=np.int64)
    return matrix, labels, scored


def _window_matrix(
    windows: Sequence[Sequence[Sequence[float]]],
) -> NDArray[np.float64]:
    return _window_tensor(windows).reshape((-1, FLAT_FEATURES))


def _window_tensor(
    windows: Sequence[Sequence[Sequence[float]]],
) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(windows)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector windows must be a rectangular numeric array") from exc
    if matrix.ndim != 3 or matrix.shape[0] == 0:
        raise ValueError("detector requires at least one feature window")
    if matrix.shape[1] != WINDOW_TICKS:
        raise ValueError(f"detector window must contain exactly {WINDOW_TICKS} ticks")
    if matrix.shape[2] != FEATURE_WIDTH:
        raise ValueError(f"detector tick must contain exactly {FEATURE_WIDTH} features")
    if any(
        isinstance(value, (bool, np.bool_))
        for value in np.asarray(windows, dtype=object).flat
    ):
        raise ValueError("detector features must be numeric")
    if matrix.dtype.kind not in "fiu":
        raise ValueError("detector features must be numeric")
    matrix = matrix.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("detector features must be finite")
    with np.errstate(over="ignore"):
        float32_matrix = matrix.astype(np.float32)
    if not np.isfinite(float32_matrix).all():
        raise ValueError("detector features must be representable as float32")
    return float32_matrix.astype(np.float64)


def _validate_detector(detector: Detector) -> None:
    if (
        isinstance(detector.window_ticks, bool)
        or not isinstance(detector.window_ticks, int)
        or isinstance(detector.feature_width, bool)
        or not isinstance(detector.feature_width, int)
        or detector.window_ticks != WINDOW_TICKS
        or detector.feature_width != FEATURE_WIDTH
    ):
        raise ValueError("detector input shape is incompatible")
    if detector.class_names != CLASS_NAMES:
        raise ValueError("detector class vocabulary is incompatible")
    _validated_contract_metadata(detector.contract_metadata)
    expected_features = (
        TEMPORAL_FEATURES
        if isinstance(detector, TemporalMLPDetector)
        else FLAT_FEATURES
    )
    if len(detector.means) != expected_features or len(detector.scales) != expected_features:
        raise ValueError("detector normalization shape is incompatible")
    if isinstance(detector, SoftmaxDetector):
        if len(detector.weights) != FLAT_FEATURES or any(
            len(row) != len(CLASS_NAMES) for row in detector.weights
        ):
            raise ValueError("detector weight shape is incompatible")
        if len(detector.biases) != len(CLASS_NAMES):
            raise ValueError("detector bias shape is incompatible")
        numeric_values = [
            *detector.means,
            *detector.scales,
            *detector.biases,
            *(value for row in detector.weights for value in row),
        ]
    elif isinstance(detector, TemporalMLPDetector):
        if detector.transform_version != TEMPORAL_TRANSFORM_VERSION:
            raise ValueError("detector temporal transform is incompatible")
        if len(detector.input_weights) != TEMPORAL_FEATURES or any(
            len(row) != HIDDEN_UNITS for row in detector.input_weights
        ):
            raise ValueError("detector input weight shape is incompatible")
        if len(detector.hidden_biases) != HIDDEN_UNITS:
            raise ValueError("detector hidden bias shape is incompatible")
        if len(detector.output_weights) != HIDDEN_UNITS or any(
            len(row) != len(CLASS_NAMES) for row in detector.output_weights
        ):
            raise ValueError("detector output weight shape is incompatible")
        if len(detector.output_biases) != len(CLASS_NAMES):
            raise ValueError("detector output bias shape is incompatible")
        numeric_values = [
            *detector.means,
            *detector.scales,
            *detector.hidden_biases,
            *detector.output_biases,
            *(value for row in detector.input_weights for value in row),
            *(value for row in detector.output_weights for value in row),
        ]
    else:
        raise ValueError("detector type is incompatible")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise ValueError("detector parameters must be finite numbers")
    if any(scale <= 0.0 for scale in detector.scales):
        raise ValueError("detector scales must be positive")


def _validated_contract_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metadata, Mapping) or set(metadata) != _CONTRACT_KEYS:
        raise ValueError("detector contract metadata is malformed")
    result = dict(metadata)
    if any(not isinstance(value, str) for value in result.values()):
        raise ValueError("detector contract metadata values must be strings")
    if result["model_input_version"] != "model_input_v1":
        raise ValueError("detector model-input version is incompatible")
    if not _is_sha256(result["selector_sha256"]) or not _is_sha256(
        result["topology_sha256"]
    ):
        raise ValueError("detector contract hashes are malformed")
    return result


def _require_all_classes(labels: NDArray[np.int64], split: str) -> None:
    if set(int(value) for value in labels) != set(range(len(CLASS_NAMES))):
        raise ValueError(f"detector {split} split must contain every class")


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _cross_entropy(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def _latency_metrics(
    rows: Sequence[Mapping[str, Any]], predictions: NDArray[np.int64]
) -> dict[str, Any]:
    detections: dict[str, list[int]] = {name: [] for name in CLASS_NAMES[1:]}
    missed: dict[str, int] = {name: 0 for name in CLASS_NAMES[1:]}
    grouped: dict[str, list[tuple[Mapping[str, Any], int]]] = {}
    for row, prediction in zip(rows, predictions):
        if row["scenario_role"] != "fault":
            continue
        grouped.setdefault(row["family_id"], []).append((row, int(prediction)))
    for family_rows in grouped.values():
        ordered = sorted(family_rows, key=lambda item: item[0]["end_tick"])
        fault_label = next(
            (
                row["label"]
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
                row["end_tick"] - row["observable_onset_tick"]
                for row, prediction in ordered
                if row["label"] == fault_label and prediction == expected_index
            ),
            None,
        )
        if latency is None:
            missed[fault_label] += 1
        else:
            detections[fault_label].append(latency)
    all_latencies = [latency for values in detections.values() for latency in values]
    return {
        "causal_stride_ticks": 5,
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


def _rolling_latency_metrics(
    labeller: Detector | RuleBaseline,
    family_evidence: Mapping[str, FamilyEvidence],
) -> dict[str, Any]:
    """Measure first correct causal detection over stride-one rolling windows."""
    detections: dict[str, list[int]] = {name: [] for name in CLASS_NAMES[1:]}
    missed: dict[str, int] = {name: 0 for name in CLASS_NAMES[1:]}
    if not family_evidence:
        raise ValueError("rolling latency requires family evidence")
    for family_id in sorted(family_evidence):
        evidence = family_evidence[family_id]
        trace = evidence.fault_model_input_trace
        if trace is None:
            raise ValueError("rolling latency evidence lacks a fault replay")
        reset = getattr(labeller, "reset", None)
        if callable(reset):
            reset()
        detected_latency: int | None = None
        for end_tick in range(WINDOW_TICKS, len(trace) + 1):
            window = [list(vector) for vector in trace[end_tick - WINDOW_TICKS : end_tick]]
            prediction = labeller.label_window(window)
            if (
                end_tick >= evidence.observable_onset_tick
                and prediction == evidence.fault_class
            ):
                detected_latency = end_tick - evidence.observable_onset_tick
                break
        if detected_latency is None:
            missed[evidence.fault_class] += 1
        else:
            detections[evidence.fault_class].append(detected_latency)
    all_latencies = [latency for values in detections.values() for latency in values]
    return {
        "causal_stride_ticks": 1,
        "transition_windows_eligible": True,
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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise ValueError(f"corpus file not found: {source}") from None
    except OSError as exc:
        raise ValueError(f"cannot read corpus file {source}: {exc}") from None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corpus line {line_number} is not valid JSON: {exc}") from None
        if not isinstance(row, dict):
            raise ValueError(f"corpus line {line_number} must be an object")
        rows.append(row)
    return rows


def _write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics["artifact_sizes_bytes"]["metrics_json"] = 0
    while True:
        text = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
        size = len(text.encode("utf-8"))
        if metrics["artifact_sizes_bytes"]["metrics_json"] == size:
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            return
        metrics["artifact_sizes_bytes"]["metrics_json"] = size


def _split_evidence(
    rows: Sequence[Mapping[str, Any]], family_count: int
) -> dict[str, Any]:
    label_counts = {name: 0 for name in (*CLASS_NAMES, EXCLUDED_TRANSITION_LABEL)}
    for row in rows:
        label_counts[row["label"]] += 1
    excluded = label_counts[EXCLUDED_TRANSITION_LABEL]
    return {
        "families": family_count,
        "rows": len(rows),
        "scored_rows": len(rows) - excluded,
        "excluded_transition_rows": excluded,
        "label_counts": label_counts,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def main(argv: Sequence[str]) -> int:
    if argv[:1] == ["train"] and len(argv) == 7:
        try:
            metrics = train_and_export(*argv[1:])
        except (ValueError, OSError, ImportError, json.JSONDecodeError) as exc:
            print(f"cannot train detector: {exc}", file=sys.stderr)
            return 2
        conclusion = metrics["evidence_conclusion"]
        print(
            "trained detector "
            f"macro_f1={metrics['model']['macro_f1']:.3f} "
            f"preferred={conclusion['preferred_method']}"
        )
        return 0
    if argv[:1] == ["predict"] and len(argv) == 3:
        try:
            predictions = predict_scenario(argv[1], argv[2])
        except (ValueError, OSError, ImportError, json.JSONDecodeError) as exc:
            print(f"cannot predict: {exc}", file=sys.stderr)
            return 2
        for prediction in predictions:
            print(json.dumps(prediction, sort_keys=True, allow_nan=False))
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main(sys.argv[1:]))
