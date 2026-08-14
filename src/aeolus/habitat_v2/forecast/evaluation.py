"""Model-neutral, fail-closed evaluation for Forecast D1 candidates."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

INPUT_MANIFEST_SHA256 = "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
TARGET_MANIFEST_SHA256 = "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"
TARGET_COUNT = 51


class EvaluationError(ValueError):
    """Caller-supplied evaluation evidence cannot be safely evaluated."""


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    sample_id: str
    family_cluster_id: str
    input_manifest_sha256: str
    target_manifest_sha256: str
    targets_f32: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidateOutput:
    sample_id: str
    status: str
    input_manifest_sha256: str
    target_manifest_sha256: str
    prediction_f32: np.ndarray | None


class CandidateAdapter(Protocol):
    def __call__(self, request: EvaluationRequest) -> CandidateOutput: ...


@dataclass(frozen=True, slots=True)
class HarmEnvelope:
    """Caller-supplied strict bounds; lower/upper crossings are harmful-positive."""

    lower_f32: np.ndarray
    upper_f32: np.ndarray
    anchor_f32: np.ndarray


@dataclass(frozen=True, slots=True)
class RatioMetric:
    value: float | None
    supported: bool


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    existing_harm_count: int


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Error polarity is lower-is-better; harm classification polarity is positive crossing."""

    polarity: str
    mae_by_horizon_target: np.ndarray
    rmse_by_horizon_target: np.ndarray
    p95_p5_scale: np.ndarray
    scale_supported: np.ndarray
    cluster_macro_normalized_mae: RatioMetric
    precision: RatioMetric
    recall: RatioMetric
    false_positive_rate: RatioMetric
    false_negative_rate: RatioMetric
    coverage: RatioMetric
    selective_error: RatioMetric
    invalid_output_count: int
    confusion: ConfusionCounts


@dataclass(frozen=True, slots=True)
class EvaluatedOutput:
    sample_id: str
    status: str
    reason: str | None
    prediction_f32: np.ndarray | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    outputs: tuple[EvaluatedOutput, ...]
    metrics: EvaluationMetrics


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _ratio(numerator: float, denominator: float) -> RatioMetric:
    return RatioMetric(None, False) if denominator == 0 else RatioMetric(float(numerator / denominator), True)


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise EvaluationError(f"{label} must be finite float32{shape}")
    return array


def _validate_request(request: EvaluationRequest) -> None:
    if type(request) is not EvaluationRequest or not request.sample_id or not request.family_cluster_id:
        raise EvaluationError("expected sample identity is malformed")
    target = np.asarray(request.targets_f32)
    if target.ndim != 2 or target.shape[1:] != (TARGET_COUNT,) or target.shape[0] not in (2, 4, 8):
        raise EvaluationError("expected target shape is unsupported")
    _array(target, target.shape, "expected target")
    if (request.input_manifest_sha256, request.target_manifest_sha256) != (INPUT_MANIFEST_SHA256, TARGET_MANIFEST_SHA256):
        raise EvaluationError("expected manifest identity drift")


def _validate_envelope(envelope: HarmEnvelope, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(envelope) is not HarmEnvelope:
        raise EvaluationError("harm envelope must be supplied explicitly")
    lower, upper = _array(envelope.lower_f32, shape, "lower envelope"), _array(envelope.upper_f32, shape, "upper envelope")
    anchor = _array(envelope.anchor_f32, (TARGET_COUNT,), "anchor")
    if (lower >= upper).any():
        raise EvaluationError("harm envelope must have strict lower/upper bounds")
    return lower, upper, anchor


def _harmful(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (values < lower) | (values > upper)


def evaluate(
    requests: Sequence[EvaluationRequest],
    adapter: CandidateAdapter,
    *,
    envelope: HarmEnvelope,
    training_targets_f32: np.ndarray,
    domain_validator: Callable[[np.ndarray], bool] | None = None,
) -> EvaluationResult:
    """Evaluate an adapter through one model-neutral callable boundary.

    Output identity and declared status are checked before its prediction field
    is read.  Any adapter failure is an INVALID_OUTPUT rather than abstention.
    """
    if isinstance(requests, (str, bytes)) or not callable(adapter) or not requests:
        raise EvaluationError("evaluation requires non-empty requests and one callable adapter")
    items = tuple(requests)
    for request in items:
        _validate_request(request)
    if len({request.sample_id for request in items}) != len(items):
        raise EvaluationError("expected sample IDs must be unique")
    shape = tuple(items[0].targets_f32.shape)
    if any(tuple(request.targets_f32.shape) != shape for request in items):
        raise EvaluationError("evaluation cannot mix target horizons")
    lower, upper, anchor = _validate_envelope(envelope, shape)
    training = np.asarray(training_targets_f32)
    if training.ndim != 3 or training.shape[1:] != shape or training.dtype != np.float32 or not np.isfinite(training).all():
        raise EvaluationError("training targets must be finite float32[N,H,51]")
    scale = np.percentile(training, 95, axis=0) - np.percentile(training, 5, axis=0)
    supported_scale = scale > 0.0
    outputs: list[EvaluatedOutput] = []
    for request in items:
        try:
            candidate = adapter(request)
        except Exception:
            outputs.append(EvaluatedOutput(request.sample_id, "INVALID_OUTPUT", "adapter_exception", None))
            continue
        # Identity and status checks precede *any* prediction access.
        if type(candidate) is not CandidateOutput or (candidate.sample_id, candidate.input_manifest_sha256, candidate.target_manifest_sha256) != (request.sample_id, request.input_manifest_sha256, request.target_manifest_sha256):
            outputs.append(EvaluatedOutput(request.sample_id, "INVALID_OUTPUT", "identity_or_manifest", None))
            continue
        if candidate.status not in {"PREDICTION", "ABSTAIN"}:
            outputs.append(EvaluatedOutput(request.sample_id, "INVALID_OUTPUT", "unsupported_status", None))
            continue
        if candidate.status == "ABSTAIN":
            if candidate.prediction_f32 is not None:
                outputs.append(EvaluatedOutput(request.sample_id, "INVALID_OUTPUT", "abstain_has_prediction", None))
            else:
                outputs.append(EvaluatedOutput(request.sample_id, "ABSTAIN", None, None))
            continue
        try:
            prediction = _array(candidate.prediction_f32, shape, "prediction")
            if domain_validator is not None and not bool(domain_validator(prediction)):
                raise EvaluationError("domain invalid")
        except Exception:
            outputs.append(EvaluatedOutput(request.sample_id, "INVALID_OUTPUT", "prediction_invalid", None))
            continue
        outputs.append(EvaluatedOutput(request.sample_id, "PREDICTION", None, _readonly(np.array(prediction, copy=True))))
    predicted = [(request, output) for request, output in zip(items, outputs, strict=True) if output.status == "PREDICTION"]
    if predicted:
        errors = np.stack([output.prediction_f32 - request.targets_f32 for request, output in predicted])
        mae, rmse = np.mean(np.abs(errors), axis=0), np.sqrt(np.mean(errors**2, axis=0))
        normalized_by_cluster: list[float] = []
        for cluster in sorted({request.family_cluster_id for request, _ in predicted}):
            cluster_errors = np.stack([np.abs(output.prediction_f32 - request.targets_f32) for request, output in predicted if request.family_cluster_id == cluster])
            if supported_scale.any():
                normalized_by_cluster.append(float(np.mean(cluster_errors[:, supported_scale] / scale[supported_scale])))
        normalized = RatioMetric(float(np.mean(normalized_by_cluster)), True) if normalized_by_cluster else RatioMetric(None, False)
        selective = RatioMetric(float(np.mean(np.abs(errors))), True)
    else:
        mae, rmse = np.full(shape, np.nan), np.full(shape, np.nan)
        normalized, selective = RatioMetric(None, False), RatioMetric(None, False)
    tp = fp = tn = fn = existing = 0
    anchor_harm = _harmful(anchor, lower[0], upper[0])
    for request, output in predicted:
        truth_harm, pred_harm = _harmful(request.targets_f32, lower, upper), _harmful(output.prediction_f32, lower, upper)
        valid = np.broadcast_to(~anchor_harm, shape)
        existing += int(np.broadcast_to(anchor_harm, shape).sum())
        tp += int(np.logical_and(pred_harm & truth_harm, valid).sum())
        fp += int(np.logical_and(pred_harm & ~truth_harm, valid).sum())
        tn += int(np.logical_and(~pred_harm & ~truth_harm, valid).sum())
        fn += int(np.logical_and(~pred_harm & truth_harm, valid).sum())
    invalid = sum(output.status == "INVALID_OUTPUT" for output in outputs)
    metrics = EvaluationMetrics(
        "harmful_crossing_positive",
        _readonly(mae.astype(np.float64)), _readonly(rmse.astype(np.float64)), _readonly(scale.astype(np.float64)), _readonly(supported_scale.astype(bool)), normalized,
        _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(fp, fp + tn), _ratio(fn, fn + tp),
        _ratio(len(predicted), len(items)), selective, invalid, ConfusionCounts(tp, fp, tn, fn, existing),
    )
    return EvaluationResult(tuple(outputs), metrics)
