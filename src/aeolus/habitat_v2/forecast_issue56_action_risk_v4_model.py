"""Deterministic development models for the Issue #56 V4 study.

The models predict catalogue-action outcomes only.  They cannot issue
commands, advance the plant, or bypass HMC.  Runtime callers must submit a
selected action through the existing HMC proposal lifecycle.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast.projection import ForecastHistory, project_proposed_action
from .forecast_issue52 import _command_vector
from .forecast_issue56_action_risk_v2 import (
    ACTION_COUNT,
    FEATURE_COUNT,
    alarm_family_slot_indices,
    v2_feature_vector,
)
from .forecast_issue56_action_risk_v3 import V3RiskSample
from .forecast_issue56_action_risk_v4_corpus import (
    V4RelativeActionTargets,
    V4RiskSample,
    V4TrajectoryMetrics,
)
from .forecast_issue55_race import deterministic_family_ids
from .forecast_issue56_action_risk_v4_features import (
    HISTORY_FEATURE_COUNT,
    V4_TEMPORAL_FEATURE_COUNT,
    v4_observable_action_mask,
    v4_temporal_feature_vector,
)


V4_MODEL_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v4_model_v2"
V4_HORIZON_KEYS = (4, 16, 32, 0)
V4_MODEL_CANDIDATES = (
    "c0_v3_refit",
    "c1_shared_hazard_ridge",
    "c2_shared_hazard_temporal",
    "c3_small_shared_mlp",
    "c4_advantage_ranker",
)
V4_ACTION_IDS = (
    "normal-occupied-v1",
    "normal-eva_transition-v1",
    "normal-contingency-v1",
    "normal-dormant-v1",
)
V4_FEATURE_VARIANTS = ("v3_708_past_only", "v4_temporal_past_only")
V4_CANDIDATE_FEATURE_VARIANTS = {
    "c0_v3_refit": "v3_708_past_only",
    "c1_shared_hazard_ridge": "v3_708_past_only",
    "c2_shared_hazard_temporal": "v4_temporal_past_only",
    "c3_small_shared_mlp": "v4_temporal_past_only",
    "c4_advantage_ranker": "v4_temporal_past_only",
}
V4_CANDIDATE_SEMANTICS = {
    "c0_v3_refit": ("ridge", "cumulative_logistic"),
    "c1_shared_hazard_ridge": ("ridge", "shared_hazard"),
    "c2_shared_hazard_temporal": ("ridge", "shared_hazard"),
    "c3_small_shared_mlp": ("small_shared_mlp", "shared_hazard"),
    "c4_advantage_ranker": ("advantage_ranker", "shared_hazard"),
}
V4_EVENT_LIMIT = 0.50
V4_EXPECTED_EXPOSURE_LIMIT = 0.50
V4_MAXIMUM_CROSSING_LIMIT = 0.25
V4_DEFAULT_EVENT_THRESHOLDS = (V4_EVENT_LIMIT,) * 4
V4_RIDGE_ALPHA = 0.1
V4_MODEL_SEEDS = (560057, 560058, 560059)
V4_LOGISTIC_ITERATIONS = 500
V4_CALIBRATION_ITERATIONS = 300
V4_CALIBRATION_RIDGE = 1e-4
V4_MLP_HIDDEN_UNITS = 16
V4_MLP_EPOCHS = 80
V4_MLP_LEARNING_RATE = 0.01
V4_MLP_L2 = 1e-4
V4_CALIBRATION_QUANTILE = 0.90
V4_MIN_VALIDATION_POSITIVES = 2
V4_MIN_VALIDATION_DECISION_COVERAGE = 0.20
V4_THRESHOLD_GRID = (0.2, 0.35, 0.5, 0.65, 0.8)
V4_THRESHOLD_RECALL_TARGET = 0.98
V4_MODEL_PROVENANCE_FIELDS = (
    "source_identity_sha256",
    "corpus_manifest_sha256",
    "hmc_binding_sha256",
    "hmc_contract_sha256",
    "scenario_manifest_sha256",
    "action_catalogue_sha256",
    "feature_manifest_sha256",
    "label_manifest_sha256",
    "model_protocol_sha256",
)
V4_UTILITY_WEIGHTS = {
    "safety_exposure": 1.0,
    "comfort_deviation": 0.25,
    "resource_composite": 0.10,
    "intervention": 0.01,
}


class Issue56V4ModelError(ValueError):
    """Raised when a V4 model, artifact, or prediction is malformed."""


@dataclass(frozen=True, slots=True)
class V4ModelSample:
    """A semantically verified V4 row without retained trace payloads."""

    base_sample: V3RiskSample
    temporal_features_f32: np.ndarray
    observable_action_mask: tuple[bool, ...]
    trajectory_metrics: V4TrajectoryMetrics
    hold_trajectory_metrics: V4TrajectoryMetrics
    relative_action_targets: V4RelativeActionTargets

    def __post_init__(self) -> None:
        if type(self.base_sample) is not V3RiskSample:
            raise Issue56V4ModelError("V4 model sample base record is invalid")
        temporal = np.asarray(self.temporal_features_f32, dtype=np.float32)
        if temporal.shape != (V4_TEMPORAL_FEATURE_COUNT,) or not np.isfinite(temporal).all():
            raise Issue56V4ModelError("V4 model temporal features are malformed")
        temporal = temporal.copy()
        temporal.setflags(write=False)
        object.__setattr__(self, "temporal_features_f32", temporal)
        if (
            type(self.observable_action_mask) is not tuple
            or len(self.observable_action_mask) != len(V4_ACTION_IDS)
            or any(type(value) is not bool for value in self.observable_action_mask)
        ):
            raise Issue56V4ModelError("V4 model observable action mask is malformed")
        if type(self.trajectory_metrics) is not V4TrajectoryMetrics:
            raise Issue56V4ModelError("V4 model action trajectory metrics are invalid")
        if type(self.hold_trajectory_metrics) is not V4TrajectoryMetrics:
            raise Issue56V4ModelError("V4 model hold trajectory metrics are invalid")
        if type(self.relative_action_targets) is not V4RelativeActionTargets:
            raise Issue56V4ModelError("V4 model relative targets are invalid")
        expected_relative = V4RelativeActionTargets(
            self.trajectory_metrics.safety_exposure
            - self.hold_trajectory_metrics.safety_exposure,
            self.trajectory_metrics.comfort_deviation
            - self.hold_trajectory_metrics.comfort_deviation,
            self.trajectory_metrics.resource_composite
            - self.hold_trajectory_metrics.resource_composite,
        )
        if self.relative_action_targets != expected_relative:
            raise Issue56V4ModelError("V4 model relative targets are inconsistent")

    @classmethod
    def from_verified(cls, sample: V4RiskSample) -> "V4ModelSample":
        if type(sample) is not V4RiskSample:
            raise Issue56V4ModelError("V4 model sample requires a verified corpus row")
        return cls(
            sample.base_sample,
            sample.temporal_features_f32,
            sample.observable_action_mask,
            sample.trajectory_metrics,
            sample.hold_trajectory_metrics,
            sample.relative_action_targets,
        )

    @property
    def family_id(self) -> str:
        return self.base_sample.family_id

    @property
    def decision_step(self) -> int:
        return self.base_sample.decision_step

    @property
    def split(self) -> str:
        return self.base_sample.split

    @property
    def action_id(self) -> str:
        return self.base_sample.action_id

    @property
    def features_f32(self) -> np.ndarray:
        return self.base_sample.features_f32

    @property
    def label(self) -> Any:
        return self.base_sample.label


def _sha(value: object) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError) as error:
        raise Issue56V4ModelError("V4 model digest input is not canonical JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Issue56V4ModelError(f"{label} must be lowercase SHA-256")
    return value


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise Issue56V4ModelError(f"V4 {label} is malformed or non-finite")
    result = array.copy()
    result.setflags(write=False)
    return result


def _strict_json_numbers(value: object) -> bool:
    if type(value) in {int, float}:
        return math.isfinite(float(value))
    return type(value) is list and all(_strict_json_numbers(item) for item in value)


def _strict_scalar(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise Issue56V4ModelError(f"V4 {label} must be a finite JSON number")
    return float(value)


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise Issue56V4ModelError(f"V4 {label} must be a boolean")
    return value


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise Issue56V4ModelError(f"V4 {label} must be an integer")
    return value


def _strict_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not _strict_json_numbers(value):
        raise Issue56V4ModelError(f"V4 {label} contains non-numeric JSON values")
    return _finite_array(value, shape, label)


def _strict_int_list(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise Issue56V4ModelError(f"V4 {label} must be a list")
    return tuple(_strict_int(item, label) for item in value)


def _strict_scalar_list(value: object, label: str) -> tuple[float, ...]:
    if type(value) is not list:
        raise Issue56V4ModelError(f"V4 {label} must be a list")
    return tuple(_strict_scalar(item, label) for item in value)


def _provenance_tuple(value: object) -> tuple[tuple[str, str], ...]:
    if value == {}:
        return ()
    if type(value) is not dict or set(value) != set(V4_MODEL_PROVENANCE_FIELDS):
        raise Issue56V4ModelError("V4 model provenance fields drift")
    return tuple(
        (field, _require_sha(value[field], f"V4 {field}"))
        for field in V4_MODEL_PROVENANCE_FIELDS
    )


def _provenance_mapping(value: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return {field: digest for field, digest in value}


def _action_index(action_id: str) -> int:
    action_ids = V4_ACTION_IDS
    try:
        return action_ids.index(action_id)
    except ValueError as error:
        raise Issue56V4ModelError("V4 action identity is not in the frozen catalogue") from error


def _condition_group(family_id: str) -> str:
    roster = deterministic_family_ids(32)
    try:
        index = roster.index(family_id)
    except ValueError as error:
        raise Issue56V4ModelError("V4 family identity is not in the frozen roster") from error
    return f"condition-group-{index // 2:04d}"


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -700.0, 700.0)
    result = np.empty_like(clipped)
    positive = clipped >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    negative = ~positive
    exp_value = np.exp(clipped[negative])
    result[negative] = exp_value / (1.0 + exp_value)
    return float(result) if np.ndim(values) == 0 else result


def _bounded_expm1(value: float) -> float:
    return math.expm1(min(700.0, max(0.0, float(value))))


def _feature_count(feature_variant: str) -> int:
    try:
        return {
            "v3_708_past_only": FEATURE_COUNT,
            "v4_temporal_past_only": V4_TEMPORAL_FEATURE_COUNT,
        }[feature_variant]
    except KeyError as error:
        raise Issue56V4ModelError("V4 feature variant is invalid") from error


def _sample_features(sample: V3RiskSample, feature_variant: str) -> np.ndarray:
    field = (
        "features_f32"
        if feature_variant == "v3_708_past_only"
        else "temporal_features_f32"
    )
    try:
        values = np.asarray(getattr(sample, field), dtype=np.float64)
    except AttributeError as error:
        raise Issue56V4ModelError("V4 sample lacks the requested feature variant") from error
    expected = _feature_count(feature_variant)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise Issue56V4ModelError("V4 sample feature vector is malformed")
    return values


def _require_v4_samples(
    samples: Sequence[V4RiskSample | V4ModelSample],
    split: str,
) -> tuple[V4RiskSample | V4ModelSample, ...]:
    items = tuple(samples)
    if not items or any(type(item) not in {V4RiskSample, V4ModelSample} for item in items):
        raise Issue56V4ModelError("V4 model operation requires semantically verified V4 samples")
    if any(item.split != split for item in items):
        raise Issue56V4ModelError(f"V4 model operation accepts {split} samples only")
    if any(item.action_id not in V4_ACTION_IDS for item in items):
        raise Issue56V4ModelError("V4 sample action is not in the frozen catalogue")
    return items


def _feature_matrix(samples: Sequence[V3RiskSample], feature_variant: str) -> np.ndarray:
    items = tuple(samples)
    if not items:
        raise Issue56V4ModelError("V4 model fitting requires samples")
    return np.stack([_sample_features(item, feature_variant) for item in items]).astype(
        np.float64
    )


def _metric(sample: V3RiskSample, horizon: int) -> Any:
    if horizon == 0:
        return sample.label.remaining_metric
    for metric in sample.label.horizon_metrics:
        if metric.horizon_steps == horizon:
            return metric
    raise Issue56V4ModelError("V4 sample lacks a requested horizon")


def _cumulative_events(samples: Sequence[V3RiskSample]) -> np.ndarray:
    values = np.asarray(
        [
            [_metric(sample, horizon).crossing_event for horizon in V4_HORIZON_KEYS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != len(V4_HORIZON_KEYS):
        raise Issue56V4ModelError("V4 cumulative event targets are malformed")
    if np.any((values < 0.0) | (values > 1.0)) or not np.isfinite(values).all():
        raise Issue56V4ModelError("V4 cumulative event targets are invalid")
    if np.any(np.diff(values, axis=1) < 0.0):
        raise Issue56V4ModelError("V4 cumulative event targets are not monotonic")
    return values


def _interval_events(samples: Sequence[V3RiskSample]) -> tuple[np.ndarray, np.ndarray]:
    cumulative = _cumulative_events(samples)
    intervals = np.empty_like(cumulative)
    intervals[:, 0] = cumulative[:, 0]
    intervals[:, 1:] = np.diff(cumulative, axis=1)
    at_risk = np.ones_like(cumulative, dtype=bool)
    at_risk[:, 1:] = cumulative[:, :-1] < 0.5
    return intervals, at_risk


def _continuous_targets(
    samples: Sequence[V3RiskSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exposure = np.asarray(
        [
            [math.log1p(_metric(sample, horizon).safety_exposure) for horizon in V4_HORIZON_KEYS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    maximum = np.asarray(
        [
            [math.log1p(_metric(sample, horizon).maximum_crossing) for horizon in V4_HORIZON_KEYS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    comfort = np.asarray(
        [math.log1p(sample.trajectory_metrics.comfort_deviation) for sample in samples],
        dtype=np.float64,
    )
    resource = np.asarray(
        [math.log1p(sample.trajectory_metrics.resource_composite) for sample in samples],
        dtype=np.float64,
    )
    relative = np.asarray(
        [
            [
                sample.relative_action_targets.safety_exposure_delta_vs_hold,
                sample.relative_action_targets.comfort_deviation_delta_vs_hold,
                sample.relative_action_targets.resource_composite_delta_vs_hold,
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )
    values = (exposure, maximum, comfort, resource, relative)
    if any(not np.isfinite(value).all() for value in values):
        raise Issue56V4ModelError("V4 regression targets are non-finite")
    return values


def _fit_conditional_ridge(
    normalized_features: np.ndarray,
    targets: np.ndarray,
    event_labels: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(targets, dtype=np.float64)
    events = np.asarray(event_labels, dtype=np.float64)
    if (
        values.ndim != 2
        or events.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(events).all()
        or np.any((events < 0.0) | (events > 1.0))
    ):
        raise Issue56V4ModelError("V4 conditional ridge targets are malformed")
    means: list[float] = []
    scales: list[float] = []
    coefficients: list[np.ndarray] = []
    for index in range(values.shape[1]):
        selected = events[:, index] >= 0.5
        if not np.any(selected):
            raise Issue56V4ModelError("V4 conditional ridge head lacks positive events")
        mean, scale, coefficient = _fit_ridge(
            normalized_features[selected], values[selected, index], alpha
        )
        means.append(float(mean[0]))
        scales.append(float(scale[0]))
        coefficients.append(coefficient[0])
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(scales, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
    )


def _fit_feature_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise Issue56V4ModelError("V4 feature normalization is non-finite")
    return mean, scale


def _normalise(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (features - mean) / scale
    if not np.isfinite(result).all():
        raise Issue56V4ModelError("V4 normalized features are non-finite")
    return result


def _fit_ridge(
    normalized_features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != normalized_features.shape[0] or not np.isfinite(values).all():
        raise Issue56V4ModelError("V4 ridge targets are malformed")
    if not math.isfinite(float(alpha)) or alpha <= 0.0:
        raise Issue56V4ModelError("V4 ridge alpha must be positive and finite")
    target_mean = np.mean(values, axis=0)
    target_scale = np.std(values, axis=0)
    target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
    normalized_targets = (values - target_mean) / target_scale
    gram = normalized_features @ normalized_features.T
    gram += np.eye(len(normalized_features), dtype=np.float64) * float(alpha)
    try:
        dual = np.linalg.solve(gram, normalized_targets)
    except np.linalg.LinAlgError as error:
        raise Issue56V4ModelError("V4 ridge fit is singular") from error
    coefficients = (dual.T @ normalized_features).astype(np.float64)
    if not np.isfinite(coefficients).all():
        raise Issue56V4ModelError("V4 ridge coefficients are non-finite")
    return target_mean, target_scale, coefficients


def _fit_logistic(
    normalized_features: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(normalized_features):
        raise Issue56V4ModelError("V4 logistic labels are malformed")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise Issue56V4ModelError("V4 logistic labels are invalid")
    positive = float(np.sum(values >= 0.5))
    negative = float(len(values) - positive)
    if positive == 0.0 or negative == 0.0:
        probability = (positive + 0.5) / (len(values) + 1.0)
        return float(math.log(probability / (1.0 - probability))), np.zeros(
            normalized_features.shape[1], dtype=np.float64
        )
    if not math.isfinite(float(alpha)) or alpha <= 0.0:
        raise Issue56V4ModelError("V4 logistic alpha must be positive and finite")
    weights = np.where(
        values >= 0.5,
        len(values) / (2.0 * positive),
        len(values) / (2.0 * negative),
    )
    intercept = float(math.log(positive / negative))
    coefficients = np.zeros(normalized_features.shape[1], dtype=np.float64)
    row_norm = max(float(np.max(np.sum(normalized_features**2, axis=1))), 1.0)
    learning_rate = 1.0 / (0.25 * row_norm + float(alpha) + 1e-8)
    for _ in range(V4_LOGISTIC_ITERATIONS):
        logits = intercept + normalized_features @ coefficients
        probabilities = np.asarray(_sigmoid(logits), dtype=np.float64)
        residual = weights * (probabilities - values)
        gradient_intercept = float(np.mean(residual))
        gradient_coefficients = normalized_features.T @ residual / len(values)
        gradient_coefficients += float(alpha) * coefficients
        next_intercept = intercept - learning_rate * gradient_intercept
        next_coefficients = coefficients - learning_rate * gradient_coefficients
        if not math.isfinite(next_intercept) or not np.isfinite(next_coefficients).all():
            raise Issue56V4ModelError("V4 logistic fit became non-finite")
        change = max(
            abs(next_intercept - intercept),
            float(np.max(np.abs(next_coefficients - coefficients))),
        )
        intercept, coefficients = next_intercept, next_coefficients
        if change <= 1e-8:
            break
    return intercept, coefficients


def _fit_logistic_heads(
    normalized_features: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    *,
    at_risk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(normalized_features):
        raise Issue56V4ModelError("V4 logistic head labels are malformed")
    intercepts: list[float] = []
    coefficients: list[np.ndarray] = []
    for index in range(values.shape[1]):
        selected = (
            np.ones(len(values), dtype=bool)
            if at_risk is None
            else np.asarray(at_risk[:, index], dtype=bool)
        )
        if not np.any(selected):
            raise Issue56V4ModelError("V4 hazard head has no at-risk rows")
        intercept, coefficient = _fit_logistic(
            normalized_features[selected],
            values[selected, index],
            alpha,
        )
        intercepts.append(intercept)
        coefficients.append(coefficient)
    return np.asarray(intercepts), np.asarray(coefficients)


def _fit_calibration_logistic(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.shape != targets.shape or not np.isfinite(values).all():
        raise Issue56V4ModelError("V4 calibration inputs are malformed")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise Issue56V4ModelError("V4 calibration labels are invalid")
    positive = int(np.sum(targets >= 0.5))
    if positive < V4_MIN_VALIDATION_POSITIVES or not np.any(targets < 0.5):
        raise Issue56V4ModelError(
            "V4 calibration requires minimum positive and negative support"
        )
    center = float(np.mean(values))
    scale = max(float(np.std(values)), 1.0)
    normalized = (values - center) / scale
    positive_rate = float(np.mean(targets))
    intercept = float(math.log(positive_rate / (1.0 - positive_rate)))
    slope = 1.0
    for _ in range(V4_CALIBRATION_ITERATIONS):
        calibrated_logits = intercept + slope * normalized
        probabilities = np.asarray(_sigmoid(calibrated_logits), dtype=np.float64)
        curvature = probabilities * (1.0 - probabilities)
        residual = probabilities - targets
        gradient = np.asarray(
            [
                np.mean(residual),
                np.mean(residual * normalized) + V4_CALIBRATION_RIDGE * slope,
            ],
            dtype=np.float64,
        )
        hessian = np.asarray(
            [
                [np.mean(curvature), np.mean(curvature * normalized)],
                [
                    np.mean(curvature * normalized),
                    np.mean(curvature * normalized * normalized) + V4_CALIBRATION_RIDGE,
                ],
            ],
            dtype=np.float64,
        )
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise Issue56V4ModelError("V4 calibration fit is singular") from error
        next_intercept = intercept - float(update[0])
        next_slope = max(0.0, slope - float(update[1]))
        if not np.isfinite([next_intercept, next_slope]).all():
            raise Issue56V4ModelError("V4 calibration fit became non-finite")
        if max(abs(next_intercept - intercept), abs(next_slope - slope)) <= 1e-8:
            intercept, slope = next_intercept, next_slope
            break
        intercept, slope = next_intercept, next_slope
    return float(intercept - slope * center / scale), float(slope / scale)


def _quantile_residual(values: np.ndarray, targets: np.ndarray) -> float:
    residual = np.abs(np.asarray(values, dtype=np.float64) - np.asarray(targets, dtype=np.float64))
    if residual.ndim != 1 or not len(residual) or not np.isfinite(residual).all():
        raise Issue56V4ModelError("V4 calibration residuals are malformed")
    return float(np.quantile(residual, V4_CALIBRATION_QUANTILE))


def _fit_pairwise_advantage(
    normalized_features: np.ndarray,
    samples: Sequence[V3RiskSample],
    alpha: float,
) -> tuple[float, np.ndarray]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(sample.family_id, sample.decision_step)].append(index)
    pair_features: list[np.ndarray] = []
    pair_labels: list[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        utilities = np.asarray(
            [
                samples[index].relative_action_targets.safety_exposure_delta_vs_hold
                + V4_UTILITY_WEIGHTS["comfort_deviation"]
                * samples[index].relative_action_targets.comfort_deviation_delta_vs_hold
                + V4_UTILITY_WEIGHTS["resource_composite"]
                * samples[index].relative_action_targets.resource_composite_delta_vs_hold
                + V4_UTILITY_WEIGHTS["intervention"]
                * float(
                    np.mean(
                        np.abs(
                            np.asarray(
                                samples[index].features_f32[
                                    HISTORY_FEATURE_COUNT * 3 : HISTORY_FEATURE_COUNT * 3
                                    + ACTION_COUNT
                                ],
                                dtype=np.float64,
                            )
                            - np.asarray(
                                samples[index].features_f32[
                                    HISTORY_FEATURE_COUNT - ACTION_COUNT : HISTORY_FEATURE_COUNT
                                ],
                                dtype=np.float64,
                            )
                        )
                    )
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                if utilities[left] == utilities[right]:
                    continue
                left_is_better = utilities[left] < utilities[right]
                difference = normalized_features[indices[left]] - normalized_features[indices[right]]
                pair_features.append(difference if left_is_better else -difference)
                pair_labels.append(1.0)
                pair_features.append(-pair_features[-1])
                pair_labels.append(0.0)
    if len(pair_features) < 2:
        raise Issue56V4ModelError("V4 advantage ranker requires non-tied action pairs")
    return _fit_logistic(
        np.asarray(pair_features, dtype=np.float64),
        np.asarray(pair_labels, dtype=np.float64),
        alpha,
    )


def _mlp_targets(
    samples: Sequence[V3RiskSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intervals, at_risk = _interval_events(samples)
    exposure, maximum, comfort, resource, relative = _continuous_targets(samples)
    cumulative = _cumulative_events(samples)
    raw = np.column_stack((intervals, exposure, maximum, comfort, resource, relative))
    continuous = raw[:, 4:]
    target_mask = np.ones_like(raw, dtype=bool)
    target_mask[:, 4:8] = cumulative >= 0.5
    target_mask[:, 8:12] = cumulative >= 0.5
    means = np.empty(continuous.shape[1], dtype=np.float64)
    scales = np.empty(continuous.shape[1], dtype=np.float64)
    for index in range(continuous.shape[1]):
        selected = target_mask[:, index + 4]
        if not np.any(selected):
            raise Issue56V4ModelError("V4 MLP conditional head lacks positive events")
        means[index] = np.mean(continuous[selected, index])
        scales[index] = np.std(continuous[selected, index])
    scales = np.where(scales > 1e-8, scales, 1.0)
    targets = raw.copy()
    targets[:, 4:] = (continuous - means) / scales
    return targets, at_risk, target_mask, means, scales


def _fit_mlp(
    normalized_features: np.ndarray,
    targets: np.ndarray,
    at_risk: np.ndarray,
    target_mask: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        targets.ndim != 2
        or targets.shape[0] != len(normalized_features)
        or targets.shape[1] != 17
        or at_risk.shape != (len(normalized_features), 4)
        or target_mask.shape != targets.shape
        or not np.isfinite(targets).all()
        or not np.isfinite(at_risk).all()
        or not np.isfinite(target_mask).all()
    ):
        raise Issue56V4ModelError("V4 MLP targets or masks are malformed")
    rng = np.random.default_rng(seed)
    hidden = V4_MLP_HIDDEN_UNITS
    input_weights = rng.normal(
        0.0,
        math.sqrt(2.0 / (normalized_features.shape[1] + hidden)),
        (normalized_features.shape[1], hidden),
    )
    hidden_biases = np.zeros(hidden, dtype=np.float64)
    output_weights = rng.normal(0.0, 0.02, (hidden, targets.shape[1]))
    output_biases = np.zeros(targets.shape[1], dtype=np.float64)
    first = [np.zeros_like(value) for value in (input_weights, hidden_biases, output_weights, output_biases)]
    second = [np.zeros_like(value) for value in (input_weights, hidden_biases, output_weights, output_biases)]
    beta1, beta2 = 0.9, 0.999
    for iteration in range(1, V4_MLP_EPOCHS + 1):
        hidden_pre = normalized_features @ input_weights + hidden_biases
        hidden_values = np.maximum(hidden_pre, 0.0)
        outputs = hidden_values @ output_weights + output_biases
        probabilities = np.asarray(_sigmoid(outputs[:, :4]), dtype=np.float64)
        gradient_outputs = np.zeros_like(outputs)
        gradient_outputs[:, :4] = (probabilities - targets[:, :4]) * at_risk
        gradient_outputs[:, 4:] = (
            0.25 * (outputs[:, 4:] - targets[:, 4:]) * target_mask[:, 4:]
        )
        gradient_outputs /= len(normalized_features)
        gradients = (
            normalized_features.T @ ((gradient_outputs @ output_weights.T) * (hidden_pre > 0.0))
            + V4_MLP_L2 * input_weights,
            np.sum((gradient_outputs @ output_weights.T) * (hidden_pre > 0.0), axis=0),
            hidden_values.T @ gradient_outputs + V4_MLP_L2 * output_weights,
            np.sum(gradient_outputs, axis=0),
        )
        values = (input_weights, hidden_biases, output_weights, output_biases)
        for index, (value, gradient) in enumerate(zip(values, gradients, strict=True)):
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
            second[index] = beta2 * second[index] + (1.0 - beta2) * gradient * gradient
            corrected_first = first[index] / (1.0 - beta1**iteration)
            corrected_second = second[index] / (1.0 - beta2**iteration)
            value -= V4_MLP_LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if not np.isfinite(input_weights).all() or not np.isfinite(output_weights).all():
            raise Issue56V4ModelError("V4 MLP fit became non-finite")
    return input_weights, hidden_biases, output_weights, output_biases


def _select_event_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, ...]:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(V4_HORIZON_KEYS)
        or targets.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
        or np.any((values < 0.0) | (values > 1.0))
        or np.any((targets < 0.0) | (targets > 1.0))
    ):
        raise Issue56V4ModelError("V4 event threshold inputs are malformed")
    thresholds: list[float] = []
    for index in range(values.shape[1]):
        positive = targets[:, index] >= 0.5
        positive_count = int(np.sum(positive))
        if positive_count < V4_MIN_VALIDATION_POSITIVES:
            raise Issue56V4ModelError("V4 threshold selection lacks positive validation support")
        selected: float | None = None
        for threshold in V4_THRESHOLD_GRID:
            predicted = values[:, index] >= threshold
            recall = float(np.sum(predicted & positive) / positive_count)
            if recall >= V4_THRESHOLD_RECALL_TARGET:
                selected = threshold
        if selected is None:
            raise Issue56V4ModelError(
                "V4 threshold selection cannot meet the validation recall target"
            )
        thresholds.append(float(selected))
    return tuple(thresholds)


@dataclass(frozen=True, slots=True)
class V4HorizonPrediction:
    """One calibrated prediction at a fixed or remaining horizon."""

    horizon_steps: int
    event_probability: float
    upper_event_probability: float
    conditional_exposure: float
    upper_conditional_exposure: float
    conditional_maximum_crossing: float
    upper_maximum_crossing: float
    upper_expected_exposure: float
    upper_expected_maximum_crossing: float

    def __post_init__(self) -> None:
        if self.horizon_steps not in V4_HORIZON_KEYS:
            raise Issue56V4ModelError("V4 prediction horizon is invalid")
        for value, label in (
            (self.event_probability, "event probability"),
            (self.upper_event_probability, "upper event probability"),
            (self.conditional_exposure, "conditional exposure"),
            (self.upper_conditional_exposure, "upper exposure"),
            (self.conditional_maximum_crossing, "maximum crossing"),
            (self.upper_maximum_crossing, "upper maximum crossing"),
            (self.upper_expected_exposure, "expected exposure"),
            (self.upper_expected_maximum_crossing, "expected maximum crossing"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V4ModelError(f"V4 {label} is invalid")
        if not 0.0 <= self.event_probability <= 1.0 or not 0.0 <= self.upper_event_probability <= 1.0:
            raise Issue56V4ModelError("V4 event probability is outside [0, 1]")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "horizon_steps": self.horizon_steps,
            "event_probability": self.event_probability,
            "upper_event_probability": self.upper_event_probability,
            "conditional_exposure": self.conditional_exposure,
            "upper_conditional_exposure": self.upper_conditional_exposure,
            "conditional_maximum_crossing": self.conditional_maximum_crossing,
            "upper_maximum_crossing": self.upper_maximum_crossing,
            "upper_expected_exposure": self.upper_expected_exposure,
            "upper_expected_maximum_crossing": self.upper_expected_maximum_crossing,
        }


@dataclass(frozen=True, slots=True)
class V4RiskPrediction:
    """Complete model output for one catalogue action."""

    horizons: tuple[V4HorizonPrediction, ...]
    comfort_deviation: float
    upper_comfort_deviation: float
    resource_composite: float
    upper_resource_composite: float
    relative_safety_exposure: float
    relative_comfort_deviation: float
    relative_resource_composite: float
    upper_relative_safety_exposure: float
    upper_relative_comfort_deviation: float
    upper_relative_resource_composite: float
    advantage_score: float
    hard_ineligible: bool
    reason: str | None

    def __post_init__(self) -> None:
        if tuple(item.horizon_steps for item in self.horizons) != V4_HORIZON_KEYS:
            raise Issue56V4ModelError("V4 prediction horizons drifted")
        for value, label in (
            (self.comfort_deviation, "comfort deviation"),
            (self.upper_comfort_deviation, "upper comfort deviation"),
            (self.resource_composite, "resource composite"),
            (self.upper_resource_composite, "upper resource composite"),
            (self.advantage_score, "advantage score"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0 and label != "advantage score":
                raise Issue56V4ModelError(f"V4 {label} is invalid")
        for value, label in (
            (self.relative_safety_exposure, "relative safety exposure"),
            (self.relative_comfort_deviation, "relative comfort deviation"),
            (self.relative_resource_composite, "relative resource composite"),
            (self.upper_relative_safety_exposure, "upper relative safety exposure"),
            (self.upper_relative_comfort_deviation, "upper relative comfort deviation"),
            (self.upper_relative_resource_composite, "upper relative resource composite"),
        ):
            if not math.isfinite(float(value)):
                raise Issue56V4ModelError(f"V4 {label} is invalid")

    def at(self, horizon: int) -> V4HorizonPrediction:
        for prediction in self.horizons:
            if prediction.horizon_steps == horizon:
                return prediction
        raise Issue56V4ModelError("V4 prediction horizon is unavailable")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "horizons": [item.to_mapping() for item in self.horizons],
            "comfort_deviation": self.comfort_deviation,
            "upper_comfort_deviation": self.upper_comfort_deviation,
            "resource_composite": self.resource_composite,
            "upper_resource_composite": self.upper_resource_composite,
            "relative_safety_exposure": self.relative_safety_exposure,
            "relative_comfort_deviation": self.relative_comfort_deviation,
            "relative_resource_composite": self.relative_resource_composite,
            "upper_relative_safety_exposure": self.upper_relative_safety_exposure,
            "upper_relative_comfort_deviation": self.upper_relative_comfort_deviation,
            "upper_relative_resource_composite": self.upper_relative_resource_composite,
            "advantage_score": self.advantage_score,
            "hard_ineligible": self.hard_ineligible,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class V4ActionScore:
    """Runtime score for one action after safety and compatibility screening."""

    action_id: str
    action_index: int
    compatible: bool
    hard_ineligible: bool
    utility_score: float
    intervention: float
    prediction: V4RiskPrediction
    reason: str | None

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise Issue56V4ModelError("V4 action score identity is invalid")
        if (
            isinstance(self.action_index, bool)
            or type(self.action_index) is not int
            or not 0 <= self.action_index < len(V4_ACTION_IDS)
            or not math.isfinite(float(self.intervention))
            or self.intervention < 0.0
            or math.isnan(float(self.utility_score))
        ):
            raise Issue56V4ModelError("V4 action score is invalid")
        if V4_ACTION_IDS[self.action_index] != self.action_id:
            raise Issue56V4ModelError("V4 action score identity does not match its index")


@dataclass(frozen=True, slots=True)
class V4RiskModel:
    """A deterministic, calibrated, advisory-only V4 model artifact."""

    candidate_id: str
    feature_variant: str
    model_kind: str
    hazard_mode: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    event_intercepts: np.ndarray
    event_coefficients: np.ndarray
    hazard_intercepts: np.ndarray
    hazard_coefficients: np.ndarray
    exposure_target_means: np.ndarray
    exposure_target_scales: np.ndarray
    exposure_coefficients: np.ndarray
    maximum_target_means: np.ndarray
    maximum_target_scales: np.ndarray
    maximum_coefficients: np.ndarray
    comfort_target_mean: float
    comfort_target_scale: float
    comfort_coefficients: np.ndarray
    resource_target_mean: float
    resource_target_scale: float
    resource_coefficients: np.ndarray
    relative_target_means: np.ndarray
    relative_target_scales: np.ndarray
    relative_coefficients: np.ndarray
    advantage_intercept: float
    advantage_coefficients: np.ndarray
    calibration_intercepts: np.ndarray
    calibration_slopes: np.ndarray
    exposure_residual_p90: np.ndarray
    maximum_residual_p90: np.ndarray
    comfort_residual_p90: float
    resource_residual_p90: float
    relative_residual_p90: np.ndarray
    alpha: float
    seed: int
    actuator_authority: bool = False
    hidden_weights: np.ndarray | None = None
    hidden_biases: np.ndarray | None = None
    output_weights: np.ndarray | None = None
    output_biases: np.ndarray | None = None
    calibration_support: tuple[int, ...] = ()
    validation_decision_coverage: float = 0.0
    event_thresholds: tuple[float, ...] = V4_DEFAULT_EVENT_THRESHOLDS
    model_sha256: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        feature_count = _feature_count(self.feature_variant)
        if self.candidate_id not in V4_MODEL_CANDIDATES:
            raise Issue56V4ModelError("V4 candidate identity is invalid")
        if V4_CANDIDATE_FEATURE_VARIANTS[self.candidate_id] != self.feature_variant:
            raise Issue56V4ModelError("V4 candidate feature variant is invalid")
        expected_kind, expected_hazard = V4_CANDIDATE_SEMANTICS[self.candidate_id]
        if self.model_kind != expected_kind:
            raise Issue56V4ModelError("V4 model kind is invalid for the candidate")
        if self.hazard_mode != expected_hazard:
            raise Issue56V4ModelError("V4 hazard mode is invalid for the candidate")
        arrays = (
            ("feature_mean", self.feature_mean, (feature_count,)),
            ("feature_scale", self.feature_scale, (feature_count,)),
            ("event_intercepts", self.event_intercepts, (4,)),
            ("event_coefficients", self.event_coefficients, (4, feature_count)),
            ("hazard_intercepts", self.hazard_intercepts, (4,)),
            ("hazard_coefficients", self.hazard_coefficients, (4, feature_count)),
            ("exposure_target_means", self.exposure_target_means, (4,)),
            ("exposure_target_scales", self.exposure_target_scales, (4,)),
            ("exposure_coefficients", self.exposure_coefficients, (4, feature_count)),
            ("maximum_target_means", self.maximum_target_means, (4,)),
            ("maximum_target_scales", self.maximum_target_scales, (4,)),
            ("maximum_coefficients", self.maximum_coefficients, (4, feature_count)),
            ("comfort_coefficients", self.comfort_coefficients, (feature_count,)),
            ("resource_coefficients", self.resource_coefficients, (feature_count,)),
            ("relative_target_means", self.relative_target_means, (3,)),
            ("relative_target_scales", self.relative_target_scales, (3,)),
            ("relative_coefficients", self.relative_coefficients, (3, feature_count)),
            ("advantage_coefficients", self.advantage_coefficients, (feature_count,)),
            ("calibration_intercepts", self.calibration_intercepts, (4,)),
            ("calibration_slopes", self.calibration_slopes, (4,)),
            ("exposure_residual_p90", self.exposure_residual_p90, (4,)),
            ("maximum_residual_p90", self.maximum_residual_p90, (4,)),
            ("relative_residual_p90", self.relative_residual_p90, (3,)),
        )
        for label, value, shape in arrays:
            object.__setattr__(self, label, _finite_array(value, shape, label))
        for label, value in (
            ("feature_scale", self.feature_scale),
            ("exposure_target_scales", self.exposure_target_scales),
            ("maximum_target_scales", self.maximum_target_scales),
            ("relative_target_scales", self.relative_target_scales),
        ):
            if np.any(value <= 0.0):
                raise Issue56V4ModelError(f"V4 {label} must be positive")
        for label, value in (
            ("comfort_target_scale", self.comfort_target_scale),
            ("resource_target_scale", self.resource_target_scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise Issue56V4ModelError(f"V4 {label} must be positive")
        if np.any(self.calibration_slopes < 0.0):
            raise Issue56V4ModelError("V4 calibration must be monotonic")
        if (
            np.any(self.exposure_residual_p90 < 0.0)
            or np.any(self.maximum_residual_p90 < 0.0)
            or self.comfort_residual_p90 < 0.0
            or self.resource_residual_p90 < 0.0
            or np.any(self.relative_residual_p90 < 0.0)
        ):
            raise Issue56V4ModelError("V4 residual bounds must be non-negative")
        if not math.isfinite(float(self.advantage_intercept)) or not math.isfinite(float(self.alpha)) or self.alpha <= 0.0:
            raise Issue56V4ModelError("V4 advantage or regularization parameter is invalid")
        if self.actuator_authority is not False:
            raise Issue56V4ModelError("V4 model cannot claim actuator authority")
        if self.model_kind == "small_shared_mlp":
            if any(
                value is None
                for value in (
                    self.hidden_weights,
                    self.hidden_biases,
                    self.output_weights,
                    self.output_biases,
                )
            ):
                raise Issue56V4ModelError("V4 MLP weights are missing")
            object.__setattr__(
                self,
                "hidden_weights",
                _finite_array(self.hidden_weights, (feature_count, V4_MLP_HIDDEN_UNITS), "hidden weights"),
            )
            object.__setattr__(
                self,
                "hidden_biases",
                _finite_array(self.hidden_biases, (V4_MLP_HIDDEN_UNITS,), "hidden biases"),
            )
            object.__setattr__(
                self,
                "output_weights",
                _finite_array(self.output_weights, (V4_MLP_HIDDEN_UNITS, 17), "output weights"),
            )
            object.__setattr__(
                self,
                "output_biases",
                _finite_array(self.output_biases, (17,), "output biases"),
            )
        if self.calibration_support and len(self.calibration_support) != 4:
            raise Issue56V4ModelError("V4 calibration support shape is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.calibration_support
        ):
            raise Issue56V4ModelError("V4 calibration support is invalid")
        if (
            not math.isfinite(float(self.validation_decision_coverage))
            or not 0.0 <= self.validation_decision_coverage <= 1.0
        ):
            raise Issue56V4ModelError("V4 validation decision coverage is invalid")
        if (
            type(self.event_thresholds) is not tuple
            or len(self.event_thresholds) != len(V4_HORIZON_KEYS)
            or any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.event_thresholds
            )
        ):
            raise Issue56V4ModelError("V4 event thresholds are invalid")
        if self.model_kind != "small_shared_mlp" and any(
            value is not None
            for value in (
                self.hidden_weights,
                self.hidden_biases,
                self.output_weights,
                self.output_biases,
            )
        ):
            raise Issue56V4ModelError("V4 non-MLP artifact contains MLP weights")
        if self.model_sha256 is not None:
            _require_sha(self.model_sha256, "V4 model artifact")
        if type(self.provenance) is not tuple:
            raise Issue56V4ModelError("V4 model provenance is malformed")
        if self.provenance and len(self.provenance) != len(V4_MODEL_PROVENANCE_FIELDS):
            raise Issue56V4ModelError("V4 model provenance is incomplete")
        if self.provenance != _provenance_tuple(_provenance_mapping(self.provenance)):
            raise Issue56V4ModelError("V4 model provenance is not canonical")

    @classmethod
    def fit(
        cls,
        samples: Sequence[V4RiskSample | V4ModelSample],
        *,
        candidate_id: str,
        feature_variant: str | None = None,
        alpha: float = V4_RIDGE_ALPHA,
        seed: int = V4_MODEL_SEEDS[0],
        provenance: Mapping[str, str] | None = None,
    ) -> "V4RiskModel":
        items = _require_v4_samples(samples, "TRAIN")
        if candidate_id not in V4_MODEL_CANDIDATES:
            raise Issue56V4ModelError("V4 candidate identity is invalid")
        expected_variant = V4_CANDIDATE_FEATURE_VARIANTS[candidate_id]
        if feature_variant is None:
            feature_variant = expected_variant
        if feature_variant != expected_variant:
            raise Issue56V4ModelError("V4 candidate feature variant is invalid")
        if len({item.family_id for item in items}) < 2:
            raise Issue56V4ModelError("V4 fitting requires at least two families")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise Issue56V4ModelError("V4 model seed is invalid")
        features = _feature_matrix(items, feature_variant)
        feature_mean, feature_scale = _fit_feature_normalization(features)
        normalized_features = _normalise(features, feature_mean, feature_scale)
        intervals, at_risk = _interval_events(items)
        cumulative = _cumulative_events(items)
        exposure, maximum, comfort, resource, relative = _continuous_targets(items)
        hazard_mode = "cumulative_logistic" if candidate_id == "c0_v3_refit" else "shared_hazard"
        model_kind = "advantage_ranker" if candidate_id == "c4_advantage_ranker" else "ridge"
        if candidate_id == "c3_small_shared_mlp":
            model_kind = "small_shared_mlp"
            (
                mlp_targets,
                mlp_at_risk,
                mlp_target_mask,
                mlp_means,
                mlp_scales,
            ) = _mlp_targets(items)
            hidden_weights, hidden_biases, output_weights, output_biases = _fit_mlp(
                normalized_features,
                mlp_targets,
                mlp_at_risk,
                mlp_target_mask,
                seed=seed,
            )
            exposure_means, exposure_scales = mlp_means[:4], mlp_scales[:4]
            maximum_means, maximum_scales = mlp_means[4:8], mlp_scales[4:8]
            comfort_mean, comfort_scale = float(mlp_means[8]), float(mlp_scales[8])
            resource_mean, resource_scale = float(mlp_means[9]), float(mlp_scales[9])
            relative_means, relative_scales = mlp_means[10:13], mlp_scales[10:13]
            event_intercepts = np.zeros(4, dtype=np.float64)
            event_coefficients = np.zeros((4, features.shape[1]), dtype=np.float64)
            hazard_intercepts = np.zeros(4, dtype=np.float64)
            hazard_coefficients = np.zeros((4, features.shape[1]), dtype=np.float64)
            exposure_coefficients = np.zeros((4, features.shape[1]), dtype=np.float64)
            maximum_coefficients = np.zeros((4, features.shape[1]), dtype=np.float64)
            comfort_coefficients = np.zeros(features.shape[1], dtype=np.float64)
            resource_coefficients = np.zeros(features.shape[1], dtype=np.float64)
            relative_coefficients = np.zeros((3, features.shape[1]), dtype=np.float64)
        else:
            if hazard_mode == "shared_hazard":
                hazard_intercepts, hazard_coefficients = _fit_logistic_heads(
                    normalized_features,
                    intervals,
                    alpha,
                    at_risk=at_risk,
                )
                event_intercepts = np.zeros(4, dtype=np.float64)
                event_coefficients = np.zeros((4, features.shape[1]), dtype=np.float64)
            else:
                event_intercepts, event_coefficients = _fit_logistic_heads(
                    normalized_features,
                    cumulative,
                    alpha,
                )
                hazard_intercepts = np.zeros(4, dtype=np.float64)
                hazard_coefficients = np.zeros((4, features.shape[1]), dtype=np.float64)
            exposure_means, exposure_scales, exposure_coefficients = _fit_conditional_ridge(
                normalized_features, exposure, cumulative, alpha
            )
            maximum_means, maximum_scales, maximum_coefficients = _fit_conditional_ridge(
                normalized_features, maximum, cumulative, alpha
            )
            comfort_fit = _fit_ridge(normalized_features, comfort, alpha)
            resource_fit = _fit_ridge(normalized_features, resource, alpha)
            relative_means, relative_scales, relative_coefficients = _fit_ridge(
                normalized_features, relative, alpha
            )
            comfort_mean, comfort_scale, comfort_coefficients = (
                float(comfort_fit[0][0]),
                float(comfort_fit[1][0]),
                comfort_fit[2][0],
            )
            resource_mean, resource_scale, resource_coefficients = (
                float(resource_fit[0][0]),
                float(resource_fit[1][0]),
                resource_fit[2][0],
            )
            hidden_weights = hidden_biases = output_weights = output_biases = None
        advantage_intercept = 0.0
        advantage_coefficients = np.zeros(features.shape[1], dtype=np.float64)
        if candidate_id == "c4_advantage_ranker":
            advantage_intercept, advantage_coefficients = _fit_pairwise_advantage(
                normalized_features,
                items,
                alpha,
            )
        return cls(
            candidate_id,
            feature_variant,
            model_kind,
            hazard_mode,
            feature_mean,
            feature_scale,
            event_intercepts,
            event_coefficients,
            hazard_intercepts,
            hazard_coefficients,
            exposure_means,
            exposure_scales,
            exposure_coefficients,
            maximum_means,
            maximum_scales,
            maximum_coefficients,
            comfort_mean,
            comfort_scale,
            comfort_coefficients,
            resource_mean,
            resource_scale,
            resource_coefficients,
            relative_means,
            relative_scales,
            relative_coefficients,
            advantage_intercept,
            advantage_coefficients,
            np.zeros(4, dtype=np.float64),
            np.ones(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            0.0,
            0.0,
            np.zeros(3, dtype=np.float64),
            alpha,
            seed,
            False,
            hidden_weights,
            hidden_biases,
            output_weights,
            output_biases,
            provenance=_provenance_tuple({} if provenance is None else provenance),
        )

    def _linear_values(self, features: np.ndarray) -> tuple[np.ndarray, ...]:
        normalized = _normalise(features, self.feature_mean, self.feature_scale)
        if self.model_kind == "small_shared_mlp":
            if self.hidden_weights is None or self.hidden_biases is None or self.output_weights is None or self.output_biases is None:
                raise Issue56V4ModelError("V4 MLP artifact is incomplete")
            hidden = np.maximum(0.0, normalized @ self.hidden_weights + self.hidden_biases)
            outputs = hidden @ self.output_weights + self.output_biases
            event = outputs[:4]
            exposure = outputs[4:8] * self.exposure_target_scales + self.exposure_target_means
            maximum = outputs[8:12] * self.maximum_target_scales + self.maximum_target_means
            comfort = np.asarray(
                [outputs[12] * self.comfort_target_scale + self.comfort_target_mean],
                dtype=np.float64,
            )
            resource = np.asarray(
                [outputs[13] * self.resource_target_scale + self.resource_target_mean],
                dtype=np.float64,
            )
            relative = outputs[14:17] * self.relative_target_scales + self.relative_target_means
        else:
            if self.hazard_mode == "shared_hazard":
                event = self.hazard_intercepts + normalized @ self.hazard_coefficients.T
            else:
                event = self.event_intercepts + normalized @ self.event_coefficients.T
            exposure = self.exposure_target_means + (
                normalized @ self.exposure_coefficients.T
            ) * self.exposure_target_scales
            maximum = self.maximum_target_means + (
                normalized @ self.maximum_coefficients.T
            ) * self.maximum_target_scales
            comfort = np.asarray(
                [
                    self.comfort_target_mean
                    + float(normalized @ self.comfort_coefficients)
                    * self.comfort_target_scale
                ],
                dtype=np.float64,
            )
            resource = np.asarray(
                [
                    self.resource_target_mean
                    + float(normalized @ self.resource_coefficients)
                    * self.resource_target_scale
                ],
                dtype=np.float64,
            )
            relative = self.relative_target_means + (
                normalized @ self.relative_coefficients.T
            ) * self.relative_target_scales
        advantage = float(self.advantage_intercept + normalized @ self.advantage_coefficients)
        return event, exposure, maximum, comfort, resource, relative, advantage

    def _event_probabilities(self, raw_event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calibrated = np.asarray(
            _sigmoid(self.calibration_intercepts + self.calibration_slopes * raw_event),
            dtype=np.float64,
        )
        if self.hazard_mode == "shared_hazard":
            cumulative = 1.0 - np.cumprod(1.0 - np.clip(calibrated, 0.0, 1.0))
        else:
            cumulative = np.maximum.accumulate(np.clip(calibrated, 0.0, 1.0))
        return cumulative, cumulative.copy()

    def predict_features(
        self,
        features: np.ndarray,
        *,
        action_index: int | None = None,
        observable_action_mask: Sequence[bool] | None = None,
    ) -> V4RiskPrediction:
        if len(self.calibration_support) != len(V4_HORIZON_KEYS):
            raise Issue56V4ModelError("V4 inference requires a calibrated model")
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (_feature_count(self.feature_variant),) or not np.isfinite(values).all():
            raise Issue56V4ModelError("V4 inference features are malformed")
        if observable_action_mask is not None and (
            len(observable_action_mask) != 4
            or any(type(value) is not bool for value in observable_action_mask)
        ):
            raise Issue56V4ModelError("V4 observable action mask is malformed")
        if action_index is not None and (
            isinstance(action_index, bool) or not 0 <= action_index < 4
        ):
            raise Issue56V4ModelError("V4 action index is invalid")
        raw_event, raw_exposure, raw_maximum, raw_comfort, raw_resource, raw_relative, advantage = self._linear_values(values)
        probabilities, upper_probabilities = self._event_probabilities(raw_event)
        horizons: list[V4HorizonPrediction] = []
        for index, horizon in enumerate(V4_HORIZON_KEYS):
            exposure_log = max(0.0, float(raw_exposure[index]))
            maximum_log = max(0.0, float(raw_maximum[index]))
            conditional_exposure = _bounded_expm1(exposure_log)
            upper_exposure = _bounded_expm1(exposure_log + self.exposure_residual_p90[index])
            conditional_maximum = _bounded_expm1(maximum_log)
            upper_maximum = _bounded_expm1(maximum_log + self.maximum_residual_p90[index])
            horizons.append(
                V4HorizonPrediction(
                    horizon,
                    float(probabilities[index]),
                    float(upper_probabilities[index]),
                    conditional_exposure,
                    upper_exposure,
                    conditional_maximum,
                    upper_maximum,
                    float(probabilities[index] * upper_exposure),
                    float(probabilities[index] * upper_maximum),
                )
            )
        comfort = _bounded_expm1(float(raw_comfort[0]))
        upper_comfort = _bounded_expm1(float(raw_comfort[0]) + self.comfort_residual_p90)
        resource = _bounded_expm1(float(raw_resource[0]))
        upper_resource = _bounded_expm1(float(raw_resource[0]) + self.resource_residual_p90)
        relative_upper = raw_relative + self.relative_residual_p90
        incompatible = (
            action_index is not None
            and observable_action_mask is not None
            and not observable_action_mask[action_index]
        )
        remaining = horizons[-1]
        hard = bool(
            any(
                probability > threshold
                for probability, threshold in zip(
                    probabilities,
                    self.event_thresholds,
                    strict=True,
                )
            )
            or remaining.upper_expected_exposure > V4_EXPECTED_EXPOSURE_LIMIT
            or remaining.upper_expected_maximum_crossing > V4_MAXIMUM_CROSSING_LIMIT
            or incompatible
        )
        reason = "observable_action_incompatible" if incompatible else None
        if reason is None and hard:
            reason = "v4_calibrated_risk_limit"
        return V4RiskPrediction(
            tuple(horizons),
            comfort,
            upper_comfort,
            resource,
            upper_resource,
            float(raw_relative[0]),
            float(raw_relative[1]),
            float(raw_relative[2]),
            float(relative_upper[0]),
            float(relative_upper[1]),
            float(relative_upper[2]),
            float(-advantage if self.candidate_id == "c4_advantage_ranker" else 0.0),
            hard,
            reason,
        )

    def predict(
        self,
        history: ForecastHistory,
        action_f32: np.ndarray,
        *,
        decision_step: int,
        alarm_family_slots: Sequence[Sequence[int]],
        action_index: int | None = None,
        observable_action_mask: Sequence[bool] | None = None,
    ) -> V4RiskPrediction:
        if self.feature_variant == "v3_708_past_only":
            features = v2_feature_vector(
                history,
                action_f32,
                decision_step=decision_step,
                alarm_family_slots=alarm_family_slots,
            )
        else:
            features = v4_temporal_feature_vector(
                history,
                action_f32,
                decision_step=decision_step,
                alarm_family_slots=alarm_family_slots,
            )
        return self.predict_features(
            features,
            action_index=action_index,
            observable_action_mask=observable_action_mask,
        )

    def calibrate(self, samples: Sequence[V4RiskSample | V4ModelSample]) -> "V4RiskModel":
        items = _require_v4_samples(samples, "VALIDATION")
        features = _feature_matrix(items, self.feature_variant)
        raw_values = [self._linear_values(row) for row in features]
        raw_event = np.stack([value[0] for value in raw_values])
        cumulative = _cumulative_events(items)
        intervals, at_risk = _interval_events(items)
        intercepts: list[float] = []
        slopes: list[float] = []
        support: list[int] = []
        for index in range(4):
            labels = intervals[:, index] if self.hazard_mode == "shared_hazard" else cumulative[:, index]
            selected = at_risk[:, index] if self.hazard_mode == "shared_hazard" else np.ones(len(items), dtype=bool)
            intercept, slope = _fit_calibration_logistic(raw_event[selected, index], labels[selected])
            intercepts.append(intercept)
            slopes.append(slope)
            support.append(int(np.sum(labels[selected] >= 0.5)))
        exposure, maximum, comfort, resource, relative = _continuous_targets(items)
        conditional = cumulative >= 0.5
        calibrated = replace(
            self,
            calibration_intercepts=np.asarray(intercepts, dtype=np.float64),
            calibration_slopes=np.asarray(slopes, dtype=np.float64),
            exposure_residual_p90=np.asarray(
                [
                    _quantile_residual(
                        np.asarray([value[1][index] for row, value in enumerate(raw_values) if conditional[row, index]]),
                        exposure[conditional[:, index], index],
                    )
                    for index in range(4)
                ],
                dtype=np.float64,
            ),
            maximum_residual_p90=np.asarray(
                [
                    _quantile_residual(
                        np.asarray([value[2][index] for row, value in enumerate(raw_values) if conditional[row, index]]),
                        maximum[conditional[:, index], index],
                    )
                    for index in range(4)
                ],
                dtype=np.float64,
            ),
            comfort_residual_p90=_quantile_residual(
                np.asarray([value[3][0] for value in raw_values]), comfort
            ),
            resource_residual_p90=_quantile_residual(
                np.asarray([value[4][0] for value in raw_values]), resource
            ),
            relative_residual_p90=np.asarray(
                [
                    _quantile_residual(
                        np.asarray([value[5][index] for value in raw_values]),
                        relative[:, index],
                    )
                    for index in range(3)
                ],
                dtype=np.float64,
            ),
            calibration_support=tuple(support),
        )
        calibrated_probabilities = np.stack(
            [calibrated._event_probabilities(value[0])[0] for value in raw_values]
        )
        calibrated = replace(
            calibrated,
            event_thresholds=_select_event_thresholds(
                calibrated_probabilities,
                cumulative,
            ),
        )
        coverage = calibrated._validation_decision_coverage(items)
        if coverage < V4_MIN_VALIDATION_DECISION_COVERAGE:
            raise Issue56V4ModelError("V4 validation decision coverage is below the minimum")
        return replace(calibrated, validation_decision_coverage=coverage)

    def _validation_decision_coverage(
        self, samples: Sequence[V4RiskSample | V4ModelSample]
    ) -> float:
        groups: dict[tuple[str, int], list[V3RiskSample]] = defaultdict(list)
        for sample in samples:
            groups[(sample.family_id, sample.decision_step)].append(sample)
        retained = 0
        for group in groups.values():
            retained += int(
                any(
                    not self.predict_features(
                        _sample_features(item, self.feature_variant),
                        action_index=_action_index(item.action_id),
                        observable_action_mask=item.observable_action_mask,
                    ).hard_ineligible
                    for item in group
                )
            )
        return retained / max(len(groups), 1)

    def score_actions(
        self,
        bundle: ForecastContracts,
        history: ForecastHistory,
        *,
        decision_step: int,
        current_command: np.ndarray,
    ) -> tuple[V4ActionScore, ...]:
        if type(bundle) is not ForecastContracts:
            raise Issue56V4ModelError("V4 action scoring requires frozen contracts")
        actions = tuple(bundle.actions)
        if (
            tuple(action.action_id for action in actions) != V4_ACTION_IDS
            or len(actions) != len(V4_ACTION_IDS)
            or len({action.action_id for action in actions}) != len(actions)
        ):
            raise Issue56V4ModelError("V4 action catalogue is malformed")
        current = np.asarray(current_command, dtype=np.float64)
        if current.shape != (ACTION_COUNT,) or not np.isfinite(current).all():
            raise Issue56V4ModelError("V4 current command is malformed")
        mask = v4_observable_action_mask(bundle, history)
        alarm_slots = alarm_family_slot_indices(bundle)
        scores: list[V4ActionScore] = []
        for index, action in enumerate(actions):
            action_vector = project_proposed_action(bundle, action.command)
            prediction = self.predict(
                history,
                action_vector,
                decision_step=decision_step,
                alarm_family_slots=alarm_slots,
                action_index=index,
                observable_action_mask=mask,
            )
            command_vector = _command_vector(
                bundle.development_scenario,
                action.command.to_mapping(),
            )
            intervention = float(np.mean(np.abs(command_vector - current)))
            if self.candidate_id == "c4_advantage_ranker":
                utility = prediction.advantage_score
            else:
                utility = (
                    prediction.horizons[-1].upper_expected_exposure
                    + V4_UTILITY_WEIGHTS["safety_exposure"]
                    * prediction.upper_relative_safety_exposure
                    + V4_UTILITY_WEIGHTS["comfort_deviation"]
                    * prediction.upper_relative_comfort_deviation
                    + V4_UTILITY_WEIGHTS["resource_composite"]
                    * prediction.upper_relative_resource_composite
                    + V4_UTILITY_WEIGHTS["intervention"] * intervention
                )
            hard = prediction.hard_ineligible or not mask[index]
            scores.append(
                V4ActionScore(
                    action.action_id,
                    index,
                    bool(mask[index]),
                    hard,
                    float(utility) if not hard else math.inf,
                    intervention,
                    prediction,
                    prediction.reason,
                )
            )
        return tuple(scores)

    def select_action(self, scores: Sequence[V4ActionScore]) -> V4ActionScore | None:
        items = tuple(scores)
        if not items:
            raise Issue56V4ModelError("V4 action selection requires scores")
        identifiers = [item.action_id for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise Issue56V4ModelError("V4 action selection received duplicate actions")
        eligible = [item for item in items if not item.hard_ineligible and item.compatible]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda item: (
                item.utility_score,
                item.prediction.horizons[-1].upper_expected_exposure,
                item.prediction.horizons[-1].event_probability,
                item.intervention,
                item.action_id,
            ),
        )

    def _body(self) -> dict[str, Any]:
        def values(array: np.ndarray) -> list[float]:
            return array.tolist()

        return {
            "schema_version": V4_MODEL_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "feature_variant": self.feature_variant,
            "model_kind": self.model_kind,
            "hazard_mode": self.hazard_mode,
            "feature_mean": values(self.feature_mean),
            "feature_scale": values(self.feature_scale),
            "event_intercepts": values(self.event_intercepts),
            "event_coefficients": values(self.event_coefficients),
            "hazard_intercepts": values(self.hazard_intercepts),
            "hazard_coefficients": values(self.hazard_coefficients),
            "exposure_target_means": values(self.exposure_target_means),
            "exposure_target_scales": values(self.exposure_target_scales),
            "exposure_coefficients": values(self.exposure_coefficients),
            "maximum_target_means": values(self.maximum_target_means),
            "maximum_target_scales": values(self.maximum_target_scales),
            "maximum_coefficients": values(self.maximum_coefficients),
            "comfort_target_mean": self.comfort_target_mean,
            "comfort_target_scale": self.comfort_target_scale,
            "comfort_coefficients": values(self.comfort_coefficients),
            "resource_target_mean": self.resource_target_mean,
            "resource_target_scale": self.resource_target_scale,
            "resource_coefficients": values(self.resource_coefficients),
            "relative_target_means": values(self.relative_target_means),
            "relative_target_scales": values(self.relative_target_scales),
            "relative_coefficients": values(self.relative_coefficients),
            "advantage_intercept": self.advantage_intercept,
            "advantage_coefficients": values(self.advantage_coefficients),
            "calibration_intercepts": values(self.calibration_intercepts),
            "calibration_slopes": values(self.calibration_slopes),
            "exposure_residual_p90": values(self.exposure_residual_p90),
            "maximum_residual_p90": values(self.maximum_residual_p90),
            "comfort_residual_p90": self.comfort_residual_p90,
            "resource_residual_p90": self.resource_residual_p90,
            "relative_residual_p90": values(self.relative_residual_p90),
            "alpha": self.alpha,
            "seed": self.seed,
            "actuator_authority": False,
            "hidden_weights": None if self.hidden_weights is None else values(self.hidden_weights),
            "hidden_biases": None if self.hidden_biases is None else values(self.hidden_biases),
            "output_weights": None if self.output_weights is None else values(self.output_weights),
            "output_biases": None if self.output_biases is None else values(self.output_biases),
            "calibration_support": list(self.calibration_support),
            "validation_decision_coverage": self.validation_decision_coverage,
            "event_thresholds": list(self.event_thresholds),
            "provenance": _provenance_mapping(self.provenance),
        }

    def to_mapping(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "model_sha256": _sha(body)}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V4RiskModel":
        expected = {
            "schema_version",
            "candidate_id",
            "feature_variant",
            "model_kind",
            "hazard_mode",
            "feature_mean",
            "feature_scale",
            "event_intercepts",
            "event_coefficients",
            "hazard_intercepts",
            "hazard_coefficients",
            "exposure_target_means",
            "exposure_target_scales",
            "exposure_coefficients",
            "maximum_target_means",
            "maximum_target_scales",
            "maximum_coefficients",
            "comfort_target_mean",
            "comfort_target_scale",
            "comfort_coefficients",
            "resource_target_mean",
            "resource_target_scale",
            "resource_coefficients",
            "relative_target_means",
            "relative_target_scales",
            "relative_coefficients",
            "advantage_intercept",
            "advantage_coefficients",
            "calibration_intercepts",
            "calibration_slopes",
            "exposure_residual_p90",
            "maximum_residual_p90",
            "comfort_residual_p90",
            "resource_residual_p90",
            "relative_residual_p90",
            "alpha",
            "seed",
            "actuator_authority",
            "hidden_weights",
            "hidden_biases",
            "output_weights",
            "output_biases",
            "calibration_support",
            "validation_decision_coverage",
            "event_thresholds",
            "provenance",
            "model_sha256",
        }
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V4ModelError("V4 model artifact fields drift")
        body = dict(mapping)
        digest = body.pop("model_sha256")
        if body.get("schema_version") != V4_MODEL_SCHEMA_VERSION or digest != _sha(body):
            raise Issue56V4ModelError("V4 model artifact digest or schema is invalid")
        if any(
            type(body[field]) is not str
            for field in ("candidate_id", "feature_variant", "model_kind", "hazard_mode")
        ):
            raise Issue56V4ModelError("V4 model artifact identities are malformed")
        feature_count = _feature_count(body["feature_variant"])
        return cls(
            body["candidate_id"],
            body["feature_variant"],
            body["model_kind"],
            body["hazard_mode"],
            _strict_array(body["feature_mean"], (feature_count,), "feature mean"),
            _strict_array(body["feature_scale"], (feature_count,), "feature scale"),
            _strict_array(body["event_intercepts"], (4,), "event intercepts"),
            _strict_array(body["event_coefficients"], (4, feature_count), "event coefficients"),
            _strict_array(body["hazard_intercepts"], (4,), "hazard intercepts"),
            _strict_array(body["hazard_coefficients"], (4, feature_count), "hazard coefficients"),
            _strict_array(body["exposure_target_means"], (4,), "exposure target means"),
            _strict_array(body["exposure_target_scales"], (4,), "exposure target scales"),
            _strict_array(body["exposure_coefficients"], (4, feature_count), "exposure coefficients"),
            _strict_array(body["maximum_target_means"], (4,), "maximum target means"),
            _strict_array(body["maximum_target_scales"], (4,), "maximum target scales"),
            _strict_array(body["maximum_coefficients"], (4, feature_count), "maximum coefficients"),
            _strict_scalar(body["comfort_target_mean"], "comfort target mean"),
            _strict_scalar(body["comfort_target_scale"], "comfort target scale"),
            _strict_array(body["comfort_coefficients"], (feature_count,), "comfort coefficients"),
            _strict_scalar(body["resource_target_mean"], "resource target mean"),
            _strict_scalar(body["resource_target_scale"], "resource target scale"),
            _strict_array(body["resource_coefficients"], (feature_count,), "resource coefficients"),
            _strict_array(body["relative_target_means"], (3,), "relative target means"),
            _strict_array(body["relative_target_scales"], (3,), "relative target scales"),
            _strict_array(body["relative_coefficients"], (3, feature_count), "relative coefficients"),
            _strict_scalar(body["advantage_intercept"], "advantage intercept"),
            _strict_array(body["advantage_coefficients"], (feature_count,), "advantage coefficients"),
            _strict_array(body["calibration_intercepts"], (4,), "calibration intercepts"),
            _strict_array(body["calibration_slopes"], (4,), "calibration slopes"),
            _strict_array(body["exposure_residual_p90"], (4,), "exposure residuals"),
            _strict_array(body["maximum_residual_p90"], (4,), "maximum residuals"),
            _strict_scalar(body["comfort_residual_p90"], "comfort residual"),
            _strict_scalar(body["resource_residual_p90"], "resource residual"),
            _strict_array(body["relative_residual_p90"], (3,), "relative residuals"),
            _strict_scalar(body["alpha"], "alpha"),
            _strict_int(body["seed"], "seed"),
            _strict_bool(body["actuator_authority"], "actuator authority"),
            None if body["hidden_weights"] is None else _strict_array(
                body["hidden_weights"], (feature_count, V4_MLP_HIDDEN_UNITS), "hidden weights"
            ),
            None if body["hidden_biases"] is None else _strict_array(
                body["hidden_biases"], (V4_MLP_HIDDEN_UNITS,), "hidden biases"
            ),
            None if body["output_weights"] is None else _strict_array(
                body["output_weights"], (V4_MLP_HIDDEN_UNITS, 17), "output weights"
            ),
            None if body["output_biases"] is None else _strict_array(
                body["output_biases"], (17,), "output biases"
            ),
            _strict_int_list(body["calibration_support"], "calibration support"),
            _strict_scalar(body["validation_decision_coverage"], "validation decision coverage"),
            _strict_scalar_list(body["event_thresholds"], "event thresholds"),
            str(digest),
            _provenance_tuple(body["provenance"]),
        )


def write_v4_model(path: str | Path, model: V4RiskModel) -> str:
    """Write a canonical JSON model artifact and return its byte digest."""

    if type(model) is not V4RiskModel:
        raise Issue56V4ModelError("V4 model writer requires an exact model type")
    if len(model.calibration_support) != len(V4_HORIZON_KEYS):
        raise Issue56V4ModelError("V4 model writer requires a calibrated model")
    if len(model.provenance) != len(V4_MODEL_PROVENANCE_FIELDS):
        raise Issue56V4ModelError("V4 model writer requires complete provenance")
    destination = Path(path)
    payload = json.dumps(
        model.to_mapping(),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=2,
    ).encode("utf-8")
    destination.write_bytes(payload)
    return _sha_bytes(payload)


def load_v4_model(path: str | Path) -> tuple[V4RiskModel, str]:
    """Load and validate a strict JSON model artifact."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        mapping = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Issue56V4ModelError("V4 model artifact is not strict JSON") from error
    model = V4RiskModel.from_mapping(mapping)
    return model, _sha_bytes(raw)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


__all__ = [
    "Issue56V4ModelError",
    "V4ActionScore",
    "V4_ACTION_IDS",
    "V4_HORIZON_KEYS",
    "V4HorizonPrediction",
    "V4ModelSample",
    "V4RiskModel",
    "V4RiskPrediction",
    "V4_MODEL_CANDIDATES",
    "V4_MODEL_SCHEMA_VERSION",
    "V4_FEATURE_VARIANTS",
    "load_v4_model",
    "write_v4_model",
]
