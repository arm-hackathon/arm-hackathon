"""Deterministic, fail-closed Forecast D1 baseline implementations.

The module consumes only Track A's public ``ForecastHistory`` projection plus a
complete proposed action.  It deliberately has no corpus or HMC dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .projection import ForecastHistory

RELEASE_TIER: Final = "DEVELOPMENT_FIXTURE_ONLY"
INPUT_MANIFEST_SHA256: Final = (
    "29d743472712dff68759477debd25aadba8a0584ad89d164bc5c583260356971"
)
TARGET_MANIFEST_SHA256: Final = (
    "26e480ca4f07d2092fc6e96fcf2f006948e9e2872ad2b0fd4ae3ac8e947c74db"
)
WINDOW_CANDIDATES: Final = frozenset((4, 8, 16))
HORIZON_CANDIDATES: Final = frozenset((2, 4, 8))
RIDGE_ALPHAS: Final = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
TARGET_COUNT: Final = 51
ACTION_COUNT: Final = 27


class BaselineError(ValueError):
    """Projected baseline evidence is outside the frozen D1 contract."""


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    """A whole-sample prediction or an honest no-prediction result."""

    status: str
    values: np.ndarray | None


@dataclass(frozen=True, slots=True)
class RidgeSample:
    sample_id: str
    family_cluster_id: str
    split_label: str
    history: ForecastHistory
    proposed_action_f32: np.ndarray
    targets_f32: np.ndarray
    input_manifest_sha256: str
    target_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DirectRidgeModel:
    """Float64 reference ridge fit; prediction returns immutable float32 Hx51."""

    alpha: float
    include_action: bool
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coef: np.ndarray
    window_steps: int
    horizon_steps: int
    input_manifest_sha256: str
    target_manifest_sha256: str

    def predict(
        self, history: ForecastHistory, proposed_action_f32: np.ndarray
    ) -> np.ndarray:
        _validate_history(history, horizon_steps=self.horizon_steps)
        if history.numeric_f32.shape[0] != self.window_steps:
            raise BaselineError("inference window differs from fitted binding")
        if (
            history.layout.input_manifest_sha256,
            history.layout.target_manifest_sha256,
        ) != (
            self.input_manifest_sha256,
            self.target_manifest_sha256,
        ):
            raise BaselineError("inference manifest identity drift")
        feature = flatten_features(
            history, proposed_action_f32, include_action=self.include_action
        )
        if feature.shape != self.feature_mean.shape:
            raise BaselineError("inference feature binding drift")
        model_dtype = self.coef.dtype
        if model_dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or any(
            array.dtype != model_dtype
            for array in (
                self.feature_mean,
                self.feature_scale,
                self.target_mean,
                self.coef,
            )
        ):
            raise BaselineError("ridge inference precision contract is invalid")
        feature_for_model = feature.astype(model_dtype, copy=False)
        value64 = (
            (feature_for_model - self.feature_mean) / self.feature_scale
        ) @ self.coef + self.target_mean
        return _as_f32(
            value64.reshape(self.horizon_steps, TARGET_COUNT), "ridge prediction"
        )


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _as_f32(value: np.ndarray, label: str) -> np.ndarray:
    array64 = np.asarray(value, dtype=np.float64)
    if (
        not np.isfinite(array64).all()
        or np.max(np.abs(array64), initial=0.0) > np.finfo(np.float32).max
    ):
        raise BaselineError(f"{label} is non-finite or overflows float32")
    result = array64.astype(np.float32)
    if not np.isfinite(result).all():
        raise BaselineError(f"{label} does not remain finite float32")
    return _readonly(result)


def _validate_history(history: ForecastHistory, *, horizon_steps: int) -> None:
    if type(history) is not ForecastHistory:
        raise BaselineError("baseline requires exact Track A ForecastHistory")
    if horizon_steps not in HORIZON_CANDIDATES:
        raise BaselineError("unsupported D1 horizon")
    rows = len(history.steps)
    if (
        rows not in WINDOW_CANDIDATES
        or history.numeric_f32.shape != (rows, 194)
        or history.status_f32.shape != (rows, 167, 5)
    ):
        raise BaselineError("history has unsupported D1 window/dimensions")
    if (
        history.mode_f32.shape != (rows, 4)
        or history.health_f32.shape != (rows, 4)
        or history.alarm_lifecycle_f32.shape != (rows, 287, 4)
    ):
        raise BaselineError("history ancillary tensors have wrong dimensions")
    if (
        history.layout.input_manifest_sha256,
        history.layout.target_manifest_sha256,
    ) != (INPUT_MANIFEST_SHA256, TARGET_MANIFEST_SHA256):
        raise BaselineError("history manifest identities drift")
    arrays = (
        history.numeric_f32,
        history.status_f32,
        history.mode_f32,
        history.health_f32,
        history.alarm_lifecycle_f32,
    )
    if any(
        array.dtype != np.float32 or not np.isfinite(array).all() for array in arrays
    ):
        raise BaselineError("history tensors must be finite float32")
    if len(history.completed_times_s) != rows or len(history.steps) != rows:
        raise BaselineError("history time dimensions drift")
    times = np.asarray(history.completed_times_s, dtype=np.float64)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise BaselineError("completed times must be finite and strictly increasing")
    statuses = history.status_f32
    if not np.all((statuses == 0.0) | (statuses == 1.0)) or not np.all(
        statuses.sum(axis=2) == 1.0
    ):
        raise BaselineError("history availability status is malformed")
    if (
        len(history.layout.operational_descriptors) != 167
        or len(history.layout.target_descriptors) != TARGET_COUNT
    ):
        raise BaselineError("history descriptor dimensions drift")


def _available(history: ForecastHistory, row: int, column: int) -> float | None:
    return (
        float(history.numeric_f32[row, column])
        if history.status_f32[row, column, 0] == 1.0
        else None
    )


def _target_operational_columns(
    history: ForecastHistory,
) -> tuple[tuple[int, int, int | None], ...]:
    descriptor_map: dict[tuple[str, str], int] = {}
    for column, raw in enumerate(history.layout.operational_descriptors):
        try:
            descriptor_map[(raw["descriptor_id"], raw["source_kind"])] = column
        except (KeyError, TypeError) as error:
            raise BaselineError("operational descriptor is malformed") from error
    slots: list[tuple[int, int, int | None]] = []
    for raw in history.layout.target_descriptors:
        try:
            target_id = raw["descriptor_id"]
        except (KeyError, TypeError) as error:
            raise BaselineError("target descriptor is malformed") from error
        if "/" in target_id and target_id.rsplit("/", 1)[1] in {
            "temperature_k",
            "pressure_pa",
            "co2_ppm",
            "o2_mole_fraction",
            "relative_humidity",
        }:
            primary = descriptor_map.get((target_id, "primary_sensor_head"))
            secondary = descriptor_map.get((target_id, "secondary_sensor_head"))
            if primary is None or secondary is None:
                raise BaselineError("environmental target lacks ordered sensor heads")
            slots.append((primary, secondary, None))
        elif target_id.endswith("/branch_airflow_m3_s"):
            zone = target_id.rsplit("/", 1)[0]
            feedback = descriptor_map.get(
                (f"branch_airflow_m3_s/{zone}", "operational_feedback_instrument")
            )
            if feedback is None:
                raise BaselineError("airflow target lacks operational feedback")
            slots.append((feedback, feedback, None))
        elif target_id in {
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        }:
            feedback = descriptor_map.get(
                (target_id, "operational_feedback_instrument")
            )
            if feedback is None:
                raise BaselineError(
                    "resource target lacks duplicate-validated operational feedback"
                )
            slots.append((feedback, feedback, None))
        else:
            raise BaselineError("unknown target descriptor")
    if len(slots) != TARGET_COUNT:
        raise BaselineError("target layout must contain exactly 51 slots")
    return tuple(slots)


def _causal_estimates(history: ForecastHistory) -> np.ndarray:
    slots = _target_operational_columns(history)
    values = np.full(
        (history.numeric_f32.shape[0], TARGET_COUNT), np.nan, dtype=np.float64
    )
    for row in range(values.shape[0]):
        for target, (first, second, _) in enumerate(slots):
            left, right = (
                _available(history, row, first),
                _available(history, row, second),
            )
            if first == second:
                values[row, target] = np.nan if left is None else left
            elif left is not None and right is not None:
                values[row, target] = (left + right) / 2.0
            elif left is not None:
                values[row, target] = left
            elif right is not None:
                values[row, target] = right
    return values


def persistence(history: ForecastHistory, *, horizon_steps: int) -> BaselinePrediction:
    _validate_history(history, horizon_steps=horizon_steps)
    latest = _causal_estimates(history)[-1]
    if not np.isfinite(latest).all():
        return BaselinePrediction("ABSTAIN", None)
    return BaselinePrediction(
        "PREDICTION",
        _as_f32(np.repeat(latest[None, :], horizon_steps, axis=0), "persistence"),
    )


def linear_extrapolation(
    history: ForecastHistory, *, horizon_steps: int, future_times_s: Sequence[float]
) -> BaselinePrediction:
    _validate_history(history, horizon_steps=horizon_steps)
    if isinstance(future_times_s, (str, bytes)) or len(future_times_s) != horizon_steps:
        raise BaselineError("future cadence must explicitly contain each horizon time")
    future = np.asarray(future_times_s, dtype=np.float64)
    if (
        not np.isfinite(future).all()
        or (np.diff(future) <= 0).any()
        or future[0] <= history.completed_times_s[-1]
    ):
        raise BaselineError("future cadence must be finite, increasing and causal")
    observations, times = (
        _causal_estimates(history),
        np.asarray(history.completed_times_s, dtype=np.float64),
    )
    result = np.empty((horizon_steps, TARGET_COUNT), dtype=np.float64)
    for target in range(TARGET_COUNT):
        mask = np.isfinite(observations[:, target])
        count = int(mask.sum())
        if count == 0:
            return BaselinePrediction("ABSTAIN", None)
        if count < 3:
            result[:, target] = observations[mask, target][-1]
        else:
            slope, intercept = np.polyfit(times[mask], observations[mask, target], 1)
            result[:, target] = intercept + slope * future
    return BaselinePrediction("PREDICTION", _as_f32(result, "linear extrapolation"))


def _action(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action)
    if (
        array.shape != (ACTION_COUNT,)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise BaselineError("proposed action must be finite float32[27]")
    return array


def flatten_features(
    history: ForecastHistory, proposed_action_f32: np.ndarray, *, include_action: bool
) -> np.ndarray:
    _validate_history(history, horizon_steps=2)
    action = _action(proposed_action_f32)
    tensors = (
        history.numeric_f32,
        history.status_f32,
        history.mode_f32,
        history.health_f32,
        history.alarm_lifecycle_f32,
    )
    if any(
        array.dtype != np.float32 or not np.isfinite(array).all() for array in tensors
    ):
        raise BaselineError("feature tensor is non-finite")
    pieces = [array.reshape(-1) for array in tensors]
    if include_action:
        pieces.append(action)
    return _readonly(np.concatenate(pieces).astype(np.float32, copy=False))


def _validate_sample(sample: RidgeSample, horizon_steps: int) -> None:
    if (
        type(sample) is not RidgeSample
        or not sample.sample_id
        or not sample.family_cluster_id
    ):
        raise BaselineError("ridge sample identity is invalid")
    if sample.split_label != "TRAIN":
        raise BaselineError("ridge fitting accepts TRAIN samples only")
    _validate_history(sample.history, horizon_steps=horizon_steps)
    if (sample.input_manifest_sha256, sample.target_manifest_sha256) != (
        INPUT_MANIFEST_SHA256,
        TARGET_MANIFEST_SHA256,
    ):
        raise BaselineError("ridge sample manifest identity drift")
    _action(sample.proposed_action_f32)
    target = np.asarray(sample.targets_f32)
    if (
        target.shape != (horizon_steps, TARGET_COUNT)
        or target.dtype != np.float32
        or not np.isfinite(target).all()
    ):
        raise BaselineError("ridge target must be finite float32[H,51]")


def _fit(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    centered_target = targets - targets.mean(axis=0)
    standardized = (features - mean) / scale
    # D1 has far more contracted features than fixture samples.  The dual
    # identity is the same closed-form ridge solution but keeps the reference
    # solve on the deterministic sample-by-sample Gram matrix.
    gram = standardized @ standardized.T
    dual = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), centered_target)
    coefficient = standardized.T @ dual
    return mean, scale, targets.mean(axis=0), coefficient


def _normalised_error(
    prediction: np.ndarray, truth: np.ndarray, training_targets: np.ndarray
) -> float:
    scale = np.percentile(training_targets, 95, axis=0) - np.percentile(
        training_targets, 5, axis=0
    )
    supported = scale > 0.0
    if not supported.any():
        raise BaselineError("validation fold has no supported target scale")
    return float(
        np.mean(
            np.abs(prediction[:, supported] - truth[:, supported]) / scale[supported]
        )
    )


def fit_direct_ridge(
    samples: Sequence[RidgeSample], *, horizon_steps: int, include_action: bool = True
) -> DirectRidgeModel:
    if (
        horizon_steps not in HORIZON_CANDIDATES
        or isinstance(samples, (str, bytes))
        or len(samples) < 3
    ):
        raise BaselineError("ridge requires supported H and at least three samples")
    items = tuple(samples)
    for item in items:
        _validate_sample(item, horizon_steps)
    if (
        len({item.sample_id for item in items}) != len(items)
        or len({item.family_cluster_id for item in items}) < 2
    ):
        raise BaselineError(
            "ridge sample/cluster identities are insufficient or duplicate"
        )
    windows = {item.history.numeric_f32.shape[0] for item in items}
    if len(windows) != 1:
        raise BaselineError("ridge cannot mix window bindings")
    feature = np.stack(
        [
            flatten_features(
                x.history, x.proposed_action_f32, include_action=include_action
            )
            for x in items
        ]
    ).astype(np.float64)
    targets = np.stack([x.targets_f32.reshape(-1) for x in items]).astype(np.float64)
    cluster_order = tuple(sorted({item.family_cluster_id for item in items}))
    errors: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        fold_errors: list[float] = []
        for cluster in cluster_order:
            validation = np.asarray([x.family_cluster_id == cluster for x in items])
            training = ~validation
            if (
                not training.any()
                or not validation.any()
                or set(np.asarray([x.family_cluster_id for x in items])[training])
                & {cluster}
            ):
                raise BaselineError("whole-cluster split leakage")
            mean, scale, target_mean, coef = _fit(
                feature[training], targets[training], alpha
            )
            predicted = ((feature[validation] - mean) / scale) @ coef + target_mean
            fold_errors.append(
                _normalised_error(predicted, targets[validation], targets[training])
            )
        errors.append((float(np.mean(fold_errors)), alpha))
    _, selected_alpha = min(errors, key=lambda item: (item[0], item[1]))
    mean, scale, target_mean, coef = _fit(feature, targets, selected_alpha)
    for array in (mean, scale, target_mean, coef):
        if not np.isfinite(array).all():
            raise BaselineError("ridge fit is non-finite")
        array.setflags(write=False)
    return DirectRidgeModel(
        selected_alpha,
        include_action,
        mean,
        scale,
        target_mean,
        coef,
        next(iter(windows)),
        horizon_steps,
        INPUT_MANIFEST_SHA256,
        TARGET_MANIFEST_SHA256,
    )


def fit_action_aware_and_blinded(
    samples: Sequence[RidgeSample], *, horizon_steps: int
) -> tuple[DirectRidgeModel, DirectRidgeModel]:
    """Fit paired diagnostics from identical samples; only action fields differ."""
    return (
        fit_direct_ridge(samples, horizon_steps=horizon_steps, include_action=True),
        fit_direct_ridge(samples, horizon_steps=horizon_steps, include_action=False),
    )
