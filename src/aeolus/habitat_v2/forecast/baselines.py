"""Deterministic, fail-closed Forecast D1 baseline implementations.

The module consumes only Track A's public ``ForecastHistory`` projection plus a
complete proposed action.  It deliberately has no corpus or HMC dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
from typing import Final

import numpy as np

from .projection import ForecastHistory

RELEASE_TIER: Final = "DEVELOPMENT_FIXTURE_ONLY"
INPUT_MANIFEST_SHA256: Final = (
    "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
)
TARGET_MANIFEST_SHA256: Final = (
    "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"
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
        value64 = (
            (feature.astype(np.float64) - self.feature_mean) / self.feature_scale
        ) @ self.coef + self.target_mean
        return _as_f32(
            value64.reshape(self.horizon_steps, TARGET_COUNT), "ridge prediction"
        )


@dataclass(frozen=True, slots=True)
class NestedInnerFold:
    """One policy-bound inner whole-cluster partition inside an outer train set."""

    index: int
    train_cluster_ids: tuple[str, ...]
    validation_cluster_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NestedOuterFold:
    """One policy-bound outer whole-cluster test partition."""

    index: int
    train_cluster_ids: tuple[str, ...]
    test_cluster_ids: tuple[str, ...]
    inner_folds: tuple[NestedInnerFold, ...]


@dataclass(frozen=True, slots=True)
class NestedRidgeFoldPlan:
    """Immutable D2 fold assignment, cryptographically bound to one policy."""

    policy_sha256: str
    outer_folds: tuple[NestedOuterFold, ...]


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


def _policy_hmac_rank(*, key: bytes, domain: bytes, fields: tuple[str, ...]) -> bytes:
    if any("\0" in field for field in fields):
        raise BaselineError("nested-fold identity fields cannot contain NUL")
    message = domain + b"\0" + b"\0".join(
        field.encode("utf-8") for field in fields
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def build_nested_ridge_fold_plan(
    *, policy: object, cluster_strata: Mapping[str, str]
) -> NestedRidgeFoldPlan:
    """Compile the ratified D2 5x4 stratified HMAC fold plan.

    This is a planning operation only: it neither materializes a scenario nor
    fits a model.  The caller must carry stratum metadata from the frozen D2
    records; parsing a free-form cluster ID would create a second split policy.
    """

    from .qualification import QualificationPolicy, validate_ratified_policy_design

    if type(policy) is not QualificationPolicy:
        raise BaselineError("nested ridge folds require an exact qualification policy")
    if policy.ratification_status != "APPROVED":
        raise BaselineError("nested ridge folds require a ratified policy")
    validate_ratified_policy_design(policy)
    if not isinstance(cluster_strata, Mapping):
        raise BaselineError("nested ridge cluster strata must be a mapping")

    by_stratum: dict[str, list[str]] = {}
    for cluster_id, stratum_id in cluster_strata.items():
        if (
            type(cluster_id) is not str
            or not cluster_id
            or type(stratum_id) is not str
            or not stratum_id
            or "\0" in cluster_id
            or "\0" in stratum_id
        ):
            raise BaselineError("nested ridge cluster and stratum identities are invalid")
        by_stratum.setdefault(stratum_id, []).append(cluster_id)

    if len(cluster_strata) != 60 or len(by_stratum) != 12:
        raise BaselineError("nested ridge folds require exactly 60 clusters in 12 strata")
    if any(len(cluster_ids) != 5 for cluster_ids in by_stratum.values()):
        raise BaselineError("nested ridge folds require exactly five clusters per stratum")

    key = bytes.fromhex(policy.policy_sha256)
    clusters_by_outer: list[list[str]] = [[] for _ in range(5)]
    for stratum_id in sorted(by_stratum):
        ranked = sorted(
            by_stratum[stratum_id],
            key=lambda cluster_id: (
                _policy_hmac_rank(
                    key=key,
                    domain=b"aeolus-forecast-d2-outer-v1",
                    fields=(stratum_id, cluster_id),
                ),
                cluster_id,
            ),
        )
        for outer_index, cluster_id in enumerate(ranked):
            clusters_by_outer[outer_index].append(cluster_id)

    all_cluster_ids = frozenset(cluster_strata)
    outer_folds: list[NestedOuterFold] = []
    for outer_index, test_ids in enumerate(clusters_by_outer):
        test_cluster_ids = tuple(sorted(test_ids))
        test_set = frozenset(test_cluster_ids)
        train_cluster_ids = tuple(sorted(all_cluster_ids - test_set))
        if len(test_cluster_ids) != 12 or len(train_cluster_ids) != 48:
            raise BaselineError("nested outer fold cardinality drifts")

        validation_by_inner: list[list[str]] = [[] for _ in range(4)]
        for stratum_id in sorted(by_stratum):
            outer_training = sorted(
                set(by_stratum[stratum_id]) - test_set
            )
            if len(outer_training) != 4:
                raise BaselineError("nested outer stratum cardinality drifts")
            ranked = sorted(
                outer_training,
                key=lambda cluster_id: (
                    _policy_hmac_rank(
                        key=key,
                        domain=b"aeolus-forecast-d2-inner-v1",
                        fields=(str(outer_index), stratum_id, cluster_id),
                    ),
                    cluster_id,
                ),
            )
            for inner_index, cluster_id in enumerate(ranked):
                validation_by_inner[inner_index].append(cluster_id)

        inner_folds: list[NestedInnerFold] = []
        for inner_index, validation_ids in enumerate(validation_by_inner):
            validation_cluster_ids = tuple(sorted(validation_ids))
            validation_set = frozenset(validation_cluster_ids)
            inner_train_cluster_ids = tuple(sorted(set(train_cluster_ids) - validation_set))
            if len(validation_cluster_ids) != 12 or len(inner_train_cluster_ids) != 36:
                raise BaselineError("nested inner fold cardinality drifts")
            if validation_set & test_set or set(inner_train_cluster_ids) & test_set:
                raise BaselineError("nested ridge fold leakage")
            inner_folds.append(
                NestedInnerFold(
                    index=inner_index,
                    train_cluster_ids=inner_train_cluster_ids,
                    validation_cluster_ids=validation_cluster_ids,
                )
            )

        if frozenset().union(
            *(set(inner.validation_cluster_ids) for inner in inner_folds)
        ) != set(train_cluster_ids):
            raise BaselineError("inner validation folds do not partition outer training")
        outer_folds.append(
            NestedOuterFold(
                index=outer_index,
                train_cluster_ids=train_cluster_ids,
                test_cluster_ids=test_cluster_ids,
                inner_folds=tuple(inner_folds),
            )
        )
    return NestedRidgeFoldPlan(
        policy_sha256=policy.policy_sha256,
        outer_folds=tuple(outer_folds),
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


@dataclass(frozen=True, slots=True)
class NestedRidgeOuterResult:
    """One outer-held-out model and its inner-only alpha-selection receipt."""

    outer_fold_index: int
    selected_alpha: float
    alpha_validation_errors: tuple[tuple[float, float], ...]
    train_cluster_ids: tuple[str, ...]
    test_cluster_ids: tuple[str, ...]
    model_training_cluster_ids: tuple[str, ...]
    model: DirectRidgeModel


@dataclass(frozen=True, slots=True)
class NestedRidgeResult:
    """D2 nested-CV models; intentionally does not choose one deployable model."""

    fold_plan: NestedRidgeFoldPlan
    outer_results: tuple[NestedRidgeOuterResult, ...]


def _ridge_model_from_fit(
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    target_mean: np.ndarray,
    coefficient: np.ndarray,
    alpha: float,
    include_action: bool,
    window_steps: int,
    horizon_steps: int,
) -> DirectRidgeModel:
    for array in (mean, scale, target_mean, coefficient):
        if not np.isfinite(array).all():
            raise BaselineError("ridge fit is non-finite")
        array.setflags(write=False)
    return DirectRidgeModel(
        alpha,
        include_action,
        mean,
        scale,
        target_mean,
        coefficient,
        window_steps,
        horizon_steps,
        INPUT_MANIFEST_SHA256,
        TARGET_MANIFEST_SHA256,
    )


def fit_direct_ridge_nested(
    samples: Sequence[RidgeSample],
    *,
    horizon_steps: int,
    policy: object,
    cluster_strata: Mapping[str, str],
    include_action: bool = True,
) -> NestedRidgeResult:
    """Fit five outer models with alpha selected only by their inner folds.

    This is a deterministic, local qualification primitive.  It cannot create
    a corpus or open the policy's generation/training permissions, and it
    deliberately returns no single refit-on-all-clusters model.
    """

    if (
        horizon_steps not in HORIZON_CANDIDATES
        or isinstance(samples, (str, bytes))
    ):
        raise BaselineError("nested ridge requires a supported horizon and samples")
    items = tuple(samples)
    for item in items:
        _validate_sample(item, horizon_steps)
    fold_plan = build_nested_ridge_fold_plan(
        policy=policy,
        cluster_strata=cluster_strata,
    )
    sample_ids = tuple(item.sample_id for item in items)
    sample_clusters = tuple(item.family_cluster_id for item in items)
    expected_clusters = frozenset(cluster_strata)
    if (
        len(items) < len(expected_clusters)
        or len(set(sample_ids)) != len(sample_ids)
        or frozenset(sample_clusters) != expected_clusters
    ):
        raise BaselineError(
            "nested ridge samples must cover each and only every planned cluster"
        )
    windows = {item.history.numeric_f32.shape[0] for item in items}
    if len(windows) != 1:
        raise BaselineError("nested ridge cannot mix window bindings")
    features = np.stack(
        [
            flatten_features(
                item.history,
                item.proposed_action_f32,
                include_action=include_action,
            )
            for item in items
        ]
    ).astype(np.float64)
    targets = np.stack(
        [item.targets_f32.reshape(-1) for item in items]
    ).astype(np.float64)
    indices_by_cluster = {
        cluster_id: np.flatnonzero(
            np.asarray([item.family_cluster_id == cluster_id for item in items])
        )
        for cluster_id in expected_clusters
    }

    outer_results: list[NestedRidgeOuterResult] = []
    for outer in fold_plan.outer_folds:
        alpha_errors: list[tuple[float, float]] = []
        for alpha in RIDGE_ALPHAS:
            inner_errors: list[float] = []
            for inner in outer.inner_folds:
                train_indices = np.concatenate(
                    [indices_by_cluster[cluster_id] for cluster_id in inner.train_cluster_ids]
                )
                mean, scale, target_mean, coefficient = _fit(
                    features[train_indices], targets[train_indices], alpha
                )
                cluster_errors: list[float] = []
                for cluster_id in inner.validation_cluster_ids:
                    validation_indices = indices_by_cluster[cluster_id]
                    prediction = (
                        (features[validation_indices] - mean) / scale
                    ) @ coefficient + target_mean
                    cluster_errors.append(
                        _normalised_error(
                            prediction,
                            targets[validation_indices],
                            targets[train_indices],
                        )
                    )
                inner_errors.append(float(np.mean(cluster_errors)))
            alpha_errors.append((float(np.mean(inner_errors)), alpha))
        _, selected_alpha = min(alpha_errors, key=lambda item: (item[0], item[1]))
        outer_train_indices = np.concatenate(
            [indices_by_cluster[cluster_id] for cluster_id in outer.train_cluster_ids]
        )
        mean, scale, target_mean, coefficient = _fit(
            features[outer_train_indices], targets[outer_train_indices], selected_alpha
        )
        model = _ridge_model_from_fit(
            mean=mean,
            scale=scale,
            target_mean=target_mean,
            coefficient=coefficient,
            alpha=selected_alpha,
            include_action=include_action,
            window_steps=next(iter(windows)),
            horizon_steps=horizon_steps,
        )
        outer_results.append(
            NestedRidgeOuterResult(
                outer_fold_index=outer.index,
                selected_alpha=selected_alpha,
                alpha_validation_errors=tuple(alpha_errors),
                train_cluster_ids=outer.train_cluster_ids,
                test_cluster_ids=outer.test_cluster_ids,
                model_training_cluster_ids=outer.train_cluster_ids,
                model=model,
            )
        )
    return NestedRidgeResult(
        fold_plan=fold_plan,
        outer_results=tuple(outer_results),
    )


def fit_action_aware_and_blinded(
    samples: Sequence[RidgeSample], *, horizon_steps: int
) -> tuple[DirectRidgeModel, DirectRidgeModel]:
    """Fit paired diagnostics from identical samples; only action fields differ."""
    return (
        fit_direct_ridge(samples, horizon_steps=horizon_steps, include_action=True),
        fit_direct_ridge(samples, horizon_steps=horizon_steps, include_action=False),
    )
