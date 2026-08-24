"""Calibrated action-conditioned safety-risk adviser for Issue #56.

The module owns offline risk labels, a small normalized ridge estimator, and a
fail-closed advisory adapter.  It deliberately has no plant or capability
authority.  HMC remains the only component allowed to arbitrate, step, and
replay the plant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from .control_trace import parse_control_trace, replay_control_trace
from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast.projection import (
    ForecastHistory,
    project_history_window,
    project_proposed_action,
)
from .forecast_issue55_race import (
    EPISODE_STEPS as ISSUE55_EPISODE_STEPS,
    HMC_IMPLEMENTATION_GIT_SHA,
    TARGET_COUNT as ISSUE55_TARGET_COUNT,
    compute_race_metrics,
    episode_nonce,
    project_true_targets,
    scenario_zone_order,
    target_bounds,
)
from .hmc import HabitatManagementComputer
from .physics import (
    InfeasibleActionError,
    ScenarioValidationError,
    advance_one_step_with_command,
    initial_state,
    validate_external_command,
)
from .scenario import Scenario
from .state import PlantState


ISSUE56_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v1"
PREREGISTRATION_ID = "habitat_v2_forecast_issue_56_preregistration_v1"
RISK_METRIC_ID = "issue56-calibrated-risk-first-ranking-v1"
MODEL_SOURCE_TYPE = "issue56-action-risk-advisory-v1"
CORPUS_ID = "issue56_action_risk_v1"
FAMILY_COUNT = 32
EPISODE_STEPS = ISSUE55_EPISODE_STEPS
HISTORY_WINDOW_STEPS = 16
RISK_HORIZON_STEPS = 32
DECISION_START_STEP = 16
DECISION_CADENCE_STEPS = 4
DECISION_END_STEP = 64
TARGET_COUNT = ISSUE55_TARGET_COUNT
ACTION_COUNT = 27
HISTORY_FEATURE_COUNT = 194
FEATURE_COUNT = HISTORY_FEATURE_COUNT * 3 + ACTION_COUNT + 1
RISK_OUTPUT_COUNT = 3
RIDGE_ALPHA = 0.1
CALIBRATION_QUANTILE = 0.90
HARD_EXPOSURE_LIMIT = 0.5
HARD_CROSSING_PROBABILITY_LIMIT = 0.5
INFERENCE_DEADLINE_MS = 250.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 560056


class Issue56RiskError(ValueError):
    """Raised when Issue #56 risk evidence or advisory data is malformed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def _sha(value: Any) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(_jsonable(value))).hexdigest()
    except (TypeError, ValueError) as error:
        raise Issue56RiskError("digest input is not finite canonical JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise Issue56RiskError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _readonly_f32(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise Issue56RiskError(f"{label} has an invalid shape or non-finite value")
    result = result.copy()
    result.setflags(write=False)
    return result


def family_split(family_ids: Sequence[str]) -> dict[str, str]:
    """Assign complete families using the frozen 60/20/20 hash split."""

    ids = tuple(family_ids)
    if not ids or len(set(ids)) != len(ids):
        raise Issue56RiskError("family roster must be non-empty and unique")
    ordered = sorted(
        ids,
        key=lambda family_id: hashlib.sha256(
            f"issue56-split-v1|{family_id}".encode("utf-8")
        ).digest(),
    )
    proportions = (0.60, 0.20, 0.20)
    labels = ("TRAIN", "VALIDATION", "EVALUATION")
    counts = [int(len(ordered) * proportion) for proportion in proportions]
    remaining = len(ordered) - sum(counts)
    order = sorted(
        range(len(labels)),
        key=lambda index: (-(len(ordered) * proportions[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    result: dict[str, str] = {}
    cursor = 0
    for label, count in zip(labels, counts, strict=True):
        for family_id in ordered[cursor : cursor + count]:
            result[family_id] = label
        cursor += count
    return dict(sorted(result.items()))


def risk_decision_steps(episode_steps: int = EPISODE_STEPS) -> tuple[int, ...]:
    """Return decisions with a complete 32-transition counterfactual horizon."""

    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, int)
        or episode_steps < DECISION_START_STEP + RISK_HORIZON_STEPS
    ):
        raise Issue56RiskError("episode is too short for the risk horizon")
    return tuple(
        step
        for step in range(DECISION_START_STEP, min(DECISION_END_STEP, episode_steps - RISK_HORIZON_STEPS + 1) + 1, DECISION_CADENCE_STEPS)
    )


def _crossings(values: np.ndarray) -> np.ndarray:
    scales, _, lowers, uppers = target_bounds()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != TARGET_COUNT or not np.isfinite(array).all():
        raise Issue56RiskError("risk target rows are malformed")
    return (
        np.maximum(0.0, lowers[None, :] - array) / scales[None, :]
        + np.maximum(0.0, array - uppers[None, :]) / scales[None, :]
    )


@dataclass(frozen=True, slots=True)
class RiskLabel:
    """Truth-backed label from one complete counterfactual action rollout."""

    action_id: str
    decision_step: int
    targets: np.ndarray
    state_digests: tuple[str, ...]
    eligible: bool
    termination_reason: str | None
    crossing_event: float
    safety_exposure: float
    maximum_crossing: float
    label_sha256: str

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise Issue56RiskError("risk label action identity is invalid")
        if isinstance(self.decision_step, bool) or not isinstance(self.decision_step, int):
            raise Issue56RiskError("risk label decision step is invalid")
        targets = np.asarray(self.targets, dtype=np.float32)
        if targets.shape != (RISK_HORIZON_STEPS, TARGET_COUNT):
            raise Issue56RiskError("risk label target shape is invalid")
        if not np.isfinite(targets).all():
            raise Issue56RiskError("risk label targets are non-finite")
        if self.eligible:
            if len(self.state_digests) != RISK_HORIZON_STEPS:
                raise Issue56RiskError("eligible risk label is incomplete")
            if self.termination_reason is not None:
                raise Issue56RiskError("eligible risk label has a termination reason")
        elif self.termination_reason is None:
            raise Issue56RiskError("ineligible risk label has no termination reason")
        for value, label in (
            (self.crossing_event, "crossing event"),
            (self.safety_exposure, "safety exposure"),
            (self.maximum_crossing, "maximum crossing"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56RiskError(f"{label} is non-finite or negative")
        if float(self.crossing_event) not in {0.0, 1.0}:
            raise Issue56RiskError("crossing event must be binary")
        _require_sha(self.label_sha256, "risk label identity")
        body = {
            "schema_version": f"{ISSUE56_SCHEMA_VERSION}.label",
            "action_id": self.action_id,
            "decision_step": self.decision_step,
            "targets": targets.tolist(),
            "state_digests": list(self.state_digests),
            "eligible": self.eligible,
            "termination_reason": self.termination_reason,
            "crossing_event": float(self.crossing_event),
            "safety_exposure": float(self.safety_exposure),
            "maximum_crossing": float(self.maximum_crossing),
        }
        if self.label_sha256 != _sha(body):
            raise Issue56RiskError("risk label digest is inconsistent")
        targets = targets.copy()
        targets.setflags(write=False)
        object.__setattr__(self, "targets", targets)


def _state_digest(state: PlantState) -> str:
    if type(state) is not PlantState:
        raise Issue56RiskError("risk state digest requires PlantState")
    return _sha(_jsonable(state))


def counterfactual_risk_label(
    scenario: Any,
    zone_ids: Sequence[str],
    state: PlantState,
    action_id: str,
    command: Mapping[str, Any],
    *,
    horizon_steps: int = RISK_HORIZON_STEPS,
) -> RiskLabel:
    """Roll one frozen command through the plant without HMC authority."""

    if horizon_steps != RISK_HORIZON_STEPS:
        raise Issue56RiskError("risk labels require exactly 32 transitions")
    if type(state) is not PlantState or type(action_id) is not str or not action_id:
        raise Issue56RiskError("counterfactual risk inputs are invalid")
    try:
        canonical = validate_external_command(scenario, command)
    except Exception as error:
        raise Issue56RiskError("counterfactual command is invalid") from error
    targets = np.full((RISK_HORIZON_STEPS, TARGET_COUNT), np.nan, dtype=np.float32)
    state_digests: list[str] = []
    cursor = state
    termination_reason: str | None = None
    for offset in range(RISK_HORIZON_STEPS):
        try:
            stepped = advance_one_step_with_command(
                scenario, cursor, canonical.to_mapping()
            )
        except (InfeasibleActionError, ScenarioValidationError) as error:
            termination_reason = type(error).__name__
            break
        targets[offset] = project_true_targets(scenario, zone_ids, stepped.state)
        state_digests.append(_state_digest(stepped.state))
        cursor = stepped.state
    if termination_reason is not None:
        targets.fill(0.0)
        event = exposure = maximum = 0.0
        eligible = False
    else:
        crossing = _crossings(targets)
        event = float(np.any(crossing > 0.0))
        exposure = float(np.sum(crossing))
        maximum = float(np.max(crossing))
        eligible = True
    body = {
        "schema_version": f"{ISSUE56_SCHEMA_VERSION}.label",
        "action_id": action_id,
        "decision_step": state.step,
        "targets": targets.tolist(),
        "state_digests": state_digests,
        "eligible": eligible,
        "termination_reason": termination_reason,
        "crossing_event": event,
        "safety_exposure": exposure,
        "maximum_crossing": maximum,
    }
    return RiskLabel(
        action_id=action_id,
        decision_step=state.step,
        targets=targets,
        state_digests=tuple(state_digests),
        eligible=eligible,
        termination_reason=termination_reason,
        crossing_event=event,
        safety_exposure=exposure,
        maximum_crossing=maximum,
        label_sha256=_sha(body),
    )


def _history_available(history: ForecastHistory) -> bool:
    if type(history) is not ForecastHistory:
        return False
    if history.numeric_f32.shape != (HISTORY_WINDOW_STEPS, HISTORY_FEATURE_COUNT):
        return False
    if not np.isfinite(history.numeric_f32).all():
        return False
    if history.status_f32.ndim != 3 or history.status_f32.shape[0] != HISTORY_WINDOW_STEPS:
        return False
    return bool(np.all(history.status_f32[:, :, 0] >= 1.0))


def feature_vector(
    history: ForecastHistory, action_f32: np.ndarray
) -> np.ndarray:
    """Construct the frozen 610-value risk feature vector."""

    if not _history_available(history):
        raise Issue56RiskError("risk features require complete finite history")
    action = np.asarray(action_f32, dtype=np.float32)
    if action.shape != (ACTION_COUNT,) or not np.isfinite(action).all():
        raise Issue56RiskError("risk action vector is malformed")
    numeric = np.asarray(history.numeric_f32, dtype=np.float64)
    values = np.concatenate(
        (
            numeric[-1],
            np.mean(numeric, axis=0),
            numeric[-1] - numeric[0],
            action.astype(np.float64),
            np.ones(1, dtype=np.float64),
        )
    ).astype(np.float32)
    return _readonly_f32(values, (FEATURE_COUNT,), "risk feature vector")


def _risk_targets(samples: Sequence["ActionRiskSample"]) -> np.ndarray:
    return np.asarray(
        [
            [
                math.log1p(sample.safety_exposure),
                sample.crossing_event,
                math.log1p(sample.maximum_crossing),
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )


def _sample_digest(
    family_id: str,
    decision_step: int,
    split: str,
    action_id: str,
    scenario_sha256: str,
    history: ForecastHistory,
    action_f32: np.ndarray,
    label: RiskLabel,
    snapshot_sha256: Sequence[str],
) -> str:
    return _sha(
        _sample_payload(
            family_id,
            decision_step,
            split,
            action_id,
            scenario_sha256,
            history.steps,
            snapshot_sha256,
            history.layout.input_manifest_sha256,
            history.layout.target_manifest_sha256,
            history.numeric_f32,
            action_f32,
            label,
        )
    )


def _sample_payload(
    family_id: str,
    decision_step: int,
    split: str,
    action_id: str,
    scenario_sha256: str,
    history_steps: Sequence[int],
    snapshot_sha256: Sequence[str],
    input_manifest_sha256: str,
    target_manifest_sha256: str,
    history_numeric_f32: np.ndarray,
    action_f32: np.ndarray,
    label: RiskLabel | None = None,
    *,
    crossing_event: float | None = None,
    safety_exposure: float | None = None,
    maximum_crossing: float | None = None,
    label_sha256: str | None = None,
) -> dict[str, Any]:
    if label is not None:
        crossing_event = label.crossing_event
        safety_exposure = label.safety_exposure
        maximum_crossing = label.maximum_crossing
        label_sha256 = label.label_sha256
    if crossing_event is None or safety_exposure is None or maximum_crossing is None or label_sha256 is None:
        raise Issue56RiskError("risk sample payload is missing label values")
    return {
        "schema_version": f"{ISSUE56_SCHEMA_VERSION}.sample",
        "family_id": family_id,
        "decision_step": decision_step,
        "split": split,
        "action_id": action_id,
        "scenario_sha256": scenario_sha256,
        "history_steps": list(history_steps),
        "snapshot_sha256": list(snapshot_sha256),
        "input_manifest_sha256": input_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "history_numeric": np.asarray(history_numeric_f32, dtype=np.float32).tobytes().hex(),
        "action": np.asarray(action_f32, dtype=np.float32).tobytes().hex(),
        "crossing_event": float(crossing_event),
        "safety_exposure": float(safety_exposure),
        "maximum_crossing": float(maximum_crossing),
        "label_sha256": label_sha256,
    }


@dataclass(frozen=True, slots=True)
class ActionRiskSample:
    """One action-conditioned history/true-risk sample with provenance."""

    family_id: str
    decision_step: int
    split: str
    action_id: str
    scenario_sha256: str
    history_steps: tuple[int, ...]
    snapshot_sha256: tuple[str, ...]
    input_manifest_sha256: str
    target_manifest_sha256: str
    history_numeric_f32: np.ndarray
    action_f32: np.ndarray
    crossing_event: float
    safety_exposure: float
    maximum_crossing: float
    label_sha256: str
    sample_sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise Issue56RiskError("risk sample family identity is invalid")
        if self.split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
            raise Issue56RiskError("risk sample split is invalid")
        if type(self.action_id) is not str or not self.action_id:
            raise Issue56RiskError("risk sample action identity is invalid")
        _require_sha(self.scenario_sha256, "risk sample scenario identity")
        _require_sha(self.input_manifest_sha256, "risk sample input manifest")
        _require_sha(self.target_manifest_sha256, "risk sample target manifest")
        _require_sha(self.label_sha256, "risk sample label identity")
        _require_sha(self.sample_sha256, "risk sample identity")
        history = np.asarray(self.history_numeric_f32, dtype=np.float32)
        action = np.asarray(self.action_f32, dtype=np.float32)
        if history.shape != (HISTORY_WINDOW_STEPS, HISTORY_FEATURE_COUNT):
            raise Issue56RiskError("risk sample history shape is invalid")
        if action.shape != (ACTION_COUNT,):
            raise Issue56RiskError("risk sample action shape is invalid")
        if not np.isfinite(history).all() or not np.isfinite(action).all():
            raise Issue56RiskError("risk sample features are non-finite")
        if len(self.history_steps) != HISTORY_WINDOW_STEPS or len(self.snapshot_sha256) != HISTORY_WINDOW_STEPS:
            raise Issue56RiskError("risk sample history provenance is incomplete")
        for snapshot in self.snapshot_sha256:
            _require_sha(snapshot, "risk sample snapshot identity")
        for value, label in (
            (self.crossing_event, "risk sample crossing event"),
            (self.safety_exposure, "risk sample exposure"),
            (self.maximum_crossing, "risk sample maximum crossing"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56RiskError(f"{label} is invalid")
        if float(self.crossing_event) not in {0.0, 1.0}:
            raise Issue56RiskError("risk sample crossing event is not binary")
        body = _sample_payload(
            self.family_id,
            self.decision_step,
            self.split,
            self.action_id,
            self.scenario_sha256,
            self.history_steps,
            self.snapshot_sha256,
            self.input_manifest_sha256,
            self.target_manifest_sha256,
            history,
            action,
            crossing_event=self.crossing_event,
            safety_exposure=self.safety_exposure,
            maximum_crossing=self.maximum_crossing,
            label_sha256=self.label_sha256,
        )
        if self.sample_sha256 != _sha(body):
            raise Issue56RiskError("risk sample digest is inconsistent")
        history = history.copy()
        action = action.copy()
        history.setflags(write=False)
        action.setflags(write=False)
        object.__setattr__(self, "history_numeric_f32", history)
        object.__setattr__(self, "action_f32", action)

    def features(self) -> np.ndarray:
        history = np.asarray(self.history_numeric_f32, dtype=np.float64)
        return _readonly_f32(
            np.concatenate(
                (
                    history[-1],
                    np.mean(history, axis=0),
                    history[-1] - history[0],
                    np.asarray(self.action_f32, dtype=np.float64),
                    np.ones(1, dtype=np.float64),
                )
            ),
            (FEATURE_COUNT,),
            "stored risk features",
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_SCHEMA_VERSION}.sample",
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "split": self.split,
            "action_id": self.action_id,
            "scenario_sha256": self.scenario_sha256,
            "history_steps": list(self.history_steps),
            "snapshot_sha256": list(self.snapshot_sha256),
            "input_manifest_sha256": self.input_manifest_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "history_numeric_f32_hex": self.history_numeric_f32.tobytes().hex(),
            "action_f32_hex": self.action_f32.tobytes().hex(),
            "crossing_event": self.crossing_event,
            "safety_exposure": self.safety_exposure,
            "maximum_crossing": self.maximum_crossing,
            "label_sha256": self.label_sha256,
            "sample_sha256": self.sample_sha256,
        }


def make_action_risk_sample(
    family_id: str,
    split: str,
    action_id: str,
    scenario_sha256: str,
    history: ForecastHistory,
    action_f32: np.ndarray,
    label: RiskLabel,
    *,
    snapshot_sha256: Sequence[str],
) -> ActionRiskSample:
    if not label.eligible:
        raise Issue56RiskError("risk samples require complete labels")
    if label.action_id != action_id or label.decision_step != history.steps[-1]:
        raise Issue56RiskError("risk sample action/decision binding is inconsistent")
    snapshot_ids = tuple(snapshot_sha256)
    if len(snapshot_ids) != HISTORY_WINDOW_STEPS:
        raise Issue56RiskError("risk sample requires 16 snapshot identities")
    action = _readonly_f32(action_f32, (ACTION_COUNT,), "risk sample action")
    sample_sha = _sample_digest(
        family_id,
        label.decision_step,
        split,
        action_id,
        scenario_sha256,
        history,
        action,
        label,
        snapshot_ids,
    )
    return ActionRiskSample(
        family_id=family_id,
        decision_step=label.decision_step,
        split=split,
        action_id=action_id,
        scenario_sha256=scenario_sha256,
        history_steps=tuple(history.steps),
        snapshot_sha256=snapshot_ids,
        input_manifest_sha256=history.layout.input_manifest_sha256,
        target_manifest_sha256=history.layout.target_manifest_sha256,
        history_numeric_f32=history.numeric_f32,
        action_f32=action,
        crossing_event=label.crossing_event,
        safety_exposure=label.safety_exposure,
        maximum_crossing=label.maximum_crossing,
        label_sha256=label.label_sha256,
        sample_sha256=sample_sha,
    )


@dataclass(frozen=True, slots=True)
class RiskPrediction:
    predicted_exposure: float
    crossing_probability: float
    predicted_maximum_crossing: float
    upper_exposure: float
    upper_maximum_crossing: float
    hard_ineligible: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ActionRiskModel:
    """Frozen normalized ridge model with validation-only residual calibration."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficients: np.ndarray
    exposure_residual_p90: float = 0.0
    maximum_residual_p90: float = 0.0
    crossing_residual_p90: float = 0.0
    alpha: float = RIDGE_ALPHA
    seed: int = 560056
    model_id: str = "issue56-action-risk-unfitted"
    actuator_authority: bool = False

    def __post_init__(self) -> None:
        arrays = (
            ("feature_mean", self.feature_mean, (FEATURE_COUNT,)),
            ("feature_scale", self.feature_scale, (FEATURE_COUNT,)),
            ("target_mean", self.target_mean, (RISK_OUTPUT_COUNT,)),
            ("target_scale", self.target_scale, (RISK_OUTPUT_COUNT,)),
            ("coefficients", self.coefficients, (FEATURE_COUNT, RISK_OUTPUT_COUNT)),
        )
        for label, value, shape in arrays:
            result = np.asarray(value, dtype=np.float64)
            if result.shape != shape or not np.isfinite(result).all():
                raise Issue56RiskError(f"risk model {label} is malformed")
            readonly = result.copy()
            readonly.setflags(write=False)
            object.__setattr__(self, label, readonly)
        if np.any(self.feature_scale <= 0.0) or np.any(self.target_scale <= 0.0):
            raise Issue56RiskError("risk model normalizers must be positive")
        for value, label in (
            (self.exposure_residual_p90, "exposure calibration"),
            (self.maximum_residual_p90, "maximum calibration"),
            (self.crossing_residual_p90, "crossing calibration"),
            (self.alpha, "ridge alpha"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56RiskError(f"{label} is invalid")
        if type(self.model_id) is not str or not self.model_id:
            raise Issue56RiskError("risk model identity is invalid")
        if self.actuator_authority is not False:
            raise Issue56RiskError("risk model cannot claim actuator authority")

    @classmethod
    def fit(
        cls,
        samples: Sequence[ActionRiskSample],
        *,
        alpha: float = RIDGE_ALPHA,
        seed: int = 560056,
    ) -> "ActionRiskModel":
        items = tuple(samples)
        if len(items) < 3 or len({item.family_id for item in items}) < 2:
            raise Issue56RiskError("risk fitting requires three samples from two families")
        if any(item.split != "TRAIN" for item in items):
            raise Issue56RiskError("risk fitting accepts TRAIN samples only")
        if not math.isfinite(float(alpha)) or alpha <= 0.0:
            raise Issue56RiskError("risk ridge alpha must be positive and finite")
        features = np.stack([item.features() for item in items]).astype(np.float64)
        targets = _risk_targets(items)
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        feature_mean[-1] = 0.0
        feature_scale[-1] = 1.0
        target_mean = np.mean(targets, axis=0)
        target_scale = np.std(targets, axis=0)
        target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
        normalized_features = (features - feature_mean) / feature_scale
        normalized_targets = (targets - target_mean) / target_scale
        gram = normalized_features.T @ normalized_features
        regularizer = np.eye(FEATURE_COUNT, dtype=np.float64) * float(alpha)
        regularizer[-1, -1] = 0.0
        try:
            coefficients = np.linalg.solve(
                gram + regularizer, normalized_features.T @ normalized_targets
            )
        except np.linalg.LinAlgError as error:
            raise Issue56RiskError("risk fit is singular") from error
        model_id = "issue56-action-risk-" + _sha(
            {
                "feature_mean": feature_mean.tolist(),
                "feature_scale": feature_scale.tolist(),
                "target_mean": target_mean.tolist(),
                "target_scale": target_scale.tolist(),
                "coefficients": coefficients.tolist(),
                "alpha": alpha,
                "seed": seed,
            }
        )[:16]
        return cls(
            feature_mean,
            feature_scale,
            target_mean,
            target_scale,
            coefficients,
            alpha=alpha,
            seed=seed,
            model_id=model_id,
        )

    def _predict_raw_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
            raise Issue56RiskError("risk inference features are malformed")
        normalized = (values - self.feature_mean) / self.feature_scale
        result = normalized @ self.coefficients
        result = result * self.target_scale + self.target_mean
        if not np.isfinite(result).all():
            raise Issue56RiskError("risk inference output is non-finite")
        return result

    def predict(self, history: ForecastHistory, action_f32: np.ndarray) -> RiskPrediction:
        raw = self._predict_raw_features(feature_vector(history, action_f32))
        exposure_log = max(0.0, float(raw[0]))
        maximum_log = max(0.0, float(raw[2]))
        exposure = math.expm1(exposure_log)
        maximum = math.expm1(maximum_log)
        probability = min(1.0, max(0.0, float(raw[1]) + self.crossing_residual_p90))
        upper_exposure = math.expm1(exposure_log + self.exposure_residual_p90)
        upper_maximum = math.expm1(maximum_log + self.maximum_residual_p90)
        hard = (
            upper_exposure > HARD_EXPOSURE_LIMIT
            or probability >= HARD_CROSSING_PROBABILITY_LIMIT
        )
        return RiskPrediction(
            predicted_exposure=exposure,
            crossing_probability=probability,
            predicted_maximum_crossing=maximum,
            upper_exposure=upper_exposure,
            upper_maximum_crossing=upper_maximum,
            hard_ineligible=hard,
            reason="calibrated_safety_risk_limit" if hard else None,
        )

    def calibrate(
        self, samples: Sequence[ActionRiskSample], *, quantile: float = CALIBRATION_QUANTILE
    ) -> "ActionRiskModel":
        items = tuple(samples)
        if not items or any(item.split != "VALIDATION" for item in items):
            raise Issue56RiskError("calibration accepts VALIDATION samples only")
        if not 0.0 < float(quantile) < 1.0:
            raise Issue56RiskError("calibration quantile must be between zero and one")
        raw = np.stack([self._predict_raw_features(item.features()) for item in items])
        truth = _risk_targets(items)
        residual = np.abs(truth - raw)
        calibrated = replace(
            self,
            exposure_residual_p90=float(np.quantile(residual[:, 0], quantile)),
            crossing_residual_p90=float(np.quantile(residual[:, 1], quantile)),
            maximum_residual_p90=float(np.quantile(residual[:, 2], quantile)),
            model_id=self.model_id + "-calibrated",
        )
        return calibrated

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "schema_version": f"{ISSUE56_SCHEMA_VERSION}.model",
            "model_id": self.model_id,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "exposure_residual_p90": self.exposure_residual_p90,
            "maximum_residual_p90": self.maximum_residual_p90,
            "crossing_residual_p90": self.crossing_residual_p90,
            "alpha": self.alpha,
            "seed": self.seed,
            "actuator_authority": False,
        }
        return {**body, "model_sha256": _sha(body)}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ActionRiskModel":
        if type(mapping) is not dict:
            raise Issue56RiskError("risk model artifact must be one object")
        expected = {
            "schema_version",
            "model_id",
            "feature_mean",
            "feature_scale",
            "target_mean",
            "target_scale",
            "coefficients",
            "exposure_residual_p90",
            "maximum_residual_p90",
            "crossing_residual_p90",
            "alpha",
            "seed",
            "actuator_authority",
            "model_sha256",
        }
        if set(mapping) != expected or mapping["schema_version"] != f"{ISSUE56_SCHEMA_VERSION}.model":
            raise Issue56RiskError("risk model artifact schema drift")
        body = dict(mapping)
        digest = body.pop("model_sha256")
        if digest != _sha(body):
            raise Issue56RiskError("risk model artifact digest is inconsistent")
        return cls(
            np.asarray(mapping["feature_mean"], dtype=np.float64),
            np.asarray(mapping["feature_scale"], dtype=np.float64),
            np.asarray(mapping["target_mean"], dtype=np.float64),
            np.asarray(mapping["target_scale"], dtype=np.float64),
            np.asarray(mapping["coefficients"], dtype=np.float64),
            exposure_residual_p90=float(mapping["exposure_residual_p90"]),
            maximum_residual_p90=float(mapping["maximum_residual_p90"]),
            crossing_residual_p90=float(mapping["crossing_residual_p90"]),
            alpha=float(mapping["alpha"]),
            seed=int(mapping["seed"]),
            model_id=str(mapping["model_id"]),
            actuator_authority=bool(mapping["actuator_authority"]),
        )


@dataclass(frozen=True, slots=True)
class RiskScore:
    action_id: str
    score: float
    hard_ineligible: bool
    upper_exposure: float
    crossing_probability: float
    upper_maximum_crossing: float
    intervention: float
    reason: str | None = None


def score_action_risk(
    model: ActionRiskModel,
    history: ForecastHistory,
    action_id: str,
    action_f32: np.ndarray,
) -> RiskScore:
    if type(model) is not ActionRiskModel:
        raise Issue56RiskError("risk scoring requires ActionRiskModel")
    if type(action_id) is not str or not action_id:
        raise Issue56RiskError("risk score action identity is invalid")
    action = _readonly_f32(action_f32, (ACTION_COUNT,), "risk score action")
    prediction = model.predict(history, action)
    current = np.asarray(history.numeric_f32[-1, -ACTION_COUNT:], dtype=np.float64)
    intervention = float(np.mean(np.abs(current - action.astype(np.float64))))
    score = prediction.upper_exposure + 0.25 * prediction.crossing_probability + 0.01 * intervention
    return RiskScore(
        action_id=action_id,
        score=score if not prediction.hard_ineligible else math.inf,
        hard_ineligible=prediction.hard_ineligible,
        upper_exposure=prediction.upper_exposure,
        crossing_probability=prediction.crossing_probability,
        upper_maximum_crossing=prediction.upper_maximum_crossing,
        intervention=intervention,
        reason=prediction.reason,
    )


def rank_action_risk(scores: Sequence[RiskScore]) -> RiskScore | None:
    if not scores:
        raise Issue56RiskError("risk ranking requires at least one candidate")
    ids = [item.action_id for item in scores]
    if len(ids) != len(set(ids)):
        raise Issue56RiskError("risk ranking received duplicate action IDs")
    eligible = [item for item in scores if not item.hard_ineligible and math.isfinite(item.score)]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item.score,
            item.upper_exposure,
            item.crossing_probability,
            item.intervention,
            0 if "hold" in item.action_id else 1,
            item.action_id,
        ),
    )


def build_risk_proposal(
    hmc: HabitatManagementComputer,
    snapshot_sha256: str,
    step: int,
    command: Mapping[str, Any],
    action_id: str,
) -> dict[str, Any]:
    if type(hmc) is not HabitatManagementComputer:
        raise Issue56RiskError("risk proposal requires the issuing HMC")
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": action_id,
        "source_type": MODEL_SOURCE_TYPE,
        "completed_observation_step": step,
        "observation_snapshot_sha256": snapshot_sha256,
        "requested_application_step": step,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": json.loads(json.dumps(dict(command), allow_nan=False)),
        "confidence": None,
    }
    return {**body, "proposal_sha256": _sha(body)}


def _history_at(
    bundle: ForecastContracts,
    snapshots: Mapping[int, tuple[Any, Any]],
    step: int,
) -> ForecastHistory:
    pairs = [
        snapshots[index]
        for index in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
    ]
    return project_history_window(bundle, pairs, window_steps=HISTORY_WINDOW_STEPS)


def collect_family_samples(
    bundle: ForecastContracts,
    scenario: Any,
    family_id: str,
    *,
    split: str,
) -> tuple[ActionRiskSample, ...]:
    """Collect all four complete counterfactual labels for one family."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise Issue56RiskError("risk collection requires the frozen forecast bundle and scenario")
    if split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
        raise Issue56RiskError("risk collection split is invalid")
    zone_ids = scenario_zone_order(scenario)
    actions = tuple(bundle.actions)
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce(family_id),
    )
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    samples: list[ActionRiskSample] = []
    for step in range(int(scenario.data["steps"])):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56RiskError("HMC terminated during risk collection")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        if step > 0:
            snapshots[step] = (snapshot, verification)
        if step in risk_decision_steps(int(scenario.data["steps"])):
            history = _history_at(bundle, snapshots, step)
            for action in actions:
                action_vector = project_proposed_action(bundle, action.command)
                label = counterfactual_risk_label(
                    scenario,
                    zone_ids,
                    shadow,
                    action.action_id,
                    action.command.to_mapping(),
                )
                if not label.eligible:
                    raise Issue56RiskError(
                        f"risk collection found incomplete label for {family_id}/{action.action_id}"
                    )
                samples.append(
                    make_action_risk_sample(
                        family_id,
                        split,
                        action.action_id,
                        scenario.scenario_sha256,
                        history,
                        action_vector,
                        label,
                        snapshot_sha256=tuple(
                            snapshots[index][0].snapshot_sha256
                            for index in range(
                                step - HISTORY_WINDOW_STEPS + 1, step + 1
                            )
                        ),
                    )
                )
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        stepped = hmc.step()
        if not hasattr(arbitration, "final_command") or not hasattr(
            stepped, "plant_receipt_digest"
        ):
            raise Issue56RiskError("HMC failed during risk collection")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        if _sha(shadow_result.receipt) != stepped.plant_receipt_digest:
            raise Issue56RiskError("risk collection shadow replay diverged")
        shadow = shadow_result.state
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract
    )
    replay = replay_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != int(scenario.data["steps"])
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56RiskError("risk collection trace did not replay to closure")
    return tuple(samples)


@dataclass(frozen=True, slots=True)
class RiskEpisodeRecord:
    family_id: str
    scenario_sha256: str
    decision_steps: tuple[int, ...]
    decision_actions: tuple[str | None, ...]
    proposal_count: int
    abstention_count: int
    hmc_rejection_count: int
    safety_exposure: float
    safety_violation_steps: int
    comfort_deviation: float
    resource_composite: float
    trace_sha256: str
    replay_committed_steps: int
    episode_sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise Issue56RiskError("risk episode family identity is invalid")
        _require_sha(self.scenario_sha256, "risk episode scenario identity")
        _require_sha(self.trace_sha256, "risk episode trace identity")
        _require_sha(self.episode_sha256, "risk episode identity")
        expected_steps = risk_decision_steps(EPISODE_STEPS)
        if self.decision_steps != expected_steps:
            raise Issue56RiskError("risk episode decision steps drifted")
        if type(self.decision_actions) is not tuple or len(self.decision_actions) != len(expected_steps):
            raise Issue56RiskError("risk episode decision actions drifted")
        if sum(item is not None for item in self.decision_actions) != self.proposal_count:
            raise Issue56RiskError("risk episode proposal count drifted")
        if self.proposal_count + self.abstention_count != len(expected_steps):
            raise Issue56RiskError("risk episode decisions are not accounted for")
        if self.replay_committed_steps != EPISODE_STEPS:
            raise Issue56RiskError("risk episode replay is incomplete")
        for value, label in (
            (self.safety_exposure, "safety exposure"),
            (self.comfort_deviation, "comfort deviation"),
            (self.resource_composite, "resource composite"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56RiskError(f"risk episode {label} is invalid")
        body = {
            "schema_version": f"{ISSUE56_SCHEMA_VERSION}.episode",
            "family_id": self.family_id,
            "scenario_sha256": self.scenario_sha256,
            "decision_steps": list(self.decision_steps),
            "decision_actions": list(self.decision_actions),
            "proposal_count": self.proposal_count,
            "abstention_count": self.abstention_count,
            "hmc_rejection_count": self.hmc_rejection_count,
            "safety_exposure": self.safety_exposure,
            "safety_violation_steps": self.safety_violation_steps,
            "comfort_deviation": self.comfort_deviation,
            "resource_composite": self.resource_composite,
            "trace_sha256": self.trace_sha256,
            "replay_committed_steps": self.replay_committed_steps,
        }
        if self.episode_sha256 != _sha(body):
            raise Issue56RiskError("risk episode digest is inconsistent")

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "schema_version": f"{ISSUE56_SCHEMA_VERSION}.episode",
            "family_id": self.family_id,
            "scenario_sha256": self.scenario_sha256,
            "decision_steps": list(self.decision_steps),
            "decision_actions": list(self.decision_actions),
            "proposal_count": self.proposal_count,
            "abstention_count": self.abstention_count,
            "hmc_rejection_count": self.hmc_rejection_count,
            "safety_exposure": self.safety_exposure,
            "safety_violation_steps": self.safety_violation_steps,
            "comfort_deviation": self.comfort_deviation,
            "resource_composite": self.resource_composite,
            "trace_sha256": self.trace_sha256,
            "replay_committed_steps": self.replay_committed_steps,
        }
        return {**body, "episode_sha256": self.episode_sha256}


def run_risk_episode(
    bundle: ForecastContracts,
    scenario: Any,
    family_id: str,
    model: ActionRiskModel,
) -> RiskEpisodeRecord:
    """Run one calibrated risk adviser episode through HMC and strict replay."""

    if type(model) is not ActionRiskModel or model.actuator_authority is not False:
        raise Issue56RiskError("risk episode requires a forecast-only model")
    zone_ids = scenario_zone_order(scenario)
    actions = tuple(bundle.actions)
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, episode_nonce(family_id))
    shadow = initial_state(scenario)
    initial_row = project_true_targets(scenario, zone_ids, shadow)
    states: dict[int, PlantState] = {0: shadow}
    snapshots: dict[int, tuple[Any, Any]] = {}
    decision_steps = risk_decision_steps(int(scenario.data["steps"]))
    decision_actions: list[str | None] = []
    proposal_count = 0
    abstention_count = 0
    hmc_rejection_count = 0
    for step in range(int(scenario.data["steps"])):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56RiskError("HMC terminated during risk episode")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        if step > 0:
            snapshots[step] = (snapshot, verification)
        proposal: dict[str, Any] | None = None
        proposed_sha: str | None = None
        if step in decision_steps:
            history = _history_at(bundle, snapshots, step)
            scores = tuple(
                score_action_risk(
                    model,
                    history,
                    action.action_id,
                    project_proposed_action(bundle, action.command),
                )
                for action in actions
            )
            selected = rank_action_risk(scores)
            if selected is None:
                abstention_count += 1
                decision_actions.append(None)
            else:
                selected_action = next(
                    action for action in actions if action.action_id == selected.action_id
                )
                proposal = build_risk_proposal(
                    hmc,
                    snapshot.snapshot_sha256,
                    step,
                    selected_action.command.to_mapping(),
                    selected.action_id,
                )
                proposal_count += 1
                decision_actions.append(selected.action_id)
                proposed_sha = validate_external_command(
                    scenario, proposal["proposed_command"]
                ).sha256
        receipt = hmc.propose(proposal, handle).to_mapping()
        if step in decision_steps:
            if proposal is None:
                if receipt["validation_outcome"] != "NO_PROPOSAL":
                    raise Issue56RiskError("risk abstention was not recorded as no proposal")
            elif (receipt["attempt_class"], receipt["validation_outcome"]) != (
                "CANONICAL_PROPOSAL",
                "VALID",
            ):
                raise Issue56RiskError("risk proposal was not admitted by HMC")
        elif receipt["validation_outcome"] != "NO_PROPOSAL":
            raise Issue56RiskError("risk proposal was issued outside a decision step")
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56RiskError("HMC terminated during risk arbitration")
        if proposed_sha is not None and arbitration.final_command_sha256 != proposed_sha:
            hmc_rejection_count += 1
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56RiskError("HMC terminated during risk step")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        if _sha(shadow_result.receipt) != stepped.plant_receipt_digest:
            raise Issue56RiskError("risk episode shadow replay diverged")
        shadow = shadow_result.state
        states[shadow.step] = shadow
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract)
    replay = replay_control_trace(trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract)
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != int(scenario.data["steps"])
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56RiskError("risk episode trace did not replay to closure")
    metrics = compute_race_metrics(
        scenario,
        zone_ids,
        initial_row,
        [states[index] for index in range(1, int(scenario.data["steps"]) + 1)],
    )
    body = {
        "schema_version": f"{ISSUE56_SCHEMA_VERSION}.episode",
        "family_id": family_id,
        "scenario_sha256": scenario.scenario_sha256,
        "decision_steps": list(decision_steps),
        "decision_actions": decision_actions,
        "proposal_count": proposal_count,
        "abstention_count": abstention_count,
        "hmc_rejection_count": hmc_rejection_count,
        "safety_exposure": float(metrics["safety_exposure"]),
        "safety_violation_steps": int(metrics["safety_violation_steps"]),
        "comfort_deviation": float(metrics["comfort_deviation"]),
        "resource_composite": float(metrics["resource_composite"]),
        "trace_sha256": _sha_bytes(trace.canonical_bytes),
        "replay_committed_steps": int(replay.committed_step_count),
    }
    return RiskEpisodeRecord(
        **{
            **{
                key: value
                for key, value in body.items()
                if key != "schema_version"
            },
            "decision_steps": tuple(decision_steps),
            "decision_actions": tuple(decision_actions),
        },
        episode_sha256=_sha(body),
    )


def calibration_metrics(
    model: ActionRiskModel, samples: Sequence[ActionRiskSample]
) -> dict[str, float | int]:
    """Report held-out risk calibration without changing the frozen model."""

    items = tuple(samples)
    if not items:
        raise Issue56RiskError("calibration metrics require samples")
    upper_hits = 0
    brier = 0.0
    exposure_error = 0.0
    for item in items:
        prediction = model.predict(
            _history_from_numeric_sample(item), item.action_f32
        )
        upper_hits += int(item.safety_exposure <= prediction.upper_exposure)
        brier += (prediction.crossing_probability - item.crossing_event) ** 2
        exposure_error += abs(prediction.predicted_exposure - item.safety_exposure)
    return {
        "sample_count": len(items),
        "upper_exposure_coverage": upper_hits / len(items),
        "crossing_brier": brier / len(items),
        "mean_absolute_exposure_error": exposure_error / len(items),
    }


def _history_from_numeric_sample(sample: ActionRiskSample) -> ForecastHistory:
    """Rebuild the minimal history object needed for offline model evaluation."""

    class _Layout:
        input_manifest_sha256 = sample.input_manifest_sha256
        target_manifest_sha256 = sample.target_manifest_sha256

    history = object.__new__(ForecastHistory)
    object.__setattr__(history, "steps", sample.history_steps)
    object.__setattr__(history, "completed_times_s", tuple(float(step) for step in sample.history_steps))
    object.__setattr__(history, "numeric_f32", sample.history_numeric_f32)
    object.__setattr__(history, "status_f32", np.ones((HISTORY_WINDOW_STEPS, 167, 5), dtype=np.float32))
    object.__setattr__(history, "mode_f32", np.zeros((HISTORY_WINDOW_STEPS, 4), dtype=np.float32))
    object.__setattr__(history, "health_f32", np.zeros((HISTORY_WINDOW_STEPS, 4), dtype=np.float32))
    object.__setattr__(history, "alarm_lifecycle_f32", np.zeros((HISTORY_WINDOW_STEPS, 287, 4), dtype=np.float32))
    object.__setattr__(history, "layout", _Layout())
    return history


__all__ = [
    "ACTION_COUNT",
    "ActionRiskModel",
    "ActionRiskSample",
    "CALIBRATION_QUANTILE",
    "CORPUS_ID",
    "DECISION_CADENCE_STEPS",
    "DECISION_START_STEP",
    "EPISODE_STEPS",
    "FEATURE_COUNT",
    "HARD_CROSSING_PROBABILITY_LIMIT",
    "HARD_EXPOSURE_LIMIT",
    "Issue56RiskError",
    "RiskEpisodeRecord",
    "RiskLabel",
    "RiskPrediction",
    "RiskScore",
    "RISK_HORIZON_STEPS",
    "RISK_METRIC_ID",
    "collect_family_samples",
    "counterfactual_risk_label",
    "feature_vector",
    "family_split",
    "make_action_risk_sample",
    "rank_action_risk",
    "risk_decision_steps",
    "run_risk_episode",
    "score_action_risk",
    "calibration_metrics",
]
