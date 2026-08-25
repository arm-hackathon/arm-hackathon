"""V2 calibrated event/severity safety filter for Issue #56.

This lane corrects the V1 risk semantics without changing the HMC authority
boundary.  It predicts a real crossing probability with a sigmoid event head,
fits severity only on positive events, and uses the result as a safety filter
for the frozen Issue #55 point adviser.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
from typing import Any

import numpy as np

from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast.projection import (
    ForecastHistory,
    project_history_window,
    project_proposed_action,
)
from .forecast_issue55_race import (
    DECISION_CADENCE_STEPS,
    DECISION_START_STEP,
    EPISODE_STEPS,
    HMC_IMPLEMENTATION_GIT_SHA,
    TARGET_COUNT,
    _crossings as true_crossings,
    compute_race_metrics,
    episode_nonce,
    scenario_zone_order,
    project_true_targets,
    target_bounds,
)
from .forecast_issue52 import _command_vector
from .forecast_issue55_race import score_point_prediction
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
from .control_trace import parse_control_trace, replay_control_trace


ISSUE56_V2_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v2_v1"
PREREGISTRATION_ID = "habitat_v2_forecast_issue_56_v2_preregistration_v1"
RISK_METRIC_ID = "issue56-risk-filtered-point-v2"
MODEL_SOURCE_TYPE = "issue56-risk-filtered-point-v2"
RISK_HORIZON_STEPS = 32
HISTORY_WINDOW_STEPS = 16
ACTION_COUNT = 27
BASE_FEATURE_COUNT = 610
MODE_FEATURE_COUNT = 4
HEALTH_FEATURE_COUNT = 4
ALARM_FEATURE_COUNT = 11
MARGIN_FEATURE_COUNT = TARGET_COUNT
POSITION_FEATURE_COUNT = 1
FEATURE_COUNT = (
    BASE_FEATURE_COUNT
    + MODE_FEATURE_COUNT
    + HEALTH_FEATURE_COUNT
    + ALARM_FEATURE_COUNT
    + MARGIN_FEATURE_COUNT
    + ACTION_COUNT
    + POSITION_FEATURE_COUNT
)
EVENT_LIMIT = 0.50
EXPECTED_EXPOSURE_LIMIT = 0.50
MAXIMUM_CROSSING_LIMIT = 0.25
CALIBRATION_QUANTILE = 0.90
EVENT_LOGIT_EPSILON = 1e-3
MAX_LOG_EXPOSURE = 700.0
RIDGE_ALPHA = 0.1
MODEL_SEED = 560057
V2_FAMILY_COUNT = 32
MIN_VALIDATION_DECISION_COVERAGE = 0.10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 560057
MODE_ORDER = ("dormant", "occupied", "eva_transition", "contingency")
HEALTH_ORDER = ("NOMINAL", "DEGRADED", "CRITICAL", "UNKNOWN")
ALARM_FAMILY_ORDER = (
    "actuator_tracking_failure",
    "high_co2",
    "high_humidity",
    "high_temperature",
    "low_battery_gauge",
    "low_oxygen",
    "low_oxygen_store_gauge",
    "low_sorbent_gauge",
    "low_temperature",
    "sensor_disagreement",
    "telemetry_unknown",
)
LABEL_TRACKS = ("effect_4", "persistent_32")


class Issue56V2RiskError(ValueError):
    """Raised when V2 risk evidence or runtime inputs are malformed."""


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
        raise Issue56V2RiskError("digest input is not canonical finite JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise Issue56V2RiskError(f"{label} is malformed or non-finite")
    result = result.copy()
    result.setflags(write=False)
    return result


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise Issue56V2RiskError(f"{label} must be a lowercase SHA-256")
    return str(value)


def v2_family_split(family_ids: Sequence[str]) -> dict[str, str]:
    ids = tuple(family_ids)
    if not ids or len(set(ids)) != len(ids):
        raise Issue56V2RiskError("V2 family roster must be unique and non-empty")
    ordered = sorted(
        ids,
        key=lambda family_id: hashlib.sha256(
            f"issue56-v2-split-v1|{family_id}".encode("utf-8")
        ).digest(),
    )
    labels = ("TRAIN", "VALIDATION", "EVALUATION")
    proportions = (0.60, 0.20, 0.20)
    counts = [int(len(ordered) * proportion) for proportion in proportions]
    for index in sorted(
        range(3),
        key=lambda item: (-(len(ordered) * proportions[item] - counts[item]), item),
    )[: len(ordered) - sum(counts)]:
        counts[index] += 1
    result: dict[str, str] = {}
    cursor = 0
    for label, count in zip(labels, counts, strict=True):
        for family_id in ordered[cursor : cursor + count]:
            result[family_id] = label
        cursor += count
    return dict(sorted(result.items()))


def v2_decision_steps(episode_steps: int = EPISODE_STEPS) -> tuple[int, ...]:
    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, int)
        or episode_steps < DECISION_START_STEP + RISK_HORIZON_STEPS
    ):
        raise Issue56V2RiskError("episode is too short for V2 risk labels")
    return tuple(
        range(
            DECISION_START_STEP,
            min(EPISODE_STEPS - RISK_HORIZON_STEPS, episode_steps - RISK_HORIZON_STEPS)
            + 1,
            DECISION_CADENCE_STEPS,
        )
    )


def _command_schedule(
    current_command: Mapping[str, Any],
    candidate_command: Mapping[str, Any],
    track: str,
) -> tuple[Mapping[str, Any], ...]:
    if track not in LABEL_TRACKS:
        raise Issue56V2RiskError("unknown V2 label track")
    if track == "persistent_32":
        return tuple(candidate_command for _ in range(RISK_HORIZON_STEPS))
    return tuple(
        candidate_command if step < DECISION_CADENCE_STEPS else current_command
        for step in range(RISK_HORIZON_STEPS)
    )


def _risk_label_values(targets: np.ndarray) -> tuple[float, float, float]:
    crossing = true_crossings(targets.astype(np.float64))
    return (
        float(np.any(crossing > 0.0)),
        float(np.sum(crossing)),
        float(np.max(crossing)),
    )


@dataclass(frozen=True, slots=True)
class V2RiskLabel:
    action_id: str
    decision_step: int
    track: str
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
            raise Issue56V2RiskError("V2 label action identity is invalid")
        if self.track not in LABEL_TRACKS:
            raise Issue56V2RiskError("V2 label track is invalid")
        if (
            isinstance(self.decision_step, bool)
            or not isinstance(self.decision_step, int)
            or self.decision_step < 0
        ):
            raise Issue56V2RiskError("V2 label decision step is invalid")
        targets = np.asarray(self.targets, dtype=np.float32)
        if targets.shape != (RISK_HORIZON_STEPS, TARGET_COUNT) or not np.isfinite(targets).all():
            raise Issue56V2RiskError("V2 label targets are malformed")
        if self.eligible and len(self.state_digests) != RISK_HORIZON_STEPS:
            raise Issue56V2RiskError("eligible V2 label lacks complete state provenance")
        if any(
            type(digest) is not str
            or len(digest) != 64
            or digest != digest.lower()
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in self.state_digests
        ):
            raise Issue56V2RiskError("V2 label state provenance is malformed")
        if self.eligible and self.termination_reason is not None:
            raise Issue56V2RiskError("eligible V2 label has termination reason")
        if not self.eligible and not self.termination_reason:
            raise Issue56V2RiskError("ineligible V2 label lacks termination reason")
        if float(self.crossing_event) not in {0.0, 1.0}:
            raise Issue56V2RiskError("V2 event label is not binary")
        for value, label in (
            (self.safety_exposure, "V2 exposure"),
            (self.maximum_crossing, "V2 maximum crossing"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V2RiskError(f"{label} is invalid")
        _require_sha(self.label_sha256, "V2 label digest")
        body = {
            "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.label",
            "action_id": self.action_id,
            "decision_step": self.decision_step,
            "track": self.track,
            "targets": targets.tolist(),
            "state_digests": list(self.state_digests),
            "eligible": self.eligible,
            "termination_reason": self.termination_reason,
            "crossing_event": self.crossing_event,
            "safety_exposure": self.safety_exposure,
            "maximum_crossing": self.maximum_crossing,
        }
        if self.label_sha256 != _sha(body):
            raise Issue56V2RiskError("V2 label digest is inconsistent")
        frozen = targets.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "targets", frozen)


def v2_counterfactual_label(
    scenario: Scenario,
    zone_ids: Sequence[str],
    state: PlantState,
    action_id: str,
    current_command: Mapping[str, Any],
    candidate_command: Mapping[str, Any],
    *,
    track: str = "effect_4",
) -> V2RiskLabel:
    if type(scenario) is not Scenario or type(state) is not PlantState:
        raise Issue56V2RiskError("V2 counterfactual requires exact scenario and state")
    try:
        current = validate_external_command(scenario, current_command).to_mapping()
        candidate = validate_external_command(scenario, candidate_command).to_mapping()
    except (ScenarioValidationError, ValueError) as error:
        raise Issue56V2RiskError("V2 counterfactual command is invalid") from error
    commands = _command_schedule(current, candidate, track)
    targets = np.zeros((RISK_HORIZON_STEPS, TARGET_COUNT), dtype=np.float32)
    state_digests: list[str] = []
    cursor = state
    termination_reason: str | None = None
    for offset, command in enumerate(commands):
        try:
            stepped = advance_one_step_with_command(scenario, cursor, command)
        except (InfeasibleActionError, ScenarioValidationError) as error:
            termination_reason = type(error).__name__
            break
        targets[offset] = project_true_targets(scenario, zone_ids, stepped.state)
        state_digests.append(_sha(_jsonable(stepped.state)))
        cursor = stepped.state
    eligible = termination_reason is None
    if not eligible:
        event = exposure = maximum = 0.0
        targets.fill(0.0)
    else:
        event, exposure, maximum = _risk_label_values(targets)
    body = {
        "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.label",
        "action_id": action_id,
        "decision_step": state.step,
        "track": track,
        "targets": targets.tolist(),
        "state_digests": state_digests,
        "eligible": eligible,
        "termination_reason": termination_reason,
        "crossing_event": event,
        "safety_exposure": exposure,
        "maximum_crossing": maximum,
    }
    return V2RiskLabel(
        action_id,
        state.step,
        track,
        targets,
        tuple(state_digests),
        eligible,
        termination_reason,
        event,
        exposure,
        maximum,
        _sha(body),
    )


def _history_complete(history: ForecastHistory) -> bool:
    return bool(
        type(history) is ForecastHistory
        and history.numeric_f32.shape == (HISTORY_WINDOW_STEPS, 194)
        and history.numeric_f32.dtype == np.float32
        and history.status_f32.shape[0] == HISTORY_WINDOW_STEPS
        and history.status_f32.dtype == np.float32
        and history.mode_f32.shape == (HISTORY_WINDOW_STEPS, 4)
        and history.mode_f32.dtype == np.float32
        and history.health_f32.shape == (HISTORY_WINDOW_STEPS, 4)
        and history.health_f32.dtype == np.float32
        and history.alarm_lifecycle_f32.shape[0] == HISTORY_WINDOW_STEPS
        and history.alarm_lifecycle_f32.dtype == np.float32
        and len(history.snapshot_sha256) == HISTORY_WINDOW_STEPS
        and all(
            type(snapshot) is str
            and len(snapshot) == 64
            and snapshot == snapshot.lower()
            and all(char in "0123456789abcdef" for char in snapshot)
            for snapshot in history.snapshot_sha256
        )
        and np.isfinite(history.numeric_f32).all()
        and np.isfinite(history.status_f32).all()
        and np.isfinite(history.mode_f32).all()
        and np.isfinite(history.health_f32).all()
        and np.isfinite(history.alarm_lifecycle_f32).all()
        and np.all((history.status_f32 == 0.0) | (history.status_f32 == 1.0))
        and np.all(history.status_f32.sum(axis=2) == 1.0)
    )


def _current_margin(history: ForecastHistory) -> np.ndarray:
    scales, _, lowers, uppers = target_bounds()
    values = _observable_target_row(history)
    return np.minimum(values - lowers, uppers - values) / scales


def _observable_target_row(history: ForecastHistory) -> np.ndarray:
    """Project current safety targets from public operational descriptors."""

    columns = {
        (str(descriptor["descriptor_id"]), str(descriptor["source_kind"])): index
        for index, descriptor in enumerate(history.layout.operational_descriptors)
    }
    values: list[float] = []
    environmental_fields = {
        "temperature_k",
        "pressure_pa",
        "co2_ppm",
        "o2_mole_fraction",
        "relative_humidity",
    }
    for descriptor in history.layout.target_descriptors:
        descriptor_id = str(descriptor["descriptor_id"])
        field = descriptor_id.rsplit("/", 1)[-1]
        if field in environmental_fields:
            key = (descriptor_id, "primary_sensor_head")
        elif field == "branch_airflow_m3_s":
            zone_id = descriptor_id.rsplit("/", 1)[0]
            key = (f"branch_airflow_m3_s/{zone_id}", "operational_feedback_instrument")
        else:
            key = (descriptor_id, "operational_feedback_instrument")
        if key not in columns:
            raise Issue56V2RiskError(
                f"observable projection lacks target descriptor {descriptor_id}"
            )
        index = columns[key]
        if history.status_f32[-1, index, 0] < 1.0:
            raise Issue56V2RiskError(
                f"current target descriptor {descriptor_id} is unavailable"
            )
        values.append(float(history.numeric_f32[-1, index]))
    return _finite_vector(np.asarray(values), (TARGET_COUNT,), "observable target row")


def alarm_family_slot_indices(bundle: ForecastContracts) -> tuple[tuple[int, ...], ...]:
    """Bind alarm feature columns to the frozen alarm manifest families."""

    if type(bundle) is not ForecastContracts:
        raise Issue56V2RiskError("alarm feature binding requires ForecastContracts")
    result = tuple(
        tuple(
            index
            for index, slot in enumerate(bundle.alarm_slots)
            if slot.family == family
        )
        for family in ALARM_FAMILY_ORDER
    )
    if any(not indices for indices in result):
        raise Issue56V2RiskError("frozen alarm manifest lacks a required family")
    return result


def _alarm_features(
    history: ForecastHistory,
    family_slot_indices: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    latest = np.asarray(history.alarm_lifecycle_f32[-1], dtype=np.float64)
    if latest.ndim != 2 or latest.shape[1] != 4 or latest.shape[0] == 0:
        raise Issue56V2RiskError("alarm projection shape is invalid")
    result = np.zeros(len(ALARM_FAMILY_ORDER), dtype=np.float64)
    if family_slot_indices is None:
        raise Issue56V2RiskError("V2 runtime requires the frozen alarm binding")
    if len(family_slot_indices) != len(ALARM_FAMILY_ORDER):
        raise Issue56V2RiskError("alarm family binding is incomplete")
    for family_index, slot_indices in enumerate(family_slot_indices):
        if any(index < 0 or index >= latest.shape[0] for index in slot_indices):
            raise Issue56V2RiskError("alarm family binding is out of range")
        result[family_index] = float(np.sum(latest[list(slot_indices), 1:]))
    return result


def v2_feature_vector(
    history: ForecastHistory,
    action_f32: np.ndarray,
    *,
    decision_step: int | None = None,
    alarm_family_slots: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Build V2 features solely from the verified forecast projection."""

    if not _history_complete(history):
        raise Issue56V2RiskError("V2 features require a complete finite history")
    action = _finite_vector(action_f32, (ACTION_COUNT,), "V2 action")
    numeric = history.numeric_f32.astype(np.float64)
    current_command = numeric[-1, -ACTION_COUNT:]
    action_delta = action - current_command
    mode = np.asarray(history.mode_f32[-1], dtype=np.float64)
    health = np.asarray(history.health_f32[-1], dtype=np.float64)
    alarms = _alarm_features(history, alarm_family_slots)
    position = float(
        (history.steps[-1] if decision_step is None else decision_step) / EPISODE_STEPS
    )
    values = np.concatenate(
        (
            numeric[-1],
            np.mean(numeric, axis=0),
            numeric[-1] - numeric[0],
            action,
            np.ones(1, dtype=np.float64),
            mode,
            health,
            alarms,
            _current_margin(history),
            action_delta,
            np.asarray([position], dtype=np.float64),
        )
    )
    return _finite_vector(values, (FEATURE_COUNT,), "V2 feature vector").astype(np.float32)


@dataclass(frozen=True, slots=True)
class V2RiskSample:
    family_id: str
    decision_step: int
    split: str
    track: str
    action_id: str
    scenario_sha256: str
    snapshot_sha256: tuple[str, ...]
    input_manifest_sha256: str
    target_manifest_sha256: str
    features_f32: np.ndarray
    crossing_event: float
    safety_exposure: float
    maximum_crossing: float
    label_sha256: str
    sample_sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise Issue56V2RiskError("V2 sample family is invalid")
        if self.split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
            raise Issue56V2RiskError("V2 sample split is invalid")
        if self.track not in LABEL_TRACKS:
            raise Issue56V2RiskError("V2 sample track is invalid")
        if (
            isinstance(self.decision_step, bool)
            or not isinstance(self.decision_step, int)
            or self.decision_step < 0
        ):
            raise Issue56V2RiskError("V2 sample decision step is invalid")
        _require_sha(self.scenario_sha256, "V2 sample scenario")
        if len(self.snapshot_sha256) != HISTORY_WINDOW_STEPS:
            raise Issue56V2RiskError("V2 sample snapshot provenance is incomplete")
        for snapshot in self.snapshot_sha256:
            _require_sha(snapshot, "V2 sample snapshot")
        _require_sha(self.input_manifest_sha256, "V2 sample input manifest")
        _require_sha(self.target_manifest_sha256, "V2 sample target manifest")
        _require_sha(self.label_sha256, "V2 sample label")
        _require_sha(self.sample_sha256, "V2 sample digest")
        features = _finite_vector(self.features_f32, (FEATURE_COUNT,), "V2 sample features")
        if float(self.crossing_event) not in {0.0, 1.0}:
            raise Issue56V2RiskError("V2 sample event is not binary")
        for value in (self.safety_exposure, self.maximum_crossing):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V2RiskError("V2 sample target is invalid")
        body = {
            "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.sample",
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "split": self.split,
            "track": self.track,
            "action_id": self.action_id,
            "scenario_sha256": self.scenario_sha256,
            "snapshot_sha256": list(self.snapshot_sha256),
            "input_manifest_sha256": self.input_manifest_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "features_f32_hex": features.astype(np.float32).tobytes().hex(),
            "crossing_event": self.crossing_event,
            "safety_exposure": self.safety_exposure,
            "maximum_crossing": self.maximum_crossing,
            "label_sha256": self.label_sha256,
        }
        if self.sample_sha256 != _sha(body):
            raise Issue56V2RiskError("V2 sample digest is inconsistent")
        frozen = features.astype(np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "features_f32", frozen)

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.sample",
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "split": self.split,
            "track": self.track,
            "action_id": self.action_id,
            "scenario_sha256": self.scenario_sha256,
            "snapshot_sha256": list(self.snapshot_sha256),
            "input_manifest_sha256": self.input_manifest_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "features_f32_hex": self.features_f32.tobytes().hex(),
            "crossing_event": self.crossing_event,
            "safety_exposure": self.safety_exposure,
            "maximum_crossing": self.maximum_crossing,
            "label_sha256": self.label_sha256,
        }
        return {**body, "sample_sha256": self.sample_sha256}


def make_v2_sample(
    family_id: str,
    split: str,
    track: str,
    action_id: str,
    scenario_sha256: str,
    history: ForecastHistory,
    features_f32: np.ndarray,
    label: V2RiskLabel,
) -> V2RiskSample:
    if not label.eligible:
        raise Issue56V2RiskError("ineligible counterfactual cannot become a finite sample")
    if label.track != track or label.action_id != action_id:
        raise Issue56V2RiskError("V2 sample label identity does not match the sample")
    if label.decision_step != history.steps[-1]:
        raise Issue56V2RiskError("V2 sample decision step does not match history")
    _require_sha(scenario_sha256, "V2 sample scenario")
    features = _finite_vector(features_f32, (FEATURE_COUNT,), "V2 sample features").astype(np.float32)
    if not _history_complete(history):
        raise Issue56V2RiskError("V2 sample requires complete source history")
    snapshots = tuple(getattr(history, "snapshot_sha256", ()))
    if len(snapshots) != HISTORY_WINDOW_STEPS:
        raise Issue56V2RiskError("V2 source history lacks snapshot identities")
    for snapshot in snapshots:
        _require_sha(snapshot, "V2 source snapshot")
    input_manifest = history.layout.input_manifest_sha256
    target_manifest = history.layout.target_manifest_sha256
    body = {
        "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.sample",
        "family_id": family_id,
        "decision_step": label.decision_step,
        "split": split,
        "track": track,
        "action_id": action_id,
        "scenario_sha256": scenario_sha256,
        "snapshot_sha256": list(snapshots),
        "input_manifest_sha256": input_manifest,
        "target_manifest_sha256": target_manifest,
        "features_f32_hex": features.tobytes().hex(),
        "crossing_event": label.crossing_event,
        "safety_exposure": label.safety_exposure,
        "maximum_crossing": label.maximum_crossing,
        "label_sha256": label.label_sha256,
    }
    return V2RiskSample(
        family_id,
        label.decision_step,
        split,
        track,
        action_id,
        scenario_sha256,
        snapshots,
        input_manifest,
        target_manifest,
        features,
        label.crossing_event,
        label.safety_exposure,
        label.maximum_crossing,
        label.label_sha256,
        _sha(body),
    )


def _fit_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise Issue56V2RiskError("V2 fit arrays are malformed")
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.where(np.std(features, axis=0) > 1e-8, np.std(features, axis=0), 1.0)
    normalized_x = (features - feature_mean) / feature_scale
    target_mean = np.mean(targets, axis=0)
    target_scale = np.where(np.std(targets, axis=0) > 1e-8, np.std(targets, axis=0), 1.0)
    normalized_y = (targets - target_mean) / target_scale
    gram = normalized_x.T @ normalized_x
    regularizer = np.eye(features.shape[1], dtype=np.float64) * alpha
    try:
        coef = np.linalg.solve(gram + regularizer, normalized_x.T @ normalized_y)
    except np.linalg.LinAlgError as error:
        raise Issue56V2RiskError("V2 ridge fit is singular") from error
    return feature_mean, feature_scale, target_mean, target_scale, coef


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _bounded_expm1(value: float) -> float:
    """Keep extreme advisory extrapolation finite and therefore rejectable."""

    return math.expm1(min(MAX_LOG_EXPOSURE, max(0.0, float(value))))


@dataclass(frozen=True, slots=True)
class V2RiskPrediction:
    event_probability: float
    upper_event_probability: float
    conditional_exposure: float
    upper_conditional_exposure: float
    conditional_maximum_crossing: float
    upper_maximum_crossing: float
    upper_expected_exposure: float
    hard_ineligible: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class V2RiskModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    event_target_mean: float
    event_target_scale: float
    event_coefficients: np.ndarray
    severity_target_mean: float
    severity_target_scale: float
    severity_coefficients: np.ndarray
    maximum_target_mean: float
    maximum_target_scale: float
    maximum_coefficients: np.ndarray
    event_calibration_offset: float = 0.0
    severity_residual_p90: float = 0.0
    maximum_residual_p90: float = 0.0
    alpha: float = RIDGE_ALPHA
    seed: int = MODEL_SEED
    model_id: str = "issue56-v2-unfitted"
    actuator_authority: bool = False

    def __post_init__(self) -> None:
        arrays = (
            ("feature_mean", self.feature_mean, (FEATURE_COUNT,)),
            ("feature_scale", self.feature_scale, (FEATURE_COUNT,)),
            ("event_coefficients", self.event_coefficients, (FEATURE_COUNT,)),
            ("severity_coefficients", self.severity_coefficients, (FEATURE_COUNT,)),
            ("maximum_coefficients", self.maximum_coefficients, (FEATURE_COUNT,)),
        )
        for label, value, shape in arrays:
            array = _finite_vector(value, shape, f"V2 model {label}")
            object.__setattr__(self, label, array)
        if np.any(self.feature_scale <= 0.0):
            raise Issue56V2RiskError("V2 feature scale must be positive")
        for value, label in (
            (self.event_target_scale, "event target scale"),
            (self.severity_target_scale, "severity target scale"),
            (self.maximum_target_scale, "maximum target scale"),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise Issue56V2RiskError(f"V2 model {label} is invalid")
        for value, label in (
            (self.event_calibration_offset, "event calibration"),
            (self.severity_residual_p90, "severity residual"),
            (self.maximum_residual_p90, "maximum residual"),
            (self.alpha, "alpha"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V2RiskError(f"V2 model {label} is invalid")
        if self.actuator_authority is not False:
            raise Issue56V2RiskError("V2 risk model cannot have actuator authority")
        for value, label in (
            (self.event_target_mean, "event target mean"),
            (self.severity_target_mean, "severity target mean"),
            (self.maximum_target_mean, "maximum target mean"),
        ):
            if not math.isfinite(float(value)):
                raise Issue56V2RiskError(f"V2 model {label} is invalid")
        if self.event_calibration_offset > 1.0:
            raise Issue56V2RiskError("V2 event calibration offset is invalid")

    @classmethod
    def fit(
        cls,
        samples: Sequence[V2RiskSample],
        *,
        track: str = "effect_4",
        alpha: float = RIDGE_ALPHA,
        seed: int = MODEL_SEED,
    ) -> "V2RiskModel":
        items = tuple(sample for sample in samples if sample.track == track)
        if len(items) < 4 or len({item.family_id for item in items}) < 2:
            raise Issue56V2RiskError("V2 fit requires at least four samples from two families")
        if any(item.split != "TRAIN" for item in items):
            raise Issue56V2RiskError("V2 fit accepts TRAIN samples only")
        x = np.stack([item.features_f32 for item in items]).astype(np.float64)
        event = np.asarray(
            [
                [
                    math.log(
                        (1.0 - EVENT_LOGIT_EPSILON) / EVENT_LOGIT_EPSILON
                    )
                    if item.crossing_event > 0.0
                    else -math.log(
                        (1.0 - EVENT_LOGIT_EPSILON) / EVENT_LOGIT_EPSILON
                    )
                ]
                for item in items
            ],
            dtype=np.float64,
        )
        positive = [item for item in items if item.crossing_event > 0.0]
        if len(positive) < 2:
            raise Issue56V2RiskError("V2 severity fit requires positive TRAIN events")
        positive_x = np.stack([item.features_f32 for item in positive]).astype(np.float64)
        severity = np.asarray([[math.log1p(item.safety_exposure)] for item in positive], dtype=np.float64)
        maximum = np.asarray([[math.log1p(item.maximum_crossing)] for item in positive], dtype=np.float64)
        # All heads share TRAIN-only feature normalization.  Severity fitting is
        # positive-only, but its normalizer remains bound to the complete TRAIN
        # feature distribution to avoid a hidden selection-dependent input scale.
        feature_mean = np.mean(x, axis=0)
        feature_scale = np.std(x, axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        normalized_x = (x - feature_mean) / feature_scale
        normalized_positive_x = (positive_x - feature_mean) / feature_scale
        event_target_mean = float(np.mean(event))
        event_target_scale = float(max(np.std(event), 1e-8))
        event_normalized_target = (event[:, 0] - event_target_mean) / event_target_scale
        event_gram = normalized_x.T @ normalized_x + np.eye(FEATURE_COUNT) * alpha
        event_coefficients = np.linalg.solve(
            event_gram, normalized_x.T @ event_normalized_target
        )
        severity_mean = float(np.mean(severity))
        severity_scale = float(max(np.std(severity), 1e-8))
        severity_target = (severity[:, 0] - severity_mean) / severity_scale
        severity_gram = normalized_positive_x.T @ normalized_positive_x + np.eye(FEATURE_COUNT) * alpha
        severity_coefficients = np.linalg.solve(
            severity_gram, normalized_positive_x.T @ severity_target
        )
        maximum_mean = float(np.mean(maximum))
        maximum_scale = float(max(np.std(maximum), 1e-8))
        maximum_target = (maximum[:, 0] - maximum_mean) / maximum_scale
        maximum_coefficients = np.linalg.solve(
            severity_gram, normalized_positive_x.T @ maximum_target
        )
        model_id = "issue56-v2-" + _sha(
            {
                "feature_mean": feature_mean.tolist(),
                "feature_scale": feature_scale.tolist(),
                "event_coefficients": event_coefficients.tolist(),
                "severity_coefficients": severity_coefficients.tolist(),
                "maximum_coefficients": maximum_coefficients.tolist(),
                "seed": seed,
            }
        )[:16]
        return cls(
            feature_mean,
            feature_scale,
            event_target_mean,
            event_target_scale,
            event_coefficients,
            severity_mean,
            severity_scale,
            severity_coefficients,
            maximum_mean,
            maximum_scale,
            maximum_coefficients,
            alpha=alpha,
            seed=seed,
            model_id=model_id,
        )

    def _linear(self, features: np.ndarray, coefficients: np.ndarray, mean: float, scale: float) -> float:
        normalized = (features.astype(np.float64) - self.feature_mean) / self.feature_scale
        return float((normalized @ coefficients) * scale + mean)

    def predict_features(self, features: np.ndarray) -> V2RiskPrediction:
        values = _finite_vector(features, (FEATURE_COUNT,), "V2 inference features")
        event_logit = self._linear(
            values,
            self.event_coefficients,
            self.event_target_mean,
            self.event_target_scale,
        )
        event_probability = _sigmoid(event_logit)
        upper_event = min(1.0, max(0.0, event_probability + self.event_calibration_offset))
        severity_log = max(0.0, self._linear(values, self.severity_coefficients, self.severity_target_mean, self.severity_target_scale))
        maximum_log = max(0.0, self._linear(values, self.maximum_coefficients, self.maximum_target_mean, self.maximum_target_scale))
        conditional = _bounded_expm1(severity_log)
        upper_conditional = _bounded_expm1(
            severity_log + self.severity_residual_p90
        )
        maximum = _bounded_expm1(maximum_log)
        upper_maximum = _bounded_expm1(maximum_log + self.maximum_residual_p90)
        upper_expected = upper_event * upper_conditional
        hard = (
            upper_event > EVENT_LIMIT
            or upper_expected > EXPECTED_EXPOSURE_LIMIT
            or upper_maximum > MAXIMUM_CROSSING_LIMIT
        )
        return V2RiskPrediction(
            event_probability,
            upper_event,
            conditional,
            upper_conditional,
            maximum,
            upper_maximum,
            upper_expected,
            hard,
            "v2_calibrated_risk_limit" if hard else None,
        )

    def predict(
        self,
        history: ForecastHistory,
        action_f32: np.ndarray,
        *,
        decision_step: int | None = None,
        alarm_family_slots: Sequence[Sequence[int]] | None = None,
    ) -> V2RiskPrediction:
        return self.predict_features(
            v2_feature_vector(
                history,
                action_f32,
                decision_step=decision_step,
                alarm_family_slots=alarm_family_slots,
            )
        )

    def calibrate(self, samples: Sequence[V2RiskSample], *, track: str = "effect_4") -> "V2RiskModel":
        items = tuple(sample for sample in samples if sample.track == track)
        if not items or any(item.split != "VALIDATION" for item in items):
            raise Issue56V2RiskError("V2 calibration accepts VALIDATION samples only")
        event_residual: list[float] = []
        severity_residual: list[float] = []
        maximum_residual: list[float] = []
        for item in items:
            values = _finite_vector(item.features_f32, (FEATURE_COUNT,), "V2 calibration feature")
            logit = self._linear(values, self.event_coefficients, self.event_target_mean, self.event_target_scale)
            event_residual.append(abs(_sigmoid(logit) - item.crossing_event))
            if item.crossing_event > 0.0:
                severity_residual.append(
                    abs(self._linear(values, self.severity_coefficients, self.severity_target_mean, self.severity_target_scale) - math.log1p(item.safety_exposure))
                )
                maximum_residual.append(
                    abs(self._linear(values, self.maximum_coefficients, self.maximum_target_mean, self.maximum_target_scale) - math.log1p(item.maximum_crossing))
                )
        if not severity_residual or not maximum_residual:
            raise Issue56V2RiskError("V2 calibration requires positive VALIDATION events")
        # A non-negative additive residual is a conservative monotonic upper
        # calibration while preserving the event head's probability semantics.
        offset = float(np.quantile(np.asarray(event_residual), CALIBRATION_QUANTILE))
        return replace(
            self,
            event_calibration_offset=offset,
            severity_residual_p90=float(np.quantile(np.asarray(severity_residual), CALIBRATION_QUANTILE)),
            maximum_residual_p90=float(np.quantile(np.asarray(maximum_residual), CALIBRATION_QUANTILE)),
            model_id=self.model_id + "-calibrated",
        )

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.model",
            "model_id": self.model_id,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "event_target_mean": self.event_target_mean,
            "event_target_scale": self.event_target_scale,
            "event_coefficients": self.event_coefficients.tolist(),
            "severity_target_mean": self.severity_target_mean,
            "severity_target_scale": self.severity_target_scale,
            "severity_coefficients": self.severity_coefficients.tolist(),
            "maximum_target_mean": self.maximum_target_mean,
            "maximum_target_scale": self.maximum_target_scale,
            "maximum_coefficients": self.maximum_coefficients.tolist(),
            "event_calibration_offset": self.event_calibration_offset,
            "severity_residual_p90": self.severity_residual_p90,
            "maximum_residual_p90": self.maximum_residual_p90,
            "alpha": self.alpha,
            "seed": self.seed,
            "actuator_authority": False,
        }
        return {**body, "model_sha256": _sha(body)}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V2RiskModel":
        if type(mapping) is not dict:
            raise Issue56V2RiskError("V2 model artifact must be an object")
        expected = {
            "schema_version",
            "model_id",
            "feature_mean",
            "feature_scale",
            "event_target_mean",
            "event_target_scale",
            "event_coefficients",
            "severity_target_mean",
            "severity_target_scale",
            "severity_coefficients",
            "maximum_target_mean",
            "maximum_target_scale",
            "maximum_coefficients",
            "event_calibration_offset",
            "severity_residual_p90",
            "maximum_residual_p90",
            "alpha",
            "seed",
            "actuator_authority",
            "model_sha256",
        }
        if set(mapping) != expected:
            raise Issue56V2RiskError("V2 model artifact fields drift")
        digest = mapping.get("model_sha256")
        body = dict(mapping)
        body.pop("model_sha256", None)
        if digest != _sha(body) or mapping.get("schema_version") != f"{ISSUE56_V2_SCHEMA_VERSION}.model":
            raise Issue56V2RiskError("V2 model artifact digest/schema is invalid")
        if mapping["actuator_authority"] is not False:
            raise Issue56V2RiskError("V2 model artifact claims actuator authority")
        return cls(
            np.asarray(mapping["feature_mean"], dtype=np.float64),
            np.asarray(mapping["feature_scale"], dtype=np.float64),
            float(mapping["event_target_mean"]),
            float(mapping["event_target_scale"]),
            np.asarray(mapping["event_coefficients"], dtype=np.float64),
            float(mapping["severity_target_mean"]),
            float(mapping["severity_target_scale"]),
            np.asarray(mapping["severity_coefficients"], dtype=np.float64),
            float(mapping["maximum_target_mean"]),
            float(mapping["maximum_target_scale"]),
            np.asarray(mapping["maximum_coefficients"], dtype=np.float64),
            float(mapping["event_calibration_offset"]),
            float(mapping["severity_residual_p90"]),
            float(mapping["maximum_residual_p90"]),
            float(mapping["alpha"]),
            int(mapping["seed"]),
            str(mapping["model_id"]),
            bool(mapping["actuator_authority"]),
        )


def load_v2_samples(rows: Sequence[Mapping[str, Any]]) -> tuple[V2RiskSample, ...]:
    """Load digest-bound V2 samples without accepting a widened row schema."""

    expected = {
        "schema_version",
        "family_id",
        "decision_step",
        "split",
        "track",
        "action_id",
        "scenario_sha256",
        "snapshot_sha256",
        "input_manifest_sha256",
        "target_manifest_sha256",
        "features_f32_hex",
        "crossing_event",
        "safety_exposure",
        "maximum_crossing",
        "label_sha256",
        "sample_sha256",
    }
    loaded: list[V2RiskSample] = []
    for row in rows:
        if type(row) is not dict or set(row) != expected:
            raise Issue56V2RiskError("V2 sample row fields drift")
        if row["schema_version"] != f"{ISSUE56_V2_SCHEMA_VERSION}.sample":
            raise Issue56V2RiskError("V2 sample row schema drift")
        try:
            raw_features = bytes.fromhex(str(row["features_f32_hex"]))
        except (TypeError, ValueError) as error:
            raise Issue56V2RiskError("V2 sample feature bytes are malformed") from error
        if len(raw_features) != FEATURE_COUNT * np.dtype(np.float32).itemsize:
            raise Issue56V2RiskError("V2 sample feature bytes have the wrong length")
        features = np.frombuffer(raw_features, dtype=np.float32).copy()
        try:
            loaded.append(
                V2RiskSample(
                    str(row["family_id"]),
                    int(row["decision_step"]),
                    str(row["split"]),
                    str(row["track"]),
                    str(row["action_id"]),
                    str(row["scenario_sha256"]),
                    tuple(str(item) for item in row["snapshot_sha256"]),
                    str(row["input_manifest_sha256"]),
                    str(row["target_manifest_sha256"]),
                    features,
                    float(row["crossing_event"]),
                    float(row["safety_exposure"]),
                    float(row["maximum_crossing"]),
                    str(row["label_sha256"]),
                    str(row["sample_sha256"]),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise Issue56V2RiskError("V2 sample row is malformed") from error
    return tuple(loaded)


@dataclass(frozen=True, slots=True)
class V2RiskScore:
    action_id: str
    hard_ineligible: bool
    point_score: float
    upper_event_probability: float
    upper_expected_exposure: float
    upper_maximum_crossing: float
    reason: str | None


def risk_filter_point_scores(
    bundle: ForecastContracts,
    history: ForecastHistory,
    model: V2RiskModel,
    point_predictor: Any,
    current_command: np.ndarray,
    *,
    decision_step: int,
) -> tuple[V2RiskScore, ...]:
    if type(bundle) is not ForecastContracts or type(model) is not V2RiskModel:
        raise Issue56V2RiskError("V2 ranking requires frozen bundle and V2 model")
    current = _finite_vector(current_command, (ACTION_COUNT,), "V2 current command")
    alarm_slots = alarm_family_slot_indices(bundle)
    scores: list[V2RiskScore] = []
    for action in bundle.actions:
        action_vector = project_proposed_action(bundle, action.command)
        prediction = model.predict(
            history,
            action_vector,
            decision_step=decision_step,
            alarm_family_slots=alarm_slots,
        )
        point_prediction = point_predictor.predict(history, action_vector)
        point_score = score_point_prediction(
            action.action_id,
            point_prediction,
            current,
            action_vector,
        )
        scores.append(
            V2RiskScore(
                action.action_id,
                prediction.hard_ineligible or point_score.hard_ineligible,
                point_score.score,
                prediction.upper_event_probability,
                prediction.upper_expected_exposure,
                prediction.upper_maximum_crossing,
                prediction.reason or point_score.reason,
            )
        )
    return tuple(scores)


def risk_only_scores(
    bundle: ForecastContracts,
    history: ForecastHistory,
    model: V2RiskModel,
    *,
    decision_step: int,
) -> tuple[V2RiskScore, ...]:
    """Score candidates by calibrated risk without using utility predictions."""

    alarm_slots = alarm_family_slot_indices(bundle)
    scores: list[V2RiskScore] = []
    for action in bundle.actions:
        action_vector = project_proposed_action(bundle, action.command)
        prediction = model.predict(
            history,
            action_vector,
            decision_step=decision_step,
            alarm_family_slots=alarm_slots,
        )
        scores.append(
            V2RiskScore(
                action.action_id,
                prediction.hard_ineligible,
                prediction.upper_expected_exposure,
                prediction.upper_event_probability,
                prediction.upper_expected_exposure,
                prediction.upper_maximum_crossing,
                prediction.reason,
            )
        )
    return tuple(scores)


def select_risk_filtered_point(scores: Sequence[V2RiskScore]) -> V2RiskScore | None:
    if not scores:
        raise Issue56V2RiskError("V2 selection requires candidates")
    ids = [score.action_id for score in scores]
    if len(ids) != len(set(ids)):
        raise Issue56V2RiskError("V2 selection received duplicate action IDs")
    eligible = [score for score in scores if not score.hard_ineligible and math.isfinite(score.point_score)]
    if not eligible:
        return None
    return min(eligible, key=lambda score: (score.point_score, score.action_id))


def select_risk_only(scores: Sequence[V2RiskScore]) -> V2RiskScore | None:
    if not scores:
        raise Issue56V2RiskError("V2 risk-only selection requires candidates")
    ids = [score.action_id for score in scores]
    if len(ids) != len(set(ids)):
        raise Issue56V2RiskError("V2 risk-only selection received duplicate action IDs")
    eligible = [score for score in scores if not score.hard_ineligible]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda score: (
            score.upper_expected_exposure,
            score.upper_event_probability,
            score.upper_maximum_crossing,
            score.action_id,
        ),
    )


def validation_non_vacuity(
    decision_candidates: Sequence[Sequence[V2RiskScore]],
    *,
    minimum_coverage: float = MIN_VALIDATION_DECISION_COVERAGE,
) -> dict[str, float | int | bool]:
    if not decision_candidates:
        raise Issue56V2RiskError("V2 non-vacuity requires validation decisions")
    retained = sum(select_risk_filtered_point(scores) is not None for scores in decision_candidates)
    coverage = retained / len(decision_candidates)
    return {
        "decision_count": len(decision_candidates),
        "retained_decision_count": retained,
        "coverage": coverage,
        "passed": retained >= 1 and coverage >= minimum_coverage,
    }


def build_v2_proposal(
    hmc: HabitatManagementComputer,
    snapshot_sha256: str,
    step: int,
    command: Mapping[str, Any],
    action_id: str,
) -> dict[str, Any]:
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


def _history_at_v2(
    bundle: ForecastContracts,
    snapshots: Mapping[int, tuple[Any, Any]],
    step: int,
) -> ForecastHistory:
    try:
        pairs = tuple(
            snapshots[index]
            for index in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
        )
    except KeyError as error:
        raise Issue56V2RiskError("V2 history window is incomplete") from error
    return project_history_window(bundle, pairs, window_steps=HISTORY_WINDOW_STEPS)


def collect_v2_family_samples(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    *,
    split: str,
) -> tuple[V2RiskSample, ...]:
    """Collect effect-4 and persistent-32 labels from one HMC-bound family."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise Issue56V2RiskError("V2 collection requires exact frozen inputs")
    if split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
        raise Issue56V2RiskError("V2 collection split is invalid")
    if int(scenario.data["steps"]) != EPISODE_STEPS:
        raise Issue56V2RiskError("V2 collection requires 96-step scenarios")
    actions = tuple(bundle.actions)
    if len(actions) != 4 or len({action.action_id for action in actions}) != 4:
        raise Issue56V2RiskError("V2 collection requires four unique actions")
    zone_ids = scenario_zone_order(scenario)
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce(family_id),
    )
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    last_command: Mapping[str, Any] | None = None
    samples: list[V2RiskSample] = []
    alarm_slots = alarm_family_slot_indices(bundle)
    decision_set = set(v2_decision_steps())
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56V2RiskError(f"HMC terminated during V2 collection at {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snapshots[step] = (snapshot, verification)
        if step in decision_set:
            if last_command is None:
                raise Issue56V2RiskError("V2 decision has no current command")
            history = _history_at_v2(bundle, snapshots, step)
            for action in actions:
                action_vector = project_proposed_action(bundle, action.command)
                features = v2_feature_vector(
                    history,
                    action_vector,
                    decision_step=step,
                    alarm_family_slots=alarm_slots,
                )
                for track in LABEL_TRACKS:
                    label = v2_counterfactual_label(
                        scenario,
                        zone_ids,
                        shadow,
                        action.action_id,
                        last_command,
                        action.command.to_mapping(),
                        track=track,
                    )
                    if not label.eligible:
                        raise Issue56V2RiskError(
                            f"V2 {track} label was infeasible at {family_id}/{step}/{action.action_id}"
                        )
                    samples.append(
                        make_v2_sample(
                            family_id,
                            split,
                            track,
                            action.action_id,
                            scenario.scenario_sha256,
                            history,
                            features,
                            label,
                        )
                    )
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56V2RiskError("HMC failed during V2 collection arbitration")
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56V2RiskError("HMC failed during V2 collection step")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        if _sha_bytes(canonical_json_bytes(shadow_result.receipt)) != stepped.plant_receipt_digest:
            raise Issue56V2RiskError("V2 collection shadow replay diverged")
        shadow = shadow_result.state
        last_command = dict(arbitration.final_command)
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract
    )
    replay = replay_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56V2RiskError("V2 collection trace failed strict replay")
    return tuple(samples)


V2_ARMS = ("risk_only_v2", "risk_filtered_point_v2")


@dataclass(frozen=True, slots=True)
class V2EpisodeRecord:
    arm: str
    family_id: str
    family_index: int
    scenario_sha256: str
    decision_steps: tuple[int, ...]
    decision_actions: tuple[str | None, ...]
    proposal_count: int
    abstention_count: int
    admitted_proposal_count: int
    hmc_rejection_count: int
    safety_exposure: float
    safety_violation_steps: int
    comfort_deviation: float
    resource_composite: float
    control_run_id: str
    trace_sha256: str
    replay_committed_steps: int
    replay_final_state_sha256: str
    episode_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in V2_ARMS:
            raise Issue56V2RiskError("V2 episode arm is invalid")
        if self.decision_steps != v2_decision_steps():
            raise Issue56V2RiskError("V2 episode decision steps drifted")
        if len(self.decision_actions) != len(self.decision_steps):
            raise Issue56V2RiskError("V2 episode decision actions drifted")
        if sum(action is not None for action in self.decision_actions) != self.proposal_count:
            raise Issue56V2RiskError("V2 proposal count does not match actions")
        if self.proposal_count + self.abstention_count != len(self.decision_steps):
            raise Issue56V2RiskError("V2 decisions are not accounted for")
        if any(
            action is not None and (type(action) is not str or not action)
            for action in self.decision_actions
        ):
            raise Issue56V2RiskError("V2 decision action identity is malformed")
        for value, label in (
            (self.family_id, "family"),
            (self.control_run_id, "control run"),
        ):
            if type(value) is not str or not value:
                raise Issue56V2RiskError(f"V2 {label} identity is invalid")
        for value, label in (
            (self.family_index, "family index"),
            (self.proposal_count, "proposal count"),
            (self.abstention_count, "abstention count"),
            (self.admitted_proposal_count, "admitted proposal count"),
            (self.hmc_rejection_count, "HMC rejection count"),
            (self.safety_violation_steps, "safety violation count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Issue56V2RiskError(f"V2 {label} is invalid")
        if self.proposal_count != self.admitted_proposal_count:
            raise Issue56V2RiskError("V2 proposal admission count is inconsistent")
        if self.hmc_rejection_count > self.admitted_proposal_count:
            raise Issue56V2RiskError("V2 HMC rejection count is inconsistent")
        for value in (
            self.safety_exposure,
            self.comfort_deviation,
            self.resource_composite,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V2RiskError("V2 episode metric is invalid")
        for value, label in (
            (self.scenario_sha256, "scenario"),
            (self.trace_sha256, "trace"),
            (self.replay_final_state_sha256, "replay final state"),
            (self.episode_sha256, "episode"),
        ):
            _require_sha(value, f"V2 {label} digest")
        if self.replay_committed_steps != EPISODE_STEPS:
            raise Issue56V2RiskError("V2 replay did not commit all steps")
        if self.episode_sha256 != _sha(self._body()):
            raise Issue56V2RiskError("V2 episode digest is inconsistent")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.episode",
            "arm": self.arm,
            "family_id": self.family_id,
            "family_index": self.family_index,
            "scenario_sha256": self.scenario_sha256,
            "decision_steps": list(self.decision_steps),
            "decision_actions": list(self.decision_actions),
            "proposal_count": self.proposal_count,
            "abstention_count": self.abstention_count,
            "admitted_proposal_count": self.admitted_proposal_count,
            "hmc_rejection_count": self.hmc_rejection_count,
            "safety_exposure": self.safety_exposure,
            "safety_violation_steps": self.safety_violation_steps,
            "comfort_deviation": self.comfort_deviation,
            "resource_composite": self.resource_composite,
            "control_run_id": self.control_run_id,
            "trace_sha256": self.trace_sha256,
            "replay_committed_steps": self.replay_committed_steps,
            "replay_final_state_sha256": self.replay_final_state_sha256,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "episode_sha256": self.episode_sha256}


def run_v2_episode(
    bundle: ForecastContracts,
    scenario: Scenario,
    arm: str,
    family_id: str,
    family_index: int,
    model: V2RiskModel,
    point_predictor: Any,
) -> V2EpisodeRecord:
    """Run a V2 advisory arm through HMC and strict replay."""

    if arm not in V2_ARMS:
        raise Issue56V2RiskError("V2 episode arm is invalid")
    if type(model) is not V2RiskModel or type(scenario) is not Scenario:
        raise Issue56V2RiskError("V2 episode inputs are invalid")
    if arm == "risk_filtered_point_v2" and not callable(
        getattr(point_predictor, "predict", None)
    ):
        raise Issue56V2RiskError("risk-filtered V2 episode requires point predictor")
    actions = tuple(bundle.actions)
    zone_ids = scenario_zone_order(scenario)
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce(family_id),
    )
    shadow = initial_state(scenario)
    initial_row = project_true_targets(scenario, zone_ids, shadow)
    states: dict[int, PlantState] = {0: shadow}
    snapshots: dict[int, tuple[Any, Any]] = {}
    last_command: Mapping[str, Any] | None = None
    decisions = v2_decision_steps()
    decision_actions: list[str | None] = []
    proposal_count = 0
    abstention_count = 0
    admitted_count = 0
    rejection_count = 0
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56V2RiskError(f"HMC terminated during V2 episode at {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snapshots[step] = (snapshot, verification)
        if last_command is None and step not in decisions:
            proposal = None
        elif step in decisions:
            if last_command is None:
                raise Issue56V2RiskError("V2 episode decision has no current command")
            history = _history_at_v2(bundle, snapshots, step)
            if arm == "risk_only_v2":
                scores = risk_only_scores(
                    bundle, history, model, decision_step=step
                )
                selected = select_risk_only(scores)
            else:
                current_vector = _command_vector(scenario, last_command)
                scores = risk_filter_point_scores(
                    bundle,
                    history,
                    model,
                    point_predictor,
                    current_vector,
                    decision_step=step,
                )
                selected = select_risk_filtered_point(scores)
            if selected is None:
                proposal = None
                abstention_count += 1
                decision_actions.append(None)
            else:
                action = next(item for item in actions if item.action_id == selected.action_id)
                proposal = build_v2_proposal(
                    hmc,
                    snapshot.snapshot_sha256,
                    step,
                    action.command.to_mapping(),
                    action.action_id,
                )
                proposal_count += 1
                decision_actions.append(selected.action_id)
        else:
            proposal = None
        receipt = hmc.propose(proposal, handle).to_mapping()
        if step in decisions:
            if proposal is None:
                if receipt["validation_outcome"] != "NO_PROPOSAL":
                    raise Issue56V2RiskError("V2 abstention receipt is not NO_PROPOSAL")
            elif (receipt["attempt_class"], receipt["validation_outcome"]) != (
                "CANONICAL_PROPOSAL",
                "VALID",
            ):
                raise Issue56V2RiskError("V2 proposal was not admitted")
            else:
                admitted_count += 1
        elif receipt["validation_outcome"] != "NO_PROPOSAL":
            raise Issue56V2RiskError("V2 proposal was issued outside a decision")
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56V2RiskError("V2 HMC arbitration failed")
        if proposal is not None:
            requested_sha = validate_external_command(
                scenario, proposal["proposed_command"]
            ).sha256
            if arbitration.final_command_sha256 != requested_sha:
                rejection_count += 1
        last_command = dict(arbitration.final_command)
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56V2RiskError("V2 HMC step failed")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        if _sha_bytes(canonical_json_bytes(shadow_result.receipt)) != stepped.plant_receipt_digest:
            raise Issue56V2RiskError("V2 episode shadow replay diverged")
        shadow = shadow_result.state
        states[shadow.step] = shadow
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract
    )
    replay = replay_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56V2RiskError("V2 episode trace failed strict replay")
    metrics = compute_race_metrics(
        scenario,
        zone_ids,
        initial_row,
        [states[index] for index in range(1, EPISODE_STEPS + 1)],
    )
    body = {
        "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.episode",
        "arm": arm,
        "family_id": family_id,
        "family_index": family_index,
        "scenario_sha256": scenario.scenario_sha256,
        "decision_steps": list(decisions),
        "decision_actions": decision_actions,
        "proposal_count": proposal_count,
        "abstention_count": abstention_count,
        "admitted_proposal_count": admitted_count,
        "hmc_rejection_count": rejection_count,
        "safety_exposure": float(metrics["safety_exposure"]),
        "safety_violation_steps": int(metrics["safety_violation_steps"]),
        "comfort_deviation": float(metrics["comfort_deviation"]),
        "resource_composite": float(metrics["resource_composite"]),
        "control_run_id": hmc.control_run_id,
        "trace_sha256": _sha_bytes(trace.canonical_bytes),
        "replay_committed_steps": int(replay.committed_step_count),
        "replay_final_state_sha256": str(replay.final_state_sha256),
    }
    return V2EpisodeRecord(
        arm=arm,
        family_id=family_id,
        family_index=family_index,
        scenario_sha256=scenario.scenario_sha256,
        decision_steps=decisions,
        decision_actions=tuple(decision_actions),
        proposal_count=proposal_count,
        abstention_count=abstention_count,
        admitted_proposal_count=admitted_count,
        hmc_rejection_count=rejection_count,
        safety_exposure=float(metrics["safety_exposure"]),
        safety_violation_steps=int(metrics["safety_violation_steps"]),
        comfort_deviation=float(metrics["comfort_deviation"]),
        resource_composite=float(metrics["resource_composite"]),
        control_run_id=hmc.control_run_id,
        trace_sha256=_sha_bytes(trace.canonical_bytes),
        replay_committed_steps=int(replay.committed_step_count),
        replay_final_state_sha256=str(replay.final_state_sha256),
        episode_sha256=_sha(body),
    )


def _history_from_sample(sample: V2RiskSample) -> np.ndarray:
    return _finite_vector(sample.features_f32, (FEATURE_COUNT,), "V2 sample feature")


def calibration_metrics_v2(model: V2RiskModel, samples: Sequence[V2RiskSample]) -> dict[str, float | int]:
    items = tuple(samples)
    if not items:
        raise Issue56V2RiskError("V2 calibration metrics require samples")
    brier = 0.0
    event_hits = 0
    upper_hits = 0
    positive_upper_hits = 0
    positive_maximum_hits = 0
    positive_count = 0
    severity_error = 0.0
    for item in items:
        prediction = model.predict_features(_history_from_sample(item))
        brier += (prediction.event_probability - item.crossing_event) ** 2
        event_hits += int((prediction.event_probability >= EVENT_LIMIT) == bool(item.crossing_event))
        upper_hits += int(item.safety_exposure <= prediction.upper_expected_exposure)
        if item.crossing_event > 0.0:
            positive_count += 1
            positive_upper_hits += int(
                item.safety_exposure <= prediction.upper_conditional_exposure
            )
            positive_maximum_hits += int(
                item.maximum_crossing <= prediction.upper_maximum_crossing
            )
            severity_error += abs(prediction.conditional_exposure - item.safety_exposure)
    return {
        "sample_count": len(items),
        "crossing_brier": brier / len(items),
        "event_threshold_accuracy": event_hits / len(items),
        "upper_expected_exposure_coverage": upper_hits / len(items),
        "positive_sample_count": positive_count,
        "positive_upper_conditional_coverage": positive_upper_hits / max(positive_count, 1),
        "positive_upper_maximum_coverage": positive_maximum_hits / max(positive_count, 1),
        "positive_mean_absolute_exposure_error": severity_error / max(positive_count, 1),
    }


__all__ = [
    "ACTION_COUNT",
    "ALARM_FAMILY_ORDER",
    "EVENT_LIMIT",
    "EVENT_LOGIT_EPSILON",
    "EXPECTED_EXPOSURE_LIMIT",
    "FEATURE_COUNT",
    "Issue56V2RiskError",
    "LABEL_TRACKS",
    "MAXIMUM_CROSSING_LIMIT",
    "MODEL_SOURCE_TYPE",
    "RISK_HORIZON_STEPS",
    "RISK_METRIC_ID",
    "V2RiskLabel",
    "V2RiskModel",
    "V2RiskPrediction",
    "V2RiskSample",
    "V2RiskScore",
    "V2_ARMS",
    "alarm_family_slot_indices",
    "build_v2_proposal",
    "calibration_metrics_v2",
    "make_v2_sample",
    "risk_filter_point_scores",
    "risk_only_scores",
    "run_v2_episode",
    "select_risk_filtered_point",
    "select_risk_only",
    "load_v2_samples",
    "select_risk_filtered_point",
    "v2_counterfactual_label",
    "v2_decision_steps",
    "v2_feature_vector",
    "v2_family_split",
    "validation_non_vacuity",
]
