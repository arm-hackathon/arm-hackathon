"""Replayable development-corpus records for the Issue #56 V4 study.

V3 samples retain a counterfactual trace digest but not the serialized trace
bytes.  This module creates a new, separate V4 corpus record that keeps the
trace bytes available for independent artifact verification.  It does not
train a model, change V3 behavior, or grant the adviser actuator authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .control_trace import (
    _trace_final_state_digest,
    parse_control_trace,
    replay_control_trace,
)
from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast.projection import project_history_window, project_proposed_action
from .forecast_issue55_race import (
    COMFORT_COLUMNS,
    EPISODE_STEPS,
    FAMILY_COUNT,
    HMC_IMPLEMENTATION_GIT_SHA,
    RESOURCE_COLUMNS,
    _crossings as true_crossings,
    build_family_scenario,
    deterministic_family_ids,
    episode_nonce,
    project_true_targets,
    scenario_zone_order,
    TARGET_COUNT,
    target_bounds,
)
from .forecast_issue56_action_risk_v2 import (
    HISTORY_WINDOW_STEPS,
    alarm_family_slot_indices,
    v2_decision_steps,
    v2_feature_vector,
)
from .forecast_issue56_action_risk_v3 import (
    ISSUE56_V3_SCHEMA_VERSION,
    V3_HORIZONS,
    V3_LABEL_TRACK,
    V3HorizonMetric,
    V3PolicyLabel,
    V3RiskSample,
    _jsonable,
    _baseline_histories,
    _clone_hmc_for_branch,
)
from .forecast_issue56_action_risk_v4_features import (
    V4_TEMPORAL_FEATURE_COUNT,
    v4_observable_action_mask,
    v4_temporal_feature_vector,
)
from .hmc import HabitatManagementComputer
from .physics import (
    advance_one_step_with_command,
    initial_state,
    operating_mode_for_application_step,
    validate_external_command,
    validate_external_step_result,
)
from .scenario import Scenario


ISSUE56_V4_CORPUS_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v4_corpus_v4"
V4_CORPUS_TRACE_DIRECTORY = "counterfactual-traces"


def v4_label_manifest() -> dict[str, Any]:
    """Return the closed label/trace pairing declaration for provenance."""

    return {
        "schema_version": f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.labels",
        "track": "hmc_persistent_remaining",
        "horizons": [4, 16, 32, "remaining"],
        "required_targets": [
            "crossing_event",
            "safety_exposure",
            "maximum_crossing",
            "comfort_deviation",
            "resource_composite",
        ],
        "relative_action_targets": [
            "safety_exposure_delta_vs_hold",
            "comfort_deviation_delta_vs_hold",
            "resource_composite_delta_vs_hold",
        ],
        "trace_pair": {
            "action": "counterfactual_action_trace",
            "hold": "no_proposal_hold_trace",
            "relative_target": "action_minus_hold",
        },
    }


class Issue56V4CorpusError(ValueError):
    """Raised when a replayable V4 corpus record is malformed."""


def _sha(value: object) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(_jsonable(value))).hexdigest()
    except (TypeError, ValueError) as error:
        raise Issue56V4CorpusError("V4 corpus digest input is not canonical JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Issue56V4CorpusError(f"{label} must be lowercase SHA-256")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise Issue56V4CorpusError(f"{label} must be a non-empty string")
    return value


def _replay_label_metrics(targets: np.ndarray) -> tuple[float, float, float]:
    crossings = true_crossings(np.asarray(targets, dtype=np.float64))
    return (
        float(np.any(crossings > 0.0)),
        float(np.sum(crossings)),
        float(np.max(crossings)),
    )


def _trace_relative_path(trace_sha256: str) -> str:
    _require_sha(trace_sha256, "V4 counterfactual trace")
    return f"{V4_CORPUS_TRACE_DIRECTORY}/{trace_sha256}.json"


def _validate_trace_relative_path(value: object, label: str) -> str:
    path = value
    path_parts = path.split("/") if type(path) is str else ()
    if (
        type(path) is not str
        or not path.startswith(f"{V4_CORPUS_TRACE_DIRECTORY}/")
        or len(path_parts) != 2
        or any(part in {"", ".", ".."} for part in path_parts)
        or "\\" in path
    ):
        raise Issue56V4CorpusError(f"{label} path is invalid")
    return path


def _trace_path_contains_symlink(path: Path, root: Path) -> bool:
    """Reject symlinked corpus components before resolving an artifact path."""

    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise Issue56V4CorpusError("V4 trace path escaped the corpus directory") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


@dataclass(frozen=True, slots=True)
class V4TrajectoryMetrics:
    """True-plant metrics for one complete remaining-horizon trajectory."""

    safety_exposure: float
    safety_violation_steps: int
    comfort_deviation: float
    resource_composite: float

    def __post_init__(self) -> None:
        if type(self.safety_violation_steps) is not int or self.safety_violation_steps < 0:
            raise Issue56V4CorpusError("V4 safety violation count is invalid")
        for value, label in (
            (self.safety_exposure, "V4 safety exposure"),
            (self.comfort_deviation, "V4 comfort deviation"),
            (self.resource_composite, "V4 resource composite"),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V4CorpusError(f"{label} is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "safety_exposure": self.safety_exposure,
            "safety_violation_steps": self.safety_violation_steps,
            "comfort_deviation": self.comfort_deviation,
            "resource_composite": self.resource_composite,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V4TrajectoryMetrics":
        expected = {
            "safety_exposure",
            "safety_violation_steps",
            "comfort_deviation",
            "resource_composite",
        }
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V4CorpusError("V4 trajectory metric fields drift")
        return cls(
            mapping["safety_exposure"],
            mapping["safety_violation_steps"],
            mapping["comfort_deviation"],
            mapping["resource_composite"],
        )


@dataclass(frozen=True, slots=True)
class V4RelativeActionTargets:
    """Action-minus-hold targets used by the V4 utility/ranking candidates."""

    safety_exposure_delta_vs_hold: float
    comfort_deviation_delta_vs_hold: float
    resource_composite_delta_vs_hold: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.safety_exposure_delta_vs_hold, "V4 relative safety exposure"),
            (self.comfort_deviation_delta_vs_hold, "V4 relative comfort deviation"),
            (self.resource_composite_delta_vs_hold, "V4 relative resource composite"),
        ):
            if not np.isfinite(float(value)):
                raise Issue56V4CorpusError(f"{label} is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "safety_exposure_delta_vs_hold": self.safety_exposure_delta_vs_hold,
            "comfort_deviation_delta_vs_hold": self.comfort_deviation_delta_vs_hold,
            "resource_composite_delta_vs_hold": self.resource_composite_delta_vs_hold,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V4RelativeActionTargets":
        expected = {
            "safety_exposure_delta_vs_hold",
            "comfort_deviation_delta_vs_hold",
            "resource_composite_delta_vs_hold",
        }
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V4CorpusError("V4 relative target fields drift")
        return cls(
            mapping["safety_exposure_delta_vs_hold"],
            mapping["comfort_deviation_delta_vs_hold"],
            mapping["resource_composite_delta_vs_hold"],
        )


@dataclass(frozen=True, slots=True)
class _V4BranchReplay:
    """Replay output shared by candidate and hold trajectories."""

    label: V3PolicyLabel | None
    trace_canonical_bytes: bytes
    trajectory_metrics: V4TrajectoryMetrics


@dataclass(frozen=True, slots=True)
class _V4SerializedTraceReplay:
    """Strict replay details needed to bind a row to its trace semantics."""

    parsed: Any
    step_state_digests: tuple[str, ...]
    target_rows: np.ndarray
    state_before_decision: Any
    step_events_by_step: Mapping[int, Mapping[str, Any]]
    arbitration_events_by_step: Mapping[int, Mapping[str, Any]]
    proposal_events_by_step: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class V4CounterfactualReplay:
    """One strictly replayed counterfactual branch and its serialized trace."""

    label: V3PolicyLabel
    trace_canonical_bytes: bytes
    trajectory_metrics: V4TrajectoryMetrics

    def __post_init__(self) -> None:
        if type(self.label) is not V3PolicyLabel:
            raise Issue56V4CorpusError("V4 counterfactual label is invalid")
        if type(self.trace_canonical_bytes) is not bytes or not self.trace_canonical_bytes:
            raise Issue56V4CorpusError("V4 counterfactual trace bytes are invalid")
        if type(self.trajectory_metrics) is not V4TrajectoryMetrics:
            raise Issue56V4CorpusError("V4 counterfactual trajectory metrics are invalid")
        if _sha_bytes(self.trace_canonical_bytes) != self.label.trace_sha256:
            raise Issue56V4CorpusError("V4 counterfactual trace digest differs from label")


@dataclass(frozen=True, slots=True)
class V4RiskSample:
    """A V3-compatible sample with retained action and hold trajectories."""

    base_sample: V3RiskSample
    counterfactual_trace_relative_path: str
    counterfactual_trace_sha256: str
    counterfactual_trace_bytes: bytes
    hold_trace_relative_path: str
    hold_trace_sha256: str
    hold_trace_bytes: bytes
    temporal_features_f32: np.ndarray
    observable_action_mask: tuple[bool, ...]
    trajectory_metrics: V4TrajectoryMetrics
    hold_trajectory_metrics: V4TrajectoryMetrics
    relative_action_targets: V4RelativeActionTargets
    sample_sha256: str

    def __post_init__(self) -> None:
        if type(self.base_sample) is not V3RiskSample:
            raise Issue56V4CorpusError("V4 sample base record is invalid")
        for path, digest, content, label in (
            (
                self.counterfactual_trace_relative_path,
                self.counterfactual_trace_sha256,
                self.counterfactual_trace_bytes,
                "V4 counterfactual trace",
            ),
            (
                self.hold_trace_relative_path,
                self.hold_trace_sha256,
                self.hold_trace_bytes,
                "V4 hold trace",
            ),
        ):
            _validate_trace_relative_path(path, label)
            if path != _trace_relative_path(digest):
                raise Issue56V4CorpusError(f"{label} path is not content-addressed")
            _require_sha(digest, label)
            if type(content) is not bytes or not content:
                raise Issue56V4CorpusError(f"{label} bytes are invalid")
            if _sha_bytes(content) != digest:
                raise Issue56V4CorpusError(f"{label} bytes do not match digest")
        if self.base_sample.label.trace_sha256 != self.counterfactual_trace_sha256:
            raise Issue56V4CorpusError("V4 label and counterfactual trace identities differ")
        temporal_features = np.asarray(self.temporal_features_f32, dtype=np.float64)
        if (
            temporal_features.shape != (V4_TEMPORAL_FEATURE_COUNT,)
            or not np.isfinite(temporal_features).all()
        ):
            raise Issue56V4CorpusError("V4 temporal feature vector is malformed")
        temporal_features = temporal_features.astype(np.float32)
        temporal_features.setflags(write=False)
        object.__setattr__(self, "temporal_features_f32", temporal_features)
        if (
            type(self.observable_action_mask) is not tuple
            or len(self.observable_action_mask) != 4
            or any(type(value) is not bool for value in self.observable_action_mask)
        ):
            raise Issue56V4CorpusError("V4 observable action mask is malformed")
        if type(self.trajectory_metrics) is not V4TrajectoryMetrics:
            raise Issue56V4CorpusError("V4 action trajectory metrics are invalid")
        if type(self.hold_trajectory_metrics) is not V4TrajectoryMetrics:
            raise Issue56V4CorpusError("V4 hold trajectory metrics are invalid")
        if type(self.relative_action_targets) is not V4RelativeActionTargets:
            raise Issue56V4CorpusError("V4 relative action targets are invalid")
        expected_relative = V4RelativeActionTargets(
            self.trajectory_metrics.safety_exposure
            - self.hold_trajectory_metrics.safety_exposure,
            self.trajectory_metrics.comfort_deviation
            - self.hold_trajectory_metrics.comfort_deviation,
            self.trajectory_metrics.resource_composite
            - self.hold_trajectory_metrics.resource_composite,
        )
        if self.relative_action_targets != expected_relative:
            raise Issue56V4CorpusError("V4 relative action targets are inconsistent")
        _require_sha(self.sample_sha256, "V4 sample")
        if self.sample_sha256 != _sha(self._body()):
            raise Issue56V4CorpusError("V4 sample digest is inconsistent")

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
    def label(self) -> V3PolicyLabel:
        return self.base_sample.label

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.sample",
            "base_sample": self.base_sample.to_mapping(),
            "counterfactual_trace_relative_path": self.counterfactual_trace_relative_path,
            "counterfactual_trace_sha256": self.counterfactual_trace_sha256,
            "hold_trace_relative_path": self.hold_trace_relative_path,
            "hold_trace_sha256": self.hold_trace_sha256,
            "temporal_features_f32_hex": self.temporal_features_f32.tobytes().hex(),
            "observable_action_mask": list(self.observable_action_mask),
            "trajectory_metrics": self.trajectory_metrics.to_mapping(),
            "hold_trajectory_metrics": self.hold_trajectory_metrics.to_mapping(),
            "relative_action_targets": self.relative_action_targets.to_mapping(),
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "sample_sha256": self.sample_sha256}

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        trace_canonical_bytes: bytes,
        hold_trace_canonical_bytes: bytes,
    ) -> "V4RiskSample":
        expected = {
            "schema_version",
            "base_sample",
            "counterfactual_trace_relative_path",
            "counterfactual_trace_sha256",
            "hold_trace_relative_path",
            "hold_trace_sha256",
            "temporal_features_f32_hex",
            "observable_action_mask",
            "trajectory_metrics",
            "hold_trajectory_metrics",
            "relative_action_targets",
            "sample_sha256",
        }
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V4CorpusError("V4 sample fields drift")
        if mapping["schema_version"] != f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.sample":
            raise Issue56V4CorpusError("V4 sample schema drift")
        if type(mapping["temporal_features_f32_hex"]) is not str:
            raise Issue56V4CorpusError("V4 temporal feature bytes are malformed")
        try:
            raw_temporal_features = bytes.fromhex(mapping["temporal_features_f32_hex"])
        except ValueError as error:
            raise Issue56V4CorpusError("V4 temporal feature bytes are malformed") from error
        if len(raw_temporal_features) != V4_TEMPORAL_FEATURE_COUNT * np.dtype(np.float32).itemsize:
            raise Issue56V4CorpusError("V4 temporal feature bytes have the wrong length")
        if type(mapping["observable_action_mask"]) is not list:
            raise Issue56V4CorpusError("V4 observable action mask is malformed")
        return cls(
            V3RiskSample.from_mapping(mapping["base_sample"]),
            mapping["counterfactual_trace_relative_path"],
            mapping["counterfactual_trace_sha256"],
            trace_canonical_bytes,
            mapping["hold_trace_relative_path"],
            mapping["hold_trace_sha256"],
            hold_trace_canonical_bytes,
            np.frombuffer(raw_temporal_features, dtype=np.float32).copy(),
            tuple(mapping["observable_action_mask"]),
            V4TrajectoryMetrics.from_mapping(mapping["trajectory_metrics"]),
            V4TrajectoryMetrics.from_mapping(mapping["hold_trajectory_metrics"]),
            V4RelativeActionTargets.from_mapping(mapping["relative_action_targets"]),
            mapping["sample_sha256"],
        )


def _replay_counterfactual_branch(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    decision_step: int,
    action_id: str,
    command: Mapping[str, Any],
    *,
    current_command_sha256: str,
    branch_hmc: HabitatManagementComputer,
    branch_state: Any,
) -> V4CounterfactualReplay:
    """Replay one candidate branch and retain its exact trace bytes."""

    replay = _replay_v4_branch(
        bundle,
        scenario,
        family_id,
        decision_step,
        action_id,
        command,
        current_command_sha256=current_command_sha256,
        branch_hmc=branch_hmc,
        branch_state=branch_state,
    )
    if replay.label is None:
        raise Issue56V4CorpusError("V4 candidate replay did not produce a label")
    return V4CounterfactualReplay(
        replay.label,
        replay.trace_canonical_bytes,
        replay.trajectory_metrics,
    )


def _trajectory_metrics(
    scenario: Scenario,
    zone_ids: tuple[str, ...],
    state_before_decision: Any,
    targets: np.ndarray,
    decision_step: int,
) -> V4TrajectoryMetrics:
    if targets.ndim != 2 or targets.shape[1] != TARGET_COUNT or not targets.shape[0]:
        raise Issue56V4CorpusError("V4 trajectory targets are malformed")
    crossings = true_crossings(np.asarray(targets, dtype=np.float64))
    _, nominals, _, _ = target_bounds()
    occupied = [
        row
        for offset, row in enumerate(targets)
        if operating_mode_for_application_step(scenario, decision_step + offset)
        == "occupied"
    ]
    comfort = float(
        np.mean(
            np.abs(
                np.stack(occupied)[:, list(COMFORT_COLUMNS)]
                - nominals[list(COMFORT_COLUMNS)][None, :]
            )
        )
        if occupied
        else 0.0
    )
    current = project_true_targets(scenario, zone_ids, state_before_decision).astype(
        np.float64
    )
    final = np.asarray(targets[-1], dtype=np.float64)
    resource = float(
        sum(max(0.0, float(current[column]) - float(final[column])) for column in RESOURCE_COLUMNS)
    )
    return V4TrajectoryMetrics(
        float(np.sum(crossings)),
        int(np.count_nonzero(np.any(crossings > 0.0, axis=1))),
        comfort,
        resource,
    )


def _strict_replay_trace_details(
    trace_canonical_bytes: bytes,
    bundle: ForecastContracts,
    scenario: Scenario,
    *,
    family_id: str,
    decision_step: int,
) -> _V4SerializedTraceReplay:
    """Replay every committed step and retain the intermediate state identities."""

    if type(trace_canonical_bytes) is not bytes or not trace_canonical_bytes:
        raise Issue56V4CorpusError("V4 serialized trace bytes are invalid")
    if decision_step not in v2_decision_steps():
        raise Issue56V4CorpusError("V4 decision step is invalid")
    try:
        parsed = parse_control_trace(
            trace_canonical_bytes,
            scenario=scenario,
            contract=bundle.hmc_contract,
        )
    except Exception as error:
        raise Issue56V4CorpusError("V4 serialized trace failed strict replay") from error
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or parsed.footer["final_sequence"] != EPISODE_STEPS
    ):
        raise Issue56V4CorpusError("V4 serialized trace replay receipt is inconsistent")
    try:
        reset_nonce = bytes.fromhex(str(parsed.header["reset_nonce_hex"]))
    except ValueError as error:
        raise Issue56V4CorpusError("V4 trace reset nonce is malformed") from error
    if reset_nonce != episode_nonce(family_id):
        raise Issue56V4CorpusError("V4 trace family identity is inconsistent")

    events = tuple(parsed.events)
    proposal_events: dict[int, Mapping[str, Any]] = {}
    arbitration_events: dict[int, Mapping[str, Any]] = {}
    step_events: dict[int, Mapping[str, Any]] = {}
    for event in events:
        receipt = event["receipt"]
        kind = event["event_kind"]
        if kind == "PROPOSAL":
            key = receipt["sequence"]
            destination = proposal_events
        elif kind == "ARBITRATION":
            key = receipt["application_step"]
            destination = arbitration_events
        elif kind == "STEP":
            key = receipt["application_step"]
            destination = step_events
        else:
            continue
        if type(key) is not int or key in destination:
            raise Issue56V4CorpusError("V4 trace contains duplicate decision events")
        destination[key] = event
    expected_steps = set(range(EPISODE_STEPS))
    if (
        set(proposal_events) != expected_steps
        or set(arbitration_events) != expected_steps
        or set(step_events) != expected_steps
    ):
        raise Issue56V4CorpusError("V4 trace decision-event coverage is incomplete")

    state = initial_state(scenario)
    state_before_decision = None
    state_digests: list[str] = []
    target_rows: list[np.ndarray] = []
    zone_ids = scenario_zone_order(scenario)
    for step in range(EPISODE_STEPS):
        if step == decision_step:
            state_before_decision = state
        step_event = step_events[step]
        step_receipt = step_event["receipt"]
        arbitration = arbitration_events[step]["receipt"]
        command = arbitration["final_command"]
        try:
            candidate = advance_one_step_with_command(scenario, state, command)
            validate_external_step_result(scenario, state, command, candidate)
        except Exception as error:
            raise Issue56V4CorpusError("V4 trace step does not replay causally") from error
        if (
            candidate.receipt["external_command_digest"]
            != step_receipt["returned_external_command_digest"]
            or candidate.receipt["external_command_digest"]
            != step_receipt["final_command_sha256"]
            or _sha_bytes(canonical_json_bytes(candidate.receipt))
            != step_receipt["plant_receipt_digest"]
        ):
            raise Issue56V4CorpusError("V4 trace step receipt is not bound to replay")
        state = candidate.state
        state_digests.append(_sha(state))
        target_rows.append(project_true_targets(scenario, zone_ids, state))
    if state_before_decision is None:
        raise Issue56V4CorpusError("V4 trace lacks the requested decision state")
    final_state_domain = str(bundle.hmc_contract.data["control_trace"]["domains"]["final_state"])
    if _trace_final_state_digest(state, final_state_domain) != parsed.footer["final_state_sha256"]:
        raise Issue56V4CorpusError("V4 trace replay final state identity differs")
    return _V4SerializedTraceReplay(
        parsed,
        tuple(state_digests),
        np.stack(target_rows).astype(np.float64),
        state_before_decision,
        step_events,
        arbitration_events,
        proposal_events,
    )


def _expected_family_scenario(
    bundle: ForecastContracts,
    family_id: str,
) -> Scenario:
    roster = deterministic_family_ids(FAMILY_COUNT)
    if family_id not in roster:
        raise Issue56V4CorpusError("V4 family is outside the development roster")
    try:
        return build_family_scenario(bundle.development_scenario, roster.index(family_id))
    except Exception as error:
        raise Issue56V4CorpusError("V4 family scenario cannot be reconstructed") from error


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise Issue56V4CorpusError(f"V4 {label} is inconsistent")


def _verify_v4_label_against_trace(
    sample: V4RiskSample,
    details: _V4SerializedTraceReplay,
    bundle: ForecastContracts,
    scenario: Scenario,
) -> None:
    """Bind an action row to the proposal, HMC receipt, plant receipt, and labels."""

    action_by_id = {action.action_id: action for action in bundle.actions}
    try:
        action = action_by_id[sample.action_id]
    except KeyError as error:
        raise Issue56V4CorpusError("V4 sample action is not in the catalogue") from error
    proposal = details.proposal_events_by_step[sample.decision_step]["receipt"]
    arbitration = details.arbitration_events_by_step[sample.decision_step]["receipt"]
    step = details.step_events_by_step[sample.decision_step]["receipt"]
    label = sample.label
    command = action.command.to_mapping()

    _require_equal(proposal["attempt_class"], "CANONICAL_PROPOSAL", "proposal attempt")
    _require_equal(proposal["validation_outcome"], "VALID", "proposal validation")
    _require_equal(proposal["source_id"], sample.action_id, "proposal action identity")
    _require_equal(
        proposal["source_type"], "issue56-risk-v4-corpus", "proposal source type"
    )
    _require_equal(proposal["sequence"], sample.decision_step, "proposal decision step")
    _require_equal(
        proposal["requested_application_step"], sample.decision_step, "proposal application step"
    )
    _require_equal(proposal["requested_command_sha256"], action.command_sha256, "proposal command")
    if type(proposal["proposal"]) is not dict:
        raise Issue56V4CorpusError("V4 proposal payload is missing")
    _require_equal(proposal["proposal"]["source_id"], sample.action_id, "proposal payload action")
    _require_equal(
        proposal["proposal"]["proposed_command"], command, "proposal payload command"
    )
    _require_equal(proposal["proposal"]["requested_application_step"], sample.decision_step, "proposal payload step")

    _require_equal(arbitration["requested_command"], command, "arbitration requested command")
    _require_equal(
        arbitration["requested_command_sha256"], action.command_sha256, "arbitration requested command identity"
    )
    _require_equal(label.current_command_sha256, details.arbitration_events_by_step[sample.decision_step - 1]["receipt"]["final_command_sha256"], "current command identity")
    _require_equal(label.requested_command_sha256, action.command_sha256, "label requested command identity")
    _require_equal(label.final_command_sha256, arbitration["final_command_sha256"], "label final command identity")
    _require_equal(label.executed_command_sha256, step["returned_external_command_digest"], "label executed command identity")
    _require_equal(step["final_command_sha256"], arbitration["final_command_sha256"], "step final command identity")
    _require_equal(step["returned_external_command_digest"], arbitration["final_command_sha256"], "step returned command identity")
    _require_equal(label.disposition, _classify_v4_disposition(arbitration), "label disposition")

    for decision, proposal_event in details.proposal_events_by_step.items():
        if decision == sample.decision_step:
            continue
        _require_equal(
            proposal_event["receipt"]["attempt_class"], "NONE", "non-decision proposal absence"
        )
        _require_equal(
            proposal_event["receipt"]["validation_outcome"], "NO_PROPOSAL", "non-decision proposal outcome"
        )

    expected_state_digests = details.step_state_digests[sample.decision_step :]
    _require_equal(tuple(label.state_digests), expected_state_digests, "state provenance")
    targets = details.target_rows[sample.decision_step :]
    expected_horizons = tuple(
        V3HorizonMetric(horizon, *_replay_label_metrics(targets[:horizon]))
        for horizon in V3_HORIZONS
    )
    _require_equal(tuple(label.horizon_metrics), expected_horizons, "horizon labels")
    _require_equal(
        label.remaining_metric,
        V3HorizonMetric(len(targets), *_replay_label_metrics(targets)),
        "remaining label",
    )
    expected_metrics = _trajectory_metrics(
        scenario,
        scenario_zone_order(scenario),
        details.state_before_decision,
        targets,
        sample.decision_step,
    )
    _require_equal(sample.trajectory_metrics, expected_metrics, "action trajectory metrics")


def _verify_v4_hold_trace(
    sample: V4RiskSample,
    details: _V4SerializedTraceReplay,
    candidate_details: _V4SerializedTraceReplay,
    scenario: Scenario,
) -> None:
    """Verify the no-proposal hold comparator and its relative-action basis."""

    decision = sample.decision_step
    proposal = details.proposal_events_by_step[decision]["receipt"]
    arbitration = details.arbitration_events_by_step[decision]["receipt"]
    step = details.step_events_by_step[decision]["receipt"]
    previous_command = details.arbitration_events_by_step[decision - 1]["receipt"][
        "final_command_sha256"
    ]
    _require_equal(proposal["attempt_class"], "NONE", "hold proposal absence")
    _require_equal(proposal["validation_outcome"], "NO_PROPOSAL", "hold proposal outcome")
    _require_equal(arbitration["requested_command"], None, "hold requested command")
    _require_equal(arbitration["requested_command_sha256"], None, "hold requested command identity")
    _require_equal(arbitration["final_command_sha256"], previous_command, "hold command persistence")
    _require_equal(step["final_command_sha256"], previous_command, "hold step command identity")
    _require_equal(step["returned_external_command_digest"], previous_command, "hold plant command identity")
    _require_equal(
        details.step_state_digests[:decision],
        candidate_details.step_state_digests[:decision],
        "candidate and hold prefix states",
    )
    _require_equal(
        _sha(details.state_before_decision),
        _sha(candidate_details.state_before_decision),
        "candidate and hold decision state",
    )
    expected_metrics = _trajectory_metrics(
        scenario,
        scenario_zone_order(scenario),
        details.state_before_decision,
        details.target_rows[decision:],
        decision,
    )
    _require_equal(sample.hold_trajectory_metrics, expected_metrics, "hold trajectory metrics")
    expected_relative = V4RelativeActionTargets(
        sample.trajectory_metrics.safety_exposure - expected_metrics.safety_exposure,
        sample.trajectory_metrics.comfort_deviation - expected_metrics.comfort_deviation,
        sample.trajectory_metrics.resource_composite - expected_metrics.resource_composite,
    )
    _require_equal(sample.relative_action_targets, expected_relative, "relative action targets")


def _verify_v4_feature_bindings(
    sample: V4RiskSample,
    bundle: ForecastContracts,
    history: Any,
) -> None:
    """Bind stored model inputs to a freshly rebuilt observable history."""

    action = next(
        (item for item in bundle.actions if item.action_id == sample.action_id),
        None,
    )
    if action is None:
        raise Issue56V4CorpusError("V4 feature action is not in the catalogue")
    alarm_slots = alarm_family_slot_indices(bundle)
    action_vector = project_proposed_action(bundle, action.command)
    try:
        expected_baseline = v2_feature_vector(
            history,
            action_vector,
            decision_step=sample.decision_step,
            alarm_family_slots=alarm_slots,
        )
        expected_temporal = v4_temporal_feature_vector(
            history,
            action_vector,
            decision_step=sample.decision_step,
            alarm_family_slots=alarm_slots,
        )
        expected_mask = v4_observable_action_mask(bundle, history)
    except Exception as error:
        raise Issue56V4CorpusError("V4 observable feature reconstruction failed") from error
    if not np.array_equal(sample.features_f32, expected_baseline):
        raise Issue56V4CorpusError("V4 baseline feature projection differs from history")
    if not np.array_equal(sample.temporal_features_f32, expected_temporal):
        raise Issue56V4CorpusError("V4 temporal feature projection differs from history")
    _require_equal(sample.observable_action_mask, expected_mask, "observable action mask")


def _v4_feature_histories(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
) -> dict[int, Any]:
    """Rebuild one family's verified history windows for semantic validation."""

    try:
        snapshots, _, _, _ = _baseline_histories(bundle, scenario, family_id)
        return {
            step: project_history_window(
                bundle,
                tuple(
                    snapshots[index]
                    for index in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
                ),
                window_steps=HISTORY_WINDOW_STEPS,
            )
            for step in v2_decision_steps()
        }
    except Exception as error:
        raise Issue56V4CorpusError("V4 feature history cannot be reconstructed") from error


def verify_v4_sample_against_trace(
    sample: V4RiskSample,
    bundle: ForecastContracts,
    scenario: Scenario,
    *,
    feature_history: Any | None = None,
) -> dict[str, Any]:
    """Verify both retained traces and all semantic sample-to-trace bindings."""

    if type(sample) is not V4RiskSample or type(bundle) is not ForecastContracts:
        raise Issue56V4CorpusError("V4 semantic verification requires exact sample and bundle types")
    expected_scenario = _expected_family_scenario(bundle, sample.family_id)
    if scenario.scenario_sha256 != expected_scenario.scenario_sha256:
        raise Issue56V4CorpusError("V4 family scenario identity differs from the roster")
    candidate = _strict_replay_trace_details(
        sample.counterfactual_trace_bytes,
        bundle,
        scenario,
        family_id=sample.family_id,
        decision_step=sample.decision_step,
    )
    hold = _strict_replay_trace_details(
        sample.hold_trace_bytes,
        bundle,
        scenario,
        family_id=sample.family_id,
        decision_step=sample.decision_step,
    )
    return _verify_v4_sample_with_trace_details(
        sample,
        candidate,
        hold,
        bundle,
        scenario,
        feature_history=feature_history,
    )


def _verify_v4_sample_with_trace_details(
    sample: V4RiskSample,
    candidate: _V4SerializedTraceReplay,
    hold: _V4SerializedTraceReplay,
    bundle: ForecastContracts,
    scenario: Scenario,
    *,
    feature_history: Any | None = None,
) -> dict[str, Any]:
    """Verify row semantics using already replayed trace details."""

    if type(sample) is not V4RiskSample or type(bundle) is not ForecastContracts:
        raise Issue56V4CorpusError("V4 semantic verification requires exact sample and bundle types")
    expected_scenario = _expected_family_scenario(bundle, sample.family_id)
    if scenario.scenario_sha256 != expected_scenario.scenario_sha256:
        raise Issue56V4CorpusError("V4 family scenario identity differs from the roster")
    _verify_v4_label_against_trace(sample, candidate, bundle, scenario)
    _verify_v4_hold_trace(sample, hold, candidate, scenario)
    history = feature_history
    if history is None:
        history = _v4_feature_histories(bundle, scenario, sample.family_id)[sample.decision_step]
    _verify_v4_feature_bindings(sample, bundle, history)
    return {
        "counterfactual_trace_sha256": sample.counterfactual_trace_sha256,
        "hold_trace_sha256": sample.hold_trace_sha256,
        "decision_step": sample.decision_step,
        "replay_committed_steps": EPISODE_STEPS,
        "semantic_bindings_verified": True,
    }


def _classify_v4_disposition(mapping: Mapping[str, Any]) -> str:
    if bool(mapping.get("emergency_override")):
        return "EMERGENCY_OVERRIDDEN"
    disposition = mapping.get("disposition")
    if disposition == "ACCEPTED":
        return "PROPOSED_ACCEPTED"
    if disposition == "MODIFIED":
        return "PROPOSED_MODIFIED"
    if disposition == "REJECTED":
        return "PROPOSED_REJECTED_TO_HOLD"
    raise Issue56V4CorpusError("V4 HMC disposition is invalid")


def _replay_v4_branch(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    decision_step: int,
    action_id: str | None,
    command: Mapping[str, Any] | None,
    *,
    current_command_sha256: str,
    branch_hmc: HabitatManagementComputer,
    branch_state: Any,
) -> _V4BranchReplay:
    """Replay a candidate or no-proposal hold branch through the HMC."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise Issue56V4CorpusError("V4 branch requires frozen bundle and scenario")
    if decision_step not in v2_decision_steps():
        raise Issue56V4CorpusError("V4 branch decision step is invalid")
    _require_identifier(family_id, "V4 family")
    if (action_id is None) != (command is None):
        raise Issue56V4CorpusError("V4 branch action and command must be paired")
    if action_id is not None:
        _require_identifier(action_id, "V4 action")
    _require_sha(current_command_sha256, "V4 current command")
    if type(branch_hmc) is not HabitatManagementComputer or branch_state is None:
        raise Issue56V4CorpusError("V4 branch continuation state is malformed")
    if branch_hmc.lifecycle_phase != "OBSERVED" or branch_state.step != decision_step:
        raise Issue56V4CorpusError("V4 branch continuation is not at the decision")
    if command is None:
        requested_command = None
    else:
        try:
            requested_command = validate_external_command(scenario, command)
        except Exception as error:
            raise Issue56V4CorpusError("V4 branch command is invalid") from error

    hmc = _clone_hmc_for_branch(branch_hmc)
    hmc._verified_snapshot_handle = None  # noqa: SLF001 - cloned capability reset
    shadow = branch_state
    zone_ids = scenario_zone_order(scenario)
    target_rows: list[np.ndarray] = []
    state_digests: list[str] = []
    requested_sha: str | None = None
    final_sha: str | None = None
    executed_sha: str | None = None
    disposition: str | None = None
    for step in range(decision_step, EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56V4CorpusError(f"V4 HMC terminated before step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        proposal = None
        if step == decision_step and action_id is not None and command is not None:
            proposal = {
                "schema_version": "aeolus_habitat_v2_control_proposal_v1",
                "control_run_id": hmc.control_run_id,
                "authority_epoch": hmc.authority_epoch,
                "source_id": action_id,
                "source_type": "issue56-risk-v4-corpus",
                "completed_observation_step": step,
                "observation_snapshot_sha256": snapshot.snapshot_sha256,
                "requested_application_step": step,
                "observable_topology_sha256": hmc.observable_topology_sha256,
                "proposed_command": command,
                "confidence": None,
            }
        if proposal is not None:
            proposal = {**proposal, "proposal_sha256": _sha(proposal)}
        hmc.propose(proposal, handle)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56V4CorpusError(f"V4 HMC terminated while arbitrating step {step}")
        if step == decision_step:
            if requested_command is not None:
                requested_sha = requested_command.sha256
            final_sha = arbitration.final_command_sha256
            if action_id is not None:
                disposition = _classify_v4_disposition(arbitration.to_mapping())
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56V4CorpusError(f"V4 HMC terminated while stepping {step}")
        shadow_result = advance_one_step_with_command(
            scenario,
            shadow,
            arbitration.final_command,
        )
        if _sha_bytes(canonical_json_bytes(shadow_result.receipt)) != stepped.plant_receipt_digest:
            raise Issue56V4CorpusError("V4 branch shadow replay diverged")
        shadow = shadow_result.state
        if step == decision_step and action_id is not None:
            executed_sha = str(shadow_result.receipt["external_command_digest"])
        target_rows.append(project_true_targets(scenario, zone_ids, shadow))
        state_digests.append(_sha(shadow))

    if (
        action_id is not None
        and (
            requested_sha is None
            or final_sha is None
            or executed_sha is None
            or disposition is None
        )
    ):
        raise Issue56V4CorpusError("V4 branch did not record decision provenance")
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract)
    replay = replay_control_trace(trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract)
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56V4CorpusError("V4 branch trace failed strict replay")

    targets = np.stack(target_rows).astype(np.float64)
    trajectory_metrics = _trajectory_metrics(
        scenario,
        zone_ids,
        branch_state,
        targets,
        decision_step,
    )
    if action_id is None:
        return _V4BranchReplay(None, trace.canonical_bytes, trajectory_metrics)
    metrics = tuple(V3HorizonMetric(horizon, *_replay_label_metrics(targets[:horizon])) for horizon in V3_HORIZONS)
    remaining_metric = V3HorizonMetric(len(targets), *_replay_label_metrics(targets))
    trace_sha256 = _sha_bytes(trace.canonical_bytes)
    body = {
        "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.label",
        "track": V3_LABEL_TRACK,
        "action_id": action_id,
        "decision_step": decision_step,
        "current_command_sha256": current_command_sha256,
        "requested_command_sha256": requested_sha,
        "final_command_sha256": final_sha,
        "executed_command_sha256": executed_sha,
        "disposition": disposition,
        "horizon_metrics": [metric.to_mapping() for metric in metrics],
        "remaining_steps": len(targets),
        "remaining_metric": remaining_metric.to_mapping(),
        "state_digests": state_digests,
        "trace_sha256": trace_sha256,
    }
    label = V3PolicyLabel(
        action_id,
        decision_step,
        current_command_sha256,
        requested_sha,
        final_sha,
        executed_sha,
        disposition,
        metrics,
        len(targets),
        remaining_metric,
        tuple(state_digests),
        trace_sha256,
        _sha(body),
    )
    return _V4BranchReplay(label, trace.canonical_bytes, trajectory_metrics)


def collect_v4_family_samples(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    *,
    split: str,
) -> tuple[V4RiskSample, ...]:
    """Collect action-conditioned features and replayable policy labels."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise Issue56V4CorpusError("V4 collection requires frozen inputs")
    if split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
        raise Issue56V4CorpusError("V4 collection split is invalid")
    if family_id not in deterministic_family_ids(32):
        raise Issue56V4CorpusError("V4 family is outside the development roster")
    actions = tuple(bundle.actions)
    if len(actions) != 4 or len({action.action_id for action in actions}) != 4:
        raise Issue56V4CorpusError("V4 collection requires four unique actions")
    snapshots, command_sha_by_step, branch_hmcs, branch_states = _baseline_histories(
        bundle,
        scenario,
        family_id,
    )
    alarm_slots = alarm_family_slot_indices(bundle)
    samples: list[V4RiskSample] = []
    for step in v2_decision_steps():
        history = project_history_window(
            bundle,
            tuple(
                snapshots[index]
                for index in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
            ),
            window_steps=HISTORY_WINDOW_STEPS,
        )
        current_sha = command_sha_by_step[step]
        hold_replay = _replay_v4_branch(
            bundle,
            scenario,
            family_id,
            step,
            None,
            None,
            current_command_sha256=current_sha,
            branch_hmc=branch_hmcs[step],
            branch_state=branch_states[step],
        )
        hold_trace_sha256 = _sha_bytes(hold_replay.trace_canonical_bytes)
        hold_trace_path = _trace_relative_path(hold_trace_sha256)
        for action in actions:
            action_vector = project_proposed_action(bundle, action.command)
            features = v2_feature_vector(
                history,
                action_vector,
                decision_step=step,
                alarm_family_slots=alarm_slots,
            )
            temporal_features = v4_temporal_feature_vector(
                history,
                action_vector,
                decision_step=step,
                alarm_family_slots=alarm_slots,
            )
            observable_action_mask = v4_observable_action_mask(bundle, history)
            replay = _replay_counterfactual_branch(
                bundle,
                scenario,
                family_id,
                step,
                action.action_id,
                action.command.to_mapping(),
                current_command_sha256=current_sha,
                branch_hmc=branch_hmcs[step],
                branch_state=branch_states[step],
            )
            base_body = {
                "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.sample",
                "family_id": family_id,
                "decision_step": step,
                "split": split,
                "action_id": action.action_id,
                "scenario_sha256": scenario.scenario_sha256,
                "features_f32_hex": np.asarray(features, dtype=np.float32).tobytes().hex(),
                "label": replay.label.to_mapping(),
            }
            base_sample = V3RiskSample(
                family_id,
                step,
                split,
                action.action_id,
                scenario.scenario_sha256,
                np.asarray(features, dtype=np.float32),
                replay.label,
                _sha(base_body),
            )
            trace_path = _trace_relative_path(replay.label.trace_sha256)
            sample_body = {
                "schema_version": f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.sample",
                "base_sample": base_sample.to_mapping(),
                "counterfactual_trace_relative_path": trace_path,
                "counterfactual_trace_sha256": replay.label.trace_sha256,
                "hold_trace_relative_path": hold_trace_path,
                "hold_trace_sha256": hold_trace_sha256,
                "temporal_features_f32_hex": temporal_features.tobytes().hex(),
                "observable_action_mask": list(observable_action_mask),
                "trajectory_metrics": replay.trajectory_metrics.to_mapping(),
                "hold_trajectory_metrics": hold_replay.trajectory_metrics.to_mapping(),
                "relative_action_targets": V4RelativeActionTargets(
                    replay.trajectory_metrics.safety_exposure
                    - hold_replay.trajectory_metrics.safety_exposure,
                    replay.trajectory_metrics.comfort_deviation
                    - hold_replay.trajectory_metrics.comfort_deviation,
                    replay.trajectory_metrics.resource_composite
                    - hold_replay.trajectory_metrics.resource_composite,
                ).to_mapping(),
            }
            samples.append(
                V4RiskSample(
                    base_sample,
                    trace_path,
                    replay.label.trace_sha256,
                    replay.trace_canonical_bytes,
                    hold_trace_path,
                    hold_trace_sha256,
                    hold_replay.trace_canonical_bytes,
                    temporal_features,
                    observable_action_mask,
                    replay.trajectory_metrics,
                    hold_replay.trajectory_metrics,
                    V4RelativeActionTargets(
                        replay.trajectory_metrics.safety_exposure
                        - hold_replay.trajectory_metrics.safety_exposure,
                        replay.trajectory_metrics.comfort_deviation
                        - hold_replay.trajectory_metrics.comfort_deviation,
                        replay.trajectory_metrics.resource_composite
                        - hold_replay.trajectory_metrics.resource_composite,
                    ),
                    _sha(sample_body),
                )
            )
    return tuple(samples)


def verify_v4_serialized_trace(
    trace_canonical_bytes: bytes,
    bundle: ForecastContracts,
    scenario: Scenario,
) -> dict[str, Any]:
    """Independently verify a serialized counterfactual trace artifact."""

    if type(trace_canonical_bytes) is not bytes or not trace_canonical_bytes:
        raise Issue56V4CorpusError("V4 serialized trace bytes are invalid")
    try:
        parsed = parse_control_trace(
            trace_canonical_bytes,
            scenario=scenario,
            contract=bundle.hmc_contract,
        )
        replay = replay_control_trace(
            trace_canonical_bytes,
            scenario=scenario,
            contract=bundle.hmc_contract,
        )
    except Exception as error:
        raise Issue56V4CorpusError("V4 serialized trace failed strict replay") from error
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56V4CorpusError("V4 serialized trace replay receipt is inconsistent")
    return {
        "trace_sha256": _sha_bytes(trace_canonical_bytes),
        "replay_committed_steps": replay.committed_step_count,
        "replay_final_state_sha256": replay.final_state_sha256,
    }


def load_v4_samples(
    rows: Sequence[Mapping[str, Any]],
    corpus_root: Path,
    bundle: ForecastContracts,
    scenarios: Mapping[str, Scenario],
) -> tuple[V4RiskSample, ...]:
    """Load rows and independently verify each retained trace artifact."""

    if isinstance(rows, (str, bytes)):
        raise Issue56V4CorpusError("V4 sample rows must be a sequence")
    if not isinstance(corpus_root, Path):
        raise Issue56V4CorpusError("V4 corpus root must be a Path")
    if type(bundle) is not ForecastContracts:
        raise Issue56V4CorpusError("V4 sample loader requires frozen contracts")
    root = corpus_root.resolve()
    trace_root = root / V4_CORPUS_TRACE_DIRECTORY
    if _trace_path_contains_symlink(trace_root, root):
        raise Issue56V4CorpusError("V4 trace directory must not be a symlink")
    samples: list[V4RiskSample] = []
    keys: set[tuple[str, int, str]] = set()
    feature_histories: dict[str, dict[int, Any]] = {}
    candidate_details: dict[tuple[str, int, str], _V4SerializedTraceReplay] = {}
    hold_details: dict[tuple[str, int, str], _V4SerializedTraceReplay] = {}
    for row in rows:
        if type(row) is not dict:
            raise Issue56V4CorpusError("V4 sample row must be an object")
        path_value = _validate_trace_relative_path(
            row.get("counterfactual_trace_relative_path"), "V4 counterfactual trace"
        )
        hold_path_value = _validate_trace_relative_path(
            row.get("hold_trace_relative_path"), "V4 hold trace"
        )
        trace_path = root / path_value
        hold_trace_path = root / hold_path_value
        if (
            _trace_path_contains_symlink(trace_path, root)
            or _trace_path_contains_symlink(hold_trace_path, root)
            or trace_path.parent != trace_root
            or hold_trace_path.parent != trace_root
        ):
            raise Issue56V4CorpusError("V4 sample trace path uses a symlink or escaped the corpus directory")
        try:
            trace_bytes = trace_path.read_bytes()
            hold_trace_bytes = hold_trace_path.read_bytes()
        except OSError as error:
            raise Issue56V4CorpusError(
                f"V4 sample trace is missing: {path_value} or {hold_path_value}"
            ) from error
        sample = V4RiskSample.from_mapping(row, trace_bytes, hold_trace_bytes)
        key = (sample.family_id, sample.decision_step, sample.action_id)
        if key in keys:
            raise Issue56V4CorpusError("V4 samples contain duplicate decision/action rows")
        keys.add(key)
        try:
            scenario = scenarios[sample.family_id]
        except KeyError as error:
            raise Issue56V4CorpusError("V4 sample family lacks a scenario") from error
        if sample.base_sample.scenario_sha256 != scenario.scenario_sha256:
            raise Issue56V4CorpusError("V4 sample scenario identity differs")
        if sample.family_id not in feature_histories:
            feature_histories[sample.family_id] = _v4_feature_histories(
                bundle,
                scenario,
                sample.family_id,
            )
        try:
            history = feature_histories[sample.family_id][sample.decision_step]
        except KeyError as error:
            raise Issue56V4CorpusError("V4 sample decision lacks a verified history") from error
        candidate_key = (
            sample.family_id,
            sample.decision_step,
            sample.counterfactual_trace_sha256,
        )
        if candidate_key not in candidate_details:
            candidate_details[candidate_key] = _strict_replay_trace_details(
                sample.counterfactual_trace_bytes,
                bundle,
                scenario,
                family_id=sample.family_id,
                decision_step=sample.decision_step,
            )
        hold_key = (sample.family_id, sample.decision_step, sample.hold_trace_sha256)
        if hold_key not in hold_details:
            hold_details[hold_key] = _strict_replay_trace_details(
                sample.hold_trace_bytes,
                bundle,
                scenario,
                family_id=sample.family_id,
                decision_step=sample.decision_step,
            )
        verified = _verify_v4_sample_with_trace_details(
            sample,
            candidate_details[candidate_key],
            hold_details[hold_key],
            bundle,
            scenario,
            feature_history=history,
        )
        if (
            verified["counterfactual_trace_sha256"] != sample.counterfactual_trace_sha256
            or verified["hold_trace_sha256"] != sample.hold_trace_sha256
        ):
            raise Issue56V4CorpusError("V4 sample trace verification identity differs")
        samples.append(sample)
    if not samples:
        raise Issue56V4CorpusError("V4 sample loader requires samples")
    return tuple(samples)


__all__ = [
    "ISSUE56_V4_CORPUS_SCHEMA_VERSION",
    "V4_CORPUS_TRACE_DIRECTORY",
    "Issue56V4CorpusError",
    "V4CounterfactualReplay",
    "V4RelativeActionTargets",
    "V4RiskSample",
    "V4TrajectoryMetrics",
    "collect_v4_family_samples",
    "load_v4_samples",
    "v4_label_manifest",
    "verify_v4_sample_against_trace",
    "verify_v4_serialized_trace",
]
