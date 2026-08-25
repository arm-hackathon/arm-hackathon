"""Policy-aligned action-risk labels for the next Issue #56 study.

V2 remains frozen.  V3 keeps the model advisory-only, but labels each catalogue
proposal by replaying the actual HMC lifecycle: the requested command is
validated, arbitrated, executed, and then retained by the no-proposal hold
policy until the end of the episode.  This prevents a short counterfactual
label from being mistaken for the command trajectory the policy will execute.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .control_trace import parse_control_trace, replay_control_trace
from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast.projection import project_history_window, project_proposed_action
from .forecast_issue55_race import (
    EPISODE_STEPS,
    HMC_IMPLEMENTATION_GIT_SHA,
    _crossings as true_crossings,
    episode_nonce,
    project_true_targets,
    scenario_zone_order,
)
from .forecast_issue56_action_risk_v2 import (
    ACTION_COUNT,
    ALARM_FAMILY_ORDER,
    FEATURE_COUNT,
    HISTORY_WINDOW_STEPS,
    alarm_family_slot_indices,
    v2_decision_steps,
    v2_feature_vector,
)
from .hmc import HabitatManagementComputer
from .physics import advance_one_step_with_command, initial_state, validate_external_command
from .scenario import Scenario


ISSUE56_V3_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v3_v1"
PREREGISTRATION_ID = "habitat_v2_forecast_issue_56_v3_preregistration_v1"
MODEL_SOURCE_TYPE = "issue56-risk-filtered-point-v3"
V3_HORIZONS = (4, 16, 32)
V3_LABEL_TRACK = "hmc_persistent_remaining"
V3_OUTCOME_TYPES = (
    "PROPOSED_ACCEPTED",
    "PROPOSED_MODIFIED",
    "PROPOSED_REJECTED_TO_HOLD",
    "EMERGENCY_OVERRIDDEN",
)


class Issue56V3RiskError(ValueError):
    """Raised when a V3 policy-aligned artifact or replay is malformed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _jsonable(value.to_mapping())
    if hasattr(value, "__dataclass_fields__"):
        return {
            str(name): _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, bytes):
        return value.hex()
    return value


def _sha(value: Any) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(_jsonable(value))).hexdigest()
    except (TypeError, ValueError) as error:
        raise Issue56V3RiskError("V3 digest input is not canonical finite JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise Issue56V3RiskError(f"{label} must be lowercase SHA-256")
    return value


def _finite_features(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise Issue56V3RiskError("V3 feature vector is malformed or non-finite")
    result = result.astype(np.float32)
    result.setflags(write=False)
    return result


def _label_metrics(targets: np.ndarray) -> tuple[float, float, float]:
    crossings = true_crossings(np.asarray(targets, dtype=np.float64))
    return (
        float(np.any(crossings > 0.0)),
        float(np.sum(crossings)),
        float(np.max(crossings)),
    )


@dataclass(frozen=True, slots=True)
class V3HorizonMetric:
    horizon_steps: int
    crossing_event: float
    safety_exposure: float
    maximum_crossing: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizon_steps, bool)
            or not isinstance(self.horizon_steps, int)
            or self.horizon_steps < 1
        ):
            raise Issue56V3RiskError("V3 horizon is invalid")
        if float(self.crossing_event) not in {0.0, 1.0}:
            raise Issue56V3RiskError("V3 horizon event is not binary")
        for value, label in (
            (self.safety_exposure, "V3 horizon exposure"),
            (self.maximum_crossing, "V3 horizon maximum crossing"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V3RiskError(f"{label} is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "horizon_steps": self.horizon_steps,
            "crossing_event": self.crossing_event,
            "safety_exposure": self.safety_exposure,
            "maximum_crossing": self.maximum_crossing,
        }


@dataclass(frozen=True, slots=True)
class V3PolicyLabel:
    action_id: str
    decision_step: int
    current_command_sha256: str
    requested_command_sha256: str
    final_command_sha256: str
    executed_command_sha256: str
    disposition: str
    horizon_metrics: tuple[V3HorizonMetric, ...]
    remaining_steps: int
    remaining_metric: V3HorizonMetric
    state_digests: tuple[str, ...]
    trace_sha256: str
    label_sha256: str

    @property
    def track(self) -> str:
        return V3_LABEL_TRACK

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise Issue56V3RiskError("V3 label action identity is invalid")
        if (
            isinstance(self.decision_step, bool)
            or not isinstance(self.decision_step, int)
            or self.decision_step not in v2_decision_steps()
        ):
            raise Issue56V3RiskError("V3 label decision step is invalid")
        for value, label in (
            (self.current_command_sha256, "V3 current command"),
            (self.requested_command_sha256, "V3 requested command"),
            (self.final_command_sha256, "V3 final command"),
            (self.executed_command_sha256, "V3 executed command"),
            (self.trace_sha256, "V3 trace"),
            (self.label_sha256, "V3 label"),
        ):
            _require_sha(value, label)
        if self.disposition not in V3_OUTCOME_TYPES:
            raise Issue56V3RiskError("V3 arbitration disposition is invalid")
        if tuple(metric.horizon_steps for metric in self.horizon_metrics) != V3_HORIZONS:
            raise Issue56V3RiskError("V3 fixed horizon metrics drifted")
        if self.remaining_steps != EPISODE_STEPS - self.decision_step:
            raise Issue56V3RiskError("V3 remaining horizon is inconsistent")
        if self.remaining_metric.horizon_steps != self.remaining_steps:
            raise Issue56V3RiskError("V3 remaining metric horizon is inconsistent")
        if len(self.state_digests) != self.remaining_steps:
            raise Issue56V3RiskError("V3 state provenance is incomplete")
        for digest in self.state_digests:
            _require_sha(digest, "V3 state")
        if self.label_sha256 != _sha(self._body()):
            raise Issue56V3RiskError("V3 label digest is inconsistent")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.label",
            "track": V3_LABEL_TRACK,
            "action_id": self.action_id,
            "decision_step": self.decision_step,
            "current_command_sha256": self.current_command_sha256,
            "requested_command_sha256": self.requested_command_sha256,
            "final_command_sha256": self.final_command_sha256,
            "executed_command_sha256": self.executed_command_sha256,
            "disposition": self.disposition,
            "horizon_metrics": [metric.to_mapping() for metric in self.horizon_metrics],
            "remaining_steps": self.remaining_steps,
            "remaining_metric": self.remaining_metric.to_mapping(),
            "state_digests": list(self.state_digests),
            "trace_sha256": self.trace_sha256,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "label_sha256": self.label_sha256}


def build_v3_proposal(
    hmc: HabitatManagementComputer,
    snapshot_sha256: str,
    step: int,
    command: Mapping[str, Any],
    action_id: str,
) -> dict[str, Any]:
    """Build one advisory proposal bound to the current verified snapshot."""

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


def _classify_outcome(arbitration: Any) -> str:
    if bool(arbitration.to_mapping().get("emergency_override")):
        return "EMERGENCY_OVERRIDDEN"
    disposition = str(arbitration.to_mapping()["disposition"])
    if disposition == "ACCEPTED":
        return "PROPOSED_ACCEPTED"
    if disposition == "MODIFIED":
        return "PROPOSED_MODIFIED"
    if disposition == "REJECTED":
        return "PROPOSED_REJECTED_TO_HOLD"
    raise Issue56V3RiskError(f"unknown HMC disposition: {disposition}")


def replay_v3_policy_branch(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    decision_step: int,
    action_id: str,
    command: Mapping[str, Any],
    *,
    current_command_sha256: str,
    verify_trace: bool = False,
) -> V3PolicyLabel:
    """Replay one proposal and the HMC no-proposal persistence policy."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise Issue56V3RiskError("V3 branch requires frozen bundle and scenario")
    if decision_step not in v2_decision_steps():
        raise Issue56V3RiskError("V3 branch decision step is invalid")
    _require_sha(current_command_sha256, "V3 current command")
    try:
        requested_command = validate_external_command(scenario, command)
    except Exception as error:
        raise Issue56V3RiskError("V3 branch command is invalid") from error
    zone_ids = scenario_zone_order(scenario)
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce(family_id),
    )
    shadow = initial_state(scenario)
    target_rows: list[np.ndarray] = []
    state_digests: list[str] = []
    requested_sha: str | None = None
    final_sha: str | None = None
    executed_sha: str | None = None
    disposition: str | None = None
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56V3RiskError(f"HMC terminated before V3 step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        proposal = (
            build_v3_proposal(
                hmc,
                snapshot.snapshot_sha256,
                step,
                requested_command.to_mapping(),
                action_id,
            )
            if step == decision_step
            else None
        )
        hmc.propose(proposal, handle)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56V3RiskError(f"HMC terminated while arbitrating V3 step {step}")
        if step == decision_step:
            requested_sha = requested_command.sha256
            final_sha = arbitration.final_command_sha256
            disposition = _classify_outcome(arbitration)
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56V3RiskError(f"HMC terminated while stepping V3 step {step}")
        shadow_result = advance_one_step_with_command(
            scenario,
            shadow,
            arbitration.final_command,
        )
        if _sha_bytes(canonical_json_bytes(shadow_result.receipt)) != stepped.plant_receipt_digest:
            raise Issue56V3RiskError("V3 branch shadow replay diverged")
        shadow = shadow_result.state
        if step >= decision_step:
            if step == decision_step:
                executed_sha = str(shadow_result.receipt["external_command_digest"])
            target_rows.append(project_true_targets(scenario, zone_ids, shadow))
            state_digests.append(_sha(shadow))
    if (
        requested_sha is None
        or final_sha is None
        or executed_sha is None
        or disposition is None
    ):
        raise Issue56V3RiskError("V3 branch did not record decision provenance")
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    trace_sha = _sha_bytes(trace.canonical_bytes)
    if verify_trace:
        parsed = parse_control_trace(
            trace.canonical_bytes,
            scenario=scenario,
            contract=bundle.hmc_contract,
        )
        replay = replay_control_trace(
            trace.canonical_bytes,
            scenario=scenario,
            contract=bundle.hmc_contract,
        )
        if (
            parsed.footer["terminal_status"] != "COMPLETED"
            or replay.committed_step_count != EPISODE_STEPS
            or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
        ):
            raise Issue56V3RiskError("V3 branch trace failed strict replay")
    targets = np.stack(target_rows).astype(np.float64)
    metrics = tuple(
        V3HorizonMetric(horizon, *_label_metrics(targets[:horizon]))
        for horizon in V3_HORIZONS
    )
    remaining_metric = V3HorizonMetric(
        len(targets),
        *_label_metrics(targets),
    )
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
        "trace_sha256": trace_sha,
    }
    return V3PolicyLabel(
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
        trace_sha,
        _sha(body),
    )


def _set_current_command(label: V3PolicyLabel, current_command_sha256: str) -> V3PolicyLabel:
    _require_sha(current_command_sha256, "V3 current command")
    body = label._body()
    body["current_command_sha256"] = current_command_sha256
    return V3PolicyLabel(
        label.action_id,
        label.decision_step,
        current_command_sha256,
        label.requested_command_sha256,
        label.final_command_sha256,
        label.executed_command_sha256,
        label.disposition,
        label.horizon_metrics,
        label.remaining_steps,
        label.remaining_metric,
        label.state_digests,
        label.trace_sha256,
        _sha(body),
    )


@dataclass(frozen=True, slots=True)
class V3RiskSample:
    family_id: str
    decision_step: int
    split: str
    action_id: str
    scenario_sha256: str
    features_f32: np.ndarray
    label: V3PolicyLabel
    sample_sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise Issue56V3RiskError("V3 sample family is invalid")
        if self.split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
            raise Issue56V3RiskError("V3 sample split is invalid")
        if self.decision_step != self.label.decision_step or self.action_id != self.label.action_id:
            raise Issue56V3RiskError("V3 sample and label identities differ")
        _require_sha(self.scenario_sha256, "V3 sample scenario")
        features = _finite_features(self.features_f32)
        object.__setattr__(self, "features_f32", features)
        _require_sha(self.sample_sha256, "V3 sample")
        body = {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.sample",
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "split": self.split,
            "action_id": self.action_id,
            "scenario_sha256": self.scenario_sha256,
            "features_f32_hex": features.tobytes().hex(),
            "label": self.label.to_mapping(),
        }
        if self.sample_sha256 != _sha(body):
            raise Issue56V3RiskError("V3 sample digest is inconsistent")

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.sample",
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "split": self.split,
            "action_id": self.action_id,
            "scenario_sha256": self.scenario_sha256,
            "features_f32_hex": self.features_f32.tobytes().hex(),
            "label": self.label.to_mapping(),
        }
        return {**body, "sample_sha256": self.sample_sha256}


def _baseline_histories(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
) -> tuple[dict[int, tuple[Any, Any]], dict[int, str]]:
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce(family_id),
    )
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    command_sha_by_step: dict[int, str] = {}
    last_command_sha: str | None = None
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56V3RiskError(f"HMC terminated during V3 baseline at {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snapshots[step] = (snapshot, verification)
        if step in v2_decision_steps():
            if last_command_sha is None:
                raise Issue56V3RiskError("V3 baseline lacks a current command")
            command_sha_by_step[step] = last_command_sha
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56V3RiskError("V3 baseline arbitration failed")
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56V3RiskError("V3 baseline step failed")
        shadow_result = advance_one_step_with_command(
            scenario,
            shadow,
            arbitration.final_command,
        )
        if _sha_bytes(canonical_json_bytes(shadow_result.receipt)) != stepped.plant_receipt_digest:
            raise Issue56V3RiskError("V3 baseline shadow replay diverged")
        shadow = shadow_result.state
        last_command_sha = arbitration.final_command_sha256
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes,
        scenario=scenario,
        contract=bundle.hmc_contract,
    )
    replay = replay_control_trace(
        trace.canonical_bytes,
        scenario=scenario,
        contract=bundle.hmc_contract,
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
    ):
        raise Issue56V3RiskError("V3 baseline trace failed strict replay")
    return snapshots, command_sha_by_step


def collect_v3_family_samples(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    *,
    split: str,
) -> tuple[V3RiskSample, ...]:
    """Collect baseline-history features and HMC-persistent labels."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise Issue56V3RiskError("V3 collection requires frozen inputs")
    if split not in {"TRAIN", "VALIDATION", "EVALUATION"}:
        raise Issue56V3RiskError("V3 collection split is invalid")
    actions = tuple(bundle.actions)
    if len(actions) != 4 or len({action.action_id for action in actions}) != 4:
        raise Issue56V3RiskError("V3 collection requires four unique actions")
    snapshots, command_sha_by_step = _baseline_histories(bundle, scenario, family_id)
    alarm_slots = alarm_family_slot_indices(bundle)
    samples: list[V3RiskSample] = []
    for step in v2_decision_steps():
        pairs = tuple(
            snapshots[index]
            for index in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
        )
        history = project_history_window(bundle, pairs, window_steps=HISTORY_WINDOW_STEPS)
        current_sha = command_sha_by_step[step]
        for action in actions:
            action_vector = project_proposed_action(bundle, action.command)
            features = v2_feature_vector(
                history,
                action_vector,
                decision_step=step,
                alarm_family_slots=alarm_slots,
            )
            label = replay_v3_policy_branch(
                bundle,
                scenario,
                family_id,
                step,
                action.action_id,
                action.command.to_mapping(),
                current_command_sha256=current_sha,
            )
            label = _set_current_command(label, current_sha)
            body = {
                "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.sample",
                "family_id": family_id,
                "decision_step": step,
                "split": split,
                "action_id": action.action_id,
                "scenario_sha256": scenario.scenario_sha256,
                "features_f32_hex": np.asarray(features, dtype=np.float32).tobytes().hex(),
                "label": label.to_mapping(),
            }
            samples.append(
                V3RiskSample(
                    family_id,
                    step,
                    split,
                    action.action_id,
                    scenario.scenario_sha256,
                    np.asarray(features, dtype=np.float32),
                    label,
                    _sha(body),
                )
            )
    return tuple(samples)


__all__ = [
    "ACTION_COUNT",
    "ALARM_FAMILY_ORDER",
    "FEATURE_COUNT",
    "Issue56V3RiskError",
    "ISSUE56_V3_SCHEMA_VERSION",
    "MODEL_SOURCE_TYPE",
    "PREREGISTRATION_ID",
    "V3_HORIZONS",
    "V3_LABEL_TRACK",
    "V3HorizonMetric",
    "V3PolicyLabel",
    "V3RiskSample",
    "build_v3_proposal",
    "collect_v3_family_samples",
    "replay_v3_policy_branch",
]
