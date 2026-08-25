"""Policy-aligned action-risk labels for the next Issue #56 study.

V2 remains frozen.  V3 keeps the model advisory-only, but labels each catalogue
proposal by replaying the actual HMC lifecycle: the requested command is
validated, arbitrated, executed, and then retained by the no-proposal hold
policy until the end of the episode.  This prevents a short counterfactual
label from being mistaken for the command trajectory the policy will execute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
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
    compute_race_metrics,
    deterministic_family_ids,
    episode_nonce,
    project_true_targets,
    rank_actions_advisory,
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
from .forecast_issue52 import _command_vector
from .forecast_issue55_race import score_point_prediction
from .hmc import HabitatManagementComputer
from .physics import advance_one_step_with_command, initial_state, validate_external_command
from .scenario import Scenario


ISSUE56_V3_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v3_v2"
PREREGISTRATION_ID = "habitat_v2_forecast_issue_56_v3_preregistration_v2"
MODEL_SOURCE_TYPE = "issue56-risk-filtered-point-v3"
V3_HORIZONS = (4, 16, 32)
V3_LABEL_TRACK = "hmc_persistent_remaining"
V3_ARMS = ("risk_only_v3", "risk_filtered_point_v3")
V3_OUTCOME_TYPES = (
    "PROPOSED_ACCEPTED",
    "PROPOSED_MODIFIED",
    "PROPOSED_REJECTED_TO_HOLD",
    "EMERGENCY_OVERRIDDEN",
)
V3_DECISION_OUTCOME_TYPES = V3_OUTCOME_TYPES + ("ABSTAINED_TO_HOLD",)


def v3_family_split(family_ids: Sequence[str]) -> dict[str, str]:
    """Split paired sensor variants using a fixed condition-stratified roster."""

    ids = tuple(family_ids)
    if not ids or len(set(ids)) != len(ids):
        raise Issue56V3RiskError("V3 family roster must be unique and non-empty")
    if ids != deterministic_family_ids(32):
        raise Issue56V3RiskError("V3 support-stratified split requires the canonical family roster")
    groups: dict[str, list[str]] = {}
    for index, family_id in enumerate(ids):
        if type(family_id) is not str or not family_id:
            raise Issue56V3RiskError("V3 family identity is invalid")
        group = f"condition-group-{index // 2:04d}"
        groups.setdefault(group, []).append(family_id)
    if len(groups) != 16 or any(len(group) != 2 for group in groups.values()):
        raise Issue56V3RiskError(
            "V3 support-stratified split requires the canonical 32-family roster"
        )
    labels = (
        "VALIDATION",
        "EVALUATION",
        "TRAIN",
        "TRAIN",
        "VALIDATION",
        "EVALUATION",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "VALIDATION",
        "EVALUATION",
        "TRAIN",
        "TRAIN",
    )
    result: dict[str, str] = {}
    for index, group in enumerate(groups.values()):
        for family_id in group:
            result[family_id] = labels[index]
    return dict(sorted(result.items()))


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
        if self.final_command_sha256 != self.executed_command_sha256:
            raise Issue56V3RiskError("V3 final and executed command identities differ")
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

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V3PolicyLabel":
        expected = {
            "schema_version",
            "track",
            "action_id",
            "decision_step",
            "current_command_sha256",
            "requested_command_sha256",
            "final_command_sha256",
            "executed_command_sha256",
            "disposition",
            "horizon_metrics",
            "remaining_steps",
            "remaining_metric",
            "state_digests",
            "trace_sha256",
            "label_sha256",
        }
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V3RiskError("V3 label fields drift")
        if (
            mapping["schema_version"] != f"{ISSUE56_V3_SCHEMA_VERSION}.label"
            or mapping["track"] != V3_LABEL_TRACK
            or type(mapping["horizon_metrics"]) is not list
            or type(mapping["remaining_metric"]) is not dict
            or type(mapping["state_digests"]) is not list
        ):
            raise Issue56V3RiskError("V3 label schema is invalid")
        metric_fields = {
            "horizon_steps",
            "crossing_event",
            "safety_exposure",
            "maximum_crossing",
        }
        metrics = []
        for item in mapping["horizon_metrics"]:
            if type(item) is not dict or set(item) != metric_fields:
                raise Issue56V3RiskError("V3 horizon metric fields drift")
            metrics.append(
                V3HorizonMetric(
                    item["horizon_steps"],
                    item["crossing_event"],
                    item["safety_exposure"],
                    item["maximum_crossing"],
                )
            )
        remaining = mapping["remaining_metric"]
        if set(remaining) != metric_fields:
            raise Issue56V3RiskError("V3 remaining metric fields drift")
        return cls(
            mapping["action_id"],
            mapping["decision_step"],
            mapping["current_command_sha256"],
            mapping["requested_command_sha256"],
            mapping["final_command_sha256"],
            mapping["executed_command_sha256"],
            mapping["disposition"],
            tuple(metrics),
            mapping["remaining_steps"],
            V3HorizonMetric(
                remaining["horizon_steps"],
                remaining["crossing_event"],
                remaining["safety_exposure"],
                remaining["maximum_crossing"],
            ),
            tuple(mapping["state_digests"]),
            mapping["trace_sha256"],
            mapping["label_sha256"],
        )


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


def _clone_hmc_for_branch(hmc: HabitatManagementComputer) -> HabitatManagementComputer:
    """Copy mutable HMC trace state without copying frozen contract mappings."""

    clone = copy.copy(hmc)
    clone._control_events = list(hmc.control_events)  # noqa: SLF001
    clone._verified_snapshot_handle = None  # noqa: SLF001
    return clone


def replay_v3_policy_branch(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
    decision_step: int,
    action_id: str,
    command: Mapping[str, Any],
    *,
    current_command_sha256: str,
    branch_hmc: HabitatManagementComputer | None = None,
    branch_state: Any | None = None,
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
    if branch_hmc is None:
        hmc = HabitatManagementComputer.reset(
            scenario,
            bundle.hmc_contract,
            episode_nonce(family_id),
        )
        shadow = initial_state(scenario)
        start_step = 0
    else:
        if type(branch_hmc) is not HabitatManagementComputer or branch_state is None:
            raise Issue56V3RiskError("V3 branch continuation state is malformed")
        if branch_hmc.lifecycle_phase != "OBSERVED" or branch_state.step != decision_step:
            raise Issue56V3RiskError("V3 branch continuation is not at the decision")
        hmc = branch_hmc
        hmc._verified_snapshot_handle = None  # noqa: SLF001 - cloned capability reset
        shadow = branch_state
        start_step = decision_step
    target_rows: list[np.ndarray] = []
    state_digests: list[str] = []
    requested_sha: str | None = None
    final_sha: str | None = None
    executed_sha: str | None = None
    disposition: str | None = None
    for step in range(start_step, EPISODE_STEPS):
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

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V3RiskSample":
        expected = {
            "schema_version",
            "family_id",
            "decision_step",
            "split",
            "action_id",
            "scenario_sha256",
            "features_f32_hex",
            "label",
            "sample_sha256",
        }
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V3RiskError("V3 sample fields drift")
        if mapping["schema_version"] != f"{ISSUE56_V3_SCHEMA_VERSION}.sample":
            raise Issue56V3RiskError("V3 sample schema drift")
        if type(mapping["features_f32_hex"]) is not str:
            raise Issue56V3RiskError("V3 sample feature bytes are malformed")
        try:
            raw = bytes.fromhex(mapping["features_f32_hex"])
        except ValueError as error:
            raise Issue56V3RiskError("V3 sample feature bytes are malformed") from error
        if len(raw) != FEATURE_COUNT * np.dtype(np.float32).itemsize:
            raise Issue56V3RiskError("V3 sample feature bytes have the wrong length")
        return cls(
            mapping["family_id"],
            mapping["decision_step"],
            mapping["split"],
            mapping["action_id"],
            mapping["scenario_sha256"],
            np.frombuffer(raw, dtype=np.float32).copy(),
            V3PolicyLabel.from_mapping(mapping["label"]),
            mapping["sample_sha256"],
        )


def load_v3_samples(rows: Sequence[Mapping[str, Any]]) -> tuple[V3RiskSample, ...]:
    if isinstance(rows, (str, bytes)):
        raise Issue56V3RiskError("V3 sample rows must be a sequence")
    return tuple(V3RiskSample.from_mapping(row) for row in rows)


def _baseline_histories(
    bundle: ForecastContracts,
    scenario: Scenario,
    family_id: str,
) -> tuple[
    dict[int, tuple[Any, Any]],
    dict[int, str],
    dict[int, HabitatManagementComputer],
    dict[int, Any],
]:
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce(family_id),
    )
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    command_sha_by_step: dict[int, str] = {}
    branch_hmcs: dict[int, HabitatManagementComputer] = {}
    branch_states: dict[int, Any] = {}
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
            branch_hmc = _clone_hmc_for_branch(hmc)
            branch_hmcs[step] = branch_hmc
            branch_states[step] = shadow
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
    return snapshots, command_sha_by_step, branch_hmcs, branch_states


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
    snapshots, command_sha_by_step, branch_hmcs, branch_states = _baseline_histories(
        bundle,
        scenario,
        family_id,
    )
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
                branch_hmc=_clone_hmc_for_branch(branch_hmcs[step]),
                branch_state=branch_states[step],
            )
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


V3_HORIZON_KEYS = (4, 16, 32, 0)
V3_EVENT_LIMIT = 0.50
V3_EXPECTED_EXPOSURE_LIMIT = 0.50
V3_MAXIMUM_CROSSING_LIMIT = 0.25
V3_RIDGE_ALPHA = 0.1
V3_MODEL_SEED = 560057
V3_LOGIT_ITERATIONS = 120
V3_LOGIT_TOLERANCE = 1e-8
V3_CALIBRATION_RIDGE = 1e-4


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-min(float(value), 700.0))
        return 1.0 / (1.0 + factor)
    factor = math.exp(max(float(value), -700.0))
    return factor / (1.0 + factor)


def _bounded_expm1(value: float) -> float:
    return math.expm1(min(700.0, max(0.0, float(value))))


def _fit_weighted_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, np.ndarray]:
    if labels.ndim != 1 or set(np.unique(labels)) != {0.0, 1.0}:
        raise Issue56V3RiskError("V3 logistic fit requires both event classes")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise Issue56V3RiskError("V3 logistic regularization is invalid")
    normalized = (features - feature_mean) / feature_scale
    count = len(labels)
    positive_count = int(np.sum(labels))
    negative_count = count - positive_count
    sample_weights = np.where(
        labels > 0.5,
        count / (2.0 * positive_count),
        count / (2.0 * negative_count),
    )
    design = np.column_stack((np.ones(count, dtype=np.float64), normalized))
    parameters = np.zeros(FEATURE_COUNT + 1, dtype=np.float64)
    parameters[0] = math.log(positive_count / negative_count)
    def objective(values: np.ndarray) -> float:
        logits = design @ values
        return float(
            np.sum(sample_weights * (np.logaddexp(0.0, logits) - labels * logits))
            + 0.5 * alpha * np.dot(values[1:], values[1:])
            + 0.5e-8 * values[0] * values[0]
        )

    for _ in range(V3_LOGIT_ITERATIONS):
        logits = design @ parameters
        probabilities = np.asarray([_sigmoid(value) for value in logits])
        residual = sample_weights * (probabilities - labels)
        gradient = design.T @ residual
        penalty = np.zeros_like(parameters)
        penalty[1:] = alpha * parameters[1:]
        penalty[0] = 1e-8 * parameters[0]
        gradient += penalty
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design)
        hessian[0, 0] += 1e-8
        hessian[1:, 1:] += np.eye(FEATURE_COUNT, dtype=np.float64) * alpha
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if not np.isfinite(direction).all():
                raise Issue56V3RiskError("V3 logistic fit is singular") from error
        current_objective = objective(parameters)
        directional_derivative = float(np.dot(gradient, direction))
        if not math.isfinite(directional_derivative) or directional_derivative <= 0.0:
            raise Issue56V3RiskError("V3 logistic fit produced an invalid search direction")
        step = 1.0
        while step >= 1e-8:
            candidate = parameters - step * direction
            if objective(candidate) <= current_objective - 1e-4 * step * directional_derivative:
                parameters = candidate
                break
            step *= 0.5
        else:
            raise Issue56V3RiskError("V3 logistic fit line search failed")
        if float(np.max(np.abs(step * direction))) <= V3_LOGIT_TOLERANCE:
            break
    else:
        raise Issue56V3RiskError("V3 logistic fit did not converge")
    if not np.isfinite(parameters).all():
        raise Issue56V3RiskError("V3 logistic fit produced non-finite coefficients")
    return float(parameters[0]), parameters[1:]


def _fit_ridge_target(
    features: np.ndarray,
    targets: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, np.ndarray]:
    if len(features) != len(targets) or not len(features):
        raise Issue56V3RiskError("V3 severity fit has insufficient rows")
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise Issue56V3RiskError("V3 severity target is malformed")
    if len(features) == 1:
        return float(values[0]), 1.0, np.zeros(FEATURE_COUNT, dtype=np.float64)
    normalized_x = (features - feature_mean) / feature_scale
    target_mean = float(np.mean(values))
    target_scale = float(max(np.std(values), 1e-8))
    normalized_y = (values - target_mean) / target_scale
    gram = normalized_x.T @ normalized_x
    gram += np.eye(FEATURE_COUNT, dtype=np.float64) * alpha
    try:
        coefficients = np.linalg.solve(gram, normalized_x.T @ normalized_y)
    except np.linalg.LinAlgError as error:
        raise Issue56V3RiskError("V3 severity fit is singular") from error
    return target_mean, target_scale, coefficients


def _fit_calibration_logistic(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if (
        logits.ndim != 1
        or labels.ndim != 1
        or logits.shape != labels.shape
        or not len(labels)
        or not np.isfinite(logits).all()
    ):
        raise Issue56V3RiskError("V3 calibration labels are malformed")
    classes = set(np.unique(labels))
    if classes != {0.0, 1.0}:
        raise Issue56V3RiskError("V3 calibration requires both event classes")
    center = float(np.mean(logits))
    scale = float(max(np.std(logits), 1.0))
    normalized_logits = (logits - center) / scale
    design = np.column_stack((np.ones(len(labels)), normalized_logits))
    positive_rate = float(np.mean(labels))
    parameters = np.asarray(
        [
            math.log(positive_rate / (1.0 - positive_rate)),
            1.0,
        ],
        dtype=np.float64,
    )
    ridge = np.asarray((1e-8, V3_CALIBRATION_RIDGE), dtype=np.float64)

    def objective(values: np.ndarray) -> float:
        linear = design @ values
        return float(
            np.sum(np.logaddexp(0.0, linear) - labels * linear)
            + 0.5 * np.dot(ridge * values, values)
        )

    converged = False
    for _ in range(V3_LOGIT_ITERATIONS):
        probabilities = np.asarray([_sigmoid(value) for value in design @ parameters])
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design)
        hessian += np.diag(ridge)
        gradient = design.T @ (probabilities - labels) + ridge * parameters
        if float(np.max(np.abs(gradient))) <= V3_LOGIT_TOLERANCE:
            converged = True
            break
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise Issue56V3RiskError("V3 event calibration is singular") from error
        if not np.isfinite(update).all():
            raise Issue56V3RiskError("V3 event calibration produced a non-finite update")
        current_objective = objective(parameters)
        directional_derivative = float(np.dot(gradient, update))
        step = 1.0
        while step >= 1e-8:
            candidate = parameters - step * update
            candidate[1] = max(0.0, candidate[1])
            if objective(candidate) <= current_objective - 1e-4 * step * directional_derivative:
                break
            step *= 0.5
        else:
            raise Issue56V3RiskError("V3 event calibration line search failed")
        if float(np.max(np.abs(candidate - parameters))) <= V3_LOGIT_TOLERANCE:
            parameters = candidate
            converged = True
            break
        parameters = candidate
    if not converged:
        raise Issue56V3RiskError("V3 event calibration did not converge")
    if not np.isfinite(parameters).all():
        raise Issue56V3RiskError("V3 event calibration is non-finite")
    return float(parameters[0] - parameters[1] * center / scale), float(parameters[1] / scale)


def _label_metric(sample: V3RiskSample, horizon: int) -> V3HorizonMetric:
    if horizon == 0:
        return sample.label.remaining_metric
    for metric in sample.label.horizon_metrics:
        if metric.horizon_steps == horizon:
            return metric
    raise Issue56V3RiskError("V3 sample lacks requested horizon")


@dataclass(frozen=True, slots=True)
class V3HorizonPrediction:
    horizon_steps: int
    event_probability: float
    conditional_exposure: float
    upper_conditional_exposure: float
    conditional_maximum_crossing: float
    upper_maximum_crossing: float
    upper_expected_exposure: float
    upper_expected_maximum_crossing: float

    def __post_init__(self) -> None:
        if self.horizon_steps not in V3_HORIZON_KEYS:
            raise Issue56V3RiskError("V3 prediction horizon is invalid")
        for value, label in (
            (self.event_probability, "V3 event probability"),
            (self.conditional_exposure, "V3 conditional exposure"),
            (self.upper_conditional_exposure, "V3 upper exposure"),
            (self.conditional_maximum_crossing, "V3 maximum crossing"),
            (self.upper_maximum_crossing, "V3 upper maximum crossing"),
            (self.upper_expected_exposure, "V3 expected exposure"),
            (
                self.upper_expected_maximum_crossing,
                "V3 expected maximum crossing",
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V3RiskError(f"{label} is invalid")
        if not 0.0 <= self.event_probability <= 1.0:
            raise Issue56V3RiskError("V3 event probability is outside [0, 1]")


@dataclass(frozen=True, slots=True)
class V3RiskPrediction:
    horizons: tuple[V3HorizonPrediction, ...]
    hard_ineligible: bool
    reason: str | None

    def __post_init__(self) -> None:
        if tuple(item.horizon_steps for item in self.horizons) != V3_HORIZON_KEYS:
            raise Issue56V3RiskError("V3 prediction horizons drifted")

    def at(self, horizon: int) -> V3HorizonPrediction:
        for item in self.horizons:
            if item.horizon_steps == horizon:
                return item
        raise Issue56V3RiskError("V3 prediction horizon is unavailable")


@dataclass(frozen=True, slots=True)
class V3RiskModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    event_intercepts: np.ndarray
    event_coefficients: np.ndarray
    severity_target_means: np.ndarray
    severity_target_scales: np.ndarray
    severity_coefficients: np.ndarray
    maximum_target_means: np.ndarray
    maximum_target_scales: np.ndarray
    maximum_coefficients: np.ndarray
    calibration_intercepts: np.ndarray
    calibration_slopes: np.ndarray
    severity_residual_p90: np.ndarray
    maximum_residual_p90: np.ndarray
    alpha: float = V3_RIDGE_ALPHA
    seed: int = V3_MODEL_SEED
    model_id: str = "issue56-v3-unfitted"
    actuator_authority: bool = False

    def __post_init__(self) -> None:
        vector_fields = (
            ("feature_mean", self.feature_mean, (FEATURE_COUNT,)),
            ("feature_scale", self.feature_scale, (FEATURE_COUNT,)),
            ("event_intercepts", self.event_intercepts, (len(V3_HORIZON_KEYS),)),
            (
                "event_coefficients",
                self.event_coefficients,
                (len(V3_HORIZON_KEYS), FEATURE_COUNT),
            ),
            (
                "severity_target_means",
                self.severity_target_means,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "severity_target_scales",
                self.severity_target_scales,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "severity_coefficients",
                self.severity_coefficients,
                (len(V3_HORIZON_KEYS), FEATURE_COUNT),
            ),
            (
                "maximum_target_means",
                self.maximum_target_means,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "maximum_target_scales",
                self.maximum_target_scales,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "maximum_coefficients",
                self.maximum_coefficients,
                (len(V3_HORIZON_KEYS), FEATURE_COUNT),
            ),
            (
                "calibration_intercepts",
                self.calibration_intercepts,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "calibration_slopes",
                self.calibration_slopes,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "severity_residual_p90",
                self.severity_residual_p90,
                (len(V3_HORIZON_KEYS),),
            ),
            (
                "maximum_residual_p90",
                self.maximum_residual_p90,
                (len(V3_HORIZON_KEYS),),
            ),
        )
        for label, value, shape in vector_fields:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise Issue56V3RiskError(f"V3 model {label} is malformed")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, label, array)
        if np.any(self.feature_scale <= 0.0) or np.any(self.severity_target_scales <= 0.0):
            raise Issue56V3RiskError("V3 model scales must be positive")
        if np.any(self.maximum_target_scales <= 0.0):
            raise Issue56V3RiskError("V3 maximum scales must be positive")
        if np.any(self.calibration_slopes < 0.0):
            raise Issue56V3RiskError("V3 calibration must be monotonic")
        if np.any(self.severity_residual_p90 < 0.0) or np.any(self.maximum_residual_p90 < 0.0):
            raise Issue56V3RiskError("V3 residual bounds must be non-negative")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise Issue56V3RiskError("V3 model alpha is invalid")
        if self.actuator_authority is not False:
            raise Issue56V3RiskError("V3 risk model cannot have actuator authority")

    @classmethod
    def fit(
        cls,
        samples: Sequence[V3RiskSample],
        *,
        alpha: float = V3_RIDGE_ALPHA,
        seed: int = V3_MODEL_SEED,
    ) -> "V3RiskModel":
        items = tuple(samples)
        if len(items) < 8 or len({item.family_id for item in items}) < 2:
            raise Issue56V3RiskError("V3 fit requires at least eight samples from two families")
        if any(item.split != "TRAIN" for item in items):
            raise Issue56V3RiskError("V3 fit accepts TRAIN samples only")
        features = np.stack([item.features_f32 for item in items]).astype(np.float64)
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        event_intercepts: list[float] = []
        event_coefficients: list[np.ndarray] = []
        severity_means: list[float] = []
        severity_scales: list[float] = []
        severity_coefficients: list[np.ndarray] = []
        maximum_means: list[float] = []
        maximum_scales: list[float] = []
        maximum_coefficients: list[np.ndarray] = []
        for horizon in V3_HORIZON_KEYS:
            metrics = tuple(_label_metric(item, horizon) for item in items)
            labels = np.asarray([metric.crossing_event for metric in metrics], dtype=np.float64)
            intercept, coefficients = _fit_weighted_logistic(
                features,
                labels,
                feature_mean,
                feature_scale,
                alpha=alpha,
            )
            event_intercepts.append(intercept)
            event_coefficients.append(coefficients)
            positive_indices = np.flatnonzero(labels > 0.5)
            if not len(positive_indices):
                raise Issue56V3RiskError(
                    f"V3 horizon {horizon or 'remaining'} lacks positive TRAIN events"
                )
            positive_metrics = [metrics[index] for index in positive_indices]
            positive_features = features[positive_indices]
            severity_mean, severity_scale, severity_coefficient = _fit_ridge_target(
                positive_features,
                np.asarray([math.log1p(metric.safety_exposure) for metric in positive_metrics]),
                feature_mean,
                feature_scale,
                alpha=alpha,
            )
            maximum_mean, maximum_scale, maximum_coefficient = _fit_ridge_target(
                positive_features,
                np.asarray([math.log1p(metric.maximum_crossing) for metric in positive_metrics]),
                feature_mean,
                feature_scale,
                alpha=alpha,
            )
            severity_means.append(severity_mean)
            severity_scales.append(severity_scale)
            severity_coefficients.append(severity_coefficient)
            maximum_means.append(maximum_mean)
            maximum_scales.append(maximum_scale)
            maximum_coefficients.append(maximum_coefficient)
        zero = np.zeros(len(V3_HORIZON_KEYS), dtype=np.float64)
        one = np.ones(len(V3_HORIZON_KEYS), dtype=np.float64)
        model_id = "issue56-v3-" + _sha(
            {
                "feature_mean": feature_mean.tolist(),
                "feature_scale": feature_scale.tolist(),
                "event_intercepts": event_intercepts,
                "event_coefficients": [item.tolist() for item in event_coefficients],
                "seed": seed,
            }
        )[:16]
        return cls(
            feature_mean,
            feature_scale,
            np.asarray(event_intercepts),
            np.asarray(event_coefficients),
            np.asarray(severity_means),
            np.asarray(severity_scales),
            np.asarray(severity_coefficients),
            np.asarray(maximum_means),
            np.asarray(maximum_scales),
            np.asarray(maximum_coefficients),
            zero,
            one,
            zero.copy(),
            zero.copy(),
            alpha,
            seed,
            model_id,
        )

    def _raw_values(self, features: np.ndarray, index: int) -> tuple[float, float, float]:
        normalized = (features - self.feature_mean) / self.feature_scale
        event_logit = float(self.event_intercepts[index] + normalized @ self.event_coefficients[index])
        severity_log = float(
            self.severity_target_means[index]
            + self.severity_target_scales[index]
            * (normalized @ self.severity_coefficients[index])
        )
        maximum_log = float(
            self.maximum_target_means[index]
            + self.maximum_target_scales[index]
            * (normalized @ self.maximum_coefficients[index])
        )
        return event_logit, severity_log, maximum_log

    def predict_features(self, features: np.ndarray) -> V3RiskPrediction:
        values = _finite_features(features).astype(np.float64)
        predictions: list[V3HorizonPrediction] = []
        for index, horizon in enumerate(V3_HORIZON_KEYS):
            event_logit, severity_log, maximum_log = self._raw_values(values, index)
            probability = _sigmoid(
                self.calibration_intercepts[index]
                + self.calibration_slopes[index] * event_logit
            )
            conditional = _bounded_expm1(severity_log)
            upper_conditional = _bounded_expm1(
                severity_log + self.severity_residual_p90[index]
            )
            maximum = _bounded_expm1(maximum_log)
            upper_maximum = _bounded_expm1(
                maximum_log + self.maximum_residual_p90[index]
            )
            predictions.append(
                V3HorizonPrediction(
                    horizon,
                    probability,
                    conditional,
                    upper_conditional,
                    maximum,
                    upper_maximum,
                    probability * upper_conditional,
                    probability * upper_maximum,
                )
            )
        immediate, remaining = predictions[0], predictions[-1]
        hard = (
            immediate.event_probability > V3_EVENT_LIMIT
            or remaining.event_probability > V3_EVENT_LIMIT
            or remaining.upper_expected_exposure > V3_EXPECTED_EXPOSURE_LIMIT
            or remaining.upper_expected_maximum_crossing > V3_MAXIMUM_CROSSING_LIMIT
        )
        return V3RiskPrediction(
            tuple(predictions),
            hard,
            "v3_calibrated_risk_limit" if hard else None,
        )

    def predict(
        self,
        history: Any,
        action_f32: np.ndarray,
        *,
        decision_step: int,
        alarm_family_slots: Sequence[Sequence[int]],
    ) -> V3RiskPrediction:
        return self.predict_features(
            v2_feature_vector(
                history,
                action_f32,
                decision_step=decision_step,
                alarm_family_slots=alarm_family_slots,
            )
        )

    def calibrate(self, samples: Sequence[V3RiskSample]) -> "V3RiskModel":
        items = tuple(samples)
        if not items or any(item.split != "VALIDATION" for item in items):
            raise Issue56V3RiskError("V3 calibration accepts VALIDATION samples only")
        intercepts: list[float] = []
        slopes: list[float] = []
        severity_residuals: list[float] = []
        maximum_residuals: list[float] = []
        for index, horizon in enumerate(V3_HORIZON_KEYS):
            metrics = tuple(_label_metric(item, horizon) for item in items)
            raw = np.asarray(
                [self._raw_values(item.features_f32, index)[0] for item in items],
                dtype=np.float64,
            )
            labels = np.asarray([metric.crossing_event for metric in metrics], dtype=np.float64)
            if set(np.unique(labels)) != {0.0, 1.0}:
                raise Issue56V3RiskError(
                    f"V3 validation horizon {horizon or 'remaining'} lacks both event classes"
                )
            intercept, slope = _fit_calibration_logistic(raw, labels)
            intercepts.append(intercept)
            slopes.append(slope)
            positive = [
                (self._raw_values(item.features_f32, index)[1], metric.safety_exposure)
                for item, metric in zip(items, metrics, strict=True)
                if metric.crossing_event > 0.5
            ]
            positive_maximum = [
                (self._raw_values(item.features_f32, index)[2], metric.maximum_crossing)
                for item, metric in zip(items, metrics, strict=True)
                if metric.crossing_event > 0.5
            ]
            if not positive or not positive_maximum:
                severity_residuals.append(0.0)
                maximum_residuals.append(0.0)
                continue
            severity_residuals.append(
                float(
                    np.quantile(
                        np.asarray(
                            [abs(log_value - math.log1p(actual)) for log_value, actual in positive]
                        ),
                        0.90,
                    )
                )
            )
            maximum_residuals.append(
                float(
                    np.quantile(
                        np.asarray(
                            [abs(log_value - math.log1p(actual)) for log_value, actual in positive_maximum]
                        ),
                        0.90,
                    )
                )
            )
        return V3RiskModel(
            self.feature_mean,
            self.feature_scale,
            self.event_intercepts,
            self.event_coefficients,
            self.severity_target_means,
            self.severity_target_scales,
            self.severity_coefficients,
            self.maximum_target_means,
            self.maximum_target_scales,
            self.maximum_coefficients,
            np.asarray(intercepts),
            np.asarray(slopes),
            np.asarray(severity_residuals),
            np.asarray(maximum_residuals),
            self.alpha,
            self.seed,
            self.model_id + "-calibrated",
        )

    def to_mapping(self) -> dict[str, Any]:
        body = {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.model",
            "model_id": self.model_id,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "event_intercepts": self.event_intercepts.tolist(),
            "event_coefficients": self.event_coefficients.tolist(),
            "severity_target_means": self.severity_target_means.tolist(),
            "severity_target_scales": self.severity_target_scales.tolist(),
            "severity_coefficients": self.severity_coefficients.tolist(),
            "maximum_target_means": self.maximum_target_means.tolist(),
            "maximum_target_scales": self.maximum_target_scales.tolist(),
            "maximum_coefficients": self.maximum_coefficients.tolist(),
            "calibration_intercepts": self.calibration_intercepts.tolist(),
            "calibration_slopes": self.calibration_slopes.tolist(),
            "severity_residual_p90": self.severity_residual_p90.tolist(),
            "maximum_residual_p90": self.maximum_residual_p90.tolist(),
            "alpha": self.alpha,
            "seed": self.seed,
            "actuator_authority": False,
            "event_fit": "weighted_logistic_balanced_then_validation_calibrated",
        }
        return {**body, "model_sha256": _sha(body)}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V3RiskModel":
        if type(mapping) is not dict:
            raise Issue56V3RiskError("V3 model artifact must be an object")
        expected = set(
            V3RiskModel(
                np.zeros(FEATURE_COUNT),
                np.ones(FEATURE_COUNT),
                np.zeros(4),
                np.zeros((4, FEATURE_COUNT)),
                np.ones(4),
                np.ones(4),
                np.zeros((4, FEATURE_COUNT)),
                np.ones(4),
                np.ones(4),
                np.zeros((4, FEATURE_COUNT)),
                np.zeros(4),
                np.ones(4),
                np.zeros(4),
                np.zeros(4),
            ).to_mapping()
        )
        if set(mapping) != expected:
            raise Issue56V3RiskError("V3 model artifact fields drift")
        body = dict(mapping)
        digest = body.pop("model_sha256")
        if digest != _sha(body) or body["schema_version"] != f"{ISSUE56_V3_SCHEMA_VERSION}.model":
            raise Issue56V3RiskError("V3 model artifact digest/schema is invalid")
        if body["actuator_authority"] is not False:
            raise Issue56V3RiskError("V3 model artifact claims actuator authority")
        return cls(
            np.asarray(body["feature_mean"], dtype=np.float64),
            np.asarray(body["feature_scale"], dtype=np.float64),
            np.asarray(body["event_intercepts"], dtype=np.float64),
            np.asarray(body["event_coefficients"], dtype=np.float64),
            np.asarray(body["severity_target_means"], dtype=np.float64),
            np.asarray(body["severity_target_scales"], dtype=np.float64),
            np.asarray(body["severity_coefficients"], dtype=np.float64),
            np.asarray(body["maximum_target_means"], dtype=np.float64),
            np.asarray(body["maximum_target_scales"], dtype=np.float64),
            np.asarray(body["maximum_coefficients"], dtype=np.float64),
            np.asarray(body["calibration_intercepts"], dtype=np.float64),
            np.asarray(body["calibration_slopes"], dtype=np.float64),
            np.asarray(body["severity_residual_p90"], dtype=np.float64),
            np.asarray(body["maximum_residual_p90"], dtype=np.float64),
            float(body["alpha"]),
            int(body["seed"]),
            str(body["model_id"]),
            bool(body["actuator_authority"]),
        )


@dataclass(frozen=True, slots=True)
class V3RiskScore:
    action_id: str
    hard_ineligible: bool
    point_score: float
    immediate: V3HorizonPrediction
    remaining: V3HorizonPrediction
    reason: str | None


def risk_filter_point_scores_v3(
    bundle: ForecastContracts,
    history: Any,
    model: V3RiskModel,
    point_predictor: Any,
    current_command: np.ndarray,
    *,
    decision_step: int,
) -> tuple[V3RiskScore, ...]:
    current = np.asarray(current_command, dtype=np.float64)
    if current.shape != (ACTION_COUNT,) or not np.isfinite(current).all():
        raise Issue56V3RiskError("V3 current command vector is invalid")
    alarm_slots = alarm_family_slot_indices(bundle)
    scores: list[V3RiskScore] = []
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
            V3RiskScore(
                action.action_id,
                prediction.hard_ineligible or point_score.hard_ineligible,
                point_score.score,
                prediction.at(4),
                prediction.at(0),
                prediction.reason or point_score.reason,
            )
        )
    return tuple(scores)


def risk_only_scores_v3(
    bundle: ForecastContracts,
    history: Any,
    model: V3RiskModel,
    *,
    decision_step: int,
) -> tuple[V3RiskScore, ...]:
    alarm_slots = alarm_family_slot_indices(bundle)
    scores: list[V3RiskScore] = []
    for action in bundle.actions:
        action_vector = project_proposed_action(bundle, action.command)
        prediction = model.predict(
            history,
            action_vector,
            decision_step=decision_step,
            alarm_family_slots=alarm_slots,
        )
        scores.append(
            V3RiskScore(
                action.action_id,
                prediction.hard_ineligible,
                prediction.at(0).upper_expected_exposure,
                prediction.at(4),
                prediction.at(0),
                prediction.reason,
            )
        )
    return tuple(scores)


def select_risk_filtered_point_v3(scores: Sequence[V3RiskScore]) -> V3RiskScore | None:
    if not scores:
        raise Issue56V3RiskError("V3 selection requires candidates")
    if len({score.action_id for score in scores}) != len(scores):
        raise Issue56V3RiskError("V3 selection received duplicate action IDs")
    eligible = [score for score in scores if not score.hard_ineligible]
    return min(eligible, key=lambda score: (score.point_score, score.action_id)) if eligible else None


def select_risk_only_v3(scores: Sequence[V3RiskScore]) -> V3RiskScore | None:
    if not scores:
        raise Issue56V3RiskError("V3 risk-only selection requires candidates")
    if len({score.action_id for score in scores}) != len(scores):
        raise Issue56V3RiskError("V3 risk-only selection received duplicate action IDs")
    eligible = [score for score in scores if not score.hard_ineligible]
    return (
        min(
            eligible,
            key=lambda score: (
                score.remaining.upper_expected_exposure,
                score.remaining.event_probability,
                score.remaining.upper_expected_maximum_crossing,
                score.action_id,
            ),
        )
        if eligible
        else None
    )


def calibration_metrics_v3(
    model: V3RiskModel,
    samples: Sequence[V3RiskSample],
) -> dict[str, Any]:
    items = tuple(samples)
    if not items:
        raise Issue56V3RiskError("V3 calibration metrics require samples")
    metrics: dict[str, Any] = {}
    for horizon in V3_HORIZON_KEYS:
        probabilities: list[float] = []
        labels: list[float] = []
        conditional_hits = 0
        positive_count = 0
        conditional_error = 0.0
        for item in items:
            index = V3_HORIZON_KEYS.index(horizon)
            prediction = model.predict_features(item.features_f32).horizons[index]
            actual = _label_metric(item, horizon)
            probabilities.append(prediction.event_probability)
            labels.append(actual.crossing_event)
            if actual.crossing_event > 0.5:
                positive_count += 1
                conditional_hits += int(
                    actual.safety_exposure <= prediction.upper_conditional_exposure
                )
                conditional_error += abs(
                    prediction.conditional_exposure - actual.safety_exposure
                )
        if positive_count == 0:
            raise Issue56V3RiskError(
                f"V3 evaluation horizon {horizon or 'remaining'} lacks positive support"
            )
        probability_array = np.asarray(probabilities, dtype=np.float64)
        label_array = np.asarray(labels, dtype=np.float64)
        metrics[str(horizon or "remaining")] = {
            "sample_count": len(items),
            "positive_count": positive_count,
            "crossing_brier": float(np.mean((probability_array - label_array) ** 2)),
            "positive_conditional_upper_coverage": conditional_hits / max(positive_count, 1),
            "positive_mean_absolute_exposure_error": conditional_error
            / max(positive_count, 1),
            "mean_probability": float(np.mean(probability_array)),
            "event_rate": float(np.mean(label_array)),
        }
    return metrics


@dataclass(frozen=True, slots=True)
class V3DecisionRecord:
    decision_step: int
    selected_action_id: str | None
    proposal_receipt_sha256: str
    validation_outcome: str
    requested_command_sha256: str | None
    final_command_sha256: str
    executed_command_sha256: str
    disposition: str
    horizon_metrics: tuple[V3HorizonMetric, ...]
    remaining_metric: V3HorizonMetric
    decision_sha256: str

    def __post_init__(self) -> None:
        if self.decision_step not in v2_decision_steps():
            raise Issue56V3RiskError("V3 decision record step is invalid")
        for value, label in (
            (self.proposal_receipt_sha256, "V3 proposal receipt"),
            (self.final_command_sha256, "V3 final command"),
            (self.executed_command_sha256, "V3 executed command"),
            (self.decision_sha256, "V3 decision"),
        ):
            _require_sha(value, label)
        if self.final_command_sha256 != self.executed_command_sha256:
            raise Issue56V3RiskError("V3 decision final and executed command differ")
        if self.selected_action_id is not None and (
            type(self.selected_action_id) is not str or not self.selected_action_id
        ):
            raise Issue56V3RiskError("V3 selected action identity is invalid")
        if self.requested_command_sha256 is not None:
            _require_sha(self.requested_command_sha256, "V3 requested command")
        if self.validation_outcome not in {"NO_PROPOSAL", "VALID"}:
            raise Issue56V3RiskError("V3 decision validation outcome is invalid")
        if self.disposition not in V3_DECISION_OUTCOME_TYPES:
            raise Issue56V3RiskError("V3 decision disposition is invalid")
        if tuple(item.horizon_steps for item in self.horizon_metrics) != V3_HORIZONS:
            raise Issue56V3RiskError("V3 decision horizons drifted")
        if self.remaining_metric.horizon_steps != EPISODE_STEPS - self.decision_step:
            raise Issue56V3RiskError("V3 decision remaining horizon drifted")
        if self.validation_outcome == "NO_PROPOSAL" and self.selected_action_id is not None:
            raise Issue56V3RiskError("V3 abstention has a selected action")
        if self.validation_outcome == "VALID" and self.selected_action_id is None:
            raise Issue56V3RiskError("V3 valid proposal lacks selected action")
        if self.decision_sha256 != _sha(self._body()):
            raise Issue56V3RiskError("V3 decision digest is inconsistent")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.decision",
            "decision_step": self.decision_step,
            "selected_action_id": self.selected_action_id,
            "proposal_receipt_sha256": self.proposal_receipt_sha256,
            "validation_outcome": self.validation_outcome,
            "requested_command_sha256": self.requested_command_sha256,
            "final_command_sha256": self.final_command_sha256,
            "executed_command_sha256": self.executed_command_sha256,
            "disposition": self.disposition,
            "horizon_metrics": [item.to_mapping() for item in self.horizon_metrics],
            "remaining_metric": self.remaining_metric.to_mapping(),
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "decision_sha256": self.decision_sha256}


@dataclass(frozen=True, slots=True)
class V3EpisodeRecord:
    arm: str
    family_id: str
    family_index: int
    scenario_sha256: str
    decision_steps: tuple[int, ...]
    decisions: tuple[V3DecisionRecord, ...]
    proposal_count: int
    abstention_count: int
    admitted_proposal_count: int
    hmc_mismatch_count: int
    safety_exposure: float
    safety_violation_steps: int
    comfort_deviation: float
    resource_battery_fraction: float
    resource_oxygen_fraction: float
    resource_sorbent_fraction: float
    resource_composite: float
    control_run_id: str
    trace_sha256: str
    replay_committed_steps: int
    replay_final_state_sha256: str
    episode_sha256: str
    authority_verified: bool
    provenance_verified: bool
    replay_verified: bool
    metrics_finite_verified: bool
    proposal_admission_verified: bool
    trace_canonical_bytes: bytes = b""

    def __post_init__(self) -> None:
        if self.arm not in {
            "rules_only_common_window",
            "point_model_common_window",
            "risk_only_v3",
            "risk_filtered_point_v3",
        }:
            raise Issue56V3RiskError("V3 episode arm is invalid")
        if type(self.family_id) is not str or not self.family_id:
            raise Issue56V3RiskError("V3 episode family is invalid")
        if isinstance(self.family_index, bool) or not isinstance(self.family_index, int) or self.family_index < 0:
            raise Issue56V3RiskError("V3 episode family index is invalid")
        _require_sha(self.scenario_sha256, "V3 episode scenario")
        if self.decision_steps != v2_decision_steps():
            raise Issue56V3RiskError("V3 episode decision steps drifted")
        if len(self.decisions) != len(self.decision_steps):
            raise Issue56V3RiskError("V3 episode decision records drifted")
        if tuple(item.decision_step for item in self.decisions) != self.decision_steps:
            raise Issue56V3RiskError("V3 decision record ordering drifted")
        if sum(item.selected_action_id is not None for item in self.decisions) != self.proposal_count:
            raise Issue56V3RiskError("V3 proposal count does not match decisions")
        if self.proposal_count + self.abstention_count != len(self.decisions):
            raise Issue56V3RiskError("V3 decisions are not accounted for")
        if self.proposal_count != self.admitted_proposal_count:
            raise Issue56V3RiskError("V3 proposals were not admitted cleanly")
        if not 0 <= self.hmc_mismatch_count <= self.admitted_proposal_count:
            raise Issue56V3RiskError("V3 HMC mismatch count is invalid")
        for value, label in (
            (self.safety_exposure, "safety exposure"),
            (self.comfort_deviation, "comfort deviation"),
            (self.resource_battery_fraction, "battery resource"),
            (self.resource_oxygen_fraction, "oxygen resource"),
            (self.resource_sorbent_fraction, "sorbent resource"),
            (self.resource_composite, "resource composite"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Issue56V3RiskError(f"V3 {label} metric is invalid")
        if (
            isinstance(self.safety_violation_steps, bool)
            or not isinstance(self.safety_violation_steps, int)
            or not 0 <= self.safety_violation_steps <= EPISODE_STEPS
        ):
            raise Issue56V3RiskError("V3 violation count is invalid")
        _require_sha(self.control_run_id, "V3 control run")
        _require_sha(self.trace_sha256, "V3 trace")
        _require_sha(self.replay_final_state_sha256, "V3 replay final state")
        _require_sha(self.episode_sha256, "V3 episode")
        if type(self.trace_canonical_bytes) is not bytes:
            raise Issue56V3RiskError("V3 trace bytes are invalid")
        for value, label in (
            (self.authority_verified, "authority"),
            (self.provenance_verified, "provenance"),
            (self.replay_verified, "replay"),
            (self.metrics_finite_verified, "metric finiteness"),
            (self.proposal_admission_verified, "proposal admission"),
        ):
            if value is not True:
                raise Issue56V3RiskError(f"V3 {label} verification did not pass")
        if self.replay_committed_steps != EPISODE_STEPS:
            raise Issue56V3RiskError("V3 replay did not commit all steps")
        if self.episode_sha256 != _sha(self._body()):
            raise Issue56V3RiskError("V3 episode digest is inconsistent")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.episode",
            "arm": self.arm,
            "family_id": self.family_id,
            "family_index": self.family_index,
            "scenario_sha256": self.scenario_sha256,
            "decision_steps": list(self.decision_steps),
            "decisions": [item.to_mapping() for item in self.decisions],
            "proposal_count": self.proposal_count,
            "abstention_count": self.abstention_count,
            "admitted_proposal_count": self.admitted_proposal_count,
            "hmc_mismatch_count": self.hmc_mismatch_count,
            "safety_exposure": self.safety_exposure,
            "safety_violation_steps": self.safety_violation_steps,
            "comfort_deviation": self.comfort_deviation,
            "resource_battery_fraction": self.resource_battery_fraction,
            "resource_oxygen_fraction": self.resource_oxygen_fraction,
            "resource_sorbent_fraction": self.resource_sorbent_fraction,
            "resource_composite": self.resource_composite,
            "control_run_id": self.control_run_id,
            "trace_sha256": self.trace_sha256,
            "replay_committed_steps": self.replay_committed_steps,
            "replay_final_state_sha256": self.replay_final_state_sha256,
            "authority_verified": self.authority_verified,
            "provenance_verified": self.provenance_verified,
            "replay_verified": self.replay_verified,
            "metrics_finite_verified": self.metrics_finite_verified,
            "proposal_admission_verified": self.proposal_admission_verified,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "episode_sha256": self.episode_sha256}


def _episode_decision_metrics(
    scenario: Scenario,
    zone_ids: Sequence[str],
    target_rows: Sequence[np.ndarray],
    decision_step: int,
) -> tuple[tuple[V3HorizonMetric, ...], V3HorizonMetric]:
    values = np.asarray(target_rows[decision_step:], dtype=np.float64)
    metrics = tuple(
        V3HorizonMetric(horizon, *_label_metrics(values[:horizon]))
        for horizon in V3_HORIZONS
    )
    return metrics, V3HorizonMetric(len(values), *_label_metrics(values))


def _decision_disposition(arbitration: Mapping[str, Any], proposal: bool) -> str:
    if not proposal:
        return "ABSTAINED_TO_HOLD"
    if bool(arbitration.get("emergency_override")):
        return "EMERGENCY_OVERRIDDEN"
    disposition = arbitration.get("disposition")
    if disposition == "ACCEPTED":
        return "PROPOSED_ACCEPTED"
    if disposition == "MODIFIED":
        return "PROPOSED_MODIFIED"
    if disposition == "REJECTED":
        return "PROPOSED_REJECTED_TO_HOLD"
    raise Issue56V3RiskError("V3 arbitration disposition drifted")


def run_v3_episode(
    bundle: ForecastContracts,
    scenario: Scenario,
    arm: str,
    family_id: str,
    family_index: int,
    model: V3RiskModel,
    point_predictor: Any,
) -> V3EpisodeRecord:
    """Run one common-window policy arm through HMC and strict replay."""

    valid_arms = {
        "rules_only_common_window",
        "point_model_common_window",
        "risk_only_v3",
        "risk_filtered_point_v3",
    }
    if arm not in valid_arms or type(model) is not V3RiskModel or type(scenario) is not Scenario:
        raise Issue56V3RiskError("V3 episode inputs are invalid")
    if arm in {"point_model_common_window", "risk_filtered_point_v3"} and not callable(
        getattr(point_predictor, "predict", None)
    ):
        raise Issue56V3RiskError("V3 point arm requires a point predictor")
    actions = tuple(bundle.actions)
    if len(actions) != 4 or len({item.action_id for item in actions}) != 4:
        raise Issue56V3RiskError("V3 episode requires four unique actions")
    decisions = v2_decision_steps()
    zone_ids = scenario_zone_order(scenario)
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, episode_nonce(family_id))
    shadow = initial_state(scenario)
    initial_row = project_true_targets(scenario, zone_ids, shadow)
    states: dict[int, Any] = {0: shadow}
    snapshots: dict[int, tuple[Any, Any]] = {}
    last_command: Mapping[str, Any] | None = None
    preliminary: list[dict[str, Any]] = []
    proposal_count = 0
    abstention_count = 0
    admitted_count = 0
    mismatch_count = 0
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue56V3RiskError(f"V3 HMC terminated at step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snapshots[step] = (snapshot, verification)
        proposal = None
        selected_action_id: str | None = None
        if step in decisions:
            if last_command is None:
                raise Issue56V3RiskError("V3 decision has no current command")
            history = project_history_window(
                bundle,
                tuple(
                    snapshots[index]
                    for index in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
                ),
                window_steps=HISTORY_WINDOW_STEPS,
            )
            current_vector = _command_vector(scenario, last_command)
            if arm == "rules_only_common_window":
                selected = None
            elif arm == "point_model_common_window":
                scores = []
                for action in actions:
                    action_vector = project_proposed_action(bundle, action.command)
                    prediction = point_predictor.predict(history, action_vector)
                    scores.append(
                        score_point_prediction(
                            action.action_id,
                            prediction,
                            current_vector,
                            _command_vector(scenario, action.command.to_mapping()),
                        )
                    )
                selected = rank_actions_advisory(scores)
            elif arm == "risk_only_v3":
                selected = select_risk_only_v3(
                    risk_only_scores_v3(bundle, history, model, decision_step=step)
                )
            else:
                selected = select_risk_filtered_point_v3(
                    risk_filter_point_scores_v3(
                        bundle,
                        history,
                        model,
                        point_predictor,
                        current_vector,
                        decision_step=step,
                    )
                )
            if selected is None:
                abstention_count += 1
            else:
                selected_action_id = selected.action_id
                action = next(item for item in actions if item.action_id == selected.action_id)
                proposal = build_v3_proposal(
                    hmc,
                    snapshot.snapshot_sha256,
                    step,
                    action.command.to_mapping(),
                    action.action_id,
                )
                proposal_count += 1
        proposal_receipt = hmc.propose(proposal, handle).to_mapping()
        if step in decisions and proposal is not None:
            if (
                proposal_receipt["attempt_class"],
                proposal_receipt["validation_outcome"],
            ) != ("CANONICAL_PROPOSAL", "VALID"):
                raise Issue56V3RiskError("V3 proposal was not admitted")
            admitted_count += 1
        elif proposal_receipt["validation_outcome"] != "NO_PROPOSAL":
            raise Issue56V3RiskError("V3 proposal was issued outside a decision")
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue56V3RiskError("V3 arbitration failed")
        arbitration_mapping = arbitration.to_mapping()
        if proposal is not None and arbitration.final_command_sha256 != proposal_receipt["requested_command_sha256"]:
            mismatch_count += 1
        stepped = hmc.step()
        if not hasattr(stepped, "plant_receipt_digest"):
            raise Issue56V3RiskError("V3 step failed")
        shadow_result = advance_one_step_with_command(scenario, shadow, arbitration.final_command)
        if _sha_bytes(canonical_json_bytes(shadow_result.receipt)) != stepped.plant_receipt_digest:
            raise Issue56V3RiskError("V3 episode shadow replay diverged")
        shadow = shadow_result.state
        states[shadow.step] = shadow
        if step in decisions:
            step_mapping = stepped.to_mapping()
            preliminary.append(
                {
                    "decision_step": step,
                    "selected_action_id": selected_action_id,
                    "proposal_receipt_sha256": proposal_receipt["proposal_receipt_sha256"],
                    "validation_outcome": proposal_receipt["validation_outcome"],
                    "requested_command_sha256": proposal_receipt["requested_command_sha256"],
                    "final_command_sha256": arbitration_mapping["final_command_sha256"],
                    "executed_command_sha256": step_mapping["returned_external_command_digest"],
                    "disposition": _decision_disposition(arbitration_mapping, proposal is not None),
                }
            )
        last_command = dict(arbitration.final_command)
    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract)
    replay = replay_control_trace(trace.canonical_bytes, scenario=scenario, contract=bundle.hmc_contract)
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise Issue56V3RiskError("V3 episode trace failed strict replay")
    target_rows = [project_true_targets(scenario, zone_ids, states[index]) for index in range(1, EPISODE_STEPS + 1)]
    records: list[V3DecisionRecord] = []
    for raw in preliminary:
        horizon_metrics, remaining_metric = _episode_decision_metrics(
            scenario,
            zone_ids,
            target_rows,
            raw["decision_step"],
        )
        body = {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.decision",
            **raw,
            "horizon_metrics": [item.to_mapping() for item in horizon_metrics],
            "remaining_metric": remaining_metric.to_mapping(),
        }
        records.append(
            V3DecisionRecord(
                raw["decision_step"],
                raw["selected_action_id"],
                raw["proposal_receipt_sha256"],
                raw["validation_outcome"],
                raw["requested_command_sha256"],
                raw["final_command_sha256"],
                raw["executed_command_sha256"],
                raw["disposition"],
                horizon_metrics,
                remaining_metric,
                _sha(body),
            )
        )
    metrics = compute_race_metrics(
        scenario,
        zone_ids,
        initial_row,
        [states[index] for index in range(1, EPISODE_STEPS + 1)],
    )
    body = {
        "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.episode",
        "arm": arm,
        "family_id": family_id,
        "family_index": family_index,
        "scenario_sha256": scenario.scenario_sha256,
        "decision_steps": list(decisions),
        "decisions": [item.to_mapping() for item in records],
        "proposal_count": proposal_count,
        "abstention_count": abstention_count,
        "admitted_proposal_count": admitted_count,
        "hmc_mismatch_count": mismatch_count,
        "safety_exposure": float(metrics["safety_exposure"]),
        "safety_violation_steps": int(metrics["safety_violation_steps"]),
        "comfort_deviation": float(metrics["comfort_deviation"]),
        "resource_battery_fraction": float(metrics["resource_battery_fraction"]),
        "resource_oxygen_fraction": float(metrics["resource_oxygen_fraction"]),
        "resource_sorbent_fraction": float(metrics["resource_sorbent_fraction"]),
        "resource_composite": float(metrics["resource_composite"]),
        "control_run_id": hmc.control_run_id,
        "trace_sha256": _sha_bytes(trace.canonical_bytes),
        "replay_committed_steps": int(replay.committed_step_count),
        "replay_final_state_sha256": str(replay.final_state_sha256),
        "authority_verified": True,
        "provenance_verified": True,
        "replay_verified": True,
        "metrics_finite_verified": True,
        "proposal_admission_verified": proposal_count == admitted_count,
    }
    return V3EpisodeRecord(
        arm,
        family_id,
        family_index,
        scenario.scenario_sha256,
        decisions,
        tuple(records),
        proposal_count,
        abstention_count,
        admitted_count,
        mismatch_count,
        float(metrics["safety_exposure"]),
        int(metrics["safety_violation_steps"]),
        float(metrics["comfort_deviation"]),
        float(metrics["resource_battery_fraction"]),
        float(metrics["resource_oxygen_fraction"]),
        float(metrics["resource_sorbent_fraction"]),
        float(metrics["resource_composite"]),
        hmc.control_run_id,
        _sha_bytes(trace.canonical_bytes),
        int(replay.committed_step_count),
        str(replay.final_state_sha256),
        _sha(body),
        True,
        True,
        True,
        True,
        proposal_count == admitted_count,
        trace.canonical_bytes,
    )


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
    "V3RiskModel",
    "V3RiskPrediction",
    "V3RiskScore",
    "V3_ARMS",
    "build_v3_proposal",
    "collect_v3_family_samples",
    "calibration_metrics_v3",
    "replay_v3_policy_branch",
    "risk_filter_point_scores_v3",
    "risk_only_scores_v3",
    "select_risk_filtered_point_v3",
    "select_risk_only_v3",
]
