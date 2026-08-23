"""Offline checkpoint reconstruction and counterfactual rollouts for Issue #52.

This module constructs an HMC to replay the scenario timeline, then uses the
deterministic plant and instrumentation seams to generate labelled data.  It
cannot mint runtime authority objects or issue proposals outside the replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from typing import Any
from types import MappingProxyType

import numpy as np

from .forecast_issue52 import (
    CADENCE_SECONDS,
    HORIZON_STEPS,
    HISTORY_STEPS,
    ISSUE52_SCHEMA_VERSION,
    CandidateCatalogue,
    CandidateFeasibility,
    CandidateSchedule,
    ForecastHistory,
    Issue52RolloutError,
    ObservationRecord,
    TargetManifest,
    TrainingSample,
    targets_from_measurement,
)
from .hmc import HabitatManagementComputer
from .health import AlarmTrack, HealthReduction, HealthTracker, reduce_health
from .hmc_contract import HMCContract, canonical_json_bytes
from .instrumentation import (
    OperationalMeasurement,
    SensorMemory,
    instrument_v5_operational_measurement,
)
from .physics import (
    CanonicalExternalCommand,
    InfeasibleActionError,
    advance_one_step_with_command,
    preflight_external_command,
    validate_external_command,
    validate_external_step_result,
)
from .scenario import (
    SCENARIO_SCHEMA_VERSION_V5,
    Scenario,
    ScenarioValidationError,
)
from .snapshot import OperationalSnapshot, SnapshotVerificationReceipt
from .state import PlantState, UtilityState, ZoneState
from .telemetry import derive_observable_topology


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


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_jsonable(value))).hexdigest()


def _array_payload(value: np.ndarray) -> list[list[float | None]]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise Issue52RolloutError("rollout evidence array must be two-dimensional")
    return [
        [None if not math.isfinite(float(item)) else float(item) for item in row]
        for row in array
    ]


def _validate_array_cells(
    values: np.ndarray, mask: np.ndarray, *, label: str
) -> None:
    if not np.isfinite(values[mask]).all():
        raise Issue52RolloutError(f"available {label} cells are non-finite")
    if np.any(~mask) and not np.isnan(values[~mask]).all():
        raise Issue52RolloutError(f"unavailable {label} cells must be NaN")


def _segment_for_step(scenario: Scenario, step: int) -> Mapping[str, Any]:
    for segment in scenario.data["timeline"]:
        if int(segment["start_step"]) <= step < int(segment["end_step"]):
            return segment
    raise Issue52RolloutError(f"no timeline segment covers step {step}")


def _measurement_mapping(measurement: OperationalMeasurement) -> dict[str, Any]:
    return {
        "schema_version": measurement.schema_version,
        "completed_step": measurement.completed_step,
        "completed_time_s": measurement.completed_time_s,
        "primary": [sample.to_mapping() for sample in measurement.primary],
        "secondary": [sample.to_mapping() for sample in measurement.secondary],
        "primary_minus_secondary": [
            sample.to_mapping() for sample in measurement.primary_minus_secondary
        ],
        "operational_feedback": [
            sample.to_mapping() for sample in measurement.operational_feedback
        ],
        "sensor_memory": _jsonable(measurement.sensor_memory),
    }


def _health_mapping(reduction: HealthReduction) -> dict[str, Any]:
    return {
        "health_state": reduction.health_state,
        "alarms": [alarm.to_mapping() for alarm in reduction.alarms],
        "tracker": _jsonable(reduction.tracker),
    }


def _state_mapping(state: PlantState) -> dict[str, Any]:
    return _jsonable(state)


def _freeze_checkpoint_state(state: PlantState) -> PlantState:
    utility = state.utility
    return PlantState(
        step=state.step,
        zones=MappingProxyType(dict(state.zones)),
        utility=UtilityState(
            co2_sorbent_remaining_mol=utility.co2_sorbent_remaining_mol,
            captured_co2_mol=utility.captured_co2_mol,
            condensed_water_mol=utility.condensed_water_mol,
            oxygen_store_mol=utility.oxygen_store_mol,
            battery_energy_wh=utility.battery_energy_wh,
            actual_airflow_m3_s=MappingProxyType(dict(utility.actual_airflow_m3_s)),
            actual_scrubber_duty=utility.actual_scrubber_duty,
            actual_condenser_duty=utility.actual_condenser_duty,
            external_heat_rejected_j=utility.external_heat_rejected_j,
            external_heat_received_j=utility.external_heat_received_j,
            actual_fan_speed_fraction=utility.actual_fan_speed_fraction,
            actual_damper_position_by_id=MappingProxyType(
                dict(utility.actual_damper_position_by_id)
            ),
            actual_cooling_removed_w=MappingProxyType(
                dict(utility.actual_cooling_removed_w)
            ),
            actual_oxygen_injection_mol_s=MappingProxyType(
                dict(utility.actual_oxygen_injection_mol_s)
            ),
            effective_scrubber_capture_ability=utility.effective_scrubber_capture_ability,
            effective_condenser_removal_ability=(
                utility.effective_condenser_removal_ability
            ),
            effective_cooling_delivery_by_zone=MappingProxyType(
                dict(utility.effective_cooling_delivery_by_zone)
            ),
            effective_oxygen_delivery_by_zone=MappingProxyType(
                dict(utility.effective_oxygen_delivery_by_zone)
            ),
            last_operational_feedback=(
                None
                if utility.last_operational_feedback is None
                else MappingProxyType(dict(utility.last_operational_feedback))
            ),
        ),
    )


def _restore_state(state: PlantState) -> PlantState:
    payload = _state_mapping(state)
    utility = payload["utility"]
    return PlantState(
        step=int(payload["step"]),
        zones={
            str(zone_id): ZoneState(**values)
            for zone_id, values in payload["zones"].items()
        },
        utility=UtilityState(
            **{
                **utility,
                "actual_airflow_m3_s": dict(utility["actual_airflow_m3_s"]),
                "actual_damper_position_by_id": dict(
                    utility["actual_damper_position_by_id"]
                ),
                "actual_cooling_removed_w": dict(utility["actual_cooling_removed_w"]),
                "actual_oxygen_injection_mol_s": dict(
                    utility["actual_oxygen_injection_mol_s"]
                ),
                "effective_cooling_delivery_by_zone": dict(
                    utility["effective_cooling_delivery_by_zone"]
                ),
                "effective_oxygen_delivery_by_zone": dict(
                    utility["effective_oxygen_delivery_by_zone"]
                ),
            }
        ),
    )


def _restore_health_tracker(tracker: HealthTracker) -> HealthTracker:
    return HealthTracker(
        completed_step=tracker.completed_step,
        tracks={track_id: AlarmTrack(**_jsonable(track)) for track_id, track in tracker.tracks.items()},
    )


def _hidden_targets(
    scenario: Scenario, state: PlantState, manifest: TargetManifest
) -> np.ndarray:
    zone_config = {
        str(zone["id"]): zone for zone in scenario.data["zones"]
    }
    equipment = scenario.data["equipment"]
    initial_oxygen = float(scenario.data["initial_utility"]["oxygen_store_mol"])
    values: list[float] = []
    for descriptor in manifest.descriptors:
        if descriptor.scope == "zone":
            zone_id, field_name = descriptor.descriptor_id.split("/", 1)
            telemetry = state.zones[zone_id].telemetry(
                volume_m3=float(zone_config[zone_id]["volume_m3"])
            )
            values.append(float(telemetry[field_name]))
        elif descriptor.descriptor_id == "battery_state_of_charge":
            values.append(
                float(state.utility.battery_energy_wh)
                / float(equipment["battery_capacity_wh"])
            )
        elif descriptor.descriptor_id == "oxygen_store_fraction":
            values.append(float(state.utility.oxygen_store_mol) / max(initial_oxygen, 1e-12))
        elif descriptor.descriptor_id == "sorbent_remaining_fraction":
            values.append(
                float(state.utility.co2_sorbent_remaining_mol)
                / float(equipment["scrubber_capacity_mol"])
            )
        else:
            raise Issue52RolloutError(
                f"unsupported hidden target descriptor {descriptor.descriptor_id}"
            )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (manifest.width,) or not np.isfinite(result).all():
        raise Issue52RolloutError("hidden target projection is malformed")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RolloutCheckpoint:
    scenario: Scenario
    contract: HMCContract
    manifest: TargetManifest
    manifest_sha256: str
    family_id: str
    decision_step: int
    state: PlantState
    sensor_memory: SensorMemory
    health_tracker: HealthTracker
    last_measurement: OperationalMeasurement
    last_final_command: CanonicalExternalCommand
    history_records: tuple[ObservationRecord, ...]
    scenario_sha256: str
    topology_sha256: str
    hmc_contract_sha256: str
    snapshot_schema_sha256: str
    deterministic_seed: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if type(self.scenario) is not Scenario or type(self.contract) is not HMCContract:
            raise Issue52RolloutError("checkpoint requires exact scenario and contract")
        try:
            self.scenario.validate_contract_identities()
        except ScenarioValidationError as error:
            raise Issue52RolloutError("checkpoint scenario identity is inconsistent") from error
        if self.scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
            raise Issue52RolloutError("checkpoint requires a V5 scenario")
        if type(self.manifest) is not TargetManifest:
            raise Issue52RolloutError("checkpoint manifest is invalid")
        if self.manifest_sha256 != self.manifest.manifest_sha256:
            raise Issue52RolloutError("checkpoint manifest identity is inconsistent")
        if self.manifest.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue52RolloutError("checkpoint manifest does not bind scenario")
        if type(self.family_id) is not str or not self.family_id:
            raise Issue52RolloutError("checkpoint family identity is invalid")
        if (
            isinstance(self.decision_step, bool)
            or not isinstance(self.decision_step, int)
            or self.decision_step < HISTORY_STEPS - 1
            or self.decision_step + HORIZON_STEPS > int(self.scenario.data["steps"])
        ):
            raise Issue52RolloutError("checkpoint decision step is not eligible")
        if type(self.state) is not PlantState or self.state.step != self.decision_step:
            raise Issue52RolloutError("checkpoint plant state does not bind decision step")
        if type(self.sensor_memory) is not SensorMemory:
            raise Issue52RolloutError("checkpoint sensor memory is invalid")
        if type(self.health_tracker) is not HealthTracker:
            raise Issue52RolloutError("checkpoint health tracker is invalid")
        if type(self.last_measurement) is not OperationalMeasurement:
            raise Issue52RolloutError("checkpoint measurement is invalid")
        if self.last_measurement.completed_step != self.decision_step:
            raise Issue52RolloutError("checkpoint measurement does not bind state")
        if type(self.last_final_command) is not CanonicalExternalCommand:
            raise Issue52RolloutError("checkpoint command is invalid")
        if type(self.history_records) is not tuple or len(self.history_records) != HISTORY_STEPS:
            raise Issue52RolloutError("checkpoint must contain exactly 16 history records")
        try:
            history = ForecastHistory.from_records(self.history_records)
        except Exception as error:
            raise Issue52RolloutError("checkpoint history is not causal") from error
        if history.latest_record.completed_step != self.decision_step:
            raise Issue52RolloutError("checkpoint history does not end at decision step")
        if self.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue52RolloutError("checkpoint scenario identity is inconsistent")
        if self.topology_sha256 != self.manifest.topology_sha256:
            raise Issue52RolloutError("checkpoint topology identity is inconsistent")
        if self.hmc_contract_sha256 != self.contract.hmc_contract_sha256:
            raise Issue52RolloutError("checkpoint HMC identity is inconsistent")
        if self.snapshot_schema_sha256 != self.contract.snapshot_schema_sha256:
            raise Issue52RolloutError("checkpoint snapshot schema identity is inconsistent")
        if any(
            record.scenario_sha256 != self.scenario_sha256
            or record.topology_sha256 != self.topology_sha256
            or record.hmc_contract_sha256 != self.hmc_contract_sha256
            or record.snapshot_schema_sha256 != self.snapshot_schema_sha256
            for record in self.history_records
        ):
            raise Issue52RolloutError("checkpoint history identities are inconsistent")
        expected = _checkpoint_digest(self)
        if self.checkpoint_sha256 != expected:
            raise Issue52RolloutError("checkpoint digest is inconsistent")

    def to_mapping(self) -> dict[str, Any]:
        return _checkpoint_payload(self)

    def clone(self) -> "RolloutCheckpoint":
        clone = object.__new__(RolloutCheckpoint)
        for field_name, value in {
            "scenario": Scenario.from_mapping(_jsonable(self.scenario.data)),
            "contract": HMCContract.from_mapping(json.loads(self.contract.canonical_bytes)),
            "manifest": self.manifest,
            "manifest_sha256": self.manifest_sha256,
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "state": _restore_state(self.state),
            "sensor_memory": self.sensor_memory,
            "health_tracker": _restore_health_tracker(self.health_tracker),
            "last_measurement": self.last_measurement,
            "last_final_command": self.last_final_command,
            "history_records": self.history_records,
            "scenario_sha256": self.scenario_sha256,
            "topology_sha256": self.topology_sha256,
            "hmc_contract_sha256": self.hmc_contract_sha256,
            "snapshot_schema_sha256": self.snapshot_schema_sha256,
            "deterministic_seed": self.deterministic_seed,
            "checkpoint_sha256": self.checkpoint_sha256,
        }.items():
            object.__setattr__(clone, field_name, value)
        clone.__post_init__()
        if _checkpoint_digest(clone) != self.checkpoint_sha256:
            raise Issue52RolloutError("checkpoint clone changed canonical identity")
        return clone


def _checkpoint_payload(checkpoint: RolloutCheckpoint) -> dict[str, Any]:
    return {
        "schema_version": f"{ISSUE52_SCHEMA_VERSION}.offline_checkpoint",
        "scenario_sha256": checkpoint.scenario_sha256,
        "topology_sha256": checkpoint.topology_sha256,
        "hmc_contract_sha256": checkpoint.hmc_contract_sha256,
        "snapshot_schema_sha256": checkpoint.snapshot_schema_sha256,
        "manifest_sha256": checkpoint.manifest_sha256,
        "family_id": checkpoint.family_id,
        "decision_step": checkpoint.decision_step,
        "deterministic_seed": checkpoint.deterministic_seed,
        "state": _state_mapping(checkpoint.state),
        "sensor_memory": _jsonable(checkpoint.sensor_memory),
        "health_tracker": _jsonable(checkpoint.health_tracker),
        "last_measurement": _measurement_mapping(checkpoint.last_measurement),
        "last_final_command": checkpoint.last_final_command.to_mapping(),
        "history_records": [
            {
                "snapshot_sha256": record.snapshot_sha256,
                "verification_receipt_sha256": record.verification_receipt_sha256,
                "control_run_id": record.control_run_id,
                "authority_epoch": record.authority_epoch,
                "topology_sha256": record.topology_sha256,
                "hmc_contract_sha256": record.hmc_contract_sha256,
                "snapshot_schema_sha256": record.snapshot_schema_sha256,
                "scenario_sha256": record.scenario_sha256,
                "previous_verification_receipt_sha256": record.previous_verification_receipt_sha256,
                "previous_control_chain_sha256": record.previous_control_chain_sha256,
                "control_chain_sha256": record.control_chain_sha256,
                "sequence": record.sequence,
                "completed_step": record.completed_step,
                "completed_time_s": record.completed_time_s,
                "mode": record.mode,
                "command_sha256": record.command_sha256,
                "target_values": record.target_values.tolist(),
                "available_mask": record.available_mask.tolist(),
            }
            for record in checkpoint.history_records
        ],
    }


def _checkpoint_digest(checkpoint: RolloutCheckpoint) -> str:
    return _digest(_checkpoint_payload(checkpoint))


def _null_root(contract: HMCContract, name: str) -> str:
    return str(contract.data["null_roots"][name]["sha256"])


def build_offline_checkpoint(
    scenario: Scenario,
    contract: HMCContract,
    *,
    decision_step: int = HISTORY_STEPS - 1,
    family_id: str = "issue52-reference-family",
) -> RolloutCheckpoint:
    """Replay the scenario timeline to an eligible, fully instrumented checkpoint."""

    if type(scenario) is not Scenario or type(contract) is not HMCContract:
        raise Issue52RolloutError("checkpoint construction requires exact inputs")
    if scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
        raise Issue52RolloutError("checkpoint construction requires V5")
    if float(scenario.data["dt_seconds"]) != CADENCE_SECONDS:
        raise Issue52RolloutError("checkpoint construction requires 60-second cadence")
    if (
        isinstance(decision_step, bool)
        or not isinstance(decision_step, int)
        or decision_step < HISTORY_STEPS - 1
        or decision_step + HORIZON_STEPS > int(scenario.data["steps"])
    ):
        raise Issue52RolloutError("scenario is too short for the requested checkpoint")
    if type(family_id) is not str or not family_id:
        raise Issue52RolloutError("checkpoint family identity is invalid")

    manifest = TargetManifest.from_scenario(scenario)
    topology = derive_observable_topology(scenario)
    if topology.sha256 != manifest.topology_sha256:
        raise Issue52RolloutError("checkpoint topology derivation is inconsistent")
    scenario.validate_contract_identities()
    try:
        hmc = HabitatManagementComputer.reset(scenario, contract, b"\x00" * 32)
    except Exception as error:  # noqa: BLE001 - mirror HMC reset acceptance
        raise Issue52RolloutError("checkpoint scenario is not HMC-reset compatible") from error
    records: list[ObservationRecord] = []
    retained_evidence: list[tuple[OperationalSnapshot, SnapshotVerificationReceipt]] = []
    for completed_step in range(decision_step + 1):
        observed = hmc.observe()
        if not isinstance(observed, tuple):
            raise Issue52RolloutError("HMC entered terminal state during checkpoint replay")
        snapshot, verification = observed
        retained_evidence.append((snapshot, verification))
        try:
            record = ObservationRecord.from_snapshot(
                snapshot,
                verification,
                manifest,
                scenario,
                control_chain_sha256=hmc.current_control_chain_sha256,
            )
            handle = hmc.verify_snapshot(snapshot, verification)
        except Exception as error:  # noqa: BLE001 - HMC evidence must be exact
            raise Issue52RolloutError("checkpoint snapshot evidence is invalid") from error
        if (
            handle.sequence != record.sequence
            or handle.snapshot_sha256 != record.snapshot_sha256
        ):
            raise Issue52RolloutError("checkpoint verified handle does not bind record")
        records.append(record)
        if completed_step == decision_step:
            break
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        stepped = hmc.step()
        if type(arbitration).__name__ != "ArbitrationReceipt" or type(stepped).__name__ != "StepReceipt":
            raise Issue52RolloutError("HMC failed during checkpoint replay")

    state = hmc._state
    measurement = hmc._last_operational_measurement
    if measurement is None:
        raise Issue52RolloutError("HMC checkpoint has no issued measurement")
    history_records = tuple(records[-HISTORY_STEPS:])
    command = validate_external_command(
        scenario, history_records[-1].command
    )
    checkpoint_scenario = Scenario.from_mapping(_jsonable(scenario.data))
    checkpoint_manifest = TargetManifest.from_scenario(checkpoint_scenario)
    checkpoint = object.__new__(RolloutCheckpoint)
    for field_name, value in {
        "scenario": checkpoint_scenario,
        "contract": HMCContract.from_mapping(json.loads(contract.canonical_bytes)),
        "manifest": checkpoint_manifest,
        "manifest_sha256": checkpoint_manifest.manifest_sha256,
        "family_id": family_id,
        "decision_step": decision_step,
        "state": _freeze_checkpoint_state(state),
        "sensor_memory": measurement.sensor_memory,
        "health_tracker": _restore_health_tracker(hmc._health_tracker),
        "last_measurement": measurement,
        "last_final_command": command,
        "history_records": history_records,
        "scenario_sha256": scenario.scenario_sha256,
        "topology_sha256": topology.sha256,
        "hmc_contract_sha256": contract.hmc_contract_sha256,
        "snapshot_schema_sha256": contract.snapshot_schema_sha256,
        "deterministic_seed": int(scenario.data["sensor_model"]["random_seed"]),
    }.items():
        object.__setattr__(checkpoint, field_name, value)
    object.__setattr__(checkpoint, "checkpoint_sha256", _checkpoint_digest(checkpoint))
    checkpoint.__post_init__()
    return checkpoint


@dataclass(frozen=True, slots=True)
class RolloutResult:
    candidate_id: str
    checkpoint_sha256: str
    schedule_sha256: str
    manifest_sha256: str
    targets: np.ndarray
    available_mask: np.ndarray
    hidden_truth: np.ndarray
    command_digests: tuple[str | None, ...]
    state_digests: tuple[str | None, ...]
    feasibility: tuple[str, ...]
    termination_reason: str | None
    eligible: bool
    rollout_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise Issue52RolloutError("rollout candidate identity is invalid")
        for label, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("schedule", self.schedule_sha256),
            ("manifest", self.manifest_sha256),
            ("rollout", self.rollout_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or value != value.lower()
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise Issue52RolloutError(f"{label} identity is invalid")
        targets = np.asarray(self.targets, dtype=np.float32)
        mask = np.asarray(self.available_mask, dtype=bool)
        hidden = np.asarray(self.hidden_truth, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[0] != HORIZON_STEPS:
            raise Issue52RolloutError("rollout target shape is invalid")
        if mask.shape != targets.shape or hidden.shape != targets.shape:
            raise Issue52RolloutError("rollout target masks do not match")
        _validate_array_cells(targets, mask, label="rollout target")
        _validate_array_cells(hidden, mask, label="hidden truth")
        if (
            type(self.command_digests) is not tuple
            or type(self.state_digests) is not tuple
            or type(self.feasibility) is not tuple
            or len(self.command_digests) != HORIZON_STEPS
            or len(self.state_digests) != HORIZON_STEPS
            or len(self.feasibility) != HORIZON_STEPS
        ):
            raise Issue52RolloutError("rollout evidence length is invalid")
        if any(status not in {"FEASIBLE", "INFEASIBLE", "UNAVAILABLE"} for status in self.feasibility):
            raise Issue52RolloutError("rollout feasibility status is invalid")
        if self.eligible != bool(self.termination_reason is None and mask.all()):
            raise Issue52RolloutError("rollout eligibility does not match evidence")
        if self.eligible and (
            any(status != "FEASIBLE" for status in self.feasibility)
            or any(digest is None for digest in self.command_digests)
            or any(digest is None for digest in self.state_digests)
        ):
            raise Issue52RolloutError("eligible rollout evidence is incomplete")
        targets = targets.copy()
        mask = mask.copy()
        hidden = hidden.copy()
        targets.setflags(write=False)
        mask.setflags(write=False)
        hidden.setflags(write=False)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "available_mask", mask)
        object.__setattr__(self, "hidden_truth", hidden)
        payload = {
            "schema_version": f"{ISSUE52_SCHEMA_VERSION}.rollout_result",
            "candidate_id": self.candidate_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "schedule_sha256": self.schedule_sha256,
            "manifest_sha256": self.manifest_sha256,
            "targets": _array_payload(targets),
            "available_mask": mask.tolist(),
            "hidden_truth": _array_payload(hidden),
            "command_digests": list(self.command_digests),
            "state_digests": list(self.state_digests),
            "feasibility": list(self.feasibility),
            "termination_reason": self.termination_reason,
            "eligible": self.eligible,
        }
        if self.rollout_sha256 != _digest(payload):
            raise Issue52RolloutError("rollout result digest is inconsistent")

    def validate_integrity(self) -> None:
        """Revalidate content-addressed rollout evidence before consuming labels."""

        self.__post_init__()


def rollout_candidate(
    checkpoint: RolloutCheckpoint,
    candidate: CandidateSchedule,
    *,
    manifest: TargetManifest | None = None,
) -> RolloutResult:
    if type(checkpoint) is not RolloutCheckpoint or type(candidate) is not CandidateSchedule:
        raise Issue52RolloutError("rollout requires exact checkpoint and candidate")
    checkpoint.__post_init__()
    bound_manifest = checkpoint.manifest if manifest is None else manifest
    if bound_manifest.manifest_sha256 != checkpoint.manifest.manifest_sha256:
        raise Issue52RolloutError("rollout manifest does not bind checkpoint")
    scenario = checkpoint.scenario
    if candidate.candidate_id == "":
        raise Issue52RolloutError("rollout candidate ID is empty")
    targets = np.full((HORIZON_STEPS, bound_manifest.width), np.nan, dtype=np.float32)
    available = np.zeros((HORIZON_STEPS, bound_manifest.width), dtype=bool)
    hidden = np.full_like(targets, np.nan)
    command_digests: list[str | None] = [None] * HORIZON_STEPS
    state_digests: list[str | None] = [None] * HORIZON_STEPS
    feasibility: list[str] = ["UNAVAILABLE"] * HORIZON_STEPS
    reconstructed = checkpoint.clone()
    state = reconstructed.state
    sensor_memory = reconstructed.sensor_memory
    health_tracker = reconstructed.health_tracker
    previous_measurement = checkpoint.last_measurement
    previous_command = checkpoint.last_final_command
    termination_reason: str | None = None
    for horizon, requested in enumerate(candidate.commands):
        if termination_reason is not None:
            break
        try:
            command = validate_external_command(scenario, requested.to_mapping())
            preflight = preflight_external_command(
                scenario, state, command.to_mapping(), state.step
            )
            if preflight.classification != "FEASIBLE":
                raise InfeasibleActionError("offline preflight classified command infeasible")
            stepped = advance_one_step_with_command(
                scenario, state, command.to_mapping()
            )
            validate_external_step_result(
                scenario, state, command.to_mapping(), stepped
            )
            measurement = instrument_v5_operational_measurement(
                scenario, stepped.state, sensor_memory
            )
            health = reduce_health(
                measurement=measurement,
                scenario=scenario,
                contract=checkpoint.contract,
                previous_tracker=health_tracker,
                previous_measurement=previous_measurement,
                last_final_command=previous_command,
            )
            projected, projected_mask = targets_from_measurement(
                measurement, bound_manifest
            )
            exact = _hidden_targets(scenario, stepped.state, bound_manifest)
            if not projected_mask.all():
                raise Issue52RolloutError("required target became unavailable")
            if health.health_state in {"UNKNOWN", "CRITICAL"}:
                raise Issue52RolloutError(
                    f"rollout health state is {health.health_state.lower()}"
                )
        except Issue52RolloutError:
            raise
        except (InfeasibleActionError, ScenarioValidationError) as error:
            termination_reason = type(error).__name__
            feasibility[horizon] = "INFEASIBLE"
            break
        targets[horizon] = projected
        hidden[horizon] = exact
        available[horizon] = projected_mask
        command_digests[horizon] = command.sha256
        state_digests[horizon] = _digest(_state_mapping(stepped.state))
        feasibility[horizon] = "FEASIBLE"
        state = stepped.state
        sensor_memory = measurement.sensor_memory
        health_tracker = health.tracker
        previous_measurement = measurement
        previous_command = command
    result_payload = {
        "schema_version": f"{ISSUE52_SCHEMA_VERSION}.rollout_result",
        "candidate_id": candidate.candidate_id,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "schedule_sha256": candidate.schedule_sha256,
        "manifest_sha256": bound_manifest.manifest_sha256,
        "targets": _array_payload(targets),
        "available_mask": available.tolist(),
        "hidden_truth": _array_payload(hidden),
        "command_digests": command_digests,
        "state_digests": state_digests,
        "feasibility": feasibility,
        "termination_reason": termination_reason,
        "eligible": bool(termination_reason is None and available.all()),
    }
    return RolloutResult(
        candidate_id=candidate.candidate_id,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        schedule_sha256=candidate.schedule_sha256,
        manifest_sha256=bound_manifest.manifest_sha256,
        targets=targets,
        available_mask=available,
        hidden_truth=hidden,
        command_digests=tuple(command_digests),
        state_digests=tuple(state_digests),
        feasibility=tuple(feasibility),
        termination_reason=termination_reason,
        eligible=bool(termination_reason is None and available.all()),
        rollout_sha256=_digest(result_payload),
    )


def rollout_catalogue(
    checkpoint: RolloutCheckpoint,
    catalogue: CandidateCatalogue,
    *,
    manifest: TargetManifest | None = None,
) -> tuple[RolloutResult, ...]:
    if type(checkpoint) is not RolloutCheckpoint or type(catalogue) is not CandidateCatalogue:
        raise Issue52RolloutError("catalogue rollout requires exact inputs")
    checkpoint.__post_init__()
    if catalogue.scenario_sha256 != checkpoint.scenario_sha256:
        raise Issue52RolloutError("catalogue scenario does not bind checkpoint")
    if catalogue.topology_sha256 != checkpoint.topology_sha256:
        raise Issue52RolloutError("catalogue topology does not bind checkpoint")
    if catalogue.base_command_sha256 != checkpoint.last_final_command.sha256:
        raise Issue52RolloutError("catalogue is not bound to checkpoint command")
    results = tuple(
        rollout_candidate(checkpoint, candidate, manifest=manifest)
        for candidate in catalogue.candidates
    )
    if any(not result.eligible for result in results):
        raise Issue52RolloutError("decision group contains an incomplete candidate")
    return results


def training_samples_from_rollouts(
    checkpoint: RolloutCheckpoint,
    catalogue: CandidateCatalogue,
    rollouts: Sequence[RolloutResult],
    *,
    family_id: str | None = None,
    split: str = "TRAIN",
) -> tuple[TrainingSample, ...]:
    if split not in {"TRAIN", "VALIDATION", "FINAL"}:
        raise Issue52RolloutError("training split is invalid")
    checkpoint.__post_init__()
    history = ForecastHistory.from_records(checkpoint.history_records)
    by_id = {item.candidate_id: item for item in rollouts}
    if len(by_id) != len(tuple(rollouts)):
        raise Issue52RolloutError("rollout candidate IDs are duplicated")
    result: list[TrainingSample] = []
    base_family = checkpoint.family_id if family_id is None else family_id
    for candidate in catalogue.candidates:
        rollout = by_id.get(candidate.candidate_id)
        if rollout is None:
            raise Issue52RolloutError(
                f"rollout set is missing {candidate.candidate_id}"
            )
        if not rollout.eligible:
            raise Issue52RolloutError("training requires a complete decision group")
        rollout.validate_integrity()
        if (
            rollout.checkpoint_sha256 != checkpoint.checkpoint_sha256
            or rollout.schedule_sha256 != candidate.schedule_sha256
            or rollout.manifest_sha256 != checkpoint.manifest.manifest_sha256
        ):
            raise Issue52RolloutError("rollout does not bind its training sample")
        result.append(
            TrainingSample(
                family_id=base_family,
                split=split,
                scenario_sha256=checkpoint.scenario_sha256,
                manifest_sha256=checkpoint.manifest.manifest_sha256,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                schedule_sha256=candidate.schedule_sha256,
                history=history,
                schedule=candidate,
                targets=rollout.targets,
            )
        )
    return tuple(result)


def assess_rollout_feasibility(
    catalogue: CandidateCatalogue,
    rollouts: Sequence[RolloutResult],
    *,
    scenario: Scenario | None = None,
) -> tuple[CandidateFeasibility, ...]:
    static_by_id: dict[str, CandidateFeasibility] = {}
    if type(scenario) is Scenario:
        from .forecast_issue52 import assess_static_feasibility

        static_by_id = {
            item.candidate_id: item
            for item in assess_static_feasibility(scenario, catalogue)
        }
    by_id = {item.candidate_id: item for item in rollouts}
    result: list[CandidateFeasibility] = []
    for candidate in catalogue.candidates:
        rollout = by_id.get(candidate.candidate_id)
        static = static_by_id.get(candidate.candidate_id)
        static_status = (
            static.static_status
            if static is not None
            else "STATICALLY_VALID"
        )
        if rollout is None:
            result.append(
                CandidateFeasibility(
                    candidate.candidate_id,
                    static_status,
                    "NOT_EVALUATED",
                    "NOT_EVALUATED",
                    rollout_reason="missing_rollout",
                )
            )
        elif rollout.eligible:
            result.append(
                CandidateFeasibility(
                    candidate.candidate_id,
                    static_status,
                    "ROLLOUT_FEASIBLE",
                    "NOT_EVALUATED",
                )
            )
        else:
            result.append(
                CandidateFeasibility(
                    candidate.candidate_id,
                    static_status,
                    "ROLLOUT_INFEASIBLE",
                    "NOT_EVALUATED",
                    rollout_reason=rollout.termination_reason,
                )
            )
    return tuple(result)


__all__ = [
    "RolloutCheckpoint",
    "RolloutResult",
    "assess_rollout_feasibility",
    "build_offline_checkpoint",
    "rollout_candidate",
    "rollout_catalogue",
    "training_samples_from_rollouts",
]
