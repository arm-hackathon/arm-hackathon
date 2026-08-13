from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .control_trace import (
    StepReceipt,
    TerminalFailureReceipt,
    ControlTrace,
    _issue_control_trace,
    _issue_step_receipt,
    _issue_terminal_failure_receipt,
)
from .health import HealthReduction, HealthTracker, reduce_health
from .hmc_contract import HMCContract, HMCContractError
from .instrumentation import (
    OperationalMeasurement,
    SensorMemory,
    instrument_v5_operational_measurement,
)
from .physics import (
    CanonicalExternalCommand,
    StepResult,
    advance_one_step_with_command,
    command_from_achieved_state,
    initial_state,
    operating_mode_for_application_step,
    preflight_external_command,
    validate_external_step_result,
)
from .proposal import (
    ControlProposal,
    ProposalReceipt,
    ProposalValidationError,
    _issue_proposal_receipt,
    parse_control_proposal,
)
from .safety import (
    ArbitrationReceipt,
    _issue_arbitration_receipt,
    apply_operating_mode_policy,
    apply_reserve_policy,
    emergency_crosses_normal_reserve_floor,
    expand_emergency_action,
)
from .scenario import SCENARIO_SCHEMA_VERSION_V5, Scenario, ScenarioValidationError
from .snapshot import (
    ControlEvent,
    OperationalSnapshot,
    SnapshotVerificationError,
    SnapshotVerificationReceipt,
    VerifiedSnapshotHandle,
    _issue_control_event,
    _issue_operational_snapshot,
    _issue_receipt_control_event,
    _issue_snapshot_verification_receipt,
    verify_issued_snapshot_pair,
)
from .state import PlantState
from .telemetry import ObservableTopology, derive_observable_topology


class HMCResetValidationError(ValueError):
    """Raised before an HMC exists when reset inputs are not closed and current."""


_SCENARIO_IDENTITY_FIELDS = (
    "canonical_bytes",
    "scenario_sha256",
    "scenario_schema_version",
    "trace_schema_version",
    "equation_contract_revision",
    "actuator_feedback_contract_revision",
    "run_id",
)


class LifecyclePhase(str, Enum):
    RESET = "RESET"
    OBSERVED = "OBSERVED"
    PROPOSED = "PROPOSED"
    ARBITRATED = "ARBITRATED"
    STEPPED = "STEPPED"
    TERMINAL = "TERMINAL"


_STEP_CAPABILITY_TOKEN = object()


@dataclass(frozen=True, init=False, slots=True)
class _StepCapability:
    owner_identity: int
    control_run_id: str
    authority_epoch: str
    scenario_sha256: str
    application_step: int
    final_command_sha256: str

    def __init__(
        self,
        *,
        owner_identity: int,
        control_run_id: str,
        authority_epoch: str,
        scenario_sha256: str,
        application_step: int,
        final_command_sha256: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _STEP_CAPABILITY_TOKEN:
            raise TypeError("step capability must be issued by an HMC")
        object.__setattr__(self, "owner_identity", owner_identity)
        object.__setattr__(self, "control_run_id", control_run_id)
        object.__setattr__(self, "authority_epoch", authority_epoch)
        object.__setattr__(self, "scenario_sha256", scenario_sha256)
        object.__setattr__(self, "application_step", application_step)
        object.__setattr__(self, "final_command_sha256", final_command_sha256)

    def __reduce__(self) -> object:
        raise TypeError("step capability is not serialisable")


@dataclass(frozen=True, slots=True)
class _StagedCycle:
    state: PlantState
    measurement: OperationalMeasurement
    health_tracker: HealthTracker
    step_receipt: StepReceipt
    plant_receipt_digest: str
    sequence: int
    snapshot: OperationalSnapshot
    verification_receipt: SnapshotVerificationReceipt
    step_event: ControlEvent
    snapshot_event: ControlEvent


def _decode_sha256(value: str, *, label: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HMCResetValidationError(f"{label} must be lowercase SHA-256 hex")
    return bytes.fromhex(value)


def _domain_hash(label: str, *parts: bytes) -> str:
    return hashlib.sha256(label.encode("utf-8") + b"".join(parts)).hexdigest()


@dataclass(slots=True)
class HabitatManagementComputer:
    _scenario: Scenario
    _contract: HMCContract
    _reset_nonce: bytes
    _state: PlantState
    _observable_topology: ObservableTopology
    _snapshot_schema_sha256: str
    _control_run_id: str
    _authority_epoch: str
    _phase: LifecyclePhase
    _sensor_memory: SensorMemory | None
    _health_tracker: HealthTracker
    _last_operational_measurement: OperationalMeasurement | None
    _cached_snapshot: OperationalSnapshot | None
    _cached_verification_receipt: SnapshotVerificationReceipt | None
    _cached_proposal_receipt: ProposalReceipt | None
    _cached_control_proposal: ControlProposal | None
    _cached_arbitration_receipt: ArbitrationReceipt | None
    _cached_final_command: CanonicalExternalCommand | None
    _step_capability: _StepCapability | None
    _last_step_receipt: StepReceipt | None
    _terminal_failure_receipt: TerminalFailureReceipt | None
    _last_plant_receipt_digest: str | None
    _sequence: int
    _control_events: list[ControlEvent]
    _current_control_chain_sha256: str
    _verified_handles: dict[tuple[int, int], VerifiedSnapshotHandle]

    @property
    def snapshot_schema_sha256(self) -> str:
        return self._snapshot_schema_sha256

    @property
    def observable_topology_sha256(self) -> str:
        return self._observable_topology.sha256

    @property
    def control_run_id(self) -> str:
        return self._control_run_id

    @property
    def authority_epoch(self) -> str:
        return self._authority_epoch

    @property
    def lifecycle_phase(self) -> str:
        return self._phase.value

    @property
    def current_control_chain_sha256(self) -> str:
        return self._current_control_chain_sha256

    @property
    def control_events(self) -> tuple[ControlEvent, ...]:
        return tuple(self._control_events)

    @classmethod
    def reset(
        cls,
        scenario: Scenario,
        contract: HMCContract,
        reset_nonce: bytes,
    ) -> HabitatManagementComputer:
        if type(scenario) is not Scenario:
            raise HMCResetValidationError("reset requires the exact Scenario type")
        if type(contract) is not HMCContract:
            raise HMCResetValidationError("reset requires the exact HMCContract type")
        if type(reset_nonce) is not bytes or len(reset_nonce) != 32:
            raise HMCResetValidationError(
                "reset nonce must be exact bytes of length 32"
            )

        try:
            parsed_scenario = Scenario.from_mapping(scenario.data)
        except ScenarioValidationError as error:
            raise HMCResetValidationError(
                "scenario does not satisfy the current closed scenario schema"
            ) from error
        for field in _SCENARIO_IDENTITY_FIELDS:
            if getattr(parsed_scenario, field) != getattr(scenario, field):
                raise HMCResetValidationError(
                    f"scenario {field} does not match the closed-schema reparse"
                )
        try:
            contract_mapping: Any = json.loads(contract.canonical_bytes)
            parsed_contract = HMCContract.from_mapping(contract_mapping)
        except (UnicodeDecodeError, json.JSONDecodeError, HMCContractError) as error:
            raise HMCResetValidationError(
                "contract does not satisfy the current closed HMC schema"
            ) from error
        if (
            parsed_contract.canonical_bytes != contract.canonical_bytes
            or parsed_contract.hmc_contract_sha256 != contract.hmc_contract_sha256
        ):
            raise HMCResetValidationError(
                "contract identity does not match the closed-schema reparse"
            )
        if parsed_scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
            raise HMCResetValidationError("HMC V1 requires a V5 scenario")

        topology = derive_observable_topology(parsed_scenario)
        reviewed_noise = parsed_contract.reviewed_noise_configuration
        scenario_noise = parsed_scenario.data["sensor_model"]
        if (
            scenario_noise["primary_noise_amplitude"]
            != reviewed_noise["primary_noise_amplitude"]
            or scenario_noise["secondary_noise_amplitude"]
            != reviewed_noise["secondary_noise_amplitude"]
        ):
            raise HMCResetValidationError(
                "scenario environmental sensor noise does not match HMC contract"
            )
        if (
            parsed_scenario.data["actuator_feedback"]["feedback_sensor_noise_amplitude"]
            != reviewed_noise["feedback_sensor_noise_amplitude"]
        ):
            raise HMCResetValidationError(
                "scenario feedback sensor noise does not match HMC contract"
            )
        snapshot_schema_sha256 = parsed_contract.snapshot_schema_sha256
        control_run_id = _domain_hash(
            "aeolus-habitat-v2-hmc-run-v1",
            _decode_sha256(parsed_scenario.scenario_sha256, label="scenario identity"),
            _decode_sha256(
                parsed_contract.hmc_contract_sha256, label="HMC contract identity"
            ),
            _decode_sha256(snapshot_schema_sha256, label="snapshot schema identity"),
            _decode_sha256(topology.sha256, label="observable topology identity"),
            reset_nonce,
        )
        authority_epoch = _domain_hash(
            "aeolus-habitat-v2-hmc-epoch-v1",
            _decode_sha256(control_run_id, label="control run identity"),
            reset_nonce,
        )
        state = initial_state(parsed_scenario)
        null_control_chain = str(
            parsed_contract.data["null_roots"]["control_chain"]["sha256"]
        )
        return cls(
            _scenario=parsed_scenario,
            _contract=parsed_contract,
            _reset_nonce=bytes(reset_nonce),
            _state=state,
            _observable_topology=topology,
            _snapshot_schema_sha256=snapshot_schema_sha256,
            _control_run_id=control_run_id,
            _authority_epoch=authority_epoch,
            _phase=LifecyclePhase.RESET,
            _sensor_memory=None,
            _health_tracker=HealthTracker.initial(),
            _last_operational_measurement=None,
            _cached_snapshot=None,
            _cached_verification_receipt=None,
            _cached_proposal_receipt=None,
            _cached_control_proposal=None,
            _cached_arbitration_receipt=None,
            _cached_final_command=None,
            _step_capability=None,
            _last_step_receipt=None,
            _terminal_failure_receipt=None,
            _last_plant_receipt_digest=None,
            _sequence=0,
            _control_events=[],
            _current_control_chain_sha256=null_control_chain,
            _verified_handles={},
        )

    def observe(
        self,
    ) -> (
        tuple[OperationalSnapshot, SnapshotVerificationReceipt] | TerminalFailureReceipt
    ):
        if self._phase in {LifecyclePhase.OBSERVED, LifecyclePhase.STEPPED}:
            if (
                self._cached_snapshot is None
                or self._cached_verification_receipt is None
            ):
                raise RuntimeError(
                    "observable lifecycle is missing its cached evidence"
                )
            self._phase = LifecyclePhase.OBSERVED
            return self._cached_snapshot, self._cached_verification_receipt
        if self._phase is not LifecyclePhase.RESET:
            raise RuntimeError(
                f"observe is not valid during lifecycle phase {self._phase.value}"
            )

        try:
            measurement = instrument_v5_operational_measurement(
                self._scenario,
                self._state,
                None,
            )
        except Exception:
            return self._enter_terminal(
                reason_code="OPERATIONAL_MEASUREMENT_INVALID",
                application_step=None,
            )
        try:
            health = reduce_health(
                measurement=measurement,
                scenario=self._scenario,
                contract=self._contract,
                previous_tracker=self._health_tracker,
                previous_measurement=self._last_operational_measurement,
                last_final_command=None,
            )
        except Exception:
            return self._enter_terminal(
                reason_code="HEALTH_REDUCTION_FAILED",
                application_step=None,
            )

        try:
            hold = command_from_achieved_state(self._scenario, self._state)
            null_roots = self._contract.data["null_roots"]
            null_plant = str(null_roots["plant_receipt"]["sha256"])
            null_step = str(null_roots["step_receipt"]["sha256"])
            null_verification = str(null_roots["verification_receipt"]["sha256"])
            primary = [sample.to_mapping() for sample in measurement.primary]
            secondary = [sample.to_mapping() for sample in measurement.secondary]
            disagreement = [
                sample.to_mapping() for sample in measurement.primary_minus_secondary
            ]
            feedback = [
                sample.to_mapping() for sample in measurement.operational_feedback
            ]
            feedback_by_id = {
                str(sample["descriptor_id"]): sample for sample in feedback
            }
            resource_gauges = [
                feedback_by_id[channel_id]
                for channel_id in (
                    self._observable_topology.operational_resource_gauge_channels
                )
            ]
            snapshot = _issue_operational_snapshot(
                {
                    "schema_version": self._contract.snapshot_schema_version,
                    "control_run_id": self._control_run_id,
                    "authority_epoch": self._authority_epoch,
                    "sequence": 0,
                    "completed_step": measurement.completed_step,
                    "completed_time_s": measurement.completed_time_s,
                    "completed_application_step": None,
                    "completed_operating_mode": None,
                    "primary_telemetry": {
                        "source_kind": "primary_sensor_head",
                        "samples": primary,
                    },
                    "secondary_telemetry": {
                        "source_kind": "secondary_sensor_head",
                        "samples": secondary,
                    },
                    "primary_minus_secondary": {
                        "source_kind": "derived_primary_minus_secondary",
                        "samples": disagreement,
                    },
                    "command_reference": {
                        "source_kind": "authoritative_command_reference",
                        "command_reference_kind": hold.command_reference_kind,
                        "command": hold.command.to_mapping(),
                    },
                    "operational_feedback": {
                        "source_kind": "operational_feedback_instrument",
                        "samples": feedback,
                    },
                    "operational_resource_gauges": {
                        "source_kind": "operational_resource_gauge",
                        "samples": resource_gauges,
                    },
                    "derived_health": {
                        "source_kind": "derived_health",
                        "health_state": health.health_state,
                    },
                    "active_operational_alarms": {
                        "source_kind": "alarm_receipt",
                        "alarms": [alarm.to_mapping() for alarm in health.alarms],
                    },
                    "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                    "snapshot_schema_sha256": self._snapshot_schema_sha256,
                    "observable_topology_sha256": self._observable_topology.sha256,
                    "completed_plant_receipt_digest": null_plant,
                    "completed_step_receipt_digest": null_step,
                }
            )
            issuer_id = _domain_hash(
                "aeolus-habitat-v2-hmc-issuer-v1",
                _decode_sha256(self._control_run_id, label="control run identity"),
                _decode_sha256(self._authority_epoch, label="authority epoch"),
                _decode_sha256(
                    self._contract.hmc_contract_sha256,
                    label="HMC contract identity",
                ),
            )
            cycle_id = _domain_hash(
                "aeolus-habitat-v2-hmc-cycle-v1",
                _decode_sha256(self._control_run_id, label="control run identity"),
                _decode_sha256(self._authority_epoch, label="authority epoch"),
                (0).to_bytes(8, "big"),
                _decode_sha256(snapshot.snapshot_sha256, label="snapshot identity"),
            )
            receipt = _issue_snapshot_verification_receipt(
                {
                    "receipt_schema_sha256": (
                        self._contract.snapshot_verification_receipt_schema_sha256
                    ),
                    "snapshot_verification_contract_sha256": (
                        self._contract.snapshot_verification_contract_sha256
                    ),
                    "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                    "snapshot_schema_sha256": self._snapshot_schema_sha256,
                    "observable_topology_sha256": self._observable_topology.sha256,
                    "control_run_id": self._control_run_id,
                    "authority_epoch": self._authority_epoch,
                    "issuer_id": issuer_id,
                    "cycle_id": cycle_id,
                    "sequence": 0,
                    "completed_step": measurement.completed_step,
                    "completed_time_s": measurement.completed_time_s,
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "completed_plant_receipt_digest": null_plant,
                    "completed_step_receipt_digest": null_step,
                    "previous_verification_receipt_digest": null_verification,
                    "event_ordinal": 0,
                    "previous_control_chain_sha256": (
                        self._current_control_chain_sha256
                    ),
                }
            )
            event = _issue_control_event(
                event_ordinal=0,
                event_kind="SNAPSHOT_VERIFICATION",
                receipt=receipt,
            )
        except Exception:
            return self._enter_terminal(
                reason_code="SNAPSHOT_ISSUANCE_FAILED",
                application_step=None,
            )
        self._sensor_memory = measurement.sensor_memory
        self._health_tracker = health.tracker
        self._last_operational_measurement = measurement
        self._cached_snapshot = snapshot
        self._cached_verification_receipt = receipt
        self._control_events.append(event)
        self._current_control_chain_sha256 = event.control_chain_sha256
        self._phase = LifecyclePhase.OBSERVED
        return snapshot, receipt

    def verify_snapshot(
        self,
        snapshot: OperationalSnapshot,
        receipt: SnapshotVerificationReceipt,
    ) -> VerifiedSnapshotHandle:
        if self._phase is LifecyclePhase.TERMINAL:
            raise RuntimeError(
                "verify_snapshot is not valid during lifecycle phase TERMINAL"
            )
        if (
            snapshot is not self._cached_snapshot
            or receipt is not self._cached_verification_receipt
            or not self._control_events
        ):
            raise SnapshotVerificationError(
                "snapshot verification requires the exact issued snapshot and receipt"
            )
        key = (id(snapshot), id(receipt))
        cached = self._verified_handles.get(key)
        if cached is not None:
            return cached
        event = self._control_events[-1]
        handle = verify_issued_snapshot_pair(
            snapshot=snapshot,
            receipt=receipt,
            event=event,
        )
        if (
            receipt.control_run_id != self._control_run_id
            or receipt.authority_epoch != self._authority_epoch
            or receipt.hmc_contract_sha256 != self._contract.hmc_contract_sha256
            or receipt.snapshot_schema_sha256 != self._snapshot_schema_sha256
            or receipt.observable_topology_sha256 != self._observable_topology.sha256
        ):
            raise SnapshotVerificationError(
                "snapshot verification identities do not match the issuing HMC"
            )
        self._verified_handles[key] = handle
        return handle

    def propose(self, proposal: object | None) -> ProposalReceipt:
        if self._phase is not LifecyclePhase.OBSERVED:
            raise RuntimeError(
                f"propose is not valid during lifecycle phase {self._phase.value}"
            )
        if self._cached_snapshot is None or self._cached_verification_receipt is None:
            raise RuntimeError("OBSERVED lifecycle is missing its cached evidence")
        event_ordinal = len(self._control_events)
        common = {
            "receipt_schema_sha256": self._contract.proposal_receipt_schema_sha256,
            "hmc_contract_sha256": self._contract.hmc_contract_sha256,
            "observable_topology_sha256": self._observable_topology.sha256,
            "control_run_id": self._control_run_id,
            "authority_epoch": self._authority_epoch,
            "sequence": self._sequence,
            "observation_snapshot_sha256": self._cached_snapshot.snapshot_sha256,
            "requested_application_step": self._state.step,
        }
        if proposal is None:
            attempt = {
                "attempt_class": "NONE",
                "attempt_evidence_sha256": str(
                    self._contract.data["null_roots"]["proposal_receipt"]["sha256"]
                ),
                "source_id": None,
                "source_type": None,
                "proposal": None,
                "proposal_sha256": None,
                "requested_command_sha256": None,
                "validation_outcome": "NO_PROPOSAL",
                "reason_code": "no_proposal",
            }
        elif type(proposal) is not dict:
            reason_code = "rejected_malformed"
            reason_bytes = reason_code.encode("utf-8")
            attempt = {
                "attempt_class": "REJECTED_INPUT",
                "attempt_evidence_sha256": hashlib.sha256(
                    b"aeolus-habitat-v2-hmc-rejected-proposal-v1"
                    + len(reason_bytes).to_bytes(8, "big")
                    + reason_bytes
                ).hexdigest(),
                "source_id": None,
                "source_type": None,
                "proposal": None,
                "proposal_sha256": None,
                "requested_command_sha256": None,
                "validation_outcome": "REJECTED",
                "reason_code": reason_code,
            }
        else:
            try:
                evidence_bytes = json.dumps(
                    proposal,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            except (TypeError, ValueError):
                reason_code = "rejected_malformed"
                reason_bytes = reason_code.encode("utf-8")
                evidence_bytes = (
                    b"aeolus-habitat-v2-hmc-rejected-proposal-v1"
                    + len(reason_bytes).to_bytes(8, "big")
                    + reason_bytes
                )
                parsed_proposal = None
            else:
                try:
                    parsed_proposal = parse_control_proposal(
                        proposal,
                        scenario=self._scenario,
                        control_run_id=self._control_run_id,
                        authority_epoch=self._authority_epoch,
                        observable_topology_sha256=self._observable_topology.sha256,
                        completed_observation_step=self._state.step,
                        observation_snapshot_sha256=(
                            self._cached_snapshot.snapshot_sha256
                        ),
                        requested_application_step=self._state.step,
                    )
                except ProposalValidationError as error:
                    reason_code = error.reason_code
                    parsed_proposal = None
            if parsed_proposal is None:
                attempt = {
                    "attempt_class": "REJECTED_INPUT",
                    "attempt_evidence_sha256": hashlib.sha256(
                        evidence_bytes
                    ).hexdigest(),
                    "source_id": None,
                    "source_type": None,
                    "proposal": None,
                    "proposal_sha256": None,
                    "requested_command_sha256": None,
                    "validation_outcome": "REJECTED",
                    "reason_code": reason_code,
                }
            else:
                attempt = {
                    "attempt_class": "CANONICAL_PROPOSAL",
                    "attempt_evidence_sha256": parsed_proposal.proposal_sha256,
                    "source_id": parsed_proposal.source_id,
                    "source_type": parsed_proposal.source_type,
                    "proposal": parsed_proposal.to_mapping(),
                    "proposal_sha256": parsed_proposal.proposal_sha256,
                    "requested_command_sha256": (
                        parsed_proposal.requested_command.sha256
                    ),
                    "validation_outcome": "VALID",
                    "reason_code": "valid",
                }
        receipt = _issue_proposal_receipt(
            {
                **common,
                **attempt,
                "event_ordinal": event_ordinal,
                "previous_control_chain_sha256": self._current_control_chain_sha256,
            }
        )
        event = _issue_receipt_control_event(
            event_ordinal=event_ordinal,
            event_kind="PROPOSAL",
            receipt_mapping=receipt.to_mapping(),
            receipt_sha256=receipt.proposal_receipt_sha256,
            previous_control_chain_sha256=receipt.previous_control_chain_sha256,
        )
        self._cached_proposal_receipt = receipt
        self._cached_control_proposal = (
            parsed_proposal if proposal is not None and type(proposal) is dict else None
        )
        # A new proposal starts a new arbitration attempt. The last completed
        # final command remains the safe-hold baseline, but prior-cycle
        # arbitration authority must not leak into current-cycle evidence.
        self._cached_arbitration_receipt = None
        self._step_capability = None
        self._control_events.append(event)
        self._current_control_chain_sha256 = event.control_chain_sha256
        self._phase = LifecyclePhase.PROPOSED
        return receipt

    def _safe_hold_command(self) -> CanonicalExternalCommand:
        if self._sequence == 0 or self._cached_final_command is None:
            return command_from_achieved_state(self._scenario, self._state).command
        return self._cached_final_command

    def _enter_terminal(
        self,
        *,
        reason_code: str,
        application_step: int | None,
        candidate_plant_receipt_digest: str | None = None,
        final_command_sha256: str | None = None,
    ) -> TerminalFailureReceipt:
        if self._phase is LifecyclePhase.TERMINAL:
            raise RuntimeError("terminal transition has already been recorded")
        null_roots = self._contract.data["null_roots"]
        last_good_snapshot_sha256 = (
            str(null_roots["snapshot"]["sha256"])
            if self._cached_snapshot is None
            else self._cached_snapshot.snapshot_sha256
        )
        last_good_verification_receipt_sha256 = (
            str(null_roots["verification_receipt"]["sha256"])
            if self._cached_verification_receipt is None
            else self._cached_verification_receipt.snapshot_verification_receipt_sha256
        )
        last_good_step_receipt_sha256 = (
            str(null_roots["step_receipt"]["sha256"])
            if self._last_step_receipt is None
            else self._last_step_receipt.step_receipt_sha256
        )
        proposal_receipt_sha256 = (
            None
            if self._cached_proposal_receipt is None
            else self._cached_proposal_receipt.proposal_receipt_sha256
        )
        arbitration_receipt_sha256 = (
            None
            if self._cached_arbitration_receipt is None
            else self._cached_arbitration_receipt.arbitration_receipt_sha256
        )
        if final_command_sha256 is None and self._cached_final_command is not None:
            final_command_sha256 = self._cached_final_command.sha256
        event_ordinal = len(self._control_events)
        receipt = _issue_terminal_failure_receipt(
            {
                "receipt_schema_sha256": self._contract.terminal_receipt_schema_sha256,
                # V1 freezes one terminal receipt schema identity and names this
                # repeated field terminal_contract_sha256. Ratification remains
                # a merge blocker, so the draft binds the explicit schema hash.
                "terminal_contract_sha256": self._contract.terminal_receipt_schema_sha256,
                "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                "observable_topology_sha256": self._observable_topology.sha256,
                "control_run_id": self._control_run_id,
                "authority_epoch": self._authority_epoch,
                "sequence": self._sequence,
                "application_step": application_step,
                "lifecycle_phase": self._phase.value,
                "last_good_snapshot_sha256": last_good_snapshot_sha256,
                "last_good_verification_receipt_sha256": (
                    last_good_verification_receipt_sha256
                ),
                "last_good_step_receipt_sha256": last_good_step_receipt_sha256,
                "proposal_receipt_sha256": proposal_receipt_sha256,
                "arbitration_receipt_sha256": arbitration_receipt_sha256,
                "final_command_sha256": final_command_sha256,
                "candidate_plant_receipt_digest": candidate_plant_receipt_digest,
                "plant_state_committed": False,
                "reason_code": reason_code,
                "event_ordinal": event_ordinal,
                "previous_control_chain_sha256": self._current_control_chain_sha256,
            },
            contract=self._contract,
        )
        event = _issue_receipt_control_event(
            event_ordinal=event_ordinal,
            event_kind="TERMINAL",
            receipt_mapping=receipt.to_mapping(),
            receipt_sha256=receipt.terminal_failure_receipt_sha256,
            previous_control_chain_sha256=self._current_control_chain_sha256,
        )
        self._step_capability = None
        self._terminal_failure_receipt = receipt
        self._control_events.append(event)
        self._current_control_chain_sha256 = event.control_chain_sha256
        self._phase = LifecyclePhase.TERMINAL
        return receipt

    def arbitrate(self) -> ArbitrationReceipt | TerminalFailureReceipt:
        if self._phase is not LifecyclePhase.PROPOSED:
            raise RuntimeError(
                f"arbitrate is not valid during lifecycle phase {self._phase.value}"
            )
        if (
            self._cached_snapshot is None
            or self._cached_proposal_receipt is None
            or not self._control_events
        ):
            raise RuntimeError("PROPOSED lifecycle is missing its cached evidence")
        application_step = self._state.step
        imminent_mode = operating_mode_for_application_step(
            self._scenario,
            application_step,
        )
        snapshot_mapping = self._cached_snapshot.to_mapping()
        health_state = snapshot_mapping["derived_health"]["health_state"]
        alarms = snapshot_mapping["active_operational_alarms"]["alarms"]
        resource_samples = snapshot_mapping["operational_resource_gauges"]["samples"]
        safe_hold = self._safe_hold_command()
        valid_proposal = (
            self._cached_proposal_receipt.attempt_class == "CANONICAL_PROPOSAL"
            and self._cached_control_proposal is not None
        )
        requested_command = (
            self._cached_control_proposal.requested_command.to_mapping()
            if valid_proposal
            else None
        )
        requested_command_sha256 = (
            self._cached_control_proposal.requested_command.sha256
            if valid_proposal
            else None
        )
        accepted_proposal_sha256 = None
        emergency_override = False
        emergency_reserve_use = False

        if health_state == "UNKNOWN":
            final_command = safe_hold
            disposition = "REJECTED"
            reason_codes = ["safe_hold_telemetry_unknown"]
            command_owner = "baseline_hold"
        else:
            emergency_command = (
                expand_emergency_action(
                    scenario=self._scenario,
                    safe_hold=safe_hold,
                    alarms=alarms,
                    catalogue=self._contract.data["safe_action_catalogue"],
                )
                if health_state == "CRITICAL"
                else None
            )
            if health_state == "CRITICAL":
                emergency_override = True
                final_command = (
                    safe_hold if emergency_command is None else emergency_command
                )
                emergency_reserve_use = (
                    False
                    if emergency_command is None
                    else emergency_crosses_normal_reserve_floor(
                        emergency_command=emergency_command,
                        safe_hold=safe_hold,
                        resource_samples=resource_samples,
                        reserve_floors=self._contract.data["safety_policy"][
                            "reserve_floors"
                        ],
                    )
                )
                disposition = "MODIFIED" if valid_proposal else "REJECTED"
                reason_codes = ["emergency_override"]
                command_owner = (
                    "baseline_hold"
                    if emergency_command is None
                    else "emergency_safe_action"
                )
            elif self._cached_proposal_receipt.attempt_class == "NONE":
                final_command = safe_hold
                requested_command = None
                requested_command_sha256 = None
                disposition = "REJECTED"
                reason_codes = ["safe_hold_no_proposal"]
                command_owner = "baseline_hold"
            elif self._cached_proposal_receipt.attempt_class == "REJECTED_INPUT":
                final_command = safe_hold
                requested_command = None
                requested_command_sha256 = None
                disposition = "REJECTED"
                reason_codes = [self._cached_proposal_receipt.reason_code]
                command_owner = "baseline_hold"
            elif valid_proposal:
                final_command, mode_modified = apply_operating_mode_policy(
                    scenario=self._scenario,
                    imminent_mode=imminent_mode,
                    requested=self._cached_control_proposal.requested_command,
                    safe_hold=safe_hold,
                )
                final_command, reserve_modified = apply_reserve_policy(
                    scenario=self._scenario,
                    candidate=final_command,
                    safe_hold=safe_hold,
                    resource_samples=resource_samples,
                    reserve_floors=self._contract.data["safety_policy"][
                        "reserve_floors"
                    ],
                )
                accepted_proposal_sha256 = self._cached_control_proposal.proposal_sha256
                disposition = (
                    "MODIFIED" if mode_modified or reserve_modified else "ACCEPTED"
                )
                reason_codes = []
                if reserve_modified:
                    reason_codes.append("modified_resource_reserve")
                if mode_modified:
                    reason_codes.append("modified_operating_mode_rule")
                if not reason_codes:
                    reason_codes.append("accepted_as_proposed")
                command_owner = "supervisor_modified"
            else:
                raise RuntimeError("proposal lifecycle evidence is inconsistent")

        def attempt_preflight(command: CanonicalExternalCommand):
            try:
                return preflight_external_command(
                    self._scenario,
                    self._state,
                    command.to_mapping(),
                    application_step,
                )
            except Exception:
                return None

        preflight = attempt_preflight(final_command)
        if (
            preflight is None
            or preflight.classification != "FEASIBLE"
            or preflight.preflight_contract_sha256
            != self._contract.preflight_contract_sha256
        ):
            if final_command.sha256 != safe_hold.sha256:
                final_command = safe_hold
                accepted_proposal_sha256 = None
                emergency_reserve_use = False
                disposition = "REJECTED"
                command_owner = "baseline_hold"
                if emergency_override:
                    reason_codes = ["emergency_override"]
                else:
                    reason_codes = ["rejected_resource_infeasible"]
                preflight = attempt_preflight(safe_hold)
            if (
                preflight is None
                or preflight.classification != "FEASIBLE"
                or preflight.preflight_contract_sha256
                != self._contract.preflight_contract_sha256
            ):
                return self._enter_terminal(
                    reason_code="SAFE_HOLD_INFEASIBLE",
                    application_step=application_step,
                    final_command_sha256=safe_hold.sha256,
                )
        event_ordinal = len(self._control_events)
        receipt = _issue_arbitration_receipt(
            {
                "receipt_schema_sha256": (
                    self._contract.arbitration_receipt_schema_sha256
                ),
                "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                "safety_policy_sha256": self._contract.safety_policy_sha256,
                "safe_action_catalogue_sha256": (
                    self._contract.safe_action_catalogue_sha256
                ),
                "preflight_contract_sha256": self._contract.preflight_contract_sha256,
                "observable_topology_sha256": self._observable_topology.sha256,
                "control_run_id": self._control_run_id,
                "authority_epoch": self._authority_epoch,
                "sequence": self._sequence,
                "observation_snapshot_sha256": self._cached_snapshot.snapshot_sha256,
                "proposal_receipt_sha256": (
                    self._cached_proposal_receipt.proposal_receipt_sha256
                ),
                "accepted_proposal_sha256": accepted_proposal_sha256,
                "requested_command": requested_command,
                "requested_command_sha256": requested_command_sha256,
                "final_command": final_command.to_mapping(),
                "final_command_sha256": final_command.sha256,
                "disposition": disposition,
                "reason_codes": reason_codes,
                "command_owner": command_owner,
                "emergency_override": emergency_override,
                "emergency_reserve_use": emergency_reserve_use,
                "imminent_application_mode": imminent_mode,
                "preflight_result": preflight.to_mapping(),
                "decision_step": self._state.step,
                "application_step": application_step,
                "event_ordinal": event_ordinal,
                "previous_control_chain_sha256": self._current_control_chain_sha256,
            },
            scenario=self._scenario,
            contract=self._contract,
        )
        event = _issue_receipt_control_event(
            event_ordinal=event_ordinal,
            event_kind="ARBITRATION",
            receipt_mapping=receipt.to_mapping(),
            receipt_sha256=receipt.arbitration_receipt_sha256,
            previous_control_chain_sha256=receipt.to_mapping()[
                "previous_control_chain_sha256"
            ],
        )
        self._cached_arbitration_receipt = receipt
        self._cached_final_command = final_command
        self._step_capability = _StepCapability(
            owner_identity=id(self),
            control_run_id=self._control_run_id,
            authority_epoch=self._authority_epoch,
            scenario_sha256=self._scenario.scenario_sha256,
            application_step=application_step,
            final_command_sha256=final_command.sha256,
            _token=_STEP_CAPABILITY_TOKEN,
        )
        self._control_events.append(event)
        self._current_control_chain_sha256 = event.control_chain_sha256
        self._phase = LifecyclePhase.ARBITRATED
        return receipt

    def _stage_completed_cycle(
        self,
        *,
        candidate: StepResult,
        final_command: CanonicalExternalCommand,
        returned_digest: str,
        plant_receipt_digest: str,
        application_step: int,
        next_measurement: OperationalMeasurement,
        next_health: HealthReduction,
    ) -> _StagedCycle:
        if (
            self._cached_proposal_receipt is None
            or self._cached_arbitration_receipt is None
            or self._cached_verification_receipt is None
        ):
            raise RuntimeError(
                "cycle staging is missing committed predecessor evidence"
            )
        previous_step_digest = (
            str(self._contract.data["null_roots"]["step_receipt"]["sha256"])
            if self._last_step_receipt is None
            else self._last_step_receipt.step_receipt_sha256
        )
        step_event_ordinal = len(self._control_events)
        step_receipt = _issue_step_receipt(
            {
                "receipt_schema_sha256": self._contract.step_receipt_schema_sha256,
                "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                "external_command_contract_sha256": (
                    self._contract.external_command_contract_sha256
                ),
                "observable_topology_sha256": self._observable_topology.sha256,
                "control_run_id": self._control_run_id,
                "authority_epoch": self._authority_epoch,
                "observation_sequence": self._sequence,
                "application_step": application_step,
                "proposal_receipt_sha256": (
                    self._cached_proposal_receipt.proposal_receipt_sha256
                ),
                "arbitration_receipt_sha256": (
                    self._cached_arbitration_receipt.arbitration_receipt_sha256
                ),
                "final_command_sha256": final_command.sha256,
                "returned_external_command_digest": returned_digest,
                "plant_receipt_digest": plant_receipt_digest,
                "application_outcome": "APPLIED",
                "previous_step_receipt_digest": previous_step_digest,
                "event_ordinal": step_event_ordinal,
                "previous_control_chain_sha256": self._current_control_chain_sha256,
            },
            contract=self._contract,
        )
        step_event = _issue_receipt_control_event(
            event_ordinal=step_event_ordinal,
            event_kind="STEP",
            receipt_mapping=step_receipt.to_mapping(),
            receipt_sha256=step_receipt.step_receipt_sha256,
            previous_control_chain_sha256=self._current_control_chain_sha256,
        )
        feedback = [
            sample.to_mapping() for sample in next_measurement.operational_feedback
        ]
        feedback_by_id = {str(sample["descriptor_id"]): sample for sample in feedback}
        next_sequence = self._sequence + 1
        next_snapshot = _issue_operational_snapshot(
            {
                "schema_version": self._contract.snapshot_schema_version,
                "control_run_id": self._control_run_id,
                "authority_epoch": self._authority_epoch,
                "sequence": next_sequence,
                "completed_step": next_measurement.completed_step,
                "completed_time_s": next_measurement.completed_time_s,
                "completed_application_step": application_step,
                "completed_operating_mode": self._cached_arbitration_receipt.to_mapping()[
                    "imminent_application_mode"
                ],
                "primary_telemetry": {
                    "source_kind": "primary_sensor_head",
                    "samples": [
                        sample.to_mapping() for sample in next_measurement.primary
                    ],
                },
                "secondary_telemetry": {
                    "source_kind": "secondary_sensor_head",
                    "samples": [
                        sample.to_mapping() for sample in next_measurement.secondary
                    ],
                },
                "primary_minus_secondary": {
                    "source_kind": "derived_primary_minus_secondary",
                    "samples": [
                        sample.to_mapping()
                        for sample in next_measurement.primary_minus_secondary
                    ],
                },
                "command_reference": {
                    "source_kind": "authoritative_command_reference",
                    "command_reference_kind": "COMPLETED_FINAL_COMMAND",
                    "command": final_command.to_mapping(),
                },
                "operational_feedback": {
                    "source_kind": "operational_feedback_instrument",
                    "samples": feedback,
                },
                "operational_resource_gauges": {
                    "source_kind": "operational_resource_gauge",
                    "samples": [
                        feedback_by_id[channel_id]
                        for channel_id in (
                            self._observable_topology.operational_resource_gauge_channels
                        )
                    ],
                },
                "derived_health": {
                    "source_kind": "derived_health",
                    "health_state": next_health.health_state,
                },
                "active_operational_alarms": {
                    "source_kind": "alarm_receipt",
                    "alarms": [alarm.to_mapping() for alarm in next_health.alarms],
                },
                "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                "snapshot_schema_sha256": self._snapshot_schema_sha256,
                "observable_topology_sha256": self._observable_topology.sha256,
                "completed_plant_receipt_digest": plant_receipt_digest,
                "completed_step_receipt_digest": step_receipt.step_receipt_sha256,
            }
        )
        issuer_id = _domain_hash(
            "aeolus-habitat-v2-hmc-issuer-v1",
            _decode_sha256(self._control_run_id, label="control run identity"),
            _decode_sha256(self._authority_epoch, label="authority epoch"),
            _decode_sha256(
                self._contract.hmc_contract_sha256,
                label="HMC contract identity",
            ),
        )
        cycle_id = _domain_hash(
            "aeolus-habitat-v2-hmc-cycle-v1",
            _decode_sha256(self._control_run_id, label="control run identity"),
            _decode_sha256(self._authority_epoch, label="authority epoch"),
            next_sequence.to_bytes(8, "big"),
            _decode_sha256(next_snapshot.snapshot_sha256, label="snapshot identity"),
        )
        snapshot_event_ordinal = step_event_ordinal + 1
        next_verification = _issue_snapshot_verification_receipt(
            {
                "receipt_schema_sha256": (
                    self._contract.snapshot_verification_receipt_schema_sha256
                ),
                "snapshot_verification_contract_sha256": (
                    self._contract.snapshot_verification_contract_sha256
                ),
                "hmc_contract_sha256": self._contract.hmc_contract_sha256,
                "snapshot_schema_sha256": self._snapshot_schema_sha256,
                "observable_topology_sha256": self._observable_topology.sha256,
                "control_run_id": self._control_run_id,
                "authority_epoch": self._authority_epoch,
                "issuer_id": issuer_id,
                "cycle_id": cycle_id,
                "sequence": next_sequence,
                "completed_step": next_measurement.completed_step,
                "completed_time_s": next_measurement.completed_time_s,
                "snapshot_sha256": next_snapshot.snapshot_sha256,
                "completed_plant_receipt_digest": plant_receipt_digest,
                "completed_step_receipt_digest": step_receipt.step_receipt_sha256,
                "previous_verification_receipt_digest": (
                    self._cached_verification_receipt.snapshot_verification_receipt_sha256
                ),
                "event_ordinal": snapshot_event_ordinal,
                "previous_control_chain_sha256": step_event.control_chain_sha256,
            }
        )
        next_snapshot_event = _issue_control_event(
            event_ordinal=snapshot_event_ordinal,
            event_kind="SNAPSHOT_VERIFICATION",
            receipt=next_verification,
        )
        return _StagedCycle(
            state=candidate.state,
            measurement=next_measurement,
            health_tracker=next_health.tracker,
            step_receipt=step_receipt,
            plant_receipt_digest=plant_receipt_digest,
            sequence=next_sequence,
            snapshot=next_snapshot,
            verification_receipt=next_verification,
            step_event=step_event,
            snapshot_event=next_snapshot_event,
        )

    def step(self) -> StepReceipt | TerminalFailureReceipt:
        return self._step_with_executor(advance_one_step_with_command)

    def _step_with_executor(
        self, executor: Any
    ) -> StepReceipt | TerminalFailureReceipt:
        """Execute one authorised cycle with the supplied trusted physics boundary."""
        if self._phase is not LifecyclePhase.ARBITRATED:
            raise RuntimeError(
                f"step is not valid during lifecycle phase {self._phase.value}"
            )
        capability = self._step_capability
        if (
            type(capability) is not _StepCapability
            or capability.owner_identity != id(self)
            or capability.control_run_id != self._control_run_id
            or capability.authority_epoch != self._authority_epoch
            or capability.scenario_sha256 != self._scenario.scenario_sha256
            or capability.application_step != self._state.step
            or self._cached_final_command is None
            or capability.final_command_sha256 != self._cached_final_command.sha256
            or self._cached_proposal_receipt is None
            or self._cached_arbitration_receipt is None
            or self._cached_snapshot is None
            or self._cached_verification_receipt is None
            or self._sensor_memory is None
            or self._last_operational_measurement is None
        ):
            raise RuntimeError("step authority capability is invalid")

        self._step_capability = None
        pre_step_state = self._state
        application_step = pre_step_state.step
        final_command = self._cached_final_command
        try:
            candidate = executor(
                self._scenario,
                pre_step_state,
                final_command.to_mapping(),
            )
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="PHYSICS_EXECUTION_FAILED",
                application_step=application_step,
                final_command_sha256=final_command.sha256,
            )
        try:
            plant_receipt_bytes = json.dumps(
                candidate.receipt,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            plant_receipt_digest = hashlib.sha256(plant_receipt_bytes).hexdigest()
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="PLANT_RECEIPT_INVALID",
                application_step=application_step,
                final_command_sha256=final_command.sha256,
            )
        try:
            returned_digest = candidate.receipt.get("external_command_digest")
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="PLANT_RECEIPT_INVALID",
                application_step=application_step,
                candidate_plant_receipt_digest=plant_receipt_digest,
                final_command_sha256=final_command.sha256,
            )
        if returned_digest != final_command.sha256:
            return self._enter_terminal(
                reason_code="COMMAND_DIGEST_MISMATCH",
                application_step=application_step,
                candidate_plant_receipt_digest=plant_receipt_digest,
                final_command_sha256=final_command.sha256,
            )
        try:
            validate_external_step_result(
                self._scenario,
                pre_step_state,
                final_command.to_mapping(),
                candidate,
            )
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="PLANT_RECEIPT_INVALID",
                application_step=application_step,
                candidate_plant_receipt_digest=plant_receipt_digest,
                final_command_sha256=final_command.sha256,
            )
        try:
            next_measurement = instrument_v5_operational_measurement(
                self._scenario,
                candidate.state,
                self._sensor_memory,
            )
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="OPERATIONAL_MEASUREMENT_INVALID",
                application_step=application_step,
                candidate_plant_receipt_digest=plant_receipt_digest,
                final_command_sha256=final_command.sha256,
            )
        try:
            next_health = reduce_health(
                measurement=next_measurement,
                scenario=self._scenario,
                contract=self._contract,
                previous_tracker=self._health_tracker,
                previous_measurement=self._last_operational_measurement,
                last_final_command=final_command,
            )
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="HEALTH_REDUCTION_FAILED",
                application_step=application_step,
                candidate_plant_receipt_digest=plant_receipt_digest,
                final_command_sha256=final_command.sha256,
            )
        try:
            staged = self._stage_completed_cycle(
                candidate=candidate,
                final_command=final_command,
                returned_digest=returned_digest,
                plant_receipt_digest=plant_receipt_digest,
                application_step=application_step,
                next_measurement=next_measurement,
                next_health=next_health,
            )
        except Exception:  # noqa: BLE001 - fail closed at trusted boundary
            return self._enter_terminal(
                reason_code="SNAPSHOT_ISSUANCE_FAILED",
                application_step=application_step,
                candidate_plant_receipt_digest=plant_receipt_digest,
                final_command_sha256=final_command.sha256,
            )

        self._state = staged.state
        self._sensor_memory = staged.measurement.sensor_memory
        self._health_tracker = staged.health_tracker
        self._last_operational_measurement = staged.measurement
        self._last_step_receipt = staged.step_receipt
        self._last_plant_receipt_digest = staged.plant_receipt_digest
        self._sequence = staged.sequence
        self._cached_snapshot = staged.snapshot
        self._cached_verification_receipt = staged.verification_receipt
        self._control_events.extend((staged.step_event, staged.snapshot_event))
        self._current_control_chain_sha256 = staged.snapshot_event.control_chain_sha256
        self._phase = LifecyclePhase.STEPPED
        return staged.step_receipt

    def export_control_trace(self, hmc_implementation_git_sha: str) -> ControlTrace:
        """Export the immutable canonical whole-control-trace artifact."""
        return _issue_control_trace(
            scenario=self._scenario,
            contract=self._contract,
            reset_nonce=self._reset_nonce,
            state=self._state,
            lifecycle_phase=self._phase.value,
            events=tuple(self._control_events),
            hmc_implementation_git_sha=hmc_implementation_git_sha,
        )

    finalize_control_trace = export_control_trace
