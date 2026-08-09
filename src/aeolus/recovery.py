"""Deterministic reserve-only recovery authority for AEOLUS."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from aeolus.trace import RECOVERY_AUTHORITY_REASONS


class AuthorityState(Enum):
    """The state of the reserve authority state machine."""

    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    PROTECT = "PROTECT"
    HANDBACK = "HANDBACK"


class ReserveCommandOwner(Enum):
    """The component that owns the reserve command channel."""

    RESERVE_OFF = "reserve_off"
    DETERMINISTIC_RECOVERY_SUPERVISOR = "deterministic_recovery_supervisor"


@dataclass(frozen=True)
class RecoverySettings:
    """Frozen thresholds and timing constants for deterministic recovery."""

    entry_residual_ratio: float = 0.10
    entry_isolation_margin: float = 0.05
    minimum_requested_fraction: float = 0.05
    entry_persistence_ticks: int = 2
    degraded_clear_persistence_ticks: int = 3
    exit_residual_ratio: float = 0.06
    handback_abort_residual_ratio: float = 0.08
    handback_abort_persistence_ticks: int = 2
    minimum_protect_dwell_ticks: int = 10
    recovery_clear_persistence_ticks: int = 10
    minimum_reserve_delivery_ratio: float = 0.90
    reserve_delivery_failure_persistence_ticks: int = 2
    reserve_command_delta: float = 0.10
    maximum_reserve_command: float = 1.00
    handback_settle_ticks: int = 5
    maximum_handback_ticks: int = 36

    def __post_init__(self) -> None:
        fractions = (
            self.entry_residual_ratio,
            self.entry_isolation_margin,
            self.minimum_requested_fraction,
            self.exit_residual_ratio,
            self.handback_abort_residual_ratio,
            self.minimum_reserve_delivery_ratio,
            self.reserve_command_delta,
            self.maximum_reserve_command,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in fractions
        ):
            raise ValueError("recovery settings must be finite numbers")
        if any(not 0.0 <= float(value) <= 1.0 for value in fractions):
            raise ValueError("recovery settings must be in 0.0..1.0")
        if self.minimum_requested_fraction <= 0.0:
            raise ValueError("recovery minimum requested fraction must be positive")
        if self.reserve_command_delta <= 0.0:
            raise ValueError("recovery reserve command delta must be positive")
        if self.maximum_reserve_command <= 0.0:
            raise ValueError("recovery maximum reserve command must be positive")

        integer_fields = (
            "entry_persistence_ticks",
            "degraded_clear_persistence_ticks",
            "minimum_protect_dwell_ticks",
            "recovery_clear_persistence_ticks",
            "reserve_delivery_failure_persistence_ticks",
            "handback_settle_ticks",
            "maximum_handback_ticks",
            "handback_abort_persistence_ticks",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"recovery {field_name} must be a positive integer"
                )

        if self.entry_persistence_ticks < 2:
            raise ValueError("recovery entry persistence must be at least two ticks")
        if self.degraded_clear_persistence_ticks < 2:
            raise ValueError(
                "recovery degraded clear persistence must be at least two ticks"
            )
        if self.reserve_delivery_failure_persistence_ticks < 2:
            raise ValueError(
                "recovery reserve delivery failure persistence must be at least two ticks"
            )
        if self.maximum_handback_ticks != 36:
            raise ValueError("recovery maximum handback ticks is frozen at 36")
        if not (
            self.exit_residual_ratio
            < self.handback_abort_residual_ratio
            < self.entry_residual_ratio
        ):
            raise ValueError("recovery hysteresis thresholds are not ordered")

    @property
    def same_target_entry_persistence_ticks(self) -> int:
        """Return the persistence required before a target can arm."""
        return self.entry_persistence_ticks

    @property
    def min_requested_fraction(self) -> float:
        """Return the primary telemetry request floor."""
        return self.minimum_requested_fraction

    @property
    def degraded_clear_persistence(self) -> int:
        """Return the clear persistence used in ``DEGRADED``."""
        return self.degraded_clear_persistence_ticks

    @property
    def minimum_protect_dwell(self) -> int:
        """Return the minimum number of fresh ``PROTECT`` ticks."""
        return self.minimum_protect_dwell_ticks

    @property
    def recovery_clear_persistence(self) -> int:
        """Return the clear persistence required for handback."""
        return self.recovery_clear_persistence_ticks

    @property
    def reserve_failure_persistence(self) -> int:
        """Return the reserve delivery failure persistence."""
        return self.reserve_delivery_failure_persistence_ticks

    @property
    def max_reserve_command(self) -> float:
        """Return the normalized reserve command ceiling."""
        return self.maximum_reserve_command

    @property
    def handback_zero_settle_ticks(self) -> int:
        """Return the physical zero acknowledgement persistence."""
        return self.handback_settle_ticks

    @property
    def maximum_handback_duration_ticks(self) -> int:
        """Return the source-enforced maximum handback duration."""
        return self.maximum_handback_ticks


@dataclass(frozen=True)
class AdvisoryAcceptanceSettings:
    """Frozen deterministic gates for one learned advisory artifact."""

    artifact_sha256: str
    minimum_probability: float
    minimum_margin: float
    minimum_residual_ratio: float = 0.04

    def __post_init__(self) -> None:
        if not _is_digest(self.artifact_sha256):
            raise ValueError("recovery advisory artifact hash is malformed")
        for name, value in (
            ("probability", self.minimum_probability),
            ("margin", self.minimum_margin),
            ("residual", self.minimum_residual_ratio),
        ):
            if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"recovery advisory {name} threshold is invalid")
        if not 0.0 < self.minimum_residual_ratio < RecoverySettings().entry_residual_ratio:
            raise ValueError("recovery advisory residual floor must be below normal entry")


@dataclass(frozen=True)
class RecoveryAdvisory:
    """One causal learned warning with no actuator-command authority."""

    run_id: str
    authority_epoch: int
    completed_tick: int
    sequence: int
    target_zone_id: str
    probability: float
    margin: float
    selector_sha256: str
    topology_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class RecoveryObservation:
    """Completed-tick telemetry accepted at the recovery authority boundary."""

    run_id: str
    authority_epoch: int
    completed_tick: int
    sequence: int
    model_input_v1: tuple[float, ...]
    selector_sha256: str
    topology_sha256: str
    zone_ids: tuple[str, ...]
    primary_outbound_ids: Mapping[str, str]
    reserve_outbound_ids: Mapping[str, str]
    reserve_return_ids: Mapping[str, str]
    co2_concentration: Mapping[str, float]
    primary_requested_airflow: Mapping[str, float]
    primary_delivered_airflow: Mapping[str, float]
    reserve_actual_position: Mapping[str, float]
    reserve_delivered_airflow: Mapping[str, float]
    applied_command_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "primary_outbound_ids",
            "reserve_outbound_ids",
            "reserve_return_ids",
            "co2_concentration",
            "primary_requested_airflow",
            "primary_delivered_airflow",
            "reserve_actual_position",
            "reserve_delivered_airflow",
        ):
            value = getattr(self, field_name)
            if isinstance(value, Mapping):
                object.__setattr__(self, field_name, MappingProxyType(dict(value)))


@dataclass(frozen=True)
class RecoveryDecision:
    """Immutable reserve command decision for one causal application tick."""

    run_id: str
    authority_epoch: int
    decision_tick: int
    observation_tick: int
    sequence: int
    state: AuthorityState
    reserve_command_owner: ReserveCommandOwner
    target_zone_id: str | None
    reserve_commands: Mapping[str, float]
    reason: str
    command_digest: str
    dwell_ticks: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.reserve_commands, Mapping):
            object.__setattr__(
                self, "reserve_commands", MappingProxyType(dict(self.reserve_commands))
            )


@dataclass(frozen=True)
class AuthorityEvent:
    """Immutable record of an authority state transition."""

    run_id: str
    authority_epoch: int
    decision_tick: int
    observation_tick: int
    sequence: int
    from_state: AuthorityState
    to_state: AuthorityState
    reason: str
    target_zone_id: str | None

    @property
    def state(self) -> AuthorityState:
        """Return the state reached by this event."""
        return self.to_state


@dataclass(frozen=True)
class _ObservationAnalysis:
    residuals: Mapping[str, float]
    candidates: tuple[str, ...]
    target: str | None
    ambiguous: bool
    advisory_target: str | None = None


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _decision_digest(
    *,
    run_id: str,
    authority_epoch: int,
    decision_tick: int,
    observation_tick: int,
    sequence: int,
    state: AuthorityState,
    owner: ReserveCommandOwner,
    target_zone_id: str | None,
    reserve_commands: Mapping[str, float],
    reason: str,
    dwell_ticks: int,
) -> str:
    return _sha256_json(
        {
            "authority_epoch": authority_epoch,
            "decision_tick": decision_tick,
            "dwell_ticks": dwell_ticks,
            "observation_tick": observation_tick,
            "reason": reason,
            "reserve_command_owner": owner.value,
            "reserve_commands": {
                zone_id: float(reserve_commands[zone_id])
                for zone_id in sorted(reserve_commands)
            },
            "run_id": run_id,
            "sequence": sequence,
            "state": state.value,
            "target_zone_id": target_zone_id,
        }
    )


def _mapping_keys(value: object) -> set[object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return set(value)
    except TypeError:
        return None


def _expected_zone_ids(config: Any) -> tuple[str, ...]:
    try:
        zones = tuple(config.non_processing_zones())
        zone_ids = tuple(zone.id for zone in zones)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("recovery configuration has no valid zone topology") from exc
    if not zone_ids or any(not isinstance(zone_id, str) or not zone_id for zone_id in zone_ids):
        raise ValueError("recovery configuration has invalid zone ids")
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError("recovery configuration has duplicate zone ids")
    return zone_ids


def _field_tuple(field: object) -> tuple[str, str, str]:
    try:
        group = field.group
        entity_id = field.entity_id
        name = field.field
    except AttributeError as exc:
        raise ValueError("recovery model-input contract has malformed fields") from exc
    if not all(isinstance(value, str) and value for value in (group, entity_id, name)):
        raise ValueError("recovery model-input contract has malformed fields")
    return group, entity_id, name


def _validate_contract(config: Any, contract: Any, zone_ids: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
    if contract is None:
        raise ValueError("recovery requires a model-input contract")
    try:
        fields = tuple(contract.fields)
    except (AttributeError, TypeError) as exc:
        raise ValueError("recovery model-input contract is malformed") from exc
    specs = tuple(_field_tuple(field) for field in fields)
    expected: list[tuple[str, str, str]] = [
        ("zones", zone_id, "sensor_co2_concentration") for zone_id in zone_ids
    ]
    for zone_id in zone_ids:
        expected.extend(
            ("actuators", zone_id, name)
            for name in ("setpoint", "actual_position", "tracking_residual", "power")
        )
    for zone_id in zone_ids:
        try:
            outbound_id = config.path_to_processing(zone_id).id
        except (AttributeError, LookupError) as exc:
            raise ValueError("recovery configuration has an invalid primary topology") from exc
        expected.extend(
            ("connections", outbound_id, name)
            for name in ("requested_airflow", "delivered_airflow", "airflow_residual")
        )
    if len(specs) != 24 or tuple(expected) != specs:
        raise ValueError("recovery model-input contract does not match primary topology")

    try:
        processing_id = config.processing_zone().id
        loops = []
        for zone_id in zone_ids:
            outbound = config.path_to_processing(zone_id)
            inbound = config.path_from_processing(zone_id)
            loops.append(
                {
                    "outbound": {
                        "from_zone": outbound.from_zone,
                        "id": outbound.id,
                        "to_zone": outbound.to_zone,
                    },
                    "return": {
                        "from_zone": inbound.from_zone,
                        "id": inbound.id,
                        "to_zone": inbound.to_zone,
                    },
                    "zone_id": zone_id,
                }
            )
    except (AttributeError, LookupError) as exc:
        raise ValueError("recovery configuration has an invalid primary topology") from exc
    topology = {
        "non_processing_zone_ids": list(zone_ids),
        "primary_loops": loops,
        "processing_zone_id": processing_id,
        "schema_version": "aeolus_topology_v1",
    }
    topology_json = _canonical_json(topology)
    topology_hash = hashlib.sha256(topology_json.encode("utf-8")).hexdigest()

    selector = {
        "dtype": "float32",
        "fields": [
            {"entity_id": entity_id, "field": name, "group": group}
            for group, entity_id, name in specs
        ],
        "shape": [24],
        "topology_hash": topology_hash,
        "schema_version": "model_input_v1",
    }
    selector_json = _canonical_json(selector)
    selector_hash = hashlib.sha256(selector_json.encode("utf-8")).hexdigest()
    if (
        getattr(contract, "topology_json", None) != topology_json
        or getattr(contract, "topology_hash", None) != topology_hash
        or getattr(contract, "selector_json", None) != selector_json
        or getattr(contract, "selector_hash", None) != selector_hash
    ):
        raise ValueError("recovery model-input contract hashes do not match topology")
    return specs


def _validate_recovery_config(config: Any, zone_ids: tuple[str, ...]) -> tuple[dict[str, str], dict[str, str], dict[str, str], float, dict[str, float]]:
    try:
        primary_connections = tuple(config.connections)
        reserve_connections = tuple(config.reserve_connections)
        reserve_capacity = config.air_system.reserve_airflow_capacity
    except AttributeError as exc:
        raise ValueError("recovery configuration is missing reserve topology") from exc
    if not _is_finite_number(reserve_capacity) or float(reserve_capacity) <= 0.0:
        raise ValueError("recovery reserve capacity must be positive and finite")
    primary_ids = {connection.id for connection in primary_connections}
    reserve_ids = {connection.id for connection in reserve_connections}
    if len(primary_ids) != len(primary_connections) or len(reserve_ids) != len(reserve_connections):
        raise ValueError("recovery connection IDs must be unique")
    if primary_ids & reserve_ids:
        raise ValueError("recovery primary and reserve ids must be disjoint")
    if len(reserve_connections) != 2 * len(zone_ids):
        raise ValueError("recovery reserve topology must contain one directed pair per zone")
    try:
        processing_id = config.processing_zone().id
    except (AttributeError, LookupError) as exc:
        raise ValueError("recovery configuration has no processing zone") from exc
    if not isinstance(processing_id, str) or not processing_id:
        raise ValueError("recovery configuration has an invalid processing zone")

    primary_outbound: dict[str, str] = {}
    reserve_outbound: dict[str, str] = {}
    reserve_return: dict[str, str] = {}
    primary_max: dict[str, float] = {}
    expected_reserve_ids: set[str] = set()
    for zone_id in zone_ids:
        try:
            primary = config.path_to_processing(zone_id)
            reserve_out = config.reserve_path_to_processing(zone_id)
            reserve_in = config.reserve_path_from_processing(zone_id)
        except (AttributeError, LookupError) as exc:
            raise ValueError("recovery configuration has incomplete reserve topology") from exc
        for connection in (primary, reserve_out, reserve_in):
            if (
                not isinstance(connection.id, str)
                or not connection.id
                or not _is_finite_number(connection.max_airflow)
                or float(connection.max_airflow) <= 0.0
            ):
                raise ValueError("recovery topology contains an invalid connection")
        if not math.isclose(
            float(reserve_out.max_airflow), float(reserve_capacity), abs_tol=1e-12
        ) or not math.isclose(
            float(reserve_in.max_airflow), float(reserve_capacity), abs_tol=1e-12
        ):
            raise ValueError("recovery reserve path max airflow must match capacity")
        primary_outbound[zone_id] = primary.id
        reserve_outbound[zone_id] = reserve_out.id
        reserve_return[zone_id] = reserve_in.id
        primary_max[zone_id] = float(primary.max_airflow)
        expected_reserve_ids.update((reserve_out.id, reserve_in.id))
    if expected_reserve_ids != reserve_ids:
        raise ValueError("recovery reserve topology contains extra or missing paths")

    try:
        minimum_command = config.control.minimum_command
    except AttributeError as exc:
        raise ValueError("recovery configuration has no primary control settings") from exc
    if not _is_finite_number(minimum_command) or float(minimum_command) < RecoverySettings().minimum_requested_fraction:
        raise ValueError("minimum primary command is below the recovery evidence floor")
    return primary_outbound, reserve_outbound, reserve_return, float(reserve_capacity), primary_max


def validate_recovery_decision(
    decision: RecoveryDecision,
    config: Any = None,
    *,
    expected_run_id: str | None = None,
    expected_authority_epoch: int | None = None,
    expected_decision_tick: int | None = None,
    expected_state: AuthorityState | None = None,
    expected_owner: ReserveCommandOwner | None = None,
    expected_reserve_command_owner: ReserveCommandOwner | None = None,
    expected_observation_tick: int | None = None,
    expected_sequence: int | None = None,
) -> dict[str, float]:
    """Validate a generated decision before reserve commands reach the plant."""
    if not isinstance(decision, RecoveryDecision):
        raise ValueError("recovery decision is malformed")
    if config is None:
        raise ValueError("recovery decision validation requires configuration")
    try:
        zone_ids = _expected_zone_ids(config)
    except ValueError:
        raise
    if expected_run_id is not None and decision.run_id != expected_run_id:
        raise ValueError("recovery decision run identity is stale")
    if expected_authority_epoch is not None and decision.authority_epoch != expected_authority_epoch:
        raise ValueError("recovery decision authority epoch is stale")
    if expected_decision_tick is not None and decision.decision_tick != expected_decision_tick:
        raise ValueError("recovery decision tick is stale")
    if expected_observation_tick is not None and decision.observation_tick != expected_observation_tick:
        raise ValueError("recovery decision observation tick is stale")
    if expected_sequence is not None and decision.sequence != expected_sequence:
        raise ValueError("recovery decision sequence is stale")
    owner_expectation = expected_owner
    if expected_reserve_command_owner is not None:
        if owner_expectation is not None and owner_expectation is not expected_reserve_command_owner:
            raise ValueError("recovery decision owner expectation conflicts")
        owner_expectation = expected_reserve_command_owner
    if expected_state is not None and decision.state is not expected_state:
        raise ValueError("recovery decision state does not match the application gate")
    if owner_expectation is not None and decision.reserve_command_owner is not owner_expectation:
        raise ValueError("recovery decision owner does not match the application gate")

    if (
        not isinstance(decision.run_id, str)
        or not decision.run_id
        or not _is_non_negative_int(decision.authority_epoch)
        or not _is_non_negative_int(decision.decision_tick)
        or decision.decision_tick < 1
        or not _is_non_negative_int(decision.observation_tick)
        or decision.observation_tick != decision.decision_tick - 1
        or not _is_non_negative_int(decision.sequence)
        or not isinstance(decision.state, AuthorityState)
        or not isinstance(decision.reserve_command_owner, ReserveCommandOwner)
        or not isinstance(decision.reason, str)
        or not decision.reason
        or not _is_non_negative_int(decision.dwell_ticks)
    ):
        raise ValueError("recovery decision has malformed identity or timing")
    if decision.reason not in RECOVERY_AUTHORITY_REASONS:
        raise ValueError("recovery decision reason is not in the fixed enum")
    if decision.reserve_command_owner is ReserveCommandOwner.RESERVE_OFF:
        if decision.state not in (AuthorityState.NOMINAL, AuthorityState.DEGRADED):
            raise ValueError("recovery decision owner is invalid for its state")
    elif decision.reserve_command_owner is ReserveCommandOwner.DETERMINISTIC_RECOVERY_SUPERVISOR:
        if decision.state not in (AuthorityState.PROTECT, AuthorityState.HANDBACK):
            raise ValueError("recovery decision owner is invalid for its state")
    else:
        raise ValueError("recovery decision has an unknown owner")

    commands = decision.reserve_commands
    if not isinstance(commands, Mapping) or _mapping_keys(commands) != set(zone_ids):
        raise ValueError("recovery decision reserve command keys do not match topology")
    normalized: dict[str, float] = {}
    for zone_id in zone_ids:
        value = commands[zone_id]
        if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("recovery decision reserve command is outside 0.0..1.0")
        normalized[zone_id] = float(value)
    if decision.reserve_command_owner is ReserveCommandOwner.RESERVE_OFF and any(
        value != 0.0 for value in normalized.values()
    ):
        raise ValueError("reserve_off decisions must force exact zero commands")

    if decision.state is AuthorityState.NOMINAL:
        if decision.target_zone_id is not None:
            raise ValueError("nominal recovery decisions cannot latch a target")
    elif decision.target_zone_id is not None:
        if decision.target_zone_id not in zone_ids:
            raise ValueError("recovery decision target is outside topology")
    elif decision.state in (AuthorityState.PROTECT, AuthorityState.HANDBACK):
        raise ValueError("protective recovery decisions require a target")

    if not _is_digest(decision.command_digest):
        raise ValueError("recovery decision command digest is malformed")
    try:
        expected_digest = _decision_digest(
            run_id=decision.run_id,
            authority_epoch=decision.authority_epoch,
            decision_tick=decision.decision_tick,
            observation_tick=decision.observation_tick,
            sequence=decision.sequence,
            state=decision.state,
            owner=decision.reserve_command_owner,
            target_zone_id=decision.target_zone_id,
            reserve_commands=normalized,
            reason=decision.reason,
            dwell_ticks=decision.dwell_ticks,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("recovery decision command digest cannot be computed") from exc
    if decision.command_digest != expected_digest:
        raise ValueError("recovery decision command digest does not match payload")
    return normalized


class DeterministicRecoverySupervisor:
    """Run the frozen reserve-only authority policy over completed telemetry."""

    def __init__(
        self,
        config: Any,
        run_id: str,
        contract: Any,
        settings: RecoverySettings | None = None,
        advisory_settings: AdvisoryAcceptanceSettings | None = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("recovery run_id must be a non-empty string")
        self.config = config
        self.settings = settings if settings is not None else RecoverySettings()
        if not isinstance(self.settings, RecoverySettings):
            raise ValueError("recovery settings are malformed")
        if advisory_settings is not None and not isinstance(
            advisory_settings, AdvisoryAcceptanceSettings
        ):
            raise ValueError("recovery advisory settings are malformed")
        self.advisory_settings = advisory_settings
        self._zone_ids = _expected_zone_ids(config)
        (
            self._primary_outbound,
            self._reserve_outbound,
            self._reserve_return,
            self._reserve_capacity,
            self._primary_max_airflow,
        ) = _validate_recovery_config(config, self._zone_ids)
        self._primary_zone_by_outbound = {
            connection_id: zone_id
            for zone_id, connection_id in self._primary_outbound.items()
        }
        self._field_specs = _validate_contract(config, contract, self._zone_ids)
        self.contract = contract
        self._selector_hash = contract.selector_hash
        self._topology_hash = contract.topology_hash
        self._run_id = run_id
        self._authority_epoch = 0
        self.reset(run_id=run_id, authority_epoch=0)

    @property
    def run_id(self) -> str:
        """Return the run identity bound to this supervisor."""
        return self._run_id

    @property
    def authority_epoch(self) -> int:
        """Return the current reserve authority epoch."""
        return self._authority_epoch

    @property
    def state(self) -> AuthorityState:
        """Return the current authority state."""
        return self._state

    @property
    def zone_ids(self) -> tuple[str, ...]:
        """Return non-processing zone IDs in topology order."""
        return self._zone_ids

    @property
    def reserve_capacity(self) -> float:
        """Return the validated reserve manifold capacity."""
        return self._reserve_capacity

    @property
    def last_decision(self) -> RecoveryDecision | None:
        """Return the last immutable decision, if the run has started."""
        return self._last_decision

    @property
    def event_history(self) -> list[AuthorityEvent]:
        """Return a copy of the ordered authority transition history."""
        return list(self._event_history)

    @property
    def reserve_failed(self) -> bool:
        """Return whether reserve delivery has latched failed this epoch."""
        return self._reserve_failed

    def primary_outbound_id(self, zone_id: str) -> str:
        """Return the primary outbound connection ID for one zone."""
        try:
            return self._primary_outbound[zone_id]
        except KeyError as exc:
            raise ValueError(f"unknown recovery zone {zone_id!r}") from exc

    def primary_return_id(self, zone_id: str) -> str:
        """Return the primary return connection ID for one zone."""
        try:
            return self.config.path_from_processing(zone_id).id
        except (AttributeError, LookupError) as exc:
            raise ValueError(f"unknown recovery zone {zone_id!r}") from exc

    def reserve_outbound_id(self, zone_id: str) -> str:
        """Return the reserve outbound connection ID for one zone."""
        try:
            return self._reserve_outbound[zone_id]
        except KeyError as exc:
            raise ValueError(f"unknown recovery zone {zone_id!r}") from exc

    def reserve_return_id(self, zone_id: str) -> str:
        """Return the reserve return connection ID for one zone."""
        try:
            return self._reserve_return[zone_id]
        except KeyError as exc:
            raise ValueError(f"unknown recovery zone {zone_id!r}") from exc

    def zone_for_primary_outbound(self, connection_id: str) -> str:
        """Return the zone represented by one primary outbound field."""
        try:
            return self._primary_zone_by_outbound[connection_id]
        except KeyError as exc:
            raise ValueError(f"unknown primary outbound connection {connection_id!r}") from exc

    def reset(self, run_id: str | None = None, authority_epoch: int | None = None) -> None:
        """Forget run state and clear the failure latch before a new run."""
        next_run_id = self._run_id if run_id is None else run_id
        if not isinstance(next_run_id, str) or not next_run_id:
            raise ValueError("recovery run_id must be a non-empty string")
        if authority_epoch is None:
            next_epoch = (
                0 if next_run_id != self._run_id else self._authority_epoch + 1
            )
        else:
            next_epoch = authority_epoch
        if not _is_non_negative_int(next_epoch):
            raise ValueError("recovery authority epoch must be a non-negative integer")
        self._run_id = next_run_id
        self._authority_epoch = next_epoch
        self._state = AuthorityState.NOMINAL
        self._target_zone_id: str | None = None
        self._entry_target: str | None = None
        self._entry_streak = 0
        self._degraded_clear_count = 0
        self._protect_age = 0
        self._protect_clear_count = 0
        self._protect_command = 0.0
        self._protect_entered_by_advisory = False
        self._reserve_delivery_failure_count = 0
        self._reserve_failed = False
        self._failure_handback = False
        self._handback_age = 0
        self._handback_zero_ack_count = 0
        self._handback_abort_streak = 0
        self._epoch_rearm_pending = False
        self._last_observation_sequence = -1
        self._last_decision: RecoveryDecision | None = None
        self._event_history: list[AuthorityEvent] = []

    def cold_start(self, decision_tick: int) -> RecoveryDecision:
        """Emit the only pre-observation decision: nominal reserve-off."""
        if self._last_decision is not None:
            raise ValueError("recovery cold_start may only be called once per reset")
        if not isinstance(decision_tick, int) or isinstance(decision_tick, bool) or decision_tick < 1:
            raise ValueError("recovery cold_start decision tick must be positive")
        self._state = AuthorityState.NOMINAL
        self._target_zone_id = None
        self._entry_target = None
        self._entry_streak = 0
        self._degraded_clear_count = 0
        self._last_observation_sequence = -1
        return self._make_decision(
            decision_tick=decision_tick,
            observation_tick=0,
            sequence=1,
            target_zone_id=None,
            commands=self._zero_commands(),
            reason="cold_start",
        )

    def decide(
        self,
        observation: RecoveryObservation,
        advisory: RecoveryAdvisory | None = None,
    ) -> RecoveryDecision:
        """Consume one fresh completed-tick observation and decide causally."""
        self._validate_observation(observation)
        self._last_observation_sequence = observation.sequence
        analysis = self._analyze(observation)
        advisory_target = self._accepted_advisory_target(
            observation, analysis, advisory
        )
        if advisory_target is not None:
            analysis = _ObservationAnalysis(
                residuals=analysis.residuals,
                candidates=analysis.candidates,
                target=analysis.target,
                ambiguous=analysis.ambiguous,
                advisory_target=advisory_target,
            )
        if self._state is AuthorityState.NOMINAL:
            return self._decide_nominal(observation, analysis)
        if self._state is AuthorityState.DEGRADED:
            return self._decide_degraded(observation, analysis)
        if self._state is AuthorityState.PROTECT:
            return self._decide_protect(observation, analysis)
        if self._state is AuthorityState.HANDBACK:
            return self._decide_handback(observation, analysis)
        raise ValueError("recovery supervisor has an invalid authority state")

    def decide_unavailable(
        self,
        completed_tick: int,
        sequence: int,
        applied_command_digest: str,
    ) -> RecoveryDecision:
        """Fail closed for one explicitly unavailable completed observation."""
        self._validate_unavailable(completed_tick, sequence, applied_command_digest)
        self._last_observation_sequence = sequence
        previous = self._last_decision
        assert previous is not None
        old_state = self._state
        if self._state is AuthorityState.NOMINAL:
            self._state = AuthorityState.DEGRADED
            self._target_zone_id = None
            self._entry_target = None
            self._entry_streak = 0
            self._degraded_clear_count = 0
            commands = self._zero_commands()
            target = None
            reason = "observation_unavailable"
        elif self._state is AuthorityState.DEGRADED:
            self._target_zone_id = None
            self._entry_target = None
            self._entry_streak = 0
            self._degraded_clear_count = 0
            commands = self._zero_commands()
            target = None
            reason = "observation_unavailable"
        elif self._state is AuthorityState.PROTECT:
            self._state = AuthorityState.HANDBACK
            self._failure_handback = False
            self._handback_age = 0
            self._protect_clear_count = 0
            self._handback_zero_ack_count = 0
            self._handback_abort_streak = 0
            commands = self._ramp_down_commands(previous.reserve_commands)
            target = self._target_zone_id
            reason = "observation_unavailable"
        else:
            self._handback_abort_streak = 0
            self._handback_age += 1
            if self._handback_age >= self.settings.maximum_handback_ticks:
                commands = self._timeout_handback(self._target_zone_id)
                target = self._target_zone_id
                reason = "handback_timeout"
            elif self._reserve_failed:
                commands = self._ramp_down_commands(previous.reserve_commands)
                target = self._target_zone_id
                reason = "reserve_failure_shutdown"
            else:
                commands = self._ramp_down_commands(previous.reserve_commands)
                target = self._target_zone_id
                reason = "observation_unavailable"
        self._record_transition(
            old_state,
            self._state,
            decision_tick=previous.decision_tick + 1,
            observation_tick=completed_tick,
            sequence=sequence + 1,
            reason=reason,
            target_zone_id=target,
        )
        return self._make_decision(
            decision_tick=previous.decision_tick + 1,
            observation_tick=completed_tick,
            sequence=sequence + 1,
            target_zone_id=target,
            commands=commands,
            reason=reason,
        )

    def _validate_observation(self, observation: RecoveryObservation) -> None:
        if not isinstance(observation, RecoveryObservation):
            raise ValueError("recovery observation is malformed")
        previous = self._last_decision
        if previous is None:
            raise ValueError("recovery observation arrived before cold_start")
        if observation.run_id != self._run_id:
            raise ValueError("recovery observation run identity is stale")
        if (
            not _is_non_negative_int(observation.authority_epoch)
            or observation.authority_epoch != self._authority_epoch
        ):
            raise ValueError("recovery observation authority epoch is stale")
        if (
            not _is_non_negative_int(observation.completed_tick)
            or observation.completed_tick != previous.decision_tick
        ):
            raise ValueError("recovery observation is not for the last applied tick")
        if (
            not _is_non_negative_int(observation.sequence)
            or observation.sequence != previous.sequence
        ):
            raise ValueError("recovery observation sequence is stale or out of order")
        if observation.sequence <= self._last_observation_sequence:
            raise ValueError("recovery observation sequence is not strictly increasing")
        if observation.selector_sha256 != self._selector_hash:
            raise ValueError("recovery observation selector hash is invalid")
        if observation.topology_sha256 != self._topology_hash:
            raise ValueError("recovery observation topology hash is invalid")
        if not isinstance(observation.zone_ids, tuple) or observation.zone_ids != self._zone_ids:
            raise ValueError("recovery observation zone topology is invalid")
        if observation.applied_command_digest != previous.command_digest:
            raise ValueError("recovery observation application digest is stale")
        for field_name, expected in (
            ("primary_outbound_ids", self._primary_outbound),
            ("reserve_outbound_ids", self._reserve_outbound),
            ("reserve_return_ids", self._reserve_return),
        ):
            mapping = getattr(observation, field_name)
            if not isinstance(mapping, Mapping) or _mapping_keys(mapping) != set(self._zone_ids):
                raise ValueError(f"recovery observation {field_name} keys are invalid")
            if any(mapping[zone_id] != expected[zone_id] for zone_id in self._zone_ids):
                raise ValueError(f"recovery observation {field_name} topology is invalid")

        value_maps = (
            ("co2_concentration", 0.0, None),
            ("primary_requested_airflow", 0.0, None),
            ("primary_delivered_airflow", 0.0, None),
            ("reserve_actual_position", 0.0, 1.0),
            ("reserve_delivered_airflow", 0.0, self._reserve_capacity),
        )
        for field_name, lower, upper in value_maps:
            mapping = getattr(observation, field_name)
            if not isinstance(mapping, Mapping) or _mapping_keys(mapping) != set(self._zone_ids):
                raise ValueError(f"recovery observation {field_name} keys are invalid")
            for zone_id in self._zone_ids:
                value = mapping[zone_id]
                if not _is_finite_number(value):
                    raise ValueError(f"recovery observation {field_name} is non-finite")
                numeric = float(value)
                if numeric < lower or (upper is not None and numeric > upper + 1e-12):
                    raise ValueError(f"recovery observation {field_name} is out of bounds")
        for zone_id in self._zone_ids:
            requested = float(observation.primary_requested_airflow[zone_id])
            delivered = float(observation.primary_delivered_airflow[zone_id])
            if delivered > requested + 1e-9:
                raise ValueError("recovery observation primary delivery exceeds request")
            reserve_requested = (
                float(observation.reserve_actual_position[zone_id])
                * self._reserve_capacity
            )
            reserve_delivered = float(observation.reserve_delivered_airflow[zone_id])
            if reserve_delivered > reserve_requested + 1e-9:
                raise ValueError("recovery observation reserve delivery exceeds request")

        if not isinstance(observation.model_input_v1, tuple) or len(observation.model_input_v1) != len(self._field_specs):
            raise ValueError("recovery observation model input has an invalid shape")
        if any(not _is_finite_number(value) for value in observation.model_input_v1):
            raise ValueError("recovery observation model input is non-finite")
        for index, (group, entity_id, field_name) in enumerate(self._field_specs):
            expected_value: float | None = None
            if group == "zones":
                expected_value = float(observation.co2_concentration[entity_id])
            elif group == "connections":
                zone_id = self.zone_for_primary_outbound(entity_id)
                if field_name == "requested_airflow":
                    expected_value = float(observation.primary_requested_airflow[zone_id])
                elif field_name == "delivered_airflow":
                    expected_value = float(observation.primary_delivered_airflow[zone_id])
                elif field_name == "airflow_residual":
                    expected_value = float(
                        observation.primary_requested_airflow[zone_id]
                        - observation.primary_delivered_airflow[zone_id]
                    )
            if expected_value is not None and not math.isclose(
                float(observation.model_input_v1[index]),
                expected_value,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError("recovery observation model input disagrees with telemetry")

    def _validate_unavailable(
        self, completed_tick: int, sequence: int, applied_command_digest: str
    ) -> None:
        previous = self._last_decision
        if previous is None:
            raise ValueError("unavailable recovery observation arrived before cold_start")
        if not _is_non_negative_int(completed_tick) or completed_tick != previous.decision_tick:
            raise ValueError("unavailable recovery observation tick is stale")
        if not _is_non_negative_int(sequence) or sequence != previous.sequence:
            raise ValueError("unavailable recovery observation sequence is stale")
        if applied_command_digest != previous.command_digest:
            raise ValueError("unavailable recovery application digest is stale")

    def _analyze(self, observation: RecoveryObservation) -> _ObservationAnalysis:
        residuals: dict[str, float] = {}
        candidates: list[str] = []
        for zone_id in self._zone_ids:
            requested = float(observation.primary_requested_airflow[zone_id])
            delivered = float(observation.primary_delivered_airflow[zone_id])
            minimum_request = self.settings.minimum_requested_fraction * self._primary_max_airflow[zone_id]
            if requested <= max(minimum_request, 1e-12):
                ratio = 0.0
            else:
                ratio = max(0.0, requested - delivered) / requested
            residuals[zone_id] = ratio
            if ratio >= self.settings.entry_residual_ratio:
                candidates.append(zone_id)
        target: str | None = None
        ambiguous = False
        if len(candidates) == 1:
            candidate = candidates[0]
            other_max = max(
                (residuals[zone_id] for zone_id in self._zone_ids if zone_id != candidate),
                default=0.0,
            )
            if residuals[candidate] - other_max >= self.settings.entry_isolation_margin:
                target = candidate
            else:
                ambiguous = True
        elif len(candidates) > 1:
            ambiguous = True
        return _ObservationAnalysis(
            residuals=residuals,
            candidates=tuple(candidates),
            target=target,
            ambiguous=ambiguous,
        )

    def _accepted_advisory_target(
        self,
        observation: RecoveryObservation,
        analysis: _ObservationAnalysis,
        advisory: RecoveryAdvisory | None,
    ) -> str | None:
        """Return a physically supported advisory target or refuse it."""
        settings = self.advisory_settings
        if settings is None or not isinstance(advisory, RecoveryAdvisory):
            return None
        if analysis.target is not None or analysis.ambiguous:
            return None
        if (
            advisory.run_id != self._run_id
            or advisory.authority_epoch != self._authority_epoch
            or advisory.completed_tick != observation.completed_tick
            or advisory.sequence != observation.sequence
            or advisory.selector_sha256 != self._selector_hash
            or advisory.topology_sha256 != self._topology_hash
            or advisory.artifact_sha256 != settings.artifact_sha256
        ):
            return None
        target = advisory.target_zone_id
        if target not in ("cabin_a", "cabin_b") or target not in self._zone_ids:
            return None
        if (
            not _is_finite_number(advisory.probability)
            or not 0.0 <= float(advisory.probability) <= 1.0
            or float(advisory.probability) < settings.minimum_probability
            or not _is_finite_number(advisory.margin)
            or not 0.0 <= float(advisory.margin) <= 1.0
            or float(advisory.margin) < settings.minimum_margin
        ):
            return None
        target_residual = float(analysis.residuals[target])
        if target_residual < settings.minimum_residual_ratio:
            return None
        other_max = max(
            (
                float(analysis.residuals[zone_id])
                for zone_id in self._zone_ids
                if zone_id != target
            ),
            default=0.0,
        )
        if target_residual - other_max < self.settings.entry_isolation_margin:
            return None
        return target

    def _decide_nominal(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis
    ) -> RecoveryDecision:
        previous = self._last_decision
        assert previous is not None
        old_state = self._state
        concern_target = analysis.target or analysis.advisory_target
        advisory_concern = analysis.target is None and analysis.advisory_target is not None
        if concern_target is not None:
            self._state = AuthorityState.DEGRADED
            self._entry_target = concern_target
            self._entry_streak = 1
            self._degraded_clear_count = 0
            self._target_zone_id = concern_target
            target = concern_target
            reason = "advisory_unique_concern" if advisory_concern else "unique_concern"
        elif analysis.ambiguous:
            self._state = AuthorityState.DEGRADED
            self._entry_target = None
            self._entry_streak = 0
            self._degraded_clear_count = 0
            self._target_zone_id = None
            target = None
            reason = "ambiguous_concern"
        else:
            target = None
            reason = "no_concern"
        self._record_transition(
            old_state,
            self._state,
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            reason=reason,
            target_zone_id=target,
        )
        return self._make_decision(
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            target_zone_id=target,
            commands=self._zero_commands(),
            reason=reason,
        )

    def _decide_degraded(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis
    ) -> RecoveryDecision:
        previous = self._last_decision
        assert previous is not None
        old_state = self._state
        commands = self._zero_commands()
        concern_target = analysis.target or analysis.advisory_target
        advisory_concern = analysis.target is None and analysis.advisory_target is not None
        if concern_target is not None:
            target_changed = concern_target != self._entry_target
            if not target_changed:
                self._entry_streak += 1
            else:
                self._entry_target = concern_target
                self._entry_streak = 1
            self._degraded_clear_count = 0
            self._target_zone_id = concern_target
            target = concern_target
            if (
                self._entry_streak >= self.settings.entry_persistence_ticks
                and not self._reserve_failed
            ):
                if self._epoch_rearm_pending:
                    self._authority_epoch += 1
                    self._epoch_rearm_pending = False
                self._state = AuthorityState.PROTECT
                self._target_zone_id = concern_target
                self._protect_age = 0
                self._protect_clear_count = 0
                self._protect_command = 0.0
                self._protect_entered_by_advisory = advisory_concern
                self._reserve_delivery_failure_count = 0
                desired = self._desired_command(observation, concern_target)
                command = self._slew_up(0.0, desired)
                self._protect_command = command
                commands = self._commands_for_target(command, concern_target)
                reason = (
                    "advisory_entry_persistence_met"
                    if advisory_concern
                    else "entry_persistence_met"
                )
            else:
                reason = (
                    "target_changed"
                    if target_changed
                    else "advisory_unique_concern"
                    if advisory_concern
                    else "unique_concern"
                )
        elif analysis.ambiguous:
            self._entry_target = None
            self._entry_streak = 0
            self._degraded_clear_count = 0
            self._target_zone_id = None
            target = None
            reason = "ambiguous_concern"
        else:
            self._entry_target = None
            self._entry_streak = 0
            self._target_zone_id = None
            self._degraded_clear_count += 1
            target = None
            if self._degraded_clear_count >= self.settings.degraded_clear_persistence_ticks:
                self._state = AuthorityState.NOMINAL
                self._degraded_clear_count = 0
                reason = "degraded_clear"
            else:
                reason = "degraded_clear"
        self._record_transition(
            old_state,
            self._state,
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            reason=reason,
            target_zone_id=target,
        )
        return self._make_decision(
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            target_zone_id=target,
            commands=commands,
            reason=reason,
        )

    def _decide_protect(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis
    ) -> RecoveryDecision:
        previous = self._last_decision
        assert previous is not None
        target = self._target_zone_id
        if target is None:
            raise ValueError("protect state has no latched target")
        old_state = self._state
        previous_value = float(previous.reserve_commands[target])
        if self._reserve_delivery_is_failed(observation, target):
            self._reserve_failed = True
            self._failure_handback = True
            self._state = AuthorityState.HANDBACK
            self._handback_age = 0
            self._handback_zero_ack_count = 0
            self._protect_clear_count = 0
            self._handback_abort_streak = 0
            commands = dict(previous.reserve_commands)
            reason = "reserve_delivery_failure"
        else:
            self._protect_age += 1
            clear_safe = self._protect_clear(observation, analysis, target)
            if clear_safe:
                self._protect_clear_count += 1
                next_value = previous_value
            else:
                self._protect_clear_count = 0
                desired = self._desired_command(observation, target)
                # Keep the largest observed shortfall as a latch, but still
                # slew from the command physically applied on the prior tick.
                self._protect_command = max(self._protect_command, desired)
                next_value = self._slew_up(previous_value, self._protect_command)
            self._protect_command = max(self._protect_command, next_value)
            commands = self._commands_for_target(next_value, target)
            if (
                self._protect_age >= self.settings.minimum_protect_dwell_ticks
                and self._protect_clear_count >= self.settings.recovery_clear_persistence_ticks
            ):
                self._state = AuthorityState.HANDBACK
                self._failure_handback = False
                self._handback_age = 0
                self._handback_zero_ack_count = 0
                self._handback_abort_streak = 0
                reason = "handback_start"
            else:
                reason = (
                    "recovery_clear"
                    if clear_safe
                    else "protect_increase"
                    if next_value > previous_value
                    else "protect_hold"
                )
        self._record_transition(
            old_state,
            self._state,
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            reason=reason,
            target_zone_id=target,
        )
        return self._make_decision(
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            target_zone_id=target,
            commands=commands,
            reason=reason,
        )

    def _decide_handback(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis
    ) -> RecoveryDecision:
        previous = self._last_decision
        assert previous is not None
        target = self._target_zone_id
        if target is None:
            raise ValueError("handback state has no latched target")
        old_state = self._state
        self._handback_age += 1
        if self._handback_age >= self.settings.maximum_handback_ticks:
            commands = self._timeout_handback(target)
            reason = "handback_timeout"
        elif (
            not self._reserve_failed
            and any(float(value) > 0.0 for value in previous.reserve_commands.values())
            and self._reserve_delivery_is_failed(observation, target)
        ):
            self._reserve_failed = True
            self._failure_handback = True
            self._protect_clear_count = 0
            self._handback_zero_ack_count = 0
            self._handback_abort_streak = 0
            commands = self._ramp_down_commands(previous.reserve_commands)
            reason = "reserve_delivery_failure"
        elif self._reserve_failed:
            self._handback_abort_streak = 0
            commands = self._ramp_down_commands(previous.reserve_commands)
            if self._zero_acknowledged(observation, previous):
                self._handback_zero_ack_count += 1
            else:
                self._handback_zero_ack_count = 0
            if self._handback_zero_ack_count >= self.settings.handback_settle_ticks:
                self._state = AuthorityState.DEGRADED
                self._target_zone_id = target
                self._entry_target = None
                self._entry_streak = 0
                self._degraded_clear_count = 0
                self._protect_command = 0.0
                commands = self._zero_commands()
                reason = "failure_latched"
            else:
                reason = "reserve_failure_shutdown"
        elif analysis.ambiguous or bool(analysis.candidates):
            self._state = AuthorityState.PROTECT
            self._protect_age = 0
            self._protect_clear_count = 0
            self._handback_zero_ack_count = 0
            self._handback_abort_streak = 0
            self._reserve_delivery_failure_count = 0
            commands = self._restore_protect_commands(previous.reserve_commands)
            reason = "handback_abort"
        elif (
            analysis.residuals[target] >= self.settings.handback_abort_residual_ratio
            or analysis.advisory_target == target
        ):
            self._handback_abort_streak += 1
            self._handback_zero_ack_count = 0
            if (
                self._handback_abort_streak
                >= self.settings.handback_abort_persistence_ticks
            ):
                self._state = AuthorityState.PROTECT
                self._protect_age = 0
                self._protect_clear_count = 0
                self._handback_abort_streak = 0
                self._reserve_delivery_failure_count = 0
                commands = self._restore_protect_commands(previous.reserve_commands)
                reason = "handback_abort"
            else:
                commands = dict(previous.reserve_commands)
                reason = "handback_wait"
        elif self._handback_safe(observation, analysis, target):
            self._handback_abort_streak = 0
            commands = self._ramp_down_commands(previous.reserve_commands)
            if self._zero_acknowledged(observation, previous):
                self._handback_zero_ack_count += 1
            else:
                self._handback_zero_ack_count = 0
            if self._handback_zero_ack_count >= self.settings.handback_settle_ticks:
                self._state = AuthorityState.NOMINAL
                self._target_zone_id = None
                self._entry_target = None
                self._entry_streak = 0
                self._degraded_clear_count = 0
                self._protect_command = 0.0
                self._handback_abort_streak = 0
                self._epoch_rearm_pending = True
                commands = self._zero_commands()
                reason = "handback_complete"
            else:
                reason = "handback_ramp"
        else:
            self._handback_abort_streak = 0
            self._handback_zero_ack_count = 0
            commands = dict(previous.reserve_commands)
            reason = "handback_ramp"
        self._record_transition(
            old_state,
            self._state,
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            reason=reason,
            target_zone_id=self._target_zone_id,
        )
        return self._make_decision(
            decision_tick=previous.decision_tick + 1,
            observation_tick=observation.completed_tick,
            sequence=previous.sequence + 1,
            target_zone_id=self._target_zone_id,
            commands=commands,
            reason=reason,
        )

    def _desired_command(self, observation: RecoveryObservation, target: str) -> float:
        shortfall = max(
            0.0,
            float(observation.primary_requested_airflow[target])
            - float(observation.primary_delivered_airflow[target]),
        )
        return max(
            0.0,
            min(
                self.settings.maximum_reserve_command,
                shortfall / self._reserve_capacity,
            ),
        )

    def _protect_clear(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis, target: str
    ) -> bool:
        return (
            not analysis.candidates
            and not analysis.ambiguous
            and analysis.advisory_target != target
            and analysis.residuals[target] <= self.settings.exit_residual_ratio
            and float(observation.co2_concentration[target])
            <= float(self.config.control.upper_threshold)
        )

    def _handback_safe(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis, target: str
    ) -> bool:
        del observation
        return (
            not analysis.candidates
            and not analysis.ambiguous
            and analysis.advisory_target != target
            and analysis.residuals[target] < self.settings.handback_abort_residual_ratio
        )

    def _handback_recurrence(
        self, observation: RecoveryObservation, analysis: _ObservationAnalysis, target: str
    ) -> bool:
        del observation
        if analysis.ambiguous or analysis.candidates:
            return True
        return analysis.residuals[target] >= self.settings.handback_abort_residual_ratio

    def _reserve_delivery_is_failed(
        self, observation: RecoveryObservation, target: str
    ) -> bool:
        position = float(observation.reserve_actual_position[target])
        if position <= 1e-12:
            self._reserve_delivery_failure_count = 0
            return False
        requested = position * self._reserve_capacity
        delivered = float(observation.reserve_delivered_airflow[target])
        ratio = delivered / requested if requested > 0.0 else 1.0
        if ratio < self.settings.minimum_reserve_delivery_ratio:
            self._reserve_delivery_failure_count += 1
        else:
            self._reserve_delivery_failure_count = 0
        return (
            self._reserve_delivery_failure_count
            >= self.settings.reserve_delivery_failure_persistence_ticks
        )

    def _zero_acknowledged(
        self, observation: RecoveryObservation, previous: RecoveryDecision
    ) -> bool:
        if any(float(value) != 0.0 for value in previous.reserve_commands.values()):
            return False
        return all(
            float(observation.reserve_actual_position[zone_id]) <= 1e-12
            and float(observation.reserve_delivered_airflow[zone_id]) <= 1e-12
            for zone_id in self._zone_ids
        )

    def _slew_up(self, previous: float, desired: float) -> float:
        return min(
            self.settings.maximum_reserve_command,
            max(previous, min(desired, previous + self.settings.reserve_command_delta)),
        )

    def _ramp_down_commands(self, previous: Mapping[str, float]) -> dict[str, float]:
        return {
            zone_id: max(
                0.0,
                float(previous[zone_id]) - self.settings.reserve_command_delta,
            )
            for zone_id in self._zone_ids
        }

    def _timeout_handback(self, target: str | None) -> dict[str, float]:
        """Latch an unacknowledged handback failure without claiming zero flow."""
        self._reserve_failed = True
        self._failure_handback = True
        self._state = AuthorityState.DEGRADED
        self._target_zone_id = target
        self._entry_target = None
        self._entry_streak = 0
        self._degraded_clear_count = 0
        self._protect_command = 0.0
        self._reserve_delivery_failure_count = 0
        self._handback_zero_ack_count = 0
        self._handback_abort_streak = 0
        return self._zero_commands()

    def _restore_protect_commands(
        self, previous: Mapping[str, float]
    ) -> dict[str, float]:
        target = self._target_zone_id
        if target is None:
            return self._zero_commands()
        previous_value = float(previous[target])
        restored = min(
            self._protect_command,
            previous_value + self.settings.reserve_command_delta,
        )
        return self._commands_for_target(restored, target)

    def _zero_commands(self) -> dict[str, float]:
        return {zone_id: 0.0 for zone_id in self._zone_ids}

    def _commands_for_target(self, command: float, target: str | None) -> dict[str, float]:
        commands = self._zero_commands()
        if target is not None:
            commands[target] = float(command)
        return commands

    def _make_decision(
        self,
        *,
        decision_tick: int,
        observation_tick: int,
        sequence: int,
        target_zone_id: str | None,
        commands: Mapping[str, float],
        reason: str,
    ) -> RecoveryDecision:
        normalized_commands = {
            zone_id: float(commands.get(zone_id, 0.0)) for zone_id in self._zone_ids
        }
        owner = (
            ReserveCommandOwner.RESERVE_OFF
            if self._state in (AuthorityState.NOMINAL, AuthorityState.DEGRADED)
            else ReserveCommandOwner.DETERMINISTIC_RECOVERY_SUPERVISOR
        )
        dwell_ticks = (
            self._protect_age
            if self._state is AuthorityState.PROTECT
            else self._handback_age
            if self._state is AuthorityState.HANDBACK
            else 0
        )
        digest = _decision_digest(
            run_id=self._run_id,
            authority_epoch=self._authority_epoch,
            decision_tick=decision_tick,
            observation_tick=observation_tick,
            sequence=sequence,
            state=self._state,
            owner=owner,
            target_zone_id=target_zone_id,
            reserve_commands=normalized_commands,
            reason=reason,
            dwell_ticks=dwell_ticks,
        )
        decision = RecoveryDecision(
            run_id=self._run_id,
            authority_epoch=self._authority_epoch,
            decision_tick=decision_tick,
            observation_tick=observation_tick,
            sequence=sequence,
            state=self._state,
            reserve_command_owner=owner,
            target_zone_id=target_zone_id,
            reserve_commands=normalized_commands,
            reason=reason,
            command_digest=digest,
            dwell_ticks=dwell_ticks,
        )
        self._last_decision = decision
        return decision

    def _record_transition(
        self,
        old_state: AuthorityState,
        new_state: AuthorityState,
        *,
        decision_tick: int,
        observation_tick: int,
        sequence: int,
        reason: str,
        target_zone_id: str | None,
    ) -> None:
        if old_state is new_state:
            return
        self._event_history.append(
            AuthorityEvent(
                run_id=self._run_id,
                authority_epoch=self._authority_epoch,
                decision_tick=decision_tick,
                observation_tick=observation_tick,
                sequence=sequence,
                from_state=old_state,
                to_state=new_state,
                reason=reason,
                target_zone_id=target_zone_id,
            )
        )
