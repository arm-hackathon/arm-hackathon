"""Fail-closed model-facing projections for the Habitat V2 Forecast D1 fixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from ..physics import CanonicalExternalCommand, validate_external_command
from ..snapshot import OperationalSnapshot, SnapshotVerificationReceipt
from ..state import PlantState
from ..telemetry import derive_observable_topology
from .contracts import ForecastContracts, canonical_json_bytes


STATUS_ORDER = (
    "AVAILABLE",
    "MISSING",
    "NON_FINITE",
    "MALFORMED",
    "DEPENDENCY_UNAVAILABLE",
)
MODE_ORDER = ("dormant", "occupied", "eva_transition", "contingency")
HEALTH_ORDER = ("NOMINAL", "DEGRADED", "CRITICAL", "UNKNOWN")
TARGET_FIELDS = (
    ("temperature_k", "K"),
    ("pressure_pa", "Pa"),
    ("co2_ppm", "ppm"),
    ("o2_mole_fraction", "mole_fraction"),
    ("relative_humidity", "fraction"),
    ("branch_airflow_m3_s", "m3_s"),
)


class ForecastProjectionError(ValueError):
    """Operational evidence cannot be safely converted to forecast tensors."""


@dataclass(frozen=True, slots=True)
class ForecastLayout:
    operational_descriptors: tuple[Mapping[str, str], ...]
    previous_command_descriptors: tuple[Mapping[str, str], ...]
    proposed_action_descriptors: tuple[Mapping[str, str], ...]
    target_descriptors: tuple[Mapping[str, str], ...]
    input_manifest_sha256: str
    target_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ForecastHistory:
    steps: tuple[int, ...]
    completed_times_s: tuple[float, ...]
    numeric_f32: np.ndarray
    status_f32: np.ndarray
    mode_f32: np.ndarray
    health_f32: np.ndarray
    alarm_lifecycle_f32: np.ndarray
    layout: ForecastLayout


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _exact(mapping: Mapping[str, Any], fields: set[str], label: str) -> None:
    unknown, missing = sorted(set(mapping) - fields), sorted(fields - set(mapping))
    if unknown or missing:
        raise ForecastProjectionError(
            f"{label} has unknown={unknown}, missing={missing}"
        )


def _f32(value: Any, label: str) -> np.float32:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ForecastProjectionError(f"{label} must be a finite non-boolean number")
    number = float(value)
    if abs(number) > float(np.finfo(np.float32).max):
        raise ForecastProjectionError(f"{label} overflows float32")
    result = np.float32(number)
    if not np.isfinite(result):
        raise ForecastProjectionError(f"{label} does not remain finite float32")
    return result


def _snapshot_mapping(
    snapshot: OperationalSnapshot,
    receipt: SnapshotVerificationReceipt,
    bundle: ForecastContracts,
) -> dict[str, Any]:
    if (
        type(snapshot) is not OperationalSnapshot
        or type(receipt) is not SnapshotVerificationReceipt
    ):
        raise ForecastProjectionError(
            "history requires exact issued snapshot/receipt pairs"
        )
    mapping = snapshot.to_mapping()
    receipt_map = receipt.to_mapping()
    if snapshot.canonical_bytes != canonical_json_bytes(
        mapping
    ) or receipt.canonical_bytes != canonical_json_bytes(receipt_map):
        raise ForecastProjectionError(
            "issued snapshot/receipt canonical bytes are inconsistent"
        )
    snapshot_body = dict(mapping)
    snapshot_body.pop("snapshot_sha256", None)
    receipt_body = dict(receipt_map)
    receipt_body.pop("snapshot_verification_receipt_sha256", None)
    if (
        hashlib.sha256(canonical_json_bytes(snapshot_body)).hexdigest()
        != snapshot.snapshot_sha256
        or mapping.get("snapshot_sha256") != snapshot.snapshot_sha256
    ):
        raise ForecastProjectionError("snapshot self hash is invalid")
    if (
        hashlib.sha256(canonical_json_bytes(receipt_body)).hexdigest()
        != receipt.snapshot_verification_receipt_sha256
    ):
        raise ForecastProjectionError(
            "snapshot verification receipt self hash is invalid"
        )
    snapshot_fields = {
        "schema_version",
        "control_run_id",
        "authority_epoch",
        "sequence",
        "completed_step",
        "completed_time_s",
        "completed_application_step",
        "completed_operating_mode",
        "primary_telemetry",
        "secondary_telemetry",
        "primary_minus_secondary",
        "command_reference",
        "operational_feedback",
        "operational_resource_gauges",
        "derived_health",
        "active_operational_alarms",
        "hmc_contract_sha256",
        "snapshot_schema_sha256",
        "observable_topology_sha256",
        "completed_plant_receipt_digest",
        "completed_step_receipt_digest",
        "snapshot_sha256",
    }
    _exact(mapping, snapshot_fields, "snapshot")
    for name, expected in (
        ("schema_version", bundle.hmc_contract.snapshot_schema_version),
        ("hmc_contract_sha256", bundle.hmc_contract.hmc_contract_sha256),
        ("snapshot_schema_sha256", bundle.hmc_contract.snapshot_schema_sha256),
        ("observable_topology_sha256", bundle.topology.sha256),
    ):
        if mapping[name] != expected:
            raise ForecastProjectionError(f"snapshot {name} drifts from frozen binding")
    for name, expected in (
        (
            "receipt_schema_sha256",
            bundle.hmc_contract.snapshot_verification_receipt_schema_sha256,
        ),
        (
            "snapshot_verification_contract_sha256",
            bundle.hmc_contract.snapshot_verification_contract_sha256,
        ),
        ("hmc_contract_sha256", bundle.hmc_contract.hmc_contract_sha256),
        ("snapshot_schema_sha256", bundle.hmc_contract.snapshot_schema_sha256),
        ("observable_topology_sha256", bundle.topology.sha256),
    ):
        if receipt_map.get(name) != expected:
            raise ForecastProjectionError(f"receipt {name} drifts from frozen binding")
    for name in (
        "control_run_id",
        "authority_epoch",
        "sequence",
        "completed_step",
        "completed_time_s",
        "snapshot_sha256",
        "completed_plant_receipt_digest",
        "completed_step_receipt_digest",
    ):
        if receipt_map.get(name) != mapping[name]:
            raise ForecastProjectionError(f"snapshot/receipt {name} is inconsistent")
    return mapping


def _sample_ids(bundle: ForecastContracts) -> tuple[tuple[str, str, str], ...]:
    zones, channels = bundle.topology.zone_ids, bundle.topology.environmental_channels
    environmental = tuple(
        (f"{zone}/{channel}", unit, kind)
        for kind in (
            "primary_sensor_head",
            "secondary_sensor_head",
            "derived_primary_minus_secondary",
        )
        for zone in zones
        for channel, unit in zip(
            channels, ("K", "Pa", "ppm", "mole_fraction", "fraction"), strict=True
        )
    )
    feedback: list[tuple[str, str, str]] = []
    unit_by_channel = {
        item["channel_id"]: item["unit"]
        for item in bundle.hmc_contract.data["snapshot_schema"][
            "operational_feedback_channels"
        ]
    }
    for channel in bundle.topology.operational_feedback_channels:
        if channel == "damper_position_by_id":
            feedback.extend(
                (
                    f"{channel}/{damper}",
                    unit_by_channel[channel],
                    "operational_feedback_instrument",
                )
                for _, damper in bundle.topology.branch_pairs
            )
        elif channel in {
            "branch_airflow_m3_s",
            "branch_differential_pressure_pa",
            "cooling_delivery_w",
            "oxygen_delivery_mol_s",
        }:
            feedback.extend(
                (
                    f"{channel}/{zone}",
                    unit_by_channel[channel],
                    "operational_feedback_instrument",
                )
                for zone in zones
            )
        else:
            feedback.append(
                (channel, unit_by_channel[channel], "operational_feedback_instrument")
            )
    return environmental + tuple(feedback)


def _command_descriptors(bundle: ForecastContracts) -> tuple[Mapping[str, str], ...]:
    items: list[Mapping[str, str]] = [
        MappingProxyType({"descriptor_id": "fan_speed_fraction", "unit": "fraction"})
    ]
    items.extend(
        MappingProxyType(
            {"descriptor_id": f"damper_position_by_id/{damper}", "unit": "fraction"}
        )
        for _, damper in bundle.topology.branch_pairs
    )
    items.extend(
        (
            MappingProxyType({"descriptor_id": "scrubber_duty", "unit": "fraction"}),
            MappingProxyType({"descriptor_id": "condenser_duty", "unit": "fraction"}),
        )
    )
    items.extend(
        MappingProxyType({"descriptor_id": f"cooling_removed_w/{zone}", "unit": "W"})
        for zone in bundle.topology.zone_ids
    )
    items.extend(
        MappingProxyType(
            {"descriptor_id": f"oxygen_injection_mol_s/{zone}", "unit": "mol_s"}
        )
        for zone in bundle.topology.zone_ids
    )
    return tuple(items)


def _validate_bundle(bundle: ForecastContracts) -> None:
    if type(bundle) is not ForecastContracts:
        raise ForecastProjectionError("projection requires the frozen contract bundle")
    derived_topology = derive_observable_topology(bundle.development_scenario)
    if (
        bundle.release_tier != "DEVELOPMENT_FIXTURE_ONLY"
        or bundle.reference_scenario.scenario_sha256
        != "a9ee8eecdb4a952ef95347edcabb7dad614280eb496877cc9cddf8a5c9f77de7"
        or bundle.development_scenario.scenario_sha256
        != "d321f86acddbdc3fb73df47f03367fc7acab0c8cfb6dbd66096d30bef5c0e3e8"
        or derived_topology != bundle.topology
        or derived_topology.sha256 != bundle.topology.sha256
        or derived_topology.to_mapping() != bundle.topology.to_mapping()
    ):
        raise ForecastProjectionError("frozen contract bundle identity has drifted")


def forecast_layout(bundle: ForecastContracts) -> ForecastLayout:
    _validate_bundle(bundle)
    operational = tuple(
        MappingProxyType(
            {"descriptor_id": sample_id, "unit": unit, "source_kind": source}
        )
        for sample_id, unit, source in _sample_ids(bundle)
    )
    command = _command_descriptors(bundle)
    target = tuple(
        MappingProxyType({"descriptor_id": f"{zone}/{field}", "unit": unit})
        for zone in bundle.topology.zone_ids
        for field, unit in TARGET_FIELDS
    ) + tuple(
        MappingProxyType({"descriptor_id": descriptor, "unit": "fraction"})
        for descriptor in (
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        )
    )
    identity = {
        "release_tier": bundle.release_tier,
        "hmc_binding_sha256": bundle.binding_sha256,
        "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
        "snapshot_schema_sha256": bundle.hmc_contract.snapshot_schema_sha256,
        "observable_topology_sha256": bundle.topology.sha256,
        "alarm_manifest_sha256": bundle.alarm_manifest_sha256,
        "action_catalogue_sha256": bundle.action_catalogue_sha256,
        "development_profile_sha256": bundle.development_profile_sha256,
        "development_record_contract_sha256": (
            bundle.development_record_contract_sha256
        ),
        "development_scenario_sha256": bundle.development_scenario.scenario_sha256,
    }
    input_manifest = {
        **identity,
        "schema_version": "aeolus_habitat_v2_forecast_input_v1",
        "operational": [dict(x) for x in operational],
        "status_order": list(STATUS_ORDER),
        "mode_order": list(MODE_ORDER),
        "health_order": list(HEALTH_ORDER),
        "alarm_slots": [
            {
                "alarm_id": slot.alarm_id,
                "family": slot.family,
                "target": slot.target,
                "severity": slot.severity,
            }
            for slot in bundle.alarm_slots
        ],
        "alarm_lifecycle_order": list(bundle.alarm_lifecycle_order),
        "previous_final_command": [dict(x) for x in command],
        "proposed_action": [dict(x) for x in command],
    }
    target_manifest = {
        **identity,
        "schema_version": "aeolus_habitat_v2_forecast_target_v1",
        "targets": [dict(x) for x in target],
    }
    return ForecastLayout(
        operational,
        command,
        command,
        target,
        hashlib.sha256(canonical_json_bytes(input_manifest)).hexdigest(),
        hashlib.sha256(canonical_json_bytes(target_manifest)).hexdigest(),
    )


def _samples(
    block: Any,
    expected_kind: str,
    expected: tuple[tuple[str, str, str], ...],
    label: str,
) -> list[dict[str, Any]]:
    if type(block) is not dict:
        raise ForecastProjectionError(f"{label} must be an object")
    _exact(block, {"source_kind", "samples"}, label)
    if block["source_kind"] != expected_kind or type(block["samples"]) is not list:
        raise ForecastProjectionError(f"{label} source kind/samples are invalid")
    if len(block["samples"]) != len(expected):
        raise ForecastProjectionError(f"{label} has wrong descriptor count")
    result: list[dict[str, Any]] = []
    for index, (sample, (descriptor, unit, _)) in enumerate(
        zip(block["samples"], expected, strict=True)
    ):
        if type(sample) is not dict:
            raise ForecastProjectionError(f"{label} sample {index} must be an object")
        _exact(
            sample,
            {"descriptor_id", "availability", "value", "unavailable_reason", "unit"},
            f"{label} sample {index}",
        )
        if sample["descriptor_id"] != descriptor or sample["unit"] != unit:
            raise ForecastProjectionError(
                f"{label} descriptor ordering/unit is invalid"
            )
        availability, value, reason = (
            sample["availability"],
            sample["value"],
            sample["unavailable_reason"],
        )
        if availability == "AVAILABLE":
            _f32(value, f"{label} sample {descriptor}")
            if reason is not None:
                raise ForecastProjectionError(
                    "available sample has an unavailable reason"
                )
        elif (
            availability == "UNAVAILABLE"
            and value is None
            and reason in STATUS_ORDER[1:]
        ):
            pass
        else:
            raise ForecastProjectionError(
                f"{label} sample {descriptor} availability is invalid"
            )
        result.append(sample)
    return result


def _command_vector(bundle: ForecastContracts, command: Any, label: str) -> np.ndarray:
    try:
        canonical = validate_external_command(bundle.development_scenario, command)
    except Exception as error:
        raise ForecastProjectionError(
            f"{label} is not a complete valid V5 command"
        ) from error
    data = canonical.to_mapping()
    values: list[np.float32] = [_f32(data["fan_speed_fraction"], label)]
    values.extend(
        _f32(data["damper_position_by_id"][damper], label)
        for _, damper in bundle.topology.branch_pairs
    )
    values.extend(
        (_f32(data["scrubber_duty"], label), _f32(data["condenser_duty"], label))
    )
    values.extend(
        _f32(data["cooling_removed_w"][zone], label)
        for zone in bundle.topology.zone_ids
    )
    values.extend(
        _f32(data["oxygen_injection_mol_s"][zone], label)
        for zone in bundle.topology.zone_ids
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (27,):
        raise AssertionError("closed V5 command layout is not 27 fields")
    return result


def project_proposed_action(
    bundle: ForecastContracts, command: CanonicalExternalCommand | Mapping[str, Any]
) -> np.ndarray:
    """Project the requested complete command, never the post-arbitration command."""
    _validate_bundle(bundle)
    if type(command) is CanonicalExternalCommand:
        mapping = command.to_mapping()
        if (
            validate_external_command(bundle.development_scenario, mapping).sha256
            != command.sha256
        ):
            raise ForecastProjectionError("canonical proposed command hash is invalid")
    elif isinstance(command, Mapping):
        mapping = command
    else:
        raise ForecastProjectionError(
            "proposed action must be a canonical command or mapping"
        )
    try:
        canonical = validate_external_command(bundle.development_scenario, mapping)
    except Exception as error:
        raise ForecastProjectionError(
            "proposed action is not a complete valid V5 command"
        ) from error
    if canonical.sha256 not in {action.command.sha256 for action in bundle.actions}:
        raise ForecastProjectionError(
            "proposed action is not a hash-bound forecast catalogue action"
        )
    return _readonly(_command_vector(bundle, canonical.to_mapping(), "proposed action"))


def project_history_window(
    bundle: ForecastContracts,
    pairs: Sequence[tuple[OperationalSnapshot, SnapshotVerificationReceipt]],
    *,
    window_steps: int,
) -> ForecastHistory:
    """Project exactly one contiguous window of issued completed observations."""
    if (
        type(bundle) is not ForecastContracts
        or isinstance(pairs, (str, bytes))
        or not isinstance(pairs, Sequence)
        or type(window_steps) is not int
        or isinstance(window_steps, bool)
        or window_steps <= 0
        or len(pairs) != window_steps
    ):
        raise ForecastProjectionError(
            "history must have exactly window_steps issued pairs"
        )
    layout = forecast_layout(bundle)
    expected_environmental = tuple(
        (f"{zone}/{channel}", unit, kind)
        for kind in (
            "primary_sensor_head",
            "secondary_sensor_head",
            "derived_primary_minus_secondary",
        )
        for zone in bundle.topology.zone_ids
        for channel, unit in zip(
            bundle.topology.environmental_channels,
            ("K", "Pa", "ppm", "mole_fraction", "fraction"),
            strict=True,
        )
    )
    expected_feedback = _sample_ids(bundle)[120:]
    numeric = np.zeros((window_steps, 194), dtype=np.float32)
    statuses = np.zeros((window_steps, 167, 5), dtype=np.float32)
    modes = np.zeros((window_steps, 4), dtype=np.float32)
    healths = np.zeros((window_steps, 4), dtype=np.float32)
    alarms = np.zeros((window_steps, 287, 4), dtype=np.float32)
    steps: list[int] = []
    times: list[float] = []
    run_id: str | None = None
    epoch: str | None = None
    resource_ids = (
        "battery_state_of_charge",
        "oxygen_store_fraction",
        "sorbent_remaining_fraction",
    )
    slot_by_id = {slot.alarm_id: slot for slot in bundle.alarm_slots}
    for row, pair in enumerate(pairs):
        if type(pair) is not tuple or len(pair) != 2:
            raise ForecastProjectionError(
                "each history row requires one snapshot/receipt tuple"
            )
        snapshot = _snapshot_mapping(pair[0], pair[1], bundle)
        step, time = snapshot["completed_step"], snapshot["completed_time_s"]
        if (
            type(step) is not int
            or isinstance(step, bool)
            or step <= 0
            or isinstance(time, bool)
            or not isinstance(time, (int, float))
            or not math.isfinite(float(time))
        ):
            raise ForecastProjectionError(
                "reset/non-finite history row is inadmissible"
            )
        if row and (step != steps[-1] + 1 or float(time) <= times[-1]):
            raise ForecastProjectionError(
                "history must have contiguous completed steps and increasing time"
            )
        if run_id is None:
            run_id, epoch = snapshot["control_run_id"], snapshot["authority_epoch"]
        if snapshot["control_run_id"] != run_id or snapshot["authority_epoch"] != epoch:
            raise ForecastProjectionError("history cannot mix HMC run/epoch identities")
        primary = _samples(
            snapshot["primary_telemetry"],
            "primary_sensor_head",
            expected_environmental[:40],
            "primary telemetry",
        )
        secondary = _samples(
            snapshot["secondary_telemetry"],
            "secondary_sensor_head",
            expected_environmental[40:80],
            "secondary telemetry",
        )
        disagreement = _samples(
            snapshot["primary_minus_secondary"],
            "derived_primary_minus_secondary",
            expected_environmental[80:],
            "primary-minus-secondary",
        )
        feedback = _samples(
            snapshot["operational_feedback"],
            "operational_feedback_instrument",
            expected_feedback,
            "operational feedback",
        )
        resources = _samples(
            snapshot["operational_resource_gauges"],
            "operational_resource_gauge",
            tuple((x, "fraction", "operational_resource_gauge") for x in resource_ids),
            "resource gauges",
        )
        feedback_by_id = {sample["descriptor_id"]: sample for sample in feedback}
        if len(feedback_by_id) != len(feedback) or any(
            feedback_by_id[name] != sample
            for name, sample in ((item["descriptor_id"], item) for item in resources)
        ):
            raise ForecastProjectionError(
                "duplicate resource gauges do not exactly match feedback"
            )
        all_samples = primary + secondary + disagreement + feedback
        for column, sample in enumerate(all_samples):
            if sample["availability"] == "AVAILABLE":
                numeric[row, column] = _f32(sample["value"], "operational value")
                statuses[row, column, 0] = 1.0
            else:
                statuses[
                    row, column, STATUS_ORDER.index(sample["unavailable_reason"])
                ] = 1.0
        command_ref = snapshot["command_reference"]
        if (
            type(command_ref) is not dict
            or set(command_ref) != {"source_kind", "command_reference_kind", "command"}
            or command_ref["source_kind"] != "authoritative_command_reference"
            or command_ref["command_reference_kind"] != "COMPLETED_FINAL_COMMAND"
        ):
            raise ForecastProjectionError(
                "history command reference is not a completed final command"
            )
        numeric[row, 167:] = _command_vector(
            bundle, command_ref["command"], "completed final command"
        )
        mode = snapshot["completed_operating_mode"]
        health = snapshot["derived_health"]
        if (
            mode not in MODE_ORDER
            or type(health) is not dict
            or set(health) != {"source_kind", "health_state"}
            or health["source_kind"] != "derived_health"
            or health["health_state"] not in HEALTH_ORDER
        ):
            raise ForecastProjectionError("mode/health input is invalid")
        modes[row, MODE_ORDER.index(mode)] = 1.0
        healths[row, HEALTH_ORDER.index(health["health_state"])] = 1.0
        active = snapshot["active_operational_alarms"]
        if (
            type(active) is not dict
            or set(active) != {"source_kind", "alarms"}
            or active["source_kind"] != "alarm_receipt"
            or type(active["alarms"]) is not list
        ):
            raise ForecastProjectionError("alarm receipt is invalid")
        observed: set[str] = set()
        for item in active["alarms"]:
            if (
                type(item) is not dict
                or set(item)
                != {"alarm_id", "family", "target", "severity", "lifecycle"}
                or item["alarm_id"] in observed
                or item["alarm_id"] not in slot_by_id
            ):
                raise ForecastProjectionError(
                    "alarm descriptor is unknown or duplicate"
                )
            slot = slot_by_id[item["alarm_id"]]
            if (item["family"], item["target"], item["severity"]) != (
                slot.family,
                slot.target,
                slot.severity,
            ) or item["lifecycle"] not in bundle.alarm_lifecycle_order[1:]:
                raise ForecastProjectionError("alarm descriptor/lifecycle is invalid")
            observed.add(item["alarm_id"])
            alarms[
                row,
                [x.alarm_id for x in bundle.alarm_slots].index(item["alarm_id"]),
                bundle.alarm_lifecycle_order.index(item["lifecycle"]),
            ] = 1.0
        for index, slot in enumerate(bundle.alarm_slots):
            if slot.alarm_id not in observed:
                alarms[row, index, 0] = 1.0
        steps.append(step)
        times.append(float(time))
    return ForecastHistory(
        tuple(steps),
        tuple(times),
        _readonly(numeric),
        _readonly(statuses),
        _readonly(modes),
        _readonly(healths),
        _readonly(alarms),
        layout,
    )


def project_physical_targets(
    bundle: ForecastContracts, states: Sequence[PlantState], *, horizon_steps: int
) -> np.ndarray:
    """Project evaluator-only shadow states without touching any HMC private state."""
    _validate_bundle(bundle)
    if (
        isinstance(states, (str, bytes))
        or not isinstance(states, Sequence)
        or type(horizon_steps) is not int
        or isinstance(horizon_steps, bool)
        or horizon_steps <= 0
        or len(states) != horizon_steps
    ):
        raise ForecastProjectionError("target horizon has wrong shape")
    zones = {
        str(item["id"]): item for item in bundle.development_scenario.data["zones"]
    }
    equipment = bundle.development_scenario.data["equipment"]
    values = np.empty((horizon_steps, 51), dtype=np.float32)
    previous_step: int | None = None
    for row, state in enumerate(states):
        if (
            type(state) is not PlantState
            or set(state.zones) != set(bundle.topology.zone_ids)
            or (previous_step is not None and state.step != previous_step + 1)
        ):
            raise ForecastProjectionError(
                "targets require contiguous exact shadow PlantState values"
            )
        if type(state.step) is not int or state.step <= 0:
            raise ForecastProjectionError("target reset state is inadmissible")
        column = 0
        for zone in bundle.topology.zone_ids:
            telemetry = state.zones[zone].telemetry(
                volume_m3=float(zones[zone]["volume_m3"])
            )
            for field, _ in TARGET_FIELDS[:-1]:
                values[row, column] = _f32(telemetry[field], f"target {zone}/{field}")
                column += 1
            values[row, column] = _f32(
                state.utility.actual_airflow_m3_s[zone],
                f"target {zone}/branch_airflow_m3_s",
            )
            column += 1
        values[row, column] = _f32(
            state.utility.battery_energy_wh / float(equipment["battery_capacity_wh"]),
            "target battery state",
        )
        column += 1
        values[row, column] = _f32(
            state.utility.oxygen_store_mol
            / float(
                bundle.development_scenario.data["initial_utility"]["oxygen_store_mol"]
            ),
            "target oxygen store",
        )
        column += 1
        values[row, column] = _f32(
            state.utility.co2_sorbent_remaining_mol
            / float(equipment["scrubber_capacity_mol"]),
            "target sorbent",
        )
        previous_step = state.step
    return _readonly(values)
