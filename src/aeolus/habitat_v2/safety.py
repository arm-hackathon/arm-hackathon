from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hmc_contract import HMCContract, canonical_json_bytes
from .physics import CanonicalExternalCommand, validate_external_command
from .scenario import Scenario
from .snapshot import _FinalIssuedType

_ARBITRATION_RECEIPT_ISSUANCE_TOKEN = object()
_ARBITRATION_RECEIPT_FIELDS = (
    "receipt_schema_sha256",
    "hmc_contract_sha256",
    "safety_policy_sha256",
    "safe_action_catalogue_sha256",
    "preflight_contract_sha256",
    "observable_topology_sha256",
    "control_run_id",
    "authority_epoch",
    "sequence",
    "observation_snapshot_sha256",
    "proposal_receipt_sha256",
    "accepted_proposal_sha256",
    "requested_command",
    "requested_command_sha256",
    "final_command",
    "final_command_sha256",
    "disposition",
    "reason_codes",
    "command_owner",
    "emergency_override",
    "emergency_reserve_use",
    "imminent_application_mode",
    "preflight_result",
    "decision_step",
    "application_step",
    "event_ordinal",
    "previous_control_chain_sha256",
    "arbitration_receipt_sha256",
)
_PRE_FLIGHT_FIELDS = {
    "classification",
    "application_step",
    "command_sha256",
    "preflight_contract_sha256",
    "preflight_result_sha256",
}
_ACTIONABLE_ENVIRONMENTAL_FAMILIES = {
    "high_co2",
    "low_oxygen",
    "high_temperature",
    "high_humidity",
}


class ArbitrationIssuanceError(ValueError):
    """Raised when deterministic arbitration evidence cannot be issued."""


@dataclass(frozen=True, init=False, slots=True)
class ArbitrationReceipt(_FinalIssuedType):
    canonical_bytes: bytes
    arbitration_receipt_sha256: str
    preflight_result: Mapping[str, Any]
    final_command: Mapping[str, Any]
    final_command_sha256: str

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _ARBITRATION_RECEIPT_ISSUANCE_TOKEN:
            raise TypeError("ArbitrationReceipt must be issued by an HMC")
        if type(mapping) is not dict or type(canonical_bytes) is not bytes:
            raise ArbitrationIssuanceError(
                "arbitration receipt issuance data is malformed"
            )
        for field in (
            "arbitration_receipt_sha256",
            "preflight_result",
            "final_command",
            "final_command_sha256",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_non_negative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _validate_preflight(
    value: object,
    *,
    contract: HMCContract,
    final_command_sha256: str,
    application_step: int,
) -> None:
    if type(value) is not dict or set(value) != _PRE_FLIGHT_FIELDS:
        raise ArbitrationIssuanceError("arbitration preflight result is malformed")
    if (
        value["classification"] not in {"FEASIBLE", "INFEASIBLE"}
        or value["application_step"] != application_step
        or value["command_sha256"] != final_command_sha256
        or value["preflight_contract_sha256"] != contract.preflight_contract_sha256
        or not _is_sha256(value["preflight_result_sha256"])
    ):
        raise ArbitrationIssuanceError("arbitration preflight result is inconsistent")
    body = dict(value)
    claimed = body.pop("preflight_result_sha256")
    if hashlib.sha256(canonical_json_bytes(body)).hexdigest() != claimed:
        raise ArbitrationIssuanceError("arbitration preflight digest is inconsistent")


def _issue_arbitration_receipt(
    content: Mapping[str, Any],
    *,
    scenario: Scenario,
    contract: HMCContract,
) -> ArbitrationReceipt:
    self_field = "arbitration_receipt_sha256"
    expected = set(_ARBITRATION_RECEIPT_FIELDS) - {self_field}
    if type(content) is not dict or set(content) != expected:
        raise ArbitrationIssuanceError("arbitration receipt content is malformed")
    if type(scenario) is not Scenario or type(contract) is not HMCContract:
        raise ArbitrationIssuanceError("arbitration issuance context is malformed")
    identity_fields = {
        "receipt_schema_sha256": contract.arbitration_receipt_schema_sha256,
        "hmc_contract_sha256": contract.hmc_contract_sha256,
        "safety_policy_sha256": contract.safety_policy_sha256,
        "safe_action_catalogue_sha256": contract.safe_action_catalogue_sha256,
        "preflight_contract_sha256": contract.preflight_contract_sha256,
    }
    if any(
        content[field] != expected_value
        for field, expected_value in identity_fields.items()
    ):
        raise ArbitrationIssuanceError("arbitration contract identity is inconsistent")
    for field in (
        "observable_topology_sha256",
        "control_run_id",
        "authority_epoch",
        "observation_snapshot_sha256",
        "proposal_receipt_sha256",
        "previous_control_chain_sha256",
    ):
        if not _is_sha256(content[field]):
            raise ArbitrationIssuanceError("arbitration digest identity is malformed")
    for field in ("sequence", "decision_step", "application_step", "event_ordinal"):
        if not _is_non_negative_integer(content[field]):
            raise ArbitrationIssuanceError("arbitration sequence identity is malformed")
    if (
        content["sequence"] != content["decision_step"]
        or content["decision_step"] != content["application_step"]
    ):
        raise ArbitrationIssuanceError("arbitration step identities are inconsistent")
    if (
        type(content["emergency_override"]) is not bool
        or type(content["emergency_reserve_use"]) is not bool
    ):
        raise ArbitrationIssuanceError("arbitration emergency flags are malformed")

    policy = contract.data["safety_policy"]
    if content["disposition"] not in policy["dispositions"]:
        raise ArbitrationIssuanceError("arbitration disposition is unsupported")
    if content["command_owner"] not in policy["command_owners"]:
        raise ArbitrationIssuanceError("arbitration command owner is unsupported")
    if content["imminent_application_mode"] not in policy["modes"]:
        raise ArbitrationIssuanceError("arbitration operating mode is unsupported")
    reason_codes = content["reason_codes"]
    priority = tuple(policy["arbitration_reason_priority"])
    if (
        type(reason_codes) is not list
        or not reason_codes
        or any(
            type(reason) is not str or reason not in priority for reason in reason_codes
        )
        or len(reason_codes) != len(set(reason_codes))
        or reason_codes != sorted(reason_codes, key=priority.index)
    ):
        raise ArbitrationIssuanceError("arbitration reason codes are malformed")

    final_command = validate_external_command(scenario, content["final_command"])
    if content["final_command_sha256"] != final_command.sha256:
        raise ArbitrationIssuanceError(
            "arbitration final command digest is inconsistent"
        )
    requested_command = content["requested_command"]
    requested_digest = content["requested_command_sha256"]
    if requested_command is None:
        if requested_digest is not None:
            raise ArbitrationIssuanceError(
                "arbitration requested command is inconsistent"
            )
    else:
        requested = validate_external_command(scenario, requested_command)
        if requested_digest != requested.sha256:
            raise ArbitrationIssuanceError(
                "arbitration requested command digest is inconsistent"
            )
    accepted = content["accepted_proposal_sha256"]
    if accepted is not None and not _is_sha256(accepted):
        raise ArbitrationIssuanceError(
            "arbitration accepted proposal identity is malformed"
        )
    _validate_preflight(
        content["preflight_result"],
        contract=contract,
        final_command_sha256=final_command.sha256,
        application_step=content["application_step"],
    )
    mapping = dict(content)
    mapping[self_field] = hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()
    canonical = canonical_json_bytes(mapping)
    return ArbitrationReceipt(
        mapping=mapping,
        canonical_bytes=canonical,
        _token=_ARBITRATION_RECEIPT_ISSUANCE_TOKEN,
    )


def apply_operating_mode_policy(
    *,
    scenario: Scenario,
    imminent_mode: str,
    requested: CanonicalExternalCommand,
    safe_hold: CanonicalExternalCommand,
) -> tuple[CanonicalExternalCommand, bool]:
    if (
        type(requested) is not CanonicalExternalCommand
        or type(safe_hold) is not CanonicalExternalCommand
    ):
        raise TypeError("mode policy requires exact canonical commands")
    if imminent_mode != "dormant":
        return requested, False
    requested_mapping = requested.to_mapping()
    hold_mapping = safe_hold.to_mapping()
    changed = False
    for field in ("cooling_removed_w", "oxygen_injection_mol_s"):
        for zone_id in sorted(requested_mapping[field]):
            if requested_mapping[field][zone_id] > hold_mapping[field][zone_id]:
                requested_mapping[field][zone_id] = hold_mapping[field][zone_id]
                changed = True
    return validate_external_command(scenario, requested_mapping), changed


def _measured_reserve_values(
    resource_samples: object,
    *,
    reserve_floors: Mapping[str, Any],
) -> dict[str, float]:
    if type(resource_samples) is not list or not isinstance(reserve_floors, Mapping):
        raise TypeError("reserve policy requires closed measured gauge data")
    values: dict[str, float] = {}
    for sample in resource_samples:
        if (
            type(sample) is not dict
            or set(sample)
            != {
                "descriptor_id",
                "unit",
                "availability",
                "value",
                "unavailable_reason",
            }
            or sample["availability"] != "AVAILABLE"
            or sample["unavailable_reason"] is not None
            or sample["unit"] != "fraction"
            or isinstance(sample["value"], bool)
            or not isinstance(sample["value"], (int, float))
        ):
            raise ValueError(
                "reserve policy measured gauge is malformed or unavailable"
            )
        descriptor_id = sample["descriptor_id"]
        if type(descriptor_id) is not str or descriptor_id in values:
            raise ValueError("reserve policy gauge identity is invalid")
        value = float(sample["value"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("reserve policy measured gauge is out of bounds")
        values[descriptor_id] = value
    if set(values) != set(reserve_floors):
        raise ValueError("reserve policy gauge topology is incomplete")
    return values


def apply_reserve_policy(
    *,
    scenario: Scenario,
    candidate: CanonicalExternalCommand,
    safe_hold: CanonicalExternalCommand,
    resource_samples: object,
    reserve_floors: Mapping[str, Any],
) -> tuple[CanonicalExternalCommand, bool]:
    values = _measured_reserve_values(
        resource_samples,
        reserve_floors=reserve_floors,
    )
    candidate_mapping = candidate.to_mapping()
    hold_mapping = safe_hold.to_mapping()
    changed = False

    def clamp_scalar(field: str) -> None:
        nonlocal changed
        if candidate_mapping[field] > hold_mapping[field]:
            candidate_mapping[field] = hold_mapping[field]
            changed = True

    def clamp_mapping(field: str) -> None:
        nonlocal changed
        for item_id in sorted(candidate_mapping[field]):
            if candidate_mapping[field][item_id] > hold_mapping[field][item_id]:
                candidate_mapping[field][item_id] = hold_mapping[field][item_id]
                changed = True

    if values["battery_state_of_charge"] < float(
        reserve_floors["battery_state_of_charge"]
    ):
        for field in ("fan_speed_fraction", "scrubber_duty", "condenser_duty"):
            clamp_scalar(field)
        for field in (
            "damper_position_by_id",
            "cooling_removed_w",
            "oxygen_injection_mol_s",
        ):
            clamp_mapping(field)
    if values["oxygen_store_fraction"] < float(reserve_floors["oxygen_store_fraction"]):
        clamp_mapping("oxygen_injection_mol_s")
    if values["sorbent_remaining_fraction"] < float(
        reserve_floors["sorbent_remaining_fraction"]
    ):
        clamp_scalar("scrubber_duty")
    return validate_external_command(scenario, candidate_mapping), changed


def _active_critical_environmental_alarms(
    alarms: object,
) -> tuple[tuple[str, str], ...]:
    if type(alarms) is not list:
        raise ValueError("operational alarm evidence is malformed")
    selected: list[tuple[str, str]] = []
    for alarm in alarms:
        if (
            type(alarm) is not dict
            or set(alarm) != {"alarm_id", "family", "target", "severity", "lifecycle"}
            or any(type(alarm[field]) is not str for field in alarm)
        ):
            raise ValueError("operational alarm evidence is malformed")
        if alarm["severity"] == "CRITICAL" and alarm["lifecycle"] == "ACTIVE":
            family = alarm["family"]
            target = alarm["target"]
            if family in _ACTIONABLE_ENVIRONMENTAL_FAMILIES:
                selected.append((family, target))
    if len(selected) != len(set(selected)):
        raise ValueError("operational alarm evidence contains duplicates")
    return tuple(sorted(selected))


def expand_emergency_action(
    *,
    scenario: Scenario,
    safe_hold: CanonicalExternalCommand,
    alarms: object,
    catalogue: Mapping[str, Any],
) -> CanonicalExternalCommand | None:
    if (
        type(scenario) is not Scenario
        or type(safe_hold) is not CanonicalExternalCommand
    ):
        raise TypeError("emergency expansion requires exact trusted inputs")
    selected = _active_critical_environmental_alarms(alarms)
    if not selected:
        return None
    templates = catalogue["templates"]
    command = safe_hold.to_mapping()
    branch_by_zone = {
        str(branch["zone_id"]): str(branch["damper_id"])
        for branch in scenario.data["air_network"]["branches"]
    }
    zone_ids = tuple(sorted(branch_by_zone))
    if set(command["cooling_removed_w"]) != set(zone_ids):
        raise ValueError("emergency expansion topology is inconsistent")
    oxygen_affected: set[str] = set()

    for family, zone_id in selected:
        if zone_id not in branch_by_zone:
            raise ValueError("emergency alarm target is outside observable topology")
        damper_id = branch_by_zone[zone_id]
        if family == "high_co2":
            template = templates["emergency_high_co2"]
            command["fan_speed_fraction"] = max(
                command["fan_speed_fraction"], float(template["fan_target"])
            )
            command["scrubber_duty"] = max(
                command["scrubber_duty"], float(template["scrubber_target"])
            )
            for candidate_damper in sorted(command["damper_position_by_id"]):
                minimum = (
                    float(template["affected_damper_target"])
                    if candidate_damper == damper_id
                    else float(template["other_damper_minimum"])
                )
                command["damper_position_by_id"][candidate_damper] = max(
                    command["damper_position_by_id"][candidate_damper], minimum
                )
        elif family == "high_temperature":
            template = templates["emergency_high_temperature"]
            command["fan_speed_fraction"] = max(
                command["fan_speed_fraction"], float(template["fan_minimum"])
            )
            command["damper_position_by_id"][damper_id] = max(
                command["damper_position_by_id"][damper_id],
                float(template["affected_damper_minimum"]),
            )
            command["cooling_removed_w"][zone_id] = max(
                command["cooling_removed_w"][zone_id],
                float(scenario.data["equipment"]["cooling_max_thermal_w_per_zone"]),
            )
        elif family == "high_humidity":
            template = templates["emergency_high_humidity"]
            command["fan_speed_fraction"] = max(
                command["fan_speed_fraction"], float(template["fan_minimum"])
            )
            command["damper_position_by_id"][damper_id] = max(
                command["damper_position_by_id"][damper_id],
                float(template["affected_damper_minimum"]),
            )
            command["condenser_duty"] = max(
                command["condenser_duty"], float(template["condenser_target"])
            )
        elif family == "low_oxygen":
            template = templates["emergency_low_oxygen"]
            command["fan_speed_fraction"] = max(
                command["fan_speed_fraction"], float(template["fan_minimum"])
            )
            command["damper_position_by_id"][damper_id] = max(
                command["damper_position_by_id"][damper_id],
                float(template["affected_damper_minimum"]),
            )
            oxygen_affected.add(zone_id)

    if oxygen_affected:
        oxygen = command["oxygen_injection_mol_s"]
        capacity = float(scenario.data["equipment"]["oxygen_injection_max_total_mol_s"])
        target = min(
            float(
                templates["emergency_low_oxygen"]["affected_oxygen_target_max_mol_s"]
            ),
            capacity,
        )
        for zone_id in sorted(oxygen_affected):
            oxygen[zone_id] = max(oxygen[zone_id], target)
        excess = max(0.0, sum(oxygen.values()) - capacity)
        for zone_id in sorted(set(zone_ids) - oxygen_affected):
            reduction = min(oxygen[zone_id], excess)
            oxygen[zone_id] -= reduction
            excess -= reduction
        if excess > 1e-12:
            return None
    return validate_external_command(scenario, command)


def emergency_crosses_normal_reserve_floor(
    *,
    emergency_command: CanonicalExternalCommand,
    safe_hold: CanonicalExternalCommand,
    resource_samples: object,
    reserve_floors: Mapping[str, Any],
) -> bool:
    values = _measured_reserve_values(
        resource_samples,
        reserve_floors=reserve_floors,
    )
    candidate = emergency_command.to_mapping()
    hold = safe_hold.to_mapping()

    def scalar_increased(field: str) -> bool:
        return candidate[field] > hold[field]

    def mapping_increased(field: str) -> bool:
        return any(candidate[field][key] > hold[field][key] for key in candidate[field])

    if values["battery_state_of_charge"] < float(
        reserve_floors["battery_state_of_charge"]
    ) and (
        any(
            scalar_increased(field)
            for field in ("fan_speed_fraction", "scrubber_duty", "condenser_duty")
        )
        or any(
            mapping_increased(field)
            for field in (
                "damper_position_by_id",
                "cooling_removed_w",
                "oxygen_injection_mol_s",
            )
        )
    ):
        return True
    if values["oxygen_store_fraction"] < float(
        reserve_floors["oxygen_store_fraction"]
    ) and mapping_increased("oxygen_injection_mol_s"):
        return True
    return values["sorbent_remaining_fraction"] < float(
        reserve_floors["sorbent_remaining_fraction"]
    ) and scalar_increased("scrubber_duty")


__all__ = (
    "ArbitrationIssuanceError",
    "ArbitrationReceipt",
    "apply_operating_mode_policy",
    "apply_reserve_policy",
    "emergency_crosses_normal_reserve_floor",
    "expand_emergency_action",
)
