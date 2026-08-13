from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

HMC_CONTRACT_SCHEMA_V1 = "aeolus_habitat_v2_hmc_contract_v1"
OPERATIONAL_SNAPSHOT_SCHEMA_V1 = "aeolus_habitat_v2_operational_snapshot_v1"
CONTROL_TRACE_SCHEMA_V1 = "aeolus_habitat_v2_control_trace_v1"

_ENVIRONMENTAL_CHANNELS = (
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
)
_TRACKED_CHANNELS = (
    "fan_speed_fraction",
    "damper_position_by_id",
    "cooling_delivery_w",
)
_RECEIPT_NAMES = (
    "snapshot_verification",
    "proposal",
    "arbitration",
    "step",
    "terminal",
)
_RECEIPT_SCHEMA_V1 = {
    "snapshot_verification": (
        "aeolus_habitat_v2_snapshot_verification_receipt_v1",
        (
            "receipt_schema_sha256",
            "snapshot_verification_contract_sha256",
            "hmc_contract_sha256",
            "snapshot_schema_sha256",
            "observable_topology_sha256",
            "control_run_id",
            "authority_epoch",
            "issuer_id",
            "cycle_id",
            "sequence",
            "completed_step",
            "completed_time_s",
            "snapshot_sha256",
            "completed_plant_receipt_digest",
            "completed_step_receipt_digest",
            "previous_verification_receipt_digest",
            "event_ordinal",
            "previous_control_chain_sha256",
            "snapshot_verification_receipt_sha256",
        ),
    ),
    "proposal": (
        "aeolus_habitat_v2_proposal_receipt_v1",
        (
            "receipt_schema_sha256",
            "hmc_contract_sha256",
            "observable_topology_sha256",
            "control_run_id",
            "authority_epoch",
            "sequence",
            "observation_snapshot_sha256",
            "requested_application_step",
            "attempt_class",
            "attempt_evidence_sha256",
            "source_id",
            "source_type",
            "proposal",
            "proposal_sha256",
            "requested_command_sha256",
            "validation_outcome",
            "reason_code",
            "event_ordinal",
            "previous_control_chain_sha256",
            "proposal_receipt_sha256",
        ),
    ),
    "arbitration": (
        "aeolus_habitat_v2_arbitration_receipt_v1",
        (
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
        ),
    ),
    "step": (
        "aeolus_habitat_v2_step_receipt_v1",
        (
            "receipt_schema_sha256",
            "hmc_contract_sha256",
            "external_command_contract_sha256",
            "observable_topology_sha256",
            "control_run_id",
            "authority_epoch",
            "observation_sequence",
            "application_step",
            "proposal_receipt_sha256",
            "arbitration_receipt_sha256",
            "final_command_sha256",
            "returned_external_command_digest",
            "plant_receipt_digest",
            "application_outcome",
            "previous_step_receipt_digest",
            "event_ordinal",
            "previous_control_chain_sha256",
            "step_receipt_sha256",
        ),
    ),
    "terminal": (
        "aeolus_habitat_v2_terminal_failure_receipt_v1",
        (
            "receipt_schema_sha256",
            "terminal_contract_sha256",
            "hmc_contract_sha256",
            "observable_topology_sha256",
            "control_run_id",
            "authority_epoch",
            "sequence",
            "application_step",
            "lifecycle_phase",
            "last_good_snapshot_sha256",
            "last_good_verification_receipt_sha256",
            "last_good_step_receipt_sha256",
            "proposal_receipt_sha256",
            "arbitration_receipt_sha256",
            "final_command_sha256",
            "candidate_plant_receipt_digest",
            "plant_state_committed",
            "reason_code",
            "event_ordinal",
            "previous_control_chain_sha256",
            "terminal_failure_receipt_sha256",
        ),
    ),
}
_CONTROL_TRACE_HEADER_FIELDS_V1 = (
    "record_type",
    "control_trace_schema_sha256",
    "hmc_implementation_git_sha",
    "hmc_contract_sha256",
    "scenario_sha256",
    "plant_run_id",
    "snapshot_schema_sha256",
    "snapshot_verification_contract_sha256",
    "observable_topology_sha256",
    "external_command_contract_sha256",
    "preflight_contract_sha256",
    "health_policy_sha256",
    "safety_policy_sha256",
    "safe_action_catalogue_sha256",
    "proposal_receipt_schema_sha256",
    "arbitration_receipt_schema_sha256",
    "step_receipt_schema_sha256",
    "terminal_receipt_schema_sha256",
    "control_run_id",
    "authority_epoch",
    "reset_nonce_hex",
    "null_control_chain_sha256",
    "control_trace_header_sha256",
)
_CONTROL_TRACE_FOOTER_FIELDS_V1 = (
    "record_type",
    "control_trace_schema_sha256",
    "control_trace_header_sha256",
    "control_run_id",
    "authority_epoch",
    "terminal_status",
    "final_sequence",
    "last_good_snapshot_sha256",
    "last_good_verification_receipt_sha256",
    "last_good_step_receipt_sha256",
    "last_good_plant_receipt_digest",
    "terminal_failure_receipt_sha256",
    "final_control_chain_sha256",
    "event_count",
    "control_trace_body_sha256",
    "final_state_sha256",
    "control_trace_footer_sha256",
)
_NULL_ROOT_NAMES = (
    "plant_receipt",
    "step_receipt",
    "verification_receipt",
    "proposal_receipt",
    "arbitration_receipt",
    "terminal_receipt",
    "final_command",
    "snapshot",
    "control_chain",
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "snapshot_schema",
    "snapshot_verification_contract",
    "external_command_contract",
    "preflight_contract",
    "reviewed_noise_configuration",
    "health_policy",
    "safety_policy",
    "safe_action_catalogue",
    "receipt_schemas",
    "control_trace",
    "null_roots",
}


class HMCContractError(ValueError):
    """Raised when the HMC contract is not the exact closed V1 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HMCContractError("HMC contract must contain finite JSON data") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(nested) for key, nested in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise HMCContractError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label=label)
    unknown = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if unknown or missing:
        raise HMCContractError(
            f"invalid {label} fields; unknown={unknown}, missing={missing}"
        )
    return mapping


def _exact_string(value: Any, *, label: str, expected: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise HMCContractError(f"{label} must be a non-empty string")
    if expected is not None and value != expected:
        raise HMCContractError(f"unsupported {label}")
    return value


def _exact_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise HMCContractError(f"{label} must be a boolean")
    return value


def _number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HMCContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise HMCContractError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise HMCContractError(f"{label} must be at least {minimum}")
    return number


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HMCContractError(f"{label} must be a positive integer")
    return value


def _ordered_strings(
    value: Any,
    *,
    label: str,
    expected: Sequence[str] | None = None,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise HMCContractError(f"{label} must be an ordered string array")
    if nonempty and not value:
        raise HMCContractError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item for item in value):
        raise HMCContractError(f"{label} must contain non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise HMCContractError(f"{label} must not contain duplicates")
    if expected is not None and result != tuple(expected):
        raise HMCContractError(f"{label} has unsupported order or values")
    return result


def _validate_descriptor_array(
    value: Any,
    *,
    label: str,
    expected_scope: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HMCContractError(f"{label} must be a non-empty descriptor array")
    channel_ids: list[str] = []
    expected_fields = (
        {"channel_id", "scope", "unit"} if expected_scope else {"channel_id", "unit"}
    )
    for index, descriptor in enumerate(value):
        item = _exact_fields(
            descriptor,
            expected_fields,
            label=f"{label}[{index}]",
        )
        channel_ids.append(
            _exact_string(item["channel_id"], label=f"{label}[{index}].channel_id")
        )
        _exact_string(item["unit"], label=f"{label}[{index}].unit")
        if expected_scope:
            scope = _exact_string(item["scope"], label=f"{label}[{index}].scope")
            if scope not in {"scalar", "zone", "damper"}:
                raise HMCContractError(f"{label}[{index}].scope is unsupported")
    if len(channel_ids) != len(set(channel_ids)):
        raise HMCContractError(f"{label} contains duplicate channel IDs")
    return tuple(channel_ids)


def _validate_snapshot_schema(value: Any) -> Mapping[str, Any]:
    schema = _exact_fields(
        value,
        {
            "schema_version",
            "channel_sample_schema_version",
            "sensor_memory_schema_version",
            "source_kinds",
            "availability_values",
            "unavailable_reasons",
            "command_reference_kinds",
            "environmental_channels",
            "operational_feedback_channels",
            "resource_gauge_channels",
        },
        label="snapshot_schema",
    )
    _exact_string(
        schema["schema_version"],
        label="snapshot_schema.schema_version",
        expected=OPERATIONAL_SNAPSHOT_SCHEMA_V1,
    )
    _exact_string(
        schema["channel_sample_schema_version"],
        label="snapshot_schema.channel_sample_schema_version",
        expected="aeolus_habitat_v2_channel_sample_v1",
    )
    _exact_string(
        schema["sensor_memory_schema_version"],
        label="snapshot_schema.sensor_memory_schema_version",
        expected="aeolus_habitat_v2_sensor_memory_v1",
    )
    _ordered_strings(
        schema["source_kinds"],
        label="snapshot_schema.source_kinds",
        expected=(
            "primary_sensor_head",
            "secondary_sensor_head",
            "derived_primary_minus_secondary",
            "operational_feedback_instrument",
            "operational_resource_gauge",
            "authoritative_command_reference",
            "derived_health",
            "alarm_receipt",
        ),
    )
    _ordered_strings(
        schema["availability_values"],
        label="snapshot_schema.availability_values",
        expected=("AVAILABLE", "UNAVAILABLE"),
    )
    _ordered_strings(
        schema["unavailable_reasons"],
        label="snapshot_schema.unavailable_reasons",
        expected=("MISSING", "NON_FINITE", "MALFORMED", "DEPENDENCY_UNAVAILABLE"),
    )
    _ordered_strings(
        schema["command_reference_kinds"],
        label="snapshot_schema.command_reference_kinds",
        expected=("INITIAL_ACHIEVED_STATE_HOLD", "COMPLETED_FINAL_COMMAND"),
    )
    environmental = _validate_descriptor_array(
        schema["environmental_channels"],
        label="snapshot_schema.environmental_channels",
        expected_scope=False,
    )
    if environmental != _ENVIRONMENTAL_CHANNELS:
        raise HMCContractError(
            "snapshot_schema.environmental_channels has unsupported order"
        )
    feedback = _validate_descriptor_array(
        schema["operational_feedback_channels"],
        label="snapshot_schema.operational_feedback_channels",
        expected_scope=True,
    )
    gauges = _ordered_strings(
        schema["resource_gauge_channels"],
        label="snapshot_schema.resource_gauge_channels",
        expected=(
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        ),
    )
    if not set(gauges).issubset(feedback):
        raise HMCContractError("resource gauges must be measured feedback channels")
    return schema


def _validate_noise_configuration(value: Any) -> Mapping[str, Any]:
    noise = _exact_fields(
        value,
        {
            "primary_noise_amplitude",
            "secondary_noise_amplitude",
            "feedback_sensor_noise_amplitude",
        },
        label="reviewed_noise_configuration",
    )
    for head in ("primary_noise_amplitude", "secondary_noise_amplitude"):
        amplitudes = _exact_fields(
            noise[head], set(_ENVIRONMENTAL_CHANNELS), label=f"noise.{head}"
        )
        for channel in _ENVIRONMENTAL_CHANNELS:
            _number(amplitudes[channel], label=f"noise.{head}.{channel}", minimum=0.0)
    _number(
        noise["feedback_sensor_noise_amplitude"],
        label="noise.feedback_sensor_noise_amplitude",
        minimum=0.0,
    )
    return noise


def _validate_threshold_fields(
    value: Any,
    *,
    label: str,
    fields: set[str],
) -> Mapping[str, Any]:
    threshold = _exact_fields(value, fields, label=label)
    for field in fields:
        if field in {"channel", "direction"}:
            continue
        _number(threshold[field], label=f"{label}.{field}", minimum=0.0)
    return threshold


def _validate_health_policy(
    value: Any,
    *,
    noise: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    policy = _exact_fields(
        value,
        {
            "schema_version",
            "health_states",
            "alarm_lifecycle",
            "severity_order",
            "persistence_rows",
            "clear_persistence_rows",
            "environmental",
            "disagreement",
            "tracking",
            "tracked_actuator_channels",
            "excluded_tracking_channels",
            "resource_gauges",
        },
        label="health_policy",
    )
    _exact_string(
        policy["schema_version"],
        label="health_policy.schema_version",
        expected="aeolus_habitat_v2_health_policy_v1",
    )
    _ordered_strings(
        policy["health_states"],
        label="health_policy.health_states",
        expected=("NOMINAL", "DEGRADED", "CRITICAL", "UNKNOWN"),
    )
    _ordered_strings(
        policy["alarm_lifecycle"],
        label="health_policy.alarm_lifecycle",
        expected=("RAISED", "ACTIVE", "CLEARED"),
    )
    _ordered_strings(
        policy["severity_order"],
        label="health_policy.severity_order",
        expected=("ADVISORY", "WARNING", "CRITICAL"),
    )
    _positive_int(policy["persistence_rows"], label="health_policy.persistence_rows")
    _positive_int(
        policy["clear_persistence_rows"],
        label="health_policy.clear_persistence_rows",
    )

    environmental = _exact_fields(
        policy["environmental"],
        {
            "high_co2",
            "low_oxygen",
            "high_temperature",
            "low_temperature",
            "high_humidity",
        },
        label="health_policy.environmental",
    )
    expected_environment = {
        "high_co2": ("co2_ppm", "HIGH"),
        "low_oxygen": ("o2_mole_fraction", "LOW"),
        "high_temperature": ("temperature_k", "HIGH"),
        "low_temperature": ("temperature_k", "LOW"),
        "high_humidity": ("relative_humidity", "HIGH"),
    }
    for family, (channel, direction) in expected_environment.items():
        rule = _validate_threshold_fields(
            environmental[family],
            label=f"health_policy.environmental.{family}",
            fields={
                "channel",
                "direction",
                "warning_enter",
                "critical_enter",
                "warning_clear",
                "critical_clear",
            },
        )
        _exact_string(
            rule["channel"],
            label=f"health_policy.environmental.{family}.channel",
            expected=channel,
        )
        _exact_string(
            rule["direction"],
            label=f"health_policy.environmental.{family}.direction",
            expected=direction,
        )
        if direction == "HIGH":
            if not float(rule["critical_enter"]) >= float(rule["warning_enter"]):
                raise HMCContractError(f"{family} critical enter is not conservative")
        elif not float(rule["critical_enter"]) <= float(rule["warning_enter"]):
            raise HMCContractError(f"{family} critical enter is not conservative")

    disagreement = _exact_fields(
        policy["disagreement"],
        set(_ENVIRONMENTAL_CHANNELS),
        label="health_policy.disagreement",
    )
    primary = _mapping(noise["primary_noise_amplitude"], label="primary noise")
    secondary = _mapping(noise["secondary_noise_amplitude"], label="secondary noise")
    for channel in _ENVIRONMENTAL_CHANNELS:
        rule = _validate_threshold_fields(
            disagreement[channel],
            label=f"health_policy.disagreement.{channel}",
            fields={
                "warning_enter",
                "critical_enter",
                "warning_clear",
                "critical_clear",
            },
        )
        envelope = float(primary[channel]) + float(secondary[channel])
        if (
            float(rule["warning_enter"]) <= envelope
            or float(rule["warning_clear"]) <= envelope
        ):
            raise HMCContractError(
                f"health_policy disagreement {channel} threshold is inside "
                "the noise-only envelope"
            )
        if float(rule["critical_enter"]) < float(rule["warning_enter"]):
            raise HMCContractError(
                f"health_policy disagreement {channel} critical threshold "
                "is not conservative"
            )

    tracking = _exact_fields(
        policy["tracking"], set(_TRACKED_CHANNELS), label="health_policy.tracking"
    )
    tracking_fields = {
        "warning_enter",
        "critical_enter",
        "warning_clear",
        "critical_clear",
        "relative_warning",
        "relative_critical",
        "relative_warning_clear",
        "relative_critical_clear",
    }
    feedback_envelope = 2.0 * float(noise["feedback_sensor_noise_amplitude"])
    for channel in _TRACKED_CHANNELS:
        rule = _validate_threshold_fields(
            tracking[channel],
            label=f"health_policy.tracking.{channel}",
            fields=tracking_fields,
        )
        if (
            float(rule["warning_enter"]) <= feedback_envelope
            or float(rule["warning_clear"]) <= feedback_envelope
        ):
            raise HMCContractError(
                f"health_policy tracking {channel} threshold is inside "
                "the noise-only envelope"
            )
        if float(rule["critical_enter"]) < float(rule["warning_enter"]):
            raise HMCContractError(
                f"health_policy tracking {channel} critical threshold "
                "is not conservative"
            )

    tracked = _ordered_strings(
        policy["tracked_actuator_channels"],
        label="health_policy.tracked_actuator_channels",
        expected=_TRACKED_CHANNELS,
    )
    excluded = _ordered_strings(
        policy["excluded_tracking_channels"],
        label="health_policy.excluded_tracking_channels",
        expected=(
            "oxygen_delivery_mol_s",
            "branch_airflow_m3_s",
            "scrubber_capture_rate_mol_s",
            "condenser_removal_rate_mol_s",
        ),
    )
    if set(tracked) & set(excluded):
        raise HMCContractError("tracked and excluded actuator channels overlap")
    resource = _validate_threshold_fields(
        policy["resource_gauges"],
        label="health_policy.resource_gauges",
        fields={
            "warning_enter",
            "critical_enter",
            "warning_clear",
            "critical_clear",
        },
    )
    if float(resource["critical_enter"]) > float(resource["warning_enter"]):
        raise HMCContractError("resource critical threshold is not conservative")
    return policy, tracked


def _validate_safety_policy(value: Any) -> Mapping[str, Any]:
    policy = _exact_fields(
        value,
        {
            "schema_version",
            "modes",
            "dispositions",
            "command_owners",
            "proposal_attempt_classes",
            "proposal_outcomes",
            "proposal_reason_codes",
            "arbitration_reason_priority",
            "reserve_floors",
            "terminal_reason_codes",
        },
        label="safety_policy",
    )
    _exact_string(
        policy["schema_version"],
        label="safety_policy.schema_version",
        expected="aeolus_habitat_v2_safety_policy_v1",
    )
    for field in (
        "modes",
        "dispositions",
        "command_owners",
        "proposal_attempt_classes",
        "proposal_outcomes",
        "proposal_reason_codes",
        "arbitration_reason_priority",
        "terminal_reason_codes",
    ):
        _ordered_strings(policy[field], label=f"safety_policy.{field}")
    floors = _exact_fields(
        policy["reserve_floors"],
        {
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        },
        label="safety_policy.reserve_floors",
    )
    for gauge, value_item in floors.items():
        floor = _number(
            value_item, label=f"safety_policy.reserve_floors.{gauge}", minimum=0.0
        )
        if floor > 1.0:
            raise HMCContractError("reserve floors must not exceed one")
    return policy


def _validate_safe_catalogue(value: Any) -> Mapping[str, Any]:
    catalogue = _exact_fields(
        value,
        {"schema_version", "safe_hold_rule", "templates"},
        label="safe_action_catalogue",
    )
    _exact_string(
        catalogue["schema_version"],
        label="safe_action_catalogue.schema_version",
        expected="aeolus_habitat_v2_safe_action_catalogue_v1",
    )
    _exact_string(
        catalogue["safe_hold_rule"],
        label="safe_action_catalogue.safe_hold_rule",
        expected="repeat_last_completed_final_command_or_initial_achieved_state",
    )
    templates = _exact_fields(
        catalogue["templates"],
        {
            "emergency_high_co2",
            "emergency_low_oxygen",
            "emergency_high_temperature",
            "emergency_high_humidity",
            "emergency_system_critical",
        },
        label="safe_action_catalogue.templates",
    )
    template_fields = {
        "emergency_high_co2": {
            "fan_target",
            "affected_damper_target",
            "other_damper_minimum",
            "scrubber_target",
        },
        "emergency_low_oxygen": {
            "affected_oxygen_target_max_mol_s",
            "fan_minimum",
            "affected_damper_minimum",
            "allocation_order",
        },
        "emergency_high_temperature": {
            "fan_minimum",
            "affected_damper_minimum",
            "cooling_target",
        },
        "emergency_high_humidity": {
            "fan_minimum",
            "affected_damper_minimum",
            "condenser_target",
        },
        "emergency_system_critical": {"merge_rule"},
    }
    for name, fields in template_fields.items():
        template = _exact_fields(
            templates[name], fields, label=f"safe_action_catalogue.templates.{name}"
        )
        for field, item in template.items():
            if isinstance(item, str):
                _exact_string(
                    item, label=f"safe_action_catalogue.templates.{name}.{field}"
                )
            else:
                number = _number(
                    item,
                    label=f"safe_action_catalogue.templates.{name}.{field}",
                    minimum=0.0,
                )
                if field != "affected_oxygen_target_max_mol_s" and number > 1.0:
                    raise HMCContractError("fractional catalogue targets exceed one")
    return catalogue


def _validate_receipt_schemas(value: Any) -> Mapping[str, Any]:
    schemas = _exact_fields(value, set(_RECEIPT_NAMES), label="receipt_schemas")
    expected_self = {
        "snapshot_verification": "snapshot_verification_receipt_sha256",
        "proposal": "proposal_receipt_sha256",
        "arbitration": "arbitration_receipt_sha256",
        "step": "step_receipt_sha256",
        "terminal": "terminal_failure_receipt_sha256",
    }
    for name in _RECEIPT_NAMES:
        expected_schema_version, expected_fields = _RECEIPT_SCHEMA_V1[name]
        schema = _exact_fields(
            schemas[name],
            {"schema_version", "fields", "self_digest_field"},
            label=f"receipt_schemas.{name}",
        )
        _exact_string(
            schema["schema_version"],
            label=f"receipt_schemas.{name}.schema_version",
            expected=expected_schema_version,
        )
        fields = _ordered_strings(
            schema["fields"],
            label=f"receipt_schemas.{name}.fields",
            expected=expected_fields,
        )
        self_field = _exact_string(
            schema["self_digest_field"],
            label=f"receipt_schemas.{name}.self_digest_field",
            expected=expected_self[name],
        )
        if fields.count(self_field) != 1 or fields[-1] != self_field:
            raise HMCContractError(
                f"receipt_schemas.{name} self digest must be the final exact field"
            )
    return schemas


def _validate_control_trace(value: Any) -> Mapping[str, Any]:
    trace = _exact_fields(
        value,
        {
            "schema_version",
            "event_kinds",
            "event_fields",
            "header_fields",
            "footer_fields",
            "terminal_statuses",
            "domains",
        },
        label="control_trace",
    )
    _exact_string(
        trace["schema_version"],
        label="control_trace.schema_version",
        expected=CONTROL_TRACE_SCHEMA_V1,
    )
    _ordered_strings(
        trace["event_kinds"],
        label="control_trace.event_kinds",
        expected=(
            "SNAPSHOT_VERIFICATION",
            "PROPOSAL",
            "ARBITRATION",
            "STEP",
            "TERMINAL",
        ),
    )
    _ordered_strings(
        trace["event_fields"],
        label="control_trace.event_fields",
        expected=(
            "record_type",
            "event_ordinal",
            "event_kind",
            "receipt_sha256",
            "previous_control_chain_sha256",
            "control_chain_sha256",
            "receipt",
        ),
    )
    _ordered_strings(
        trace["header_fields"],
        label="control_trace.header_fields",
        expected=_CONTROL_TRACE_HEADER_FIELDS_V1,
    )
    _ordered_strings(
        trace["footer_fields"],
        label="control_trace.footer_fields",
        expected=_CONTROL_TRACE_FOOTER_FIELDS_V1,
    )
    _ordered_strings(
        trace["terminal_statuses"],
        label="control_trace.terminal_statuses",
        expected=("COMPLETED", "TERMINAL_FAILURE"),
    )
    domains = _exact_fields(
        trace["domains"],
        {"chain", "body", "final_state"},
        label="control_trace.domains",
    )
    expected_domains = {
        "chain": "aeolus-habitat-v2-hmc-control-chain-v1",
        "body": "aeolus-habitat-v2-hmc-control-body-v1",
        "final_state": "aeolus-habitat-v2-hmc-final-state-v1",
    }
    for name, expected in expected_domains.items():
        _exact_string(
            domains[name], label=f"control_trace.domains.{name}", expected=expected
        )
    return trace


def _validate_null_roots(value: Any) -> Mapping[str, Any]:
    roots = _exact_fields(value, set(_NULL_ROOT_NAMES), label="null_roots")
    for name in _NULL_ROOT_NAMES:
        root = _exact_fields(
            roots[name], {"label", "sha256"}, label=f"null_roots.{name}"
        )
        label = _exact_string(root["label"], label=f"null_roots.{name}.label")
        digest = _exact_string(root["sha256"], label=f"null_roots.{name}.sha256")
        expected = hashlib.sha256(label.encode("utf-8")).hexdigest()
        if digest != expected:
            raise HMCContractError(f"null_roots.{name} digest does not match label")
    return roots


def _validate_simple_contract(
    value: Any,
    *,
    label: str,
    expected_fields: set[str],
    schema_version: str,
) -> Mapping[str, Any]:
    contract = _exact_fields(value, expected_fields, label=label)
    _exact_string(
        contract["schema_version"],
        label=f"{label}.schema_version",
        expected=schema_version,
    )
    return contract


@dataclass(frozen=True, slots=True)
class HMCContract:
    schema_version: str
    snapshot_schema_version: str
    control_trace_schema_version: str
    canonical_bytes: bytes
    hmc_contract_sha256: str
    snapshot_schema_sha256: str
    snapshot_verification_contract_sha256: str
    external_command_contract_sha256: str
    preflight_contract_sha256: str
    health_policy_sha256: str
    safety_policy_sha256: str
    safe_action_catalogue_sha256: str
    proposal_receipt_schema_sha256: str
    arbitration_receipt_schema_sha256: str
    step_receipt_schema_sha256: str
    terminal_receipt_schema_sha256: str
    snapshot_verification_receipt_schema_sha256: str
    control_trace_schema_sha256: str
    reviewed_noise_configuration: Mapping[str, Any]
    tracked_actuator_channels: tuple[str, ...]
    data: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> HMCContract:
        values = _exact_fields(mapping, _TOP_LEVEL_FIELDS, label="HMC contract")
        schema_version = _exact_string(
            values["schema_version"],
            label="HMC contract schema_version",
            expected=HMC_CONTRACT_SCHEMA_V1,
        )
        snapshot = _validate_snapshot_schema(values["snapshot_schema"])
        verification = _validate_simple_contract(
            values["snapshot_verification_contract"],
            label="snapshot_verification_contract",
            expected_fields={
                "schema_version",
                "issuer_domain",
                "cycle_domain",
                "exact_runtime_capability_required",
                "cache_one_pair_per_cycle",
            },
            schema_version="aeolus_habitat_v2_snapshot_verification_contract_v1",
        )
        _exact_string(
            verification["issuer_domain"],
            label="snapshot_verification_contract.issuer_domain",
            expected="aeolus-habitat-v2-hmc-issuer-v1",
        )
        _exact_string(
            verification["cycle_domain"],
            label="snapshot_verification_contract.cycle_domain",
            expected="aeolus-habitat-v2-hmc-cycle-v1",
        )
        if not _exact_bool(
            verification["exact_runtime_capability_required"],
            label="snapshot_verification_contract.exact_runtime_capability_required",
        ) or not _exact_bool(
            verification["cache_one_pair_per_cycle"],
            label="snapshot_verification_contract.cache_one_pair_per_cycle",
        ):
            raise HMCContractError(
                "snapshot verification capability rules must be enabled"
            )

        external = _validate_simple_contract(
            values["external_command_contract"],
            label="external_command_contract",
            expected_fields={
                "schema_version",
                "v1_v2_fields",
                "v3_v4_v5_fields",
                "canonical_json",
            },
            schema_version="aeolus_habitat_v2_external_command_v1",
        )
        _ordered_strings(
            external["v1_v2_fields"], label="external_command_contract.v1_v2_fields"
        )
        _ordered_strings(
            external["v3_v4_v5_fields"],
            label="external_command_contract.v3_v4_v5_fields",
        )
        if not _exact_bool(
            external["canonical_json"], label="external_command_contract.canonical_json"
        ):
            raise HMCContractError("external command canonical JSON must be enabled")

        preflight = _validate_simple_contract(
            values["preflight_contract"],
            label="preflight_contract",
            expected_fields={"schema_version", "classifications", "fields"},
            schema_version="aeolus_habitat_v2_hmc_preflight_v1",
        )
        _ordered_strings(
            preflight["classifications"],
            label="preflight_contract.classifications",
            expected=("FEASIBLE", "INFEASIBLE"),
        )
        _ordered_strings(
            preflight["fields"],
            label="preflight_contract.fields",
            expected=(
                "classification",
                "application_step",
                "command_sha256",
                "preflight_contract_sha256",
                "preflight_result_sha256",
            ),
        )

        noise = _validate_noise_configuration(values["reviewed_noise_configuration"])
        health, tracked = _validate_health_policy(values["health_policy"], noise=noise)
        safety = _validate_safety_policy(values["safety_policy"])
        catalogue = _validate_safe_catalogue(values["safe_action_catalogue"])
        receipt_schemas = _validate_receipt_schemas(values["receipt_schemas"])
        control_trace = _validate_control_trace(values["control_trace"])
        _validate_null_roots(values["null_roots"])

        canonical = canonical_json_bytes(values)
        receipt_hashes = {
            name: canonical_sha256(receipt_schemas[name]) for name in _RECEIPT_NAMES
        }
        frozen = _freeze(json.loads(canonical))
        return cls(
            schema_version=schema_version,
            snapshot_schema_version=str(snapshot["schema_version"]),
            control_trace_schema_version=str(control_trace["schema_version"]),
            canonical_bytes=canonical,
            hmc_contract_sha256=hashlib.sha256(canonical).hexdigest(),
            snapshot_schema_sha256=canonical_sha256(snapshot),
            snapshot_verification_contract_sha256=canonical_sha256(verification),
            external_command_contract_sha256=canonical_sha256(external),
            preflight_contract_sha256=canonical_sha256(preflight),
            health_policy_sha256=canonical_sha256(health),
            safety_policy_sha256=canonical_sha256(safety),
            safe_action_catalogue_sha256=canonical_sha256(catalogue),
            proposal_receipt_schema_sha256=receipt_hashes["proposal"],
            arbitration_receipt_schema_sha256=receipt_hashes["arbitration"],
            step_receipt_schema_sha256=receipt_hashes["step"],
            terminal_receipt_schema_sha256=receipt_hashes["terminal"],
            snapshot_verification_receipt_schema_sha256=receipt_hashes[
                "snapshot_verification"
            ],
            control_trace_schema_sha256=canonical_sha256(control_trace),
            reviewed_noise_configuration=_freeze(
                json.loads(canonical_json_bytes(noise))
            ),
            tracked_actuator_channels=tracked,
            data=frozen,
        )


def load_hmc_contract(path: str | Path) -> HMCContract:
    contract_path = Path(path)
    raw = contract_path.read_bytes()
    try:
        mapping = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HMCContractError("HMC contract must be valid UTF-8 JSON") from error
    contract = HMCContract.from_mapping(mapping)
    if raw.rstrip(b"\n") != contract.canonical_bytes or raw not in {
        contract.canonical_bytes,
        contract.canonical_bytes + b"\n",
    }:
        raise HMCContractError("HMC contract file must use canonical JSON")
    return contract
