from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

SCENARIO_SCHEMA_VERSION_V1 = "aeolus_habitat_v2_scenario_v1"
SCENARIO_SCHEMA_VERSION_V2 = "aeolus_habitat_v2_scenario_v2"
SCENARIO_SCHEMA_VERSION_V3 = "aeolus_habitat_v2_scenario_v3"
SCENARIO_SCHEMA_VERSION_V4 = "aeolus_habitat_v2_scenario_v4"
TRACE_SCHEMA_VERSION_V1 = "aeolus_habitat_v2_trace_v1"
TRACE_SCHEMA_VERSION_V2 = "aeolus_habitat_v2_trace_v2"
TRACE_SCHEMA_VERSION_V3 = "aeolus_habitat_v2_trace_v3"
TRACE_SCHEMA_VERSION_V4 = "aeolus_habitat_v2_trace_v4"
# V1 aliases are retained for callers that import the original contract names.
SCENARIO_SCHEMA_VERSION = SCENARIO_SCHEMA_VERSION_V1
TRACE_SCHEMA_VERSION = TRACE_SCHEMA_VERSION_V1
EQUATION_CONTRACT_REVISION = "aeolus_habitat_v2_equations_v1"
EQUATION_CONTRACT_REVISION_V2 = "aeolus_habitat_v2_equations_v2"
EQUATION_CONTRACT_REVISION_V3 = "aeolus_habitat_v2_equations_v3"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "dt_seconds",
    "steps",
    "zones",
    "equipment",
    "initial_utility",
    "timeline",
}
_ZONE_FIELDS = {
    "id",
    "volume_m3",
    "thermal_capacity_j_per_k",
    "passive_thermal_conductance_w_per_k",
    "sink_temperature_k",
    "initial",
}
_ZONE_INITIAL_FIELDS = {
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
}
_EQUIPMENT_FIELDS = {
    "max_total_airflow_m3_s",
    "max_zone_airflow_m3_s",
    "airflow_slew_m3_s2",
    "scrubber_duty_slew_per_s",
    "condenser_duty_slew_per_s",
    "scrubber_max_co2_mol_s",
    "scrubber_capacity_mol",
    "scrubber_power_w_full",
    "condenser_max_water_mol_s",
    "condenser_power_w_full",
    "cooling_max_thermal_w_per_zone",
    "cooling_coefficient_of_performance",
    "oxygen_injection_max_total_mol_s",
    "oxygen_injection_power_w_per_mol_s",
    "fan_power_w_per_m3_s",
    "base_load_w",
    "battery_capacity_wh",
    "battery_max_charge_w",
    "battery_max_discharge_w",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "air_density_kg_m3",
    "air_specific_heat_j_kg_k",
}
_INITIAL_UTILITY_FIELDS = {
    "co2_sorbent_remaining_mol",
    "captured_co2_mol",
    "condensed_water_mol",
    "oxygen_store_mol",
    "battery_energy_wh",
    "actual_airflow_m3_s",
    "actual_scrubber_duty",
    "actual_condenser_duty",
    "external_heat_rejected_j",
    "external_heat_received_j",
}
_TIMELINE_FIELDS_V1 = {
    "start_step",
    "end_step",
    "generation_w",
    "loads",
    "command",
}
_TIMELINE_FIELDS_V2 = _TIMELINE_FIELDS_V1 | {"operating_mode"}
_OPERATING_MODES = {"occupied", "eva_transition", "contingency", "dormant"}
_ZONE_LOAD_FIELDS = {
    "co2_generation_mol_s",
    "o2_consumption_mol_s",
    "water_vapor_generation_mol_s",
    "sensible_heat_w",
}
_COMMAND_FIELDS = {
    "airflow_m3_s",
    "scrubber_duty",
    "condenser_duty",
    "cooling_removed_w",
    "oxygen_injection_mol_s",
}

_TOP_LEVEL_FIELDS_V3 = _TOP_LEVEL_FIELDS | {"air_network"}
_TOP_LEVEL_FIELDS_V4 = _TOP_LEVEL_FIELDS_V3 | {"sensor_model", "fault_profiles"}
_ZONE_FIELDS_V3 = _ZONE_FIELDS | {"geometry"}
_ZONE_GEOMETRY_FIELDS = {"center_m", "size_m"}
_EQUIPMENT_FIELDS_V3 = _EQUIPMENT_FIELDS - {
    "max_total_airflow_m3_s",
    "max_zone_airflow_m3_s",
    "airflow_slew_m3_s2",
    "fan_power_w_per_m3_s",
}
_INITIAL_UTILITY_FIELDS_V3 = (
    _INITIAL_UTILITY_FIELDS - {"actual_airflow_m3_s"}
) | {"actual_fan_speed_fraction", "actual_damper_position_by_id"}
_COMMAND_FIELDS_V3 = (_COMMAND_FIELDS - {"airflow_m3_s"}) | {
    "fan_speed_fraction",
    "damper_position_by_id",
}
_AIR_NETWORK_FIELDS = {
    "supply_plenum_position_m",
    "return_plenum_position_m",
    "fan",
    "shared_resistance",
    "branches",
}
_FAN_FIELDS = {
    "id",
    "rated_free_delivery_m3_s",
    "rated_shutoff_pressure_pa",
    "total_efficiency",
    "speed_slew_fraction_per_s",
    "position_m",
}
_SHARED_RESISTANCE_FIELDS = {
    "supply_trunk_pa_s2_m6",
    "return_trunk_pa_s2_m6",
    "filter_pa_s2_m6",
}
_BRANCH_FIELDS = {
    "zone_id",
    "damper_id",
    "open_supply_resistance_pa_s2_m6",
    "return_resistance_pa_s2_m6",
    "damper_leak_fraction",
    "damper_slew_fraction_per_s",
    "supply_diffuser_position_m",
    "return_grille_position_m",
    "damper_position_m",
    "duct_polyline_m",
}
_SENSOR_CHANNELS = {
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
}
_SENSOR_MODEL_FIELDS = {
    "random_seed",
    "primary_noise_amplitude",
    "secondary_noise_amplitude",
}
_FAN_DEGRADATION_FIELDS = {
    "id",
    "type",
    "start_step",
    "end_step",
    "start_multiplier",
    "end_multiplier",
}
_BRANCH_RESISTANCE_FIELDS = {
    "id",
    "type",
    "zone_id",
    "start_step",
    "end_step",
    "start_multiplier",
    "end_multiplier",
}


class ScenarioValidationError(ValueError):
    """Raised when Habitat V2 scenario input violates its contract."""


def _normalise_json(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScenarioValidationError(f"{path} must be finite")
        return value
    if isinstance(value, list):
        return [
            _normalise_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScenarioValidationError(f"{path} keys must be strings")
            normalised[key] = _normalise_json(item, path=f"{path}.{key}")
        return normalised
    raise ScenarioValidationError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], *, label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ScenarioValidationError(f"unknown {label} fields: {', '.join(unknown)}")
    missing = sorted(allowed - set(value))
    if missing:
        raise ScenarioValidationError(f"missing {label} fields: {', '.join(missing)}")


def _top_level_fields_for_schema(schema_version: str) -> set[str]:
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        return _TOP_LEVEL_FIELDS
    if schema_version == SCENARIO_SCHEMA_VERSION_V3:
        return _TOP_LEVEL_FIELDS_V3
    if schema_version == SCENARIO_SCHEMA_VERSION_V4:
        return _TOP_LEVEL_FIELDS_V4
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _zone_fields_for_schema(schema_version: str) -> set[str]:
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        return _ZONE_FIELDS
    if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
        return _ZONE_FIELDS_V3
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _equipment_fields_for_schema(schema_version: str) -> set[str]:
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        return _EQUIPMENT_FIELDS
    if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
        return _EQUIPMENT_FIELDS_V3
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _initial_utility_fields_for_schema(schema_version: str) -> set[str]:
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        return _INITIAL_UTILITY_FIELDS
    if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
        return _INITIAL_UTILITY_FIELDS_V3
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _command_fields_for_schema(schema_version: str) -> set[str]:
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        return _COMMAND_FIELDS
    if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
        return _COMMAND_FIELDS_V3
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _timeline_fields_for_schema(schema_version: str) -> set[str]:
    if schema_version == SCENARIO_SCHEMA_VERSION_V1:
        return _TIMELINE_FIELDS_V1
    if schema_version in {
        SCENARIO_SCHEMA_VERSION_V2,
        SCENARIO_SCHEMA_VERSION_V3,
        SCENARIO_SCHEMA_VERSION_V4,
    }:
        return _TIMELINE_FIELDS_V2
    raise ScenarioValidationError(
        "schema_version must be "
        f"{SCENARIO_SCHEMA_VERSION_V1!r}, {SCENARIO_SCHEMA_VERSION_V2!r}, "
        f"{SCENARIO_SCHEMA_VERSION_V3!r}, or {SCENARIO_SCHEMA_VERSION_V4!r}"
    )


def _trace_schema_for_scenario(schema_version: str) -> str:
    if schema_version == SCENARIO_SCHEMA_VERSION_V1:
        return TRACE_SCHEMA_VERSION_V1
    if schema_version == SCENARIO_SCHEMA_VERSION_V2:
        return TRACE_SCHEMA_VERSION_V2
    if schema_version == SCENARIO_SCHEMA_VERSION_V3:
        return TRACE_SCHEMA_VERSION_V3
    if schema_version == SCENARIO_SCHEMA_VERSION_V4:
        return TRACE_SCHEMA_VERSION_V4
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _equation_contract_for_scenario(schema_version: str) -> str:
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        return EQUATION_CONTRACT_REVISION
    if schema_version == SCENARIO_SCHEMA_VERSION_V3:
        return EQUATION_CONTRACT_REVISION_V2
    if schema_version == SCENARIO_SCHEMA_VERSION_V4:
        return EQUATION_CONTRACT_REVISION_V3
    raise ScenarioValidationError(f"unsupported scenario schema {schema_version!r}")


def _validate_nested_schema(scenario: Mapping[str, Any]) -> None:
    schema_version = str(scenario["schema_version"])
    for zone in scenario["zones"]:
        _reject_unknown_fields(
            zone, _zone_fields_for_schema(schema_version), label="zone"
        )
        _reject_unknown_fields(
            zone["initial"], _ZONE_INITIAL_FIELDS, label="zone initial state"
        )
        if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
            _reject_unknown_fields(
                zone["geometry"], _ZONE_GEOMETRY_FIELDS, label="zone geometry"
            )

    _reject_unknown_fields(
        scenario["equipment"],
        _equipment_fields_for_schema(schema_version),
        label="equipment",
    )
    _reject_unknown_fields(
        scenario["initial_utility"],
        _initial_utility_fields_for_schema(schema_version),
        label="initial utility",
    )

    if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
        network = scenario["air_network"]
        _reject_unknown_fields(network, _AIR_NETWORK_FIELDS, label="air network")
        _reject_unknown_fields(network["fan"], _FAN_FIELDS, label="air network fan")
        _reject_unknown_fields(
            network["shared_resistance"],
            _SHARED_RESISTANCE_FIELDS,
            label="air network shared resistance",
        )
        for branch in network["branches"]:
            _reject_unknown_fields(
                branch, _BRANCH_FIELDS, label="air network branch"
            )

    if schema_version == SCENARIO_SCHEMA_VERSION_V4:
        sensor_model = scenario["sensor_model"]
        _reject_unknown_fields(
            sensor_model, _SENSOR_MODEL_FIELDS, label="sensor model"
        )
        for head in ("primary_noise_amplitude", "secondary_noise_amplitude"):
            _reject_unknown_fields(
                sensor_model[head],
                _SENSOR_CHANNELS,
                label=f"sensor model {head}",
            )
        if not isinstance(scenario["fault_profiles"], list):
            raise ScenarioValidationError("fault_profiles must be an array")
        for profile in scenario["fault_profiles"]:
            if not isinstance(profile, Mapping):
                raise ScenarioValidationError("fault profile must be an object")
            profile_type = profile.get("type")
            if profile_type == "fan_speed_degradation":
                profile_fields = _FAN_DEGRADATION_FIELDS
                profile_label = "fan speed degradation profile"
            elif profile_type == "branch_resistance_increase":
                profile_fields = _BRANCH_RESISTANCE_FIELDS
                profile_label = "branch resistance increase profile"
            else:
                raise ScenarioValidationError(
                    f"unsupported fault profile type {profile_type!r}"
                )
            _reject_unknown_fields(
                profile,
                profile_fields,
                label=profile_label,
            )

    for segment in scenario["timeline"]:
        _reject_unknown_fields(
            segment,
            _timeline_fields_for_schema(schema_version),
            label="timeline segment",
        )
        for load in segment["loads"].values():
            _reject_unknown_fields(load, _ZONE_LOAD_FIELDS, label="zone load")
        _reject_unknown_fields(
            segment["command"],
            _command_fields_for_schema(schema_version),
            label="plant command",
        )


def _require_zone_keys(
    value: Mapping[str, Any], zone_ids: set[str], *, path: str
) -> None:
    if set(value) != zone_ids:
        raise ScenarioValidationError(
            f"zone topology mismatch at {path}: expected {sorted(zone_ids)!r}, "
            f"got {sorted(value)!r}"
        )


def _validate_topology(scenario: Mapping[str, Any]) -> None:
    schema_version = str(scenario["schema_version"])
    zones = scenario["zones"]
    zone_ids = {zone["id"] for zone in zones}

    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        expected_zone_ids = {"crew_cabin", "work_airlock"}
        if len(zones) != 2 or zone_ids != expected_zone_ids:
            raise ScenarioValidationError(
                "zone topology must contain exactly crew_cabin and work_airlock"
            )
        _require_zone_keys(
            scenario["initial_utility"]["actual_airflow_m3_s"],
            zone_ids,
            path="initial_utility.actual_airflow_m3_s",
        )
    else:
        if not 2 <= len(zones) <= 16 or len(zone_ids) != len(zones):
            raise ScenarioValidationError(
                "scenario-v3 zone topology must contain 2..16 unique zone ids"
            )
        if any(not isinstance(zone_id, str) or not zone_id.strip() for zone_id in zone_ids):
            raise ScenarioValidationError("scenario-v3 zone ids must be non-empty strings")
        branches = scenario["air_network"]["branches"]
        branch_zone_ids = [branch["zone_id"] for branch in branches]
        damper_ids = [branch["damper_id"] for branch in branches]
        if len(branches) != len(zones) or set(branch_zone_ids) != zone_ids:
            raise ScenarioValidationError(
                "scenario-v3 requires exactly one air-network branch per zone"
            )
        if len(set(damper_ids)) != len(damper_ids):
            raise ScenarioValidationError(
                "scenario-v3 air-network damper ids must be unique"
            )
        initial_dampers = scenario["initial_utility"][
            "actual_damper_position_by_id"
        ]
        if set(initial_dampers) != set(damper_ids):
            raise ScenarioValidationError(
                "initial damper state must match air-network damper ids"
            )

    for index, segment in enumerate(scenario["timeline"]):
        _require_zone_keys(segment["loads"], zone_ids, path=f"timeline[{index}].loads")
        command = segment["command"]
        for field in ("cooling_removed_w", "oxygen_injection_mol_s"):
            _require_zone_keys(
                command[field], zone_ids, path=f"timeline[{index}].command.{field}"
            )
        if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
            _require_zone_keys(
                command["airflow_m3_s"],
                zone_ids,
                path=f"timeline[{index}].command.airflow_m3_s",
            )
        else:
            expected_damper_ids = {
                branch["damper_id"] for branch in scenario["air_network"]["branches"]
            }
            if set(command["damper_position_by_id"]) != expected_damper_ids:
                raise ScenarioValidationError(
                    f"damper topology mismatch at timeline[{index}].command"
                )


def _invalid(path: str, message: str) -> None:
    raise ScenarioValidationError(f"invalid scenario value at {path}: {message}")


def _number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(path, "must be a number")
    return float(value)


def _positive(value: Any, *, path: str) -> float:
    number = _number(value, path=path)
    if number <= 0.0:
        _invalid(path, "must be greater than zero")
    return number


def _nonnegative(value: Any, *, path: str) -> float:
    number = _number(value, path=path)
    if number < 0.0:
        _invalid(path, "must be non-negative")
    return number


def _fraction(value: Any, *, path: str, maximum: float = 1.0) -> float:
    number = _number(value, path=path)
    if not 0.0 <= number <= maximum:
        _invalid(path, f"must be between 0 and {maximum}")
    return number


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _invalid(path, "must be a positive integer")
    return value


def _vector3(value: Any, *, path: str, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 3:
        _invalid(path, "must be an array of exactly three numbers")
    numbers = tuple(_number(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if positive and any(number <= 0.0 for number in numbers):
        _invalid(path, "all dimensions must be greater than zero")
    return numbers


def _validate_values(scenario: Mapping[str, Any]) -> None:
    schema_version = str(scenario["schema_version"])
    if not isinstance(scenario["name"], str) or not scenario["name"].strip():
        _invalid("name", "must be a non-empty string")
    _positive(scenario["dt_seconds"], path="dt_seconds")
    steps = _positive_int(scenario["steps"], path="steps")

    if schema_version == SCENARIO_SCHEMA_VERSION_V4:
        seen_profile_ids: set[str] = set()
        fan_intervals: list[tuple[int, int]] = []
        branch_intervals_by_zone: dict[str, list[tuple[int, int]]] = {}
        declared_zone_ids = {str(zone["id"]) for zone in scenario["zones"]}
        for index, profile in enumerate(scenario["fault_profiles"]):
            prefix = f"fault_profiles[{index}]"
            profile_id = profile["id"]
            if not isinstance(profile_id, str) or not profile_id.strip():
                _invalid(f"{prefix}.id", "must be a non-empty string")
            if profile_id in seen_profile_ids:
                _invalid(f"{prefix}.id", "must be unique")
            seen_profile_ids.add(profile_id)
            start_step = profile["start_step"]
            end_step = profile["end_step"]
            if isinstance(start_step, bool) or not isinstance(start_step, int):
                _invalid(f"{prefix}.start_step", "must be an integer")
            if isinstance(end_step, bool) or not isinstance(end_step, int):
                _invalid(f"{prefix}.end_step", "must be an integer")
            if not 1 <= start_step < end_step <= steps + 1:
                _invalid(prefix, "interval must satisfy 1 <= start < end <= steps + 1")
            profile_type = profile["type"]
            for field in ("start_multiplier", "end_multiplier"):
                multiplier = _number(profile[field], path=f"{prefix}.{field}")
                if profile_type == "fan_speed_degradation":
                    if not 0.0 < multiplier <= 1.0:
                        _invalid(f"{prefix}.{field}", "must be in (0, 1]")
                elif multiplier < 1.0:
                    _invalid(f"{prefix}.{field}", "must be at least 1")

            if profile_type == "fan_speed_degradation":
                for prior_start, prior_end in fan_intervals:
                    if max(start_step, prior_start) < min(end_step, prior_end):
                        _invalid(prefix, "fan degradation profiles may not overlap")
                fan_intervals.append((start_step, end_step))
                continue

            zone_id = profile["zone_id"]
            if (
                not isinstance(zone_id, str)
                or not zone_id.strip()
                or zone_id not in declared_zone_ids
            ):
                _invalid(f"{prefix}.zone_id", "must identify a declared zone")
            branch_intervals = branch_intervals_by_zone.setdefault(zone_id, [])
            for prior_start, prior_end in branch_intervals:
                if max(start_step, prior_start) < min(end_step, prior_end):
                    _invalid(
                        prefix,
                        f"branch resistance profiles for {zone_id} may not overlap",
                    )
            branch_intervals.append((start_step, end_step))

    for zone_index, zone in enumerate(scenario["zones"]):
        prefix = f"zones[{zone_index}]"
        _positive(zone["volume_m3"], path=f"{prefix}.volume_m3")
        _positive(
            zone["thermal_capacity_j_per_k"],
            path=f"{prefix}.thermal_capacity_j_per_k",
        )
        _nonnegative(
            zone["passive_thermal_conductance_w_per_k"],
            path=f"{prefix}.passive_thermal_conductance_w_per_k",
        )
        sink_temperature = _number(
            zone["sink_temperature_k"], path=f"{prefix}.sink_temperature_k"
        )
        if not 150.0 <= sink_temperature <= 330.0:
            _invalid(f"{prefix}.sink_temperature_k", "outside supported 150..330 K")

        initial = zone["initial"]
        temperature = _number(
            initial["temperature_k"], path=f"{prefix}.initial.temperature_k"
        )
        if not 250.0 <= temperature <= 330.0:
            _invalid(f"{prefix}.initial.temperature_k", "outside supported 250..330 K")
        pressure = _number(initial["pressure_pa"], path=f"{prefix}.initial.pressure_pa")
        if not 34_500.0 <= pressure <= 103_000.0:
            _invalid(
                f"{prefix}.initial.pressure_pa",
                "outside the declared 34.5..103 kPa initial range",
            )
        co2_ppm = _number(initial["co2_ppm"], path=f"{prefix}.initial.co2_ppm")
        if not 0.0 <= co2_ppm < 1_000_000.0:
            _invalid(f"{prefix}.initial.co2_ppm", "must be in [0, 1000000)")
        o2_fraction = _number(
            initial["o2_mole_fraction"],
            path=f"{prefix}.initial.o2_mole_fraction",
        )
        if not 0.0 < o2_fraction <= 0.70:
            _invalid(
                f"{prefix}.initial.o2_mole_fraction",
                "must be in (0, 0.70] to retain at least 30% diluent",
            )
        _fraction(
            initial["relative_humidity"],
            path=f"{prefix}.initial.relative_humidity",
        )
        if schema_version in {SCENARIO_SCHEMA_VERSION_V3, SCENARIO_SCHEMA_VERSION_V4}:
            _vector3(zone["geometry"]["center_m"], path=f"{prefix}.geometry.center_m")
            _vector3(
                zone["geometry"]["size_m"],
                path=f"{prefix}.geometry.size_m",
                positive=True,
            )

    equipment = scenario["equipment"]
    positive_equipment_fields = {
        "scrubber_duty_slew_per_s",
        "condenser_duty_slew_per_s",
        "cooling_coefficient_of_performance",
        "battery_capacity_wh",
        "air_density_kg_m3",
        "air_specific_heat_j_kg_k",
    }
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        positive_equipment_fields |= {
            "max_total_airflow_m3_s",
            "max_zone_airflow_m3_s",
            "airflow_slew_m3_s2",
        }
    for field in sorted(positive_equipment_fields):
        _positive(equipment[field], path=f"equipment.{field}")

    efficiency_fields = {"battery_charge_efficiency", "battery_discharge_efficiency"}
    for field in sorted(
        _equipment_fields_for_schema(schema_version)
        - positive_equipment_fields
        - efficiency_fields
    ):
        _nonnegative(equipment[field], path=f"equipment.{field}")
    for field in sorted(efficiency_fields):
        efficiency = _number(equipment[field], path=f"equipment.{field}")
        if not 0.0 < efficiency <= 1.0:
            _invalid(f"equipment.{field}", "must be in (0, 1]")
    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        if equipment["max_zone_airflow_m3_s"] > equipment["max_total_airflow_m3_s"]:
            _invalid(
                "equipment.max_zone_airflow_m3_s",
                "cannot exceed max_total_airflow_m3_s",
            )

    utility = scenario["initial_utility"]
    utility_inventory_fields = {
        "co2_sorbent_remaining_mol",
        "captured_co2_mol",
        "condensed_water_mol",
        "oxygen_store_mol",
        "battery_energy_wh",
        "external_heat_rejected_j",
        "external_heat_received_j",
    }
    for field in sorted(utility_inventory_fields):
        _nonnegative(utility[field], path=f"initial_utility.{field}")
    for field in ("actual_scrubber_duty", "actual_condenser_duty"):
        _fraction(utility[field], path=f"initial_utility.{field}")
    if utility["battery_energy_wh"] > equipment["battery_capacity_wh"]:
        _invalid("initial_utility.battery_energy_wh", "exceeds battery capacity")
    if utility["co2_sorbent_remaining_mol"] > equipment["scrubber_capacity_mol"]:
        _invalid(
            "initial_utility.co2_sorbent_remaining_mol",
            "exceeds scrubber capacity",
        )

    if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
        initial_flows = utility["actual_airflow_m3_s"]
        for zone_id, flow in initial_flows.items():
            flow_value = _nonnegative(
                flow, path=f"initial_utility.actual_airflow_m3_s.{zone_id}"
            )
            if flow_value > equipment["max_zone_airflow_m3_s"]:
                _invalid(
                    f"initial_utility.actual_airflow_m3_s.{zone_id}",
                    "exceeds per-zone airflow capacity",
                )
        if sum(initial_flows.values()) > equipment["max_total_airflow_m3_s"]:
            _invalid(
                "initial_utility.actual_airflow_m3_s",
                "exceeds total airflow capacity",
            )
    else:
        _fraction(
            utility["actual_fan_speed_fraction"],
            path="initial_utility.actual_fan_speed_fraction",
        )
        for damper_id, position in utility["actual_damper_position_by_id"].items():
            _fraction(
                position,
                path=f"initial_utility.actual_damper_position_by_id.{damper_id}",
            )

        network = scenario["air_network"]
        _vector3(
            network["supply_plenum_position_m"],
            path="air_network.supply_plenum_position_m",
        )
        _vector3(
            network["return_plenum_position_m"],
            path="air_network.return_plenum_position_m",
        )
        fan = network["fan"]
        if not isinstance(fan["id"], str) or not fan["id"].strip():
            _invalid("air_network.fan.id", "must be a non-empty string")
        for field in (
            "rated_free_delivery_m3_s",
            "rated_shutoff_pressure_pa",
            "speed_slew_fraction_per_s",
        ):
            _positive(fan[field], path=f"air_network.fan.{field}")
        fan_efficiency = _number(
            fan["total_efficiency"], path="air_network.fan.total_efficiency"
        )
        if not 0.0 < fan_efficiency <= 1.0:
            _invalid("air_network.fan.total_efficiency", "must be in (0, 1]")
        _vector3(fan["position_m"], path="air_network.fan.position_m")

        shared_resistance = network["shared_resistance"]
        for field in sorted(_SHARED_RESISTANCE_FIELDS):
            _nonnegative(
                shared_resistance[field],
                path=f"air_network.shared_resistance.{field}",
            )
        if sum(float(shared_resistance[field]) for field in _SHARED_RESISTANCE_FIELDS) <= 0.0:
            _invalid(
                "air_network.shared_resistance",
                "at least one shared resistance must be greater than zero",
            )

        component_ids = {str(fan["id"])}
        for branch_index, branch in enumerate(network["branches"]):
            branch_prefix = f"air_network.branches[{branch_index}]"
            for field in ("zone_id", "damper_id"):
                if not isinstance(branch[field], str) or not branch[field].strip():
                    _invalid(f"{branch_prefix}.{field}", "must be a non-empty string")
            if branch["damper_id"] in component_ids:
                _invalid(f"{branch_prefix}.damper_id", "component ids must be unique")
            component_ids.add(branch["damper_id"])
            for field in (
                "open_supply_resistance_pa_s2_m6",
                "return_resistance_pa_s2_m6",
                "damper_slew_fraction_per_s",
            ):
                _positive(branch[field], path=f"{branch_prefix}.{field}")
            leak_fraction = _number(
                branch["damper_leak_fraction"],
                path=f"{branch_prefix}.damper_leak_fraction",
            )
            if not 0.0 < leak_fraction <= 1.0:
                _invalid(
                    f"{branch_prefix}.damper_leak_fraction", "must be in (0, 1]"
                )
            for field in (
                "supply_diffuser_position_m",
                "return_grille_position_m",
                "damper_position_m",
            ):
                _vector3(branch[field], path=f"{branch_prefix}.{field}")
            polyline = branch["duct_polyline_m"]
            if not isinstance(polyline, list) or len(polyline) < 2:
                _invalid(
                    f"{branch_prefix}.duct_polyline_m",
                    "must contain at least two 3D points",
                )
            for point_index, point in enumerate(polyline):
                _vector3(
                    point,
                    path=f"{branch_prefix}.duct_polyline_m[{point_index}]",
                )

    if schema_version == SCENARIO_SCHEMA_VERSION_V4:
        sensor_model = scenario["sensor_model"]
        seed = sensor_model["random_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            _invalid("sensor_model.random_seed", "must be an integer")
        for head in ("primary_noise_amplitude", "secondary_noise_amplitude"):
            for channel in sorted(_SENSOR_CHANNELS):
                _nonnegative(
                    sensor_model[head][channel],
                    path=f"sensor_model.{head}.{channel}",
                )

    timeline = scenario["timeline"]
    if not timeline:
        _invalid("timeline", "must contain at least one segment")
    expected_start = 0
    for index, segment in enumerate(timeline):
        prefix = f"timeline[{index}]"
        start = segment["start_step"]
        end = segment["end_step"]
        if isinstance(start, bool) or not isinstance(start, int):
            _invalid(f"{prefix}.start_step", "must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            _invalid(f"{prefix}.end_step", "must be an integer")
        if start != expected_start or not start < end <= steps:
            _invalid(prefix, "segments must cover steps contiguously from 0 to steps")
        expected_start = end
        _nonnegative(segment["generation_w"], path=f"{prefix}.generation_w")
        if schema_version in {
            SCENARIO_SCHEMA_VERSION_V2,
            SCENARIO_SCHEMA_VERSION_V3,
            SCENARIO_SCHEMA_VERSION_V4,
        }:
            mode = segment["operating_mode"]
            if not isinstance(mode, str) or mode not in _OPERATING_MODES:
                _invalid(
                    f"{prefix}.operating_mode",
                    "must be one of occupied, eva_transition, contingency, dormant",
                )

        for zone_id, load in segment["loads"].items():
            for field in _ZONE_LOAD_FIELDS:
                _nonnegative(load[field], path=f"{prefix}.loads.{zone_id}.{field}")

        command = segment["command"]
        for field in ("scrubber_duty", "condenser_duty"):
            _fraction(command[field], path=f"{prefix}.command.{field}")
        if schema_version in {SCENARIO_SCHEMA_VERSION_V1, SCENARIO_SCHEMA_VERSION_V2}:
            flows = command["airflow_m3_s"]
            for zone_id, flow in flows.items():
                flow_value = _nonnegative(
                    flow, path=f"{prefix}.command.airflow_m3_s.{zone_id}"
                )
                if flow_value > equipment["max_zone_airflow_m3_s"]:
                    _invalid(
                        f"{prefix}.command.airflow_m3_s.{zone_id}",
                        "exceeds per-zone airflow capacity",
                    )
            if sum(flows.values()) > equipment["max_total_airflow_m3_s"]:
                _invalid(
                    f"{prefix}.command.airflow_m3_s",
                    "exceeds total airflow capacity",
                )
        else:
            _fraction(
                command["fan_speed_fraction"],
                path=f"{prefix}.command.fan_speed_fraction",
            )
            for damper_id, position in command["damper_position_by_id"].items():
                _fraction(
                    position,
                    path=f"{prefix}.command.damper_position_by_id.{damper_id}",
                )
        cooling = command["cooling_removed_w"]
        for zone_id, value in cooling.items():
            cooling_value = _nonnegative(
                value, path=f"{prefix}.command.cooling_removed_w.{zone_id}"
            )
            if cooling_value > equipment["cooling_max_thermal_w_per_zone"]:
                _invalid(
                    f"{prefix}.command.cooling_removed_w.{zone_id}",
                    "exceeds per-zone cooling capacity",
                )
        oxygen = command["oxygen_injection_mol_s"]
        for zone_id, value in oxygen.items():
            _nonnegative(
                value, path=f"{prefix}.command.oxygen_injection_mol_s.{zone_id}"
            )
        if sum(oxygen.values()) > equipment["oxygen_injection_max_total_mol_s"]:
            _invalid(
                f"{prefix}.command.oxygen_injection_mol_s",
                "exceeds total oxygen-injection capacity",
            )
    if expected_start != steps:
        _invalid("timeline", "segments must end at steps")


def derive_run_id(
    *,
    scenario_sha256: str,
    scenario_schema_version: str = SCENARIO_SCHEMA_VERSION,
    trace_schema_version: str = TRACE_SCHEMA_VERSION,
    equation_contract_revision: str = EQUATION_CONTRACT_REVISION,
) -> str:
    lineage_payload = {
        "equation_contract_revision": equation_contract_revision,
        "scenario_schema_version": scenario_schema_version,
        "scenario_sha256": scenario_sha256,
        "trace_schema_version": trace_schema_version,
    }
    return hashlib.sha256(_canonical_bytes(lineage_payload)).hexdigest()


@dataclass(frozen=True)
class Scenario:
    data: Mapping[str, Any]
    canonical_bytes: bytes
    scenario_sha256: str
    scenario_schema_version: str
    trace_schema_version: str
    equation_contract_revision: str
    run_id: str

    def validate_contract_identities(self) -> None:
        try:
            current_canonical_bytes = _canonical_bytes(self.data)
        except (TypeError, ValueError) as error:
            raise ScenarioValidationError(
                "scenario data cannot be canonicalized"
            ) from error
        if current_canonical_bytes != self.canonical_bytes:
            raise ScenarioValidationError(
                "scenario data does not match stored canonical bytes"
            )
        expected_scenario_sha256 = hashlib.sha256(self.canonical_bytes).hexdigest()
        if self.scenario_sha256 != expected_scenario_sha256:
            raise ScenarioValidationError(
                "scenario digest does not match stored canonical bytes"
            )
        if self.data.get("schema_version") != self.scenario_schema_version:
            raise ScenarioValidationError(
                "scenario schema identity does not match parsed scenario data"
            )
        expected_trace_schema = _trace_schema_for_scenario(self.scenario_schema_version)
        if self.trace_schema_version != expected_trace_schema:
            raise ScenarioValidationError(
                "trace schema does not match scenario schema identity"
            )
        expected_equation_contract = _equation_contract_for_scenario(
            self.scenario_schema_version
        )
        if self.equation_contract_revision != expected_equation_contract:
            raise ScenarioValidationError("unsupported equation contract identity")
        expected_run_id = derive_run_id(
            scenario_sha256=self.scenario_sha256,
            scenario_schema_version=self.scenario_schema_version,
            trace_schema_version=self.trace_schema_version,
            equation_contract_revision=self.equation_contract_revision,
        )
        if self.run_id != expected_run_id:
            raise ScenarioValidationError("run_id does not match scenario identities")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Scenario":
        if not isinstance(mapping, Mapping):
            raise ScenarioValidationError("scenario must be a JSON object")

        normalised = _normalise_json(mapping)
        schema_version = normalised.get("schema_version")
        if not isinstance(schema_version, str):
            raise ScenarioValidationError("schema_version must be a string")
        _timeline_fields_for_schema(schema_version)
        top_level_fields = _top_level_fields_for_schema(schema_version)
        unknown_fields = sorted(set(normalised) - top_level_fields)
        if unknown_fields:
            raise ScenarioValidationError(
                f"unknown top-level fields: {', '.join(unknown_fields)}"
            )
        missing_fields = sorted(top_level_fields - set(normalised))
        if missing_fields:
            raise ScenarioValidationError(
                f"missing top-level fields: {', '.join(missing_fields)}"
            )

        _validate_nested_schema(normalised)
        _validate_topology(normalised)
        _validate_values(normalised)

        canonical = _canonical_bytes(normalised)
        scenario_sha256 = hashlib.sha256(canonical).hexdigest()
        trace_schema_version = _trace_schema_for_scenario(schema_version)
        equation_contract_revision = _equation_contract_for_scenario(schema_version)
        run_id = derive_run_id(
            scenario_sha256=scenario_sha256,
            scenario_schema_version=schema_version,
            trace_schema_version=trace_schema_version,
            equation_contract_revision=equation_contract_revision,
        )
        return cls(
            data=normalised,
            canonical_bytes=canonical,
            scenario_sha256=scenario_sha256,
            scenario_schema_version=schema_version,
            trace_schema_version=trace_schema_version,
            equation_contract_revision=equation_contract_revision,
            run_id=run_id,
        )
