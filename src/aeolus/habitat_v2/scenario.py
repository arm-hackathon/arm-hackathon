from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

SCENARIO_SCHEMA_VERSION = "aeolus_habitat_v2_scenario_v1"
TRACE_SCHEMA_VERSION = "aeolus_habitat_v2_trace_v1"
EQUATION_CONTRACT_REVISION = "aeolus_habitat_v2_equations_v1"

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
_TIMELINE_FIELDS = {
    "start_step",
    "end_step",
    "generation_w",
    "loads",
    "command",
}
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


def _validate_nested_schema(scenario: Mapping[str, Any]) -> None:
    for zone in scenario["zones"]:
        _reject_unknown_fields(zone, _ZONE_FIELDS, label="zone")
        _reject_unknown_fields(
            zone["initial"], _ZONE_INITIAL_FIELDS, label="zone initial state"
        )

    _reject_unknown_fields(scenario["equipment"], _EQUIPMENT_FIELDS, label="equipment")
    _reject_unknown_fields(
        scenario["initial_utility"],
        _INITIAL_UTILITY_FIELDS,
        label="initial utility",
    )

    for segment in scenario["timeline"]:
        _reject_unknown_fields(segment, _TIMELINE_FIELDS, label="timeline segment")
        for load in segment["loads"].values():
            _reject_unknown_fields(load, _ZONE_LOAD_FIELDS, label="zone load")
        _reject_unknown_fields(
            segment["command"], _COMMAND_FIELDS, label="plant command"
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
    expected_zone_ids = {"crew_cabin", "work_airlock"}
    zones = scenario["zones"]
    zone_ids = {zone["id"] for zone in zones}
    if len(zones) != 2 or zone_ids != expected_zone_ids:
        raise ScenarioValidationError(
            "zone topology must contain exactly crew_cabin and work_airlock"
        )

    _require_zone_keys(
        scenario["initial_utility"]["actual_airflow_m3_s"],
        zone_ids,
        path="initial_utility.actual_airflow_m3_s",
    )
    for index, segment in enumerate(scenario["timeline"]):
        _require_zone_keys(segment["loads"], zone_ids, path=f"timeline[{index}].loads")
        command = segment["command"]
        for field in (
            "airflow_m3_s",
            "cooling_removed_w",
            "oxygen_injection_mol_s",
        ):
            _require_zone_keys(
                command[field], zone_ids, path=f"timeline[{index}].command.{field}"
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


def _validate_values(scenario: Mapping[str, Any]) -> None:
    if not isinstance(scenario["name"], str) or not scenario["name"].strip():
        _invalid("name", "must be a non-empty string")
    _positive(scenario["dt_seconds"], path="dt_seconds")
    steps = _positive_int(scenario["steps"], path="steps")

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

    equipment = scenario["equipment"]
    for field in (
        "max_total_airflow_m3_s",
        "max_zone_airflow_m3_s",
        "airflow_slew_m3_s2",
        "scrubber_duty_slew_per_s",
        "condenser_duty_slew_per_s",
        "cooling_coefficient_of_performance",
        "battery_capacity_wh",
        "air_density_kg_m3",
        "air_specific_heat_j_kg_k",
    ):
        _positive(equipment[field], path=f"equipment.{field}")
    for field in _EQUIPMENT_FIELDS - {
        "max_total_airflow_m3_s",
        "max_zone_airflow_m3_s",
        "airflow_slew_m3_s2",
        "scrubber_duty_slew_per_s",
        "condenser_duty_slew_per_s",
        "cooling_coefficient_of_performance",
        "battery_capacity_wh",
        "battery_charge_efficiency",
        "battery_discharge_efficiency",
        "air_density_kg_m3",
        "air_specific_heat_j_kg_k",
    }:
        _nonnegative(equipment[field], path=f"equipment.{field}")
    for field in ("battery_charge_efficiency", "battery_discharge_efficiency"):
        efficiency = _number(equipment[field], path=f"equipment.{field}")
        if not 0.0 < efficiency <= 1.0:
            _invalid(f"equipment.{field}", "must be in (0, 1]")
    if equipment["max_zone_airflow_m3_s"] > equipment["max_total_airflow_m3_s"]:
        _invalid(
            "equipment.max_zone_airflow_m3_s",
            "cannot exceed max_total_airflow_m3_s",
        )

    utility = scenario["initial_utility"]
    for field in _INITIAL_UTILITY_FIELDS - {
        "actual_airflow_m3_s",
        "actual_scrubber_duty",
        "actual_condenser_duty",
    }:
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

        for zone_id, load in segment["loads"].items():
            for field in _ZONE_LOAD_FIELDS:
                _nonnegative(load[field], path=f"{prefix}.loads.{zone_id}.{field}")

        command = segment["command"]
        for field in ("scrubber_duty", "condenser_duty"):
            _fraction(command[field], path=f"{prefix}.command.{field}")
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
    run_id: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Scenario":
        if not isinstance(mapping, Mapping):
            raise ScenarioValidationError("scenario must be a JSON object")

        normalised = _normalise_json(mapping)
        unknown_fields = sorted(set(normalised) - _TOP_LEVEL_FIELDS)
        if unknown_fields:
            raise ScenarioValidationError(
                f"unknown top-level fields: {', '.join(unknown_fields)}"
            )
        missing_fields = sorted(_TOP_LEVEL_FIELDS - set(normalised))
        if missing_fields:
            raise ScenarioValidationError(
                f"missing top-level fields: {', '.join(missing_fields)}"
            )

        schema_version = normalised.get("schema_version")
        if schema_version != SCENARIO_SCHEMA_VERSION:
            raise ScenarioValidationError(
                f"schema_version must be {SCENARIO_SCHEMA_VERSION!r}"
            )
        _validate_nested_schema(normalised)
        _validate_topology(normalised)
        _validate_values(normalised)

        canonical = _canonical_bytes(normalised)
        scenario_sha256 = hashlib.sha256(canonical).hexdigest()
        run_id = derive_run_id(scenario_sha256=scenario_sha256)
        return cls(
            data=normalised,
            canonical_bytes=canonical,
            scenario_sha256=scenario_sha256,
            run_id=run_id,
        )
