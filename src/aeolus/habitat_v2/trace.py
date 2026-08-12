from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .scenario import (
    Scenario,
    TRACE_SCHEMA_VERSION_V1,
    TRACE_SCHEMA_VERSION_V2,
    derive_run_id,
)

_ZONE_IDS = {"crew_cabin", "work_airlock"}
_TELEMETRY_FIELDS = {
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
}
_ACTION_FIELDS = {
    "airflow_m3_s",
    "scrubber_duty",
    "condenser_duty",
    "cooling_removed_w",
    "oxygen_injection_mol_s",
}
_RESOURCE_FIELDS = {
    "co2_sorbent_remaining_mol",
    "oxygen_store_mol",
    "battery_energy_wh",
    "captured_co2_mol",
    "condensed_water_mol",
    "external_heat_received_j",
    "external_heat_rejected_j",
}
_LOAD_FIELDS = {
    "co2_generation_mol_s",
    "o2_consumption_mol_s",
    "water_vapor_generation_mol_s",
    "sensible_heat_w",
}
_ACCOUNTING_FIELDS = {
    "species_sources",
    "species_accounting",
    "recirculation",
    "oxygen_injected_mol",
    "passive_condensation_mol",
    "thermal",
    "electrical",
}
_INVARIANT_FIELDS = {"passed"}
_TRACE_FIELDS_V1 = {
    "schema_version",
    "lineage",
    "step",
    "time_s",
    "telemetry",
    "commanded_action",
    "actual_action",
    "resource_state",
    "realised_loads",
    "accounting_receipt",
    "invariant_status",
}
_TRACE_FIELDS_V2 = _TRACE_FIELDS_V1 | {"applied_operating_mode"}
_LINEAGE_FIELDS = {
    "run_id",
    "scenario_sha256",
    "scenario_schema_version",
    "trace_schema_version",
    "equation_contract_revision",
}


class TraceValidationError(ValueError):
    """Raised when trace bytes do not satisfy the frozen lineage contract."""


def _trace_fields_for_scenario(scenario: Scenario) -> set[str]:
    if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V1:
        return _TRACE_FIELDS_V1
    if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V2:
        return _TRACE_FIELDS_V2
    raise TraceValidationError("unsupported trace schema in parsed scenario")


def _reject_json_constant(value: str) -> None:
    raise TraceValidationError(f"non-finite JSON constant: {value}")


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise TraceValidationError(
            f"invalid {label} fields; unknown={unknown}, missing={missing}"
        )


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceValidationError(f"{label} must be finite numeric data")
    number = float(value)
    if not math.isfinite(number):
        raise TraceValidationError(f"{label} must be finite numeric data")
    return number


def _validate_numeric_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _validate_numeric_tree(nested, label=f"{label} {key}")
        return
    _finite_number(value, label=label)


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(f"{label} must be an object")
    return value


def _validate_action(value: Any, *, label: str) -> None:
    action = _as_mapping(value, label=label)
    _exact_fields(action, _ACTION_FIELDS, label=label)
    for field in ("scrubber_duty", "condenser_duty"):
        _finite_number(action[field], label=f"{label} {field}")
    for field in (
        "airflow_m3_s",
        "cooling_removed_w",
        "oxygen_injection_mol_s",
    ):
        zone_values = _as_mapping(action[field], label=f"{label} {field}")
        _exact_fields(zone_values, _ZONE_IDS, label=f"{label} {field}")
        _validate_numeric_tree(zone_values, label=f"{label} {field}")


def _validate_loads(value: Any, *, label: str) -> None:
    loads = _as_mapping(value, label=label)
    _exact_fields(loads, _ZONE_IDS, label=label)
    for zone_id in sorted(_ZONE_IDS):
        zone_loads = _as_mapping(loads[zone_id], label=f"{label} {zone_id}")
        _exact_fields(zone_loads, _LOAD_FIELDS, label=f"{label} {zone_id}")
        _validate_numeric_tree(zone_loads, label=f"{label} {zone_id}")


def _timeline_segment_for_step(scenario: Scenario, *, step: int) -> Mapping[str, Any]:
    for segment in scenario.data["timeline"]:
        if int(segment["start_step"]) <= step < int(segment["end_step"]):
            return segment
    raise TraceValidationError(f"scenario timeline has no segment for step {step}")


def validate_trace_bytes(
    data: bytes, *, scenario: Scenario
) -> tuple[Mapping[str, Any], ...]:
    if not data:
        raise TraceValidationError("trace is empty")
    if not data.endswith(b"\n"):
        raise TraceValidationError("trace must end with a newline")

    encoded_lines = data.splitlines()
    expected_row_count = int(scenario.data["steps"]) + 1
    if len(encoded_lines) != expected_row_count:
        raise TraceValidationError(
            f"expected {expected_row_count} rows, got {len(encoded_lines)}"
        )

    rows: list[Mapping[str, Any]] = []
    expected_lineage: Mapping[str, Any] | None = None
    for index, encoded_line in enumerate(encoded_lines):
        try:
            row = json.loads(
                encoded_line.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TraceValidationError(f"invalid JSON at row {index}") from error
        if not isinstance(row, Mapping):
            raise TraceValidationError(f"row {index} must be an object")
        _exact_fields(row, _trace_fields_for_scenario(scenario), label=f"row {index}")
        if row["schema_version"] != scenario.trace_schema_version:
            raise TraceValidationError(f"unsupported trace schema at row {index}")
        if isinstance(row["step"], bool) or not isinstance(row["step"], int):
            raise TraceValidationError(f"step at row {index} must be an integer")
        if row["step"] != index:
            raise TraceValidationError(f"non-consecutive step at row {index}")
        expected_time_s = index * float(scenario.data["dt_seconds"])
        actual_time_s = _finite_number(row["time_s"], label=f"time_s at row {index}")
        if actual_time_s != expected_time_s:
            raise TraceValidationError(f"unexpected time_s at row {index}")

        if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V2:
            applied_mode = row["applied_operating_mode"]
            if index == 0 and applied_mode is not None:
                raise TraceValidationError("applied operating mode row 0 must be null")
            if index > 0 and not isinstance(applied_mode, str):
                raise TraceValidationError(
                    f"applied operating mode row {index} must be a string"
                )

        telemetry = row["telemetry"]
        if not isinstance(telemetry, Mapping):
            raise TraceValidationError(f"telemetry at row {index} must be an object")
        _exact_fields(telemetry, _ZONE_IDS, label=f"telemetry row {index}")
        for zone_id in sorted(_ZONE_IDS):
            zone_telemetry = telemetry[zone_id]
            if not isinstance(zone_telemetry, Mapping):
                raise TraceValidationError(
                    f"telemetry row {index} {zone_id} must be an object"
                )
            _exact_fields(
                zone_telemetry,
                _TELEMETRY_FIELDS,
                label=f"telemetry row {index} {zone_id}",
            )
            _validate_numeric_tree(
                zone_telemetry, label=f"telemetry row {index} {zone_id}"
            )

        if index == 0:
            if row["commanded_action"] is not None:
                raise TraceValidationError("commanded action row 0 must be null")
            if row["realised_loads"] is not None:
                raise TraceValidationError("loads row 0 must be null")
            if row["accounting_receipt"] is not None:
                raise TraceValidationError("accounting receipt row 0 must be null")
        else:
            _validate_action(
                row["commanded_action"], label=f"commanded action row {index}"
            )
            _validate_loads(row["realised_loads"], label=f"loads row {index}")
            segment = _timeline_segment_for_step(scenario, step=index - 1)
            if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V2 and (
                row["applied_operating_mode"] != segment["operating_mode"]
            ):
                raise TraceValidationError(
                    f"applied operating mode row {index} does not match scenario timeline"
                )
            if row["commanded_action"] != segment["command"]:
                raise TraceValidationError(
                    f"commanded action row {index} does not match scenario timeline"
                )
            if row["realised_loads"] != segment["loads"]:
                raise TraceValidationError(
                    f"loads row {index} does not match scenario timeline"
                )
            accounting_receipt = _as_mapping(
                row["accounting_receipt"],
                label=f"accounting receipt row {index}",
            )
            _exact_fields(
                accounting_receipt,
                _ACCOUNTING_FIELDS,
                label=f"accounting receipt row {index}",
            )
            _validate_numeric_tree(
                accounting_receipt, label=f"accounting receipt row {index}"
            )

        _validate_action(row["actual_action"], label=f"actual action row {index}")
        resource_state = _as_mapping(
            row["resource_state"], label=f"resource state row {index}"
        )
        _exact_fields(
            resource_state, _RESOURCE_FIELDS, label=f"resource state row {index}"
        )
        _validate_numeric_tree(resource_state, label=f"resource state row {index}")
        invariant_status = _as_mapping(
            row["invariant_status"], label=f"invariant status row {index}"
        )
        _exact_fields(
            invariant_status,
            _INVARIANT_FIELDS,
            label=f"invariant status row {index}",
        )
        if invariant_status["passed"] is not True:
            raise TraceValidationError(f"invariant status failed at row {index}")

        lineage = row["lineage"]
        if not isinstance(lineage, Mapping):
            raise TraceValidationError(f"lineage at row {index} must be an object")
        _exact_fields(lineage, _LINEAGE_FIELDS, label=f"lineage row {index}")
        for field in _LINEAGE_FIELDS:
            if not isinstance(lineage[field], str):
                raise TraceValidationError(
                    f"lineage row {index} {field} must be a string"
                )
        if lineage["scenario_schema_version"] != scenario.scenario_schema_version:
            raise TraceValidationError("unsupported scenario schema in lineage")
        if lineage["trace_schema_version"] != scenario.trace_schema_version:
            raise TraceValidationError("unsupported trace schema in lineage")
        if lineage["equation_contract_revision"] != scenario.equation_contract_revision:
            raise TraceValidationError("unsupported equation contract in lineage")
        if lineage["scenario_sha256"] != scenario.scenario_sha256:
            raise TraceValidationError(
                f"scenario digest does not match parsed scenario at row {index}"
            )
        expected_run_id = derive_run_id(
            scenario_sha256=str(lineage["scenario_sha256"]),
            scenario_schema_version=str(lineage["scenario_schema_version"]),
            trace_schema_version=str(lineage["trace_schema_version"]),
            equation_contract_revision=str(lineage["equation_contract_revision"]),
        )
        if lineage["run_id"] != expected_run_id:
            raise TraceValidationError("run_id does not match lineage")
        if expected_lineage is None:
            expected_lineage = dict(lineage)
        elif lineage != expected_lineage:
            raise TraceValidationError(f"lineage changes at row {index}")
        rows.append(row)

    # Structural validation gives precise diagnostics. Exact deterministic replay
    # then binds every observable, action, resource and accounting byte to the
    # parsed scenario rather than accepting a merely well-shaped forged trace.
    from .runner import run_scenario

    expected_trace_bytes = run_scenario(scenario).trace_bytes
    if data != expected_trace_bytes:
        raise TraceValidationError("trace does not match deterministic replay")

    return tuple(rows)
