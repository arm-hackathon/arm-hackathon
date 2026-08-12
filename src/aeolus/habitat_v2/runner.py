from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from .physics import StepResult, advance_one_step, initial_state
from .scenario import (
    EQUATION_CONTRACT_REVISION,
    SCENARIO_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    Scenario,
)
from .state import PlantState

_STATE_TOLERANCE = 1e-12


class AccountingInvariantError(RuntimeError):
    """Raised when a completed step's accounting receipt does not close."""


class StateInvariantError(RuntimeError):
    """Raised when a completed step violates a physical or resource invariant."""


def _finite_accounting_value(
    value: Any,
    *,
    path: str,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise AccountingInvariantError(f"{path} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AccountingInvariantError(
            f"{path} must be finite numeric data"
        ) from error
    if not math.isfinite(number):
        raise AccountingInvariantError(f"{path} must be finite numeric data")
    if non_negative and number < 0.0:
        raise AccountingInvariantError(f"{path} must be non-negative")
    return number


@dataclass(frozen=True)
class SimulationRun:
    final_state: PlantState
    rows: tuple[Mapping[str, Any], ...]
    trace_bytes: bytes


def validate_accounting_receipt(receipt: Mapping[str, Any]) -> None:
    species = receipt["species_accounting"]
    tolerance_mol = _finite_accounting_value(
        species["tolerance_mol"],
        path="species_accounting.tolerance_mol",
        non_negative=True,
    )
    for field in (
        "co2_residual_mol",
        "o2_residual_mol",
        "water_residual_mol",
        "inert_residual_mol",
    ):
        residual = _finite_accounting_value(
            species[field], path=f"species_accounting.{field}"
        )
        if abs(residual) > tolerance_mol:
            raise AccountingInvariantError(
                f"{field} exceeds declared species tolerance"
            )

    thermal = receipt["thermal"]
    system_receipt_scale_j = 0.0
    for zone_id in sorted(thermal["zones"]):
        zone = thermal["zones"][zone_id]
        zone_receipt_scale_j = sum(
            abs(
                _finite_accounting_value(
                    zone[field], path=f"thermal.zones.{zone_id}.{field}"
                )
            )
            for field in (
                "metabolic_heat_added_j",
                "recirculation_heat_added_j",
                "cooling_heat_removed_j",
                "passive_heat_rejected_j",
                "passive_heat_received_j",
                "zone_thermal_energy_delta_j",
            )
        )
        system_receipt_scale_j += zone_receipt_scale_j
        receipt_scale_j = max(1.0, zone_receipt_scale_j)
        tolerance_j = max(1e-6, 1e-10 * receipt_scale_j)
        zone_residual_j = _finite_accounting_value(
            zone["zone_thermal_residual_j"],
            path=f"thermal.zones.{zone_id}.zone_thermal_residual_j",
        )
        if abs(zone_residual_j) > tolerance_j:
            raise AccountingInvariantError(
                f"{zone_id} thermal residual exceeds declared tolerance"
            )
    system_tolerance_j = max(1e-6, 1e-10 * max(1.0, system_receipt_scale_j))
    system_residual_j = _finite_accounting_value(
        thermal["system_residual_j"], path="thermal.system_residual_j"
    )
    if abs(system_residual_j) > system_tolerance_j:
        raise AccountingInvariantError(
            "system thermal residual exceeds declared tolerance"
        )

    electrical = receipt["electrical"]
    electrical_scale_wh = max(
        1.0,
        sum(
            abs(
                _finite_accounting_value(
                    electrical[field], path=f"electrical.{field}"
                )
            )
            for field in (
                "generation_wh",
                "battery_withdrawn_wh",
                "served_load_wh",
                "battery_charge_stored_wh",
                "curtailed_generation_wh",
                "charge_conversion_loss_wh",
                "discharge_conversion_loss_wh",
            )
        ),
    )
    electrical_tolerance_wh = max(1e-12, 1e-10 * electrical_scale_wh)
    electrical_residual_wh = _finite_accounting_value(
        electrical["residual_wh"], path="electrical.residual_wh"
    )
    if abs(electrical_residual_wh) > electrical_tolerance_wh:
        raise AccountingInvariantError("electrical residual exceeds declared tolerance")


def _assert_finite(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite(nested, path=f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise StateInvariantError(f"non-finite state at {path}")


def _state_payload(scenario: Scenario, state: PlantState) -> dict[str, Any]:
    zone_configs = {str(zone["id"]): zone for zone in scenario.data["zones"]}
    zones: dict[str, Any] = {}
    for zone_id in sorted(state.zones):
        zone = state.zones[zone_id]
        zones[zone_id] = {
            "co2_mol": zone.co2_mol,
            "o2_mol": zone.o2_mol,
            "water_vapor_mol": zone.water_vapor_mol,
            "inert_mol": zone.inert_mol,
            "temperature_k": zone.temperature_k,
            "telemetry": zone.telemetry(
                volume_m3=float(zone_configs[zone_id]["volume_m3"])
            ),
        }
    utility = state.utility
    return {
        "zones": zones,
        "utility": {
            "co2_sorbent_remaining_mol": utility.co2_sorbent_remaining_mol,
            "oxygen_store_mol": utility.oxygen_store_mol,
            "battery_energy_wh": utility.battery_energy_wh,
            "captured_co2_mol": utility.captured_co2_mol,
            "condensed_water_mol": utility.condensed_water_mol,
            "external_heat_received_j": utility.external_heat_received_j,
            "external_heat_rejected_j": utility.external_heat_rejected_j,
            "actual_airflow_m3_s": {
                zone_id: float(utility.actual_airflow_m3_s[zone_id])
                for zone_id in sorted(utility.actual_airflow_m3_s)
            },
            "actual_scrubber_duty": utility.actual_scrubber_duty,
            "actual_condenser_duty": utility.actual_condenser_duty,
        },
    }


def _assert_state_invariants(scenario: Scenario, state: PlantState) -> None:
    payload = _state_payload(scenario, state)
    _assert_finite(payload, path="state")
    for zone_id, zone in state.zones.items():
        for field in ("co2_mol", "o2_mol", "water_vapor_mol", "inert_mol"):
            if getattr(zone, field) < -_STATE_TOLERANCE:
                raise StateInvariantError(f"negative {field} in {zone_id}")
        if zone.temperature_k <= 0.0:
            raise StateInvariantError(f"non-positive temperature in {zone_id}")
        relative_humidity = payload["zones"][zone_id]["telemetry"]["relative_humidity"]
        if not -_STATE_TOLERANCE <= relative_humidity <= 1.0 + _STATE_TOLERANCE:
            raise StateInvariantError(f"relative humidity outside [0, 1] in {zone_id}")

    equipment = scenario.data["equipment"]
    utility = state.utility
    for field in (
        "co2_sorbent_remaining_mol",
        "oxygen_store_mol",
        "battery_energy_wh",
        "captured_co2_mol",
        "condensed_water_mol",
        "external_heat_received_j",
        "external_heat_rejected_j",
    ):
        if getattr(utility, field) < -_STATE_TOLERANCE:
            raise StateInvariantError(f"negative utility inventory: {field}")
    if (
        utility.battery_energy_wh
        > float(equipment["battery_capacity_wh"]) + _STATE_TOLERANCE
    ):
        raise StateInvariantError("battery exceeds declared capacity")
    if not 0.0 <= utility.actual_scrubber_duty <= 1.0:
        raise StateInvariantError("scrubber duty outside [0, 1]")
    if not 0.0 <= utility.actual_condenser_duty <= 1.0:
        raise StateInvariantError("condenser duty outside [0, 1]")
    if any(value < 0.0 for value in utility.actual_airflow_m3_s.values()):
        raise StateInvariantError("negative actual airflow")


def _trace_telemetry(scenario: Scenario, state: PlantState) -> dict[str, Any]:
    payload = _state_payload(scenario, state)
    observable_fields = (
        "temperature_k",
        "pressure_pa",
        "co2_ppm",
        "o2_mole_fraction",
        "relative_humidity",
    )
    return {
        zone_id: {
            field: payload["zones"][zone_id]["telemetry"][field]
            for field in observable_fields
        }
        for zone_id in sorted(payload["zones"])
    }


def _trace_resources(state: PlantState) -> dict[str, float]:
    utility = state.utility
    return {
        "co2_sorbent_remaining_mol": utility.co2_sorbent_remaining_mol,
        "oxygen_store_mol": utility.oxygen_store_mol,
        "battery_energy_wh": utility.battery_energy_wh,
        "captured_co2_mol": utility.captured_co2_mol,
        "condensed_water_mol": utility.condensed_water_mol,
        "external_heat_received_j": utility.external_heat_received_j,
        "external_heat_rejected_j": utility.external_heat_rejected_j,
    }


def _trace_actual_action(
    state: PlantState,
    *,
    segment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    zone_ids = sorted(state.zones)
    command = segment["command"] if segment is not None else None
    return {
        "airflow_m3_s": {
            zone_id: float(state.utility.actual_airflow_m3_s[zone_id])
            for zone_id in zone_ids
        },
        "scrubber_duty": state.utility.actual_scrubber_duty,
        "condenser_duty": state.utility.actual_condenser_duty,
        "cooling_removed_w": {
            zone_id: (
                float(command["cooling_removed_w"][zone_id])
                if command is not None
                else 0.0
            )
            for zone_id in zone_ids
        },
        "oxygen_injection_mol_s": {
            zone_id: (
                float(command["oxygen_injection_mol_s"][zone_id])
                if command is not None
                else 0.0
            )
            for zone_id in zone_ids
        },
    }


def _trace_command(segment: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if segment is None:
        return None
    command = segment["command"]
    return {
        "airflow_m3_s": {
            zone_id: float(command["airflow_m3_s"][zone_id])
            for zone_id in sorted(command["airflow_m3_s"])
        },
        "scrubber_duty": float(command["scrubber_duty"]),
        "condenser_duty": float(command["condenser_duty"]),
        "cooling_removed_w": {
            zone_id: float(command["cooling_removed_w"][zone_id])
            for zone_id in sorted(command["cooling_removed_w"])
        },
        "oxygen_injection_mol_s": {
            zone_id: float(command["oxygen_injection_mol_s"][zone_id])
            for zone_id in sorted(command["oxygen_injection_mol_s"])
        },
    }


def _row(
    scenario: Scenario,
    state: PlantState,
    *,
    segment: Mapping[str, Any] | None,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "lineage": {
            "run_id": scenario.run_id,
            "scenario_sha256": scenario.scenario_sha256,
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "equation_contract_revision": EQUATION_CONTRACT_REVISION,
        },
        "step": state.step,
        "time_s": state.step * float(scenario.data["dt_seconds"]),
        "telemetry": _trace_telemetry(scenario, state),
        "commanded_action": _trace_command(segment),
        "actual_action": _trace_actual_action(state, segment=segment),
        "resource_state": _trace_resources(state),
        "realised_loads": segment["loads"] if segment is not None else None,
        "accounting_receipt": receipt,
        "invariant_status": {"passed": True},
    }


def _canonical_trace_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def run_scenario(scenario: Scenario) -> SimulationRun:
    state = initial_state(scenario)
    _assert_state_invariants(scenario, state)
    rows: list[Mapping[str, Any]] = [_row(scenario, state, segment=None, receipt=None)]
    while state.step < int(scenario.data["steps"]):
        segment = next(
            segment
            for segment in scenario.data["timeline"]
            if segment["start_step"] <= state.step < segment["end_step"]
        )
        result: StepResult = advance_one_step(scenario, state)
        validate_accounting_receipt(result.receipt)
        state = result.state
        _assert_state_invariants(scenario, state)
        rows.append(_row(scenario, state, segment=segment, receipt=result.receipt))
    return SimulationRun(
        final_state=state,
        rows=tuple(rows),
        trace_bytes=_canonical_trace_bytes(rows),
    )
