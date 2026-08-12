from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from . import physics as physics_module
from .physics import StepResult, advance_one_step, initial_state
from .scenario import (
    Scenario,
    TRACE_SCHEMA_VERSION_V2,
    TRACE_SCHEMA_VERSION_V3,
)
from .state import PlantState

_STATE_TOLERANCE = 1e-12


class AccountingInvariantError(RuntimeError):
    """Raised when a completed step's accounting receipt does not close."""


class StateInvariantError(RuntimeError):
    """Raised when a completed step violates a physical or resource invariant."""


@dataclass(frozen=True)
class SimulationRun:
    final_state: PlantState
    rows: tuple[Mapping[str, Any], ...]
    trace_bytes: bytes


def _require_causal_receipt_match(
    actual: Any, expected: Any, *, path: str
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AccountingInvariantError(
                f"{path} does not match causal recomputation"
            )
        for key in sorted(expected):
            _require_causal_receipt_match(
                actual[key], expected[key], path=f"{path}.{key}"
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise AccountingInvariantError(
                f"{path} does not match causal recomputation"
            )
        actual_value = float(actual)
        expected_value = float(expected)
        tolerance = max(
            1e-12,
            1e-10 * max(1.0, abs(actual_value), abs(expected_value)),
        )
        if (
            not math.isfinite(actual_value)
            or abs(actual_value - expected_value) > tolerance
        ):
            raise AccountingInvariantError(
                f"{path} does not match causal recomputation"
            )
        return
    if actual != expected:
        raise AccountingInvariantError(f"{path} does not match causal recomputation")


def validate_accounting_receipt(
    receipt: Mapping[str, Any],
    *,
    scenario: Scenario | None = None,
    pre_step_state: PlantState | None = None,
) -> None:
    species = receipt["species_accounting"]
    tolerance_mol = float(species["tolerance_mol"])
    for field in (
        "co2_residual_mol",
        "o2_residual_mol",
        "water_residual_mol",
        "inert_residual_mol",
    ):
        if abs(float(species[field])) > tolerance_mol:
            raise AccountingInvariantError(
                f"{field} exceeds declared species tolerance"
            )

    thermal = receipt["thermal"]
    system_receipt_scale_j = 0.0
    for zone_id in sorted(thermal["zones"]):
        zone = thermal["zones"][zone_id]
        zone_receipt_scale_j = sum(
            abs(float(zone[field]))
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
        if abs(float(zone["zone_thermal_residual_j"])) > tolerance_j:
            raise AccountingInvariantError(
                f"{zone_id} thermal residual exceeds declared tolerance"
            )
    system_tolerance_j = max(1e-6, 1e-10 * max(1.0, system_receipt_scale_j))
    if abs(float(thermal["system_residual_j"])) > system_tolerance_j:
        raise AccountingInvariantError(
            "system thermal residual exceeds declared tolerance"
        )

    electrical = receipt["electrical"]
    electrical_scale_wh = max(
        1.0,
        sum(
            abs(float(electrical[field]))
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
    if abs(float(electrical["residual_wh"])) > electrical_tolerance_wh:
        raise AccountingInvariantError("electrical residual exceeds declared tolerance")

    network = receipt.get("air_network")
    if network is None:
        if (
            scenario is not None
            and scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V3
        ):
            raise AccountingInvariantError(
                "scenario-v3 accounting requires an air-network receipt"
            )
        return
    if scenario is None:
        raise AccountingInvariantError(
            "air-network accounting requires the parsed scenario contract"
        )
    if pre_step_state is None:
        raise AccountingInvariantError(
            "air-network accounting requires the pre-step plant state"
        )

    def finite_network_value(field: str) -> float:
        value = float(network[field])
        if not math.isfinite(value):
            raise AccountingInvariantError(f"air-network {field} must be finite")
        return value

    density = finite_network_value("air_density_kg_m3")
    efficiency = finite_network_value("total_efficiency")
    if density <= 0.0:
        raise AccountingInvariantError("air-network density must be positive")
    if not 0.0 < efficiency <= 1.0:
        raise AccountingInvariantError("air-network efficiency must be in (0, 1]")

    declared_density = float(scenario.data["equipment"]["air_density_kg_m3"])
    declared_efficiency = float(
        scenario.data["air_network"]["fan"]["total_efficiency"]
    )
    parameter_tolerance = 1e-12
    if abs(density - declared_density) > parameter_tolerance:
        raise AccountingInvariantError(
            "air-network receipt does not use the declared reference density"
        )
    if abs(efficiency - declared_efficiency) > parameter_tolerance:
        raise AccountingInvariantError(
            "air-network receipt does not use the declared fan efficiency"
        )

    fan_pressure_pa = finite_network_value("fan_pressure_rise_pa")
    shared_pressure_pa = finite_network_value("shared_pressure_loss_pa")
    total_flow_m3_s = finite_network_value("total_flow_m3_s")
    fan_air_power_w = finite_network_value("fan_air_power_w")
    fan_electrical_power_w = finite_network_value("fan_electrical_power_w")
    if min(
        fan_pressure_pa,
        shared_pressure_pa,
        total_flow_m3_s,
        fan_air_power_w,
        fan_electrical_power_w,
    ) < 0.0:
        raise AccountingInvariantError(
            "air-network pressure, flow, and power must be non-negative"
        )

    zone_flow = network["zone_flow_m3_s"]
    zone_mass_flow = network["zone_mass_flow_kg_s"]
    branch_pressure = network["branch_pressure_loss_pa"]
    mass_residual = network["mass_balance_residual_kg_s"]
    if not (
        set(zone_flow)
        == set(zone_mass_flow)
        == set(branch_pressure)
        == set(mass_residual)
    ):
        raise AccountingInvariantError("air-network zone receipt ids do not match")

    flow_sum = sum(float(zone_flow[zone_id]) for zone_id in sorted(zone_flow))
    flow_tolerance = max(
        1e-12,
        1e-10 * max(1.0, abs(total_flow_m3_s), abs(flow_sum)),
    )
    if abs(total_flow_m3_s - flow_sum) > flow_tolerance:
        raise AccountingInvariantError(
            "air-network total flow does not equal branch flow sum"
        )

    for zone_id in sorted(zone_flow):
        flow = float(zone_flow[zone_id])
        mass_flow = float(zone_mass_flow[zone_id])
        residual = float(mass_residual[zone_id])
        pressure = float(branch_pressure[zone_id])
        if not all(
            math.isfinite(value) for value in (flow, mass_flow, residual, pressure)
        ):
            raise AccountingInvariantError("air-network zone receipts must be finite")
        if flow < 0.0 or mass_flow < 0.0 or pressure < 0.0:
            raise AccountingInvariantError(
                "air-network zone flow, mass, and pressure must be non-negative"
            )

        expected_mass_flow = density * flow
        mass_tolerance = max(
            1e-12,
            1e-10 * max(1.0, abs(mass_flow), abs(expected_mass_flow)),
        )
        if abs(mass_flow - expected_mass_flow) > mass_tolerance:
            raise AccountingInvariantError(
                f"{zone_id} mass flow does not equal reference density times volumetric flow"
            )
        if abs(residual) > mass_tolerance:
            raise AccountingInvariantError(
                f"{zone_id} fixed-density supply-return residual exceeds tolerance"
            )

        recomputed_pressure_residual = (
            fan_pressure_pa - shared_pressure_pa - pressure
        )
        pressure_tolerance = max(
            1e-9,
            1e-10
            * max(
                1.0,
                abs(fan_pressure_pa),
                abs(shared_pressure_pa),
                abs(pressure),
            ),
        )
        if abs(recomputed_pressure_residual) > pressure_tolerance:
            raise AccountingInvariantError(
                f"{zone_id} fan/system pressure residual exceeds tolerance"
            )

    recorded_pressure_residual = finite_network_value(
        "operating_point_residual_pa"
    )
    pressure_tolerance = max(
        1e-9,
        1e-10 * max(1.0, abs(fan_pressure_pa), abs(shared_pressure_pa)),
    )
    if abs(recorded_pressure_residual) > pressure_tolerance:
        raise AccountingInvariantError(
            "recorded fan/system pressure residual exceeds tolerance"
        )

    expected_air_power_w = fan_pressure_pa * total_flow_m3_s
    air_power_tolerance_w = max(
        1e-9,
        1e-10 * max(1.0, abs(fan_air_power_w), abs(expected_air_power_w)),
    )
    if abs(fan_air_power_w - expected_air_power_w) > air_power_tolerance_w:
        raise AccountingInvariantError("fan air power does not equal pressure times flow")

    expected_electrical_power_w = fan_air_power_w / efficiency
    electrical_power_tolerance_w = max(
        1e-9,
        1e-10
        * max(
            1.0,
            abs(fan_electrical_power_w),
            abs(expected_electrical_power_w),
        ),
    )
    if (
        abs(fan_electrical_power_w - expected_electrical_power_w)
        > electrical_power_tolerance_w
    ):
        raise AccountingInvariantError(
            "fan electrical power does not equal air power divided by efficiency"
        )

    recorded_fan_load_wh = float(electrical["fan_load_wh"])
    if not math.isfinite(recorded_fan_load_wh):
        raise AccountingInvariantError("electrical fan load must be finite")
    expected_fan_load_wh = (
        fan_electrical_power_w * float(scenario.data["dt_seconds"]) / 3600.0
    )
    fan_load_tolerance_wh = max(
        1e-12,
        1e-10
        * max(1.0, abs(recorded_fan_load_wh), abs(expected_fan_load_wh)),
    )
    if abs(recorded_fan_load_wh - expected_fan_load_wh) > fan_load_tolerance_wh:
        raise AccountingInvariantError(
            "electrical fan load does not match air-network fan power"
        )

    recomputed_receipt = physics_module.advance_one_step(
        scenario, pre_step_state
    ).receipt
    _require_causal_receipt_match(
        network,
        recomputed_receipt["air_network"],
        path="air-network receipt",
    )
    _require_causal_receipt_match(
        electrical["fan_load_wh"],
        recomputed_receipt["electrical"]["fan_load_wh"],
        path="electrical fan load",
    )


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
    scenario: Scenario,
    state: PlantState,
    *,
    segment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    zone_ids = sorted(state.zones)
    command = segment["command"] if segment is not None else None
    action: dict[str, Any] = {
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
    if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V3:
        action["fan_speed_fraction"] = state.utility.actual_fan_speed_fraction
        action["damper_position_by_id"] = {
            damper_id: float(state.utility.actual_damper_position_by_id[damper_id])
            for damper_id in sorted(state.utility.actual_damper_position_by_id)
        }
    return action


def _trace_command(
    scenario: Scenario, segment: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if segment is None:
        return None
    command = segment["command"]
    action: dict[str, Any] = {
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
    if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V3:
        action["fan_speed_fraction"] = float(command["fan_speed_fraction"])
        action["damper_position_by_id"] = {
            damper_id: float(command["damper_position_by_id"][damper_id])
            for damper_id in sorted(command["damper_position_by_id"])
        }
    else:
        action["airflow_m3_s"] = {
            zone_id: float(command["airflow_m3_s"][zone_id])
            for zone_id in sorted(command["airflow_m3_s"])
        }
    return action


def _row(
    scenario: Scenario,
    state: PlantState,
    *,
    segment: Mapping[str, Any] | None,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    accounting_receipt = receipt
    if receipt is not None and scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V3:
        accounting_receipt = {
            key: value for key, value in receipt.items() if key != "air_network"
        }
    row = {
        "schema_version": scenario.trace_schema_version,
        "lineage": {
            "run_id": scenario.run_id,
            "scenario_sha256": scenario.scenario_sha256,
            "scenario_schema_version": scenario.scenario_schema_version,
            "trace_schema_version": scenario.trace_schema_version,
            "equation_contract_revision": scenario.equation_contract_revision,
        },
        "step": state.step,
        "time_s": state.step * float(scenario.data["dt_seconds"]),
        "telemetry": _trace_telemetry(scenario, state),
        "commanded_action": _trace_command(scenario, segment),
        "actual_action": _trace_actual_action(scenario, state, segment=segment),
        "resource_state": _trace_resources(state),
        "realised_loads": segment["loads"] if segment is not None else None,
        "accounting_receipt": accounting_receipt,
        "invariant_status": {"passed": True},
    }
    if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V2:
        row["applied_operating_mode"] = (
            None if segment is None else segment["operating_mode"]
        )
    if scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V3:
        row["applied_operating_mode"] = (
            None if segment is None else segment["operating_mode"]
        )
        row["air_network_receipt"] = (
            None if receipt is None else receipt["air_network"]
        )
    return row


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
    scenario.validate_contract_identities()
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
        validate_accounting_receipt(
            result.receipt,
            scenario=scenario,
            pre_step_state=state,
        )
        state = result.state
        _assert_state_invariants(scenario, state)
        rows.append(_row(scenario, state, segment=segment, receipt=result.receipt))
    return SimulationRun(
        final_state=state,
        rows=tuple(rows),
        trace_bytes=_canonical_trace_bytes(rows),
    )
