from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .actuators import achieve_actuator_state
from .air_network import (
    AirNetworkResult,
    AirNetworkSpec,
    BranchSpec,
    FanSpec,
    solve_air_network,
)
from .faults import physical_fault_effects
from .instrumentation import measure_operational_feedback
from .scenario import (
    SCENARIO_SCHEMA_VERSION_V3,
    SCENARIO_SCHEMA_VERSION_V4,
    SCENARIO_SCHEMA_VERSION_V5,
    Scenario,
    ScenarioValidationError,
    command_fields_for_schema,
)
from .state import (
    GAS_CONSTANT_J_PER_MOL_K,
    PlantState,
    UtilityState,
    ZoneState,
    saturation_vapor_pressure_pa,
)


@dataclass(frozen=True)
class StepResult:
    state: PlantState
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CanonicalExternalCommand:
    """A closed command validated without consulting scenario timeline data."""

    scenario_schema_version: str
    canonical_bytes: bytes
    sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class AchievedStateCommandReference:
    command_reference_kind: str
    command: CanonicalExternalCommand


@dataclass(frozen=True, slots=True)
class PreflightResult:
    classification: str
    application_step: int
    command_sha256: str
    preflight_contract_sha256: str
    preflight_result_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "application_step": self.application_step,
            "command_sha256": self.command_sha256,
            "preflight_contract_sha256": self.preflight_contract_sha256,
            "preflight_result_sha256": self.preflight_result_sha256,
        }


class InfeasibleActionError(RuntimeError):
    """Raised before state advance when a valid command cannot be served."""


def _segment_for_step(scenario: Scenario, step: int) -> Mapping[str, Any]:
    for segment in scenario.data["timeline"]:
        if segment["start_step"] <= step < segment["end_step"]:
            return segment
    raise ScenarioValidationError(f"no timeline segment covers step {step}")


def _slew(current: float, target: float, maximum_delta: float) -> float:
    difference = target - current
    if abs(difference) <= maximum_delta:
        return target
    return current + math.copysign(maximum_delta, difference)


def _air_network_spec(
    scenario: Scenario,
    *,
    open_supply_resistance_multiplier_by_zone: Mapping[str, float] | None = None,
) -> AirNetworkSpec:
    network = scenario.data["air_network"]
    fan = network["fan"]
    shared = network["shared_resistance"]
    resistance_multiplier_by_zone = (
        {}
        if open_supply_resistance_multiplier_by_zone is None
        else open_supply_resistance_multiplier_by_zone
    )
    return AirNetworkSpec(
        fan=FanSpec(
            component_id=str(fan["id"]),
            rated_free_delivery_m3_s=float(fan["rated_free_delivery_m3_s"]),
            rated_shutoff_pressure_pa=float(fan["rated_shutoff_pressure_pa"]),
            total_efficiency=float(fan["total_efficiency"]),
        ),
        shared_resistance_pa_s2_m6=sum(float(value) for value in shared.values()),
        air_density_kg_m3=float(scenario.data["equipment"]["air_density_kg_m3"]),
        branches=tuple(
            BranchSpec(
                zone_id=str(branch["zone_id"]),
                damper_id=str(branch["damper_id"]),
                open_supply_resistance_pa_s2_m6=float(
                    branch["open_supply_resistance_pa_s2_m6"]
                )
                * float(resistance_multiplier_by_zone.get(str(branch["zone_id"]), 1.0)),
                return_resistance_pa_s2_m6=float(branch["return_resistance_pa_s2_m6"]),
                damper_leak_fraction=float(branch["damper_leak_fraction"]),
            )
            for branch in sorted(
                network["branches"], key=lambda value: str(value["zone_id"])
            )
        ),
    )


def _air_network_receipt(
    result: AirNetworkResult,
    *,
    requested_fan_speed_fraction: float,
    actual_fan_speed_fraction: float,
    effective_fan_speed_fraction: float | None,
    requested_damper_position_by_id: Mapping[str, float],
    actual_damper_position_by_id: Mapping[str, float],
) -> dict[str, Any]:
    receipt = {
        "requested_fan_speed_fraction": requested_fan_speed_fraction,
        "actual_fan_speed_fraction": actual_fan_speed_fraction,
        "requested_damper_position_by_id": {
            damper_id: float(requested_damper_position_by_id[damper_id])
            for damper_id in sorted(requested_damper_position_by_id)
        },
        "actual_damper_position_by_id": {
            damper_id: float(actual_damper_position_by_id[damper_id])
            for damper_id in sorted(actual_damper_position_by_id)
        },
        "fan_pressure_rise_pa": result.fan_pressure_rise_pa,
        "shared_pressure_loss_pa": result.shared_pressure_loss_pa,
        "branch_pressure_loss_pa": dict(result.branch_pressure_loss_pa),
        "total_flow_m3_s": result.total_flow_m3_s,
        "zone_flow_m3_s": dict(result.zone_flow_m3_s),
        "zone_mass_flow_kg_s": dict(result.zone_mass_flow_kg_s),
        "fan_air_power_w": result.fan_air_power_w,
        "fan_electrical_power_w": result.fan_electrical_power_w,
        "total_efficiency": result.total_efficiency,
        "air_density_kg_m3": result.air_density_kg_m3,
        "operating_point_residual_pa": result.operating_point_residual_pa,
        "mass_balance_residual_kg_s": dict(result.mass_balance_residual_kg_s),
    }
    if effective_fan_speed_fraction is not None:
        receipt["effective_fan_speed_fraction"] = effective_fan_speed_fraction
    return receipt


def _operational_feedback_truth(
    scenario: Scenario,
    state: PlantState,
    *,
    network_result: AirNetworkResult,
    recirculation_receipt: Mapping[str, Any],
    dt_seconds: float,
    effective_fan_speed_fraction: float | None = None,
) -> dict[str, Mapping[str, float] | float]:
    voltage = float(scenario.data["actuator_feedback"]["dc_bus_voltage_v"])
    equipment = scenario.data["equipment"]
    return {
        "fan_speed_fraction": float(
            state.utility.actual_fan_speed_fraction
            if effective_fan_speed_fraction is None
            else effective_fan_speed_fraction
        ),
        "fan_dc_bus_current_a": network_result.fan_electrical_power_w / voltage,
        "damper_position_by_id": {
            damper_id: float(state.utility.actual_damper_position_by_id[damper_id])
            for damper_id in sorted(state.utility.actual_damper_position_by_id)
        },
        "branch_airflow_m3_s": {
            zone_id: float(network_result.zone_flow_m3_s[zone_id])
            for zone_id in sorted(network_result.zone_flow_m3_s)
        },
        "branch_differential_pressure_pa": {
            zone_id: float(network_result.branch_pressure_loss_pa[zone_id])
            for zone_id in sorted(network_result.branch_pressure_loss_pa)
        },
        "scrubber_capture_rate_mol_s": float(recirculation_receipt["co2_captured_mol"])
        / dt_seconds,
        "condenser_removal_rate_mol_s": float(
            recirculation_receipt["water_condensed_mol"]
        )
        / dt_seconds,
        "cooling_delivery_w": {
            zone_id: float(state.utility.effective_cooling_delivery_by_zone[zone_id])
            for zone_id in sorted(state.utility.effective_cooling_delivery_by_zone)
        },
        "oxygen_delivery_mol_s": {
            zone_id: float(state.utility.effective_oxygen_delivery_by_zone[zone_id])
            for zone_id in sorted(state.utility.effective_oxygen_delivery_by_zone)
        },
        "battery_state_of_charge": float(state.utility.battery_energy_wh)
        / float(equipment["battery_capacity_wh"]),
        "oxygen_store_fraction": float(state.utility.oxygen_store_mol)
        / max(1e-12, float(scenario.data["initial_utility"]["oxygen_store_mol"])),
        "sorbent_remaining_fraction": float(state.utility.co2_sorbent_remaining_mol)
        / float(equipment["scrubber_capacity_mol"]),
    }


def _recirculate(
    zones: Mapping[str, ZoneState],
    *,
    zone_configs: Mapping[str, Mapping[str, Any]],
    airflow_m3_s: Mapping[str, float],
    scrubber_duty: float,
    condenser_duty: float,
    equipment: Mapping[str, Any],
    sorbent_remaining_mol: float,
    dt_seconds: float,
    scrubber_capture_ability: float = 1.0,
    condenser_removal_ability: float = 1.0,
    include_effectiveness_receipt: bool = False,
) -> tuple[dict[str, ZoneState], dict[str, Any]]:
    species = ("co2_mol", "o2_mol", "water_vapor_mol", "inert_mol")
    extracted: dict[str, dict[str, float]] = {}
    remaining: dict[str, ZoneState] = {}
    pool = {name: 0.0 for name in species}
    extracted_totals: dict[str, float] = {}

    for zone_id in sorted(zones):
        zone = zones[zone_id]
        volume_m3 = float(zone_configs[zone_id]["volume_m3"])
        exchange_fraction = 1.0 - math.exp(
            -float(airflow_m3_s[zone_id]) * dt_seconds / volume_m3
        )
        zone_extracted = {
            name: getattr(zone, name) * exchange_fraction for name in species
        }
        extracted[zone_id] = zone_extracted
        extracted_totals[zone_id] = sum(zone_extracted.values())
        for name in species:
            pool[name] += zone_extracted[name]
        remaining[zone_id] = replace(
            zone,
            **{name: getattr(zone, name) - zone_extracted[name] for name in species},
        )

    total_extracted = sum(extracted_totals.values())
    if total_extracted == 0.0:
        receipt = {
            "exchange_fraction": {zone_id: 0.0 for zone_id in zones},
            "co2_captured_mol": 0.0,
            "water_condensed_mol": 0.0,
            "species_residual_mol": {name: 0.0 for name in species},
        }
        if include_effectiveness_receipt:
            receipt["scrubber_capture_ability"] = scrubber_capture_ability
            receipt["condenser_removal_ability"] = condenser_removal_ability
        return dict(zones), receipt

    co2_captured_mol = min(
        pool["co2_mol"],
        float(equipment["scrubber_max_co2_mol_s"])
        * scrubber_duty
        * scrubber_capture_ability
        * dt_seconds,
        sorbent_remaining_mol,
    )
    water_condensed_mol = min(
        pool["water_vapor_mol"],
        float(equipment["condenser_max_water_mol_s"])
        * condenser_duty
        * condenser_removal_ability
        * dt_seconds,
    )
    pool["co2_mol"] -= co2_captured_mol
    pool["water_vapor_mol"] -= water_condensed_mol

    mixed: dict[str, ZoneState] = {}
    for zone_id in sorted(zones):
        return_fraction = extracted_totals[zone_id] / total_extracted
        zone = remaining[zone_id]
        mixed[zone_id] = replace(
            zone,
            **{
                name: getattr(zone, name) + pool[name] * return_fraction
                for name in species
            },
        )

    removed = {
        "co2_mol": co2_captured_mol,
        "water_vapor_mol": water_condensed_mol,
        "o2_mol": 0.0,
        "inert_mol": 0.0,
    }
    residual = {
        name: sum(getattr(zone, name) for zone in mixed.values())
        + removed[name]
        - sum(getattr(zone, name) for zone in zones.values())
        for name in species
    }
    receipt = {
        "extracted_mol": extracted,
        "co2_captured_mol": co2_captured_mol,
        "water_condensed_mol": water_condensed_mol,
        "species_residual_mol": residual,
    }
    if include_effectiveness_receipt:
        receipt["scrubber_capture_ability"] = scrubber_capture_ability
        receipt["condenser_removal_ability"] = condenser_removal_ability
    return mixed, receipt


def _recirculation_heat_transfer_j(
    zones: Mapping[str, ZoneState],
    *,
    airflow_m3_s: Mapping[str, float],
    equipment: Mapping[str, Any],
    dt_seconds: float,
) -> dict[str, float]:
    zone_ids = sorted(zones)
    total_airflow_m3_s = sum(float(airflow_m3_s[zone_id]) for zone_id in zone_ids)
    if total_airflow_m3_s == 0.0:
        return {zone_id: 0.0 for zone_id in zone_ids}

    mixed_temperature_k = (
        sum(
            float(airflow_m3_s[zone_id]) * zones[zone_id].temperature_k
            for zone_id in zone_ids
        )
        / total_airflow_m3_s
    )
    density_kg_m3 = float(equipment["air_density_kg_m3"])
    specific_heat_j_kg_k = float(equipment["air_specific_heat_j_kg_k"])
    transfers = {
        zone_id: (
            density_kg_m3
            * specific_heat_j_kg_k
            * float(airflow_m3_s[zone_id])
            * (mixed_temperature_k - zones[zone_id].temperature_k)
            * dt_seconds
        )
        for zone_id in zone_ids
    }
    closure_error_j = sum(transfers.values())
    transfers[zone_ids[-1]] -= closure_error_j
    return transfers


def _apply_zone_thermal_balance(
    zones: Mapping[str, ZoneState],
    *,
    zone_configs: Mapping[str, Mapping[str, Any]],
    loads: Mapping[str, Mapping[str, Any]],
    cooling_removed_w: Mapping[str, float],
    recirculation_heat_added_j: Mapping[str, float],
    dt_seconds: float,
) -> tuple[dict[str, ZoneState], dict[str, Any]]:
    updated: dict[str, ZoneState] = {}
    zone_receipts: dict[str, Mapping[str, float]] = {}
    for zone_id in sorted(zones):
        zone = zones[zone_id]
        config = zone_configs[zone_id]
        thermal_capacity_j_per_k = float(config["thermal_capacity_j_per_k"])
        metabolic_heat_added_j = float(loads[zone_id]["sensible_heat_w"]) * dt_seconds
        signed_passive_heat_j = (
            float(config["passive_thermal_conductance_w_per_k"])
            * (float(config["sink_temperature_k"]) - zone.temperature_k)
            * dt_seconds
        )
        passive_heat_received_j = max(0.0, signed_passive_heat_j)
        passive_heat_rejected_j = max(0.0, -signed_passive_heat_j)
        cooling_heat_removed_j = float(cooling_removed_w[zone_id]) * dt_seconds
        zone_recirculation_heat_added_j = float(recirculation_heat_added_j[zone_id])
        expected_energy_delta_j = (
            metabolic_heat_added_j
            + zone_recirculation_heat_added_j
            + passive_heat_received_j
            - passive_heat_rejected_j
            - cooling_heat_removed_j
        )
        next_temperature_k = (
            zone.temperature_k + expected_energy_delta_j / thermal_capacity_j_per_k
        )
        zone_thermal_energy_delta_j = (
            next_temperature_k - zone.temperature_k
        ) * thermal_capacity_j_per_k
        zone_thermal_residual_j = zone_thermal_energy_delta_j - expected_energy_delta_j
        updated[zone_id] = replace(zone, temperature_k=next_temperature_k)
        zone_receipts[zone_id] = {
            "metabolic_heat_added_j": metabolic_heat_added_j,
            "recirculation_heat_added_j": zone_recirculation_heat_added_j,
            "cooling_heat_removed_j": cooling_heat_removed_j,
            "passive_heat_rejected_j": passive_heat_rejected_j,
            "passive_heat_received_j": passive_heat_received_j,
            "zone_thermal_energy_delta_j": zone_thermal_energy_delta_j,
            "zone_thermal_residual_j": zone_thermal_residual_j,
        }

    return updated, {
        "zones": zone_receipts,
        "system_residual_j": sum(
            receipt["zone_thermal_residual_j"] for receipt in zone_receipts.values()
        ),
        "external_heat_received_j": sum(
            receipt["passive_heat_received_j"] for receipt in zone_receipts.values()
        ),
        "external_heat_rejected_j": sum(
            receipt["passive_heat_rejected_j"] + receipt["cooling_heat_removed_j"]
            for receipt in zone_receipts.values()
        ),
    }


def _apply_passive_condensation(
    zones: Mapping[str, ZoneState],
    *,
    zone_configs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, ZoneState], dict[str, float]]:
    updated: dict[str, ZoneState] = {}
    condensed_by_zone: dict[str, float] = {}
    for zone_id in sorted(zones):
        zone = zones[zone_id]
        volume_m3 = float(zone_configs[zone_id]["volume_m3"])
        saturated_water_mol = (
            saturation_vapor_pressure_pa(zone.temperature_k)
            * volume_m3
            / (GAS_CONSTANT_J_PER_MOL_K * zone.temperature_k)
        )
        condensed_mol = max(0.0, zone.water_vapor_mol - saturated_water_mol)
        updated[zone_id] = replace(
            zone,
            water_vapor_mol=zone.water_vapor_mol - condensed_mol,
        )
        condensed_by_zone[zone_id] = condensed_mol
    return updated, condensed_by_zone


def _electrical_balance(
    *,
    battery_energy_wh: float,
    equipment: Mapping[str, Any],
    command: Mapping[str, Any],
    actual_airflow_m3_s: Mapping[str, float],
    fan_electrical_power_w: float | None,
    actual_scrubber_duty: float,
    actual_condenser_duty: float,
    generation_w: float,
    dt_seconds: float,
    achieved_cooling_removed_w: Mapping[str, float] | None = None,
    achieved_oxygen_injection_mol_s: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    duration_hours = dt_seconds / 3600.0
    cooling_command = (
        command["cooling_removed_w"]
        if achieved_cooling_removed_w is None
        else achieved_cooling_removed_w
    )
    oxygen_command = (
        command["oxygen_injection_mol_s"]
        if achieved_oxygen_injection_mol_s is None
        else achieved_oxygen_injection_mol_s
    )
    load_power_w = {
        "fixed_load_wh": float(equipment["base_load_w"]),
        "fan_load_wh": (
            fan_electrical_power_w
            if fan_electrical_power_w is not None
            else float(equipment["fan_power_w_per_m3_s"])
            * sum(float(value) for value in actual_airflow_m3_s.values())
        ),
        "scrubber_load_wh": (
            float(equipment["scrubber_power_w_full"]) * actual_scrubber_duty
        ),
        "condenser_load_wh": (
            float(equipment["condenser_power_w_full"]) * actual_condenser_duty
        ),
        "cooling_load_wh": (
            sum(float(value) for value in cooling_command.values())
            / float(equipment["cooling_coefficient_of_performance"])
        ),
        "oxygen_injection_load_wh": (
            float(equipment["oxygen_injection_power_w_per_mol_s"])
            * sum(float(value) for value in oxygen_command.values())
        ),
    }
    load_energy_wh = {
        name: power_w * duration_hours for name, power_w in load_power_w.items()
    }
    served_load_wh = sum(load_energy_wh.values())
    generation_wh = float(generation_w) * duration_hours

    battery_charge_input_wh = 0.0
    battery_charge_stored_wh = 0.0
    battery_withdrawn_wh = 0.0
    battery_bus_output_wh = 0.0
    charge_conversion_loss_wh = 0.0
    discharge_conversion_loss_wh = 0.0
    curtailed_generation_wh = 0.0

    if generation_wh >= served_load_wh:
        surplus_wh = generation_wh - served_load_wh
        charge_efficiency = float(equipment["battery_charge_efficiency"])
        capacity_headroom_wh = (
            float(equipment["battery_capacity_wh"]) - battery_energy_wh
        )
        maximum_charge_input_wh = min(
            float(equipment["battery_max_charge_w"]) * duration_hours,
            capacity_headroom_wh / charge_efficiency,
        )
        battery_charge_input_wh = min(surplus_wh, maximum_charge_input_wh)
        battery_charge_stored_wh = battery_charge_input_wh * charge_efficiency
        charge_conversion_loss_wh = battery_charge_input_wh - battery_charge_stored_wh
        curtailed_generation_wh = surplus_wh - battery_charge_input_wh
    else:
        deficit_at_bus_wh = served_load_wh - generation_wh
        discharge_efficiency = float(equipment["battery_discharge_efficiency"])
        required_withdrawal_wh = deficit_at_bus_wh / discharge_efficiency
        maximum_withdrawal_wh = min(
            battery_energy_wh,
            float(equipment["battery_max_discharge_w"]) * duration_hours,
        )
        tolerance_wh = max(1e-12, 1e-10 * max(1.0, served_load_wh))
        if required_withdrawal_wh > maximum_withdrawal_wh + tolerance_wh:
            raise InfeasibleActionError(
                "electrical demand exceeds generation and battery capability"
            )
        battery_withdrawn_wh = required_withdrawal_wh
        battery_bus_output_wh = battery_withdrawn_wh * discharge_efficiency
        discharge_conversion_loss_wh = battery_withdrawn_wh - battery_bus_output_wh

    battery_energy_delta_wh = battery_charge_stored_wh - battery_withdrawn_wh
    next_battery_energy_wh = battery_energy_wh + battery_energy_delta_wh
    residual_wh = (
        generation_wh
        + battery_withdrawn_wh
        - (
            served_load_wh
            + battery_charge_stored_wh
            + curtailed_generation_wh
            + charge_conversion_loss_wh
            + discharge_conversion_loss_wh
        )
    )
    receipt = {
        "generation_wh": generation_wh,
        **load_energy_wh,
        "served_load_wh": served_load_wh,
        "battery_charge_input_wh": battery_charge_input_wh,
        "battery_charge_stored_wh": battery_charge_stored_wh,
        "battery_withdrawn_wh": battery_withdrawn_wh,
        "battery_bus_output_wh": battery_bus_output_wh,
        "charge_conversion_loss_wh": charge_conversion_loss_wh,
        "discharge_conversion_loss_wh": discharge_conversion_loss_wh,
        "curtailed_generation_wh": curtailed_generation_wh,
        "battery_energy_delta_wh": battery_energy_delta_wh,
        "residual_wh": residual_wh,
    }
    return next_battery_energy_wh, receipt


def initial_state(scenario: Scenario) -> PlantState:
    zones: dict[str, ZoneState] = {}
    for zone_config in scenario.data["zones"]:
        initial = zone_config["initial"]
        temperature_k = float(initial["temperature_k"])
        pressure_pa = float(initial["pressure_pa"])
        volume_m3 = float(zone_config["volume_m3"])
        total_moles = (
            pressure_pa * volume_m3 / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        )
        water_partial_pressure_pa = float(
            initial["relative_humidity"]
        ) * saturation_vapor_pressure_pa(temperature_k)
        water_fraction = water_partial_pressure_pa / pressure_pa
        co2_fraction = float(initial["co2_ppm"]) / 1_000_000.0
        o2_fraction = float(initial["o2_mole_fraction"])
        inert_fraction = 1.0 - water_fraction - co2_fraction - o2_fraction
        if inert_fraction <= 0.0:
            raise ScenarioValidationError(
                f"invalid scenario value at zones.{zone_config['id']}.initial: "
                "species fractions leave no inert gas"
            )
        zones[str(zone_config["id"])] = ZoneState(
            co2_mol=total_moles * co2_fraction,
            o2_mol=total_moles * o2_fraction,
            water_vapor_mol=total_moles * water_fraction,
            inert_mol=total_moles * inert_fraction,
            temperature_k=temperature_k,
        )

    utility = scenario.data["initial_utility"]
    if scenario.scenario_schema_version in {
        SCENARIO_SCHEMA_VERSION_V3,
        SCENARIO_SCHEMA_VERSION_V4,
        SCENARIO_SCHEMA_VERSION_V5,
    }:
        initial_fan_speed = float(utility["actual_fan_speed_fraction"])
        initial_dampers = {
            str(damper_id): float(position)
            for damper_id, position in utility["actual_damper_position_by_id"].items()
        }
        initial_network_result = solve_air_network(
            _air_network_spec(scenario),
            fan_speed_fraction=initial_fan_speed,
            damper_position_by_id=initial_dampers,
        )
        initial_airflow = dict(initial_network_result.zone_flow_m3_s)
    else:
        initial_fan_speed = None
        initial_dampers = {}
        initial_airflow = {
            str(zone_id): float(value)
            for zone_id, value in utility["actual_airflow_m3_s"].items()
        }

    initial_utility = UtilityState(
        co2_sorbent_remaining_mol=float(utility["co2_sorbent_remaining_mol"]),
        captured_co2_mol=float(utility["captured_co2_mol"]),
        condensed_water_mol=float(utility["condensed_water_mol"]),
        oxygen_store_mol=float(utility["oxygen_store_mol"]),
        battery_energy_wh=float(utility["battery_energy_wh"]),
        actual_airflow_m3_s=initial_airflow,
        actual_scrubber_duty=float(utility["actual_scrubber_duty"]),
        actual_condenser_duty=float(utility["actual_condenser_duty"]),
        external_heat_rejected_j=float(utility["external_heat_rejected_j"]),
        external_heat_received_j=float(utility["external_heat_received_j"]),
        actual_fan_speed_fraction=initial_fan_speed,
        actual_damper_position_by_id=initial_dampers,
        actual_cooling_removed_w=(
            {
                str(zone_id): float(value)
                for zone_id, value in utility["actual_cooling_removed_w"].items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else {}
        ),
        actual_oxygen_injection_mol_s=(
            {
                str(zone_id): float(value)
                for zone_id, value in utility["actual_oxygen_injection_mol_s"].items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else {}
        ),
        effective_cooling_delivery_by_zone=(
            {
                str(zone_id): float(value)
                for zone_id, value in utility["actual_cooling_removed_w"].items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else {}
        ),
        effective_oxygen_delivery_by_zone=(
            {
                str(zone_id): float(value)
                for zone_id, value in utility["actual_oxygen_injection_mol_s"].items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else {}
        ),
    )
    initial_state_value = PlantState(
        step=0,
        zones=zones,
        utility=initial_utility,
    )
    if scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
        return initial_state_value
    feedback_truth = _operational_feedback_truth(
        scenario,
        initial_state_value,
        network_result=initial_network_result,
        recirculation_receipt={"co2_captured_mol": 0.0, "water_condensed_mol": 0.0},
        dt_seconds=float(scenario.data["dt_seconds"]),
    )
    initial_feedback, _ = measure_operational_feedback(
        scenario,
        truth=feedback_truth,
        step=0,
    )
    return replace(
        initial_state_value,
        utility=replace(
            initial_state_value.utility,
            last_operational_feedback=initial_feedback,
        ),
    )


def _advance_one_step_canonical(
    scenario: Scenario,
    state: PlantState,
    *,
    command_override: CanonicalExternalCommand | None = None,
    external_command_digest: str | None = None,
) -> StepResult:
    if state.step >= scenario.data["steps"]:
        raise ScenarioValidationError("cannot advance beyond configured steps")

    segment = _segment_for_step(scenario, state.step)
    dt_seconds = float(scenario.data["dt_seconds"])
    next_zones: dict[str, ZoneState] = {}
    source_receipt: dict[str, dict[str, float]] = {}
    for zone_id, zone in state.zones.items():
        load = segment["loads"][zone_id]
        co2_added_mol = float(load["co2_generation_mol_s"]) * dt_seconds
        o2_consumed_mol = float(load["o2_consumption_mol_s"]) * dt_seconds
        water_added_mol = float(load["water_vapor_generation_mol_s"]) * dt_seconds
        if o2_consumed_mol > zone.o2_mol:
            raise InfeasibleActionError(
                f"oxygen load exceeds inventory in zone {zone_id}"
            )
        next_zones[zone_id] = replace(
            zone,
            co2_mol=zone.co2_mol + co2_added_mol,
            o2_mol=zone.o2_mol - o2_consumed_mol,
            water_vapor_mol=zone.water_vapor_mol + water_added_mol,
        )
        source_receipt[zone_id] = {
            "co2_added_mol": co2_added_mol,
            "o2_consumed_mol": o2_consumed_mol,
            "water_added_mol": water_added_mol,
        }

    equipment = scenario.data["equipment"]
    canonical_command = (
        validate_external_command(scenario, segment["command"])
        if command_override is None
        else command_override
    )
    if canonical_command.scenario_schema_version != scenario.scenario_schema_version:
        raise ScenarioValidationError(
            "canonical external command schema does not match scenario"
        )
    if external_command_sha256(canonical_command) != canonical_command.sha256:
        raise ScenarioValidationError("canonical external command digest mismatch")
    command = canonical_command.to_mapping()
    network_result: AirNetworkResult | None = None
    if scenario.scenario_schema_version in {
        SCENARIO_SCHEMA_VERSION_V3,
        SCENARIO_SCHEMA_VERSION_V4,
        SCENARIO_SCHEMA_VERSION_V5,
    }:
        network = scenario.data["air_network"]
        current_fan_speed = state.utility.actual_fan_speed_fraction
        if current_fan_speed is None:
            raise ScenarioValidationError(
                "scenario-v3 state is missing fan actuator state"
            )
        actual_fan_speed = _slew(
            current_fan_speed,
            float(command["fan_speed_fraction"]),
            float(network["fan"]["speed_slew_fraction_per_s"]) * dt_seconds,
        )
        branch_by_damper = {
            str(branch["damper_id"]): branch for branch in network["branches"]
        }
        physical_faults = physical_fault_effects(
            scenario,
            emitted_step=state.step + 1,
            previous_damper_position_by_id=(state.utility.actual_damper_position_by_id),
        )
        jammed_damper_ids = set(physical_faults.jammed_damper_ids)
        actual_dampers = {
            damper_id: (
                float(state.utility.actual_damper_position_by_id[damper_id])
                if damper_id in jammed_damper_ids
                else _slew(
                    float(state.utility.actual_damper_position_by_id[damper_id]),
                    float(command["damper_position_by_id"][damper_id]),
                    float(branch_by_damper[damper_id]["damper_slew_fraction_per_s"])
                    * dt_seconds,
                )
            )
            for damper_id in sorted(branch_by_damper)
        }
        effective_fan_speed = actual_fan_speed * physical_faults.fan_speed_multiplier
        network_result = solve_air_network(
            _air_network_spec(
                scenario,
                open_supply_resistance_multiplier_by_zone=(
                    physical_faults.open_supply_resistance_multiplier_by_zone
                ),
            ),
            fan_speed_fraction=effective_fan_speed,
            damper_position_by_id=actual_dampers,
        )
        actual_airflow_m3_s = dict(network_result.zone_flow_m3_s)
    else:
        actual_fan_speed = None
        actual_dampers = {}
        maximum_flow_delta = float(equipment["airflow_slew_m3_s2"]) * dt_seconds
        actual_airflow_m3_s = {
            zone_id: _slew(
                float(state.utility.actual_airflow_m3_s[zone_id]),
                float(command["airflow_m3_s"][zone_id]),
                maximum_flow_delta,
            )
            for zone_id in sorted(next_zones)
        }
    actual_scrubber_duty = _slew(
        state.utility.actual_scrubber_duty,
        float(command["scrubber_duty"]),
        float(equipment["scrubber_duty_slew_per_s"]) * dt_seconds,
    )
    actual_condenser_duty = _slew(
        state.utility.actual_condenser_duty,
        float(command["condenser_duty"]),
        float(equipment["condenser_duty_slew_per_s"]) * dt_seconds,
    )
    if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5:
        achievement = achieve_actuator_state(
            current_cooling_removed_w=state.utility.actual_cooling_removed_w,
            current_oxygen_injection_mol_s=state.utility.actual_oxygen_injection_mol_s,
            requested_cooling_removed_w=command["cooling_removed_w"],
            requested_oxygen_injection_mol_s=command["oxygen_injection_mol_s"],
            cooling_slew_w_per_s=float(
                scenario.data["actuator_feedback"]["cooling_slew_w_per_s"]
            ),
            oxygen_slew_mol_s2=float(
                scenario.data["actuator_feedback"]["oxygen_slew_mol_s2"]
            ),
            dt_seconds=dt_seconds,
            oxygen_max_total_mol_s=float(
                equipment["oxygen_injection_max_total_mol_s"]
            ),
        )
        achieved_cooling_removed_w = achievement.cooling_removed_w
        achieved_oxygen_injection_mol_s = achievement.oxygen_injection_mol_s
        effective_cooling_delivery_by_zone = {
            zone_id: value
            * float(
                physical_faults.cooling_delivery_multiplier_by_zone.get(zone_id, 1.0)
            )
            for zone_id, value in achieved_cooling_removed_w.items()
        }
        effective_oxygen_delivery_by_zone = {
            zone_id: value
            * float(
                physical_faults.oxygen_delivery_multiplier_by_zone.get(zone_id, 1.0)
            )
            for zone_id, value in achieved_oxygen_injection_mol_s.items()
        }
        scrubber_capture_ability = physical_faults.scrubber_capture_multiplier
        condenser_removal_ability = physical_faults.condenser_removal_multiplier
    else:
        achieved_cooling_removed_w = command["cooling_removed_w"]
        achieved_oxygen_injection_mol_s = command["oxygen_injection_mol_s"]
        effective_cooling_delivery_by_zone = achieved_cooling_removed_w
        effective_oxygen_delivery_by_zone = achieved_oxygen_injection_mol_s
        scrubber_capture_ability = 1.0
        condenser_removal_ability = 1.0
    zone_configs = {str(zone["id"]): zone for zone in scenario.data["zones"]}
    next_zones, recirculation_receipt = _recirculate(
        next_zones,
        zone_configs=zone_configs,
        airflow_m3_s=actual_airflow_m3_s,
        scrubber_duty=actual_scrubber_duty,
        condenser_duty=actual_condenser_duty,
        equipment=equipment,
        sorbent_remaining_mol=state.utility.co2_sorbent_remaining_mol,
        dt_seconds=dt_seconds,
        scrubber_capture_ability=scrubber_capture_ability,
        condenser_removal_ability=condenser_removal_ability,
        include_effectiveness_receipt=(
            scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
        ),
    )
    co2_captured_mol = float(recirculation_receipt["co2_captured_mol"])
    water_condensed_mol = float(recirculation_receipt["water_condensed_mol"])
    oxygen_injected_by_zone = {
        zone_id: float(effective_oxygen_delivery_by_zone[zone_id]) * dt_seconds
        for zone_id in sorted(next_zones)
    }
    total_oxygen_injected_mol = sum(oxygen_injected_by_zone.values())
    if total_oxygen_injected_mol > state.utility.oxygen_store_mol:
        raise InfeasibleActionError("oxygen injection exceeds stored oxygen")
    next_zones = {
        zone_id: replace(
            zone,
            o2_mol=zone.o2_mol + oxygen_injected_by_zone[zone_id],
        )
        for zone_id, zone in next_zones.items()
    }
    recirculation_heat_added_j = _recirculation_heat_transfer_j(
        next_zones,
        airflow_m3_s=actual_airflow_m3_s,
        equipment=equipment,
        dt_seconds=dt_seconds,
    )
    next_zones, thermal_receipt = _apply_zone_thermal_balance(
        next_zones,
        zone_configs=zone_configs,
        loads=segment["loads"],
        cooling_removed_w=effective_cooling_delivery_by_zone,
        recirculation_heat_added_j=recirculation_heat_added_j,
        dt_seconds=dt_seconds,
    )
    next_zones, passive_condensation_mol = _apply_passive_condensation(
        next_zones,
        zone_configs=zone_configs,
    )
    total_passive_condensation_mol = sum(passive_condensation_mol.values())
    next_battery_energy_wh, electrical_receipt = _electrical_balance(
        battery_energy_wh=state.utility.battery_energy_wh,
        equipment=equipment,
        command=command,
        actual_airflow_m3_s=actual_airflow_m3_s,
        fan_electrical_power_w=(
            None if network_result is None else network_result.fan_electrical_power_w
        ),
        actual_scrubber_duty=actual_scrubber_duty,
        actual_condenser_duty=actual_condenser_duty,
        generation_w=float(segment["generation_w"]),
        dt_seconds=dt_seconds,
        achieved_cooling_removed_w=(
            achieved_cooling_removed_w
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else None
        ),
        achieved_oxygen_injection_mol_s=(
            achieved_oxygen_injection_mol_s
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else None
        ),
    )
    electrical_heat_rejected_j = (
        electrical_receipt["served_load_wh"]
        + electrical_receipt["charge_conversion_loss_wh"]
        + electrical_receipt["discharge_conversion_loss_wh"]
    ) * 3600.0
    thermal_receipt = {
        **thermal_receipt,
        "external_heat_rejected_j": (
            float(thermal_receipt["external_heat_rejected_j"])
            + electrical_heat_rejected_j
        ),
    }
    species_before = {
        "co2_mol": sum(zone.co2_mol for zone in state.zones.values()),
        "o2_mol": sum(zone.o2_mol for zone in state.zones.values()),
        "water_mol": sum(zone.water_vapor_mol for zone in state.zones.values()),
        "inert_mol": sum(zone.inert_mol for zone in state.zones.values()),
    }
    species_after = {
        "co2_mol": sum(zone.co2_mol for zone in next_zones.values()),
        "o2_mol": sum(zone.o2_mol for zone in next_zones.values()),
        "water_mol": sum(zone.water_vapor_mol for zone in next_zones.values()),
        "inert_mol": sum(zone.inert_mol for zone in next_zones.values()),
    }
    expected_species_delta = {
        "co2_mol": (
            sum(item["co2_added_mol"] for item in source_receipt.values())
            - co2_captured_mol
        ),
        "o2_mol": (
            -sum(item["o2_consumed_mol"] for item in source_receipt.values())
            + total_oxygen_injected_mol
        ),
        "water_mol": (
            sum(item["water_added_mol"] for item in source_receipt.values())
            - water_condensed_mol
            - total_passive_condensation_mol
        ),
        "inert_mol": 0.0,
    }
    species_receipt_scale_mol = max(
        1.0, *(abs(value) for value in expected_species_delta.values())
    )
    species_tolerance_mol = max(1e-12, 1e-10 * species_receipt_scale_mol)
    species_accounting = {
        "co2_residual_mol": (
            species_after["co2_mol"]
            - species_before["co2_mol"]
            - expected_species_delta["co2_mol"]
        ),
        "o2_residual_mol": (
            species_after["o2_mol"]
            - species_before["o2_mol"]
            - expected_species_delta["o2_mol"]
        ),
        "water_residual_mol": (
            species_after["water_mol"]
            - species_before["water_mol"]
            - expected_species_delta["water_mol"]
        ),
        "inert_residual_mol": (
            species_after["inert_mol"] - species_before["inert_mol"]
        ),
        "receipt_scale_mol": species_receipt_scale_mol,
        "tolerance_mol": species_tolerance_mol,
    }

    next_utility = replace(
        state.utility,
        co2_sorbent_remaining_mol=(
            state.utility.co2_sorbent_remaining_mol - co2_captured_mol
        ),
        captured_co2_mol=state.utility.captured_co2_mol + co2_captured_mol,
        condensed_water_mol=(
            state.utility.condensed_water_mol
            + water_condensed_mol
            + total_passive_condensation_mol
        ),
        oxygen_store_mol=(state.utility.oxygen_store_mol - total_oxygen_injected_mol),
        battery_energy_wh=next_battery_energy_wh,
        external_heat_received_j=(
            state.utility.external_heat_received_j
            + float(thermal_receipt["external_heat_received_j"])
        ),
        external_heat_rejected_j=(
            state.utility.external_heat_rejected_j
            + float(thermal_receipt["external_heat_rejected_j"])
        ),
        actual_airflow_m3_s=actual_airflow_m3_s,
        actual_scrubber_duty=actual_scrubber_duty,
        actual_condenser_duty=actual_condenser_duty,
        actual_fan_speed_fraction=actual_fan_speed,
        actual_damper_position_by_id=actual_dampers,
        actual_cooling_removed_w=(
            {
                zone_id: float(value)
                for zone_id, value in achieved_cooling_removed_w.items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else state.utility.actual_cooling_removed_w
        ),
        actual_oxygen_injection_mol_s=(
            {
                zone_id: float(value)
                for zone_id, value in achieved_oxygen_injection_mol_s.items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else state.utility.actual_oxygen_injection_mol_s
        ),
        effective_scrubber_capture_ability=(
            scrubber_capture_ability
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else state.utility.effective_scrubber_capture_ability
        ),
        effective_condenser_removal_ability=(
            condenser_removal_ability
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else state.utility.effective_condenser_removal_ability
        ),
        effective_cooling_delivery_by_zone=(
            {
                zone_id: float(value)
                for zone_id, value in effective_cooling_delivery_by_zone.items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else state.utility.effective_cooling_delivery_by_zone
        ),
        effective_oxygen_delivery_by_zone=(
            {
                zone_id: float(value)
                for zone_id, value in effective_oxygen_delivery_by_zone.items()
            }
            if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5
            else state.utility.effective_oxygen_delivery_by_zone
        ),
    )

    step_receipt: dict[str, Any] = {
        "species_sources": source_receipt,
        "species_accounting": species_accounting,
        "recirculation": recirculation_receipt,
        "oxygen_injected_mol": oxygen_injected_by_zone,
        "passive_condensation_mol": passive_condensation_mol,
        "thermal": thermal_receipt,
        "electrical": electrical_receipt,
    }
    if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5:
        step_receipt["actuators"] = {
            "fan": {
                "requested_fraction": float(command["fan_speed_fraction"]),
                "achieved_fraction": float(actual_fan_speed),
                "effective_fraction": float(effective_fan_speed),
            },
            "dampers": {
                "requested_by_id": {
                    damper_id: float(command["damper_position_by_id"][damper_id])
                    for damper_id in sorted(command["damper_position_by_id"])
                },
                "achieved_by_id": {
                    damper_id: float(actual_dampers[damper_id])
                    for damper_id in sorted(actual_dampers)
                },
                "effective_by_id": {
                    damper_id: float(actual_dampers[damper_id])
                    for damper_id in sorted(actual_dampers)
                },
            },
            "scrubber": {
                "requested_duty": float(command["scrubber_duty"]),
                "achieved_duty": float(actual_scrubber_duty),
                "effectiveness_multiplier": float(scrubber_capture_ability),
                "effective_duty": float(actual_scrubber_duty)
                * float(scrubber_capture_ability),
            },
            "condenser": {
                "requested_duty": float(command["condenser_duty"]),
                "achieved_duty": float(actual_condenser_duty),
                "effectiveness_multiplier": float(condenser_removal_ability),
                "effective_duty": float(actual_condenser_duty)
                * float(condenser_removal_ability),
            },
            "cooling": {
                "requested_w": {
                    zone_id: float(command["cooling_removed_w"][zone_id])
                    for zone_id in sorted(command["cooling_removed_w"])
                },
                "achieved_w": {
                    zone_id: float(achieved_cooling_removed_w[zone_id])
                    for zone_id in sorted(achieved_cooling_removed_w)
                },
                "effective_w": {
                    zone_id: float(effective_cooling_delivery_by_zone[zone_id])
                    for zone_id in sorted(effective_cooling_delivery_by_zone)
                },
            },
            "oxygen": {
                "requested_mol_s": {
                    zone_id: float(command["oxygen_injection_mol_s"][zone_id])
                    for zone_id in sorted(command["oxygen_injection_mol_s"])
                },
                "achieved_mol_s": {
                    zone_id: float(achieved_oxygen_injection_mol_s[zone_id])
                    for zone_id in sorted(achieved_oxygen_injection_mol_s)
                },
                "effective_mol_s": {
                    zone_id: float(effective_oxygen_delivery_by_zone[zone_id])
                    for zone_id in sorted(effective_oxygen_delivery_by_zone)
                },
            },
        }
    if network_result is not None and actual_fan_speed is not None:
        step_receipt["air_network"] = _air_network_receipt(
            network_result,
            requested_fan_speed_fraction=float(command["fan_speed_fraction"]),
            actual_fan_speed_fraction=actual_fan_speed,
            effective_fan_speed_fraction=(
                effective_fan_speed
                if scenario.scenario_schema_version
                in {SCENARIO_SCHEMA_VERSION_V4, SCENARIO_SCHEMA_VERSION_V5}
                else None
            ),
            requested_damper_position_by_id=command["damper_position_by_id"],
            actual_damper_position_by_id=actual_dampers,
        )
    if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V4:
        step_receipt["active_faults"] = list(physical_faults.active_faults)

    next_state = PlantState(
        step=state.step + 1,
        zones=next_zones,
        utility=next_utility,
    )
    if scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5:
        feedback_truth = _operational_feedback_truth(
            scenario,
            next_state,
            network_result=network_result,
            recirculation_receipt=recirculation_receipt,
            dt_seconds=dt_seconds,
            effective_fan_speed_fraction=effective_fan_speed,
        )
        operational_feedback, feedback_faults = measure_operational_feedback(
            scenario,
            truth=feedback_truth,
            step=next_state.step,
            previous=state.utility.last_operational_feedback,
        )
        next_state = replace(
            next_state,
            utility=replace(
                next_state.utility,
                last_operational_feedback=operational_feedback,
            ),
        )
        step_receipt["operational_feedback"] = operational_feedback
        step_receipt["active_faults"] = sorted(
            [dict(value) for value in physical_faults.active_faults]
            + [dict(value) for value in feedback_faults],
            key=lambda value: str(value["fault_id"]),
        )
        step_receipt["realised_loads"] = segment["loads"]
        if external_command_digest is not None:
            step_receipt["external_command_digest"] = external_command_digest

    return StepResult(
        state=next_state,
        receipt=step_receipt,
    )


def _external_command_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioValidationError(
            f"external command {path} must be finite numeric data"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ScenarioValidationError(
            f"external command {path} must be finite numeric data"
        )
    return number


def _external_command_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ScenarioValidationError(f"external command {path} must be an object")
    return value


def _canonical_command_mapping_bytes(command: Mapping[str, Any]) -> bytes:
    return json.dumps(
        command,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_external_command_bytes(command: CanonicalExternalCommand) -> bytes:
    if type(command) is not CanonicalExternalCommand:
        raise TypeError("canonical command identity requires CanonicalExternalCommand")
    return command.canonical_bytes


def external_command_sha256(command: CanonicalExternalCommand) -> str:
    return hashlib.sha256(canonical_external_command_bytes(command)).hexdigest()


_PREFLIGHT_CONTRACT = {
    "schema_version": "aeolus_habitat_v2_hmc_preflight_v1",
    "classifications": ["FEASIBLE", "INFEASIBLE"],
    "fields": [
        "classification",
        "application_step",
        "command_sha256",
        "preflight_contract_sha256",
        "preflight_result_sha256",
    ],
}
_PREFLIGHT_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_command_mapping_bytes(_PREFLIGHT_CONTRACT)
).hexdigest()


def validate_external_command(
    scenario: Scenario, command: Mapping[str, Any]
) -> CanonicalExternalCommand:
    if not isinstance(command, Mapping):
        raise ScenarioValidationError("external command must be an object")
    expected_fields = set(command_fields_for_schema(scenario.scenario_schema_version))
    unknown = sorted(set(command) - expected_fields)
    missing = sorted(expected_fields - set(command))
    if unknown or missing:
        raise ScenarioValidationError(
            f"invalid external command fields; unknown={unknown}, missing={missing}"
        )
    zone_ids = tuple(sorted(str(zone["id"]) for zone in scenario.data["zones"]))
    zone_id_set = set(zone_ids)
    normalised: dict[str, Any] = {
        "scrubber_duty": _external_command_number(
            command["scrubber_duty"], path="scrubber_duty"
        ),
        "condenser_duty": _external_command_number(
            command["condenser_duty"], path="condenser_duty"
        ),
    }
    for field in ("cooling_removed_w", "oxygen_injection_mol_s"):
        values = _external_command_mapping(command[field], path=field)
        if set(values) != zone_id_set:
            raise ScenarioValidationError(f"external command {field} topology mismatch")
        normalised[field] = {
            zone_id: _external_command_number(
                values[zone_id], path=f"{field}.{zone_id}"
            )
            for zone_id in zone_ids
        }
    if scenario.scenario_schema_version in {
        SCENARIO_SCHEMA_VERSION_V3,
        SCENARIO_SCHEMA_VERSION_V4,
        SCENARIO_SCHEMA_VERSION_V5,
    }:
        damper_ids = tuple(
            sorted(
                str(branch["damper_id"])
                for branch in scenario.data["air_network"]["branches"]
            )
        )
        damper_values = _external_command_mapping(
            command["damper_position_by_id"], path="damper_position_by_id"
        )
        if set(damper_values) != set(damper_ids):
            raise ScenarioValidationError("external command damper topology mismatch")
        normalised["fan_speed_fraction"] = _external_command_number(
            command["fan_speed_fraction"], path="fan_speed_fraction"
        )
        normalised["damper_position_by_id"] = {
            damper_id: _external_command_number(
                damper_values[damper_id], path=f"damper_position_by_id.{damper_id}"
            )
            for damper_id in damper_ids
        }
    else:
        airflow_values = _external_command_mapping(
            command["airflow_m3_s"], path="airflow_m3_s"
        )
        if set(airflow_values) != zone_id_set:
            raise ScenarioValidationError("external command airflow topology mismatch")
        normalised["airflow_m3_s"] = {
            zone_id: _external_command_number(
                airflow_values[zone_id], path=f"airflow_m3_s.{zone_id}"
            )
            for zone_id in zone_ids
        }
    equipment = scenario.data["equipment"]
    for field in ("scrubber_duty", "condenser_duty"):
        value = normalised[field]
        if not 0.0 <= value <= 1.0:
            raise ScenarioValidationError(f"external command {field} is out of bounds")
    for field in ("cooling_removed_w", "oxygen_injection_mol_s"):
        for zone_id, value in normalised[field].items():
            if value < 0.0:
                raise ScenarioValidationError(
                    f"external command {field}.{zone_id} is out of bounds"
                )
    if any(
        value > float(equipment["cooling_max_thermal_w_per_zone"])
        for value in normalised["cooling_removed_w"].values()
    ):
        raise ScenarioValidationError("external command cooling exceeds capacity")
    if sum(normalised["oxygen_injection_mol_s"].values()) > float(
        equipment["oxygen_injection_max_total_mol_s"]
    ):
        raise ScenarioValidationError("external command oxygen exceeds capacity")
    if scenario.scenario_schema_version in {
        SCENARIO_SCHEMA_VERSION_V3,
        SCENARIO_SCHEMA_VERSION_V4,
        SCENARIO_SCHEMA_VERSION_V5,
    }:
        for value in normalised["damper_position_by_id"].values():
            if not 0.0 <= value <= 1.0:
                raise ScenarioValidationError(
                    "external command damper position is out of bounds"
                )
        fan_speed = normalised["fan_speed_fraction"]
        if not 0.0 <= fan_speed <= 1.0:
            raise ScenarioValidationError("external command fan speed is out of bounds")
    else:
        airflow = normalised["airflow_m3_s"]
        for value in airflow.values():
            if value < 0.0:
                raise ScenarioValidationError(
                    "external command airflow is out of bounds"
                )
            if value > float(equipment["max_zone_airflow_m3_s"]):
                raise ScenarioValidationError(
                    "external command zone airflow exceeds capacity"
                )
        if sum(airflow.values()) > float(equipment["max_total_airflow_m3_s"]):
            raise ScenarioValidationError(
                "external command total airflow exceeds capacity"
            )
    canonical_bytes = _canonical_command_mapping_bytes(normalised)
    canonical = CanonicalExternalCommand(
        scenario_schema_version=scenario.scenario_schema_version,
        canonical_bytes=canonical_bytes,
        sha256=hashlib.sha256(canonical_bytes).hexdigest(),
    )
    if canonical.sha256 != external_command_sha256(canonical):
        raise AssertionError("canonical external command digest mismatch")
    return canonical


def command_from_achieved_state(
    scenario: Scenario, state: PlantState
) -> AchievedStateCommandReference:
    if scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
        raise ScenarioValidationError("achieved-state HMC hold requires V5")
    if set(state.zones) != {str(zone["id"]) for zone in scenario.data["zones"]}:
        raise ScenarioValidationError("achieved-state hold topology mismatch")
    fan_speed = state.utility.actual_fan_speed_fraction
    if fan_speed is None:
        raise ScenarioValidationError("achieved-state hold requires fan state")
    command = validate_external_command(
        scenario,
        {
            "fan_speed_fraction": fan_speed,
            "damper_position_by_id": dict(state.utility.actual_damper_position_by_id),
            "scrubber_duty": state.utility.actual_scrubber_duty,
            "condenser_duty": state.utility.actual_condenser_duty,
            "cooling_removed_w": dict(state.utility.actual_cooling_removed_w),
            "oxygen_injection_mol_s": dict(state.utility.actual_oxygen_injection_mol_s),
        },
    )
    return AchievedStateCommandReference(
        command_reference_kind="INITIAL_ACHIEVED_STATE_HOLD",
        command=command,
    )


def _preflight_result(
    *,
    classification: str,
    application_step: int,
    command_sha256: str,
) -> PreflightResult:
    content = {
        "classification": classification,
        "application_step": application_step,
        "command_sha256": command_sha256,
        "preflight_contract_sha256": _PREFLIGHT_CONTRACT_SHA256,
    }
    digest = hashlib.sha256(_canonical_command_mapping_bytes(content)).hexdigest()
    return PreflightResult(
        **content,
        preflight_result_sha256=digest,
    )


def preflight_external_command(
    scenario: Scenario,
    state: PlantState,
    command: Mapping[str, Any],
    application_step: int,
) -> PreflightResult:
    if (
        isinstance(application_step, bool)
        or not isinstance(application_step, int)
        or application_step < 0
        or application_step != state.step
    ):
        raise ScenarioValidationError(
            "preflight application step must equal current state step"
        )
    canonical = validate_external_command(scenario, command)
    try:
        _advance_one_step_canonical(
            scenario,
            state,
            command_override=canonical,
            external_command_digest=canonical.sha256,
        )
    except (InfeasibleActionError, ScenarioValidationError):
        classification = "INFEASIBLE"
    else:
        classification = "FEASIBLE"
    return _preflight_result(
        classification=classification,
        application_step=application_step,
        command_sha256=canonical.sha256,
    )


def operating_mode_for_application_step(
    scenario: Scenario,
    application_step: int,
) -> str:
    if (
        isinstance(application_step, bool)
        or not isinstance(application_step, int)
        or application_step < 0
    ):
        raise ScenarioValidationError("application step must be a non-negative integer")
    segment = _segment_for_step(scenario, application_step)
    mode = segment.get("operating_mode")
    if type(mode) is not str:
        raise ScenarioValidationError("application step is missing an operating mode")
    return mode


def advance_one_step(scenario: Scenario, state: PlantState) -> StepResult:
    return _advance_one_step_canonical(scenario, state)


def advance_one_step_with_command(
    scenario: Scenario,
    state: PlantState,
    command: Mapping[str, Any],
) -> StepResult:
    if state.step >= scenario.data["steps"]:
        raise ScenarioValidationError("cannot advance beyond configured steps")
    validated_command = validate_external_command(scenario, command)
    return _advance_one_step_canonical(
        scenario,
        state,
        command_override=validated_command,
        external_command_digest=validated_command.sha256,
    )


def validate_external_step_result(
    scenario: Scenario,
    pre_step_state: PlantState,
    command: Mapping[str, Any],
    candidate: StepResult,
) -> None:
    if type(scenario) is not Scenario or type(pre_step_state) is not PlantState:
        raise ScenarioValidationError("external step validation requires exact inputs")
    if type(candidate) is not StepResult or not isinstance(candidate.receipt, Mapping):
        raise ScenarioValidationError("external step candidate is malformed")
    try:
        candidate_receipt_bytes = _canonical_command_mapping_bytes(candidate.receipt)
    except (TypeError, ValueError) as error:
        raise ScenarioValidationError(
            "external step receipt is not finite canonical JSON"
        ) from error
    replay = advance_one_step_with_command(scenario, pre_step_state, command)
    replay_receipt_bytes = _canonical_command_mapping_bytes(replay.receipt)
    if (
        candidate.state != replay.state
        or candidate_receipt_bytes != replay_receipt_bytes
    ):
        raise ScenarioValidationError("external step result fails causal replay")
