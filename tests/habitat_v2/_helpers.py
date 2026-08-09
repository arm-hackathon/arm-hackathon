from __future__ import annotations

from copy import deepcopy
from typing import Any


def reference_scenario_mapping() -> dict[str, Any]:
    return {
        "schema_version": "aeolus_habitat_v2_scenario_v1",
        "name": "two-zone-reference",
        "dt_seconds": 60.0,
        "steps": 4,
        "zones": [
            {
                "id": "crew_cabin",
                "volume_m3": 40.0,
                "thermal_capacity_j_per_k": 5_000_000.0,
                "passive_thermal_conductance_w_per_k": 12.0,
                "sink_temperature_k": 280.0,
                "initial": {
                    "temperature_k": 295.15,
                    "pressure_pa": 72_000.0,
                    "co2_ppm": 800.0,
                    "o2_mole_fraction": 0.30,
                    "relative_humidity": 0.45,
                },
            },
            {
                "id": "work_airlock",
                "volume_m3": 30.0,
                "thermal_capacity_j_per_k": 4_000_000.0,
                "passive_thermal_conductance_w_per_k": 10.0,
                "sink_temperature_k": 280.0,
                "initial": {
                    "temperature_k": 295.15,
                    "pressure_pa": 72_000.0,
                    "co2_ppm": 800.0,
                    "o2_mole_fraction": 0.30,
                    "relative_humidity": 0.45,
                },
            },
        ],
        "equipment": {
            "max_total_airflow_m3_s": 0.20,
            "max_zone_airflow_m3_s": 0.15,
            "airflow_slew_m3_s2": 0.01,
            "scrubber_duty_slew_per_s": 0.01,
            "condenser_duty_slew_per_s": 0.01,
            "scrubber_max_co2_mol_s": 0.0010,
            "scrubber_capacity_mol": 1_000.0,
            "scrubber_power_w_full": 500.0,
            "condenser_max_water_mol_s": 0.0030,
            "condenser_power_w_full": 400.0,
            "cooling_max_thermal_w_per_zone": 1_000.0,
            "cooling_coefficient_of_performance": 2.5,
            "oxygen_injection_max_total_mol_s": 0.0010,
            "oxygen_injection_power_w_per_mol_s": 100_000.0,
            "fan_power_w_per_m3_s": 1_000.0,
            "base_load_w": 500.0,
            "battery_capacity_wh": 10_000.0,
            "battery_max_charge_w": 2_000.0,
            "battery_max_discharge_w": 3_000.0,
            "battery_charge_efficiency": 0.95,
            "battery_discharge_efficiency": 0.95,
            "air_density_kg_m3": 0.85,
            "air_specific_heat_j_kg_k": 1_005.0,
        },
        "initial_utility": {
            "co2_sorbent_remaining_mol": 500.0,
            "captured_co2_mol": 0.0,
            "condensed_water_mol": 0.0,
            "oxygen_store_mol": 100.0,
            "battery_energy_wh": 5_000.0,
            "actual_airflow_m3_s": {
                "crew_cabin": 0.0,
                "work_airlock": 0.0,
            },
            "actual_scrubber_duty": 0.0,
            "actual_condenser_duty": 0.0,
            "external_heat_rejected_j": 0.0,
            "external_heat_received_j": 0.0,
        },
        "timeline": [
            {
                "start_step": 0,
                "end_step": 4,
                "generation_w": 3_000.0,
                "loads": {
                    "crew_cabin": {
                        "co2_generation_mol_s": 0.00025,
                        "o2_consumption_mol_s": 0.00030,
                        "water_vapor_generation_mol_s": 0.0012,
                        "sensible_heat_w": 110.0,
                    },
                    "work_airlock": {
                        "co2_generation_mol_s": 0.00025,
                        "o2_consumption_mol_s": 0.00030,
                        "water_vapor_generation_mol_s": 0.0012,
                        "sensible_heat_w": 110.0,
                    },
                },
                "command": {
                    "airflow_m3_s": {
                        "crew_cabin": 0.08,
                        "work_airlock": 0.06,
                    },
                    "scrubber_duty": 0.60,
                    "condenser_duty": 0.50,
                    "cooling_removed_w": {
                        "crew_cabin": 120.0,
                        "work_airlock": 120.0,
                    },
                    "oxygen_injection_mol_s": {
                        "crew_cabin": 0.00030,
                        "work_airlock": 0.00030,
                    },
                },
            }
        ],
    }


def reversed_object_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: reversed_object_keys(nested)
            for key, nested in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [reversed_object_keys(item) for item in value]
    return deepcopy(value)
