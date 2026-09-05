"""Bounded one-at-a-time parameter sensitivity for Habitat V2 (Issue #71).

Replays the frozen forecast development fixture as race families under the
no-proposal hold policy, perturbing one generator-variable parameter at a time
to its declared band edges (`contracts/habitat_v2_physics_provenance_v1.json`)
and reporting the effect on decision-relevant outputs. Parameters whose
perturbation moves total safety exposure by at least the declared reversal
share, or flips crossing-event presence, are flagged for the Scenario Family
Generator v2 (Issue #72).

The crossing metric is computed inline against the declared evaluation bounds
rather than reusing a production scorer, keeping this analysis an independent
measurement. Output is a write-once JSON receipt under `out/` plus a ranked
table on stdout.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast_issue55_race import (
    RESOURCE_FIELD_BOUNDS,
    ZONE_FIELD_BOUNDS,
    N_ZONES,
    build_family_scenario,
    deterministic_family_ids,
    episode_nonce,
    project_true_targets,
    scenario_zone_order,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics_provenance import (
    load_physics_provenance_manifest,
    parameter_by_id,
    sample_band,
)
from aeolus.habitat_v2.scenario import Scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 96
BASE_FAMILY_INDICES = (0, 3)
REVERSAL_EXPOSURE_SHARE = 0.25
OUTPUT_DIR = REPO_ROOT / "out" / "habitat-v2-parameter-sensitivity"
RECEIPT_PATH = OUTPUT_DIR / "sensitivity-receipt.json"


def _bounds_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scales: list[float] = []
    nominals: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for _zone in range(N_ZONES):
        for _name, scale, nominal, lower, upper in ZONE_FIELD_BOUNDS:
            scales.append(scale)
            nominals.append(nominal)
            lowers.append(lower)
            uppers.append(upper)
    for _name, scale, nominal, lower, upper in RESOURCE_FIELD_BOUNDS:
        scales.append(scale)
        nominals.append(nominal)
        lowers.append(lower)
        uppers.append(upper)
    return (
        np.array(scales),
        np.array(nominals),
        np.array(lowers),
        np.array(uppers),
    )


SCALES, _NOMINALS, LOWERS, UPPERS = _bounds_arrays()
O2_COLUMN_START = 3
O2_COLUMN_STRIDE = 6
CO2_COLUMN_START = 2
TEMP_COLUMN_START = 0


def _crossing_metrics(rows: np.ndarray) -> dict[str, Any]:
    excess = np.maximum(
        np.maximum(LOWERS - rows, rows - UPPERS), 0.0
    ) / SCALES
    per_step = excess.sum(axis=1)
    o2_columns = [
        zone * O2_COLUMN_STRIDE + O2_COLUMN_START for zone in range(N_ZONES)
    ]
    co2_columns = [
        zone * O2_COLUMN_STRIDE + CO2_COLUMN_START for zone in range(N_ZONES)
    ]
    temp_columns = [zone * O2_COLUMN_STRIDE + TEMP_COLUMN_START for zone in range(N_ZONES)]
    return {
        "total_safety_exposure": float(per_step.sum()),
        "violation_steps": int(np.count_nonzero(per_step > 0.0)),
        "eventful": bool(np.any(per_step > 0.0)),
        "o2_upper_margin_min": float(np.min(0.30 - rows[:, o2_columns].max(axis=1))),
        "co2_upper_margin_ppm_min": float(np.min(5000.0 - rows[:, co2_columns].max(axis=1))),
        "temp_margin_min": float(
            np.minimum(
                330.0 - rows[:, temp_columns].max(axis=1),
                rows[:, temp_columns].min(axis=1) - 250.0,
            ).min()
        ),
        "battery_soc_delta": float(rows[-1, -3] - rows[0, -3]),
        "o2_store_delta": float(rows[-1, -2] - rows[0, -2]),
        "sorbent_delta": float(rows[-1, -1] - rows[0, -1]),
    }


def _replay(bundle: Any, scenario: Scenario, family_id: str) -> dict[str, Any]:
    zone_ids = scenario_zone_order(scenario)
    hmc = HabitatManagementComputer.reset(
        scenario, bundle.hmc_contract, episode_nonce(family_id)
    )
    rows = []
    for _step in range(EPISODE_STEPS):
        snapshot, verification = hmc.observe()
        handle = hmc.verify_snapshot(snapshot, verification)
        rows.append(project_true_targets(scenario, zone_ids, hmc._state))
        hmc.propose(None, handle)
        hmc.arbitrate()
        hmc.step()
    return _crossing_metrics(np.stack(rows).astype(np.float64))


def _all_zones(data: dict) -> list[dict]:
    return data["zones"]


def _metabolic_factor(record: dict, direction: float) -> float:
    perturbed = sample_band(record, direction)
    return perturbed / float(record["value"])


APPLIERS: dict[str, Any] = {
    "zone_volume_m3": lambda data, value: [
        zone.update(volume_m3=value) for zone in _all_zones(data)
    ],
    "zone_thermal_capacity_j_per_k": lambda data, value: [
        zone.update(thermal_capacity_j_per_k=value) for zone in _all_zones(data)
    ],
    "passive_thermal_conductance_w_per_k": lambda data, value: [
        zone.update(passive_thermal_conductance_w_per_k=value) for zone in _all_zones(data)
    ],
    "sink_temperature_k": lambda data, value: [
        zone.update(sink_temperature_k=value) for zone in _all_zones(data)
    ],
    "initial_pressure_pa": lambda data, value: [
        zone["initial"].update(pressure_pa=value) for zone in _all_zones(data)
    ],
    "initial_co2_ppm": lambda data, value: [
        zone["initial"].update(co2_ppm=value) for zone in _all_zones(data)
    ],
    "initial_relative_humidity": lambda data, value: [
        zone["initial"].update(relative_humidity=value) for zone in _all_zones(data)
    ],
    "initial_temperature_k": lambda data, value: [
        zone["initial"].update(temperature_k=value) for zone in _all_zones(data)
    ],
    "initial_battery_energy_wh": lambda data, value: data["initial_utility"].update(
        battery_energy_wh=value
    ),
    "initial_oxygen_store_mol": lambda data, value: data["initial_utility"].update(
        oxygen_store_mol=value
    ),
    "initial_co2_sorbent_mol": lambda data, value: data["initial_utility"].update(
        co2_sorbent_remaining_mol=value
    ),
    "air_density_kg_m3": lambda data, value: data["equipment"].update(
        air_density_kg_m3=value
    ),
    "base_load_w": lambda data, value: data["equipment"].update(base_load_w=value),
    "battery_capacity_wh": lambda data, value: data["equipment"].update(
        battery_capacity_wh=value
    ),
    "condenser_max_water_mol_s": lambda data, value: data["equipment"].update(
        condenser_max_water_mol_s=value
    ),
    "cooling_coefficient_of_performance": lambda data, value: data["equipment"].update(
        cooling_coefficient_of_performance=value
    ),
    "cooling_max_thermal_w_per_zone": lambda data, value: data["equipment"].update(
        cooling_max_thermal_w_per_zone=value
    ),
    "oxygen_injection_max_total_mol_s": lambda data, value: data["equipment"].update(
        oxygen_injection_max_total_mol_s=value
    ),
    "scrubber_capacity_mol": lambda data, value: data["equipment"].update(
        scrubber_capacity_mol=value
    ),
    "scrubber_max_co2_mol_s": lambda data, value: data["equipment"].update(
        scrubber_max_co2_mol_s=value
    ),
    "fan_rated_free_delivery_m3_s": lambda data, value: data["air_network"][
        "fan"
    ].update(rated_free_delivery_m3_s=value),
    "fan_rated_shutoff_pressure_pa": lambda data, value: data["air_network"][
        "fan"
    ].update(rated_shutoff_pressure_pa=value),
    "fan_total_efficiency": lambda data, value: data["air_network"]["fan"].update(
        total_efficiency=value
    ),
    "branch_open_supply_resistance_pa_s2_m6": lambda data, value: [
        branch.update(open_supply_resistance_pa_s2_m6=value)
        for branch in data["air_network"]["branches"]
    ],
    "branch_return_resistance_pa_s2_m6": lambda data, value: [
        branch.update(return_resistance_pa_s2_m6=value)
        for branch in data["air_network"]["branches"]
    ],
    "damper_leak_fraction": lambda data, value: [
        branch.update(damper_leak_fraction=value)
        for branch in data["air_network"]["branches"]
    ],
    "shared_supply_trunk_resistance_pa_s2_m6": lambda data, value: data[
        "air_network"
    ]["shared_resistance"].update(supply_trunk_pa_s2_m6=value),
    "shared_filter_resistance_pa_s2_m6": lambda data, value: data["air_network"][
        "shared_resistance"
    ].update(filter_pa_s2_m6=value),
}

LOAD_FACTOR_PARAMETERS = {
    "occupied_zone_co2_generation_mol_s": ("co2_generation_mol_s", (0.00025,)),
    "occupied_zone_o2_consumption_mol_s": ("o2_consumption_mol_s", (0.0003,)),
    "occupied_zone_water_vapor_generation_mol_s": (
        "water_vapor_generation_mol_s",
        (0.0012,),
    ),
    "occupied_zone_sensible_heat_w": ("sensible_heat_w", (110.0,)),
}


def _apply_load_factor(data: dict, key: str, reference_values: tuple[float, ...], factor: float) -> None:
    for segment in data["timeline"]:
        for zone_loads in segment["loads"].values():
            current = float(zone_loads.get(key, 0.0))
            if any(
                abs(current - reference) < 1e-12 for reference in reference_values
            ):
                zone_loads[key] = current * factor


def _perturbed_scenario(
    bundle: Any, family_index: int, parameter_id: str, value: float, record: dict
) -> Scenario:
    scenario = build_family_scenario(bundle.development_scenario, family_index)
    data = copy.deepcopy(scenario.data)
    if parameter_id in APPLIERS:
        APPLIERS[parameter_id](data, value)
        if parameter_id == "scrubber_capacity_mol":
            # Feasibility coupling: the scenario contract requires the initial
            # sorbent charge to fit the declared capacity, so a downward
            # capacity perturbation clamps the initial charge with it.
            charge = float(data["initial_utility"]["co2_sorbent_remaining_mol"])
            data["initial_utility"]["co2_sorbent_remaining_mol"] = min(charge, value)
    elif parameter_id in LOAD_FACTOR_PARAMETERS:
        key, references = LOAD_FACTOR_PARAMETERS[parameter_id]
        _apply_load_factor(data, key, references, value / float(record["value"]))
    else:
        raise KeyError(parameter_id)
    return Scenario.from_mapping(data)


def main() -> int:
    if RECEIPT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {RECEIPT_PATH}")
    bundle = load_forecast_contracts(REPO_ROOT)
    manifest, manifest_sha = load_physics_provenance_manifest(REPO_ROOT)
    roster = deterministic_family_ids(32)

    perturbable = [
        parameter_id
        for parameter_id in list(APPLIERS) + list(LOAD_FACTOR_PARAMETERS)
        if parameter_by_id(manifest, parameter_id)["generator_variable"]
    ]

    baselines: dict[int, dict[str, Any]] = {}
    for family_index in BASE_FAMILY_INDICES:
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        baselines[family_index] = _replay(bundle, scenario, roster[family_index])
        print(f"baseline family {family_index}: {json.dumps(baselines[family_index], sort_keys=True)}")

    rows = []
    for parameter_id in sorted(perturbable):
        record = parameter_by_id(manifest, parameter_id)
        for direction in (-1.0, 1.0):
            value = sample_band(record, direction)
            scenario = _perturbed_scenario(
                bundle, BASE_FAMILY_INDICES[0], parameter_id, value, record
            )
            metrics = _replay(bundle, scenario, roster[BASE_FAMILY_INDICES[0]])
            base = baselines[BASE_FAMILY_INDICES[0]]
            base_exposure = base["total_safety_exposure"]
            exposure_change = metrics["total_safety_exposure"] - base_exposure
            relative_change = (
                exposure_change / base_exposure if base_exposure > 0.0 else 0.0
            )
            reversal = (
                abs(relative_change) >= REVERSAL_EXPOSURE_SHARE
                or metrics["eventful"] != base["eventful"]
            )
            rows.append(
                {
                    "parameter_id": parameter_id,
                    "direction": direction,
                    "perturbed_value": value,
                    "total_safety_exposure": metrics["total_safety_exposure"],
                    "exposure_change": exposure_change,
                    "relative_change": relative_change,
                    "violation_steps": metrics["violation_steps"],
                    "eventful": metrics["eventful"],
                    "o2_upper_margin_min": metrics["o2_upper_margin_min"],
                    "co2_upper_margin_ppm_min": metrics["co2_upper_margin_ppm_min"],
                    "temp_margin_min": metrics["temp_margin_min"],
                    "battery_soc_delta": metrics["battery_soc_delta"],
                    "o2_store_delta": metrics["o2_store_delta"],
                    "sorbent_delta": metrics["sorbent_delta"],
                    "reversal_flag": reversal,
                }
            )
            print(
                f"{parameter_id:44s} dir={direction:+.0f} value={value:12.5g} "
                f"exposure={metrics['total_safety_exposure']:.6f} rel={relative_change:+.2%} "
                f"steps={metrics['violation_steps']:3d} reversal={int(reversal)}"
            )

    receipt = {
        "schema_version": "aeolus_habitat_v2_parameter_sensitivity_receipt_v1",
        "provenance_manifest_sha256": manifest_sha,
        "episode_steps": EPISODE_STEPS,
        "base_family_indices": list(BASE_FAMILY_INDICES),
        "reversal_exposure_share": REVERSAL_EXPOSURE_SHARE,
        "baselines": {str(k): v for k, v in baselines.items()},
        "rows": rows,
    }
    raw = canonical_json_bytes(receipt)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_bytes(raw)
    flagged = sorted({row["parameter_id"] for row in rows if row["reversal_flag"]})
    print("\nreversal-flagged parameters:")
    for parameter_id in flagged:
        print(f"  {parameter_id}")
    print(f"receipt: {RECEIPT_PATH}")
    print(f"receipt sha256: {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
