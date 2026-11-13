"""Shared fixtures for the issue 92 batched research plant tests.

Only checked-in development scenarios are read.  Scenario mappings are deep
copied and modified in memory; nothing is written to disk.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.batched_plant_contract import (
    CONTROL_FIELDS,
    DISTURBANCE_FIELDS,
    STATE_FIELDS,
    SUPPORTED_ZONE_IDS,
    ZONE_COUNT,
    AchievedControl,
    Disturbance,
    PlantParameters,
    PlantStateArrays,
)
from aeolus.habitat_v2.batched_plant_reference import (
    ScalarReferenceStep,
    scalar_reference_step,
)
from aeolus.habitat_v2.physics import initial_state
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.state import PlantState

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "scenarios"

DEVELOPMENT_SCENARIO = "habitat_v2_forecast_development.json"

# Every checked-in scenario inside the supported domain (schema v5, eight zones).
SUPPORTED_SCENARIO_FILES: tuple[str, ...] = (
    "habitat_v2_forecast_development.json",
    "habitat_v2_actuator_feedback.json",
    "habitat_v2_observability/ambiguous.json",
    "habitat_v2_observability/branch_resistance.json",
    "habitat_v2_observability/damper_jam.json",
    "habitat_v2_observability/fan_degradation.json",
    "habitat_v2_observability/healthy_elevated.json",
    "habitat_v2_observability/healthy_nominal.json",
    "habitat_v2_observability/primary_sensor_drift.json",
    "habitat_v2_observability/primary_sensor_stuck.json",
)

# Checked-in scenarios outside it, with the reason each must be rejected.
UNSUPPORTED_SCENARIO_FILES: tuple[str, ...] = (
    "habitat_v2_reference.json",  # schema v1, two zones
    "habitat_v2_operating_modes.json",  # schema v2, two zones
    "habitat_v2_air_network.json",  # schema v3
    "habitat_v2_compound_faults.json",  # schema v4
)


def scenario_mapping(name: str = DEVELOPMENT_SCENARIO) -> dict[str, Any]:
    return json.loads((SCENARIO_DIR / name).read_text(encoding="utf-8"))


def load_scenario(name: str = DEVELOPMENT_SCENARIO) -> Scenario:
    return Scenario.from_mapping(scenario_mapping(name))


def stress_scenario_mapping(steps: int = 240) -> dict[str, Any]:
    """A long development-fixture variant that exercises more mechanisms.

    Three timeline segments change the requested command, loads and
    generation, so actuators spend many steps slew-limited.  The fault list
    adds the physical faults no checked-in scenario uses (scrubber, condenser
    and oxygen delivery degradation) alongside fan, branch, damper and cooling
    faults.  It remains a closed-schema v5 scenario validated by the
    production parser.
    """

    mapping = copy.deepcopy(scenario_mapping(DEVELOPMENT_SCENARIO))
    base = mapping["timeline"][0]
    first = copy.deepcopy(base)
    second = copy.deepcopy(base)
    third = copy.deepcopy(base)
    cut_a = steps // 3
    cut_b = (2 * steps) // 3
    first.update(start_step=0, end_step=cut_a)
    second.update(start_step=cut_a, end_step=cut_b, generation_w=2600.0)
    third.update(start_step=cut_b, end_step=steps, generation_w=7000.0)

    second["command"]["fan_speed_fraction"] = 0.25
    second["command"]["scrubber_duty"] = 0.2
    second["command"]["condenser_duty"] = 1.0
    for index, zone_id in enumerate(SUPPORTED_ZONE_IDS):
        second["command"]["cooling_removed_w"][zone_id] = 900.0 if index % 2 else 0.0
        second["command"]["damper_position_by_id"][f"{zone_id}_supply_damper"] = (
            0.05 + 0.1 * index
        )
        second["loads"][zone_id]["sensible_heat_w"] = 150.0 + 90.0 * index
        second["loads"][zone_id]["water_vapor_generation_mol_s"] = 0.004
    second["command"]["oxygen_injection_mol_s"] = {
        zone_id: 0.0 for zone_id in SUPPORTED_ZONE_IDS
    }
    second["command"]["oxygen_injection_mol_s"]["laboratory"] = 0.003

    third["command"]["fan_speed_fraction"] = 0.0
    third["command"]["scrubber_duty"] = 1.0
    third["command"]["condenser_duty"] = 0.0
    for zone_id in SUPPORTED_ZONE_IDS:
        third["command"]["cooling_removed_w"][zone_id] = 0.0
        third["loads"][zone_id]["co2_generation_mol_s"] = 0.0011

    mapping["steps"] = steps
    mapping["timeline"] = [first, second, third]
    mapping["fault_profiles"] = [
        {
            "id": "stress-fan",
            "type": "fan_speed_degradation",
            "start_step": 5,
            "end_step": cut_a,
            "start_multiplier": 0.9,
            "end_multiplier": 0.5,
        },
        {
            "id": "stress-branch",
            "type": "branch_resistance_increase",
            "zone_id": "laboratory",
            "start_step": 10,
            "end_step": cut_b,
            "start_multiplier": 1.5,
            "end_multiplier": 6.0,
        },
        {
            "id": "stress-damper",
            "type": "damper_jam",
            "damper_id": "airlock_suitport_supply_damper",
            "start_step": cut_a - 2,
            "end_step": cut_a + 20,
        },
        {
            "id": "stress-scrubber",
            "type": "scrubber_capture_degradation",
            "start_step": 3,
            "end_step": cut_b,
            "start_multiplier": 1.0,
            "end_multiplier": 0.1,
        },
        {
            "id": "stress-condenser",
            "type": "condenser_removal_degradation",
            "start_step": cut_a,
            "end_step": cut_b + 5,
            "start_multiplier": 0.8,
            "end_multiplier": 0.0,
        },
        {
            "id": "stress-cooling",
            "type": "cooling_delivery_degradation",
            "zone_id": "crew_quarters_a",
            "start_step": cut_a,
            "end_step": cut_b,
            "start_multiplier": 0.9,
            "end_multiplier": 0.2,
        },
        {
            "id": "stress-oxygen",
            "type": "oxygen_delivery_degradation",
            "zone_id": "laboratory",
            "start_step": cut_a + 4,
            "end_step": cut_b,
            "start_multiplier": 0.95,
            "end_multiplier": 0.3,
        },
    ]
    return mapping


def stress_scenario(steps: int = 240) -> Scenario:
    return Scenario.from_mapping(stress_scenario_mapping(steps))


def scalar_trajectory(
    scenario: Scenario, steps: int | None = None
) -> tuple[PlantState, tuple[ScalarReferenceStep, ...]]:
    """Run the authoritative scalar plant and keep every step's reference."""

    state = initial_state(scenario)
    horizon = scenario.data["steps"] if steps is None else steps
    references = []
    for _ in range(horizon):
        reference = scalar_reference_step(scenario, state)
        assert reference.infeasible_code == 0, "fixture scenario became infeasible"
        references.append(reference)
        state = reference.plant_state
    return initial_state(scenario), tuple(references)


def random_batch(
    parameters: PlantParameters, worlds: int, seed: int
) -> tuple[PlantStateArrays, AchievedControl, Disturbance]:
    """Deterministic in-domain synthetic batch with a recorded seed.

    Ranges are deliberately wide (including zero airflow, saturated duties and
    near-empty stores) but every value is inside the declared domain.  Some
    worlds may be infeasible; that is intended.
    """

    rng = np.random.default_rng(seed)
    zone = (worlds, ZONE_COUNT)

    def u(low: float, high: float, shape: tuple[int, ...]) -> np.ndarray:
        return rng.uniform(low, high, size=shape).astype(np.float64)

    state = PlantStateArrays(
        co2_mol=u(0.0, 3.0, zone),
        o2_mol=u(50.0, 400.0, zone),
        water_vapor_mol=u(0.0, 30.0, zone),
        inert_mol=u(200.0, 900.0, zone),
        temperature_k=u(275.0, 315.0, zone),
        co2_sorbent_remaining_mol=u(0.0, 1500.0, (worlds,)) * (u(0, 1, (worlds,)) > 0.15),
        captured_co2_mol=u(0.0, 500.0, (worlds,)),
        condensed_water_mol=u(0.0, 500.0, (worlds,)),
        oxygen_store_mol=u(0.0, 500.0, (worlds,)),
        battery_energy_wh=u(0.0, parameters.battery_capacity_wh, (worlds,)),
        external_heat_rejected_j=u(0.0, 1e8, (worlds,)),
        external_heat_received_j=u(0.0, 1e8, (worlds,)),
    )
    fan_on = (u(0, 1, (worlds,)) > 0.2).astype(np.float64)
    achieved_cooling = u(0.0, parameters.cooling_max_thermal_w_per_zone, zone)
    oxygen_share = u(0.0, 1.0, zone)
    oxygen_share = oxygen_share / np.sum(oxygen_share, axis=-1, keepdims=True)
    achieved_oxygen = (
        oxygen_share
        * u(0.0, 0.999 * parameters.oxygen_injection_max_total_mol_s, (worlds, 1))
    )
    control = AchievedControl(
        zone_airflow_m3_s=u(0.0, 0.12, zone) * fan_on[:, None],
        achieved_cooling_removed_w=achieved_cooling,
        effective_cooling_delivery_w=achieved_cooling * u(0.0, 1.0, zone),
        achieved_oxygen_injection_mol_s=achieved_oxygen,
        effective_oxygen_delivery_mol_s=achieved_oxygen * u(0.0, 1.0, zone),
        fan_electrical_power_w=u(0.0, 400.0, (worlds,)) * fan_on,
        achieved_scrubber_duty=u(0.0, 1.0, (worlds,)),
        achieved_condenser_duty=u(0.0, 1.0, (worlds,)),
        scrubber_capture_ability=u(0.0, 1.0, (worlds,)),
        condenser_removal_ability=u(0.0, 1.0, (worlds,)),
    )
    disturbance = Disturbance(
        co2_generation_mol_s=u(0.0, 0.002, zone),
        o2_consumption_mol_s=u(0.0, 0.002, zone),
        water_vapor_generation_mol_s=u(0.0, 0.006, zone),
        sensible_heat_w=u(0.0, 900.0, zone),
        generation_w=u(0.0, 8000.0, (worlds,)),
    )
    return state, control, disturbance


def replace_fields(value: Any, **changes: np.ndarray) -> Any:
    """Return a copy of an interface value with some arrays replaced."""

    names = {
        PlantStateArrays: STATE_FIELDS,
        AchievedControl: CONTROL_FIELDS,
        Disturbance: DISTURBANCE_FIELDS,
    }[type(value)]
    data = {name: getattr(value, name).copy() for name in names}
    data.update(changes)
    return type(value)(**data)
