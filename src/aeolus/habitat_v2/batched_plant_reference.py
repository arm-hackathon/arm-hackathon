"""CPU reference side of the batched research plant step (issue 92).

Two things live here, both NumPy ``float64`` and both free of Torch:

``scalar_reference_step``
    Drives the authoritative scalar plant (``physics.advance_one_step`` or
    ``advance_one_step_with_command``) for one world and reports the state,
    achieved control and disturbance of that step in the research interface.
    Requested commands, slew limits, the oxygen cap projection, fault
    schedules and the air-network solve are all resolved there.

``step_arrays`` / ``rollout_arrays``
    An array reference of the thermal, species and battery update over any
    leading batch shape.  Airflow per zone and fan electrical power are
    explicit inputs; there is no airflow solve here.  It mirrors the scalar
    plant's operation order, including left-to-right zone sums in sorted zone
    order, so that a world's result does not depend on batch size, order,
    neighbours or chunking.

Neither is an authority.  ``physics.py`` and the deterministic HMC remain the
only source of canonical simulation and replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .batched_plant_contract import (
    FEASIBLE,
    INFEASIBLE_ELECTRICAL,
    INFEASIBLE_OXYGEN_LOAD,
    INFEASIBLE_OXYGEN_STORE,
    OUTSIDE_SUPPORTED_DOMAIN,
    REFERENCE_ROLLOUT_MAX_STEPS,
    STATE_FIELDS,
    SUPPORTED_ZONE_IDS,
    ZONE_COUNT,
    AchievedControl,
    BatchedPlantDomainError,
    Disturbance,
    PlantParameters,
    PlantStateArrays,
    require_supported_scenario,
    state_field_domain,
    validate_step_inputs,
    zone_sum,
)
from .physics import (
    InfeasibleActionError,
    advance_one_step,
    advance_one_step_with_command,
)
from .scenario import Scenario
from .state import PlantState

# Hand-entered here so the array reference does not call the production
# helper it is compared against (CODATA 2018 molar gas constant; Murphy and
# Koop 2005 saturation pressure over liquid water).
_GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324


def saturation_vapor_pressure_pa(temperature_k: np.ndarray) -> np.ndarray:
    log_pressure = (
        54.842763
        - 6763.22 / temperature_k
        - 4.210 * np.log(temperature_k)
        + 0.000367 * temperature_k
        + np.tanh(0.0415 * (temperature_k - 218.8))
        * (
            53.878
            - 1331.22 / temperature_k
            - 9.44523 * np.log(temperature_k)
            + 0.014025 * temperature_k
        )
    )
    return np.exp(log_pressure)


@dataclass(frozen=True)
class StepDiagnostics:
    """Per-world accounting for one step.  Zero for a held world."""

    infeasible_code: np.ndarray
    co2_captured_mol: np.ndarray
    water_condensed_mol: np.ndarray
    passive_condensation_mol: np.ndarray
    oxygen_injected_mol: np.ndarray
    recirculation_heat_added_j: np.ndarray
    served_load_wh: np.ndarray
    generation_wh: np.ndarray
    battery_charge_stored_wh: np.ndarray
    battery_withdrawn_wh: np.ndarray
    charge_conversion_loss_wh: np.ndarray
    discharge_conversion_loss_wh: np.ndarray
    curtailed_generation_wh: np.ndarray
    external_heat_received_step_j: np.ndarray
    external_heat_rejected_step_j: np.ndarray


@dataclass(frozen=True)
class StepOutput:
    state: PlantStateArrays
    diagnostics: StepDiagnostics


def step_arrays(
    parameters: PlantParameters,
    state: PlantStateArrays,
    control: AchievedControl,
    disturbance: Disturbance,
    *,
    dt_seconds: float,
) -> StepOutput:
    """Advance every world by one step.  Inputs are validated and not mutated.

    A world the scalar plant would refuse (``InfeasibleActionError``) keeps
    its input state and gets a nonzero ``infeasible_code``; its diagnostics
    are zero.  So does a world whose result would be nonfinite or outside the
    declared state domain (``OUTSIDE_SUPPORTED_DOMAIN``).  Other worlds are
    unaffected.
    """

    dt = validate_step_inputs(parameters, state, control, disturbance, dt_seconds)
    with np.errstate(all="ignore"):
        return _step_validated(parameters, state, control, disturbance, dt)


def _step_validated(
    parameters: PlantParameters,
    state: PlantStateArrays,
    control: AchievedControl,
    disturbance: Disturbance,
    dt: float,
) -> StepOutput:
    p = parameters
    airflow = control.zone_airflow_m3_s

    # 1. Declared species sources.
    o2_consumed = disturbance.o2_consumption_mol_s * dt
    infeasible_load = np.any(o2_consumed > state.o2_mol, axis=-1)
    species = {
        "co2_mol": state.co2_mol + disturbance.co2_generation_mol_s * dt,
        "o2_mol": state.o2_mol - o2_consumed,
        "water_vapor_mol": (
            state.water_vapor_mol + disturbance.water_vapor_generation_mol_s * dt
        ),
        "inert_mol": state.inert_mol,
    }

    # 2. Well-mixed recirculation with scrubber capture and active condensing.
    exchange_fraction = 1.0 - np.exp(-airflow * dt / p.volume_m3)
    extracted = {name: value * exchange_fraction for name, value in species.items()}
    remaining = {name: species[name] - extracted[name] for name in species}
    zone_extracted = (
        extracted["co2_mol"] + extracted["o2_mol"] + extracted["water_vapor_mol"]
    ) + extracted["inert_mol"]
    pool = {name: zone_sum(value) for name, value in extracted.items()}
    total_extracted = zone_sum(zone_extracted)
    exchanging = total_extracted != 0.0
    co2_captured = np.minimum(
        np.minimum(
            pool["co2_mol"],
            p.scrubber_max_co2_mol_s
            * control.achieved_scrubber_duty
            * control.scrubber_capture_ability
            * dt,
        ),
        state.co2_sorbent_remaining_mol,
    )
    water_condensed = np.minimum(
        pool["water_vapor_mol"],
        p.condenser_max_water_mol_s
        * control.achieved_condenser_duty
        * control.condenser_removal_ability
        * dt,
    )
    co2_captured = np.where(exchanging, co2_captured, 0.0)
    water_condensed = np.where(exchanging, water_condensed, 0.0)
    pool["co2_mol"] = pool["co2_mol"] - co2_captured
    pool["water_vapor_mol"] = pool["water_vapor_mol"] - water_condensed
    safe_total = np.where(exchanging, total_extracted, 1.0)
    return_fraction = zone_extracted / safe_total[..., None]
    mixed = {
        name: np.where(
            exchanging[..., None],
            remaining[name] + pool[name][..., None] * return_fraction,
            species[name],
        )
        for name in species
    }

    # 3. Oxygen injection from the store (effective delivery).
    oxygen_injected = control.effective_oxygen_delivery_mol_s * dt
    total_oxygen_injected = zone_sum(oxygen_injected)
    infeasible_store = total_oxygen_injected > state.oxygen_store_mol
    mixed["o2_mol"] = mixed["o2_mol"] + oxygen_injected

    # 4. Sensible heat carried by recirculated air, on pre-step temperatures.
    temperature = state.temperature_k
    total_airflow = zone_sum(airflow)
    flowing = total_airflow != 0.0
    safe_airflow = np.where(flowing, total_airflow, 1.0)
    mixed_temperature = zone_sum(airflow * temperature) / safe_airflow
    recirculation_heat = (
        p.air_density_kg_m3
        * p.air_specific_heat_j_kg_k
        * airflow
        * (mixed_temperature[..., None] - temperature)
        * dt
    )
    closure = np.zeros(recirculation_heat.shape, dtype=np.float64)
    closure[..., -1] = zone_sum(recirculation_heat)
    recirculation_heat = np.where(
        flowing[..., None], recirculation_heat - closure, 0.0
    )

    # 5. Explicit-Euler zone thermal balance.
    metabolic_heat = disturbance.sensible_heat_w * dt
    signed_passive = (
        p.passive_thermal_conductance_w_per_k
        * (p.sink_temperature_k - temperature)
        * dt
    )
    passive_received = np.maximum(0.0, signed_passive)
    passive_rejected = np.maximum(0.0, -signed_passive)
    cooling_heat = control.effective_cooling_delivery_w * dt
    energy_delta = (
        metabolic_heat
        + recirculation_heat
        + passive_received
        - passive_rejected
        - cooling_heat
    )
    next_temperature = temperature + energy_delta / p.thermal_capacity_j_per_k
    heat_received = zone_sum(passive_received)
    heat_rejected = zone_sum(passive_rejected + cooling_heat)

    # 6. Passive condensation at the new temperature (no latent heat).
    # A world whose new temperature left the supported band is held below;
    # a placeholder keeps the saturation formula inside its valid range there.
    low_k, high_k = state_field_domain(p, "temperature_k")
    temperature_in_band = np.all(
        (next_temperature >= low_k) & (next_temperature <= high_k), axis=-1
    )
    saturation_temperature = np.where(
        temperature_in_band[..., None], next_temperature, 300.0
    )
    saturated_water = (
        saturation_vapor_pressure_pa(saturation_temperature)
        * p.volume_m3
        / (_GAS_CONSTANT_J_PER_MOL_K * saturation_temperature)
    )
    passive_condensation = np.maximum(0.0, mixed["water_vapor_mol"] - saturated_water)
    mixed["water_vapor_mol"] = mixed["water_vapor_mol"] - passive_condensation

    # 7. Electrical balance.  Loads follow achieved actuation, not effective.
    hours = dt / 3600.0
    served_load = p.base_load_w * hours
    served_load = served_load + control.fan_electrical_power_w * hours
    served_load = served_load + (
        p.scrubber_power_w_full * control.achieved_scrubber_duty
    ) * hours
    served_load = served_load + (
        p.condenser_power_w_full * control.achieved_condenser_duty
    ) * hours
    served_load = served_load + (
        zone_sum(control.achieved_cooling_removed_w)
        / p.cooling_coefficient_of_performance
    ) * hours
    served_load = served_load + (
        p.oxygen_injection_power_w_per_mol_s
        * zone_sum(control.achieved_oxygen_injection_mol_s)
    ) * hours
    generation = disturbance.generation_w * hours
    battery = state.battery_energy_wh
    charging = generation >= served_load

    surplus = generation - served_load
    maximum_charge_input = np.minimum(
        p.battery_max_charge_w * hours,
        (p.battery_capacity_wh - battery) / p.battery_charge_efficiency,
    )
    charge_input = np.where(charging, np.minimum(surplus, maximum_charge_input), 0.0)
    charge_stored = charge_input * p.battery_charge_efficiency
    charge_loss = charge_input - charge_stored
    curtailed = np.where(charging, surplus - charge_input, 0.0)

    required_withdrawal = (served_load - generation) / p.battery_discharge_efficiency
    maximum_withdrawal = np.minimum(battery, p.battery_max_discharge_w * hours)
    electrical_tolerance = np.maximum(1e-12, 1e-10 * np.maximum(1.0, served_load))
    infeasible_electrical = ~charging & (
        required_withdrawal > maximum_withdrawal + electrical_tolerance
    )
    withdrawn = np.where(charging, 0.0, required_withdrawal)
    bus_output = withdrawn * p.battery_discharge_efficiency
    discharge_loss = withdrawn - bus_output
    next_battery = battery + (charge_stored - withdrawn)
    heat_rejected = heat_rejected + (served_load + charge_loss + discharge_loss) * 3600.0

    # 8. Candidate next state.
    result = {
        "co2_mol": mixed["co2_mol"],
        "o2_mol": mixed["o2_mol"],
        "water_vapor_mol": mixed["water_vapor_mol"],
        "inert_mol": mixed["inert_mol"],
        "temperature_k": next_temperature,
        "co2_sorbent_remaining_mol": state.co2_sorbent_remaining_mol - co2_captured,
        "captured_co2_mol": state.captured_co2_mol + co2_captured,
        "condensed_water_mol": (
            state.condensed_water_mol + water_condensed + zone_sum(passive_condensation)
        ),
        "oxygen_store_mol": state.oxygen_store_mol - total_oxygen_injected,
        "battery_energy_wh": next_battery,
        "external_heat_rejected_j": state.external_heat_rejected_j + heat_rejected,
        "external_heat_received_j": state.external_heat_received_j + heat_received,
    }
    in_domain = temperature_in_band
    for name, value in result.items():
        low, high = state_field_domain(p, name)
        ok = np.isfinite(value) & (value >= low)
        if high is not None:
            ok = ok & (value <= high)
        in_domain = in_domain & (np.all(ok, axis=-1) if ok.ndim > in_domain.ndim else ok)

    # 9. Outcome code.  Scalar-plant refusals come first, in the plant's own
    # evaluation order; a result outside the supported domain comes last.
    code = np.where(
        infeasible_load,
        INFEASIBLE_OXYGEN_LOAD,
        np.where(
            infeasible_store,
            INFEASIBLE_OXYGEN_STORE,
            np.where(
                infeasible_electrical,
                INFEASIBLE_ELECTRICAL,
                np.where(in_domain, FEASIBLE, OUTSIDE_SUPPORTED_DOMAIN),
            ),
        ),
    ).astype(np.int64)
    feasible = code == FEASIBLE
    feasible_zone = feasible[..., None]

    def keep(candidate: np.ndarray, previous: np.ndarray) -> np.ndarray:
        mask = feasible_zone if candidate.ndim == feasible_zone.ndim else feasible
        return np.where(mask, candidate, previous)

    def accounted(value: np.ndarray) -> np.ndarray:
        mask = feasible_zone if value.ndim == feasible_zone.ndim else feasible
        return np.where(mask, value, 0.0)

    next_state = PlantStateArrays(
        **{name: keep(value, getattr(state, name)) for name, value in result.items()}
    )
    diagnostics = StepDiagnostics(
        infeasible_code=code,
        co2_captured_mol=accounted(co2_captured),
        water_condensed_mol=accounted(water_condensed),
        passive_condensation_mol=accounted(passive_condensation),
        oxygen_injected_mol=accounted(oxygen_injected),
        recirculation_heat_added_j=accounted(recirculation_heat),
        served_load_wh=accounted(served_load),
        generation_wh=accounted(generation),
        battery_charge_stored_wh=accounted(charge_stored),
        battery_withdrawn_wh=accounted(withdrawn),
        charge_conversion_loss_wh=accounted(charge_loss),
        discharge_conversion_loss_wh=accounted(discharge_loss),
        curtailed_generation_wh=accounted(curtailed),
        external_heat_received_step_j=accounted(heat_received),
        external_heat_rejected_step_j=accounted(heat_rejected),
    )
    return StepOutput(state=next_state, diagnostics=diagnostics)


@dataclass(frozen=True)
class RolloutOutput:
    """Bounded open-loop rollout result.

    ``first_infeasible_step`` is ``-1`` for a world that stayed feasible.  A
    world that becomes infeasible is frozen at its last feasible state for the
    rest of the horizon; later controls and disturbances are not applied to it.
    """

    final_state: PlantStateArrays
    first_infeasible_step: np.ndarray
    infeasible_code: np.ndarray
    steps_applied: np.ndarray


def rollout_arrays(
    parameters: PlantParameters,
    state: PlantStateArrays,
    controls: tuple[AchievedControl, ...],
    disturbances: tuple[Disturbance, ...],
    *,
    dt_seconds: float,
) -> RolloutOutput:
    """Apply a declared sequence of achieved controls and disturbances.

    The sequence is open loop: achieved controls must already include slew,
    cap, fault and airflow effects for every step.  This is a research
    rollout, not an HMC episode.
    """

    if type(controls) is not tuple or type(disturbances) is not tuple:
        raise BatchedPlantDomainError("controls and disturbances must be tuples")
    if len(controls) != len(disturbances) or not controls:
        raise BatchedPlantDomainError(
            "controls and disturbances must be nonempty and of equal length"
        )
    if len(controls) > REFERENCE_ROLLOUT_MAX_STEPS:
        raise BatchedPlantDomainError(
            f"rollout horizon exceeds {REFERENCE_ROLLOUT_MAX_STEPS} steps"
        )
    batch_shape = state.batch_shape
    first_infeasible = np.full(batch_shape, -1, dtype=np.int64)
    code = np.zeros(batch_shape, dtype=np.int64)
    steps_applied = np.zeros(batch_shape, dtype=np.int64)
    current = state
    for index, (control, disturbance) in enumerate(zip(controls, disturbances)):
        output = step_arrays(
            parameters, current, control, disturbance, dt_seconds=dt_seconds
        )
        live = code == FEASIBLE
        newly_infeasible = live & (output.diagnostics.infeasible_code != FEASIBLE)
        first_infeasible = np.where(newly_infeasible, index, first_infeasible)
        code = np.where(newly_infeasible, output.diagnostics.infeasible_code, code)
        advanced = live & ~newly_infeasible
        steps_applied = steps_applied + advanced.astype(np.int64)
        merged = {}
        for name in STATE_FIELDS:
            candidate = getattr(output.state, name)
            previous = getattr(current, name)
            mask = advanced[..., None] if candidate.ndim > advanced.ndim else advanced
            merged[name] = np.where(mask, candidate, previous)
        current = PlantStateArrays(**merged)
    return RolloutOutput(
        final_state=current,
        first_infeasible_step=first_infeasible,
        infeasible_code=code,
        steps_applied=steps_applied,
    )


# --------------------------------------------------------------------------
# Scalar plant adapter
# --------------------------------------------------------------------------

_INFEASIBLE_MESSAGE_CODES: tuple[tuple[str, int], ...] = (
    ("oxygen load exceeds inventory", INFEASIBLE_OXYGEN_LOAD),
    ("oxygen injection exceeds stored oxygen", INFEASIBLE_OXYGEN_STORE),
    ("electrical demand exceeds", INFEASIBLE_ELECTRICAL),
)


@dataclass(frozen=True)
class ScalarReferenceStep:
    """One authoritative scalar step expressed in the research interface.

    All arrays carry a leading world axis of length one.  For an infeasible
    step ``plant_state`` and ``state`` are the unchanged input and ``control``
    is ``None``: the scalar plant raised before it resolved actuation.
    """

    infeasible_code: int
    plant_state: PlantState
    state: PlantStateArrays
    control: AchievedControl | None
    disturbance: Disturbance
    receipt: Mapping[str, Any] | None


def _zone_row(values: Mapping[str, Any], key: str | None = None) -> np.ndarray:
    return np.array(
        [
            [
                float(values[zone_id] if key is None else values[zone_id][key])
                for zone_id in SUPPORTED_ZONE_IDS
            ]
        ],
        dtype=np.float64,
    )


def _world(value: Any) -> np.ndarray:
    return np.array([float(value)], dtype=np.float64)


def disturbance_for_step(scenario: Scenario, step: int) -> Disturbance:
    """Resolve the timeline loads and generation that apply to ``step``."""

    require_supported_scenario(scenario)
    for segment in scenario.data["timeline"]:
        if segment["start_step"] <= step < segment["end_step"]:
            loads = segment["loads"]
            return Disturbance(
                co2_generation_mol_s=_zone_row(loads, "co2_generation_mol_s"),
                o2_consumption_mol_s=_zone_row(loads, "o2_consumption_mol_s"),
                water_vapor_generation_mol_s=_zone_row(
                    loads, "water_vapor_generation_mol_s"
                ),
                sensible_heat_w=_zone_row(loads, "sensible_heat_w"),
                generation_w=_world(segment["generation_w"]),
            )
    raise BatchedPlantDomainError(f"no timeline segment covers step {step}")


def scalar_reference_step(
    scenario: Scenario,
    plant_state: PlantState,
    command: Mapping[str, Any] | None = None,
) -> ScalarReferenceStep:
    """Run the authoritative scalar plant for one world and one step.

    ``command`` is a *requested* external command for the scalar plant (or
    ``None`` to use the scenario timeline).  The returned ``control`` is what
    the plant actually achieved and delivered, which is the only form the
    research step accepts.
    """

    require_supported_scenario(scenario)
    disturbance = disturbance_for_step(scenario, plant_state.step)
    before = PlantStateArrays.from_plant_states((plant_state,))
    try:
        result = (
            advance_one_step(scenario, plant_state)
            if command is None
            else advance_one_step_with_command(scenario, plant_state, command)
        )
    except InfeasibleActionError as exc:
        message = str(exc)
        for fragment, code in _INFEASIBLE_MESSAGE_CODES:
            if fragment in message:
                return ScalarReferenceStep(
                    infeasible_code=code,
                    plant_state=plant_state,
                    state=before,
                    control=None,
                    disturbance=disturbance,
                    receipt=None,
                )
        raise
    utility = result.state.utility
    control = AchievedControl(
        zone_airflow_m3_s=_zone_row(utility.actual_airflow_m3_s),
        achieved_cooling_removed_w=_zone_row(utility.actual_cooling_removed_w),
        effective_cooling_delivery_w=_zone_row(
            utility.effective_cooling_delivery_by_zone
        ),
        achieved_oxygen_injection_mol_s=_zone_row(
            utility.actual_oxygen_injection_mol_s
        ),
        effective_oxygen_delivery_mol_s=_zone_row(
            utility.effective_oxygen_delivery_by_zone
        ),
        fan_electrical_power_w=_world(
            result.receipt["air_network"]["fan_electrical_power_w"]
        ),
        achieved_scrubber_duty=_world(utility.actual_scrubber_duty),
        achieved_condenser_duty=_world(utility.actual_condenser_duty),
        scrubber_capture_ability=_world(utility.effective_scrubber_capture_ability),
        condenser_removal_ability=_world(utility.effective_condenser_removal_ability),
    )
    return ScalarReferenceStep(
        infeasible_code=FEASIBLE,
        plant_state=result.state,
        state=PlantStateArrays.from_plant_states((result.state,)),
        control=control,
        disturbance=disturbance,
        receipt=result.receipt,
    )


def concatenate_worlds(items: tuple[Any, ...]) -> Any:
    """Concatenate interface values of one type along the leading world axis."""

    if not items:
        raise BatchedPlantDomainError("at least one item is required")
    kind = type(items[0])
    if kind not in (PlantStateArrays, AchievedControl, Disturbance):
        raise BatchedPlantDomainError("unsupported interface type")
    if any(type(item) is not kind for item in items):
        raise BatchedPlantDomainError("items must share one interface type")
    names = tuple(kind.__dataclass_fields__)
    return kind(
        **{
            name: np.concatenate([getattr(item, name) for item in items], axis=0)
            for name in names
        }
    )


def take_worlds(value: Any, index: Any) -> Any:
    """Select worlds along the leading axis with a slice or integer array."""

    kind = type(value)
    if kind not in (PlantStateArrays, AchievedControl, Disturbance):
        raise BatchedPlantDomainError("unsupported interface type")
    return kind(
        **{
            name: np.ascontiguousarray(getattr(value, name)[index])
            for name in kind.__dataclass_fields__
        }
    )


__all__ = [
    "RolloutOutput",
    "ScalarReferenceStep",
    "StepDiagnostics",
    "StepOutput",
    "ZONE_COUNT",
    "concatenate_worlds",
    "disturbance_for_step",
    "rollout_arrays",
    "saturation_vapor_pressure_pa",
    "scalar_reference_step",
    "step_arrays",
    "take_worlds",
]
