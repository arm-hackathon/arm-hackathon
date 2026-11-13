"""Conservation, closed-form and reproducibility checks for the NumPy reference.

``step_arrays`` and ``rollout_arrays`` are the array *reference oracle* of the
issue 92 research interface, not a batched backend.  Every expected value in
the conservation and closed-form tests is written out here from the physical
statement of the model with plain arithmetic.  Neither the scalar plant nor a
helper of the module under test is called to produce an expectation.

Reproducibility is asserted bit-exact (``numpy.array_equal``): the declared
reproducibility tolerance of the float64 NumPy reference is exact equality.

These tests show internal consistency of a research oracle on synthetic
in-domain batches.  They are not evidence about any physical system.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, Callable

import numpy as np
import pytest

from aeolus.habitat_v2.batched_plant_contract import (
    CONTROL_FIELDS,
    DISTURBANCE_FIELDS,
    FEASIBLE,
    INFEASIBLE_ELECTRICAL,
    INFEASIBLE_OXYGEN_LOAD,
    INFEASIBLE_OXYGEN_STORE,
    SPECIES_FIELDS,
    STATE_FIELDS,
    ZONE_COUNT,
    AchievedControl,
    Disturbance,
    PlantParameters,
    PlantStateArrays,
    reference_tolerances,
)
from aeolus.habitat_v2.batched_plant_reference import (
    StepOutput,
    concatenate_worlds,
    rollout_arrays,
    step_arrays,
    take_worlds,
)

from ._batched_plant_helpers import load_scenario, random_batch, replace_fields

DT_SECONDS = 60.0
RECORDED_SEEDS: tuple[int, ...] = (11, 92, 2024, 31337)
CONSERVATION_WORLDS = 256
ROLLOUT_REPRODUCIBILITY_STEPS = 12
MODES: tuple[str, ...] = ("step", "rollout")

# Hand-entered CODATA 2018 molar gas constant, independent of the module under test.
GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324

Batch = tuple[PlantStateArrays, AchievedControl, Disturbance]
# One state plus a sequence of per-step controls and disturbances.
Problem = tuple[PlantStateArrays, tuple[AchievedControl, ...], tuple[Disturbance, ...]]

_INFEASIBLE_KINDS: tuple[int, ...] = (
    INFEASIBLE_OXYGEN_LOAD,
    INFEASIBLE_OXYGEN_STORE,
    INFEASIBLE_ELECTRICAL,
)
_CONSTRUCTED_INFEASIBLE_WORLDS: tuple[int, ...] = (5, 50, 150)

_FIELD_NAMES: dict[type, tuple[str, ...]] = {
    PlantStateArrays: STATE_FIELDS,
    AchievedControl: CONTROL_FIELDS,
    Disturbance: DISTURBANCE_FIELDS,
}


@pytest.fixture(scope="module")
def parameters() -> PlantParameters:
    return PlantParameters.from_scenario(load_scenario())


# --------------------------------------------------------------------------
# Local helpers
# --------------------------------------------------------------------------


def _plant_style_tolerance(scale: np.ndarray | float) -> np.ndarray:
    """``max(1e-12, 1e-10 * scale)``, the accounting tolerance form of the plant."""

    return np.maximum(1e-12, 1e-10 * np.asarray(scale, dtype=np.float64))


def _species_tolerance(scale: np.ndarray | float) -> np.ndarray:
    """Plant-style species tolerance, never wider than the 1e-9 mol closure cap."""

    return np.minimum(_plant_style_tolerance(np.maximum(1.0, scale)), 1e-9)


def _energy_tolerance_j(scale: np.ndarray | float) -> np.ndarray:
    return np.maximum(1e-6, 1e-10 * np.asarray(scale, dtype=np.float64))


def _step(parameters: PlantParameters, batch: Batch) -> StepOutput:
    return step_arrays(parameters, *batch, dt_seconds=DT_SECONDS)


def _step_fields(output: StepOutput) -> dict[str, np.ndarray]:
    result = {f"state.{name}": getattr(output.state, name) for name in STATE_FIELDS}
    for item in fields(output.diagnostics):
        result[f"diagnostics.{item.name}"] = getattr(output.diagnostics, item.name)
    return result


def _input_arrays(problem: Problem) -> dict[str, np.ndarray]:
    state, controls, disturbances = problem
    result = {f"state.{name}": getattr(state, name) for name in STATE_FIELDS}
    for index, control in enumerate(controls):
        for name in CONTROL_FIELDS:
            result[f"control[{index}].{name}"] = getattr(control, name)
    for index, disturbance in enumerate(disturbances):
        for name in DISTURBANCE_FIELDS:
            result[f"disturbance[{index}].{name}"] = getattr(disturbance, name)
    return result


def _make_problem(
    parameters: PlantParameters, worlds: int, seed: int, mode: str
) -> Problem:
    """A state and a declared sequence; step ``k`` uses recorded seed ``seed + k``.

    ``random_batch`` worlds are almost always feasible for a single step, so
    refusals are constructed: every fifth world from world 3 is refused at
    step 0 (cycling through the three kinds), and in a rollout every fifth
    world from world 1 is refused at a later step by an oxygen load larger
    than its zone inventory.
    """

    steps = 1 if mode == "step" else ROLLOUT_REPRODUCIBILITY_STEPS
    state, _, _ = random_batch(parameters, worlds, seed)
    sequence = [random_batch(parameters, worlds, seed + index) for index in range(steps)]
    controls = [item[1] for item in sequence]
    disturbances = [item[2] for item in sequence]
    first: Batch = (state, controls[0], disturbances[0])
    for world in range(3, worlds, 5):
        first = _make_world_infeasible(first, world, _INFEASIBLE_KINDS[(world // 5) % 3])
    state, controls[0], disturbances[0] = first
    if steps > 1:
        consumption = [item.o2_consumption_mol_s.copy() for item in disturbances]
        for world in range(1, worlds, 5):
            consumption[1 + (world // 5) % (steps - 1)][world] = 1e6
        disturbances = [
            replace_fields(item, o2_consumption_mol_s=values)
            for item, values in zip(disturbances, consumption)
        ]
    return state, tuple(controls), tuple(disturbances)


def _conservation_batch(parameters: PlantParameters, seed: int) -> Batch:
    """A recorded-seed batch with one constructed world of each refusal kind."""

    batch = random_batch(parameters, CONSERVATION_WORLDS, seed)
    for world, code in zip(_CONSTRUCTED_INFEASIBLE_WORLDS, _INFEASIBLE_KINDS):
        batch = _make_world_infeasible(batch, world, code)
    return batch


def _run(parameters: PlantParameters, problem: Problem, mode: str) -> dict[str, np.ndarray]:
    state, controls, disturbances = problem
    if mode == "step":
        assert len(controls) == 1
        return _step_fields(
            step_arrays(
                parameters, state, controls[0], disturbances[0], dt_seconds=DT_SECONDS
            )
        )
    output = rollout_arrays(
        parameters, state, controls, disturbances, dt_seconds=DT_SECONDS
    )
    result = {
        f"state.{name}": getattr(output.final_state, name) for name in STATE_FIELDS
    }
    result["first_infeasible_step"] = output.first_infeasible_step
    result["infeasible_code"] = output.infeasible_code
    result["steps_applied"] = output.steps_applied
    return result


def _code_of(result: dict[str, np.ndarray]) -> np.ndarray:
    if "infeasible_code" in result:
        return result["infeasible_code"]
    return result["diagnostics.infeasible_code"]


def _take_problem(problem: Problem, index: Any) -> Problem:
    state, controls, disturbances = problem
    return (
        take_worlds(state, index),
        tuple(take_worlds(item, index) for item in controls),
        tuple(take_worlds(item, index) for item in disturbances),
    )


def _concatenate_problems(problems: list[Problem]) -> Problem:
    steps = len(problems[0][1])
    return (
        concatenate_worlds(tuple(item[0] for item in problems)),
        tuple(
            concatenate_worlds(tuple(item[1][step] for item in problems))
            for step in range(steps)
        ),
        tuple(
            concatenate_worlds(tuple(item[2][step] for item in problems))
            for step in range(steps)
        ),
    )


def _map_value(value: Any, transform: Callable[[np.ndarray], np.ndarray]) -> Any:
    kind = type(value)
    return kind(**{name: transform(getattr(value, name)) for name in _FIELD_NAMES[kind]})


def _map_problem(
    problem: Problem, transform: Callable[[np.ndarray], np.ndarray]
) -> Problem:
    state, controls, disturbances = problem
    return (
        _map_value(state, transform),
        tuple(_map_value(item, transform) for item in controls),
        tuple(_map_value(item, transform) for item in disturbances),
    )


def _select(result: dict[str, np.ndarray], index: Any) -> dict[str, np.ndarray]:
    return {name: value[index] for name, value in result.items()}


def _assert_bit_identical(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], context: str
) -> None:
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        candidate = actual[name]
        assert candidate.dtype == reference.dtype, f"{context}: dtype of {name}"
        assert candidate.shape == reference.shape, f"{context}: shape of {name}"
        assert np.array_equal(candidate, reference), (
            f"{context}: {name} is not bit-identical; "
            f"max |difference| = {np.max(np.abs(candidate - reference))!r}"
        )


def _hand_served_load_wh(
    parameters: PlantParameters, control: AchievedControl
) -> np.ndarray:
    """Electrical load from achieved actuation, written from the load statement."""

    load_w = (
        parameters.base_load_w
        + control.fan_electrical_power_w
        + parameters.scrubber_power_w_full * control.achieved_scrubber_duty
        + parameters.condenser_power_w_full * control.achieved_condenser_duty
        + np.sum(control.achieved_cooling_removed_w, axis=-1)
        / parameters.cooling_coefficient_of_performance
        + parameters.oxygen_injection_power_w_per_mol_s
        * np.sum(control.achieved_oxygen_injection_mol_s, axis=-1)
    )
    return load_w * (DT_SECONDS / 3600.0)


def _hand_battery_flows(
    parameters: PlantParameters, batch: Batch
) -> dict[str, np.ndarray]:
    """Battery bookkeeping per world from the stated charge and discharge rules."""

    state, control, disturbance = batch
    hours = DT_SECONDS / 3600.0
    served = _hand_served_load_wh(parameters, control)
    generation = disturbance.generation_w * hours
    surplus = generation - served
    charging = surplus >= 0.0
    headroom_input = (
        parameters.battery_capacity_wh - state.battery_energy_wh
    ) / parameters.battery_charge_efficiency
    charge_input = np.where(
        charging,
        np.minimum(
            np.minimum(surplus, parameters.battery_max_charge_w * hours), headroom_input
        ),
        0.0,
    )
    stored = charge_input * parameters.battery_charge_efficiency
    withdrawn = np.where(
        charging, 0.0, -surplus / parameters.battery_discharge_efficiency
    )
    return {
        "served": served,
        "generation": generation,
        "stored": stored,
        "charge_loss": charge_input - stored,
        "curtailed": np.where(charging, surplus - charge_input, 0.0),
        "withdrawn": withdrawn,
        "discharge_loss": withdrawn * (1.0 - parameters.battery_discharge_efficiency),
    }


def _hand_saturation_pressure_pa(temperature_k: float) -> float:
    """Murphy and Koop (2005) eq. 10, liquid water, transcribed with ``math``."""

    t = temperature_k
    return math.exp(
        54.842763
        - 6763.22 / t
        - 4.210 * math.log(t)
        + 0.000367 * t
        + math.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * math.log(t) + 0.014025 * t)
    )


def _without_actuation(batch: Batch, **control_changes: np.ndarray) -> Batch:
    """Zero the scrubber and condenser duties and apply control replacements."""

    state, control, disturbance = batch
    zeros = np.zeros_like(control.achieved_scrubber_duty)
    changes: dict[str, np.ndarray] = {
        "achieved_scrubber_duty": zeros,
        "achieved_condenser_duty": zeros.copy(),
    }
    changes.update(control_changes)
    return state, replace_fields(control, **changes), disturbance


def _extreme_problem(
    parameters: PlantParameters, worlds: int, seed: int, steps: int, all_infeasible: bool
) -> Problem:
    """Worlds at the corners of the declared domain.

    Each value is drawn as either the low or the high end of an in-domain
    range.  With ``all_infeasible`` every world asks for more oxygen than a
    zone holds, so it is refused at step 0 and held for a whole rollout.
    """

    rng = np.random.default_rng(seed)
    zone = (worlds, ZONE_COUNT)

    def corner(low: float, high: float, shape: tuple[int, ...]) -> np.ndarray:
        return np.where(rng.integers(0, 2, size=shape) == 1, high, low).astype(np.float64)

    state = PlantStateArrays(
        co2_mol=corner(0.0, 1e6, zone),
        o2_mol=corner(0.0, 1e7, zone),
        water_vapor_mol=corner(0.0, 1e5, zone),
        inert_mol=corner(0.0, 1e7, zone),
        temperature_k=corner(200.0, 332.0, zone),
        co2_sorbent_remaining_mol=corner(0.0, 1e9, (worlds,)),
        captured_co2_mol=corner(0.0, 1e9, (worlds,)),
        condensed_water_mol=corner(0.0, 1e9, (worlds,)),
        oxygen_store_mol=corner(0.0, 1e9, (worlds,)),
        battery_energy_wh=corner(0.0, parameters.battery_capacity_wh, (worlds,)),
        external_heat_rejected_j=corner(0.0, 1e15, (worlds,)),
        external_heat_received_j=corner(0.0, 1e15, (worlds,)),
    )
    cooling = corner(0.0, parameters.cooling_max_thermal_w_per_zone, zone)
    oxygen = corner(
        0.0, 0.999 * parameters.oxygen_injection_max_total_mol_s / ZONE_COUNT, zone
    )
    control = AchievedControl(
        zone_airflow_m3_s=corner(0.0, 100.0, zone),
        achieved_cooling_removed_w=cooling,
        effective_cooling_delivery_w=cooling * corner(0.0, 1.0, zone),
        achieved_oxygen_injection_mol_s=oxygen,
        effective_oxygen_delivery_mol_s=oxygen * corner(0.0, 1.0, (worlds, 1)),
        fan_electrical_power_w=corner(0.0, 1e6, (worlds,)),
        achieved_scrubber_duty=corner(0.0, 1.0, (worlds,)),
        achieved_condenser_duty=corner(0.0, 1.0, (worlds,)),
        scrubber_capture_ability=corner(0.0, 1.0, (worlds,)),
        condenser_removal_ability=corner(0.0, 1.0, (worlds,)),
    )
    consumption = (
        np.full(zone, 1e9, dtype=np.float64)
        if all_infeasible
        else np.zeros(zone, dtype=np.float64) + corner(0.0, 1e9, (worlds, 1))
    )
    disturbance = Disturbance(
        co2_generation_mol_s=corner(0.0, 10.0, zone),
        o2_consumption_mol_s=consumption,
        water_vapor_generation_mol_s=corner(0.0, 10.0, zone),
        sensible_heat_w=corner(0.0, 1e6, zone),
        generation_w=corner(0.0, 1e9, (worlds,)),
    )
    return state, (control,) * steps, (disturbance,) * steps


# --------------------------------------------------------------------------
# Batch composition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_conservation_batches_contain_feasible_worlds_and_one_constructed_world_of_each_infeasible_kind(
    parameters: PlantParameters, seed: int
) -> None:
    raw_code = _step(
        parameters, random_batch(parameters, CONSERVATION_WORLDS, seed)
    ).diagnostics.infeasible_code
    code = _step(parameters, _conservation_batch(parameters, seed)).diagnostics.infeasible_code
    assert code.dtype == np.int64
    assert code.shape == (CONSERVATION_WORLDS,)
    assert set(np.unique(code).tolist()) == {FEASIBLE, *_INFEASIBLE_KINDS}
    constructed = np.zeros(CONSERVATION_WORLDS, dtype=bool)
    for world, kind in zip(_CONSTRUCTED_INFEASIBLE_WORLDS, _INFEASIBLE_KINDS):
        assert int(code[world]) == kind
        constructed[world] = True
    # Constructing refusals in three worlds changes no other world's outcome.
    assert np.array_equal(code[~constructed], raw_code[~constructed])
    assert int(np.sum(code == FEASIBLE)) >= CONSERVATION_WORLDS // 2


# --------------------------------------------------------------------------
# Conservation laws with hand-written expectations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_species_totals_change_only_by_declared_sources_capture_condensation_and_injection(
    parameters: PlantParameters, seed: int
) -> None:
    batch = _conservation_batch(parameters, seed)
    state, _, disturbance = batch
    output = _step(parameters, batch)
    diagnostics = output.diagnostics
    feasible = diagnostics.infeasible_code == FEASIBLE
    assert np.any(feasible)

    expected = {
        "co2_mol": (
            np.sum(disturbance.co2_generation_mol_s * DT_SECONDS, axis=-1)
            - diagnostics.co2_captured_mol
        ),
        "o2_mol": (
            -np.sum(disturbance.o2_consumption_mol_s * DT_SECONDS, axis=-1)
            + np.sum(diagnostics.oxygen_injected_mol, axis=-1)
        ),
        "water_vapor_mol": (
            np.sum(disturbance.water_vapor_generation_mol_s * DT_SECONDS, axis=-1)
            - diagnostics.water_condensed_mol
            - np.sum(diagnostics.passive_condensation_mol, axis=-1)
        ),
        "inert_mol": np.zeros(CONSERVATION_WORLDS, dtype=np.float64),
    }
    scale = np.maximum.reduce([np.abs(value) for value in expected.values()])
    tolerance = _species_tolerance(scale)
    for name in SPECIES_FIELDS:
        delta = np.sum(getattr(output.state, name), axis=-1) - np.sum(
            getattr(state, name), axis=-1
        )
        residual = np.abs(delta - expected[name])
        assert np.all(residual[feasible] <= tolerance[feasible]), (
            name,
            float(np.max(residual[feasible])),
        )
    # Removed amounts are physical quantities: never negative, never more
    # than the equipment could move in one step.
    assert np.all(diagnostics.co2_captured_mol >= 0.0)
    assert np.all(diagnostics.water_condensed_mol >= 0.0)
    assert np.all(diagnostics.passive_condensation_mol >= 0.0)
    assert np.all(
        diagnostics.co2_captured_mol
        <= parameters.scrubber_max_co2_mol_s * DT_SECONDS * (1.0 + 1e-12)
    )
    assert np.all(
        diagnostics.water_condensed_mol
        <= parameters.condenser_max_water_mol_s * DT_SECONDS * (1.0 + 1e-12)
    )


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_stores_conserve_sorbent_plus_captured_and_track_injection_and_condensation(
    parameters: PlantParameters, seed: int
) -> None:
    batch = _conservation_batch(parameters, seed)
    state, control, _ = batch
    output = _step(parameters, batch)
    diagnostics = output.diagnostics
    after = output.state
    feasible = diagnostics.infeasible_code == FEASIBLE
    assert np.any(feasible)

    sorbent_total_before = state.co2_sorbent_remaining_mol + state.captured_co2_mol
    sorbent_total_after = after.co2_sorbent_remaining_mol + after.captured_co2_mol
    residual = np.abs(sorbent_total_after - sorbent_total_before)
    assert np.all(
        residual[feasible] <= _species_tolerance(sorbent_total_before)[feasible]
    )
    captured_delta = after.captured_co2_mol - state.captured_co2_mol
    assert np.all(
        np.abs(captured_delta - diagnostics.co2_captured_mol)[feasible]
        <= _species_tolerance(state.captured_co2_mol)[feasible]
    )
    assert np.all(
        diagnostics.co2_captured_mol[feasible]
        <= state.co2_sorbent_remaining_mol[feasible]
    )

    hand_injected = np.sum(control.effective_oxygen_delivery_mol_s * DT_SECONDS, axis=-1)
    store_decrease = state.oxygen_store_mol - after.oxygen_store_mol
    assert np.all(
        np.abs(store_decrease - hand_injected)[feasible]
        <= _species_tolerance(state.oxygen_store_mol)[feasible]
    )
    assert np.all(
        np.abs(np.sum(diagnostics.oxygen_injected_mol, axis=-1) - hand_injected)[feasible]
        <= _species_tolerance(hand_injected)[feasible]
    )

    condensed_increase = after.condensed_water_mol - state.condensed_water_mol
    hand_condensed = diagnostics.water_condensed_mol + np.sum(
        diagnostics.passive_condensation_mol, axis=-1
    )
    assert np.all(
        np.abs(condensed_increase - hand_condensed)[feasible]
        <= _species_tolerance(state.condensed_water_mol)[feasible]
    )


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_zone_thermal_energy_changes_by_sensible_passive_and_effective_cooling_only(
    parameters: PlantParameters, seed: int
) -> None:
    batch = _conservation_batch(parameters, seed)
    state, control, disturbance = batch
    output = _step(parameters, batch)
    feasible = output.diagnostics.infeasible_code == FEASIBLE
    assert np.any(feasible)
    capacity = parameters.thermal_capacity_j_per_k

    stored_delta = np.sum(
        capacity * (output.state.temperature_k - state.temperature_k), axis=-1
    )
    sensible = disturbance.sensible_heat_w * DT_SECONDS
    passive = (
        parameters.passive_thermal_conductance_w_per_k
        * (parameters.sink_temperature_k - state.temperature_k)
        * DT_SECONDS
    )
    cooling = control.effective_cooling_delivery_w * DT_SECONDS
    expected = np.sum(sensible + passive - cooling, axis=-1)
    # Scale is the stored thermal energy being differenced, sum_z C_z * T_z.
    scale = np.sum(capacity * state.temperature_k, axis=-1)
    residual = np.abs(stored_delta - expected)
    assert np.all(residual[feasible] <= _energy_tolerance_j(scale)[feasible]), float(
        np.max(residual[feasible])
    )

    recirculation = output.diagnostics.recirculation_heat_added_j
    recirculation_total = np.array(
        [math.fsum(row) for row in recirculation.tolist()], dtype=np.float64
    )
    assert np.all(np.abs(recirculation_total) <= 1e-9), float(
        np.max(np.abs(recirculation_total))
    )
    # The batch must actually exercise recirculation heat for this to mean anything.
    assert np.any(np.abs(recirculation[feasible]) > 1.0)


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_external_heat_accumulators_match_hand_computation_including_electrical_heat(
    parameters: PlantParameters, seed: int
) -> None:
    batch = _conservation_batch(parameters, seed)
    state, control, _ = batch
    output = _step(parameters, batch)
    diagnostics = output.diagnostics
    feasible = diagnostics.infeasible_code == FEASIBLE
    assert np.any(feasible)

    signed_passive = (
        parameters.passive_thermal_conductance_w_per_k
        * (parameters.sink_temperature_k - state.temperature_k)
        * DT_SECONDS
    )
    hand_received = np.sum(np.where(signed_passive > 0.0, signed_passive, 0.0), axis=-1)
    hand_passive_rejected = np.sum(
        np.where(signed_passive < 0.0, -signed_passive, 0.0), axis=-1
    )
    flows = _hand_battery_flows(parameters, batch)
    hand_rejected = (
        hand_passive_rejected
        + np.sum(control.effective_cooling_delivery_w * DT_SECONDS, axis=-1)
        + (flows["served"] + flows["charge_loss"] + flows["discharge_loss"]) * 3600.0
    )

    assert np.all(
        np.abs(diagnostics.external_heat_received_step_j - hand_received)[feasible]
        <= _energy_tolerance_j(hand_received)[feasible]
    )
    assert np.all(
        np.abs(diagnostics.external_heat_rejected_step_j - hand_rejected)[feasible]
        <= _energy_tolerance_j(hand_rejected)[feasible]
    )

    tolerances = reference_tolerances("float64", "one_step")
    for name, hand_step in (
        ("external_heat_received_j", hand_received),
        ("external_heat_rejected_j", hand_rejected),
    ):
        reference = getattr(state, name) + hand_step
        difference = np.abs(getattr(output.state, name) - reference)
        allowed = tolerances[name].allowed(reference)
        assert np.all(difference[feasible] <= allowed[feasible]), name
        # Accumulators are monotone: a feasible step never lowers them.
        assert np.all(
            getattr(output.state, name)[feasible] >= getattr(state, name)[feasible]
        )


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_electrical_energy_balance_closes_and_battery_follows_stored_minus_withdrawn(
    parameters: PlantParameters, seed: int
) -> None:
    batch = _conservation_batch(parameters, seed)
    state, _, _ = batch
    output = _step(parameters, batch)
    d = output.diagnostics
    feasible = d.infeasible_code == FEASIBLE
    assert np.any(feasible)

    supply = d.generation_wh + d.battery_withdrawn_wh
    use = (
        d.served_load_wh
        + d.battery_charge_stored_wh
        + d.curtailed_generation_wh
        + d.charge_conversion_loss_wh
        + d.discharge_conversion_loss_wh
    )
    scale = np.maximum(1.0, np.maximum(np.abs(supply), np.abs(use)))
    assert np.all(np.abs(supply - use)[feasible] <= _plant_style_tolerance(scale)[feasible])

    flows = _hand_battery_flows(parameters, batch)
    for name, actual in (
        ("served", d.served_load_wh),
        ("generation", d.generation_wh),
        ("stored", d.battery_charge_stored_wh),
        ("charge_loss", d.charge_conversion_loss_wh),
        ("curtailed", d.curtailed_generation_wh),
        ("withdrawn", d.battery_withdrawn_wh),
        ("discharge_loss", d.discharge_conversion_loss_wh),
    ):
        hand = flows[name]
        hand_scale = np.maximum(1.0, np.maximum(flows["served"], flows["generation"]))
        assert np.all(
            np.abs(actual - hand)[feasible] <= _plant_style_tolerance(hand_scale)[feasible]
        ), name
        assert np.all(actual >= 0.0), name
    # Both branches must be present in the feasible set.
    assert np.any(d.battery_charge_stored_wh[feasible] > 0.0)
    assert np.any(d.battery_withdrawn_wh[feasible] > 0.0)
    assert not np.any(
        (d.battery_charge_stored_wh > 0.0) & (d.battery_withdrawn_wh > 0.0)
    )

    battery_delta = output.state.battery_energy_wh - state.battery_energy_wh
    hand_delta = flows["stored"] - flows["withdrawn"]
    battery_scale = np.maximum(1.0, state.battery_energy_wh)
    assert np.all(
        np.abs(battery_delta - hand_delta)[feasible]
        <= _plant_style_tolerance(battery_scale)[feasible]
    )
    assert np.all(
        np.abs(battery_delta - (d.battery_charge_stored_wh - d.battery_withdrawn_wh))[
            feasible
        ]
        <= _plant_style_tolerance(battery_scale)[feasible]
    )
    assert np.all(
        output.state.battery_energy_wh[feasible]
        <= parameters.battery_capacity_wh + 1e-9
    )
    assert np.all(
        d.battery_withdrawn_wh[feasible]
        <= parameters.battery_max_discharge_w * DT_SECONDS / 3600.0 + 1e-9
    )
    assert np.all(
        d.battery_charge_stored_wh[feasible]
        <= parameters.battery_max_charge_w
        * parameters.battery_charge_efficiency
        * DT_SECONDS
        / 3600.0
        + 1e-9
    )


# --------------------------------------------------------------------------
# Closed-form cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_zero_airflow_zones_evolve_independently_with_hand_euler_temperature_and_sources_only(
    parameters: PlantParameters, seed: int
) -> None:
    worlds = 128
    state, control, disturbance = random_batch(parameters, worlds, seed)
    zone_zeros = np.zeros((worlds, ZONE_COUNT), dtype=np.float64)
    control = replace_fields(
        control,
        zone_airflow_m3_s=zone_zeros,
        fan_electrical_power_w=np.zeros(worlds, dtype=np.float64),
        achieved_cooling_removed_w=zone_zeros.copy(),
        effective_cooling_delivery_w=zone_zeros.copy(),
    )
    output = _step(parameters, (state, control, disturbance))
    d = output.diagnostics
    feasible = d.infeasible_code == FEASIBLE
    assert np.any(feasible)
    after = output.state

    hand_temperature = state.temperature_k + (
        disturbance.sensible_heat_w
        + parameters.passive_thermal_conductance_w_per_k
        * (parameters.sink_temperature_k - state.temperature_k)
    ) * DT_SECONDS / parameters.thermal_capacity_j_per_k
    assert np.all(np.abs(after.temperature_k - hand_temperature)[feasible] <= 1e-12)

    # No exchange: scrubber and condenser see an empty pool and remove nothing.
    assert np.all(d.co2_captured_mol == 0.0)
    assert np.all(d.water_condensed_mol == 0.0)
    assert np.all(d.recirculation_heat_added_j == 0.0)
    assert np.array_equal(
        after.co2_sorbent_remaining_mol, state.co2_sorbent_remaining_mol
    )
    assert np.array_equal(after.captured_co2_mol, state.captured_co2_mol)

    # A single multiply and add or subtract per zone: IEEE arithmetic leaves
    # one correctly rounded answer, so these are exact.
    hand_co2 = state.co2_mol + disturbance.co2_generation_mol_s * DT_SECONDS
    hand_o2 = (
        state.o2_mol - disturbance.o2_consumption_mol_s * DT_SECONDS
    ) + control.effective_oxygen_delivery_mol_s * DT_SECONDS
    assert np.array_equal(after.co2_mol[feasible], hand_co2[feasible])
    assert np.array_equal(after.o2_mol[feasible], hand_o2[feasible])
    assert np.array_equal(after.inert_mol, state.inert_mol)

    hand_water = (
        state.water_vapor_mol + disturbance.water_vapor_generation_mol_s * DT_SECONDS
    )
    tolerance = reference_tolerances("float64", "one_step")["water_vapor_mol"]
    assert np.all(
        np.abs(after.water_vapor_mol + d.passive_condensation_mol - hand_water)[feasible]
        <= tolerance.allowed(hand_water)[feasible]
    )
    dry = feasible[:, None] & (d.passive_condensation_mol == 0.0)
    assert np.any(dry)
    assert np.array_equal(after.water_vapor_mol[dry], hand_water[dry])


def test_passive_condensation_caps_vapor_at_hand_transcribed_saturation_inventory(
    parameters: PlantParameters,
) -> None:
    # Liquid-water triple point, 611.657 Pa: an anchor for the transcription
    # that does not come from this repository.
    assert abs(_hand_saturation_pressure_pa(273.16) - 611.657) < 0.05

    state, control, disturbance = random_batch(parameters, 4, RECORDED_SEEDS[0])
    zone_zeros = np.zeros((4, ZONE_COUNT), dtype=np.float64)
    world_zeros = np.zeros(4, dtype=np.float64)
    # Zones start at their sink temperature with no heat input and no
    # cooling, so the temperature does not move and saturation is known.
    start_temperature = np.broadcast_to(
        parameters.sink_temperature_k, (4, ZONE_COUNT)
    ).copy()
    state = replace_fields(
        state,
        temperature_k=start_temperature,
        water_vapor_mol=np.full((4, ZONE_COUNT), 30.0, dtype=np.float64),
    )
    control = replace_fields(
        control,
        zone_airflow_m3_s=zone_zeros,
        fan_electrical_power_w=world_zeros,
        achieved_cooling_removed_w=zone_zeros.copy(),
        effective_cooling_delivery_w=zone_zeros.copy(),
        achieved_oxygen_injection_mol_s=zone_zeros.copy(),
        effective_oxygen_delivery_mol_s=zone_zeros.copy(),
    )
    disturbance = replace_fields(
        disturbance,
        sensible_heat_w=zone_zeros.copy(),
        water_vapor_generation_mol_s=zone_zeros.copy(),
        o2_consumption_mol_s=zone_zeros.copy(),
        generation_w=np.full(4, 8000.0, dtype=np.float64),
    )
    output = _step(parameters, (state, control, disturbance))
    assert np.all(output.diagnostics.infeasible_code == FEASIBLE)
    assert np.array_equal(output.state.temperature_k, start_temperature)

    hand_saturated = np.array(
        [
            _hand_saturation_pressure_pa(float(temperature))
            * float(volume)
            / (GAS_CONSTANT_J_PER_MOL_K * float(temperature))
            for temperature, volume in zip(
                parameters.sink_temperature_k, parameters.volume_m3
            )
        ],
        dtype=np.float64,
    )
    assert np.all(hand_saturated < 30.0)
    tolerance = reference_tolerances("float64", "one_step")["water_vapor_mol"]
    assert np.all(
        np.abs(output.state.water_vapor_mol - hand_saturated)
        <= tolerance.allowed(hand_saturated)
    )
    hand_condensed = float(np.sum(30.0 - hand_saturated))
    increase = output.state.condensed_water_mol - state.condensed_water_mol
    assert np.all(np.abs(increase - hand_condensed) <= _species_tolerance(600.0))


def test_uniform_composition_and_temperature_are_a_fixed_point_of_recirculation(
    parameters: PlantParameters,
) -> None:
    worlds = 6
    rng = np.random.default_rng(424242)
    fractions = {
        "co2_mol": 0.001,
        "o2_mol": 0.21,
        "water_vapor_mol": 0.004,
        "inert_mol": 0.785,
    }
    total_moles = (
        parameters.volume_m3 * rng.uniform(30.0, 45.0, size=(worlds, ZONE_COUNT))
    ).astype(np.float64)
    state, control, disturbance = random_batch(parameters, worlds, 424242)
    zone_zeros = np.zeros((worlds, ZONE_COUNT), dtype=np.float64)
    state = replace_fields(
        state,
        temperature_k=np.full((worlds, ZONE_COUNT), 293.0, dtype=np.float64),
        **{name: total_moles * fraction for name, fraction in fractions.items()},
    )
    airflow = rng.uniform(0.01, 0.12, size=(worlds, ZONE_COUNT)).astype(np.float64)
    control = replace_fields(
        control,
        zone_airflow_m3_s=airflow,
        achieved_scrubber_duty=np.zeros(worlds, dtype=np.float64),
        achieved_condenser_duty=np.zeros(worlds, dtype=np.float64),
        achieved_oxygen_injection_mol_s=zone_zeros,
        effective_oxygen_delivery_mol_s=zone_zeros.copy(),
    )
    disturbance = replace_fields(
        disturbance,
        co2_generation_mol_s=zone_zeros.copy(),
        o2_consumption_mol_s=zone_zeros.copy(),
        water_vapor_generation_mol_s=zone_zeros.copy(),
        generation_w=np.full(worlds, 8000.0, dtype=np.float64),
    )
    output = _step(parameters, (state, control, disturbance))
    d = output.diagnostics
    assert np.all(d.infeasible_code == FEASIBLE)
    assert np.all(d.passive_condensation_mol == 0.0)
    assert np.all(d.co2_captured_mol == 0.0)
    assert np.all(d.water_condensed_mol == 0.0)

    after_total = np.zeros((worlds, ZONE_COUNT), dtype=np.float64)
    for name in SPECIES_FIELDS:
        after_total = after_total + getattr(output.state, name)
    for name, fraction in fractions.items():
        before = getattr(state, name)
        after = getattr(output.state, name)
        assert np.all(np.abs(after - before) <= 1e-12 * np.abs(before)), name
        assert np.all(np.abs(after / after_total - fraction) <= 1e-12 * fraction), name
    grand_before = np.sum(total_moles, axis=-1)
    assert np.all(
        np.abs(np.sum(after_total, axis=-1) - grand_before)
        <= _species_tolerance(grand_before)
    )

    # Equal temperatures carry no net heat between zones; each zone then
    # follows its own explicit-Euler balance.
    assert np.all(np.abs(d.recirculation_heat_added_j) <= 1e-6)
    hand_temperature = 293.0 + (
        disturbance.sensible_heat_w
        + parameters.passive_thermal_conductance_w_per_k
        * (parameters.sink_temperature_k - 293.0)
        - control.effective_cooling_delivery_w
    ) * DT_SECONDS / parameters.thermal_capacity_j_per_k
    tolerance = reference_tolerances("float64", "one_step")["temperature_k"]
    assert np.all(
        np.abs(output.state.temperature_k - hand_temperature)
        <= tolerance.allowed(hand_temperature)
    )


@pytest.mark.parametrize("active_zone", [0, 3, ZONE_COUNT - 1])
def test_single_nonzero_airflow_zone_gets_back_exactly_what_it_gave(
    parameters: PlantParameters, active_zone: int
) -> None:
    worlds = 64
    batch = random_batch(parameters, worlds, RECORDED_SEEDS[1])
    still = np.zeros((worlds, ZONE_COUNT), dtype=np.float64)
    single = still.copy()
    single[:, active_zone] = np.linspace(0.01, 0.12, worlds)
    reference = _step(parameters, _without_actuation(batch, zone_airflow_m3_s=still))
    candidate = _step(parameters, _without_actuation(batch, zone_airflow_m3_s=single))
    assert np.array_equal(
        candidate.diagnostics.infeasible_code, reference.diagnostics.infeasible_code
    )
    feasible = reference.diagnostics.infeasible_code == FEASIBLE
    assert np.any(feasible)
    for name in STATE_FIELDS:
        expected = getattr(reference.state, name)[feasible]
        actual = getattr(candidate.state, name)[feasible]
        assert np.all(np.abs(actual - expected) <= 1e-12 * np.abs(expected)), name
    assert np.all(candidate.diagnostics.co2_captured_mol == 0.0)
    assert np.all(candidate.diagnostics.water_condensed_mol == 0.0)
    assert np.all(np.abs(candidate.diagnostics.recirculation_heat_added_j) <= 1e-9)


def test_two_active_zones_match_hand_computed_exchange_fraction_and_mixed_pool(
    parameters: PlantParameters,
) -> None:
    zone_a, zone_b = 2, ZONE_COUNT - 1
    airflow_a, airflow_b = 0.05, 0.08
    index = np.arange(ZONE_COUNT, dtype=np.float64)
    row = (1, ZONE_COUNT)
    airflow = np.zeros(row, dtype=np.float64)
    airflow[0, zone_a] = airflow_a
    airflow[0, zone_b] = airflow_b
    state = PlantStateArrays(
        co2_mol=(0.4 + 0.3 * index).reshape(row),
        o2_mol=(100.0 + 20.0 * index).reshape(row),
        water_vapor_mol=(0.5 + 0.1 * index).reshape(row),
        inert_mol=(400.0 + 50.0 * index).reshape(row),
        temperature_k=(288.0 + 2.0 * index).reshape(row),
        co2_sorbent_remaining_mol=np.array([100.0]),
        captured_co2_mol=np.array([7.0]),
        condensed_water_mol=np.array([3.0]),
        oxygen_store_mol=np.array([50.0]),
        battery_energy_wh=np.array([0.5 * parameters.battery_capacity_wh]),
        external_heat_rejected_j=np.array([0.0]),
        external_heat_received_j=np.array([0.0]),
    )
    cooling_delivered_w = 300.0
    oxygen_delivered_mol_s = 0.0002
    scrubber_duty, scrubber_ability = 0.8, 0.5
    condenser_duty, condenser_ability = 1.0, 1.0
    control = AchievedControl(
        zone_airflow_m3_s=airflow,
        achieved_cooling_removed_w=np.full(row, 500.0),
        effective_cooling_delivery_w=np.full(row, cooling_delivered_w),
        achieved_oxygen_injection_mol_s=np.full(row, 0.0003),
        effective_oxygen_delivery_mol_s=np.full(row, oxygen_delivered_mol_s),
        fan_electrical_power_w=np.array([50.0]),
        achieved_scrubber_duty=np.array([scrubber_duty]),
        achieved_condenser_duty=np.array([condenser_duty]),
        scrubber_capture_ability=np.array([scrubber_ability]),
        condenser_removal_ability=np.array([condenser_ability]),
    )
    generation = {"co2_mol": 0.001, "o2_mol": -0.0015, "water_vapor_mol": 0.002}
    disturbance = Disturbance(
        co2_generation_mol_s=np.full(row, generation["co2_mol"]),
        o2_consumption_mol_s=np.full(row, -generation["o2_mol"]),
        water_vapor_generation_mol_s=np.full(row, generation["water_vapor_mol"]),
        sensible_heat_w=(100.0 + 10.0 * index).reshape(row),
        generation_w=np.array([6000.0]),
    )
    output = _step(parameters, (state, control, disturbance))
    d = output.diagnostics
    assert int(d.infeasible_code[0]) == FEASIBLE
    assert np.all(d.passive_condensation_mol == 0.0)

    # Hand computation with Python floats.
    dt = DT_SECONDS
    volume = [float(value) for value in parameters.volume_m3]
    flow = {zone_a: airflow_a, zone_b: airflow_b}
    fraction = {
        zone: 1.0 - math.exp(-flow[zone] * dt / volume[zone]) for zone in flow
    }
    sourced = {
        name: [
            float(getattr(state, name)[0, zone]) + generation.get(name, 0.0) * dt
            for zone in range(ZONE_COUNT)
        ]
        for name in SPECIES_FIELDS
    }
    extracted = {
        name: {zone: sourced[name][zone] * fraction[zone] for zone in flow}
        for name in SPECIES_FIELDS
    }
    pool = {name: extracted[name][zone_a] + extracted[name][zone_b] for name in SPECIES_FIELDS}
    zone_extracted = {
        zone: sum(extracted[name][zone] for name in SPECIES_FIELDS) for zone in flow
    }
    total_extracted = zone_extracted[zone_a] + zone_extracted[zone_b]
    scrubber_limit = (
        parameters.scrubber_max_co2_mol_s * scrubber_duty * scrubber_ability * dt
    )
    condenser_limit = (
        parameters.condenser_max_water_mol_s * condenser_duty * condenser_ability * dt
    )
    hand_captured = min(pool["co2_mol"], scrubber_limit, 100.0)
    hand_condensed = min(pool["water_vapor_mol"], condenser_limit)
    # The fixture is built so each limit binds once: the scrubber is rate
    # limited and the condenser empties the pool.
    assert hand_captured == scrubber_limit < pool["co2_mol"]
    assert hand_condensed == pool["water_vapor_mol"] < condenser_limit
    pool["co2_mol"] -= hand_captured
    pool["water_vapor_mol"] -= hand_condensed

    tolerances = reference_tolerances("float64", "one_step")
    for name in SPECIES_FIELDS:
        hand = list(sourced[name])
        for zone in flow:
            hand[zone] = (
                sourced[name][zone]
                - extracted[name][zone]
                + pool[name] * zone_extracted[zone] / total_extracted
            )
        if name == "o2_mol":
            hand = [value + oxygen_delivered_mol_s * dt for value in hand]
        expected = np.array([hand], dtype=np.float64)
        difference = np.abs(getattr(output.state, name) - expected)
        assert np.all(difference <= tolerances[name].allowed(expected)), name

    assert abs(float(d.co2_captured_mol[0]) - hand_captured) <= 1e-12
    assert abs(float(d.water_condensed_mol[0]) - hand_condensed) <= 1e-12
    assert abs(float(output.state.captured_co2_mol[0]) - (7.0 + hand_captured)) <= 1e-12
    assert (
        abs(float(output.state.co2_sorbent_remaining_mol[0]) - (100.0 - hand_captured))
        <= 1e-12
    )
    assert abs(float(output.state.condensed_water_mol[0]) - (3.0 + hand_condensed)) <= 1e-12
    hand_store = 50.0 - ZONE_COUNT * oxygen_delivered_mol_s * dt
    assert abs(float(output.state.oxygen_store_mol[0]) - hand_store) <= 1e-12

    # Sensible heat: flow-weighted mixed temperature of the two active zones.
    temperature = [float(value) for value in state.temperature_k[0]]
    mixed_temperature = (
        airflow_a * temperature[zone_a] + airflow_b * temperature[zone_b]
    ) / (airflow_a + airflow_b)
    carried = parameters.air_density_kg_m3 * parameters.air_specific_heat_j_kg_k
    heat = [0.0] * ZONE_COUNT
    heat[zone_a] = carried * airflow_a * (mixed_temperature - temperature[zone_a]) * dt
    heat[zone_b] = carried * airflow_b * (mixed_temperature - temperature[zone_b]) * dt
    assert heat[zone_a] > 1000.0 and heat[zone_b] < -1000.0
    hand_temperature = np.array(
        [
            [
                temperature[zone]
                + (
                    float(disturbance.sensible_heat_w[0, zone]) * dt
                    + heat[zone]
                    + float(parameters.passive_thermal_conductance_w_per_k[zone])
                    * (float(parameters.sink_temperature_k[zone]) - temperature[zone])
                    * dt
                    - cooling_delivered_w * dt
                )
                / float(parameters.thermal_capacity_j_per_k[zone])
                for zone in range(ZONE_COUNT)
            ]
        ],
        dtype=np.float64,
    )
    assert np.all(
        np.abs(output.state.temperature_k - hand_temperature)
        <= tolerances["temperature_k"].allowed(hand_temperature)
    )
    assert np.all(
        np.abs(d.recirculation_heat_added_j - np.array([heat])) <= _energy_tolerance_j(0.0)
    )


# --------------------------------------------------------------------------
# Nonnegative inventory, finite state and held worlds
# --------------------------------------------------------------------------


def _assert_state_is_finite_and_nonnegative(
    state: PlantStateArrays, mask: np.ndarray, context: str
) -> None:
    for name in STATE_FIELDS:
        values = getattr(state, name)[mask]
        assert np.all(np.isfinite(values)), f"{context}: {name} is not finite"
        if name == "temperature_k":
            assert np.all(values > 0.0), f"{context}: temperature is not positive"
        else:
            assert np.all(values >= -1e-12), (
                f"{context}: {name} is negative, min {float(np.min(values))!r}"
            )


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_feasible_worlds_keep_finite_nonnegative_inventory_and_positive_temperature(
    parameters: PlantParameters, seed: int
) -> None:
    batch = _conservation_batch(parameters, seed)
    output = _step(parameters, batch)
    feasible = output.diagnostics.infeasible_code == FEASIBLE
    assert np.any(feasible)
    _assert_state_is_finite_and_nonnegative(output.state, feasible, f"seed {seed}")
    for item in fields(output.diagnostics):
        assert np.all(np.isfinite(getattr(output.diagnostics, item.name))), item.name


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_sixty_step_repeated_rollout_stays_finite_nonnegative_and_freezes_infeasible_worlds(
    parameters: PlantParameters, seed: int
) -> None:
    worlds = 96
    steps = 60
    state, control, disturbance = random_batch(parameters, worlds, seed)
    controls = (control,) * steps
    disturbances = (disturbance,) * steps
    output = rollout_arrays(
        parameters, state, controls, disturbances, dt_seconds=DT_SECONDS
    )
    every = np.ones(worlds, dtype=bool)
    _assert_state_is_finite_and_nonnegative(output.final_state, every, f"seed {seed}")
    assert np.all(
        output.final_state.battery_energy_wh <= parameters.battery_capacity_wh + 1e-9
    )

    first = output.first_infeasible_step
    held = first >= 0
    assert np.any(held) and np.any(~held)
    assert np.all(output.infeasible_code[~held] == FEASIBLE)
    assert np.all(first[~held] == -1)
    assert np.all(output.steps_applied[~held] == steps)
    assert np.all(output.infeasible_code[held] != FEASIBLE)
    assert np.array_equal(output.steps_applied[held], first[held])
    assert np.all(first[held] < steps)

    # A held world must equal its own state after exactly ``first`` applied
    # steps, which a rollout truncated to that many steps exposes.
    for step in np.unique(first[held]).tolist():
        if step == 0:
            prefix_state = state
        else:
            prefix = rollout_arrays(
                parameters,
                state,
                controls[:step],
                disturbances[:step],
                dt_seconds=DT_SECONDS,
            )
            prefix_state = prefix.final_state
        selected = first == step
        if step > 0:
            assert np.all(prefix.first_infeasible_step[selected] == -1)
            assert np.all(prefix.steps_applied[selected] == step)
        for name in STATE_FIELDS:
            assert np.array_equal(
                getattr(output.final_state, name)[selected],
                getattr(prefix_state, name)[selected],
            ), (step, name)


# --------------------------------------------------------------------------
# Reproducibility and lane isolation (bit-exact)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_world_result_is_bit_identical_alone_and_inside_batches_of_2_17_and_1024(
    parameters: PlantParameters, mode: str
) -> None:
    base = _make_problem(parameters, 1024, 7001, mode)
    full = _run(parameters, base, mode)
    code = _code_of(full)
    assert set(np.unique(code).tolist()) == {FEASIBLE, *_INFEASIBLE_KINDS}
    if mode == "rollout":
        held_at = full["first_infeasible_step"]
        assert np.any(held_at == 0) and np.any(held_at > 0) and np.any(held_at == -1)
    for size in (1, 2, 17, 1024):
        leading = _run(parameters, _take_problem(base, slice(0, size)), mode)
        _assert_bit_identical(leading, _select(full, slice(0, size)), f"size {size}")
    infeasible_world = int(np.flatnonzero(code != FEASIBLE)[0])
    for world in (0, 1, 16, 511, 1023, infeasible_world):
        alone = _run(parameters, _take_problem(base, np.array([world])), mode)
        _assert_bit_identical(
            alone, _select(full, slice(world, world + 1)), f"world {world} alone"
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [2, 17, 1024])
def test_world_result_is_bit_identical_at_first_middle_and_last_batch_position(
    parameters: PlantParameters, mode: str, size: int
) -> None:
    base = _make_problem(parameters, 1025, 7002, mode)
    target = 1024
    alone = _run(parameters, _take_problem(base, np.array([target])), mode)
    for position in sorted({0, size // 2, size - 1}):
        index = np.arange(size - 1)
        index = np.concatenate([index[:position], [target], index[position:]])
        assert index.shape == (size,) and int(index[position]) == target
        placed = _run(parameters, _take_problem(base, index), mode)
        _assert_bit_identical(
            _select(placed, slice(position, position + 1)),
            alone,
            f"size {size} position {position}",
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", [5, 6])
def test_random_world_permutation_gives_bit_identical_inverse_permuted_outputs(
    parameters: PlantParameters, mode: str, seed: int
) -> None:
    worlds = 1024
    base = _make_problem(parameters, worlds, 7003, mode)
    full = _run(parameters, base, mode)
    permutation = np.random.default_rng(seed).permutation(worlds)
    assert not np.array_equal(permutation, np.arange(worlds))
    inverse = np.argsort(permutation)
    permuted = _run(parameters, _take_problem(base, permutation), mode)
    _assert_bit_identical(_select(permuted, inverse), full, f"permutation seed {seed}")


@pytest.mark.parametrize(
    ("mode", "all_infeasible"),
    [("step", False), ("step", True), ("rollout", True)],
)
def test_extreme_in_domain_or_infeasible_neighbours_do_not_change_a_world(
    parameters: PlantParameters, mode: str, all_infeasible: bool
) -> None:
    size = 17
    targets = (0, 8, 16)
    base = _make_problem(parameters, size, 7004, mode)
    baseline = _run(parameters, base, mode)
    steps = len(base[1])
    extreme = _extreme_problem(parameters, size, 7005, steps, all_infeasible)
    extreme_code = _code_of(_run(parameters, extreme, mode))
    if all_infeasible:
        assert np.all(extreme_code == INFEASIBLE_OXYGEN_LOAD)
    else:
        assert np.any(extreme_code == FEASIBLE) and np.any(extreme_code != FEASIBLE)

    keep = np.zeros(size, dtype=bool)
    keep[list(targets)] = True

    def splice(original: np.ndarray, replacement: np.ndarray) -> np.ndarray:
        mask = keep if original.ndim == 1 else keep[:, None]
        return np.where(mask, original, replacement)

    def merge(original: Any, replacement: Any) -> Any:
        kind = type(original)
        return kind(
            **{
                name: splice(getattr(original, name), getattr(replacement, name))
                for name in _FIELD_NAMES[kind]
            }
        )

    mixed: Problem = (
        merge(base[0], extreme[0]),
        tuple(merge(a, b) for a, b in zip(base[1], extreme[1])),
        tuple(merge(a, b) for a, b in zip(base[2], extreme[2])),
    )
    result = _run(parameters, mixed, mode)
    index = np.array(targets)
    _assert_bit_identical(
        _select(result, index), _select(baseline, index), "extreme neighbours"
    )
    for world in targets:
        alone = _run(parameters, _take_problem(base, np.array([world])), mode)
        _assert_bit_identical(
            _select(result, slice(world, world + 1)), alone, f"world {world} alone"
        )
    # The replaced neighbours behave as they do in their own batch.
    assert np.array_equal(_code_of(result)[~keep], extreme_code[~keep])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunk", [1, 3, 7, 64])
def test_chunked_evaluation_concatenates_to_the_bit_identical_full_batch_result(
    parameters: PlantParameters, mode: str, chunk: int
) -> None:
    worlds = 200
    base = _make_problem(parameters, worlds, 7006, mode)
    full = _run(parameters, base, mode)
    pieces = [
        _take_problem(base, slice(start, min(start + chunk, worlds)))
        for start in range(0, worlds, chunk)
    ]
    assert sum(piece[0].batch_shape[0] for piece in pieces) == worlds
    results = [_run(parameters, piece, mode) for piece in pieces]
    joined = {
        name: np.concatenate([result[name] for result in results], axis=0)
        for name in full
    }
    _assert_bit_identical(joined, full, f"chunk {chunk}")

    # concatenate_worlds is the inverse of the chunking on the inputs too.
    rebuilt = _concatenate_problems(pieces)
    for name, value in _input_arrays(base).items():
        assert np.array_equal(_input_arrays(rebuilt)[name], value), name
    _assert_bit_identical(_run(parameters, rebuilt, mode), full, f"rebuilt {chunk}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("leading", [(4, 6), (2, 3, 4), (24, 1), (1, 24)])
def test_flat_and_reshaped_leading_batch_axes_give_bit_identical_results(
    parameters: PlantParameters, mode: str, leading: tuple[int, ...]
) -> None:
    worlds = 24
    base = _make_problem(parameters, worlds, 7007, mode)
    flat = _run(parameters, base, mode)

    def reshape(array: np.ndarray) -> np.ndarray:
        return array.reshape(leading + array.shape[1:]).copy()

    shaped = _run(parameters, _map_problem(base, reshape), mode)
    for name, value in shaped.items():
        assert value.shape[: len(leading)] == leading, name
    restored = {
        name: value.reshape((worlds,) + value.shape[len(leading) :])
        for name, value in shaped.items()
    }
    _assert_bit_identical(restored, flat, f"leading {leading}")


def _strided(array: np.ndarray) -> np.ndarray:
    """Every other element of a larger buffer along each axis."""

    big = np.full(tuple(2 * extent for extent in array.shape), 1.0, dtype=np.float64)
    view = big[tuple(slice(None, None, 2) for _ in array.shape)]
    view[...] = array
    return view


def _fortran(array: np.ndarray) -> np.ndarray:
    return np.asfortranarray(array)


def _negative_zone_stride(array: np.ndarray) -> np.ndarray:
    """Same values, walked backwards in memory along the last axis."""

    return np.ascontiguousarray(array[..., ::-1])[..., ::-1]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "layout", [_strided, _fortran, _negative_zone_stride], ids=lambda item: item.__name__
)
def test_non_contiguous_fortran_and_negative_stride_inputs_give_bit_identical_results(
    parameters: PlantParameters, mode: str, layout: Callable[[np.ndarray], np.ndarray]
) -> None:
    base = _make_problem(parameters, 131, 7008, mode)
    reference = _run(parameters, base, mode)
    relaid = _map_problem(base, layout)
    relaid_arrays = _input_arrays(relaid)
    for name, value in _input_arrays(base).items():
        assert np.array_equal(relaid_arrays[name], value), name
    zone_arrays = [value for value in relaid_arrays.values() if value.ndim == 2]
    if layout is _fortran:
        assert all(
            value.flags.f_contiguous and not value.flags.c_contiguous
            for value in zone_arrays
        )
    else:
        assert not any(value.flags.c_contiguous for value in relaid_arrays.values())
    if layout is _negative_zone_stride:
        assert all(value.strides[-1] < 0 for value in relaid_arrays.values())
    _assert_bit_identical(_run(parameters, relaid, mode), reference, layout.__name__)


@pytest.mark.parametrize("mode", MODES)
def test_reversed_world_axis_views_give_bit_identical_reversed_outputs(
    parameters: PlantParameters, mode: str
) -> None:
    base = _make_problem(parameters, 131, 7009, mode)
    reference = _run(parameters, base, mode)
    reversed_problem = _map_problem(base, lambda array: array[::-1])
    assert all(value.strides[0] < 0 for value in _input_arrays(reversed_problem).values())
    result = _run(parameters, reversed_problem, mode)
    _assert_bit_identical(_select(result, slice(None, None, -1)), reference, "reversed")


# --------------------------------------------------------------------------
# No shared mutable state, no RNG coupling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_outputs_never_share_memory_with_inputs_and_inputs_are_left_unchanged(
    parameters: PlantParameters, mode: str
) -> None:
    problem = _make_problem(parameters, 33, 7010, mode)
    inputs = _input_arrays(problem)
    snapshots = {name: value.copy() for name, value in inputs.items()}
    # Read-only inputs: any in-place write inside the call would raise.
    for value in inputs.values():
        value.setflags(write=False)
    result = _run(parameters, problem, mode)
    assert np.any(_code_of(result) == FEASIBLE) and np.any(_code_of(result) != FEASIBLE)
    for name, value in inputs.items():
        assert np.array_equal(value, snapshots[name]), f"input {name} changed"
    for output_name, output_value in result.items():
        for input_name, input_value in inputs.items():
            assert not np.shares_memory(output_value, input_value), (
                output_name,
                input_name,
            )
    names = list(result)
    for position, first_name in enumerate(names):
        for second_name in names[position + 1 :]:
            assert not np.shares_memory(result[first_name], result[second_name]), (
                first_name,
                second_name,
            )


@pytest.mark.parametrize("mode", MODES)
def test_repeated_calls_are_bit_identical_and_mutating_an_output_does_not_leak(
    parameters: PlantParameters, mode: str
) -> None:
    problem = _make_problem(parameters, 33, 7011, mode)
    snapshots = {name: value.copy() for name, value in _input_arrays(problem).items()}
    first = _run(parameters, problem, mode)
    kept = {name: value.copy() for name, value in first.items()}
    for value in first.values():
        if value.flags.writeable:
            value[...] = -12345 if value.dtype == np.int64 else -12345.678
    second = _run(parameters, problem, mode)
    _assert_bit_identical(second, kept, "second call after mutating first outputs")
    for name, value in _input_arrays(problem).items():
        assert np.array_equal(value, snapshots[name]), f"input {name} changed"
    third = _run(parameters, problem, mode)
    _assert_bit_identical(third, kept, "third call")


@pytest.mark.parametrize("mode", MODES)
def test_results_do_not_depend_on_or_advance_the_global_numpy_rng(
    parameters: PlantParameters, mode: str
) -> None:
    problem = _make_problem(parameters, 33, 7012, mode)
    saved = np.random.get_state()
    try:
        np.random.seed(0)
        first = _run(parameters, problem, mode)
        np.random.seed(987654321)
        np.random.random(1000)
        second = _run(parameters, problem, mode)
        _assert_bit_identical(second, first, "different global RNG seeds")

        np.random.seed(1234)
        expected_draws = np.random.random(8)
        np.random.seed(1234)
        _run(parameters, problem, mode)
        assert np.array_equal(np.random.random(8), expected_draws)
    finally:
        np.random.set_state(saved)


# --------------------------------------------------------------------------
# Held (infeasible) worlds
# --------------------------------------------------------------------------


def _make_world_infeasible(batch: Batch, world: int, code: int) -> Batch:
    """Force one world into exactly one refusal kind, leaving every other world alone."""

    state, control, disturbance = batch
    consumption = disturbance.o2_consumption_mol_s.copy()
    delivered = control.effective_oxygen_delivery_mol_s.copy()
    achieved = control.achieved_oxygen_injection_mol_s.copy()
    store = state.oxygen_store_mol.copy()
    battery = state.battery_energy_wh.copy()
    generation = disturbance.generation_w.copy()

    # Start from a world that cannot be refused for oxygen or electrical reasons.
    consumption[world] = 0.0
    achieved[world] = 0.0002
    delivered[world] = 0.0001
    store[world] = 400.0
    battery[world] = 0.5 * float(np.max(battery)) + 1.0
    generation[world] = 8000.0
    if code == INFEASIBLE_OXYGEN_LOAD:
        consumption[world, 3] = 2.0 * float(state.o2_mol[world, 3]) / DT_SECONDS
        # Also violates the later checks: the earliest refusal must win.
        store[world] = 0.0
        battery[world] = 0.0
        generation[world] = 0.0
    elif code == INFEASIBLE_OXYGEN_STORE:
        store[world] = 0.5 * float(np.sum(delivered[world])) * DT_SECONDS
        battery[world] = 0.0
        generation[world] = 0.0
    elif code == INFEASIBLE_ELECTRICAL:
        battery[world] = 0.0
        generation[world] = 0.0
    else:
        raise AssertionError(code)
    return (
        replace_fields(state, oxygen_store_mol=store, battery_energy_wh=battery),
        replace_fields(
            control,
            achieved_oxygen_injection_mol_s=achieved,
            effective_oxygen_delivery_mol_s=delivered,
        ),
        replace_fields(disturbance, o2_consumption_mol_s=consumption, generation_w=generation),
    )


@pytest.mark.parametrize(
    "code", [INFEASIBLE_OXYGEN_LOAD, INFEASIBLE_OXYGEN_STORE, INFEASIBLE_ELECTRICAL]
)
@pytest.mark.parametrize("world", [0, 9, 20])
def test_infeasible_world_is_held_with_zero_diagnostics_and_leaves_neighbours_bit_identical(
    parameters: PlantParameters, code: int, world: int
) -> None:
    worlds = 21
    batch = random_batch(parameters, worlds, RECORDED_SEEDS[2])
    baseline = _step_fields(_step(parameters, batch))
    modified = _make_world_infeasible(batch, world, code)
    output = _step(parameters, modified)
    result = _step_fields(output)

    assert int(output.diagnostics.infeasible_code[world]) == code
    for name in STATE_FIELDS:
        assert np.array_equal(
            getattr(output.state, name)[world], getattr(modified[0], name)[world]
        ), f"held world changed {name}"
    for item in fields(output.diagnostics):
        if item.name == "infeasible_code":
            continue
        value = getattr(output.diagnostics, item.name)[world]
        assert np.all(value == 0.0), f"held world has nonzero {item.name}"

    others = np.arange(worlds) != world
    assert np.any(baseline["diagnostics.infeasible_code"][others] == FEASIBLE)
    _assert_bit_identical(
        _select(result, others), _select(baseline, others), "neighbours of held world"
    )


def test_rollout_holds_a_constructed_infeasible_world_and_advances_its_neighbours_unchanged(
    parameters: PlantParameters,
) -> None:
    worlds = 21
    world = 9
    steps = ROLLOUT_REPRODUCIBILITY_STEPS
    base = _make_problem(parameters, worlds, 7013, "rollout")
    baseline = _run(parameters, base, "rollout")
    state, controls, disturbances = base
    held_state, held_control, held_disturbance = _make_world_infeasible(
        (state, controls[0], disturbances[0]), world, INFEASIBLE_ELECTRICAL
    )
    modified: Problem = (
        held_state,
        (held_control,) + controls[1:],
        (held_disturbance,) + disturbances[1:],
    )
    result = _run(parameters, modified, "rollout")
    assert int(result["infeasible_code"][world]) == INFEASIBLE_ELECTRICAL
    assert int(result["first_infeasible_step"][world]) == 0
    assert int(result["steps_applied"][world]) == 0
    for name in STATE_FIELDS:
        assert np.array_equal(
            result[f"state.{name}"][world], getattr(held_state, name)[world]
        ), name
    others = np.arange(worlds) != world
    assert np.any(baseline["steps_applied"][others] == steps)
    _assert_bit_identical(
        _select(result, others), _select(baseline, others), "neighbours in rollout"
    )


def test_rollout_does_not_abort_the_batch_after_a_world_accepted_inside_the_electrical_tolerance(
    parameters: PlantParameters,
) -> None:
    """A step the plant accepts must leave a state the next step accepts.

    The plant accepts a battery withdrawal that exceeds the available energy
    by up to ``max(1e-12, 1e-10 * max(1, served_load_wh))``.  World ``edge``
    has an empty battery and a generation shortfall of 3e-8 W, well inside
    that tolerance, so it is feasible.  Every other world is untouched and
    must advance exactly as it does without the edge world.
    """

    worlds = 9
    edge = 4
    state, control, disturbance = random_batch(parameters, worlds, RECORDED_SEEDS[3])
    load_w = _hand_served_load_wh(parameters, control) * (3600.0 / DT_SECONDS)
    battery = state.battery_energy_wh.copy()
    generation = disturbance.generation_w.copy()
    battery[edge] = 0.0
    generation[edge] = float(load_w[edge]) - 3e-8
    edge_state = replace_fields(state, battery_energy_wh=battery)
    edge_disturbance = replace_fields(disturbance, generation_w=generation)

    one_step = _step(parameters, (edge_state, control, edge_disturbance))
    assert int(one_step.diagnostics.infeasible_code[edge]) == FEASIBLE
    assert 0.0 < float(one_step.diagnostics.battery_withdrawn_wh[edge]) < 1e-8

    baseline = rollout_arrays(
        parameters, state, (control,) * 2, (disturbance,) * 2, dt_seconds=DT_SECONDS
    )
    result = rollout_arrays(
        parameters,
        edge_state,
        (control,) * 2,
        (edge_disturbance,) * 2,
        dt_seconds=DT_SECONDS,
    )
    others = np.arange(worlds) != edge
    assert np.all(baseline.steps_applied == 2)
    assert np.array_equal(result.steps_applied[others], baseline.steps_applied[others])
    for name in STATE_FIELDS:
        assert np.array_equal(
            getattr(result.final_state, name)[others],
            getattr(baseline.final_state, name)[others],
        ), name
    assert int(result.steps_applied[edge]) >= 1
