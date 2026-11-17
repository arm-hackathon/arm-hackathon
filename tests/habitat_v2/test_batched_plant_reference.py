"""NumPy reference oracle versus the authoritative scalar plant (issue 92).

Every comparison here runs ``physics.py`` (through ``scalar_reference_step``)
and checks ``step_arrays`` / ``rollout_arrays`` against it under the
predeclared ``reference_tolerances("float64", ...)`` table.  Two kinds of
disagreement are reported separately:

* a *decision disagreement* (infeasible code or a branch indicator such as
  capture-limited-by-sorbent, passive-condensation-active or
  charging-versus-discharging), which must be exactly zero; and
* a *numeric error* on a ``STATE_FIELDS`` entry, which must be inside
  ``Tolerance.allowed(reference)``.

The array reference is an oracle for later backends, not an authority, and
nothing here is evidence about hardware or about an HMC episode.
"""

from __future__ import annotations

import copy
import functools
import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pytest

from aeolus.habitat_v2.batched_plant_contract import (
    BATTERY_STATE_SLACK_WH,
    FEASIBLE,
    INFEASIBLE_ELECTRICAL,
    INFEASIBLE_OXYGEN_LOAD,
    INFEASIBLE_OXYGEN_STORE,
    OUTSIDE_SUPPORTED_DOMAIN,
    STATE_FIELDS,
    SUPPORTED_SERVED_LOAD_WH_PER_STEP,
    SUPPORTED_ZONE_IDS,
    AchievedControl,
    PlantParameters,
    PlantStateArrays,
    Tolerance,
    reference_tolerances,
    validate_step_inputs,
)
from aeolus.habitat_v2.batched_plant_reference import (
    ScalarReferenceStep,
    StepOutput,
    concatenate_worlds,
    rollout_arrays,
    scalar_reference_step,
    step_arrays,
)
from aeolus.habitat_v2.physics import initial_state
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.state import (
    GAS_CONSTANT_J_PER_MOL_K,
    PlantState,
    saturation_vapor_pressure_pa,
)

from ._batched_plant_helpers import (
    SUPPORTED_SCENARIO_FILES,
    load_scenario,
    replace_fields,
    scalar_trajectory,
    scenario_mapping,
    stress_scenario,
    stress_scenario_mapping,
)

ONE_STEP = reference_tolerances("float64", "one_step")
ROLLOUT = reference_tolerances("float64", "rollout")
STRESS_STEPS = 240
ZONE_INDEX = {zone_id: index for index, zone_id in enumerate(SUPPORTED_ZONE_IDS)}


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------


def state_max_abs_errors(
    candidate: PlantStateArrays, reference: PlantStateArrays
) -> dict[str, float]:
    """Per-variable maximum absolute error over every world and zone."""

    return {
        name: float(
            np.max(np.abs(getattr(candidate, name) - getattr(reference, name)))
        )
        for name in STATE_FIELDS
    }


def state_tolerance_exceedances(
    candidate: PlantStateArrays,
    reference: PlantStateArrays,
    tolerances: Mapping[str, Tolerance],
) -> dict[str, float]:
    """Variables whose error is NOT inside the tolerance, with the max error."""

    exceeded: dict[str, float] = {}
    for name in STATE_FIELDS:
        expected = getattr(reference, name)
        error = np.abs(getattr(candidate, name) - expected)
        if not bool(np.all(error <= tolerances[name].allowed(expected))):
            exceeded[name] = float(np.max(error))
    return exceeded


def assert_state_within_tolerance(
    candidate: PlantStateArrays,
    reference: PlantStateArrays,
    tolerances: Mapping[str, Tolerance],
    context: str,
) -> dict[str, float]:
    """Assert every ``STATE_FIELDS`` entry is inside its tolerance.

    Returns the per-variable maximum absolute error so callers can aggregate.
    """

    assert set(tolerances) == set(STATE_FIELDS)
    errors = state_max_abs_errors(candidate, reference)
    failures = []
    for name in STATE_FIELDS:
        expected = getattr(reference, name)
        error = np.abs(getattr(candidate, name) - expected)
        allowed = tolerances[name].allowed(expected)
        if not bool(np.all(error <= allowed)):
            failures.append(
                f"{name}: max abs error {errors[name]!r} exceeds allowed "
                f"{float(np.min(allowed))!r}"
            )
    assert not failures, f"numeric error ({context}): " + "; ".join(failures)
    return errors


def decision_disagreements(
    output: StepOutput,
    before: PlantStateArrays,
    reference: ScalarReferenceStep,
    world: int = 0,
) -> dict[str, int]:
    """Count threshold and branch decisions that differ from the scalar plant.

    Independent of numeric error: each entry counts worlds or zones where the
    oracle took a different branch than ``physics.py`` did.  For a step the
    scalar plant refused, only the infeasible code is defined.
    """

    diagnostics = output.diagnostics
    counts = {
        "infeasible_code": int(
            int(diagnostics.infeasible_code[world]) != reference.infeasible_code
        )
    }
    if reference.receipt is None:
        return counts
    recirculation = reference.receipt["recirculation"]
    electrical = reference.receipt["electrical"]
    sorbent_before = float(before.co2_sorbent_remaining_mol[world])
    world_decisions = {
        "capture_limited_by_sorbent": (
            float(recirculation["co2_captured_mol"]) == sorbent_before,
            float(diagnostics.co2_captured_mol[world]) == sorbent_before,
        ),
        "scrubber_capture_active": (
            float(recirculation["co2_captured_mol"]) > 0.0,
            float(diagnostics.co2_captured_mol[world]) > 0.0,
        ),
        "active_condensing_active": (
            float(recirculation["water_condensed_mol"]) > 0.0,
            float(diagnostics.water_condensed_mol[world]) > 0.0,
        ),
        "generation_covers_load_charging_branch": (
            float(electrical["generation_wh"]) >= float(electrical["served_load_wh"]),
            float(diagnostics.generation_wh[world])
            >= float(diagnostics.served_load_wh[world]),
        ),
        "battery_charging": (
            float(electrical["battery_charge_stored_wh"]) > 0.0,
            float(diagnostics.battery_charge_stored_wh[world]) > 0.0,
        ),
        "battery_discharging": (
            float(electrical["battery_withdrawn_wh"]) > 0.0,
            float(diagnostics.battery_withdrawn_wh[world]) > 0.0,
        ),
        "generation_curtailed": (
            float(electrical["curtailed_generation_wh"]) > 0.0,
            float(diagnostics.curtailed_generation_wh[world]) > 0.0,
        ),
    }
    for name, (scalar_decision, oracle_decision) in world_decisions.items():
        counts[name] = int(scalar_decision != oracle_decision)
    passive = reference.receipt["passive_condensation_mol"]
    injected = reference.receipt["oxygen_injected_mol"]
    counts["passive_condensation_active"] = sum(
        int(
            (float(passive[zone_id]) > 0.0)
            != (float(diagnostics.passive_condensation_mol[world, index]) > 0.0)
        )
        for index, zone_id in enumerate(SUPPORTED_ZONE_IDS)
    )
    counts["oxygen_injection_active"] = sum(
        int(
            (float(injected[zone_id]) > 0.0)
            != (float(diagnostics.oxygen_injected_mol[world, index]) > 0.0)
        )
        for index, zone_id in enumerate(SUPPORTED_ZONE_IDS)
    )
    return counts


def assert_decisions_agree(
    output: StepOutput,
    before: PlantStateArrays,
    reference: ScalarReferenceStep,
    context: str,
    world: int = 0,
) -> dict[str, int]:
    counts = decision_disagreements(output, before, reference, world)
    disagreeing = {name: count for name, count in counts.items() if count}
    assert not disagreeing, f"decision disagreement ({context}): {disagreeing}"
    return counts


def assert_bit_identical(
    actual: PlantStateArrays, expected: PlantStateArrays, context: str
) -> None:
    for name in STATE_FIELDS:
        assert (
            getattr(actual, name).tobytes() == getattr(expected, name).tobytes()
        ), f"{context}: {name} is not bit-identical"


def assert_diagnostics_zero(output: StepOutput, world: int, context: str) -> None:
    diagnostics = output.diagnostics
    for name in diagnostics.__dataclass_fields__:
        if name == "infeasible_code":
            continue
        value = getattr(diagnostics, name)[world]
        assert bool(np.all(value == 0.0)), f"{context}: {name} is not zero when held"


# --------------------------------------------------------------------------
# Scenario and state builders (in memory only)
# --------------------------------------------------------------------------


def _dt(scenario: Scenario) -> float:
    return float(scenario.data["dt_seconds"])


def _arrays(plant_state: PlantState) -> PlantStateArrays:
    return PlantStateArrays.from_plant_states((plant_state,))


def _development_mapping(**segment_changes: Any) -> dict[str, Any]:
    mapping = copy.deepcopy(scenario_mapping())
    mapping["timeline"][0].update(copy.deepcopy(segment_changes))
    return mapping


def _fan_off_mapping() -> dict[str, Any]:
    mapping = copy.deepcopy(scenario_mapping())
    assert mapping["initial_utility"]["actual_fan_speed_fraction"] == 0.0
    mapping["timeline"][0]["command"]["fan_speed_fraction"] = 0.0
    return mapping


def _all_off_mapping(generation_w: float) -> dict[str, Any]:
    mapping = _fan_off_mapping()
    segment = mapping["timeline"][0]
    command = segment["command"]
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    for zone_id in SUPPORTED_ZONE_IDS:
        command["cooling_removed_w"][zone_id] = 0.0
        command["oxygen_injection_mol_s"][zone_id] = 0.0
    segment["generation_w"] = generation_w
    return mapping


def _with_zone(plant_state: PlantState, zone_id: str, **changes: float) -> PlantState:
    zones = dict(plant_state.zones)
    zones[zone_id] = replace(zones[zone_id], **changes)
    return replace(plant_state, zones=zones)


def _with_utility(plant_state: PlantState, **changes: float) -> PlantState:
    return replace(plant_state, utility=replace(plant_state.utility, **changes))


def _feasible_twin(scenario: Scenario, plant_state: PlantState) -> PlantState:
    """Same command and actuator state, with ample inventories.

    Achieved actuation, airflow and fan power do not depend on zone oxygen,
    the oxygen store or the battery, so the twin's achieved control is the
    control the refused step would have resolved.
    """

    capacity = float(scenario.data["equipment"]["battery_capacity_wh"])
    utility = plant_state.utility
    return replace(
        plant_state,
        zones={
            zone_id: replace(zone, o2_mol=max(zone.o2_mol, 100.0))
            for zone_id, zone in plant_state.zones.items()
        },
        utility=replace(
            utility,
            oxygen_store_mol=max(utility.oxygen_store_mol, 100.0),
            battery_energy_wh=max(utility.battery_energy_wh, 0.5 * capacity),
        ),
    )


def _achieved_control(
    scenario: Scenario,
    plant_state: PlantState,
    reference: ScalarReferenceStep,
    command: Mapping[str, Any] | None = None,
) -> AchievedControl:
    if reference.control is not None:
        return reference.control
    twin = scalar_reference_step(
        scenario, _feasible_twin(scenario, plant_state), command
    )
    assert twin.infeasible_code == FEASIBLE, "feasible twin was refused"
    assert twin.control is not None
    for name in reference.disturbance.__dataclass_fields__:
        assert np.array_equal(
            getattr(twin.disturbance, name), getattr(reference.disturbance, name)
        )
    return twin.control


def compare_one_step(
    scenario: Scenario,
    plant_state: PlantState,
    context: str,
    command: Mapping[str, Any] | None = None,
) -> tuple[ScalarReferenceStep, StepOutput]:
    """Run scalar plant and oracle from one state; assert decisions then numerics."""

    reference = scalar_reference_step(scenario, plant_state, command)
    control = _achieved_control(scenario, plant_state, reference, command)
    before = _arrays(plant_state)
    output = step_arrays(
        PlantParameters.from_scenario(scenario),
        before,
        control,
        reference.disturbance,
        dt_seconds=_dt(scenario),
    )
    assert_decisions_agree(output, before, reference, context)
    assert_state_within_tolerance(output.state, reference.state, ONE_STEP, context)
    if reference.infeasible_code != FEASIBLE:
        assert_bit_identical(output.state, before, f"{context} held state")
        assert_diagnostics_zero(output, 0, context)
    return reference, output


@functools.lru_cache(maxsize=None)
def _trajectory(
    name: str,
) -> tuple[Scenario, PlantState, tuple[ScalarReferenceStep, ...]]:
    scenario = stress_scenario(STRESS_STEPS) if name == "stress" else load_scenario(name)
    initial, references = scalar_trajectory(scenario)
    return scenario, initial, references


def _pre_state(
    initial: PlantState, references: tuple[ScalarReferenceStep, ...], index: int
) -> PlantState:
    return initial if index == 0 else references[index - 1].plant_state


def _teacher_forced_worst_errors(name: str) -> dict[str, float]:
    """Teacher-force the oracle at every scalar state; return worst errors."""

    scenario, initial, references = _trajectory(name)
    parameters = PlantParameters.from_scenario(scenario)
    worst = dict.fromkeys(STATE_FIELDS, 0.0)
    for index, reference in enumerate(references):
        assert reference.control is not None
        before = _arrays(_pre_state(initial, references, index))
        output = step_arrays(
            parameters,
            before,
            reference.control,
            reference.disturbance,
            dt_seconds=_dt(scenario),
        )
        context = f"{name} teacher-forced step {index}"
        assert_decisions_agree(output, before, reference, context)
        errors = assert_state_within_tolerance(
            output.state, reference.state, ONE_STEP, context
        )
        worst = {key: max(worst[key], errors[key]) for key in STATE_FIELDS}
    return worst


# --------------------------------------------------------------------------
# Teacher-forced and free-running trajectories
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_file", SUPPORTED_SCENARIO_FILES)
def test_one_step_teacher_forced_oracle_matches_scalar_plant_at_every_step_of_supported_scenario(
    scenario_file: str,
) -> None:
    scenario, _, references = _trajectory(scenario_file)
    assert len(references) == scenario.data["steps"]
    worst = _teacher_forced_worst_errors(scenario_file)
    assert set(worst) == set(STATE_FIELDS)
    assert all(math.isfinite(value) for value in worst.values())


def test_teacher_forced_oracle_stays_within_one_step_tolerance_at_every_step_of_long_stress_rollout() -> None:
    _, _, references = _trajectory("stress")
    assert len(references) == STRESS_STEPS
    # The helper asserts decisions and the one-step tolerance at each step.
    worst = _teacher_forced_worst_errors("stress")
    assert set(worst) == set(STATE_FIELDS)
    assert all(math.isfinite(value) for value in worst.values())


def test_free_running_rollout_of_long_stress_scenario_matches_scalar_final_state_within_rollout_tolerance() -> None:
    scenario, initial, references = _trajectory("stress")
    controls = tuple(reference.control for reference in references)
    assert all(control is not None for control in controls)
    disturbances = tuple(reference.disturbance for reference in references)
    output = rollout_arrays(
        PlantParameters.from_scenario(scenario),
        _arrays(initial),
        controls,
        disturbances,
        dt_seconds=_dt(scenario),
    )
    assert output.steps_applied.tolist() == [STRESS_STEPS]
    assert output.infeasible_code.tolist() == [FEASIBLE], "decision disagreement"
    assert output.first_infeasible_step.tolist() == [-1]
    assert references[-1].plant_state.step == STRESS_STEPS
    assert_state_within_tolerance(
        output.final_state,
        references[-1].state,
        ROLLOUT,
        "stress free-running rollout final state",
    )


def test_free_running_rollout_prefixes_match_scalar_plant_within_rollout_tolerance() -> None:
    scenario, initial, references = _trajectory("stress")
    parameters = PlantParameters.from_scenario(scenario)
    for horizon in (1, 7, 80, 85, 160, 165, 239):
        output = rollout_arrays(
            parameters,
            _arrays(initial),
            tuple(reference.control for reference in references[:horizon]),
            tuple(reference.disturbance for reference in references[:horizon]),
            dt_seconds=_dt(scenario),
        )
        assert output.steps_applied.tolist() == [horizon]
        assert output.infeasible_code.tolist() == [FEASIBLE], "decision disagreement"
        assert_state_within_tolerance(
            output.final_state,
            references[horizon - 1].state,
            ROLLOUT,
            f"stress free-running rollout horizon {horizon}",
        )


def test_stress_trajectory_exercises_both_electrical_branches_fan_off_and_degraded_delivery() -> None:
    """Guard that the long comparison is not vacuous."""

    _, _, references = _trajectory("stress")
    receipts = [reference.receipt for reference in references]
    assert any(r["electrical"]["battery_withdrawn_wh"] > 0.0 for r in receipts)
    assert any(r["electrical"]["battery_charge_stored_wh"] > 0.0 for r in receipts)
    assert any(r["recirculation"]["co2_captured_mol"] > 0.0 for r in receipts)
    assert any(r["recirculation"]["water_condensed_mol"] > 0.0 for r in receipts)
    controls = [reference.control for reference in references]
    assert any(bool(np.all(c.zone_airflow_m3_s == 0.0)) for c in controls)
    assert any(float(c.scrubber_capture_ability[0]) < 1.0 for c in controls)
    assert any(float(c.condenser_removal_ability[0]) < 1.0 for c in controls)
    assert any(
        bool(np.any(c.effective_cooling_delivery_w < c.achieved_cooling_removed_w))
        for c in controls
    )
    assert any(
        bool(
            np.any(
                c.effective_oxygen_delivery_mol_s < c.achieved_oxygen_injection_mol_s
            )
        )
        for c in controls
    )


# --------------------------------------------------------------------------
# Diagnostics versus the scalar receipt
# --------------------------------------------------------------------------

# The contract predeclares tolerances for state variables only.  Each
# diagnostic is checked under the tolerance of the state variable it
# accumulates into.
_WORLD_DIAGNOSTIC_SOURCES: tuple[tuple[str, str, str, str], ...] = (
    ("co2_captured_mol", "recirculation", "co2_captured_mol", "captured_co2_mol"),
    (
        "water_condensed_mol",
        "recirculation",
        "water_condensed_mol",
        "condensed_water_mol",
    ),
    ("served_load_wh", "electrical", "served_load_wh", "battery_energy_wh"),
    ("generation_wh", "electrical", "generation_wh", "battery_energy_wh"),
    (
        "battery_withdrawn_wh",
        "electrical",
        "battery_withdrawn_wh",
        "battery_energy_wh",
    ),
    (
        "battery_charge_stored_wh",
        "electrical",
        "battery_charge_stored_wh",
        "battery_energy_wh",
    ),
    (
        "curtailed_generation_wh",
        "electrical",
        "curtailed_generation_wh",
        "battery_energy_wh",
    ),
    (
        "charge_conversion_loss_wh",
        "electrical",
        "charge_conversion_loss_wh",
        "battery_energy_wh",
    ),
    (
        "discharge_conversion_loss_wh",
        "electrical",
        "discharge_conversion_loss_wh",
        "battery_energy_wh",
    ),
    (
        "external_heat_received_step_j",
        "thermal",
        "external_heat_received_j",
        "external_heat_received_j",
    ),
    (
        "external_heat_rejected_step_j",
        "thermal",
        "external_heat_rejected_j",
        "external_heat_rejected_j",
    ),
)
_ZONE_DIAGNOSTIC_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("passive_condensation_mol", "passive_condensation_mol", "condensed_water_mol"),
    ("oxygen_injected_mol", "oxygen_injected_mol", "oxygen_store_mol"),
)


def _diagnostic_failures(
    output: StepOutput, reference: ScalarReferenceStep
) -> list[str]:
    assert reference.receipt is not None
    failures = []
    diagnostics = output.diagnostics
    for name, section, key, tolerance_name in _WORLD_DIAGNOSTIC_SOURCES:
        expected = np.array([float(reference.receipt[section][key])])
        error = np.abs(getattr(diagnostics, name) - expected)
        if not bool(np.all(error <= ONE_STEP[tolerance_name].allowed(expected))):
            failures.append(f"{name}: abs error {float(error[0])!r}")
    for name, key, tolerance_name in _ZONE_DIAGNOSTIC_SOURCES:
        expected = np.array(
            [[float(reference.receipt[key][zone_id]) for zone_id in SUPPORTED_ZONE_IDS]]
        )
        error = np.abs(getattr(diagnostics, name) - expected)
        if not bool(np.all(error <= ONE_STEP[tolerance_name].allowed(expected))):
            failures.append(f"{name}: max abs error {float(np.max(error))!r}")
    return failures


@pytest.mark.parametrize(
    "name", ("stress", "habitat_v2_forecast_development.json")
)
def test_step_diagnostics_agree_with_scalar_receipt_at_every_step(name: str) -> None:
    scenario, initial, references = _trajectory(name)
    parameters = PlantParameters.from_scenario(scenario)
    for index, reference in enumerate(references):
        output = step_arrays(
            parameters,
            _arrays(_pre_state(initial, references, index)),
            reference.control,
            reference.disturbance,
            dt_seconds=_dt(scenario),
        )
        failures = _diagnostic_failures(output, reference)
        assert not failures, f"numeric error ({name} diagnostics step {index}): {failures}"


def test_recirculation_heat_diagnostic_matches_scalar_thermal_receipt_per_zone() -> None:
    scenario, initial, references = _trajectory("stress")
    parameters = PlantParameters.from_scenario(scenario)
    tolerance = ONE_STEP["external_heat_rejected_j"]
    for index in (0, 1, 40, 79, 80, 120, 159, 160, 161, 200, 239):
        reference = references[index]
        output = step_arrays(
            parameters,
            _arrays(_pre_state(initial, references, index)),
            reference.control,
            reference.disturbance,
            dt_seconds=_dt(scenario),
        )
        expected = np.array(
            [
                [
                    float(
                        reference.receipt["thermal"]["zones"][zone_id][
                            "recirculation_heat_added_j"
                        ]
                    )
                    for zone_id in SUPPORTED_ZONE_IDS
                ]
            ]
        )
        error = np.abs(output.diagnostics.recirculation_heat_added_j - expected)
        assert bool(np.all(error <= tolerance.allowed(expected))), (
            f"numeric error (recirculation heat step {index}): "
            f"{float(np.max(error))!r}"
        )


# --------------------------------------------------------------------------
# Fault ordering: faults at emitted step state.step + 1, loads at state.step
# --------------------------------------------------------------------------


def test_delivery_faults_starting_at_emitted_step_apply_while_loads_still_use_previous_segment() -> None:
    mapping = stress_scenario_mapping(STRESS_STEPS)
    profiles = {profile["id"]: profile for profile in mapping["fault_profiles"]}
    cooling_fault = profiles["stress-cooling"]
    condenser_fault = profiles["stress-condenser"]
    fault_step = cooling_fault["start_step"]
    assert condenser_fault["start_step"] == fault_step
    # The fault's first emitted step coincides with the timeline cut, so the
    # transition out of state.step == fault_step - 1 pairs the *old* segment's
    # loads with the *new* fault.
    assert mapping["timeline"][0]["end_step"] == fault_step
    old_generation = mapping["timeline"][0]["generation_w"]
    new_generation = mapping["timeline"][1]["generation_w"]
    assert old_generation != new_generation

    scenario, initial, references = _trajectory("stress")
    zone = ZONE_INDEX[cooling_fault["zone_id"]]
    before_fault = references[fault_step - 2]
    at_fault = references[fault_step - 1]
    pre_state = _pre_state(initial, references, fault_step - 1)
    assert pre_state.step == fault_step - 1
    assert at_fault.plant_state.step == fault_step

    assert float(before_fault.control.condenser_removal_ability[0]) == 1.0
    assert (
        before_fault.control.effective_cooling_delivery_w[0, zone]
        == before_fault.control.achieved_cooling_removed_w[0, zone]
    )

    achieved = float(at_fault.control.achieved_cooling_removed_w[0, zone])
    effective = float(at_fault.control.effective_cooling_delivery_w[0, zone])
    assert achieved > 0.0
    assert effective < achieved
    assert effective == achieved * cooling_fault["start_multiplier"]
    ability = float(at_fault.control.condenser_removal_ability[0])
    assert ability == condenser_fault["start_multiplier"]
    assert ability < 1.0
    assert float(at_fault.control.achieved_condenser_duty[0]) > 0.0
    assert float(at_fault.disturbance.generation_w[0]) == old_generation
    assert float(references[fault_step].disturbance.generation_w[0]) == new_generation

    parameters = PlantParameters.from_scenario(scenario)
    before = _arrays(pre_state)
    faulted = step_arrays(
        parameters,
        before,
        at_fault.control,
        at_fault.disturbance,
        dt_seconds=_dt(scenario),
    )
    assert_decisions_agree(faulted, before, at_fault, "first delivery-fault step")
    assert_state_within_tolerance(
        faulted.state, at_fault.state, ONE_STEP, "first delivery-fault step"
    )

    unfaulted_control = replace_fields(
        at_fault.control,
        effective_cooling_delivery_w=at_fault.control.achieved_cooling_removed_w.copy(),
        effective_oxygen_delivery_mol_s=(
            at_fault.control.achieved_oxygen_injection_mol_s.copy()
        ),
        scrubber_capture_ability=np.array([1.0]),
        condenser_removal_ability=np.array([1.0]),
    )
    unfaulted = step_arrays(
        parameters,
        before,
        unfaulted_control,
        at_fault.disturbance,
        dt_seconds=_dt(scenario),
    )
    exceeded = state_tolerance_exceedances(unfaulted.state, at_fault.state, ONE_STEP)
    assert {"temperature_k", "condensed_water_mol", "water_vapor_mol"} <= set(
        exceeded
    ), f"un-faulted control did not change the result: {exceeded}"


def test_scrubber_fault_ability_follows_emitted_step_profile_value() -> None:
    mapping = stress_scenario_mapping(STRESS_STEPS)
    profile = {item["id"]: item for item in mapping["fault_profiles"]}[
        "stress-scrubber"
    ]
    start = profile["start_step"]
    span = profile["end_step"] - start - 1
    _, _, references = _trajectory("stress")
    for state_step in (start - 2, start - 1, start, start + 1, start + 20):
        emitted = state_step + 1
        expected = (
            1.0
            if emitted < start
            else profile["start_multiplier"]
            + (profile["end_multiplier"] - profile["start_multiplier"])
            * ((emitted - start) / float(span))
        )
        ability = float(references[state_step].control.scrubber_capture_ability[0])
        assert ability == expected, f"state.step {state_step}"
    assert float(references[start].control.scrubber_capture_ability[0]) < 1.0


def test_fan_fault_first_reduces_airflow_on_the_transition_into_its_start_step_and_unfaulted_airflow_breaks_parity() -> None:
    mapping = stress_scenario_mapping(STRESS_STEPS)
    fan_fault = {item["id"]: item for item in mapping["fault_profiles"]}["stress-fan"]
    fault_step = fan_fault["start_step"]
    unfaulted_mapping = copy.deepcopy(mapping)
    unfaulted_mapping["fault_profiles"] = [
        profile
        for profile in unfaulted_mapping["fault_profiles"]
        if profile["id"] != "stress-fan"
    ]
    unfaulted_scenario = Scenario.from_mapping(unfaulted_mapping)

    scenario, initial, references = _trajectory("stress")
    parameters = PlantParameters.from_scenario(scenario)

    # One transition earlier (emitted step fault_step - 1) the fault is not
    # yet evaluated: removing it changes nothing, bit for bit.
    earlier_state = _pre_state(initial, references, fault_step - 2)
    earlier_twin = scalar_reference_step(unfaulted_scenario, earlier_state)
    assert np.array_equal(
        earlier_twin.control.zone_airflow_m3_s,
        references[fault_step - 2].control.zone_airflow_m3_s,
    )

    pre_state = _pre_state(initial, references, fault_step - 1)
    assert pre_state.step == fault_step - 1
    at_fault = references[fault_step - 1]
    twin = scalar_reference_step(unfaulted_scenario, pre_state)
    assert twin.infeasible_code == FEASIBLE
    assert bool(
        np.all(at_fault.control.zone_airflow_m3_s < twin.control.zone_airflow_m3_s)
    )
    assert float(at_fault.control.fan_electrical_power_w[0]) < float(
        twin.control.fan_electrical_power_w[0]
    )

    before = _arrays(pre_state)
    faulted = step_arrays(
        parameters,
        before,
        at_fault.control,
        at_fault.disturbance,
        dt_seconds=_dt(scenario),
    )
    assert_decisions_agree(faulted, before, at_fault, "first fan-fault step")
    assert_state_within_tolerance(
        faulted.state, at_fault.state, ONE_STEP, "first fan-fault step"
    )
    unfaulted = step_arrays(
        parameters, before, twin.control, at_fault.disturbance, dt_seconds=_dt(scenario)
    )
    exceeded = state_tolerance_exceedances(unfaulted.state, at_fault.state, ONE_STEP)
    assert {"co2_mol", "temperature_k", "battery_energy_wh"} <= set(
        exceeded
    ), f"un-faulted airflow did not change the result: {exceeded}"


# --------------------------------------------------------------------------
# Actuator limits: requested is not achieved
# --------------------------------------------------------------------------


def _far_command() -> dict[str, Any]:
    command = copy.deepcopy(scenario_mapping()["timeline"][0]["command"])
    command["fan_speed_fraction"] = 1.0
    command["scrubber_duty"] = 1.0
    command["condenser_duty"] = 1.0
    for zone_id in SUPPORTED_ZONE_IDS:
        command["cooling_removed_w"][zone_id] = 1000.0
        command["oxygen_injection_mol_s"][zone_id] = 0.0
    command["oxygen_injection_mol_s"]["laboratory"] = 0.003
    return command


def test_slew_limited_achieved_control_matches_scalar_plant_and_requested_values_do_not() -> None:
    scenario = load_scenario()
    parameters = PlantParameters.from_scenario(scenario)
    command = _far_command()
    requested_cooling = np.array(
        [[command["cooling_removed_w"][zone_id] for zone_id in SUPPORTED_ZONE_IDS]]
    )
    requested_oxygen = np.array(
        [[command["oxygen_injection_mol_s"][zone_id] for zone_id in SUPPORTED_ZONE_IDS]]
    )
    plant_state = initial_state(scenario)
    slew_limited_steps = 0
    for step in range(3):
        context = f"far requested command step {step}"
        reference, _ = compare_one_step(scenario, plant_state, context, command)
        assert reference.infeasible_code == FEASIBLE
        actuators = reference.receipt["actuators"]
        control = reference.control
        assert actuators["scrubber"]["requested_duty"] == 1.0
        assert float(control.achieved_scrubber_duty[0]) == (
            actuators["scrubber"]["achieved_duty"]
        )
        assert bool(np.all(control.achieved_cooling_removed_w <= requested_cooling))
        assert bool(np.all(control.achieved_oxygen_injection_mol_s <= requested_oxygen))
        limited = (
            actuators["fan"]["achieved_fraction"] < actuators["fan"]["requested_fraction"]
            and float(control.achieved_scrubber_duty[0]) < 1.0
            and float(control.achieved_condenser_duty[0]) < 1.0
            and bool(np.all(control.achieved_cooling_removed_w < requested_cooling))
            and float(control.achieved_oxygen_injection_mol_s[0, ZONE_INDEX["laboratory"]])
            < 0.003
        )
        if limited:
            slew_limited_steps += 1
            requested_as_achieved = replace_fields(
                control,
                achieved_scrubber_duty=np.array([1.0]),
                achieved_condenser_duty=np.array([1.0]),
                achieved_cooling_removed_w=requested_cooling.copy(),
                effective_cooling_delivery_w=requested_cooling.copy(),
                achieved_oxygen_injection_mol_s=requested_oxygen.copy(),
                effective_oxygen_delivery_mol_s=requested_oxygen.copy(),
            )
            wrong = step_arrays(
                parameters,
                _arrays(plant_state),
                requested_as_achieved,
                reference.disturbance,
                dt_seconds=_dt(scenario),
            )
            exceeded = state_tolerance_exceedances(
                wrong.state, reference.state, ONE_STEP
            )
            assert {
                "temperature_k",
                "o2_mol",
                "oxygen_store_mol",
                "battery_energy_wh",
                "external_heat_rejected_j",
            } <= set(exceeded), f"{context}: requested values matched: {exceeded}"
        plant_state = reference.plant_state
    assert slew_limited_steps >= 1


def test_first_step_achieved_values_equal_one_slew_increment_from_actuator_state() -> None:
    mapping = scenario_mapping()
    scenario = Scenario.from_mapping(copy.deepcopy(mapping))
    dt = _dt(scenario)
    reference = scalar_reference_step(scenario, initial_state(scenario), _far_command())
    control = reference.control
    equipment = mapping["equipment"]
    feedback = mapping["actuator_feedback"]
    assert float(control.achieved_scrubber_duty[0]) == (
        equipment["scrubber_duty_slew_per_s"] * dt
    )
    assert float(control.achieved_condenser_duty[0]) == (
        equipment["condenser_duty_slew_per_s"] * dt
    )
    assert bool(
        np.all(control.achieved_cooling_removed_w == feedback["cooling_slew_w_per_s"] * dt)
    )
    oxygen = control.achieved_oxygen_injection_mol_s[0]
    assert oxygen[ZONE_INDEX["laboratory"]] == feedback["oxygen_slew_mol_s2"] * dt
    assert float(np.sum(oxygen)) == feedback["oxygen_slew_mol_s2"] * dt
    assert reference.receipt["actuators"]["fan"]["achieved_fraction"] == (
        mapping["air_network"]["fan"]["speed_slew_fraction_per_s"] * dt
    )


# --------------------------------------------------------------------------
# Fan off
# --------------------------------------------------------------------------


def test_fan_off_gives_exactly_zero_airflow_and_no_cross_zone_transfer() -> None:
    scenario = Scenario.from_mapping(_fan_off_mapping())
    plant_state = initial_state(scenario)
    assert plant_state.utility.actual_fan_speed_fraction == 0.0
    for step in range(4):
        context = f"fan off step {step}"
        before = _arrays(plant_state)
        reference, output = compare_one_step(scenario, plant_state, context)
        assert reference.infeasible_code == FEASIBLE
        control = reference.control
        assert control.zone_airflow_m3_s.tolist() == [[0.0] * len(SUPPORTED_ZONE_IDS)]
        assert control.fan_electrical_power_w.tolist() == [0.0]
        assert float(control.achieved_scrubber_duty[0]) > 0.0
        # No exchange: nothing is captured or actively condensed, and each
        # zone keeps exactly its own inert inventory.
        assert reference.receipt["recirculation"]["co2_captured_mol"] == 0.0
        assert output.diagnostics.co2_captured_mol.tolist() == [0.0]
        assert output.diagnostics.water_condensed_mol.tolist() == [0.0]
        assert bool(np.all(output.diagnostics.recirculation_heat_added_j == 0.0))
        assert output.state.inert_mol.tobytes() == before.inert_mol.tobytes()
        assert reference.state.inert_mol.tobytes() == before.inert_mol.tobytes()
        assert (
            output.state.captured_co2_mol.tobytes() == before.captured_co2_mol.tobytes()
        )
        assert (
            output.state.co2_sorbent_remaining_mol.tobytes()
            == before.co2_sorbent_remaining_mol.tobytes()
        )
        # Each zone's CO2 changes only by its own declared source.
        expected_co2 = before.co2_mol + reference.disturbance.co2_generation_mol_s * _dt(
            scenario
        )
        assert output.state.co2_mol.tobytes() == expected_co2.tobytes()
        assert reference.state.co2_mol.tobytes() == expected_co2.tobytes()
        plant_state = reference.plant_state


# --------------------------------------------------------------------------
# Infeasible worlds
# --------------------------------------------------------------------------

_INFEASIBILITY_ORDER = ("load", "store", "electrical")
_INFEASIBILITY_CODE = {
    "load": INFEASIBLE_OXYGEN_LOAD,
    "store": INFEASIBLE_OXYGEN_STORE,
    "electrical": INFEASIBLE_ELECTRICAL,
}


def _infeasible_case(causes: tuple[str, ...]) -> tuple[Scenario, PlantState]:
    mapping = (
        _development_mapping(generation_w=0.0)
        if "electrical" in causes
        else _development_mapping()
    )
    scenario = Scenario.from_mapping(mapping)
    plant_state = initial_state(scenario)
    if "electrical" in causes:
        plant_state = _with_utility(plant_state, battery_energy_wh=0.0)
    if "store" in causes:
        plant_state = _with_utility(plant_state, oxygen_store_mol=0.001)
    if "load" in causes:
        plant_state = _with_zone(plant_state, "laboratory", o2_mol=0.001)
    return scenario, plant_state


_INFEASIBLE_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("load",),
    ("store",),
    ("electrical",),
    ("load", "store"),
    ("load", "electrical"),
    ("store", "electrical"),
    ("load", "store", "electrical"),
)


@pytest.mark.parametrize(
    "causes", _INFEASIBLE_COMBINATIONS, ids=lambda causes: "+".join(causes)
)
def test_infeasible_world_gets_scalar_plant_classification_precedence_and_bit_identical_held_state(
    causes: tuple[str, ...],
) -> None:
    scenario, plant_state = _infeasible_case(causes)
    first_cause = next(name for name in _INFEASIBILITY_ORDER if name in causes)
    reference, output = compare_one_step(
        scenario, plant_state, f"infeasible {'+'.join(causes)}"
    )
    assert reference.control is None
    assert reference.receipt is None
    assert reference.infeasible_code == _INFEASIBILITY_CODE[first_cause]
    assert output.diagnostics.infeasible_code.tolist() == [
        _INFEASIBILITY_CODE[first_cause]
    ]
    assert_bit_identical(output.state, _arrays(plant_state), "held infeasible world")
    assert_bit_identical(reference.state, _arrays(plant_state), "scalar refused state")


def test_each_single_cause_is_feasible_once_only_that_inventory_is_restored() -> None:
    """The three causes are independent: fixing the one cause makes it feasible."""

    for cause in _INFEASIBILITY_ORDER:
        scenario, plant_state = _infeasible_case((cause,))
        refused = scalar_reference_step(scenario, plant_state)
        assert refused.infeasible_code == _INFEASIBILITY_CODE[cause]
        reference, output = compare_one_step(
            scenario, _feasible_twin(scenario, plant_state), f"restored {cause}"
        )
        assert reference.infeasible_code == FEASIBLE
        assert output.diagnostics.infeasible_code.tolist() == [FEASIBLE]


def test_infeasible_worlds_are_flagged_per_world_without_disturbing_a_feasible_neighbour() -> None:
    feasible_scenario = load_scenario()
    feasible_state = initial_state(feasible_scenario)
    feasible_reference = scalar_reference_step(feasible_scenario, feasible_state)
    assert feasible_reference.infeasible_code == FEASIBLE
    parameters = PlantParameters.from_scenario(feasible_scenario)

    states = [_arrays(feasible_state)]
    controls = [feasible_reference.control]
    disturbances = [feasible_reference.disturbance]
    references = [feasible_reference]
    for causes in _INFEASIBLE_COMBINATIONS:
        scenario, plant_state = _infeasible_case(causes)
        reference = scalar_reference_step(scenario, plant_state)
        states.append(_arrays(plant_state))
        controls.append(_achieved_control(scenario, plant_state, reference))
        disturbances.append(reference.disturbance)
        references.append(reference)
    batch_state = concatenate_worlds(tuple(states))
    output = step_arrays(
        parameters,
        batch_state,
        concatenate_worlds(tuple(controls)),
        concatenate_worlds(tuple(disturbances)),
        dt_seconds=_dt(feasible_scenario),
    )
    scalar_codes = [reference.infeasible_code for reference in references]
    assert scalar_codes == [0, 1, 2, 3, 1, 1, 2, 1]
    assert output.diagnostics.infeasible_code.tolist() == scalar_codes, (
        "decision disagreement (batched infeasible codes)"
    )
    for world, reference in enumerate(references):
        assert_decisions_agree(
            output, batch_state, reference, f"batched world {world}", world=world
        )
        for name in STATE_FIELDS:
            candidate = getattr(output.state, name)[world : world + 1]
            expected = getattr(reference.state, name)
            error = np.abs(candidate - expected)
            assert bool(np.all(error <= ONE_STEP[name].allowed(expected))), (
                f"numeric error (batched world {world}): {name}"
            )
            if reference.infeasible_code != FEASIBLE:
                assert (
                    candidate.tobytes()
                    == getattr(batch_state, name)[world : world + 1].tobytes()
                )
        if reference.infeasible_code != FEASIBLE:
            assert_diagnostics_zero(output, world, f"batched world {world}")
    solo = step_arrays(
        parameters,
        states[0],
        controls[0],
        disturbances[0],
        dt_seconds=_dt(feasible_scenario),
    )
    for name in STATE_FIELDS:
        assert (
            getattr(output.state, name)[0:1].tobytes()
            == getattr(solo.state, name).tobytes()
        )


def test_rollout_freezes_world_at_scalar_plant_refusal_step() -> None:
    """Drain the oxygen store so the scalar plant refuses a later step."""

    scenario = load_scenario()
    first = scalar_reference_step(scenario, initial_state(scenario))
    injected_first = sum(
        float(first.receipt["oxygen_injected_mol"][zone_id])
        for zone_id in SUPPORTED_ZONE_IDS
    )
    assert injected_first > 0.0
    start = _with_utility(
        initial_state(scenario), oxygen_store_mol=2.5 * injected_first
    )
    plant_state = start
    controls = []
    disturbances = []
    refused_at = -1
    last_feasible = _arrays(start)
    for index in range(6):
        reference = scalar_reference_step(scenario, plant_state)
        controls.append(_achieved_control(scenario, plant_state, reference))
        disturbances.append(reference.disturbance)
        if reference.infeasible_code != FEASIBLE:
            assert reference.infeasible_code == INFEASIBLE_OXYGEN_STORE
            refused_at = index
            break
        plant_state = reference.plant_state
        last_feasible = reference.state
    assert refused_at == 2
    # Later inputs must not be applied to a frozen world.
    controls.append(first.control)
    disturbances.append(first.disturbance)
    output = rollout_arrays(
        PlantParameters.from_scenario(scenario),
        _arrays(start),
        tuple(controls),
        tuple(disturbances),
        dt_seconds=_dt(scenario),
    )
    assert output.infeasible_code.tolist() == [INFEASIBLE_OXYGEN_STORE], (
        "decision disagreement (rollout refusal code)"
    )
    assert output.first_infeasible_step.tolist() == [refused_at]
    assert output.steps_applied.tolist() == [refused_at]
    assert_state_within_tolerance(
        output.final_state, last_feasible, ROLLOUT, "rollout frozen at refusal"
    )


# --------------------------------------------------------------------------
# Threshold-adjacent decisions
# --------------------------------------------------------------------------


def _neighbours(value: float) -> dict[str, float]:
    return {
        "below": math.nextafter(value, -math.inf),
        "at": value,
        "above": math.nextafter(value, math.inf),
    }


@pytest.mark.parametrize(
    ("side", "expected_code"),
    (("below", INFEASIBLE_OXYGEN_LOAD), ("at", FEASIBLE), ("above", FEASIBLE)),
)
def test_oxygen_load_threshold_is_strictly_greater_than_zone_inventory(
    side: str, expected_code: int
) -> None:
    mapping = scenario_mapping()
    scenario = Scenario.from_mapping(copy.deepcopy(mapping))
    zone_id = "laboratory"
    consumed = float(
        mapping["timeline"][0]["loads"][zone_id]["o2_consumption_mol_s"]
    ) * _dt(scenario)
    assert consumed > 0.0
    plant_state = _with_zone(
        initial_state(scenario), zone_id, o2_mol=_neighbours(consumed)[side]
    )
    reference, output = compare_one_step(
        scenario, plant_state, f"zone oxygen {side} consumed"
    )
    assert reference.infeasible_code == expected_code
    assert output.diagnostics.infeasible_code.tolist() == [expected_code]


@pytest.mark.parametrize(
    ("side", "expected_code"),
    (("below", INFEASIBLE_OXYGEN_STORE), ("at", FEASIBLE), ("above", FEASIBLE)),
)
def test_oxygen_store_threshold_is_strictly_greater_than_total_injection(
    side: str, expected_code: int
) -> None:
    scenario = load_scenario()
    baseline = scalar_reference_step(scenario, initial_state(scenario))
    total_injected = sum(
        float(baseline.receipt["oxygen_injected_mol"][zone_id])
        for zone_id in SUPPORTED_ZONE_IDS
    )
    assert total_injected > 0.0
    plant_state = _with_utility(
        initial_state(scenario), oxygen_store_mol=_neighbours(total_injected)[side]
    )
    reference, output = compare_one_step(
        scenario, plant_state, f"oxygen store {side} total injection"
    )
    assert reference.infeasible_code == expected_code
    assert output.diagnostics.infeasible_code.tolist() == [expected_code]
    if side == "at":
        assert reference.plant_state.utility.oxygen_store_mol == 0.0
        assert output.state.oxygen_store_mol.tolist() == [0.0]


def _discharging_baseline() -> tuple[Scenario, float, float]:
    """A zero-generation scenario and its required withdrawal and tolerance."""

    scenario = Scenario.from_mapping(_development_mapping(generation_w=0.0))
    baseline = scalar_reference_step(scenario, initial_state(scenario))
    assert baseline.infeasible_code == FEASIBLE
    electrical = baseline.receipt["electrical"]
    required = float(electrical["battery_withdrawn_wh"])
    served = float(electrical["served_load_wh"])
    assert required > 0.0
    discharge_limit = float(scenario.data["equipment"]["battery_max_discharge_w"]) * (
        _dt(scenario) / 3600.0
    )
    assert required < discharge_limit
    return scenario, required, max(1e-12, 1e-10 * max(1.0, served))


def _largest_refused_battery(required: float, tolerance: float) -> float:
    battery = required - tolerance
    while required > battery + tolerance:
        battery = math.nextafter(battery, math.inf)
    while not required > battery + tolerance:
        battery = math.nextafter(battery, -math.inf)
    return battery


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("largest_refused", INFEASIBLE_ELECTRICAL),
        ("smallest_accepted", FEASIBLE),
        ("inside_tolerance", FEASIBLE),
        ("exactly_required", FEASIBLE),
        ("above_required", FEASIBLE),
        ("well_below", INFEASIBLE_ELECTRICAL),
    ),
)
def test_battery_withdrawal_limit_includes_the_scalar_plant_tolerance(
    case: str, expected_code: int
) -> None:
    scenario, required, tolerance = _discharging_baseline()
    refused = _largest_refused_battery(required, tolerance)
    battery = {
        "largest_refused": refused,
        "smallest_accepted": math.nextafter(refused, math.inf),
        "inside_tolerance": required - 0.5 * tolerance,
        "exactly_required": required,
        "above_required": math.nextafter(required, math.inf),
        "well_below": required - 4.0 * tolerance,
    }[case]
    assert 0.0 < battery
    plant_state = _with_utility(initial_state(scenario), battery_energy_wh=battery)
    reference, output = compare_one_step(scenario, plant_state, f"battery {case}")
    assert reference.infeasible_code == expected_code
    assert output.diagnostics.infeasible_code.tolist() == [expected_code]
    if expected_code == FEASIBLE:
        assert reference.receipt["electrical"]["battery_withdrawn_wh"] == required
    if case == "exactly_required":
        assert reference.plant_state.utility.battery_energy_wh == 0.0
        assert output.state.battery_energy_wh.tolist() == [0.0]


def _generation_bracketing(served_load_wh: float, hours: float) -> tuple[float, float]:
    """Adjacent generation powers with energy just below and at/above the load."""

    generation = served_load_wh / hours
    while generation * hours >= served_load_wh:
        generation = math.nextafter(generation, -math.inf)
    below = generation
    while generation * hours < served_load_wh:
        generation = math.nextafter(generation, math.inf)
    assert math.nextafter(below, math.inf) == generation
    return below, generation


@pytest.mark.parametrize("side", ("below", "at_or_above"))
def test_generation_adjacent_to_served_load_takes_the_scalar_plant_electrical_branch(
    side: str,
) -> None:
    scenario = load_scenario()
    hours = _dt(scenario) / 3600.0
    baseline = scalar_reference_step(scenario, initial_state(scenario))
    served = float(baseline.receipt["electrical"]["served_load_wh"])
    below, at_or_above = _generation_bracketing(served, hours)
    generation = below if side == "below" else at_or_above
    adjacent = Scenario.from_mapping(_development_mapping(generation_w=generation))
    reference, output = compare_one_step(
        adjacent, initial_state(adjacent), f"generation {side} served load"
    )
    electrical = reference.receipt["electrical"]
    assert electrical["served_load_wh"] == served
    if side == "below":
        assert electrical["generation_wh"] < served
        assert electrical["battery_withdrawn_wh"] > 0.0
        assert float(output.diagnostics.battery_withdrawn_wh[0]) > 0.0
        assert electrical["battery_charge_stored_wh"] == 0.0
    else:
        assert electrical["generation_wh"] >= served
        assert electrical["battery_withdrawn_wh"] == 0.0
        assert output.diagnostics.battery_withdrawn_wh.tolist() == [0.0]
        assert output.diagnostics.discharge_conversion_loss_wh.tolist() == [0.0]


def test_generation_exactly_equal_to_served_load_takes_the_charging_branch_with_zero_flow() -> None:
    base_load_w = float(scenario_mapping()["equipment"]["base_load_w"])
    scenario = Scenario.from_mapping(_all_off_mapping(base_load_w))
    plant_state = initial_state(scenario)
    reference, output = compare_one_step(
        scenario, plant_state, "generation exactly equals served load"
    )
    electrical = reference.receipt["electrical"]
    assert electrical["generation_wh"] == electrical["served_load_wh"]
    assert electrical["battery_withdrawn_wh"] == 0.0
    assert electrical["battery_charge_stored_wh"] == 0.0
    assert electrical["curtailed_generation_wh"] == 0.0
    diagnostics = output.diagnostics
    assert diagnostics.generation_wh.tolist() == diagnostics.served_load_wh.tolist()
    assert diagnostics.battery_withdrawn_wh.tolist() == [0.0]
    assert diagnostics.battery_charge_stored_wh.tolist() == [0.0]
    assert diagnostics.curtailed_generation_wh.tolist() == [0.0]
    battery = plant_state.utility.battery_energy_wh
    assert reference.plant_state.utility.battery_energy_wh == battery
    assert output.state.battery_energy_wh.tolist() == [battery]

    hours = _dt(scenario) / 3600.0
    below, at_or_above = _generation_bracketing(
        float(electrical["served_load_wh"]), hours
    )
    assert at_or_above <= base_load_w
    short = Scenario.from_mapping(_all_off_mapping(below))
    short_reference, short_output = compare_one_step(
        short, initial_state(short), "generation one ulp short of served load"
    )
    assert short_reference.receipt["electrical"]["battery_withdrawn_wh"] > 0.0
    assert float(short_output.diagnostics.battery_withdrawn_wh[0]) > 0.0


def _capture_capacity(scenario: Scenario, baseline: ScalarReferenceStep) -> float:
    return (
        float(scenario.data["equipment"]["scrubber_max_co2_mol_s"])
        * float(baseline.control.achieved_scrubber_duty[0])
        * float(baseline.control.scrubber_capture_ability[0])
        * _dt(scenario)
    )


@pytest.mark.parametrize("side", ("below", "at", "above"))
def test_sorbent_adjacent_to_capture_capacity_limits_capture_like_the_scalar_plant(
    side: str,
) -> None:
    scenario = load_scenario()
    baseline = scalar_reference_step(scenario, initial_state(scenario))
    capacity = _capture_capacity(scenario, baseline)
    assert capacity > 0.0
    assert baseline.receipt["recirculation"]["co2_captured_mol"] == capacity, (
        "baseline step is not capacity limited"
    )
    sorbent = _neighbours(capacity)[side]
    plant_state = _with_utility(
        initial_state(scenario), co2_sorbent_remaining_mol=sorbent
    )
    reference, output = compare_one_step(
        scenario, plant_state, f"sorbent {side} capture capacity"
    )
    captured = float(reference.receipt["recirculation"]["co2_captured_mol"])
    assert captured == min(capacity, sorbent)
    assert output.diagnostics.co2_captured_mol.tolist() == [captured]
    remaining = reference.plant_state.utility.co2_sorbent_remaining_mol
    assert output.state.co2_sorbent_remaining_mol.tolist() == [remaining]
    if side == "above":
        assert remaining > 0.0
    else:
        assert remaining == 0.0


def test_sorbent_smaller_than_capture_capacity_is_exhausted_then_captures_nothing() -> None:
    scenario = load_scenario()
    baseline = scalar_reference_step(scenario, initial_state(scenario))
    capacity = _capture_capacity(scenario, baseline)
    sorbent = capacity / 3.0
    plant_state = _with_utility(
        initial_state(scenario), co2_sorbent_remaining_mol=sorbent
    )
    first, first_output = compare_one_step(
        scenario, plant_state, "sorbent exhausted this step"
    )
    assert first.receipt["recirculation"]["co2_captured_mol"] == sorbent
    assert first_output.diagnostics.co2_captured_mol.tolist() == [sorbent]
    assert first.plant_state.utility.co2_sorbent_remaining_mol == 0.0
    assert first_output.state.co2_sorbent_remaining_mol.tolist() == [0.0]
    assert first_output.state.captured_co2_mol.tolist() == [sorbent]

    second, second_output = compare_one_step(
        scenario, first.plant_state, "sorbent already exhausted"
    )
    assert float(second.control.achieved_scrubber_duty[0]) > 0.0
    assert bool(np.all(second.control.zone_airflow_m3_s > 0.0))
    assert second.receipt["recirculation"]["co2_captured_mol"] == 0.0
    assert second_output.diagnostics.co2_captured_mol.tolist() == [0.0]
    assert second_output.state.co2_sorbent_remaining_mol.tolist() == [0.0]
    assert second_output.state.captured_co2_mol.tolist() == [sorbent]


@pytest.mark.parametrize("side", ("below", "at", "above"))
def test_water_adjacent_to_saturation_condenses_passively_like_the_scalar_plant(
    side: str,
) -> None:
    # Fan off and a zone with no water source: the zone's water reaches the
    # passive-condensation comparison unchanged.
    mapping = _fan_off_mapping()
    zone_id = "equipment_power_bay"
    assert (
        mapping["timeline"][0]["loads"][zone_id]["water_vapor_generation_mol_s"] == 0.0
    )
    scenario = Scenario.from_mapping(mapping)
    baseline = scalar_reference_step(scenario, initial_state(scenario))
    assert bool(np.all(baseline.control.zone_airflow_m3_s == 0.0))
    next_temperature = baseline.plant_state.zones[zone_id].temperature_k
    volume_m3 = next(
        float(zone["volume_m3"]) for zone in mapping["zones"] if zone["id"] == zone_id
    )
    saturated = (
        saturation_vapor_pressure_pa(next_temperature)
        * volume_m3
        / (GAS_CONSTANT_J_PER_MOL_K * next_temperature)
    )
    water = _neighbours(saturated)[side]
    plant_state = _with_zone(initial_state(scenario), zone_id, water_vapor_mol=water)
    reference, output = compare_one_step(
        scenario, plant_state, f"water {side} saturation"
    )
    index = ZONE_INDEX[zone_id]
    assert reference.plant_state.zones[zone_id].temperature_k == next_temperature
    condensed = float(reference.receipt["passive_condensation_mol"][zone_id])
    oracle_condensed = float(output.diagnostics.passive_condensation_mol[0, index])
    if side == "above":
        assert condensed > 0.0
        assert reference.plant_state.zones[zone_id].water_vapor_mol == saturated
    else:
        assert condensed == 0.0
        assert reference.plant_state.zones[zone_id].water_vapor_mol == water
    assert (oracle_condensed > 0.0) == (condensed > 0.0), (
        "decision disagreement (passive condensation active): "
        f"scalar {condensed!r}, oracle {oracle_condensed!r}"
    )
    other_zones = [
        float(reference.receipt["passive_condensation_mol"][other])
        for other in SUPPORTED_ZONE_IDS
        if other != zone_id
    ]
    assert other_zones == [0.0] * (len(SUPPORTED_ZONE_IDS) - 1)


def test_water_well_above_saturation_condenses_the_same_amount_as_the_scalar_plant() -> None:
    mapping = _fan_off_mapping()
    zone_id = "equipment_power_bay"
    scenario = Scenario.from_mapping(mapping)
    start = initial_state(scenario)
    plant_state = _with_zone(
        start, zone_id, water_vapor_mol=3.0 * start.zones[zone_id].water_vapor_mol
    )
    reference, output = compare_one_step(
        scenario, plant_state, "water well above saturation"
    )
    condensed = float(reference.receipt["passive_condensation_mol"][zone_id])
    assert condensed > 1.0
    oracle = output.diagnostics.passive_condensation_mol[0, ZONE_INDEX[zone_id]]
    expected = np.array(condensed)
    assert abs(float(oracle) - condensed) <= float(
        ONE_STEP["condensed_water_mol"].allowed(expected)
    ), "numeric error (passive condensation amount)"


@pytest.mark.parametrize("side", ("below", "at"))
def test_battery_at_capacity_curtails_generation_like_the_scalar_plant(
    side: str,
) -> None:
    scenario = load_scenario()
    capacity = float(scenario.data["equipment"]["battery_capacity_wh"])
    battery = _neighbours(capacity)[side]
    plant_state = _with_utility(initial_state(scenario), battery_energy_wh=battery)
    reference, output = compare_one_step(
        scenario, plant_state, f"battery {side} capacity"
    )
    electrical = reference.receipt["electrical"]
    surplus = electrical["generation_wh"] - electrical["served_load_wh"]
    assert surplus > 0.0
    assert electrical["curtailed_generation_wh"] > 0.0
    assert float(output.diagnostics.curtailed_generation_wh[0]) > 0.0
    assert reference.plant_state.utility.battery_energy_wh <= capacity
    assert float(output.state.battery_energy_wh[0]) <= capacity
    if side == "at":
        assert electrical["battery_charge_stored_wh"] == 0.0
        assert electrical["curtailed_generation_wh"] == surplus
        assert output.diagnostics.battery_charge_stored_wh.tolist() == [0.0]
        assert output.state.battery_energy_wh.tolist() == [capacity]
    else:
        assert electrical["battery_charge_stored_wh"] > 0.0
        assert float(output.diagnostics.battery_charge_stored_wh[0]) > 0.0


def test_battery_with_ample_headroom_stores_surplus_without_curtailment_like_the_scalar_plant() -> None:
    scenario = load_scenario()
    reference, output = compare_one_step(
        scenario, initial_state(scenario), "battery with headroom"
    )
    electrical = reference.receipt["electrical"]
    assert electrical["battery_charge_stored_wh"] > 0.0
    assert electrical["curtailed_generation_wh"] == 0.0
    assert output.diagnostics.curtailed_generation_wh.tolist() == [0.0]


# --------------------------------------------------------------------------
# States the scalar plant reaches through its electrical tolerance
# --------------------------------------------------------------------------


def _state_after_withdrawal_inside_tolerance() -> tuple[
    Scenario, PlantState, ScalarReferenceStep
]:
    """A feasible scalar step whose withdrawal exceeds the battery by < tolerance.

    ``physics.py`` accepts it and leaves ``battery_energy_wh`` a hair below
    zero (about -2e-9 Wh).  That state is reachable by the authority.
    """

    scenario, required, tolerance = _discharging_baseline()
    start = _with_utility(
        initial_state(scenario), battery_energy_wh=required - 0.5 * tolerance
    )
    first = scalar_reference_step(scenario, start)
    assert first.infeasible_code == FEASIBLE
    battery = first.plant_state.utility.battery_energy_wh
    assert -tolerance <= battery < 0.0
    return scenario, start, first


def test_oracle_steps_from_scalar_reachable_battery_state_inside_withdrawal_tolerance() -> None:
    _, _, first = _state_after_withdrawal_inside_tolerance()
    battery = first.plant_state.utility.battery_energy_wh
    # Same slightly negative battery, now with generation above the load: the
    # scalar plant charges from it and stays feasible.
    charging = load_scenario()
    plant_state = _with_utility(initial_state(charging), battery_energy_wh=battery)
    scalar = scalar_reference_step(charging, plant_state)
    assert scalar.infeasible_code == FEASIBLE
    assert scalar.plant_state.utility.battery_energy_wh > 0.0
    compare_one_step(charging, plant_state, "battery inside withdrawal tolerance")


def test_rollout_continues_past_its_own_battery_state_inside_withdrawal_tolerance() -> None:
    scenario, start, first = _state_after_withdrawal_inside_tolerance()
    second = scalar_reference_step(scenario, first.plant_state)
    assert second.infeasible_code == INFEASIBLE_ELECTRICAL
    output = rollout_arrays(
        PlantParameters.from_scenario(scenario),
        _arrays(start),
        (first.control, _achieved_control(scenario, first.plant_state, second)),
        (first.disturbance, second.disturbance),
        dt_seconds=_dt(scenario),
    )
    assert output.infeasible_code.tolist() == [INFEASIBLE_ELECTRICAL], (
        "decision disagreement (rollout after withdrawal inside tolerance)"
    )
    assert output.first_infeasible_step.tolist() == [1]
    assert output.steps_applied.tolist() == [1]
    assert_state_within_tolerance(
        output.final_state, first.state, ROLLOUT, "rollout held after refusal"
    )


# --------------------------------------------------------------------------
# The tolerance tables must be able to tell a real step from no step
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_one_step_tolerances_reject_a_candidate_that_returns_its_input_unchanged(
    precision: str,
) -> None:
    from aeolus.habitat_v2.batched_plant_contract import (
        STATE_FIELDS as fields,
    )
    from aeolus.habitat_v2.batched_plant_contract import (
        PlantStateArrays as arrays,
    )
    from aeolus.habitat_v2.batched_plant_contract import (
        reference_tolerances as tolerances_for,
    )

    initial, references = scalar_trajectory(load_scenario())
    table = tolerances_for(precision, "one_step")
    previous = arrays.from_plant_states((initial,))
    # Every sink is colder than every zone in the development fixture, so no
    # passive heat is ever received and this accumulator truly never changes.
    unchanged_by_physics = {"external_heat_received_j"}
    for index, reference in enumerate(references):
        for name in fields:
            error = np.abs(getattr(previous, name) - getattr(reference.state, name))
            within = bool(np.all(error <= table[name].allowed(getattr(reference.state, name))))
            if name in unchanged_by_physics:
                assert within
            else:
                assert not within, (
                    f"{precision} one-step tolerance for {name} accepts an "
                    f"unchanged state at step {index}"
                )
        previous = reference.state


# --------------------------------------------------------------------------
# Passive heat received: scalar parity with sinks warmer than the zones
# --------------------------------------------------------------------------


def _warm_sink_scenario() -> Scenario:
    mapping = copy.deepcopy(scenario_mapping())
    for index, zone in enumerate(mapping["zones"]):
        # Half the zones warm, half left cold, so both passive branches run.
        if index % 2 == 0:
            zone["sink_temperature_k"] = 315.0
    return Scenario.from_mapping(mapping)


def test_oracle_matches_scalar_plant_when_sinks_are_warmer_than_zones() -> None:
    scenario = _warm_sink_scenario()
    parameters = PlantParameters.from_scenario(scenario)
    initial, references = scalar_trajectory(scenario)
    assert references[-1].state.external_heat_received_j[0] > 1.0e4, (
        "fixture must exercise the passive-heat-received branch"
    )
    assert references[-1].plant_state.utility.external_heat_rejected_j > 0.0

    previous = _arrays(initial)
    for index, reference in enumerate(references):
        output = step_arrays(
            parameters,
            previous,
            reference.control,
            reference.disturbance,
            dt_seconds=_dt(scenario),
        )
        assert output.diagnostics.infeasible_code.tolist() == [FEASIBLE]
        assert_state_within_tolerance(
            output.state, reference.state, ONE_STEP, f"warm sinks, step {index}"
        )
        received = reference.receipt["thermal"]["external_heat_received_j"]
        assert output.diagnostics.external_heat_received_step_j[0] == pytest.approx(
            received, rel=1e-12, abs=1e-6
        )
        assert received > 0.0
        previous = reference.state

    rollout = rollout_arrays(
        parameters,
        _arrays(initial),
        tuple(reference.control for reference in references),
        tuple(reference.disturbance for reference in references),
        dt_seconds=_dt(scenario),
    )
    assert rollout.infeasible_code.tolist() == [FEASIBLE]
    assert_state_within_tolerance(
        rollout.final_state, references[-1].state, ROLLOUT, "warm sinks, rollout"
    )


# --------------------------------------------------------------------------
# Battery slack against the scalar plant's load-dependent withdrawal tolerance
# --------------------------------------------------------------------------


def _heavy_load_mapping(base_load_w: float, battery_wh: float) -> dict[str, Any]:
    """All actuators off, no generation, long step, one large fixed load."""

    mapping = copy.deepcopy(scenario_mapping())
    mapping["dt_seconds"] = 600.0
    mapping["steps"] = 2
    mapping["equipment"]["base_load_w"] = base_load_w
    mapping["equipment"]["battery_capacity_wh"] = 200000.0
    mapping["equipment"]["battery_max_discharge_w"] = 2.0e6
    mapping["initial_utility"]["battery_energy_wh"] = battery_wh
    segment = mapping["timeline"][0]
    segment["end_step"] = 2
    segment["generation_w"] = 0.0
    command = segment["command"]
    command["fan_speed_fraction"] = 0.0
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    for zone_id in SUPPORTED_ZONE_IDS:
        command["cooling_removed_w"][zone_id] = 0.0
        command["oxygen_injection_mol_s"][zone_id] = 0.0
    return mapping


def _battery_inside_scalar_tolerance(base_load_w: float) -> tuple[Scenario, float]:
    """Scenario whose battery sits below the required withdrawal, inside tolerance."""

    probe = Scenario.from_mapping(_heavy_load_mapping(base_load_w, 150000.0))
    electrical = scalar_reference_step(probe, initial_state(probe)).receipt["electrical"]
    served = float(electrical["served_load_wh"])
    required = float(electrical["battery_withdrawn_wh"])
    tolerance = max(1e-12, 1e-10 * max(1.0, served))
    scenario = Scenario.from_mapping(
        _heavy_load_mapping(base_load_w, required - 0.9 * tolerance)
    )
    return scenario, served


@pytest.mark.parametrize("base_load_w", [600.0, 6000.0, 29000.0])
def test_scalar_plant_battery_overshoot_is_a_legal_next_input_inside_the_load_cap(
    base_load_w: float,
) -> None:
    scenario, served = _battery_inside_scalar_tolerance(base_load_w)
    assert served <= SUPPORTED_SERVED_LOAD_WH_PER_STEP
    parameters = PlantParameters.from_scenario(scenario)
    initial = initial_state(scenario)
    reference = scalar_reference_step(scenario, initial)
    assert reference.infeasible_code == FEASIBLE
    overshoot = reference.plant_state.utility.battery_energy_wh
    assert -BATTERY_STATE_SLACK_WH <= overshoot < 0.0

    output = step_arrays(
        parameters,
        _arrays(initial),
        reference.control,
        reference.disturbance,
        dt_seconds=_dt(scenario),
    )
    assert output.diagnostics.infeasible_code.tolist() == [FEASIBLE]
    assert_state_within_tolerance(
        output.state, reference.state, ONE_STEP, "battery overshoot step"
    )
    # Both the scalar plant's own next state and the oracle's are legal inputs,
    # and both sides then refuse the following step for the same reason.
    for state in (reference.state, output.state):
        validate_step_inputs(
            parameters, state, reference.control, reference.disturbance, _dt(scenario)
        )
    following = scalar_reference_step(scenario, reference.plant_state)
    assert following.infeasible_code == INFEASIBLE_ELECTRICAL
    held = step_arrays(
        parameters,
        output.state,
        reference.control,
        following.disturbance,
        dt_seconds=_dt(scenario),
    )
    assert held.diagnostics.infeasible_code.tolist() == [INFEASIBLE_ELECTRICAL]


def test_served_load_above_the_cap_is_reported_outside_the_supported_domain() -> None:
    # Reviewer's case: 200 kW base load over a 600 s step.  The scalar plant
    # accepts it and leaves the battery further below zero than the declared
    # slack, so this load is outside the research domain by declaration.
    scenario, served = _battery_inside_scalar_tolerance(200000.0)
    assert served > SUPPORTED_SERVED_LOAD_WH_PER_STEP
    parameters = PlantParameters.from_scenario(scenario)
    initial = initial_state(scenario)
    reference = scalar_reference_step(scenario, initial)
    assert reference.infeasible_code == FEASIBLE
    assert reference.plant_state.utility.battery_energy_wh < -BATTERY_STATE_SLACK_WH

    before = _arrays(initial)
    output = step_arrays(
        parameters,
        before,
        reference.control,
        reference.disturbance,
        dt_seconds=_dt(scenario),
    )
    assert output.diagnostics.infeasible_code.tolist() == [OUTSIDE_SUPPORTED_DOMAIN]
    assert_bit_identical(output.state, before, "held world above the load cap")


def test_no_in_cap_discharge_can_leave_the_battery_below_the_declared_slack() -> None:
    parameters = PlantParameters.from_scenario(
        Scenario.from_mapping(_heavy_load_mapping(600.0, 1000.0))
    )
    rng = np.random.default_rng(920511)
    worlds = 4096
    zeros = np.zeros((worlds, len(SUPPORTED_ZONE_IDS)))
    fan_power = rng.uniform(0.0, 60000.0, worlds)
    hours = 600.0 / 3600.0
    served = (parameters.base_load_w + fan_power) * hours
    required = served / parameters.battery_discharge_efficiency
    tolerance = np.maximum(1e-12, 1e-10 * np.maximum(1.0, served))
    battery = np.maximum(0.0, required - rng.uniform(0.0, 1.0, worlds) * tolerance)
    state = _arrays(initial_state(load_scenario()))
    batch_state = concatenate_worlds((state,) * worlds)
    batch_state = replace_fields(batch_state, battery_energy_wh=battery)
    control = AchievedControl(
        zone_airflow_m3_s=zeros.copy(),
        achieved_cooling_removed_w=zeros.copy(),
        effective_cooling_delivery_w=zeros.copy(),
        achieved_oxygen_injection_mol_s=zeros.copy(),
        effective_oxygen_delivery_mol_s=zeros.copy(),
        fan_electrical_power_w=fan_power,
        achieved_scrubber_duty=np.zeros(worlds),
        achieved_condenser_duty=np.zeros(worlds),
        scrubber_capture_ability=np.ones(worlds),
        condenser_removal_ability=np.ones(worlds),
    )
    from aeolus.habitat_v2.batched_plant_contract import Disturbance

    disturbance = Disturbance(
        co2_generation_mol_s=zeros.copy(),
        o2_consumption_mol_s=zeros.copy(),
        water_vapor_generation_mol_s=zeros.copy(),
        sensible_heat_w=zeros.copy(),
        generation_w=np.zeros(worlds),
    )
    output = step_arrays(
        parameters, batch_state, control, disturbance, dt_seconds=600.0
    )
    in_cap = served <= SUPPORTED_SERVED_LOAD_WH_PER_STEP
    assert int(np.sum(in_cap)) > 1000 and int(np.sum(~in_cap)) > 1000
    codes = output.diagnostics.infeasible_code
    assert np.all(codes[in_cap] == FEASIBLE)
    assert np.all(codes[~in_cap] == OUTSIDE_SUPPORTED_DOMAIN)
    assert float(np.min(output.state.battery_energy_wh[in_cap])) < 0.0
    assert np.all(output.state.battery_energy_wh >= -BATTERY_STATE_SLACK_WH)
