"""Supported-domain and interface contract of the batched research plant step.

These tests exercise the fail-closed boundary of issue 92: which scenarios,
arrays, achieved controls, disturbances and timesteps the research interface
accepts, and which it refuses.  They make no numerical-parity claim; that is
covered by the sibling parity tests against the scalar plant.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from aeolus.habitat_v2.batched_plant_contract import (
    CONTROL_FIELDS,
    DISTURBANCE_FIELDS,
    EXCLUDED_MECHANISMS,
    FEASIBLE,
    INFEASIBLE_CODE_NAMES,
    INFEASIBLE_ELECTRICAL,
    INFEASIBLE_OXYGEN_LOAD,
    INFEASIBLE_OXYGEN_STORE,
    OUTSIDE_SUPPORTED_DOMAIN,
    OXYGEN_CAP_RELATIVE_SLACK,
    REFERENCE_ROLLOUT_MAX_STEPS,
    SPECIES_FIELDS,
    STATE_FIELDS,
    SUPPORTED_DT_SECONDS,
    SUPPORTED_SCENARIO_SCHEMA_VERSION,
    SUPPORTED_TEMPERATURE_K,
    SUPPORTED_ZONE_IDS,
    ZONE_COUNT,
    BATTERY_STATE_SLACK_WH,
    AchievedControl,
    BatchedPlantDomainError,
    Disturbance,
    PlantParameters,
    PlantStateArrays,
    Tolerance,
    reference_tolerances,
    require_supported_dt,
    require_supported_scenario,
    validate_step_inputs,
)
from aeolus.habitat_v2.batched_plant_reference import rollout_arrays, step_arrays
from aeolus.habitat_v2.physics import initial_state
from aeolus.habitat_v2.scenario import Scenario, ScenarioValidationError

from ._batched_plant_helpers import (
    DEVELOPMENT_SCENARIO,
    REPO_ROOT,
    SUPPORTED_SCENARIO_FILES,
    UNSUPPORTED_SCENARIO_FILES,
    load_scenario,
    random_batch,
    replace_fields,
    scenario_mapping,
)

WORLDS = 3
BATCH_SEED = 9201
DT_SECONDS = 60.0

# Written out here, independently of the module under test, so a silently
# dropped or renamed parameter is caught.
ZONE_PARAMETER_KEYS: tuple[str, ...] = (
    "volume_m3",
    "thermal_capacity_j_per_k",
    "passive_thermal_conductance_w_per_k",
    "sink_temperature_k",
)
EQUIPMENT_PARAMETER_KEYS: tuple[str, ...] = (
    "air_density_kg_m3",
    "air_specific_heat_j_kg_k",
    "scrubber_max_co2_mol_s",
    "condenser_max_water_mol_s",
    "oxygen_injection_max_total_mol_s",
    "cooling_max_thermal_w_per_zone",
    "base_load_w",
    "scrubber_power_w_full",
    "condenser_power_w_full",
    "cooling_coefficient_of_performance",
    "oxygen_injection_power_w_per_mol_s",
    "battery_capacity_wh",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "battery_max_charge_w",
    "battery_max_discharge_w",
)
STATE_ZONE_FIELDS: tuple[str, ...] = (
    "co2_mol",
    "o2_mol",
    "water_vapor_mol",
    "inert_mol",
    "temperature_k",
)
STATE_WORLD_FIELDS: tuple[str, ...] = (
    "co2_sorbent_remaining_mol",
    "captured_co2_mol",
    "condensed_water_mol",
    "oxygen_store_mol",
    "battery_energy_wh",
    "external_heat_rejected_j",
    "external_heat_received_j",
)
CONTROL_ZONE_FIELDS: tuple[str, ...] = (
    "zone_airflow_m3_s",
    "achieved_cooling_removed_w",
    "effective_cooling_delivery_w",
    "achieved_oxygen_injection_mol_s",
    "effective_oxygen_delivery_mol_s",
)
CONTROL_WORLD_FIELDS: tuple[str, ...] = (
    "fan_electrical_power_w",
    "achieved_scrubber_duty",
    "achieved_condenser_duty",
    "scrubber_capture_ability",
    "condenser_removal_ability",
)
DISTURBANCE_ZONE_FIELDS: tuple[str, ...] = (
    "co2_generation_mol_s",
    "o2_consumption_mol_s",
    "water_vapor_generation_mol_s",
    "sensible_heat_w",
)
DISTURBANCE_WORLD_FIELDS: tuple[str, ...] = ("generation_w",)

ALL_FIELDS: tuple[tuple[str, str], ...] = (
    tuple(("state", name) for name in STATE_FIELDS)
    + tuple(("control", name) for name in CONTROL_FIELDS)
    + tuple(("disturbance", name) for name in DISTURBANCE_FIELDS)
)
ZONE_FIELDS: tuple[tuple[str, str], ...] = (
    tuple(("state", name) for name in STATE_ZONE_FIELDS)
    + tuple(("control", name) for name in CONTROL_ZONE_FIELDS)
    + tuple(("disturbance", name) for name in DISTURBANCE_ZONE_FIELDS)
)
WORLD_FIELDS: tuple[tuple[str, str], ...] = (
    tuple(("state", name) for name in STATE_WORLD_FIELDS)
    + tuple(("control", name) for name in CONTROL_WORLD_FIELDS)
    + tuple(("disturbance", name) for name in DISTURBANCE_WORLD_FIELDS)
)


def _field_id(value: Any) -> str:
    return f"{value[0]}.{value[1]}" if isinstance(value, tuple) else str(value)


Batch = dict[str, Any]


@pytest.fixture(scope="module")
def parameters() -> PlantParameters:
    return PlantParameters.from_scenario(load_scenario())


@pytest.fixture(scope="module")
def batch(parameters: PlantParameters) -> Batch:
    state, control, disturbance = random_batch(parameters, WORLDS, BATCH_SEED)
    return {"state": state, "control": control, "disturbance": disturbance}


def _validate(parameters: PlantParameters, batch: Batch, **replaced: Any) -> float:
    merged = {**batch, **replaced}
    return validate_step_inputs(
        parameters,
        merged["state"],
        merged["control"],
        merged["disturbance"],
        DT_SECONDS,
    )


def _with_element(array: np.ndarray, value: float, index: int = 0) -> np.ndarray:
    changed = array.copy()
    changed.flat[index] = value
    return changed


# --------------------------------------------------------------------------
# Supported and unsupported scenarios
# --------------------------------------------------------------------------


def test_helper_field_layout_matches_the_declared_interface_field_order() -> None:
    assert STATE_FIELDS == STATE_ZONE_FIELDS + STATE_WORLD_FIELDS
    assert CONTROL_FIELDS == CONTROL_ZONE_FIELDS + CONTROL_WORLD_FIELDS
    assert DISTURBANCE_FIELDS == DISTURBANCE_ZONE_FIELDS + DISTURBANCE_WORLD_FIELDS
    assert SPECIES_FIELDS == STATE_ZONE_FIELDS[:4]
    for kind, names in (
        (PlantStateArrays, STATE_FIELDS),
        (AchievedControl, CONTROL_FIELDS),
        (Disturbance, DISTURBANCE_FIELDS),
    ):
        assert tuple(item.name for item in dataclasses.fields(kind)) == names
    assert tuple(item.name for item in dataclasses.fields(PlantParameters)) == (
        ("zone_ids",) + ZONE_PARAMETER_KEYS + EQUIPMENT_PARAMETER_KEYS
    )


def test_supported_zone_ids_equal_sorted_zone_ids_of_the_development_scenario() -> None:
    mapping = scenario_mapping(DEVELOPMENT_SCENARIO)
    expected = tuple(sorted(str(zone["id"]) for zone in mapping["zones"]))
    assert SUPPORTED_ZONE_IDS == expected
    assert ZONE_COUNT == len(expected) == 8
    assert len(set(SUPPORTED_ZONE_IDS)) == ZONE_COUNT
    assert mapping["schema_version"] == SUPPORTED_SCENARIO_SCHEMA_VERSION


@pytest.mark.parametrize("name", SUPPORTED_SCENARIO_FILES)
def test_every_supported_scenario_is_accepted_with_parameters_equal_to_its_json_in_sorted_zone_order(
    name: str,
) -> None:
    mapping = scenario_mapping(name)
    scenario = Scenario.from_mapping(copy.deepcopy(mapping))

    assert require_supported_scenario(scenario) is None
    parameters = PlantParameters.from_scenario(scenario)

    zones = {str(zone["id"]): zone for zone in mapping["zones"]}
    sorted_ids = tuple(sorted(zones))
    assert parameters.zone_ids == sorted_ids == SUPPORTED_ZONE_IDS
    for key in ZONE_PARAMETER_KEYS:
        array = getattr(parameters, key)
        assert type(array) is np.ndarray
        assert array.dtype == np.float64
        assert array.shape == (ZONE_COUNT,)
        expected = [float(zones[zone_id][key]) for zone_id in sorted_ids]
        assert array.tolist() == expected, key
    for key in EQUIPMENT_PARAMETER_KEYS:
        value = getattr(parameters, key)
        assert type(value) is float
        assert value == float(mapping["equipment"][key]), key
    low, high = SUPPORTED_DT_SECONDS
    assert low <= float(mapping["dt_seconds"]) <= high


@pytest.mark.parametrize("name", UNSUPPORTED_SCENARIO_FILES)
def test_every_unsupported_checked_in_scenario_is_rejected_with_a_domain_error(
    name: str,
) -> None:
    scenario = load_scenario(name)
    assert scenario.scenario_schema_version != SUPPORTED_SCENARIO_SCHEMA_VERSION
    with pytest.raises(BatchedPlantDomainError):
        require_supported_scenario(scenario)
    with pytest.raises(BatchedPlantDomainError):
        PlantParameters.from_scenario(scenario)


def _renamed_zone(mapping: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(mapping)
    assert "laboratory_annex" not in text
    return json.loads(text.replace("laboratory", "laboratory_annex"))


def _mentions_zone(value: Any, zone_id: str) -> bool:
    return isinstance(value, str) and (
        value == zone_id or value.startswith(zone_id + "_")
    )


def _without_zone(value: Any, zone_id: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_zone(item, zone_id)
            for key, item in value.items()
            if not _mentions_zone(key, zone_id)
        }
    if isinstance(value, list):
        return [
            _without_zone(item, zone_id)
            for item in value
            if not _mentions_zone(item, zone_id)
            and not (
                isinstance(item, dict)
                and any(_mentions_zone(entry, zone_id) for entry in item.values())
            )
        ]
    return value


def _removed_zone(mapping: dict[str, Any]) -> dict[str, Any]:
    return _without_zone(mapping, "laboratory")


def _with_dt(dt_seconds: float) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(mapping: dict[str, Any]) -> dict[str, Any]:
        changed = copy.deepcopy(mapping)
        changed["dt_seconds"] = dt_seconds
        return changed

    return mutate


V5_MUTATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "renamed_zone": _renamed_zone,
    "removed_zone": _removed_zone,
    "dt_below_supported_range": _with_dt(0.5),
    "dt_above_supported_range": _with_dt(900.0),
}
# Mutations the production parser accepts today, so the research contract is
# the boundary that has to refuse them.
V5_MUTATIONS_REACHING_THE_CONTRACT: tuple[str, ...] = tuple(V5_MUTATIONS)


def _parse_or_none(mapping: dict[str, Any]) -> Scenario | None:
    try:
        return Scenario.from_mapping(mapping)
    except ScenarioValidationError:
        return None


@pytest.mark.parametrize("mutation", sorted(V5_MUTATIONS))
def test_v5_mapping_outside_the_fixed_topology_or_timestep_is_refused_before_the_step(
    mutation: str,
) -> None:
    mapping = V5_MUTATIONS[mutation](scenario_mapping(DEVELOPMENT_SCENARIO))
    assert mapping["schema_version"] == SUPPORTED_SCENARIO_SCHEMA_VERSION
    scenario = _parse_or_none(mapping)
    if scenario is None:
        # The closed-schema production parser already fails closed.
        assert mutation not in V5_MUTATIONS_REACHING_THE_CONTRACT
        return
    assert mutation in V5_MUTATIONS_REACHING_THE_CONTRACT
    with pytest.raises(BatchedPlantDomainError):
        require_supported_scenario(scenario)
    with pytest.raises(BatchedPlantDomainError):
        PlantParameters.from_scenario(scenario)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "scenarios/habitat_v2_forecast_development.json",
        42,
        SimpleNamespace(
            scenario_schema_version=SUPPORTED_SCENARIO_SCHEMA_VERSION, data={}
        ),
    ],
    ids=["none", "path_string", "int", "duck_typed_namespace"],
)
def test_non_scenario_objects_are_rejected_as_scenarios(value: Any) -> None:
    with pytest.raises(BatchedPlantDomainError):
        require_supported_scenario(value)
    with pytest.raises(BatchedPlantDomainError):
        PlantParameters.from_scenario(value)


def test_raw_scenario_mapping_is_rejected_until_it_is_a_validated_scenario() -> None:
    mapping = scenario_mapping(DEVELOPMENT_SCENARIO)
    with pytest.raises(BatchedPlantDomainError):
        require_supported_scenario(mapping)  # type: ignore[arg-type]
    with pytest.raises(BatchedPlantDomainError):
        PlantParameters.from_scenario(mapping)  # type: ignore[arg-type]
    require_supported_scenario(Scenario.from_mapping(mapping))


def test_plant_parameters_reject_zone_ids_in_any_order_other_than_sorted(
    parameters: PlantParameters,
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        dataclasses.replace(parameters, zone_ids=tuple(reversed(SUPPORTED_ZONE_IDS)))
    with pytest.raises(BatchedPlantDomainError):
        dataclasses.replace(parameters, zone_ids=SUPPORTED_ZONE_IDS[:-1])
    with pytest.raises(BatchedPlantDomainError):
        dataclasses.replace(
            parameters, zone_ids=SUPPORTED_ZONE_IDS[:-1] + ("laboratory_annex",)
        )


@pytest.mark.parametrize("key", ZONE_PARAMETER_KEYS)
@pytest.mark.parametrize(
    "case", ["float32", "list", "wrong_zone_count", "extra_axis", "nan", "inf"]
)
def test_plant_parameters_reject_malformed_zone_parameter_arrays(
    parameters: PlantParameters, key: str, case: str
) -> None:
    array = getattr(parameters, key)
    bad: Any = {
        "float32": array.astype(np.float32),
        "list": array.tolist(),
        "wrong_zone_count": array[:-1].copy(),
        "extra_axis": array[None, :].copy(),
        "nan": _with_element(array, np.nan),
        "inf": _with_element(array, np.inf),
    }[case]
    with pytest.raises(BatchedPlantDomainError):
        dataclasses.replace(parameters, **{key: bad})


@pytest.mark.parametrize("key", EQUIPMENT_PARAMETER_KEYS)
@pytest.mark.parametrize(
    "bad", [-1.0, float("nan"), float("inf"), True, "1.0", None], ids=repr
)
def test_plant_parameters_reject_malformed_equipment_scalars(
    parameters: PlantParameters, key: str, bad: Any
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        dataclasses.replace(parameters, **{key: bad})


# --------------------------------------------------------------------------
# State stacking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUPPORTED_SCENARIO_FILES)
def test_plant_state_arrays_from_plant_states_round_trips_initial_state_values(
    name: str,
) -> None:
    plant_state = initial_state(load_scenario(name))
    arrays = PlantStateArrays.from_plant_states((plant_state,))

    assert arrays.batch_shape == (1,)
    for field in STATE_ZONE_FIELDS:
        array = getattr(arrays, field)
        assert array.dtype == np.float64
        assert array.shape == (1, ZONE_COUNT)
        expected = [
            float(getattr(plant_state.zones[zone_id], field))
            for zone_id in sorted(plant_state.zones)
        ]
        assert array[0].tolist() == expected, field
    for field in STATE_WORLD_FIELDS:
        array = getattr(arrays, field)
        assert array.dtype == np.float64
        assert array.shape == (1,)
        assert array[0] == float(getattr(plant_state.utility, field)), field


def test_plant_state_arrays_from_plant_states_stacks_worlds_on_a_leading_axis() -> None:
    first = initial_state(load_scenario(SUPPORTED_SCENARIO_FILES[0]))
    second = initial_state(load_scenario(SUPPORTED_SCENARIO_FILES[1]))
    stacked = PlantStateArrays.from_plant_states((first, second))
    assert stacked.batch_shape == (2,)
    for index, plant_state in enumerate((first, second)):
        single = PlantStateArrays.from_plant_states((plant_state,))
        for field in STATE_FIELDS:
            np.testing.assert_array_equal(
                getattr(stacked, field)[index], getattr(single, field)[0]
            )


def test_plant_state_arrays_from_plant_states_rejects_empty_and_foreign_inputs() -> None:
    plant_state = initial_state(load_scenario())
    with pytest.raises(BatchedPlantDomainError):
        PlantStateArrays.from_plant_states(())
    with pytest.raises(BatchedPlantDomainError):
        PlantStateArrays.from_plant_states((plant_state, None))  # type: ignore[arg-type]
    with pytest.raises(BatchedPlantDomainError):
        PlantStateArrays.from_plant_states(({"zones": {}},))  # type: ignore[arg-type]
    two_zone_state = initial_state(load_scenario(UNSUPPORTED_SCENARIO_FILES[0]))
    with pytest.raises(BatchedPlantDomainError):
        PlantStateArrays.from_plant_states((two_zone_state,))


# --------------------------------------------------------------------------
# Array layout, dtype, type and finiteness for every field
# --------------------------------------------------------------------------


def test_the_shared_synthetic_batch_is_inside_the_declared_domain(
    parameters: PlantParameters, batch: Batch
) -> None:
    assert _validate(parameters, batch) == DT_SECONDS
    for owner, name in ZONE_FIELDS:
        assert getattr(batch[owner], name).shape == (WORLDS, ZONE_COUNT)
    for owner, name in WORLD_FIELDS:
        assert getattr(batch[owner], name).shape == (WORLDS,)
    for value in batch.values():
        assert value.batch_shape == (WORLDS,)


ZONE_SHAPE_CASES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "one_zone_too_few": lambda a: a[:, :-1].copy(),
    "one_zone_too_many": lambda a: np.concatenate([a, a[:, :1]], axis=-1),
    "missing_zone_axis": lambda a: a[:, 0].copy(),
    "extra_trailing_axis": lambda a: a[:, :, None].copy(),
    "zone_axis_first": lambda a: a.T.copy(),
    "one_world_too_few": lambda a: a[:-1].copy(),
}
WORLD_SHAPE_CASES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "extra_trailing_axis": lambda a: a[:, None].copy(),
    "broadcast_over_zone_axis": lambda a: np.repeat(a[:, None], ZONE_COUNT, axis=1),
    "collapsed_to_zero_dimensions": lambda a: np.array(a[0], dtype=np.float64),
    "one_world_too_few": lambda a: a[:-1].copy(),
}


@pytest.mark.parametrize("field", ZONE_FIELDS, ids=_field_id)
@pytest.mark.parametrize("case", sorted(ZONE_SHAPE_CASES))
def test_per_zone_field_with_a_malformed_shape_is_rejected(
    batch: Batch, field: tuple[str, str], case: str
) -> None:
    owner, name = field
    bad = ZONE_SHAPE_CASES[case](getattr(batch[owner], name))
    assert bad.dtype == np.float64
    with pytest.raises(BatchedPlantDomainError):
        replace_fields(batch[owner], **{name: bad})


@pytest.mark.parametrize("field", WORLD_FIELDS, ids=_field_id)
@pytest.mark.parametrize("case", sorted(WORLD_SHAPE_CASES))
def test_per_world_field_with_a_malformed_shape_is_rejected(
    batch: Batch, field: tuple[str, str], case: str
) -> None:
    owner, name = field
    bad = WORLD_SHAPE_CASES[case](getattr(batch[owner], name))
    assert bad.dtype == np.float64
    with pytest.raises(BatchedPlantDomainError):
        replace_fields(batch[owner], **{name: bad})


DTYPE_CASES: dict[str, Callable[[np.ndarray], Any]] = {
    "float32": lambda a: a.astype(np.float32),
    "int64": lambda a: a.astype(np.int64),
    "bool": lambda a: a.astype(np.bool_),
    "object": lambda a: a.astype(object),
}


@pytest.mark.parametrize("field", ALL_FIELDS, ids=_field_id)
@pytest.mark.parametrize("case", sorted(DTYPE_CASES))
def test_every_field_with_a_dtype_other_than_float64_is_rejected(
    batch: Batch, field: tuple[str, str], case: str
) -> None:
    owner, name = field
    original = getattr(batch[owner], name)
    bad = DTYPE_CASES[case](original)
    assert bad.shape == original.shape
    with pytest.raises(BatchedPlantDomainError):
        replace_fields(batch[owner], **{name: bad})


NON_NDARRAY_CASES: dict[str, Callable[[np.ndarray], Any]] = {
    "nested_list": lambda a: a.tolist(),
    "tuple": lambda a: tuple(a.tolist()),
    "python_scalar": lambda a: float(a.flat[0]),
    "numpy_scalar": lambda a: np.float64(a.flat[0]),
    "none": lambda a: None,
    "ndarray_subclass": lambda a: np.ma.MaskedArray(a.copy()),
}


@pytest.mark.parametrize("field", ALL_FIELDS, ids=_field_id)
@pytest.mark.parametrize("case", sorted(NON_NDARRAY_CASES))
def test_every_field_that_is_not_a_plain_numpy_ndarray_is_rejected(
    batch: Batch, field: tuple[str, str], case: str
) -> None:
    owner, name = field
    bad = NON_NDARRAY_CASES[case](getattr(batch[owner], name))
    assert type(bad) is not np.ndarray
    with pytest.raises(BatchedPlantDomainError):
        replace_fields(batch[owner], **{name: bad})


@pytest.mark.parametrize("field", ALL_FIELDS, ids=_field_id)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "+inf", "-inf"]
)
@pytest.mark.parametrize("position", ["first", "last"])
def test_every_field_containing_a_nonfinite_value_is_rejected(
    batch: Batch, field: tuple[str, str], value: float, position: str
) -> None:
    owner, name = field
    index = 0 if position == "first" else -1
    bad = _with_element(getattr(batch[owner], name), value, index)
    with pytest.raises(BatchedPlantDomainError):
        replace_fields(batch[owner], **{name: bad})


@pytest.mark.parametrize("odd_one_out", ["state", "control", "disturbance"])
def test_mismatched_batch_shapes_between_state_control_and_disturbance_are_rejected(
    parameters: PlantParameters, batch: Batch, odd_one_out: str
) -> None:
    state, control, disturbance = random_batch(parameters, WORLDS - 1, BATCH_SEED)
    smaller = {"state": state, "control": control, "disturbance": disturbance}
    mixed = {**batch, odd_one_out: smaller[odd_one_out]}
    assert _validate(parameters, smaller) == DT_SECONDS
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, mixed)
    with pytest.raises(BatchedPlantDomainError):
        step_arrays(
            parameters,
            mixed["state"],
            mixed["control"],
            mixed["disturbance"],
            dt_seconds=DT_SECONDS,
        )


def test_a_batch_shape_that_only_broadcasts_is_rejected_rather_than_broadcast(
    parameters: PlantParameters, batch: Batch
) -> None:
    state, control, disturbance = random_batch(parameters, 1, BATCH_SEED)
    assert state.batch_shape == (1,)
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=control)
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, disturbance=disturbance)
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, state=state)


@pytest.mark.parametrize("slot", ["parameters", "state", "disturbance"])
@pytest.mark.parametrize("impostor", ["none", "dict", "namespace"])
def test_step_inputs_of_the_wrong_interface_type_are_rejected(
    parameters: PlantParameters, batch: Batch, slot: str, impostor: str
) -> None:
    arguments: dict[str, Any] = {"parameters": parameters, **batch}
    genuine = arguments[slot]
    as_dict = {
        item.name: getattr(genuine, item.name) for item in dataclasses.fields(genuine)
    }
    arguments[slot] = {
        "none": None,
        "dict": as_dict,
        "namespace": SimpleNamespace(**as_dict),
    }[impostor]
    with pytest.raises(BatchedPlantDomainError):
        validate_step_inputs(
            arguments["parameters"],
            arguments["state"],
            arguments["control"],
            arguments["disturbance"],
            DT_SECONDS,
        )
    with pytest.raises(BatchedPlantDomainError):
        step_arrays(
            arguments["parameters"],
            arguments["state"],
            arguments["control"],
            arguments["disturbance"],
            dt_seconds=DT_SECONDS,
        )


# --------------------------------------------------------------------------
# Value domain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SPECIES_FIELDS + STATE_WORLD_FIELDS)
@pytest.mark.parametrize("value", [-1.0, -1e-12], ids=["minus_one", "minus_1e-12"])
def test_negative_moles_and_stores_in_the_state_are_rejected(
    parameters: PlantParameters, batch: Batch, name: str, value: float
) -> None:
    state = batch["state"]
    if name == "battery_energy_wh":
        # The battery carries the declared slack the scalar plant can produce.
        value = min(value, -2.0 * BATTERY_STATE_SLACK_WH)
    bad = replace_fields(state, **{name: _with_element(getattr(state, name), value, -1)})
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, state=bad)


@pytest.mark.parametrize("name", SPECIES_FIELDS + STATE_WORLD_FIELDS)
def test_zero_moles_and_stores_in_the_state_are_accepted(
    parameters: PlantParameters, batch: Batch, name: str
) -> None:
    state = batch["state"]
    zeroed = replace_fields(state, **{name: np.zeros_like(getattr(state, name))})
    assert _validate(parameters, batch, state=zeroed) == DT_SECONDS


@pytest.mark.parametrize(
    "temperature_k",
    [
        float(np.nextafter(SUPPORTED_TEMPERATURE_K[0], -np.inf)),
        SUPPORTED_TEMPERATURE_K[0] - 1.0,
        0.0,
        -10.0,
        float(np.nextafter(SUPPORTED_TEMPERATURE_K[1], np.inf)),
        SUPPORTED_TEMPERATURE_K[1] + 1.0,
        1.0e4,
    ],
    ids=repr,
)
def test_temperature_outside_the_supported_band_is_rejected(
    parameters: PlantParameters, batch: Batch, temperature_k: float
) -> None:
    state = batch["state"]
    bad = replace_fields(
        state, temperature_k=_with_element(state.temperature_k, temperature_k, 5)
    )
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, state=bad)


@pytest.mark.parametrize("bound", SUPPORTED_TEMPERATURE_K, ids=repr)
def test_temperature_exactly_at_a_supported_bound_is_accepted(
    parameters: PlantParameters, batch: Batch, bound: float
) -> None:
    state = batch["state"]
    edge = replace_fields(
        state, temperature_k=np.full_like(state.temperature_k, bound)
    )
    assert _validate(parameters, batch, state=edge) == DT_SECONDS


def test_battery_energy_above_capacity_is_rejected_and_exactly_at_capacity_is_accepted(
    parameters: PlantParameters, batch: Batch
) -> None:
    state = batch["state"]
    capacity = parameters.battery_capacity_wh
    full = replace_fields(
        state, battery_energy_wh=np.full_like(state.battery_energy_wh, capacity)
    )
    assert _validate(parameters, batch, state=full) == DT_SECONDS
    inside_slack = (
        float(np.nextafter(capacity, np.inf)),
        capacity + BATTERY_STATE_SLACK_WH,
        -BATTERY_STATE_SLACK_WH,
    )
    for value in inside_slack:
        edge = replace_fields(
            state,
            battery_energy_wh=_with_element(state.battery_energy_wh, value, 1),
        )
        assert _validate(parameters, batch, state=edge) == DT_SECONDS
    outside_slack = (
        float(np.nextafter(capacity + BATTERY_STATE_SLACK_WH, np.inf)),
        float(np.nextafter(-BATTERY_STATE_SLACK_WH, -np.inf)),
        capacity * 2.0,
    )
    for excess in outside_slack:
        over = replace_fields(
            state,
            battery_energy_wh=_with_element(state.battery_energy_wh, excess, 1),
        )
        with pytest.raises(BatchedPlantDomainError):
            _validate(parameters, batch, state=over)


UNIT_INTERVAL_CONTROL_FIELDS: tuple[str, ...] = (
    "achieved_scrubber_duty",
    "achieved_condenser_duty",
    "scrubber_capture_ability",
    "condenser_removal_ability",
)


@pytest.mark.parametrize("name", UNIT_INTERVAL_CONTROL_FIELDS)
def test_duty_or_ability_above_one_is_rejected_and_exactly_one_is_accepted(
    parameters: PlantParameters, batch: Batch, name: str
) -> None:
    control = batch["control"]
    saturated = replace_fields(control, **{name: np.ones_like(getattr(control, name))})
    assert _validate(parameters, batch, control=saturated) == DT_SECONDS
    for excess in (float(np.nextafter(1.0, 2.0)), 1.5):
        over = replace_fields(
            control, **{name: _with_element(getattr(control, name), excess, -1)}
        )
        with pytest.raises(BatchedPlantDomainError):
            _validate(parameters, batch, control=over)


@pytest.mark.parametrize("name", CONTROL_FIELDS)
@pytest.mark.parametrize("value", [-1.0, -1e-12], ids=["minus_one", "minus_1e-12"])
def test_negative_value_in_any_control_field_is_rejected(
    parameters: PlantParameters, batch: Batch, name: str, value: float
) -> None:
    control = batch["control"]
    bad = replace_fields(
        control, **{name: _with_element(getattr(control, name), value, -1)}
    )
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=bad)


@pytest.mark.parametrize("name", DISTURBANCE_FIELDS)
@pytest.mark.parametrize("value", [-1.0, -1e-12], ids=["minus_one", "minus_1e-12"])
def test_negative_value_in_any_disturbance_field_is_rejected(
    parameters: PlantParameters, batch: Batch, name: str, value: float
) -> None:
    disturbance = batch["disturbance"]
    bad = replace_fields(
        disturbance, **{name: _with_element(getattr(disturbance, name), value, -1)}
    )
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, disturbance=bad)


def test_all_zero_control_and_disturbance_are_accepted(
    parameters: PlantParameters, batch: Batch
) -> None:
    control = replace_fields(
        batch["control"],
        **{
            name: np.zeros_like(getattr(batch["control"], name))
            for name in CONTROL_FIELDS
        },
    )
    disturbance = replace_fields(
        batch["disturbance"],
        **{
            name: np.zeros_like(getattr(batch["disturbance"], name))
            for name in DISTURBANCE_FIELDS
        },
    )
    assert (
        _validate(parameters, batch, control=control, disturbance=disturbance)
        == DT_SECONDS
    )


def test_achieved_cooling_above_the_per_zone_maximum_is_rejected_and_at_it_accepted(
    parameters: PlantParameters, batch: Batch
) -> None:
    control = batch["control"]
    maximum = parameters.cooling_max_thermal_w_per_zone
    at_maximum = replace_fields(
        control,
        achieved_cooling_removed_w=np.full_like(
            control.achieved_cooling_removed_w, maximum
        ),
    )
    assert _validate(parameters, batch, control=at_maximum) == DT_SECONDS
    for excess in (float(np.nextafter(maximum, np.inf)), maximum * 2.0):
        over = replace_fields(
            control,
            achieved_cooling_removed_w=_with_element(
                control.achieved_cooling_removed_w, excess, -1
            ),
        )
        with pytest.raises(BatchedPlantDomainError):
            _validate(parameters, batch, control=over)


def test_effective_cooling_above_achieved_cooling_is_rejected_and_equal_is_accepted(
    parameters: PlantParameters, batch: Batch
) -> None:
    control = batch["control"]
    equal = replace_fields(
        control, effective_cooling_delivery_w=control.achieved_cooling_removed_w.copy()
    )
    assert _validate(parameters, batch, control=equal) == DT_SECONDS
    achieved = float(control.achieved_cooling_removed_w.flat[-1])
    over = replace_fields(
        control,
        effective_cooling_delivery_w=_with_element(
            equal.effective_cooling_delivery_w,
            float(np.nextafter(achieved, np.inf)),
            -1,
        ),
    )
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=over)


def test_effective_oxygen_above_achieved_oxygen_is_rejected_and_equal_is_accepted(
    parameters: PlantParameters, batch: Batch
) -> None:
    control = batch["control"]
    equal = replace_fields(
        control,
        effective_oxygen_delivery_mol_s=control.achieved_oxygen_injection_mol_s.copy(),
    )
    assert _validate(parameters, batch, control=equal) == DT_SECONDS
    achieved = float(control.achieved_oxygen_injection_mol_s.flat[-1])
    over = replace_fields(
        control,
        effective_oxygen_delivery_mol_s=_with_element(
            equal.effective_oxygen_delivery_mol_s,
            float(np.nextafter(achieved, np.inf)),
            -1,
        ),
    )
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=over)
    no_injection = np.zeros_like(control.achieved_oxygen_injection_mol_s)
    delivered_without_injection = replace_fields(
        control,
        achieved_oxygen_injection_mol_s=no_injection,
        effective_oxygen_delivery_mol_s=_with_element(no_injection, 1e-9, 0),
    )
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=delivered_without_injection)


def _oxygen_control(control: AchievedControl, achieved: np.ndarray) -> AchievedControl:
    return replace_fields(
        control,
        achieved_oxygen_injection_mol_s=achieved,
        effective_oxygen_delivery_mol_s=np.zeros_like(achieved),
    )


@pytest.mark.parametrize("world", range(WORLDS))
def test_total_achieved_oxygen_above_the_cap_is_rejected_for_any_world(
    parameters: PlantParameters, batch: Batch, world: int
) -> None:
    control = batch["control"]
    cap = parameters.oxygen_injection_max_total_mol_s
    assert cap > 0.0
    above = cap * (1.0 + 1000.0 * OXYGEN_CAP_RELATIVE_SLACK)
    assert above > cap

    in_one_zone = np.zeros_like(control.achieved_oxygen_injection_mol_s)
    in_one_zone[world, -1] = above
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=_oxygen_control(control, in_one_zone))

    spread = np.zeros_like(control.achieved_oxygen_injection_mol_s)
    spread[world, :] = 2.0 * cap / ZONE_COUNT
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, batch, control=_oxygen_control(control, spread))


def test_total_achieved_oxygen_exactly_at_the_cap_is_accepted(
    parameters: PlantParameters, batch: Batch
) -> None:
    control = batch["control"]
    cap = parameters.oxygen_injection_max_total_mol_s
    in_one_zone = np.zeros_like(control.achieved_oxygen_injection_mol_s)
    in_one_zone[:, 0] = cap
    assert (
        _validate(parameters, batch, control=_oxygen_control(control, in_one_zone))
        == DT_SECONDS
    )
    two_exact_halves = np.zeros_like(control.achieved_oxygen_injection_mol_s)
    two_exact_halves[:, 2] = cap / 2.0
    two_exact_halves[:, 5] = cap / 2.0
    assert (
        _validate(parameters, batch, control=_oxygen_control(control, two_exact_halves))
        == DT_SECONDS
    )


BAD_DT_VALUES: list[Any] = [
    True,
    False,
    "60",
    "60.0",
    None,
    float("nan"),
    float("inf"),
    float("-inf"),
    0,
    0.0,
    -60.0,
    -1,
    float(np.nextafter(SUPPORTED_DT_SECONDS[0], -np.inf)),
    0.5,
    float(np.nextafter(SUPPORTED_DT_SECONDS[1], np.inf)),
    601,
    3600.0,
    60 + 0j,
    [60.0],
    (60.0,),
    np.array(60.0),
    np.array([60.0]),
]


@pytest.mark.parametrize("dt_seconds", BAD_DT_VALUES, ids=repr)
def test_timestep_of_the_wrong_type_or_outside_the_supported_range_is_rejected(
    parameters: PlantParameters, batch: Batch, dt_seconds: Any
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        require_supported_dt(dt_seconds)
    with pytest.raises(BatchedPlantDomainError):
        validate_step_inputs(
            parameters,
            batch["state"],
            batch["control"],
            batch["disturbance"],
            dt_seconds,
        )
    with pytest.raises(BatchedPlantDomainError):
        step_arrays(
            parameters,
            batch["state"],
            batch["control"],
            batch["disturbance"],
            dt_seconds=dt_seconds,
        )
    with pytest.raises(BatchedPlantDomainError):
        rollout_arrays(
            parameters,
            batch["state"],
            (batch["control"],),
            (batch["disturbance"],),
            dt_seconds=dt_seconds,
        )


@pytest.mark.parametrize(
    "dt_seconds",
    [
        SUPPORTED_DT_SECONDS[0],
        SUPPORTED_DT_SECONDS[1],
        int(SUPPORTED_DT_SECONDS[0]),
        int(SUPPORTED_DT_SECONDS[1]),
        60,
        60.0,
        float(np.nextafter(SUPPORTED_DT_SECONDS[0], np.inf)),
        float(np.nextafter(SUPPORTED_DT_SECONDS[1], -np.inf)),
    ],
    ids=repr,
)
def test_timestep_at_and_inside_the_supported_bounds_is_accepted_as_a_float(
    parameters: PlantParameters, batch: Batch, dt_seconds: Any
) -> None:
    accepted = require_supported_dt(dt_seconds)
    assert type(accepted) is float
    assert accepted == float(dt_seconds)
    validated = validate_step_inputs(
        parameters, batch["state"], batch["control"], batch["disturbance"], dt_seconds
    )
    assert type(validated) is float
    assert validated == float(dt_seconds)


def test_supported_ranges_are_ordered_finite_and_positive() -> None:
    for low, high in (SUPPORTED_TEMPERATURE_K, SUPPORTED_DT_SECONDS):
        assert np.isfinite(low) and np.isfinite(high)
        assert 0.0 < low < high
    assert 0.0 < OXYGEN_CAP_RELATIVE_SLACK <= 1e-9
    assert REFERENCE_ROLLOUT_MAX_STEPS >= 1


# --------------------------------------------------------------------------
# Requested commands cannot be passed
# --------------------------------------------------------------------------


def _timeline_command() -> dict[str, Any]:
    return copy.deepcopy(scenario_mapping(DEVELOPMENT_SCENARIO)["timeline"][0]["command"])


def _control_impostors(control: AchievedControl) -> dict[str, Any]:
    as_dict = {name: getattr(control, name) for name in CONTROL_FIELDS}

    class AchievedControlSubclass(AchievedControl):
        pass

    return {
        "scenario_timeline_command_mapping": _timeline_command(),
        "dict_of_achieved_arrays": as_dict,
        "namespace_of_achieved_arrays": SimpleNamespace(**as_dict),
        "achieved_control_subclass": AchievedControlSubclass(**as_dict),
        "tuple_of_arrays": tuple(as_dict.values()),
        "none": None,
    }


IMPOSTOR_NAMES: tuple[str, ...] = (
    "scenario_timeline_command_mapping",
    "dict_of_achieved_arrays",
    "namespace_of_achieved_arrays",
    "achieved_control_subclass",
    "tuple_of_arrays",
    "none",
)


@pytest.mark.parametrize("impostor", IMPOSTOR_NAMES)
def test_requested_command_or_any_non_achieved_control_object_is_rejected_as_control(
    parameters: PlantParameters, batch: Batch, impostor: str
) -> None:
    impostors = _control_impostors(batch["control"])
    assert tuple(impostors) == IMPOSTOR_NAMES
    control = impostors[impostor]
    assert type(control) is not AchievedControl
    with pytest.raises(BatchedPlantDomainError):
        validate_step_inputs(
            parameters, batch["state"], control, batch["disturbance"], DT_SECONDS
        )
    with pytest.raises(BatchedPlantDomainError):
        step_arrays(
            parameters,
            batch["state"],
            control,
            batch["disturbance"],
            dt_seconds=DT_SECONDS,
        )
    with pytest.raises(BatchedPlantDomainError):
        rollout_arrays(
            parameters,
            batch["state"],
            (control,),
            (batch["disturbance"],),
            dt_seconds=DT_SECONDS,
        )


def test_the_timeline_command_mapping_used_as_an_impostor_is_a_requested_command() -> None:
    command = _timeline_command()
    assert isinstance(command, dict)
    assert "fan_speed_fraction" in command
    assert not set(command) & set(CONTROL_FIELDS)


def test_achieved_control_has_no_requested_or_command_field_and_cannot_be_given_one(
    batch: Batch,
) -> None:
    names = tuple(item.name for item in dataclasses.fields(AchievedControl))
    assert names == CONTROL_FIELDS
    for name in names:
        assert "requested" not in name, name
        assert "command" not in name, name
        assert "setpoint" not in name, name
    for name in set(CONTROL_FIELDS) - {"zone_airflow_m3_s", "fan_electrical_power_w"}:
        assert (
            name.startswith(("achieved_", "effective_")) or name.endswith("_ability")
        ), name
    assert "zone_airflow_m3_s" in names
    assert "fan_electrical_power_w" in names

    arrays = {name: getattr(batch["control"], name) for name in CONTROL_FIELDS}
    for extra in ("requested_fan_speed_fraction", "command", "fan_speed_fraction"):
        with pytest.raises(TypeError):
            AchievedControl(**arrays, **{extra: arrays["fan_electrical_power_w"]})
    for missing in CONTROL_FIELDS:
        partial = {name: value for name, value in arrays.items() if name != missing}
        with pytest.raises(TypeError):
            AchievedControl(**partial)


def test_plant_state_arrays_carry_no_actuator_feedback_or_seed_field() -> None:
    names = tuple(item.name for item in dataclasses.fields(PlantStateArrays))
    assert names == STATE_FIELDS
    for name in names:
        for forbidden in (
            "actual_",
            "last_operational_feedback",
            "feedback",
            "seed",
            "requested",
            "command",
            "effective_",
            "achieved_",
        ):
            assert forbidden not in name, name


def test_disturbance_carries_no_step_index_seed_or_command_field() -> None:
    names = tuple(item.name for item in dataclasses.fields(Disturbance))
    assert names == DISTURBANCE_FIELDS
    for name in names:
        for forbidden in ("step", "seed", "requested", "command", "fault"):
            assert forbidden not in name, name


# --------------------------------------------------------------------------
# Predeclared tolerances, exclusions and outcome codes
# --------------------------------------------------------------------------

PRECISIONS: tuple[str, ...] = ("float64", "float32")
HORIZONS: tuple[str, ...] = ("one_step", "rollout")


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_reference_tolerances_cover_exactly_the_state_fields(
    precision: str, horizon: str
) -> None:
    table = reference_tolerances(precision, horizon)
    assert set(table) == set(STATE_FIELDS)
    assert len(table) == len(STATE_FIELDS)
    for name, tolerance in table.items():
        assert type(tolerance) is Tolerance, name
        assert np.isfinite(tolerance.absolute) and np.isfinite(tolerance.relative)
        assert tolerance.absolute > 0.0, name
        assert tolerance.relative >= 0.0, name


@pytest.mark.parametrize(
    ("precision", "horizon"),
    [
        ("float16", "one_step"),
        ("double", "rollout"),
        ("FLOAT64", "one_step"),
        ("float64", "two_step"),
        ("float64", "ROLLOUT"),
        ("one_step", "float64"),
        ("", ""),
        (None, "one_step"),
        ("float64", None),
        (64, "one_step"),
    ],
)
def test_reference_tolerances_reject_unknown_precision_or_horizon(
    precision: Any, horizon: Any
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        reference_tolerances(precision, horizon)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", STATE_FIELDS)
def test_every_rollout_tolerance_is_at_least_its_one_step_tolerance(
    precision: str, name: str
) -> None:
    one_step = reference_tolerances(precision, "one_step")[name]
    rollout = reference_tolerances(precision, "rollout")[name]
    assert rollout.absolute >= one_step.absolute
    assert rollout.relative >= one_step.relative


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("name", STATE_FIELDS)
def test_float64_tolerances_are_no_looser_than_float32_tolerances(
    horizon: str, name: str
) -> None:
    double = reference_tolerances("float64", horizon)[name]
    single = reference_tolerances("float32", horizon)[name]
    assert double.absolute <= single.absolute
    assert double.relative <= single.relative


def test_tolerance_allowed_error_is_absolute_plus_relative_times_reference_magnitude() -> None:
    tolerance = Tolerance(absolute=1e-3, relative=1e-2)
    reference = np.array([-200.0, 0.0, 50.0], dtype=np.float64)
    np.testing.assert_array_equal(
        tolerance.allowed(reference),
        1e-3 + 1e-2 * np.abs(reference),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tolerance.absolute = 1.0  # type: ignore[misc]


def test_tolerance_tables_cannot_be_loosened_through_the_returned_mapping_objects() -> None:
    table = reference_tolerances("float64", "one_step")
    tolerance = table["temperature_k"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        tolerance.absolute = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        tolerance.relative = 1.0  # type: ignore[misc]
    assert reference_tolerances("float64", "one_step")["temperature_k"] == tolerance


@pytest.mark.parametrize(
    "mechanism",
    [
        "requested_command_slew",
        "oxygen_total_cap_projection",
        "fault_schedule_evaluation",
        "air_network_solve",
        "sensor_and_feedback_models",
        "hmc_arbitration_and_preflight",
    ],
)
def test_excluded_mechanisms_document_each_required_exclusion(mechanism: str) -> None:
    assert mechanism in EXCLUDED_MECHANISMS
    description = EXCLUDED_MECHANISMS[mechanism]
    assert isinstance(description, str)
    assert description.strip()


def test_every_excluded_mechanism_has_a_nonempty_text_reason() -> None:
    assert EXCLUDED_MECHANISMS
    for mechanism, description in EXCLUDED_MECHANISMS.items():
        assert isinstance(mechanism, str) and mechanism.strip()
        assert isinstance(description, str) and description.strip(), mechanism


def test_outcome_code_names_cover_codes_zero_to_four_with_distinct_names() -> None:
    assert (
        FEASIBLE,
        INFEASIBLE_OXYGEN_LOAD,
        INFEASIBLE_OXYGEN_STORE,
        INFEASIBLE_ELECTRICAL,
    ) == (0, 1, 2, 3)
    assert OUTSIDE_SUPPORTED_DOMAIN == 4
    assert set(INFEASIBLE_CODE_NAMES) == {0, 1, 2, 3, 4}
    names = [INFEASIBLE_CODE_NAMES[code] for code in range(5)]
    assert all(isinstance(name, str) and name for name in names)
    assert len(set(names)) == 5
    assert INFEASIBLE_CODE_NAMES[FEASIBLE] == "feasible"


# --------------------------------------------------------------------------
# Optional-dependency boundary
# --------------------------------------------------------------------------

_SOURCE_DIR = REPO_ROOT / "src"

_TORCH_PROBE = """
import json
import sys


class TorchImportRecorder:
    attempts = []

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            self.attempts.append(name)
        return None


sys.meta_path.insert(0, TorchImportRecorder())
before = "torch" in sys.modules
import aeolus.habitat_v2.batched_plant_contract as contract
import aeolus.habitat_v2.batched_plant_reference as reference

print(
    json.dumps(
        {
            "torch_loaded_before": before,
            "torch_modules": sorted(
                name
                for name in sys.modules
                if name == "torch" or name.startswith("torch.")
            ),
            "torch_import_attempts": TorchImportRecorder.attempts,
            "contract_file": contract.__file__,
            "reference_file": reference.__file__,
        }
    )
)
"""

_PACKAGE_PROBE = """
import json
import sys

import aeolus.habitat_v2

print(
    json.dumps(
        {
            "batched_plant_modules": sorted(
                name for name in sys.modules if "batched_plant" in name
            ),
            "torch_modules": sorted(
                name
                for name in sys.modules
                if name == "torch" or name.startswith("torch.")
            ),
        }
    )
)
"""


def _run_child(code: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_SOURCE_DIR)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_the_batched_plant_modules_does_not_import_or_look_for_torch() -> None:
    report = _run_child(_TORCH_PROBE)
    assert report["torch_loaded_before"] is False
    assert report["torch_modules"] == []
    assert report["torch_import_attempts"] == []
    package_dir = (_SOURCE_DIR / "aeolus" / "habitat_v2").resolve()
    assert Path(report["contract_file"]).resolve().parent == package_dir
    assert Path(report["reference_file"]).resolve().parent == package_dir


def test_importing_the_habitat_v2_package_alone_imports_neither_batched_plant_module() -> None:
    report = _run_child(_PACKAGE_PROBE)
    assert report["batched_plant_modules"] == []
    assert report["torch_modules"] == []


# --------------------------------------------------------------------------
# Inputs are not mutated; rollout sequence validation
# --------------------------------------------------------------------------


def _snapshot(parameters: PlantParameters, batch: Batch) -> dict[str, np.ndarray]:
    arrays = {
        f"parameters.{key}": getattr(parameters, key) for key in ZONE_PARAMETER_KEYS
    }
    for owner, name in ALL_FIELDS:
        arrays[f"{owner}.{name}"] = getattr(batch[owner], name)
    return arrays


def _assert_unchanged(
    live: dict[str, np.ndarray], copies: dict[str, np.ndarray], scalars: tuple[Any, ...],
    parameters: PlantParameters,
) -> None:
    for key, array in live.items():
        assert array.dtype == copies[key].dtype, key
        assert array.shape == copies[key].shape, key
        assert array.tobytes() == copies[key].tobytes(), key
    assert scalars == tuple(getattr(parameters, key) for key in EQUIPMENT_PARAMETER_KEYS)
    assert parameters.zone_ids == SUPPORTED_ZONE_IDS


@pytest.mark.parametrize("seed", [9201, 9202, 9203])
def test_validate_step_inputs_and_step_arrays_do_not_mutate_their_inputs(seed: int) -> None:
    parameters = PlantParameters.from_scenario(load_scenario())
    state, control, disturbance = random_batch(parameters, 16, seed)
    batch = {"state": state, "control": control, "disturbance": disturbance}
    live = _snapshot(parameters, batch)
    copies = {key: array.copy() for key, array in live.items()}
    scalars = tuple(getattr(parameters, key) for key in EQUIPMENT_PARAMETER_KEYS)

    assert validate_step_inputs(parameters, state, control, disturbance, DT_SECONDS) == 60.0
    _assert_unchanged(live, copies, scalars, parameters)

    output = step_arrays(parameters, state, control, disturbance, dt_seconds=DT_SECONDS)
    _assert_unchanged(live, copies, scalars, parameters)
    assert type(output.state) is PlantStateArrays
    assert output.state is not state
    assert output.state.batch_shape == (16,)
    assert output.diagnostics.infeasible_code.shape == (16,)
    assert set(np.unique(output.diagnostics.infeasible_code)) <= set(
        INFEASIBLE_CODE_NAMES
    )

    rollout = rollout_arrays(
        parameters,
        state,
        (control, control),
        (disturbance, disturbance),
        dt_seconds=DT_SECONDS,
    )
    _assert_unchanged(live, copies, scalars, parameters)
    assert rollout.final_state.batch_shape == (16,)


def test_rejected_inputs_are_left_unmutated_by_validation_and_step(
    parameters: PlantParameters, batch: Batch
) -> None:
    state = batch["state"]
    bad_state = replace_fields(
        state, temperature_k=_with_element(state.temperature_k, 1000.0, 0)
    )
    bad_batch = {**batch, "state": bad_state}
    live = _snapshot(parameters, bad_batch)
    copies = {key: array.copy() for key, array in live.items()}
    scalars = tuple(getattr(parameters, key) for key in EQUIPMENT_PARAMETER_KEYS)
    with pytest.raises(BatchedPlantDomainError):
        _validate(parameters, bad_batch)
    with pytest.raises(BatchedPlantDomainError):
        step_arrays(
            parameters,
            bad_state,
            batch["control"],
            batch["disturbance"],
            dt_seconds=DT_SECONDS,
        )
    _assert_unchanged(live, copies, scalars, parameters)


def _rollout(
    parameters: PlantParameters, batch: Batch, controls: Any, disturbances: Any
) -> Any:
    return rollout_arrays(
        parameters, batch["state"], controls, disturbances, dt_seconds=DT_SECONDS
    )


def test_rollout_arrays_rejects_empty_sequences(
    parameters: PlantParameters, batch: Batch
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        _rollout(parameters, batch, (), ())


@pytest.mark.parametrize(("controls", "disturbances"), [(2, 1), (1, 2), (0, 1), (1, 0)])
def test_rollout_arrays_rejects_sequences_of_unequal_length(
    parameters: PlantParameters, batch: Batch, controls: int, disturbances: int
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        _rollout(
            parameters,
            batch,
            (batch["control"],) * controls,
            (batch["disturbance"],) * disturbances,
        )


@pytest.mark.parametrize("which", ["controls", "disturbances", "both"])
@pytest.mark.parametrize("container", ["list", "generator", "single_object", "none"])
def test_rollout_arrays_rejects_sequences_that_are_not_tuples(
    parameters: PlantParameters, batch: Batch, which: str, container: str
) -> None:
    def wrap(value: Any) -> Any:
        return {
            "list": [value],
            "generator": (item for item in (value,)),
            "single_object": value,
            "none": None,
        }[container]

    controls: Any = (batch["control"],)
    disturbances: Any = (batch["disturbance"],)
    if which in ("controls", "both"):
        controls = wrap(batch["control"])
    if which in ("disturbances", "both"):
        disturbances = wrap(batch["disturbance"])
    with pytest.raises(BatchedPlantDomainError):
        _rollout(parameters, batch, controls, disturbances)


@pytest.mark.parametrize("extra_steps", [1, 2, 1000])
def test_rollout_arrays_rejects_a_horizon_over_the_declared_reference_maximum(
    parameters: PlantParameters, batch: Batch, extra_steps: int
) -> None:
    steps = REFERENCE_ROLLOUT_MAX_STEPS + extra_steps
    with pytest.raises(BatchedPlantDomainError):
        _rollout(
            parameters,
            batch,
            (batch["control"],) * steps,
            (batch["disturbance"],) * steps,
        )


def test_rollout_arrays_accepts_a_horizon_exactly_at_the_declared_reference_maximum(
    parameters: PlantParameters, batch: Batch
) -> None:
    steps = REFERENCE_ROLLOUT_MAX_STEPS
    output = _rollout(
        parameters,
        batch,
        (batch["control"],) * steps,
        (batch["disturbance"],) * steps,
    )
    assert output.final_state.batch_shape == (WORLDS,)
    assert output.steps_applied.shape == (WORLDS,)
    assert bool(np.all(output.steps_applied >= 0))
    assert bool(np.all(output.steps_applied <= steps))
    feasible = output.infeasible_code == FEASIBLE
    assert bool(np.all(output.steps_applied[feasible] == steps))
    assert bool(np.all(output.first_infeasible_step[feasible] == -1))
    assert bool(
        np.all(
            output.first_infeasible_step[~feasible] == output.steps_applied[~feasible]
        )
    )


def test_rollout_arrays_rejects_a_sequence_containing_a_requested_command_mapping(
    parameters: PlantParameters, batch: Batch
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        _rollout(
            parameters,
            batch,
            (batch["control"], _timeline_command()),
            (batch["disturbance"], batch["disturbance"]),
        )


# --------------------------------------------------------------------------
# Closure of the state domain and immutability of the declared contract
# --------------------------------------------------------------------------


def test_declared_tables_and_code_names_are_read_only_mappings() -> None:
    table = reference_tolerances("float64", "one_step")
    with pytest.raises(TypeError):
        table["co2_mol"] = Tolerance(1e9, 1.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        INFEASIBLE_CODE_NAMES[9] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        EXCLUDED_MECHANISMS["air_network_solve"] = ""  # type: ignore[index]


@pytest.mark.parametrize(
    "terms", [(-1.0, 0.0), (0.0, -1.0), (float("nan"), 0.0), (0.0, float("inf"))]
)
def test_tolerance_rejects_negative_or_nonfinite_terms(
    terms: tuple[float, float],
) -> None:
    with pytest.raises(BatchedPlantDomainError):
        Tolerance(*terms)


def test_values_written_into_a_container_after_construction_are_still_rejected(
    parameters: PlantParameters, batch: Batch
) -> None:
    for key, name, value in (
        ("state", "co2_mol", np.nan),
        ("control", "scrubber_capture_ability", np.nan),
        ("disturbance", "generation_w", np.inf),
    ):
        fresh = {k: replace_fields(v) for k, v in batch.items()}
        getattr(fresh[key], name).flat[0] = value
        with pytest.raises(BatchedPlantDomainError):
            validate_step_inputs(
                parameters,
                fresh["state"],
                fresh["control"],
                fresh["disturbance"],
                DT_SECONDS,
            )
    tampered = PlantParameters.from_scenario(load_scenario())
    tampered.volume_m3[:] = -50.0
    with pytest.raises(BatchedPlantDomainError):
        _validate(tampered, batch)


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_a_batch_with_no_worlds_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(BatchedPlantDomainError):
        Disturbance(
            co2_generation_mol_s=np.zeros(shape + (ZONE_COUNT,)),
            o2_consumption_mol_s=np.zeros(shape + (ZONE_COUNT,)),
            water_vapor_generation_mol_s=np.zeros(shape + (ZONE_COUNT,)),
            sensible_heat_w=np.zeros(shape + (ZONE_COUNT,)),
            generation_w=np.zeros(shape),
        )


def test_a_scenario_edited_after_validation_is_rejected() -> None:
    scenario = load_scenario()
    scenario.data["equipment"]["base_load_w"] = 0.0
    with pytest.raises(BatchedPlantDomainError):
        require_supported_scenario(scenario)
    with pytest.raises(BatchedPlantDomainError):
        PlantParameters.from_scenario(scenario)


def test_oversized_integers_fail_as_domain_errors_not_overflow() -> None:
    with pytest.raises(BatchedPlantDomainError):
        require_supported_dt(10**400)


def _world_leaving_the_temperature_band(
    parameters: PlantParameters, heat_w: float
) -> tuple[PlantStateArrays, AchievedControl, Disturbance]:
    state, control, disturbance = random_batch(parameters, 3, 4242)
    ample = replace_fields(
        state,
        oxygen_store_mol=np.full(3, 500.0),
        battery_energy_wh=np.full(3, 12000.0),
    )
    heat = disturbance.sensible_heat_w.copy()
    heat[1, :] = heat_w
    loaded = replace_fields(
        disturbance, sensible_heat_w=heat, generation_w=np.full(3, 8000.0)
    )
    return ample, control, loaded


@pytest.mark.parametrize("heat_w", [4.0e6, 1.0e10, 1.0e306])
def test_a_world_whose_result_leaves_the_state_domain_is_held_and_flagged_alone(
    parameters: PlantParameters, heat_w: float
) -> None:
    state, control, disturbance = _world_leaving_the_temperature_band(
        parameters, heat_w
    )
    output = step_arrays(
        parameters, state, control, disturbance, dt_seconds=DT_SECONDS
    )
    assert output.diagnostics.infeasible_code.tolist() == [
        FEASIBLE,
        OUTSIDE_SUPPORTED_DOMAIN,
        FEASIBLE,
    ]
    for name in STATE_FIELDS:
        assert np.array_equal(getattr(output.state, name)[1], getattr(state, name)[1])
        assert np.all(np.isfinite(getattr(output.state, name)))
    assert output.diagnostics.served_load_wh[1] == 0.0

    calm_heat = disturbance.sensible_heat_w.copy()
    calm_heat[1, :] = 0.0
    calm = step_arrays(
        parameters,
        state,
        control,
        replace_fields(disturbance, sensible_heat_w=calm_heat),
        dt_seconds=DT_SECONDS,
    )
    for name in STATE_FIELDS:
        for world in (0, 2):
            assert np.array_equal(
                getattr(output.state, name)[world], getattr(calm.state, name)[world]
            )


def test_rollout_freezes_a_world_that_leaves_the_state_domain_and_keeps_neighbours(
    parameters: PlantParameters,
) -> None:
    state, control, disturbance = _world_leaving_the_temperature_band(
        parameters, 4.0e6
    )
    zero_oxygen = np.zeros_like(control.achieved_oxygen_injection_mol_s)
    control = replace_fields(
        control,
        achieved_oxygen_injection_mol_s=zero_oxygen,
        effective_oxygen_delivery_mol_s=zero_oxygen,
    )
    disturbance = replace_fields(
        disturbance, o2_consumption_mol_s=np.zeros_like(disturbance.sensible_heat_w)
    )
    rollout = rollout_arrays(
        parameters, state, (control,) * 3, (disturbance,) * 3, dt_seconds=DT_SECONDS
    )
    assert rollout.infeasible_code[1] == OUTSIDE_SUPPORTED_DOMAIN
    assert rollout.first_infeasible_step[1] == 0
    assert rollout.steps_applied[1] == 0
    assert np.array_equal(rollout.final_state.temperature_k[1], state.temperature_k[1])
    assert rollout.steps_applied[0] > 0
