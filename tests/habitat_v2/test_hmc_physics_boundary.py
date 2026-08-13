from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus.habitat_v2 import physics
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.state import PlantState

SCENARIOS = (
    "habitat_v2_reference.json",
    "habitat_v2_operating_modes.json",
    "habitat_v2_air_network.json",
    "habitat_v2_compound_faults.json",
    "habitat_v2_actuator_feedback.json",
)


def _scenario(name: str) -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / name
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _reachable_local_function_nodes(entry_name: str) -> dict[str, ast.FunctionDef]:
    module_ast = ast.parse(inspect.getsource(physics))
    definitions = {
        node.name: node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reachable: dict[str, ast.FunctionDef] = {}
    pending = [entry_name]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        node = definitions[name]
        reachable[name] = node
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in definitions
            ):
                pending.append(child.func.id)
    return reachable


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_public_command_validation_is_timeline_independent(
    scenario_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(scenario_name)
    command = scenario.data["timeline"][0]["command"]

    def timeline_must_not_be_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("command validation reached the scenario timeline")

    monkeypatch.setattr(physics, "_segment_for_step", timeline_must_not_be_read)

    canonical = physics.validate_external_command(scenario, command)

    assert canonical.to_mapping() == command
    assert canonical.sha256 == hashlib.sha256(canonical.canonical_bytes).hexdigest()
    assert json.loads(canonical.canonical_bytes) == command

    reachable = _reachable_local_function_nodes("validate_external_command")
    forbidden_names = {"_segment_for_step"}
    forbidden_literals = {"timeline", "loads", "generation_w", "operating_mode"}
    for function_name, node in reachable.items():
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        literals = {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        assert names.isdisjoint(forbidden_names), function_name
        assert literals.isdisjoint(forbidden_literals), function_name


def test_initial_hold_uses_only_v5_achieved_state_and_is_explicitly_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario("habitat_v2_actuator_feedback.json")
    state = physics.initial_state(scenario)

    def timeline_must_not_be_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("achieved-state hold reached the scenario timeline")

    monkeypatch.setattr(physics, "_segment_for_step", timeline_must_not_be_read)

    hold = physics.command_from_achieved_state(scenario, state)
    command = hold.command.to_mapping()

    assert hold.command_reference_kind == "INITIAL_ACHIEVED_STATE_HOLD"
    assert command == {
        "fan_speed_fraction": state.utility.actual_fan_speed_fraction,
        "damper_position_by_id": dict(state.utility.actual_damper_position_by_id),
        "scrubber_duty": state.utility.actual_scrubber_duty,
        "condenser_duty": state.utility.actual_condenser_duty,
        "cooling_removed_w": dict(state.utility.actual_cooling_removed_w),
        "oxygen_injection_mol_s": dict(state.utility.actual_oxygen_injection_mol_s),
    }
    assert physics.validate_external_command(scenario, command) == hold.command


def _preflight_mapping_without_digest(result: object) -> dict[str, object]:
    mapping = result.to_mapping()  # type: ignore[attr-defined]
    digest = mapping.pop("preflight_result_sha256")
    assert isinstance(digest, str)
    return mapping


def test_preflight_is_closed_deterministic_and_does_not_mutate_state() -> None:
    scenario = _scenario("habitat_v2_actuator_feedback.json")
    state = physics.initial_state(scenario)
    hold = physics.command_from_achieved_state(scenario, state)

    result = physics.preflight_external_command(
        scenario,
        state,
        hold.command.to_mapping(),
        application_step=0,
    )

    assert result.classification == "FEASIBLE"
    assert set(result.to_mapping()) == {
        "classification",
        "application_step",
        "command_sha256",
        "preflight_contract_sha256",
        "preflight_result_sha256",
    }
    contract_mapping = json.loads(
        (Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected_preflight_contract_sha256 = hashlib.sha256(
        json.dumps(
            contract_mapping["preflight_contract"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert result.command_sha256 == hold.command.sha256
    assert result.preflight_contract_sha256 == expected_preflight_contract_sha256
    assert (
        result.preflight_result_sha256
        == hashlib.sha256(
            json.dumps(
                _preflight_mapping_without_digest(result),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert state == physics.initial_state(scenario)


def test_preflight_collapses_physical_infeasibility_without_leaking_detail() -> None:
    scenario = _scenario("habitat_v2_actuator_feedback.json")
    original = physics.initial_state(scenario)
    empty_oxygen_state = PlantState(
        step=original.step,
        zones=original.zones,
        utility=replace(original.utility, oxygen_store_mol=0.0),
    )
    command = physics.command_from_achieved_state(
        scenario, original
    ).command.to_mapping()
    first_zone = min(command["oxygen_injection_mol_s"])
    command["oxygen_injection_mol_s"][first_zone] = 0.001

    result = physics.preflight_external_command(
        scenario,
        empty_oxygen_state,
        command,
        application_step=0,
    )

    assert result.classification == "INFEASIBLE"
    assert set(result.to_mapping()) == {
        "classification",
        "application_step",
        "command_sha256",
        "preflight_contract_sha256",
        "preflight_result_sha256",
    }
    assert empty_oxygen_state.utility.oxygen_store_mol == 0.0
