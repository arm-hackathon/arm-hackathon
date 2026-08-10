from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

import aeolus.habitat_v2.runner as runner_module
from aeolus.habitat_v2.runner import run_scenario
from aeolus.habitat_v2.trace import TraceValidationError, validate_trace_bytes
from aeolus.habitat_v2.scenario import (
    EQUATION_CONTRACT_REVISION,
    SCENARIO_SCHEMA_VERSION_V2,
    TRACE_SCHEMA_VERSION_V1,
    TRACE_SCHEMA_VERSION_V2,
    Scenario,
    ScenarioValidationError,
    derive_run_id,
)

from ._helpers import reference_scenario_mapping


def scenario_v2_mapping() -> dict:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["schema_version"] = "aeolus_habitat_v2_scenario_v2"
    for segment in mapping["timeline"]:
        segment["operating_mode"] = "occupied"
    return mapping


def four_mode_scenario_mapping() -> dict:
    mapping = scenario_v2_mapping()
    segment = mapping["timeline"][0]
    mapping["timeline"] = [
        {
            **deepcopy(segment),
            "start_step": step,
            "end_step": step + 1,
            "operating_mode": mode,
        }
        for step, mode in enumerate(
            ("occupied", "eva_transition", "contingency", "dormant")
        )
    ]
    return mapping


def _encode_rows(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def _physical_rows_without_mode_context(rows: tuple[dict, ...]) -> list[dict]:
    physical_rows = deepcopy(list(rows))
    for row in physical_rows:
        del row["lineage"]
        del row["applied_operating_mode"]
    return physical_rows


def test_scenario_v1_rejects_mode_while_scenario_v2_requires_it() -> None:
    v1 = reference_scenario_mapping()
    v1["timeline"][0]["operating_mode"] = "occupied"
    with pytest.raises(
        ScenarioValidationError, match="unknown timeline segment fields"
    ):
        Scenario.from_mapping(v1)

    v2 = scenario_v2_mapping()
    del v2["timeline"][0]["operating_mode"]
    with pytest.raises(
        ScenarioValidationError, match="missing timeline segment fields"
    ):
        Scenario.from_mapping(v2)


def test_scenario_v2_checks_operating_mode_on_later_segments() -> None:
    mapping = four_mode_scenario_mapping()
    del mapping["timeline"][3]["operating_mode"]

    with pytest.raises(
        ScenarioValidationError, match="missing timeline segment fields"
    ):
        Scenario.from_mapping(mapping)


def test_unknown_scenario_contract_version_fails_closed() -> None:
    mapping = reference_scenario_mapping()
    mapping["schema_version"] = "aeolus_habitat_v2_scenario_v3"

    with pytest.raises(ScenarioValidationError, match="schema_version must be"):
        Scenario.from_mapping(mapping)


def test_mismatched_scenario_trace_contract_fails_before_physics(monkeypatch) -> None:
    scenario = Scenario.from_mapping(scenario_v2_mapping())
    mismatched = replace(scenario, trace_schema_version=TRACE_SCHEMA_VERSION_V1)

    def fail_if_physics_starts(_scenario: Scenario) -> None:
        raise AssertionError("physics started before contract identity validation")

    monkeypatch.setattr(runner_module, "initial_state", fail_if_physics_starts)
    with pytest.raises(ScenarioValidationError, match="trace schema"):
        runner_module.run_scenario(mismatched)


@pytest.mark.parametrize("mutation", ["forged_digest", "mutated_data"])
def test_scenario_data_and_digest_are_bound_before_physics(
    monkeypatch, mutation: str
) -> None:
    scenario = Scenario.from_mapping(scenario_v2_mapping())
    if mutation == "forged_digest":
        forged_digest = "0" * 64
        scenario = replace(
            scenario,
            scenario_sha256=forged_digest,
            run_id=derive_run_id(
                scenario_sha256=forged_digest,
                scenario_schema_version=scenario.scenario_schema_version,
                trace_schema_version=scenario.trace_schema_version,
                equation_contract_revision=scenario.equation_contract_revision,
            ),
        )
    else:
        scenario.data["timeline"][0]["operating_mode"] = "dormant"

    def fail_if_physics_starts(_scenario: Scenario) -> None:
        raise AssertionError("physics started with unbound scenario identity")

    monkeypatch.setattr(runner_module, "initial_state", fail_if_physics_starts)
    with pytest.raises(ScenarioValidationError, match="canonical|digest"):
        runner_module.run_scenario(scenario)


@pytest.mark.parametrize("mode", ["sleep", 3, None])
def test_scenario_v2_rejects_unknown_or_non_string_operating_mode(mode: object) -> None:
    mapping = scenario_v2_mapping()
    mapping["timeline"][0]["operating_mode"] = mode

    with pytest.raises(ScenarioValidationError, match="operating_mode"):
        Scenario.from_mapping(mapping)


def test_scenario_v2_carries_its_own_contract_identities_for_lineage() -> None:
    scenario = Scenario.from_mapping(scenario_v2_mapping())

    assert scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V2
    assert scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V2
    assert scenario.equation_contract_revision == EQUATION_CONTRACT_REVISION
    assert scenario.run_id == derive_run_id(
        scenario_sha256=scenario.scenario_sha256,
        scenario_schema_version=scenario.scenario_schema_version,
        trace_schema_version=scenario.trace_schema_version,
        equation_contract_revision=scenario.equation_contract_revision,
    )


def test_original_v1_scenario_run_and_trace_identities_are_preserved() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    run = run_scenario(scenario)

    assert (
        scenario.scenario_sha256
        == "8bd3586ace18d008417122b127258b90ad255e622fe329eb70a580d38ed7b48d"
    )
    assert (
        scenario.run_id
        == "e0ff08d2e00a06bfabf82ddfca43ca67d19f9ade9c778f186a10d31f92c64c75"
    )
    assert (
        hashlib.sha256(run.trace_bytes).hexdigest()
        == "a94b098cf8707cde6383319be913032de53053d033fe6a7d2f0a07efad6260fb"
    )


def test_mode_only_changes_lineage_without_changing_physical_evolution() -> None:
    first_mapping = four_mode_scenario_mapping()
    second_mapping = deepcopy(first_mapping)
    for segment in second_mapping["timeline"]:
        segment["operating_mode"] = "occupied"

    first_scenario = Scenario.from_mapping(first_mapping)
    second_scenario = Scenario.from_mapping(second_mapping)
    first_run = run_scenario(first_scenario)
    second_run = run_scenario(second_scenario)

    assert first_scenario.scenario_sha256 != second_scenario.scenario_sha256
    assert first_scenario.run_id != second_scenario.run_id
    assert _physical_rows_without_mode_context(
        first_run.rows
    ) == _physical_rows_without_mode_context(second_run.rows)


def test_trace_v2_records_the_mode_of_the_interval_that_produced_each_row() -> None:
    run = run_scenario(Scenario.from_mapping(four_mode_scenario_mapping()))

    assert [row["applied_operating_mode"] for row in run.rows] == [
        None,
        "occupied",
        "eva_transition",
        "contingency",
        "dormant",
    ]


def test_trace_v2_validates_against_its_parsed_scenario() -> None:
    scenario = Scenario.from_mapping(four_mode_scenario_mapping())
    run = run_scenario(scenario)

    assert validate_trace_bytes(run.trace_bytes, scenario=scenario) == run.rows


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_trace_v2_operating_mode_field_is_closed_schema(mutation: str) -> None:
    scenario = Scenario.from_mapping(four_mode_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    if mutation == "missing":
        del rows[3]["applied_operating_mode"]
    else:
        rows[3]["unreviewed_mode_context"] = "occupied"

    with pytest.raises(TraceValidationError, match="invalid row 3 fields"):
        validate_trace_bytes(_encode_rows(rows), scenario=scenario)


def test_trace_v2_rejects_a_forged_interval_operating_mode() -> None:
    scenario = Scenario.from_mapping(four_mode_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[2]["applied_operating_mode"] = "dormant"

    with pytest.raises(TraceValidationError, match="applied operating mode row 2"):
        validate_trace_bytes(_encode_rows(rows), scenario=scenario)


def test_trace_v2_requires_null_operating_mode_for_the_initial_row() -> None:
    scenario = Scenario.from_mapping(four_mode_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[0]["applied_operating_mode"] = "occupied"

    with pytest.raises(
        TraceValidationError,
        match="applied operating mode row 0 must be null",
    ):
        validate_trace_bytes(_encode_rows(rows), scenario=scenario)


@pytest.mark.parametrize("mode", [None, 3, True])
def test_trace_v2_rejects_non_string_post_step_operating_mode(mode: object) -> None:
    scenario = Scenario.from_mapping(four_mode_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[1]["applied_operating_mode"] = mode

    with pytest.raises(
        TraceValidationError,
        match="applied operating mode row 1 must be a string",
    ):
        validate_trace_bytes(_encode_rows(rows), scenario=scenario)
