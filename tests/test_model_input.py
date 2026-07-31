"""Gate 1 contracts for the fixed AEOLUS model-input tensor."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aeolus.config import load_scenario, parse_scenario
from aeolus.model_input import (
    assert_model_contract_compatible,
    build_model_input_contract,
    model_artifact_metadata,
    model_input_v1,
)
from aeolus.scenario import run_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_SCENARIO = REPO_ROOT / "scenarios" / "standard_habitat.json"


def _with_rehashed_topology(contract, topology: dict, *, fields=None):
    selected_fields = contract.fields if fields is None else fields
    topology_json = json.dumps(
        topology, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    topology_hash = hashlib.sha256(topology_json.encode()).hexdigest()
    selector = json.loads(contract.selector_json)
    selector["fields"] = [field.as_dict() for field in selected_fields]
    selector["topology_hash"] = topology_hash
    selector_json = json.dumps(
        selector, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return replace(
        contract,
        fields=selected_fields,
        selector_json=selector_json,
        selector_hash=hashlib.sha256(selector_json.encode()).hexdigest(),
        topology_json=topology_json,
        topology_hash=topology_hash,
    )


def test_model_input_v1_has_r2_order_shape_and_float32_dtype():
    config = load_scenario(STANDARD_SCENARIO)
    record = run_scenario(config)[0]

    tensor = model_input_v1(record, build_model_input_contract(config))

    expected = np.array(
        [
            record.zones["cabin_a"]["sensor_co2_concentration"],
            record.zones["cabin_b"]["sensor_co2_concentration"],
            record.zones["lab"]["sensor_co2_concentration"],
            record.actuators["cabin_a"]["setpoint"],
            record.actuators["cabin_a"]["actual_position"],
            record.actuators["cabin_a"]["tracking_residual"],
            record.actuators["cabin_a"]["power"],
            record.actuators["cabin_b"]["setpoint"],
            record.actuators["cabin_b"]["actual_position"],
            record.actuators["cabin_b"]["tracking_residual"],
            record.actuators["cabin_b"]["power"],
            record.actuators["lab"]["setpoint"],
            record.actuators["lab"]["actual_position"],
            record.actuators["lab"]["tracking_residual"],
            record.actuators["lab"]["power"],
            record.connections["cabin_a_to_processing"]["requested_airflow"],
            record.connections["cabin_a_to_processing"]["delivered_airflow"],
            record.connections["cabin_a_to_processing"]["airflow_residual"],
            record.connections["cabin_b_to_processing"]["requested_airflow"],
            record.connections["cabin_b_to_processing"]["delivered_airflow"],
            record.connections["cabin_b_to_processing"]["airflow_residual"],
            record.connections["lab_to_processing"]["requested_airflow"],
            record.connections["lab_to_processing"]["delivered_airflow"],
            record.connections["lab_to_processing"]["airflow_residual"],
        ],
        dtype=np.float32,
    )

    assert tensor.shape == (24,)
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor, expected)


def test_model_input_contract_hashes_are_canonical_and_topology_bound():
    config = load_scenario(STANDARD_SCENARIO)
    first = build_model_input_contract(config)
    second = build_model_input_contract(config)

    assert first.selector_json == second.selector_json
    assert first.topology_json == second.topology_json
    assert first.selector_hash == second.selector_hash
    assert first.topology_hash == second.topology_hash
    assert first.selector_hash == hashlib.sha256(first.selector_json.encode()).hexdigest()
    assert first.topology_hash == hashlib.sha256(first.topology_json.encode()).hexdigest()

    selector = json.loads(first.selector_json)
    topology = json.loads(first.topology_json)
    assert selector["schema_version"] == "model_input_v1"
    assert "version" not in selector
    assert topology["schema_version"] == "aeolus_topology_v1"
    assert "version" not in topology

    changed_selector = json.loads(first.selector_json)
    changed_selector["fields"][0]["field"] = "different_observable"
    changed_selector_json = json.dumps(
        changed_selector, sort_keys=True, separators=(",", ":")
    )
    assert hashlib.sha256(changed_selector_json.encode()).hexdigest() != first.selector_hash

    renamed_scenario = json.loads(STANDARD_SCENARIO.read_text(encoding="utf-8"))
    renamed_scenario["connections"][0]["id"] = "arbitrary_outbound_meter"
    renamed_contract = build_model_input_contract(parse_scenario(renamed_scenario))

    assert renamed_contract.topology_hash != first.topology_hash
    assert renamed_contract.selector_hash != first.selector_hash


def test_model_artifact_contract_mismatches_fail_closed():
    contract = build_model_input_contract(load_scenario(STANDARD_SCENARIO))
    metadata = model_artifact_metadata(contract)

    assert_model_contract_compatible(metadata, contract)

    for malformed in (
        None,
        {},
        {"model_input_version": "model_input_v1"},
        {**metadata, "unexpected": "value"},
        {**metadata, "selector_sha256": "0" * 64},
        {**metadata, "topology_sha256": "f" * 64},
        {**metadata, "model_input_version": "model_input_v2"},
    ):
        with pytest.raises(ValueError):
            assert_model_contract_compatible(malformed, contract)


def test_model_input_v1_rejects_finite_values_that_overflow_float32():
    config = load_scenario(STANDARD_SCENARIO)
    record = run_scenario(config)[0]
    record.zones["cabin_a"]["sensor_co2_concentration"] = 1e40

    with pytest.raises(ValueError, match="non-finite float32"):
        model_input_v1(record, build_model_input_contract(config))


@pytest.mark.parametrize("mutation", ("extra", "missing_return"))
def test_model_input_v1_rejects_connection_ids_outside_contract_topology(
    mutation: str,
):
    config = load_scenario(STANDARD_SCENARIO)
    record = copy.deepcopy(run_scenario(config)[0])

    if mutation == "extra":
        record.connections["uncontracted_connection"] = dict(
            record.connections["processing_to_cabin_a"]
        )
    else:
        record.connections.pop("processing_to_cabin_a")

    with pytest.raises(ValueError, match=r"record topology.*connections"):
        model_input_v1(record, build_model_input_contract(config))


@pytest.mark.parametrize(
    ("mutation", "group"),
    (
        ("extra_zone", "zones"),
        ("missing_processing_zone", "zones"),
        ("extra_actuator", "actuators"),
        ("missing_actuator", "actuators"),
    ),
)
def test_model_input_v1_rejects_zone_or_actuator_ids_outside_contract_topology(
    mutation: str,
    group: str,
):
    config = load_scenario(STANDARD_SCENARIO)
    record = copy.deepcopy(run_scenario(config)[0])

    if mutation == "extra_zone":
        record.zones["uncontracted_zone"] = dict(record.zones["cabin_a"])
    elif mutation == "missing_processing_zone":
        record.zones.pop("processing")
    elif mutation == "extra_actuator":
        record.actuators["uncontracted_actuator"] = dict(record.actuators["cabin_a"])
    else:
        record.actuators.pop("cabin_a")

    with pytest.raises(ValueError, match=rf"record topology.*{group}"):
        model_input_v1(record, build_model_input_contract(config))


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_topology_key",
        "extra_topology_key",
        "missing_loop_key",
        "extra_loop_key",
        "missing_edge_key",
        "extra_edge_key",
        "empty_zone_id",
        "empty_return_id",
        "null_return_id",
        "processing_zone_collision",
        "outbound_endpoint_mismatch",
        "return_endpoint_mismatch",
        "duplicate_zone_id",
        "duplicate_outbound_id",
        "duplicate_return_id",
        "cross_direction_duplicate_edge_id",
    ),
)
def test_model_input_contract_rejects_malformed_topology_structure(mutation: str):
    contract = build_model_input_contract(load_scenario(STANDARD_SCENARIO))
    topology = json.loads(contract.topology_json)
    first_loop = topology["primary_loops"][0]
    fields = contract.fields

    if mutation == "missing_topology_key":
        topology.pop("processing_zone_id")
    elif mutation == "extra_topology_key":
        topology["unexpected"] = "value"
    elif mutation == "missing_loop_key":
        first_loop.pop("return")
    elif mutation == "extra_loop_key":
        first_loop["unexpected"] = "value"
    elif mutation == "missing_edge_key":
        first_loop["outbound"].pop("id")
    elif mutation == "extra_edge_key":
        first_loop["outbound"]["unexpected"] = "value"
    elif mutation == "empty_zone_id":
        topology["non_processing_zone_ids"][0] = ""
        first_loop["zone_id"] = ""
        first_loop["outbound"]["from_zone"] = ""
        first_loop["return"]["to_zone"] = ""
    elif mutation == "empty_return_id":
        first_loop["return"]["id"] = ""
    elif mutation == "null_return_id":
        first_loop["return"]["id"] = None
    elif mutation == "processing_zone_collision":
        original_zone_id = topology["non_processing_zone_ids"][0]
        processing_id = topology["processing_zone_id"]
        topology["non_processing_zone_ids"][0] = processing_id
        first_loop["zone_id"] = processing_id
        first_loop["outbound"]["from_zone"] = processing_id
        first_loop["return"]["to_zone"] = processing_id
        fields = tuple(
            replace(field, entity_id=processing_id)
            if field.entity_id == original_zone_id and field.group in {"zones", "actuators"}
            else field
            for field in contract.fields
        )
    elif mutation == "outbound_endpoint_mismatch":
        first_loop["outbound"]["to_zone"] = "lab"
    elif mutation == "return_endpoint_mismatch":
        first_loop["return"]["from_zone"] = "lab"
    elif mutation == "duplicate_zone_id":
        topology["non_processing_zone_ids"][1] = topology["non_processing_zone_ids"][0]
        topology["primary_loops"][1]["zone_id"] = topology["primary_loops"][0]["zone_id"]
    elif mutation == "duplicate_outbound_id":
        topology["primary_loops"][1]["outbound"]["id"] = first_loop["outbound"]["id"]
    elif mutation == "duplicate_return_id":
        topology["primary_loops"][1]["return"]["id"] = first_loop["return"]["id"]
    elif mutation == "cross_direction_duplicate_edge_id":
        first_loop["return"]["id"] = first_loop["outbound"]["id"]

    with pytest.raises(ValueError, match="topology is malformed"):
        model_artifact_metadata(_with_rehashed_topology(contract, topology, fields=fields))


def test_model_input_v1_excludes_processing_sensor_and_return_leg_values():
    config = load_scenario(STANDARD_SCENARIO)
    contract = build_model_input_contract(config)
    record = run_scenario(config)[0]
    expected = model_input_v1(record, contract)

    changed = copy.deepcopy(record)
    changed.zones["processing"]["sensor_co2_concentration"] = 1234.0
    changed.connections["processing_to_cabin_a"].update(
        requested_airflow=9.0,
        delivered_airflow=7.0,
        airflow_residual=2.0,
    )

    np.testing.assert_array_equal(model_input_v1(changed, contract), expected)


def test_model_input_v1_never_reads_hidden_truth_or_accepts_health_telemetry():
    config = load_scenario(STANDARD_SCENARIO)
    contract = build_model_input_contract(config)
    record = run_scenario(config)[0]
    expected = model_input_v1(record, contract)

    object.__setattr__(
        record,
        "hidden_fault_truth",
        {
            "fault_type": "blocked_path",
            "target_metadata": {"connection_id": "cabin_a_to_processing"},
            "schedule": {"start_tick": 20},
            "random_seed": 7,
        },
    )

    np.testing.assert_array_equal(model_input_v1(record, contract), expected)

    poisoned = copy.deepcopy(record)
    poisoned.connections["cabin_a_to_processing"]["health"] = 0.4
    with pytest.raises(ValueError, match="unexpected telemetry"):
        model_input_v1(poisoned, contract)


def test_model_input_v1_rejects_malformed_or_incompatible_contracts():
    config = load_scenario(STANDARD_SCENARIO)
    record = run_scenario(config)[0]
    contract = build_model_input_contract(config)

    malformed = replace(contract, fields=contract.fields[:-1])
    with pytest.raises(ValueError, match="unexpected field count"):
        model_input_v1(record, malformed)
    with pytest.raises(ValueError, match="unexpected field count"):
        model_artifact_metadata(malformed)

    reordered = replace(contract, fields=tuple(reversed(contract.fields)))
    with pytest.raises(ValueError, match="does not match its topology"):
        model_input_v1(record, reordered)
    with pytest.raises(ValueError, match="does not match its topology"):
        model_artifact_metadata(reordered)

    altered_fields = (*contract.fields[:-1], replace(contract.fields[-1], field="requested_airflow"))
    altered_selector = json.loads(contract.selector_json)
    altered_selector["fields"][-1]["field"] = "requested_airflow"
    altered_selector_json = json.dumps(
        altered_selector, sort_keys=True, separators=(",", ":")
    )
    altered = replace(
        contract,
        fields=altered_fields,
        selector_json=altered_selector_json,
        selector_hash=hashlib.sha256(altered_selector_json.encode()).hexdigest(),
    )
    with pytest.raises(ValueError, match="does not match its topology"):
        model_input_v1(record, altered)

    renamed_scenario = json.loads(STANDARD_SCENARIO.read_text(encoding="utf-8"))
    renamed_scenario["connections"][0]["id"] = "different_outbound_meter"
    incompatible = build_model_input_contract(parse_scenario(renamed_scenario))
    with pytest.raises(ValueError, match=r"record topology.*connections"):
        model_input_v1(record, incompatible)
