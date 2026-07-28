"""Gate 1 contracts for the fixed ICARUS model-input tensor."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from icarus.config import load_scenario, parse_scenario
from icarus.model_input import (
    assert_model_contract_compatible,
    build_model_input_contract,
    model_artifact_metadata,
    model_input_v1,
)
from icarus.scenario import run_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_SCENARIO = REPO_ROOT / "scenarios" / "standard_habitat.json"


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
    assert topology["schema_version"] == "icarus_topology_v1"
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
    with pytest.raises(ValueError, match="does not satisfy"):
        model_input_v1(record, incompatible)
