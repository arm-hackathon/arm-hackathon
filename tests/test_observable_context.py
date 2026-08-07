"""Fail-closed V6 observable-context projection for specialist diagnostics."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aeolus.config import load_scenario
from aeolus.observable_context import (
    OBSERVABLE_CONTEXT_DTYPE,
    OBSERVABLE_CONTEXT_VERSION,
    assert_observable_context_compatible,
    build_observable_context_contract,
    observable_context_metadata,
    observable_context_v1,
)
from aeolus.scenario import run_scenario

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "scenarios" / "standard_habitat.json"


def test_observable_context_v1_has_declared_full_control_shape_and_order():
    config = load_scenario(STANDARD)
    record = run_scenario(config)[0]
    contract = build_observable_context_contract(config)

    tensor = observable_context_v1(record, contract)

    assert OBSERVABLE_CONTEXT_VERSION == "observable_context_v1"
    assert OBSERVABLE_CONTEXT_DTYPE == "float32"
    assert tensor.dtype == np.float32
    assert tensor.shape == (46,)
    np.testing.assert_array_equal(
        tensor[:7],
        np.array(
            [
                record.zones["cabin_a"]["sensor_co2_concentration"],
                record.zones["cabin_b"]["sensor_co2_concentration"],
                record.zones["lab"]["sensor_co2_concentration"],
                record.actuators["cabin_a"]["setpoint"],
                record.actuators["cabin_a"]["actual_position"],
                record.actuators["cabin_a"]["tracking_residual"],
                record.actuators["cabin_a"]["moving"],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        tensor[-4:],
        np.array(
            [
                record.system["shared_airflow_capacity"],
                record.system["total_requested_airflow"],
                record.system["total_delivered_airflow"],
                record.system["capacity_scale"],
            ],
            dtype=np.float32,
        ),
    )


def test_observable_context_excludes_occupancy_and_hidden_plant_fields():
    config = load_scenario(STANDARD)
    record = run_scenario(config)[0]
    contract = build_observable_context_contract(config)
    expected = observable_context_v1(record, contract)
    altered = copy.deepcopy(record)
    altered.zones["cabin_a"]["occupancy_multiplier"] = 9.0
    altered.zones["cabin_a"]["co2_mass"] = 123.0
    altered.zones["cabin_a"]["source_co2_mass"] = 456.0

    np.testing.assert_array_equal(observable_context_v1(altered, contract), expected)
    assert all(field.field not in {"occupancy_multiplier", "co2_mass", "source_co2_mass"} for field in contract.fields)


def test_observable_context_rejects_runtime_topology_and_contract_drift():
    config = load_scenario(STANDARD)
    record = run_scenario(config)[0]
    contract = build_observable_context_contract(config)
    broken_record = copy.deepcopy(record)
    broken_record.connections.pop("processing_to_cabin_a")
    with pytest.raises(ValueError, match="topology"):
        observable_context_v1(broken_record, contract)

    with pytest.raises(ValueError, match="context selector"):
        observable_context_v1(record, replace(contract, selector_hash="0" * 64))


def test_observable_context_metadata_is_exact_and_bound_to_contract():
    contract = build_observable_context_contract(load_scenario(STANDARD))
    metadata = observable_context_metadata(contract)

    assert_observable_context_compatible(metadata, contract)
    for malformed in (None, {}, {**metadata, "extra": "x"}, {**metadata, "selector_sha256": "0" * 64}):
        with pytest.raises(ValueError):
            assert_observable_context_compatible(malformed, contract)
