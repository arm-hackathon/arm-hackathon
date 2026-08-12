from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

import aeolus.habitat_v2.runner as runner_module
from aeolus.habitat_v2.physics import StepResult, advance_one_step, initial_state
from aeolus.habitat_v2.runner import (
    AccountingInvariantError,
    run_scenario,
    validate_accounting_receipt,
)
from aeolus.habitat_v2.scenario import (
    EQUATION_CONTRACT_REVISION,
    SCENARIO_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    Scenario,
    derive_run_id,
)
from aeolus.habitat_v2.trace import TraceValidationError, validate_trace_bytes

from ._helpers import reference_scenario_mapping, reversed_object_keys


def encode_rows(rows) -> bytes:
    return b"".join(
        json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def test_identical_runs_emit_byte_identical_trace_with_lineage_on_every_row() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())

    first = run_scenario(scenario)
    second = run_scenario(scenario)

    assert first.trace_bytes == second.trace_bytes
    rows = [json.loads(line) for line in first.trace_bytes.splitlines()]
    assert len(rows) == int(scenario.data["steps"]) + 1
    assert [row["step"] for row in rows] == list(range(int(scenario.data["steps"]) + 1))
    for row in rows:
        assert set(row) == {
            "schema_version",
            "lineage",
            "step",
            "time_s",
            "telemetry",
            "commanded_action",
            "actual_action",
            "resource_state",
            "realised_loads",
            "accounting_receipt",
            "invariant_status",
        }
        assert row["lineage"] == {
            "run_id": scenario.run_id,
            "scenario_sha256": scenario.scenario_sha256,
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "equation_contract_revision": EQUATION_CONTRACT_REVISION,
        }
        assert row["invariant_status"] == {"passed": True}
    assert rows[0]["commanded_action"] is None
    assert rows[0]["realised_loads"] is None
    assert rows[0]["accounting_receipt"] is None


def test_object_key_order_does_not_change_trace_bytes() -> None:
    first = Scenario.from_mapping(reference_scenario_mapping())
    reordered = Scenario.from_mapping(
        reversed_object_keys(reference_scenario_mapping())
    )

    assert run_scenario(first).trace_bytes == run_scenario(reordered).trace_bytes


def test_trace_telemetry_exposes_only_operational_environmental_channels() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    row = run_scenario(scenario).rows[1]

    assert set(row["telemetry"]) == {"crew_cabin", "work_airlock"}
    for zone in row["telemetry"].values():
        assert set(zone) == {
            "temperature_k",
            "pressure_pa",
            "co2_ppm",
            "o2_mole_fraction",
            "relative_humidity",
        }


def test_loader_rejects_stale_run_id_after_lineage_tampering() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    run = run_scenario(scenario)
    rows = [json.loads(line) for line in run.trace_bytes.splitlines()]
    rows[1]["lineage"]["run_id"] = "0" * 64
    tampered = encode_rows(rows)

    with pytest.raises(TraceValidationError, match="run_id does not match lineage"):
        validate_trace_bytes(tampered, scenario=scenario)


def test_loader_rejects_self_consistent_lineage_for_wrong_scenario() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    forged_digest = "1" * 64
    forged_run_id = derive_run_id(scenario_sha256=forged_digest)
    for row in rows:
        row["lineage"]["scenario_sha256"] = forged_digest
        row["lineage"]["run_id"] = forged_run_id

    with pytest.raises(TraceValidationError, match="scenario digest does not match"):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


def test_loader_rejects_truncated_trace() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    initial_row_only = run_scenario(scenario).trace_bytes.splitlines(keepends=True)[0]

    with pytest.raises(TraceValidationError, match="expected 5 rows"):
        validate_trace_bytes(initial_row_only, scenario=scenario)


def test_loader_rejects_non_deterministic_time_axis() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[1]["time_s"] = 61.0

    with pytest.raises(TraceValidationError, match="unexpected time_s at row 1"):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


def test_loader_rejects_wrong_type_telemetry_scalar() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[1]["telemetry"]["crew_cabin"]["temperature_k"] = "not-a-number"

    with pytest.raises(TraceValidationError, match="temperature_k must be finite"):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


@pytest.mark.parametrize(
    "path",
    [
        ("commanded_action", "scrubber_duty"),
        ("realised_loads", "crew_cabin", "co2_generation_mol_s"),
    ],
)
def test_loader_rejects_timeline_action_or_load_mutation(path) -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    target = rows[1]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] += 0.01

    with pytest.raises(TraceValidationError, match="does not match scenario timeline"):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


def test_loader_rejects_finite_numeric_physical_trace_mutation() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[1]["telemetry"]["crew_cabin"]["temperature_k"] += 0.01

    with pytest.raises(TraceValidationError, match="deterministic replay"):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


def test_loader_rejects_unknown_later_row_telemetry_field() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    rows[1]["telemetry"]["crew_cabin"]["hidden_health"] = 1.0

    with pytest.raises(TraceValidationError, match="telemetry row 1 crew_cabin"):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (("commanded_action",), "commanded action row 1"),
        (("actual_action",), "actual action row 1"),
        (("resource_state",), "resource state row 1"),
        (("realised_loads", "crew_cabin"), "loads row 1 crew_cabin"),
        (("accounting_receipt",), "accounting receipt row 1"),
        (("invariant_status",), "invariant status row 1"),
    ],
)
def test_loader_rejects_unknown_nested_trace_fields(path, label) -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    rows = [
        json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
    ]
    target = rows[1]
    for part in path:
        target = target[part]
    target["unreviewed_field"] = 1.0

    with pytest.raises(TraceValidationError, match=label):
        validate_trace_bytes(encode_rows(rows), scenario=scenario)


def test_receipt_validator_rejects_species_residual_beyond_tolerance() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt["species_accounting"]["co2_residual_mol"] = (
        2.0 * receipt["species_accounting"]["tolerance_mol"]
    )

    with pytest.raises(AccountingInvariantError, match="co2_residual_mol"):
        validate_accounting_receipt(receipt)


def test_receipt_validator_rejects_zone_thermal_residual_beyond_tolerance() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt["thermal"]["zones"]["crew_cabin"]["zone_thermal_residual_j"] = 1.0

    with pytest.raises(AccountingInvariantError, match="crew_cabin thermal residual"):
        validate_accounting_receipt(receipt)


def test_receipt_validator_rejects_system_thermal_residual_beyond_tolerance() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt["thermal"]["system_residual_j"] = 1.0

    with pytest.raises(AccountingInvariantError, match="system thermal residual"):
        validate_accounting_receipt(receipt)


def test_receipt_validator_rejects_electrical_residual_beyond_tolerance() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt["electrical"]["residual_wh"] = 1.0

    with pytest.raises(AccountingInvariantError, match="electrical residual"):
        validate_accounting_receipt(receipt)


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "path",
    [
        ("species_accounting", "tolerance_mol"),
        ("species_accounting", "co2_residual_mol"),
        ("species_accounting", "o2_residual_mol"),
        ("species_accounting", "water_residual_mol"),
        ("species_accounting", "inert_residual_mol"),
        ("thermal", "system_residual_j"),
        ("electrical", "generation_wh"),
        ("electrical", "battery_withdrawn_wh"),
        ("electrical", "served_load_wh"),
        ("electrical", "battery_charge_stored_wh"),
        ("electrical", "curtailed_generation_wh"),
        ("electrical", "charge_conversion_loss_wh"),
        ("electrical", "discharge_conversion_loss_wh"),
        ("electrical", "residual_wh"),
    ],
)
def test_receipt_validator_rejects_non_finite_accounting_values(
    path, non_finite
) -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt[path[0]][path[1]] = non_finite

    with pytest.raises(AccountingInvariantError, match="finite numeric data"):
        validate_accounting_receipt(receipt)


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    [
        "metabolic_heat_added_j",
        "recirculation_heat_added_j",
        "cooling_heat_removed_j",
        "passive_heat_rejected_j",
        "passive_heat_received_j",
        "zone_thermal_energy_delta_j",
        "zone_thermal_residual_j",
    ],
)
def test_receipt_validator_rejects_non_finite_zone_thermal_values(
    field, non_finite
) -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt["thermal"]["zones"]["crew_cabin"][field] = non_finite

    with pytest.raises(AccountingInvariantError, match="finite numeric data"):
        validate_accounting_receipt(receipt)


def test_receipt_validator_rejects_negative_species_tolerance() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    receipt = deepcopy(advance_one_step(scenario, initial_state(scenario)).receipt)
    receipt["species_accounting"]["tolerance_mol"] = -1.0

    with pytest.raises(AccountingInvariantError, match="must be non-negative"):
        validate_accounting_receipt(receipt)


def test_runner_rejects_bad_accounting_before_emitting_passed_row(monkeypatch) -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    real_advance = advance_one_step

    def advance_with_bad_receipt(scenario_arg, state_arg) -> StepResult:
        result = real_advance(scenario_arg, state_arg)
        receipt = deepcopy(result.receipt)
        receipt["electrical"]["residual_wh"] = 1.0
        return StepResult(state=result.state, receipt=receipt)

    monkeypatch.setattr(runner_module, "advance_one_step", advance_with_bad_receipt)

    with pytest.raises(AccountingInvariantError, match="electrical residual"):
        run_scenario(scenario)
