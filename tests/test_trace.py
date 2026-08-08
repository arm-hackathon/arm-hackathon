"""Tests for the JSONL trace writer."""

import json
from pathlib import Path

import pytest

from aeolus.config import load_scenario
from aeolus.trace import (
    RECOVERY_TRACE_VERSION,
    RecoveryTickRecord,
    RecoveryTraceWriter,
    TickRecord,
    TraceWriter,
)

RECOVERY_SCENARIO = (
    Path(__file__).resolve().parents[1] / "scenarios" / "recovery_habitat.json"
)


@pytest.fixture
def recovery_config():
    return load_scenario(RECOVERY_SCENARIO)


def _record(tick: int) -> TickRecord:
    return TickRecord(
        tick=tick,
        zones={
            "cabin_a": {
                "co2_mass": 10.0 + tick,
                "co2_concentration": 0.10 + tick / 100.0,
                "sensor_co2_concentration": 0.11 + tick / 100.0,
                "source_co2_mass": 1.01,
                "occupancy_multiplier": 1.0,
            },
            "cabin_b": {
                "co2_mass": 9.0 + tick,
                "co2_concentration": 0.09 + tick / 100.0,
                "sensor_co2_concentration": 0.10 + tick / 100.0,
                "source_co2_mass": 0.99,
                "occupancy_multiplier": 1.0,
            },
            "lab": {
                "co2_mass": 0.0,
                "co2_concentration": 0.0,
                "sensor_co2_concentration": 0.0,
                "source_co2_mass": 0.0,
                "occupancy_multiplier": 0.0,
            },
            "processing": {
                "co2_mass": 0.0,
                "co2_concentration": 0.0,
                "sensor_co2_concentration": 0.0,
                "source_co2_mass": 0.0,
                "occupancy_multiplier": 1.0,
                "captured_co2": 0.5 * tick,
            },
        },
        connections={
            "cabin_a_to_processing": {
                "requested_airflow": 12.0,
                "delivered_airflow": 10.0,
                "airflow_residual": 2.0,
            },
            "processing_to_cabin_a": {
                "requested_airflow": 12.0,
                "delivered_airflow": 10.0,
                "airflow_residual": 2.0,
            },
        },
        system={
            "shared_airflow_capacity": 18.0,
            "total_requested_airflow": 20.0,
            "total_delivered_airflow": 18.0,
            "capacity_scale": 0.9,
        },
        actuators={
            "cabin_a": {
                "setpoint": 1.0,
                "actual_position": 0.8,
                "tracking_residual": 0.2,
                "moving": 1.0,
                "movement_seconds": float(tick),
                "power": 1.0,
                "direction": 1.0,
            }
        },
    )


def test_trace_output_is_valid_jsonl_with_zone_and_connection_fields(tmp_path):
    records = [_record(tick) for tick in range(1, 6)]
    path = tmp_path / "trace.jsonl"

    with TraceWriter(path) as writer:
        for record in records:
            writer.write(record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)  # one row per tick

    rows = [json.loads(line) for line in lines]  # raises if any line is not valid JSON
    for row in rows:
        assert set(row) == {"tick", "zones", "connections", "actuators", "system"}
        assert row["zones"]["cabin_a"]["co2_mass"] > 0.0
        assert row["zones"]["processing"]["captured_co2"] >= 0.0
        for connection in row["connections"].values():
            assert set(connection) == {
                "requested_airflow",
                "delivered_airflow",
                "airflow_residual",
            }
        assert row["actuators"]["cabin_a"]["actual_position"] == 0.8
        assert row["system"]["capacity_scale"] == 0.9
    assert [row["tick"] for row in rows] == [1, 2, 3, 4, 5]


def test_same_records_twice_produce_byte_identical_traces(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    for path in (first_path, second_path):
        with TraceWriter(path) as writer:
            for tick in range(1, 6):
                writer.write(_record(tick))

    assert first_path.read_bytes() == second_path.read_bytes()


def test_writer_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deep" / "trace.jsonl"

    with TraceWriter(path) as writer:
        writer.write(_record(1))

    assert path.exists()


def _recovery_plant_record(config):
    return TickRecord(
        tick=1,
        zones={
            zone.id: {
                "co2_mass": 0.0,
                "co2_concentration": 0.0,
                "sensor_co2_concentration": 0.0,
                "source_co2_mass": 0.0,
                "occupancy_multiplier": 1.0,
                **({"captured_co2": 0.0} if zone.preset == "air_processing" else {}),
            }
            for zone in config.zones
        },
        connections={
            connection.id: {
                "requested_airflow": 0.0,
                "delivered_airflow": 0.0,
                "airflow_residual": 0.0,
            }
            for connection in config.connections
        },
        actuators={
            zone.id: {
                "setpoint": 0.0,
                "actual_position": 0.0,
                "tracking_residual": 0.0,
                "moving": 0.0,
                "movement_seconds": 0.0,
                "power": 0.0,
                "direction": 0.0,
            }
            for zone in config.non_processing_zones()
        },
        system={
            "shared_airflow_capacity": 24.0,
            "total_requested_airflow": 0.0,
            "total_delivered_airflow": 0.0,
            "capacity_scale": 1.0,
        },
    )


def _reserve_telemetry(config):
    active_ids = {
        config.reserve_path_to_processing("cabin_a").id,
        config.reserve_path_from_processing("cabin_a").id,
    }
    return {
        "connections": {
            connection.id: {
                "requested_airflow": 1.0 if connection.id in active_ids else 0.0,
                "delivered_airflow": 0.9 if connection.id in active_ids else 0.0,
                "airflow_residual": 0.1 if connection.id in active_ids else 0.0,
            }
            for connection in config.reserve_connections
        },
        "actuators": {
            zone.id: {
                "setpoint": 0.25 if zone.id == "cabin_a" else 0.0,
                "actual_position": 0.2 if zone.id == "cabin_a" else 0.0,
                "tracking_residual": 0.05 if zone.id == "cabin_a" else 0.0,
                "moving": zone.id == "cabin_a",
                "movement_seconds": 1.0 if zone.id == "cabin_a" else 0.0,
                "power": 1.0 if zone.id == "cabin_a" else 0.0,
            }
            for zone in config.non_processing_zones()
        },
        "system": {
            "reserve_airflow_capacity": 4.0,
            "total_requested_airflow": 1.0,
            "total_delivered_airflow": 0.9,
            "capacity_scale": 1.0,
            "total_power": 1.0,
        },
    }


def _authority_telemetry():
    return {
        "run_id": "run-1",
        "authority_epoch": 0,
        "decision_tick": 1,
        "sequence": 1,
        "state": "NOMINAL",
        "reserve_command_owner": "reserve_off",
        "target_zone_id": None,
        "reason": "cold_start",
        "dwell_ticks": 1,
        "observation_tick": 0,
        "command_digest": "a" * 64,
        "applied_command_digest": "a" * 64,
    }


def test_recovery_writer_emits_strict_versioned_wrapper(
    tmp_path, recovery_config
):
    output = tmp_path / "recovery.jsonl"
    recovery = RecoveryTickRecord(
        plant=_recovery_plant_record(recovery_config),
        reserve=_reserve_telemetry(recovery_config),
        authority=_authority_telemetry(),
    )

    with RecoveryTraceWriter(output, recovery_config) as writer:
        writer.write(recovery)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert set(row) == {"schema_version", "plant", "reserve", "authority"}
    assert row["schema_version"] == RECOVERY_TRACE_VERSION
    assert row["plant"]["tick"] == 1
    assert set(row["reserve"]) == {"connections", "actuators", "system"}
    assert set(row["authority"]) == set(_authority_telemetry())


def test_recovery_writer_rejects_unknown_or_non_finite_telemetry(
    tmp_path, recovery_config
):
    reserve = _reserve_telemetry(recovery_config)
    reserve["system"]["hidden_health"] = 1.0
    record = RecoveryTickRecord(
        plant=_recovery_plant_record(recovery_config),
        reserve=reserve,
        authority=_authority_telemetry(),
    )
    with RecoveryTraceWriter(tmp_path / "bad-key.jsonl", recovery_config) as writer:
        with pytest.raises(ValueError, match="reserve system"):
            writer.write(record)

    reserve = _reserve_telemetry(recovery_config)
    outbound_id = recovery_config.reserve_path_to_processing("cabin_a").id
    reserve["connections"][outbound_id]["delivered_airflow"] = float("nan")
    record = RecoveryTickRecord(
        plant=_recovery_plant_record(recovery_config),
        reserve=reserve,
        authority=_authority_telemetry(),
    )
    with RecoveryTraceWriter(tmp_path / "non-finite.jsonl", recovery_config) as writer:
        with pytest.raises(ValueError, match="finite"):
            writer.write(record)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda reserve, authority, config: reserve["connections"].update(
                {"invented_path": reserve["connections"].pop(next(iter(reserve["connections"])))}
            ),
            "topology",
        ),
        (
            lambda reserve, authority, config: reserve["system"].update(
                total_requested_airflow=2.0
            ),
            "total requested",
        ),
        (
            lambda reserve, authority, config: authority.update(reason="free_form"),
            "reason",
        ),
        (
            lambda reserve, authority, config: authority.update(
                target_zone_id="invented_zone"
            ),
            "target",
        ),
    ],
)
def test_recovery_writer_rejects_topology_authority_and_total_mismatches(
    tmp_path, recovery_config, mutate, match
):
    reserve = _reserve_telemetry(recovery_config)
    authority = _authority_telemetry()
    mutate(reserve, authority, recovery_config)
    record = RecoveryTickRecord(
        plant=_recovery_plant_record(recovery_config),
        reserve=reserve,
        authority=authority,
    )

    with RecoveryTraceWriter(tmp_path / "mismatch.jsonl", recovery_config) as writer:
        with pytest.raises(ValueError, match=match):
            writer.write(record)
