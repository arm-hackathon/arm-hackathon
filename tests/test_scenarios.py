"""Tests for running the standard habitat scenario graph."""

import json

from icarus.__main__ import main
from icarus.config import load_scenario
from icarus.scenario import STANDARD_RUN, run_scenario


def test_run_produces_one_record_per_tick_in_tick_order(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    records = run_scenario(config)

    assert len(records) == STANDARD_RUN.total_ticks
    assert [r.tick for r in records] == list(range(1, STANDARD_RUN.total_ticks + 1))


def test_every_record_covers_every_zone_and_connection(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    zone_ids = {z.id for z in config.zones}
    connection_ids = {c.id for c in config.connections}

    for record in run_scenario(config):
        assert set(record.zones) == zone_ids
        for zone_id in zone_ids:
            assert "co2" in record.zones[zone_id]
        assert "captured_co2" in record.zones[config.processing_zone().id]
        assert set(record.connections) == connection_ids
        for entry in record.connections.values():
            assert set(entry) == {"airflow"}


def test_same_scenario_twice_produces_identical_records(standard_scenario_path):
    first = run_scenario(load_scenario(standard_scenario_path))
    second = run_scenario(load_scenario(standard_scenario_path))

    assert first == second


def test_same_scenario_file_twice_produces_byte_identical_traces(
    standard_scenario_path, tmp_path
):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    run_scenario(load_scenario(standard_scenario_path), trace_path=first_path)
    run_scenario(load_scenario(standard_scenario_path), trace_path=second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_nominal_scenario_keeps_primary_fan_airflow_constant(
    standard_scenario_path,
):
    records = run_scenario(load_scenario(standard_scenario_path))

    outbound = {
        record.connections["cabin_a_to_processing"]["airflow"]
        for record in records
    }
    inbound = {
        record.connections["processing_to_cabin_a"]["airflow"]
        for record in records
    }
    assert outbound == {10.0}
    assert inbound == {10.0}


def test_gradual_primary_fan_degradation_is_deterministic(
    degradation_scenario_path, tmp_path
):
    config = load_scenario(degradation_scenario_path)
    first_path = tmp_path / "degraded-first.jsonl"
    second_path = tmp_path / "degraded-second.jsonl"

    first = run_scenario(config, trace_path=first_path)
    second = run_scenario(
        load_scenario(degradation_scenario_path), trace_path=second_path
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    airflow_by_tick = {
        record.tick: record.connections["cabin_a_to_processing"]["airflow"]
        for record in first
    }
    assert airflow_by_tick[19] == 10.0
    assert airflow_by_tick[20] == 10.0
    assert airflow_by_tick[50] == 7.0
    assert airflow_by_tick[80] == 4.0
    assert airflow_by_tick[120] == 4.0
    assert [airflow_by_tick[tick] for tick in range(20, 81)] == sorted(
        (airflow_by_tick[tick] for tick in range(20, 81)), reverse=True
    )
    for record in first:
        assert (
            record.connections["processing_to_cabin_a"]["airflow"]
            == record.connections["cabin_a_to_processing"]["airflow"]
        )


def test_degradation_trace_exposes_only_observable_consequences(
    standard_scenario_path, degradation_scenario_path, tmp_path
):
    nominal = run_scenario(load_scenario(standard_scenario_path))
    trace_path = tmp_path / "degraded.jsonl"
    degraded = run_scenario(
        load_scenario(degradation_scenario_path), trace_path=trace_path
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    tick_80 = rows[79]
    assert tick_80["connections"]["cabin_a_to_processing"] == {"airflow": 4.0}
    assert tick_80["connections"]["processing_to_cabin_a"] == {"airflow": 4.0}
    assert (
        degraded[79].zones["cabin_a"]["co2"]
        > nominal[79].zones["cabin_a"]["co2"]
    )
    for row in rows:
        for connection in row["connections"].values():
            assert set(connection) == {"airflow"}
        serialised = json.dumps(row, sort_keys=True)
        assert "health" not in serialised
        assert "fault" not in serialised
        assert "effectiveness" not in serialised


def test_healthy_standard_habitat_keeps_crew_cabins_below_ceiling_after_warmup(
    standard_scenario_path,
):
    config = load_scenario(standard_scenario_path)
    records = run_scenario(config)

    late = [r for r in records if r.tick > STANDARD_RUN.warmup_ticks]
    assert late, "scenario should leave ticks to check after the warm-up window"
    crew_cabins = [z for z in config.non_processing_zones() if z.preset == "crew_cabin"]
    assert {z.id for z in crew_cabins} == {"cabin_a", "cabin_b"}
    for zone in crew_cabins:
        peak = max(r.zones[zone.id]["co2"] for r in late)
        assert peak < STANDARD_RUN.crew_cabin_co2_ceiling


def test_run_writes_jsonl_trace_with_one_row_per_tick(standard_scenario_path, tmp_path):
    config = load_scenario(standard_scenario_path)
    path = tmp_path / "standard.jsonl"

    records = run_scenario(config, trace_path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == STANDARD_RUN.total_ticks == len(records)
    row = json.loads(lines[0])
    assert set(row) == {"tick", "zones", "connections"}
    assert set(row["zones"]) == {z.id for z in config.zones}
    assert set(row["connections"]) == {c.id for c in config.connections}


def test_main_entrypoint_runs_explicit_scenario_file_and_writes_trace(
    standard_scenario_path, tmp_path, capsys
):
    path = tmp_path / "out.jsonl"

    exit_code = main([str(standard_scenario_path), str(path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert str(standard_scenario_path) in out
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == STANDARD_RUN.total_ticks
    row = json.loads(lines[0])
    assert set(row["zones"]) == {"cabin_a", "cabin_b", "lab", "processing"}


def test_main_entrypoint_rejects_invalid_scenario_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"version": 99, "zones": [], "connections": []}', encoding="utf-8"
    )

    assert main([str(bad), str(tmp_path / "out.jsonl")]) == 2
    assert "version" in capsys.readouterr().err


def test_main_entrypoint_rejects_missing_scenario_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json"), str(tmp_path / "out.jsonl")]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_entrypoint_rejects_wrong_argument_count(capsys):
    assert main([]) == 2
    assert "Usage" in capsys.readouterr().err
