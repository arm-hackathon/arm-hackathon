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
            assert "airflow" in entry
            assert "health" in entry


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
