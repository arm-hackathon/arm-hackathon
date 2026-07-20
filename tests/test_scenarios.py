"""Tests for the named scenarios and their deterministic replay."""

import json

from icarus.__main__ import main
from icarus.scenario import SCENARIOS, run_scenario


def test_degraded_scenario_has_higher_room_a_co2_than_nominal_after_fault_window():
    spec = SCENARIOS["primary_fan_degradation"]
    nominal = run_scenario("nominal")
    degraded = run_scenario("primary_fan_degradation")

    nominal_late = [r.room_a_co2 for r in nominal if r.tick > spec.fault_tick]
    degraded_late = [r.room_a_co2 for r in degraded if r.tick > spec.fault_tick]

    assert len(nominal_late) == len(degraded_late) > 0
    assert degraded_late[-1] > nominal_late[-1]
    assert sum(degraded_late) / len(degraded_late) > sum(nominal_late) / len(nominal_late)


def test_fault_applies_at_declared_tick_and_airflow_drops_below_nominal():
    spec = SCENARIOS["primary_fan_degradation"]
    nominal_by_tick = {r.tick: r for r in run_scenario("nominal")}
    degraded = run_scenario("primary_fan_degradation")

    for record in degraded:
        if record.tick < spec.fault_tick:
            assert record.main_fan_effectiveness == 1.0
        else:
            assert record.main_fan_effectiveness == spec.degraded_main_fan_effectiveness
            assert record.airflow < nominal_by_tick[record.tick].airflow


def test_same_named_scenario_twice_produces_identical_records():
    first = run_scenario("primary_fan_degradation")
    second = run_scenario("primary_fan_degradation")

    assert first == second


def test_nominal_stays_below_declared_ceiling_after_warmup():
    spec = SCENARIOS["nominal"]
    late = [r.room_a_co2 for r in run_scenario("nominal") if r.tick > spec.warmup_ticks]

    assert late, "scenario should leave ticks to check after the warm-up window"
    assert max(late) < spec.nominal_co2_ceiling


def test_scenario_trace_file_has_one_jsonl_row_per_tick(tmp_path):
    spec = SCENARIOS["primary_fan_degradation"]
    path = tmp_path / "degradation.jsonl"

    records = run_scenario("primary_fan_degradation", trace_path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == spec.total_ticks == len(records)
    required_fields = {"tick", "room_a_co2", "room_b_co2", "main_fan_effectiveness", "airflow"}
    for line in lines:
        assert required_fields <= json.loads(line).keys()


def test_main_entrypoint_writes_trace_and_prints_summary(tmp_path, capsys):
    path = tmp_path / "out.jsonl"

    exit_code = main(["primary_fan_degradation", str(path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "primary_fan_degradation" in out
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == SCENARIOS["primary_fan_degradation"].total_ticks


def test_main_entrypoint_rejects_unknown_scenario(capsys):
    assert main(["no_such_scenario", "ignored.jsonl"]) == 2
    assert "no_such_scenario" in capsys.readouterr().err
