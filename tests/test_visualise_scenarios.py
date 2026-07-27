"""Scenario-v7 trace acceptance and malformed-telemetry rejection for the visualiser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icarus.config import load_scenario
from icarus.scenario import run_scenario
from icarus.visualise import load_trace, write_visualisation


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_NAMES = (
    "standard_habitat.json",
    "high_demand_healthy.json",
    "primary_fan_degradation.json",
    "blocked_path.json",
    "frozen_sensor.json",
)


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_visualiser_accepts_each_v7_scenario_trace(scenario_name, tmp_path):
    trace_path = tmp_path / f"{scenario_name}.jsonl"
    report_path = tmp_path / f"{scenario_name}.html"
    scenario_path = REPO_ROOT / "scenarios" / scenario_name

    run_scenario(load_scenario(scenario_path), trace_path=trace_path)
    rows = load_trace(trace_path)
    result = write_visualisation(trace_path, report_path)

    assert len(rows) == 120
    assert result == report_path
    assert "Airflow residual" in report_path.read_text(encoding="utf-8")


def test_visualiser_rejects_undeclared_connection_field(tmp_path):
    scenario_path = REPO_ROOT / "scenarios" / "primary_fan_degradation.json"
    trace_path = tmp_path / "trace.jsonl"
    run_scenario(load_scenario(scenario_path), trace_path=trace_path)
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["connections"]["cabin_a_to_processing"]["hidden_effectiveness"] = 0.4
    trace_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected field"):
        load_trace(trace_path)
