from __future__ import annotations

import json
from pathlib import Path

from aeolus.habitat_v2.__main__ import main
from aeolus.habitat_v2.config import load_scenario_file
from aeolus.habitat_v2.runner import run_scenario
from aeolus.habitat_v2.trace import validate_trace_bytes

from ._helpers import reference_scenario_mapping


REFERENCE_SCENARIO_PATH = (
    Path(__file__).resolve().parents[2] / "scenarios" / "habitat_v2_reference.json"
)


def test_checked_in_reference_scenario_loads_and_runs() -> None:
    scenario = load_scenario_file(REFERENCE_SCENARIO_PATH)
    run = run_scenario(scenario)

    assert scenario.data["name"] == "two-zone-reference"
    assert run.final_state.step == scenario.data["steps"]
    assert validate_trace_bytes(run.trace_bytes, scenario=scenario) == run.rows


def test_cli_writes_valid_trace_and_refuses_overwrite(tmp_path, capsys) -> None:
    scenario_path = tmp_path / "scenario.json"
    trace_path = tmp_path / "trace.jsonl"
    scenario_path.write_text(json.dumps(reference_scenario_mapping()), encoding="utf-8")

    scenario = load_scenario_file(scenario_path)
    assert main([str(scenario_path), str(trace_path)]) == 0
    original_trace = trace_path.read_bytes()
    rows = validate_trace_bytes(original_trace, scenario=scenario)
    assert len(rows) == reference_scenario_mapping()["steps"] + 1
    output = capsys.readouterr().out
    assert "run_id=" in output
    assert "crew_cabin co2_ppm=" in output
    assert "battery_energy_wh=" in output

    assert main([str(scenario_path), str(trace_path)]) == 2
    assert trace_path.read_bytes() == original_trace
