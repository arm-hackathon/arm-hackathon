"""The model-feature projection must exclude hidden scenario truth."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from icarus.config import load_scenario
from icarus.scenario import run_scenario
from icarus.trace import TraceWriter, model_feature_row


REPO_ROOT = Path(__file__).resolve().parents[1]
DEGRADATION_PATH = REPO_ROOT / "scenarios" / "primary_fan_degradation.json"


def test_model_feature_projection_has_only_allowlisted_observable_signals(tmp_path):
    trace_path = tmp_path / "degradation.jsonl"
    records = run_scenario(load_scenario(DEGRADATION_PATH), trace_path=trace_path)

    features = model_feature_row(records[-1])
    assert set(features) == {"zones", "actuators", "connections"}
    assert all(set(values) == {"sensor_co2_concentration"} for values in features["zones"].values())
    assert all(
        set(values) == {"setpoint", "actual_position", "tracking_residual", "power"}
        for values in features["actuators"].values()
    )
    assert all(
        set(values) == {"requested_airflow", "delivered_airflow", "airflow_residual"}
        for values in features["connections"].values()
    )

    feature_text = json.dumps(features, sort_keys=True)
    for hidden_name in (
        "fault",
        "effectiveness",
        "health",
        "random_seed",
        "source_noise",
        "occupancy_multiplier",
        "source_co2_mass",
    ):
        assert hidden_name not in feature_text

    trace_text = trace_path.read_text(encoding="utf-8")
    for hidden_name in ("fault", "effectiveness", "health", "random_seed", "source_noise"):
        assert hidden_name not in trace_text


def test_trace_writer_rejects_hidden_or_undeclared_connection_telemetry(tmp_path):
    record = copy.deepcopy(run_scenario(load_scenario(DEGRADATION_PATH))[0])
    record.connections["cabin_a_to_processing"]["health"] = 0.4

    with TraceWriter(tmp_path / "trace.jsonl") as writer:
        with pytest.raises(ValueError, match="unexpected telemetry"):
            writer.write(record)
