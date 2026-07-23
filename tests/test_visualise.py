"""Tests for the standalone trace visualiser."""

import json

from icarus.trace import TickRecord, TraceWriter
from icarus.visualise import load_trace, main, write_visualisation


def _write_trace(path) -> None:
    with TraceWriter(path) as writer:
        for tick in range(1, 4):
            writer.write(
                TickRecord(
                    tick=tick,
                    zones={
                        "cabin": {
                            "co2_mass": 5.0 * tick,
                            "co2_concentration": 0.05 * tick,
                            "sensor_co2_concentration": 0.051 * tick,
                            "source_co2_mass": 1.0,
                            "occupancy_multiplier": 1.0,
                        },
                        "processing": {
                            "co2_mass": 0.0,
                            "co2_concentration": 0.0,
                            "sensor_co2_concentration": 0.0,
                            "source_co2_mass": 0.0,
                            "occupancy_multiplier": 1.0,
                            "captured_co2": 1.5 * tick,
                        },
                    },
                    connections={
                        "cabin_to_processing": {
                            "requested_airflow": 12.0,
                            "airflow": 10.0,
                            "health": 1.0,
                        }
                    },
                    actuators={
                        "cabin": {
                            "setpoint": 1.0,
                            "actual_position": 0.1 * tick,
                            "tracking_residual": 1.0 - 0.1 * tick,
                            "moving": 1.0,
                            "movement_seconds": float(tick),
                            "power": 1.0,
                            "direction": 1.0,
                        }
                    },
                    system={
                        "shared_airflow_capacity": 10.0,
                        "total_requested_airflow": 12.0,
                        "total_actual_airflow": 10.0,
                        "capacity_scale": 10.0 / 12.0,
                    },
                )
            )


def test_visualiser_writes_self_contained_html(tmp_path):
    trace = tmp_path / "trace.jsonl"
    report = tmp_path / "nested" / "report.html"
    _write_trace(trace)

    result = write_visualisation(trace, report)

    html = report.read_text(encoding="utf-8")
    assert result == report
    assert "ICARUS Trace Visualiser" in html
    assert "CO₂ concentration" in html
    assert "Requested and allocated airflow" in html
    assert "Actuator setpoint and actual position" in html
    assert "__TRACE_DATA__" not in html
    assert "https://" not in html


def test_load_trace_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    try:
        load_trace(path)
    except ValueError as exc:
        assert str(exc) == "trace contains no records"
    else:
        raise AssertionError("empty trace should be rejected")


def test_load_trace_reports_malformed_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    valid = {
        "tick": 1,
        "zones": {
            "cabin": {
                "co2_mass": 1.0,
                "co2_concentration": 0.01,
                "sensor_co2_concentration": 0.011,
                "source_co2_mass": 1.0,
                "occupancy_multiplier": 1.0,
            }
        },
        "connections": {
            "duct": {
                "requested_airflow": 1.2,
                "airflow": 1.0,
                "health": 1.0,
            }
        },
        "actuators": {
            "cabin": {
                "setpoint": 1.0,
                "actual_position": 0.5,
                "tracking_residual": 0.5,
                "moving": 1.0,
                "movement_seconds": 1.0,
                "power": 1.0,
                "direction": 1.0,
            }
        },
        "system": {
            "shared_airflow_capacity": 1.0,
            "total_requested_airflow": 1.2,
            "total_actual_airflow": 1.0,
            "capacity_scale": 1.0 / 1.2,
        },
    }
    path.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")

    try:
        load_trace(path)
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("malformed trace should be rejected")


def test_visualiser_cli_writes_report(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    report = tmp_path / "report.html"
    _write_trace(trace)

    assert main([str(trace), str(report)]) == 0

    assert report.exists()
    assert "visualised trace=" in capsys.readouterr().out


def test_visualiser_cli_rejects_missing_trace(tmp_path, capsys):
    report = tmp_path / "report.html"

    assert main([str(tmp_path / "missing.jsonl"), str(report)]) == 2

    assert "trace file not found" in capsys.readouterr().err
    assert not report.exists()


def test_load_trace_rejects_non_integer_tick(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["tick"] = 1.5
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    try:
        load_trace(trace)
    except ValueError as exc:
        assert "tick must be a positive integer" in str(exc)
    else:
        raise AssertionError("fractional tick should be rejected")


def test_load_trace_rejects_non_consecutive_ticks(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[1]["tick"] = 3
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    try:
        load_trace(trace)
    except ValueError as exc:
        assert "expected tick 2" in str(exc)
    else:
        raise AssertionError("non-consecutive tick should be rejected")


def test_load_trace_rejects_schema_drift(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    del rows[1]["zones"]["cabin"]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    try:
        load_trace(trace)
    except ValueError as exc:
        assert "zones do not match" in str(exc)
    else:
        raise AssertionError("schema drift should be rejected")


def test_load_trace_rejects_negative_airflow(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["connections"]["cabin_to_processing"]["airflow"] = -0.1
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    try:
        load_trace(trace)
    except ValueError as exc:
        assert "airflow must not be negative" in str(exc)
    else:
        raise AssertionError("negative airflow should be rejected")
