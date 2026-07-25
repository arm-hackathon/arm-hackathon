"""Tests for the JSONL trace writer."""

import json
from pathlib import Path

import pytest

from icarus.scenario import STANDARD_RUN
from icarus.trace import TickRecord, TraceWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_TRACE_PATH = REPO_ROOT / "traces" / "standard_habitat.jsonl"
DEGRADATION_TRACE_PATH = REPO_ROOT / "traces" / "primary_fan_degradation.jsonl"


def _record(tick: int) -> TickRecord:
    return TickRecord(
        tick=tick,
        zones={
            "cabin_a": {"co2": 10.0 + tick},
            "cabin_b": {"co2": 9.0 + tick},
            "lab": {"co2": 0.0},
            "processing": {"co2": 0.0, "captured_co2": 0.5 * tick},
        },
        connections={
            "cabin_a_to_processing": {"airflow": 10.0},
            "processing_to_cabin_a": {"airflow": 10.0},
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
        assert set(row) == {"tick", "zones", "connections"}
        assert row["zones"]["cabin_a"]["co2"] > 0.0
        assert row["zones"]["processing"]["captured_co2"] >= 0.0
        for connection in row["connections"].values():
            assert set(connection) == {"airflow"}
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


def test_writer_rejects_hidden_connection_health(tmp_path):
    record = _record(1)
    record.connections["cabin_a_to_processing"]["health"] = 0.5

    with TraceWriter(tmp_path / "trace.jsonl") as writer:
        with pytest.raises(ValueError, match="only observable airflow"):
            writer.write(record)


def _assert_checked_in_trace_clean(trace_path: Path) -> None:
    assert trace_path.exists(), (f'missing checked-in trace: {trace_path}')
    rows = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]

    assert len(rows) == STANDARD_RUN.total_ticks
    assert [row['tick'] for row in rows] == list(range(1, STANDARD_RUN.total_ticks + 1))

    for row in rows:
        assert set(row) == {'tick', 'zones', 'connections'}
        for telemetry in row['connections'].values():
            assert set(telemetry) == {'airflow'}, (f'checked-in trace leaks non-airflow connection telemetry: {trace_path.name}')
        serialised = json.dumps(row, sort_keys=True)
        assert 'health' not in serialised
        assert 'fault' not in serialised
        assert 'effectiveness' not in serialised


@pytest.mark.parametrize('trace_path', [STANDARD_TRACE_PATH, DEGRADATION_TRACE_PATH], ids=['standard', 'degradation'])
def test_checked_in_trace_exposes_only_observable_telemetry(trace_path: Path):
    """Regression guard for the disallowed 'health' field once shipped in the trace."""
    _assert_checked_in_trace_clean(trace_path)
