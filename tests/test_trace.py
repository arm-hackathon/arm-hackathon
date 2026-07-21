"""Tests for the JSONL trace writer."""

import json

from icarus.trace import TickRecord, TraceWriter


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
            "cabin_a_to_processing": {"airflow": 10.0, "health": 1.0},
            "processing_to_cabin_a": {"airflow": 10.0, "health": 1.0},
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
            assert set(connection) == {"airflow", "health"}
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
