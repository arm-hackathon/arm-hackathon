"""Tests for the JSONL trace writer."""

import json

from icarus.trace import TickRecord, TraceWriter


def test_trace_output_is_valid_jsonl_with_required_fields(tmp_path):
    records = [
        TickRecord(
            tick=tick,
            room_a_co2=10.0 + tick,
            room_b_co2=0.5 * tick,
            main_fan_effectiveness=1.0,
            airflow=10.0,
        )
        for tick in range(1, 6)
    ]
    path = tmp_path / "trace.jsonl"

    with TraceWriter(path) as writer:
        for record in records:
            writer.write(record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)  # one row per tick

    required_fields = {"tick", "room_a_co2", "room_b_co2", "main_fan_effectiveness", "airflow"}
    rows = [json.loads(line) for line in lines]  # raises if any line is not valid JSON
    for row in rows:
        assert required_fields <= row.keys()
    assert [row["tick"] for row in rows] == [1, 2, 3, 4, 5]
