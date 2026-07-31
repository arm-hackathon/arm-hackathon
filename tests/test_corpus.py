"""Leakage-safe labelled corpus generation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.config import parse_scenario
from aeolus.corpus import (
    DEFAULT_STRIDE_TICKS,
    DEFAULT_WINDOW_TICKS,
    build_corpus_rows,
    generate_corpus,
    label_for_window,
    main,
)
from aeolus.scenario import run_scenario
from aeolus.trace import model_feature_row

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "scenarios"
ALL_SCENARIO_PATHS = (
    SCENARIOS / "standard_habitat.json",
    SCENARIOS / "high_demand_healthy.json",
    SCENARIOS / "primary_fan_degradation.json",
    SCENARIOS / "blocked_path.json",
    SCENARIOS / "frozen_sensor.json",
)
FORBIDDEN_FEATURE_NAMES = {
    "health",
    "effectiveness",
    "blocked_effectiveness",
    "end_effectiveness",
    "random_seed",
    "source_noise",
    "seed",
    "frozen",
}


def _degradation_config(standard_doc):
    standard_doc["version"] = 7
    standard_doc["fault_profiles"] = [
        {
            "type": "gradual_primary_fan_degradation",
            "connection_id": "cabin_a_to_processing",
            "start_tick": 20,
            "end_tick": 80,
            "end_effectiveness": 0.4,
        }
    ]
    return parse_scenario(standard_doc)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_label_for_window_marks_fault_active_at_window_end(standard_doc):
    config = _degradation_config(standard_doc)

    assert label_for_window(config, 20) == "nominal"
    assert label_for_window(config, 21) == "gradual_primary_fan_degradation"
    assert label_for_window(config, 120) == "gradual_primary_fan_degradation"


def test_label_for_window_marks_blocked_and_frozen_from_start_tick(standard_doc):
    standard_doc["version"] = 7
    standard_doc["fault_profiles"] = [
        {
            "type": "blocked_path",
            "connection_id": "cabin_b_to_processing",
            "start_tick": 30,
            "blocked_effectiveness": 0.05,
        }
    ]
    blocked_config = parse_scenario(standard_doc)

    assert label_for_window(blocked_config, 29) == "nominal"
    assert label_for_window(blocked_config, 30) == "blocked_path"


def test_label_for_window_rejects_multiple_active_faults(standard_doc):
    standard_doc["version"] = 7
    standard_doc["fault_profiles"] = [
        {
            "type": "gradual_primary_fan_degradation",
            "connection_id": "cabin_a_to_processing",
            "start_tick": 20,
            "end_tick": 80,
            "end_effectiveness": 0.4,
        },
        {
            "type": "blocked_path",
            "connection_id": "cabin_b_to_processing",
            "start_tick": 30,
            "blocked_effectiveness": 0.05,
        },
    ]
    config = parse_scenario(standard_doc)

    assert label_for_window(config, 25) == "gradual_primary_fan_degradation"
    with pytest.raises(ValueError, match="more than one fault"):
        label_for_window(config, 40)


def test_build_corpus_windows_cover_records_with_stride(standard_doc):
    config = _degradation_config(standard_doc)
    records = run_scenario(config)

    rows = build_corpus_rows(config, records, scenario_name="degradation")

    expected_windows = (len(records) - DEFAULT_WINDOW_TICKS) // DEFAULT_STRIDE_TICKS + 1
    assert len(rows) == expected_windows == 23
    first, last = rows[0], rows[-1]
    assert (first["start_tick"], first["end_tick"]) == (1, 10)
    assert (last["start_tick"], last["end_tick"]) == (111, 120)
    assert [row["window_index"] for row in rows] == list(range(23))
    for row in rows:
        assert len(row["features"]) == DEFAULT_WINDOW_TICKS
        assert row["scenario"] == "degradation"


def test_corpus_features_are_exactly_the_model_feature_projection(standard_doc):
    config = _degradation_config(standard_doc)
    records = run_scenario(config)

    rows = build_corpus_rows(config, records, scenario_name="degradation")

    for row in rows:
        start = row["start_tick"] - 1
        expected = [model_feature_row(record) for record in records[start : start + 10]]
        assert row["features"] == expected
        key_set = set(_walk_keys(row["features"]))
        assert key_set.isdisjoint(FORBIDDEN_FEATURE_NAMES)
        for feature in row["features"]:
            assert set(feature) == {"zones", "actuators", "connections"}
            for zone_values in feature["zones"].values():
                assert set(zone_values) == {"sensor_co2_concentration"}
            for actuator_values in feature["actuators"].values():
                assert set(actuator_values) == {
                    "setpoint",
                    "actual_position",
                    "tracking_residual",
                    "power",
                }
            for connection_values in feature["connections"].values():
                assert set(connection_values) == {
                    "requested_airflow",
                    "delivered_airflow",
                    "airflow_residual",
                }


def test_generate_corpus_is_byte_identical_across_runs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_corpus(ALL_SCENARIO_PATHS, first)
    generate_corpus(tuple(reversed(ALL_SCENARIO_PATHS)), second)

    assert (first / "corpus.jsonl").read_bytes() == (
        second / "corpus.jsonl"
    ).read_bytes()
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()


def test_generate_corpus_labels_every_planned_fault_class(tmp_path):
    manifest = generate_corpus(ALL_SCENARIO_PATHS, tmp_path)

    assert manifest["corpus_version"] == 1
    assert manifest["window_ticks"] == DEFAULT_WINDOW_TICKS
    assert manifest["stride_ticks"] == DEFAULT_STRIDE_TICKS
    assert manifest["total_windows"] == 115
    assert manifest["label_counts"] == {
        "nominal": 57,
        "gradual_primary_fan_degradation": 20,
        "blocked_path": 19,
        "frozen_sensor": 19,
    }
    assert [entry["name"] for entry in manifest["scenarios"]] == [
        "blocked_path",
        "frozen_sensor",
        "high_demand_healthy",
        "primary_fan_degradation",
        "standard_habitat",
    ]
    for entry in manifest["scenarios"]:
        assert entry["ticks"] == 120
        assert entry["windows"] == 23
        assert len(entry["sha256"]) == 64


def test_corpus_cli_writes_corpus_and_manifest(tmp_path):
    out_dir = tmp_path / "corpus"
    exit_code = main([str(out_dir), *(str(path) for path in ALL_SCENARIO_PATHS)])

    assert exit_code == 0
    corpus_lines = (out_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus_lines) == 115
    first_row = json.loads(corpus_lines[0])
    assert set(first_row) == {
        "scenario",
        "window_index",
        "start_tick",
        "end_tick",
        "label",
        "features",
    }
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_windows"] == 115


def test_corpus_cli_rejects_missing_arguments():
    assert main([]) == 2
    assert main(["only-out-dir"]) == 2
