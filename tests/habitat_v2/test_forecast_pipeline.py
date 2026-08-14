from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_generates_and_replays_atomic_development_fixture(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pipeline import ForecastPipelineError, generate_development_fixture, validate_development_packet

    root = Path(__file__).resolve().parents[2]
    first = generate_development_fixture(root, tmp_path, "packet-a")
    second = generate_development_fixture(root, tmp_path, "packet-b")
    bundle = load_forecast_contracts(root)
    sample = json.loads((tmp_path / "packet-a" / "development-fixture-only" / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert first["release_tier"] == "DEVELOPMENT_FIXTURE_ONLY"
    assert first["sample_count"] == 4
    assert first["shadow_receipt_matches"] == 96
    assert first["file_sha256"] == second["file_sha256"]
    assert validate_development_packet(tmp_path / "packet-a", bundle)["strict_trace_replays"] == 4
    assert set(sample["input_tensors"]) == {"history_numeric", "history_availability", "history_mode_one_hot", "history_health_one_hot", "history_alarm_lifecycle_one_hot", "history_final_command", "proposed_action"}
    assert len(sample["input_tensors"]["history_numeric"]) == 4
    assert len(sample["input_tensors"]["history_numeric"][0]) == 194
    assert len(sample["target_truth"]) == 2
    assert len(sample["target_truth"][0]) == 51
    with pytest.raises(ForecastPipelineError, match="destination"):
        generate_development_fixture(root, tmp_path, "packet-a")


def test_generation_rejects_unsafe_destination_name(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pipeline import ForecastPipelineError, generate_development_fixture

    with pytest.raises(ForecastPipelineError, match="destination"):
        generate_development_fixture(Path(__file__).resolve().parents[2], tmp_path, "../escape")
