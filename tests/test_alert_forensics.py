"""Stateful historical healthy-alert forensic reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeolus.alert_forensics import (
    FORENSIC_EVIDENCE_ROLE,
    build_alert_forensics_report,
    canonical_alert_forensics_sha256,
    summarize_forensic_window,
    validate_alert_forensics_report,
)
from aeolus.config import load_scenario
from aeolus.scenario import run_scenario

_SHA = "a" * 64
_COMMIT = "b" * 40
_STANDARD_SCENARIO = Path("scenarios/standard_habitat.json")


def _row(
    end_tick: int,
    prediction_context: float,
    *,
    stream_id: str = "validation-s1300-profile-reference",
    family_id: str = "validation-s1300-profile-t20-cabin_a-frozen",
) -> dict[str, object]:
    return {
        "family_id": family_id,
        "scenario_role": "reference",
        "stream_id": stream_id,
        "operating_profile_id": "v5-profile",
        "start_tick": end_tick - 9,
        "end_tick": end_tick,
        "context": {
            "sensor_slope": prediction_context,
            "system_capacity_scale": 1.0,
        },
    }


def _report(rows: list[dict[str, object]], predictions: list[str]) -> dict[str, object]:
    return build_alert_forensics_report(
        rows,
        predictions,
        source_commit=_COMMIT,
        source_manifest_sha256=_SHA,
        family_manifest_sha256="c" * 64,
        method_name="temporal_cnn_balanced_gated",
        method_sha256="d" * 64,
    )


def test_alert_forensics_groups_contiguous_healthy_alert_windows_into_one_episode():
    report = _report(
        [_row(10, 0.0), _row(11, 0.1), _row(12, 0.2), _row(13, 0.3), _row(14, 0.4)],
        ["nominal", "frozen_sensor", "blocked_path", "nominal", "frozen_sensor"],
    )

    assert report["evidence_role"] == FORENSIC_EVIDENCE_ROLE
    assert report["input_row_count"] == 5
    assert report["healthy_alert_episode_count"] == 2
    assert report["episodes"] == [
        {
            "end_context": {"sensor_slope": 0.2, "system_capacity_scale": 1.0},
            "end_tick": 12,
            "family_id": "validation-s1300-profile-t20-cabin_a-frozen",
            "operating_profile_id": "v5-profile",
            "predicted_labels": ["blocked_path", "frozen_sensor"],
            "start_context": {"sensor_slope": 0.1, "system_capacity_scale": 1.0},
            "start_tick": 11,
            "stream_id": "validation-s1300-profile-reference",
            "window_count": 2,
        },
        {
            "end_context": {"sensor_slope": 0.4, "system_capacity_scale": 1.0},
            "end_tick": 14,
            "family_id": "validation-s1300-profile-t20-cabin_a-frozen",
            "operating_profile_id": "v5-profile",
            "predicted_labels": ["frozen_sensor"],
            "start_context": {"sensor_slope": 0.4, "system_capacity_scale": 1.0},
            "start_tick": 14,
            "stream_id": "validation-s1300-profile-reference",
            "window_count": 1,
        },
    ]


def test_alert_forensics_resets_episode_state_at_stream_boundary():
    report = _report(
        [_row(10, 0.0), _row(11, 0.1), _row(10, 0.2, stream_id="validation-s1301-profile-reference")],
        ["frozen_sensor", "frozen_sensor", "frozen_sensor"],
    )

    assert [(episode["stream_id"], episode["window_count"]) for episode in report["episodes"]] == [
        ("validation-s1300-profile-reference", 2),
        ("validation-s1301-profile-reference", 1),
    ]


def test_alert_forensics_rejects_non_reference_rows_and_forged_method_receipts():
    non_reference = _row(10, 0.0)
    non_reference["scenario_role"] = "fault"
    with pytest.raises(ValueError, match="reference"):
        _report([non_reference], ["nominal"])

    with pytest.raises(ValueError, match="method SHA-256"):
        build_alert_forensics_report(
            [_row(10, 0.0)],
            ["nominal"],
            source_commit=_COMMIT,
            source_manifest_sha256=_SHA,
            family_manifest_sha256="c" * 64,
            method_name="rules",
            method_sha256="not-a-digest",
        )


def test_alert_forensics_context_is_bound_by_canonical_report_digest():
    first = _report([_row(10, 0.1)], ["frozen_sensor"])
    second = _report([_row(10, 0.2)], ["frozen_sensor"])

    validate_alert_forensics_report(first)
    assert canonical_alert_forensics_sha256(first) != canonical_alert_forensics_sha256(second)


def test_forensic_context_uses_observable_telemetry_not_hidden_plant_truth():
    records = run_scenario(load_scenario(_STANDARD_SCENARIO))

    context = summarize_forensic_window(records[:10])

    assert set(context) == {"actuators", "connections", "system", "zones"}
    assert set(context["zones"]["cabin_a"]) == {
        "occupancy_delta",
        "occupancy_end",
        "occupancy_start",
        "sensor_delta",
        "sensor_end",
        "sensor_range",
        "sensor_start",
        "sensor_tail_delta_3",
        "sensor_tail_range_3",
    }
    assert "co2_mass" not in str(context)
    assert "source_co2_mass" not in str(context)
    assert context["connections"]
    assert all(
        set(connection) == {
            "delivered_end",
            "requested_end",
            "residual_end",
            "residual_ratio_peak",
        }
        for connection in context["connections"].values()
    )
    assert context["system"]["capacity_scale_min"] <= context["system"]["capacity_scale_end"]
