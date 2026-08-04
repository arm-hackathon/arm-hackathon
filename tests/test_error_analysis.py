"""Strict historical-forensic error-report contracts."""

from __future__ import annotations

import json

import pytest

from aeolus.error_analysis import (
    build_error_report,
    reject_forensic_report_input,
    write_error_report,
)
from aeolus.families import FamilyEvidence


def test_build_error_report_groups_scored_errors_by_forensic_identity():
    family_id = "validation-s500-primary-high-wind-t55-cabin_a-blocked-e650"
    family_evidence = {
        family_id: FamilyEvidence(
            family_id=family_id,
            split="validation",
            fault_class="blocked_path",
            observable_onset_tick=55,
            reference_scenario_sha256="c" * 64,
            fault_scenario_sha256="d" * 64,
        )
    }
    common = {
        "family_id": family_id,
        "scenario_role": "fault",
        "split": "validation",
    }
    rows = [
        {**common, "label": "blocked_path", "end_tick": 60, "features": []},
        {**common, "label": "blocked_path", "end_tick": 65, "features": []},
        {**common, "label": "blocked_path", "end_tick": 70, "features": []},
        {
            **common,
            "label": "excluded_transition",
            "end_tick": 55,
            "features": [],
        },
    ]

    report = build_error_report(
        rows,
        ["nominal", "nominal", "blocked_path", "nominal"],
        family_evidence,
        source_manifest_sha256="a" * 64,
        source_model_sha256="b" * 64,
    )

    assert report == {
        "format": "aeolus_forensic_error_report_v1",
        "evidence_role": "historical_forensic_only",
        "source_manifest_sha256": "a" * 64,
        "source_model_sha256": "b" * 64,
        "input_row_count": 4,
        "scored_row_count": 3,
        "excluded_transition_row_count": 1,
        "error_count": 2,
        "groups": [
            {
                "true_class": "blocked_path",
                "predicted_class": "nominal",
                "family_id": family_id,
                "scenario_role": "fault",
                "operating_profile_id": "primary-high-wind",
                "count": 2,
            }
        ],
    }


def test_build_error_report_uses_explicit_operating_profile_identity():
    family_id = "historical-family-without-sweep-encoding"
    rows = [
        {
            "family_id": family_id,
            "scenario_role": "reference",
            "split": "final",
            "label": "nominal",
            "operating_profile_id": "explicit-storm-profile",
            "features": [],
        }
    ]
    family_evidence = {
        family_id: {
            "family_id": family_id,
            "split": "final",
            "operating_profile_id": "explicit-storm-profile",
        }
    }

    report = build_error_report(
        rows,
        ["blocked_path"],
        family_evidence,
        source_manifest_sha256="1" * 64,
        source_model_sha256="2" * 64,
    )

    groups = report["groups"]
    assert isinstance(groups, list)
    assert groups[0]["operating_profile_id"] == "explicit-storm-profile"


def test_v4_input_guard_rejects_historical_forensic_reports():
    family_id = "legacy-family"
    report = build_error_report(
        [
            {
                "family_id": family_id,
                "scenario_role": "fault",
                "label": "blocked_path",
                "operating_profile_id": "legacy-profile",
            }
        ],
        ["nominal"],
        {family_id: {"operating_profile_id": "legacy-profile"}},
        source_manifest_sha256="a" * 64,
        source_model_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="historical forensic"):
        reject_forensic_report_input(report)


@pytest.mark.parametrize(
    ("manifest_hash", "model_hash"),
    (("A" * 64, "b" * 64), ("a" * 64, "B" * 64), ("a" * 63, "b" * 64)),
)
def test_build_error_report_requires_lowercase_sha256_bindings(
    manifest_hash, model_hash
):
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_error_report(
            [],
            [],
            {},
            source_manifest_sha256=manifest_hash,
            source_model_sha256=model_hash,
        )


def test_write_error_report_enforces_strict_finite_json(tmp_path):
    report = {
        "format": "aeolus_forensic_error_report_v1",
        "evidence_role": "historical_forensic_only",
        "source_manifest_sha256": "a" * 64,
        "source_model_sha256": "b" * 64,
        "input_row_count": 1,
        "scored_row_count": 1,
        "excluded_transition_row_count": 0,
        "error_count": 1,
        "groups": [
            {
                "true_class": "blocked_path",
                "predicted_class": "nominal",
                "family_id": "final-s900-high-noise-t55-lab-blocked-e650",
                "scenario_role": "fault",
                "operating_profile_id": "high-noise",
                "count": 1,
            }
        ],
    }
    output_path = tmp_path / "forensic-report.json"

    write_error_report(output_path, report)

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert output_path.read_bytes().endswith(b"\n")

    unknown_field = {**report, "unexpected": True}
    with pytest.raises(ValueError, match="schema"):
        write_error_report(tmp_path / "unknown.json", unknown_field)

    non_finite = {
        **report,
        "groups": [{**report["groups"][0], "count": float("nan")}],
    }
    with pytest.raises(ValueError, match="non-finite"):
        write_error_report(tmp_path / "non-finite.json", non_finite)
