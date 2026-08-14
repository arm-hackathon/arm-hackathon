from __future__ import annotations

import pytest

from aeolus.habitat_v2.forecast.timing import (
    TimingError,
    emit_baseline_gate_receipt,
    emit_timing_receipt,
    validate_candidate_timing,
)

INPUT = "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
TARGET = "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"


def test_candidate_validation_and_exact_stop_receipts_are_deterministic() -> None:
    validate_candidate_timing(4, 2)
    with pytest.raises(TimingError):
        validate_candidate_timing(3, 2)
    first = emit_timing_receipt(
        4,
        2,
        timing_evidence={"pilot": "fixture"},
        input_manifest_sha256=INPUT,
        target_manifest_sha256=TARGET,
    )
    second = emit_timing_receipt(
        4,
        2,
        timing_evidence={"pilot": "fixture"},
        input_manifest_sha256=INPUT,
        target_manifest_sha256=TARGET,
    )
    assert first.outcome == "STOP_UNDERPOWERED"
    assert first.release_tier == "DEVELOPMENT_FIXTURE_ONLY"
    assert first.receipt_sha256 == second.receipt_sha256
    assert (
        emit_baseline_gate_receipt(
            baseline_evidence={"baseline": "persistence"},
            input_manifest_sha256=INPUT,
            target_manifest_sha256=TARGET,
        ).outcome
        == "STOP_UNDERPOWERED"
    )


def test_unsupported_receipt_outcomes_reject() -> None:
    with pytest.raises(TimingError):
        emit_timing_receipt(
            4,
            2,
            timing_evidence={},
            input_manifest_sha256=INPUT,
            target_manifest_sha256=TARGET,
            outcome="SELECTED",
        )
    with pytest.raises(TimingError):
        emit_baseline_gate_receipt(
            baseline_evidence={},
            input_manifest_sha256=INPUT,
            target_manifest_sha256=TARGET,
            outcome="PROCEED_TO_EXPERIMENT_FREEZE",
        )
