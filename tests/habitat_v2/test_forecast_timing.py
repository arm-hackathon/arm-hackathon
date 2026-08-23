from __future__ import annotations

import pytest

from aeolus.habitat_v2.forecast.timing import (
    TimingError,
    emit_baseline_gate_receipt,
    emit_timing_receipt,
    validate_candidate_timing,
)

INPUT = "29d743472712dff68759477debd25aadba8a0584ad89d164bc5c583260356971"
TARGET = "26e480ca4f07d2092fc6e96fcf2f006948e9e2872ad2b0fd4ae3ac8e947c74db"


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


def test_receipt_evidence_requires_an_exact_dict() -> None:
    class Evidence(dict[str, object]):
        pass

    with pytest.raises(TimingError, match="one canonical object"):
        emit_timing_receipt(
            4,
            2,
            timing_evidence=Evidence(pilot="fixture"),
            input_manifest_sha256=INPUT,
            target_manifest_sha256=TARGET,
        )
