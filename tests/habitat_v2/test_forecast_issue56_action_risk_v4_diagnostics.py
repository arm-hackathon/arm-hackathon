from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_diagnostics import (
    ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION,
    V4_PROVENANCE_FIELDS,
    Issue56V4DiagnosticsError,
    V4CandidateObservation,
    V4ExecutedObservation,
    bootstrap_equal_weight_group_mean,
    candidate_screening_metrics,
    equal_weight_group_mean,
    executed_action_metrics,
    observation_manifest_sha256,
    provenance_manifest_sha256,
    validate_v4_protocol,
    validate_condition_groups,
)
from scripts.diagnose_action_risk_v4 import V4DiagnosticRunError, _verify_decision_digest


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _candidate(
    group: str,
    family: str,
    step: int,
    action: str,
    *,
    rejected: bool,
    dangerous: bool,
) -> V4CandidateObservation:
    body = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.candidate",
        "condition_group_id": group,
        "family_id": family,
        "decision_step": step,
        "action_id": action,
        "model_rejected": rejected,
        "dangerous": dangerous,
    }
    return V4CandidateObservation(
        group,
        family,
        step,
        action,
        rejected,
        dangerous,
        _digest(body),
    )


def _executed(
    group: str,
    family: str,
    arm: str,
    step: int,
    *,
    selected: str | None,
    dangerous: bool,
    requested: str | None,
    final: str,
    disposition: str,
) -> V4ExecutedObservation:
    body = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.executed",
        "condition_group_id": group,
        "family_id": family,
        "arm": arm,
        "decision_step": step,
        "selected_action_id": selected,
        "actual_dangerous": dangerous,
        "requested_command_sha256": requested,
        "final_command_sha256": final,
        "executed_command_sha256": final,
        "disposition": disposition,
    }
    return V4ExecutedObservation(
        group,
        family,
        arm,
        step,
        selected,
        dangerous,
        requested,
        final,
        final,
        disposition,
        _digest(body),
    )


def test_candidate_metrics_are_separate_from_executed_metrics() -> None:
    candidates = (
        _candidate("group-a", "family-a0", 16, "action-0", rejected=True, dangerous=True),
        _candidate("group-a", "family-a0", 16, "action-1", rejected=False, dangerous=True),
        _candidate("group-a", "family-a1", 16, "action-0", rejected=False, dangerous=False),
    )
    metrics = candidate_screening_metrics(candidates)

    assert metrics["sample_count"] == 3
    assert metrics["dangerous_event_recall"] == 0.5
    assert metrics["false_safe_rate"] == 0.5

    executed = (
        _executed(
            "group-a",
            "family-a0",
            "risk-v4",
            16,
            selected="action-0",
            dangerous=True,
            requested="a" * 64,
            final="b" * 64,
            disposition="PROPOSED_MODIFIED",
        ),
        _executed(
            "group-a",
            "family-a1",
            "risk-v4",
            16,
            selected=None,
            dangerous=True,
            requested=None,
            final="c" * 64,
            disposition="ABSTAINED_TO_HOLD",
        ),
    )
    executed_metrics = executed_action_metrics(executed)

    assert executed_metrics["decision_count"] == 2
    assert executed_metrics["proposal_count"] == 1
    assert executed_metrics["abstention_count"] == 1
    assert executed_metrics["selected_action_false_safe_rate"] == 1.0
    assert executed_metrics["hmc_mismatch_count"] == 1
    assert executed_metrics["disposition_counts"]["PROPOSED_MODIFIED"] == 1
    assert executed_metrics["disposition_counts"]["ABSTAINED_TO_HOLD"] == 1


def test_observations_round_trip_and_manifest_is_order_independent() -> None:
    candidate = _candidate(
        "group-a", "family-a0", 16, "action-0", rejected=True, dangerous=True
    )
    executed = _executed(
        "group-a",
        "family-a0",
        "risk-v4",
        16,
        selected="action-0",
        dangerous=True,
        requested="a" * 64,
        final="b" * 64,
        disposition="PROPOSED_MODIFIED",
    )

    assert V4CandidateObservation.from_mapping(candidate.to_mapping()) == candidate
    assert V4ExecutedObservation.from_mapping(executed.to_mapping()) == executed
    assert observation_manifest_sha256((candidate, executed)) == observation_manifest_sha256(
        (executed, candidate)
    )

    tampered = dict(candidate.to_mapping())
    tampered["dangerous"] = False
    with pytest.raises(Issue56V4DiagnosticsError, match="digest"):
        V4CandidateObservation.from_mapping(tampered)


def test_condition_groups_require_two_families_and_group_means_are_equal_weighted() -> None:
    observations = (
        _candidate("group-a", "family-a0", 16, "action-0", rejected=False, dangerous=False),
        _candidate("group-a", "family-a1", 16, "action-0", rejected=False, dangerous=False),
        _candidate("group-b", "family-b0", 16, "action-0", rejected=False, dangerous=False),
        _candidate("group-b", "family-b1", 16, "action-0", rejected=False, dangerous=False),
    )

    assert validate_condition_groups(observations) == {
        "group-a": ("family-a0", "family-a1"),
        "group-b": ("family-b0", "family-b1"),
    }
    aggregate = equal_weight_group_mean({"group-a": (1.0, 3.0), "group-b": (10.0,)})
    assert aggregate["group_means"] == {"group-a": 2.0, "group-b": 10.0}
    assert aggregate["equal_weight_mean"] == 6.0

    with pytest.raises(Issue56V4DiagnosticsError, match="paired families"):
        validate_condition_groups(observations[:1])


def test_group_bootstrap_is_deterministic_and_validates_support() -> None:
    values = {"group-a": 1.0, "group-b": 3.0, "group-c": 8.0}
    first = bootstrap_equal_weight_group_mean(values, resamples=100)
    second = bootstrap_equal_weight_group_mean(values, resamples=100)

    assert first == second
    assert first["point_estimate"] == 4.0
    assert first["group_ids"] == ["group-a", "group-b", "group-c"]
    with pytest.raises(Issue56V4DiagnosticsError, match="at least two"):
        bootstrap_equal_weight_group_mean({"group-a": 1.0}, resamples=10)


def test_provenance_manifest_requires_every_bound_identity() -> None:
    manifest = {field: "a" * 64 for field in V4_PROVENANCE_FIELDS}
    digest = provenance_manifest_sha256(manifest)

    assert len(digest) == 64
    missing = dict(manifest)
    del missing["hmc_binding_sha256"]
    with pytest.raises(Issue56V4DiagnosticsError, match="provenance fields"):
        provenance_manifest_sha256(missing)


def test_v4_protocol_contract_is_explicitly_pre_model_and_fail_closed() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert validate_v4_protocol(protocol) == protocol
    tampered = dict(protocol)
    tampered["scope"] = dict(protocol["scope"])
    tampered["scope"]["training_authorized"] = True
    with pytest.raises(Issue56V4DiagnosticsError, match="authorizes learned-model work"):
        validate_v4_protocol(tampered)


def test_v4_adapter_rejects_tampered_serialized_decision() -> None:
    body = {
        "decision_step": 16,
        "selected_action_id": "action-0",
        "requested_command_sha256": "a" * 64,
        "final_command_sha256": "b" * 64,
        "executed_command_sha256": "b" * 64,
        "disposition": "PROPOSED_MODIFIED",
    }
    decision = {**body, "decision_sha256": _digest(body)}

    _verify_decision_digest(decision)
    tampered = dict(decision)
    tampered["disposition"] = "PROPOSED_ACCEPTED"
    with pytest.raises(V4DiagnosticRunError, match="digest"):
        _verify_decision_digest(tampered)
