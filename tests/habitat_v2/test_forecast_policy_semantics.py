from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POLICY = (
    ROOT
    / "docs/plans/2026-08-14-habitat-v2-forecast-statistical-policy-proposal-v1.json"
)
POLICY_SHA256 = "91e662707c3b4d139cb5bf78f01ef411d4609b12452d8592f30822eaa6e7eced"
POLICY_BYTES_SHA256 = "170f8aeaaf8fb938eecb32106c365bab673133f04515e6fada9b6d4a8d07457b"


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def write_self_consistent_policy(
    path: Path, value: dict[str, object]
) -> tuple[str, str]:
    value.pop("policy_sha256", None)
    value["policy_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    raw = canonical(value) + b"\n"
    path.write_bytes(raw)
    return value["policy_sha256"], hashlib.sha256(raw).hexdigest()  # type: ignore[return-value]


def test_approved_policy_design_has_closed_nested_semantics_and_no_generation_authority() -> (
    None
):
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        require_approved_policy,
        validate_ratified_policy_design,
    )

    policy = load_qualification_policy(
        POLICY,
        expected_policy_sha256=POLICY_SHA256,
        expected_policy_bytes_sha256=POLICY_BYTES_SHA256,
    )
    validate_ratified_policy_design(policy)
    assert require_approved_policy(policy) is policy
    assert policy.permissions["pilot_generation_allowed"] is False
    assert policy.permissions["scenario_generation_allowed"] is False
    assert policy.permissions["canonical_generation_allowed"] is False
    assert policy.permissions["model_training_allowed"] is False

    from dataclasses import replace

    substituted_bytes = replace(policy, policy_bytes_sha256="0" * 64)
    with pytest.raises(QualificationPolicyError, match="bytes are not compiled"):
        require_approved_policy(substituted_bytes)


def test_self_consistent_bootstrap_substitution_fails_semantic_closure(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        validate_ratified_policy_design,
    )

    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["timing"]["bootstrap"]["resamples"] = 9_999
    path = tmp_path / "bootstrap-substitute.json"
    semantic, raw = write_self_consistent_policy(path, value)
    policy = load_qualification_policy(
        path,
        expected_policy_sha256=semantic,
        expected_policy_bytes_sha256=raw,
    )
    with pytest.raises(QualificationPolicyError, match="bootstrap resamples"):
        validate_ratified_policy_design(policy)


def test_self_consistent_unknown_nested_policy_field_is_rejected(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        validate_ratified_policy_design,
    )

    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["timing"]["support"]["post_hoc_exception"] = True
    path = tmp_path / "unknown-nested.json"
    semantic, raw = write_self_consistent_policy(path, value)
    policy = load_qualification_policy(
        path,
        expected_policy_sha256=semantic,
        expected_policy_bytes_sha256=raw,
    )
    with pytest.raises(QualificationPolicyError, match="timing.support.*unknown"):
        validate_ratified_policy_design(policy)
