from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def policy_mapping(
    *,
    pilot_cluster_count: int = 60,
    ratification_status: str = "DRAFT_FOR_REVIEW",
    model_training_allowed: bool = False,
    permission_overrides: dict[str, bool] | None = None,
) -> dict[str, object]:
    permissions = {
        "model_training_allowed": model_training_allowed,
        "learned_action_authority_allowed": False,
        "final_set_access_allowed": False,
        "publication_allowed": False,
        "pilot_generation_allowed": False,
        "scenario_generation_allowed": False,
        "canonical_generation_allowed": False,
        "validation_access_allowed": False,
        "hmc_is_sole_actuator_authority": True,
        "proceed_requires_separate_experiment_freeze": True,
    }
    if permission_overrides is not None:
        permissions.update(permission_overrides)
    value: dict[str, object] = {
        "schema_version": "aeolus_habitat_v2_forecast_qualification_policy_v1",
        "ratification_status": ratification_status,
        "foundation": {"d1_candidate_sha": "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3"},
        "timing": {"pilot_cluster_count": pilot_cluster_count},
        "baseline": {"bootstrap_resamples": 10_000},
        "permissions": permissions,
        "stop_precedence": {"timing": ["STOP_EVIDENCE_INVALID"]},
    }
    value["policy_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def write_policy(path: Path, value: object) -> None:
    path.write_bytes(canonical(value) + b"\n")


def policy_file_sha256(value: object) -> str:
    return hashlib.sha256(canonical(value) + b"\n").hexdigest()


def test_expected_policy_identity_rejects_self_consistent_substitution(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
    )

    original_path = tmp_path / "original.json"
    original = policy_mapping()
    write_policy(original_path, original)
    loaded = load_qualification_policy(
        original_path,
        expected_policy_sha256=original["policy_sha256"],
        expected_policy_bytes_sha256=policy_file_sha256(original),
    )
    assert loaded.policy_sha256 == original["policy_sha256"]
    assert loaded.policy_bytes_sha256 == policy_file_sha256(original)

    substitute_path = tmp_path / "substitute.json"
    substitute = policy_mapping(pilot_cluster_count=61)
    write_policy(substitute_path, substitute)

    with pytest.raises(QualificationPolicyError, match="expected policy identity"):
        load_qualification_policy(
            substitute_path,
            expected_policy_sha256=original["policy_sha256"],
            expected_policy_bytes_sha256=policy_file_sha256(original),
        )


def test_semantically_equivalent_noncanonical_policy_bytes_are_rejected(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
    )

    value = policy_mapping()
    path = tmp_path / "reformatted.json"
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(QualificationPolicyError, match="policy byte identity"):
        load_qualification_policy(
            path,
            expected_policy_sha256=value["policy_sha256"],
            expected_policy_bytes_sha256=policy_file_sha256(value),
        )


def test_draft_policy_cannot_authorize_qualification(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        require_approved_policy,
    )

    path = tmp_path / "draft.json"
    value = policy_mapping()
    write_policy(path, value)
    draft = load_qualification_policy(
        path,
        expected_policy_sha256=value["policy_sha256"],
        expected_policy_bytes_sha256=policy_file_sha256(value),
    )

    with pytest.raises(QualificationPolicyError, match="not approved"):
        require_approved_policy(draft)


def test_approved_policy_cannot_authorize_model_training(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        require_approved_policy,
    )

    path = tmp_path / "overreaching.json"
    value = policy_mapping(
        ratification_status="APPROVED",
        model_training_allowed=True,
    )
    write_policy(path, value)
    policy = load_qualification_policy(
        path,
        expected_policy_sha256=value["policy_sha256"],
        expected_policy_bytes_sha256=policy_file_sha256(value),
    )

    with pytest.raises(QualificationPolicyError, match="model training"):
        require_approved_policy(policy)


@pytest.mark.parametrize(
    "forbidden_permission",
    (
        "learned_action_authority_allowed",
        "final_set_access_allowed",
        "publication_allowed",
        "pilot_generation_allowed",
        "scenario_generation_allowed",
        "canonical_generation_allowed",
        "validation_access_allowed",
    ),
)
def test_approved_policy_cannot_grant_forbidden_authority(
    tmp_path: Path,
    forbidden_permission: str,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        require_approved_policy,
    )

    path = tmp_path / f"overreaching-{forbidden_permission}.json"
    value = policy_mapping(
        ratification_status="APPROVED",
        permission_overrides={forbidden_permission: True},
    )
    write_policy(path, value)
    policy = load_qualification_policy(
        path,
        expected_policy_sha256=value["policy_sha256"],
        expected_policy_bytes_sha256=policy_file_sha256(value),
    )

    with pytest.raises(QualificationPolicyError, match="forbidden authority"):
        require_approved_policy(policy)


@pytest.mark.parametrize(
    "required_invariant",
    (
        "hmc_is_sole_actuator_authority",
        "proceed_requires_separate_experiment_freeze",
    ),
)
def test_approved_policy_cannot_clear_required_safety_invariant(
    tmp_path: Path,
    required_invariant: str,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        require_approved_policy,
    )

    path = tmp_path / f"unsafe-{required_invariant}.json"
    value = policy_mapping(
        ratification_status="APPROVED",
        permission_overrides={required_invariant: False},
    )
    write_policy(path, value)
    policy = load_qualification_policy(
        path,
        expected_policy_sha256=value["policy_sha256"],
        expected_policy_bytes_sha256=policy_file_sha256(value),
    )

    with pytest.raises(QualificationPolicyError, match="safety invariant"):
        require_approved_policy(policy)


def test_approved_policy_with_empty_nested_sections_has_no_authority(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        QualificationPolicyError,
        load_qualification_policy,
        require_approved_policy,
    )

    path = tmp_path / "empty-nested-sections.json"
    value = policy_mapping(ratification_status="APPROVED")
    for field in ("foundation", "timing", "baseline", "stop_precedence"):
        value[field] = {}
    value.pop("policy_sha256")
    value["policy_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    write_policy(path, value)
    policy = load_qualification_policy(
        path,
        expected_policy_sha256=value["policy_sha256"],
        expected_policy_bytes_sha256=policy_file_sha256(value),
    )

    with pytest.raises(QualificationPolicyError, match="not compiled for approval"):
        require_approved_policy(policy)


def test_resource_estimate_exposes_dense_ridge_campaign_risk() -> None:
    from aeolus.habitat_v2.forecast.qualification import (
        estimate_ridge_resources,
    )

    full = estimate_ridge_resources(
        sample_count=52_416,
        window_steps=16,
        horizon_steps=8,
        representation="FULL_CONTRACT_FLAT",
    )
    compact = estimate_ridge_resources(
        sample_count=52_416,
        window_steps=16,
        horizon_steps=8,
        representation="TARGET_HISTORY_PLUS_ACTION",
    )

    assert full.feature_count == 34_987
    assert compact.feature_count == 843
    assert full.dual_gram_f64_bytes == 52_416 * 52_416 * 8
    assert full.primal_gram_f64_bytes == 34_987 * 34_987 * 8
    assert compact.primal_gram_f64_bytes == 843 * 843 * 8
    assert full.design_f64_bytes > compact.design_f64_bytes * 40
