"""Fail-closed authority boundary for Habitat V2 forecast qualification policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Final

from .contracts import (
    ForecastContractError,
    _freeze,
    _strict_json_bytes,
    canonical_json_bytes,
)

POLICY_SCHEMA_VERSION: Final = "aeolus_habitat_v2_forecast_qualification_policy_v1"
RATIFICATION_STATUSES: Final = frozenset(("DRAFT_FOR_REVIEW", "APPROVED"))
WINDOW_STEPS: Final = frozenset((4, 8, 16))
HORIZON_STEPS: Final = frozenset((2, 4, 8))
RIDGE_REPRESENTATIONS: Final = frozenset(
    ("FULL_CONTRACT_FLAT", "TARGET_HISTORY_PLUS_ACTION")
)
_FULL_FEATURES_PER_STEP: Final = 194 + 167 * 5 + 4 + 4 + 287 * 4
_TARGET_COUNT: Final = 51
_ACTION_COUNT: Final = 27
_POLICY_FIELDS: Final = frozenset(
    (
        "schema_version",
        "ratification_status",
        "foundation",
        "timing",
        "baseline",
        "permissions",
        "stop_precedence",
        "policy_sha256",
    )
)
_FORBIDDEN_APPROVED_PERMISSIONS: Final = (
    "learned_action_authority_allowed",
    "final_set_access_allowed",
    "publication_allowed",
    "pilot_generation_allowed",
    "scenario_generation_allowed",
    "canonical_generation_allowed",
    "validation_access_allowed",
)
_REQUIRED_APPROVED_INVARIANTS: Final = (
    "hmc_is_sole_actuator_authority",
    "proceed_requires_separate_experiment_freeze",
)
_PERMISSION_FIELDS: Final = frozenset(
    (
        "model_training_allowed",
        *_FORBIDDEN_APPROVED_PERMISSIONS,
        *_REQUIRED_APPROVED_INVARIANTS,
    )
)
_APPROVED_POLICY_SHA256S: Final = frozenset(
    {"91e662707c3b4d139cb5bf78f01ef411d4609b12452d8592f30822eaa6e7eced"}
)
_APPROVED_POLICY_BYTES_SHA256S: Final = frozenset(
    {"170f8aeaaf8fb938eecb32106c365bab673133f04515e6fada9b6d4a8d07457b"}
)


class QualificationPolicyError(ValueError):
    """The qualification policy is missing, substituted or outside its contract."""


@dataclass(frozen=True, slots=True)
class QualificationPolicy:
    path: Path
    schema_version: str
    ratification_status: str
    foundation: Mapping[str, Any]
    timing: Mapping[str, Any]
    baseline: Mapping[str, Any]
    permissions: Mapping[str, Any]
    stop_precedence: Mapping[str, Any]
    policy_sha256: str
    policy_bytes_sha256: str


@dataclass(frozen=True, slots=True)
class RidgeResourceEstimate:
    sample_count: int
    window_steps: int
    horizon_steps: int
    representation: str
    feature_count: int
    target_count: int
    minimum_f32_tensor_bytes: int
    design_f64_bytes: int
    targets_f64_bytes: int
    dual_gram_f64_bytes: int
    primal_gram_f64_bytes: int
    coefficient_f64_bytes: int


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise QualificationPolicyError(f"{label} must be lowercase SHA-256")
    return value


def estimate_ridge_resources(
    *,
    sample_count: int,
    window_steps: int,
    horizon_steps: int,
    representation: str,
) -> RidgeResourceEstimate:
    """Return allocation sizes without allocating campaign tensors or matrices."""

    if type(sample_count) is not int or sample_count < 1:
        raise QualificationPolicyError("sample_count must be a positive integer")
    if type(window_steps) is not int or window_steps not in WINDOW_STEPS:
        raise QualificationPolicyError("window_steps is outside the frozen timing grid")
    if type(horizon_steps) is not int or horizon_steps not in HORIZON_STEPS:
        raise QualificationPolicyError(
            "horizon_steps is outside the frozen timing grid"
        )
    if representation not in RIDGE_REPRESENTATIONS:
        raise QualificationPolicyError("ridge representation is unsupported")

    features_per_step = (
        _FULL_FEATURES_PER_STEP
        if representation == "FULL_CONTRACT_FLAT"
        else _TARGET_COUNT
    )
    feature_count = window_steps * features_per_step + _ACTION_COUNT
    target_count = horizon_steps * _TARGET_COUNT
    return RidgeResourceEstimate(
        sample_count=sample_count,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        representation=representation,
        feature_count=feature_count,
        target_count=target_count,
        minimum_f32_tensor_bytes=sample_count * (feature_count + target_count) * 4,
        design_f64_bytes=sample_count * feature_count * 8,
        targets_f64_bytes=sample_count * target_count * 8,
        dual_gram_f64_bytes=sample_count * sample_count * 8,
        primal_gram_f64_bytes=feature_count * feature_count * 8,
        coefficient_f64_bytes=feature_count * target_count * 8,
    )


def load_qualification_policy(
    path: str | Path,
    *,
    expected_policy_sha256: str,
    expected_policy_bytes_sha256: str,
) -> QualificationPolicy:
    """Load one exact policy; its internal self-hash is not external authority."""

    candidate = Path(path).resolve()
    expected = _sha256(expected_policy_sha256, label="expected policy identity")
    expected_bytes = _sha256(
        expected_policy_bytes_sha256,
        label="expected policy byte identity",
    )
    try:
        raw = candidate.read_bytes()
    except OSError as error:
        raise QualificationPolicyError("qualification policy cannot be read") from error
    actual_bytes = hashlib.sha256(raw).hexdigest()
    try:
        value = _strict_json_bytes(raw, label=candidate.name)
    except ForecastContractError as error:
        raise QualificationPolicyError(
            "qualification policy is not strict JSON"
        ) from error

    unknown = sorted(set(value) - _POLICY_FIELDS)
    missing = sorted(_POLICY_FIELDS - set(value))
    if unknown or missing:
        raise QualificationPolicyError(
            f"qualification policy has unknown={unknown}, missing={missing}"
        )
    if value["schema_version"] != POLICY_SCHEMA_VERSION:
        raise QualificationPolicyError("qualification policy schema is unsupported")
    if value["ratification_status"] not in RATIFICATION_STATUSES:
        raise QualificationPolicyError(
            "qualification policy ratification status is invalid"
        )
    for field in (
        "foundation",
        "timing",
        "baseline",
        "permissions",
        "stop_precedence",
    ):
        if type(value[field]) is not dict:
            raise QualificationPolicyError(
                f"qualification policy {field} must be an object"
            )

    declared = _sha256(value["policy_sha256"], label="policy_sha256")
    body = dict(value)
    body.pop("policy_sha256")
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if declared != actual:
        raise QualificationPolicyError("qualification policy self-hash is invalid")
    if declared != expected:
        raise QualificationPolicyError(
            "qualification policy differs from expected policy identity"
        )
    if actual_bytes != expected_bytes:
        raise QualificationPolicyError(
            "qualification policy differs from expected policy byte identity"
        )

    return QualificationPolicy(
        path=candidate,
        schema_version=value["schema_version"],
        ratification_status=value["ratification_status"],
        foundation=_freeze(value["foundation"]),
        timing=_freeze(value["timing"]),
        baseline=_freeze(value["baseline"]),
        permissions=_freeze(value["permissions"]),
        stop_precedence=_freeze(value["stop_precedence"]),
        policy_sha256=declared,
        policy_bytes_sha256=actual_bytes,
    )


def _exact_nested_mapping(
    value: Any,
    expected_fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationPolicyError(f"{label} must be an object")
    actual = set(value)
    unknown = sorted(actual - expected_fields)
    missing = sorted(expected_fields - actual)
    if unknown or missing:
        raise QualificationPolicyError(
            f"{label} has unknown={unknown}, missing={missing}"
        )
    return value


def validate_ratified_policy_design(policy: QualificationPolicy) -> None:
    """Validate the complete human-ratified D2 design, without opening authority."""

    if type(policy) is not QualificationPolicy:
        raise QualificationPolicyError("policy design requires the exact policy type")

    foundation = _exact_nested_mapping(
        policy.foundation,
        frozenset(
            {
                "action_catalogue",
                "alarm_manifest",
                "d1_candidate_git_sha",
                "d1_development_profile",
                "d1_record_contract",
                "hmc_binding",
                "hmc_foundation_git_sha",
                "observable_topology_sha256",
                "observation_cadence_seconds",
                "package_version",
                "pilot_profile_action_proposal",
                "pilot_roster_proposal",
                "required_exact_bindings_before_approval",
            }
        ),
        label="foundation",
    )
    if (
        foundation["d1_candidate_git_sha"] != "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3"
        or foundation["hmc_foundation_git_sha"]
        != "79d6a718e0d44122a763bb72f9c8ed929f39fd23"
        or foundation["observable_topology_sha256"]
        != "b0246a9dc8f847c3236068c8e1eeeddb31809a680e6133eaf038ea197d6e10e6"
        or foundation["observation_cadence_seconds"] != 60.0
        or foundation["package_version"] != "0.8.0"
    ):
        raise QualificationPolicyError("policy foundation identity drifts")
    for field in (
        "action_catalogue",
        "alarm_manifest",
        "d1_development_profile",
        "d1_record_contract",
        "hmc_binding",
    ):
        binding = _exact_nested_mapping(
            foundation[field],
            frozenset({"path", "raw_bytes_sha256", "semantic_sha256"}),
            label=f"foundation.{field}",
        )
        if type(binding["path"]) is not str or not binding["path"]:
            raise QualificationPolicyError(f"foundation.{field}.path is invalid")
        _sha256(binding["raw_bytes_sha256"], label=f"foundation.{field}.raw bytes")
        _sha256(binding["semantic_sha256"], label=f"foundation.{field}.semantic")

    from .pilot import (
        APPROVED_PROFILE_ACTION_BYTES_SHA256,
        APPROVED_PROFILE_ACTION_SHA256,
        APPROVED_ROSTER_BYTES_SHA256,
        APPROVED_ROSTER_SHA256,
    )

    proposal_expectations = {
        "pilot_roster_proposal": (
            APPROVED_ROSTER_SHA256,
            APPROVED_ROSTER_BYTES_SHA256,
        ),
        "pilot_profile_action_proposal": (
            APPROVED_PROFILE_ACTION_SHA256,
            APPROVED_PROFILE_ACTION_BYTES_SHA256,
        ),
    }
    for field, (semantic, raw) in proposal_expectations.items():
        binding = _exact_nested_mapping(
            foundation[field],
            frozenset({"ratification_status", "raw_bytes_sha256", "semantic_sha256"}),
            label=f"foundation.{field}",
        )
        if binding != {
            "ratification_status": "APPROVED",
            "raw_bytes_sha256": raw,
            "semantic_sha256": semantic,
        }:
            raise QualificationPolicyError(f"foundation.{field} identity drifts")
    required_bindings = (
        "ratified pilot profile/action packet semantic and raw byte identities",
        "pilot identity and matched-control record contract identities",
        "permanent pilot exclusion validator/source identity",
        "evaluator and timing implementation source-blob manifest identities",
        "compact and full-reference ridge source-blob manifest identities",
        "bounded resource benchmark receipt identity",
        "frozen critical envelope/group manifest identity",
    )
    if (
        tuple(foundation["required_exact_bindings_before_approval"])
        != required_bindings
    ):
        raise QualificationPolicyError("required implementation binding roster drifts")

    timing = _exact_nested_mapping(
        policy.timing,
        frozenset(
            {
                "admissibility",
                "aggregation_order",
                "analysis_unit",
                "bootstrap",
                "critical_inferential_universe",
                "folds",
                "history_steps",
                "horizon_steps",
                "matched_action_and_noise",
                "metric_semantics",
                "pilot",
                "selection",
                "support",
                "target_scale",
            }
        ),
        label="timing",
    )
    if (
        tuple(timing["history_steps"]) != (4, 8, 16)
        or tuple(timing["horizon_steps"]) != (2, 4, 8)
        or timing["analysis_unit"] != "family_cluster_id"
    ):
        raise QualificationPolicyError("timing grid or analysis unit drifts")

    admissibility = _exact_nested_mapping(
        timing["admissibility"],
        frozenset(
            {
                "action_aware_vs_blinded_material_advantage",
                "affirmative_noninferiority_required",
                "all_four_action_effect_claims_required",
                "coverage_maximum_allowed_loss",
                "fnr_fpr_maximum_allowed_increase",
                "history_equivalence_requires_upper_lcb_and_adjusted_test",
                "history_maximum_material_longer_gain",
                "no_invalid_outputs",
                "ridge_maximum_allowed_normalized_mae_loss",
            }
        ),
        label="timing.admissibility",
    )
    if admissibility != {
        "action_aware_vs_blinded_material_advantage": 0.005,
        "affirmative_noninferiority_required": True,
        "all_four_action_effect_claims_required": True,
        "coverage_maximum_allowed_loss": 0.02,
        "fnr_fpr_maximum_allowed_increase": 0.02,
        "history_equivalence_requires_upper_lcb_and_adjusted_test": True,
        "history_maximum_material_longer_gain": 0.005,
        "no_invalid_outputs": True,
        "ridge_maximum_allowed_normalized_mae_loss": 0.005,
    }:
        raise QualificationPolicyError("timing admissibility margins drift")

    bootstrap = _exact_nested_mapping(
        timing["bootstrap"],
        frozenset(
            {
                "one_sided_lower_quantile",
                "one_sided_upper_quantile",
                "p_value",
                "quantile_method",
                "receipt_fields",
                "resamples",
                "rng",
                "seed",
                "stratified_by_mode_load",
                "two_sided_quantiles",
                "unit",
            }
        ),
        label="timing.bootstrap",
    )
    if bootstrap["resamples"] != 10_000:
        raise QualificationPolicyError("bootstrap resamples must equal 10000")
    if (
        bootstrap["rng"] != "numpy.random.Generator(PCG64DXSM)"
        or bootstrap["quantile_method"] != "linear"
        or bootstrap["unit"] != "whole family cluster"
        or bootstrap["stratified_by_mode_load"] is not True
        or bootstrap["one_sided_lower_quantile"] != 0.05
        or bootstrap["one_sided_upper_quantile"] != 0.95
        or tuple(bootstrap["two_sided_quantiles"]) != (0.025, 0.975)
        or tuple(bootstrap["receipt_fields"])
        != (
            "seed_digest",
            "analysis_id",
            "ordered_eligible_cluster_ids",
            "all_resample_index_vectors",
        )
    ):
        raise QualificationPolicyError("bootstrap procedure drifts")

    universe = _exact_nested_mapping(
        timing["critical_inferential_universe"],
        frozenset(
            {
                "claim_count",
                "claims",
                "descriptive_only",
                "familywise_alpha",
                "holm_method",
                "unsupported_claim",
            }
        ),
        label="timing.critical_inferential_universe",
    )
    claims = _exact_nested_mapping(
        universe["claims"],
        frozenset(
            {
                "action_aware_vs_blinded",
                "action_effect",
                "coverage_fnr_fpr_noninferiority",
                "ridge_vs_persistence_and_linear_noninferiority",
                "shorter_vs_longer_history_equivalence",
            }
        ),
        label="timing.critical_inferential_universe.claims",
    )
    if (
        dict(claims)
        != {
            "action_aware_vs_blinded": 9,
            "action_effect": 36,
            "coverage_fnr_fpr_noninferiority": 27,
            "ridge_vs_persistence_and_linear_noninferiority": 18,
            "shorter_vs_longer_history_equivalence": 9,
        }
        or universe["claim_count"] != 99
        or sum(claims.values()) != 99
        or universe["familywise_alpha"] != 0.05
        or universe["holm_method"] != "step_down"
    ):
        raise QualificationPolicyError("99-claim Holm family drifts")

    folds = _exact_nested_mapping(
        timing["folds"],
        frozenset(
            {
                "all_preprocessing_scale_and_alpha_selection_train_only",
                "alpha_tie_break",
                "inner_assignment",
                "inner_fold_count",
                "inner_train_clusters",
                "inner_validation_clusters",
                "outer_assignment",
                "outer_fold_count",
                "outer_test_clusters",
                "outer_train_clusters",
                "same_outer_assignment_for_all_timing_pairs",
            }
        ),
        label="timing.folds",
    )
    if (
        folds["outer_fold_count"],
        folds["outer_train_clusters"],
        folds["outer_test_clusters"],
        folds["inner_fold_count"],
        folds["inner_train_clusters"],
        folds["inner_validation_clusters"],
    ) != (5, 48, 12, 4, 36, 12) or any(
        folds[field] is not True
        for field in (
            "all_preprocessing_scale_and_alpha_selection_train_only",
            "same_outer_assignment_for_all_timing_pairs",
        )
    ):
        raise QualificationPolicyError("nested cluster-fold design drifts")

    matched = _exact_nested_mapping(
        timing["matched_action_and_noise"],
        frozenset(
            {
                "control",
                "control_reuse",
                "effect",
                "minimum_material_effect",
                "noise_floor",
                "rule",
            }
        ),
        label="timing.matched_action_and_noise",
    )
    if (
        matched["minimum_material_effect"] != 0.02
        or "four actions" not in matched["rule"]
    ):
        raise QualificationPolicyError("matched action-information rule drifts")
    metrics = _exact_nested_mapping(
        timing["metric_semantics"],
        frozenset(
            {
                "abstention",
                "deterministic_repeat_drift",
                "harmful_crossing",
                "invalid_output",
            }
        ),
        label="timing.metric_semantics",
    )
    if (
        "Inclusive envelope boundary" not in metrics["harmful_crossing"]
        or "FN for positive truth" not in metrics["abstention"]
        or metrics["invalid_output"]
        != "STOP_EVIDENCE_INVALID; never abstention or underpower."
    ):
        raise QualificationPolicyError(
            "harm, abstention or invalid-output semantics drift"
        )

    pilot = _exact_nested_mapping(
        timing["pilot"],
        frozenset(
            {
                "action_runs",
                "anchors",
                "cluster_count",
                "clusters_per_stratum",
                "matched_no_proposal_control_runs",
                "members_per_family",
                "mode_load_strata",
                "normal_action_count",
                "permanent_exclusion_scope",
                "repetitions_per_cluster",
                "total_hmc_runs",
            }
        ),
        label="timing.pilot",
    )
    if (
        pilot["action_runs"],
        tuple(pilot["anchors"]),
        pilot["cluster_count"],
        pilot["clusters_per_stratum"],
        pilot["matched_no_proposal_control_runs"],
        pilot["members_per_family"],
        pilot["mode_load_strata"],
        pilot["normal_action_count"],
        pilot["repetitions_per_cluster"],
        pilot["total_hmc_runs"],
    ) != (18_720, (16, 40, 64), 60, 5, 4_680, 13, 12, 4, 2, 23_400):
        raise QualificationPolicyError("pilot arithmetic drifts")

    support = _exact_nested_mapping(
        timing["support"],
        frozenset(
            {
                "action_pair_clusters_per_action",
                "action_pair_clusters_per_stratum",
                "aggregate_regression_clusters",
                "aggregate_regression_per_stratum",
                "coverage_clusters",
                "coverage_clusters_per_stratum",
                "crossing_negative_clusters",
                "crossing_negative_per_stratum_minimum",
                "crossing_positive_clusters",
                "crossing_positive_per_stratum_minimum",
                "missing_or_ineligible_mandatory_cluster",
            }
        ),
        label="timing.support",
    )
    if dict(support) != {
        "action_pair_clusters_per_action": 60,
        "action_pair_clusters_per_stratum": 5,
        "aggregate_regression_clusters": 60,
        "aggregate_regression_per_stratum": 5,
        "coverage_clusters": 60,
        "coverage_clusters_per_stratum": 5,
        "crossing_negative_clusters": 24,
        "crossing_negative_per_stratum_minimum": 1,
        "crossing_positive_clusters": 24,
        "crossing_positive_per_stratum_minimum": 1,
        "missing_or_ineligible_mandatory_cluster": "STOP_UNDERPOWERED",
    }:
        raise QualificationPolicyError("timing support rules drift")
    target_scale = _exact_nested_mapping(
        timing["target_scale"],
        frozenset(
            {
                "all_51_targets_required",
                "heldout_outside_training_percentiles",
                "native_unit_crossing_metrics_ignore_normalization_scale",
                "quantile_method",
                "source",
                "statistic",
                "zero_or_nonfinite_scale",
            }
        ),
        label="timing.target_scale",
    )
    if (
        target_scale["all_51_targets_required"] is not True
        or target_scale["native_unit_crossing_metrics_ignore_normalization_scale"]
        is not True
        or target_scale["quantile_method"] != "linear"
        or target_scale["statistic"] != "P95_MINUS_P5"
        or target_scale["zero_or_nonfinite_scale"] != "STOP_UNDERPOWERED"
    ):
        raise QualificationPolicyError("target-scale contract drifts")

    if tuple(timing["aggregation_order"]) != (
        "point",
        "sample",
        "equal action-anchor-repetition mean",
        "equal supported target-offset mean",
        "equal cluster mean",
    ) or tuple(timing["selection"]) != (
        "For each H choose W=4 only if admissible and equivalent to W=8 and W=16.",
        "Otherwise choose W=8 only if admissible and equivalent to W=16.",
        "Otherwise choose W=16 if admissible.",
        "Select the largest H with a selected W.",
        "If no H has a selected W emit STOP_NO_DEFENSIBLE_TIMING.",
    ):
        raise QualificationPolicyError("aggregation or timing selection order drifts")

    baseline = _exact_nested_mapping(
        policy.baseline,
        frozenset(
            {
                "future_candidate_margin",
                "headroom",
                "oracle",
                "ridge_replacement",
                "simple_comparator_selection",
                "validation_access",
                "validation_support",
            }
        ),
        label="baseline",
    )
    headroom = _exact_nested_mapping(
        baseline["headroom"],
        frozenset({"definition", "minimum_absolute_headroom", "radius", "rule"}),
        label="baseline.headroom",
    )
    oracle = _exact_nested_mapping(
        baseline["oracle"],
        frozenset({"role", "selectable"}),
        label="baseline.oracle",
    )
    validation_support = _exact_nested_mapping(
        baseline["validation_support"],
        frozenset(
            {
                "aggregate_regression_clusters",
                "aggregate_regression_per_stratum",
                "crossing_negative_clusters",
                "crossing_negative_per_stratum_minimum",
                "crossing_positive_clusters",
                "crossing_positive_per_stratum_minimum",
            }
        ),
        label="baseline.validation_support",
    )
    _exact_nested_mapping(
        baseline["future_candidate_margin"],
        frozenset({"crossing_and_coverage_are_guard_metrics", "error_metrics", "rule"}),
        label="baseline.future_candidate_margin",
    )
    if (
        headroom["minimum_absolute_headroom"] != 0.02
        or oracle["selectable"] is not False
        or dict(validation_support)
        != {
            "aggregate_regression_clusters": 36,
            "aggregate_regression_per_stratum": 3,
            "crossing_negative_clusters": 18,
            "crossing_negative_per_stratum_minimum": 1,
            "crossing_positive_clusters": 18,
            "crossing_positive_per_stratum_minimum": 1,
        }
        or tuple(baseline["simple_comparator_selection"])
        != (
            "higher whole-sample coverage",
            "lower cluster-macro normalized MAE",
            "persistence on exact tie",
        )
    ):
        raise QualificationPolicyError("baseline gate design drifts")

    expected_permissions = {
        "canonical_generation_allowed": False,
        "final_set_access_allowed": False,
        "hmc_is_sole_actuator_authority": True,
        "learned_action_authority_allowed": False,
        "model_training_allowed": False,
        "pilot_generation_allowed": False,
        "proceed_requires_separate_experiment_freeze": True,
        "publication_allowed": False,
        "scenario_generation_allowed": False,
        "validation_access_allowed": False,
    }
    if dict(policy.permissions) != expected_permissions:
        raise QualificationPolicyError("qualification authority boundary drifts")
    if dict(policy.stop_precedence) != {
        "baseline": (
            "STOP_EVIDENCE_INVALID",
            "STOP_VALIDATION_ACCESS_INVALID",
            "STOP_UNDERPOWERED",
            "STOP_NO_ACTION_INFORMATION",
            "STOP_NO_DEFENSIBLE_HEADROOM",
            "PROCEED_TO_EXPERIMENT_FREEZE",
        ),
        "resource_preflight_failure": "REFUSE_BEFORE_GENERATION_WITHOUT_SCIENTIFIC_OUTCOME",
        "timing": (
            "STOP_POLICY_UNRATIFIED",
            "STOP_EVIDENCE_INVALID",
            "STOP_UNDERPOWERED",
            "STOP_NO_DEFENSIBLE_TIMING",
            "SELECTED",
        ),
    }:
        raise QualificationPolicyError("stop precedence drifts")


def require_approved_policy(policy: QualificationPolicy) -> QualificationPolicy:
    """Return the exact policy only when its reviewed ratification gate is open."""

    if type(policy) is not QualificationPolicy:
        raise QualificationPolicyError("qualification requires the exact policy type")
    if policy.ratification_status != "APPROVED":
        raise QualificationPolicyError("qualification policy is not approved")
    unknown_permissions = sorted(set(policy.permissions) - _PERMISSION_FIELDS)
    missing_permissions = sorted(_PERMISSION_FIELDS - set(policy.permissions))
    if unknown_permissions or missing_permissions:
        raise QualificationPolicyError(
            "qualification policy permission contract has "
            f"unknown={unknown_permissions}, missing={missing_permissions}"
        )
    if policy.permissions.get("model_training_allowed") is not False:
        raise QualificationPolicyError(
            "qualification policy cannot authorize model training"
        )
    for field in _FORBIDDEN_APPROVED_PERMISSIONS:
        if policy.permissions.get(field) is not False:
            raise QualificationPolicyError(
                f"qualification policy grants forbidden authority: {field}"
            )
    for field in _REQUIRED_APPROVED_INVARIANTS:
        if policy.permissions.get(field) is not True:
            raise QualificationPolicyError(
                f"qualification policy clears required safety invariant: {field}"
            )
    if policy.policy_sha256 not in _APPROVED_POLICY_SHA256S:
        raise QualificationPolicyError(
            "qualification policy is not compiled for approval"
        )
    if policy.policy_bytes_sha256 not in _APPROVED_POLICY_BYTES_SHA256S:
        raise QualificationPolicyError(
            "qualification policy bytes are not compiled for approval"
        )
    validate_ratified_policy_design(policy)
    return policy
