"""Validation for the authorized Issue #56 V4 model-study protocol."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


ISSUE56_V4_MODEL_PROTOCOL_SCHEMA_VERSION = (
    "aeolus_habitat_v2_risk_issue_56_v4_model_preregistration_v2"
)
ISSUE56_V4_MODEL_PROTOCOL_ID = "habitat_v2_forecast_issue_56_v4_model_preregistration_v2"
ISSUE56_V4_MODEL_PROTOCOL_STATUS = "AUTHORIZED_DEVELOPMENT_MODEL_STUDY"
ISSUE56_V4_MODEL_PROTOCOL_FILENAME = (
    "habitat_v2_forecast_issue_56_v4_model_preregistration_v2.json"
)
ISSUE56_V4_MODEL_PROTOCOL_V3_SCHEMA_VERSION = (
    "aeolus_habitat_v2_risk_issue_56_v4_model_preregistration_v3"
)
ISSUE56_V4_MODEL_PROTOCOL_V3_ID = "habitat_v2_forecast_issue_56_v4_model_preregistration_v3"
ISSUE56_V4_MODEL_PROTOCOL_V3_FILENAME = (
    "habitat_v2_forecast_issue_56_v4_model_preregistration_v3.json"
)
V4_MODEL_CANDIDATE_IDS = (
    "c0_v3_refit",
    "c1_shared_hazard_ridge",
    "c2_shared_hazard_temporal",
    "c3_small_shared_mlp",
    "c4_advantage_ranker",
)
V4_MODEL_V3_CANDIDATE_IDS = (
    "c0_v3_refit",
    "c5_action_conditioned_ridge",
    "c6_action_conditioned_temporal",
)
ISSUE56_V4_MODEL_PROTOCOL_V4_SCHEMA_VERSION = (
    "aeolus_habitat_v2_risk_issue_56_v4_model_preregistration_v4"
)
ISSUE56_V4_MODEL_PROTOCOL_V4_ID = "habitat_v2_forecast_issue_56_v4_model_preregistration_v4"
ISSUE56_V4_MODEL_PROTOCOL_V4_FILENAME = (
    "habitat_v2_forecast_issue_56_v4_model_preregistration_v4.json"
)
V4_MODEL_V4_CANDIDATE_IDS = (
    "c0_v3_refit",
    "c5_action_conditioned_ridge",
    "c6_action_conditioned_temporal",
    "c7_action_conditioned_cumulative",
)
V4_MODEL_V4_STAGE_B_RULE = "stage_a_passer_else_best_safety_passing_usefulness"
ISSUE56_V4_MODEL_PROTOCOL_V5_SCHEMA_VERSION = (
    "aeolus_habitat_v2_risk_issue_56_v4_model_preregistration_v5"
)
ISSUE56_V4_MODEL_PROTOCOL_V5_ID = "habitat_v2_forecast_issue_56_v4_model_preregistration_v5"
ISSUE56_V4_MODEL_PROTOCOL_V5_FILENAME = (
    "habitat_v2_forecast_issue_56_v4_model_preregistration_v5.json"
)
V4_MODEL_V5_SELECTION_CONTRACT = "context_gated_select_v1"
V4_MODEL_V5_ELIGIBILITY = "risk_screen_passed_and_context_gated"
V4_MODEL_V5_CONTEXT_GATES = {
    "critical_health_action": "abstain",
    "dormant_admission": "nominal_o2_excess_only",
    "non_nominal_fallback": "current_operating_mode_action",
    "o2_upper_bound": 0.30,
    "o2_nominal_margin": 0.015,
}
V4_MODEL_V3_STAGE_B_ARMS = (
    "rules_only_common_window",
    "point_model_common_window",
    "risk_filtered_point_v3",
    "risk_v4_model_common_window",
)
V4_MODEL_FEATURE_VARIANT_IDS = ("v3_708_past_only", "v4_temporal_past_only")
V4_MODEL_SPLIT_COUNTS = {"TRAIN": 20, "VALIDATION": 6, "EVALUATION": 6}


class Issue56V4ModelProtocolError(ValueError):
    """Raised when the authorized V4 model protocol is malformed."""


def _strict_json(raw: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Issue56V4ModelProtocolError("V4 model protocol JSON is invalid") from error
    if type(value) is not dict:
        raise Issue56V4ModelProtocolError("V4 model protocol root must be an object")
    return value


def _exact(mapping: object, fields: set[str], label: str) -> dict[str, Any]:
    if type(mapping) is not dict or set(mapping) != fields:
        raise Issue56V4ModelProtocolError(f"V4 model protocol {label} fields drift")
    return mapping


def _require_list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise Issue56V4ModelProtocolError(f"V4 model protocol {label} must be a list")
    return value


def validate_v4_model_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the authorized V4 model-study contract."""

    root = _exact(
        protocol,
        {
            "schema_version",
            "preregistration_id",
            "status",
            "authorization",
            "authority",
            "population",
            "data_contract",
            "feature_variants",
            "label_contract",
            "candidate_models",
            "training",
            "calibration",
            "policy",
            "evaluation",
            "artifact",
            "non_goals",
        },
        "root",
    )
    if (
        root["schema_version"] != ISSUE56_V4_MODEL_PROTOCOL_SCHEMA_VERSION
        or root["preregistration_id"] != ISSUE56_V4_MODEL_PROTOCOL_ID
        or root["status"] != ISSUE56_V4_MODEL_PROTOCOL_STATUS
    ):
        raise Issue56V4ModelProtocolError("V4 model protocol identity drift")

    authorization = _exact(
        root["authorization"],
        {"authorized_by_user", "authorized_at", "authorization_scope", "protected_final_suite_allowed"},
        "authorization",
    )
    if (
        authorization["authorized_by_user"] is not True
        or type(authorization["authorized_at"]) is not str
        or not authorization["authorized_at"]
        or authorization["protected_final_suite_allowed"] is not False
    ):
        raise Issue56V4ModelProtocolError("V4 authorization boundary is invalid")
    if set(_require_list(authorization["authorization_scope"], "authorization scope")) != {
        "learned_model_implementation",
        "training_on_development_data",
        "development_artifact_export",
        "development_threshold_selection",
        "opt_in_runtime_integration",
    }:
        raise Issue56V4ModelProtocolError("V4 authorization scope is incomplete")

    authority = _exact(
        root["authority"],
        {"final_command_authority", "plant_step_authority", "replay_authority", "model_actuator_authority", "v3_immutable"},
        "authority",
    )
    if (
        authority["final_command_authority"] != "HMC"
        or authority["plant_step_authority"] != "HMC"
        or authority["replay_authority"] != "HMC_and_strict_replay_validator"
        or authority["model_actuator_authority"] is not False
        or authority["v3_immutable"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 authority boundary drifted")

    population = _exact(
        root["population"],
        {
            "family_count",
            "condition_group_count",
            "families_per_condition_group",
            "statistical_unit",
            "paired_sensor_variants_stay_together",
            "splits",
            "decision_steps_per_family",
            "actions_per_decision",
            "expected_samples",
        },
        "population",
    )
    if (
        population["family_count"] != 32
        or population["condition_group_count"] != 16
        or population["families_per_condition_group"] != 2
        or population["statistical_unit"] != "condition_group"
        or population["paired_sensor_variants_stay_together"] is not True
        or population["decision_steps_per_family"] != 13
        or population["actions_per_decision"] != 4
        or population["splits"] != {
            "TRAIN": "20 families / 10 condition groups",
            "VALIDATION": "6 families / 3 condition groups",
            "EVALUATION": "6 families / 3 condition groups",
        }
        or population["expected_samples"]
        != {"TRAIN": 1040, "VALIDATION": 312, "EVALUATION": 312, "TOTAL": 1664}
    ):
        raise Issue56V4ModelProtocolError("V4 population contract drifted")

    data_contract = _exact(
        root["data_contract"],
        {
            "corpus_schema_version",
            "counterfactual_trace_bytes_required",
            "hold_trace_bytes_required",
            "relative_targets_are_action_minus_no_proposal_hold",
            "temporal_feature_manifest_required",
            "observable_action_mask_required",
            "future_states_are_labels_only",
            "runtime_projection_source",
            "prohibited_runtime_inputs",
            "required_provenance",
        },
        "data contract",
    )
    if (
        data_contract["corpus_schema_version"]
        != "aeolus_habitat_v2_risk_issue_56_v4_corpus_v4"
        or data_contract["counterfactual_trace_bytes_required"] is not True
        or data_contract["hold_trace_bytes_required"] is not True
        or data_contract["relative_targets_are_action_minus_no_proposal_hold"] is not True
        or data_contract["temporal_feature_manifest_required"] is not True
        or data_contract["observable_action_mask_required"] is not True
        or data_contract["future_states_are_labels_only"] is not True
        or data_contract["runtime_projection_source"] != "verified_complete_history_window"
    ):
        raise Issue56V4ModelProtocolError("V4 data boundary drifted")
    prohibited = set(_require_list(data_contract["prohibited_runtime_inputs"], "prohibited inputs"))
    if not {"hidden_fault_truth", "future_measurements", "hmc_arbitration_outcome"}.issubset(
        prohibited
    ):
        raise Issue56V4ModelProtocolError("V4 prohibited input boundary is incomplete")
    required_provenance = _require_list(data_contract["required_provenance"], "provenance")
    if set(required_provenance) != {
        "source_identity_sha256",
        "corpus_manifest_sha256",
        "hmc_binding_sha256",
        "hmc_contract_sha256",
        "scenario_manifest_sha256",
        "action_catalogue_sha256",
        "feature_manifest_sha256",
        "label_manifest_sha256",
    }:
        raise Issue56V4ModelProtocolError("V4 provenance contract is incomplete")

    feature_variants = _require_list(root["feature_variants"], "feature variants")
    if tuple(item.get("id") for item in feature_variants) != V4_MODEL_FEATURE_VARIANT_IDS:
        raise Issue56V4ModelProtocolError("V4 feature variants drifted")
    if any(
        type(item) is not dict
        or set(item)
        != {"id", "description", "feature_count", "include", "exclude", "mask"}
        or type(item.get("include")) is not list
        or type(item.get("exclude")) is not list
        or type(item.get("feature_count")) is not int
        or item.get("feature_count") <= 0
        or "future_measurements" not in item["exclude"]
        or "hidden_truth" not in item["exclude"]
        or "hmc_outcomes" not in item["exclude"]
        or item.get("mask") != "validated_catalogue_actions"
        for item in feature_variants
    ):
        raise Issue56V4ModelProtocolError("V4 feature leakage boundary is incomplete")
    if (
        feature_variants[0]["feature_count"] != 708
        or feature_variants[1]["feature_count"] != 1290
    ):
        raise Issue56V4ModelProtocolError("V4 feature counts drifted")

    label_contract = _exact(
        root["label_contract"],
        {"risk_track", "hazard_horizons", "required_targets", "relative_action_targets", "trace_semantics_must_match_labels"},
        "label contract",
    )
    if (
        label_contract["risk_track"] != "hmc_persistent_remaining"
        or label_contract["hazard_horizons"] != [4, 16, 32, "remaining"]
        or label_contract["required_targets"]
        != [
            "crossing_event",
            "safety_exposure",
            "maximum_crossing",
            "comfort_deviation",
            "resource_composite",
        ]
        or label_contract["relative_action_targets"]
        != [
            "safety_exposure_delta_vs_hold",
            "comfort_deviation_delta_vs_hold",
            "resource_composite_delta_vs_hold",
        ]
        or label_contract["trace_semantics_must_match_labels"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 label contract drifted")

    candidates = _require_list(root["candidate_models"], "candidate models")
    if tuple(item.get("id") for item in candidates) != V4_MODEL_CANDIDATE_IDS:
        raise Issue56V4ModelProtocolError("V4 candidate model roster drifted")
    if any(
        type(item) is not dict
        or type(item.get("kind")) is not str
        or item.get("feature_variant") not in V4_MODEL_FEATURE_VARIANT_IDS
        for item in candidates
    ):
        raise Issue56V4ModelProtocolError("V4 candidate model descriptor is invalid")

    training = _exact(
        root["training"],
        {
            "normalization_fit_split",
            "model_fit_split",
            "calibration_split",
            "evaluation_split",
            "seeds",
            "deterministic_serialization",
            "no_evaluation_threshold_tuning",
            "no_protected_data_access",
        },
        "training",
    )
    if (
        training["normalization_fit_split"] != "TRAIN"
        or training["model_fit_split"] != "TRAIN"
        or training["calibration_split"] != "VALIDATION"
        or training["evaluation_split"] != "EVALUATION"
        or training["seeds"] != [560057, 560058, 560059]
        or training["deterministic_serialization"] is not True
        or training["no_evaluation_threshold_tuning"] is not True
        or training["no_protected_data_access"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 training contract drifted")

    calibration = _exact(
        root["calibration"],
        {
            "method",
            "confidence_level",
            "threshold_selection_split",
            "threshold_grid",
            "minimum_positive_labels_per_horizon",
            "minimum_validation_decision_coverage",
        },
        "calibration",
    )
    if (
        calibration["method"] != "grouped_monotonic_hazard_and_upper_residual_quantiles"
        or calibration["confidence_level"] != 0.9
        or calibration["threshold_selection_split"] != "VALIDATION"
        or calibration["threshold_grid"] != [0.2, 0.35, 0.5, 0.65, 0.8]
        or calibration["minimum_positive_labels_per_horizon"] != 2
        or calibration["minimum_validation_decision_coverage"] != 0.2
    ):
        raise Issue56V4ModelProtocolError("V4 calibration contract drifted")

    policy = _exact(
        root["policy"],
        {
            "candidate_screening_scope",
            "selection_scope",
            "abstention_allowed",
            "hmc_compatibility_mask",
            "selection_order",
            "model_proposal_source_type",
        },
        "policy",
    )
    if (
        policy["candidate_screening_scope"] != "all_catalogue_actions"
        or policy["selection_scope"] != "compatible_safe_candidates"
        or policy["abstention_allowed"] is not True
        or policy["hmc_compatibility_mask"] != "validated_catalogue_actions"
        or policy["selection_order"]
        != ["safety_gate", "calibrated_expected_utility", "hmc_compatibility", "resource_cost", "action_id"]
        or policy["model_proposal_source_type"] != "issue56-risk-v4-model"
    ):
        raise Issue56V4ModelProtocolError("V4 policy contract drifted")

    evaluation = _exact(
        root["evaluation"],
        {"bootstrap_seed", "bootstrap_resamples", "bootstrap_unit", "required_metrics", "gates", "negative_results_are_published_unchanged", "evaluation_runs"},
        "evaluation",
    )
    if (
        evaluation["bootstrap_seed"] != 560057
        or evaluation["bootstrap_resamples"] != 10000
        or evaluation["bootstrap_unit"] != "condition_group"
        or evaluation["negative_results_are_published_unchanged"] is not True
        or evaluation["evaluation_runs"] != 1
    ):
        raise Issue56V4ModelProtocolError("V4 evaluation contract drifted")
    required_metrics = set(_require_list(evaluation["required_metrics"], "evaluation metrics"))
    if not {"useful_action_rate", "hmc_mismatch_rate", "selected_action_false_safe_rate"}.issubset(
        required_metrics
    ):
        raise Issue56V4ModelProtocolError("V4 evaluation metrics are incomplete")
    gates = _exact(
        evaluation["gates"],
        {
            "authority_violations",
            "replay_failures",
            "provenance_violations",
            "non_finite_metrics",
            "proposal_admission_failures",
            "minimum_useful_action_count",
            "minimum_distinct_selected_actions",
            "maximum_abstention_rate",
            "maximum_hmc_mismatch_rate",
            "maximum_inference_latency_p99_ms",
            "minimum_dangerous_event_recall",
        },
        "evaluation gates",
    )
    if gates != {
        "authority_violations": 0,
        "replay_failures": 0,
        "provenance_violations": 0,
        "non_finite_metrics": 0,
        "proposal_admission_failures": 0,
        "minimum_useful_action_count": 16,
        "minimum_distinct_selected_actions": 2,
        "maximum_abstention_rate": 0.8,
        "maximum_hmc_mismatch_rate": 0.1,
        "maximum_inference_latency_p99_ms": 250.0,
        "minimum_dangerous_event_recall": 0.98,
    }:
        raise Issue56V4ModelProtocolError("V4 evaluation gates drifted")

    artifact = _exact(
        root["artifact"],
        {"release_tier", "actuator_authority", "artifact_replacement_requires_separate_review", "v3_artifacts_unchanged", "runtime_integration_is_opt_in"},
        "artifact",
    )
    if (
        artifact["release_tier"] != "DEVELOPMENT_EVIDENCE_ONLY"
        or artifact["actuator_authority"] is not False
        or artifact["artifact_replacement_requires_separate_review"] is not True
        or artifact["v3_artifacts_unchanged"] is not True
        or artifact["runtime_integration_is_opt_in"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 artifact boundary drifted")
    if type(root["non_goals"]) is not list or not root["non_goals"]:
        raise Issue56V4ModelProtocolError("V4 model protocol non-goals are missing")
    return dict(root)


def load_v4_model_protocol(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load and hash the exact authorized V4 model protocol bytes."""

    path = Path(root).resolve() / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_FILENAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Issue56V4ModelProtocolError("V4 model protocol is unreadable") from error
    return validate_v4_model_protocol(_strict_json(raw)), hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Issue56V4ModelProtocolError(f"V4 protocol v3 {label} must be lowercase SHA-256")
    return value


def _validate_v4_study_protocol(
    protocol: Mapping[str, Any],
    *,
    tag: str,
    schema_version: str,
    preregistration_id: str,
    parent_evidence_fields: set[str],
    parent_results_key: str,
    candidate_ids: tuple[str, ...],
    stage_b_rule: str,
    selection_contract: str = "composite_point_select_v1",
    eligibility: str = "risk_screen_passed_and_predicted_safety_improvement",
    context_gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on an authorized V4 model-study protocol revision."""

    root = _exact(
        protocol,
        {
            "schema_version",
            "preregistration_id",
            "status",
            "authorization",
            "authority",
            "parent_evidence",
            "population",
            "corpus_requirement",
            "data_contract",
            "feature_variants",
            "label_contract",
            "candidate_models",
            "training",
            "calibration",
            "policy",
            "evaluation",
            "artifact",
            "non_goals",
        },
        f"{tag} root",
    )
    if (
        root["schema_version"] != schema_version
        or root["preregistration_id"] != preregistration_id
        or root["status"] != ISSUE56_V4_MODEL_PROTOCOL_STATUS
    ):
        raise Issue56V4ModelProtocolError(f"V4 protocol {tag} identity drift")

    authorization = _exact(
        root["authorization"],
        {"authorized_by_user", "authorized_at", "authorization_scope", "protected_final_suite_allowed"},
        "v3 authorization",
    )
    if (
        authorization["authorized_by_user"] is not True
        or type(authorization["authorized_at"]) is not str
        or not authorization["authorized_at"]
        or authorization["protected_final_suite_allowed"] is not False
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 authorization boundary is invalid")
    if set(_require_list(authorization["authorization_scope"], "v3 authorization scope")) != {
        "learned_model_implementation",
        "training_on_development_data",
        "development_artifact_export",
        "development_threshold_selection",
        "opt_in_runtime_integration",
    }:
        raise Issue56V4ModelProtocolError("V4 protocol v3 authorization scope is incomplete")

    authority = _exact(
        root["authority"],
        {"final_command_authority", "plant_step_authority", "replay_authority", "model_actuator_authority", "v3_immutable"},
        "v3 authority",
    )
    if (
        authority["final_command_authority"] != "HMC"
        or authority["plant_step_authority"] != "HMC"
        or authority["replay_authority"] != "HMC_and_strict_replay_validator"
        or authority["model_actuator_authority"] is not False
        or authority["v3_immutable"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 authority boundary drifted")

    parent_evidence = _exact(
        root["parent_evidence"],
        parent_evidence_fields,
        f"{tag} parent evidence",
    )
    _require_sha256(parent_evidence[parent_results_key], "parent results digest")
    if "oracle_ceiling_useful_decisions" in parent_evidence_fields:
        if (
            type(parent_evidence["oracle_ceiling_useful_decisions"]) is not int
            or parent_evidence["oracle_ceiling_useful_decisions"] < 16
            or type(parent_evidence["oracle_ceiling_distinct_actions"]) is not int
            or parent_evidence["oracle_ceiling_distinct_actions"] < 2
            or parent_evidence["oracle_policy"] != "composite_point_delta_argmin_over_useful_and_safe_actions"
        ):
            raise Issue56V4ModelProtocolError(f"V4 protocol {tag} parent evidence is invalid")
    if "revision_rationale" in parent_evidence_fields and (
        type(parent_evidence["revision_rationale"]) is not str
        or not parent_evidence["revision_rationale"]
    ):
        raise Issue56V4ModelProtocolError(f"V4 protocol {tag} revision rationale is missing")

    population = _exact(
        root["population"],
        {
            "family_count",
            "condition_group_count",
            "families_per_condition_group",
            "statistical_unit",
            "paired_sensor_variants_stay_together",
            "splits",
            "decision_steps_per_family",
            "actions_per_decision",
            "expected_samples",
        },
        "v3 population",
    )
    if (
        population["family_count"] != 32
        or population["condition_group_count"] != 16
        or population["families_per_condition_group"] != 2
        or population["statistical_unit"] != "condition_group"
        or population["paired_sensor_variants_stay_together"] is not True
        or population["decision_steps_per_family"] != 13
        or population["actions_per_decision"] != 4
        or population["splits"]
        != {
            "TRAIN": "20 families / 10 condition groups",
            "VALIDATION": "6 families / 3 condition groups",
            "EVALUATION": "6 families / 3 condition groups",
        }
        or population["expected_samples"]
        != {"TRAIN": 1040, "VALIDATION": 312, "EVALUATION": 312, "TOTAL": 1664}
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 population contract drifted")

    corpus_requirement = _exact(
        root["corpus_requirement"],
        {
            "corpus_schema_version",
            "corpus_preregistration_id",
            "family_count",
            "sample_counts",
            "trace_count",
            "counterfactual_trace_bytes_present",
            "hold_trace_bytes_present",
        },
        "v3 corpus requirement",
    )
    if (
        corpus_requirement["corpus_schema_version"]
        != "aeolus_habitat_v2_risk_issue_56_v4_corpus_v4"
        or corpus_requirement["corpus_preregistration_id"] != ISSUE56_V4_MODEL_PROTOCOL_ID
        or corpus_requirement["family_count"] != 32
        or corpus_requirement["sample_counts"]
        != {"TRAIN": 1040, "VALIDATION": 312, "EVALUATION": 312, "TOTAL": 1664}
        or corpus_requirement["trace_count"] != 1696
        or corpus_requirement["counterfactual_trace_bytes_present"] is not True
        or corpus_requirement["hold_trace_bytes_present"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 corpus requirement drifted")

    data_contract = _exact(
        root["data_contract"],
        {
            "corpus_schema_version",
            "counterfactual_trace_bytes_required",
            "hold_trace_bytes_required",
            "relative_targets_are_action_minus_no_proposal_hold",
            "temporal_feature_manifest_required",
            "observable_action_mask_required",
            "future_states_are_labels_only",
            "runtime_projection_source",
            "prohibited_runtime_inputs",
            "required_provenance",
        },
        "v3 data contract",
    )
    if (
        data_contract["corpus_schema_version"]
        != "aeolus_habitat_v2_risk_issue_56_v4_corpus_v4"
        or data_contract["counterfactual_trace_bytes_required"] is not True
        or data_contract["hold_trace_bytes_required"] is not True
        or data_contract["relative_targets_are_action_minus_no_proposal_hold"] is not True
        or data_contract["temporal_feature_manifest_required"] is not True
        or data_contract["observable_action_mask_required"] is not True
        or data_contract["future_states_are_labels_only"] is not True
        or data_contract["runtime_projection_source"] != "verified_complete_history_window"
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 data boundary drifted")
    prohibited = set(
        _require_list(data_contract["prohibited_runtime_inputs"], "v3 prohibited inputs")
    )
    if not {"hidden_fault_truth", "future_measurements", "hmc_arbitration_outcome"}.issubset(
        prohibited
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 prohibited input boundary is incomplete")

    feature_variants = _require_list(root["feature_variants"], "v3 feature variants")
    if tuple(item.get("id") for item in feature_variants) != V4_MODEL_FEATURE_VARIANT_IDS:
        raise Issue56V4ModelProtocolError("V4 protocol v3 feature variants drifted")
    if (
        feature_variants[0].get("feature_count") != 708
        or feature_variants[1].get("feature_count") != 1290
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 feature counts drifted")

    label_contract = _exact(
        root["label_contract"],
        {"risk_track", "hazard_horizons", "required_targets", "relative_action_targets", "trace_semantics_must_match_labels"},
        "v3 label contract",
    )
    if (
        label_contract["risk_track"] != "hmc_persistent_remaining"
        or label_contract["hazard_horizons"] != [4, 16, 32, "remaining"]
        or label_contract["relative_action_targets"]
        != [
            "safety_exposure_delta_vs_hold",
            "comfort_deviation_delta_vs_hold",
            "resource_composite_delta_vs_hold",
        ]
        or label_contract["trace_semantics_must_match_labels"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 label contract drifted")

    candidates = _require_list(root["candidate_models"], f"{tag} candidate models")
    if tuple(item.get("id") for item in candidates) != candidate_ids:
        raise Issue56V4ModelProtocolError(f"V4 protocol {tag} candidate roster drifted")
    if any(
        type(item) is not dict
        or type(item.get("kind")) is not str
        or item.get("feature_variant") not in V4_MODEL_FEATURE_VARIANT_IDS
        for item in candidates
    ):
        raise Issue56V4ModelProtocolError(f"V4 protocol {tag} candidate descriptor is invalid")

    training = _exact(
        root["training"],
        {
            "normalization_fit_split",
            "model_fit_split",
            "calibration_split",
            "evaluation_split",
            "seeds",
            "deterministic_serialization",
            "no_evaluation_threshold_tuning",
            "no_protected_data_access",
        },
        "v3 training",
    )
    if (
        training["normalization_fit_split"] != "TRAIN"
        or training["model_fit_split"] != "TRAIN"
        or training["calibration_split"] != "VALIDATION"
        or training["evaluation_split"] != "EVALUATION"
        or training["seeds"] != [560057, 560058, 560059]
        or training["deterministic_serialization"] is not True
        or training["no_evaluation_threshold_tuning"] is not True
        or training["no_protected_data_access"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 training contract drifted")

    calibration = _exact(
        root["calibration"],
        {
            "method",
            "confidence_level",
            "threshold_selection_split",
            "threshold_grid",
            "minimum_positive_labels_per_horizon",
            "minimum_validation_decision_coverage",
        },
        "v3 calibration",
    )
    if (
        calibration["method"] != "grouped_monotonic_hazard_and_upper_residual_quantiles"
        or calibration["confidence_level"] != 0.9
        or calibration["threshold_selection_split"] != "VALIDATION"
        or calibration["threshold_grid"] != [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
        or calibration["minimum_positive_labels_per_horizon"] != 2
        or calibration["minimum_validation_decision_coverage"] != 0.2
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 calibration contract drifted")

    policy_fields = {
        "selection_contract",
        "candidate_screening_scope",
        "eligibility",
        "ranking_order",
        "abstention_allowed",
        "hmc_compatibility_mask",
        "model_proposal_source_type",
    }
    if context_gates is not None:
        policy_fields.add("context_gates")
    policy = _exact(root["policy"], policy_fields, f"{tag} policy")
    if (
        policy["selection_contract"] != selection_contract
        or policy["candidate_screening_scope"] != "all_catalogue_actions"
        or policy["eligibility"] != eligibility
        or policy["ranking_order"] != ["composite_point_delta", "action_id"]
        or policy["abstention_allowed"] is not True
        or policy["hmc_compatibility_mask"] != "validated_catalogue_actions"
        or policy["model_proposal_source_type"] != "issue56-risk-v4-model"
    ):
        raise Issue56V4ModelProtocolError(f"V4 protocol {tag} policy contract drifted")
    if context_gates is not None:
        actual_context_gates = policy.get("context_gates")
        if type(actual_context_gates) is not dict or actual_context_gates != dict(context_gates):
            raise Issue56V4ModelProtocolError(f"V4 protocol {tag} context gates drifted")

    evaluation = _exact(
        root["evaluation"],
        {"stage_a_offline", "stage_b_hmc_replay", "negative_results_are_published_unchanged", "evaluation_runs"},
        "v3 evaluation",
    )
    if (
        evaluation["negative_results_are_published_unchanged"] is not True
        or evaluation["evaluation_runs"] != 1
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 evaluation contract drifted")

    stage_a = _exact(
        evaluation["stage_a_offline"],
        {"bootstrap_seed", "bootstrap_resamples", "bootstrap_unit", "required_metrics", "gates"},
        "v3 stage A",
    )
    if (
        stage_a["bootstrap_seed"] != 560057
        or stage_a["bootstrap_resamples"] != 10000
        or stage_a["bootstrap_unit"] != "condition_group"
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 stage A bootstrap drifted")
    stage_a_metrics = set(_require_list(stage_a["required_metrics"], "v3 stage A metrics"))
    if not {
        "useful_action_count",
        "distinct_selected_action_count",
        "selected_action_false_safe_rate",
        "dangerous_event_recall",
        "abstention_rate",
    }.issubset(stage_a_metrics):
        raise Issue56V4ModelProtocolError("V4 protocol v3 stage A metrics are incomplete")
    stage_a_gates = _exact(
        stage_a["gates"],
        {
            "authority_violations",
            "replay_failures",
            "provenance_violations",
            "non_finite_metrics",
            "proposal_admission_failures",
            "minimum_useful_action_count",
            "minimum_distinct_selected_actions",
            "maximum_abstention_rate",
            "maximum_inference_latency_p99_ms",
            "minimum_dangerous_event_recall",
        },
        "v3 stage A gates",
    )
    if stage_a_gates != {
        "authority_violations": 0,
        "replay_failures": 0,
        "provenance_violations": 0,
        "non_finite_metrics": 0,
        "proposal_admission_failures": 0,
        "minimum_useful_action_count": 16,
        "minimum_distinct_selected_actions": 2,
        "maximum_abstention_rate": 0.8,
        "maximum_inference_latency_p99_ms": 250.0,
        "minimum_dangerous_event_recall": 0.98,
    }:
        raise Issue56V4ModelProtocolError("V4 protocol v3 stage A gates drifted")

    stage_b = _exact(
        evaluation["stage_b_hmc_replay"],
        {
            "arms",
            "stage_b_candidate_rule",
            "v3_baseline_refit_required",
            "bootstrap_seed",
            "bootstrap_resamples",
            "gates",
            "superiority_over_v3",
        },
        "v3 stage B",
    )
    if (
        tuple(stage_b["arms"]) != V4_MODEL_V3_STAGE_B_ARMS
        or stage_b["stage_b_candidate_rule"] != stage_b_rule
        or stage_b["v3_baseline_refit_required"] is not True
        or stage_b["bootstrap_seed"] != 560057
        or stage_b["bootstrap_resamples"] != 10000
    ):
        raise Issue56V4ModelProtocolError(f"V4 protocol {tag} stage B contract drifted")
    stage_b_gates = _exact(
        stage_b["gates"],
        {
            "authority_violations",
            "replay_failures",
            "provenance_violations",
            "non_finite_metrics",
            "proposal_admission_failures",
            "safety_vs_rules_only_point_and_ci_upper_nonpositive",
            "maximum_hmc_mismatch_rate",
            "maximum_inference_latency_p99_ms",
        },
        "v3 stage B gates",
    )
    if stage_b_gates != {
        "authority_violations": 0,
        "replay_failures": 0,
        "provenance_violations": 0,
        "non_finite_metrics": 0,
        "proposal_admission_failures": 0,
        "safety_vs_rules_only_point_and_ci_upper_nonpositive": True,
        "maximum_hmc_mismatch_rate": 0.1,
        "maximum_inference_latency_p99_ms": 250.0,
    }:
        raise Issue56V4ModelProtocolError("V4 protocol v3 stage B gates drifted")
    superiority = _exact(
        stage_b["superiority_over_v3"],
        {"safety_exposure_paired_point_difference_maximum", "admitted_proposal_count_must_exceed_v3"},
        "v3 superiority",
    )
    if (
        superiority["safety_exposure_paired_point_difference_maximum"] != 0.0
        or superiority["admitted_proposal_count_must_exceed_v3"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 superiority contract drifted")

    artifact = _exact(
        root["artifact"],
        {"release_tier", "actuator_authority", "artifact_replacement_requires_separate_review", "v3_artifacts_unchanged", "runtime_integration_is_opt_in"},
        "v3 artifact",
    )
    if (
        artifact["release_tier"] != "DEVELOPMENT_EVIDENCE_ONLY"
        or artifact["actuator_authority"] is not False
        or artifact["artifact_replacement_requires_separate_review"] is not True
        or artifact["v3_artifacts_unchanged"] is not True
        or artifact["runtime_integration_is_opt_in"] is not True
    ):
        raise Issue56V4ModelProtocolError("V4 protocol v3 artifact boundary drifted")
    if type(root["non_goals"]) is not list or not root["non_goals"]:
        raise Issue56V4ModelProtocolError("V4 protocol v3 non-goals are missing")
    return dict(root)


def validate_v4_model_protocol_v3(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the authorized V4 model-study protocol revision 3."""

    return _validate_v4_study_protocol(
        protocol,
        tag="v3",
        schema_version=ISSUE56_V4_MODEL_PROTOCOL_V3_SCHEMA_VERSION,
        preregistration_id=ISSUE56_V4_MODEL_PROTOCOL_V3_ID,
        parent_evidence_fields={
            "v4_v2_results_sha256",
            "oracle_ceiling_useful_decisions",
            "oracle_ceiling_distinct_actions",
            "oracle_policy",
        },
        parent_results_key="v4_v2_results_sha256",
        candidate_ids=V4_MODEL_V3_CANDIDATE_IDS,
        stage_b_rule="lowest_index_stage_a_gate_passing_candidate",
    )


def validate_v4_model_protocol_v4(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the authorized V4 model-study protocol revision 4."""

    return _validate_v4_study_protocol(
        protocol,
        tag="v4",
        schema_version=ISSUE56_V4_MODEL_PROTOCOL_V4_SCHEMA_VERSION,
        preregistration_id=ISSUE56_V4_MODEL_PROTOCOL_V4_ID,
        parent_evidence_fields={
            "v4_v3_results_sha256",
            "oracle_ceiling_useful_decisions",
            "oracle_ceiling_distinct_actions",
            "oracle_policy",
            "revision_rationale",
        },
        parent_results_key="v4_v3_results_sha256",
        candidate_ids=V4_MODEL_V4_CANDIDATE_IDS,
        stage_b_rule=V4_MODEL_V4_STAGE_B_RULE,
    )


def load_v4_model_protocol_v3(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load and hash the exact authorized V4 model protocol revision 3 bytes."""

    path = Path(root).resolve() / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V3_FILENAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Issue56V4ModelProtocolError("V4 protocol v3 is unreadable") from error
    return validate_v4_model_protocol_v3(_strict_json(raw)), hashlib.sha256(raw).hexdigest()


def load_v4_model_protocol_v4(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load and hash the exact authorized V4 model protocol revision 4 bytes."""

    path = Path(root).resolve() / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V4_FILENAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Issue56V4ModelProtocolError("V4 protocol v4 is unreadable") from error
    return validate_v4_model_protocol_v4(_strict_json(raw)), hashlib.sha256(raw).hexdigest()


def validate_v4_model_protocol_v5(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the authorized V4 model-study protocol revision 5."""

    return _validate_v4_study_protocol(
        protocol,
        tag="v5",
        schema_version=ISSUE56_V4_MODEL_PROTOCOL_V5_SCHEMA_VERSION,
        preregistration_id=ISSUE56_V4_MODEL_PROTOCOL_V5_ID,
        parent_evidence_fields={
            "v4_v4_results_sha256",
            "revision_rationale",
        },
        parent_results_key="v4_v4_results_sha256",
        candidate_ids=V4_MODEL_V4_CANDIDATE_IDS,
        stage_b_rule=V4_MODEL_V4_STAGE_B_RULE,
        selection_contract=V4_MODEL_V5_SELECTION_CONTRACT,
        eligibility=V4_MODEL_V5_ELIGIBILITY,
        context_gates=V4_MODEL_V5_CONTEXT_GATES,
    )


def load_v4_model_protocol_v5(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load and hash the exact authorized V4 model protocol revision 5 bytes."""

    path = Path(root).resolve() / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V5_FILENAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Issue56V4ModelProtocolError("V4 protocol v5 is unreadable") from error
    return validate_v4_model_protocol_v5(_strict_json(raw)), hashlib.sha256(raw).hexdigest()


__all__ = [
    "ISSUE56_V4_MODEL_PROTOCOL_FILENAME",
    "ISSUE56_V4_MODEL_PROTOCOL_ID",
    "ISSUE56_V4_MODEL_PROTOCOL_SCHEMA_VERSION",
    "ISSUE56_V4_MODEL_PROTOCOL_STATUS",
    "ISSUE56_V4_MODEL_PROTOCOL_V3_FILENAME",
    "ISSUE56_V4_MODEL_PROTOCOL_V3_ID",
    "ISSUE56_V4_MODEL_PROTOCOL_V3_SCHEMA_VERSION",
    "ISSUE56_V4_MODEL_PROTOCOL_V4_FILENAME",
    "ISSUE56_V4_MODEL_PROTOCOL_V4_ID",
    "ISSUE56_V4_MODEL_PROTOCOL_V4_SCHEMA_VERSION",
    "ISSUE56_V4_MODEL_PROTOCOL_V5_FILENAME",
    "ISSUE56_V4_MODEL_PROTOCOL_V5_ID",
    "ISSUE56_V4_MODEL_PROTOCOL_V5_SCHEMA_VERSION",
    "Issue56V4ModelProtocolError",
    "V4_MODEL_CANDIDATE_IDS",
    "V4_MODEL_FEATURE_VARIANT_IDS",
    "V4_MODEL_SPLIT_COUNTS",
    "V4_MODEL_V3_CANDIDATE_IDS",
    "V4_MODEL_V3_STAGE_B_ARMS",
    "V4_MODEL_V4_CANDIDATE_IDS",
    "V4_MODEL_V4_STAGE_B_RULE",
    "V4_MODEL_V5_CONTEXT_GATES",
    "V4_MODEL_V5_ELIGIBILITY",
    "V4_MODEL_V5_SELECTION_CONTRACT",
    "load_v4_model_protocol",
    "load_v4_model_protocol_v3",
    "load_v4_model_protocol_v4",
    "load_v4_model_protocol_v5",
    "validate_v4_model_protocol",
    "validate_v4_model_protocol_v3",
    "validate_v4_model_protocol_v4",
    "validate_v4_model_protocol_v5",
]
