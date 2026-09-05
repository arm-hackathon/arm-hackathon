"""BDM-v1 benchmark and evidence contract loading, validation, and enforcement.

This module owns the machine-readable research contract frozen by Issue #70
(`contracts/habitat_v2_bdm_v1_benchmark_contract_v1.json`). It is deliberately
fail-closed: every section, field record, frozen list, and threshold marker is
checked exactly, and the enforcement helpers reject prohibited model inputs,
non-causal windows, partition/group overlap, and undeclared targets or metrics
before any downstream study (Issues #72-#77) may consume data.

Nothing in this module trains a model, generates scenarios, or accesses any
protected blind population. HMC remains the sole final-command, plant-step,
and replay authority for every study governed by this contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

BDM_V1_CONTRACT_FILENAME = "habitat_v2_bdm_v1_benchmark_contract_v1.json"
BDM_V1_CONTRACT_ID = "habitat_v2_bdm_v1_benchmark_contract_v1"
BDM_V1_CONTRACT_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_benchmark_contract_v1"
BDM_V1_CONTRACT_STATUS = "ACCEPTED_RESEARCH_CONTRACT"
BDM_V1_TBD_MARKER = "TBD_FROM_PILOT"
BDM_V1_PHYSICS_PROVENANCE_CONTRACT_ID = "habitat_v2_physics_provenance_v1"

BDM_V1_PROHIBITED_INPUT_ITEMS = (
    "hidden_physical_truth",
    "fault_labels_or_future_fault_schedules",
    "simulator_seeds",
    "internal_noise_or_bias_state",
    "future_measurements",
    "counterfactual_outcomes",
    "evaluator_only_reserve_or_audit_state",
    "future_hmc_arbitration_results",
    "undeclared_future_crew_or_environmental_loads",
)

BDM_V1_CATALOGUE_ACTIONS = (
    "normal-occupied-v1",
    "normal-eva_transition-v1",
    "normal-contingency-v1",
    "normal-dormant-v1",
)

BDM_V1_HORIZON_STEP_KEYS = (4, 16, 32)
BDM_V1_EPISODE_REMAINING_HORIZON_KEY = 0

BDM_V1_DECISION_TARGETS = (
    "crossing_event",
    "safety_exposure",
    "maximum_crossing",
    "comfort_deviation",
    "resource_composite",
)

BDM_V1_ROSTER_ARM_IDS = (
    "hmc_rules_only",
    "hold_current_command",
    "c8_o2_excess_guard",
    "c9_o2_guard_statistical",
    "action_agnostic_ridge",
    "action_conditioned_linear",
    "bdm_v1",
)

BDM_V1_LOWER_IS_BETTER_METRICS = (
    "safety_exposure",
    "maximum_crossing",
    "counterfactual_safety_exposure_error",
    "action_value_delta_error",
    "finite_catalogue_regret",
    "harmful_admission_rate",
    "comfort_deviation",
    "resource_composite",
    "resource_depletion",
    "intervention_count",
    "actuator_wear",
    "inference_latency_p99_ms",
)
BDM_V1_HIGHER_IS_BETTER_METRICS = (
    "beneficial_action_precision",
    "beneficial_action_recall",
    "useful_opportunity_recall",
    "action_ranking_quality",
    "interval_coverage_at_declared_level",
    "selected_action_diversity_within_declared_bounds",
)

BDM_V1_DECISION_METRIC_HIERARCHY = (
    "hard_admissibility_violation_counts_must_be_zero",
    "per_family_hard_safety_non_inferiority",
    "paired_aggregate_safety_benefit_or_equal_safety_with_predeclared_resource_benefit",
    "action_value_delta_error_and_finite_catalogue_regret",
    "beneficial_action_precision_and_recall",
    "resource_depletion",
    "comfort_deviation",
    "intervention_cost_and_actuator_wear",
)

BDM_V1_PARTITIONS = ("TRAIN", "DEV", "CALIBRATION", "BLIND_FINAL")
BDM_V1_GROUPING_KEYS = (
    "causal_scenario_template",
    "physical_parameter_band",
    "fault_mechanism_composition",
    "operating_schedule",
    "action_opportunity",
    "sensor_failure_bundle",
)

BDM_V1_DECISION_THRESHOLD_NAMES = (
    "safety_non_inferiority_margin",
    "action_ranking_quality_minimum",
    "finite_catalogue_regret_maximum",
    "action_value_delta_error_maximum",
    "beneficial_action_recall_minimum",
    "harmful_admission_rate_maximum",
    "interval_coverage_minimum",
    "useful_opportunity_recall_minimum",
    "maximum_abstention_on_useful_opportunities",
    "minimum_selected_action_diversity",
)

BDM_V1_STOP_CRITERIA = (
    "BDM-v1 cannot beat the action-conditioned linear baseline on action ranking",
    "improvement disappears when the deterministic guard is removed",
    "results depend mostly on one dormant/O2-boundary pattern",
    "uncertainty is badly miscalibrated under family shift",
    "apparent success comes from near-total abstention",
    "one plausible simulator assumption reverses the result",
    "leakage or family overlap is found",
    "any HMC authority or replay inconsistency appears",
    "the blind confirmation fails",
)

BDM_V1_FIELD_DTYPES = (
    "float32",
    "int32",
    "bool",
    "categorical_one_hot",
    "command_vector",
    "descriptor",
)
BDM_V1_FIELD_TIMINGS = (
    "causal_window_completed_steps",
    "static",
    "per_scoring",
    "declared_known_future",
)
BDM_V1_FIELD_MISSINGNESS = (
    "not_applicable",
    "mask_and_staleness_required",
    "mask_required",
)
BDM_V1_FIELD_OBSERVABILITY = (
    "operational_telemetry",
    "derived_from_operational_telemetry",
    "declared_static_context",
    "candidate_encoding",
)
_FIELD_RECORD_KEYS = frozenset(
    {"name", "dtype", "shape", "unit", "timing", "missingness", "observability", "provenance"}
)


class BdmV1ContractError(ValueError):
    """Raised when the BDM-v1 benchmark contract or its enforcement inputs are invalid."""


def _strict_json(raw: bytes) -> Any:
    """Parse canonical contract JSON, rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise BdmV1ContractError(f"duplicate contract key {key!r}")
            seen.add(key)
        return dict(pairs)

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise BdmV1ContractError("BDM-v1 contract is not valid JSON") from error


def _exact(mapping: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(mapping) is not dict:
        raise BdmV1ContractError(f"BDM-v1 contract {label} must be an object")
    if set(mapping) != fields:
        raise BdmV1ContractError(f"BDM-v1 contract {label} field set drifted")
    return mapping


def _require_str_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise BdmV1ContractError(f"BDM-v1 contract {label} must be a list of strings")
    return value


def _validate_field_record(record: Any, class_id: str, seen_names: set[str]) -> None:
    fields = _exact(record, _FIELD_RECORD_KEYS, f"{class_id} field record")
    name = fields["name"]
    if type(name) is not str or not name or name in seen_names:
        raise BdmV1ContractError(f"BDM-v1 contract field name {name!r} is invalid or duplicated")
    seen_names.add(name)
    for key, allowed in (
        ("dtype", BDM_V1_FIELD_DTYPES),
        ("timing", BDM_V1_FIELD_TIMINGS),
        ("missingness", BDM_V1_FIELD_MISSINGNESS),
        ("observability", BDM_V1_FIELD_OBSERVABILITY),
    ):
        if fields[key] not in allowed:
            raise BdmV1ContractError(f"BDM-v1 contract field {name!r} has invalid {key}")
    for key in ("shape", "unit", "provenance"):
        if type(fields[key]) is not str or not fields[key]:
            raise BdmV1ContractError(f"BDM-v1 contract field {name!r} has invalid {key}")


def validate_bdm_v1_benchmark_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the frozen BDM-v1 benchmark and evidence contract."""

    root = _exact(
        contract,
        {
            "schema_version",
            "contract_id",
            "status",
            "authorization",
            "design_record",
            "primary_hypothesis",
            "non_claims",
            "evidence_status_matrix",
            "input_schema",
            "prohibited_fields",
            "action_catalogue",
            "horizons",
            "labels",
            "comparison_roster",
            "metrics",
            "split_custody",
            "thresholds",
            "stop_criteria",
            "stop_response",
            "claim_boundaries",
            "provenance_bindings",
        },
        "root",
    )
    if (
        root["schema_version"] != BDM_V1_CONTRACT_SCHEMA_VERSION
        or root["contract_id"] != BDM_V1_CONTRACT_ID
        or root["status"] != BDM_V1_CONTRACT_STATUS
        or type(root["design_record"]) is not str
        or type(root["primary_hypothesis"]) is not str
        or type(root["stop_response"]) is not str
    ):
        raise BdmV1ContractError("BDM-v1 contract identity drifted")
    _require_str_list(root["non_claims"], "non_claims")

    authorization = _exact(
        root["authorization"],
        {"authorized_by", "authorized_via_issue", "authorized_at", "scope"},
        "authorization",
    )
    if (
        authorization["authorized_by"] != "repository_owner"
        or authorization["authorized_via_issue"] != 70
        or type(authorization["authorized_at"]) is not str
    ):
        raise BdmV1ContractError("BDM-v1 contract authorization drifted")
    _require_str_list(authorization["scope"], "authorization scope")

    matrix = _exact(
        root["evidence_status_matrix"],
        {"implemented", "historical", "proposed", "not_claimed"},
        "evidence status matrix",
    )
    for category in ("implemented", "historical", "proposed"):
        entries = matrix[category]
        if type(entries) is not list or not entries:
            raise BdmV1ContractError(f"BDM-v1 contract evidence category {category} is empty")
        for entry in entries:
            fields = _exact(entry, {"id", "summary", "evidence"}, f"{category} entry")
            if any(type(fields[key]) is not str or not fields[key] for key in fields):
                raise BdmV1ContractError(f"BDM-v1 contract {category} entry is malformed")
    not_claimed = _require_str_list(matrix["not_claimed"], "not_claimed")
    if not not_claimed:
        raise BdmV1ContractError("BDM-v1 contract not_claimed list is empty")

    input_schema = _exact(root["input_schema"], {"causal_window", "field_classes"}, "input schema")
    causal_window = _exact(
        input_schema["causal_window"],
        {"max_history_steps", "timing_rule", "window_unit", "future_information"},
        "causal window",
    )
    if (
        type(causal_window["max_history_steps"]) is not int
        or causal_window["max_history_steps"] <= 0
        or causal_window["timing_rule"] != "completed_telemetry_at_or_before_decision_step"
    ):
        raise BdmV1ContractError("BDM-v1 contract causal window drifted")
    field_classes = input_schema["field_classes"]
    if type(field_classes) is not list or not field_classes:
        raise BdmV1ContractError("BDM-v1 contract field classes are empty")
    seen_names: set[str] = set()
    for field_class in field_classes:
        class_fields = _exact(
            field_class, {"class_id", "description", "fields"}, "input field class"
        )
        class_id = class_fields["class_id"]
        if type(class_id) is not str or not class_id:
            raise BdmV1ContractError("BDM-v1 contract field class id is invalid")
        records = class_fields["fields"]
        if type(records) is not list or not records:
            raise BdmV1ContractError(f"BDM-v1 contract field class {class_id} has no fields")
        for record in records:
            _validate_field_record(record, class_id, seen_names)

    prohibited = _exact(
        root["prohibited_fields"], {"items", "enforcement"}, "prohibited fields"
    )
    items = _require_str_list(prohibited["items"], "prohibited items")
    if tuple(items) != BDM_V1_PROHIBITED_INPUT_ITEMS:
        raise BdmV1ContractError("BDM-v1 contract prohibited input list drifted")
    if type(prohibited["enforcement"]) is not str or not prohibited["enforcement"]:
        raise BdmV1ContractError("BDM-v1 contract prohibited enforcement drifted")

    catalogue = _exact(
        root["action_catalogue"],
        {"actions", "abstention_allowed", "catalogue_binding", "mutable_in_study"},
        "action catalogue",
    )
    actions = _require_str_list(catalogue["actions"], "catalogue actions")
    if (
        tuple(actions) != BDM_V1_CATALOGUE_ACTIONS
        or catalogue["abstention_allowed"] is not True
        or catalogue["mutable_in_study"] is not False
    ):
        raise BdmV1ContractError("BDM-v1 contract action catalogue drifted")

    horizons = _exact(
        root["horizons"],
        {"step_keys", "episode_remaining_key", "semantics", "source"},
        "horizons",
    )
    step_keys = horizons["step_keys"]
    if (
        type(step_keys) is not list
        or tuple(step_keys) != BDM_V1_HORIZON_STEP_KEYS
        or horizons["episode_remaining_key"] != BDM_V1_EPISODE_REMAINING_HORIZON_KEY
    ):
        raise BdmV1ContractError("BDM-v1 contract horizons drifted")

    labels = _exact(
        root["labels"],
        {
            "trajectory_targets",
            "decision_targets",
            "action_minus_hold_targets",
            "counterfactual_rollouts",
            "counterfactual_construction",
        },
        "labels",
    )
    decision_targets = _require_str_list(labels["decision_targets"], "decision targets")
    if (
        tuple(decision_targets) != BDM_V1_DECISION_TARGETS
        or labels["action_minus_hold_targets"] is not True
        or labels["counterfactual_rollouts"] != "labels_only_never_runtime_inputs"
    ):
        raise BdmV1ContractError("BDM-v1 contract labels drifted")

    roster = _exact(
        root["comparison_roster"], {"arms", "ablation_arms_from_issue_73"}, "comparison roster"
    )
    arms = roster["arms"]
    if type(arms) is not list or len(arms) != len(BDM_V1_ROSTER_ARM_IDS):
        raise BdmV1ContractError("BDM-v1 contract roster arms drifted")
    for arm, expected_id in zip(arms, BDM_V1_ROSTER_ARM_IDS):
        arm_fields = _exact(
            arm,
            {"arm_id", "kind", "source", "advisory_behind_hmc"},
            "roster arm",
        )
        if arm_fields["arm_id"] != expected_id or type(arm_fields["kind"]) is not str:
            raise BdmV1ContractError("BDM-v1 contract roster arms drifted")
    _require_str_list(roster["ablation_arms_from_issue_73"], "ablation arms")

    metrics = _exact(
        root["metrics"],
        {
            "polarity",
            "decision_metric_hierarchy",
            "aggregate_forecast_error_status",
            "independent_statistical_unit",
            "non_independent_units",
            "pairing_rule",
            "bootstrap",
        },
        "metrics",
    )
    polarity = _exact(
        metrics["polarity"], {"lower_is_better", "higher_is_better"}, "metric polarity"
    )
    lower = _require_str_list(polarity["lower_is_better"], "lower_is_better")
    higher = _require_str_list(polarity["higher_is_better"], "higher_is_better")
    hierarchy = _require_str_list(
        metrics["decision_metric_hierarchy"], "decision metric hierarchy"
    )
    non_independent = _require_str_list(
        metrics["non_independent_units"], "non-independent units"
    )
    bootstrap = _exact(
        metrics["bootstrap"], {"unit", "seed_convention", "resamples_default"}, "bootstrap"
    )
    if (
        tuple(lower) != BDM_V1_LOWER_IS_BETTER_METRICS
        or tuple(higher) != BDM_V1_HIGHER_IS_BETTER_METRICS
        or tuple(hierarchy) != BDM_V1_DECISION_METRIC_HIERARCHY
        or metrics["independent_statistical_unit"] != "causal_group"
        or bootstrap["unit"] != "causal_group"
        or type(bootstrap["resamples_default"]) is not int
        or set(non_independent)
        != {"paired_sensor_variants", "counterfactual_action_branches", "decision_steps_within_one_family"}
    ):
        raise BdmV1ContractError("BDM-v1 contract metrics drifted")

    custody = _exact(
        root["split_custody"],
        {
            "partitions",
            "group_disjoint",
            "grouping_keys",
            "same_group_rule",
            "sealed_manifest_rule",
            "blind_size_rule",
            "calibration_separation",
        },
        "split custody",
    )
    partitions = _exact(
        custody["partitions"], set(BDM_V1_PARTITIONS), "custody partitions"
    )
    grouping_keys = _require_str_list(custody["grouping_keys"], "grouping keys")
    if (
        custody["group_disjoint"] is not True
        or tuple(grouping_keys) != BDM_V1_GROUPING_KEYS
        or any(type(use) is not str or not use for use in partitions.values())
    ):
        raise BdmV1ContractError("BDM-v1 contract split custody drifted")

    thresholds = _exact(
        root["thresholds"],
        {"model_path_latency_p99_ms_maximum", "decision_thresholds", "pilot_definition"},
        "thresholds",
    )
    if thresholds["model_path_latency_p99_ms_maximum"] != 250.0:
        raise BdmV1ContractError("BDM-v1 contract latency ceiling drifted")
    decision_thresholds = _exact(
        thresholds["decision_thresholds"],
        set(BDM_V1_DECISION_THRESHOLD_NAMES),
        "decision thresholds",
    )
    for name, value in decision_thresholds.items():
        if value == BDM_V1_TBD_MARKER:
            continue
        if type(value) is bool or not isinstance(value, (int, float)):
            raise BdmV1ContractError(
                f"BDM-v1 contract threshold {name} must be numeric or {BDM_V1_TBD_MARKER}"
            )
    pilot = _exact(
        thresholds["pilot_definition"],
        {"data", "freeze_rule", "no_result_derived_value_may_be_marked_frozen"},
        "pilot definition",
    )
    if (
        pilot["data"] != "DEV_partition_only"
        or pilot["no_result_derived_value_may_be_marked_frozen"] is not True
    ):
        raise BdmV1ContractError("BDM-v1 contract pilot definition drifted")

    stop_criteria = _require_str_list(root["stop_criteria"], "stop criteria")
    if tuple(stop_criteria) != BDM_V1_STOP_CRITERIA:
        raise BdmV1ContractError("BDM-v1 contract stop criteria drifted")

    claim_boundaries = _require_str_list(root["claim_boundaries"], "claim boundaries")
    if not claim_boundaries or not any(
        "advisory-only" in item for item in claim_boundaries
    ):
        raise BdmV1ContractError("BDM-v1 contract claim boundaries drifted")

    bindings = _exact(
        root["provenance_bindings"],
        {
            "physics_provenance_contract_id",
            "scenario_schema_versions",
            "trace_schema_version",
            "reviewed_hmc_package",
            "forecast_fixture_binding",
            "issue_56_reference",
        },
        "provenance bindings",
    )
    issue_56 = _exact(
        bindings["issue_56_reference"],
        {"final_protocol", "guard_candidate", "hybrid_candidate"},
        "issue 56 reference",
    )
    if (
        bindings["physics_provenance_contract_id"] != BDM_V1_PHYSICS_PROVENANCE_CONTRACT_ID
        or issue_56["guard_candidate"] != "c8_o2_excess_guard"
        or issue_56["hybrid_candidate"] != "c9_o2_guard_statistical"
    ):
        raise BdmV1ContractError("BDM-v1 contract provenance bindings drifted")
    return dict(root)


def load_bdm_v1_benchmark_contract(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load, validate, and hash the frozen BDM-v1 benchmark contract bytes."""

    path = Path(root).resolve() / "contracts" / BDM_V1_CONTRACT_FILENAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BdmV1ContractError("BDM-v1 benchmark contract is unreadable") from error
    return validate_bdm_v1_benchmark_contract(_strict_json(raw)), hashlib.sha256(raw).hexdigest()


def declared_input_field_names(contract: Mapping[str, Any]) -> frozenset[str]:
    """Return every declared model-facing input field name in the contract."""

    names: set[str] = set()
    for field_class in contract["input_schema"]["field_classes"]:
        for record in field_class["fields"]:
            names.add(record["name"])
    return frozenset(names)


def validate_model_input_fields(
    field_names: Iterable[str], contract: Mapping[str, Any]
) -> None:
    """Reject prohibited or undeclared model-facing input fields."""

    declared = declared_input_field_names(contract)
    prohibited = set(contract["prohibited_fields"]["items"])
    for name in field_names:
        if name in prohibited:
            raise BdmV1ContractError(f"prohibited model input field {name!r}")
        if name not in declared:
            raise BdmV1ContractError(f"undeclared model input field {name!r}")


def validate_causal_window(
    window_steps: Sequence[int], decision_step: int, contract: Mapping[str, Any]
) -> None:
    """Reject non-causal or over-long observation windows."""

    if type(decision_step) is not int or decision_step < 0:
        raise BdmV1ContractError("decision step must be a non-negative integer")
    if not window_steps:
        raise BdmV1ContractError("causal window must contain at least one completed step")
    max_history = int(contract["input_schema"]["causal_window"]["max_history_steps"])
    if len(window_steps) > max_history:
        raise BdmV1ContractError(
            f"causal window exceeds {max_history} completed steps"
        )
    earliest_allowed = decision_step - max_history + 1
    for step in window_steps:
        if type(step) is not int:
            raise BdmV1ContractError("causal window steps must be integers")
        if step > decision_step:
            raise BdmV1ContractError(
                f"causal window contains future step {step} after decision step {decision_step}"
            )
        if step < earliest_allowed:
            raise BdmV1ContractError(
                f"causal window step {step} falls outside the declared history bound"
            )


def validate_group_disjointness(partition_groups: Mapping[str, Iterable[str]]) -> None:
    """Reject partition overlap or duplicated causal groups across custody partitions."""

    unknown = set(partition_groups) - set(BDM_V1_PARTITIONS)
    if unknown:
        raise BdmV1ContractError(f"unknown custody partitions {sorted(unknown)}")
    seen: dict[str, str] = {}
    for partition in BDM_V1_PARTITIONS:
        for group_id in partition_groups.get(partition, ()):
            if type(group_id) is not str or not group_id:
                raise BdmV1ContractError("custody group ids must be non-empty strings")
            if group_id in seen:
                raise BdmV1ContractError(
                    f"causal group {group_id!r} appears in both {seen[group_id]} and {partition}"
                )
            seen[group_id] = partition


def validate_targets_declared(target_names: Iterable[str], contract: Mapping[str, Any]) -> None:
    """Reject undeclared label targets."""

    allowed = set(contract["labels"]["decision_targets"]) | {"trajectory_targets"}
    for name in target_names:
        if name not in allowed:
            raise BdmV1ContractError(f"undeclared label target {name!r}")


def validate_metrics_declared(metric_names: Iterable[str], contract: Mapping[str, Any]) -> None:
    """Reject undeclared or polarity-less metrics."""

    polarity = contract["metrics"]["polarity"]
    allowed = set(polarity["lower_is_better"]) | set(polarity["higher_is_better"])
    for name in metric_names:
        if name not in allowed:
            raise BdmV1ContractError(f"undeclared metric {name!r}")


def threshold_is_frozen(contract: Mapping[str, Any], threshold_name: str) -> bool:
    """Report whether a decision threshold has a frozen numeric value."""

    if threshold_name not in BDM_V1_DECISION_THRESHOLD_NAMES:
        raise BdmV1ContractError(f"unknown decision threshold {threshold_name!r}")
    value = contract["thresholds"]["decision_thresholds"][threshold_name]
    return value != BDM_V1_TBD_MARKER


__all__ = [
    "BDM_V1_CATALOGUE_ACTIONS",
    "BDM_V1_CONTRACT_FILENAME",
    "BDM_V1_CONTRACT_ID",
    "BDM_V1_CONTRACT_SCHEMA_VERSION",
    "BDM_V1_CONTRACT_STATUS",
    "BDM_V1_DECISION_TARGETS",
    "BDM_V1_DECISION_THRESHOLD_NAMES",
    "BDM_V1_GROUPING_KEYS",
    "BDM_V1_HORIZON_STEP_KEYS",
    "BDM_V1_EPISODE_REMAINING_HORIZON_KEY",
    "BDM_V1_LOWER_IS_BETTER_METRICS",
    "BDM_V1_HIGHER_IS_BETTER_METRICS",
    "BDM_V1_DECISION_METRIC_HIERARCHY",
    "BDM_V1_PARTITIONS",
    "BDM_V1_PHYSICS_PROVENANCE_CONTRACT_ID",
    "BDM_V1_PROHIBITED_INPUT_ITEMS",
    "BDM_V1_ROSTER_ARM_IDS",
    "BDM_V1_STOP_CRITERIA",
    "BDM_V1_TBD_MARKER",
    "BdmV1ContractError",
    "declared_input_field_names",
    "load_bdm_v1_benchmark_contract",
    "threshold_is_frozen",
    "validate_bdm_v1_benchmark_contract",
    "validate_causal_window",
    "validate_group_disjointness",
    "validate_metrics_declared",
    "validate_model_input_fields",
    "validate_targets_declared",
]
