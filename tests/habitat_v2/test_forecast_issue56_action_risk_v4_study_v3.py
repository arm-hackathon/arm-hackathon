"""Tests for the Issue #56 V4 protocol revision 3 study surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast_issue55_race import EPISODE_STEPS, deterministic_family_ids
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import FEATURE_COUNT
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    Issue56V3RiskError,
    V3HorizonMetric,
    V3PolicyLabel,
    V3RiskSample,
    V3_HORIZONS,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import V4RiskSample
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_features import (
    V4_TEMPORAL_FEATURE_COUNT,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model import (
    Issue56V4ModelError,
    V4_ACTION_IDS,
    V4_COMPOSITE_SELECTION_WEIGHTS,
    V4_HORIZON_KEYS,
    V4_THRESHOLD_GRID_EXTENDED,
    V4ActionScore,
    V4HorizonPrediction,
    V4ModelSample,
    V4RiskModel,
    V4RiskPrediction,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
    ISSUE56_V4_MODEL_PROTOCOL_V3_FILENAME,
    load_v4_model_protocol_v3,
    validate_v4_model_protocol_v3,
    Issue56V4ModelProtocolError,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(value: object) -> str:
    from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_protocol_v3_loads_and_rejects_drift() -> None:
    protocol, digest = load_v4_model_protocol_v3(REPO_ROOT)
    assert protocol["preregistration_id"] == "habitat_v2_forecast_issue_56_v4_model_preregistration_v3"
    assert len(digest) == 64
    assert protocol["calibration"]["threshold_grid"] == [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
    assert protocol["policy"]["selection_contract"] == "composite_point_select_v1"

    drifted = json.loads(json.dumps(protocol))
    drifted["evaluation"]["stage_a_offline"]["gates"]["minimum_useful_action_count"] = 8
    with pytest.raises(Issue56V4ModelProtocolError, match="stage A gates"):
        validate_v4_model_protocol_v3(drifted)

    drifted_identity = json.loads(json.dumps(protocol))
    drifted_identity["preregistration_id"] = "not-the-protocol"
    with pytest.raises(Issue56V4ModelProtocolError, match="identity"):
        validate_v4_model_protocol_v3(drifted_identity)

    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V3_FILENAME).is_file()


def _make_sample(index: int, split: str, action_index: int) -> V4RiskSample:
    family = f"{split.lower()}-{index % 2}"
    value = float(index)
    event_pattern = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    )[index % 10]
    metrics = tuple(
        V3HorizonMetric(horizon, event_pattern[horizon_index], 0.0, 0.0)
        for horizon_index, horizon in enumerate(V3_HORIZONS)
    )
    remaining = V3HorizonMetric(EPISODE_STEPS - 16, event_pattern[3], 0.0, 0.0)
    action_id = V4_ACTION_IDS[action_index]
    action_bytes = f"action-trace-{split}-{index}-{action_index}".encode("ascii")
    hold_bytes = f"hold-trace-{split}-{index}".encode("ascii")
    action_sha = hashlib.sha256(action_bytes).hexdigest()
    hold_sha = hashlib.sha256(hold_bytes).hexdigest()
    state_digests = tuple("a" * 64 for _ in range(EPISODE_STEPS - 16))
    label_body = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v3_v2.label",
        "track": "hmc_persistent_remaining",
        "action_id": action_id,
        "decision_step": 16,
        "current_command_sha256": "b" * 64,
        "requested_command_sha256": "c" * 64,
        "final_command_sha256": "d" * 64,
        "executed_command_sha256": "d" * 64,
        "disposition": "PROPOSED_ACCEPTED",
        "horizon_metrics": [metric.to_mapping() for metric in metrics],
        "remaining_steps": EPISODE_STEPS - 16,
        "remaining_metric": remaining.to_mapping(),
        "state_digests": list(state_digests),
        "trace_sha256": action_sha,
    }
    label = V3PolicyLabel(
        action_id,
        16,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "d" * 64,
        "PROPOSED_ACCEPTED",
        metrics,
        EPISODE_STEPS - 16,
        remaining,
        state_digests,
        action_sha,
        _digest(label_body),
    )
    features = np.full(FEATURE_COUNT, value + action_index * 0.25, dtype=np.float32)
    base_body = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v3_v2.sample",
        "family_id": family,
        "decision_step": 16,
        "split": split,
        "action_id": action_id,
        "scenario_sha256": "f" * 64,
        "features_f32_hex": features.tobytes().hex(),
        "label": label.to_mapping(),
    }
    base_sample = V3RiskSample(
        family,
        16,
        split,
        action_id,
        "f" * 64,
        features,
        label,
        _digest(base_body),
    )
    safety_exposure = float(action_index)
    hold_safety = 1.5
    comfort = float(action_index) * 10.0
    trajectory = {
        "safety_exposure": safety_exposure,
        "safety_violation_steps": 0,
        "comfort_deviation": comfort,
        "resource_composite": 0.0,
    }
    hold_trajectory = {
        "safety_exposure": hold_safety,
        "safety_violation_steps": 0,
        "comfort_deviation": 0.0,
        "resource_composite": 0.0,
    }
    relative = {
        "safety_exposure_delta_vs_hold": safety_exposure - hold_safety,
        "comfort_deviation_delta_vs_hold": comfort,
        "resource_composite_delta_vs_hold": 0.0,
    }
    mapping = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v4_corpus_v4.sample",
        "base_sample": base_sample.to_mapping(),
        "counterfactual_trace_relative_path": f"counterfactual-traces/{action_sha}.json",
        "counterfactual_trace_sha256": action_sha,
        "hold_trace_relative_path": f"counterfactual-traces/{hold_sha}.json",
        "hold_trace_sha256": hold_sha,
        "temporal_features_f32_hex": np.resize(
            features, (V4_TEMPORAL_FEATURE_COUNT,)
        ).tobytes().hex(),
        "observable_action_mask": [True, True, True, True],
        "trajectory_metrics": trajectory,
        "hold_trajectory_metrics": hold_trajectory,
        "relative_action_targets": relative,
    }
    mapping["sample_sha256"] = _digest(mapping)
    return V4RiskSample.from_mapping(mapping, action_bytes, hold_bytes)


def _dataset(split: str, count: int) -> tuple[V4RiskSample, ...]:
    samples = []
    for index in range(count):
        for action_index in range(len(V4_ACTION_IDS)):
            samples.append(_make_sample(index, split, action_index))
    return tuple(samples)


def test_action_conditioned_candidate_learns_per_action_relative_heads() -> None:
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    model = V4RiskModel.fit(train, candidate_id="c5_action_conditioned_ridge").calibrate(
        validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED
    )
    assert model.model_kind == "action_conditioned_ridge"
    assert model.relative_action_coefficients is not None
    assert model.relative_action_coefficients.shape == (4, 3, FEATURE_COUNT)

    features = np.asarray(train[0].features_f32, dtype=np.float32)
    relative_by_action = []
    for action_index in range(len(V4_ACTION_IDS)):
        prediction = model.predict_features(features, action_index=action_index)
        relative_by_action.append(prediction.relative_safety_exposure)
    assert len(set(round(value, 6) for value in relative_by_action)) >= 2

    restored = V4RiskModel.from_mapping(model.to_mapping())
    assert restored.to_mapping() == model.to_mapping()
    model_sample = V4ModelSample.from_verified(train[0])
    assert model.predict_features(
        model_sample.features_f32, action_index=0
    ).horizons[-1].horizon_steps == V4_HORIZON_KEYS[-1]

    with pytest.raises(Issue56V4ModelError, match="action index"):
        model.predict_features(features)


def test_action_conditioned_model_head_validation() -> None:
    from dataclasses import replace

    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    grid = V4_THRESHOLD_GRID_EXTENDED
    reference = V4RiskModel.fit(train, candidate_id="c0_v3_refit").calibrate(
        validation, threshold_grid=grid
    )
    assert reference.relative_action_coefficients is None
    with pytest.raises(Issue56V4ModelError, match="unexpected per-action heads"):
        replace(reference, relative_action_coefficients=np.zeros((4, 3, FEATURE_COUNT)))
    conditioned = V4RiskModel.fit(train, candidate_id="c5_action_conditioned_ridge").calibrate(
        validation, threshold_grid=grid
    )
    with pytest.raises(Issue56V4ModelError, match="lacks per-action heads"):
        replace(conditioned, relative_action_coefficients=None)


def _prediction(relative_safety: float, relative_comfort: float = 0.0) -> V4RiskPrediction:
    horizons = tuple(
        V4HorizonPrediction(horizon, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for horizon in V4_HORIZON_KEYS
    )
    return V4RiskPrediction(
        horizons,
        0.0,
        0.0,
        0.0,
        0.0,
        relative_safety,
        relative_comfort,
        0.0,
        relative_safety,
        relative_comfort,
        0.0,
        0.0,
        False,
        None,
    )


def _score(action_index: int, prediction: V4RiskPrediction, utility: float) -> V4ActionScore:
    return V4ActionScore(
        V4_ACTION_IDS[action_index],
        action_index,
        True,
        prediction.hard_ineligible,
        utility,
        0.0,
        prediction,
        None,
    )


def test_select_action_composite_requires_predicted_improvement() -> None:
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    model = V4RiskModel.fit(train, candidate_id="c5_action_conditioned_ridge").calibrate(
        validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED
    )

    no_improvement = [
        _score(index, _prediction(relative_safety=0.5), 0.5)
        for index in range(len(V4_ACTION_IDS))
    ]
    assert model.select_action_composite(no_improvement) is None

    improving = [
        _score(index, _prediction(relative_safety=float(index) - 2.0), float(index) - 2.0)
        for index in range(len(V4_ACTION_IDS))
    ]
    selected = model.select_action_composite(improving)
    assert selected is not None
    assert selected.action_index == 0

    tie = [
        _score(2, _prediction(relative_safety=-1.0), -1.0),
        _score(3, _prediction(relative_safety=-1.0), -1.0),
    ]
    assert model.select_action_composite(tie).action_index == 2

    weights = V4_COMPOSITE_SELECTION_WEIGHTS
    comfort_wins = [
        _score(
            0,
            _prediction(relative_safety=-1.0, relative_comfort=100.0),
            -1.0 * weights["safety_exposure"] + 100.0 * weights["comfort_deviation"],
        ),
        _score(
            1,
            _prediction(relative_safety=-1.5, relative_comfort=0.0),
            -1.5 * weights["safety_exposure"],
        ),
    ]
    assert model.select_action_composite(comfort_wins).action_index == 1


def test_v3_episode_record_accepts_v4_arm_name() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
        V4_MODEL_ARM,
        V4_MODEL_SOURCE_TYPE,
    )

    assert V4_MODEL_ARM == "risk_v4_model_common_window"
    assert V4_MODEL_SOURCE_TYPE == "issue56-risk-v4-model"
    roster = deterministic_family_ids(32)
    assert len(roster) == 32


def test_v3_module_rejects_invalid_v4_arm_model() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
        V4_MODEL_ARM,
        run_v3_episode,
    )

    bundle = load_forecast_contracts(REPO_ROOT)
    from aeolus.habitat_v2.forecast_issue55_race import build_family_scenario

    scenario = build_family_scenario(bundle.development_scenario, 2)
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    v3_model = _minimal_v3_model(train, validation)
    with pytest.raises(Issue56V3RiskError, match="calibrated V4 model"):
        run_v3_episode(
            bundle,
            scenario,
            V4_MODEL_ARM,
            deterministic_family_ids(32)[2],
            2,
            v3_model,
            None,
            v4_model=None,
        )


def _minimal_v3_model(train, validation):
    from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import V3RiskModel

    base = [item.base_sample for item in train]
    base_validation = [item.base_sample for item in validation]
    return V3RiskModel.fit(base).calibrate(base_validation)


def test_c7_action_conditioned_cumulative_fits_and_round_trips() -> None:
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    model = V4RiskModel.fit(train, candidate_id="c7_action_conditioned_cumulative").calibrate(
        validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED
    )
    assert model.model_kind == "action_conditioned_ridge"
    assert model.hazard_mode == "cumulative_logistic"
    assert model.relative_action_coefficients is not None
    assert model.relative_action_coefficients.shape == (4, 3, FEATURE_COUNT)
    restored = V4RiskModel.from_mapping(model.to_mapping())
    assert restored.to_mapping() == model.to_mapping()
    features = train[0].features_f32
    relative = [
        model.predict_features(features, action_index=index).relative_safety_exposure
        for index in range(4)
    ]
    assert len({round(value, 6) for value in relative}) >= 2


def test_protocol_v4_loads_and_revises_stage_b_rule() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V4_FILENAME,
        load_v4_model_protocol_v4,
        validate_v4_model_protocol_v4,
    )

    protocol, digest = load_v4_model_protocol_v4(REPO_ROOT)
    assert protocol["preregistration_id"] == "habitat_v2_forecast_issue_56_v4_model_preregistration_v4"
    assert len(digest) == 64
    assert protocol["evaluation"]["stage_b_hmc_replay"]["stage_b_candidate_rule"] == (
        "stage_a_passer_else_best_safety_passing_usefulness"
    )
    candidate_ids = [item["id"] for item in protocol["candidate_models"]]
    assert candidate_ids == [
        "c0_v3_refit",
        "c5_action_conditioned_ridge",
        "c6_action_conditioned_temporal",
        "c7_action_conditioned_cumulative",
    ]
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V4_FILENAME).is_file()

    drifted = json.loads(json.dumps(protocol))
    drifted["evaluation"]["stage_b_hmc_replay"]["stage_b_candidate_rule"] = "something_else"
    with pytest.raises(Issue56V4ModelProtocolError, match="stage B"):
        validate_v4_model_protocol_v4(drifted)

    drifted_candidates = json.loads(json.dumps(protocol))
    drifted_candidates["candidate_models"] = drifted_candidates["candidate_models"][:2]
    with pytest.raises(Issue56V4ModelProtocolError, match="candidate roster"):
        validate_v4_model_protocol_v4(drifted_candidates)


def test_protocol_v5_context_gated_selection() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V5_FILENAME,
        load_v4_model_protocol_v5,
        validate_v4_model_protocol_v5,
    )

    protocol, digest = load_v4_model_protocol_v5(REPO_ROOT)
    assert protocol["preregistration_id"] == "habitat_v2_forecast_issue_56_v4_model_preregistration_v5"
    assert len(digest) == 64
    assert protocol["policy"]["selection_contract"] == "context_gated_select_v1"
    assert protocol["policy"]["context_gates"]["dormant_admission"] == "nominal_o2_excess_only"
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V5_FILENAME).is_file()

    drifted = json.loads(json.dumps(protocol))
    drifted["policy"]["context_gates"]["o2_nominal_margin"] = 0.5
    with pytest.raises(Issue56V4ModelProtocolError, match="context gates"):
        validate_v4_model_protocol_v5(drifted)


def test_protocol_v6_redesigned_evaluation_split() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import v3_family_split
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V6_FILENAME,
        V4_MODEL_V6_SPLIT_PROTOCOL,
        condition_group_labels_for_split,
        load_v4_model_protocol_v6,
        v6_family_split,
        validate_v4_model_protocol_v6,
    )

    protocol, digest = load_v4_model_protocol_v6(REPO_ROOT)
    assert (
        protocol["preregistration_id"]
        == "habitat_v2_forecast_issue_56_v4_model_preregistration_v6"
    )
    assert len(digest) == 64
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V6_FILENAME).is_file()

    # The strict superiority gate is carried over unchanged from V5.
    superiority = protocol["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"]
    assert superiority == {
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "admitted_proposal_count_must_exceed_v3": True,
    }

    roster = deterministic_family_ids(32)
    split = v6_family_split(roster)
    assert protocol["population"]["split_protocol"] == V4_MODEL_V6_SPLIT_PROTOCOL
    assert protocol["population"]["condition_group_labels"] == list(
        condition_group_labels_for_split(split)
    )
    assert protocol["corpus_requirement"]["split_protocol"] == V4_MODEL_V6_SPLIT_PROTOCOL

    # Split counts are unchanged; only condition groups g0003 and g0005 swap.
    assert {
        label: sum(1 for value in split.values() if value == label)
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
    } == {"TRAIN": 20, "VALIDATION": 6, "EVALUATION": 6}
    baseline = v3_family_split(roster)
    changed = {fid for fid in roster if split[fid] != baseline[fid]}
    assert len(changed) == 4
    assert {split[fid] for fid in changed} == {"TRAIN", "EVALUATION"}
    for index in range(0, 32, 2):
        pair = roster[index : index + 2]
        assert split[pair[0]] == split[pair[1]]

    drifted = json.loads(json.dumps(protocol))
    drifted["population"]["condition_group_labels"][3] = "TRAIN"
    with pytest.raises(Issue56V4ModelProtocolError, match="split contract"):
        validate_v4_model_protocol_v6(drifted)

    drifted_gate = json.loads(json.dumps(protocol))
    drifted_gate["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"][
        "admitted_proposal_count_must_exceed_v3"
    ] = False
    with pytest.raises(Issue56V4ModelProtocolError, match="superiority"):
        validate_v4_model_protocol_v6(drifted_gate)

    with pytest.raises(Issue56V4ModelProtocolError):
        v6_family_split(roster[:-1])


def test_family_split_dispatcher_rejects_unknown_protocol() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        V4_MODEL_V3_SPLIT_PROTOCOL,
        V4_MODEL_V6_SPLIT_PROTOCOL,
        family_split_for_protocol,
    )
    from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import v3_family_split

    roster = deterministic_family_ids(32)
    assert family_split_for_protocol(V4_MODEL_V3_SPLIT_PROTOCOL, roster) == v3_family_split(roster)
    assert family_split_for_protocol(V4_MODEL_V6_SPLIT_PROTOCOL, roster) != v3_family_split(roster)
    with pytest.raises(Issue56V4ModelProtocolError, match="unknown V4 family split protocol"):
        family_split_for_protocol("not-a-split", roster)


def test_protocol_v7_revised_superiority_and_stage_a_gates() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V7_FILENAME,
        load_v4_model_protocol_v7,
        validate_v4_model_protocol_v7,
    )

    protocol, digest = load_v4_model_protocol_v7(REPO_ROOT)
    assert (
        protocol["preregistration_id"]
        == "habitat_v2_forecast_issue_56_v4_model_preregistration_v7"
    )
    assert len(digest) == 64
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V7_FILENAME).is_file()

    # The distinct-action Stage A gate is dropped; the useful gate is unchanged.
    stage_a_gates = protocol["evaluation"]["stage_a_offline"]["gates"]
    assert "minimum_distinct_selected_actions" not in stage_a_gates
    assert stage_a_gates["minimum_useful_action_count"] == 16
    assert stage_a_gates["minimum_dangerous_event_recall"] == 0.98

    # Superiority now credits equal-or-more admissions with strictly better safety.
    superiority = protocol["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"]
    assert superiority == {
        "admitted_proposal_count_must_be_at_least_v3": True,
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "safety_exposure_paired_point_difference_must_be_strictly_negative": True,
        "safety_exposure_paired_ci_upper_maximum": 0.0,
        "maximum_hmc_mismatch_count": 0,
    }

    # The redesigned v6 split is carried over unchanged.
    assert protocol["population"]["split_protocol"] == "issue56_v4_model_split_v6"
    assert protocol["corpus_requirement"]["split_protocol"] == "issue56_v4_model_split_v6"

    drifted = json.loads(json.dumps(protocol))
    drifted["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"][
        "safety_exposure_paired_point_difference_must_be_strictly_negative"
    ] = False
    with pytest.raises(Issue56V4ModelProtocolError, match="superiority"):
        validate_v4_model_protocol_v7(drifted)

    drifted_gates = json.loads(json.dumps(protocol))
    drifted_gates["evaluation"]["stage_a_offline"]["gates"]["minimum_useful_action_count"] = 8
    with pytest.raises(Issue56V4ModelProtocolError, match="stage A gates"):
        validate_v4_model_protocol_v7(drifted_gates)


def test_superiority_helper_enforces_strictly_better_safety() -> None:
    import importlib.util

    spec_path = REPO_ROOT / "scripts" / "run_action_risk_v4_study_v3.py"
    module_spec = importlib.util.spec_from_file_location("study_v3_under_test", spec_path)
    study = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(study)

    v7_spec = {
        "admitted_proposal_count_must_be_at_least_v3": True,
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "safety_exposure_paired_point_difference_must_be_strictly_negative": True,
        "safety_exposure_paired_ci_upper_maximum": 0.0,
        "maximum_hmc_mismatch_count": 0,
    }
    better = {"point_difference": -1e-05, "ci_lower": -3e-05, "ci_upper": -1e-07}
    equal = {"point_difference": 0.0, "ci_lower": 0.0, "ci_upper": 1e-06}
    assert study._superiority_over_v3(v7_spec, better, 4, 4, 0)["achieved"] is True
    assert study._superiority_over_v3(v7_spec, equal, 4, 4, 0)["achieved"] is False
    assert study._superiority_over_v3(v7_spec, better, 3, 4, 0)["achieved"] is False
    assert study._superiority_over_v3(v7_spec, better, 4, 4, 1)["achieved"] is False

    legacy_spec = {
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "admitted_proposal_count_must_exceed_v3": True,
    }
    assert study._superiority_over_v3(legacy_spec, better, 5, 4, 0)["achieved"] is True
    assert study._superiority_over_v3(legacy_spec, better, 4, 4, 0)["achieved"] is False

    v8_spec = {
        "admitted_proposal_count_must_exceed_v3": True,
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "early_intervention_alternative": {
            "admitted_proposal_count_must_be_at_least_v3": True,
            "safety_exposure_paired_point_difference_must_be_strictly_negative": True,
            "safety_exposure_paired_ci_upper_maximum": 0.0,
            "maximum_hmc_mismatch_count": 0,
        },
    }
    # Primary clause: more admissions with no-worse safety.
    assert study._superiority_over_v3(v8_spec, better, 5, 4, 0)["achieved"] is True
    # Alternative clause: equal admissions, strictly better safety, zero mismatches.
    assert study._superiority_over_v3(v8_spec, better, 4, 4, 0)["achieved"] is True
    # Neither clause: equal admissions with non-negative safety.
    assert study._superiority_over_v3(v8_spec, equal, 4, 4, 0)["achieved"] is False
    # Neither clause: fewer admissions.
    assert study._superiority_over_v3(v8_spec, better, 3, 4, 0)["achieved"] is False
    # Neither clause: mismatches break the alternative.
    assert study._superiority_over_v3(v8_spec, better, 4, 4, 1)["achieved"] is False


def test_protocol_v8_fixture_revision_split_and_superiority() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V8_FILENAME,
        V4_MODEL_V8_SPLIT_PROTOCOL,
        condition_group_labels_for_split,
        load_v4_model_protocol_v8,
        v8_family_split,
        validate_v4_model_protocol_v8,
    )

    protocol, digest = load_v4_model_protocol_v8(REPO_ROOT)
    assert (
        protocol["preregistration_id"]
        == "habitat_v2_forecast_issue_56_v4_model_preregistration_v8"
    )
    assert len(digest) == 64
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V8_FILENAME).is_file()

    roster = deterministic_family_ids(32)
    split = v8_family_split(roster)
    assert protocol["population"]["split_protocol"] == V4_MODEL_V8_SPLIT_PROTOCOL
    assert protocol["population"]["condition_group_labels"] == list(
        condition_group_labels_for_split(split)
    )
    assert protocol["corpus_requirement"]["split_protocol"] == V4_MODEL_V8_SPLIT_PROTOCOL

    # Split counts unchanged; evaluation spans three operating and plant conditions.
    assert {
        label: sum(1 for value in split.values() if value == label)
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
    } == {"TRAIN": 20, "VALIDATION": 6, "EVALUATION": 6}
    eval_groups = [
        index // 2 for index, fid in enumerate(roster) if split[fid] == "EVALUATION"
    ]
    assert sorted(set(eval_groups)) == [1, 6, 11]
    assert {group % 4 for group in (1, 6, 11)} == {1, 2, 3}
    assert {group // 4 for group in (1, 6, 11)} == {0, 1, 2}

    superiority = protocol["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"]
    assert superiority["admitted_proposal_count_must_exceed_v3"] is True
    assert "early_intervention_alternative" in superiority

    drifted = json.loads(json.dumps(protocol))
    drifted["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"][
        "early_intervention_alternative"
    ]["maximum_hmc_mismatch_count"] = 1
    with pytest.raises(Issue56V4ModelProtocolError, match="superiority"):
        validate_v4_model_protocol_v8(drifted)

    drifted_split = json.loads(json.dumps(protocol))
    drifted_split["population"]["condition_group_labels"][1] = "TRAIN"
    with pytest.raises(Issue56V4ModelProtocolError, match="split contract"):
        validate_v4_model_protocol_v8(drifted_split)


def _context_model() -> "V4RiskModel":
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    return V4RiskModel.fit(train, candidate_id="c7_action_conditioned_cumulative").calibrate(
        validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED
    )


def test_select_action_context_abstains_when_critical() -> None:
    model = _context_model()
    scores = [
        _score(index, _prediction(relative_safety=-1.0), -1.0) for index in range(4)
    ]
    context = {"critical_health": True, "nominal_o2_excess": False, "operating_mode": "occupied"}
    assert model.select_action_context(scores, context) is None


def test_select_action_context_nominal_uses_composite() -> None:
    model = _context_model()
    scores = [
        _score(0, _prediction(relative_safety=-2.0), -2.0),
        _score(1, _prediction(relative_safety=-1.0), -1.0),
        _score(2, _prediction(relative_safety=0.5), 0.5),
        _score(3, _prediction(relative_safety=1.0), 1.0),
    ]
    context = {"critical_health": False, "nominal_o2_excess": True, "operating_mode": "occupied"}
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_index == 0


def test_select_action_context_non_nominal_picks_mode_action() -> None:
    model = _context_model()
    # dormant (index 3) would be the composite favourite but context is non-nominal.
    scores = [
        _score(0, _prediction(relative_safety=0.1), 0.1),
        _score(1, _prediction(relative_safety=0.2), 0.2),
        _score(2, _prediction(relative_safety=0.3), 0.3),
        _score(3, _prediction(relative_safety=-1.0), -1.0),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": False,
        "operating_mode": "contingency",
    }
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_id == "normal-contingency-v1"

    occupied_context = {
        "critical_health": False,
        "nominal_o2_excess": False,
        "operating_mode": "occupied",
    }
    selected_occupied = model.select_action_context(scores, occupied_context)
    assert selected_occupied is not None and selected_occupied.action_id == "normal-occupied-v1"


def _c8_model() -> "V4RiskModel":
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    return V4RiskModel.fit(train, candidate_id="c8_o2_excess_guard").calibrate(
        validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED
    )


def test_select_action_context_c8_guard_proposes_dormant_on_o2_excess() -> None:
    model = _c8_model()
    scores = [
        _score(0, _prediction(relative_safety=0.5), 0.5),
        _score(1, _prediction(relative_safety=0.6), 0.6),
        _score(2, _prediction(relative_safety=0.7), 0.7),
        _score(3, _prediction(relative_safety=0.8), 0.8),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": True,
        "operating_mode": "occupied",
        "current_action_id": "normal-occupied-v1",
    }
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_id == "normal-dormant-v1"


def test_select_action_context_c8_guard_suppressed_when_already_dormant() -> None:
    model = _c8_model()
    scores = [
        _score(0, _prediction(relative_safety=-2.0), -2.0),
        _score(1, _prediction(relative_safety=0.2), 0.2),
        _score(2, _prediction(relative_safety=0.3), 0.3),
        _score(3, _prediction(relative_safety=0.1), 0.1),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": True,
        "operating_mode": "occupied",
        "current_action_id": "normal-dormant-v1",
    }
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_index == 0


def test_select_action_context_c8_guard_inactive_without_o2_excess() -> None:
    model = _c8_model()
    scores = [
        _score(0, _prediction(relative_safety=0.1), 0.1),
        _score(1, _prediction(relative_safety=0.2), 0.2),
        _score(2, _prediction(relative_safety=0.3), 0.3),
        _score(3, _prediction(relative_safety=-1.0), -1.0),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": False,
        "operating_mode": "contingency",
        "current_action_id": None,
    }
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_id == "normal-contingency-v1"


def test_protocol_v9_per_family_superiority_and_guard() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V9_FILENAME,
        V4_MODEL_V8_SPLIT_PROTOCOL,
        load_v4_model_protocol_v9,
        validate_v4_model_protocol_v9,
    )

    protocol, digest = load_v4_model_protocol_v9(REPO_ROOT)
    assert (
        protocol["preregistration_id"]
        == "habitat_v2_forecast_issue_56_v4_model_preregistration_v9"
    )
    assert len(digest) == 64
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V9_FILENAME).is_file()

    assert tuple(item["id"] for item in protocol["candidate_models"]) == (
        "c0_v3_refit",
        "c3_small_shared_mlp",
        "c8_o2_excess_guard",
    )
    stage_b = protocol["evaluation"]["stage_b_hmc_replay"]
    assert stage_b["stage_b_candidate_rule"] == "replay_all_stage_a_passers_v1"
    assert stage_b["superiority_over_v3"] == {
        "family_losses_maximum": 0,
        "family_wins_minimum": 4,
        "admitted_proposal_count_must_be_at_least_v3": True,
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "maximum_hmc_mismatch_count": 0,
    }
    assert protocol["policy"]["o2_excess_guard"] == {
        "trigger": "nominal_o2_excess",
        "action": "normal-dormant-v1",
        "model_confirmation": "none",
    }
    assert protocol["population"]["split_protocol"] == V4_MODEL_V8_SPLIT_PROTOCOL
    assert protocol["corpus_requirement"]["split_protocol"] == V4_MODEL_V8_SPLIT_PROTOCOL

    drifted = json.loads(json.dumps(protocol))
    drifted["evaluation"]["stage_b_hmc_replay"]["superiority_over_v3"]["family_wins_minimum"] = 3
    with pytest.raises(Issue56V4ModelProtocolError, match="superiority"):
        validate_v4_model_protocol_v9(drifted)

    drifted_guard = json.loads(json.dumps(protocol))
    drifted_guard["policy"]["o2_excess_guard"]["action"] = "normal-occupied-v1"
    with pytest.raises(Issue56V4ModelProtocolError, match="o2 excess guard"):
        validate_v4_model_protocol_v9(drifted_guard)


def test_superiority_helper_per_family_grading() -> None:
    import importlib.util
    import types

    spec_path = REPO_ROOT / "scripts" / "run_action_risk_v4_study_v3.py"
    module_spec = importlib.util.spec_from_file_location("study_v9_under_test", spec_path)
    study = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(study)

    v9_spec = {
        "family_losses_maximum": 0,
        "family_wins_minimum": 4,
        "admitted_proposal_count_must_be_at_least_v3": True,
        "safety_exposure_paired_point_difference_maximum": 0.0,
        "maximum_hmc_mismatch_count": 0,
    }
    families = ("fam-a", "fam-b", "fam-c", "fam-d", "fam-e", "fam-f")
    v4_exposures = (0.0001, 0.0002, 0.0003, 0.0004, 0.0006, 0.0007)
    v3_exposures = (0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007)

    def make_records(v4_list, v3_list):
        by_pair = {}
        for fid, e4, e3 in zip(families, v4_list, v3_list):
            by_pair[("risk_v4_model_common_window", fid)] = types.SimpleNamespace(
                safety_exposure=e4
            )
            by_pair[("risk_filtered_point_v3", fid)] = types.SimpleNamespace(
                safety_exposure=e3
            )
        return by_pair

    safety = {"point_difference": -0.0003, "ci_lower": -0.0005, "ci_upper": -0.0001}
    result = study._superiority_over_v3(
        v9_spec, safety, 4, 2, 0, by_pair=make_records(v4_exposures, v3_exposures),
        evaluation_ids=families,
    )
    assert result["family_wins"] == 4
    assert result["family_ties"] == 2
    assert result["family_losses"] == 0
    assert result["achieved"] is True

    losing_v4 = (0.0002, 0.0004, 0.0003, 0.0004, 0.0005, 0.0006)
    result_loss = study._superiority_over_v3(
        v9_spec, safety, 4, 2, 0, by_pair=make_records(losing_v4, v3_exposures),
        evaluation_ids=families,
    )
    assert result_loss["family_losses"] == 1
    assert result_loss["achieved"] is False

    result_few_wins = study._superiority_over_v3(
        v9_spec, safety, 4, 2, 0,
        by_pair=make_records((0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0006), v3_exposures),
        evaluation_ids=families,
    )
    assert result_few_wins["family_wins"] == 3
    assert result_few_wins["achieved"] is False

    result_mismatch = study._superiority_over_v3(
        v9_spec, safety, 4, 2, 1, by_pair=make_records(v4_exposures, v3_exposures),
        evaluation_ids=families,
    )
    assert result_mismatch["achieved"] is False


def _c9_model() -> "V4RiskModel":
    train = _dataset("TRAIN", 10)
    validation = _dataset("VALIDATION", 10)
    return V4RiskModel.fit(train, candidate_id="c9_o2_guard_statistical").calibrate(
        validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED
    )


def test_select_action_context_c9_statistical_dormant_under_eva() -> None:
    model = _c9_model()
    scores = [
        _score(0, _prediction(relative_safety=0.5), 0.5),
        _score(1, _prediction(relative_safety=0.6), 0.6),
        _score(2, _prediction(relative_safety=0.7), 0.7),
        _score(3, _prediction(relative_safety=-1.0), -1.0),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": False,
        "operating_mode": "eva_transition",
        "current_action_id": "normal-eva_transition-v1",
    }
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_id == "normal-dormant-v1"


def test_select_action_context_c9_suppressed_when_dormant_current() -> None:
    model = _c9_model()
    scores = [
        _score(0, _prediction(relative_safety=-2.0), -2.0),
        _score(1, _prediction(relative_safety=0.2), 0.2),
        _score(2, _prediction(relative_safety=0.3), 0.3),
        _score(3, _prediction(relative_safety=-1.0), -1.0),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": False,
        "operating_mode": "eva_transition",
        "current_action_id": "normal-dormant-v1",
    }
    assert model.select_action_context(scores, context) is None


def test_select_action_context_c9_guard_priority_on_o2_excess() -> None:
    model = _c9_model()
    scores = [
        _score(0, _prediction(relative_safety=-2.0), -2.0),
        _score(1, _prediction(relative_safety=0.6), 0.6),
        _score(2, _prediction(relative_safety=0.7), 0.7),
        _score(3, _prediction(relative_safety=0.9), 0.9),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": True,
        "operating_mode": "occupied",
        "current_action_id": "normal-occupied-v1",
    }
    selected = model.select_action_context(scores, context)
    assert selected is not None and selected.action_id == "normal-dormant-v1"


def test_select_action_context_c9_abstains_without_improvement() -> None:
    model = _c9_model()
    scores = [
        _score(0, _prediction(relative_safety=0.5), 0.5),
        _score(1, _prediction(relative_safety=0.6), 0.6),
        _score(2, _prediction(relative_safety=0.7), 0.7),
        _score(3, _prediction(relative_safety=0.8), 0.8),
    ]
    context = {
        "critical_health": False,
        "nominal_o2_excess": False,
        "operating_mode": "eva_transition",
        "current_action_id": "normal-eva_transition-v1",
    }
    assert model.select_action_context(scores, context) is None


def test_protocol_v10_statistical_dormant_and_roster() -> None:
    from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
        ISSUE56_V4_MODEL_PROTOCOL_V10_FILENAME,
        load_v4_model_protocol_v10,
        validate_v4_model_protocol_v10,
    )

    protocol, digest = load_v4_model_protocol_v10(REPO_ROOT)
    assert (
        protocol["preregistration_id"]
        == "habitat_v2_forecast_issue_56_v4_model_preregistration_v10"
    )
    assert len(digest) == 64
    assert (REPO_ROOT / "contracts" / ISSUE56_V4_MODEL_PROTOCOL_V10_FILENAME).is_file()

    assert tuple(item["id"] for item in protocol["candidate_models"]) == (
        "c8_o2_excess_guard",
        "c9_o2_guard_statistical",
    )
    assert protocol["policy"]["selection_contract"] == (
        "context_gated_select_v2_statistical_dormant"
    )
    assert protocol["policy"]["statistical_dormant"] == {
        "candidates": ["c9_o2_guard_statistical"],
        "admission": "calibrated_screen_and_predicted_relative_improvement",
        "repeat_suppression": "current_command_is_dormant",
        "outside_o2_excess_scope": "all_operating_modes",
    }
    assert protocol["evaluation"]["stage_b_hmc_replay"]["stage_b_candidate_rule"] == (
        "replay_all_stage_a_passers_v1"
    )

    drifted = json.loads(json.dumps(protocol))
    drifted["policy"]["statistical_dormant"]["candidates"] = []
    with pytest.raises(Issue56V4ModelProtocolError, match="statistical dormant"):
        validate_v4_model_protocol_v10(drifted)

    drifted_roster = json.loads(json.dumps(protocol))
    drifted_roster["candidate_models"] = drifted_roster["candidate_models"][:1]
    with pytest.raises(Issue56V4ModelProtocolError, match="candidate roster"):
        validate_v4_model_protocol_v10(drifted_roster)
