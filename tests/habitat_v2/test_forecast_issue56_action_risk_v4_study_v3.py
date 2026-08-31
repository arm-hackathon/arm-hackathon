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
