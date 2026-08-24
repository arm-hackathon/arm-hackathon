from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import ForecastContracts, load_forecast_contracts
from aeolus.habitat_v2.forecast.projection import ForecastHistory, ForecastLayout
from aeolus.habitat_v2.forecast_issue55_race import build_family_scenario
from aeolus.habitat_v2.forecast_issue56_action_risk import (
    ACTION_COUNT,
    FEATURE_COUNT,
    HARD_CROSSING_PROBABILITY_LIMIT,
    HARD_EXPOSURE_LIMIT,
    RISK_HORIZON_STEPS,
    ActionRiskModel,
    ActionRiskSample,
    Issue56RiskError,
    RiskLabel,
    RiskScore,
    build_risk_proposal,
    counterfactual_risk_label,
    family_split,
    feature_vector,
    rank_action_risk,
    risk_decision_steps,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics import initial_state


REPO_ROOT = Path(__file__).resolve().parents[2]


def _history(value: float = 0.0) -> ForecastHistory:
    numeric = np.full((16, 194), value, dtype=np.float32)
    numeric[:, 167:] = 0.25
    status = np.zeros((16, 167, 5), dtype=np.float32)
    status[:, :, 0] = 1.0
    return ForecastHistory(
        tuple(range(1, 17)),
        tuple(float(step * 60) for step in range(1, 17)),
        numeric,
        status,
        np.zeros((16, 4), dtype=np.float32),
        np.zeros((16, 4), dtype=np.float32),
        np.zeros((16, 287, 4), dtype=np.float32),
        ForecastLayout(
            "a" * 64,
            (),
            (),
            (),
            "b" * 64,
            "c" * 64,
        ),
    )


def _label(action_id: str, exposure: float, event: float) -> RiskLabel:
    targets = np.zeros((RISK_HORIZON_STEPS, 51), dtype=np.float32)
    state_digests = tuple("d" * 64 for _ in range(RISK_HORIZON_STEPS))
    body = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v1.label",
        "action_id": action_id,
        "decision_step": 16,
        "targets": targets.tolist(),
        "state_digests": list(state_digests),
        "eligible": True,
        "termination_reason": None,
        "crossing_event": event,
        "safety_exposure": exposure,
        "maximum_crossing": exposure,
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return RiskLabel(
        action_id,
        16,
        targets,
        state_digests,
        True,
        None,
        event,
        exposure,
        exposure,
        digest,
    )


def _sample(family_id: str, value: float, exposure: float, event: float) -> ActionRiskSample:
    history = np.full((16, 194), value, dtype=np.float32)
    history[:, 167:] = 0.25 + value
    action = np.full(ACTION_COUNT, 0.25 + value, dtype=np.float32)
    label = _label("action", exposure, event)
    snapshot_ids = tuple("e" * 64 for _ in range(16))
    payload = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v1.sample",
        "family_id": family_id,
        "decision_step": 16,
        "split": "TRAIN",
        "action_id": "action",
        "scenario_sha256": "f" * 64,
        "history_steps": list(range(1, 17)),
        "snapshot_sha256": list(snapshot_ids),
        "input_manifest_sha256": "a" * 64,
        "target_manifest_sha256": "b" * 64,
        "history_numeric": history.tobytes().hex(),
        "action": action.tobytes().hex(),
        "crossing_event": event,
        "safety_exposure": exposure,
        "maximum_crossing": exposure,
        "label_sha256": label.label_sha256,
    }
    sample_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return ActionRiskSample(
        family_id,
        16,
        "TRAIN",
        "action",
        "f" * 64,
        tuple(range(1, 17)),
        snapshot_ids,
        "a" * 64,
        "b" * 64,
        history,
        action,
        event,
        exposure,
        exposure,
        label.label_sha256,
        sample_sha,
    )


def test_family_split_is_deterministic_and_whole_family() -> None:
    family_ids = tuple(f"family-{index}" for index in range(10))
    first = family_split(family_ids)
    assert first == family_split(family_ids)
    assert set(first.values()) == {"TRAIN", "VALIDATION", "EVALUATION"}
    assert set(first) == set(family_ids)


def test_risk_decision_steps_have_complete_horizon() -> None:
    assert risk_decision_steps() == tuple(range(16, 65, 4))
    with pytest.raises(Issue56RiskError, match="too short"):
        risk_decision_steps(47)


def test_feature_vector_is_action_conditioned_and_fail_closed() -> None:
    history = _history()
    action = np.full(ACTION_COUNT, 0.5, dtype=np.float32)
    result = feature_vector(history, action)
    assert result.shape == (FEATURE_COUNT,)
    assert result[-ACTION_COUNT - 1 : -1].tolist() == action.tolist()
    unavailable = _history()
    unavailable.status_f32[:, 0, 0] = 0.0
    unavailable.status_f32[:, 0, 1] = 1.0
    with pytest.raises(Issue56RiskError, match="complete finite history"):
        feature_vector(unavailable, action)


def test_model_fits_train_only_calibrates_validation_and_round_trips() -> None:
    train = (
        _sample("family-a", 0.0, 0.0, 0.0),
        _sample("family-a", 0.01, 0.1, 0.0),
        _sample("family-b", 0.02, 1.0, 1.0),
        _sample("family-b", 0.03, 2.0, 1.0),
    )
    model = ActionRiskModel.fit(train)
    validation = tuple(
        ActionRiskSample(
            sample.family_id,
            sample.decision_step,
            "VALIDATION",
            sample.action_id,
            sample.scenario_sha256,
            sample.history_steps,
            sample.snapshot_sha256,
            sample.input_manifest_sha256,
            sample.target_manifest_sha256,
            sample.history_numeric_f32,
            sample.action_f32,
            sample.crossing_event,
            sample.safety_exposure,
            sample.maximum_crossing,
            sample.label_sha256,
            hashlib.sha256(
                json.dumps(
                    {
                        "schema_version": "aeolus_habitat_v2_risk_issue_56_v1.sample",
                        "family_id": sample.family_id,
                        "decision_step": sample.decision_step,
                        "split": "VALIDATION",
                        "action_id": sample.action_id,
                        "scenario_sha256": sample.scenario_sha256,
                        "history_steps": list(sample.history_steps),
                        "snapshot_sha256": list(sample.snapshot_sha256),
                        "input_manifest_sha256": sample.input_manifest_sha256,
                        "target_manifest_sha256": sample.target_manifest_sha256,
                        "history_numeric": sample.history_numeric_f32.tobytes().hex(),
                        "action": sample.action_f32.tobytes().hex(),
                        "crossing_event": sample.crossing_event,
                        "safety_exposure": sample.safety_exposure,
                        "maximum_crossing": sample.maximum_crossing,
                        "label_sha256": sample.label_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )
        for sample in train[:2]
    )
    calibrated = model.calibrate(validation)
    restored = ActionRiskModel.from_mapping(calibrated.to_mapping())
    assert restored.to_mapping() == calibrated.to_mapping()
    assert restored.actuator_authority is False


def test_risk_ranker_abstains_when_all_candidates_fail_hard_gate() -> None:
    scores = tuple(
        RiskScore(
            action_id,
            float("inf"),
            True,
            HARD_EXPOSURE_LIMIT + 1.0,
            HARD_CROSSING_PROBABILITY_LIMIT,
            1.0,
            0.1,
            "calibrated_safety_risk_limit",
        )
        for action_id in ("a", "b")
    )
    assert rank_action_risk(scores) is None
    with pytest.raises(Issue56RiskError, match="duplicate"):
        rank_action_risk((scores[0], scores[0]))


def test_counterfactual_label_is_deterministic_and_true_plant_bound() -> None:
    bundle = load_forecast_contracts(REPO_ROOT)
    scenario = build_family_scenario(bundle.development_scenario, 0)
    action = bundle.actions[0]
    first = counterfactual_risk_label(
        scenario,
        tuple(bundle.topology.zone_ids),
        initial_state(scenario),
        action.action_id,
        action.command.to_mapping(),
    )
    second = counterfactual_risk_label(
        scenario,
        tuple(bundle.topology.zone_ids),
        initial_state(scenario),
        action.action_id,
        action.command.to_mapping(),
    )
    assert first.eligible
    assert first.label_sha256 == second.label_sha256
    assert first.crossing_event in {0.0, 1.0}
    assert np.isfinite(first.targets).all()


def test_advisory_proposal_is_accepted_only_by_hmc() -> None:
    bundle: ForecastContracts = load_forecast_contracts(REPO_ROOT)
    hmc = HabitatManagementComputer.reset(
        bundle.development_scenario,
        bundle.hmc_contract,
        b"r" * 32,
    )
    observed = hmc.observe()
    assert isinstance(observed, tuple)
    snapshot, verification = observed
    handle = hmc.verify_snapshot(snapshot, verification)
    action = bundle.actions[0]
    proposal = build_risk_proposal(
        hmc,
        snapshot.snapshot_sha256,
        0,
        action.command.to_mapping(),
        action.action_id,
    )
    receipt = hmc.propose(proposal, handle).to_mapping()
    assert receipt["validation_outcome"] == "VALID"
    assert receipt["source_type"] == "issue56-action-risk-advisory-v1"
