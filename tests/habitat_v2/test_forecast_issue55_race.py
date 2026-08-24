"""Tests for the Issue #55 three-way controller race module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model
from aeolus.habitat_v2.forecast_issue55_race import (
    ADVISORY_RANKING_METRIC_ID,
    ARMS,
    CORPUS_ID,
    EPISODE_STEPS,
    ORACLE_SELECTION_METRIC_ID,
    PREREGISTRATION_ID,
    RACE_SCHEMA_VERSION,
    Issue55RaceError,
    OracleScore,
    aggregate_race_results,
    bootstrap_gap_closure,
    build_family_scenario,
    compute_race_metrics,
    decision_steps,
    deterministic_family_ids,
    episode_nonce,
    oracle_lookahead_scores,
    project_true_targets,
    rank_actions_advisory,
    run_race_episode,
    scenario_zone_order,
    score_point_prediction,
    select_oracle_action,
    target_bounds,
)
from aeolus.habitat_v2.physics import initial_state

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_DIGEST = (
    "17C601D7F15A21804AA68B26024C96D44642491E07A9BD75BDE805E027C773CF"
)
MLP_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
MLP_ARTIFACT_SHA = "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"


def _flat_prediction(values: float) -> np.ndarray:
    return np.full((8, 51), values, dtype=np.float32)


def _nominal_prediction(offset_columns: dict[int, float] | None = None) -> np.ndarray:
    _, nominals, _, _ = target_bounds()
    prediction = np.tile(nominals, (8, 1))
    if offset_columns:
        for column, value in offset_columns.items():
            prediction[:, column] = nominals[column] + value
    return prediction.astype(np.float32)


def _vectors() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(27, dtype=np.float64), np.zeros(27, dtype=np.float64)


class TestPreregistrationBinding:
    def test_preregistration_digest_is_frozen(self) -> None:
        path = (
            REPO_ROOT
            / "contracts"
            / "habitat_v2_forecast_issue_55_preregistration_v1.json"
        )
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest().upper()
        assert digest == PREREGISTRATION_DIGEST

    def test_module_constants_bind_preregistration(self) -> None:
        assert PREREGISTRATION_ID == "habitat_v2_forecast_issue_55_preregistration_v1"
        assert RACE_SCHEMA_VERSION == "aeolus_habitat_v2_race_issue_55_v1"
        assert ADVISORY_RANKING_METRIC_ID == "issue55-advisory-point-ranking-v1"
        assert ORACLE_SELECTION_METRIC_ID == "issue55-oracle-lookahead-v1"
        assert CORPUS_ID == "issue55_race_v1"
        assert ARMS == ("rules_only", "model_advised", "oracle_instrument")


class TestProtocolDerivation:
    def test_decision_steps_are_preregistered(self) -> None:
        steps = decision_steps(EPISODE_STEPS)
        assert steps == tuple(range(16, 85, 4))
        assert len(steps) == 18
        assert all(step + 8 <= EPISODE_STEPS - 1 for step in steps)

    def test_decision_steps_filter_incomplete_lookahead(self) -> None:
        assert decision_steps(40) == (16, 20, 24, 28)

    def test_decision_steps_reject_short_episodes(self) -> None:
        with pytest.raises(Issue55RaceError, match="too short"):
            decision_steps(24)

    def test_family_ids_are_deterministic_and_unique(self) -> None:
        first = deterministic_family_ids(32)
        second = deterministic_family_ids(32)
        assert first == second
        assert len(set(first)) == 32
        assert all(fid.startswith("issue55f") for fid in first)

    def test_nonce_is_family_bound_and_arm_independent(self) -> None:
        nonce = episode_nonce("issue55fabc")
        assert nonce == episode_nonce("issue55fabc")
        assert nonce != episode_nonce("issue55fother")
        assert len(nonce) == 32

    def test_target_bounds_match_preregistered_values(self) -> None:
        scales, nominals, lowers, uppers = target_bounds()
        assert scales.shape == nominals.shape == lowers.shape == uppers.shape == (51,)
        assert nominals[0] == pytest.approx(295.15)
        assert scales[0] == pytest.approx(10.0)
        assert lowers[0] == pytest.approx(250.0)
        assert uppers[0] == pytest.approx(330.0)
        assert nominals[48] == pytest.approx(0.75)
        assert lowers[50] == pytest.approx(0.0)


class TestTrueTargetProjection:
    def test_initial_state_projection_is_well_formed(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        scenario = bundle.development_scenario
        zone_ids = scenario_zone_order(scenario)
        row = project_true_targets(scenario, zone_ids, initial_state(scenario))
        assert row.shape == (51,)
        assert row.dtype == np.float32
        assert np.isfinite(row).all()
        assert 0.0 <= float(row[48]) <= 1.0
        assert 0.0 <= float(row[49]) <= 1.0
        assert 0.0 <= float(row[50]) <= 1.0

    def test_projection_rejects_wrong_zone_count(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        scenario = bundle.development_scenario
        with pytest.raises(Issue55RaceError, match="eight zone ids"):
            project_true_targets(
                scenario, ("z0",), initial_state(scenario)
            )


class TestAdvisoryRanking:
    def test_nominal_prediction_scores_zero_without_intervention(self) -> None:
        current, candidate = _vectors()
        score = score_point_prediction(
            "normal-occupied-v1", _nominal_prediction(), current, candidate
        )
        assert score.hard_ineligible is False
        assert score.tracking == pytest.approx(0.0, abs=1e-6)
        assert score.safety == pytest.approx(0.0, abs=1e-6)
        assert score.score == pytest.approx(0.0, abs=1e-6)

    def test_prediction_outside_bounds_is_hard_ineligible(self) -> None:
        current, candidate = _vectors()
        score = score_point_prediction(
            "normal-occupied-v1", _flat_prediction(1e6), current, candidate
        )
        assert score.hard_ineligible is True
        assert score.score == float("inf")
        assert score.reason == "predicted_hard_bound_crossing"

    def test_ranking_prefers_lower_score(self) -> None:
        current, candidate = _vectors()
        better = score_point_prediction(
            "a-normal", _nominal_prediction(), current, candidate
        )
        worse = score_point_prediction(
            "b-normal", _nominal_prediction({0: 20.0}), current, candidate
        )
        assert worse.hard_ineligible is False
        selected = rank_actions_advisory([worse, better])
        assert selected is not None
        assert selected.action_id == "a-normal"

    def test_ranking_returns_none_when_all_ineligible(self) -> None:
        current, candidate = _vectors()
        scores = [
            score_point_prediction(
                f"{index}-normal", _flat_prediction(1e6), current, candidate
            )
            for index in range(2)
        ]
        assert rank_actions_advisory(scores) is None

    def test_ranking_tie_break_prefers_hold_then_id(self) -> None:
        current, candidate = _vectors()
        first = score_point_prediction(
            "b-hold", _nominal_prediction(), current, candidate
        )
        second = score_point_prediction(
            "a-normal", _nominal_prediction(), current, candidate
        )
        selected = rank_actions_advisory([second, first])
        assert selected is not None
        assert selected.action_id == "b-hold"
        third = score_point_prediction("a-x", _nominal_prediction(), current, candidate)
        fourth = score_point_prediction("b-x", _nominal_prediction(), current, candidate)
        assert rank_actions_advisory([fourth, third]).action_id == "a-x"

    def test_ranking_rejects_duplicates(self) -> None:
        current, candidate = _vectors()
        score = score_point_prediction("a", _nominal_prediction(), current, candidate)
        with pytest.raises(Issue55RaceError, match="duplicate action ids"):
            rank_actions_advisory([score, score])


class TestOracleSelection:
    def test_selects_minimum_score(self) -> None:
        scores = (
            OracleScore("a", 2.0, 1.0, 0.5, 0.1, 8, False),
            OracleScore("b", 1.0, 0.5, 0.25, 0.05, 8, False),
            OracleScore("c", float("inf"), float("inf"), float("inf"), float("inf"), 3, True),
        )
        assert select_oracle_action(scores).action_id == "b"

    def test_returns_none_when_all_excluded(self) -> None:
        scores = (
            OracleScore("a", float("inf"), float("inf"), float("inf"), float("inf"), 2, True),
            OracleScore("b", float("inf"), float("inf"), float("inf"), float("inf"), 0, True),
        )
        assert select_oracle_action(scores) is None

    def test_lookahead_on_real_scenario_produces_four_entries(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        scenario = build_family_scenario(bundle.development_scenario, 0)
        zone_ids = scenario_zone_order(scenario)
        scores = oracle_lookahead_scores(
            scenario, zone_ids, initial_state(scenario), tuple(bundle.actions)
        )
        assert len(scores) == 4
        assert {item.action_id for item in scores} == {
            action.action_id for action in bundle.actions
        }
        for item in scores:
            if item.excluded:
                assert item.feasible_steps < 8
                assert item.score == float("inf")
            else:
                assert item.feasible_steps == 8
                assert item.score >= 0.0


class TestBootstrapGapClosure:
    def test_midway_model_closes_half_the_gap(self) -> None:
        rules = [2.0] * 8
        model = [1.0] * 8
        oracle = [0.0] * 8
        result = bootstrap_gap_closure(rules, model, oracle, resamples=200)
        assert result["status"] == "ESTIMATED"
        assert result["point_estimate"] == pytest.approx(0.5)

    def test_equal_rules_and_oracle_is_degenerate(self) -> None:
        result = bootstrap_gap_closure([1.0, 1.0], [1.0, 0.5], [1.0, 1.0], resamples=50)
        assert result["status"] == "DEGENERATE_GAP"
        assert result["degenerate_gap"] is True
        assert result["point_estimate"] is None

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(Issue55RaceError, match="equal-length"):
            bootstrap_gap_closure([1.0, 2.0], [1.0], [0.5, 0.5])


class TestAggregation:
    def test_rejects_empty_records(self) -> None:
        with pytest.raises(Issue55RaceError, match="requires episode records"):
            aggregate_race_results([])


class TestFamilyScenario:
    def test_family_variants_are_extended_and_deterministic(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        base = bundle.development_scenario
        family_zero = build_family_scenario(base, 0)
        family_one = build_family_scenario(base, 1)
        assert family_zero.data["steps"] == EPISODE_STEPS
        assert family_one.data["steps"] == EPISODE_STEPS
        assert family_zero.scenario_sha256 != family_one.scenario_sha256
        assert (
            build_family_scenario(base, 0).scenario_sha256
            == family_zero.scenario_sha256
        )
        assert (
            family_one.data["sensor_model"]["random_seed"]
            == family_zero.data["sensor_model"]["random_seed"] + 1000
        )


def _run_family_episode(bundle, arm: str, family_index: int, teacher=None):
    scenario = build_family_scenario(bundle.development_scenario, family_index)
    family_id = deterministic_family_ids(32)[family_index]
    return run_race_episode(bundle, scenario, arm, family_id, family_index, teacher)


class TestEpisodeIntegration:
    def test_rules_only_episode_is_replayable_and_never_proposes(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        record = _run_family_episode(bundle, "rules_only", 0)
        assert record.arm == "rules_only"
        assert record.proposal_count == 0
        assert record.abstention_count == 0
        assert record.admitted_proposal_count == 0
        assert record.hmc_rejection_count == 0
        assert record.decision_actions == (None,) * 18
        assert record.replay_committed_steps == EPISODE_STEPS
        assert record.episode_steps == EPISODE_STEPS
        assert record.safety_exposure >= 0.0
        assert 0 <= record.safety_violation_steps <= EPISODE_STEPS
        assert record.comfort_deviation >= 0.0
        assert record.resource_composite >= 0.0
        assert len(record.episode_sha256) == 64

    def test_rules_only_episode_is_deterministic(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        first = _run_family_episode(bundle, "rules_only", 1)
        second = _run_family_episode(bundle, "rules_only", 1)
        assert first.episode_sha256 == second.episode_sha256
        assert first.trace_sha256 == second.trace_sha256

    def test_model_advised_episode_proposes_through_hmc(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        teacher = load_live_mlp_model(
            MLP_ARTIFACT_PATH, expected_sha256=MLP_ARTIFACT_SHA
        )
        record = _run_family_episode(bundle, "model_advised", 0, teacher)
        assert record.arm == "model_advised"
        assert record.proposal_count + record.abstention_count == 18
        assert record.admitted_proposal_count == record.proposal_count
        assert 0 <= record.hmc_rejection_count <= record.proposal_count
        assert sum(entry is not None for entry in record.decision_actions) == (
            record.proposal_count
        )
        catalogue_ids = {action.action_id for action in bundle.actions}
        assert all(
            entry is None or entry in catalogue_ids
            for entry in record.decision_actions
        )
        assert record.replay_committed_steps == EPISODE_STEPS

    def test_oracle_episode_proposes_through_hmc(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        record = _run_family_episode(bundle, "oracle_instrument", 0)
        assert record.arm == "oracle_instrument"
        assert record.proposal_count + record.abstention_count == 18
        assert record.admitted_proposal_count == record.proposal_count
        assert sum(entry is not None for entry in record.decision_actions) == (
            record.proposal_count
        )
        assert record.replay_committed_steps == EPISODE_STEPS

    def test_teacher_is_rejected_for_non_model_arms(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        teacher = load_live_mlp_model(
            MLP_ARTIFACT_PATH, expected_sha256=MLP_ARTIFACT_SHA
        )
        with pytest.raises(Issue55RaceError, match="only the model-advised arm"):
            _run_family_episode(bundle, "rules_only", 0, teacher)

    def test_metrics_require_contiguous_states(self) -> None:
        bundle = load_forecast_contracts(REPO_ROOT)
        scenario = build_family_scenario(bundle.development_scenario, 0)
        zone_ids = scenario_zone_order(scenario)
        initial_row = project_true_targets(
            scenario, zone_ids, initial_state(scenario)
        )
        with pytest.raises(Issue55RaceError, match="contiguous"):
            compute_race_metrics(
                scenario, zone_ids, initial_row, [initial_state(scenario)]
            )
