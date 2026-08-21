from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aeolus.habitat_v2.forecast_issue52 import (
    CATALOGUE_SIZE,
    HISTORY_STEPS,
    HORIZON_STEPS,
    ActionConditionedLinearForecaster,
    CandidateCatalogue,
    ForecastTrajectory,
    Issue52AdvisorySource,
    TargetManifest,
    extend_scenario_for_issue52,
    normalized_mae,
    rank_candidates,
)
from aeolus.habitat_v2.forecast_issue52_rollout import (
    assess_rollout_feasibility,
    build_offline_checkpoint,
    rollout_catalogue,
    training_samples_from_rollouts,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.scenario import Scenario


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    parsed = Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    return extend_scenario_for_issue52(parsed)


def _contract():
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def _complete_cycle(
    hmc: HabitatManagementComputer, adviser: Issue52AdvisorySource
):
    observed = hmc.observe()
    assert isinstance(observed, tuple)
    snapshot, verification = observed
    decision, proposal = adviser.submit(hmc, snapshot, verification)
    arbitration = hmc.arbitrate()
    assert type(arbitration).__name__ == "ArbitrationReceipt"
    stepped = hmc.step()
    assert type(stepped).__name__ == "StepReceipt"
    return decision, proposal, arbitration


def test_manifest_and_catalogue_are_topology_bound_and_deterministic() -> None:
    scenario = _scenario()

    manifest = TargetManifest.from_scenario(scenario)
    catalogue = CandidateCatalogue.from_scenario(scenario)

    assert manifest.width == 3 * len(scenario.data["zones"]) + 3
    assert len(manifest.descriptors) == len({item.descriptor_id for item in manifest.descriptors})
    assert [item.descriptor_id for item in manifest.descriptors[-3:]] == [
        "battery_state_of_charge",
        "oxygen_store_fraction",
        "sorbent_remaining_fraction",
    ]
    assert len(catalogue.candidates) == CATALOGUE_SIZE
    assert [item.candidate_id for item in catalogue.candidates][0] == "candidate_hold"
    assert all(len(item.commands) == HORIZON_STEPS for item in catalogue.candidates)
    assert catalogue.catalogue_sha256 == CandidateCatalogue.from_scenario(
        scenario
    ).catalogue_sha256


def test_offline_checkpoint_rollout_is_deterministic_and_complete() -> None:
    scenario = _scenario()
    contract = _contract()
    first = build_offline_checkpoint(scenario, contract, family_id="family-a")
    second = build_offline_checkpoint(scenario, contract, family_id="family-a")
    catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=first.last_final_command
    )

    rollouts = rollout_catalogue(first, catalogue)
    repeated = rollout_catalogue(second, catalogue)

    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert len(first.history_records) == HISTORY_STEPS
    assert len(rollouts) == CATALOGUE_SIZE
    assert all(result.targets.shape == (HORIZON_STEPS, first.manifest.width) for result in rollouts)
    assert all(result.eligible for result in rollouts)
    assert [result.rollout_sha256 for result in rollouts] == [
        result.rollout_sha256 for result in repeated
    ]
    assert {
        result.rollout_status
        for result in assess_rollout_feasibility(catalogue, rollouts)
    } == {"ROLLOUT_FEASIBLE"}


def test_training_and_metric_bind_complete_rollouts() -> None:
    scenario = _scenario()
    checkpoint = build_offline_checkpoint(scenario, _contract(), family_id="family-a")
    catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=checkpoint.last_final_command
    )
    rollouts = rollout_catalogue(checkpoint, catalogue)
    samples = training_samples_from_rollouts(checkpoint, catalogue, rollouts)

    assert len(samples) == CATALOGUE_SIZE
    assert normalized_mae(
        rollouts[0].targets,
        rollouts[0].targets,
        checkpoint.manifest,
        start_horizon=9,
        end_horizon=HORIZON_STEPS,
    ) == 0.0
    with pytest.raises(ValueError, match="two families"):
        ActionConditionedLinearForecaster.fit_for_scenario(
            scenario, checkpoint.manifest, samples
        )


def test_ranker_fails_closed_for_invalid_or_ambiguous_forecasts() -> None:
    scenario = _scenario()
    checkpoint = build_offline_checkpoint(scenario, _contract())
    catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=checkpoint.last_final_command
    )
    history = checkpoint.history_records
    forecast_history = __import__(
        "aeolus.habitat_v2.forecast_issue52", fromlist=["ForecastHistory"]
    ).ForecastHistory.from_records(history)
    width = checkpoint.manifest.width
    mean = np.repeat(forecast_history.latest[None, :], HORIZON_STEPS, axis=0)
    mean = np.clip(mean, 0.01, 0.99)
    mean[:, : len(scenario.data["zones"]) * 3 : 3] = 800.0
    mean[:, 1 : len(scenario.data["zones"]) * 3 : 3] = 295.15
    mean[:, 2 : len(scenario.data["zones"]) * 3 : 3] = 0.45
    lower = mean - 0.001
    upper = mean + 0.001
    trajectories = {
        item.candidate_id: ForecastTrajectory(
            "PREDICTION", mean, lower, upper, "test-model"
        )
        for item in catalogue.candidates
    }

    ambiguous = rank_candidates(
        catalogue,
        checkpoint.manifest,
        forecast_history,
        trajectories,
        scenario,
        ambiguity_margin=1.0,
    )
    assert ambiguous.outcome == "ABSTAINED"
    assert ambiguous.reason == "candidate_margin_ambiguous"
    assert width == mean.shape[1]

    trajectories[catalogue.candidates[0].candidate_id] = ForecastTrajectory(
        "INVALID_OUTPUT", None, None, None, "test-model", "invalid"
    )
    invalid = rank_candidates(
        catalogue,
        checkpoint.manifest,
        forecast_history,
        trajectories,
        scenario,
        ambiguity_margin=0.0,
    )
    assert invalid.outcome == "INVALID_OUTPUT"
    assert invalid.reason == "forecast_invalid_output"


def test_adviser_warms_up_then_only_submits_current_first_action() -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"i" * 32)
    adviser = Issue52AdvisorySource.create(
        scenario,
        enabled=True,
        ambiguity_margin=0.0,
        inference_deadline_ms=10_000.0,
    )

    decisions = []
    for _ in range(HISTORY_STEPS):
        decision, proposal, arbitration = _complete_cycle(hmc, adviser)
        decisions.append((decision, proposal, arbitration))

    assert [decision.outcome for decision, _, _ in decisions[: HISTORY_STEPS - 1]] == [
        "WARMUP_NO_PROPOSAL"
    ] * (HISTORY_STEPS - 1)
    assert all(
        proposal.attempt_class == "NONE" for _, proposal, _ in decisions[: HISTORY_STEPS - 1]
    )
    final_decision, final_proposal, final_arbitration = decisions[-1]
    assert final_decision.outcome in {"SELECTED_HOLD", "SELECTED_CANDIDATE", "ABSTAINED"}
    if final_decision.proposal is None:
        assert final_proposal.attempt_class == "NONE"
    else:
        assert final_proposal.attempt_class == "CANONICAL_PROPOSAL"
        proposed = final_decision.proposal["proposed_command"]
        assert proposed == next(
            candidate.first_command.to_mapping()
            for candidate in adviser.catalogue.candidates
            if candidate.candidate_id == final_decision.candidate_id
        )
    assert final_arbitration.final_command_sha256
    assert len(adviser.history.records) == HISTORY_STEPS


def test_adviser_timeout_submits_no_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"t" * 32)
    adviser = Issue52AdvisorySource.create(
        scenario,
        enabled=True,
        inference_deadline_ms=0.000001,
    )
    ticks = iter(range(0, 100_000_000, 1_000_000))
    monkeypatch.setattr(
        "aeolus.habitat_v2.forecast_issue52.time.perf_counter_ns",
        lambda: next(ticks),
    )

    decisions = [_complete_cycle(hmc, adviser) for _ in range(HISTORY_STEPS)]

    decision, proposal, _ = decisions[-1]
    assert decision.outcome == "TIMEOUT_NO_PROPOSAL"
    assert proposal.attempt_class == "NONE"


def test_disabled_adviser_does_not_change_hmc_proposal_path() -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"d" * 32)
    adviser = Issue52AdvisorySource.create(scenario, enabled=False)
    observed = hmc.observe()
    assert isinstance(observed, tuple)

    decision, proposal = adviser.submit(hmc, *observed)

    assert decision.outcome == "DISABLED"
    assert proposal.attempt_class == "NONE"
    assert hmc.arbitrate().to_mapping()["command_owner"] == "baseline_hold"


def test_adviser_rejects_same_topology_different_scenario() -> None:
    scenario = _scenario()
    altered = deepcopy(scenario.data)
    altered["timeline"][0]["generation_w"] += 1.0
    other_scenario = Scenario.from_mapping(altered)
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"s" * 32)
    adviser = Issue52AdvisorySource.create(other_scenario, enabled=True)
    observed = hmc.observe()
    assert isinstance(observed, tuple)

    decision, proposal = adviser.submit(hmc, *observed)

    assert decision.outcome == "INVALID_OUTPUT"
    assert proposal.attempt_class == "NONE"


def test_ranker_rejects_safety_interval_crossing_and_partial_invalid_batch() -> None:
    scenario = _scenario()
    checkpoint = build_offline_checkpoint(scenario, _contract())
    catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=checkpoint.last_final_command
    )
    history = __import__(
        "aeolus.habitat_v2.forecast_issue52", fromlist=["ForecastHistory"]
    ).ForecastHistory.from_records(checkpoint.history_records)
    width = checkpoint.manifest.width
    mean = np.zeros((HORIZON_STEPS, width), dtype=np.float32)
    for index, descriptor in enumerate(checkpoint.manifest.descriptors):
        mean[:, index] = descriptor.nominal
    lower = mean.copy()
    upper = mean.copy()
    temperature_index = next(
        index
        for index, descriptor in enumerate(checkpoint.manifest.descriptors)
        if descriptor.descriptor_id.endswith("/temperature_k")
    )
    mean[:, temperature_index] = 302.9
    upper[:, temperature_index] = 303.1
    trajectories = {
        candidate.candidate_id: ForecastTrajectory(
            "PREDICTION", mean, lower, upper, "test-model"
        )
        for candidate in catalogue.candidates
    }

    decision = rank_candidates(
        catalogue,
        checkpoint.manifest,
        history,
        trajectories,
        scenario,
        contract=_contract(),
    )

    assert decision.outcome == "ABSTAINED"
    assert all(score.hard_ineligible for score in decision.scores)


def test_rollout_training_rejects_foreign_checkpoint_labels() -> None:
    scenario = extend_scenario_for_issue52(_scenario(), minimum_steps=48)
    contract = _contract()
    first = build_offline_checkpoint(scenario, contract, decision_step=15)
    second = build_offline_checkpoint(scenario, contract, decision_step=16)
    first_catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=first.last_final_command
    )
    second_catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=second.last_final_command
    )

    with pytest.raises(ValueError, match="rollout does not bind"):
        training_samples_from_rollouts(
            first,
            first_catalogue,
            rollout_catalogue(second, second_catalogue),
        )


def test_reconcile_arbitration_reclassifies_rejection_and_emergency_without_double_counting() -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"r" * 32)
    adviser = Issue52AdvisorySource.create(scenario, enabled=True)
    observed = hmc.observe()
    assert isinstance(observed, tuple)
    decision, _ = adviser.submit(hmc, *observed)
    assert decision.outcome == "WARMUP_NO_PROPOSAL"
    hmc.arbitrate()
    hmc.step()

    proposed = "0123456789abcdef" * 4
    rejected_receipt = SimpleNamespace(
        to_mapping=lambda: {
            "arbitration_receipt_sha256": proposed,
            "emergency_override": False,
            "disposition": "REJECTED",
            "command_owner": "baseline_hold",
        }
    )
    selected = SimpleNamespace(
        outcome="SELECTED_CANDIDATE",
        candidate_id="candidate_hold",
        proposal={"proposed_command": {"fan_speed_fraction": 0.5}},
        ranked=None,
        history_status="ACCEPTED",
        reason=None,
        latency_ms=1.5,
    )
    adviser._outcome_counts["SELECTED_CANDIDATE"] = 1

    reclassified = adviser.reconcile_arbitration(selected, rejected_receipt)

    assert reclassified.outcome == "HMC_REJECTED_TO_HOLD"
    assert reclassified.hmc_receipt_sha256 == proposed
    assert adviser.outcome_counts["SELECTED_CANDIDATE"] == 0
    assert adviser.outcome_counts["HMC_REJECTED_TO_HOLD"] == 1

    emergency_receipt = SimpleNamespace(
        to_mapping=lambda: {
            "arbitration_receipt_sha256": proposed,
            "emergency_override": True,
            "disposition": "MODIFIED",
            "command_owner": "emergency_safe_action",
        }
    )
    adviser._outcome_counts["SELECTED_CANDIDATE"] = 1

    emergency = adviser.reconcile_arbitration(selected, emergency_receipt)

    assert emergency.outcome == "HMC_EMERGENCY_OVERRIDDEN"
    assert adviser.outcome_counts["SELECTED_CANDIDATE"] == 0
    assert adviser.outcome_counts["HMC_EMERGENCY_OVERRIDDEN"] == 1
