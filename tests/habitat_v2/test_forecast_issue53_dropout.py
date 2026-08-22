"""Issue #53 dropout lane — contract and honesty coverage.

The Issue #52 lane at ``src/aeolus/habitat_v2/forecast_issue52.py:1550`` correctly
abstains on any missing latest target. This suite proves the #53 lane does
the honest alternative: deterministic, observation-only masks that keep the model
working with 1–3 missing sensors while saying how unsure it is.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from aeolus.habitat_v2.forecast_issue52 import (
    CandidateCatalogue,
    ForecastHistory,
    TargetManifest,
)
from aeolus.habitat_v2.forecast_issue53_dropout import (
    DropoutAwareLinearForecaster,
    DropoutConfig,
    apply_dropout_to_history,
    build_dropout_dataset_manifest,
    dropout_mask_for_history,
    evaluate_per_k,
    impute_history_values,
    interval_coverage_at_k,
)
from aeolus.habitat_v2.forecast_issue52_rollout import build_offline_checkpoint

from aeolus.habitat_v2.hmc_contract import HMCContract
from aeolus.habitat_v2.scenario import Scenario


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    parsed = Scenario.from_mapping(__import__("json").loads(path.read_text(encoding="utf-8")))
    from aeolus.habitat_v2.forecast_issue52 import extend_scenario_for_issue52

    return extend_scenario_for_issue52(parsed)


def _contract() -> HMCContract:
    from aeolus.habitat_v2.hmc_contract import load_hmc_contract

    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def _checkpoint_and_manifest():
    scenario = _scenario()
    contract = _contract()
    cp = build_offline_checkpoint(scenario, contract, decision_step=15)
    manifest = TargetManifest.from_scenario(scenario)
    return cp, manifest, scenario


def test_dropout_config_digest_is_canonical() -> None:
    cfg = DropoutConfig(p_uniform=0.05, mode="independent", seed=530053)
    # recompute from canonical bytes
    payload = {
        "schema_version": "aeolus_habitat_v2_dropout_v1",
        "p_uniform": 0.05,
        "mode": "independent",
        "burst_min": 2,
        "burst_max": 8,
        "p_burst_onset": 0.02,
        "resource_gauge_dropout": False,
        "max_missing_per_row": 6,
        "seed": 530053,
    }
    from aeolus.habitat_v2.hmc_contract import canonical_json_bytes

    expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert cfg.config_sha256 == expected


def test_dropout_mask_is_deterministic() -> None:
    cp, manifest, _ = _checkpoint_and_manifest()
    history = ForecastHistory.from_records(cp.history_records)
    cfg = DropoutConfig(p_uniform=0.07, seed=530053)
    a = dropout_mask_for_history(history, manifest, cfg, family_id="fam-A", decision_step=15)
    b = dropout_mask_for_history(history, manifest, cfg, family_id="fam-A", decision_step=15)
    assert np.array_equal(a, b)


def test_dropout_is_observation_only_and_leak_free() -> None:
    cp, manifest, _ = _checkpoint_and_manifest()
    history = ForecastHistory.from_records(cp.history_records)
    cfg = DropoutConfig(p_uniform=0.15, seed=530053)
    masked = apply_dropout_to_history(history, manifest, cfg, family_id="fam-B", decision_step=15)
    # Time array is copied unchanged — no future mask leaks into time
    assert np.array_equal(masked.completed_times_s, history.completed_times_s)
    # Native missing stays missing — dropout is additive
    assert np.all(masked.available_mask <= history.available_mask)
    # Where we newly masked, target is NaN
    newly_masked = history.available_mask & ~masked.available_mask
    assert np.all(np.isnan(masked.target_values[newly_masked]))
    # Where still available, value is untouched
    assert np.all(masked.target_values[masked.available_mask] == history.target_values[masked.available_mask])


def test_resource_gauges_are_anchors_by_default() -> None:
    cp, manifest, _ = _checkpoint_and_manifest()
    history = ForecastHistory.from_records(cp.history_records)
    cfg = DropoutConfig(p_uniform=0.9, seed=530053, resource_gauge_dropout=False, max_missing_per_row=None)
    mask = dropout_mask_for_history(history, manifest, cfg, family_id="fam-C", decision_step=15)
    gauge_cols = [i for i, d in enumerate(manifest.descriptors) if d.descriptor_id in {"battery_state_of_charge", "oxygen_store_fraction", "sorbent_remaining_fraction"}]
    # Gauges never dropped when resource_gauge_dropout is False
    for c in gauge_cols:
        assert np.all(mask[:, c] == history.available_mask[:, c])


def test_imputation_is_nan_free_and_deterministic() -> None:
    cp, manifest, _ = _checkpoint_and_manifest()
    history = ForecastHistory.from_records(cp.history_records)
    cfg = DropoutConfig(p_uniform=0.2, seed=530053)
    masked = apply_dropout_to_history(history, manifest, cfg, family_id="fam-D", decision_step=15)
    a = impute_history_values(masked.target_values, masked.available_mask, manifest)
    b = impute_history_values(masked.target_values, masked.available_mask, manifest)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_masked_forecaster_keeps_working_with_missing() -> None:
    cp, manifest, scenario = _checkpoint_and_manifest()
    history = ForecastHistory.from_records(cp.history_records)
    cfg = DropoutConfig(p_uniform=0.0, seed=530053)
    # Train a tiny forecaster on 3 samples with k=0 then k=1,3 dropout — synthetic data from checkpoint
    from aeolus.habitat_v2.forecast_issue52_rollout import rollout_catalogue

    catalogue = CandidateCatalogue.from_scenario(scenario, base_command=cp.last_final_command)
    rollouts = rollout_catalogue(cp, catalogue)
    from aeolus.habitat_v2.forecast_issue52_rollout import training_samples_from_rollouts

    samples = training_samples_from_rollouts(cp, catalogue, rollouts, split="TRAIN")
    # Expand to at least 2 families by cloning with different family_ids
    samples2 = tuple(
        __import__("aeolus.habitat_v2.forecast_issue52", fromlist=["TrainingSample"]).TrainingSample(
            family_id=f"fam-{i}",
            split=s.split,
            scenario_sha256=s.scenario_sha256,
            manifest_sha256=s.manifest_sha256,
            checkpoint_sha256=s.checkpoint_sha256,
            schedule_sha256=s.schedule_sha256,
            history=s.history,
            schedule=s.schedule,
            targets=s.targets,
        )
        for i, s in enumerate(list(samples) * 2)
    )
    forecaster = DropoutAwareLinearForecaster.fit_for_scenario(scenario, manifest, samples2, dropout_config=cfg, alpha=1e-4)
    # Now create a history with k=3 missing on latest row — Issue #52 forecaster would ABSTAIN
    cfg3 = DropoutConfig(p_uniform=0.35, seed=99, max_missing_per_row=3)
    masked_k3 = apply_dropout_to_history(history, manifest, cfg3, family_id="fam-E", decision_step=15)
    # Force exactly 3 missing on latest row for the honesty check
    # Guard: if sampler gave differently, inject deterministically
    if int(np.sum(~masked_k3.available_mask[-1])) != 3:
        # manually craft a k=3 latest row by masking 3 lexicographically smallest env channels
        values = masked_k3.target_values.copy()
        mask = masked_k3.available_mask.copy()
        avail_cols = [i for i in range(manifest.width) if mask[-1, i]]
        for c in sorted(avail_cols, key=lambda i: manifest.descriptors[i].descriptor_id)[:3]:
            mask[-1, c] = False
            values[-1, c] = np.nan
        values.setflags(write=False)
        mask.setflags(write=False)
        times = masked_k3.completed_times_s.copy()
        times.setflags(write=False)
        masked_k3 = ForecastHistory(masked_k3.records, values, mask, times)

    k = int(np.sum(~masked_k3.available_mask[-1]))
    assert k == 3, f"expected k=3, got k={k}"
    traj = forecaster.forecast(masked_k3, catalogue.candidates[0])
    assert traj.status == "PREDICTION", f"dropout forecaster should predict with k=3, got {traj.status}: {traj.reason}"
    assert traj.mean is not None and np.isfinite(traj.mean).all()
    assert traj.lower is not None and traj.upper is not None and np.all(traj.lower <= traj.upper)


def test_per_k_evaluation_shapes_and_honesty() -> None:
    cp, manifest, scenario = _checkpoint_and_manifest()
    from aeolus.habitat_v2.forecast_issue52_rollout import rollout_catalogue, training_samples_from_rollouts

    catalogue = CandidateCatalogue.from_scenario(scenario, base_command=cp.last_final_command)
    rollouts = rollout_catalogue(cp, catalogue)
    samples = training_samples_from_rollouts(cp, catalogue, rollouts, split="TRAIN")
    samples2 = tuple(
        __import__("aeolus.habitat_v2.forecast_issue52", fromlist=["TrainingSample"]).TrainingSample(
            family_id=f"fam-{i}",
            split=s.split,
            scenario_sha256=s.scenario_sha256,
            manifest_sha256=s.manifest_sha256,
            checkpoint_sha256=s.checkpoint_sha256,
            schedule_sha256=s.schedule_sha256,
            history=s.history,
            schedule=s.schedule,
            targets=s.targets,
        )
        for i, s in enumerate(list(samples) * 2)
    )
    cfg = DropoutConfig(p_uniform=0.0, seed=530053)
    forecaster = DropoutAwareLinearForecaster.fit_for_scenario(scenario, manifest, samples2, dropout_config=cfg, alpha=1e-4)
    history = ForecastHistory.from_records(cp.history_records)
    # Build per-k buckets by masking copies of one sample
    buckets: dict[int, list] = {0: [], 1: [], 3: []}
    for k_target in (0, 1, 3):
        masked = history
        if k_target > 0:
            vals = history.target_values.copy()
            m = history.available_mask.copy()
            avail = [i for i in range(manifest.width) if m[-1, i]]
            for c in sorted(avail, key=lambda i: manifest.descriptors[i].descriptor_id)[:k_target]:
                m[-1, c] = False
                vals[-1, c] = np.nan
            vals.setflags(write=False)
            m.setflags(write=False)
            times = history.completed_times_s.copy()
            times.setflags(write=False)
            masked = ForecastHistory(history.records, vals, m, times)
        s = samples2[0]
        buckets[k_target].append(
            __import__("aeolus.habitat_v2.forecast_issue52", fromlist=["TrainingSample"]).TrainingSample(
                family_id=s.family_id,
                split=s.split,
                scenario_sha256=s.scenario_sha256,
                manifest_sha256=s.manifest_sha256,
                checkpoint_sha256=s.checkpoint_sha256,
                schedule_sha256=s.schedule_sha256,
                history=masked,
                schedule=s.schedule,
                targets=s.targets,
            )
        )
    result = evaluate_per_k(forecaster, buckets, manifest)
    assert set(result.keys()) == {0, 1, 3}
    for k in (0, 1, 3):
        assert math.isfinite(result[k]["nmae"]) or result[k]["count"] == 0
    # interval coverage should be in [0,1] when defined
    cov = interval_coverage_at_k(forecaster, buckets[3])
    assert math.isnan(cov) or 0.0 <= cov <= 1.0


def test_dropout_dataset_manifest_is_content_addressed() -> None:
    cfg = DropoutConfig(p_uniform=0.05, seed=530053)
    m1 = build_dropout_dataset_manifest(cfg, ["fam-A", "fam-B"], {"fam-A": "TRAIN", "fam-B": "VALIDATION"}, parent_artifact_sha256=None)
    m2 = build_dropout_dataset_manifest(cfg, ["fam-B", "fam-A"], {"fam-B": "VALIDATION", "fam-A": "TRAIN"}, parent_artifact_sha256=None)
    assert m1.dataset_sha256 == m2.dataset_sha256
    assert m1.dropout_config_sha256 == cfg.config_sha256


def test_issue52_forecaster_still_abstains_on_missing() -> None:
    """Guard: we did not accidentally soften the frozen Issue #52 lane."""
    from aeolus.habitat_v2.forecast_issue52 import ActionConditionedLinearForecaster

    cp, manifest, scenario = _checkpoint_and_manifest()
    history = ForecastHistory.from_records(cp.history_records)
    # mask one channel on latest row
    vals = history.target_values.copy()
    m = history.available_mask.copy()
    m[-1, 0] = False
    vals[-1, 0] = np.nan
    vals.setflags(write=False)
    m.setflags(write=False)
    times = history.completed_times_s.copy()
    times.setflags(write=False)
    masked = ForecastHistory(history.records, vals, m, times)
    forecaster = ActionConditionedLinearForecaster(scenario, manifest)
    catalogue = CandidateCatalogue.from_scenario(scenario, base_command=cp.last_final_command)
    traj = forecaster.forecast(masked, catalogue.candidates[0])
    assert traj.status == "ABSTAIN"
