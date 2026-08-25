from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.forecast.projection import (
    project_history_window,
    project_proposed_action,
)
from aeolus.habitat_v2.forecast_issue55_race import (
    build_family_scenario,
    episode_nonce,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import (
    FEATURE_COUNT,
    ISSUE56_V2_SCHEMA_VERSION,
    Issue56V2RiskError,
    V2RiskModel,
    V2RiskSample,
    V2RiskScore,
    alarm_family_slot_indices,
    load_v2_samples,
    select_risk_filtered_point,
    select_risk_only,
    v2_counterfactual_label,
    v2_feature_vector,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics import initial_state


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sample(
    family_id: str,
    split: str,
    value: float,
    event: float,
    exposure: float,
    maximum: float,
) -> V2RiskSample:
    features = np.full(FEATURE_COUNT, value, dtype=np.float32)
    snapshot_ids = tuple("e" * 64 for _ in range(16))
    label_sha256 = "d" * 64
    body = {
        "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.sample",
        "family_id": family_id,
        "decision_step": 16,
        "split": split,
        "track": "effect_4",
        "action_id": "normal-occupied-v1",
        "scenario_sha256": "f" * 64,
        "snapshot_sha256": list(snapshot_ids),
        "input_manifest_sha256": "a" * 64,
        "target_manifest_sha256": "b" * 64,
        "features_f32_hex": features.tobytes().hex(),
        "crossing_event": event,
        "safety_exposure": exposure,
        "maximum_crossing": maximum,
        "label_sha256": label_sha256,
    }
    return V2RiskSample(
        family_id,
        16,
        split,
        "effect_4",
        "normal-occupied-v1",
        "f" * 64,
        snapshot_ids,
        "a" * 64,
        "b" * 64,
        features,
        event,
        exposure,
        maximum,
        label_sha256,
        _sha(body),
    )


def _zero_model(**overrides: object) -> V2RiskModel:
    values: dict[str, object] = {
        "feature_mean": np.zeros(FEATURE_COUNT, dtype=np.float64),
        "feature_scale": np.ones(FEATURE_COUNT, dtype=np.float64),
        "event_target_mean": 0.0,
        "event_target_scale": 1.0,
        "event_coefficients": np.zeros(FEATURE_COUNT, dtype=np.float64),
        "severity_target_mean": 0.0,
        "severity_target_scale": 1.0,
        "severity_coefficients": np.zeros(FEATURE_COUNT, dtype=np.float64),
        "maximum_target_mean": 0.0,
        "maximum_target_scale": 1.0,
        "maximum_coefficients": np.zeros(FEATURE_COUNT, dtype=np.float64),
        "model_id": "test-v2-model",
    }
    values.update(overrides)
    return V2RiskModel(**values)


def _history_at_step_16() -> tuple[object, object, object]:
    bundle = load_forecast_contracts(REPO_ROOT)
    scenario = build_family_scenario(bundle.development_scenario, 0)
    hmc = HabitatManagementComputer.reset(
        scenario,
        bundle.hmc_contract,
        episode_nonce("issue56-v2-test-history"),
    )
    snapshots: dict[int, tuple[object, object]] = {}
    for step in range(17):
        observed = hmc.observe()
        assert isinstance(observed, tuple)
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snapshots[step] = (snapshot, verification)
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        assert hasattr(arbitration, "final_command")
        stepped = hmc.step()
        assert hasattr(stepped, "plant_receipt_digest")
    history = project_history_window(
        bundle,
        tuple(snapshots[index] for index in range(1, 17)),
        window_steps=16,
    )
    return bundle, scenario, history


def test_v2_feature_projection_binds_public_targets_and_snapshot_provenance() -> None:
    bundle, _, history = _history_at_step_16()
    action = project_proposed_action(bundle, bundle.actions[0].command)
    features = v2_feature_vector(
        history,
        action,
        decision_step=16,
        alarm_family_slots=alarm_family_slot_indices(bundle),
    )
    assert features.shape == (FEATURE_COUNT,)
    assert len(history.snapshot_sha256) == 16
    assert history.steps == tuple(range(1, 17))
    with pytest.raises(Issue56V2RiskError, match="alarm binding"):
        v2_feature_vector(history, action, decision_step=16)


def test_v2_counterfactual_labels_keep_track_identity_and_provenance() -> None:
    bundle = load_forecast_contracts(REPO_ROOT)
    scenario = build_family_scenario(bundle.development_scenario, 0)
    current = bundle.actions[0]
    candidate = bundle.actions[1]
    effect = v2_counterfactual_label(
        scenario,
        tuple(bundle.topology.zone_ids),
        initial_state(scenario),
        candidate.action_id,
        current.command.to_mapping(),
        candidate.command.to_mapping(),
        track="effect_4",
    )
    persistent = v2_counterfactual_label(
        scenario,
        tuple(bundle.topology.zone_ids),
        initial_state(scenario),
        candidate.action_id,
        current.command.to_mapping(),
        candidate.command.to_mapping(),
        track="persistent_32",
    )
    assert effect.eligible and persistent.eligible
    assert effect.track == "effect_4"
    assert persistent.track == "persistent_32"
    assert len(effect.state_digests) == len(persistent.state_digests) == 32
    assert effect.label_sha256 != persistent.label_sha256


def test_v2_model_fit_calibration_and_artifact_round_trip() -> None:
    train = (
        _sample("family-a", "TRAIN", 0.0, 0.0, 0.0, 0.0),
        _sample("family-a", "TRAIN", 0.1, 1.0, 1.0, 0.2),
        _sample("family-b", "TRAIN", 0.2, 0.0, 0.0, 0.0),
        _sample("family-b", "TRAIN", 0.3, 1.0, 3.0, 0.4),
    )
    validation = (
        _sample("family-c", "VALIDATION", 0.15, 0.0, 0.0, 0.0),
        _sample("family-c", "VALIDATION", 0.25, 1.0, 2.0, 0.3),
    )
    model = V2RiskModel.fit(train)
    calibrated = model.calibrate(validation)
    prediction = calibrated.predict_features(train[0].features_f32)
    restored = V2RiskModel.from_mapping(calibrated.to_mapping())
    assert 0.0 <= prediction.event_probability <= 1.0
    assert 0.0 <= prediction.upper_event_probability <= 1.0
    assert restored.to_mapping() == calibrated.to_mapping()
    with pytest.raises(Issue56V2RiskError, match="VALIDATION"):
        model.calibrate(train)


def test_v2_extreme_severity_is_finite_and_rejected() -> None:
    model = _zero_model(severity_target_mean=10_000.0)
    prediction = model.predict_features(np.zeros(FEATURE_COUNT, dtype=np.float32))
    assert np.isfinite(prediction.upper_expected_exposure)
    assert np.isfinite(prediction.upper_maximum_crossing)
    assert prediction.hard_ineligible


def test_v2_sample_loader_and_rankers_fail_closed() -> None:
    sample = _sample("family-a", "TRAIN", 0.0, 0.0, 0.0, 0.0)
    restored = load_v2_samples([sample.to_mapping()])
    assert len(restored) == 1
    assert restored[0].sample_sha256 == sample.sample_sha256
    malformed = sample.to_mapping()
    malformed["unexpected"] = True
    with pytest.raises(Issue56V2RiskError, match="fields drift"):
        load_v2_samples([malformed])

    eligible = V2RiskScore("a", False, 1.0, 0.1, 0.1, 0.1, None)
    rejected = V2RiskScore("b", True, 0.0, 1.0, 10.0, 1.0, "risk")
    assert select_risk_filtered_point((rejected, eligible)) == eligible
    assert select_risk_only((rejected, eligible)) == eligible
    with pytest.raises(Issue56V2RiskError, match="duplicate"):
        select_risk_only((eligible, eligible))
