from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast_issue55_race import (
    build_family_scenario,
    deterministic_family_ids,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    EPISODE_STEPS,
    ISSUE56_V3_SCHEMA_VERSION,
    Issue56V3RiskError,
    V3_HORIZONS,
    V3_LABEL_TRACK,
    V3HorizonMetric,
    V3PolicyLabel,
    V3RiskModel,
    V3RiskSample,
    collect_v3_family_samples,
    load_v3_samples,
    run_v3_episode,
    v3_family_split,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import FEATURE_COUNT


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v3_split_preserves_pairs_and_has_all_three_strata() -> None:
    family_ids = deterministic_family_ids(32)
    split = v3_family_split(family_ids)

    assert {split[family_id] for family_id in family_ids} == {
        "TRAIN",
        "VALIDATION",
        "EVALUATION",
    }
    assert {
        label: tuple(split.values()).count(label)
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
    } == {"TRAIN": 20, "VALIDATION": 6, "EVALUATION": 6}
    for index in range(0, len(family_ids), 2):
        assert split[family_ids[index]] == split[family_ids[index + 1]]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _synthetic_sample(family: str, split: str, value: float, event: float) -> V3RiskSample:
    metrics = tuple(
        V3HorizonMetric(horizon, event, event * (value + 1.0), event * (value + 0.5))
        for horizon in V3_HORIZONS
    )
    remaining = V3HorizonMetric(
        EPISODE_STEPS - 16,
        event,
        event * (value + 2.0),
        event * (value + 1.0),
    )
    digests = tuple("a" * 64 for _ in range(EPISODE_STEPS - 16))
    label_body = {
        "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.label",
        "track": V3_LABEL_TRACK,
        "action_id": "action-v3",
        "decision_step": 16,
        "current_command_sha256": "b" * 64,
        "requested_command_sha256": "c" * 64,
        "final_command_sha256": "d" * 64,
        "executed_command_sha256": "d" * 64,
        "disposition": "PROPOSED_ACCEPTED",
        "horizon_metrics": [metric.to_mapping() for metric in metrics],
        "remaining_steps": EPISODE_STEPS - 16,
        "remaining_metric": remaining.to_mapping(),
        "state_digests": list(digests),
        "trace_sha256": "e" * 64,
    }
    label = V3PolicyLabel(
        "action-v3",
        16,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "d" * 64,
        "PROPOSED_ACCEPTED",
        metrics,
        EPISODE_STEPS - 16,
        remaining,
        digests,
        "e" * 64,
        _digest(label_body),
    )
    features = np.full(FEATURE_COUNT, value, dtype=np.float32)
    body = {
        "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.sample",
        "family_id": family,
        "decision_step": 16,
        "split": split,
        "action_id": "action-v3",
        "scenario_sha256": "f" * 64,
        "features_f32_hex": features.tobytes().hex(),
        "label": label.to_mapping(),
    }
    return V3RiskSample(
        family,
        16,
        split,
        "action-v3",
        "f" * 64,
        features,
        label,
        _digest(body),
    )


def _zero_model() -> V3RiskModel:
    return V3RiskModel(
        np.zeros(FEATURE_COUNT),
        np.ones(FEATURE_COUNT),
        np.zeros(4),
        np.zeros((4, FEATURE_COUNT)),
        np.zeros(4),
        np.ones(4),
        np.zeros((4, FEATURE_COUNT)),
        np.zeros(4),
        np.ones(4),
        np.zeros((4, FEATURE_COUNT)),
        np.zeros(4),
        np.ones(4),
        np.zeros(4),
        np.zeros(4),
    )


def test_v3_labels_bind_hmc_persistence_and_all_command_identities() -> None:
    bundle = load_forecast_contracts(REPO_ROOT)
    scenario = build_family_scenario(bundle.development_scenario, 0)

    samples = collect_v3_family_samples(
        bundle,
        scenario,
        "issue55-v3-test-family",
        split="TRAIN",
    )

    assert len(samples) == 13 * 4
    assert {sample.label.track for sample in samples} == {V3_LABEL_TRACK}
    for sample in samples:
        label = sample.label
        assert label.to_mapping()["schema_version"] == f"{ISSUE56_V3_SCHEMA_VERSION}.label"
        assert tuple(metric.horizon_steps for metric in label.horizon_metrics) == V3_HORIZONS
        assert label.remaining_steps == EPISODE_STEPS - label.decision_step
        assert len(label.state_digests) == label.remaining_steps
        assert label.final_command_sha256 == label.executed_command_sha256
        assert label.requested_command_sha256 != label.current_command_sha256
        assert label.trace_sha256 != label.label_sha256


def test_v3_logistic_multihorizon_model_calibrates_and_round_trips() -> None:
    train = tuple(
        _synthetic_sample(
            f"family-{index % 2}",
            "TRAIN",
            float(index),
            float(index % 2),
        )
        for index in range(8)
    )
    validation = tuple(
        _synthetic_sample(
            f"validation-{index % 2}",
            "VALIDATION",
            float(index) + 0.5,
            float(index % 2),
        )
        for index in range(4)
    )
    model = V3RiskModel.fit(train)
    calibrated = model.calibrate(validation)
    prediction = calibrated.predict_features(train[0].features_f32)
    restored = V3RiskModel.from_mapping(calibrated.to_mapping())
    assert tuple(item.horizon_steps for item in prediction.horizons) == (4, 16, 32, 0)
    assert all(0.0 <= item.event_probability <= 1.0 for item in prediction.horizons)
    assert np.all(calibrated.calibration_slopes >= 0.0)
    assert restored.to_mapping() == calibrated.to_mapping()
    assert load_v3_samples([train[0].to_mapping()])[0].sample_sha256 == train[0].sample_sha256
    with pytest.raises(Issue56V3RiskError, match="VALIDATION"):
        model.calibrate(train)


def test_v3_calibration_rejects_one_class_validation_horizon() -> None:
    train = tuple(
        _synthetic_sample(
            f"family-{index % 2}",
            "TRAIN",
            float(index),
            float(index % 2),
        )
        for index in range(8)
    )
    validation = tuple(
        _synthetic_sample(f"validation-{index % 2}", "VALIDATION", float(index), 0.0)
        for index in range(4)
    )

    with pytest.raises(Issue56V3RiskError, match="both event classes"):
        V3RiskModel.fit(train).calibrate(validation)


def test_v3_fit_accepts_one_positive_severity_row_per_horizon() -> None:
    train = tuple(
        _synthetic_sample(
            f"family-{index % 2}",
            "TRAIN",
            float(index),
            float(index == 0),
        )
        for index in range(8)
    )

    model = V3RiskModel.fit(train)

    assert np.allclose(model.severity_target_scales, 1.0)
    assert np.allclose(model.maximum_target_scales, 1.0)
    assert np.allclose(model.severity_coefficients, 0.0)
    assert np.allclose(model.maximum_coefficients, 0.0)


def test_v3_episode_records_common_window_and_replay_evidence() -> None:
    bundle = load_forecast_contracts(REPO_ROOT)
    scenario = build_family_scenario(bundle.development_scenario, 0)
    record = run_v3_episode(
        bundle,
        scenario,
        "risk_only_v3",
        "issue55-v3-episode-family",
        0,
        _zero_model(),
        None,
    )
    assert record.decision_steps == tuple(range(16, 65, 4))
    assert len(record.decisions) == 13
    assert record.replay_verified
    assert record.authority_verified
    assert record.provenance_verified
    assert record.trace_canonical_bytes
