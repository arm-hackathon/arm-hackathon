"""Focused tests for the V7 named-fault escalation cycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.config import load_scenario
from aeolus.residual_features import ResidualFeatureProjector
from aeolus.scenario import run_scenario
from aeolus.specialists_v7 import (
    V7EscalatedRulePolicy,
    V7GatedResidualPolicy,
    residual_window_vector,
)
from aeolus.v7_centroid import V7ResidualCentroid

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
V6_DIR = SCENARIOS / "v6"


def _windows(scenario_name: str, zone: str = "cabin_a", start: int = 0, count: int = 200):
    config = load_scenario(scenario_name)
    records = tuple(run_scenario(config))
    for i in range(start, min(start + count, len(records) - 10)):
        yield config, records[i : i + 10], zone


def test_residual_window_vector_has_stable_width():
    config = load_scenario(V6_DIR / "room-balanced.json")
    records = tuple(run_scenario(config))[:10]
    projector = ResidualFeatureProjector(config)
    zone_ids = tuple(zone.id for zone in config.non_processing_zones())
    vector = residual_window_vector(projector, records, zone_ids)
    assert vector.ndim == 1
    from aeolus.residual_features import PhysicalResidualFeatures, SensorResidualFeatures
    from dataclasses import fields

    expected = len(zone_ids) * (len(fields(SensorResidualFeatures)) + len(fields(PhysicalResidualFeatures)))
    assert vector.size == expected
    assert np.isfinite(vector).all()


def test_residual_centroid_roundtrip_artifact():
    vectors = [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [9.0, 8.0, 7.0],
        [9.2, 8.1, 7.2],
    ]
    labels = ["nominal", "nominal", "blocked_path", "blocked_path"]
    centroid = V7ResidualCentroid.fit(vectors, labels, feature_width=3)
    probabilities = centroid.predict_probabilities([[1.05, 2.05, 3.05]])
    assert probabilities.shape == (1, 4)
    restored = V7ResidualCentroid.from_dict(centroid.as_dict())
    again = restored.predict_probabilities([[1.05, 2.05, 3.05]])
    np.testing.assert_allclose(probabilities, again)


def test_residual_centroid_rejects_width_mismatch():
    with pytest.raises(ValueError, match="width"):
        V7ResidualCentroid.fit([[1.0, 2.0]], ["nominal"], feature_width=3)


def test_escalated_rules_name_blocked_as_blocked():
    scenario = V6_DIR / "room-balanced.json"
    # inject a blocked fault directly in a copy scenario document
    from aeolus.config import parse_scenario

    document = json.loads(scenario.read_text())
    document["fault_profiles"] = [
        {"type": "blocked_path", "connection_id": "cabin_a_to_processing", "start_tick": 30, "blocked_effectiveness": 0.65}
    ]
    config = parse_scenario(document)
    records = tuple(run_scenario(config))
    policy = V7EscalatedRulePolicy(config)
    post_onset = records[39:49]
    label = policy.label_window(post_onset)
    assert label == "blocked_path"


def test_escalated_rules_name_gradual_as_gradual():
    scenario = V6_DIR / "room-balanced.json"
    from aeolus.config import parse_scenario

    document = json.loads(scenario.read_text())
    document["fault_profiles"] = [
        {
            "type": "gradual_primary_fan_degradation",
            "connection_id": "cabin_a_to_processing",
            "start_tick": 30,
            "end_tick": 60,
            "end_effectiveness": 0.75,
        }
    ]
    config = parse_scenario(document)
    records = tuple(run_scenario(config))
    policy = V7EscalatedRulePolicy(config)
    # Replay windows sequentially (stride 1) like the stateful evaluator does.
    labels = [policy.label_window(records[i : i + 10]) for i in range(len(records) - 10)]
    named = [label for label in labels if label != "nominal"]
    assert named, "gradual fault produced no named detection"
    assert all(label == "gradual_primary_fan_degradation" for label in named), named


def test_escalated_rules_keep_healthy_nominal():
    config = load_scenario(V6_DIR / "room-balanced.json")
    records = tuple(run_scenario(config))
    policy = V7EscalatedRulePolicy(config)
    for start in (0, 20, 40):
        assert policy.label_window(records[start : start + 10]) == "nominal"


def test_gated_residual_policy_abstains_without_confidence():
    config = load_scenario(V6_DIR / "room-balanced.json")
    records = tuple(run_scenario(config))
    vectors = [[1.0, 2.0], [9.0, 8.0]]
    labels = ["nominal", "blocked_path"]
    centroid = V7ResidualCentroid.fit(vectors, labels, feature_width=2)
    policy = V7GatedResidualPolicy(config, centroid, min_confidence=0.99)
    # healthy window: no concern -> nominal regardless of classifier
    assert policy.label_window(records[0:10]) == "nominal"
