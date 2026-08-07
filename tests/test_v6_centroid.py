"""Deterministic learned V6 centroid candidate."""

from __future__ import annotations

import numpy as np

from aeolus.v6_centroid import V6CentroidClassifier


def test_centroid_classifier_fits_only_declared_classes_and_returns_probabilities():
    rows = [
        {"label": "nominal", "features": [[0.0, 0.0], [0.0, 0.0]]},
        {"label": "nominal", "features": [[0.1, 0.0], [0.0, 0.1]]},
        {"label": "frozen_sensor", "features": [[3.0, 3.0], [3.0, 3.0]]},
        {"label": "frozen_sensor", "features": [[3.1, 3.0], [3.0, 3.1]]},
    ]

    classifier = V6CentroidClassifier.fit(rows, window_ticks=2, feature_width=2)
    probabilities = classifier.predict_probabilities([[[0.0, 0.0], [0.0, 0.0]]])

    assert probabilities.shape == (1, 4)
    assert np.isclose(probabilities.sum(), 1.0)
    assert classifier.class_names == (
        "nominal",
        "frozen_sensor",
        "blocked_path",
        "gradual_primary_fan_degradation",
    )
    assert classifier.predict_label([[0.0, 0.0], [0.0, 0.0]]) == "nominal"
    assert classifier.predict_label([[3.0, 3.0], [3.0, 3.0]]) == "frozen_sensor"


def test_centroid_classifier_rejects_transition_and_unknown_training_labels():
    rows = [{"label": "excluded_transition", "features": [[0.0], [0.0]]}]

    try:
        V6CentroidClassifier.fit(rows, window_ticks=2, feature_width=1)
    except ValueError as exc:
        assert "no usable" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("transition-only fit rows must be rejected")
