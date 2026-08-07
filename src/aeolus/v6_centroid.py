"""Deterministic V6 observable-context centroid classifier.

This is an intentionally small learned comparator. It learns normalisation and
class centroids from fit rows only; calibration chooses the abstention threshold.
It is not a deployment claim or a substitute for a richer specialist model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

V6_CLASS_NAMES = (
    "nominal",
    "frozen_sensor",
    "blocked_path",
    "gradual_primary_fan_degradation",
)
_EXCLUDED_LABEL = "excluded_transition"


@dataclass(frozen=True)
class V6CentroidClassifier:
    """A deterministic nearest-centroid classifier over fixed V6 windows."""

    window_ticks: int
    feature_width: int
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    centroids: NDArray[np.float64]
    present_classes: NDArray[np.bool_]

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return the fixed, V6-compatible prediction vocabulary."""
        return V6_CLASS_NAMES

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        window_ticks: int,
        feature_width: int,
    ) -> "V6CentroidClassifier":
        """Fit normalisation and centroids from non-transition rows only."""
        if window_ticks < 1 or feature_width < 1:
            raise ValueError("centroid classifier window dimensions must be positive")
        vectors: list[NDArray[np.float64]] = []
        labels: list[int] = []
        for row in rows:
            label = row.get("label")
            if label == _EXCLUDED_LABEL:
                continue
            if label not in V6_CLASS_NAMES:
                raise ValueError(f"centroid classifier received unsupported label {label!r}")
            vectors.append(_flatten_window(row.get("features"), window_ticks, feature_width))
            labels.append(V6_CLASS_NAMES.index(str(label)))
        if not vectors:
            raise ValueError("centroid classifier has no usable fit rows")
        matrix = np.stack(vectors)
        target = np.asarray(labels, dtype=np.int64)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        normalized = (matrix - means) / scales
        centroids = np.zeros((len(V6_CLASS_NAMES), normalized.shape[1]), dtype=np.float64)
        present = np.zeros(len(V6_CLASS_NAMES), dtype=np.bool_)
        for class_index in range(len(V6_CLASS_NAMES)):
            members = normalized[target == class_index]
            if len(members):
                centroids[class_index] = members.mean(axis=0)
                present[class_index] = True
        return cls(
            window_ticks=window_ticks,
            feature_width=feature_width,
            means=means,
            scales=scales,
            centroids=centroids,
            present_classes=present,
        )

    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
        """Return fixed-vocabulary probabilities for validated causal windows."""
        if not windows:
            return np.empty((0, len(V6_CLASS_NAMES)), dtype=np.float64)
        matrix = np.stack(
            [_flatten_window(window, self.window_ticks, self.feature_width) for window in windows]
        )
        normalized = (matrix - self.means) / self.scales
        squared_distance = np.mean((normalized[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        logits = -squared_distance
        logits[:, ~self.present_classes] = -np.inf
        maximum = np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits - maximum)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ValueError("centroid classifier cannot score a window without a fitted class")
        return probabilities

    def predict_label(self, window: Sequence[Sequence[float]]) -> str:
        """Return the highest-probability fixed V6 class for one window."""
        probabilities = self.predict_probabilities([window])[0]
        return V6_CLASS_NAMES[int(np.argmax(probabilities))]

    def as_dict(self) -> dict[str, object]:
        """Return a strict JSON-safe artifact representation."""
        return {
            "schema_version": "aeolus_v6_centroid_v1",
            "class_names": list(V6_CLASS_NAMES),
            "window_ticks": self.window_ticks,
            "feature_width": self.feature_width,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "centroids": self.centroids.tolist(),
            "present_classes": self.present_classes.astype(bool).tolist(),
        }


def _flatten_window(value: object, window_ticks: int, feature_width: int) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"centroid classifier window is non-numeric: {exc}") from None
    if matrix.shape != (window_ticks, feature_width) or not np.isfinite(matrix).all():
        raise ValueError("centroid classifier window has incompatible shape or non-finite values")
    return matrix.reshape(-1)
