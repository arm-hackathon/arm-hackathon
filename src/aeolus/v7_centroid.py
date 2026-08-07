"""Deterministic V7 residual-vector centroid classifier.

The V6 centroid consumed raw ``observable_context_v1`` windows and failed to
transfer across room families: every held-out window fell below the abstention
threshold. V7 trains the same nearest-centroid geometry on ``residual_window_vector``
output (per-zone sensor + physical residual/trend features, normalised by
capacity/request), which is designed to be room-family invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

V7_CLASS_NAMES = (
    "nominal",
    "frozen_sensor",
    "blocked_path",
    "gradual_primary_fan_degradation",
)


@dataclass(frozen=True)
class V7ResidualCentroid:
    """A deterministic nearest-centroid classifier over fixed-width vectors."""

    feature_width: int
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    centroids: NDArray[np.float64]
    present_classes: NDArray[np.bool_]

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return the fixed V7 prediction vocabulary."""
        return V7_CLASS_NAMES

    @classmethod
    def fit(
        cls,
        vectors: Sequence[Sequence[float]],
        labels: Sequence[str],
        *,
        feature_width: int,
    ) -> "V7ResidualCentroid":
        """Fit normalisation and centroids from labelled residual vectors."""
        if feature_width < 1:
            raise ValueError("v7 residual centroid feature width must be positive")
        if len(vectors) != len(labels) or not vectors:
            raise ValueError("v7 residual centroid requires aligned non-empty vectors and labels")
        matrix = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
        if matrix.shape[1] != feature_width:
            raise ValueError(
                f"v7 residual centroid received width {matrix.shape[1]}, expected {feature_width}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("v7 residual centroid vectors must be finite")
        target = np.asarray([V7_CLASS_NAMES.index(str(label)) for label in labels], dtype=np.int64)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        normalized = (matrix - means) / scales
        centroids = np.zeros((len(V7_CLASS_NAMES), normalized.shape[1]), dtype=np.float64)
        present = np.zeros(len(V7_CLASS_NAMES), dtype=np.bool_)
        for class_index in range(len(V7_CLASS_NAMES)):
            members = normalized[target == class_index]
            if len(members):
                centroids[class_index] = members.mean(axis=0)
                present[class_index] = True
        return cls(
            feature_width=feature_width,
            means=means,
            scales=scales,
            centroids=centroids,
            present_classes=present,
        )

    def predict_probabilities(
        self, vectors: Sequence[Sequence[float]]
    ) -> NDArray[np.float64]:
        """Return fixed-vocabulary probabilities for validated residual vectors."""
        if not vectors:
            return np.empty((0, len(V7_CLASS_NAMES)), dtype=np.float64)
        matrix = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
        if matrix.shape[1] != self.feature_width:
            raise ValueError(
                f"v7 residual centroid received width {matrix.shape[1]}, expected {self.feature_width}"
            )
        normalized = (matrix - self.means) / self.scales
        squared_distance = np.mean((normalized[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        logits = -squared_distance
        logits[:, ~self.present_classes] = -np.inf
        maximum = np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits - maximum)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ValueError("v7 residual centroid cannot score a vector without a fitted class")
        return probabilities

    def as_dict(self) -> dict[str, object]:
        """Return a strict JSON-safe artifact representation."""
        return {
            "schema_version": "aeolus_v7_residual_centroid_v1",
            "class_names": list(V7_CLASS_NAMES),
            "feature_width": self.feature_width,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "centroids": self.centroids.tolist(),
            "present_classes": self.present_classes.tolist(),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> "V7ResidualCentroid":
        """Reconstruct a centroid from its JSON-safe representation."""
        if document.get("schema_version") != "aeolus_v7_residual_centroid_v1":
            raise ValueError("v7 residual centroid schema_version is unsupported")
        raw_present = document["present_classes"]
        if not isinstance(raw_present, list):
            raise ValueError("v7 residual centroid present_classes must be a list")
        return cls(
            feature_width=int(document["feature_width"]),
            means=np.asarray(document["means"], dtype=np.float64),
            scales=np.asarray(document["scales"], dtype=np.float64),
            centroids=np.asarray(document["centroids"], dtype=np.float64),
            present_classes=np.asarray([bool(value) for value in raw_present], dtype=np.bool_),
        )
