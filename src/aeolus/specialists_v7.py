"""V7 named-fault escalation policies over the V6 concern layer.

V6 failed because the concern layer was blind by design (never named a fault)
and the raw-context centroid could not transfer across room families. V7 keeps
the precision-1.0 specialist concerns as a gate and adds:

- ``V7EscalatedRulePolicy``: a calibrated hand-written baseline that names
  faults inside concern windows (blocked vs gradual via a jump/ramp ratio,
  frozen via a trend/expected-change test);
- ``V7GatedResidualPolicy``: a learned candidate that only classifies inside
  concern windows, using residual/trend features normalised across room
  families instead of raw observable context.

Both policies expose ``reset()`` and ``label_window(records)`` and are
consumed by the existing stateful V6 evaluator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from aeolus.config import HabitatConfig
from aeolus.residual_features import PhysicalResidualFeatures, ResidualFeatureProjector, SensorResidualFeatures
from aeolus.trace import TickRecord

V7_CLASS_NAMES = (
    "nominal",
    "frozen_sensor",
    "blocked_path",
    "gradual_primary_fan_degradation",
)


class V7Classifier(Protocol):
    """A residual-vector classifier exposing the V6 centroid contract."""

    @property
    def class_names(self) -> tuple[str, ...]: ...

    def predict_probabilities(
        self, vectors: Sequence[Sequence[float]]
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class V7EscalationParameters:
    """Immutable, calibration-selected thresholds for named-fault escalation."""

    sensor_trend_abs_max: float = 0.0025
    expected_change_proxy: float = 0.1
    sensor_max_delta: float = 0.02
    settled_residual_threshold: float = 0.25

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("v7 escalation thresholds must be finite numbers")
            if float(value) < 0.0:
                raise ValueError("v7 escalation thresholds must be non-negative")


def residual_window_vector(
    projector: ResidualFeatureProjector,
    records: Sequence[TickRecord],
    zone_ids: Sequence[str],
) -> NDArray[np.float64]:
    """Return a fixed-width residual vector: per zone, sensor then physical features.

    Fields follow the dataclass declaration order so the width is stable across
    runs. Sensor (13) + physical (13) per zone, zones in topology order.
    """
    sensor_names = tuple(f.name for f in fields(SensorResidualFeatures))
    physical_names = tuple(f.name for f in fields(PhysicalResidualFeatures))
    parts: list[float] = []
    for zone_id in zone_ids:
        sensor = asdict(projector.sensor_features(records, zone_id))
        physical = asdict(projector.physical_features(records, zone_id))
        parts.extend(float(sensor[name]) for name in sensor_names)
        parts.extend(float(physical[name]) for name in physical_names)
    if not parts or not all(math.isfinite(value) for value in parts):
        raise ValueError("v7 residual window vector must be finite and non-empty")
    return np.asarray(parts, dtype=np.float64)


class V7EscalatedRulePolicy:
    """Calibrated hand-written baseline: specialists gate, rules name faults.

    Physical escalation is stateful: the first isolated persistent residual
    window locks the mode (blocked = step already settled, gradual = residual
    still rising), and later settled windows inherit that mode. This is the
    only way to name a fault after its onset window has passed.

    A locked or decidable physical escalation outranks a coincident sensor
    pattern: during a gradual ramp tail the shared loop also shifts the
    corroborating sensors, so a simultaneous ``sensor`` hit must not downgrade
    the already-decided physical fault to ``uncertain``.
    """

    def __init__(
        self,
        config: HabitatConfig,
        parameters: V7EscalationParameters | None = None,
    ) -> None:
        self._projector = ResidualFeatureProjector(config)
        self._parameters = parameters or V7EscalationParameters()
        self._zone_ids = tuple(zone.id for zone in config.non_processing_zones())
        self._mode: dict[str, str] = {}

    def reset(self) -> None:
        """Clear the per-zone escalation mode at each replay-stream boundary."""
        self._mode = {}

    def _physical_assessment(self, records: Sequence[TickRecord]) -> tuple[bool, str]:
        """Return (concern, best zone) for isolated persistent residual."""
        best: tuple[float, str] = (0.0, "")
        for zone_id in self._zone_ids:
            feature = self._projector.physical_features(records, zone_id)
            if feature.normalized_residual < 0.1:
                continue
            if feature.transient_proxy > 0.2:
                continue
            if feature.isolation_ratio < 1.0:
                continue
            if feature.residual_persistence < 0.5:
                continue
            score = feature.normalized_residual * feature.isolation_ratio * feature.residual_persistence
            if score > best[0]:
                best = (score, zone_id)
        return (best[1] != "", best[1])

    def _sensor_assessment(self, records: Sequence[TickRecord]) -> tuple[bool, str]:
        """Return (concern, best zone) for flat trend with corroborated change."""
        best: tuple[float, str] = (0.0, "")
        for zone_id in self._zone_ids:
            feature = self._projector.sensor_features(records, zone_id)
            if abs(feature.sensor_slope) > self._parameters.sensor_trend_abs_max:
                continue
            if feature.sensor_max_delta > self._parameters.sensor_max_delta:
                continue
            if feature.expected_change_proxy < self._parameters.expected_change_proxy:
                continue
            score = feature.expected_change_proxy * (1.0 - abs(feature.sensor_slope))
            if score > best[0]:
                best = (score, zone_id)
        return (best[1] != "", best[1])

    def _escalate_physical(self, records: Sequence[TickRecord], zone_id: str) -> str:
        """Name a physical concern; lock the mode on the first concern window."""
        if zone_id in self._mode:
            return self._mode[zone_id]
        feature = self._projector.physical_features(records, zone_id)
        # A blocked path settles fast: the residual is already high and flat.
        # A gradual ramp is still rising through the concern threshold.
        mode = (
            "blocked_path"
            if feature.normalized_residual >= self._parameters.settled_residual_threshold
            else "gradual_primary_fan_degradation"
        )
        self._mode[zone_id] = mode
        return mode

    def label_window(self, records: Sequence[TickRecord]) -> str:
        """Gate by concerns; name the fault inside a concern window.

        Physical evidence is decisive: once an isolated persistent residual
        appears, the mode is locked and named (blocked vs gradual), and later
        windows inherit it even if the sensor side also corroborates. The
        frozen-sensor name is reserved for windows with sensor evidence only.
        """
        physical, physical_zone = self._physical_assessment(records)
        sensor, _sensor_zone = self._sensor_assessment(records)
        if physical:
            return self._escalate_physical(records, physical_zone)
        if sensor:
            return "frozen_sensor"
        return "nominal"


class V7GatedResidualPolicy:
    """Learned candidate: centroid over residual vectors, gated by concerns."""

    def __init__(
        self,
        config: HabitatConfig,
        classifier: V7Classifier,
        *,
        min_confidence: float,
        parameters: V7EscalationParameters | None = None,
    ) -> None:
        self._projector = ResidualFeatureProjector(config)
        self._parameters = parameters or V7EscalationParameters()
        self._zone_ids = tuple(zone.id for zone in config.non_processing_zones())
        self._classifier = classifier
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("v7 policy minimum confidence must be in [0, 1]")
        self._min_confidence = min_confidence

    def reset(self) -> None:
        """The gated centroid retains no replay state."""

    def _physical_concern(self, records: Sequence[TickRecord]) -> bool:
        return any(
            self._projector.physical_features(records, zone_id).normalized_residual >= 0.1
            for zone_id in self._zone_ids
        )

    def _sensor_concern(self, records: Sequence[TickRecord]) -> bool:
        return any(
            self._projector.sensor_features(records, zone_id).expected_change_proxy
            >= self._parameters.expected_change_proxy
            for zone_id in self._zone_ids
        )

    def label_window(self, records: Sequence[TickRecord]) -> str:
        """Emit a named class only inside a concern window with sufficient confidence."""
        physical = self._physical_concern(records)
        sensor = self._sensor_concern(records)
        if not physical and not sensor:
            return "nominal"
        vector = residual_window_vector(self._projector, records, self._zone_ids)
        probabilities = self._classifier.predict_probabilities([vector.tolist()])[0]
        index = int(np.argmax(probabilities))
        if float(probabilities[index]) < self._min_confidence:
            return "uncertain"
        return self._classifier.class_names[index]
