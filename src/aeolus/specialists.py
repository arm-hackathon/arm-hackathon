"""Conservative V6 observable-only diagnostic specialists.

These rules issue evidence concerns, not simulator-truth diagnoses. They are the
baseline that later learned specialists must beat under the same residual and
observable-context contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from aeolus.config import HabitatConfig
from aeolus.residual_features import ResidualFeatureProjector
from aeolus.trace import TickRecord


@dataclass(frozen=True)
class ConditionalRuleParameters:
    """Immutable thresholds for the auditable V6 conditional baseline."""

    sensor_max_delta: float = 0.0015
    expected_change_proxy: float = 0.1
    residual_threshold: float = 0.1
    isolation_threshold: float = 1.0
    persistence_threshold: float = 0.5
    transient_threshold: float = 0.2

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("conditional-rule thresholds must be finite numbers")
            if float(value) < 0.0:
                raise ValueError("conditional-rule thresholds must be non-negative")
        if self.persistence_threshold > 1.0 or self.transient_threshold > 1.0:
            raise ValueError("conditional-rule fractional thresholds must be at most one")


@dataclass(frozen=True)
class SensorHealthAssessment:
    """Inspectable measurement-integrity evidence from an observable window."""

    score: float
    concern: bool
    zone_id: str | None
    reason_code: str


@dataclass(frozen=True)
class PhysicalFlowAssessment:
    """Inspectable physical-flow evidence from an observable window."""

    score: float
    concern: bool
    loop_zone_id: str | None
    reason_code: str


class SensorHealthSpecialist:
    """Require a flat measured sensor plus observable response opportunity."""

    def __init__(
        self,
        projector: ResidualFeatureProjector,
        parameters: ConditionalRuleParameters | None = None,
    ) -> None:
        self._projector = projector
        self._parameters = parameters or ConditionalRuleParameters()

    def reset(self) -> None:
        """Reset retained state; this stateless baseline retains none."""

    def assess_window(self, window: Sequence[TickRecord]) -> SensorHealthAssessment:
        """Return the strongest corroborated sensor-health concern, if any."""
        candidates: list[SensorHealthAssessment] = []
        saw_flat = False
        for zone_id in self._zone_ids:
            feature = self._projector.sensor_features(window, zone_id)
            if feature.sensor_max_delta > self._parameters.sensor_max_delta:
                continue
            saw_flat = True
            if feature.expected_change_proxy < self._parameters.expected_change_proxy:
                continue
            score = feature.expected_change_proxy * (1.0 - min(1.0, feature.sensor_max_delta / max(self._parameters.sensor_max_delta, 1e-12)))
            candidates.append(SensorHealthAssessment(score, True, zone_id, "flat_sensor_with_corroboration"))
        if candidates:
            return max(candidates, key=lambda assessment: (assessment.score, assessment.zone_id or ""))
        return SensorHealthAssessment(0.0, False, None, "no_expected_change" if saw_flat else "sensor_not_flat")

    @property
    def _zone_ids(self) -> tuple[str, ...]:
        topology = self._projector.context_contract.model_input_contract
        return tuple(field.entity_id for field in topology.fields if field.group == "zones")


class PhysicalFlowSpecialist:
    """Require isolated, persistent, settled observable delivery shortfall."""

    def __init__(
        self,
        projector: ResidualFeatureProjector,
        parameters: ConditionalRuleParameters | None = None,
    ) -> None:
        self._projector = projector
        self._parameters = parameters or ConditionalRuleParameters()

    def reset(self) -> None:
        """Reset retained state; this stateless baseline retains none."""

    def assess_window(self, window: Sequence[TickRecord]) -> PhysicalFlowAssessment:
        """Return the strongest local physical-flow concern, if any."""
        candidates: list[PhysicalFlowAssessment] = []
        shared_transient = False
        for zone_id in self._zone_ids:
            feature = self._projector.physical_features(window, zone_id)
            if feature.normalized_residual < self._parameters.residual_threshold:
                continue
            if feature.transient_proxy > self._parameters.transient_threshold:
                shared_transient = True
                continue
            if feature.isolation_ratio < self._parameters.isolation_threshold:
                continue
            if feature.residual_persistence < self._parameters.persistence_threshold:
                continue
            score = feature.normalized_residual * feature.isolation_ratio * feature.residual_persistence
            candidates.append(PhysicalFlowAssessment(score, True, zone_id, "isolated_persistent_residual"))
        if candidates:
            return max(candidates, key=lambda assessment: (assessment.score, assessment.loop_zone_id or ""))
        reason = "shared_capacity_transient" if shared_transient else "insufficient_physical_evidence"
        return PhysicalFlowAssessment(0.0, False, None, reason)

    @property
    def _zone_ids(self) -> tuple[str, ...]:
        topology = self._projector.context_contract.model_input_contract
        return tuple(field.entity_id for field in topology.fields if field.group == "zones")


class V6DecisionPolicy:
    """Merge independent concerns; ambiguity is explicit and non-operational."""

    def __init__(
        self,
        config: HabitatConfig,
        parameters: ConditionalRuleParameters | None = None,
    ) -> None:
        projector = ResidualFeatureProjector(config)
        self.sensor_specialist = SensorHealthSpecialist(projector, parameters)
        self.physical_specialist = PhysicalFlowSpecialist(projector, parameters)

    def reset(self) -> None:
        """Clear all specialist state at each replay-stream boundary."""
        self.sensor_specialist.reset()
        self.physical_specialist.reset()

    def label_window(self, window: Sequence[TickRecord]) -> Literal[
        "nominal", "uncertain", "sensor_health_concern", "physical_flow_concern"
    ]:
        """Return an observable-only concern state for one causal window."""
        sensor = self.sensor_specialist.assess_window(window)
        physical = self.physical_specialist.assess_window(window)
        if sensor.concern and physical.concern:
            return "uncertain"
        if sensor.concern:
            return "sensor_health_concern"
        if physical.concern:
            return "physical_flow_concern"
        return "nominal"
