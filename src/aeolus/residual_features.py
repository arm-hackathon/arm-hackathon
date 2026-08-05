"""Transparent, causal residual features for V6 diagnostic specialists.

The projector consumes only fields admitted by ``observable_context_v1``.
It does not inspect simulator state, scenario configuration, labels, seeds, or
fault declarations. ``expected_change_proxy`` is an opportunity signal: an
observable actuator, delivery, or sibling-actuator reconfiguration occurred.
It is not an estimate of hidden CO2 dynamics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aeolus.config import HabitatConfig
from aeolus.observable_context import (
    ObservableContextContract,
    build_observable_context_contract,
    observable_context_v1,
)
from aeolus.trace import TickRecord

_EPSILON = 1e-12


@dataclass(frozen=True)
class SensorResidualFeatures:
    """Causal sensor-health evidence for one non-processing zone."""

    sensor_slope: float
    sensor_range: float
    sensor_max_delta: float
    actuator_setpoint_span: float
    actuator_actual_span: float
    actuator_tracking_residual_mean: float
    actuator_moving_fraction: float
    outbound_request_change: float
    outbound_delivery_change: float
    outbound_residual_change: float
    sibling_actuator_span_max: float
    expected_change_proxy: float


@dataclass(frozen=True)
class PhysicalResidualFeatures:
    """Causal physical-flow evidence for one non-processing outbound loop."""

    request: float
    delivery: float
    residual: float
    normalized_residual: float
    residual_slope: float
    residual_max_jump: float
    residual_persistence: float
    isolation_ratio: float
    capacity_headroom: float
    capacity_contention: float
    actuator_moving_fraction: float
    transient_proxy: float
    settled_proxy: float


class ResidualFeatureProjector:
    """Topology-bound V6 causal feature projector for the fixed hub."""

    def __init__(self, config: HabitatConfig) -> None:
        self._contract = build_observable_context_contract(config)
        self._zone_ids = tuple(zone.id for zone in config.non_processing_zones())
        self._outbound_by_zone = {
            zone_id: config.path_to_processing(zone_id).id for zone_id in self._zone_ids
        }

    @property
    def context_contract(self) -> ObservableContextContract:
        """Return the exact observable input contract used for validation."""
        return self._contract

    def sensor_features(
        self, window: Sequence[TickRecord], zone_id: str
    ) -> SensorResidualFeatures:
        """Return sensor-health evidence from one causal observable window."""
        records = self._validated_window(window)
        self._require_zone(zone_id)
        sensor = _series(records, "zones", zone_id, "sensor_co2_concentration")
        setpoint = _series(records, "actuators", zone_id, "setpoint")
        actual = _series(records, "actuators", zone_id, "actual_position")
        tracking = _series(records, "actuators", zone_id, "tracking_residual")
        moving = _series(records, "actuators", zone_id, "moving")
        outbound_id = self._outbound_by_zone[zone_id]
        request = _series(records, "connections", outbound_id, "requested_airflow")
        delivery = _series(records, "connections", outbound_id, "delivered_airflow")
        residual = _series(records, "connections", outbound_id, "airflow_residual")
        capacity = _series(records, "system", "system", "shared_airflow_capacity")
        sibling_spans = [
            _span(_series(records, "actuators", sibling_id, "actual_position"))
            for sibling_id in self._zone_ids
            if sibling_id != zone_id
        ]
        actuator_span = _span(actual)
        delivery_fraction_change = abs(delivery[-1] - delivery[0]) / max(capacity[-1], _EPSILON)
        sibling_span = max(sibling_spans, default=0.0)
        return SensorResidualFeatures(
            sensor_slope=_slope(sensor),
            sensor_range=_span(sensor),
            sensor_max_delta=_max_jump(sensor),
            actuator_setpoint_span=_span(setpoint),
            actuator_actual_span=actuator_span,
            actuator_tracking_residual_mean=_mean(tracking),
            actuator_moving_fraction=_fraction_positive(moving),
            outbound_request_change=request[-1] - request[0],
            outbound_delivery_change=delivery[-1] - delivery[0],
            outbound_residual_change=residual[-1] - residual[0],
            sibling_actuator_span_max=sibling_span,
            expected_change_proxy=max(actuator_span, delivery_fraction_change, sibling_span),
        )

    def physical_features(
        self, window: Sequence[TickRecord], zone_id: str
    ) -> PhysicalResidualFeatures:
        """Return physical-airflow evidence from one causal observable window."""
        records = self._validated_window(window)
        self._require_zone(zone_id)
        outbound_id = self._outbound_by_zone[zone_id]
        request = _series(records, "connections", outbound_id, "requested_airflow")
        delivery = _series(records, "connections", outbound_id, "delivered_airflow")
        residual = _series(records, "connections", outbound_id, "airflow_residual")
        capacity = _series(records, "system", "system", "shared_airflow_capacity")
        total_delivery = _series(records, "system", "system", "total_delivered_airflow")
        capacity_scale = _series(records, "system", "system", "capacity_scale")
        moving = _series(records, "actuators", zone_id, "moving")
        normalized = _normalized_residual(residual[-1], request[-1])
        other_normalized = [
            _normalized_residual(
                _series(records, "connections", connection_id, "airflow_residual")[-1],
                _series(records, "connections", connection_id, "requested_airflow")[-1],
            )
            for sibling_id, connection_id in self._outbound_by_zone.items()
            if sibling_id != zone_id
        ]
        max_other = max(other_normalized, default=0.0)
        isolation = 0.0 if normalized <= _EPSILON else 1.0 if max_other <= _EPSILON else normalized / max_other
        headroom = max(0.0, (capacity[-1] - total_delivery[-1]) / capacity[-1])
        contention = 1.0 - capacity_scale[-1]
        moving_fraction = _fraction_positive(moving)
        transient = max(contention, moving_fraction)
        return PhysicalResidualFeatures(
            request=request[-1],
            delivery=delivery[-1],
            residual=residual[-1],
            normalized_residual=normalized,
            residual_slope=_slope(residual),
            residual_max_jump=_max_jump(residual),
            residual_persistence=_fraction_positive(residual),
            isolation_ratio=isolation,
            capacity_headroom=headroom,
            capacity_contention=contention,
            actuator_moving_fraction=moving_fraction,
            transient_proxy=transient,
            settled_proxy=1.0 - transient,
        )

    def _validated_window(self, window: Sequence[TickRecord]) -> tuple[TickRecord, ...]:
        if not isinstance(window, Sequence) or len(window) < 2:
            raise ValueError("residual features require at least two causal tick records")
        records = tuple(window)
        previous_tick = 0
        for record in records:
            if not isinstance(record, TickRecord):
                raise ValueError("residual feature window must contain TickRecord values")
            if record.tick <= previous_tick:
                raise ValueError("residual feature window ticks must be strictly increasing")
            observable_context_v1(record, self._contract)
            previous_tick = record.tick
        return records

    def _require_zone(self, zone_id: str) -> None:
        if zone_id not in self._outbound_by_zone:
            raise ValueError("residual features require a known non-processing zone")


def _series(
    records: Sequence[TickRecord], group: str, entity_id: str, field: str
) -> tuple[float, ...]:
    if group == "system":
        values = tuple(float(record.system[field]) for record in records)
    else:
        values = tuple(float(getattr(record, group)[entity_id][field]) for record in records)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("residual feature input must be finite")
    return values


def _span(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _slope(values: Sequence[float]) -> float:
    return (values[-1] - values[0]) / (len(values) - 1)


def _max_jump(values: Sequence[float]) -> float:
    return max(abs(right - left) for left, right in zip(values, values[1:]))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _fraction_positive(values: Sequence[float]) -> float:
    return sum(value > _EPSILON for value in values) / len(values)


def _normalized_residual(residual: float, request: float) -> float:
    return 0.0 if request <= _EPSILON else residual / request
