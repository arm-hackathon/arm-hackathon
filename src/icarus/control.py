"""Deterministic CO2-demand control for habitat ventilation actuators."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class CO2ControlSettings:
    """Bounds for converting a CO2 sensor reading into an actuator command.

    Readings at or below ``lower_threshold`` request ``minimum_command``;
    readings at or above ``upper_threshold`` request ``maximum_command``.
    Values between the thresholds are mapped linearly. Commands are normalised
    to the inclusive range 0.0..1.0.
    """

    lower_threshold: float
    upper_threshold: float
    minimum_command: float
    maximum_command: float

    def __post_init__(self) -> None:
        values = (
            self.lower_threshold,
            self.upper_threshold,
            self.minimum_command,
            self.maximum_command,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("CO2 control settings must be finite")
        if self.lower_threshold < 0.0:
            raise ValueError("CO2 lower threshold must not be negative")
        if self.upper_threshold <= self.lower_threshold:
            raise ValueError("CO2 upper threshold must exceed the lower threshold")
        if not 0.0 <= self.minimum_command <= 1.0:
            raise ValueError("minimum actuator command must be in 0.0..1.0")
        if not 0.0 <= self.maximum_command <= 1.0:
            raise ValueError("maximum actuator command must be in 0.0..1.0")
        if self.maximum_command < self.minimum_command:
            raise ValueError("maximum actuator command must not be below the minimum")


@dataclass(frozen=True)
class CO2SensorReading:
    """One idealised CO2 sensor observation in abstract simulation units."""

    zone_id: str
    value: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or self.value < 0.0
        ):
            raise ValueError("CO2 sensor reading must be finite and non-negative")


class ProportionalCO2Controller:
    """Map each zone's CO2 reading to a bounded ventilation command."""

    def __init__(self, settings: CO2ControlSettings) -> None:
        self.settings = settings

    def command_for(self, reading: CO2SensorReading) -> float:
        """Return a normalised actuator command for one sensor reading."""
        span = self.settings.upper_threshold - self.settings.lower_threshold
        demand = _clamp(
            (reading.value - self.settings.lower_threshold) / span,
            0.0,
            1.0,
        )
        command_span = (
            self.settings.maximum_command - self.settings.minimum_command
        )
        return self.settings.minimum_command + demand * command_span
