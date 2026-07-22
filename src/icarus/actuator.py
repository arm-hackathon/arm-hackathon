"""Rate-limited ventilation actuator dynamics for the ICARUS plant."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ActuatorSettings:
    """Shared characteristics of the simulated ventilation actuators."""

    full_stroke_seconds: float
    moving_power: float
    holding_power: float

    def __post_init__(self) -> None:
        values = (self.full_stroke_seconds, self.moving_power, self.holding_power)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("actuator settings must be finite numbers")
        if self.full_stroke_seconds <= 0.0:
            raise ValueError("actuator full stroke time must be positive")
        if self.moving_power < 0.0 or self.holding_power < 0.0:
            raise ValueError("actuator power values must not be negative")


@dataclass(frozen=True)
class ActuatorState:
    """Commanded and measured state of one normalised actuator."""

    setpoint: float = 0.0
    actual_position: float = 0.0
    movement_seconds: float = 0.0
    power: float = 0.0
    direction: int = 0

    @property
    def tracking_residual(self) -> float:
        return self.setpoint - self.actual_position

    @property
    def moving(self) -> bool:
        return self.direction != 0


class RateLimitedActuator:
    """Move towards each setpoint at a fixed maximum stroke rate."""

    def __init__(self, settings: ActuatorSettings) -> None:
        self.settings = settings

    def step(
        self,
        state: ActuatorState,
        setpoint: float,
        *,
        seconds: float = 1.0,
    ) -> ActuatorState:
        """Advance one actuator while respecting its full-stroke time."""
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("actuator timestep must be finite and positive")
        target = _bounded(setpoint)
        maximum_change = seconds / self.settings.full_stroke_seconds
        error = target - state.actual_position
        change = max(-maximum_change, min(maximum_change, error))
        position = _bounded(state.actual_position + change)
        direction = 1 if change > 0.0 else -1 if change < 0.0 else 0
        movement_seconds = state.movement_seconds + seconds if direction else 0.0
        power = (
            self.settings.moving_power
            if direction
            else self.settings.holding_power
        )
        return ActuatorState(
            setpoint=target,
            actual_position=position,
            movement_seconds=movement_seconds,
            power=power,
            direction=direction,
        )
