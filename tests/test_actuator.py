"""Tests for gradual ventilation actuator movement."""

import pytest

from icarus.actuator import ActuatorSettings, ActuatorState, RateLimitedActuator


@pytest.fixture
def actuator() -> RateLimitedActuator:
    return RateLimitedActuator(
        ActuatorSettings(
            full_stroke_seconds=20.0,
            moving_power=1.0,
            holding_power=0.05,
        )
    )


def test_actuator_approaches_setpoint_at_declared_stroke_rate(actuator):
    first = actuator.step(ActuatorState(), 1.0)
    second = actuator.step(first, 1.0)

    assert first.setpoint == 1.0
    assert first.actual_position == pytest.approx(0.05)
    assert second.actual_position == pytest.approx(0.10)
    assert first.tracking_residual == pytest.approx(0.95)
    assert first.direction == 1
    assert first.moving is True


def test_actuator_does_not_overshoot_setpoint(actuator):
    state = actuator.step(ActuatorState(actual_position=0.48), 0.5)

    assert state.actual_position == pytest.approx(0.5)
    assert state.tracking_residual == pytest.approx(0.0)


def test_actuator_can_close_gradually(actuator):
    state = actuator.step(ActuatorState(actual_position=1.0), 0.0)

    assert state.actual_position == pytest.approx(0.95)
    assert state.direction == -1


def test_movement_time_accumulates_then_resets_when_settled(actuator):
    moving = actuator.step(ActuatorState(), 1.0)
    moving = actuator.step(moving, 1.0)
    settled = actuator.step(ActuatorState(actual_position=0.5), 0.5)

    assert moving.movement_seconds == 2.0
    assert moving.power == 1.0
    assert settled.movement_seconds == 0.0
    assert settled.power == 0.05


def test_setpoint_is_bounded_to_normalised_range(actuator):
    high = actuator.step(ActuatorState(), 2.0)
    low = actuator.step(ActuatorState(actual_position=1.0), -1.0)

    assert high.setpoint == 1.0
    assert low.setpoint == 0.0


@pytest.mark.parametrize(
    "settings",
    [
        (0.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (20.0, -1.0, 0.0),
        (20.0, 1.0, -1.0),
    ],
)
def test_invalid_actuator_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ActuatorSettings(*settings)
