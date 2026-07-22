"""Tests for sensor-driven ventilation control."""

import pytest

from icarus.control import (
    CO2ControlSettings,
    CO2SensorReading,
    ProportionalCO2Controller,
)


@pytest.fixture
def controller() -> ProportionalCO2Controller:
    return ProportionalCO2Controller(
        CO2ControlSettings(
            lower_threshold=0.0,
            upper_threshold=20.0,
            minimum_command=0.0,
            maximum_command=1.0,
        )
    )


def test_command_is_proportional_to_co2_between_thresholds(controller):
    command = controller.command_for(CO2SensorReading("cabin", 10.0))

    assert command == pytest.approx(0.5)


def test_command_is_bounded_at_both_ends(controller):
    assert controller.command_for(CO2SensorReading("cabin", 0.0)) == 0.0
    assert controller.command_for(CO2SensorReading("cabin", 30.0)) == 1.0


def test_minimum_and_maximum_commands_are_respected():
    controller = ProportionalCO2Controller(
        CO2ControlSettings(
            lower_threshold=5.0,
            upper_threshold=15.0,
            minimum_command=0.2,
            maximum_command=0.8,
        )
    )

    assert controller.command_for(CO2SensorReading("cabin", 0.0)) == 0.2
    assert controller.command_for(CO2SensorReading("cabin", 20.0)) == 0.8


@pytest.mark.parametrize(
    "settings",
    [
        (-1.0, 20.0, 0.0, 1.0),
        (20.0, 20.0, 0.0, 1.0),
        (0.0, 20.0, -0.1, 1.0),
        (0.0, 20.0, 0.0, 1.1),
        (0.0, 20.0, 0.8, 0.2),
    ],
)
def test_invalid_control_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        CO2ControlSettings(*settings)
