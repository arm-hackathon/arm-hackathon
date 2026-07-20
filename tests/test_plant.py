"""Tests for the two-room ventilation plant model."""

import pytest

from icarus.plant import (
    PlantConfig,
    PlantState,
    main_fan_airflow,
    step_plant,
    with_main_fan_effectiveness,
)


def test_new_nominal_plant_has_expected_starting_state():
    config = PlantConfig()
    state = PlantState()

    assert state.tick == 0
    assert state.room_a_co2 == 0.0
    assert state.room_b_co2 == 0.0
    assert state.main_fan_effectiveness == 1.0
    assert main_fan_airflow(config, state) == config.fan_base_airflow


def test_weak_main_fan_produces_lower_airflow_than_healthy_fan():
    config = PlantConfig()
    healthy = PlantState()
    weak = with_main_fan_effectiveness(healthy, 0.35)

    assert main_fan_airflow(config, weak) < main_fan_airflow(config, healthy)


def test_one_tick_conserves_co2_between_the_rooms():
    config = PlantConfig()
    state = step_plant(config, PlantState())

    assert state.tick == 1
    assert 0.0 < state.room_a_co2 < config.crew_co2_per_second
    assert state.room_b_co2 > 0.0
    assert state.room_a_co2 + state.room_b_co2 == pytest.approx(config.crew_co2_per_second)
