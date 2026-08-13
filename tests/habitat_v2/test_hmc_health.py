from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus.habitat_v2.health import (
    HealthReducerError,
    HealthTracker,
    reduce_health,
)
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.instrumentation import (
    instrument_v5_operational_measurement,
)
from aeolus.habitat_v2.physics import (
    command_from_achieved_state,
    initial_state,
    validate_external_command,
)
from aeolus.habitat_v2.scenario import Scenario


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _scenario_without_faults() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    mapping["fault_profiles"] = []
    return Scenario.from_mapping(mapping)


def _contract():
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def test_healthy_reset_measurement_reduces_to_nominal_without_alarm() -> None:
    scenario = _scenario()
    measurement = instrument_v5_operational_measurement(
        scenario,
        initial_state(scenario),
        None,
    )

    result = reduce_health(
        measurement=measurement,
        scenario=scenario,
        contract=_contract(),
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )

    assert result.health_state == "NOMINAL"
    assert result.alarms == ()
    assert result.tracker.completed_step == 0


def test_unavailable_required_slots_fail_closed_to_stable_unknown_alarms() -> None:
    scenario = _scenario()
    state = initial_state(scenario)
    feedback = dict(state.utility.last_operational_feedback or {})
    feedback.pop("fan_speed_fraction")
    feedback["battery_state_of_charge"] = float("nan")
    malformed = replace(
        state,
        utility=replace(state.utility, last_operational_feedback=feedback),
    )
    measurement = instrument_v5_operational_measurement(scenario, malformed, None)

    result = reduce_health(
        measurement=measurement,
        scenario=scenario,
        contract=_contract(),
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )

    assert result.health_state == "UNKNOWN"
    assert tuple(alarm.alarm_id for alarm in result.alarms) == (
        "telemetry_unknown/battery_state_of_charge/critical",
        "telemetry_unknown/fan_speed_fraction/critical",
    )
    assert [alarm.to_mapping() for alarm in result.alarms] == [
        {
            "alarm_id": "telemetry_unknown/battery_state_of_charge/critical",
            "family": "telemetry_unknown",
            "target": "battery_state_of_charge",
            "severity": "CRITICAL",
            "lifecycle": "ACTIVE",
        },
        {
            "alarm_id": "telemetry_unknown/fan_speed_fraction/critical",
            "family": "telemetry_unknown",
            "target": "fan_speed_fraction",
            "severity": "CRITICAL",
            "lifecycle": "ACTIVE",
        },
    ]
    rendered = json.dumps(
        [alarm.to_mapping() for alarm in result.alarms],
        allow_nan=False,
        sort_keys=True,
    )
    assert "NaN" not in rendered
    assert "MISSING" not in rendered
    assert "NON_FINITE" not in rendered


def test_structurally_invalid_measurement_is_rejected_not_converted_to_unknown() -> (
    None
):
    scenario = _scenario()
    with pytest.raises(HealthReducerError, match="exact OperationalMeasurement"):
        reduce_health(
            measurement={},  # type: ignore[arg-type]
            scenario=scenario,
            contract=_contract(),
            previous_tracker=HealthTracker.initial(),
            previous_measurement=None,
            last_final_command=None,
        )


def _state_with_zone_co2_ppm(
    scenario: Scenario,
    *,
    zone_id: str,
    co2_ppm: float,
    step: int,
) -> object:
    base = initial_state(scenario)
    zone = base.zones[zone_id]
    fraction = co2_ppm / 1_000_000.0
    other_moles = zone.o2_mol + zone.water_vapor_mol + zone.inert_mol
    zones = dict(base.zones)
    zones[zone_id] = replace(
        zone,
        co2_mol=fraction * other_moles / (1.0 - fraction),
    )
    return replace(
        base,
        step=step,
        zones=zones,
    )


def _state_with_feedback_value(
    scenario: Scenario,
    *,
    descriptor_id: str,
    value: float,
    step: int,
) -> object:
    base = initial_state(scenario)
    feedback = dict(base.utility.last_operational_feedback or {})
    if "/" in descriptor_id:
        channel_id, resource_id = descriptor_id.split("/", 1)
        resource_values = dict(feedback[channel_id])
        resource_values[resource_id] = value
        feedback[channel_id] = resource_values
    else:
        feedback[descriptor_id] = value
    return replace(
        base,
        step=step,
        utility=replace(base.utility, last_operational_feedback=feedback),
    )


def _command_with_fan_target(scenario: Scenario, target: float):
    command = _initial_hold_command(scenario).to_mapping()
    command["fan_speed_fraction"] = target
    return validate_external_command(scenario, command)


def _command_with_cooling_target(
    scenario: Scenario,
    *,
    zone_id: str,
    target_w: float,
):
    command = _initial_hold_command(scenario).to_mapping()
    command["cooling_removed_w"][zone_id] = target_w
    return validate_external_command(scenario, command)


def _initial_hold_command(scenario: Scenario):
    return command_from_achieved_state(scenario, initial_state(scenario)).command


def _command_with_damper_target(
    scenario: Scenario,
    *,
    damper_id: str,
    target: float,
):
    command = _initial_hold_command(scenario).to_mapping()
    command["damper_position_by_id"][damper_id] = target
    return validate_external_command(scenario, command)


def test_co2_alarm_has_two_row_enter_and_clear_persistence_with_one_cleared_row() -> (
    None
):
    scenario = _scenario_without_faults()
    contract = _contract()
    command = _initial_hold_command(scenario)
    values = (2700.0, 2800.0, 2000.0, 1900.0, 1800.0)
    tracker = HealthTracker.initial()
    previous = None
    lifecycles = []
    health_states = []

    for step, value in enumerate(values):
        measurement = instrument_v5_operational_measurement(
            scenario,
            _state_with_zone_co2_ppm(
                scenario,
                zone_id="common_galley",
                co2_ppm=value,
                step=step,
            ),
            None if previous is None else previous.sensor_memory,
        )
        result = reduce_health(
            measurement=measurement,
            scenario=scenario,
            contract=contract,
            previous_tracker=tracker,
            previous_measurement=previous,
            last_final_command=None if step == 0 else command,
        )
        lifecycles.append(
            tuple(
                alarm.lifecycle
                for alarm in result.alarms
                if alarm.alarm_id == "high_co2/common_galley/warning"
            )
        )
        health_states.append(result.health_state)
        tracker = result.tracker
        previous = measurement

    assert lifecycles == [
        ("RAISED",),
        ("ACTIVE",),
        ("ACTIVE",),
        ("CLEARED",),
        (),
    ]
    assert health_states == [
        "DEGRADED",
        "DEGRADED",
        "DEGRADED",
        "NOMINAL",
        "NOMINAL",
    ]


def test_environmental_alarm_requires_same_row_two_head_corroboration() -> None:
    mapping = json.loads(
        (
            Path(__file__).parents[2]
            / "scenarios"
            / "habitat_v2_actuator_feedback.json"
        ).read_text(encoding="utf-8")
    )
    mapping["fault_profiles"] = [
        {
            "id": "health-test-primary-only-co2-bias",
            "type": "sensor_bias_drift",
            "zone_id": "common_galley",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 2,
            "start_bias": 2000.0,
            "end_bias": 2000.0,
        }
    ]
    scenario = Scenario.from_mapping(mapping)
    contract = _contract()
    reset_measurement = instrument_v5_operational_measurement(
        scenario,
        initial_state(scenario),
        None,
    )
    reset_result = reduce_health(
        measurement=reset_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )
    step_one_state = replace(initial_state(scenario), step=1)
    measurement = instrument_v5_operational_measurement(
        scenario,
        step_one_state,
        reset_measurement.sensor_memory,
    )
    command = _initial_hold_command(scenario)

    result = reduce_health(
        measurement=measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=reset_result.tracker,
        previous_measurement=reset_measurement,
        last_final_command=command,
    )

    assert all(alarm.family != "high_co2" for alarm in result.alarms)
    assert any(alarm.family == "sensor_disagreement" for alarm in result.alarms)


@pytest.mark.parametrize(
    ("descriptor_id", "family"),
    (
        ("battery_state_of_charge", "low_battery_gauge"),
        ("oxygen_store_fraction", "low_oxygen_store_gauge"),
        ("sorbent_remaining_fraction", "low_sorbent_gauge"),
    ),
)
def test_resource_warning_uses_issued_measured_gauge_not_hidden_inventory(
    descriptor_id: str,
    family: str,
) -> None:
    scenario = _scenario_without_faults()
    contract = _contract()
    command = _initial_hold_command(scenario)
    tracker = HealthTracker.initial()
    previous = None
    result = None

    for step in (0, 1):
        state = _state_with_feedback_value(
            scenario,
            descriptor_id=descriptor_id,
            value=0.20,
            step=step,
        )
        # Hidden inventory remains at its healthy initial value. Only the issued
        # measured feedback gauge is low.
        assert state.utility.battery_energy_wh == 12000.0
        assert state.utility.oxygen_store_mol == 500.0
        assert state.utility.co2_sorbent_remaining_mol == 1500.0
        measurement = instrument_v5_operational_measurement(
            scenario,
            state,
            None if previous is None else previous.sensor_memory,
        )
        result = reduce_health(
            measurement=measurement,
            scenario=scenario,
            contract=contract,
            previous_tracker=tracker,
            previous_measurement=previous,
            last_final_command=None if step == 0 else command,
        )
        tracker = result.tracker
        previous = measurement

    assert result is not None
    alarm_id = f"{family}/{descriptor_id}/warning"
    assert tuple(
        (alarm.alarm_id, alarm.lifecycle)
        for alarm in result.alarms
        if alarm.alarm_id == alarm_id
    ) == ((alarm_id, "ACTIVE"),)
    assert result.health_state == "DEGRADED"


def test_fan_tracking_uses_slew_expected_achievement_not_raw_command() -> None:
    scenario = _scenario_without_faults()
    contract = _contract()
    command = _command_with_fan_target(scenario, 1.0)
    reset_state = _state_with_feedback_value(
        scenario,
        descriptor_id="fan_speed_fraction",
        value=0.20,
        step=0,
    )
    reset_measurement = instrument_v5_operational_measurement(
        scenario,
        reset_state,
        None,
    )
    reset_result = reduce_health(
        measurement=reset_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )
    completed_state = _state_with_feedback_value(
        scenario,
        descriptor_id="fan_speed_fraction",
        value=0.719,
        step=1,
    )
    completed_measurement = instrument_v5_operational_measurement(
        scenario,
        completed_state,
        reset_measurement.sensor_memory,
    )

    result = reduce_health(
        measurement=completed_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=reset_result.tracker,
        previous_measurement=reset_measurement,
        last_final_command=command,
    )

    assert tuple(
        (alarm.alarm_id, alarm.lifecycle)
        for alarm in result.alarms
        if alarm.family == "actuator_tracking_failure"
    ) == (
        (
            "actuator_tracking_failure/fan_speed_fraction/primary_supply_fan/warning",
            "RAISED",
        ),
    )
    assert result.health_state == "DEGRADED"


def test_cooling_tracking_uses_slew_expected_achievement_not_raw_command() -> None:
    scenario = _scenario_without_faults()
    contract = _contract()
    zone_id = "laboratory"
    command = _command_with_cooling_target(
        scenario,
        zone_id=zone_id,
        target_w=1000.0,
    )
    reset_state = initial_state(scenario)
    reset_measurement = instrument_v5_operational_measurement(
        scenario,
        reset_state,
        None,
    )
    reset_result = reduce_health(
        measurement=reset_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )
    completed_state = _state_with_feedback_value(
        scenario,
        descriptor_id=f"cooling_delivery_w/{zone_id}",
        value=249.9,
        step=1,
    )
    completed_measurement = instrument_v5_operational_measurement(
        scenario,
        completed_state,
        reset_measurement.sensor_memory,
    )

    result = reduce_health(
        measurement=completed_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=reset_result.tracker,
        previous_measurement=reset_measurement,
        last_final_command=command,
    )

    assert tuple(
        (alarm.alarm_id, alarm.lifecycle)
        for alarm in result.alarms
        if alarm.family == "actuator_tracking_failure"
    ) == (
        (
            "actuator_tracking_failure/cooling_delivery_w/laboratory/warning",
            "RAISED",
        ),
    )
    assert result.health_state == "DEGRADED"


def test_damper_tracking_is_independent_per_declared_damper() -> None:
    scenario = _scenario_without_faults()
    contract = _contract()
    damper_id = "laboratory_supply_damper"
    descriptor_id = f"damper_position_by_id/{damper_id}"
    command = _command_with_damper_target(
        scenario,
        damper_id=damper_id,
        target=1.0,
    )
    reset_state = _state_with_feedback_value(
        scenario,
        descriptor_id=descriptor_id,
        value=0.20,
        step=0,
    )
    reset_measurement = instrument_v5_operational_measurement(
        scenario,
        reset_state,
        None,
    )
    reset_result = reduce_health(
        measurement=reset_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )
    completed_state = _state_with_feedback_value(
        scenario,
        descriptor_id=descriptor_id,
        value=0.899,
        step=1,
    )
    completed_measurement = instrument_v5_operational_measurement(
        scenario,
        completed_state,
        reset_measurement.sensor_memory,
    )

    result = reduce_health(
        measurement=completed_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=reset_result.tracker,
        previous_measurement=reset_measurement,
        last_final_command=command,
    )

    assert tuple(
        (alarm.alarm_id, alarm.lifecycle)
        for alarm in result.alarms
        if alarm.family == "actuator_tracking_failure"
    ) == (
        (
            f"actuator_tracking_failure/{descriptor_id}/warning",
            "RAISED",
        ),
    )
    assert result.health_state == "DEGRADED"


@pytest.mark.parametrize(
    "descriptor_id",
    (
        "oxygen_delivery_mol_s/laboratory",
        "branch_airflow_m3_s/laboratory",
        "scrubber_capture_rate_mol_s",
        "condenser_removal_rate_mol_s",
    ),
)
def test_contract_excluded_channels_never_create_tracking_alarms(
    descriptor_id: str,
) -> None:
    scenario = _scenario_without_faults()
    contract = _contract()
    reset_state = initial_state(scenario)
    reset_measurement = instrument_v5_operational_measurement(
        scenario,
        reset_state,
        None,
    )
    reset_result = reduce_health(
        measurement=reset_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=HealthTracker.initial(),
        previous_measurement=None,
        last_final_command=None,
    )
    completed_state = _state_with_feedback_value(
        scenario,
        descriptor_id=descriptor_id,
        value=1000.0,
        step=1,
    )
    completed_measurement = instrument_v5_operational_measurement(
        scenario,
        completed_state,
        reset_measurement.sensor_memory,
    )

    result = reduce_health(
        measurement=completed_measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=reset_result.tracker,
        previous_measurement=reset_measurement,
        last_final_command=_initial_hold_command(scenario),
    )

    assert all(alarm.family != "actuator_tracking_failure" for alarm in result.alarms)
    assert result.health_state == "NOMINAL"
