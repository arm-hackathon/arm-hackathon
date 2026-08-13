from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aeolus.habitat_v2.instrumentation import (
    ChannelSample,
    OperationalInstrumentationError,
    instrument_v5_operational_measurement,
)
from aeolus.habitat_v2.physics import advance_one_step, initial_state
from aeolus.habitat_v2.runner import run_scenario
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.telemetry import (
    ENVIRONMENTAL_CHANNELS,
    OPERATIONAL_FEEDBACK_CHANNELS,
    derive_observable_topology,
)


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _sample_mapping(samples: tuple[ChannelSample, ...]) -> dict[str, ChannelSample]:
    return {sample.descriptor_id: sample for sample in samples}


def _available_nested(
    samples: tuple[ChannelSample, ...],
) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = {}
    for sample in samples:
        assert sample.availability == "AVAILABLE"
        assert sample.value is not None
        resource_id, channel_id = sample.descriptor_id.split("/", 1)
        nested.setdefault(resource_id, {})[channel_id] = sample.value
    return nested


def _available_feedback(samples: tuple[ChannelSample, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for sample in samples:
        assert sample.availability == "AVAILABLE"
        assert sample.value is not None
        if "/" not in sample.descriptor_id:
            output[sample.descriptor_id] = sample.value
            continue
        channel_id, resource_id = sample.descriptor_id.split("/", 1)
        output.setdefault(channel_id, {})[resource_id] = sample.value
    return output


def _expected_environmental_ids(scenario: Scenario) -> tuple[str, ...]:
    topology = derive_observable_topology(scenario)
    return tuple(
        f"{zone_id}/{channel_id}"
        for zone_id in topology.zone_ids
        for channel_id in ENVIRONMENTAL_CHANNELS
    )


def _expected_feedback_ids(scenario: Scenario) -> tuple[str, ...]:
    topology = derive_observable_topology(scenario)
    damper_ids = tuple(damper_id for _, damper_id in topology.branch_pairs)
    result: list[str] = []
    for channel_id in OPERATIONAL_FEEDBACK_CHANNELS:
        if channel_id == "damper_position_by_id":
            result.extend(f"{channel_id}/{damper_id}" for damper_id in damper_ids)
        elif channel_id in {
            "branch_airflow_m3_s",
            "branch_differential_pressure_pa",
            "cooling_delivery_w",
            "oxygen_delivery_mol_s",
        }:
            result.extend(f"{channel_id}/{zone_id}" for zone_id in topology.zone_ids)
        else:
            result.append(channel_id)
    return tuple(result)


def test_v5_operational_adapter_matches_every_approved_trace_row() -> None:
    scenario = _scenario()
    run = run_scenario(scenario)
    state = initial_state(scenario)
    memory = None
    measurements = []

    while True:
        measurement = instrument_v5_operational_measurement(scenario, state, memory)
        measurements.append(measurement)
        memory = measurement.sensor_memory
        if state.step == int(scenario.data["steps"]):
            break
        state = advance_one_step(scenario, state).state

    assert len(measurements) == len(run.rows)
    expected_environmental_ids = _expected_environmental_ids(scenario)
    expected_feedback_ids = _expected_feedback_ids(scenario)
    dt_seconds = float(scenario.data["dt_seconds"])

    for sequence, (measurement, trace_row) in enumerate(
        zip(measurements, run.rows, strict=True)
    ):
        assert tuple(sample.descriptor_id for sample in measurement.primary) == (
            expected_environmental_ids
        )
        assert tuple(sample.descriptor_id for sample in measurement.secondary) == (
            expected_environmental_ids
        )
        assert (
            tuple(
                sample.descriptor_id for sample in measurement.primary_minus_secondary
            )
            == expected_environmental_ids
        )
        assert (
            tuple(sample.descriptor_id for sample in measurement.operational_feedback)
            == expected_feedback_ids
        )
        assert measurement.sensor_memory.primary is measurement.primary
        assert measurement.sensor_memory.secondary is measurement.secondary
        assert (
            measurement.sensor_memory.operational_feedback
            is measurement.operational_feedback
        )
        assert measurement.completed_step == sequence
        assert measurement.completed_time_s == sequence * dt_seconds
        assert _available_nested(measurement.primary) == trace_row["telemetry"]
        assert _available_nested(measurement.secondary) == {
            zone_id: trace_row["sensor_disagreement"][zone_id]["secondary"]
            for zone_id in sorted(trace_row["sensor_disagreement"])
        }
        assert _available_nested(measurement.primary_minus_secondary) == {
            zone_id: trace_row["sensor_disagreement"][zone_id][
                "primary_minus_secondary"
            ]
            for zone_id in sorted(trace_row["sensor_disagreement"])
        }
        assert (
            _available_feedback(measurement.operational_feedback)
            == trace_row["operational_feedback"]
        )


def test_v5_operational_adapter_normalises_known_missing_and_non_finite_feedback() -> (
    None
):
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
    samples = _sample_mapping(measurement.operational_feedback)

    assert samples["fan_speed_fraction"].availability == "UNAVAILABLE"
    assert samples["fan_speed_fraction"].unavailable_reason == "MISSING"
    assert samples["fan_speed_fraction"].value is None
    assert samples["battery_state_of_charge"].availability == "UNAVAILABLE"
    assert samples["battery_state_of_charge"].unavailable_reason == "NON_FINITE"
    assert samples["battery_state_of_charge"].value is None


def test_v5_operational_adapter_rejects_unknown_feedback_identity() -> None:
    scenario = _scenario()
    state = initial_state(scenario)
    feedback = dict(state.utility.last_operational_feedback or {})
    feedback["hidden_truth_probe"] = 1.0
    malformed = replace(
        state,
        utility=replace(state.utility, last_operational_feedback=feedback),
    )

    with pytest.raises(OperationalInstrumentationError, match="unknown feedback"):
        instrument_v5_operational_measurement(scenario, malformed, None)


def test_issued_channel_samples_are_final() -> None:
    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedChannelSample(ChannelSample):
            pass
