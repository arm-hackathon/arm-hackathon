"""Tests for running the standard habitat scenario graph."""

import json

import pytest

from aeolus.__main__ import main
from aeolus.config import load_scenario, parse_scenario
from aeolus.plant import initial_state, step_habitat
from aeolus.measurement import deterministic_measurement_drift
from aeolus.scenario import (
    RunSpec,
    STANDARD_RUN,
    deterministic_measurement_sample,
    run_scenario,
)


def test_run_produces_one_record_per_tick_in_tick_order(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    records = run_scenario(config)

    assert len(records) == STANDARD_RUN.total_ticks
    assert [r.tick for r in records] == list(range(1, STANDARD_RUN.total_ticks + 1))


def test_warmup_settles_state_without_appearing_in_trace(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    warmed = run_scenario(config, run=STANDARD_RUN)[0]
    cold = run_scenario(
        config,
        run=RunSpec(
            total_ticks=1,
            warmup_ticks=0,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
    )[0]

    assert warmed.tick == 1
    assert warmed.zones["cabin_a"]["co2_concentration"] > cold.zones["cabin_a"][
        "co2_concentration"
    ]
    assert (
        warmed.actuators["cabin_a"]["actual_position"]
        > cold.actuators["cabin_a"]["actual_position"]
    )
    warmup_state = initial_state(config)
    for warmup_index in range(STANDARD_RUN.warmup_ticks):
        warmup_state, _ = step_habitat(
            config,
            warmup_state,
            source_tick=warmup_index - STANDARD_RUN.warmup_ticks,
            occupancy_tick=1,
        )
    assert (
        warmed.zones["processing"]["captured_co2"]
        < warmup_state.captured_co2
    )


def test_every_record_covers_every_zone_and_connection(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    zone_ids = {z.id for z in config.zones}
    connection_ids = {c.id for c in config.connections}
    actuator_ids = {z.id for z in config.non_processing_zones()}

    for record in run_scenario(config):
        assert set(record.zones) == zone_ids
        for zone_id in zone_ids:
            assert "co2_mass" in record.zones[zone_id]
            assert "co2_concentration" in record.zones[zone_id]
            assert "sensor_co2_concentration" in record.zones[zone_id]
            assert "source_co2_mass" in record.zones[zone_id]
            assert "occupancy_multiplier" in record.zones[zone_id]
        assert "captured_co2" in record.zones[config.processing_zone().id]
        assert set(record.connections) == connection_ids
        for entry in record.connections.values():
            assert set(entry) == {
                "requested_airflow",
                "delivered_airflow",
                "airflow_residual",
            }
        assert set(record.actuators) == actuator_ids
        for entry in record.actuators.values():
            assert {
                "setpoint",
                "actual_position",
                "tracking_residual",
                "moving",
                "movement_seconds",
                "power",
                "direction",
            } == set(entry)
        assert set(record.system) == {
            "shared_airflow_capacity",
            "total_requested_airflow",
            "total_delivered_airflow",
            "capacity_scale",
        }


def test_same_scenario_twice_produces_identical_records(standard_scenario_path):
    first = run_scenario(load_scenario(standard_scenario_path))
    second = run_scenario(load_scenario(standard_scenario_path))

    assert first == second


def test_seeded_cabin_sources_vary_independently_within_epsilon(
    standard_scenario_path,
):
    records = run_scenario(load_scenario(standard_scenario_path))
    cabin_a = [record.zones["cabin_a"]["source_co2_mass"] for record in records]
    cabin_b = [record.zones["cabin_b"]["source_co2_mass"] for record in records]

    assert all(0.82 <= source <= 1.18 for source in cabin_a[:40])
    assert all(1.42 <= source <= 1.78 for source in cabin_a[40:80])
    assert all(0.52 <= source <= 0.88 for source in cabin_a[80:])
    assert len(set(cabin_a)) > 1
    assert cabin_a != cabin_b


def test_noisy_measurements_are_deterministic_and_preserve_trace_invariants(
    standard_doc, tmp_path,
):
    clean = run_scenario(parse_scenario(standard_doc))
    standard_doc["telemetry"] = {
        "airflow_noise_fraction": 0.015,
        "airflow_bias_fraction": 0.005,
        "airflow_drift_fraction": 0.01,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.0,
        "co2_sensor_bias_fraction": 0.0,
        "co2_sensor_drift_fraction": 0.0,
    }
    config = parse_scenario(standard_doc)

    first_path = tmp_path / "first-noisy.jsonl"
    second_path = tmp_path / "second-noisy.jsonl"
    first = run_scenario(config, trace_path=first_path)
    second = run_scenario(config, trace_path=second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert any(
        record.actuators["cabin_a"]["actual_position"]
        != clean[index].actuators["cabin_a"]["actual_position"]
        for index, record in enumerate(first)
    )
    for record in first:
        assert (
            record.system["total_delivered_airflow"]
            <= record.system["shared_airflow_capacity"]
        )
        for values in record.connections.values():
            assert 0.0 <= values["delivered_airflow"] <= values["requested_airflow"]
            assert values["airflow_residual"] == pytest.approx(
                values["requested_airflow"] - values["delivered_airflow"]
            )


def test_measurement_settings_do_not_change_hidden_plant_state(standard_doc):
    clean = parse_scenario(standard_doc)
    standard_doc["telemetry"] = {
        "airflow_noise_fraction": 0.1,
        "airflow_bias_fraction": 0.1,
        "airflow_drift_fraction": 0.1,
        "actuator_position_noise_fraction": 0.1,
        "co2_sensor_noise_fraction": 0.0,
        "co2_sensor_bias_fraction": 0.0,
        "co2_sensor_drift_fraction": 0.0,
    }
    noisy = parse_scenario(standard_doc)
    clean_state = initial_state(clean)
    noisy_state = initial_state(noisy)

    for _ in range(20):
        clean_state, clean_airflows = step_habitat(clean, clean_state)
        noisy_state, noisy_airflows = step_habitat(noisy, noisy_state)

    assert noisy_state == clean_state
    assert noisy_airflows == clean_airflows


def test_co2_measurement_noise_drives_controller_without_directly_changing_mass(
    standard_doc,
):
    clean = parse_scenario(standard_doc)
    standard_doc["telemetry"]["co2_sensor_bias_fraction"] = 0.5
    noisy = parse_scenario(standard_doc)
    clean_state, _ = step_habitat(clean, initial_state(clean))
    noisy_state, _ = step_habitat(noisy, initial_state(noisy))

    assert noisy_state.source_co2_mass == clean_state.source_co2_mass
    assert noisy_state.sensor_co2_concentration != clean_state.sensor_co2_concentration
    assert any(
        noisy_state.actuators[zone].setpoint != clean_state.actuators[zone].setpoint
        for zone in noisy_state.actuators
    )


def test_measurement_samples_are_bounded_and_independent_by_key(standard_doc):
    first_config = parse_scenario(standard_doc)
    standard_doc["simulation"]["random_seed"] += 1
    second_config = parse_scenario(standard_doc)
    samples = {
        deterministic_measurement_sample(config, entity, channel, tick)
        for config in (first_config, second_config)
        for entity in ("cabin_a", "cabin_b")
        for channel in ("actuator_position", "airflow_noise", "airflow_bias")
        for tick in (0, 1, 2)
    }

    assert len(samples) == 36
    assert all(-1.0 <= sample < 1.0 for sample in samples)


def test_measurement_drift_is_bounded_continuous_and_anchor_exact(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    values = [
        deterministic_measurement_drift(config, "cabin_a", "co2_sensor", tick)
        for tick in range(61)
    ]

    assert all(-1.0 <= value < 1.0 for value in values)
    assert max(abs(later - earlier) for earlier, later in zip(values, values[1:])) < 0.1
    for tick in (0, 20, 40, 60):
        assert values[tick] == deterministic_measurement_sample(
            config, "cabin_a", "co2_sensor:drift-anchor", tick
        )


def test_frozen_sensor_holds_latent_value_while_readout_noise_continues(standard_doc):
    standard_doc["telemetry"].update(
        co2_sensor_noise_fraction=0.05,
        co2_sensor_bias_fraction=0.02,
        co2_sensor_drift_fraction=0.03,
    )
    config = parse_scenario(standard_doc)
    state = initial_state(config)
    readings = []
    truths = []
    latent_values = []
    for _ in range(8):
        state, _ = step_habitat(config, state, frozen_zones={"lab"})
        readings.append(state.sensor_co2_concentration["lab"])
        truths.append(state.zone_co2_mass["lab"])
        latent_values.append(state.frozen_sensor_readings["lab"])

    assert len(set(readings)) > 1
    assert len(set(latent_values)) == 1
    assert len(set(truths)) > 1


def test_trace_proves_co2_sensor_controls_actuator(standard_scenario_path):
    records = run_scenario(load_scenario(standard_scenario_path))

    early = records[0]
    strongest = max(
        records,
        key=lambda record: record.actuators["cabin_a"]["setpoint"],
    )
    path = "cabin_a_to_processing"
    assert (
        strongest.zones["cabin_a"]["sensor_co2_concentration"]
        > early.zones["cabin_a"]["sensor_co2_concentration"]
    )
    assert (
        strongest.actuators["cabin_a"]["setpoint"]
        > early.actuators["cabin_a"]["setpoint"]
    )
    assert (
        strongest.actuators["cabin_a"]["actual_position"]
        > early.actuators["cabin_a"]["actual_position"]
    )
    assert (
        strongest.connections[path]["delivered_airflow"]
        > early.connections[path]["delivered_airflow"]
    )


def test_same_scenario_file_twice_produces_byte_identical_traces(
    standard_scenario_path, tmp_path
):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    run_scenario(load_scenario(standard_scenario_path), trace_path=first_path)
    run_scenario(load_scenario(standard_scenario_path), trace_path=second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_full_run_conserves_generated_co2_mass(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    records = run_scenario(config)

    generated = sum(
        zone["source_co2_mass"]
        for record in records
        for zone in record.zones.values()
    )
    final_airborne = sum(
        zone["co2_mass"] for zone in records[-1].zones.values()
    )
    captured = records[-1].zones["processing"]["captured_co2"]
    # The first sensor reading is taken after the first measured source is
    # added, so subtracting that source recovers the unrecorded warm-up mass.
    initial_airborne = sum(
        records[0].zones[zone.id]["sensor_co2_concentration"] * zone.air_volume
        - records[0].zones[zone.id]["source_co2_mass"]
        for zone in config.zones
    )

    assert final_airborne + captured == pytest.approx(
        initial_airborne + generated
    )


def test_healthy_standard_habitat_keeps_crew_cabins_below_ceiling_after_warmup(
    standard_scenario_path,
):
    config = load_scenario(standard_scenario_path)
    records = run_scenario(config)

    late = [r for r in records if r.tick > STANDARD_RUN.warmup_ticks]
    assert late, "scenario should leave ticks to check after the warm-up window"
    crew_cabins = [z for z in config.non_processing_zones() if z.preset == "crew_cabin"]
    assert {z.id for z in crew_cabins} == {"cabin_a", "cabin_b"}
    for zone in crew_cabins:
        peak = max(r.zones[zone.id]["co2_concentration"] for r in late)
        assert peak < STANDARD_RUN.crew_cabin_co2_concentration_ceiling


def test_run_writes_jsonl_trace_with_one_row_per_tick(standard_scenario_path, tmp_path):
    config = load_scenario(standard_scenario_path)
    path = tmp_path / "standard.jsonl"

    records = run_scenario(config, trace_path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == STANDARD_RUN.total_ticks == len(records)
    row = json.loads(lines[0])
    assert set(row) == {"tick", "zones", "connections", "actuators", "system"}
    assert set(row["zones"]) == {z.id for z in config.zones}
    assert set(row["connections"]) == {c.id for c in config.connections}
    assert set(row["actuators"]) == {
        z.id for z in config.non_processing_zones()
    }


def test_main_entrypoint_runs_explicit_scenario_file_and_writes_trace(
    standard_scenario_path, tmp_path, capsys
):
    path = tmp_path / "out.jsonl"

    exit_code = main([str(standard_scenario_path), str(path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert str(standard_scenario_path) in out
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == STANDARD_RUN.total_ticks
    row = json.loads(lines[0])
    assert set(row["zones"]) == {"cabin_a", "cabin_b", "lab", "processing"}


def test_main_entrypoint_rejects_invalid_scenario_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"version": 99, "zones": [], "connections": []}', encoding="utf-8"
    )

    assert main([str(bad), str(tmp_path / "out.jsonl")]) == 2
    assert "version" in capsys.readouterr().err


def test_main_entrypoint_rejects_missing_scenario_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json"), str(tmp_path / "out.jsonl")]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_entrypoint_rejects_wrong_argument_count(capsys):
    assert main([]) == 2
    assert "Usage" in capsys.readouterr().err
