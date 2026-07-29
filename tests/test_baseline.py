"""Rule-baseline detector contracts over model-feature windows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icarus.baseline import RuleBaseline
from icarus.config import load_scenario, parse_scenario

LOOPS = ("cabin_a", "cabin_b", "lab")
REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_SCENARIO = REPO_ROOT / "scenarios" / "standard_habitat.json"


def _baseline() -> RuleBaseline:
    return RuleBaseline(load_scenario(STANDARD_SCENARIO))


def _tick(
    *,
    sensor: float,
    loop_ratios: dict[str, float] | None = None,
    requested: float = 10.0,
) -> dict:
    """One feature tick with three hub loops (both legs each), all healthy by default."""
    ratios = {zone: 0.0 for zone in LOOPS}
    if loop_ratios:
        ratios.update(loop_ratios)
    connections = {}
    for zone in LOOPS:
        residual = ratios[zone] * requested
        entry = {
            "requested_airflow": requested,
            "delivered_airflow": requested - residual,
            "airflow_residual": residual,
        }
        connections[f"{zone}_to_processing"] = dict(entry)
        connections[f"processing_to_{zone}"] = dict(entry)
    zones = {
        zone: {"sensor_co2_concentration": sensor}
        for zone in LOOPS
    }
    zones["processing"] = {"sensor_co2_concentration": 0.0}
    actuators = {
        zone: {
            "setpoint": 1.0,
            "actual_position": 1.0,
            "tracking_residual": 0.0,
            "power": 0.05,
        }
        for zone in LOOPS
    }
    return {
        "zones": zones,
        "actuators": actuators,
        "connections": connections,
    }


def _window(ticks: list[dict]) -> list[dict]:
    return ticks


def test_healthy_window_is_nominal():
    window = _window([_tick(sensor=0.1 + 0.001 * i) for i in range(10)])
    assert _baseline().label_window(window) == "nominal"


def test_frozen_sensor_is_detected_from_zero_variance():
    window = _window([_tick(sensor=0.11) for _ in range(10)])
    assert _baseline().label_window(window) == "frozen_sensor"


def test_sensor_constant_for_part_of_window_is_not_frozen():
    ticks = [_tick(sensor=0.1 + 0.01 * i) for i in range(6)]
    ticks += [_tick(sensor=0.16) for _ in range(4)]
    assert _baseline().label_window(_window(ticks)) == "nominal"


def test_gradual_residual_ramp_is_degradation():
    window = [
        _tick(sensor=0.2 + 0.001 * i, loop_ratios={"cabin_a": 0.04 * i})
        for i in range(10)
    ]
    assert _baseline().label_window(window) == "gradual_primary_fan_degradation"


def test_sudden_residual_step_is_blocked_path():
    window = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.95 if i >= 6 else 0.0},
        )
        for i in range(10)
    ]
    assert _baseline().label_window(window) == "blocked_path"


def test_residual_blip_shorter_than_persistence_is_nominal():
    window = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.1 if i >= 8 else 0.0},
        )
        for i in range(10)
    ]
    assert _baseline().label_window(window) == "nominal"


def test_zero_requested_flow_does_not_crash_or_false_positive():
    window = [_tick(sensor=0.05 + 0.001 * i, requested=0.0) for i in range(10)]
    assert _baseline().label_window(window) == "nominal"


def test_capacity_contention_shared_by_all_loops_is_not_a_fault():
    # Shared-capacity allocation cuts every loop proportionally: every loop
    # shows the same residual ratio, so no loop is isolated.
    window = [
        _tick(sensor=0.2 + 0.001 * i, loop_ratios={zone: 0.14 for zone in LOOPS})
        for i in range(10)
    ]
    assert _baseline().label_window(window) == "nominal"


def test_one_isolated_loop_among_contention_is_a_fault():
    window = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.3, "cabin_b": 0.14, "lab": 0.14},
        )
        for i in range(10)
    ]
    assert _baseline().label_window(window) == "gradual_primary_fan_degradation"


def test_jump_memory_keeps_blockage_after_onset_leaves_the_window():
    detector = _baseline()
    onset = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.95 if i >= 5 else 0.0},
        )
        for i in range(10)
    ]
    steady = [
        _tick(sensor=0.4 + 0.001 * i, loop_ratios={"cabin_a": 0.95}) for i in range(10)
    ]

    assert detector.label_window(onset) == "blocked_path"
    assert detector.label_window(steady) == "blocked_path"

    detector.reset()
    assert detector.label_window(steady) == "gradual_primary_fan_degradation"


def test_graph_derived_pairing_survives_renamed_connection_ids():
    standard = load_scenario(STANDARD_SCENARIO)
    renamed_scenario = json.loads(STANDARD_SCENARIO.read_text(encoding="utf-8"))
    renamed_ids = {
        "cabin_a_to_processing": "outbound_meter_alpha",
        "processing_to_cabin_a": "return_leg_alpha",
    }
    for connection in renamed_scenario["connections"]:
        connection["id"] = renamed_ids.get(connection["id"], connection["id"])
    renamed = parse_scenario(renamed_scenario)

    standard_window = [
        _tick(
            sensor=0.2 + 0.001 * tick,
            loop_ratios={"cabin_a": 0.3, "cabin_b": 0.14, "lab": 0.14},
        )
        for tick in range(10)
    ]
    renamed_window = []
    for tick in standard_window:
        connections = dict(tick["connections"])
        for old_id, new_id in renamed_ids.items():
            connections[new_id] = connections.pop(old_id)
        renamed_window.append({**tick, "connections": connections})

    assert RuleBaseline(standard).label_window(standard_window) == "gradual_primary_fan_degradation"
    assert RuleBaseline(renamed).label_window(renamed_window) == "gradual_primary_fan_degradation"


@pytest.mark.parametrize("mutation", ("missing", "unexpected"))
def test_every_tick_must_match_configured_connection_topology(mutation: str):
    window = [_tick(sensor=0.1 + 0.001 * tick) for tick in range(10)]
    later_connections = window[5]["connections"]
    if mutation == "missing":
        later_connections.pop("lab_to_processing")
    else:
        later_connections["late_extra_connection"] = {
            "requested_airflow": 10.0,
            "delivered_airflow": 10.0,
            "airflow_residual": 0.0,
        }

    with pytest.raises(ValueError, match=rf"tick 6.*{mutation}"):
        _baseline().label_window(window)


def test_unexpected_mixed_type_connection_ids_raise_controlled_value_error():
    window = [_tick(sensor=0.1 + 0.001 * tick) for tick in range(10)]
    window[5]["connections"].update(
        {
            "late_extra_connection": {
                "requested_airflow": 10.0,
                "delivered_airflow": 10.0,
                "airflow_residual": 0.0,
            },
            7: {
                "requested_airflow": 10.0,
                "delivered_airflow": 10.0,
                "airflow_residual": 0.0,
            },
        }
    )

    with pytest.raises(ValueError, match=r"tick 6.*unexpected"):
        _baseline().label_window(window)


@pytest.mark.parametrize(
    ("mutation", "group", "reason"),
    (
        ("rogue_constant_sensor_loop", "zones", "unexpected"),
        ("missing_zone", "zones", "missing"),
        ("extra_actuator", "actuators", "unexpected"),
        ("missing_actuator", "actuators", "missing"),
    ),
)
def test_every_tick_must_match_configured_zone_and_actuator_topology(
    mutation: str,
    group: str,
    reason: str,
):
    window = [_tick(sensor=0.1 + 0.001 * tick) for tick in range(10)]
    if mutation == "rogue_constant_sensor_loop":
        for tick in window:
            tick["zones"]["rogue"] = {"sensor_co2_concentration": 0.5}
            tick["actuators"]["rogue"] = dict(tick["actuators"]["cabin_a"])
    elif mutation == "missing_zone":
        window[5]["zones"].pop("lab")
    elif mutation == "extra_actuator":
        window[5]["actuators"]["rogue"] = dict(window[5]["actuators"]["cabin_a"])
    else:
        window[5]["actuators"].pop("lab")

    with pytest.raises(ValueError, match=rf"tick [16].*{group}.*{reason}"):
        _baseline().label_window(window)
