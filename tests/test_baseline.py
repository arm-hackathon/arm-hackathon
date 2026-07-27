"""Rule-baseline detector contracts over model-feature windows."""

from __future__ import annotations

from icarus.baseline import RuleBaseline

LOOPS = ("cabin_a", "cabin_b", "lab")


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
    return {
        "zones": {
            "cabin_a": {"sensor_co2_concentration": sensor},
            "processing": {"sensor_co2_concentration": 0.0},
        },
        "actuators": {
            "cabin_a": {
                "setpoint": 1.0,
                "actual_position": 1.0,
                "tracking_residual": 0.0,
                "power": 0.05,
            }
        },
        "connections": connections,
    }


def _window(ticks: list[dict]) -> list[dict]:
    return ticks


def test_healthy_window_is_nominal():
    window = _window([_tick(sensor=0.1 + 0.001 * i) for i in range(10)])
    assert RuleBaseline().label_window(window) == "nominal"


def test_frozen_sensor_is_detected_from_zero_variance():
    window = _window([_tick(sensor=0.11) for _ in range(10)])
    assert RuleBaseline().label_window(window) == "frozen_sensor"


def test_sensor_constant_for_part_of_window_is_not_frozen():
    ticks = [_tick(sensor=0.1 + 0.01 * i) for i in range(6)]
    ticks += [_tick(sensor=0.16) for _ in range(4)]
    assert RuleBaseline().label_window(_window(ticks)) == "nominal"


def test_gradual_residual_ramp_is_degradation():
    window = [
        _tick(sensor=0.2 + 0.001 * i, loop_ratios={"cabin_a": 0.04 * i})
        for i in range(10)
    ]
    assert RuleBaseline().label_window(window) == "gradual_primary_fan_degradation"


def test_sudden_residual_step_is_blocked_path():
    window = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.95 if i >= 6 else 0.0},
        )
        for i in range(10)
    ]
    assert RuleBaseline().label_window(window) == "blocked_path"


def test_residual_blip_shorter_than_persistence_is_nominal():
    window = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.1 if i >= 8 else 0.0},
        )
        for i in range(10)
    ]
    assert RuleBaseline().label_window(window) == "nominal"


def test_zero_requested_flow_does_not_crash_or_false_positive():
    window = [_tick(sensor=0.05 + 0.001 * i, requested=0.0) for i in range(10)]
    assert RuleBaseline().label_window(window) == "nominal"


def test_capacity_contention_shared_by_all_loops_is_not_a_fault():
    # Shared-capacity allocation cuts every loop proportionally: every loop
    # shows the same residual ratio, so no loop is isolated.
    window = [
        _tick(sensor=0.2 + 0.001 * i, loop_ratios={zone: 0.14 for zone in LOOPS})
        for i in range(10)
    ]
    assert RuleBaseline().label_window(window) == "nominal"


def test_one_isolated_loop_among_contention_is_a_fault():
    window = [
        _tick(
            sensor=0.2 + 0.001 * i,
            loop_ratios={"cabin_a": 0.3, "cabin_b": 0.14, "lab": 0.14},
        )
        for i in range(10)
    ]
    assert RuleBaseline().label_window(window) == "gradual_primary_fan_degradation"


def test_jump_memory_keeps_blockage_after_onset_leaves_the_window():
    detector = RuleBaseline()
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
