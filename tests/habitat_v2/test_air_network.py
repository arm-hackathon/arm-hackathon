from __future__ import annotations

from dataclasses import replace

import pytest

from aeolus.habitat_v2.air_network import (
    AirNetworkSpec,
    AirNetworkValidationError,
    BranchSpec,
    FanSpec,
    solve_air_network,
)


def one_branch_spec() -> AirNetworkSpec:
    return AirNetworkSpec(
        fan=FanSpec(
            component_id="supply_fan",
            rated_free_delivery_m3_s=0.40,
            rated_shutoff_pressure_pa=500.0,
            total_efficiency=0.70,
        ),
        shared_resistance_pa_s2_m6=1_000.0,
        air_density_kg_m3=0.85,
        branches=(
            BranchSpec(
                zone_id="laboratory",
                damper_id="laboratory_supply_damper",
                open_supply_resistance_pa_s2_m6=2_000.0,
                return_resistance_pa_s2_m6=1_000.0,
                damper_leak_fraction=0.05,
            ),
        ),
    )


def test_zero_speed_has_no_flow_pressure_or_power() -> None:
    result = solve_air_network(
        one_branch_spec(),
        fan_speed_fraction=0.0,
        damper_position_by_id={"laboratory_supply_damper": 1.0},
    )

    assert result.fan_pressure_rise_pa == 0.0
    assert result.total_flow_m3_s == 0.0
    assert result.fan_air_power_w == 0.0
    assert result.fan_electrical_power_w == 0.0
    assert result.zone_flow_m3_s == {"laboratory": 0.0}
    assert result.zone_mass_flow_kg_s == {"laboratory": 0.0}


def test_one_open_branch_solves_fan_and_system_operating_point() -> None:
    result = solve_air_network(
        one_branch_spec(),
        fan_speed_fraction=1.0,
        damper_position_by_id={"laboratory_supply_damper": 1.0},
    )

    assert result.total_flow_m3_s == pytest.approx(0.264906471413009)
    assert result.zone_flow_m3_s == {
        "laboratory": pytest.approx(result.total_flow_m3_s)
    }
    assert result.zone_mass_flow_kg_s == {
        "laboratory": pytest.approx(result.total_flow_m3_s * 0.85)
    }
    assert result.fan_pressure_rise_pa == pytest.approx(280.7017543859649)
    assert result.shared_pressure_loss_pa == pytest.approx(
        1_000.0 * result.total_flow_m3_s**2
    )
    assert result.branch_pressure_loss_pa == {
        "laboratory": pytest.approx(3_000.0 * result.total_flow_m3_s**2)
    }
    assert result.fan_air_power_w == pytest.approx(
        result.fan_pressure_rise_pa * result.total_flow_m3_s
    )
    assert result.fan_electrical_power_w == pytest.approx(
        result.fan_air_power_w / 0.70
    )
    assert abs(result.operating_point_residual_pa) <= 1e-9
    assert result.mass_balance_residual_kg_s == {"laboratory": 0.0}


def test_solver_rejects_out_of_range_fan_speed() -> None:
    with pytest.raises(AirNetworkValidationError, match="fan_speed_fraction"):
        solve_air_network(
            one_branch_spec(),
            fan_speed_fraction=1.01,
            damper_position_by_id={"laboratory_supply_damper": 1.0},
        )


def test_solver_requires_exact_damper_command_ids() -> None:
    with pytest.raises(AirNetworkValidationError, match="damper command ids"):
        solve_air_network(
            one_branch_spec(),
            fan_speed_fraction=1.0,
            damper_position_by_id={"unknown_damper": 1.0},
        )


@pytest.mark.parametrize("position", [-0.01, 1.01, float("nan")])
def test_solver_rejects_invalid_damper_positions(position: float) -> None:
    with pytest.raises(AirNetworkValidationError, match="damper position"):
        solve_air_network(
            one_branch_spec(),
            fan_speed_fraction=1.0,
            damper_position_by_id={"laboratory_supply_damper": position},
        )


def test_solver_rejects_duplicate_zone_and_damper_ids() -> None:
    original = one_branch_spec()
    duplicate = AirNetworkSpec(
        fan=original.fan,
        shared_resistance_pa_s2_m6=original.shared_resistance_pa_s2_m6,
        air_density_kg_m3=original.air_density_kg_m3,
        branches=original.branches + original.branches,
    )

    with pytest.raises(AirNetworkValidationError, match="unique"):
        solve_air_network(
            duplicate,
            fan_speed_fraction=1.0,
            damper_position_by_id={"laboratory_supply_damper": 1.0},
        )


def test_solver_rejects_nonpositive_air_density() -> None:
    invalid = replace(one_branch_spec(), air_density_kg_m3=0.0)

    with pytest.raises(AirNetworkValidationError, match="air_density_kg_m3"):
        solve_air_network(
            invalid,
            fan_speed_fraction=1.0,
            damper_position_by_id={"laboratory_supply_damper": 1.0},
        )


def test_solver_rejects_zero_damper_leak_fraction() -> None:
    original = one_branch_spec()
    invalid = replace(
        original,
        branches=(replace(original.branches[0], damper_leak_fraction=0.0),),
    )

    with pytest.raises(AirNetworkValidationError, match="damper_leak_fraction"):
        solve_air_network(
            invalid,
            fan_speed_fraction=1.0,
            damper_position_by_id={"laboratory_supply_damper": 0.0},
        )


@pytest.mark.parametrize(
    ("fan", "message"),
    [
        (replace(one_branch_spec().fan, rated_free_delivery_m3_s=0.0), "rated_free_delivery"),
        (replace(one_branch_spec().fan, rated_shutoff_pressure_pa=0.0), "rated_shutoff"),
        (replace(one_branch_spec().fan, total_efficiency=0.0), "total_efficiency"),
        (replace(one_branch_spec().fan, total_efficiency=1.01), "total_efficiency"),
    ],
)
def test_solver_rejects_invalid_fan_spec(fan: FanSpec, message: str) -> None:
    invalid = replace(one_branch_spec(), fan=fan)

    with pytest.raises(AirNetworkValidationError, match=message):
        solve_air_network(
            invalid,
            fan_speed_fraction=0.0,
            damper_position_by_id={"laboratory_supply_damper": 1.0},
        )


@pytest.mark.parametrize(
    "invalid",
    [
        replace(one_branch_spec(), shared_resistance_pa_s2_m6=-1.0),
        replace(
            one_branch_spec(),
            branches=(
                replace(
                    one_branch_spec().branches[0],
                    open_supply_resistance_pa_s2_m6=0.0,
                ),
            ),
        ),
        replace(
            one_branch_spec(),
            branches=(
                replace(
                    one_branch_spec().branches[0],
                    return_resistance_pa_s2_m6=0.0,
                ),
            ),
        ),
    ],
)
def test_solver_rejects_invalid_resistance(invalid: AirNetworkSpec) -> None:
    with pytest.raises(AirNetworkValidationError, match="resistance"):
        solve_air_network(
            invalid,
            fan_speed_fraction=0.0,
            damper_position_by_id={"laboratory_supply_damper": 1.0},
        )
