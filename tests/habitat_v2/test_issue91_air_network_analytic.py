"""CPU tests for the closed-form batched air-network candidate (Issue #91).

Everything here runs without PyTorch or a GPU. Expected values come from three
legs that do not share code with the candidate: the reviewed bisection solver,
an 80-digit decimal root of the contract's implicit fan/system equation written
in this file, and direct substitution into the physical laws.
"""

from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from aeolus.habitat_v2 import air_network_analytic as analytic
from aeolus.habitat_v2.air_network import (
    AirNetworkSpec,
    BranchSpec,
    FanSpec,
    solve_air_network,
)
from aeolus.habitat_v2.air_network_analytic import (
    BRANCH_FIELDS,
    CONTRACT_CANONICAL_SHA256,
    SCALAR_FIELDS,
    AirNetworkAnalyticError,
    AirNetworkBatch,
    AirNetworkBatchResult,
    air_network_batch_sha256,
    batch_from_operating_points,
    compare_results,
    comparison_scales,
    generate_air_network_cases,
    load_air_network_batched_contract,
    solve_air_network_analytic,
    solve_air_network_analytic_scalar,
    validate_air_network_batch,
    validate_air_network_batched_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT, CONTRACT_SHA256 = load_air_network_batched_contract(REPO_ROOT)
PROFILES = CONTRACT["case_profiles"]
TOLERANCES = CONTRACT["tolerances"]


def _spec_from_batch(batch: AirNetworkBatch, world: int) -> tuple[AirNetworkSpec, dict[str, float]]:
    branches = tuple(
        BranchSpec(
            zone_id=f"zone_{index:02d}",
            damper_id=f"damper_{index:02d}",
            open_supply_resistance_pa_s2_m6=float(
                batch.open_supply_resistance_pa_s2_m6[world, index]
            ),
            return_resistance_pa_s2_m6=float(
                batch.return_resistance_pa_s2_m6[world, index]
            ),
            damper_leak_fraction=float(batch.damper_leak_fraction[world, index]),
        )
        for index in range(batch.branch_count)
    )
    spec = AirNetworkSpec(
        fan=FanSpec(
            component_id="fan",
            rated_free_delivery_m3_s=float(batch.rated_free_delivery_m3_s[world]),
            rated_shutoff_pressure_pa=float(batch.rated_shutoff_pressure_pa[world]),
            total_efficiency=float(batch.total_efficiency[world]),
        ),
        shared_resistance_pa_s2_m6=float(batch.shared_resistance_pa_s2_m6[world]),
        air_density_kg_m3=float(batch.air_density_kg_m3[world]),
        branches=branches,
    )
    dampers = {
        f"damper_{index:02d}": float(batch.damper_position_fraction[world, index])
        for index in range(batch.branch_count)
    }
    return spec, dampers


def _reference_batch_result(batch: AirNetworkBatch) -> AirNetworkBatchResult:
    """Run the reviewed scalar bisection solver once per world."""

    rows = []
    for world in range(batch.batch_size):
        spec, dampers = _spec_from_batch(batch, world)
        result = solve_air_network(
            spec,
            fan_speed_fraction=float(batch.fan_speed_fraction[world]),
            damper_position_by_id=dampers,
        )
        zones = sorted(result.zone_flow_m3_s)
        rows.append(
            (
                result.fan_pressure_rise_pa,
                result.shared_pressure_loss_pa,
                result.branch_pressure_loss_pa[zones[0]],
                result.total_flow_m3_s,
                result.fan_air_power_w,
                result.fan_electrical_power_w,
                result.operating_point_residual_pa,
                [result.zone_flow_m3_s[zone] for zone in zones],
                [result.zone_mass_flow_kg_s[zone] for zone in zones],
            )
        )
    columns = list(zip(*rows))
    return AirNetworkBatchResult(*(np.array(column, dtype=np.float64) for column in columns))


def _high_precision_root(batch: AirNetworkBatch) -> AirNetworkBatchResult:
    """Independent oracle: bisect the contract's implicit equation in decimal.

    Solves ``P_fan(Q(p_b), s) - R_shared * Q(p_b)**2 - p_b = 0`` with
    ``q_i(p_b) = sqrt(p_b / R_i)`` at 80 significant digits. It never uses the
    equivalent-resistance closed form under test.
    """

    rows = []
    with localcontext() as context:
        context.prec = 80
        for world in range(batch.batch_size):
            speed = Decimal(float(batch.fan_speed_fraction[world]))
            free = Decimal(float(batch.rated_free_delivery_m3_s[world])) * speed
            shutoff = Decimal(float(batch.rated_shutoff_pressure_pa[world])) * speed * speed
            shared_r = Decimal(float(batch.shared_resistance_pa_s2_m6[world]))
            density = Decimal(float(batch.air_density_kg_m3[world]))
            efficiency = Decimal(float(batch.total_efficiency[world]))
            inverse_roots = []
            for index in range(batch.branch_count):
                leak = Decimal(float(batch.damper_leak_fraction[world, index]))
                position = Decimal(float(batch.damper_position_fraction[world, index]))
                area = leak + (1 - leak) * position
                resistance = Decimal(
                    float(batch.open_supply_resistance_pa_s2_m6[world, index])
                ) / (area * area) + Decimal(float(batch.return_resistance_pa_s2_m6[world, index]))
                inverse_roots.append(1 / resistance.sqrt())
            if speed == 0:
                rows.append((0.0,) * 7 + ([0.0] * batch.branch_count,) * 2)
                continue

            def branch_flows(pressure: Decimal) -> list[Decimal]:
                root = pressure.sqrt()
                return [root * inverse for inverse in inverse_roots]

            def fan_pressure(total: Decimal) -> Decimal:
                return shutoff * max(Decimal(0), 1 - (total / free) ** 2)

            low, high = Decimal(0), shutoff
            for _ in range(420):
                middle = (low + high) / 2
                total = sum(branch_flows(middle))
                if fan_pressure(total) - shared_r * total * total - middle > 0:
                    low = middle
                else:
                    high = middle
            pressure = (low + high) / 2
            flows = branch_flows(pressure)
            total = sum(flows)
            fan = fan_pressure(total)
            rows.append(
                (
                    float(fan),
                    float(shared_r * total * total),
                    float(pressure),
                    float(total),
                    float(fan * total),
                    float(fan * total / efficiency),
                    0.0,
                    [float(flow) for flow in flows],
                    [float(flow * density) for flow in flows],
                )
            )
    columns = list(zip(*rows))
    return AirNetworkBatchResult(*(np.array(column, dtype=np.float64) for column in columns))


def _failures(report: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {name: entry for name, entry in report.items() if not entry["passed"]}


def _one_branch_spec() -> AirNetworkSpec:
    return AirNetworkSpec(
        fan=FanSpec("fan", 0.40, 500.0, 0.70),
        shared_resistance_pa_s2_m6=1_000.0,
        air_density_kg_m3=0.85,
        branches=(BranchSpec("laboratory", "laboratory_damper", 2_000.0, 1_000.0, 0.05),),
    )


def _replace(batch: AirNetworkBatch, **changes: np.ndarray) -> AirNetworkBatch:
    values = {name: getattr(batch, name) for name in SCALAR_FIELDS + BRANCH_FIELDS}
    values.update(changes)
    return AirNetworkBatch(**values)


def test_pinned_single_branch_operating_point_from_existing_unit_test() -> None:
    result = solve_air_network_analytic_scalar(
        _one_branch_spec(),
        fan_speed_fraction=1.0,
        damper_position_by_id={"laboratory_damper": 1.0},
    )

    # Constants pinned by tests/habitat_v2/test_air_network.py for the reviewed solver.
    assert result.total_flow_m3_s == pytest.approx(0.264906471413009, rel=1e-13)
    assert result.fan_pressure_rise_pa == pytest.approx(280.7017543859649, rel=1e-13)
    assert result.shared_pressure_loss_pa == pytest.approx(
        1_000.0 * 0.264906471413009**2, rel=1e-13
    )
    assert result.branch_pressure_loss_pa == {
        "laboratory": pytest.approx(3_000.0 * 0.264906471413009**2, rel=1e-13)
    }
    assert result.mass_balance_residual_kg_s == {"laboratory": 0.0}
    assert result.total_efficiency == 0.70 and result.air_density_kg_m3 == 0.85


@pytest.mark.parametrize("branch_count", (1, 2, 8, 16))
def test_candidate_matches_reviewed_scalar_solver_on_fixture_band(branch_count: int) -> None:
    batch = generate_air_network_cases(
        PROFILES["fixture_band"], seed=9100 + branch_count, batch_size=192,
        branch_count=branch_count,
    )
    report = compare_results(
        solve_air_network_analytic(batch),
        _reference_batch_result(batch),
        batch,
        TOLERANCES["numpy_fp64_vs_scalar_reference"]["variables"],
    )

    assert _failures(report) == {}


@pytest.mark.parametrize("branch_count", (1, 3, 8, 16))
def test_candidate_matches_independent_high_precision_root_on_stress_domain(
    branch_count: int,
) -> None:
    batch = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=9200 + branch_count, batch_size=96,
        branch_count=branch_count,
    )
    report = compare_results(
        solve_air_network_analytic(batch),
        _high_precision_root(batch),
        batch,
        TOLERANCES["numpy_fp64_vs_high_precision_root"]["variables"],
    )

    assert _failures(report) == {}


def test_outputs_satisfy_the_physical_laws_by_direct_substitution() -> None:
    batch = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=9300, batch_size=512, branch_count=8
    )
    result = solve_air_network_analytic(batch)
    speed = batch.fan_speed_fraction

    area = batch.damper_leak_fraction + (1.0 - batch.damper_leak_fraction) * (
        batch.damper_position_fraction
    )
    resistance = (
        batch.open_supply_resistance_pa_s2_m6 / area**2 + batch.return_resistance_pa_s2_m6
    )
    # Branch law: every parallel branch sees the same pressure R_i * q_i**2.
    np.testing.assert_allclose(
        resistance * result.zone_flow_m3_s**2,
        np.broadcast_to(result.branch_pressure_loss_pa[:, None], resistance.shape),
        rtol=1e-12, atol=1e-300,
    )
    # Volume continuity at the supply node.
    np.testing.assert_allclose(
        result.zone_flow_m3_s.sum(axis=1), result.total_flow_m3_s, rtol=1e-13, atol=0.0
    )
    # Shared loss law.
    np.testing.assert_allclose(
        batch.shared_resistance_pa_s2_m6 * result.total_flow_m3_s**2,
        result.shared_pressure_loss_pa, rtol=1e-12, atol=1e-300,
    )
    # Fan curve at the solved flow balances the system, relative to shutoff pressure.
    shutoff = batch.rated_shutoff_pressure_pa * speed**2
    running = speed > 0.0
    normalized = result.total_flow_m3_s[running] / (
        batch.rated_free_delivery_m3_s[running] * speed[running]
    )
    # Below free delivery in exact arithmetic; rounding may add a few ulps.
    assert bool((normalized <= 1.0 + 4.0 * np.finfo(np.float64).eps).all())
    fan_curve = shutoff[running] * (1.0 - normalized**2)
    system = (
        result.shared_pressure_loss_pa[running] + result.branch_pressure_loss_pa[running]
    )
    assert float(np.max(np.abs(fan_curve - system) / shutoff[running])) <= 1e-12
    assert float(np.max(np.abs(result.operating_point_residual_pa[running]) / shutoff[running])) <= 1e-12
    # The reported residual is the clamped fan curve minus the losses, not a
    # constant: recompute it from the returned outputs.
    recomputed = shutoff[running] * np.maximum(0.0, 1.0 - normalized**2) - system
    assert float(
        np.max(np.abs(result.operating_point_residual_pa[running] - recomputed) / shutoff[running])
    ) <= 8.0 * np.finfo(np.float64).eps
    assert np.count_nonzero(result.operating_point_residual_pa) > 100
    # Pressure rise equals total loss; power and mass definitions.
    np.testing.assert_allclose(
        result.fan_pressure_rise_pa,
        result.shared_pressure_loss_pa + result.branch_pressure_loss_pa,
        rtol=1e-13, atol=0.0,
    )
    assert np.array_equal(
        result.fan_air_power_w, result.fan_pressure_rise_pa * result.total_flow_m3_s
    )
    assert np.array_equal(
        result.fan_electrical_power_w, result.fan_air_power_w / batch.total_efficiency
    )
    assert np.array_equal(
        result.zone_mass_flow_kg_s, result.zone_flow_m3_s * batch.air_density_kg_m3[:, None]
    )
    assert bool((result.fan_electrical_power_w >= result.fan_air_power_w).all())
    for name in analytic.COMPARISON_VARIABLES:
        values = getattr(result, name)
        assert bool(np.isfinite(values).all()), name
        if name != "operating_point_residual_pa":
            assert bool((values >= 0.0).all()), name


def test_fan_affinity_laws_hold_across_twelve_decades_of_speed() -> None:
    base = generate_air_network_cases(
        PROFILES["fixture_band"], seed=9400, batch_size=16, branch_count=8
    )
    unit = solve_air_network_analytic(
        _replace(base, fan_speed_fraction=np.ones(16))
    )
    for speed in (0.5, 1e-3, 1e-6, 1e-12, 1e-30, 1e-100):
        scaled = solve_air_network_analytic(
            _replace(base, fan_speed_fraction=np.full(16, speed))
        )
        np.testing.assert_allclose(scaled.total_flow_m3_s, unit.total_flow_m3_s * speed, rtol=1e-14)
        np.testing.assert_allclose(scaled.zone_flow_m3_s, unit.zone_flow_m3_s * speed, rtol=1e-14)
        np.testing.assert_allclose(
            scaled.fan_pressure_rise_pa, unit.fan_pressure_rise_pa * speed**2, rtol=1e-14
        )
        np.testing.assert_allclose(
            scaled.fan_air_power_w, unit.fan_air_power_w * speed**3, rtol=1e-14, atol=1e-300
        )


def test_tiny_positive_speed_matches_reference_where_the_reference_is_representable() -> None:
    spec, dampers = _one_branch_spec(), {"laboratory_damper": 0.4}
    for speed in (1e-6, 1e-30, 1e-100):
        reference = solve_air_network(spec, fan_speed_fraction=speed, damper_position_by_id=dampers)
        candidate = solve_air_network_analytic_scalar(
            spec, fan_speed_fraction=speed, damper_position_by_id=dampers
        )
        assert candidate.total_flow_m3_s == pytest.approx(reference.total_flow_m3_s, rel=1e-12)
        assert candidate.fan_pressure_rise_pa == pytest.approx(
            reference.fan_pressure_rise_pa, rel=1e-12
        )
        assert candidate.fan_electrical_power_w == pytest.approx(
            reference.fan_electrical_power_w, rel=1e-12
        )


def test_reference_underflow_limitations_are_reported_not_hidden() -> None:
    spec, dampers = _one_branch_spec(), {"laboratory_damper": 0.4}
    unit = solve_air_network_analytic_scalar(
        spec, fan_speed_fraction=1.0, damper_position_by_id=dampers
    )

    # s**2 underflows to zero in FP64: the bisection bracket collapses and the
    # reviewed solver reports zero flow for a running fan. The candidate keeps
    # flow linear in speed; its pressure and power underflow to exact zero.
    speed = 1e-170
    reference = solve_air_network(spec, fan_speed_fraction=speed, damper_position_by_id=dampers)
    candidate = solve_air_network_analytic_scalar(
        spec, fan_speed_fraction=speed, damper_position_by_id=dampers
    )
    assert reference.total_flow_m3_s == 0.0
    assert candidate.total_flow_m3_s == pytest.approx(unit.total_flow_m3_s * speed, rel=1e-14)
    assert candidate.fan_pressure_rise_pa == 0.0
    assert candidate.fan_electrical_power_w == 0.0

    # Free delivery itself underflows to zero: the reviewed solver divides by it.
    smallest = 5e-324
    with pytest.raises(ZeroDivisionError):
        solve_air_network(spec, fan_speed_fraction=smallest, damper_position_by_id=dampers)
    candidate = solve_air_network_analytic_scalar(
        spec, fan_speed_fraction=smallest, damper_position_by_id=dampers
    )
    assert math.isfinite(candidate.total_flow_m3_s) and candidate.total_flow_m3_s >= 0.0
    assert candidate.operating_point_residual_pa == 0.0

    # A leak fraction whose square underflows makes the reviewed solver divide
    # by zero; it is outside the candidate's supported domain and is rejected.
    tiny_leak = AirNetworkSpec(
        fan=spec.fan, shared_resistance_pa_s2_m6=1_000.0, air_density_kg_m3=0.85,
        branches=(BranchSpec("laboratory", "laboratory_damper", 2_000.0, 1_000.0, 1e-200),),
    )
    with pytest.raises(ZeroDivisionError):
        solve_air_network(
            tiny_leak, fan_speed_fraction=1.0, damper_position_by_id={"laboratory_damper": 0.0}
        )
    with pytest.raises(AirNetworkAnalyticError, match="damper_leak_fraction"):
        solve_air_network_analytic_scalar(
            tiny_leak, fan_speed_fraction=1.0, damper_position_by_id={"laboratory_damper": 0.0}
        )


def test_fan_off_is_exactly_zero_and_still_validates_the_specification() -> None:
    batch = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=9500, batch_size=32, branch_count=8
    )
    for zero in (0.0, -0.0):
        result = solve_air_network_analytic(
            _replace(batch, fan_speed_fraction=np.full(32, zero))
        )
        for name in analytic.COMPARISON_VARIABLES:
            values = getattr(result, name)
            assert not values.any(), name
            assert not np.signbit(values).any(), name

    # Fan-dominated world found by review: rounding puts the unit-speed flow one
    # ulp above free delivery, which used to leak a -0.0 residual at fan-off.
    fan_dominated = AirNetworkBatch(
        rated_free_delivery_m3_s=np.array([0.00010160200152086619] * 2),
        rated_shutoff_pressure_pa=np.array([6538735.763003341] * 2),
        total_efficiency=np.array([0.423934216603324] * 2),
        shared_resistance_pa_s2_m6=np.array([0.0, -0.0]),
        air_density_kg_m3=np.array([0.85] * 2),
        fan_speed_fraction=np.array([0.0, -0.0]),
        open_supply_resistance_pa_s2_m6=np.array([[0.013209198462124311]] * 2),
        return_resistance_pa_s2_m6=np.array([[0.03824337072633355]] * 2),
        damper_leak_fraction=np.array([[0.00032751340371570256]] * 2),
        damper_position_fraction=np.array([[0.9111857685020996]] * 2),
    )
    result = solve_air_network_analytic(fan_dominated)
    for name in analytic.COMPARISON_VARIABLES:
        values = getattr(result, name)
        assert not values.any() and not np.signbit(values).any(), name
    running = solve_air_network_analytic(
        _replace(fan_dominated, fan_speed_fraction=np.array([1.0, 1.0]))
    )
    # Here the clamped fan-curve side is exactly zero, so the residual is minus
    # the (tiny) system loss: bounded relative to shutoff pressure, not zero.
    assert bool(
        (np.abs(running.operating_point_residual_pa) <= 1e-12 * 6538735.763003341).all()
    )
    assert not np.signbit(running.shared_pressure_loss_pa).any()

    invalid = AirNetworkSpec(
        fan=FanSpec("fan", 0.40, 500.0, 1.01),
        shared_resistance_pa_s2_m6=1_000.0, air_density_kg_m3=0.85,
        branches=_one_branch_spec().branches,
    )
    with pytest.raises(AirNetworkAnalyticError, match="total_efficiency"):
        solve_air_network_analytic_scalar(
            invalid, fan_speed_fraction=0.0, damper_position_by_id={"laboratory_damper": 1.0}
        )


def test_resistance_extremes_and_asymmetric_branches_stay_finite_and_ordered() -> None:
    low, high = analytic.SUPPORTED_DOMAIN["open_supply_resistance_pa_s2_m6"]
    leak_low = analytic.SUPPORTED_DOMAIN["damper_leak_fraction"][0]
    batch = AirNetworkBatch(
        rated_free_delivery_m3_s=np.array([0.75, 0.75]),
        rated_shutoff_pressure_pa=np.array([900.0, 900.0]),
        total_efficiency=np.array([0.68, 0.68]),
        shared_resistance_pa_s2_m6=np.array([0.0, 1e9]),
        air_density_kg_m3=np.array([0.85, 0.85]),
        fan_speed_fraction=np.array([1.0, 1.0]),
        open_supply_resistance_pa_s2_m6=np.array([[low, high, 7000.0], [high, low, 7000.0]]),
        return_resistance_pa_s2_m6=np.array([[low, high, 3000.0], [high, low, 3000.0]]),
        damper_leak_fraction=np.array([[1.0, leak_low, 0.08], [leak_low, 1.0, 0.08]]),
        damper_position_fraction=np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
    )
    result = solve_air_network_analytic(batch)

    report = compare_results(
        result, _high_precision_root(batch), batch,
        TOLERANCES["numpy_fp64_vs_high_precision_root"]["variables"],
    )
    assert _failures(report) == {}
    # The near-blocked branch carries the least flow; the near-open one the most.
    assert result.zone_flow_m3_s[0, 1] < result.zone_flow_m3_s[0, 2] < result.zone_flow_m3_s[0, 0]
    assert result.zone_flow_m3_s[1, 0] < result.zone_flow_m3_s[1, 2] < result.zone_flow_m3_s[1, 1]
    # With no shared loss the branch pressure is the whole fan pressure rise.
    assert result.shared_pressure_loss_pa[0] == 0.0
    assert result.branch_pressure_loss_pa[0] == result.fan_pressure_rise_pa[0]


def test_closing_a_damper_or_adding_resistance_never_increases_flow() -> None:
    batch = generate_air_network_cases(
        PROFILES["fixture_band"], seed=9600, batch_size=128, branch_count=8
    )
    batch = _replace(
        batch,
        fan_speed_fraction=np.full(128, 0.8),
        damper_position_fraction=np.clip(batch.damper_position_fraction, 0.2, 1.0),
    )
    before = solve_air_network_analytic(batch)

    closed = batch.damper_position_fraction.copy()
    closed[:, 3] *= 0.5
    after_close = solve_air_network_analytic(_replace(batch, damper_position_fraction=closed))
    assert bool((after_close.zone_flow_m3_s[:, 3] < before.zone_flow_m3_s[:, 3]).all())
    assert bool((after_close.total_flow_m3_s < before.total_flow_m3_s).all())
    others = [index for index in range(8) if index != 3]
    assert bool(
        (after_close.zone_flow_m3_s[:, others] > before.zone_flow_m3_s[:, others]).all()
    )

    blocked = batch.open_supply_resistance_pa_s2_m6.copy()
    blocked[:, 5] *= 3.0
    after_block = solve_air_network_analytic(
        _replace(batch, open_supply_resistance_pa_s2_m6=blocked)
    )
    assert bool((after_block.total_flow_m3_s < before.total_flow_m3_s).all())

    after_shared = solve_air_network_analytic(
        _replace(batch, shared_resistance_pa_s2_m6=batch.shared_resistance_pa_s2_m6 * 2.0)
    )
    assert bool((after_shared.total_flow_m3_s < before.total_flow_m3_s).all())


def test_identical_parallel_branches_collapse_to_one_equivalent_branch() -> None:
    count = 4
    many = AirNetworkBatch(
        rated_free_delivery_m3_s=np.array([0.75]),
        rated_shutoff_pressure_pa=np.array([900.0]),
        total_efficiency=np.array([0.68]),
        shared_resistance_pa_s2_m6=np.array([1000.0]),
        air_density_kg_m3=np.array([0.85]),
        fan_speed_fraction=np.array([0.9]),
        open_supply_resistance_pa_s2_m6=np.full((1, count), 8000.0),
        return_resistance_pa_s2_m6=np.full((1, count), 2000.0),
        damper_leak_fraction=np.full((1, count), 1.0),
        damper_position_fraction=np.full((1, count), 1.0),
    )
    # N equal quadratic resistances in parallel are one resistance R / N**2.
    single = _replace(
        many,
        open_supply_resistance_pa_s2_m6=np.array([[8000.0 / count**2]]),
        return_resistance_pa_s2_m6=np.array([[2000.0 / count**2]]),
        damper_leak_fraction=np.array([[1.0]]),
        damper_position_fraction=np.array([[1.0]]),
    )
    many_result, single_result = solve_air_network_analytic(many), solve_air_network_analytic(single)

    assert many_result.total_flow_m3_s[0] == pytest.approx(single_result.total_flow_m3_s[0], rel=1e-14)
    assert many_result.fan_electrical_power_w[0] == pytest.approx(
        single_result.fan_electrical_power_w[0], rel=1e-14
    )
    np.testing.assert_allclose(
        many_result.zone_flow_m3_s[0], many_result.total_flow_m3_s[0] / count, rtol=1e-14
    )


def test_worlds_are_isolated_from_batch_size_order_and_neighbours() -> None:
    batch = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=9700, batch_size=64, branch_count=8
    )
    full = solve_air_network_analytic(batch)

    def take(indices: np.ndarray) -> AirNetworkBatch:
        return AirNetworkBatch(
            **{
                name: np.ascontiguousarray(getattr(batch, name)[indices])
                for name in SCALAR_FIELDS + BRANCH_FIELDS
            }
        )

    order = np.random.Generator(np.random.PCG64(1)).permutation(64)
    permuted = solve_air_network_analytic(take(order))
    for name in analytic.COMPARISON_VARIABLES:
        assert np.array_equal(getattr(permuted, name), getattr(full, name)[order]), name
    for world in (0, 17, 63):
        alone = solve_air_network_analytic(take(np.array([world])))
        for name in analytic.COMPARISON_VARIABLES:
            assert np.array_equal(getattr(alone, name)[0], getattr(full, name)[world]), name

    # Reordering branches reorders per-branch outputs; world totals move only by
    # summation-order rounding.
    reversed_branches = AirNetworkBatch(
        **{name: getattr(batch, name) for name in SCALAR_FIELDS},
        **{name: np.ascontiguousarray(getattr(batch, name)[:, ::-1]) for name in BRANCH_FIELDS},
    )
    flipped = solve_air_network_analytic(reversed_branches)
    np.testing.assert_allclose(flipped.zone_flow_m3_s[:, ::-1], full.zone_flow_m3_s, rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(flipped.total_flow_m3_s, full.total_flow_m3_s, rtol=1e-13, atol=0.0)


def test_solve_does_not_mutate_inputs() -> None:
    batch = generate_air_network_cases(
        PROFILES["fixture_band"], seed=9800, batch_size=8, branch_count=8
    )
    before = air_network_batch_sha256(batch)
    solve_air_network_analytic(batch)

    assert air_network_batch_sha256(batch) == before


def _valid_batch() -> AirNetworkBatch:
    return generate_air_network_cases(
        PROFILES["fixture_band"], seed=9900, batch_size=4, branch_count=3
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fan_speed_fraction", math.nan, "fan_speed_fraction in world 2"),
        ("fan_speed_fraction", math.inf, "fan_speed_fraction in world 2"),
        ("fan_speed_fraction", 1.0000001, "fan_speed_fraction in world 2"),
        ("fan_speed_fraction", -1e-300, "fan_speed_fraction in world 2"),
        ("rated_free_delivery_m3_s", 0.0, "rated_free_delivery_m3_s in world 2"),
        ("rated_shutoff_pressure_pa", -900.0, "rated_shutoff_pressure_pa in world 2"),
        ("rated_shutoff_pressure_pa", 1e300, "rated_shutoff_pressure_pa in world 2"),
        ("total_efficiency", 1.01, "total_efficiency in world 2"),
        ("total_efficiency", 0.0, "total_efficiency in world 2"),
        ("shared_resistance_pa_s2_m6", -1.0, "shared_resistance_pa_s2_m6 in world 2"),
        ("air_density_kg_m3", 0.0, "air_density_kg_m3 in world 2"),
        ("open_supply_resistance_pa_s2_m6", 0.0, "open_supply_resistance_pa_s2_m6 in world 2"),
        ("open_supply_resistance_pa_s2_m6", math.nan, "open_supply_resistance_pa_s2_m6 in world 2"),
        ("return_resistance_pa_s2_m6", -math.inf, "return_resistance_pa_s2_m6 in world 2"),
        ("damper_leak_fraction", 0.0, "damper_leak_fraction in world 2"),
        ("damper_leak_fraction", 1.5, "damper_leak_fraction in world 2"),
        ("damper_position_fraction", -0.01, "damper_position_fraction in world 2"),
        ("damper_position_fraction", 1.01, "damper_position_fraction in world 2"),
        ("damper_position_fraction", math.nan, "damper_position_fraction in world 2"),
    ),
)
def test_invalid_or_nonfinite_values_fail_closed(field: str, value: float, message: str) -> None:
    batch = _valid_batch()
    array = getattr(batch, field).copy()
    array[2] = value

    with pytest.raises(AirNetworkAnalyticError, match=message):
        solve_air_network_analytic(_replace(batch, **{field: array}))


def test_malformed_layouts_fail_closed() -> None:
    batch = _valid_batch()

    cases = {
        "float32 dtype": {"fan_speed_fraction": batch.fan_speed_fraction.astype(np.float32)},
        "boolean dtype": {"damper_position_fraction": batch.damper_position_fraction > 0.5},
        "integer dtype": {"fan_speed_fraction": np.ones(4, dtype=np.int64)},
        "python list": {"fan_speed_fraction": [0.5, 0.5, 0.5, 0.5]},
        "world count mismatch": {"total_efficiency": batch.total_efficiency[:3].copy()},
        "world field with branch axis": {"air_density_kg_m3": np.ones((4, 1))},
        "branch field without branch axis": {"damper_leak_fraction": np.full(4, 0.1)},
        "branch count mismatch": {
            "return_resistance_pa_s2_m6": np.full((4, 2), 3000.0)
        },
        "branch world mismatch": {
            "return_resistance_pa_s2_m6": np.full((3, 3), 3000.0)
        },
    }
    for label, change in cases.items():
        with pytest.raises(AirNetworkAnalyticError):
            validate_air_network_batch(_replace(batch, **change))
        with pytest.raises(AirNetworkAnalyticError):
            solve_air_network_analytic(_replace(batch, **change)), label

    def shaped(worlds: int, branches: int) -> AirNetworkBatch:
        return AirNetworkBatch(
            **{name: np.full(worlds, 0.5) for name in SCALAR_FIELDS},
            **{name: np.full((worlds, branches), 0.5) for name in BRANCH_FIELDS},
        )

    validate_air_network_batch(_replace(shaped(2, 16), **{
        "open_supply_resistance_pa_s2_m6": np.full((2, 16), 7000.0),
        "return_resistance_pa_s2_m6": np.full((2, 16), 3000.0),
    }))
    for worlds, branches in ((0, 3), (2, 0), (2, 17)):
        with pytest.raises(AirNetworkAnalyticError, match="shape"):
            validate_air_network_batch(shaped(worlds, branches))
    with pytest.raises(AirNetworkAnalyticError, match="AirNetworkBatch"):
        validate_air_network_batch({"fan_speed_fraction": np.ones(1)})  # type: ignore[arg-type]


def test_operating_point_packing_rejects_inconsistent_identifiers_and_types() -> None:
    spec = _one_branch_spec()
    with pytest.raises(AirNetworkAnalyticError, match="damper command ids"):
        batch_from_operating_points([(spec, 1.0, {"other_damper": 1.0})])
    with pytest.raises(AirNetworkAnalyticError, match="real numbers"):
        batch_from_operating_points([(spec, True, {"laboratory_damper": 1.0})])
    with pytest.raises(AirNetworkAnalyticError, match="real numbers"):
        batch_from_operating_points([(spec, 1.0, {"laboratory_damper": "1.0"})])
    with pytest.raises(AirNetworkAnalyticError, match="at least one operating point"):
        batch_from_operating_points([])
    duplicate = AirNetworkSpec(
        fan=spec.fan, shared_resistance_pa_s2_m6=1_000.0, air_density_kg_m3=0.85,
        branches=spec.branches + spec.branches,
    )
    with pytest.raises(AirNetworkAnalyticError, match="unique"):
        batch_from_operating_points([(duplicate, 1.0, {"laboratory_damper": 1.0})])
    empty = AirNetworkSpec(
        fan=spec.fan, shared_resistance_pa_s2_m6=1_000.0, air_density_kg_m3=0.85, branches=()
    )
    with pytest.raises(AirNetworkAnalyticError, match="at least one branch"):
        batch_from_operating_points([(empty, 1.0, {})])
    other = AirNetworkSpec(
        fan=spec.fan, shared_resistance_pa_s2_m6=1_000.0, air_density_kg_m3=0.85,
        branches=(BranchSpec("galley", "galley_damper", 2_000.0, 1_000.0, 0.05),),
    )
    with pytest.raises(AirNetworkAnalyticError, match="same zone ids"):
        batch_from_operating_points(
            [(spec, 1.0, {"laboratory_damper": 1.0}), (other, 1.0, {"galley_damper": 1.0})]
        )


def test_branch_axis_is_sorted_by_zone_id_like_the_reference_result() -> None:
    spec = AirNetworkSpec(
        fan=FanSpec("fan", 0.75, 900.0, 0.68),
        shared_resistance_pa_s2_m6=450.0, air_density_kg_m3=0.85,
        branches=(
            BranchSpec("beta", "d_b", 5500.0, 3000.0, 0.08),
            BranchSpec("alpha", "d_a", 8000.0, 2000.0, 0.08),
        ),
    )
    # Insertion order deliberately differs from zone order: lookup is by id.
    dampers = {"d_b": 0.25, "d_a": 1.0}
    batch, zone_ids = batch_from_operating_points([(spec, 0.9, dampers)])
    reference = solve_air_network(spec, fan_speed_fraction=0.9, damper_position_by_id=dampers)
    candidate = solve_air_network_analytic_scalar(
        spec, fan_speed_fraction=0.9, damper_position_by_id=dampers
    )

    assert zone_ids == ("alpha", "beta") == tuple(reference.zone_flow_m3_s)
    assert batch.open_supply_resistance_pa_s2_m6.tolist() == [[8000.0, 5500.0]]
    assert batch.return_resistance_pa_s2_m6.tolist() == [[2000.0, 3000.0]]
    assert batch.damper_position_fraction.tolist() == [[1.0, 0.25]]
    assert candidate.zone_flow_m3_s["alpha"] > candidate.zone_flow_m3_s["beta"]
    for name in (
        "fan_pressure_rise_pa", "shared_pressure_loss_pa", "total_flow_m3_s",
        "fan_air_power_w", "fan_electrical_power_w", "total_efficiency", "air_density_kg_m3",
    ):
        assert getattr(candidate, name) == pytest.approx(getattr(reference, name), rel=1e-12), name
    for name in ("branch_pressure_loss_pa", "zone_flow_m3_s", "zone_mass_flow_kg_s"):
        mapping = getattr(candidate, name)
        assert list(mapping) == ["alpha", "beta"], name
        for zone in zone_ids:
            assert mapping[zone] == pytest.approx(getattr(reference, name)[zone], rel=1e-12), name
    assert candidate.mass_balance_residual_kg_s == reference.mass_balance_residual_kg_s
    assert abs(candidate.operating_point_residual_pa) <= 1e-12 * 900.0


def test_every_solver_call_of_the_checked_in_scenario_runs_is_reproduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from aeolus.habitat_v2 import physics
    from aeolus.habitat_v2.config import load_scenario_file
    from aeolus.habitat_v2.runner import run_scenario

    calls: list[tuple[AirNetworkSpec, float, dict[str, float], object]] = []

    def recording_solver(spec, *, fan_speed_fraction, damper_position_by_id):
        result = solve_air_network(
            spec,
            fan_speed_fraction=fan_speed_fraction,
            damper_position_by_id=damper_position_by_id,
        )
        calls.append((spec, fan_speed_fraction, dict(damper_position_by_id), result))
        return result

    monkeypatch.setattr(physics, "solve_air_network", recording_solver)
    scenario_paths = [
        path
        for path in sorted((REPO_ROOT / "scenarios").glob("habitat_v2*.json"))
        + sorted((REPO_ROOT / "scenarios/habitat_v2_observability").glob("*.json"))
        if "air_network" in json.loads(path.read_text(encoding="utf-8"))
    ]
    assert len(scenario_paths) == 12
    for path in scenario_paths:
        run_scenario(load_scenario_file(path))

    # Faulted, slewing and fan-off plant states, exactly as the plant issued them.
    assert len(calls) > 300
    assert len({call[1] for call in calls}) > 10, "runs must exercise several fan speeds"
    assert any(call[1] == 0.0 for call in calls)
    batch, zone_ids = batch_from_operating_points([call[:3] for call in calls])
    assert len(zone_ids) == 8
    candidate = solve_air_network_analytic(batch)
    reference = AirNetworkBatchResult(
        fan_pressure_rise_pa=np.array([call[3].fan_pressure_rise_pa for call in calls]),
        shared_pressure_loss_pa=np.array([call[3].shared_pressure_loss_pa for call in calls]),
        branch_pressure_loss_pa=np.array(
            [call[3].branch_pressure_loss_pa[zone_ids[0]] for call in calls]
        ),
        total_flow_m3_s=np.array([call[3].total_flow_m3_s for call in calls]),
        fan_air_power_w=np.array([call[3].fan_air_power_w for call in calls]),
        fan_electrical_power_w=np.array([call[3].fan_electrical_power_w for call in calls]),
        operating_point_residual_pa=np.array(
            [call[3].operating_point_residual_pa for call in calls]
        ),
        zone_flow_m3_s=np.array(
            [[call[3].zone_flow_m3_s[zone] for zone in zone_ids] for call in calls]
        ),
        zone_mass_flow_kg_s=np.array(
            [[call[3].zone_mass_flow_kg_s[zone] for zone in zone_ids] for call in calls]
        ),
    )
    report = compare_results(
        candidate, reference, batch, TOLERANCES["numpy_fp64_vs_scalar_reference"]["variables"]
    )

    assert _failures(report) == {}


def test_generator_is_deterministic_seed_separated_and_inside_the_domain() -> None:
    first = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=91, batch_size=64, branch_count=8
    )
    again = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=91, batch_size=64, branch_count=8
    )
    other = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=92, batch_size=64, branch_count=8
    )

    assert air_network_batch_sha256(first) == air_network_batch_sha256(again)
    assert air_network_batch_sha256(first) != air_network_batch_sha256(other)
    # Pinned bytes: the generator uses only Generator.random and exact IEEE
    # operations, so this identity is expected to hold across platforms.
    assert air_network_batch_sha256(first) == PINNED_STRESS_SHA256
    assert air_network_batch_sha256(
        generate_air_network_cases(
            PROFILES["fixture_band"], seed=91, batch_size=64, branch_count=8
        )
    ) == PINNED_FIXTURE_BAND_SHA256
    assert first.batch_size == 64 and first.branch_count == 8
    # The stress profile must actually reach the declared special cases.
    assert (first.fan_speed_fraction == 0.0).any() or (other.fan_speed_fraction == 0.0).any()
    assert (first.damper_position_fraction == 0.0).any()
    assert (first.damper_position_fraction == 1.0).any()
    assert (first.shared_resistance_pa_s2_m6 == 0.0).any()
    assert first.fan_speed_fraction[first.fan_speed_fraction > 0.0].min() < 1e-6
    assert first.open_supply_resistance_pa_s2_m6.max() / first.open_supply_resistance_pa_s2_m6.min() > 1e8

    for profile_id, profile in PROFILES.items():
        batch = generate_air_network_cases(profile, seed=5, batch_size=2048, branch_count=16)
        validate_air_network_batch(batch)
        for name, rule in profile["variables"].items():
            values = getattr(batch, name)
            if rule["distribution"] == "uniform":
                low, high = rule["low"], rule["high"]
            else:
                low, high = 10.0 ** rule["low_exponent"], 10.0 ** rule["high_exponent"]
            special = (values == 0.0) | (values == 1.0)
            assert bool(((values >= low) & (values <= high) | special).all()), (profile_id, name)


PINNED_STRESS_SHA256 = "d05cd1f8c9771d117856aea1d7a71e37348bf3d88464b853bb53e322e41bca09"
PINNED_FIXTURE_BAND_SHA256 = "d4a59d45ee4078305bd47cdcaa3b8e85f16a667862af9587e3e03378173e2b23"


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({"seed": -1, "batch_size": 4, "branch_count": 8}, "seed"),
        ({"seed": True, "batch_size": 4, "branch_count": 8}, "seed"),
        ({"seed": 1.5, "batch_size": 4, "branch_count": 8}, "seed"),
        ({"seed": 1, "batch_size": 0, "branch_count": 8}, "batch_size"),
        ({"seed": 1, "batch_size": 4, "branch_count": 0}, "branch_count"),
        ({"seed": 1, "batch_size": 4, "branch_count": 17}, "branch_count"),
    ),
)
def test_generator_rejects_invalid_requests(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(AirNetworkAnalyticError, match=message):
        generate_air_network_cases(PROFILES["fixture_band"], **arguments)  # type: ignore[arg-type]


def test_generator_rejects_drifted_profiles() -> None:
    missing = copy.deepcopy(PROFILES["fixture_band"])
    del missing["variables"]["air_density_kg_m3"]
    with pytest.raises(AirNetworkAnalyticError, match="variable set drifted"):
        generate_air_network_cases(missing, seed=1, batch_size=4, branch_count=8)

    unknown = copy.deepcopy(PROFILES["fixture_band"])
    unknown["variables"]["air_density_kg_m3"]["distribution"] = "normal"
    with pytest.raises(AirNetworkAnalyticError, match="unknown distribution"):
        generate_air_network_cases(unknown, seed=1, batch_size=4, branch_count=8)

    outside = copy.deepcopy(PROFILES["fixture_band"])
    outside["variables"]["fan_speed_fraction"]["high"] = 2.0
    with pytest.raises(AirNetworkAnalyticError, match="fan_speed_fraction"):
        generate_air_network_cases(outside, seed=1, batch_size=64, branch_count=8)


def test_comparison_reports_violations_and_nonfinite_candidates() -> None:
    batch = _valid_batch()
    reference = solve_air_network_analytic(batch)
    tolerances = TOLERANCES["torch_fp64_vs_numpy_fp64"]["variables"]
    assert _failures(compare_results(reference, reference, batch, tolerances)) == {}

    def altered(field: str, factor: float) -> AirNetworkBatchResult:
        values = {name: getattr(reference, name) for name in analytic.COMPARISON_VARIABLES}
        changed = values[field].copy()
        changed[1] = changed[1] * factor
        values[field] = changed
        return AirNetworkBatchResult(**values)

    report = compare_results(altered("total_flow_m3_s", 1.0 + 1e-9), reference, batch, tolerances)
    assert set(_failures(report)) == {"total_flow_m3_s"}
    assert report["total_flow_m3_s"]["violation_count"] == 1
    assert report["total_flow_m3_s"]["max_error_over_allowed"] > 1.0

    report = compare_results(altered("zone_flow_m3_s", math.nan), reference, batch, tolerances)
    assert report["zone_flow_m3_s"]["passed"] is False
    assert report["zone_flow_m3_s"]["candidate_finite"] is False

    with pytest.raises(AirNetworkAnalyticError, match="variable set drifted"):
        compare_results(reference, reference, batch, {"total_flow_m3_s": {}})
    drifted = copy.deepcopy(tolerances)
    drifted["total_flow_m3_s"]["extra"] = 0.0
    with pytest.raises(AirNetworkAnalyticError, match="field set drifted"):
        compare_results(reference, reference, batch, drifted)


def _write_contract(root: Path, raw: bytes) -> Path:
    path = root / analytic.CONTRACT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return root


def test_contract_digest_is_pinned_so_tolerances_cannot_drift_silently(tmp_path: Path) -> None:
    assert CONTRACT["status"] == "FROZEN_BEFORE_BENCHMARK_RESULTS"
    assert CONTRACT_SHA256 == CONTRACT_CANONICAL_SHA256
    assert CONTRACT["authority"]["canonical_replay_or_hash_authority"] is False
    assert CONTRACT["authority"]["actuator_authority"] is False
    assert tuple(TOLERANCES) == analytic.COMPARISON_IDS
    assert tuple(PROFILES) == analytic.CASE_PROFILE_IDS

    # A structurally valid contract with one loosened tolerance is refused.
    loosened = copy.deepcopy(CONTRACT)
    loosened["tolerances"]["torch_fp32_vs_numpy_fp64"]["variables"]["total_flow_m3_s"][
        "relative_to_reference"
    ] = 1e-3
    validate_air_network_batched_contract(copy.deepcopy(loosened))
    with pytest.raises(AirNetworkAnalyticError, match="pinned frozen digest"):
        load_air_network_batched_contract(
            _write_contract(tmp_path / "loosened", json.dumps(loosened).encode("utf-8"))
        )
    # Reformatting alone (different key order, indentation, CRLF) is accepted.
    reformatted = json.dumps(CONTRACT, indent=4, sort_keys=True).replace("\n", "\r\n")
    reloaded, digest = load_air_network_batched_contract(
        _write_contract(tmp_path / "reformatted", reformatted.encode("utf-8"))
    )
    assert reloaded == CONTRACT and digest == CONTRACT_CANONICAL_SHA256


def test_contract_loader_rejects_malformed_bytes(tmp_path: Path) -> None:
    raw = json.dumps(CONTRACT)
    cases = {
        "duplicate": (raw[:-1] + ', "issue": 91}', "repeats the key"),
        "nan": (raw.replace("1e-37", "NaN", 1), "non-JSON constant"),
        "infinity": (raw.replace("1e-37", "Infinity", 1), "non-JSON constant"),
        "truncated": (raw[: len(raw) // 2], "missing or not valid JSON"),
        "array": ("[]", "field set drifted"),
        "not-utf8": (None, "missing or not valid JSON"),
    }
    for label, (text, message) in cases.items():
        payload = b"\xff\xfe" if text is None else text.encode("utf-8")
        with pytest.raises(AirNetworkAnalyticError, match=message):
            load_air_network_batched_contract(_write_contract(tmp_path / label, payload))
    with pytest.raises(AirNetworkAnalyticError, match="missing or not valid JSON"):
        load_air_network_batched_contract(tmp_path / "does-not-exist")


def test_contract_validation_fails_closed_on_every_section() -> None:
    def mutated(mutation) -> dict:
        contract = copy.deepcopy(CONTRACT)
        mutation(contract)
        return contract

    fp32 = lambda c: c["tolerances"]["torch_fp32_vs_numpy_fp64"]  # noqa: E731
    band = lambda c: c["case_profiles"]["fixture_band"]["variables"]  # noqa: E731
    stress = lambda c: c["case_profiles"]["supported_domain_stress"]["variables"]  # noqa: E731
    mutations = (
        (lambda c: c.pop("non_claims"), "field set drifted"),
        (lambda c: c.update(extra=1), "field set drifted"),
        (lambda c: c.update(schema_version="v2"), "unsupported"),
        (lambda c: c.update(status="DRAFT"), "identity, status or issue"),
        (lambda c: c.update(issue=True), "identity, status or issue"),
        (lambda c: c.update(contract_id="other"), "identity, status or issue"),
        (lambda c: c.update(authority=[]), "authority field set"),
        (lambda c: c["authority"].update(actuator_authority=True), "must not grant any authority"),
        (lambda c: c["authority"].update(canonical_replay_or_hash_authority=1), "must not grant"),
        (lambda c: c["authority"].update(role="controller"), "must not grant any authority"),
        (lambda c: c["reference_solver"].pop("git_blob_sha1"), "reference solver field set"),
        (lambda c: c["reference_solver"].update(method=""), "non-empty string"),
        (lambda c: c.update(layout="flat"), "layout field set"),
        (lambda c: c["layout"]["branch_fields"].reverse(), "layout drifts"),
        (lambda c: c["layout"].update(host_dtype="float32"), "layout drifts"),
        (lambda c: c["supported_domain"].update(fan_speed_fraction=[0.0, 2.0]), "supported domain drifts"),
        (lambda c: c["supported_domain"].update(fan_speed_fraction="0..1"), "supported domain drifts"),
        (lambda c: c["supported_domain"].pop("air_density_kg_m3"), "supported domain field set"),
        (lambda c: c.update(branch_count_range=[1, 32]), "supported domain drifts"),
        (lambda c: c.update(branch_count_range=[1.0, 16.0]), "supported domain drifts"),
        (lambda c: c["comparison_scales"].pop("total_flow_m3_s"), "comparison scale set"),
        (lambda c: c["comparison_scales"].update(total_flow_m3_s=3), "non-empty string"),
        (lambda c: c.update(comparison_form=None), "non-empty string"),
        (lambda c: c["tolerances"].pop("torch_fp64_vs_numpy_fp64"), "tolerance set field set"),
        (lambda c: c["tolerances"].update(extra_comparison={}), "tolerance set field set"),
        (lambda c: fp32(c).pop("rationale"), "torch_fp32_vs_numpy_fp64 field set"),
        (lambda c: fp32(c)["variables"].pop("total_flow_m3_s"), "variable set"),
        (lambda c: fp32(c)["variables"]["total_flow_m3_s"].update(absolute=-1.0), r"must be in \[0, 1\)"),
        (lambda c: fp32(c)["variables"]["total_flow_m3_s"].update(absolute=1.0), r"must be in \[0, 1\)"),
        (lambda c: fp32(c)["variables"]["total_flow_m3_s"].update(absolute=True), "real number"),
        (lambda c: fp32(c)["variables"]["total_flow_m3_s"].update(absolute="0"), "real number"),
        (lambda c: fp32(c)["variables"]["total_flow_m3_s"].pop("absolute"), "field set drifted"),
        (lambda c: fp32(c).update(case_profile="unknown"), "undeclared case profile"),
        (lambda c: c["case_profiles"].pop("fixture_band"), "case profiles field set"),
        (lambda c: c["case_profiles"]["fixture_band"].pop("description"), "fixture_band field set"),
        (lambda c: band(c).pop("total_efficiency"), "variable set drifted"),
        (lambda c: band(c).update(total_efficiency=[0.5, 0.85]), "must be an object"),
        (lambda c: band(c)["total_efficiency"].pop("low"), "field set drifted"),
        (lambda c: band(c)["total_efficiency"].update(extra=1), "field set drifted"),
        (lambda c: band(c)["total_efficiency"].update(distribution="normal"), "unknown distribution"),
        (lambda c: band(c)["total_efficiency"].update(low="0.5"), "real number"),
        (lambda c: band(c)["total_efficiency"].update(low=True), "real number"),
        (lambda c: band(c)["total_efficiency"].update(low=0.9), "empty or outside"),
        (lambda c: band(c)["total_efficiency"].update(high=1.5), "empty or outside"),
        (lambda c: band(c)["total_efficiency"].update(exact_zero_probability=0.1), "special value is outside"),
        (lambda c: band(c)["fan_speed_fraction"].update(exact_zero_probability=0.7, exact_one_probability=0.7), "probabilities are invalid"),
        (lambda c: band(c)["fan_speed_fraction"].update(exact_zero_probability=-0.1), "probabilities are invalid"),
        (lambda c: stress(c)["air_density_kg_m3"].update(low_exponent=-3.0), "exponents must be integers"),
        (lambda c: stress(c)["air_density_kg_m3"].update(low_exponent=-40), "outside the decade table"),
        (lambda c: stress(c)["air_density_kg_m3"].update(high_exponent=-3), "outside the decade table"),
        (lambda c: stress(c)["air_density_kg_m3"].update(high_exponent=5), "empty or outside"),
        (lambda c: stress(c)["air_density_kg_m3"].update(exact_one_probability=0.1), None),
        (lambda c: stress(c)["rated_free_delivery_m3_s"].update(exact_zero_probability=0.1), "special value is outside"),
        (lambda c: c.update(non_claims=[]), "non-empty list"),
        (lambda c: c.update(non_claims=[1]), "non-empty string"),
    )
    for index, (mutation, message) in enumerate(mutations):
        if message is None:  # a legal edit: validation passes, only the pin protects it
            validate_air_network_batched_contract(mutated(mutation))
            continue
        with pytest.raises(AirNetworkAnalyticError, match=message):
            validate_air_network_batched_contract(mutated(mutation))
            pytest.fail(f"mutation {index} was accepted")
    with pytest.raises(AirNetworkAnalyticError, match="field set drifted"):
        validate_air_network_batched_contract([])


def test_reviewed_reference_solver_bytes_are_the_ones_the_contract_binds() -> None:
    raw = (REPO_ROOT / "src/aeolus/habitat_v2/air_network.py").read_bytes()
    normalised = raw.replace(b"\r\n", b"\n")
    blob = hashlib.sha1(f"blob {len(normalised)}\0".encode("ascii") + normalised).hexdigest()

    assert blob == CONTRACT["reference_solver"]["git_blob_sha1"]
    assert hashlib.sha256(normalised).hexdigest() == (
        CONTRACT["reference_solver"]["sha256_lf_normalised"]
    )


def test_candidate_has_no_plant_hmc_or_torch_dependency() -> None:
    source = inspect.getsource(analytic)
    for forbidden in ("from .hmc", "from .physics", "from .runner", "from .trace", "import torch"):
        assert forbidden not in source, forbidden

    script = (
        "import sys\n"
        "import aeolus.habitat_v2\n"
        "import aeolus.habitat_v2.air_network_analytic\n"
        "import aeolus.habitat_v2.air_network_batched_torch\n"
        "assert 'torch' not in sys.modules, 'core import pulled in torch'\n"
    )
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=REPO_ROOT, env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_comparison_scales_are_the_declared_physical_scales() -> None:
    batch = AirNetworkBatch(
        rated_free_delivery_m3_s=np.array([0.8, 2.0]),
        rated_shutoff_pressure_pa=np.array([1000.0, 50.0]),
        total_efficiency=np.array([0.5, 0.25]),
        shared_resistance_pa_s2_m6=np.array([1000.0, 10.0]),
        air_density_kg_m3=np.array([0.9, 1.2]),
        fan_speed_fraction=np.array([0.5, 0.1]),
        open_supply_resistance_pa_s2_m6=np.full((2, 3), 7000.0),
        return_resistance_pa_s2_m6=np.full((2, 3), 3000.0),
        damper_leak_fraction=np.full((2, 3), 0.08),
        damper_position_fraction=np.full((2, 3), 0.5),
    )
    scales = comparison_scales(batch)

    # Hand-computed: pressure P0*s^2, flow Q0*s, power P0*Q0*s^3 (/efficiency), mass rho*Q0*s.
    expected = {
        "fan_pressure_rise_pa": [250.0, 0.5],
        "shared_pressure_loss_pa": [250.0, 0.5],
        "branch_pressure_loss_pa": [250.0, 0.5],
        "operating_point_residual_pa": [250.0, 0.5],
        "total_flow_m3_s": [0.4, 0.2],
        "fan_air_power_w": [100.0, 0.1],
        "fan_electrical_power_w": [200.0, 0.4],
        "zone_flow_m3_s": [[0.4], [0.2]],
        "zone_mass_flow_kg_s": [[0.36], [0.24]],
    }
    assert set(scales) == set(expected) == set(analytic.COMPARISON_VARIABLES)
    for name, values in expected.items():
        np.testing.assert_allclose(scales[name], np.array(values), rtol=1e-14, err_msg=name)

    # A scale-relative tolerance must tighten with speed: an error of 1e-10 of
    # the unit-speed flow passes at full speed and fails at 1e-3 speed.
    scale_only = {
        name: {"absolute": 0.0, "relative_to_reference": 0.0, "relative_to_scale": 1e-9}
        for name in analytic.COMPARISON_VARIABLES
    }
    for speed, should_pass in ((1.0, True), (1e-3, False)):
        slowed = _replace(batch, fan_speed_fraction=np.full(2, speed))
        reference = solve_air_network_analytic(slowed)
        values = {name: getattr(reference, name) for name in analytic.COMPARISON_VARIABLES}
        values["total_flow_m3_s"] = values["total_flow_m3_s"] + 1e-10 * batch.rated_free_delivery_m3_s
        report = compare_results(AirNetworkBatchResult(**values), reference, slowed, scale_only)
        assert report["total_flow_m3_s"]["passed"] is should_pass, speed


def test_comparison_report_stays_serialisable_when_a_zero_allowance_is_violated() -> None:
    batch = _replace(_valid_batch(), fan_speed_fraction=np.zeros(4))
    reference = solve_air_network_analytic(batch)
    values = {name: getattr(reference, name) for name in analytic.COMPARISON_VARIABLES}
    values["total_flow_m3_s"] = values["total_flow_m3_s"] + 1e-6
    scale_only = {
        name: {"absolute": 0.0, "relative_to_reference": 0.0, "relative_to_scale": 1e-12}
        for name in analytic.COMPARISON_VARIABLES
    }
    report = compare_results(AirNetworkBatchResult(**values), reference, batch, scale_only)

    assert report["total_flow_m3_s"]["passed"] is False
    assert report["total_flow_m3_s"]["violation_count"] == 4
    assert report["total_flow_m3_s"]["max_error_over_allowed"] is None
    json.dumps(report, allow_nan=False)


def test_speed_scaling_keeps_pressure_accurate_when_speed_squared_is_subnormal() -> None:
    # s*s = 1e-312 is subnormal (about 1e-12 relative precision) while the
    # pressure P0*s^2 = 1e-305 is a normal number: scaling as (s*x)*s keeps it
    # accurate, scaling as x*(s*s) would not.
    speed = 1e-156
    batch = AirNetworkBatch(
        rated_free_delivery_m3_s=np.array([1e4]),
        rated_shutoff_pressure_pa=np.array([1e7]),
        total_efficiency=np.array([1.0]),
        shared_resistance_pa_s2_m6=np.array([1e9]),
        air_density_kg_m3=np.array([1.0]),
        fan_speed_fraction=np.array([1.0]),
        open_supply_resistance_pa_s2_m6=np.array([[1e9]]),
        return_resistance_pa_s2_m6=np.array([[1e9]]),
        damper_leak_fraction=np.array([[1.0]]),
        damper_position_fraction=np.array([[1.0]]),
    )
    unit = solve_air_network_analytic(batch)
    tiny = solve_air_network_analytic(_replace(batch, fan_speed_fraction=np.array([speed])))

    assert unit.fan_pressure_rise_pa[0] > 0.99e7
    for name in ("fan_pressure_rise_pa", "shared_pressure_loss_pa", "branch_pressure_loss_pa"):
        expected = (getattr(unit, name)[0] * speed) * speed
        assert expected > 1e-306, name
        assert getattr(tiny, name)[0] == pytest.approx(expected, rel=1e-14), name
    report = compare_results(
        tiny, _high_precision_root(_replace(batch, fan_speed_fraction=np.array([speed]))),
        _replace(batch, fan_speed_fraction=np.array([speed])),
        TOLERANCES["numpy_fp64_vs_high_precision_root"]["variables"],
    )
    assert _failures(report) == {}


def test_residual_tolerance_holds_at_subnormal_pressure_scale() -> None:
    batch = generate_air_network_cases(
        PROFILES["supported_domain_stress"], seed=9950, batch_size=48, branch_count=8
    )
    batch = _replace(batch, fan_speed_fraction=np.full(48, 1e-160))
    report = compare_results(
        solve_air_network_analytic(batch), _high_precision_root(batch), batch,
        TOLERANCES["numpy_fp64_vs_high_precision_root"]["variables"],
    )

    assert _failures(report) == {}


def test_unconverged_reference_bisection_off_the_fixture_band_is_reported() -> None:
    # Fan-dominated corner of the supported domain at full speed: the branch
    # pressure is about 8e-21 of shutoff pressure, which 100 bisection steps of
    # [0, shutoff] cannot resolve. The reviewed solver departs from the exact
    # root there; the candidate does not. Agreement with the reviewed solver is
    # therefore declared for the fixture band only.
    batch = AirNetworkBatch(
        rated_free_delivery_m3_s=np.array([1e-4]),
        rated_shutoff_pressure_pa=np.array([1e7]),
        total_efficiency=np.array([1e-3]),
        shared_resistance_pa_s2_m6=np.array([1e9]),
        air_density_kg_m3=np.array([1e-3]),
        fan_speed_fraction=np.array([1.0]),
        open_supply_resistance_pa_s2_m6=np.full((1, 16), 1e-3),
        return_resistance_pa_s2_m6=np.full((1, 16), 1e-3),
        damper_leak_fraction=np.full((1, 16), 1e-5),
        damper_position_fraction=np.full((1, 16), 1.0),
    )
    exact = _high_precision_root(batch)
    candidate = solve_air_network_analytic(batch)
    reference = _reference_batch_result(batch)
    assert exact.branch_pressure_loss_pa[0] / 1e7 < 1e-19

    assert _failures(
        compare_results(
            candidate, exact, batch, TOLERANCES["numpy_fp64_vs_high_precision_root"]["variables"]
        )
    ) == {}
    reference_vs_exact = compare_results(
        reference, exact, batch, TOLERANCES["numpy_fp64_vs_scalar_reference"]["variables"]
    )
    assert reference_vs_exact["fan_pressure_rise_pa"]["passed"] is False
    assert reference_vs_exact["total_flow_m3_s"]["passed"] is False
    assert reference_vs_exact["branch_pressure_loss_pa"]["passed"] is True
    assert abs(reference.operating_point_residual_pa[0]) > 1e-5
    assert abs(candidate.operating_point_residual_pa[0]) <= 1e-12 * 1e7
