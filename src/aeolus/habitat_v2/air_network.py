from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


class AirNetworkValidationError(ValueError):
    """Raised when an air-network specification or command is invalid."""


@dataclass(frozen=True)
class FanSpec:
    component_id: str
    rated_free_delivery_m3_s: float
    rated_shutoff_pressure_pa: float
    total_efficiency: float


@dataclass(frozen=True)
class BranchSpec:
    zone_id: str
    damper_id: str
    open_supply_resistance_pa_s2_m6: float
    return_resistance_pa_s2_m6: float
    damper_leak_fraction: float


@dataclass(frozen=True)
class AirNetworkSpec:
    fan: FanSpec
    shared_resistance_pa_s2_m6: float
    air_density_kg_m3: float
    branches: tuple[BranchSpec, ...]


@dataclass(frozen=True)
class AirNetworkResult:
    fan_pressure_rise_pa: float
    shared_pressure_loss_pa: float
    branch_pressure_loss_pa: Mapping[str, float]
    total_flow_m3_s: float
    zone_flow_m3_s: Mapping[str, float]
    zone_mass_flow_kg_s: Mapping[str, float]
    fan_air_power_w: float
    fan_electrical_power_w: float
    total_efficiency: float
    air_density_kg_m3: float
    operating_point_residual_pa: float
    mass_balance_residual_kg_s: Mapping[str, float]


def solve_air_network(
    spec: AirNetworkSpec,
    *,
    fan_speed_fraction: float,
    damper_position_by_id: Mapping[str, float],
) -> AirNetworkResult:
    if (
        isinstance(spec.air_density_kg_m3, bool)
        or not isinstance(spec.air_density_kg_m3, (int, float))
        or not math.isfinite(float(spec.air_density_kg_m3))
        or float(spec.air_density_kg_m3) <= 0.0
    ):
        raise AirNetworkValidationError(
            "air_density_kg_m3 must be finite and greater than zero"
        )

    fan = spec.fan
    if (
        isinstance(fan.rated_free_delivery_m3_s, bool)
        or not isinstance(fan.rated_free_delivery_m3_s, (int, float))
        or not math.isfinite(float(fan.rated_free_delivery_m3_s))
        or float(fan.rated_free_delivery_m3_s) <= 0.0
    ):
        raise AirNetworkValidationError(
            "rated_free_delivery_m3_s must be finite and greater than zero"
        )
    if (
        isinstance(fan.rated_shutoff_pressure_pa, bool)
        or not isinstance(fan.rated_shutoff_pressure_pa, (int, float))
        or not math.isfinite(float(fan.rated_shutoff_pressure_pa))
        or float(fan.rated_shutoff_pressure_pa) <= 0.0
    ):
        raise AirNetworkValidationError(
            "rated_shutoff_pressure_pa must be finite and greater than zero"
        )
    if (
        isinstance(fan.total_efficiency, bool)
        or not isinstance(fan.total_efficiency, (int, float))
        or not math.isfinite(float(fan.total_efficiency))
        or not 0.0 < float(fan.total_efficiency) <= 1.0
    ):
        raise AirNetworkValidationError(
            "total_efficiency must be finite and in (0, 1]"
        )
    if (
        isinstance(spec.shared_resistance_pa_s2_m6, bool)
        or not isinstance(spec.shared_resistance_pa_s2_m6, (int, float))
        or not math.isfinite(float(spec.shared_resistance_pa_s2_m6))
        or float(spec.shared_resistance_pa_s2_m6) < 0.0
    ):
        raise AirNetworkValidationError(
            "shared resistance must be finite and non-negative"
        )
    if not spec.branches:
        raise AirNetworkValidationError("air network must declare at least one branch")

    if (
        isinstance(fan_speed_fraction, bool)
        or not isinstance(fan_speed_fraction, (int, float))
        or not math.isfinite(float(fan_speed_fraction))
        or not 0.0 <= float(fan_speed_fraction) <= 1.0
    ):
        raise AirNetworkValidationError(
            "fan_speed_fraction must be finite and between 0 and 1"
        )

    branch_zone_ids = [branch.zone_id for branch in spec.branches]
    branch_damper_ids = [branch.damper_id for branch in spec.branches]
    if (
        len(set(branch_zone_ids)) != len(branch_zone_ids)
        or len(set(branch_damper_ids)) != len(branch_damper_ids)
    ):
        raise AirNetworkValidationError("branch zone and damper ids must be unique")
    for branch in spec.branches:
        if (
            not branch.zone_id
            or not branch.damper_id
            or isinstance(branch.open_supply_resistance_pa_s2_m6, bool)
            or not isinstance(
                branch.open_supply_resistance_pa_s2_m6, (int, float)
            )
            or not math.isfinite(float(branch.open_supply_resistance_pa_s2_m6))
            or float(branch.open_supply_resistance_pa_s2_m6) <= 0.0
            or isinstance(branch.return_resistance_pa_s2_m6, bool)
            or not isinstance(branch.return_resistance_pa_s2_m6, (int, float))
            or not math.isfinite(float(branch.return_resistance_pa_s2_m6))
            or float(branch.return_resistance_pa_s2_m6) <= 0.0
        ):
            raise AirNetworkValidationError(
                "branch ids must be non-empty and resistance values must be finite and positive"
            )
        if (
            isinstance(branch.damper_leak_fraction, bool)
            or not isinstance(branch.damper_leak_fraction, (int, float))
            or not math.isfinite(float(branch.damper_leak_fraction))
            or not 0.0 < float(branch.damper_leak_fraction) <= 1.0
        ):
            raise AirNetworkValidationError(
                "damper_leak_fraction must be finite and in (0, 1]"
            )

    expected_damper_ids = set(branch_damper_ids)
    if set(damper_position_by_id) != expected_damper_ids:
        raise AirNetworkValidationError(
            "damper command ids must exactly match configured damper ids"
        )
    for damper_id, position in damper_position_by_id.items():
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(float(position))
            or not 0.0 <= float(position) <= 1.0
        ):
            raise AirNetworkValidationError(
                f"damper position for {damper_id!r} must be finite and between 0 and 1"
            )

    zone_ids = tuple(sorted(branch.zone_id for branch in spec.branches))
    zeros = {zone_id: 0.0 for zone_id in zone_ids}
    if fan_speed_fraction == 0.0:
        return AirNetworkResult(
            fan_pressure_rise_pa=0.0,
            shared_pressure_loss_pa=0.0,
            branch_pressure_loss_pa=zeros,
            total_flow_m3_s=0.0,
            zone_flow_m3_s=zeros,
            zone_mass_flow_kg_s=zeros,
            fan_air_power_w=0.0,
            fan_electrical_power_w=0.0,
            total_efficiency=spec.fan.total_efficiency,
            air_density_kg_m3=spec.air_density_kg_m3,
            operating_point_residual_pa=0.0,
            mass_balance_residual_kg_s=zeros,
        )

    resistance_by_zone: dict[str, float] = {}
    for branch in spec.branches:
        position = float(damper_position_by_id[branch.damper_id])
        area_fraction = branch.damper_leak_fraction + (
            1.0 - branch.damper_leak_fraction
        ) * position
        resistance_by_zone[branch.zone_id] = (
            branch.open_supply_resistance_pa_s2_m6 / area_fraction**2
            + branch.return_resistance_pa_s2_m6
        )

    rated_free_flow = spec.fan.rated_free_delivery_m3_s
    free_flow = rated_free_flow * fan_speed_fraction
    shutoff_pressure = (
        spec.fan.rated_shutoff_pressure_pa * fan_speed_fraction**2
    )

    def flows_for_branch_pressure(branch_pressure_pa: float) -> dict[str, float]:
        return {
            zone_id: math.sqrt(branch_pressure_pa / resistance_by_zone[zone_id])
            for zone_id in zone_ids
        }

    def pressure_residual(branch_pressure_pa: float) -> float:
        total_flow = sum(flows_for_branch_pressure(branch_pressure_pa).values())
        normalized_flow = total_flow / free_flow
        fan_pressure = shutoff_pressure * max(0.0, 1.0 - normalized_flow**2)
        system_pressure = (
            spec.shared_resistance_pa_s2_m6 * total_flow**2
            + branch_pressure_pa
        )
        return fan_pressure - system_pressure

    lower_pressure = 0.0
    upper_pressure = shutoff_pressure
    for _ in range(100):
        branch_pressure = (lower_pressure + upper_pressure) / 2.0
        if pressure_residual(branch_pressure) > 0.0:
            lower_pressure = branch_pressure
        else:
            upper_pressure = branch_pressure
    branch_pressure = (lower_pressure + upper_pressure) / 2.0
    zone_flow = flows_for_branch_pressure(branch_pressure)
    total_flow = sum(zone_flow.values())
    shared_pressure = spec.shared_resistance_pa_s2_m6 * total_flow**2
    normalized_flow = total_flow / free_flow
    fan_pressure = shutoff_pressure * max(0.0, 1.0 - normalized_flow**2)
    fan_air_power = fan_pressure * total_flow

    return AirNetworkResult(
        fan_pressure_rise_pa=fan_pressure,
        shared_pressure_loss_pa=shared_pressure,
        branch_pressure_loss_pa={zone_id: branch_pressure for zone_id in zone_ids},
        total_flow_m3_s=total_flow,
        zone_flow_m3_s=zone_flow,
        zone_mass_flow_kg_s={
            zone_id: flow * spec.air_density_kg_m3
            for zone_id, flow in zone_flow.items()
        },
        fan_air_power_w=fan_air_power,
        fan_electrical_power_w=fan_air_power / spec.fan.total_efficiency,
        total_efficiency=spec.fan.total_efficiency,
        air_density_kg_m3=spec.air_density_kg_m3,
        operating_point_residual_pa=(
            fan_pressure - shared_pressure - branch_pressure
        ),
        mass_balance_residual_kg_s=zeros,
    )
