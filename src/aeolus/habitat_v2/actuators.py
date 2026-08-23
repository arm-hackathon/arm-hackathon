"""Deterministic requested-to-achieved actuator response for Habitat V2 V5."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class ActuatorAchievement:
    """The actuator state reached after policy and physical slew limits."""

    cooling_removed_w: Mapping[str, float]
    oxygen_injection_mol_s: Mapping[str, float]


def _slew(current: float, requested: float, maximum_delta: float) -> float:
    difference = requested - current
    if abs(difference) <= maximum_delta:
        return requested
    return current + math.copysign(maximum_delta, difference)


def achieve_actuator_state(
    *,
    current_cooling_removed_w: Mapping[str, float],
    current_oxygen_injection_mol_s: Mapping[str, float],
    requested_cooling_removed_w: Mapping[str, float],
    requested_oxygen_injection_mol_s: Mapping[str, float],
    cooling_slew_w_per_s: float,
    oxygen_slew_mol_s2: float,
    dt_seconds: float,
    oxygen_max_total_mol_s: float | None = None,
) -> ActuatorAchievement:
    """Apply bounded response to cooling and oxygen command maps."""

    cooling_delta = float(cooling_slew_w_per_s) * float(dt_seconds)
    oxygen_delta = float(oxygen_slew_mol_s2) * float(dt_seconds)
    if not math.isfinite(cooling_delta) or not math.isfinite(oxygen_delta):
        raise ValueError("actuator slew limits must be finite")
    cooling = {
        zone_id: _slew(
            float(current_cooling_removed_w[zone_id]),
            float(requested_cooling_removed_w[zone_id]),
            cooling_delta,
        )
        for zone_id in sorted(requested_cooling_removed_w)
    }
    oxygen = {
        zone_id: _slew(
            float(current_oxygen_injection_mol_s[zone_id]),
            float(requested_oxygen_injection_mol_s[zone_id]),
            oxygen_delta,
        )
        for zone_id in sorted(requested_oxygen_injection_mol_s)
    }
    if oxygen_max_total_mol_s is not None:
        capacity = float(oxygen_max_total_mol_s)
        if not math.isfinite(capacity) or capacity < 0.0:
            raise ValueError("oxygen capacity must be finite and non-negative")
        current = {
            zone_id: float(current_oxygen_injection_mol_s[zone_id])
            for zone_id in oxygen
        }
        total_current = math.fsum(current.values())
        if total_current > capacity and not math.isclose(
            total_current, capacity, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("current oxygen actuator state exceeds shared capacity")
        lower = {
            zone_id: max(0.0, current[zone_id] - oxygen_delta)
            for zone_id in oxygen
        }
        excess = math.fsum(oxygen.values()) - capacity
        if excess > 0.0:
            # Leave one representable float below the cap so callers using a
            # normal sum cannot observe a capacity overshoot from round-off.
            target = math.nextafter(capacity, -math.inf) if capacity > 0.0 else 0.0
            excess = math.fsum(oxygen.values()) - target
            for zone_id in sorted(
                oxygen,
                key=lambda item: (-oxygen[item], item),
            ):
                reduction = min(max(0.0, oxygen[zone_id] - lower[zone_id]), excess)
                oxygen[zone_id] -= reduction
                excess -= reduction
                if excess <= 0.0:
                    break
        total_achieved = math.fsum(oxygen.values())
        if total_achieved > capacity:
            raise ValueError("oxygen shared-capacity projection failed")
    return ActuatorAchievement(
        cooling_removed_w=cooling,
        oxygen_injection_mol_s=oxygen,
    )
