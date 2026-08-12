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
) -> ActuatorAchievement:
    """Apply bounded response to cooling and oxygen command maps."""

    cooling_delta = float(cooling_slew_w_per_s) * float(dt_seconds)
    oxygen_delta = float(oxygen_slew_mol_s2) * float(dt_seconds)
    if not math.isfinite(cooling_delta) or not math.isfinite(oxygen_delta):
        raise ValueError("actuator slew limits must be finite")
    return ActuatorAchievement(
        cooling_removed_w={
            zone_id: _slew(
                float(current_cooling_removed_w[zone_id]),
                float(requested_cooling_removed_w[zone_id]),
                cooling_delta,
            )
            for zone_id in sorted(requested_cooling_removed_w)
        },
        oxygen_injection_mol_s={
            zone_id: _slew(
                float(current_oxygen_injection_mol_s[zone_id]),
                float(requested_oxygen_injection_mol_s[zone_id]),
                oxygen_delta,
            )
            for zone_id in sorted(requested_oxygen_injection_mol_s)
        },
    )
