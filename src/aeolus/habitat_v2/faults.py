from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .scenario import SCENARIO_SCHEMA_VERSION_V4, Scenario


@dataclass(frozen=True)
class PhysicalFaultEffects:
    fan_speed_multiplier: float
    open_supply_resistance_multiplier_by_zone: Mapping[str, float]
    jammed_damper_ids: tuple[str, ...]
    active_faults: tuple[Mapping[str, Any], ...]


def _linear_profile_value(profile: Mapping[str, Any], *, emitted_step: int) -> float:
    start_step = int(profile["start_step"])
    end_step = int(profile["end_step"])
    start_value = float(profile["start_multiplier"])
    end_value = float(profile["end_multiplier"])
    count = end_step - start_step
    if count == 1:
        return end_value
    progress = (emitted_step - start_step) / float(count - 1)
    return start_value + (end_value - start_value) * progress


def physical_fault_effects(
    scenario: Scenario,
    *,
    emitted_step: int,
    previous_damper_position_by_id: Mapping[str, float],
) -> PhysicalFaultEffects:
    if scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V4:
        return PhysicalFaultEffects(
            fan_speed_multiplier=1.0,
            open_supply_resistance_multiplier_by_zone={},
            jammed_damper_ids=(),
            active_faults=(),
        )

    multiplier = 1.0
    branch_multiplier_by_zone: dict[str, float] = {}
    jammed_damper_ids: list[str] = []
    active: list[Mapping[str, Any]] = []
    fan_id = str(scenario.data["air_network"]["fan"]["id"])
    for profile in scenario.data["fault_profiles"]:
        if not (
            int(profile["start_step"])
            <= emitted_step
            < int(profile["end_step"])
        ):
            continue
        if profile["type"] == "fan_speed_degradation":
            multiplier = _linear_profile_value(profile, emitted_step=emitted_step)
            active.append(
                {
                    "fault_id": str(profile["id"]),
                    "fault_type": "fan_speed_degradation",
                    "target_id": fan_id,
                    "effect_name": "fan_speed_multiplier",
                    "effect_value": multiplier,
                }
            )
        elif profile["type"] == "branch_resistance_increase":
            zone_id = str(profile["zone_id"])
            branch_multiplier = _linear_profile_value(
                profile, emitted_step=emitted_step
            )
            branch_multiplier_by_zone[zone_id] = branch_multiplier
            active.append(
                {
                    "fault_id": str(profile["id"]),
                    "fault_type": "branch_resistance_increase",
                    "target_id": zone_id,
                    "effect_name": "open_supply_resistance_multiplier",
                    "effect_value": branch_multiplier,
                }
            )
        elif profile["type"] == "damper_jam":
            damper_id = str(profile["damper_id"])
            held_position = float(previous_damper_position_by_id[damper_id])
            jammed_damper_ids.append(damper_id)
            active.append(
                {
                    "fault_id": str(profile["id"]),
                    "fault_type": "damper_jam",
                    "target_id": damper_id,
                    "effect_name": "held_damper_position",
                    "effect_value": held_position,
                }
            )
    return PhysicalFaultEffects(
        fan_speed_multiplier=multiplier,
        open_supply_resistance_multiplier_by_zone={
            zone_id: branch_multiplier_by_zone[zone_id]
            for zone_id in sorted(branch_multiplier_by_zone)
        },
        jammed_damper_ids=tuple(sorted(jammed_damper_ids)),
        active_faults=tuple(sorted(active, key=lambda item: str(item["fault_id"]))),
    )
