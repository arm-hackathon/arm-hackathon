from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .scenario import (
    SCENARIO_SCHEMA_VERSION_V4,
    SCENARIO_SCHEMA_VERSION_V5,
    Scenario,
)


@dataclass(frozen=True)
class PhysicalFaultEffects:
    fan_speed_multiplier: float
    open_supply_resistance_multiplier_by_zone: Mapping[str, float]
    jammed_damper_ids: tuple[str, ...]
    active_faults: tuple[Mapping[str, Any], ...]
    scrubber_capture_multiplier: float = 1.0
    condenser_removal_multiplier: float = 1.0
    cooling_delivery_multiplier_by_zone: Mapping[str, float] = field(
        default_factory=dict
    )
    oxygen_delivery_multiplier_by_zone: Mapping[str, float] = field(
        default_factory=dict
    )


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
    if scenario.scenario_schema_version not in {
        SCENARIO_SCHEMA_VERSION_V4,
        SCENARIO_SCHEMA_VERSION_V5,
    }:
        return PhysicalFaultEffects(
            fan_speed_multiplier=1.0,
            open_supply_resistance_multiplier_by_zone={},
            jammed_damper_ids=(),
            active_faults=(),
        )

    fan_speed_multiplier = 1.0
    branch_multiplier_by_zone: dict[str, float] = {}
    jammed_damper_ids: list[str] = []
    active: list[Mapping[str, Any]] = []
    scrubber_multiplier = 1.0
    condenser_multiplier = 1.0
    cooling_multiplier_by_zone: dict[str, float] = {}
    oxygen_multiplier_by_zone: dict[str, float] = {}
    fan_id = str(scenario.data["air_network"]["fan"]["id"])
    for profile in scenario.data["fault_profiles"]:
        if not (
            int(profile["start_step"])
            <= emitted_step
            < int(profile["end_step"])
        ):
            continue
        if profile["type"] == "fan_speed_degradation":
            fan_speed_multiplier = _linear_profile_value(
                profile, emitted_step=emitted_step
            )
            active.append(
                {
                    "fault_id": str(profile["id"]),
                    "fault_type": "fan_speed_degradation",
                    "target_id": fan_id,
                    "effect_name": "fan_speed_multiplier",
                    "effect_value": fan_speed_multiplier,
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
        elif scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V5 and str(
            profile["type"]
        ) in {
            "scrubber_capture_degradation",
            "condenser_removal_degradation",
            "scrubber_effectiveness_degradation",
            "condenser_effectiveness_degradation",
            "cooling_delivery_degradation",
            "oxygen_delivery_degradation",
            "cooling_effectiveness_degradation",
            "oxygen_effectiveness_degradation",
        }:
            profile_type = str(profile["type"])
            multiplier = _linear_profile_value(profile, emitted_step=emitted_step)
            if profile_type.startswith("scrubber_"):
                scrubber_multiplier = multiplier
                active.append(
                    {
                        "fault_id": str(profile["id"]),
                        "fault_type": profile_type,
                        "target_id": "scrubber",
                        "effect_name": "capture_ability_multiplier",
                        "effect_value": multiplier,
                    }
                )
            elif profile_type.startswith("condenser_"):
                condenser_multiplier = multiplier
                active.append(
                    {
                        "fault_id": str(profile["id"]),
                        "fault_type": profile_type,
                        "target_id": "condenser",
                        "effect_name": "removal_ability_multiplier",
                        "effect_value": multiplier,
                    }
                )
            elif profile_type.startswith("cooling_"):
                zone_id = str(profile["zone_id"])
                cooling_multiplier_by_zone[zone_id] = multiplier
                active.append(
                    {
                        "fault_id": str(profile["id"]),
                        "fault_type": profile_type,
                        "target_id": zone_id,
                        "effect_name": "cooling_delivery_multiplier",
                        "effect_value": multiplier,
                    }
                )
            elif profile_type.startswith("oxygen_"):
                zone_id = str(profile["zone_id"])
                oxygen_multiplier_by_zone[zone_id] = multiplier
                active.append(
                    {
                        "fault_id": str(profile["id"]),
                        "fault_type": profile_type,
                        "target_id": zone_id,
                        "effect_name": "oxygen_delivery_multiplier",
                        "effect_value": multiplier,
                    }
                )
    return PhysicalFaultEffects(
        fan_speed_multiplier=fan_speed_multiplier,
        open_supply_resistance_multiplier_by_zone={
            zone_id: branch_multiplier_by_zone[zone_id]
            for zone_id in sorted(branch_multiplier_by_zone)
        },
        jammed_damper_ids=tuple(sorted(jammed_damper_ids)),
        active_faults=tuple(sorted(active, key=lambda item: str(item["fault_id"]))),
        scrubber_capture_multiplier=scrubber_multiplier,
        condenser_removal_multiplier=condenser_multiplier,
        cooling_delivery_multiplier_by_zone={
            zone_id: cooling_multiplier_by_zone[zone_id]
            for zone_id in sorted(cooling_multiplier_by_zone)
        },
        oxygen_delivery_multiplier_by_zone={
            zone_id: oxygen_multiplier_by_zone[zone_id]
            for zone_id in sorted(oxygen_multiplier_by_zone)
        },
    )
