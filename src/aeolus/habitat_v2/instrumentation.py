"""Stateless deterministic operational feedback instrumentation for V5."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping


def _sample(*, seed: int, component_id: str, channel_id: str, step: int) -> float:
    payload = f"{seed}\0{component_id}\0{channel_id}\0{step}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 2.0 * integer / float(1 << 64) - 1.0


def _linear_bias(profile: Mapping[str, Any], *, step: int) -> float:
    start = int(profile["start_step"])
    end = int(profile["end_step"])
    if end - start == 1:
        return float(profile["end_bias"])
    progress = (step - start) / float(end - start - 1)
    return float(profile["start_bias"]) + (
        float(profile["end_bias"]) - float(profile["start_bias"])
    ) * progress


def _noise_amplitude(config: Mapping[str, Any], channel: str) -> float:
    value = config["feedback_sensor_noise_amplitude"]
    if isinstance(value, Mapping):
        return float(value[channel])
    return float(value)


def _bounds(channel: str) -> tuple[float, float]:
    if channel in {
        "fan_speed_fraction",
        "damper_position_by_id",
        "battery_state_of_charge",
        "oxygen_store_fraction",
        "sorbent_remaining_fraction",
    }:
        return 0.0, 1.0
    return 0.0, math.inf


def _target_value(
    truth: Mapping[str, Mapping[str, float] | float],
    channel: str,
    resource_id: str,
) -> float:
    value = truth[channel]
    if isinstance(value, Mapping):
        return float(value[resource_id])
    return float(value)


def _feedback_faults(
    scenario: Any,
    *,
    truth: Mapping[str, Mapping[str, float] | float],
    step: int,
    previous: Mapping[str, Any] | None,
) -> tuple[dict[str, Mapping[str, float] | float], list[Mapping[str, Any]]]:
    observations: dict[str, Mapping[str, float] | float] = {
        channel: (
            {key: float(value) for key, value in value.items()}
            if isinstance(value, Mapping)
            else float(value)
        )
        for channel, value in truth.items()
    }
    active: list[Mapping[str, Any]] = []
    for profile in scenario.data["fault_profiles"]:
        if profile["type"] not in {
            "feedback_sensor_bias_drift",
            "feedback_sensor_stuck",
        }:
            continue
        if not int(profile["start_step"]) <= step < int(profile["end_step"]):
            continue
        channel = str(profile["channel"])
        resource_id = str(profile["resource_id"])
        target_id = f"{channel}/{resource_id}"
        if profile["type"] == "feedback_sensor_bias_drift":
            bias = _linear_bias(profile, step=step)
            if isinstance(observations[channel], Mapping):
                observations[channel][resource_id] += bias  # type: ignore[index]
            else:
                observations[channel] = float(observations[channel]) + bias
            effect_name = "additive_feedback_sensor_bias"
            effect_value = bias
        else:
            if previous is None:
                raise ValueError("feedback sensor stuck fault requires prior feedback")
            held = _target_value(previous, channel, resource_id)
            if isinstance(observations[channel], Mapping):
                observations[channel][resource_id] = held  # type: ignore[index]
            else:
                observations[channel] = held
            effect_name = "held_feedback_sensor_observation"
            effect_value = held
        active.append(
            {
                "fault_id": str(profile["id"]),
                "fault_type": str(profile["type"]),
                "target_id": target_id,
                "effect_name": effect_name,
                "effect_value": effect_value,
            }
        )
    return observations, active


def _apply_noise(
    scenario: Any,
    values: Mapping[str, Mapping[str, float] | float],
    *,
    step: int,
) -> dict[str, Mapping[str, float] | float]:
    config = scenario.data["actuator_feedback"]
    seed = int(scenario.data["sensor_model"]["random_seed"])
    output: dict[str, Mapping[str, float] | float] = {}
    for channel, value in values.items():
        amplitude = _noise_amplitude(config, channel)
        if isinstance(value, Mapping):
            output[channel] = {
                resource_id: _clamp(
                    channel,
                    float(raw)
                    + amplitude
                    * _sample(
                        seed=seed,
                        component_id=str(resource_id),
                        channel_id=channel,
                        step=step,
                    ),
                )
                for resource_id, raw in sorted(value.items())
            }
        else:
            output[channel] = _clamp(
                channel,
                float(value)
                + amplitude
                * _sample(
                    seed=seed,
                    component_id=_scalar_resource_id(channel),
                    channel_id=channel,
                    step=step,
                ),
            )
    return output


def _scalar_resource_id(channel: str) -> str:
    return {
        "fan_speed_fraction": "primary_supply_fan",
        "fan_dc_bus_current_a": "primary_supply_fan",
        "scrubber_capture_rate_mol_s": "scrubber",
        "condenser_removal_rate_mol_s": "condenser",
        "battery_state_of_charge": "battery",
        "oxygen_store_fraction": "oxygen_store",
        "sorbent_remaining_fraction": "sorbent",
    }.get(channel, channel)


def _clamp(channel: str, value: float) -> float:
    lower, upper = _bounds(channel)
    return min(upper, max(lower, value))


def measure_operational_feedback(
    scenario: Any,
    *,
    truth: Mapping[str, Mapping[str, float] | float],
    step: int,
    previous: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Measure truth using deterministic noise and V5 feedback sensor faults."""

    noisy = _apply_noise(scenario, truth, step=step)
    measured, active = _feedback_faults(
        scenario,
        truth=noisy,
        step=step,
        previous=previous,
    )
    return {
        channel: (
            {
                resource_id: _clamp(channel, float(value))
                for resource_id, value in sorted(raw.items())
            }
            if isinstance(raw, Mapping)
            else _clamp(channel, float(raw))
        )
        for channel, raw in measured.items()
    }, tuple(sorted(active, key=lambda item: str(item["fault_id"])))


instrument_operational_feedback = measure_operational_feedback
