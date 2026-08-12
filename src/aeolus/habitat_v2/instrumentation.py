"""Stateless deterministic operational feedback instrumentation for V5."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .scenario import SCENARIO_SCHEMA_VERSION_V5, Scenario
from .state import PlantState
from .telemetry import (
    ENVIRONMENTAL_CHANNELS,
    OPERATIONAL_FEEDBACK_CHANNELS,
    ObservableTopology,
    derive_observable_topology,
)

CHANNEL_SAMPLE_SCHEMA_V1 = "aeolus_habitat_v2_channel_sample_v1"
SENSOR_MEMORY_SCHEMA_V1 = "aeolus_habitat_v2_sensor_memory_v1"
OPERATIONAL_MEASUREMENT_SCHEMA_V1 = "aeolus_habitat_v2_operational_measurement_v1"
_MEASUREMENT_ISSUANCE_TOKEN = object()
_MISSING = object()
_DEPENDENCY_UNAVAILABLE = object()

_ENVIRONMENTAL_UNITS = {
    "temperature_k": "K",
    "pressure_pa": "Pa",
    "co2_ppm": "ppm",
    "o2_mole_fraction": "mole_fraction",
    "relative_humidity": "fraction",
}
_FEEDBACK_UNITS = {
    "fan_speed_fraction": "fraction",
    "fan_dc_bus_current_a": "A",
    "damper_position_by_id": "fraction",
    "branch_airflow_m3_s": "m3_s",
    "branch_differential_pressure_pa": "Pa",
    "scrubber_capture_rate_mol_s": "mol_s",
    "condenser_removal_rate_mol_s": "mol_s",
    "cooling_delivery_w": "W",
    "oxygen_delivery_mol_s": "mol_s",
    "battery_state_of_charge": "fraction",
    "oxygen_store_fraction": "fraction",
    "sorbent_remaining_fraction": "fraction",
}
_FEEDBACK_SCOPES = {
    "fan_speed_fraction": "scalar",
    "fan_dc_bus_current_a": "scalar",
    "damper_position_by_id": "damper",
    "branch_airflow_m3_s": "zone",
    "branch_differential_pressure_pa": "zone",
    "scrubber_capture_rate_mol_s": "scalar",
    "condenser_removal_rate_mol_s": "scalar",
    "cooling_delivery_w": "zone",
    "oxygen_delivery_mol_s": "zone",
    "battery_state_of_charge": "scalar",
    "oxygen_store_fraction": "scalar",
    "sorbent_remaining_fraction": "scalar",
}


class OperationalInstrumentationError(ValueError):
    """Raised when operational instrumentation violates its closed topology."""


class _FinalType:
    _sealed = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if any(getattr(base, "_sealed", False) for base in cls.__bases__):
            raise TypeError(f"{cls.__name__} cannot subclass a final issued type")
        cls._sealed = True


@dataclass(frozen=True, slots=True)
class ChannelSample(_FinalType):
    descriptor_id: str
    availability: str
    value: float | None
    unavailable_reason: str | None
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor_id, str) or not self.descriptor_id:
            raise OperationalInstrumentationError(
                "channel sample descriptor_id must be a non-empty string"
            )
        if not isinstance(self.unit, str) or not self.unit:
            raise OperationalInstrumentationError(
                "channel sample unit must be a non-empty string"
            )
        if self.availability == "AVAILABLE":
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
                or self.unavailable_reason is not None
            ):
                raise OperationalInstrumentationError(
                    "AVAILABLE channel samples require one finite value"
                )
            object.__setattr__(self, "value", float(self.value))
            return
        if self.availability == "UNAVAILABLE":
            if self.value is not None or self.unavailable_reason not in {
                "MISSING",
                "NON_FINITE",
                "MALFORMED",
                "DEPENDENCY_UNAVAILABLE",
            }:
                raise OperationalInstrumentationError(
                    "UNAVAILABLE channel samples require one closed reason"
                )
            return
        raise OperationalInstrumentationError(
            "channel availability must be AVAILABLE or UNAVAILABLE"
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "descriptor_id": self.descriptor_id,
            "availability": self.availability,
            "value": self.value,
            "unavailable_reason": self.unavailable_reason,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class SensorMemory(_FinalType):
    schema_version: str
    completed_step: int
    primary: tuple[ChannelSample, ...]
    secondary: tuple[ChannelSample, ...]
    operational_feedback: tuple[ChannelSample, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SENSOR_MEMORY_SCHEMA_V1:
            raise OperationalInstrumentationError("unsupported sensor memory schema")
        if (
            isinstance(self.completed_step, bool)
            or not isinstance(self.completed_step, int)
            or self.completed_step < 0
        ):
            raise OperationalInstrumentationError(
                "sensor memory completed step must be a non-negative integer"
            )
        for label, samples in (
            ("primary", self.primary),
            ("secondary", self.secondary),
            ("operational_feedback", self.operational_feedback),
        ):
            if type(samples) is not tuple or any(
                type(sample) is not ChannelSample for sample in samples
            ):
                raise OperationalInstrumentationError(
                    f"sensor memory {label} must contain exact ChannelSample values"
                )


@dataclass(frozen=True, init=False, slots=True)
class OperationalMeasurement(_FinalType):
    schema_version: str
    completed_step: int
    completed_time_s: float
    primary: tuple[ChannelSample, ...]
    secondary: tuple[ChannelSample, ...]
    primary_minus_secondary: tuple[ChannelSample, ...]
    operational_feedback: tuple[ChannelSample, ...]
    sensor_memory: SensorMemory

    def __init__(
        self,
        *,
        schema_version: str,
        completed_step: int,
        completed_time_s: float,
        primary: tuple[ChannelSample, ...],
        secondary: tuple[ChannelSample, ...],
        primary_minus_secondary: tuple[ChannelSample, ...],
        operational_feedback: tuple[ChannelSample, ...],
        sensor_memory: SensorMemory,
        _token: object | None = None,
    ) -> None:
        if _token is not _MEASUREMENT_ISSUANCE_TOKEN:
            raise TypeError(
                "OperationalMeasurement must be issued by operational instrumentation"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "completed_step", completed_step)
        object.__setattr__(self, "completed_time_s", completed_time_s)
        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "secondary", secondary)
        object.__setattr__(self, "primary_minus_secondary", primary_minus_secondary)
        object.__setattr__(self, "operational_feedback", operational_feedback)
        object.__setattr__(self, "sensor_memory", sensor_memory)
        if self.schema_version != OPERATIONAL_MEASUREMENT_SCHEMA_V1:
            raise OperationalInstrumentationError(
                "unsupported operational measurement schema"
            )
        if (
            isinstance(self.completed_step, bool)
            or not isinstance(self.completed_step, int)
            or self.completed_step < 0
            or isinstance(self.completed_time_s, bool)
            or not isinstance(self.completed_time_s, (int, float))
            or not math.isfinite(float(self.completed_time_s))
            or float(self.completed_time_s) < 0.0
        ):
            raise OperationalInstrumentationError(
                "operational measurement step/time is invalid"
            )
        object.__setattr__(self, "completed_time_s", float(self.completed_time_s))


def _sample_from_raw(
    descriptor_id: str,
    raw: Any,
    *,
    unit: str,
) -> ChannelSample:
    if raw is _MISSING:
        return ChannelSample(descriptor_id, "UNAVAILABLE", None, "MISSING", unit)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return ChannelSample(descriptor_id, "UNAVAILABLE", None, "MALFORMED", unit)
    value = float(raw)
    if not math.isfinite(value):
        return ChannelSample(descriptor_id, "UNAVAILABLE", None, "NON_FINITE", unit)
    return ChannelSample(descriptor_id, "AVAILABLE", value, None, unit)


def _dependency_sample(descriptor_id: str, *, unit: str) -> ChannelSample:
    return ChannelSample(
        descriptor_id,
        "UNAVAILABLE",
        None,
        "DEPENDENCY_UNAVAILABLE",
        unit,
    )


def _sample(*, seed: int, component_id: str, channel_id: str, step: int) -> float:
    payload = f"{seed}\0{component_id}\0{channel_id}\0{step}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 2.0 * integer / float(1 << 64) - 1.0


def _linear_bias(profile: Mapping[str, Any], *, step: int) -> float:
    start = int(profile["start_step"])
    end = int(profile["end_step"])
    if end - start == 1:
        return float(profile["end_bias"])
    progress = (step - start) / float(end - start - 1)
    return (
        float(profile["start_bias"])
        + (float(profile["end_bias"]) - float(profile["start_bias"])) * progress
    )


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


def _environmental_sensor_sample(
    *,
    seed: int,
    zone_id: str,
    sensor_head: str,
    channel_id: str,
    step: int,
) -> float:
    payload = f"{seed}\0{zone_id}\0{sensor_head}\0{channel_id}\0{step}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 2.0 * integer / float(1 << 64) - 1.0


def _clamp_environmental_value(channel_id: str, value: float) -> float:
    lower, upper = {
        "temperature_k": (0.0, math.inf),
        "pressure_pa": (0.0, math.inf),
        "co2_ppm": (0.0, 1_000_000.0),
        "o2_mole_fraction": (0.0, 1.0),
        "relative_humidity": (0.0, 1.0),
    }[channel_id]
    return min(upper, max(lower, value))


def _environmental_truth(
    scenario: Scenario,
    state: PlantState,
    topology: ObservableTopology,
) -> dict[str, dict[str, float]]:
    if set(state.zones) != set(topology.zone_ids):
        raise OperationalInstrumentationError(
            "plant-state zone topology does not match observable topology"
        )
    zone_config = {str(zone["id"]): zone for zone in scenario.data["zones"]}
    return {
        zone_id: {
            channel_id: float(
                state.zones[zone_id].telemetry(
                    volume_m3=float(zone_config[zone_id]["volume_m3"])
                )[channel_id]
            )
            for channel_id in ENVIRONMENTAL_CHANNELS
        }
        for zone_id in topology.zone_ids
    }


def _healthy_environmental_head(
    scenario: Scenario,
    truth: Mapping[str, Mapping[str, float]],
    *,
    sensor_head: str,
    step: int,
) -> dict[str, dict[str, float | object]]:
    model = scenario.data["sensor_model"]
    amplitudes = model[f"{sensor_head}_noise_amplitude"]
    seed = int(model["random_seed"])
    return {
        zone_id: {
            channel_id: float(truth[zone_id][channel_id])
            + float(amplitudes[channel_id])
            * _environmental_sensor_sample(
                seed=seed,
                zone_id=zone_id,
                sensor_head=sensor_head,
                channel_id=channel_id,
                step=step,
            )
            for channel_id in ENVIRONMENTAL_CHANNELS
        }
        for zone_id in sorted(truth)
    }


def _samples_by_id(samples: tuple[ChannelSample, ...]) -> dict[str, ChannelSample]:
    result = {sample.descriptor_id: sample for sample in samples}
    if len(result) != len(samples):
        raise OperationalInstrumentationError(
            "sensor memory contains duplicate descriptor IDs"
        )
    return result


def _apply_environmental_sensor_faults(
    scenario: Scenario,
    *,
    step: int,
    primary: dict[str, dict[str, float | object]],
    secondary: dict[str, dict[str, float | object]],
    previous_memory: SensorMemory | None,
) -> None:
    heads = {"primary": primary, "secondary": secondary}
    previous_heads = {
        "primary": (
            {} if previous_memory is None else _samples_by_id(previous_memory.primary)
        ),
        "secondary": (
            {} if previous_memory is None else _samples_by_id(previous_memory.secondary)
        ),
    }
    for profile in scenario.data["fault_profiles"]:
        profile_type = str(profile["type"])
        if profile_type not in {"sensor_bias_drift", "sensor_stuck"}:
            continue
        if not int(profile["start_step"]) <= step < int(profile["end_step"]):
            continue
        zone_id = str(profile["zone_id"])
        sensor_head = str(profile["sensor_head"])
        channel_id = str(profile["channel"])
        if profile_type == "sensor_bias_drift":
            current = heads[sensor_head][zone_id][channel_id]
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                heads[sensor_head][zone_id][channel_id] = _DEPENDENCY_UNAVAILABLE
            else:
                heads[sensor_head][zone_id][channel_id] = float(current) + _linear_bias(
                    profile, step=step
                )
            continue
        previous = previous_heads[sensor_head].get(f"{zone_id}/{channel_id}")
        if previous is None or previous.availability != "AVAILABLE":
            heads[sensor_head][zone_id][channel_id] = _DEPENDENCY_UNAVAILABLE
        else:
            heads[sensor_head][zone_id][channel_id] = float(previous.value)


def _environmental_samples(
    values: Mapping[str, Mapping[str, float | object]],
    topology: ObservableTopology,
) -> tuple[ChannelSample, ...]:
    samples: list[ChannelSample] = []
    for zone_id in topology.zone_ids:
        zone_values = values.get(zone_id)
        if zone_values is None:
            raise OperationalInstrumentationError(
                f"environmental instrument missing zone {zone_id}"
            )
        unknown = sorted(set(zone_values) - set(ENVIRONMENTAL_CHANNELS))
        if unknown:
            raise OperationalInstrumentationError(
                f"environmental instrument has unknown channels {unknown}"
            )
        for channel_id in ENVIRONMENTAL_CHANNELS:
            descriptor_id = f"{zone_id}/{channel_id}"
            raw = zone_values.get(channel_id, _MISSING)
            if raw is _DEPENDENCY_UNAVAILABLE:
                sample = _dependency_sample(
                    descriptor_id, unit=_ENVIRONMENTAL_UNITS[channel_id]
                )
            else:
                sample = _sample_from_raw(
                    descriptor_id,
                    raw,
                    unit=_ENVIRONMENTAL_UNITS[channel_id],
                )
                if sample.availability == "AVAILABLE":
                    sample = ChannelSample(
                        descriptor_id,
                        "AVAILABLE",
                        _clamp_environmental_value(channel_id, float(sample.value)),
                        None,
                        _ENVIRONMENTAL_UNITS[channel_id],
                    )
            samples.append(sample)
    return tuple(samples)


def _difference_samples(
    primary: tuple[ChannelSample, ...],
    secondary: tuple[ChannelSample, ...],
) -> tuple[ChannelSample, ...]:
    if tuple(sample.descriptor_id for sample in primary) != tuple(
        sample.descriptor_id for sample in secondary
    ):
        raise OperationalInstrumentationError(
            "primary and secondary descriptor ordering differs"
        )
    result: list[ChannelSample] = []
    for first, second in zip(primary, secondary, strict=True):
        if first.unit != second.unit:
            raise OperationalInstrumentationError(
                "primary and secondary descriptor units differ"
            )
        if first.availability != "AVAILABLE" or second.availability != "AVAILABLE":
            result.append(_dependency_sample(first.descriptor_id, unit=first.unit))
        else:
            result.append(
                ChannelSample(
                    first.descriptor_id,
                    "AVAILABLE",
                    float(first.value) - float(second.value),
                    None,
                    first.unit,
                )
            )
    return tuple(result)


def _feedback_resource_ids(
    topology: ObservableTopology,
    channel_id: str,
) -> tuple[str, ...] | None:
    scope = _FEEDBACK_SCOPES[channel_id]
    if scope == "scalar":
        return None
    if scope == "zone":
        return topology.zone_ids
    if scope == "damper":
        return tuple(damper_id for _, damper_id in topology.branch_pairs)
    raise OperationalInstrumentationError(f"unsupported feedback scope {scope}")


def _feedback_samples(
    raw_feedback: Mapping[str, object] | None,
    topology: ObservableTopology,
) -> tuple[ChannelSample, ...]:
    feedback: Mapping[str, object] = {} if raw_feedback is None else raw_feedback
    if not isinstance(feedback, Mapping) or any(
        not isinstance(key, str) for key in feedback
    ):
        raise OperationalInstrumentationError(
            "operational feedback must be a string-keyed object"
        )
    unknown = sorted(set(feedback) - set(OPERATIONAL_FEEDBACK_CHANNELS))
    if unknown:
        raise OperationalInstrumentationError(f"unknown feedback IDs {unknown}")
    samples: list[ChannelSample] = []
    for channel_id in OPERATIONAL_FEEDBACK_CHANNELS:
        unit = _FEEDBACK_UNITS[channel_id]
        resource_ids = _feedback_resource_ids(topology, channel_id)
        raw_channel = feedback.get(channel_id, _MISSING)
        if resource_ids is None:
            samples.append(_sample_from_raw(channel_id, raw_channel, unit=unit))
            continue
        if raw_channel is _MISSING:
            samples.extend(
                _sample_from_raw(f"{channel_id}/{resource_id}", _MISSING, unit=unit)
                for resource_id in resource_ids
            )
            continue
        if not isinstance(raw_channel, Mapping) or any(
            not isinstance(key, str) for key in raw_channel
        ):
            samples.extend(
                _sample_from_raw(f"{channel_id}/{resource_id}", raw_channel, unit=unit)
                for resource_id in resource_ids
            )
            continue
        unknown_resources = sorted(set(raw_channel) - set(resource_ids))
        if unknown_resources:
            raise OperationalInstrumentationError(
                f"unknown feedback resource IDs for {channel_id}: {unknown_resources}"
            )
        samples.extend(
            _sample_from_raw(
                f"{channel_id}/{resource_id}",
                raw_channel.get(resource_id, _MISSING),
                unit=unit,
            )
            for resource_id in resource_ids
        )
    return tuple(samples)


def _validate_previous_memory(
    previous_memory: SensorMemory | None,
    *,
    state_step: int,
) -> None:
    if state_step == 0:
        if previous_memory is not None:
            raise OperationalInstrumentationError(
                "reset measurement must not receive prior sensor memory"
            )
        return
    if type(previous_memory) is not SensorMemory:
        raise OperationalInstrumentationError(
            "completed measurement requires exact prior SensorMemory"
        )
    if previous_memory.completed_step != state_step - 1:
        raise OperationalInstrumentationError(
            "prior sensor memory does not precede the completed state"
        )


def instrument_v5_operational_measurement(
    scenario: Scenario,
    state: PlantState,
    previous_sensor_memory: SensorMemory | None,
) -> OperationalMeasurement:
    """Issue one topology-complete operational measurement for a completed V5 row."""

    if type(scenario) is not Scenario:
        raise OperationalInstrumentationError(
            "operational instrumentation requires the exact Scenario type"
        )
    if scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
        raise OperationalInstrumentationError(
            "operational instrumentation requires a V5 scenario"
        )
    if type(state) is not PlantState:
        raise OperationalInstrumentationError(
            "operational instrumentation requires the exact PlantState type"
        )
    _validate_previous_memory(previous_sensor_memory, state_step=state.step)
    topology = derive_observable_topology(scenario)
    truth = _environmental_truth(scenario, state, topology)
    primary_values = _healthy_environmental_head(
        scenario, truth, sensor_head="primary", step=state.step
    )
    secondary_values = _healthy_environmental_head(
        scenario, truth, sensor_head="secondary", step=state.step
    )
    _apply_environmental_sensor_faults(
        scenario,
        step=state.step,
        primary=primary_values,
        secondary=secondary_values,
        previous_memory=previous_sensor_memory,
    )
    primary = _environmental_samples(primary_values, topology)
    secondary = _environmental_samples(secondary_values, topology)
    disagreement = _difference_samples(primary, secondary)
    feedback = _feedback_samples(state.utility.last_operational_feedback, topology)
    memory = SensorMemory(
        schema_version=SENSOR_MEMORY_SCHEMA_V1,
        completed_step=state.step,
        primary=primary,
        secondary=secondary,
        operational_feedback=feedback,
    )
    return OperationalMeasurement(
        schema_version=OPERATIONAL_MEASUREMENT_SCHEMA_V1,
        completed_step=state.step,
        completed_time_s=state.step * float(scenario.data["dt_seconds"]),
        primary=primary,
        secondary=secondary,
        primary_minus_secondary=disagreement,
        operational_feedback=feedback,
        sensor_memory=memory,
        _token=_MEASUREMENT_ISSUANCE_TOKEN,
    )
