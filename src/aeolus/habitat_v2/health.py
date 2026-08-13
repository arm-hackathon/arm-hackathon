from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .hmc_contract import HMCContract
from .instrumentation import (
    ChannelSample,
    OperationalMeasurement,
)
from .physics import CanonicalExternalCommand
from .scenario import Scenario
from .telemetry import (
    ENVIRONMENTAL_CHANNELS,
    OPERATIONAL_FEEDBACK_CHANNELS,
    ObservableTopology,
    derive_observable_topology,
)


class HealthReducerError(ValueError):
    """Raised when operational measurement evidence is structurally invalid."""


@dataclass(frozen=True, slots=True)
class OperationalAlarm:
    alarm_id: str
    family: str
    target: str
    severity: str
    lifecycle: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "alarm_id": self.alarm_id,
            "family": self.family,
            "target": self.target,
            "severity": self.severity,
            "lifecycle": self.lifecycle,
        }


@dataclass(frozen=True, slots=True)
class AlarmTrack:
    qualifying_rows: int
    clear_rows: int
    lifecycle: str | None


@dataclass(frozen=True, slots=True)
class HealthTracker:
    completed_step: int
    tracks: Mapping[str, AlarmTrack]

    @classmethod
    def initial(cls) -> HealthTracker:
        return cls(completed_step=-1, tracks=MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class HealthReduction:
    health_state: str
    alarms: tuple[OperationalAlarm, ...]
    tracker: HealthTracker


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
_RESOURCE_GAUGE_FAMILIES = {
    "battery_state_of_charge": "low_battery_gauge",
    "oxygen_store_fraction": "low_oxygen_store_gauge",
    "sorbent_remaining_fraction": "low_sorbent_gauge",
}


def _expected_environmental_ids(topology: ObservableTopology) -> tuple[str, ...]:
    return tuple(
        f"{zone_id}/{channel_id}"
        for zone_id in topology.zone_ids
        for channel_id in ENVIRONMENTAL_CHANNELS
    )


def _expected_feedback_ids(topology: ObservableTopology) -> tuple[str, ...]:
    damper_ids = tuple(damper_id for _, damper_id in topology.branch_pairs)
    result: list[str] = []
    for channel_id in OPERATIONAL_FEEDBACK_CHANNELS:
        if channel_id == "damper_position_by_id":
            result.extend(f"{channel_id}/{damper_id}" for damper_id in damper_ids)
        elif channel_id in {
            "branch_airflow_m3_s",
            "branch_differential_pressure_pa",
            "cooling_delivery_w",
            "oxygen_delivery_mol_s",
        }:
            result.extend(f"{channel_id}/{zone_id}" for zone_id in topology.zone_ids)
        else:
            result.append(channel_id)
    return tuple(result)


def _unit_for_environmental(descriptor_id: str) -> str:
    _, channel_id = descriptor_id.split("/", 1)
    return _ENVIRONMENTAL_UNITS[channel_id]


def _unit_for_feedback(descriptor_id: str) -> str:
    channel_id = descriptor_id.split("/", 1)[0]
    return _FEEDBACK_UNITS[channel_id]


def _validate_sample_array(
    samples: tuple[ChannelSample, ...],
    *,
    expected_ids: tuple[str, ...],
    unit_lookup: Any,
    label: str,
) -> None:
    if type(samples) is not tuple or any(
        type(sample) is not ChannelSample for sample in samples
    ):
        raise HealthReducerError(f"{label} must contain exact ChannelSample values")
    actual_ids = tuple(sample.descriptor_id for sample in samples)
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise HealthReducerError(f"{label} descriptor topology or order is invalid")
    for sample in samples:
        if sample.unit != unit_lookup(sample.descriptor_id):
            raise HealthReducerError(f"{label} descriptor unit is invalid")


def _validate_measurement(
    measurement: OperationalMeasurement,
    *,
    scenario: Scenario,
    contract: HMCContract,
    previous_tracker: HealthTracker,
    topology: ObservableTopology,
) -> None:
    if type(measurement) is not OperationalMeasurement:
        raise HealthReducerError(
            "health reducer requires the exact OperationalMeasurement"
        )
    if type(scenario) is not Scenario or type(contract) is not HMCContract:
        raise HealthReducerError("health reducer requires exact parsed inputs")
    if type(previous_tracker) is not HealthTracker:
        raise HealthReducerError("health reducer requires the exact HealthTracker")
    if measurement.completed_step != previous_tracker.completed_step + 1:
        raise HealthReducerError("health measurement does not follow the tracker")
    expected_environmental = _expected_environmental_ids(topology)
    expected_feedback = _expected_feedback_ids(topology)
    _validate_sample_array(
        measurement.primary,
        expected_ids=expected_environmental,
        unit_lookup=_unit_for_environmental,
        label="primary telemetry",
    )
    _validate_sample_array(
        measurement.secondary,
        expected_ids=expected_environmental,
        unit_lookup=_unit_for_environmental,
        label="secondary telemetry",
    )
    _validate_sample_array(
        measurement.primary_minus_secondary,
        expected_ids=expected_environmental,
        unit_lookup=_unit_for_environmental,
        label="primary-minus-secondary telemetry",
    )
    _validate_sample_array(
        measurement.operational_feedback,
        expected_ids=expected_feedback,
        unit_lookup=_unit_for_feedback,
        label="operational feedback",
    )
    for primary, secondary, disagreement in zip(
        measurement.primary,
        measurement.secondary,
        measurement.primary_minus_secondary,
        strict=True,
    ):
        if (
            primary.descriptor_id != secondary.descriptor_id
            or primary.descriptor_id != disagreement.descriptor_id
        ):
            raise HealthReducerError("environmental descriptor arrays are misaligned")
        if (
            primary.availability == "AVAILABLE"
            and secondary.availability == "AVAILABLE"
        ):
            if (
                disagreement.availability != "AVAILABLE"
                or abs(
                    float(disagreement.value)
                    - (float(primary.value) - float(secondary.value))
                )
                > 1e-12
            ):
                raise HealthReducerError("disagreement sample is inconsistent")
        elif disagreement.availability != "UNAVAILABLE" or (
            disagreement.unavailable_reason != "DEPENDENCY_UNAVAILABLE"
        ):
            raise HealthReducerError("unavailable disagreement is inconsistent")


def _unknown_alarms(
    measurement: OperationalMeasurement,
) -> tuple[OperationalAlarm, ...]:
    unavailable = {
        sample.descriptor_id
        for samples in (
            measurement.primary,
            measurement.secondary,
            measurement.primary_minus_secondary,
            measurement.operational_feedback,
        )
        for sample in samples
        if sample.availability == "UNAVAILABLE"
    }
    return tuple(
        OperationalAlarm(
            alarm_id=f"telemetry_unknown/{descriptor_id}/critical",
            family="telemetry_unknown",
            target=descriptor_id,
            severity="CRITICAL",
            lifecycle="ACTIVE",
        )
        for descriptor_id in sorted(unavailable)
    )


def _sample_values(
    samples: tuple[ChannelSample, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for sample in samples:
        if sample.availability != "AVAILABLE" or sample.value is None:
            continue
        values[sample.descriptor_id] = float(sample.value)
    return values


def _threshold_crossed(value: float, threshold: float, *, direction: str) -> bool:
    if direction == "HIGH":
        return value >= threshold
    if direction == "LOW":
        return value <= threshold
    raise HealthReducerError(f"unsupported threshold direction {direction}")


def _threshold_cleared(value: float, threshold: float, *, direction: str) -> bool:
    if direction == "HIGH":
        return value < threshold
    if direction == "LOW":
        return value > threshold
    raise HealthReducerError(f"unsupported threshold direction {direction}")


def _slew_expected(current: float, target: float, maximum_delta: float) -> float:
    difference = target - current
    if abs(difference) <= maximum_delta:
        return target
    return current + maximum_delta if difference > 0.0 else current - maximum_delta


def _advance_track(
    *,
    alarm_id: str,
    family: str,
    target: str,
    severity: str,
    enter: bool,
    clear: bool,
    previous: AlarmTrack | None,
    enter_persistence: int,
    clear_persistence: int,
) -> tuple[AlarmTrack | None, OperationalAlarm | None]:
    if previous is not None and previous.lifecycle == "CLEARED":
        previous = None
    if previous is None:
        if not enter:
            return None, None
        next_track = AlarmTrack(
            qualifying_rows=1,
            clear_rows=0,
            lifecycle="RAISED",
        )
    elif previous.lifecycle == "RAISED":
        if not enter:
            return None, None
        qualifying_rows = previous.qualifying_rows + 1
        next_track = AlarmTrack(
            qualifying_rows=qualifying_rows,
            clear_rows=0,
            lifecycle=("ACTIVE" if qualifying_rows >= enter_persistence else "RAISED"),
        )
    elif previous.lifecycle == "ACTIVE":
        clear_rows = previous.clear_rows + 1 if clear else 0
        next_track = AlarmTrack(
            qualifying_rows=previous.qualifying_rows,
            clear_rows=clear_rows,
            lifecycle=("CLEARED" if clear_rows >= clear_persistence else "ACTIVE"),
        )
    else:
        raise HealthReducerError(f"invalid prior alarm lifecycle for {alarm_id}")
    alarm = OperationalAlarm(
        alarm_id=alarm_id,
        family=family,
        target=target,
        severity=severity,
        lifecycle=str(next_track.lifecycle),
    )
    return next_track, alarm


def _alarm_sort_key(alarm: OperationalAlarm) -> tuple[str, str, int]:
    severity_rank = {"ADVISORY": 0, "WARNING": 1, "CRITICAL": 2}
    return (alarm.family, alarm.target, severity_rank[alarm.severity])


def _threshold_alarms(
    measurement: OperationalMeasurement,
    *,
    scenario: Scenario,
    contract: HMCContract,
    previous_tracker: HealthTracker,
    previous_measurement: OperationalMeasurement | None,
    last_final_command: CanonicalExternalCommand | None,
    topology: ObservableTopology,
) -> tuple[tuple[OperationalAlarm, ...], Mapping[str, AlarmTrack]]:
    primary = _sample_values(measurement.primary)
    secondary = _sample_values(measurement.secondary)
    disagreement = _sample_values(measurement.primary_minus_secondary)
    feedback = _sample_values(measurement.operational_feedback)
    policy = contract.data["health_policy"]
    persistence = int(policy["persistence_rows"])
    clear_persistence = int(policy["clear_persistence_rows"])
    next_tracks: dict[str, AlarmTrack] = {}
    alarms: list[OperationalAlarm] = []

    environmental = policy["environmental"]
    for family in sorted(environmental):
        rule = environmental[family]
        channel_id = str(rule["channel"])
        direction = str(rule["direction"])
        for zone_id in topology.zone_ids:
            descriptor_id = f"{zone_id}/{channel_id}"
            first = primary[descriptor_id]
            second = secondary[descriptor_id]
            for severity in ("WARNING", "CRITICAL"):
                severity_key = severity.lower()
                alarm_id = f"{family}/{zone_id}/{severity_key}"
                enter_threshold = float(rule[f"{severity_key}_enter"])
                clear_threshold = float(rule[f"{severity_key}_clear"])
                enter = _threshold_crossed(
                    first, enter_threshold, direction=direction
                ) and _threshold_crossed(second, enter_threshold, direction=direction)
                clear = _threshold_cleared(
                    first, clear_threshold, direction=direction
                ) and _threshold_cleared(second, clear_threshold, direction=direction)
                track, alarm = _advance_track(
                    alarm_id=alarm_id,
                    family=str(family),
                    target=zone_id,
                    severity=severity,
                    enter=enter,
                    clear=clear,
                    previous=previous_tracker.tracks.get(alarm_id),
                    enter_persistence=persistence,
                    clear_persistence=clear_persistence,
                )
                if track is not None:
                    next_tracks[alarm_id] = track
                if alarm is not None:
                    alarms.append(alarm)

    disagreement_policy = policy["disagreement"]
    for zone_id in topology.zone_ids:
        for channel_id in ENVIRONMENTAL_CHANNELS:
            descriptor_id = f"{zone_id}/{channel_id}"
            magnitude = abs(disagreement[descriptor_id])
            rule = disagreement_policy[channel_id]
            for severity in ("WARNING", "CRITICAL"):
                severity_key = severity.lower()
                alarm_id = f"sensor_disagreement/{descriptor_id}/{severity_key}"
                track, alarm = _advance_track(
                    alarm_id=alarm_id,
                    family="sensor_disagreement",
                    target=descriptor_id,
                    severity=severity,
                    enter=magnitude >= float(rule[f"{severity_key}_enter"]),
                    clear=magnitude < float(rule[f"{severity_key}_clear"]),
                    previous=previous_tracker.tracks.get(alarm_id),
                    enter_persistence=persistence,
                    clear_persistence=clear_persistence,
                )
                if track is not None:
                    next_tracks[alarm_id] = track
                if alarm is not None:
                    alarms.append(alarm)

    resource_rule = policy["resource_gauges"]
    for descriptor_id, family in _RESOURCE_GAUGE_FAMILIES.items():
        measured = feedback[descriptor_id]
        for severity in ("WARNING", "CRITICAL"):
            severity_key = severity.lower()
            alarm_id = f"{family}/{descriptor_id}/{severity_key}"
            track, alarm = _advance_track(
                alarm_id=alarm_id,
                family=family,
                target=descriptor_id,
                severity=severity,
                enter=measured <= float(resource_rule[f"{severity_key}_enter"]),
                clear=measured > float(resource_rule[f"{severity_key}_clear"]),
                previous=previous_tracker.tracks.get(alarm_id),
                enter_persistence=persistence,
                clear_persistence=clear_persistence,
            )
            if track is not None:
                next_tracks[alarm_id] = track
            if alarm is not None:
                alarms.append(alarm)

    if previous_measurement is not None and last_final_command is not None:
        previous_feedback = _sample_values(previous_measurement.operational_feedback)
        command = last_final_command.to_mapping()
        channel_id = "fan_speed_fraction"
        fan_id = str(scenario.data["air_network"]["fan"]["id"])
        target = f"{channel_id}/{fan_id}"
        maximum_delta = float(
            scenario.data["air_network"]["fan"]["speed_slew_fraction_per_s"]
        ) * float(scenario.data["dt_seconds"])
        expected = _slew_expected(
            previous_feedback[channel_id],
            float(command[channel_id]),
            maximum_delta,
        )
        error = abs(expected - feedback[channel_id])
        tracking_rule = policy["tracking"][channel_id]
        for severity in ("WARNING", "CRITICAL"):
            severity_key = severity.lower()
            alarm_id = f"actuator_tracking_failure/{target}/{severity_key}"
            track, alarm = _advance_track(
                alarm_id=alarm_id,
                family="actuator_tracking_failure",
                target=target,
                severity=severity,
                enter=error >= float(tracking_rule[f"{severity_key}_enter"]),
                clear=error < float(tracking_rule[f"{severity_key}_clear"]),
                previous=previous_tracker.tracks.get(alarm_id),
                enter_persistence=persistence,
                clear_persistence=clear_persistence,
            )
            if track is not None:
                next_tracks[alarm_id] = track
            if alarm is not None:
                alarms.append(alarm)

        channel_id = "damper_position_by_id"
        tracking_rule = policy["tracking"][channel_id]
        branch_by_damper = {
            str(branch["damper_id"]): branch
            for branch in scenario.data["air_network"]["branches"]
        }
        for _, damper_id in topology.branch_pairs:
            descriptor_id = f"{channel_id}/{damper_id}"
            maximum_delta = float(
                branch_by_damper[damper_id]["damper_slew_fraction_per_s"]
            ) * float(scenario.data["dt_seconds"])
            expected = _slew_expected(
                previous_feedback[descriptor_id],
                float(command["damper_position_by_id"][damper_id]),
                maximum_delta,
            )
            error = abs(expected - feedback[descriptor_id])
            target = descriptor_id
            for severity in ("WARNING", "CRITICAL"):
                severity_key = severity.lower()
                alarm_id = f"actuator_tracking_failure/{target}/{severity_key}"
                track, alarm = _advance_track(
                    alarm_id=alarm_id,
                    family="actuator_tracking_failure",
                    target=target,
                    severity=severity,
                    enter=error >= float(tracking_rule[f"{severity_key}_enter"]),
                    clear=error < float(tracking_rule[f"{severity_key}_clear"]),
                    previous=previous_tracker.tracks.get(alarm_id),
                    enter_persistence=persistence,
                    clear_persistence=clear_persistence,
                )
                if track is not None:
                    next_tracks[alarm_id] = track
                if alarm is not None:
                    alarms.append(alarm)

        channel_id = "cooling_delivery_w"
        tracking_rule = policy["tracking"][channel_id]
        cooling_delta = float(
            scenario.data["actuator_feedback"]["cooling_slew_w_per_s"]
        ) * float(scenario.data["dt_seconds"])
        for zone_id in topology.zone_ids:
            descriptor_id = f"{channel_id}/{zone_id}"
            expected = _slew_expected(
                previous_feedback[descriptor_id],
                float(command["cooling_removed_w"][zone_id]),
                cooling_delta,
            )
            error = abs(expected - feedback[descriptor_id])
            target = descriptor_id
            for severity in ("WARNING", "CRITICAL"):
                severity_key = severity.lower()
                enter_threshold = max(
                    float(tracking_rule[f"{severity_key}_enter"]),
                    float(tracking_rule[f"relative_{severity_key}"]) * abs(expected),
                )
                clear_threshold = max(
                    float(tracking_rule[f"{severity_key}_clear"]),
                    float(tracking_rule[f"relative_{severity_key}_clear"])
                    * abs(expected),
                )
                alarm_id = f"actuator_tracking_failure/{target}/{severity_key}"
                track, alarm = _advance_track(
                    alarm_id=alarm_id,
                    family="actuator_tracking_failure",
                    target=target,
                    severity=severity,
                    enter=error >= enter_threshold,
                    clear=error < clear_threshold,
                    previous=previous_tracker.tracks.get(alarm_id),
                    enter_persistence=persistence,
                    clear_persistence=clear_persistence,
                )
                if track is not None:
                    next_tracks[alarm_id] = track
                if alarm is not None:
                    alarms.append(alarm)
    return tuple(sorted(alarms, key=_alarm_sort_key)), MappingProxyType(next_tracks)


def _aggregate_health(alarms: tuple[OperationalAlarm, ...]) -> str:
    if any(alarm.family == "telemetry_unknown" for alarm in alarms):
        return "UNKNOWN"
    if any(
        alarm.severity == "CRITICAL" and alarm.lifecycle == "ACTIVE" for alarm in alarms
    ):
        return "CRITICAL"
    if any(alarm.lifecycle in {"RAISED", "ACTIVE"} for alarm in alarms):
        return "DEGRADED"
    return "NOMINAL"


def reduce_health(
    *,
    measurement: OperationalMeasurement,
    scenario: Scenario,
    contract: HMCContract,
    previous_tracker: HealthTracker,
    previous_measurement: OperationalMeasurement | None,
    last_final_command: CanonicalExternalCommand | None,
) -> HealthReduction:
    topology = derive_observable_topology(scenario)
    _validate_measurement(
        measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=previous_tracker,
        topology=topology,
    )
    if measurement.completed_step == 0:
        if previous_measurement is not None or last_final_command is not None:
            raise HealthReducerError(
                "reset health reduction must not receive prior measurement or command"
            )
    else:
        if type(previous_measurement) is not OperationalMeasurement:
            raise HealthReducerError(
                "completed health reduction requires prior OperationalMeasurement"
            )
        if previous_measurement.completed_step != measurement.completed_step - 1:
            raise HealthReducerError("prior health measurement has the wrong step")
        if type(last_final_command) is not CanonicalExternalCommand:
            raise HealthReducerError(
                "completed health reduction requires the exact last final command"
            )
    unknown = _unknown_alarms(measurement)
    tracker = HealthTracker(
        completed_step=measurement.completed_step,
        tracks=MappingProxyType(dict(previous_tracker.tracks)),
    )
    if unknown:
        return HealthReduction(
            health_state="UNKNOWN",
            alarms=unknown,
            tracker=tracker,
        )
    alarms, tracks = _threshold_alarms(
        measurement,
        scenario=scenario,
        contract=contract,
        previous_tracker=previous_tracker,
        previous_measurement=previous_measurement,
        last_final_command=last_final_command,
        topology=topology,
    )
    tracker = HealthTracker(
        completed_step=measurement.completed_step,
        tracks=tracks,
    )
    return HealthReduction(
        health_state=_aggregate_health(alarms),
        alarms=alarms,
        tracker=tracker,
    )
