"""Strict stateful forensics for historical healthy-alert episodes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from aeolus.trace import TickRecord

REPORT_FORMAT = "aeolus_healthy_alert_forensics_v1"
FORENSIC_EVIDENCE_ROLE = "historical_forensic_only"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_REPORT_KEYS = frozenset(
    {
        "format",
        "evidence_role",
        "source_commit",
        "source_manifest_sha256",
        "family_manifest_sha256",
        "method_name",
        "method_sha256",
        "input_row_count",
        "healthy_alert_episode_count",
        "episodes",
    }
)
_ROW_KEYS = frozenset(
    {
        "family_id",
        "scenario_role",
        "stream_id",
        "operating_profile_id",
        "start_tick",
        "end_tick",
        "context",
    }
)
_EPISODE_KEYS = frozenset(
    {
        "family_id",
        "stream_id",
        "operating_profile_id",
        "start_tick",
        "end_tick",
        "window_count",
        "predicted_labels",
        "start_context",
        "end_context",
    }
)


def build_alert_forensics_report(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    source_commit: str,
    source_manifest_sha256: str,
    family_manifest_sha256: str,
    method_name: str,
    method_sha256: str,
) -> dict[str, object]:
    """Build one immutable healthy-reference episode ledger from causal rows."""
    _require_commit(source_commit, "source commit")
    _require_sha256(source_manifest_sha256, "source manifest SHA-256")
    _require_sha256(family_manifest_sha256, "family manifest SHA-256")
    _require_non_empty_string(method_name, "method name")
    _require_sha256(method_sha256, "method SHA-256")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("forensic rows must be a sequence")
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise ValueError("forensic predictions must be a sequence")
    if len(rows) != len(predictions):
        raise ValueError("forensic rows and predictions must have equal length")

    episodes: list[dict[str, object]] = []
    current_stream: str | None = None
    last_end_tick: int | None = None
    open_episode: _OpenEpisode | None = None
    closed_streams: set[str] = set()

    for row_number, (row, prediction) in enumerate(zip(rows, predictions, strict=True), start=1):
        parsed = _parse_row(row, row_number)
        _require_non_empty_string(prediction, f"forensic prediction {row_number}")
        stream_id = parsed["stream_id"]
        if stream_id != current_stream:
            if open_episode is not None:
                episodes.append(open_episode.as_dict())
                open_episode = None
            if stream_id in closed_streams:
                raise ValueError("forensic stream rows must be contiguous")
            if current_stream is not None:
                closed_streams.add(current_stream)
            current_stream = stream_id
            last_end_tick = None
        if last_end_tick is not None and parsed["end_tick"] <= last_end_tick:
            raise ValueError("forensic stream rows must be strictly ordered by end_tick")
        last_end_tick = parsed["end_tick"]

        if prediction == "nominal":
            if open_episode is not None:
                episodes.append(open_episode.as_dict())
                open_episode = None
            continue
        if open_episode is None:
            open_episode = _OpenEpisode.from_row(parsed, prediction)
        else:
            open_episode.add(parsed, prediction)

    if open_episode is not None:
        episodes.append(open_episode.as_dict())

    report: dict[str, object] = {
        "format": REPORT_FORMAT,
        "evidence_role": FORENSIC_EVIDENCE_ROLE,
        "source_commit": source_commit,
        "source_manifest_sha256": source_manifest_sha256,
        "family_manifest_sha256": family_manifest_sha256,
        "method_name": method_name,
        "method_sha256": method_sha256,
        "input_row_count": len(rows),
        "healthy_alert_episode_count": len(episodes),
        "episodes": episodes,
    }
    validate_alert_forensics_report(report)
    return report



def summarize_forensic_window(records: Sequence[TickRecord]) -> dict[str, object]:
    """Summarise observable simulator telemetry for one historical window.

    This is a diagnostic projection, not a model input. In particular, it may
    retain simulator occupancy context but deliberately excludes plant/source
    mass and every hidden fault parameter.
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ValueError("forensic window requires non-empty trace records")
    if any(not isinstance(record, TickRecord) for record in records):
        raise ValueError("forensic window records must be TickRecord instances")
    ticks = [record.tick for record in records]
    if ticks != sorted(ticks) or len(set(ticks)) != len(ticks):
        raise ValueError("forensic window ticks must be strictly ordered")

    first, last = records[0], records[-1]
    if set(first.zones) != set(last.zones):
        raise ValueError("forensic window zone topology drifted")
    if set(first.actuators) != set(last.actuators):
        raise ValueError("forensic window actuator topology drifted")
    if set(first.connections) != set(last.connections):
        raise ValueError("forensic window connection topology drifted")

    zones = {
        zone_id: _zone_context(records, zone_id)
        for zone_id in sorted(first.zones)
    }
    actuators = {
        actuator_id: _actuator_context(records, actuator_id)
        for actuator_id in sorted(first.actuators)
    }
    connections = {
        connection_id: _connection_context(records, connection_id)
        for connection_id in sorted(first.connections)
    }
    return {
        "zones": zones,
        "actuators": actuators,
        "connections": connections,
        "system": {
            "capacity_scale_start": float(first.system["capacity_scale"]),
            "capacity_scale_end": float(last.system["capacity_scale"]),
            "capacity_scale_min": float(
                min(record.system["capacity_scale"] for record in records)
            ),
            "shared_airflow_capacity": float(last.system["shared_airflow_capacity"]),
            "total_requested_end": float(last.system["total_requested_airflow"]),
            "total_delivered_end": float(last.system["total_delivered_airflow"]),
        },
    }


def _zone_context(records: Sequence[TickRecord], zone_id: str) -> dict[str, float]:
    sensors = [record.zones[zone_id]["sensor_co2_concentration"] for record in records]
    occupancy = [record.zones[zone_id]["occupancy_multiplier"] for record in records]
    tail = sensors[-3:]
    return {
        "sensor_start": float(sensors[0]),
        "sensor_end": float(sensors[-1]),
        "sensor_delta": float(sensors[-1] - sensors[0]),
        "sensor_range": float(max(sensors) - min(sensors)),
        "sensor_tail_delta_3": float(tail[-1] - tail[0]),
        "sensor_tail_range_3": float(max(tail) - min(tail)),
        "occupancy_start": float(occupancy[0]),
        "occupancy_end": float(occupancy[-1]),
        "occupancy_delta": float(occupancy[-1] - occupancy[0]),
    }


def _actuator_context(records: Sequence[TickRecord], actuator_id: str) -> dict[str, float]:
    values = [record.actuators[actuator_id] for record in records]
    return {
        "setpoint_start": float(values[0]["setpoint"]),
        "setpoint_end": float(values[-1]["setpoint"]),
        "position_start": float(values[0]["actual_position"]),
        "position_end": float(values[-1]["actual_position"]),
        "tracking_residual_end": float(values[-1]["tracking_residual"]),
        "tracking_residual_peak": float(
            max(abs(value["tracking_residual"]) for value in values)
        ),
        "moving_tick_count": float(sum(value["moving"] > 0.0 for value in values)),
    }


def _connection_context(records: Sequence[TickRecord], connection_id: str) -> dict[str, float]:
    values = [record.connections[connection_id] for record in records]
    ratios = [
        value["airflow_residual"] / value["requested_airflow"]
        if value["requested_airflow"] > 0.0
        else 0.0
        for value in values
    ]
    return {
        "requested_end": float(values[-1]["requested_airflow"]),
        "delivered_end": float(values[-1]["delivered_airflow"]),
        "residual_end": float(values[-1]["airflow_residual"]),
        "residual_ratio_peak": float(max(ratios)),
    }


def canonical_alert_forensics_sha256(report: object) -> str:
    """Return the digest binding all validated episode context and receipts."""
    validate_alert_forensics_report(report)
    return hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()


def validate_alert_forensics_report(report: object) -> None:
    """Reject malformed, unauditable, or non-historical healthy-alert reports."""
    _require_finite_json(report, "forensic report")
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise ValueError("forensic report schema mismatch")
    if report["format"] != REPORT_FORMAT:
        raise ValueError("forensic report format is unsupported")
    if report["evidence_role"] != FORENSIC_EVIDENCE_ROLE:
        raise ValueError("forensic report evidence role is unsupported")
    _require_commit(report["source_commit"], "source commit")
    _require_sha256(report["source_manifest_sha256"], "source manifest SHA-256")
    _require_sha256(report["family_manifest_sha256"], "family manifest SHA-256")
    _require_non_empty_string(report["method_name"], "method name")
    _require_sha256(report["method_sha256"], "method SHA-256")
    if not _is_non_negative_int(report["input_row_count"]):
        raise ValueError("forensic input row count must be a non-negative integer")
    episodes = report["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("forensic episodes must be a list")
    if report["healthy_alert_episode_count"] != len(episodes):
        raise ValueError("forensic episode count is inconsistent")

    previous_identity: tuple[str, int, str] | None = None
    for episode in episodes:
        _validate_episode(episode)
        identity = (
            episode["stream_id"],
            episode["start_tick"],
            episode["family_id"],
        )
        if previous_identity is not None and identity <= previous_identity:
            raise ValueError("forensic episodes must be uniquely ordered")
        previous_identity = identity


class _OpenEpisode:
    def __init__(self, row: dict[str, Any], prediction: str) -> None:
        self.family_id = row["family_id"]
        self.stream_id = row["stream_id"]
        self.operating_profile_id = row["operating_profile_id"]
        self.start_tick = row["end_tick"]
        self.end_tick = row["end_tick"]
        self.window_count = 1
        self.predicted_labels = {prediction}
        self.start_context = row["context"]
        self.end_context = row["context"]

    @classmethod
    def from_row(cls, row: dict[str, Any], prediction: str) -> _OpenEpisode:
        return cls(row, prediction)

    def add(self, row: dict[str, Any], prediction: str) -> None:
        if (
            row["family_id"] != self.family_id
            or row["stream_id"] != self.stream_id
            or row["operating_profile_id"] != self.operating_profile_id
        ):
            raise ValueError("forensic episode cannot cross stream identity")
        self.end_tick = row["end_tick"]
        self.window_count += 1
        self.predicted_labels.add(prediction)
        self.end_context = row["context"]

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "stream_id": self.stream_id,
            "operating_profile_id": self.operating_profile_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "window_count": self.window_count,
            "predicted_labels": sorted(self.predicted_labels),
            "start_context": self.start_context,
            "end_context": self.end_context,
        }


def _parse_row(row: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
        raise ValueError(f"forensic row {row_number} schema mismatch")
    if row["scenario_role"] != "reference":
        raise ValueError("healthy-alert forensics accepts reference streams only")
    parsed = dict(row)
    for field in ("family_id", "stream_id", "operating_profile_id"):
        _require_non_empty_string(parsed[field], f"forensic row {row_number} {field}")
    for field in ("start_tick", "end_tick"):
        if not _is_positive_int(parsed[field]):
            raise ValueError(f"forensic row {row_number} {field} must be a positive integer")
    if parsed["start_tick"] > parsed["end_tick"]:
        raise ValueError(f"forensic row {row_number} tick range is invalid")
    if not isinstance(parsed["context"], Mapping):
        raise ValueError(f"forensic row {row_number} context must be an object")
    _require_finite_json(parsed["context"], f"forensic row {row_number} context")
    return parsed


def _validate_episode(episode: object) -> None:
    if not isinstance(episode, dict) or set(episode) != _EPISODE_KEYS:
        raise ValueError("forensic episode schema mismatch")
    for field in ("family_id", "stream_id", "operating_profile_id"):
        _require_non_empty_string(episode[field], f"forensic episode {field}")
    if not _is_positive_int(episode["start_tick"]) or not _is_positive_int(episode["end_tick"]):
        raise ValueError("forensic episode ticks must be positive integers")
    if episode["end_tick"] < episode["start_tick"]:
        raise ValueError("forensic episode tick range is invalid")
    if not _is_positive_int(episode["window_count"]):
        raise ValueError("forensic episode window count must be a positive integer")
    labels = episode["predicted_labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or labels != sorted(set(labels))
        or any(not isinstance(label, str) or not label or label == "nominal" for label in labels)
    ):
        raise ValueError("forensic episode predicted labels are malformed")
    for field in ("start_context", "end_context"):
        if not isinstance(episode[field], Mapping):
            raise ValueError(f"forensic episode {field} must be an object")
        _require_finite_json(episode[field], f"forensic episode {field}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _require_finite_json(value: object, description: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} must not contain non-finite numbers")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{description} keys must be strings")
            _require_finite_json(nested, description)
    elif isinstance(value, list):
        for nested in value:
            _require_finite_json(nested, description)
    elif value is None or isinstance(value, (str, bool, int, float)):
        return
    else:
        raise ValueError(f"{description} contains a non-JSON value")


def _require_sha256(value: object, description: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


def _require_commit(value: object, description: str) -> None:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase git commit")


def _require_non_empty_string(value: object, description: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
