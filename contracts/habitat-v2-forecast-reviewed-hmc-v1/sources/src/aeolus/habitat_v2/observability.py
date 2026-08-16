"""Closed operational projection for completed Habitat V2 V5 traces.

Only the declared, ordered operational feature manifest reaches qualification.
Raw rows are validated before projection; evaluator-only receipts never cross
this boundary.  A caller supplies a fixture identity explicitly because matched
counterparts intentionally share their scenario ``name``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .scenario import Scenario, TRACE_SCHEMA_VERSION_V5
from .trace import validate_trace_bytes

OPERATIONAL_FEATURE_MANIFEST_ID = "aeolus_habitat_v2_operational_feature_manifest_v1"
# The mutable literals exist only long enough to calculate the public immutable
# manifest identity. Consumers receive mapping proxies, so the published hash
# cannot drift after module import.
_FEATURE_DESCRIPTOR_DATA = (
    {"ordinal": 0, "path": "primary_telemetry.<zone_id>.{temperature_k,pressure_pa,co2_ppm,o2_mole_fraction,relative_humidity}", "source": "primary_sensor_observation", "units": "K|Pa|ppm|mole_fraction|fraction", "completed_timing": "completed_step", "decision_treatment": "deliberately_unscored"},
    {"ordinal": 1, "path": "secondary_telemetry.<zone_id>.{temperature_k,pressure_pa,co2_ppm,o2_mole_fraction,relative_humidity}", "source": "secondary_sensor_observation", "units": "K|Pa|ppm|mole_fraction|fraction", "completed_timing": "completed_step", "decision_treatment": "deliberately_unscored"},
    {"ordinal": 2, "path": "primary_minus_secondary.<zone_id>.{temperature_k,pressure_pa,co2_ppm,o2_mole_fraction,relative_humidity}", "source": "derived_sensor_disagreement", "units": "K|Pa|ppm|mole_fraction|fraction", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 3, "path": "commanded_action", "source": "completed_command_receipt", "units": "mixed_si", "completed_timing": "completed_step", "decision_treatment": "asserted_equal"},
    {"ordinal": 4, "path": "actual_action", "source": "achieved_actuator_receipt", "units": "mixed_si", "completed_timing": "completed_step", "decision_treatment": "deliberately_unscored"},
    {"ordinal": 5, "path": "operational_feedback.fan_speed_fraction", "source": "feedback_sensor", "units": "fraction", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 6, "path": "operational_feedback.fan_dc_bus_current_a", "source": "feedback_sensor", "units": "A", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 7, "path": "operational_feedback.damper_position_by_id.<component_id>", "source": "feedback_sensor", "units": "fraction", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 8, "path": "operational_feedback.branch_airflow_m3_s.<zone_id>", "source": "feedback_sensor", "units": "m3_s", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 9, "path": "operational_feedback.branch_differential_pressure_pa.<zone_id>", "source": "feedback_sensor", "units": "Pa", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 10, "path": "operational_feedback.{scrubber_capture_rate_mol_s,condenser_removal_rate_mol_s,oxygen_delivery_mol_s}", "source": "feedback_sensor", "units": "mol_s", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 11, "path": "operational_feedback.cooling_delivery_w.<zone_id>", "source": "feedback_sensor", "units": "W", "completed_timing": "completed_step", "decision_treatment": "compared"},
    {"ordinal": 12, "path": "operational_feedback.{battery_state_of_charge,oxygen_store_fraction,sorbent_remaining_fraction}", "source": "feedback_sensor", "units": "fraction", "completed_timing": "completed_step", "decision_treatment": "compared"},
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


OPERATIONAL_FEATURE_MANIFEST_SHA256 = hashlib.sha256(_canonical_bytes(_FEATURE_DESCRIPTOR_DATA)).hexdigest()
OPERATIONAL_FEATURE_MANIFEST = tuple(MappingProxyType(dict(item)) for item in _FEATURE_DESCRIPTOR_DATA)
_OPERATIONAL_FIELDS = frozenset({"step", "time_s", "mode", "primary_telemetry", "secondary_telemetry", "primary_minus_secondary", "commanded_action", "actual_action", "operational_feedback"})
_RAW_TRACE_CONSTRUCTION_TOKEN = object()


class OperationalProjectionError(ValueError):
    """Raised when an operational projection is not a closed typed V5 view."""


def _freeze_numeric_tree(value: Any, *, label: str) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise OperationalProjectionError(f"{label} keys must be strings")
        return MappingProxyType({key: _freeze_numeric_tree(value[key], label=f"{label}.{key}") for key in sorted(value)})
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperationalProjectionError(f"{label} must contain finite numeric values")
    number = float(value)
    if not math.isfinite(number):
        raise OperationalProjectionError(f"{label} must contain finite numeric values")
    return number


def _freeze_json_tree(value: Any, *, label: str) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise OperationalProjectionError(f"{label} keys must be strings")
        return MappingProxyType({key: _freeze_json_tree(value[key], label=f"{label}.{key}") for key in sorted(value)})
    if isinstance(value, list):
        return tuple(_freeze_json_tree(item, label=f"{label}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise OperationalProjectionError(f"{label} must contain finite JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    return value


@dataclass(frozen=True, init=False)
class RawV5Trace:
    """Validated raw V5 bytes; the only accepted projection input."""

    _rows: tuple[Mapping[str, Any], ...]
    fixture_id: str
    trace_sha256: str
    scenario_sha256: str
    run_id: str

    def __init__(
        self,
        rows: tuple[Mapping[str, Any], ...],
        fixture_id: str,
        trace_sha256: str,
        scenario_sha256: str,
        run_id: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _RAW_TRACE_CONSTRUCTION_TOKEN:
            raise TypeError("RawV5Trace must be constructed from validated trace bytes")
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "trace_sha256", trace_sha256)
        object.__setattr__(self, "scenario_sha256", scenario_sha256)
        object.__setattr__(self, "run_id", run_id)

    @classmethod
    def from_trace_bytes(cls, trace_bytes: bytes, *, scenario: Scenario, fixture_id: str) -> "RawV5Trace":
        if not isinstance(trace_bytes, bytes):
            raise OperationalProjectionError("trace bytes must be bytes")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise OperationalProjectionError("fixture_id must be a non-empty explicit string")
        if scenario.trace_schema_version != TRACE_SCHEMA_VERSION_V5:
            raise OperationalProjectionError("operational qualification requires a V5 scenario")
        rows = tuple(
            _freeze_json_tree(row, label=f"raw row {index}")
            for index, row in enumerate(validate_trace_bytes(trace_bytes, scenario=scenario))
        )
        return cls(
            rows,
            fixture_id,
            hashlib.sha256(trace_bytes).hexdigest(),
            scenario.scenario_sha256,
            scenario.run_id,
            _token=_RAW_TRACE_CONSTRUCTION_TOKEN,
        )


@dataclass(frozen=True)
class OperationalRow:
    """One completed V5 operational observation; no evaluator truth survives."""

    step: int
    time_s: float
    mode: str | None
    primary_telemetry: Mapping[str, Any]
    secondary_telemetry: Mapping[str, Any]
    primary_minus_secondary: Mapping[str, Any]
    commanded_action: Mapping[str, Any] | None
    actual_action: Mapping[str, Any]
    operational_feedback: Mapping[str, Any]

    def as_canonical_mapping(self) -> dict[str, Any]:
        return {"step": self.step, "time_s": self.time_s, "mode": self.mode, "primary_telemetry": _thaw(self.primary_telemetry), "secondary_telemetry": _thaw(self.secondary_telemetry), "primary_minus_secondary": _thaw(self.primary_minus_secondary), "commanded_action": _thaw(self.commanded_action) if self.commanded_action is not None else None, "actual_action": _thaw(self.actual_action), "operational_feedback": _thaw(self.operational_feedback)}

    @classmethod
    def from_canonical_mapping(cls, value: Any) -> "OperationalRow":
        if not isinstance(value, Mapping):
            raise OperationalProjectionError("operational row must be an object")
        unknown, missing = sorted(set(value) - _OPERATIONAL_FIELDS), sorted(_OPERATIONAL_FIELDS - set(value))
        if unknown or missing:
            raise OperationalProjectionError(f"operational row has unknown={unknown}, missing={missing}")
        step, time_s, mode, command = value["step"], value["time_s"], value["mode"], value["commanded_action"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise OperationalProjectionError("operational step must be a non-negative integer")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(float(time_s)):
            raise OperationalProjectionError("operational time must be finite")
        if mode is not None and not isinstance(mode, str):
            raise OperationalProjectionError("operational mode must be a string or null")
        if command is not None and not isinstance(command, Mapping):
            raise OperationalProjectionError("operational commanded action must be an object or null")
        maps = ("primary_telemetry", "secondary_telemetry", "primary_minus_secondary", "actual_action", "operational_feedback")
        if any(not isinstance(value[field], Mapping) for field in maps):
            raise OperationalProjectionError("operational feature fields must be objects")
        return cls(step, float(time_s), mode, _freeze_numeric_tree(value["primary_telemetry"], label="primary telemetry"), _freeze_numeric_tree(value["secondary_telemetry"], label="secondary telemetry"), _freeze_numeric_tree(value["primary_minus_secondary"], label="sensor disagreement"), None if command is None else _freeze_numeric_tree(command, label="commanded action"), _freeze_numeric_tree(value["actual_action"], label="actual action"), _freeze_numeric_tree(value["operational_feedback"], label="operational feedback"))


@dataclass(frozen=True)
class OperationalTrace:
    """Strict, closed sequence passed to the operational evaluator."""

    rows: tuple[OperationalRow, ...]
    feature_manifest_id: str = OPERATIONAL_FEATURE_MANIFEST_ID
    feature_manifest_sha256: str = OPERATIONAL_FEATURE_MANIFEST_SHA256
    fixture_id: str = ""
    source_trace_sha256: str = ""
    scenario_sha256: str = ""
    run_id: str = ""
    _validated_projection_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def is_validated_projection(self) -> bool:
        return self._validated_projection_token is _RAW_TRACE_CONSTRUCTION_TOKEN

    def __post_init__(self) -> None:
        if self.feature_manifest_id != OPERATIONAL_FEATURE_MANIFEST_ID or self.feature_manifest_sha256 != OPERATIONAL_FEATURE_MANIFEST_SHA256:
            raise OperationalProjectionError("unsupported operational feature manifest")
        if not self.fixture_id:
            raise OperationalProjectionError("operational trace requires explicit fixture_id")
        for label, value in (("source trace", self.source_trace_sha256), ("scenario", self.scenario_sha256), ("run", self.run_id)):
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise OperationalProjectionError(f"operational trace requires a 64-hex {label} provenance identity")
        if not self.rows:
            raise OperationalProjectionError("operational trace must contain rows")
        for expected, row in enumerate(self.rows):
            if row.step != expected:
                raise OperationalProjectionError("operational rows must have contiguous completed steps")

    @classmethod
    def from_canonical_rows(cls, rows: Sequence[Any], *, fixture_id: str = "unlabelled", source_trace_sha256: str = "", scenario_sha256: str = "", run_id: str = "") -> "OperationalTrace":
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise OperationalProjectionError("operational rows must be a sequence")
        return cls(tuple(OperationalRow.from_canonical_mapping(row) for row in rows), fixture_id=fixture_id, source_trace_sha256=source_trace_sha256, scenario_sha256=scenario_sha256, run_id=run_id)

    def as_canonical_mapping(self) -> dict[str, Any]:
        return {"feature_manifest_id": self.feature_manifest_id, "feature_manifest_sha256": self.feature_manifest_sha256, "fixture_id": self.fixture_id, "source_trace_sha256": self.source_trace_sha256, "scenario_sha256": self.scenario_sha256, "run_id": self.run_id, "rows": [row.as_canonical_mapping() for row in self.rows]}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.as_canonical_mapping())


def project_v5_trace(raw_trace: RawV5Trace) -> OperationalTrace:
    """Project a validated V5 trace into exactly the declared operational fields."""
    if not isinstance(raw_trace, RawV5Trace):
        raise OperationalProjectionError("project_v5_trace requires RawV5Trace, not a Mapping")
    rows: list[OperationalRow] = []
    for raw in raw_trace._rows:
        disagreement = raw["sensor_disagreement"]
        if not isinstance(disagreement, Mapping):
            raise OperationalProjectionError("V5 sensor disagreement must be an object")
        rows.append(OperationalRow.from_canonical_mapping({"step": raw["step"], "time_s": raw["time_s"], "mode": raw["applied_operating_mode"], "primary_telemetry": raw["telemetry"], "secondary_telemetry": {zone: disagreement[zone]["secondary"] for zone in sorted(disagreement)}, "primary_minus_secondary": {zone: disagreement[zone]["primary_minus_secondary"] for zone in sorted(disagreement)}, "commanded_action": raw["commanded_action"], "actual_action": raw["actual_action"], "operational_feedback": raw["operational_feedback"]}))
    return OperationalTrace(
        tuple(rows),
        fixture_id=raw_trace.fixture_id,
        source_trace_sha256=raw_trace.trace_sha256,
        scenario_sha256=raw_trace.scenario_sha256,
        run_id=raw_trace.run_id,
        _validated_projection_token=_RAW_TRACE_CONSTRUCTION_TOKEN,
    )
