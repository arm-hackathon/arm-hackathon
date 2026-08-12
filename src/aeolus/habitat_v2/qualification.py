"""Matched operational observability qualification over provenance-bound V5 traces.

The evaluator consumes only an operational projection. Scenario identities bind a
pair to its traces, but scenario/fault truth is never inspected for a decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .observability import (
    OPERATIONAL_FEATURE_MANIFEST_ID,
    OPERATIONAL_FEATURE_MANIFEST_SHA256,
    OperationalTrace,
)
from .scenario import ACTUATOR_FEEDBACK_CONTRACT_REVISION_V1, Scenario

OBSERVABILITY_QUALIFICATION_ID = "aeolus_habitat_v2_observability_qualification_v1"
OBSERVABILITY_REPORT_ID = "aeolus_habitat_v2_observability_report_v1"
DECISION_CONTRACT_ID = "aeolus_habitat_v2_observability_decision_v1"
OUTCOMES = frozenset(("NO_OBSERVABLE_CONCERN", "ABNORMAL_OPERATION", "SUBSYSTEM_LOCALISED", "UNKNOWN"))
SUBSYSTEMS = frozenset(("air_network", "air_quality", "thermal_control", "oxygen_delivery", "co2_scrubbing", "humidity_condensation", "instrumentation", "resource_gauges", "UNKNOWN"))
WINDOW_STEPS = 2
PERSISTENCE_STEPS = WINDOW_STEPS

# Only descriptors declared ``compared`` below have finite paired tolerances.
# primary/secondary telemetry and actual action are deliberately unscored in a
# pair; commanded action is an exact structural assertion.
_CHANNEL_TOLERANCES = {
    "primary_minus_secondary.co2_ppm": 10.0,
    "primary_minus_secondary.temperature_k": 0.02,
    "primary_minus_secondary.pressure_pa": 12.0,
    "primary_minus_secondary.o2_mole_fraction": 0.00008,
    "primary_minus_secondary.relative_humidity": 0.004,
    "operational_feedback.fan_speed_fraction": 0.025,
    "operational_feedback.fan_dc_bus_current_a": 0.25,
    "operational_feedback.damper_position_by_id": 0.025,
    "operational_feedback.branch_airflow_m3_s": 0.001,
    "operational_feedback.branch_differential_pressure_pa": 3.0,
    "operational_feedback.scrubber_capture_rate_mol_s": 0.00005,
    "operational_feedback.condenser_removal_rate_mol_s": 0.00005,
    "operational_feedback.cooling_delivery_w": 15.0,
    "operational_feedback.oxygen_delivery_mol_s": 0.00005,
    "operational_feedback.battery_state_of_charge": 0.01,
    "operational_feedback.oxygen_store_fraction": 0.01,
    "operational_feedback.sorbent_remaining_fraction": 0.01,
}
_OPERATIONAL_DECISION_BOUNDARIES = {
    "primary_telemetry.co2_ppm": (0.0, 2_000.0),
    "primary_telemetry.temperature_k": (280.0, 310.0),
    "primary_telemetry.pressure_pa": (50_000.0, 120_000.0),
    "primary_telemetry.o2_mole_fraction": (0.19, 0.35),
    "primary_telemetry.relative_humidity": (0.10, 0.80),
}
_PAIR_MANIFEST_CONSTRUCTION_TOKEN = object()


class PairValidationError(ValueError):
    """Raised when nominal and treatment scenarios are not an exact match."""


class QualificationError(ValueError):
    """Raised for invalid evaluator input or result identity substitution."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


DECISION_TOLERANCE_CONTRACT_SHA256 = _sha(
    {"tolerances": _CHANNEL_TOLERANCES, "window_steps": WINDOW_STEPS}
)
OPERATIONAL_DECISION_BOUNDARY_SHA256 = _sha(
    {
        "decision_contract_id": DECISION_CONTRACT_ID,
        "boundaries": _OPERATIONAL_DECISION_BOUNDARIES,
        "window_steps": WINDOW_STEPS,
    }
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha256(value: Any, *, label: str, error_type: type[ValueError]) -> None:
    if not _is_sha256(value):
        raise error_type(f"{label} identity must be a non-empty 64-hex SHA-256")


def _scenario_from(value: Scenario | Mapping[str, Any]) -> Scenario:
    if isinstance(value, Scenario):
        return value
    if isinstance(value, Mapping):
        return Scenario.from_mapping(value)
    raise PairValidationError("pair members must be parsed Scenario values")


def _remove_declared_treatment_faults(data: Mapping[str, Any], treatment_fault_ids: Sequence[str]) -> dict[str, Any]:
    faults = data.get("fault_profiles")
    if not isinstance(faults, list):
        raise PairValidationError("V5 pair requires fault_profiles")
    declared = set(treatment_fault_ids)
    copied = dict(data)
    copied["fault_profiles"] = [profile for profile in faults if profile.get("id") not in declared]
    return copied


def _path_difference(left: Any, right: Any, path: str = "scenario") -> str | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return path
        for key in sorted(left):
            difference = _path_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _path_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else path


def _profile_interval(profiles: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    intervals = {(profile.get("start_step"), profile.get("end_step")) for profile in profiles}
    if len(intervals) != 1:
        raise PairValidationError("declared treatment profiles require a shared half-open interval")
    start_step, end_step = next(iter(intervals))
    if (
        isinstance(start_step, bool)
        or not isinstance(start_step, int)
        or isinstance(end_step, bool)
        or not isinstance(end_step, int)
        or start_step < 0
        or end_step <= start_step
    ):
        raise PairValidationError("treatment profiles require a valid half-open integer interval")
    return start_step, end_step


@dataclass(frozen=True)
class PairManifest:
    contract_id: str
    healthy_scenario_sha256: str
    fault_scenario_sha256: str
    healthy_run_id: str
    fault_run_id: str
    treatment_fault_ids: tuple[str, ...]
    treatment_fault_profiles_sha256: str
    treatment_start_step: int
    treatment_end_step: int
    structural_baseline_sha256: str
    actuator_feedback_contract_revision: str
    actuator_feedback_config_sha256: str
    operational_feature_manifest_id: str
    operational_feature_manifest_sha256: str
    decision_contract_id: str
    decision_tolerance_contract_sha256: str
    pair_manifest_sha256: str
    _validated_pair_token: object | None = dataclass_field(default=None, init=False, repr=False, compare=False)

    @property
    def is_validated_pair(self) -> bool:
        return self._validated_pair_token is _PAIR_MANIFEST_CONSTRUCTION_TOKEN

    def as_canonical_mapping(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name, definition in self.__dataclass_fields__.items()
            if definition.init
        }

    @classmethod
    def from_canonical_mapping(cls, value: Any) -> "PairManifest":
        expected_fields = {
            name for name, definition in cls.__dataclass_fields__.items() if definition.init
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise PairValidationError("pair manifest fields are closed")
        if value["contract_id"] != OBSERVABILITY_QUALIFICATION_ID:
            raise PairValidationError("unsupported qualification contract")
        if value["actuator_feedback_contract_revision"] != ACTUATOR_FEEDBACK_CONTRACT_REVISION_V1:
            raise PairValidationError("unsupported actuator feedback contract")
        if value["operational_feature_manifest_id"] != OPERATIONAL_FEATURE_MANIFEST_ID or value["operational_feature_manifest_sha256"] != OPERATIONAL_FEATURE_MANIFEST_SHA256:
            raise PairValidationError("unsupported feature manifest")
        if value["decision_contract_id"] != DECISION_CONTRACT_ID:
            raise PairValidationError("unsupported decision contract")
        if value["decision_tolerance_contract_sha256"] != DECISION_TOLERANCE_CONTRACT_SHA256:
            raise PairValidationError("unsupported decision tolerance contract")
        ids = value["treatment_fault_ids"]
        if not isinstance(ids, (list, tuple)) or not ids or not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids):
            raise PairValidationError("treatment fault ids must be explicit, non-empty, and unique")
        if (
            isinstance(value["treatment_start_step"], bool)
            or not isinstance(value["treatment_start_step"], int)
            or isinstance(value["treatment_end_step"], bool)
            or not isinstance(value["treatment_end_step"], int)
            or value["treatment_start_step"] < 0
            or value["treatment_end_step"] <= value["treatment_start_step"]
        ):
            raise PairValidationError("treatment interval must be a valid half-open integer interval")
        for field in ("healthy_scenario_sha256", "fault_scenario_sha256", "healthy_run_id", "fault_run_id", "treatment_fault_profiles_sha256", "structural_baseline_sha256", "actuator_feedback_config_sha256", "operational_feature_manifest_sha256", "decision_tolerance_contract_sha256", "pair_manifest_sha256"):
            _require_sha256(value[field], label=field, error_type=PairValidationError)
        content = {field: value[field] for field in expected_fields if field != "pair_manifest_sha256"}
        content["treatment_fault_ids"] = tuple(ids)
        if value["pair_manifest_sha256"] != _sha(content):
            raise PairValidationError("pair manifest identity does not bind contents")
        return cls(**content, pair_manifest_sha256=value["pair_manifest_sha256"])


def build_pair_manifest(*, healthy: Scenario | Mapping[str, Any], fault: Scenario | None = None, fault_mapping: Mapping[str, Any] | None = None, treatment_fault_ids: Sequence[str] | None = None) -> PairManifest:
    if (fault is None) == (fault_mapping is None):
        raise PairValidationError("provide exactly one parsed fault scenario or fault mapping")
    if treatment_fault_ids is None:
        raise PairValidationError("treatment fault ids must be explicit; inference from all profiles is forbidden")
    healthy_scenario = _scenario_from(healthy)
    try:
        fault_scenario = _scenario_from(fault if fault is not None else fault_mapping)  # type: ignore[arg-type]
    except ValueError as error:
        raise PairValidationError(f"structural invalid treatment scenario: {error}") from error
    profiles = fault_scenario.data.get("fault_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise PairValidationError("fault counterpart requires declared treatment fault profiles")
    profile_by_id = {profile.get("id"): profile for profile in profiles if isinstance(profile, Mapping) and isinstance(profile.get("id"), str)}
    ids = tuple(treatment_fault_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(item, str) or not item for item in ids) or set(ids) - set(profile_by_id):
        raise PairValidationError("treatment fault ids must explicitly and uniquely name fault profiles")
    treatment_profiles = [profile_by_id[fault_id] for fault_id in ids]
    treatment_start_step, treatment_end_step = _profile_interval(treatment_profiles)
    healthy_profiles = healthy_scenario.data.get("fault_profiles")
    if not isinstance(healthy_profiles, list) or any(profile.get("id") in set(ids) for profile in healthy_profiles):
        raise PairValidationError("healthy counterpart must not contain declared treatment faults")
    healthy_clean = _remove_declared_treatment_faults(healthy_scenario.data, ids)
    fault_clean = _remove_declared_treatment_faults(fault_scenario.data, ids)
    mismatch = _path_difference(healthy_clean, fault_clean)
    if mismatch is not None:
        raise PairValidationError(f"structural pair mismatch at {mismatch}; only declared treatment fault IDs may differ")
    content = {
        "contract_id": OBSERVABILITY_QUALIFICATION_ID,
        "healthy_scenario_sha256": healthy_scenario.scenario_sha256,
        "fault_scenario_sha256": fault_scenario.scenario_sha256,
        "healthy_run_id": healthy_scenario.run_id,
        "fault_run_id": fault_scenario.run_id,
        "treatment_fault_ids": ids,
        "treatment_fault_profiles_sha256": _sha(treatment_profiles),
        "treatment_start_step": treatment_start_step,
        "treatment_end_step": treatment_end_step,
        "structural_baseline_sha256": _sha(healthy_clean),
        "actuator_feedback_contract_revision": ACTUATOR_FEEDBACK_CONTRACT_REVISION_V1,
        "actuator_feedback_config_sha256": _sha(healthy_scenario.data["actuator_feedback"]),
        "operational_feature_manifest_id": OPERATIONAL_FEATURE_MANIFEST_ID,
        "operational_feature_manifest_sha256": OPERATIONAL_FEATURE_MANIFEST_SHA256,
        "decision_contract_id": DECISION_CONTRACT_ID,
        "decision_tolerance_contract_sha256": DECISION_TOLERANCE_CONTRACT_SHA256,
    }
    manifest = PairManifest(**content, pair_manifest_sha256=_sha(content))
    object.__setattr__(manifest, "_validated_pair_token", _PAIR_MANIFEST_CONSTRUCTION_TOKEN)
    return manifest


def _flatten(value: Mapping[str, Any], prefix: str) -> Iterable[tuple[str, float]]:
    for key in sorted(value):
        nested, path = value[key], f"{prefix}.{key}"
        if isinstance(nested, Mapping):
            yield from _flatten(nested, path)
        elif isinstance(nested, bool) or not isinstance(nested, (int, float)) or not math.isfinite(float(nested)):
            raise QualificationError(f"non-numeric operational feature at {path}")
        else:
            yield path, float(nested)


def tolerance_for_path(path: str) -> float:
    """Return finite tolerance for a declared compared feature, infinity otherwise."""
    parts = path.split(".")
    for stem, tolerance in _CHANNEL_TOLERANCES.items():
        stem_parts = stem.split(".")
        if parts == stem_parts or (len(parts) >= len(stem_parts) + 1 and parts[0] == stem_parts[0] and parts[-1] == stem_parts[-1] and stem_parts[0] == "primary_minus_secondary"):
            return tolerance
        if len(parts) >= len(stem_parts) and parts[:len(stem_parts)] == stem_parts:
            return tolerance
    return math.inf


def _subsystem(path: str) -> str:
    if path.startswith("primary_minus_secondary"):
        return "instrumentation"
    if ".fan_" in path or ".damper_" in path or ".branch_" in path:
        return "air_network"
    if ".cooling_" in path:
        return "thermal_control"
    if ".oxygen_" in path:
        return "oxygen_delivery"
    if ".scrubber_" in path or path.endswith(".co2_ppm"):
        return "co2_scrubbing"
    if ".condenser_" in path or path.endswith(".relative_humidity"):
        return "humidity_condensation"
    if any(token in path for token in ("battery", "sorbent", "store_fraction")):
        return "resource_gauges"
    return "UNKNOWN"


def _persistent_channels(
    divergent: Mapping[str, Sequence[int]],
    *,
    start_step: int | None = None,
    end_step: int | None = None,
) -> dict[str, int]:
    persistent: dict[str, int] = {}
    for path, steps in divergent.items():
        seen = set(steps)
        for start in steps:
            if start_step is not None and start < start_step:
                continue
            if end_step is not None and start + WINDOW_STEPS > end_step:
                continue
            if all(step in seen for step in range(start, start + WINDOW_STEPS)):
                persistent[path] = start
                break
    return persistent


def _phase_divergent_row_count(divergent_steps: set[int], bounds: tuple[int, int]) -> int:
    start_step, end_step = bounds
    return sum(step in divergent_steps for step in range(start_step, end_step))


def _first_clearance_decision(
    divergent_steps: set[int],
    *,
    treatment_end_step: int,
    row_count: int,
) -> int | None:
    for start_step in range(treatment_end_step, row_count - WINDOW_STEPS + 1):
        if all(step not in divergent_steps for step in range(start_step, start_step + WINDOW_STEPS)):
            return start_step + WINDOW_STEPS - 1
    return None


def _report_hash_content(report: "ObservabilityReport") -> dict[str, Any]:
    content = report.as_canonical_mapping()
    content.pop("report_sha256")
    return content


@dataclass(frozen=True)
class ObservabilityReport:
    report_contract_id: str
    decision_contract_id: str
    feature_manifest_id: str
    feature_manifest_sha256: str
    pair_manifest_sha256: str
    healthy_fixture_id: str
    fault_fixture_id: str
    healthy_run_id: str
    fault_run_id: str
    healthy_trace_sha256: str
    fault_trace_sha256: str
    treatment_start_step: int
    treatment_end_step: int
    window_steps: int
    persistence_steps: int
    step_duration_seconds: float
    tolerance_contract_sha256: str
    outcome: str
    abnormality_detected: bool
    localisation: str
    exact_identification: str
    first_divergence_step: int | None
    decision_step: int | None
    earliest_divergence_step: int | None
    detection_latency_steps: int | None
    detection_latency_seconds: float | None
    persistent_channels: tuple[str, ...]
    phase_bounds: Mapping[str, tuple[int, int]]
    phase_persistent_concern: Mapping[str, bool]
    phase_divergent_row_counts: Mapping[str, int]
    clearance_decision_step: int | None
    post_recovery_stable: bool
    report_sha256: str

    @property
    def subsystem(self) -> str:
        """Compatibility alias; localisation is the separately scoped answer."""
        return self.localisation

    def as_canonical_mapping(self) -> dict[str, Any]:
        value = {field: getattr(self, field) for field in self.__dataclass_fields__}
        value["phase_bounds"] = {
            phase: list(bounds) for phase, bounds in sorted(self.phase_bounds.items())
        }
        value["phase_persistent_concern"] = dict(sorted(self.phase_persistent_concern.items()))
        value["phase_divergent_row_counts"] = dict(sorted(self.phase_divergent_row_counts.items()))
        return value

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.as_canonical_mapping())

    @classmethod
    def from_canonical_mapping(cls, value: Any) -> "ObservabilityReport":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise QualificationError("report fields are closed")
        if value["report_contract_id"] != OBSERVABILITY_REPORT_ID or value["decision_contract_id"] != DECISION_CONTRACT_ID:
            raise QualificationError("unsupported report or decision contract")
        if value["window_steps"] != WINDOW_STEPS or value["persistence_steps"] != PERSISTENCE_STEPS:
            raise QualificationError("unsupported fixed report window contract")
        if value["tolerance_contract_sha256"] != DECISION_TOLERANCE_CONTRACT_SHA256:
            raise QualificationError("unsupported report tolerance contract")
        if value["feature_manifest_id"] != OPERATIONAL_FEATURE_MANIFEST_ID or value["feature_manifest_sha256"] != OPERATIONAL_FEATURE_MANIFEST_SHA256:
            raise QualificationError("unsupported feature manifest")
        for field in ("feature_manifest_sha256", "pair_manifest_sha256", "healthy_run_id", "fault_run_id", "healthy_trace_sha256", "fault_trace_sha256", "tolerance_contract_sha256", "report_sha256"):
            _require_sha256(value[field], label=field, error_type=QualificationError)
        if not all(isinstance(value[field], str) and value[field] for field in ("healthy_fixture_id", "fault_fixture_id")):
            raise QualificationError("report fixture identities must be non-empty strings")
        channels = value["persistent_channels"]
        if (
            not isinstance(channels, (list, tuple))
            or not all(isinstance(channel, str) and channel for channel in channels)
            or len(set(channels)) != len(channels)
            or tuple(channels) != tuple(sorted(channels))
        ):
            raise QualificationError("report persistent channels must be sorted unique strings")
        latency_seconds = value["detection_latency_seconds"]
        if latency_seconds is not None and (
            isinstance(latency_seconds, bool)
            or not isinstance(latency_seconds, (int, float))
            or not math.isfinite(float(latency_seconds))
            or latency_seconds < 0
        ):
            raise QualificationError("report latency seconds must be null or finite and non-negative")
        step_duration = value["step_duration_seconds"]
        if isinstance(step_duration, bool) or not isinstance(step_duration, (int, float)) or not math.isfinite(float(step_duration)) or step_duration <= 0:
            raise QualificationError("report step duration must be finite and positive")
        if not isinstance(value["post_recovery_stable"], bool):
            raise QualificationError("report stable-tail flag must be boolean")
        if value["outcome"] not in OUTCOMES or value["localisation"] not in SUBSYSTEMS or value["exact_identification"] != "UNKNOWN" or not isinstance(value["abnormality_detected"], bool):
            raise QualificationError("unsupported outcome or identification answer")
        if value["outcome"] == "NO_OBSERVABLE_CONCERN" and (value["abnormality_detected"] or value["localisation"] != "UNKNOWN"):
            raise QualificationError("NO_OBSERVABLE_CONCERN outcome must have no detection or localisation")
        if value["outcome"] == "SUBSYSTEM_LOCALISED" and (not value["abnormality_detected"] or value["localisation"] == "UNKNOWN"):
            raise QualificationError("localised outcome requires detected concern and known localisation")
        if value["outcome"] in {"ABNORMAL_OPERATION", "UNKNOWN"} and not value["abnormality_detected"]:
            raise QualificationError("abnormal or unknown outcome requires separately detected abnormality")
        first = value["first_divergence_step"]
        decision = value["decision_step"]
        earliest = value["earliest_divergence_step"]
        latency = value["detection_latency_steps"]
        if earliest != first:
            raise QualificationError("report earliest divergence contradicts first divergence")
        if value["abnormality_detected"]:
            if first is None or decision != first + value["window_steps"] - 1 or latency != decision - value["treatment_start_step"]:
                raise QualificationError("report decision and latency receipts are inconsistent")
            if value["detection_latency_seconds"] is None or not math.isclose(
                value["detection_latency_seconds"],
                latency * float(step_duration),
            ):
                raise QualificationError("report decision latency seconds are inconsistent")
        elif any(item is not None for item in (first, decision, latency, value["detection_latency_seconds"])):
            raise QualificationError("undetected report must have null decision and latency receipts")
        for field in ("treatment_start_step", "treatment_end_step", "window_steps", "persistence_steps"):
            if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
                raise QualificationError(f"{field} must be a non-negative integer")
        if value["treatment_end_step"] <= value["treatment_start_step"]:
            raise QualificationError("report treatment interval must be half-open and non-empty")
        for field in ("first_divergence_step", "decision_step", "earliest_divergence_step", "detection_latency_steps"):
            if value[field] is not None and (isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0):
                raise QualificationError(f"{field} must be null or a non-negative integer")
        phases = ("baseline", "treatment", "recovery", "post_recovery")
        bounds = value["phase_bounds"]
        concerns = value["phase_persistent_concern"]
        counts = value["phase_divergent_row_counts"]
        if not all(isinstance(item, Mapping) and set(item) == set(phases) for item in (bounds, concerns, counts)):
            raise QualificationError("report temporal phase fields are closed")
        frozen_bounds: dict[str, tuple[int, int]] = {}
        for phase in phases:
            interval = bounds[phase]
            if (
                not isinstance(interval, (list, tuple))
                or len(interval) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in interval)
                or interval[0] < 0
                or interval[1] < interval[0]
            ):
                raise QualificationError("report temporal phase bounds are invalid")
            frozen_bounds[phase] = (interval[0], interval[1])
            if not isinstance(concerns[phase], bool):
                raise QualificationError("report temporal concern flags must be boolean")
            if (
                isinstance(counts[phase], bool)
                or not isinstance(counts[phase], int)
                or not 0 <= counts[phase] <= interval[1] - interval[0]
            ):
                raise QualificationError("report temporal divergent-row counts are invalid")
        expected_bounds = {
            "baseline": (0, value["treatment_start_step"]),
            "treatment": (value["treatment_start_step"], value["treatment_end_step"]),
            "recovery": (value["treatment_end_step"], frozen_bounds["post_recovery"][0]),
            "post_recovery": frozen_bounds["post_recovery"],
        }
        if frozen_bounds != expected_bounds or frozen_bounds["post_recovery"][1] <= frozen_bounds["post_recovery"][0]:
            raise QualificationError("report temporal phase bounds are inconsistent")
        if concerns["baseline"] or counts["baseline"] != 0:
            raise QualificationError("report temporal baseline must match before treatment")
        if concerns["treatment"] is not value["abnormality_detected"]:
            raise QualificationError("report temporal treatment concern must match abnormality answer")
        if value["post_recovery_stable"] is not (counts["post_recovery"] == 0):
            raise QualificationError("report temporal stable-tail flag is inconsistent")
        clearance = value["clearance_decision_step"]
        if clearance is not None and (
            isinstance(clearance, bool)
            or not isinstance(clearance, int)
            or clearance < value["treatment_end_step"] + value["window_steps"] - 1
            or clearance >= frozen_bounds["post_recovery"][1]
        ):
            raise QualificationError("report temporal clearance decision is invalid")
        prepared = {field: value[field] for field in cls.__dataclass_fields__}
        prepared["phase_bounds"] = MappingProxyType(frozen_bounds)
        prepared["phase_persistent_concern"] = MappingProxyType(dict(concerns))
        prepared["phase_divergent_row_counts"] = MappingProxyType(dict(counts))
        report = cls(**prepared)
        if report.report_sha256 != _sha(_report_hash_content(report)):
            raise QualificationError("report identity does not bind contents")
        return report


def qualify_pair(healthy: OperationalTrace, fault: OperationalTrace, *, pair_manifest: PairManifest | None, treatment_start_step: int | None = None) -> ObservabilityReport:
    """Compare provenance-bound operational traces using a completed-row window."""
    if not isinstance(healthy, OperationalTrace) or not isinstance(fault, OperationalTrace):
        raise QualificationError("qualify_pair requires OperationalTrace values, not Mapping inputs")
    if not healthy.is_validated_projection or not fault.is_validated_projection:
        raise QualificationError("qualify_pair requires traces from validated V5 projection")
    if not isinstance(pair_manifest, PairManifest):
        raise QualificationError("qualify_pair requires a bound PairManifest")
    if not pair_manifest.is_validated_pair:
        raise QualificationError("qualify_pair requires a validated pair manifest")
    PairManifest.from_canonical_mapping(pair_manifest.as_canonical_mapping())
    if healthy.fixture_id == fault.fixture_id:
        raise QualificationError("matched fixture identities must be distinct")
    if healthy.scenario_sha256 != pair_manifest.healthy_scenario_sha256 or fault.scenario_sha256 != pair_manifest.fault_scenario_sha256:
        raise QualificationError("operational trace scenario provenance does not match pair manifest")
    if healthy.run_id != pair_manifest.healthy_run_id or fault.run_id != pair_manifest.fault_run_id:
        raise QualificationError("operational trace run provenance does not match pair manifest")
    if healthy.feature_manifest_sha256 != OPERATIONAL_FEATURE_MANIFEST_SHA256 or fault.feature_manifest_sha256 != OPERATIONAL_FEATURE_MANIFEST_SHA256:
        raise QualificationError("unsupported operational feature manifest")
    if len(healthy.rows) != len(fault.rows):
        raise QualificationError("matched operational traces must have equal row counts")
    if len(fault.rows) < 2:
        raise QualificationError("matched operational traces require at least two completed rows")
    step_duration_seconds = fault.rows[1].time_s - fault.rows[0].time_s
    if not math.isfinite(step_duration_seconds) or step_duration_seconds <= 0:
        raise QualificationError("matched operational traces require a finite positive step duration")
    for previous, current in zip(fault.rows, fault.rows[1:]):
        if not math.isclose(current.time_s - previous.time_s, step_duration_seconds):
            raise QualificationError("matched operational traces require a fixed step duration")
    start_step = pair_manifest.treatment_start_step if treatment_start_step is None else treatment_start_step
    end_step = pair_manifest.treatment_end_step
    if start_step != pair_manifest.treatment_start_step:
        raise QualificationError("latency start must be the manifest's completed treatment start")
    if end_step + WINDOW_STEPS * 2 > len(fault.rows):
        raise QualificationError("matched operational traces lack bounded recovery and post-recovery rows")
    divergent: dict[str, list[int]] = {}
    for healthy_row, fault_row in zip(healthy.rows, fault.rows):
        if healthy_row.step != fault_row.step or healthy_row.time_s != fault_row.time_s or healthy_row.mode != fault_row.mode:
            raise QualificationError("matched operational timing/mode must be equal")
        if healthy_row.commanded_action != fault_row.commanded_action:
            raise QualificationError("declared asserted-equal commanded action differs")
        for section in ("primary_minus_secondary", "operational_feedback"):
            healthy_features = dict(_flatten(getattr(healthy_row, section), section))
            fault_features = dict(_flatten(getattr(fault_row, section), section))
            if set(healthy_features) != set(fault_features):
                raise QualificationError("matched operational feature shapes must be equal")
            for path in sorted(healthy_features):
                if abs(healthy_features[path] - fault_features[path]) > tolerance_for_path(path):
                    divergent.setdefault(path, []).append(healthy_row.step)
    persistent = _persistent_channels(divergent, start_step=start_step, end_step=end_step)
    first = min(persistent.values()) if persistent else None
    decision_step = first + WINDOW_STEPS - 1 if first is not None else None
    channels = tuple(sorted(path for path, step in persistent.items() if step == first))
    divergent_steps = {step for steps in divergent.values() for step in steps}
    post_recovery_start = len(fault.rows) - WINDOW_STEPS
    phase_bounds = {
        "baseline": (0, start_step),
        "treatment": (start_step, end_step),
        "recovery": (end_step, post_recovery_start),
        "post_recovery": (post_recovery_start, len(fault.rows)),
    }
    phase_divergent_row_counts = {
        phase: _phase_divergent_row_count(divergent_steps, bounds)
        for phase, bounds in phase_bounds.items()
    }
    phase_persistent_concern = {
        phase: bool(
            _persistent_channels(
                divergent,
                start_step=bounds[0],
                end_step=bounds[1],
            )
        )
        for phase, bounds in phase_bounds.items()
    }
    clearance_decision_step = _first_clearance_decision(
        divergent_steps,
        treatment_end_step=end_step,
        row_count=len(fault.rows),
    )
    post_recovery_stable = phase_divergent_row_counts["post_recovery"] == 0
    if first is None:
        outcome, detected, localisation, latency_steps, latency_seconds = "NO_OBSERVABLE_CONCERN", False, "UNKNOWN", None, None
    else:
        candidates = {_subsystem(path) for path in channels}
        detected = True
        latency_steps = decision_step - start_step
        latency_seconds = latency_steps * step_duration_seconds
        if "UNKNOWN" in candidates or ("instrumentation" in candidates and len(candidates) > 1):
            outcome, localisation = "UNKNOWN", "UNKNOWN"
        elif len(candidates) == 1:
            outcome, localisation = "SUBSYSTEM_LOCALISED", next(iter(candidates))
        else:
            outcome, localisation = "ABNORMAL_OPERATION", "UNKNOWN"
    content = {
        "report_contract_id": OBSERVABILITY_REPORT_ID,
        "decision_contract_id": DECISION_CONTRACT_ID,
        "feature_manifest_id": OPERATIONAL_FEATURE_MANIFEST_ID,
        "feature_manifest_sha256": OPERATIONAL_FEATURE_MANIFEST_SHA256,
        "pair_manifest_sha256": pair_manifest.pair_manifest_sha256,
        "healthy_fixture_id": healthy.fixture_id,
        "fault_fixture_id": fault.fixture_id,
        "healthy_run_id": healthy.run_id,
        "fault_run_id": fault.run_id,
        "healthy_trace_sha256": healthy.source_trace_sha256,
        "fault_trace_sha256": fault.source_trace_sha256,
        "treatment_start_step": start_step,
        "treatment_end_step": end_step,
        "window_steps": WINDOW_STEPS,
        "persistence_steps": PERSISTENCE_STEPS,
        "step_duration_seconds": step_duration_seconds,
        "tolerance_contract_sha256": pair_manifest.decision_tolerance_contract_sha256,
        "outcome": outcome,
        "abnormality_detected": detected,
        "localisation": localisation,
        "exact_identification": "UNKNOWN",
        "first_divergence_step": first,
        "decision_step": decision_step,
        "earliest_divergence_step": first,
        "detection_latency_steps": latency_steps,
        "detection_latency_seconds": latency_seconds,
        "persistent_channels": channels,
        "phase_bounds": MappingProxyType(phase_bounds),
        "phase_persistent_concern": MappingProxyType(phase_persistent_concern),
        "phase_divergent_row_counts": MappingProxyType(phase_divergent_row_counts),
        "clearance_decision_step": clearance_decision_step,
        "post_recovery_stable": post_recovery_stable,
    }
    provisional = ObservabilityReport(**content, report_sha256="")
    final = ObservabilityReport(**content, report_sha256=_sha(_report_hash_content(provisional)))
    return ObservabilityReport.from_canonical_mapping(final.as_canonical_mapping())


@dataclass(frozen=True)
class HardNegativeResult:
    fixture_id: str
    source_trace_sha256: str
    scenario_sha256: str
    run_id: str
    feature_manifest_sha256: str
    decision_contract_id: str
    operational_decision_boundary_sha256: str
    outcome: str
    false_concern: bool
    persistent_channels: tuple[str, ...]
    result_sha256: str

    def as_canonical_mapping(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_canonical_mapping(cls, value: Any) -> "HardNegativeResult":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise QualificationError("hard-negative result fields are closed")
        if not isinstance(value["fixture_id"], str) or not value["fixture_id"]:
            raise QualificationError("hard-negative fixture identity must be non-empty")
        for field in (
            "source_trace_sha256",
            "scenario_sha256",
            "run_id",
            "feature_manifest_sha256",
            "operational_decision_boundary_sha256",
            "result_sha256",
        ):
            _require_sha256(value[field], label=field, error_type=QualificationError)
        if value["feature_manifest_sha256"] != OPERATIONAL_FEATURE_MANIFEST_SHA256:
            raise QualificationError("hard-negative feature manifest is unsupported")
        if value["decision_contract_id"] != DECISION_CONTRACT_ID:
            raise QualificationError("hard-negative decision contract is unsupported")
        if value["operational_decision_boundary_sha256"] != OPERATIONAL_DECISION_BOUNDARY_SHA256:
            raise QualificationError("hard-negative operational boundary contract is unsupported")
        if value["outcome"] not in {"NO_OBSERVABLE_CONCERN", "ABNORMAL_OPERATION"} or not isinstance(value["false_concern"], bool):
            raise QualificationError("hard-negative outcome is invalid")
        if (value["outcome"] == "ABNORMAL_OPERATION") is not value["false_concern"]:
            raise QualificationError("hard-negative outcome polarity is inconsistent")
        channels = value["persistent_channels"]
        if (
            not isinstance(channels, (list, tuple))
            or not all(isinstance(channel, str) and channel for channel in channels)
            or len(set(channels)) != len(channels)
            or tuple(channels) != tuple(sorted(channels))
        ):
            raise QualificationError("hard-negative persistent channels must be sorted unique strings")
        content = {field: value[field] for field in cls.__dataclass_fields__ if field != "result_sha256"}
        content["persistent_channels"] = tuple(channels)
        if value["result_sha256"] != _sha(content):
            raise QualificationError("hard-negative result identity does not bind contents")
        return cls(**content, result_sha256=value["result_sha256"])


def evaluate_hard_negative(trace: OperationalTrace) -> HardNegativeResult:
    """Evaluate a healthy trace against the explicitly narrower direct boundary."""
    if not isinstance(trace, OperationalTrace):
        raise QualificationError("hard negative requires OperationalTrace")
    if not trace.is_validated_projection:
        raise QualificationError("hard negative requires a trace from validated V5 projection")
    divergent: dict[str, list[int]] = {}
    for row in trace.rows:
        for path, number in _flatten(row.primary_telemetry, "primary_telemetry"):
            key = ".".join((path.split(".")[0], path.split(".")[-1]))
            boundary = _OPERATIONAL_DECISION_BOUNDARIES.get(key)
            if boundary is not None and not boundary[0] <= number <= boundary[1]:
                divergent.setdefault(path, []).append(row.step)
    channels = tuple(sorted(_persistent_channels(divergent)))
    content = {
        "fixture_id": trace.fixture_id,
        "source_trace_sha256": trace.source_trace_sha256,
        "scenario_sha256": trace.scenario_sha256,
        "run_id": trace.run_id,
        "feature_manifest_sha256": trace.feature_manifest_sha256,
        "decision_contract_id": DECISION_CONTRACT_ID,
        "operational_decision_boundary_sha256": OPERATIONAL_DECISION_BOUNDARY_SHA256,
        "outcome": "ABNORMAL_OPERATION" if channels else "NO_OBSERVABLE_CONCERN",
        "false_concern": bool(channels),
        "persistent_channels": channels,
    }
    final = HardNegativeResult(**content, result_sha256=_sha(content))
    return HardNegativeResult.from_canonical_mapping(final.as_canonical_mapping())


@dataclass(frozen=True)
class QualificationCase:
    report: ObservabilityReport
    expected_concern: bool
    expected_subsystem: str | None
    localisation_eligible: bool


@dataclass(frozen=True)
class AggregateMetrics:
    qualification_case_manifest_sha256: str
    hard_negative_manifest_sha256: str
    concern_coverage_numerator: int
    concern_coverage_denominator: int
    healthy_false_concern_count: int
    healthy_hard_negative_denominator: int
    eligible_localisation_numerator: int
    eligible_localisation_denominator: int
    exact_identification_numerator: int
    exact_identification_denominator: int
    latency_detected_count: int
    latency_null_non_detection_count: int
    ambiguous_abstention_numerator: int
    ambiguous_abstention_denominator: int
    overclaim_count: int
    overclaim_denominator: int
    aggregate_sha256: str

    def as_canonical_mapping(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_canonical_mapping(cls, value: Any) -> "AggregateMetrics":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise QualificationError("aggregate metric fields are closed")
        for field in cls.__dataclass_fields__:
            if field in {"aggregate_sha256", "qualification_case_manifest_sha256", "hard_negative_manifest_sha256"}:
                continue
            if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
                raise QualificationError("aggregate metrics must be non-negative integer counts")
        for numerator, denominator in (
            ("concern_coverage_numerator", "concern_coverage_denominator"),
            ("healthy_false_concern_count", "healthy_hard_negative_denominator"),
            ("eligible_localisation_numerator", "eligible_localisation_denominator"),
            ("exact_identification_numerator", "exact_identification_denominator"),
            ("ambiguous_abstention_numerator", "ambiguous_abstention_denominator"),
            ("overclaim_count", "overclaim_denominator"),
        ):
            if value[numerator] > value[denominator]:
                raise QualificationError("aggregate numerator exceeds its denominator")
        if value["latency_detected_count"] + value["latency_null_non_detection_count"] != value["concern_coverage_denominator"]:
            raise QualificationError("aggregate latency denominator is inconsistent")
        for field in ("qualification_case_manifest_sha256", "hard_negative_manifest_sha256", "aggregate_sha256"):
            _require_sha256(value[field], label=field, error_type=QualificationError)
        content = {field: value[field] for field in cls.__dataclass_fields__ if field != "aggregate_sha256"}
        if value["aggregate_sha256"] != _sha(content):
            raise QualificationError("aggregate identity does not bind contents")
        return cls(**content, aggregate_sha256=value["aggregate_sha256"])


def aggregate_qualification_metrics(cases: Sequence[QualificationCase], *, hard_negatives: Sequence[HardNegativeResult]) -> AggregateMetrics:
    for result in hard_negatives:
        HardNegativeResult.from_canonical_mapping(result.as_canonical_mapping())
    for case in cases:
        if not isinstance(case, QualificationCase) or not isinstance(case.report, ObservabilityReport):
            raise QualificationError("qualification case must contain a typed report")
        ObservabilityReport.from_canonical_mapping(case.report.as_canonical_mapping())
        if not isinstance(case.expected_concern, bool) or not isinstance(case.localisation_eligible, bool):
            raise QualificationError("qualification case polarity flags must be boolean")
        if case.expected_subsystem is not None and case.expected_subsystem not in SUBSYSTEMS - {"UNKNOWN"}:
            raise QualificationError("qualification case expected subsystem is invalid")
        if case.localisation_eligible is not (case.expected_subsystem is not None):
            raise QualificationError("qualification case localisation eligibility is inconsistent")
    concern_cases = [case for case in cases if case.expected_concern]
    eligible = [case for case in cases if case.localisation_eligible]
    ambiguous = [case for case in cases if case.expected_concern and not case.localisation_eligible]
    content = {
        "qualification_case_manifest_sha256": _sha(
            [
                {
                    "report_sha256": case.report.report_sha256,
                    "expected_concern": case.expected_concern,
                    "expected_subsystem": case.expected_subsystem,
                    "localisation_eligible": case.localisation_eligible,
                }
                for case in cases
            ]
        ),
        "hard_negative_manifest_sha256": _sha(
            [result.result_sha256 for result in hard_negatives]
        ),
        "concern_coverage_numerator": sum(case.report.abnormality_detected for case in concern_cases),
        "concern_coverage_denominator": len(concern_cases),
        "healthy_false_concern_count": sum(result.false_concern for result in hard_negatives),
        "healthy_hard_negative_denominator": len(hard_negatives),
        "eligible_localisation_numerator": sum(case.report.localisation == case.expected_subsystem and case.report.outcome == "SUBSYSTEM_LOCALISED" for case in eligible),
        "eligible_localisation_denominator": len(eligible),
        "exact_identification_numerator": 0,
        "exact_identification_denominator": 0,
        "latency_detected_count": sum(case.report.detection_latency_steps is not None for case in concern_cases),
        "latency_null_non_detection_count": sum(case.report.detection_latency_steps is None for case in concern_cases),
        "ambiguous_abstention_numerator": sum(case.report.outcome == "UNKNOWN" and case.report.abnormality_detected for case in ambiguous),
        "ambiguous_abstention_denominator": len(ambiguous),
        "overclaim_count": sum(case.report.outcome == "SUBSYSTEM_LOCALISED" and (not case.localisation_eligible or case.report.localisation != case.expected_subsystem) for case in cases),
        "overclaim_denominator": len(cases),
    }
    final = AggregateMetrics(**content, aggregate_sha256=_sha(content))
    return AggregateMetrics.from_canonical_mapping(final.as_canonical_mapping())
