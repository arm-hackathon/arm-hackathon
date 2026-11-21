"""Budgeted, development-only fault stress-search for Habitat Plant V2 (Issue #93).

This module is a research instrument.  It proposes physically bounded fault
scenarios on one declared development fixture, runs each through the
authoritative CPU plant under the deterministic HMC with no adviser, scores a
pre-declared objective from privileged simulator truth, and keeps a hash-chained
audit of every attempted candidate.

Boundaries that the code enforces rather than merely documents:

* Search outputs are biased development discoveries.  They are not IID samples,
  not a validation or test population and not evidence of generalisation.
* The harness never builds, reads or runs a BDM-v1 roster family and never
  opens the custody registry; candidates live in their own name, seed and
  nonce namespace on a fixture that differs from every roster scenario.
* Simulator truth is used only by the evaluator.  There is no adviser
  interface here: the HMC receives ``propose(None)`` on every cycle.
* No accelerated backend is accepted.  ``screening.backend`` must be
  ``"none"`` until a validated batched backend exists.
* Wall-clock values never enter a hashed payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any

import numpy as np

from .bdm_v1_families import (
    BdmV1FamilyError,
    DT_SECONDS,
    GENERATOR_FAULT_DURATION_RANGE,
    GENERATOR_FAULT_MULTIPLIER_BANDS,
    GENERATOR_FAULT_ONSET_CLASSES,
    GENERATOR_RAMP_SHAPES,
    SENSOR_SEED_BASE,
    load_base_scenario_data,
)
from .control_trace import (
    ControlTraceError,
    TerminalFailureReceipt,
    parse_control_trace,
    replay_control_trace,
)
from .forecast_issue55_race import (
    HMC_IMPLEMENTATION_GIT_SHA,
    Issue55RaceError,
    ZONE_FIELD_BOUNDS,
    ZONE_FIELD_COUNT,
    compute_race_metrics,
    project_true_targets,
    scenario_zone_order,
)
from .hmc import HabitatManagementComputer, HMCResetValidationError
from .hmc_contract import (
    HMCContract,
    HMCContractError,
    canonical_json_bytes,
    load_hmc_contract,
)
from .physics import advance_one_step_with_command, initial_state
from .scenario import Scenario, ScenarioValidationError

HARNESS_VERSION = "stress_search_v1"
SPEC_SCHEMA_VERSION = "aeolus_habitat_v2_stress_search_spec_v1"
AUDIT_SCHEMA_VERSION = "aeolus_habitat_v2_stress_search_audit_v1"
MANIFEST_SCHEMA_VERSION = "aeolus_habitat_v2_stress_search_manifest_v1"
REPLAY_RECEIPT_SCHEMA_VERSION = "aeolus_habitat_v2_stress_search_replay_receipt_v1"
COMPARISON_SCHEMA_VERSION = "aeolus_habitat_v2_stress_search_comparison_v1"

SPEC_FILE_NAME = "spec.json"
AUDIT_FILE_NAME = "audit.jsonl"
MANIFEST_FILE_NAME = "manifest.json"
CASES_DIRECTORY_NAME = "cases"
REPLAY_RECEIPT_FILE_NAME = "replay-receipt.json"
COMPARISON_FILE_NAME = "comparison.json"
COMPARISON_REPORT_FILE_NAME = "comparison.md"
ENVIRONMENT_FILE_NAME = "environment.json"
LOCK_FILE_NAME = "run.lock"

POPULATION_SCOPE = "DEVELOPMENT_ONLY"
PROTECTED_ACCESS = "prohibited"
POLICY_ID = "hmc_hold_no_adviser_v1"
INITIAL_ACHIEVED_STATE_RULE = "scheduled_command"
SCREENING_BACKEND_NONE = "none"
CLI_SCRIPT_RELATIVE_PATH = "scripts/run_fault_stress_search.py"

METHOD_RANDOM = "random_search"
METHOD_CROSS_ENTROPY = "cross_entropy"
METHODS = (METHOD_RANDOM, METHOD_CROSS_ENTROPY)

OBJECTIVES = {
    "peak_hazard_shift_v1": "peak_hazard_shift",
    "hazard_exposure_delta_v1": "hazard_exposure_delta",
}
OBJECTIVE_UNITS = {
    "peak_hazard_shift_v1": (
        "Issue 55 channel scale units (dimensionless); largest fault-minus-twin "
        "movement toward an HMC warning-enter threshold over steps, zones and hazard families"
    ),
    "hazard_exposure_delta_v1": (
        "Issue 55 channel scale units summed over zone-steps (dimensionless, not "
        "multiplied by dt); truth excess beyond HMC warning-enter thresholds, fault minus twin"
    ),
}
TIER_METRICS = ("peak_hazard_shift", "hazard_exposure_delta", "residual_shift_at_end")
TIER_OPERATORS = (">", ">=")

# Fault vocabulary this harness supports.  ``target`` names which declared
# scenario entity a fault of that type must identify.
ZONE_TARGET = "zone"
DAMPER_TARGET = "damper"
GLOBAL_TARGET = "global"
SUPPORTED_FAULT_TARGETS: Mapping[str, str] = {
    "fan_speed_degradation": GLOBAL_TARGET,
    "branch_resistance_increase": ZONE_TARGET,
    "cooling_delivery_degradation": ZONE_TARGET,
    "oxygen_delivery_degradation": ZONE_TARGET,
    "scrubber_effectiveness_degradation": GLOBAL_TARGET,
    "condenser_effectiveness_degradation": GLOBAL_TARGET,
    "damper_jam": DAMPER_TARGET,
}
# For these types a *larger* multiplier is the more severe fault.
_SEVERITY_INCREASES_WITH_MULTIPLIER = frozenset({"branch_resistance_increase"})
_MULTIPLIER_FREE_TYPES = frozenset({"damper_jam"})
GOVERNING_ONSET_RANGE = (
    min(low for low, _ in GENERATOR_FAULT_ONSET_CLASSES.values()),
    max(high for _, high in GENERATOR_FAULT_ONSET_CLASSES.values()),
)
GOVERNING_DURATION_RANGE = tuple(GENERATOR_FAULT_DURATION_RANGE)
MAX_FAULTS_LIMIT = 4
ROSTER_SENSOR_SEED_CEILING = SENSOR_SEED_BASE + 1_000_000

DISPOSITION_EVALUATED = "EVALUATED"
DISPOSITION_INVALID_PARAMETERS = "REJECTED_INVALID_PARAMETERS"
DISPOSITION_SCENARIO_VALIDATION = "REJECTED_SCENARIO_VALIDATION"
DISPOSITION_DUPLICATE = "DUPLICATE_EPISODE"
DISPOSITION_TERMINAL = "TERMINAL_FAILURE"
DISPOSITION_NONFINITE = "NONFINITE_OUTPUT"
DISPOSITION_SIMULATOR_ERROR = "SIMULATOR_ERROR"
DISPOSITIONS = (
    DISPOSITION_EVALUATED,
    DISPOSITION_INVALID_PARAMETERS,
    DISPOSITION_SCENARIO_VALIDATION,
    DISPOSITION_DUPLICATE,
    DISPOSITION_TERMINAL,
    DISPOSITION_NONFINITE,
    DISPOSITION_SIMULATOR_ERROR,
)
# Dispositions that consumed one simulation call.
_SIMULATED_DISPOSITIONS = frozenset(
    {
        DISPOSITION_EVALUATED,
        DISPOSITION_TERMINAL,
        DISPOSITION_NONFINITE,
        DISPOSITION_SIMULATOR_ERROR,
    }
)

OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_TERMINAL = "TERMINAL_FAILURE"
OUTCOME_NONFINITE = "NONFINITE_OUTPUT"
OUTCOME_ERROR = "SIMULATOR_ERROR"

END_SIMULATION_BUDGET = "BUDGET_SIMULATION_CALLS_EXHAUSTED"
END_ATTEMPT_BUDGET = "BUDGET_CANDIDATE_ATTEMPTS_EXHAUSTED"
END_WALL_BUDGET = "BUDGET_WALL_TIME_EXHAUSTED"
END_BASELINE_FAILED = "BASELINE_FAILED"
END_CANCELLED = "CANCELLED"
END_ABORTED = "ABORTED_ERROR"
# End reasons decided by a deterministic counter; only these are comparable
# across runs.  A wall-time stop depends on machine speed and load.
DETERMINISTIC_END_REASONS = (END_SIMULATION_BUDGET, END_ATTEMPT_BUDGET)
FINAL_END_REASONS = (
    END_SIMULATION_BUDGET,
    END_ATTEMPT_BUDGET,
    END_WALL_BUDGET,
    END_BASELINE_FAILED,
)

INTERRUPT_DEADLINE = "DEADLINE"
INTERRUPT_CANCELLED = "CANCELLED"

DATA_BOUNDARY = (
    "Development-only stress-search output. Candidates are adversarially selected "
    "with privileged simulator truth; they are biased discoveries, not IID samples, "
    "not a validation, final or BLIND_FINAL population and not proof of "
    "generalisation. Adding any case to a training corpus needs a separately "
    "reviewed split and provenance decision. All members of one causal family "
    "must stay in the same partition under any later split."
)

_NULL_SHA = "0" * 64
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
FIXTURE_ID_PREFIX = "stress-search-"
SPLIT_RULE = "every episode of one causal_family_id stays in one partition"
GPU_SCREENING_STATUS = {
    "enabled": False,
    "status": "OPEN_NOT_RUN",
    "note": "no validated batched backend and no DGX run; CPU plant only",
}
REPLAY_STATUS_PASS = "PASS"
REPLAY_STATUS_FAIL = "FAIL"
REPLAY_STATUS_NO_CASES = "NO_CASES"
_RNG_NAMESPACE = "aeolus-habitat-v2-stress-search-v1"
_UNIFORM_STD = 1.0 / math.sqrt(12.0)
_CATEGORY_FLOOR = 0.02
_TRUNCATION_RETRIES = 8

# Source files whose LF-normalised bytes define "the same code" for resume and
# replay.  The list is the harness itself plus every module on the simulated
# path it calls.
SOURCE_IDENTITY_PATHS = (
    "contracts/habitat_v2_hmc_v1.json",
    "scenarios/habitat_v2_forecast_development.json",
    "src/aeolus/habitat_v2/actuators.py",
    "src/aeolus/habitat_v2/air_network.py",
    "src/aeolus/habitat_v2/bdm_v1_families.py",
    "src/aeolus/habitat_v2/control_trace.py",
    "src/aeolus/habitat_v2/faults.py",
    "src/aeolus/habitat_v2/forecast_issue55_race.py",
    "src/aeolus/habitat_v2/health.py",
    "src/aeolus/habitat_v2/hmc.py",
    "src/aeolus/habitat_v2/hmc_contract.py",
    "src/aeolus/habitat_v2/instrumentation.py",
    "src/aeolus/habitat_v2/physics.py",
    "src/aeolus/habitat_v2/proposal.py",
    "src/aeolus/habitat_v2/safety.py",
    "src/aeolus/habitat_v2/scenario.py",
    "src/aeolus/habitat_v2/snapshot.py",
    "src/aeolus/habitat_v2/state.py",
    "src/aeolus/habitat_v2/stress_search.py",
    "src/aeolus/habitat_v2/telemetry.py",
)


class StressSearchError(ValueError):
    """Base error; every failure in this module fails closed."""


class StressSearchSpecError(StressSearchError):
    """The search specification is malformed, unsupported or out of bounds."""


class StressSearchCandidateError(StressSearchError):
    """Candidate parameters violate the declared search space."""


class StressSearchAuditError(StressSearchError):
    """The audit log is malformed, tampered with or inconsistent."""


class StressSearchManifestError(StressSearchError):
    """A candidate manifest is malformed or does not match this environment."""


class StressSearchOutputError(StressSearchError):
    """A write-once output already exists with different content."""


class EpisodeInterrupted(Exception):
    """Raised inside an episode when a deadline or cancellation fires."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


# ---------------------------------------------------------------------------
# Canonical JSON and strict parsing
# ---------------------------------------------------------------------------


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes; non-finite numbers and unsupported types fail."""

    try:
        return canonical_json_bytes(value)
    except Exception as error:  # HMCContractError wraps TypeError/ValueError
        raise StressSearchError(f"value is not canonical JSON: {error}") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(data: bytes | str, *, label: str, error: type[StressSearchError]) -> Any:
    """Parse JSON rejecting duplicate keys and NaN/Infinity."""

    try:
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, ValueError) as failure:
        raise error(f"{label} is not strict JSON: {failure}") from failure


def _exact_fields(
    mapping: Any, fields: Sequence[str], label: str, error: type[StressSearchError]
) -> None:
    if type(mapping) is not dict:
        raise error(f"{label} must be an object")
    unknown = sorted(set(mapping) - set(fields))
    missing = sorted(set(fields) - set(mapping))
    if unknown:
        raise error(f"unknown {label} fields: {', '.join(unknown)}")
    if missing:
        raise error(f"missing {label} fields: {', '.join(missing)}")


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _is_sha(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


# ---------------------------------------------------------------------------
# Search specification
# ---------------------------------------------------------------------------

_SPEC_FIELDS = (
    "schema_version",
    "spec_id",
    "status",
    "population",
    "fixture",
    "policy",
    "search_space",
    "objective",
    "discovery_tiers",
    "methods",
    "budget",
    "seeds",
    "selection",
    "replay",
    "screening",
)
_FIXTURE_FIELDS = (
    "fixture_id",
    "base_scenario_relative_path",
    "base_scenario_sha256",
    "horizon_steps",
    "dt_seconds",
    "initial_achieved_state",
    "sensor_random_seed",
)
_POLICY_FIELDS = ("policy_id", "hmc_contract_relative_path", "hmc_contract_sha256")
_SEARCH_SPACE_FIELDS = (
    "min_faults",
    "max_faults",
    "fault_types",
    "onset_step_range",
    "duration_steps_range",
    "latest_end_step",
    "ramp_shapes",
    "multiplier_decimals",
)
_OBJECTIVE_FIELDS = ("objective_id", "direction", "units", "baseline")
_TIER_FIELDS = ("tier_id", "metric", "operator", "value")
_CEM_FIELDS = ("population_size", "elite_count", "smoothing", "min_sigma")
_BUDGET_FIELDS = ("max_simulation_calls", "max_candidate_attempts", "max_wall_seconds")
_SELECTION_FIELDS = ("shortlist_size", "max_per_causal_family", "minimum_objective_exclusive")
_REPLAY_FIELDS = ("metric_absolute_tolerance", "require_identical_trace_sha256")

BASE_SCENARIO_RELATIVE_PATH = "scenarios/habitat_v2_forecast_development.json"
HMC_CONTRACT_RELATIVE_PATH = "contracts/habitat_v2_hmc_v1.json"


def _int_range(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2 or not all(_is_int(item) for item in value):
        raise StressSearchSpecError(f"{label} must be [low, high] integers")
    low, high = value
    if low > high:
        raise StressSearchSpecError(f"{label} must satisfy low <= high")
    return low, high


def _float_range(value: Any, label: str) -> tuple[float, float]:
    if type(value) is not list or len(value) != 2 or not all(_is_number(item) for item in value):
        raise StressSearchSpecError(f"{label} must be [low, high] finite numbers")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise StressSearchSpecError(f"{label} must satisfy low <= high")
    return low, high


@dataclass(frozen=True)
class SearchSpec:
    """A validated research-only search specification."""

    data: Mapping[str, Any]
    spec_sha256: str

    @property
    def horizon_steps(self) -> int:
        return int(self.data["fixture"]["horizon_steps"])

    @property
    def fault_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.data["search_space"]["fault_types"]))

    @property
    def seeds(self) -> tuple[str, ...]:
        return tuple(self.data["seeds"])

    @property
    def objective_metric(self) -> str:
        return OBJECTIVES[self.data["objective"]["objective_id"]]

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.data))


def validate_spec_mapping(mapping: Any) -> SearchSpec:
    """Validate a spec mapping against the closed schema and governing bands.

    Declared ranges may be narrower than the reviewed BDM-v1 generator
    vocabulary but never wider; unsupported fault types, sensor or feedback
    defects and accelerated screening are rejected.
    """

    if type(mapping) is not dict:
        raise StressSearchSpecError("spec must be an object")
    try:
        data = json.loads(canonical_bytes(mapping))
    except StressSearchError as error:
        raise StressSearchSpecError(str(error)) from error
    _exact_fields(data, _SPEC_FIELDS, "spec", StressSearchSpecError)
    if data["schema_version"] != SPEC_SCHEMA_VERSION:
        raise StressSearchSpecError("unsupported spec schema_version")
    for name in ("spec_id", "status"):
        if type(data[name]) is not str or not data[name]:
            raise StressSearchSpecError(f"spec {name} must be a non-empty string")

    population = data["population"]
    _exact_fields(
        population, ("scope", "protected_population_access"), "population", StressSearchSpecError
    )
    if population["scope"] != POPULATION_SCOPE:
        raise StressSearchSpecError(f"population scope must be {POPULATION_SCOPE}")
    if population["protected_population_access"] != PROTECTED_ACCESS:
        raise StressSearchSpecError("protected population access must be prohibited")

    fixture = data["fixture"]
    _exact_fields(fixture, _FIXTURE_FIELDS, "fixture", StressSearchSpecError)
    fixture_id = fixture["fixture_id"]
    if (
        type(fixture_id) is not str
        or _NAME_PATTERN.fullmatch(fixture_id) is None
        or not fixture_id.startswith(FIXTURE_ID_PREFIX)
    ):
        raise StressSearchSpecError(
            f"fixture_id must start with {FIXTURE_ID_PREFIX!r} and use only letters, "
            "digits, '-' and '_' so scenario names stay in the harness namespace"
        )
    if fixture["base_scenario_relative_path"] != BASE_SCENARIO_RELATIVE_PATH:
        raise StressSearchSpecError(
            f"only the development base fixture {BASE_SCENARIO_RELATIVE_PATH} is supported"
        )
    if not _is_sha(fixture["base_scenario_sha256"]):
        raise StressSearchSpecError("base_scenario_sha256 must be a sha256 hex digest")
    horizon = fixture["horizon_steps"]
    if not _is_int(horizon) or horizon < 2:
        raise StressSearchSpecError("horizon_steps must be an integer >= 2")
    if not _is_number(fixture["dt_seconds"]) or float(fixture["dt_seconds"]) != DT_SECONDS:
        raise StressSearchSpecError(f"dt_seconds must be {DT_SECONDS}")
    if fixture["initial_achieved_state"] != INITIAL_ACHIEVED_STATE_RULE:
        raise StressSearchSpecError(
            f"initial_achieved_state must be {INITIAL_ACHIEVED_STATE_RULE!r}"
        )
    seed = fixture["sensor_random_seed"]
    if not _is_int(seed) or seed < 0:
        raise StressSearchSpecError("sensor_random_seed must be a non-negative integer")
    if SENSOR_SEED_BASE <= seed < ROSTER_SENSOR_SEED_CEILING:
        raise StressSearchSpecError(
            "sensor_random_seed lies in the BDM-v1 roster seed range and is refused"
        )

    policy = data["policy"]
    _exact_fields(policy, _POLICY_FIELDS, "policy", StressSearchSpecError)
    if policy["policy_id"] != POLICY_ID:
        raise StressSearchSpecError(f"only the fixed policy {POLICY_ID} is supported")
    if policy["hmc_contract_relative_path"] != HMC_CONTRACT_RELATIVE_PATH:
        raise StressSearchSpecError(f"only {HMC_CONTRACT_RELATIVE_PATH} is supported")
    if not _is_sha(policy["hmc_contract_sha256"]):
        raise StressSearchSpecError("hmc_contract_sha256 must be a sha256 hex digest")

    space = data["search_space"]
    _exact_fields(space, _SEARCH_SPACE_FIELDS, "search_space", StressSearchSpecError)
    min_faults, max_faults = space["min_faults"], space["max_faults"]
    if not _is_int(min_faults) or not _is_int(max_faults):
        raise StressSearchSpecError("min_faults and max_faults must be integers")
    if not 1 <= min_faults <= max_faults <= MAX_FAULTS_LIMIT:
        raise StressSearchSpecError(
            f"fault count must satisfy 1 <= min_faults <= max_faults <= {MAX_FAULTS_LIMIT}"
        )
    decimals = space["multiplier_decimals"]
    if not _is_int(decimals) or not 1 <= decimals <= 9:
        raise StressSearchSpecError("multiplier_decimals must be an integer in 1..9")
    fault_types = space["fault_types"]
    if type(fault_types) is not dict or not fault_types:
        raise StressSearchSpecError("fault_types must be a non-empty object")
    for fault_type, declaration in fault_types.items():
        if fault_type not in SUPPORTED_FAULT_TARGETS:
            raise StressSearchSpecError(
                f"unsupported fault type {fault_type!r}; supported: "
                + ", ".join(sorted(SUPPORTED_FAULT_TARGETS))
            )
        if fault_type in _MULTIPLIER_FREE_TYPES:
            _exact_fields(declaration, (), f"fault type {fault_type}", StressSearchSpecError)
            continue
        _exact_fields(
            declaration, ("multiplier_range",), f"fault type {fault_type}", StressSearchSpecError
        )
        low, high = _float_range(declaration["multiplier_range"], f"{fault_type} multiplier_range")
        band_low, band_high = GENERATOR_FAULT_MULTIPLIER_BANDS[fault_type]
        if low < band_low or high > band_high:
            raise StressSearchSpecError(
                f"{fault_type} multiplier_range [{low}, {high}] exceeds the governing "
                f"band [{band_low}, {band_high}]"
            )
        if round(low, decimals) != low or round(high, decimals) != high:
            raise StressSearchSpecError(
                f"{fault_type} multiplier_range endpoints must be representable at "
                f"{decimals} decimals"
            )
    onset_low, onset_high = _int_range(space["onset_step_range"], "onset_step_range")
    if onset_low < GOVERNING_ONSET_RANGE[0] or onset_high > GOVERNING_ONSET_RANGE[1]:
        raise StressSearchSpecError(
            f"onset_step_range exceeds the governing range {list(GOVERNING_ONSET_RANGE)}"
        )
    duration_low, duration_high = _int_range(space["duration_steps_range"], "duration_steps_range")
    if (
        duration_low < GOVERNING_DURATION_RANGE[0]
        or duration_high > GOVERNING_DURATION_RANGE[1]
    ):
        raise StressSearchSpecError(
            f"duration_steps_range exceeds the governing range {list(GOVERNING_DURATION_RANGE)}"
        )
    latest_end = space["latest_end_step"]
    if not _is_int(latest_end) or latest_end > horizon - 1:
        raise StressSearchSpecError("latest_end_step must be an integer <= horizon_steps - 1")
    if onset_low + duration_low > latest_end:
        raise StressSearchSpecError(
            "search space is empty: no onset and duration fit before latest_end_step"
        )
    shapes = space["ramp_shapes"]
    if (
        type(shapes) is not list
        or not shapes
        or len(set(shapes)) != len(shapes)
        or any(shape not in GENERATOR_RAMP_SHAPES for shape in shapes)
    ):
        raise StressSearchSpecError(
            "ramp_shapes must be unique members of " + ", ".join(GENERATOR_RAMP_SHAPES)
        )

    objective = data["objective"]
    _exact_fields(objective, _OBJECTIVE_FIELDS, "objective", StressSearchSpecError)
    if objective["objective_id"] not in OBJECTIVES:
        raise StressSearchSpecError(
            "objective_id must be one of " + ", ".join(sorted(OBJECTIVES))
        )
    if objective["direction"] != "maximize":
        raise StressSearchSpecError("objective direction must be 'maximize'")
    if objective["units"] != OBJECTIVE_UNITS[objective["objective_id"]]:
        raise StressSearchSpecError("objective units do not match the declared objective")
    if objective["baseline"] != "healthy_twin":
        raise StressSearchSpecError("objective baseline must be 'healthy_twin'")

    tiers = data["discovery_tiers"]
    if type(tiers) is not list or not tiers:
        raise StressSearchSpecError("discovery_tiers must be a non-empty list")
    tier_ids: set[str] = set()
    for tier in tiers:
        _exact_fields(tier, _TIER_FIELDS, "discovery tier", StressSearchSpecError)
        if type(tier["tier_id"]) is not str or not tier["tier_id"] or tier["tier_id"] in tier_ids:
            raise StressSearchSpecError("discovery tier_id must be a unique non-empty string")
        tier_ids.add(tier["tier_id"])
        if tier["metric"] not in TIER_METRICS or tier["operator"] not in TIER_OPERATORS:
            raise StressSearchSpecError("discovery tier metric or operator is unsupported")
        if not _is_number(tier["value"]):
            raise StressSearchSpecError("discovery tier value must be a finite number")

    methods = data["methods"]
    _exact_fields(methods, METHODS, "methods", StressSearchSpecError)
    _exact_fields(methods[METHOD_RANDOM], (), "random_search settings", StressSearchSpecError)
    cem = methods[METHOD_CROSS_ENTROPY]
    _exact_fields(cem, _CEM_FIELDS, "cross_entropy settings", StressSearchSpecError)
    if not _is_int(cem["population_size"]) or not 2 <= cem["population_size"] <= 256:
        raise StressSearchSpecError("population_size must be an integer in 2..256")
    if not _is_int(cem["elite_count"]) or not 1 <= cem["elite_count"] < cem["population_size"]:
        raise StressSearchSpecError("elite_count must satisfy 1 <= elite_count < population_size")
    if not _is_number(cem["smoothing"]) or not 0.0 < float(cem["smoothing"]) <= 1.0:
        raise StressSearchSpecError("smoothing must be in (0, 1]")
    if not _is_number(cem["min_sigma"]) or not 0.0 < float(cem["min_sigma"]) <= _UNIFORM_STD:
        raise StressSearchSpecError("min_sigma must be in (0, 1/sqrt(12)]")

    budget = data["budget"]
    _exact_fields(budget, _BUDGET_FIELDS, "budget", StressSearchSpecError)
    if not _is_int(budget["max_simulation_calls"]) or not 2 <= budget["max_simulation_calls"] <= 100_000:
        raise StressSearchSpecError("max_simulation_calls must be an integer in 2..100000")
    if (
        not _is_int(budget["max_candidate_attempts"])
        or not 1 <= budget["max_candidate_attempts"] <= 1_000_000
    ):
        raise StressSearchSpecError("max_candidate_attempts must be an integer in 1..1000000")
    if not _is_number(budget["max_wall_seconds"]) or float(budget["max_wall_seconds"]) <= 0.0:
        raise StressSearchSpecError("max_wall_seconds must be a positive finite number")

    seeds = data["seeds"]
    if (
        type(seeds) is not list
        or not seeds
        or any(type(item) is not str or _NAME_PATTERN.fullmatch(item) is None for item in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise StressSearchSpecError(
            "seeds must be unique names of letters, digits, '-' and '_' (at most 64 characters)"
        )

    selection = data["selection"]
    _exact_fields(selection, _SELECTION_FIELDS, "selection", StressSearchSpecError)
    if not _is_int(selection["shortlist_size"]) or not 0 <= selection["shortlist_size"] <= 64:
        raise StressSearchSpecError("shortlist_size must be an integer in 0..64")
    if not _is_int(selection["max_per_causal_family"]) or selection["max_per_causal_family"] < 1:
        raise StressSearchSpecError("max_per_causal_family must be an integer >= 1")
    if not _is_number(selection["minimum_objective_exclusive"]):
        raise StressSearchSpecError("minimum_objective_exclusive must be a finite number")

    replay = data["replay"]
    _exact_fields(replay, _REPLAY_FIELDS, "replay", StressSearchSpecError)
    tolerance = replay["metric_absolute_tolerance"]
    if not _is_number(tolerance) or float(tolerance) < 0.0:
        raise StressSearchSpecError("metric_absolute_tolerance must be a finite number >= 0")
    if type(replay["require_identical_trace_sha256"]) is not bool:
        raise StressSearchSpecError("require_identical_trace_sha256 must be a boolean")

    screening = data["screening"]
    _exact_fields(screening, ("backend",), "screening", StressSearchSpecError)
    if screening["backend"] != SCREENING_BACKEND_NONE:
        raise StressSearchSpecError(
            "accelerated screening is refused: no validated batched backend exists; "
            "screening.backend must be 'none'"
        )
    return SearchSpec(data=data, spec_sha256=canonical_sha256(data))


def load_spec_file(path: str | Path) -> SearchSpec:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise StressSearchSpecError(f"cannot read spec: {error}") from error
    return validate_spec_mapping(
        strict_json_loads(raw, label="spec", error=StressSearchSpecError)
    )


# ---------------------------------------------------------------------------
# Environment: fixture, contract, hazard table, source identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HazardFamily:
    family: str
    channel: str
    direction: str
    warning_enter: float
    scale: float
    column: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "channel": self.channel,
            "direction": self.direction,
            "warning_enter": self.warning_enter,
            "scale": self.scale,
        }


def hazard_table(contract: HMCContract) -> tuple[HazardFamily, ...]:
    """Hazard families: HMC warning-enter thresholds with Issue 55 scales."""

    columns = {name: index for index, (name, *_rest) in enumerate(ZONE_FIELD_BOUNDS)}
    scales = {name: float(scale) for name, scale, *_rest in ZONE_FIELD_BOUNDS}
    environmental = contract.data["health_policy"]["environmental"]
    families = []
    for family in sorted(environmental):
        entry = environmental[family]
        channel = entry["channel"]
        direction = entry["direction"]
        if channel not in columns or direction not in ("HIGH", "LOW"):
            raise StressSearchError(f"hazard family {family} is not scoreable")
        families.append(
            HazardFamily(
                family=family,
                channel=channel,
                direction=direction,
                warning_enter=float(entry["warning_enter"]),
                scale=scales[channel],
                column=columns[channel],
            )
        )
    if not families:
        raise StressSearchError("HMC contract declares no environmental hazard families")
    return tuple(families)


def _normalised_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def collect_source_identity(repo_root: str | Path) -> dict[str, Any]:
    """Digest the harness and simulated-path sources (line-ending independent)."""

    root = Path(repo_root)
    files = []
    for relative in SOURCE_IDENTITY_PATHS:
        path = root / relative
        if not path.is_file():
            raise StressSearchError(f"source identity file is missing: {relative}")
        files.append({"relative_path": relative, "sha256_lf": _normalised_file_sha256(path)})
    return {"files": files, "source_sha256": canonical_sha256(files)}


def collect_runtime_environment(repo_root: str | Path) -> dict[str, Any]:
    """Unhashed provenance: commit, runtime and platform.  Never a local path."""

    def _git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "source_commit": commit.strip() if commit else None,
        "source_worktree_dirty": None if status is None else bool(status.strip()),
        "container_image_digest": None,
        "container_image_note": "not containerised; runtime identified by the fields below",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@dataclass(frozen=True)
class SearchEnvironment:
    """Everything a run binds to: spec, fixture, contract, hazards, sources."""

    spec: SearchSpec
    contract: HMCContract
    twin_data: Mapping[str, Any]
    twin_scenario: Scenario
    zone_ids: tuple[str, ...]
    damper_ids: tuple[str, ...]
    hazards: tuple[HazardFamily, ...]
    source_identity: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "harness_version": HARNESS_VERSION,
            "spec_sha256": self.spec.spec_sha256,
            "base_scenario_sha256": self.spec.data["fixture"]["base_scenario_sha256"],
            "fixture_scenario_sha256": self.twin_scenario.scenario_sha256,
            "hmc_contract_sha256": self.contract.hmc_contract_sha256,
            "hmc_implementation_git_sha": HMC_IMPLEMENTATION_GIT_SHA,
            "hazard_table": [hazard.to_mapping() for hazard in self.hazards],
            "source_sha256": self.source_identity["source_sha256"],
        }


def build_fixture_data(spec: SearchSpec, base_data: Mapping[str, Any]) -> dict[str, Any]:
    """The healthy twin: base fixture, one segment to the horizon, nominal hold.

    Setting the initial achieved actuation to the scheduled command makes the
    HMC hold rule repeat the nominal command, so physical faults act on running
    equipment.  Sensor noise amplitudes are left at the reviewed HMC values.
    """

    fixture = spec.data["fixture"]
    data = copy.deepcopy(dict(base_data))
    horizon = int(fixture["horizon_steps"])
    segment = copy.deepcopy(data["timeline"][0])
    segment["start_step"] = 0
    segment["end_step"] = horizon
    data["timeline"] = [segment]
    data["steps"] = horizon
    data["dt_seconds"] = float(fixture["dt_seconds"])
    command = segment["command"]
    utility = data["initial_utility"]
    utility["actual_fan_speed_fraction"] = command["fan_speed_fraction"]
    utility["actual_scrubber_duty"] = command["scrubber_duty"]
    utility["actual_condenser_duty"] = command["condenser_duty"]
    utility["actual_damper_position_by_id"] = dict(command["damper_position_by_id"])
    utility["actual_cooling_removed_w"] = dict(command["cooling_removed_w"])
    utility["actual_oxygen_injection_mol_s"] = dict(command["oxygen_injection_mol_s"])
    data["sensor_model"]["random_seed"] = int(fixture["sensor_random_seed"])
    data["fault_profiles"] = []
    data["name"] = f"{fixture['fixture_id']}-healthy-twin"
    return data


def build_environment(spec: SearchSpec, repo_root: str | Path) -> SearchEnvironment:
    """Bind a spec to the checked-in base fixture, HMC contract and sources."""

    root = Path(repo_root)
    try:
        base_data, base_sha = load_base_scenario_data(root)
    except (BdmV1FamilyError, OSError, ValueError) as error:
        raise StressSearchSpecError(f"cannot load the base fixture: {error}") from error
    if base_sha != spec.data["fixture"]["base_scenario_sha256"]:
        raise StressSearchSpecError(
            "base scenario digest does not match the spec; the fixture has drifted"
        )
    try:
        contract = load_hmc_contract(root / HMC_CONTRACT_RELATIVE_PATH)
    except (HMCContractError, OSError, ValueError) as error:
        raise StressSearchSpecError(f"cannot load the HMC contract: {error}") from error
    if contract.hmc_contract_sha256 != spec.data["policy"]["hmc_contract_sha256"]:
        raise StressSearchSpecError("HMC contract digest does not match the spec")
    twin_data = build_fixture_data(spec, base_data)
    try:
        twin = Scenario.from_mapping(twin_data)
    except ScenarioValidationError as error:
        raise StressSearchSpecError(f"fixture is not a valid scenario: {error}") from error
    zone_ids = scenario_zone_order(twin)
    damper_ids = tuple(
        sorted(branch["damper_id"] for branch in twin.data["air_network"]["branches"])
    )
    return SearchEnvironment(
        spec=spec,
        contract=contract,
        twin_data=twin_data,
        twin_scenario=twin,
        zone_ids=tuple(zone_ids),
        damper_ids=damper_ids,
        hazards=hazard_table(contract),
        source_identity=collect_source_identity(root),
    )


# ---------------------------------------------------------------------------
# Candidate parameters, identity and scenario construction
# ---------------------------------------------------------------------------

_FAULT_FIELDS = ("type", "target", "onset_step", "duration_steps", "ramp_shape", "multiplier")


def _fault_sort_key(fault: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        fault["type"],
        fault["target"] or "",
        fault["onset_step"],
        fault["duration_steps"],
        fault["ramp_shape"] or "",
        -1.0 if fault["multiplier"] is None else float(fault["multiplier"]),
    )


def canonical_faults(faults: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order-independent canonical form of a candidate's fault list."""

    return sorted(
        ({name: fault[name] for name in _FAULT_FIELDS} for fault in faults),
        key=_fault_sort_key,
    )


def validate_candidate_parameters(
    environment: SearchEnvironment, faults: Any
) -> list[dict[str, Any]]:
    """Check candidate parameters against the declared search space.

    Returns the canonical fault list.  Anything outside the declared bounds is
    rejected; nothing is clipped into range.
    """

    space = environment.spec.data["search_space"]
    if type(faults) is not list:
        raise StressSearchCandidateError("faults must be a list")
    if not space["min_faults"] <= len(faults) <= space["max_faults"]:
        raise StressSearchCandidateError(
            f"fault count {len(faults)} is outside "
            f"[{space['min_faults']}, {space['max_faults']}]"
        )
    onset_low, onset_high = space["onset_step_range"]
    duration_low, duration_high = space["duration_steps_range"]
    decimals = space["multiplier_decimals"]
    for fault in faults:
        if type(fault) is not dict or set(fault) != set(_FAULT_FIELDS):
            raise StressSearchCandidateError(
                "each fault must have exactly the fields " + ", ".join(_FAULT_FIELDS)
            )
        fault_type = fault["type"]
        if type(fault_type) is not str or fault_type not in space["fault_types"]:
            raise StressSearchCandidateError(f"fault type {fault_type!r} is not permitted")
        target_kind = SUPPORTED_FAULT_TARGETS[fault_type]
        target = fault["target"]
        if target_kind == GLOBAL_TARGET:
            if target is not None:
                raise StressSearchCandidateError(f"{fault_type} takes no target")
        elif target_kind == ZONE_TARGET:
            if type(target) is not str or target not in environment.zone_ids:
                raise StressSearchCandidateError(f"{fault_type} target is not a declared zone")
        elif type(target) is not str or target not in environment.damper_ids:
            raise StressSearchCandidateError(f"{fault_type} target is not a declared damper")
        onset, duration = fault["onset_step"], fault["duration_steps"]
        if not _is_int(onset) or not onset_low <= onset <= onset_high:
            raise StressSearchCandidateError(
                f"onset_step {onset!r} is outside [{onset_low}, {onset_high}]"
            )
        if not _is_int(duration) or not duration_low <= duration <= duration_high:
            raise StressSearchCandidateError(
                f"duration_steps {duration!r} is outside [{duration_low}, {duration_high}]"
            )
        if onset + duration > space["latest_end_step"]:
            raise StressSearchCandidateError(
                f"fault end step {onset + duration} exceeds latest_end_step "
                f"{space['latest_end_step']}"
            )
        if fault_type in _MULTIPLIER_FREE_TYPES:
            if fault["multiplier"] is not None or fault["ramp_shape"] is not None:
                raise StressSearchCandidateError(f"{fault_type} takes no multiplier or ramp shape")
            continue
        if fault["ramp_shape"] not in space["ramp_shapes"]:
            raise StressSearchCandidateError(f"ramp_shape {fault['ramp_shape']!r} is not permitted")
        multiplier = fault["multiplier"]
        if type(multiplier) is not float or not math.isfinite(multiplier):
            raise StressSearchCandidateError("multiplier must be a finite float")
        low, high = space["fault_types"][fault_type]["multiplier_range"]
        if not float(low) <= multiplier <= float(high):
            raise StressSearchCandidateError(
                f"{fault_type} multiplier {multiplier} is outside [{low}, {high}]"
            )
        if round(multiplier, decimals) != multiplier:
            raise StressSearchCandidateError(
                f"multiplier {multiplier!r} is not quantised to {decimals} decimals"
            )
    return canonical_faults(faults)


def episode_identity(
    environment: SearchEnvironment, faults: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Causal episode and family identities for a canonical fault list.

    ``episode_id`` identifies the exact causal content (fixture, policy and
    every fault parameter) independent of fault order, names or ids.
    ``causal_family_id`` identifies the mechanism composition: the same fault
    types on the same targets with any magnitude, timing or ramp are related
    perturbations of one family and must stay together under any later split.
    """

    canonical = canonical_faults(faults)
    binding = {
        "harness_version": HARNESS_VERSION,
        "fixture_scenario_sha256": environment.twin_scenario.scenario_sha256,
        "hmc_contract_sha256": environment.contract.hmc_contract_sha256,
        "policy_id": POLICY_ID,
    }
    mechanisms = sorted({(fault["type"], fault["target"] or "") for fault in canonical})
    family_key = {
        "causal_scenario_template": environment.spec.data["fixture"]["fixture_id"],
        "fault_mechanism_composition": "+".join(sorted({fault["type"] for fault in canonical})),
        "fault_mechanism_targets": [list(item) for item in mechanisms],
        "operating_schedule": "steady",
        "sensor_failure_bundle": "none",
    }
    return {
        "episode_id": canonical_sha256({**binding, "faults": canonical}),
        "causal_family_id": canonical_sha256({**binding, "family_key": family_key}),
        "causal_family_key": family_key,
    }


def _multiplier_endpoints(shape: str, multiplier: float) -> tuple[float, float]:
    if shape == "worsening":
        return 1.0, multiplier
    if shape == "recovery":
        return multiplier, 1.0
    return multiplier, multiplier


def fault_profiles_for(faults: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Scenario ``fault_profiles`` entries for a canonical fault list."""

    profiles = []
    for index, fault in enumerate(canonical_faults(faults)):
        profile: dict[str, Any] = {
            "id": f"ss1-{index}-{fault['type']}",
            "type": fault["type"],
            "start_step": fault["onset_step"],
            "end_step": fault["onset_step"] + fault["duration_steps"],
        }
        target_kind = SUPPORTED_FAULT_TARGETS[fault["type"]]
        if target_kind == ZONE_TARGET:
            profile["zone_id"] = fault["target"]
        elif target_kind == DAMPER_TARGET:
            profile["damper_id"] = fault["target"]
        if fault["type"] not in _MULTIPLIER_FREE_TYPES:
            start, end = _multiplier_endpoints(fault["ramp_shape"], float(fault["multiplier"]))
            profile["start_multiplier"] = start
            profile["end_multiplier"] = end
        profiles.append(profile)
    return profiles


def build_candidate_scenario(
    environment: SearchEnvironment, faults: Sequence[Mapping[str, Any]], episode_id: str
) -> Scenario:
    """Build and validate the candidate scenario with the existing validator."""

    data = copy.deepcopy(dict(environment.twin_data))
    data["fault_profiles"] = fault_profiles_for(faults)
    data["name"] = f"{environment.spec.data['fixture']['fixture_id']}-{episode_id[:16]}"
    try:
        return Scenario.from_mapping(data)
    except ScenarioValidationError:
        raise
    except (KeyError, TypeError, AttributeError) as error:
        raise ScenarioValidationError(f"malformed scenario container: {error!r}") from error


def episode_nonce(episode_id: str) -> bytes:
    return hashlib.sha256(b"stress-search-v1|" + episode_id.encode("ascii")).digest()


# ---------------------------------------------------------------------------
# Authoritative episode
# ---------------------------------------------------------------------------


@dataclass
class EpisodeOutcome:
    """Result of one evaluator call.  ``rows`` is privileged truth [T, 51]."""

    status: str
    committed_steps: int
    rows: np.ndarray | None = None
    race_metrics: Mapping[str, Any] | None = None
    terminal_reason: str | None = None
    error: str | None = None
    trace_sha256: str | None = None
    control_run_id: str | None = None
    final_state_sha256: str | None = None
    actuation_sha256: str | None = None
    emergency_action_steps: int = 0
    replay_verified: bool = False
    actuation: list[dict[str, Any]] = field(default_factory=list)
    trace_bytes: bytes | None = None


Evaluator = Callable[[Scenario, str, Callable[[], None] | None], EpisodeOutcome]


def run_authoritative_episode(
    scenario: Scenario,
    contract: HMCContract,
    episode_id: str,
    *,
    interrupt_check: Callable[[], None] | None = None,
    verify_replay: bool = False,
    keep_artifacts: bool = False,
) -> EpisodeOutcome:
    """Run one episode on the authoritative CPU plant under the HMC, no adviser.

    A shadow plant recomputes each committed step from the HMC final command
    and must match the HMC's plant receipt digest, which is how privileged
    truth is obtained without touching HMC internals.  HMC terminal failures
    are data, not exceptions.  ``interrupt_check`` may raise
    :class:`EpisodeInterrupted` between steps.
    """

    try:
        hmc = HabitatManagementComputer.reset(scenario, contract, episode_nonce(episode_id))
    except HMCResetValidationError as error:
        return EpisodeOutcome(
            status=OUTCOME_ERROR, committed_steps=0, error=f"HMC reset rejected: {error}"
        )
    zone_ids = scenario_zone_order(scenario)
    shadow = initial_state(scenario)
    try:
        initial_row = project_true_targets(scenario, zone_ids, shadow)
    except Issue55RaceError as error:
        return EpisodeOutcome(
            status=OUTCOME_NONFINITE,
            committed_steps=0,
            error=f"initial plant truth is not finite: {error}",
        )
    nonfinite_error: str | None = None
    states: list[Any] = []
    rows: list[np.ndarray] = []
    actuation: list[dict[str, Any]] = []
    emergency_steps = 0
    terminal_reason: str | None = None
    for step_index in range(int(scenario.data["steps"])):
        if interrupt_check is not None:
            interrupt_check()
        observed = hmc.observe()
        if isinstance(observed, TerminalFailureReceipt):
            terminal_reason = observed.reason_code
            break
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        if isinstance(arbitration, TerminalFailureReceipt):
            terminal_reason = arbitration.reason_code
            break
        arbitration_mapping = arbitration.to_mapping()
        step_receipt = hmc.step()
        if isinstance(step_receipt, TerminalFailureReceipt):
            terminal_reason = step_receipt.reason_code
            break
        result = advance_one_step_with_command(scenario, shadow, dict(arbitration.final_command))
        receipt_digest = hashlib.sha256(canonical_json_bytes(result.receipt)).hexdigest()
        if receipt_digest != step_receipt.plant_receipt_digest:
            return EpisodeOutcome(
                status=OUTCOME_ERROR,
                committed_steps=len(states),
                error=f"shadow plant receipt diverged from the HMC at step {step_index + 1}",
            )
        shadow = result.state
        states.append(shadow)
        try:
            rows.append(
                np.asarray(project_true_targets(scenario, zone_ids, shadow), dtype=np.float64)
            )
        except Issue55RaceError as error:
            # The existing truth projection refuses non-finite plant state.
            nonfinite_error = f"plant truth is not finite at step {step_index + 1}: {error}"
        owner = arbitration_mapping["command_owner"]
        if owner == "emergency_safe_action":
            emergency_steps += 1
        actuation.append(
            {
                "application_step": step_index,
                "adviser_requested_command": arbitration_mapping["requested_command"],
                "final_command": arbitration_mapping["final_command"],
                "final_command_sha256": arbitration.final_command_sha256,
                "command_owner": owner,
                "disposition": arbitration_mapping["disposition"],
                "reason_codes": arbitration_mapping["reason_codes"],
                "emergency_override": arbitration_mapping["emergency_override"],
                "plant_actuators": result.receipt["actuators"],
                "active_faults": result.receipt["active_faults"],
            }
        )
        if nonfinite_error is not None:
            break
    if nonfinite_error is not None:
        # The HMC stopped before its horizon, so no control trace can be
        # exported; the failure is recorded without one.
        return EpisodeOutcome(
            status=OUTCOME_NONFINITE,
            committed_steps=len(states),
            error=nonfinite_error,
            emergency_action_steps=emergency_steps,
            control_run_id=hmc.control_run_id,
            actuation_sha256=canonical_sha256(actuation),
        )
    outcome = EpisodeOutcome(
        status=OUTCOME_COMPLETED if terminal_reason is None else OUTCOME_TERMINAL,
        committed_steps=len(states),
        terminal_reason=terminal_reason,
        emergency_action_steps=emergency_steps,
        control_run_id=hmc.control_run_id,
    )
    try:
        outcome.actuation_sha256 = canonical_sha256(actuation)
        trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
        outcome.trace_sha256 = hashlib.sha256(trace.canonical_bytes).hexdigest()
        outcome.final_state_sha256 = trace.footer["final_state_sha256"]
        if verify_replay:
            parse_control_trace(trace.canonical_bytes, scenario=scenario, contract=contract)
            replayed = replay_control_trace(
                trace.canonical_bytes, scenario=scenario, contract=contract
            )
            if (
                replayed.committed_step_count != len(states)
                or replayed.final_state_sha256 != outcome.final_state_sha256
            ):
                raise ControlTraceError("authoritative replay disagrees with the episode")
            outcome.replay_verified = True
    except (ControlTraceError, StressSearchError) as error:
        outcome.status = OUTCOME_ERROR
        outcome.error = f"control trace export or replay failed: {error}"
        return outcome
    if keep_artifacts:
        outcome.actuation = actuation
        outcome.trace_bytes = trace.canonical_bytes
    if rows:
        outcome.rows = np.stack(rows)
        if not np.isfinite(outcome.rows).all():
            outcome.status = OUTCOME_NONFINITE
            outcome.error = "plant truth contains non-finite values"
            return outcome
    if outcome.status == OUTCOME_COMPLETED:
        try:
            outcome.race_metrics = dict(
                compute_race_metrics(scenario, zone_ids, initial_row, states)
            )
        except ValueError as error:
            outcome.status = OUTCOME_NONFINITE
            outcome.error = f"Issue 55 metrics rejected the episode: {error}"
    return outcome


def authoritative_evaluator(
    contract: HMCContract, *, verify_replay: bool = False, keep_artifacts: bool = False
) -> Evaluator:
    """The default evaluator: the real CPU plant and HMC."""

    def evaluate(
        scenario: Scenario, episode_id: str, interrupt_check: Callable[[], None] | None
    ) -> EpisodeOutcome:
        return run_authoritative_episode(
            scenario,
            contract,
            episode_id,
            interrupt_check=interrupt_check,
            verify_replay=verify_replay,
            keep_artifacts=keep_artifacts,
        )

    return evaluate


# ---------------------------------------------------------------------------
# Scoring (privileged truth, evaluator only)
# ---------------------------------------------------------------------------


def hazard_excess(rows: np.ndarray, hazards: Sequence[HazardFamily]) -> np.ndarray:
    """Signed distance past each warning threshold, in scale units.

    Shape [T, zones, families].  Positive means truth is beyond the threshold
    in the hazardous direction; negative is the remaining margin.
    """

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < ZONE_FIELD_COUNT:
        raise StressSearchError("truth rows must be [steps, targets]")
    zone_count = (values.shape[1] - 3) // ZONE_FIELD_COUNT
    zones = values[:, : zone_count * ZONE_FIELD_COUNT].reshape(
        values.shape[0], zone_count, ZONE_FIELD_COUNT
    )
    layers = []
    for hazard in hazards:
        sign = 1.0 if hazard.direction == "HIGH" else -1.0
        layers.append(sign * (zones[:, :, hazard.column] - hazard.warning_enter) / hazard.scale)
    return np.stack(layers, axis=2)


def score_against_twin(
    environment: SearchEnvironment, rows: np.ndarray, twin_rows: np.ndarray
) -> dict[str, Any]:
    """Declared metrics for a completed episode against the healthy twin.

    Raises :class:`StressSearchError` when shapes differ or any value is
    non-finite; the caller records that as a failure, never as a score.
    """

    fault_values = np.asarray(rows, dtype=np.float64)
    twin_values = np.asarray(twin_rows, dtype=np.float64)
    if fault_values.shape != twin_values.shape:
        raise StressSearchError("episode and twin truth have different shapes")
    if not np.isfinite(fault_values).all() or not np.isfinite(twin_values).all():
        raise StressSearchError("truth contains non-finite values")
    fault_excess = hazard_excess(fault_values, environment.hazards)
    twin_excess = hazard_excess(twin_values, environment.hazards)
    shift = fault_excess - twin_excess
    flat_index = int(np.argmax(shift))
    step_index, zone_index, family_index = np.unravel_index(flat_index, shift.shape)
    exposure = float(np.sum(np.maximum(0.0, fault_excess)))
    twin_exposure = float(np.sum(np.maximum(0.0, twin_excess)))
    metrics: dict[str, Any] = {
        "peak_hazard_shift": float(shift[step_index, zone_index, family_index]),
        "peak_hazard_shift_step": int(step_index) + 1,
        "peak_hazard_shift_zone": environment.zone_ids[int(zone_index)],
        "peak_hazard_shift_family": environment.hazards[int(family_index)].family,
        "hazard_exposure": exposure,
        "hazard_exposure_delta": exposure - twin_exposure,
        "peak_hazard_excess": float(np.max(fault_excess)),
        "residual_shift_at_end": float(np.max(shift[-1])),
    }
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise StressSearchError(f"metric {key} is non-finite")
    return metrics


def tier_met(tier: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    value = float(metrics[tier["metric"]])
    threshold = float(tier["value"])
    return value > threshold if tier["operator"] == ">" else value >= threshold


# ---------------------------------------------------------------------------
# Seeded draws
# ---------------------------------------------------------------------------


def unit_draw(*parts: str | int) -> float:
    """Stateless uniform draw in [0, 1) keyed by a namespaced string."""

    payload = "|".join((_RNG_NAMESPACE, *(str(part) for part in parts)))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _pick_index(unit: float, weights: Sequence[float]) -> int:
    total = float(sum(weights))
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight / total
        if unit < cumulative:
            return index
    return len(weights) - 1


def _unit_to_int(unit: float, low: int, high: int) -> int:
    span = high - low + 1
    return low + min(int(unit * span), span - 1)


class _ContinuousDimension:
    """Unit-interval dimension: uniform until first update, then truncated normal."""

    def __init__(self) -> None:
        self.uniform = True
        self.mean = 0.5
        self.sigma = _UNIFORM_STD

    def sample(self, key: tuple[str | int, ...]) -> float:
        if self.uniform:
            return unit_draw(*key, 0)
        value = self.mean
        for retry in range(_TRUNCATION_RETRIES):
            first = max(unit_draw(*key, retry, "a"), 1e-300)
            second = unit_draw(*key, retry, "b")
            normal = math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)
            value = self.mean + self.sigma * normal
            if 0.0 <= value < 1.0:
                return value
        return min(max(value, 0.0), math.nextafter(1.0, 0.0))

    def update(self, values: Sequence[float], smoothing: float, min_sigma: float) -> None:
        if not values:
            return
        mean = float(sum(values) / len(values))
        variance = float(sum((value - mean) ** 2 for value in values) / len(values))
        self.mean = (1.0 - smoothing) * self.mean + smoothing * mean
        self.sigma = max(min_sigma, (1.0 - smoothing) * self.sigma + smoothing * math.sqrt(variance))
        self.uniform = False

    def state(self) -> dict[str, Any]:
        return {"uniform": self.uniform, "mean": self.mean, "sigma": self.sigma}


class _CategoricalDimension:
    def __init__(self, size: int) -> None:
        self.probabilities = [1.0 / size] * size

    def sample(self, key: tuple[str | int, ...]) -> int:
        return _pick_index(unit_draw(*key), self.probabilities)

    def update(self, indices: Sequence[int], smoothing: float) -> None:
        if not indices:
            return
        size = len(self.probabilities)
        frequencies = [0.0] * size
        for index in indices:
            frequencies[index] += 1.0 / len(indices)
        blended = [
            (1.0 - smoothing) * old + smoothing * new
            for old, new in zip(self.probabilities, frequencies)
        ]
        floor = min(_CATEGORY_FLOOR, 1.0 / (2.0 * size))
        floored = [max(value, floor) for value in blended]
        total = sum(floored)
        self.probabilities = [value / total for value in floored]

    def state(self) -> list[float]:
        return list(self.probabilities)


class _Slot:
    def __init__(self, environment: SearchEnvironment) -> None:
        types = environment.spec.fault_types
        self.active = _CategoricalDimension(2)
        self.fault_type = _CategoricalDimension(len(types))
        self.zone = _CategoricalDimension(len(environment.zone_ids))
        self.damper = _CategoricalDimension(len(environment.damper_ids))
        self.shape = _CategoricalDimension(len(environment.spec.data["search_space"]["ramp_shapes"]))
        self.onset = _ContinuousDimension()
        self.duration = _ContinuousDimension()
        self.severity = _ContinuousDimension()


class CandidateSampler:
    """Seeded sampler over the declared search space.

    ``random_search`` never adapts.  ``cross_entropy`` starts from the same
    uniform distribution and, after each generation of ``population_size``
    attempts, moves toward that generation's elite candidates with fixed
    smoothing.  All state is a pure function of the recorded attempt outcomes,
    which is what makes resume exact.
    """

    def __init__(self, environment: SearchEnvironment, method: str, seed: str) -> None:
        if method not in METHODS:
            raise StressSearchError(f"unknown method {method!r}")
        if seed not in environment.spec.seeds:
            raise StressSearchError(f"seed {seed!r} is not declared in the spec")
        self._environment = environment
        self._method = method
        self._seed = seed
        self._space = environment.spec.data["search_space"]
        self._types = environment.spec.fault_types
        self._slots = [_Slot(environment) for _ in range(self._space["max_faults"])]
        self._settings = environment.spec.data["methods"][METHOD_CROSS_ENTROPY]
        self._generation: list[tuple[list[dict[str, Any]], float | None]] = []
        self.generation_index = 0
        self.updates_applied = 0
        self.updates_skipped = 0

    def _key(self, attempt: int, slot: int, name: str) -> tuple[str | int, ...]:
        return (
            self._environment.spec.spec_sha256,
            self._method,
            self._seed,
            attempt,
            slot,
            name,
        )

    def propose(self, attempt: int) -> list[dict[str, Any]]:
        """Raw slot-ordered parameters for an attempt (may violate end bounds)."""

        space = self._space
        faults: list[dict[str, Any]] = []
        for slot_index, slot in enumerate(self._slots):
            if slot_index >= space["min_faults"]:
                if slot.active.sample(self._key(attempt, slot_index, "active")) == 0:
                    continue
            fault_type = self._types[slot.fault_type.sample(self._key(attempt, slot_index, "type"))]
            target_kind = SUPPORTED_FAULT_TARGETS[fault_type]
            target: str | None = None
            if target_kind == ZONE_TARGET:
                target = self._environment.zone_ids[
                    slot.zone.sample(self._key(attempt, slot_index, "zone"))
                ]
            elif target_kind == DAMPER_TARGET:
                target = self._environment.damper_ids[
                    slot.damper.sample(self._key(attempt, slot_index, "damper"))
                ]
            onset = _unit_to_int(
                slot.onset.sample(self._key(attempt, slot_index, "onset")),
                *space["onset_step_range"],
            )
            duration = _unit_to_int(
                slot.duration.sample(self._key(attempt, slot_index, "duration")),
                *space["duration_steps_range"],
            )
            shape: str | None = None
            multiplier: float | None = None
            if fault_type not in _MULTIPLIER_FREE_TYPES:
                shape = space["ramp_shapes"][
                    slot.shape.sample(self._key(attempt, slot_index, "shape"))
                ]
                severity = slot.severity.sample(self._key(attempt, slot_index, "severity"))
                multiplier = severity_to_multiplier(
                    fault_type,
                    severity,
                    space["fault_types"][fault_type]["multiplier_range"],
                    space["multiplier_decimals"],
                )
            faults.append(
                {
                    "type": fault_type,
                    "target": target,
                    "onset_step": onset,
                    "duration_steps": duration,
                    "ramp_shape": shape,
                    "multiplier": multiplier,
                    "slot": slot_index,
                }
            )
        return faults

    def observe(self, raw_faults: list[dict[str, Any]], objective: float | None) -> None:
        """Feed back one attempt's outcome (``None`` when it has no score)."""

        if self._method != METHOD_CROSS_ENTROPY:
            return
        self._generation.append((raw_faults, objective))
        if len(self._generation) < self._settings["population_size"]:
            return
        scored = [
            (objective_value, index)
            for index, (_faults, objective_value) in enumerate(self._generation)
            if objective_value is not None
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        # A generation is ``population_size`` attempts of any disposition, which
        # keeps the sampler a pure function of the recorded attempts.  When no
        # more candidates were scored than there are elite slots the update
        # would apply no selection, so the distribution is left unchanged.
        if len(scored) > self._settings["elite_count"]:
            elites = [
                self._generation[index][0]
                for _value, index in scored[: self._settings["elite_count"]]
            ]
            self._update(elites)
            self.updates_applied += 1
        else:
            self.updates_skipped += 1
        self._generation = []
        self.generation_index += 1

    def _update(self, elites: Sequence[list[dict[str, Any]]]) -> None:
        smoothing = float(self._settings["smoothing"])
        min_sigma = float(self._settings["min_sigma"])
        space = self._space
        for slot_index, slot in enumerate(self._slots):
            members = [
                next((fault for fault in elite if fault["slot"] == slot_index), None)
                for elite in elites
            ]
            if slot_index >= space["min_faults"]:
                slot.active.update([0 if member is None else 1 for member in members], smoothing)
            present = [member for member in members if member is not None]
            if not present:
                continue
            slot.fault_type.update(
                [self._types.index(member["type"]) for member in present], smoothing
            )
            slot.zone.update(
                [
                    self._environment.zone_ids.index(member["target"])
                    for member in present
                    if SUPPORTED_FAULT_TARGETS[member["type"]] == ZONE_TARGET
                ],
                smoothing,
            )
            slot.damper.update(
                [
                    self._environment.damper_ids.index(member["target"])
                    for member in present
                    if SUPPORTED_FAULT_TARGETS[member["type"]] == DAMPER_TARGET
                ],
                smoothing,
            )
            with_multiplier = [
                member for member in present if member["type"] not in _MULTIPLIER_FREE_TYPES
            ]
            slot.shape.update(
                [space["ramp_shapes"].index(member["ramp_shape"]) for member in with_multiplier],
                smoothing,
            )
            slot.onset.update(
                [
                    _int_to_unit(member["onset_step"], *space["onset_step_range"])
                    for member in present
                ],
                smoothing,
                min_sigma,
            )
            slot.duration.update(
                [
                    _int_to_unit(member["duration_steps"], *space["duration_steps_range"])
                    for member in present
                ],
                smoothing,
                min_sigma,
            )
            slot.severity.update(
                [
                    multiplier_to_severity(
                        member["type"],
                        member["multiplier"],
                        space["fault_types"][member["type"]]["multiplier_range"],
                    )
                    for member in with_multiplier
                ],
                smoothing,
                min_sigma,
            )

    def state(self) -> dict[str, Any]:
        """Distribution state, for tests and the audit footer."""

        return {
            "generation_index": self.generation_index,
            "updates_applied": self.updates_applied,
            "updates_skipped_no_selection": self.updates_skipped,
            "pending_in_generation": len(self._generation),
            "slots": [
                {
                    "active": slot.active.state(),
                    "fault_type": slot.fault_type.state(),
                    "zone": slot.zone.state(),
                    "damper": slot.damper.state(),
                    "shape": slot.shape.state(),
                    "onset": slot.onset.state(),
                    "duration": slot.duration.state(),
                    "severity": slot.severity.state(),
                }
                for slot in self._slots
            ],
        }


def _int_to_unit(value: int, low: int, high: int) -> float:
    return (value - low + 0.5) / (high - low + 1)


def severity_to_multiplier(
    fault_type: str, severity: float, multiplier_range: Sequence[float], decimals: int
) -> float:
    """Map severity in [0, 1] onto the declared band; 1 is the most severe end."""

    low, high = float(multiplier_range[0]), float(multiplier_range[1])
    if fault_type in _SEVERITY_INCREASES_WITH_MULTIPLIER:
        value = low + severity * (high - low)
    else:
        value = high - severity * (high - low)
    return min(max(round(value, decimals), low), high)


def multiplier_to_severity(
    fault_type: str, multiplier: float, multiplier_range: Sequence[float]
) -> float:
    low, high = float(multiplier_range[0]), float(multiplier_range[1])
    if high == low:
        return 0.5
    fraction = (float(multiplier) - low) / (high - low)
    severity = fraction if fault_type in _SEVERITY_INCREASES_WITH_MULTIPLIER else 1.0 - fraction
    return min(max(severity, 0.0), 1.0)


def strip_slots(raw_faults: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{name: fault[name] for name in _FAULT_FIELDS} for fault in raw_faults]


# ---------------------------------------------------------------------------
# Write-once files and the hash-chained audit log
# ---------------------------------------------------------------------------


def write_once(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` exactly once; identical content is accepted.

    The bytes are written to a temporary sibling, synced, then linked into
    place without clobbering, so the final name never holds a partial file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise StressSearchOutputError(
                    f"refusing to overwrite existing output {path.name} with different content"
                ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class RunLock:
    """Exclusive single-writer lock on a run directory.

    It is an operating-system lock on an open handle, so it is released when
    the process ends for any reason and can never go stale.
    """

    def __init__(self, directory: Path) -> None:
        self._path = directory / LOCK_FILE_NAME
        self._handle: Any = None

    def __enter__(self) -> RunLock:
        handle = self._path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise StressSearchOutputError(
                "another process is writing this run directory; refusing a second writer"
            ) from None
        self._handle = handle
        return self

    def __exit__(self, *_exc: object) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            handle.close()


def _record_sha(sequence: int, previous: str, record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {"sequence": sequence, "previous_record_sha256": previous, "record": record}
    )


_TIMING_FIELDS = ("session_elapsed_seconds", "cumulative_wall_seconds")
_ZERO_TIMING = {"session_elapsed_seconds": 0.0, "cumulative_wall_seconds": 0.0}


def _validated_timing(timing: Any, line_number: int) -> float:
    """Check an entry's timing block and return its cumulative wall seconds."""

    if type(timing) is not dict or set(timing) != set(_TIMING_FIELDS):
        raise StressSearchAuditError(f"audit line {line_number} has a malformed timing block")
    for name in _TIMING_FIELDS:
        value = timing[name]
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
            raise StressSearchAuditError(
                f"audit line {line_number} timing {name} must be a finite number >= 0"
            )
    return float(timing["cumulative_wall_seconds"])


class AuditLog:
    """Append-only, hash-chained JSONL.

    ``timing`` carries wall-clock values and is deliberately outside the hash
    chain so the chained records stay deterministic.  It is still validated on
    load (present, finite, non-decreasing) so consumed wall time cannot
    silently fall back to zero.
    """

    def __init__(self, path: Path, entries: list[dict[str, Any]]) -> None:
        self.path = path
        self.entries = entries

    @classmethod
    def create(cls, path: Path) -> AuditLog:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb"):
                pass
        except FileExistsError:
            raise StressSearchOutputError(
                f"refusing to overwrite existing audit log {path.name}"
            ) from None
        return cls(path, [])

    @classmethod
    def load(cls, path: Path) -> AuditLog:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise StressSearchAuditError(f"cannot read audit log: {error}") from error
        if raw and not raw.endswith(b"\n"):
            raise StressSearchAuditError(
                "audit log ends with a partial record (the process died while appending); "
                "the log is write-once and is not repaired, so this run must be discarded"
            )
        entries: list[dict[str, Any]] = []
        previous = _NULL_SHA
        previous_wall = 0.0
        for line_number, line in enumerate(raw.split(b"\n")[:-1]):
            entry = strict_json_loads(
                line, label=f"audit line {line_number + 1}", error=StressSearchAuditError
            )
            _exact_fields(
                entry,
                ("sequence", "previous_record_sha256", "record", "record_sha256", "timing"),
                "audit entry",
                StressSearchAuditError,
            )
            try:
                canonical = canonical_bytes(entry)
            except StressSearchError as error:
                raise StressSearchAuditError(
                    f"audit line {line_number + 1} is not canonical JSON: {error}"
                ) from error
            if canonical != line:
                raise StressSearchAuditError(f"audit line {line_number + 1} is not canonical JSON")
            if entry["sequence"] != line_number or entry["previous_record_sha256"] != previous:
                raise StressSearchAuditError(f"audit chain is broken at line {line_number + 1}")
            if type(entry["record"]) is not dict:
                raise StressSearchAuditError(f"audit line {line_number + 1} is malformed")
            wall = _validated_timing(entry["timing"], line_number + 1)
            if wall < previous_wall:
                raise StressSearchAuditError(
                    f"audit wall time decreases at line {line_number + 1}"
                )
            previous_wall = wall
            expected = _record_sha(line_number, previous, entry["record"])
            if entry["record_sha256"] != expected:
                raise StressSearchAuditError(
                    f"audit record digest mismatch at line {line_number + 1}"
                )
            previous = expected
            entries.append(entry)
        return cls(path, entries)

    @property
    def head_sha256(self) -> str:
        return self.entries[-1]["record_sha256"] if self.entries else _NULL_SHA

    def records(self, kind: str) -> list[dict[str, Any]]:
        return [entry["record"] for entry in self.entries if entry["record"].get("kind") == kind]

    def append(self, record: Mapping[str, Any], timing: Mapping[str, Any]) -> None:
        sequence = len(self.entries)
        previous = self.head_sha256
        _validated_timing(dict(timing), sequence + 1)
        entry = {
            "sequence": sequence,
            "previous_record_sha256": previous,
            "record": json.loads(canonical_bytes(record)),
            "record_sha256": _record_sha(sequence, previous, record),
            "timing": dict(timing),
        }
        with self.path.open("ab") as handle:
            handle.write(canonical_bytes(entry) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# The search driver
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """What a search session did; ``end_reason`` is final unless cancelled."""

    output_dir: Path
    method: str
    seed: str
    end_reason: str
    final: bool
    simulation_calls_used: int
    candidate_attempts_used: int
    audit_head_sha256: str
    manifest_sha256: str | None = None


def _outcome_public(outcome: EpisodeOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "committed_steps": outcome.committed_steps,
        "terminal_reason": outcome.terminal_reason,
        "error": outcome.error,
        "trace_sha256": outcome.trace_sha256,
        "control_run_id": outcome.control_run_id,
        "final_state_sha256": outcome.final_state_sha256,
        "actuation_sha256": outcome.actuation_sha256,
        "emergency_action_steps": outcome.emergency_action_steps,
        "race_metrics": None if outcome.race_metrics is None else dict(outcome.race_metrics),
    }


_OUTCOME_STATUSES = (OUTCOME_COMPLETED, OUTCOME_TERMINAL, OUTCOME_NONFINITE, OUTCOME_ERROR)


def _safe_text(value: object) -> str:
    """Error text that is always encodable as canonical JSON."""

    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _sanitised_outcome(outcome: Any, expected_width: int | None = None) -> EpisodeOutcome:
    """Refuse an evaluator result that could not be recorded or scored faithfully."""

    if not isinstance(outcome, EpisodeOutcome):
        return EpisodeOutcome(
            status=OUTCOME_ERROR, committed_steps=0, error="evaluator returned no EpisodeOutcome"
        )
    problem: str | None = None
    if outcome.status not in _OUTCOME_STATUSES:
        problem = f"unknown outcome status {outcome.status!r}"
    elif type(outcome.committed_steps) is not int or outcome.committed_steps < 0:
        problem = "committed_steps is not a non-negative integer"
    else:
        try:
            canonical_bytes(_outcome_public(outcome))
        except (StressSearchError, TypeError, ValueError) as error:
            problem = f"outcome is not canonical JSON: {_safe_text(error)}"
    if problem is None and outcome.rows is not None:
        try:
            rows = np.asarray(outcome.rows, dtype=np.float64)
        except (TypeError, ValueError) as error:
            problem = f"truth rows are not numeric: {_safe_text(error)}"
        else:
            if rows.ndim != 2 or rows.shape[0] != outcome.committed_steps:
                problem = "truth rows are not [committed_steps, targets]"
            elif expected_width is not None and rows.shape[1] != expected_width:
                problem = f"truth rows have {rows.shape[1]} targets, expected {expected_width}"
            else:
                outcome.rows = rows
    if problem is None:
        return outcome
    return EpisodeOutcome(
        status=OUTCOME_NONFINITE, committed_steps=0, error=f"evaluator outcome rejected: {problem}"
    )


class _Session:
    """One live or replayed pass over the deterministic attempt sequence."""

    def __init__(
        self,
        *,
        environment: SearchEnvironment,
        method: str,
        seed: str,
        audit: AuditLog,
        evaluator: Evaluator,
        clock: Callable[[], float],
        cancel_requested: Callable[[], bool],
        recorded_entries: list[dict[str, Any]],
        prior_wall_seconds: float,
    ) -> None:
        self.environment = environment
        self.spec = environment.spec
        self.method = method
        self.seed = seed
        self.audit = audit
        self.evaluator = evaluator
        self.clock = clock
        self.cancel_requested = cancel_requested
        self.recorded = [
            entry["record"]
            for entry in recorded_entries
            if entry["record"]["kind"] in ("baseline", "candidate")
        ]
        self.recorded_index = 0
        self.prior_wall_seconds = prior_wall_seconds
        self.session_started = clock()
        self.sampler = CandidateSampler(environment, method, seed)
        self.simulation_calls = 0
        self.attempts = 0
        self.seen: dict[str, dict[str, Any]] = {}
        self.twin_rows: np.ndarray | None = None
        self.twin_rows_sha256: str | None = None
        self.budget = self.spec.data["budget"]
        self.truth_width = ZONE_FIELD_COUNT * len(environment.zone_ids) + 3

    # -- time -------------------------------------------------------------
    def session_elapsed(self) -> float:
        return max(0.0, self.clock() - self.session_started)

    def wall_used(self) -> float:
        return self.prior_wall_seconds + self.session_elapsed()

    def _timing(self) -> dict[str, Any]:
        return {
            "session_elapsed_seconds": round(self.session_elapsed(), 6),
            "cumulative_wall_seconds": round(self.wall_used(), 6),
        }

    def _interrupt_check(self) -> None:
        if self.cancel_requested():
            raise EpisodeInterrupted(INTERRUPT_CANCELLED)
        if self.wall_used() >= float(self.budget["max_wall_seconds"]):
            raise EpisodeInterrupted(INTERRUPT_DEADLINE)

    def _next_recorded(self, kind: str) -> dict[str, Any] | None:
        if self.recorded_index >= len(self.recorded):
            return None
        record = self.recorded[self.recorded_index]
        if record["kind"] != kind:
            raise StressSearchAuditError(
                f"audit sequence diverged: expected a {kind} record, found {record['kind']}"
            )
        self.recorded_index += 1
        return record

    def _counters(self) -> dict[str, Any]:
        return {
            "simulation_calls_used": self.simulation_calls,
            "candidate_attempts_used": self.attempts,
        }

    # -- baseline ---------------------------------------------------------
    def _evaluate(
        self, scenario: Scenario, episode_id: str, *, interruptible: bool = True
    ) -> EpisodeOutcome:
        try:
            outcome = self.evaluator(
                scenario, episode_id, self._interrupt_check if interruptible else None
            )
        except EpisodeInterrupted:
            raise
        except KeyboardInterrupt:
            raise EpisodeInterrupted(INTERRUPT_CANCELLED) from None
        except Exception as error:  # a simulator failure is a recorded outcome
            return EpisodeOutcome(
                status=OUTCOME_ERROR,
                committed_steps=0,
                error=_safe_text(f"{type(error).__name__}: {error}"),
            )
        return _sanitised_outcome(outcome, self.truth_width)

    def run_baseline(self) -> str | None:
        """Evaluate (or re-derive) the healthy twin.  Returns an end reason on failure."""

        twin = self.environment.twin_scenario
        twin_episode_id = canonical_sha256(
            {"harness_version": HARNESS_VERSION, "healthy_twin": twin.scenario_sha256}
        )
        recorded = self._next_recorded("baseline")
        # Twin truth is not stored in the audit, so a resumed session recomputes
        # it; the recorded digest proves it is the same baseline.  That
        # recomputation is not charged to the simulation budget a second time.
        # The single baseline episode is bounded and is never interrupted, so a
        # run always has either a complete baseline record or none.
        outcome = self._evaluate(twin, twin_episode_id, interruptible=False)
        rows_sha = (
            None
            if outcome.rows is None
            else hashlib.sha256(np.ascontiguousarray(outcome.rows).tobytes()).hexdigest()
        )
        record = {
            "kind": "baseline",
            "episode_id": twin_episode_id,
            "scenario_sha256": twin.scenario_sha256,
            "outcome": _outcome_public(outcome),
            "truth_rows_sha256": rows_sha,
            "simulation_calls_used": 1,
        }
        self.simulation_calls = 1
        if recorded is not None:
            if canonical_bytes(recorded) != canonical_bytes(record):
                raise StressSearchAuditError(
                    "healthy twin no longer reproduces the recorded baseline"
                )
        else:
            self.audit.append(record, self._timing())
        if (
            outcome.status != OUTCOME_COMPLETED
            or outcome.rows is None
            or outcome.committed_steps != self.spec.horizon_steps
            or not np.isfinite(np.asarray(outcome.rows, dtype=np.float64)).all()
        ):
            return END_BASELINE_FAILED
        self.twin_rows = outcome.rows
        self.twin_rows_sha256 = rows_sha
        return None

    # -- candidates -------------------------------------------------------
    def _candidate_record(self, attempt: int, raw: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate one attempt and return its audit record (no I/O)."""

        parameters = strip_slots(raw)
        record: dict[str, Any] = {
            "kind": "candidate",
            "attempt_index": attempt,
            "generation_index": self.sampler.generation_index,
            "parameters": parameters,
            "disposition": None,
            "reason": None,
            "episode_id": None,
            "causal_family_id": None,
            "causal_family_key": None,
            "scenario_sha256": None,
            "duplicate_of_attempt_index": None,
            "outcome": None,
            "metrics": None,
            "objective": None,
        }
        try:
            canonical = validate_candidate_parameters(self.environment, parameters)
        except StressSearchCandidateError as error:
            record["disposition"] = DISPOSITION_INVALID_PARAMETERS
            record["reason"] = str(error)
            return record
        identity = episode_identity(self.environment, canonical)
        record.update(identity)
        first = self.seen.get(identity["episode_id"])
        if first is not None:
            record["disposition"] = DISPOSITION_DUPLICATE
            record["reason"] = "same causal episode as an earlier attempt; not re-simulated"
            record["duplicate_of_attempt_index"] = first["attempt_index"]
            record["scenario_sha256"] = first["scenario_sha256"]
            record["objective"] = first["objective"]
            return record
        try:
            scenario = build_candidate_scenario(
                self.environment, canonical, identity["episode_id"]
            )
        except ScenarioValidationError as error:
            record["disposition"] = DISPOSITION_SCENARIO_VALIDATION
            record["reason"] = str(error)
            return record
        record["scenario_sha256"] = scenario.scenario_sha256
        outcome = self._evaluate(scenario, identity["episode_id"])
        record["outcome"] = _outcome_public(outcome)
        if outcome.status == OUTCOME_TERMINAL:
            record["disposition"] = DISPOSITION_TERMINAL
            record["reason"] = f"HMC terminal failure: {outcome.terminal_reason}"
        elif outcome.status == OUTCOME_NONFINITE:
            record["disposition"] = DISPOSITION_NONFINITE
            record["reason"] = outcome.error
        elif outcome.status != OUTCOME_COMPLETED:
            record["disposition"] = DISPOSITION_SIMULATOR_ERROR
            record["reason"] = outcome.error
        elif outcome.rows is None or outcome.committed_steps != self.spec.horizon_steps:
            record["disposition"] = DISPOSITION_SIMULATOR_ERROR
            record["reason"] = "evaluator reported completion without a full truth trajectory"
        else:
            try:
                metrics = score_against_twin(self.environment, outcome.rows, self.twin_rows)
            except (StressSearchError, ValueError, TypeError, IndexError) as error:
                record["disposition"] = DISPOSITION_NONFINITE
                record["reason"] = _safe_text(error)
            else:
                record["disposition"] = DISPOSITION_EVALUATED
                record["metrics"] = metrics
                record["objective"] = float(metrics[self.spec.objective_metric])
        return record

    def run(self) -> tuple[str, dict[str, Any] | None]:
        """Drive attempts until a budget, deadline or cancellation ends the session.

        Returns ``(end_reason, incomplete)`` where ``incomplete`` describes an
        attempt that was interrupted mid-episode and therefore not recorded as
        a candidate.
        """

        while True:
            replaying = self.recorded_index < len(self.recorded)
            if self.simulation_calls >= self.budget["max_simulation_calls"]:
                return END_SIMULATION_BUDGET, None
            if self.attempts >= self.budget["max_candidate_attempts"]:
                return END_ATTEMPT_BUDGET, None
            if not replaying:
                if self.cancel_requested():
                    return END_CANCELLED, None
                if self.wall_used() >= float(self.budget["max_wall_seconds"]):
                    return END_WALL_BUDGET, None
            attempt = self.attempts
            raw = self.sampler.propose(attempt)
            if replaying:
                record = self._next_recorded("candidate")
                assert record is not None
                if record["attempt_index"] != attempt or canonical_bytes(
                    record["parameters"]
                ) != canonical_bytes(strip_slots(raw)):
                    raise StressSearchAuditError(
                        f"audit sequence diverged at attempt {attempt}: the regenerated "
                        "candidate does not match the recorded one"
                    )
            else:
                try:
                    record = self._candidate_record(attempt, raw)
                except EpisodeInterrupted as interrupted:
                    incomplete = {
                        "attempt_index": attempt,
                        "parameters": strip_slots(raw),
                        "cause": interrupted.cause,
                    }
                    reason = (
                        END_WALL_BUDGET
                        if interrupted.cause == INTERRUPT_DEADLINE
                        else END_CANCELLED
                    )
                    return reason, incomplete
            self.attempts += 1
            if record["disposition"] in _SIMULATED_DISPOSITIONS:
                self.simulation_calls += 1
            if not replaying:
                record = {**record, **self._counters()}
                self.audit.append(record, self._timing())
            elif (
                record["simulation_calls_used"] != self.simulation_calls
                or record["candidate_attempts_used"] != self.attempts
            ):
                raise StressSearchAuditError(
                    f"audit budget counters are inconsistent at attempt {attempt}"
                )
            if record["episode_id"] is not None and record["disposition"] != DISPOSITION_DUPLICATE:
                if record["disposition"] in _SIMULATED_DISPOSITIONS:
                    self.seen[record["episode_id"]] = record
            self.sampler.observe(raw, record["objective"])


def _prior_wall_seconds(audit: AuditLog) -> float:
    """Wall time already consumed: the last (largest) cumulative value recorded."""

    if not audit.entries:
        return 0.0
    return float(audit.entries[-1]["timing"]["cumulative_wall_seconds"])


def _header_record(environment: SearchEnvironment, method: str, seed: str) -> dict[str, Any]:
    return {
        "kind": "run_header",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "method": method,
        "seed": seed,
        "identity": environment.identity(),
        "budget": dict(environment.spec.data["budget"]),
        "data_boundary": DATA_BOUNDARY,
    }


def _run_session(
    *,
    environment: SearchEnvironment,
    output_dir: Path,
    audit: AuditLog,
    method: str,
    seed: str,
    evaluator: Evaluator | None,
    confirm_evaluator: Evaluator | None,
    clock: Callable[[], float],
    cancel_requested: Callable[[], bool],
    runtime_environment: Mapping[str, Any] | None,
) -> RunSummary:
    recorded_entries = list(audit.entries)
    prior_wall = _prior_wall_seconds(audit)
    session = _Session(
        environment=environment,
        method=method,
        seed=seed,
        audit=audit,
        evaluator=evaluator or authoritative_evaluator(environment.contract),
        clock=clock,
        cancel_requested=cancel_requested,
        recorded_entries=recorded_entries,
        prior_wall_seconds=prior_wall,
    )
    session_index = len(audit.records("session_start"))
    audit.append(
        {
            "kind": "session_start",
            "session_index": session_index,
            "resumed_after_sequence": len(recorded_entries) - 1,
        },
        session._timing(),
    )
    incomplete: dict[str, Any] | None = None
    try:
        try:
            end_reason = session.run_baseline()
        except EpisodeInterrupted:
            # The baseline is not a candidate; nothing was recorded for it, so
            # the next session simply runs it again.
            end_reason = END_CANCELLED
        if end_reason is None:
            end_reason, incomplete = session.run()
    except BaseException as error:
        # Every session_start gets a matching session_end, even when the
        # session dies on an integrity error or a second Ctrl-C.  The failure
        # may have landed inside an append, so the in-memory log is first
        # re-read from disk; if the file no longer loads, nothing is appended.
        try:
            audit.entries = AuditLog.load(audit.path).entries
            audit.append(
                {
                    "kind": "session_end",
                    "session_index": session_index,
                    "end_reason": END_ABORTED,
                    "final": False,
                    "error_type": type(error).__name__,
                    **session._counters(),
                },
                session._timing(),
            )
        except Exception:  # the original failure is the one worth reporting
            pass
        raise
    if incomplete is not None:
        audit.append(
            {
                "kind": "incomplete",
                **incomplete,
                "note": "interrupted mid-episode; not counted against any budget, no trace",
                **session._counters(),
            },
            session._timing(),
        )
    final = end_reason in FINAL_END_REASONS
    audit.append(
        {
            "kind": "session_end",
            "session_index": session_index,
            "end_reason": end_reason,
            "final": final,
            **session._counters(),
        },
        session._timing(),
    )
    manifest_sha: str | None = None
    if final:
        audit.append(
            {
                "kind": "run_footer",
                "end_reason": end_reason,
                **session._counters(),
                "sampler_state": session.sampler.state(),
            },
            session._timing(),
        )
        manifest_sha = finalize_run(
            environment=environment,
            output_dir=output_dir,
            audit=audit,
            confirm_evaluator=confirm_evaluator,
            runtime_environment=runtime_environment,
        )
    return RunSummary(
        output_dir=output_dir,
        method=method,
        seed=seed,
        end_reason=end_reason,
        final=final,
        simulation_calls_used=session.simulation_calls,
        candidate_attempts_used=session.attempts,
        audit_head_sha256=audit.head_sha256,
        manifest_sha256=manifest_sha,
    )


def _never_cancelled() -> bool:
    return False


def run_search(
    *,
    spec: SearchSpec,
    repo_root: str | Path,
    output_dir: str | Path,
    method: str,
    seed: str,
    evaluator: Evaluator | None = None,
    confirm_evaluator: Evaluator | None = None,
    clock: Callable[[], float] = time.monotonic,
    cancel_requested: Callable[[], bool] = _never_cancelled,
    runtime_environment: Mapping[str, Any] | None = None,
) -> RunSummary:
    """Start a new search run in a directory that must not already exist."""

    if method not in METHODS:
        raise StressSearchError(f"unknown method {method!r}; choose one of {', '.join(METHODS)}")
    if seed not in spec.seeds:
        raise StressSearchError(f"seed {seed!r} is not declared in the spec")
    output = Path(output_dir)
    if output.exists():
        raise StressSearchOutputError(
            f"refusing to overwrite existing output directory {output.name}"
        )
    environment = build_environment(spec, repo_root)
    output.mkdir(parents=True)
    with RunLock(output):
        write_once(output / SPEC_FILE_NAME, canonical_bytes(spec.to_mapping()))
        audit = AuditLog.create(output / AUDIT_FILE_NAME)
        audit.append(_header_record(environment, method, seed), _ZERO_TIMING)
        return _run_session(
            environment=environment,
            output_dir=output,
            audit=audit,
            method=method,
            seed=seed,
            evaluator=evaluator,
            confirm_evaluator=confirm_evaluator,
            clock=clock,
            cancel_requested=cancel_requested,
            runtime_environment=runtime_environment,
        )


def load_run(
    output_dir: str | Path, repo_root: str | Path
) -> tuple[SearchEnvironment, AuditLog, dict[str, Any]]:
    """Load a run directory and verify it against the current environment."""

    output = Path(output_dir)
    spec = load_spec_file(output / SPEC_FILE_NAME)
    audit = AuditLog.load(output / AUDIT_FILE_NAME)
    if not audit.entries or audit.entries[0]["record"].get("kind") != "run_header":
        raise StressSearchAuditError("audit log has no run header")
    header = audit.entries[0]["record"]
    environment = build_environment(spec, repo_root)
    expected = _header_record(environment, header.get("method"), header.get("seed"))
    if canonical_bytes(header) != canonical_bytes(expected):
        raise StressSearchAuditError(
            "run identity does not match this environment (spec, fixture, HMC contract, "
            "hazard table or source files differ); refusing to continue"
        )
    if header["method"] not in METHODS or header["seed"] not in spec.seeds:
        raise StressSearchAuditError("run header names an undeclared method or seed")
    return environment, audit, header


def resume_search(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    evaluator: Evaluator | None = None,
    confirm_evaluator: Evaluator | None = None,
    clock: Callable[[], float] = time.monotonic,
    cancel_requested: Callable[[], bool] = _never_cancelled,
    runtime_environment: Mapping[str, Any] | None = None,
) -> RunSummary:
    """Continue an interrupted run.

    The recorded attempts are re-derived from the seed and checked one by one
    against the audit, restoring the sampler state and both budget counters
    exactly; nothing recorded is re-simulated and nothing is skipped.  Wall
    time already consumed stays consumed.  A run that already ended on a
    budget cannot be resumed, though an unfinished finalisation is completed.
    Only one process may write a run directory at a time.
    """

    output = Path(output_dir)
    if not output.is_dir():
        raise StressSearchAuditError("run directory does not exist")
    if not (output / SPEC_FILE_NAME).is_file() or not (output / AUDIT_FILE_NAME).is_file():
        raise StressSearchAuditError(
            f"not a run directory: {SPEC_FILE_NAME} and {AUDIT_FILE_NAME} are required"
        )
    with RunLock(output):
        environment, audit, header = load_run(output, repo_root)
        footers = audit.records("run_footer")
        if not footers:
            return _run_session(
                environment=environment,
                output_dir=output,
                audit=audit,
                method=header["method"],
                seed=header["seed"],
                evaluator=evaluator,
                confirm_evaluator=confirm_evaluator,
                clock=clock,
                cancel_requested=cancel_requested,
                runtime_environment=runtime_environment,
            )
        if (output / MANIFEST_FILE_NAME).exists():
            # Fails closed with a manifest error if the file is not this run's
            # valid manifest; otherwise the run is simply finished.
            load_run_manifest(output / MANIFEST_FILE_NAME)
            raise StressSearchAuditError("run is already complete; nothing to resume")
        manifest_sha = finalize_run(
            environment=environment,
            output_dir=output,
            audit=audit,
            confirm_evaluator=confirm_evaluator,
            runtime_environment=runtime_environment,
        )
        footer = footers[-1]
        return RunSummary(
            output_dir=output,
            method=header["method"],
            seed=header["seed"],
            end_reason=footer["end_reason"],
            final=True,
            simulation_calls_used=footer["simulation_calls_used"],
            candidate_attempts_used=footer["candidate_attempts_used"],
            audit_head_sha256=audit.head_sha256,
            manifest_sha256=manifest_sha,
        )


# ---------------------------------------------------------------------------
# Selection, confirmation and the candidate manifest
# ---------------------------------------------------------------------------

_CANDIDATE_SUMMARY_FIELDS = (
    "attempt_index",
    "generation_index",
    "parameters",
    "disposition",
    "reason",
    "episode_id",
    "causal_family_id",
    "scenario_sha256",
    "duplicate_of_attempt_index",
    "metrics",
    "objective",
    "simulation_calls_used",
)
_CASE_CORE_FIELDS = (
    "rank",
    "attempt_index",
    "generation_index",
    "simulation_calls_used",
    "episode_id",
    "causal_family_id",
    "causal_family_key",
    "parameters",
    "scenario_sha256",
    "objective",
    "metrics",
    "search_outcome",
)
_CASE_FIELDS = (*_CASE_CORE_FIELDS, "confirmation", "files")
_CONFIRMATION_FIELDS = (
    "status",
    "pipeline",
    "outcome",
    "replay_verified",
    "trace_sha256_identical",
    "metric_comparisons",
)
_FILE_ENTRY_FIELDS = ("relative_path", "sha256", "byte_length")
_CASE_FILE_NAMES = ("scenario.json", "control-trace.json", "actuation.jsonl")
_MANIFEST_FIELDS = (
    "schema_version",
    "data_boundary",
    "method",
    "seed",
    "identity",
    "spec",
    "budget_consumption",
    "audit",
    "selected_cases",
    "rejected_contenders",
    "causal_families",
    "split_rule",
    "replay_command",
    "gpu_screening",
    "manifest_sha256",
)
_CONFIRMATION_CALLS_FIELD = "confirmation_calls_outside_search_budget"


def select_candidates(
    spec: SearchSpec, candidates: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Apply the declared selection rule.

    Returns the selected candidate records in rank order and, for every other
    attempt, the reason it was not selected.
    """

    selection = spec.data["selection"]
    eligible = [
        candidate for candidate in candidates if candidate["disposition"] == DISPOSITION_EVALUATED
    ]
    ranked = sorted(eligible, key=lambda item: (-float(item["objective"]), item["episode_id"]))
    selected: list[dict[str, Any]] = []
    reasons: dict[int, str] = {}
    per_family: dict[str, int] = {}
    for candidate in ranked:
        attempt = candidate["attempt_index"]
        if float(candidate["objective"]) <= float(selection["minimum_objective_exclusive"]):
            reasons[attempt] = "objective not above minimum_objective_exclusive"
        elif len(selected) >= selection["shortlist_size"]:
            reasons[attempt] = "ranked below the shortlist"
        elif per_family.get(candidate["causal_family_id"], 0) >= selection["max_per_causal_family"]:
            reasons[attempt] = "a higher-ranked member of the same causal family was selected"
        else:
            selected.append(dict(candidate))
            per_family[candidate["causal_family_id"]] = (
                per_family.get(candidate["causal_family_id"], 0) + 1
            )
    for candidate in candidates:
        attempt = candidate["attempt_index"]
        if candidate["disposition"] != DISPOSITION_EVALUATED:
            reasons[attempt] = f"not eligible: {candidate['disposition']}"
    return selected, reasons


def _metrics_within_tolerance(
    reported: Mapping[str, Any], reproduced: Mapping[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    """Per-metric comparison; string and integer fields must match exactly."""

    comparisons = []
    for key in sorted(set(reported) | set(reproduced)):
        left, right = reported.get(key), reproduced.get(key)
        if type(left) is float and type(right) is float:
            difference = abs(left - right)
            comparisons.append(
                {
                    "metric": key,
                    "reported": left,
                    "reproduced": right,
                    "absolute_difference": difference,
                    "within_tolerance": difference <= tolerance,
                }
            )
        else:
            comparisons.append(
                {
                    "metric": key,
                    "reported": left,
                    "reproduced": right,
                    "absolute_difference": None,
                    "within_tolerance": left == right and type(left) is type(right),
                }
            )
    return comparisons


def _confirming(evaluator: Evaluator, scenario: Scenario, episode_id: str) -> EpisodeOutcome:
    """Call a confirmation evaluator; a failure is an outcome, never a crash."""

    try:
        outcome = evaluator(scenario, episode_id, None)
    except Exception as error:
        return EpisodeOutcome(
            status=OUTCOME_ERROR,
            committed_steps=0,
            error=_safe_text(f"{type(error).__name__}: {error}"),
        )
    return _sanitised_outcome(outcome)


def confirm_case(
    environment: SearchEnvironment,
    parameters: Sequence[Mapping[str, Any]],
    twin_rows: np.ndarray,
    evaluator: Evaluator,
) -> tuple[Scenario, dict[str, Any], EpisodeOutcome, dict[str, Any] | None]:
    """Re-run one case through the full authoritative pipeline."""

    if type(parameters) is not list or any(type(item) is not dict for item in parameters):
        raise StressSearchCandidateError("case parameters must be a list of objects")
    canonical = validate_candidate_parameters(environment, [dict(item) for item in parameters])
    identity = episode_identity(environment, canonical)
    scenario = build_candidate_scenario(environment, canonical, identity["episode_id"])
    outcome = _confirming(evaluator, scenario, identity["episode_id"])
    metrics: dict[str, Any] | None = None
    if (
        outcome.status == OUTCOME_COMPLETED
        and outcome.rows is not None
        and outcome.committed_steps == environment.spec.horizon_steps
    ):
        try:
            metrics = score_against_twin(environment, outcome.rows, twin_rows)
        except (StressSearchError, ValueError, TypeError, IndexError):
            metrics = None
    return scenario, identity, outcome, metrics


def evaluate_twin(environment: SearchEnvironment, evaluator: Evaluator) -> EpisodeOutcome:
    twin = environment.twin_scenario
    twin_episode_id = canonical_sha256(
        {"harness_version": HARNESS_VERSION, "healthy_twin": twin.scenario_sha256}
    )
    outcome = _confirming(evaluator, twin, twin_episode_id)
    if (
        outcome.status != OUTCOME_COMPLETED
        or outcome.rows is None
        or outcome.committed_steps != environment.spec.horizon_steps
        or not np.isfinite(np.asarray(outcome.rows, dtype=np.float64)).all()
    ):
        raise StressSearchError("healthy twin did not complete on the authoritative pipeline")
    return outcome


def replay_command(manifest_relative_path: str, output_relative_path: str) -> list[str]:
    """Exact, repo-relative command that replays a manifest."""

    return [
        "uv",
        "run",
        "--locked",
        "--python",
        "3.11",
        "--extra",
        "dev",
        "python",
        CLI_SCRIPT_RELATIVE_PATH,
        "replay",
        "--manifest",
        manifest_relative_path,
        "--output",
        output_relative_path,
    ]


def _case_payloads(scenario: Scenario, outcome: EpisodeOutcome) -> dict[str, bytes]:
    """The artifact bytes a confirmed or replayed case produces."""

    payloads = {"scenario.json": scenario.canonical_bytes}
    if outcome.trace_bytes is not None:
        payloads["control-trace.json"] = outcome.trace_bytes
    if outcome.actuation:
        payloads["actuation.jsonl"] = b"".join(
            canonical_bytes(row) + b"\n" for row in outcome.actuation
        )
    return payloads


def manifest_core(spec: SearchSpec, audit: AuditLog) -> dict[str, Any]:
    """Every manifest section that is a pure function of the spec and the audit.

    Finalisation writes these sections and replay recomputes them, which is
    what binds a manifest's case list, contender list and budget figures to
    the hash-chained audit beside it.
    """

    if not audit.entries or audit.entries[0]["record"].get("kind") != "run_header":
        raise StressSearchAuditError("audit log has no run header")
    header = audit.entries[0]["record"]
    footer_entries = [
        entry for entry in audit.entries if entry["record"].get("kind") == "run_footer"
    ]
    if len(footer_entries) != 1 or audit.entries[-1] is not footer_entries[0]:
        raise StressSearchAuditError(
            "a manifest needs an audit that ends with exactly one run footer"
        )
    footer = footer_entries[0]["record"]
    if header["identity"]["spec_sha256"] != spec.spec_sha256:
        raise StressSearchAuditError("audit header was produced under a different spec")
    candidates = audit.records("candidate")
    selected, reasons = select_candidates(spec, candidates)
    cases = [
        {
            "rank": rank,
            "attempt_index": candidate["attempt_index"],
            "generation_index": candidate["generation_index"],
            "simulation_calls_used": candidate["simulation_calls_used"],
            "episode_id": candidate["episode_id"],
            "causal_family_id": candidate["causal_family_id"],
            "causal_family_key": candidate["causal_family_key"],
            "parameters": candidate["parameters"],
            "scenario_sha256": candidate["scenario_sha256"],
            "objective": candidate["objective"],
            "metrics": candidate["metrics"],
            "search_outcome": candidate["outcome"],
        }
        for rank, candidate in enumerate(selected)
    ]
    families: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        family_id = candidate["causal_family_id"]
        if family_id is None:
            continue
        group = families.setdefault(
            family_id,
            {
                "causal_family_id": family_id,
                "causal_family_key": candidate["causal_family_key"],
                "episode_ids": [],
            },
        )
        if candidate["episode_id"] not in group["episode_ids"]:
            group["episode_ids"].append(candidate["episode_id"])
    selected_attempts = {case["attempt_index"] for case in cases}
    contenders = [
        {
            **{name: candidate[name] for name in _CANDIDATE_SUMMARY_FIELDS},
            "not_selected_reason": reasons.get(candidate["attempt_index"]),
        }
        for candidate in candidates
        if candidate["attempt_index"] not in selected_attempts
    ]
    dispositions = {name: 0 for name in DISPOSITIONS}
    for candidate in candidates:
        dispositions[candidate["disposition"]] += 1
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_boundary": DATA_BOUNDARY,
        "method": header["method"],
        "seed": header["seed"],
        "identity": header["identity"],
        "spec": spec.to_mapping(),
        "budget_consumption": {
            "end_reason": footer["end_reason"],
            "simulation_calls_used": footer["simulation_calls_used"],
            "candidate_attempts_used": footer["candidate_attempts_used"],
            "sessions": len(audit.records("session_start")),
            "incomplete_attempts": len(audit.records("incomplete")),
            "dispositions": dispositions,
        },
        "audit": {
            "relative_path": AUDIT_FILE_NAME,
            "footer_record_sha256": footer_entries[0]["record_sha256"],
        },
        "selected_cases": cases,
        "rejected_contenders": contenders,
        "causal_families": sorted(families.values(), key=lambda item: item["causal_family_id"]),
        "split_rule": SPLIT_RULE,
        "replay_command": replay_command(
            f"out/<run-directory>/{MANIFEST_FILE_NAME}", "out/<new-replay-directory>"
        ),
        "gpu_screening": dict(GPU_SCREENING_STATUS),
    }


def finalize_run(
    *,
    environment: SearchEnvironment,
    output_dir: Path,
    audit: AuditLog,
    confirm_evaluator: Evaluator | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> str:
    """Select, confirm on the full authoritative pipeline, and write the manifest.

    Confirmation re-runs each selected case with control-trace export, strict
    parse and replay.  A case whose confirmation disagrees with its search-time
    result stays in the manifest marked ``CONFIRMATION_MISMATCH``; it is
    reported, not dropped.  Confirmation runs are bounded by the shortlist size
    and are counted separately from the search budget.
    """

    spec = environment.spec
    manifest = manifest_core(spec, audit)
    evaluator = confirm_evaluator or authoritative_evaluator(
        environment.contract, verify_replay=True, keep_artifacts=True
    )
    tolerance = float(spec.data["replay"]["metric_absolute_tolerance"])
    require_trace = bool(spec.data["replay"]["require_identical_trace_sha256"])
    confirmation_calls = 0
    if manifest["selected_cases"]:
        twin_outcome = evaluate_twin(environment, evaluator)
        confirmation_calls += 1
        baseline_records = audit.records("baseline")
        twin_sha = hashlib.sha256(np.ascontiguousarray(twin_outcome.rows).tobytes()).hexdigest()
        if not baseline_records or baseline_records[0]["truth_rows_sha256"] != twin_sha:
            raise StressSearchAuditError("healthy twin changed between search and confirmation")
        for case in manifest["selected_cases"]:
            scenario, identity, outcome, metrics = confirm_case(
                environment, case["parameters"], twin_outcome.rows, evaluator
            )
            confirmation_calls += 1
            comparisons = (
                []
                if metrics is None
                else _metrics_within_tolerance(case["metrics"], metrics, tolerance)
            )
            trace_identical = (
                outcome.trace_sha256 is not None
                and outcome.trace_sha256 == case["search_outcome"]["trace_sha256"]
            )
            confirmed = (
                metrics is not None
                and bool(comparisons)
                and outcome.replay_verified
                and identity["episode_id"] == case["episode_id"]
                and scenario.scenario_sha256 == case["scenario_sha256"]
                and all(item["within_tolerance"] for item in comparisons)
                and (trace_identical or not require_trace)
            )
            case_dir = Path(CASES_DIRECTORY_NAME) / case["episode_id"]
            files: dict[str, Any] = {}
            for name, payload in sorted(_case_payloads(scenario, outcome).items()):
                write_once(output_dir / case_dir / name, payload)
                files[name] = {
                    "relative_path": (case_dir / name).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_length": len(payload),
                }
            case["confirmation"] = {
                "status": "CONFIRMED" if confirmed else "CONFIRMATION_MISMATCH",
                "pipeline": CONFIRMATION_PIPELINE,
                "outcome": _outcome_public(outcome),
                "replay_verified": outcome.replay_verified,
                "trace_sha256_identical": trace_identical,
                "metric_comparisons": comparisons,
            }
            case["files"] = files
    manifest["budget_consumption"][_CONFIRMATION_CALLS_FIELD] = confirmation_calls
    manifest["manifest_sha256"] = None
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if runtime_environment is not None:
        # Written before the manifest so a crash between the two is repaired
        # by re-running finalisation.  Unhashed provenance only.
        environment_path = output_dir / ENVIRONMENT_FILE_NAME
        if not environment_path.exists():
            write_once(
                environment_path,
                canonical_bytes(
                    {
                        "runtime_environment": dict(runtime_environment),
                        "source_identity": dict(environment.source_identity),
                        "wall_seconds_provisional": round(_prior_wall_seconds(audit), 3),
                        "timing_note": (
                            "wall time is provisional; other work may share the machine"
                        ),
                    }
                ),
            )
    write_once(output_dir / MANIFEST_FILE_NAME, canonical_bytes(manifest))
    return manifest["manifest_sha256"]


# ---------------------------------------------------------------------------
# Manifest loading and authoritative replay
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Strictly load a manifest file: closed schema, canonical bytes, self-hash.

    This checks the document on its own.  :func:`load_run_manifest` also binds
    it to the audit log beside it and is what replay and comparison use.
    """

    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise StressSearchManifestError(f"cannot read manifest: {error}") from error
    manifest = strict_json_loads(raw, label="manifest", error=StressSearchManifestError)
    _exact_fields(manifest, _MANIFEST_FIELDS, "manifest", StressSearchManifestError)
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise StressSearchManifestError("unsupported manifest schema_version")
    if canonical_bytes(manifest) != raw:
        raise StressSearchManifestError("manifest bytes are not canonical JSON")
    expected = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest["manifest_sha256"] != expected:
        raise StressSearchManifestError("manifest self-hash mismatch")
    if manifest["data_boundary"] != DATA_BOUNDARY:
        raise StressSearchManifestError("manifest data boundary statement was altered")
    if manifest["method"] not in METHODS:
        raise StressSearchManifestError("manifest names an unknown method")
    if manifest["gpu_screening"] != GPU_SCREENING_STATUS:
        raise StressSearchManifestError(
            "manifest claims GPU screening; this harness is CPU only"
        )
    if manifest["split_rule"] != SPLIT_RULE:
        raise StressSearchManifestError("manifest split rule was altered")
    if (
        type(manifest["selected_cases"]) is not list
        or type(manifest["rejected_contenders"]) is not list
    ):
        raise StressSearchManifestError("manifest case lists are malformed")
    for case in manifest["selected_cases"]:
        _exact_fields(case, _CASE_FIELDS, "selected case", StressSearchManifestError)
        _exact_fields(
            case["confirmation"],
            _CONFIRMATION_FIELDS,
            "case confirmation",
            StressSearchManifestError,
        )
        if case["confirmation"]["status"] not in ("CONFIRMED", "CONFIRMATION_MISMATCH"):
            raise StressSearchManifestError("case confirmation status is unknown")
        if not _is_sha(case["episode_id"]):
            raise StressSearchManifestError("case episode_id is not a sha256 digest")
        files = case["files"]
        if type(files) is not dict or "scenario.json" not in files:
            raise StressSearchManifestError("case files must include scenario.json")
        for name, entry in files.items():
            if name not in _CASE_FILE_NAMES:
                raise StressSearchManifestError(f"unexpected case file {name!r}")
            _exact_fields(entry, _FILE_ENTRY_FIELDS, "case file entry", StressSearchManifestError)
            if entry["relative_path"] != f"{CASES_DIRECTORY_NAME}/{case['episode_id']}/{name}":
                raise StressSearchManifestError(
                    f"case file {name!r} path leaves its case directory"
                )
            if not _is_sha(entry["sha256"]) or not _is_int(entry["byte_length"]):
                raise StressSearchManifestError(f"case file {name!r} digest entry is malformed")
    return manifest


_OUTCOME_PUBLIC_FIELDS = (
    "status",
    "committed_steps",
    "terminal_reason",
    "error",
    "trace_sha256",
    "control_run_id",
    "final_state_sha256",
    "actuation_sha256",
    "emergency_action_steps",
    "race_metrics",
)
_COMPARISON_FIELDS = (
    "metric",
    "reported",
    "reproduced",
    "absolute_difference",
    "within_tolerance",
)
CONFIRMATION_PIPELINE = "hmc_shadow_plant_export_parse_replay"


def _validate_confirmation(case: Mapping[str, Any], spec: SearchSpec) -> None:
    """A confirmation block must be internally consistent with its case.

    The block is produced by a re-run, not by the audit, so it is checked
    against its own parts: the status must follow from the recorded replay
    flag, trace identity and metric comparisons, and those comparisons must
    restate the audit-bound metrics of the case.
    """

    block = case["confirmation"]
    if block["pipeline"] != CONFIRMATION_PIPELINE:
        raise StressSearchManifestError("case confirmation pipeline is unknown")
    if type(block["replay_verified"]) is not bool or type(block["trace_sha256_identical"]) is not bool:
        raise StressSearchManifestError("case confirmation flags must be booleans")
    _exact_fields(
        block["outcome"], _OUTCOME_PUBLIC_FIELDS, "confirmation outcome", StressSearchManifestError
    )
    if block["outcome"]["status"] not in _OUTCOME_STATUSES:
        raise StressSearchManifestError("confirmation outcome status is unknown")
    trace = block["outcome"]["trace_sha256"]
    if block["trace_sha256_identical"] != (
        trace is not None and trace == case["search_outcome"]["trace_sha256"]
    ):
        raise StressSearchManifestError("confirmation trace identity flag is inconsistent")
    comparisons = block["metric_comparisons"]
    if type(comparisons) is not list:
        raise StressSearchManifestError("confirmation metric_comparisons must be a list")
    tolerance = float(spec.data["replay"]["metric_absolute_tolerance"])
    reproduced: dict[str, Any] = {}
    for item in comparisons:
        _exact_fields(item, _COMPARISON_FIELDS, "metric comparison", StressSearchManifestError)
        if type(item["metric"]) is not str or item["metric"] in reproduced:
            raise StressSearchManifestError("metric comparison names must be unique strings")
        if item["reproduced"] is not None:
            reproduced[item["metric"]] = item["reproduced"]
    if comparisons and canonical_bytes(comparisons) != canonical_bytes(
        _metrics_within_tolerance(case["metrics"], reproduced, tolerance)
    ):
        raise StressSearchManifestError(
            "confirmation metric comparisons do not restate the case metrics"
        )
    confirmed = (
        block["replay_verified"]
        and bool(comparisons)
        and all(item["within_tolerance"] for item in comparisons)
        and (
            block["trace_sha256_identical"]
            or not spec.data["replay"]["require_identical_trace_sha256"]
        )
    )
    if block["status"] != ("CONFIRMED" if confirmed else "CONFIRMATION_MISMATCH"):
        raise StressSearchManifestError(
            "case confirmation status does not follow from its recorded evidence"
        )


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest and bind it to the hash-chained audit beside it.

    Every section that the audit determines (method, seed, identity, budget
    figures, selected cases, rejected contenders, causal families) is
    recomputed from the audit and must match, so cases cannot be removed,
    relabelled or re-scored by editing the manifest and re-sealing its hash.
    """

    manifest_file = Path(path)
    manifest = load_manifest(manifest_file)
    try:
        spec = validate_spec_mapping(manifest["spec"])
        audit_binding = manifest["audit"]
        _exact_fields(
            audit_binding,
            ("relative_path", "footer_record_sha256"),
            "manifest audit binding",
            StressSearchManifestError,
        )
        if audit_binding["relative_path"] != AUDIT_FILE_NAME:
            raise StressSearchManifestError("manifest audit path is not the run's audit log")
        audit = AuditLog.load(manifest_file.parent / AUDIT_FILE_NAME)
        core = manifest_core(spec, audit)
        _exact_fields(
            manifest["budget_consumption"],
            (*core["budget_consumption"], _CONFIRMATION_CALLS_FIELD),
            "manifest budget_consumption",
            StressSearchManifestError,
        )
        for name, expected in core.items():
            actual = manifest[name]
            if name == "budget_consumption":
                actual = {
                    key: value for key, value in actual.items() if key != _CONFIRMATION_CALLS_FIELD
                }
            elif name == "selected_cases":
                actual = [
                    {key: case[key] for key in _CASE_CORE_FIELDS} for case in manifest[name]
                ]
            if canonical_bytes(actual) != canonical_bytes(expected):
                raise StressSearchManifestError(
                    f"manifest section {name!r} does not match the audit log"
                )
        # Confirmation is produced by a re-run, not by the audit, so it is
        # checked for internal consistency once the audit-bound parts agree.
        case_count = len(manifest["selected_cases"])
        calls = manifest["budget_consumption"][_CONFIRMATION_CALLS_FIELD]
        if not _is_int(calls) or calls != (case_count + 1 if case_count else 0):
            raise StressSearchManifestError(
                "confirmation call count must be an integer: one twin run plus one per selected case"
            )
        for case in manifest["selected_cases"]:
            _validate_confirmation(case, spec)
    except StressSearchManifestError:
        raise
    except StressSearchError as error:
        raise StressSearchManifestError(f"manifest does not bind to its audit: {error}") from error
    except (KeyError, TypeError, AttributeError, IndexError, ValueError, ArithmeticError) as error:
        raise StressSearchManifestError(f"manifest is malformed: {error!r}") from error
    return manifest


def _stored_case_files(
    manifest_file: Path, case: Mapping[str, Any]
) -> dict[str, bytes]:
    """Read a case's stored artifacts, checking each against its digest entry."""

    stored: dict[str, bytes] = {}
    for name, entry in sorted(case["files"].items()):
        target = manifest_file.parent / CASES_DIRECTORY_NAME / case["episode_id"] / name
        if not target.is_file():
            raise StressSearchManifestError(
                f"stored {name} for case {case['episode_id']!r} is missing"
            )
        data = target.read_bytes()
        if (
            len(data) != entry["byte_length"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise StressSearchManifestError(
                f"stored {name} for case {case['episode_id']!r} does not match its recorded digest"
            )
        stored[name] = data
    return stored


def replay_manifest(
    *,
    manifest_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path | None = None,
    evaluator: Evaluator | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay every selected case and compare with the reported metrics.

    Fails closed (raises) when the manifest is malformed, does not bind to the
    audit beside it, has altered or missing case files, or was produced by a
    different spec, fixture, HMC contract or source tree.  The receipt
    ``status`` is ``PASS`` only if there is at least one selected case and every
    case reproduces its metrics within the declared tolerance, its trace
    identity (when required) and its stored artifacts.  A manifest with no
    selected cases yields ``NO_CASES``, never ``PASS``.
    """

    output = None if output_dir is None else Path(output_dir)
    if output is not None and output.exists():
        raise StressSearchOutputError(
            f"refusing to overwrite existing output directory {output.name}"
        )
    manifest_file = Path(manifest_path)
    manifest = load_run_manifest(manifest_file)
    spec = validate_spec_mapping(manifest["spec"])
    if manifest["seed"] not in spec.seeds:
        raise StressSearchManifestError("manifest seed is not declared in its spec")
    try:
        environment = build_environment(spec, repo_root)
    except StressSearchSpecError as error:
        raise StressSearchManifestError(
            f"manifest does not match this environment: {error}"
        ) from error
    if canonical_bytes(manifest["identity"]) != canonical_bytes(environment.identity()):
        raise StressSearchManifestError(
            "manifest identity does not match this environment (spec, fixture, HMC "
            "contract, hazard table or source files differ)"
        )
    runner = evaluator or authoritative_evaluator(
        environment.contract, verify_replay=True, keep_artifacts=True
    )
    tolerance = float(spec.data["replay"]["metric_absolute_tolerance"])
    require_trace = bool(spec.data["replay"]["require_identical_trace_sha256"])
    objective_metric = spec.objective_metric
    results: list[dict[str, Any]] = []
    twin_outcome = evaluate_twin(environment, runner) if manifest["selected_cases"] else None
    for case in manifest["selected_cases"]:
        stored = _stored_case_files(manifest_file, case)
        try:
            scenario, identity, outcome, metrics = confirm_case(
                environment, case["parameters"], twin_outcome.rows, runner
            )
        except (StressSearchCandidateError, ScenarioValidationError) as error:
            raise StressSearchManifestError(
                f"selected case {case['episode_id']!r} is outside the declared search space: {error}"
            ) from error
        if (
            identity["episode_id"] != case["episode_id"]
            or identity["causal_family_id"] != case["causal_family_id"]
            or scenario.scenario_sha256 != case["scenario_sha256"]
        ):
            raise StressSearchManifestError(
                f"selected case {case['episode_id']!r} identity does not match its parameters"
            )
        if stored["scenario.json"] != scenario.canonical_bytes:
            raise StressSearchManifestError(
                f"stored scenario for case {case['episode_id']!r} does not match its parameters"
            )
        regenerated = _case_payloads(scenario, outcome)
        artifacts_identical = regenerated == stored
        comparisons = (
            [] if metrics is None else _metrics_within_tolerance(case["metrics"], metrics, tolerance)
        )
        objective_consistent = (
            type(case["metrics"]) is dict
            and case["metrics"].get(objective_metric) == case["objective"]
        )
        trace_identical = (
            outcome.trace_sha256 is not None
            and outcome.trace_sha256 == case["search_outcome"]["trace_sha256"]
        )
        reproduced = (
            metrics is not None
            and bool(comparisons)
            and objective_consistent
            and outcome.replay_verified
            and all(item["within_tolerance"] for item in comparisons)
            and ((trace_identical and artifacts_identical) or not require_trace)
        )
        results.append(
            {
                "episode_id": case["episode_id"],
                "status": "REPRODUCED" if reproduced else "MISMATCH",
                "replay_verified": outcome.replay_verified,
                "outcome": _outcome_public(outcome),
                "stored_scenario_bytes_identical": True,
                "stored_artifacts_identical": artifacts_identical,
                "objective_consistent_with_metrics": objective_consistent,
                "trace_sha256_identical": trace_identical,
                "trace_sha256_required": require_trace,
                "metric_comparisons": comparisons,
            }
        )
    if not results:
        status = REPLAY_STATUS_NO_CASES
    elif all(item["status"] == "REPRODUCED" for item in results):
        status = REPLAY_STATUS_PASS
    else:
        status = REPLAY_STATUS_FAIL
    receipt: dict[str, Any] = {
        "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
        "data_boundary": DATA_BOUNDARY,
        "manifest_sha256": manifest["manifest_sha256"],
        "identity": environment.identity(),
        "metric_absolute_tolerance": tolerance,
        "case_count": len(results),
        "reproduced_count": sum(1 for item in results if item["status"] == "REPRODUCED"),
        "status": status,
        "cases": results,
        "runtime_environment": None if runtime_environment is None else dict(runtime_environment),
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in ("receipt_sha256", "runtime_environment")
        }
    )
    if output is not None:
        try:
            output.mkdir(parents=True)
        except FileExistsError:
            raise StressSearchOutputError(
                f"refusing to overwrite existing output directory {output.name}"
            ) from None
        write_once(output / REPLAY_RECEIPT_FILE_NAME, canonical_bytes(receipt))
    return receipt


# ---------------------------------------------------------------------------
# Matched-budget comparison
# ---------------------------------------------------------------------------


def summarize_run(
    manifest: Mapping[str, Any], *, max_simulation_calls: int | None = None
) -> dict[str, Any]:
    """Discovery and cost figures for one run, computed from its manifest.

    With ``max_simulation_calls`` the figures cover only the attempts made
    while the simulation-call counter was at or below that value, which is how
    two runs that consumed different numbers of calls are compared at a common
    count.  Only ``EVALUATED`` candidates can meet a discovery tier; terminal
    failures, non-finite outputs and simulator errors are counted separately.
    """

    spec = manifest["spec"]
    objective_metric = OBJECTIVES[spec["objective"]["objective_id"]]
    attempts = sorted(
        [
            *manifest["rejected_contenders"],
            *(
                {
                    **{name: case.get(name) for name in _CANDIDATE_SUMMARY_FIELDS},
                    "disposition": DISPOSITION_EVALUATED,
                }
                for case in manifest["selected_cases"]
            ),
        ],
        key=lambda item: item["attempt_index"],
    )
    if max_simulation_calls is not None:
        attempts = [
            item for item in attempts if item["simulation_calls_used"] <= max_simulation_calls
        ]
    evaluated = [item for item in attempts if item["disposition"] == DISPOSITION_EVALUATED]
    terminal = [item for item in attempts if item["disposition"] == DISPOSITION_TERMINAL]
    consumption = manifest["budget_consumption"]
    calls = (
        consumption["simulation_calls_used"]
        if max_simulation_calls is None
        else min(consumption["simulation_calls_used"], max_simulation_calls)
    )
    tiers = []
    for tier in spec["discovery_tiers"]:
        hits = [item for item in evaluated if tier_met(tier, item["metrics"])]
        families = len({item["causal_family_id"] for item in hits})
        tiers.append(
            {
                "tier_id": tier["tier_id"],
                "discovered_episodes": len({item["episode_id"] for item in hits}),
                "discovered_causal_families": families,
                "simulation_calls_to_first_discovery": min(
                    (item["simulation_calls_used"] for item in hits), default=None
                ),
                "discovered_families_per_simulation_call": families / calls if calls else None,
            }
        )
    dispositions = {name: 0 for name in DISPOSITIONS}
    for item in attempts:
        dispositions[item["disposition"]] += 1
    return {
        "method": manifest["method"],
        "seed": manifest["seed"],
        "manifest_sha256": manifest["manifest_sha256"],
        "end_reason": consumption["end_reason"],
        "simulation_calls_used": calls,
        "candidate_attempts_used": len(attempts),
        "sessions": consumption["sessions"],
        "incomplete_attempts": consumption["incomplete_attempts"],
        _CONFIRMATION_CALLS_FIELD: consumption[_CONFIRMATION_CALLS_FIELD],
        "dispositions": dispositions,
        "evaluated_candidates": len(evaluated),
        "distinct_causal_families_evaluated": len({item["causal_family_id"] for item in evaluated}),
        "terminal_failure_episodes": len({item["episode_id"] for item in terminal}),
        "objective_metric": objective_metric,
        "best_objective": max((float(item["objective"]) for item in evaluated), default=None),
        "selected_cases": len(manifest["selected_cases"]),
        "confirmed_cases": sum(
            1 for case in manifest["selected_cases"] if case["confirmation"]["status"] == "CONFIRMED"
        ),
        "tiers": tiers,
    }


def _aggregate(values: Sequence[float | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(present),
        "mean": sum(present) / len(present),
        "min": min(present),
        "max": max(present),
    }


def compare_runs(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Matched-budget comparison of both methods over every declared seed.

    Fails closed unless the manifests share one spec and identity, cover
    exactly one run per (method, declared seed), and every run ended on a
    deterministic budget (simulation calls or candidate attempts).  A run cut
    short by wall time or a failed baseline depends on the machine and is
    refused, so a missing, negative or truncated run cannot slip in or be left
    out.  Per-seed paired figures are computed at the common simulation-call
    count of the two methods.  The result is descriptive; it makes no
    inferential claim that either method is better.
    """

    if not manifests:
        raise StressSearchManifestError("comparison requires at least one manifest")
    reference = manifests[0]
    try:
        spec = validate_spec_mapping(reference["spec"])
    except StressSearchSpecError as error:
        raise StressSearchManifestError(f"comparison manifest spec is invalid: {error}") from error
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for manifest in manifests:
        if canonical_bytes(manifest["identity"]) != canonical_bytes(reference["identity"]):
            raise StressSearchManifestError(
                "comparison manifests do not share one spec, fixture and source identity"
            )
        key = (manifest["method"], manifest["seed"])
        if key in by_key:
            raise StressSearchManifestError(f"duplicate run for method and seed {key}")
        by_key[key] = manifest
    expected = {(method, seed) for method in METHODS for seed in spec.seeds}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise StressSearchManifestError(
            f"comparison needs every declared seed for both methods; missing {missing}, unexpected {extra}"
        )
    unmatched = sorted(
        key
        for key, manifest in by_key.items()
        if manifest["budget_consumption"]["end_reason"] not in DETERMINISTIC_END_REASONS
    )
    if unmatched:
        raise StressSearchManifestError(
            "comparison refuses runs that did not end on a deterministic budget (wall-time "
            f"stops and failed baselines depend on the machine): {unmatched}"
        )
    runs = {key: summarize_run(manifest) for key, manifest in by_key.items()}
    tier_ids = [tier["tier_id"] for tier in spec.data["discovery_tiers"]]
    per_method: dict[str, Any] = {}
    for method in METHODS:
        method_runs = [runs[(method, seed)] for seed in spec.seeds]
        per_method[method] = {
            "simulation_calls_used": _aggregate([run["simulation_calls_used"] for run in method_runs]),
            "candidate_attempts_used": _aggregate(
                [run["candidate_attempts_used"] for run in method_runs]
            ),
            "best_objective": _aggregate([run["best_objective"] for run in method_runs]),
            "terminal_failure_episodes": _aggregate(
                [run["terminal_failure_episodes"] for run in method_runs]
            ),
            "tiers": {
                tier_id: {
                    "seeds_with_discovery": sum(
                        1
                        for run in method_runs
                        if run["tiers"][index]["discovered_causal_families"] > 0
                    ),
                    "discovered_causal_families": _aggregate(
                        [run["tiers"][index]["discovered_causal_families"] for run in method_runs]
                    ),
                    "discovered_families_per_simulation_call": _aggregate(
                        [
                            run["tiers"][index]["discovered_families_per_simulation_call"]
                            for run in method_runs
                        ]
                    ),
                    "simulation_calls_to_first_discovery": _aggregate(
                        [
                            run["tiers"][index]["simulation_calls_to_first_discovery"]
                            for run in method_runs
                        ]
                    ),
                }
                for index, tier_id in enumerate(tier_ids)
            },
        }
    paired = []
    for seed in spec.seeds:
        calls = {method: runs[(method, seed)]["simulation_calls_used"] for method in METHODS}
        common = min(calls.values())
        random_run = summarize_run(by_key[(METHOD_RANDOM, seed)], max_simulation_calls=common)
        adaptive_run = summarize_run(
            by_key[(METHOD_CROSS_ENTROPY, seed)], max_simulation_calls=common
        )
        difference = (
            None
            if random_run["best_objective"] is None or adaptive_run["best_objective"] is None
            else adaptive_run["best_objective"] - random_run["best_objective"]
        )
        paired.append(
            {
                "seed": seed,
                "simulation_calls_used": calls,
                "simulation_calls_matched": len(set(calls.values())) == 1,
                "compared_at_simulation_calls": common,
                "best_objective": {
                    METHOD_RANDOM: random_run["best_objective"],
                    METHOD_CROSS_ENTROPY: adaptive_run["best_objective"],
                },
                "best_objective_cross_entropy_minus_random": difference,
                "tier_family_difference_cross_entropy_minus_random": {
                    tier_id: adaptive_run["tiers"][index]["discovered_causal_families"]
                    - random_run["tiers"][index]["discovered_causal_families"]
                    for index, tier_id in enumerate(tier_ids)
                },
            }
        )
    differences = [
        item["best_objective_cross_entropy_minus_random"]
        for item in paired
        if item["best_objective_cross_entropy_minus_random"] is not None
    ]
    comparison: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "data_boundary": DATA_BOUNDARY,
        "identity": reference["identity"],
        "spec_id": spec.data["spec_id"],
        "spec_status": spec.data["status"],
        "objective": spec.data["objective"],
        "discovery_tiers": spec.data["discovery_tiers"],
        "budget": spec.data["budget"],
        "seeds": list(spec.seeds),
        "all_seeds_used_equal_simulation_calls": all(
            item["simulation_calls_matched"] for item in paired
        ),
        "runs": [runs[(method, seed)] for method in METHODS for seed in spec.seeds],
        "per_method": per_method,
        "paired_by_seed": paired,
        "paired_best_objective_summary": {
            "seeds_cross_entropy_higher": sum(1 for value in differences if value > 0.0),
            "seeds_random_higher": sum(1 for value in differences if value < 0.0),
            "seeds_tied": sum(1 for value in differences if value == 0.0),
            "mean_difference": sum(differences) / len(differences) if differences else None,
        },
        "interpretation": (
            "Descriptive matched-budget figures over the declared seeds only. No "
            "significance test was run and no claim is made that either method is "
            "better in general. Seeds are repeated searches on one fixture, not "
            "independent populations. Best objective is the quantity the adaptive "
            "method optimises, so it is the comparison most favourable to it; the "
            "discovery tiers are the declared outcome. Only fully evaluated "
            "candidates can meet a tier; terminal failures, non-finite outputs and "
            "simulator errors are counted separately and are never selected."
        ),
        "comparison_sha256": None,
    }
    comparison["comparison_sha256"] = canonical_sha256(
        {key: value for key, value in comparison.items() if key != "comparison_sha256"}
    )
    return comparison


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Human-readable matched-budget report."""

    lines = [
        "# Fault stress-search: matched-budget comparison",
        "",
        f"Spec `{comparison['spec_id']}` (status `{comparison['spec_status']}`), "
        f"objective `{comparison['objective']['objective_id']}`.",
        "",
        f"Budget per run: {comparison['budget']['max_simulation_calls']} simulation calls, "
        f"{comparison['budget']['max_candidate_attempts']} candidate attempts, "
        f"{comparison['budget']['max_wall_seconds']} s wall time. "
        f"Seeds: {', '.join(comparison['seeds'])}.",
        "",
        comparison["interpretation"],
        "",
        "## Per run",
        "",
        "| Method | Seed | End reason | Sim calls | Attempts | Evaluated | Families | Terminal | Best objective | "
        + " | ".join(
            f"{tier['tier_id']} families (first at call)" for tier in comparison["runs"][0]["tiers"]
        )
        + " |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        + " | ".join("---" for _ in comparison["runs"][0]["tiers"])
        + " |",
    ]
    for run in comparison["runs"]:
        tier_cells = " | ".join(
            f"{tier['discovered_causal_families']} ({_format(tier['simulation_calls_to_first_discovery'])})"
            for tier in run["tiers"]
        )
        lines.append(
            f"| {run['method']} | {run['seed']} | {run['end_reason']} | "
            f"{run['simulation_calls_used']} | {run['candidate_attempts_used']} | "
            f"{run['evaluated_candidates']} | {run['distinct_causal_families_evaluated']} | "
            f"{run['terminal_failure_episodes']} | "
            f"{_format(run['best_objective'])} | {tier_cells} |"
        )
    lines += ["", "## Per method (mean, min to max over seeds)", ""]
    lines.append("| Method | Best objective | " + " | ".join(
        f"{tier_id}: seeds with discovery; families per call"
        for tier_id in comparison["per_method"][METHOD_RANDOM]["tiers"]
    ) + " |")
    lines.append("| --- | --- | " + " | ".join(
        "---" for _ in comparison["per_method"][METHOD_RANDOM]["tiers"]
    ) + " |")
    for method in METHODS:
        summary = comparison["per_method"][method]
        best = summary["best_objective"]
        cells = " | ".join(
            f"{tier['seeds_with_discovery']} of {len(comparison['seeds'])}; "
            f"{_format(tier['discovered_families_per_simulation_call']['mean'], 5)}"
            for tier in summary["tiers"].values()
        )
        lines.append(
            f"| {method} | {_format(best['mean'])} ({_format(best['min'])} to {_format(best['max'])}) | {cells} |"
        )
    paired = comparison["paired_best_objective_summary"]
    lines += [
        "",
        "## Paired by seed (best objective at the common simulation-call count)",
        "",
        "| Seed | Compared at calls | Calls matched | Random | Cross-entropy | Difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparison["paired_by_seed"]:
        lines.append(
            f"| {item['seed']} | {item['compared_at_simulation_calls']} | "
            f"{'yes' if item['simulation_calls_matched'] else 'no'} | "
            f"{_format(item['best_objective'][METHOD_RANDOM])} | "
            f"{_format(item['best_objective'][METHOD_CROSS_ENTROPY])} | "
            f"{_format(item['best_objective_cross_entropy_minus_random'])} |"
        )
    lines += [
        "",
        f"Cross-entropy higher on {paired['seeds_cross_entropy_higher']} seeds, random higher on "
        f"{paired['seeds_random_higher']}, tied on {paired['seeds_tied']}; mean difference "
        f"{_format(paired['mean_difference'])}.",
        "",
        "Tier definitions: "
        + "; ".join(
            f"`{tier['tier_id']}` is `{tier['metric']} {tier['operator']} {tier['value']}`"
            for tier in comparison["discovery_tiers"]
        )
        + ". 'Families' counts distinct causal families; 'first at call' is the "
        "simulation-call count (including the one healthy-twin call) at the first discovery.",
        "",
        comparison["data_boundary"],
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "AUDIT_FILE_NAME",
    "AuditLog",
    "CandidateSampler",
    "DATA_BOUNDARY",
    "EpisodeInterrupted",
    "EpisodeOutcome",
    "HARNESS_VERSION",
    "MANIFEST_FILE_NAME",
    "METHODS",
    "METHOD_CROSS_ENTROPY",
    "METHOD_RANDOM",
    "RunLock",
    "RunSummary",
    "SearchEnvironment",
    "SearchSpec",
    "StressSearchAuditError",
    "StressSearchCandidateError",
    "StressSearchError",
    "StressSearchManifestError",
    "StressSearchOutputError",
    "StressSearchSpecError",
    "authoritative_evaluator",
    "build_candidate_scenario",
    "build_environment",
    "build_fixture_data",
    "canonical_bytes",
    "canonical_faults",
    "canonical_sha256",
    "collect_runtime_environment",
    "collect_source_identity",
    "compare_runs",
    "episode_identity",
    "finalize_run",
    "hazard_excess",
    "hazard_table",
    "load_manifest",
    "load_run",
    "load_run_manifest",
    "manifest_core",
    "load_spec_file",
    "render_comparison_markdown",
    "replay_manifest",
    "resume_search",
    "run_authoritative_episode",
    "run_search",
    "score_against_twin",
    "select_candidates",
    "summarize_run",
    "unit_draw",
    "validate_candidate_parameters",
    "validate_spec_mapping",
    "write_once",
]
