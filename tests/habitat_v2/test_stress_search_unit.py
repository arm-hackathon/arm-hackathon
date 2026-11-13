"""Unit and property tests for the Issue 93 fault stress-search harness.

Scope and honesty boundary
--------------------------
Spec validation, candidate bounds, sampler determinism, identity, audit,
selection, manifest and comparison tests exercise the real library code
directly.

Every test that drives ``run_search`` / ``resume_search`` in this module uses a
FABRICATED evaluator (``FakeEvaluator``).  It never runs the plant or the HMC:
it tiles one real healthy-twin truth row (from ``project_true_targets`` on the
fixture's ``initial_state``) and adds a small CO2 offset that is a pure
function of the fault parameters.  Those tests check the driver's bookkeeping
only (budgets, audit records, deduplication, deadline, cancellation, resume,
failure recording).  They are NOT integration evidence, say nothing about the
physics, and their scores, selections and comparisons are meaningless as
results.  Real-simulator coverage lives in a separate integration test file.

Nothing here reads a BDM-v1 roster family, the custody registry or any
protected population, and everything is written under ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from aeolus.habitat_v2 import stress_search as ss
from aeolus.habitat_v2.bdm_v1_families import (
    GENERATOR_FAULT_MULTIPLIER_BANDS,
    SENSOR_SEED_BASE,
)
from aeolus.habitat_v2.forecast_issue55_race import (
    ZONE_FIELD_BOUNDS,
    ZONE_FIELD_COUNT,
    project_true_targets,
    scenario_zone_order,
)
from aeolus.habitat_v2.physics import initial_state
from aeolus.habitat_v2.scenario import Scenario, ScenarioValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "scripts" / "stress_search_spec_v1.json"
DECLARED_EVIDENCE_SEEDS = (
    "issue93-s1",
    "issue93-s2",
    "issue93-s3",
    "issue93-s4",
    "issue93-s5",
)
TEST_SEEDS = ["test-s1", "test-s2"]
CO2_COLUMN = [name for name, *_rest in ZONE_FIELD_BOUNDS].index("co2_ppm")
ZONE_A = "crew_quarters_a"
ZONE_B = "laboratory"
DAMPER_A = "crew_quarters_a_supply_damper"
DAMPER_B = "laboratory_supply_damper"


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def checked_in_mapping() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def wide_mapping() -> dict[str, Any]:
    """The checked-in search space under a test-only identity (no simulation)."""

    data = copy.deepcopy(checked_in_mapping())
    data["spec_id"] = "stress-search-unit-wide-v1"
    data["fixture"]["fixture_id"] = "stress-search-unit-wide-v1"
    data["seeds"] = list(TEST_SEEDS)
    return data


def small_mapping() -> dict[str, Any]:
    """A 32-step spec for fabricated-evaluator driver tests."""

    data = wide_mapping()
    data["spec_id"] = "stress-search-unit-small-v1"
    data["fixture"]["fixture_id"] = "stress-search-unit-small-v1"
    data["fixture"]["horizon_steps"] = 32
    space = data["search_space"]
    space["onset_step_range"] = [8, 12]
    space["duration_steps_range"] = [16, 18]
    space["latest_end_step"] = 31
    space["max_faults"] = 2
    data["methods"]["cross_entropy"]["population_size"] = 4
    data["methods"]["cross_entropy"]["elite_count"] = 2
    data["budget"] = {
        "max_simulation_calls": 10,
        "max_candidate_attempts": 60,
        "max_wall_seconds": 1.0e6,
    }
    data["selection"]["shortlist_size"] = 3
    return data


def mixed_mapping() -> dict[str, Any]:
    """A tiny space that yields rejections, duplicates and evaluations."""

    data = small_mapping()
    data["spec_id"] = "stress-search-unit-mixed-v1"
    space = data["search_space"]
    space["fault_types"] = {
        "fan_speed_degradation": {"multiplier_range": [0.80, 0.98]},
        "damper_jam": {},
    }
    space["onset_step_range"] = [8, 8]
    space["duration_steps_range"] = [16, 17]
    space["latest_end_step"] = 24
    data["budget"]["max_simulation_calls"] = 500
    data["budget"]["max_candidate_attempts"] = 80
    return data


def tiny_mapping() -> dict[str, Any]:
    """Eight distinct episodes in total, so a fixed seed repeats itself often."""

    data = small_mapping()
    data["spec_id"] = "stress-search-unit-tiny-v1"
    space = data["search_space"]
    space["fault_types"] = {"damper_jam": {}}
    space["max_faults"] = 1
    space["onset_step_range"] = [8, 8]
    space["duration_steps_range"] = [16, 16]
    space["latest_end_step"] = 24
    data["budget"]["max_simulation_calls"] = 500
    data["budget"]["max_candidate_attempts"] = 60
    return data


def single_fault_mapping() -> dict[str, Any]:
    data = small_mapping()
    data["spec_id"] = "stress-search-unit-single-v1"
    data["search_space"]["max_faults"] = 1
    return data


def make_spec(mapping: Mapping[str, Any]) -> ss.SearchSpec:
    return ss.validate_spec_mapping(copy.deepcopy(dict(mapping)))


def make_environment(mapping: Mapping[str, Any]) -> ss.SearchEnvironment:
    return ss.build_environment(make_spec(mapping), REPO_ROOT)


@pytest.fixture(scope="module")
def wide_environment() -> ss.SearchEnvironment:
    return make_environment(wide_mapping())


@pytest.fixture(scope="module")
def small_environment() -> ss.SearchEnvironment:
    return make_environment(small_mapping())


def _descend(data: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        data = data[key]
    return data


def _set(data: dict[str, Any], path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    _descend(data, path[:-1])[path[-1]] = value
    return data


# ---------------------------------------------------------------------------
# Fabricated evaluator (bookkeeping tests only; see the module docstring)
# ---------------------------------------------------------------------------


class FakeClock:
    """Injected monotonic clock; time moves only when a test advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def fabricated_rows(scenario: Scenario, gain: float = 1.0) -> np.ndarray:
    """A [horizon, 51] truth array: a tiled real twin row plus a CO2 offset.

    The offset is a deterministic function of the fault parameters and is far
    too small to reach any discovery tier at ``gain == 1``.
    """

    zone_ids = scenario_zone_order(scenario)
    row = np.asarray(
        project_true_targets(scenario, zone_ids, initial_state(scenario)), dtype=np.float64
    )
    rows = np.tile(row, (int(scenario.data["steps"]), 1))
    dampers = sorted(branch["damper_id"] for branch in scenario.data["air_network"]["branches"])
    for profile in scenario.data["fault_profiles"]:
        start, end = int(profile["start_step"]), int(profile["end_step"])
        multiplier = float(profile.get("end_multiplier", 1.0)) + float(
            profile.get("start_multiplier", 1.0)
        )
        target = 0.0
        if "zone_id" in profile:
            target = 1.0 + list(zone_ids).index(profile["zone_id"])
        elif "damper_id" in profile:
            target = 1.0 + dampers.index(profile["damper_id"])
        delta = 0.5 * start + 0.5 * (end - start) + 10.0 * abs(2.0 - multiplier) + 0.25 * target
        rows[start:, CO2_COLUMN] += gain * delta
    return rows


class FakeEvaluator:
    """Fabricated stand-in for the authoritative evaluator.  Not a simulator."""

    def __init__(
        self,
        *,
        clock: FakeClock | None = None,
        seconds_per_episode: float = 0.0,
        modes: Mapping[int, str] | None = None,
        twin_mode: str | None = None,
        cancel_at_candidate_call: int | None = None,
        cancel_phase: str = "mid",
        gain: float = 1.0,
        faithful: bool = False,
        trace_salt: str = "",
    ) -> None:
        self.clock = clock
        self.seconds_per_episode = seconds_per_episode
        self.modes = dict(modes or {})
        self.twin_mode = twin_mode
        self.cancel_at_candidate_call = cancel_at_candidate_call
        self.cancel_phase = cancel_phase
        self.gain = gain
        # ``faithful`` makes the fabricated outcome carry a fabricated trace
        # digest, trace bytes, actuation rows and ``replay_verified``, so the
        # confirmation and replay status logic can be driven to its positive
        # branch.  It is still not a simulator and proves nothing about replay.
        self.faithful = faithful
        self.trace_salt = trace_salt
        self.cancelled = False
        self.twin_calls = 0
        self.candidate_calls = 0
        self.started: list[str] = []
        self.completed: list[str] = []

    def cancel_requested(self) -> bool:
        return self.cancelled

    def __call__(
        self, scenario: Scenario, episode_id: str, interrupt_check: Callable[[], None] | None
    ) -> ss.EpisodeOutcome:
        is_twin = not scenario.data["fault_profiles"]
        if is_twin:
            self.twin_calls += 1
            mode = self.twin_mode
        else:
            self.candidate_calls += 1
            self.started.append(episode_id)
            mode = self.modes.get(self.candidate_calls)
        cancels_here = not is_twin and self.cancel_at_candidate_call == self.candidate_calls
        for half in range(2):
            if cancels_here and half == 1 and self.cancel_phase == "mid":
                self.cancelled = True
            if interrupt_check is not None:
                interrupt_check()
            if self.clock is not None:
                self.clock.advance(self.seconds_per_episode / 2.0)
        if cancels_here and self.cancel_phase == "end":
            self.cancelled = True
        if not is_twin:
            self.completed.append(episode_id)
        steps = int(scenario.data["steps"])
        if mode == "raise":
            raise RuntimeError("fabricated simulator crash")
        if mode == "keyboard_interrupt":
            raise KeyboardInterrupt
        if mode == "episode_interrupted":
            raise ss.EpisodeInterrupted(ss.INTERRUPT_CANCELLED)
        if mode == "not_outcome":
            return {"status": ss.OUTCOME_COMPLETED}  # type: ignore[return-value]
        if mode == "none_outcome":
            return None  # type: ignore[return-value]
        if mode == "terminal":
            return ss.EpisodeOutcome(
                status=ss.OUTCOME_TERMINAL,
                committed_steps=5,
                rows=fabricated_rows(scenario, self.gain)[:5],
                terminal_reason="FABRICATED_TERMINAL",
            )
        if mode == "nonfinite_status":
            return ss.EpisodeOutcome(
                status=ss.OUTCOME_NONFINITE,
                committed_steps=steps,
                error="plant truth contains non-finite values",
            )
        rows = fabricated_rows(scenario, self.gain)
        if mode == "nan_rows":
            rows[3, CO2_COLUMN] = float("nan")
        if mode == "inf_rows":
            rows[-1, 0] = float("inf")
        if mode == "short":
            return ss.EpisodeOutcome(
                status=ss.OUTCOME_COMPLETED, committed_steps=steps - 1, rows=rows[:-1]
            )
        if mode == "no_rows":
            return ss.EpisodeOutcome(status=ss.OUTCOME_COMPLETED, committed_steps=steps)
        outcome = ss.EpisodeOutcome(status=ss.OUTCOME_COMPLETED, committed_steps=steps, rows=rows)
        if mode == "nan_metrics":
            outcome.race_metrics = {"fabricated_metric": float("nan")}
        if mode == "inf_metrics":
            outcome.race_metrics = {"fabricated_metric": [1.0, float("-inf")]}
        if mode == "unserialisable_metrics":
            outcome.race_metrics = {"fabricated_metric": {1, 2}}
        if mode == "bad_status":
            outcome.status = "FABRICATED_UNKNOWN_STATUS"
        if mode == "negative_steps":
            outcome.committed_steps = -1
        if mode == "bool_steps":
            outcome.committed_steps = True
        if mode == "string_steps":
            outcome.committed_steps = str(steps)  # type: ignore[assignment]
        if self.faithful:
            outcome.trace_bytes = ss.canonical_bytes(
                {"fabricated_trace_for": episode_id, "salt": self.trace_salt}
            )
            outcome.trace_sha256 = hashlib.sha256(outcome.trace_bytes).hexdigest()
            outcome.actuation = [{"application_step": 0, "fabricated": True}]
            outcome.replay_verified = True
        return outcome


def run_fake(
    output_dir: Path,
    mapping: Mapping[str, Any],
    *,
    method: str = ss.METHOD_RANDOM,
    seed: str = "test-s1",
    evaluator: FakeEvaluator | None = None,
    clock: FakeClock | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> tuple[ss.RunSummary, FakeEvaluator]:
    fake = evaluator or FakeEvaluator()
    summary = ss.run_search(
        spec=make_spec(mapping),
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        method=method,
        seed=seed,
        evaluator=fake,
        confirm_evaluator=confirming_fake(fake),
        clock=clock or FakeClock(),
        cancel_requested=cancel_requested or fake.cancel_requested,
        runtime_environment=runtime_environment,
    )
    return summary, fake


def confirming_fake(fake: FakeEvaluator) -> FakeEvaluator:
    return FakeEvaluator(gain=fake.gain, faithful=fake.faithful, trace_salt=fake.trace_salt)


def resume_fake(
    output_dir: Path,
    *,
    evaluator: FakeEvaluator | None = None,
    clock: FakeClock | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> tuple[ss.RunSummary, FakeEvaluator]:
    fake = evaluator or FakeEvaluator()
    summary = ss.resume_search(
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        evaluator=fake,
        confirm_evaluator=confirming_fake(fake),
        clock=clock or FakeClock(),
        cancel_requested=fake.cancel_requested,
        runtime_environment=runtime_environment,
    )
    return summary, fake


def load_audit(output_dir: Path) -> ss.AuditLog:
    return ss.AuditLog.load(output_dir / ss.AUDIT_FILE_NAME)


def candidates_of(output_dir: Path) -> list[dict[str, Any]]:
    return load_audit(output_dir).records("candidate")


def manifest_of(output_dir: Path) -> dict[str, Any]:
    return ss.load_manifest(output_dir / ss.MANIFEST_FILE_NAME)


def run_manifest_of(output_dir: Path) -> dict[str, Any]:
    """The manifest loaded the way replay and comparison load it (audit-bound)."""

    return ss.load_run_manifest(output_dir / ss.MANIFEST_FILE_NAME)


def simulated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kinds = (
        ss.DISPOSITION_EVALUATED,
        ss.DISPOSITION_TERMINAL,
        ss.DISPOSITION_NONFINITE,
        ss.DISPOSITION_SIMULATOR_ERROR,
    )
    return [record for record in records if record["disposition"] in kinds]


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_checked_in_spec_loads_and_binds_to_the_environment() -> None:
    spec = ss.load_spec_file(SPEC_PATH)
    assert spec.seeds == DECLARED_EVIDENCE_SEEDS
    assert spec.data["population"] == {
        "scope": "DEVELOPMENT_ONLY",
        "protected_population_access": "prohibited",
    }
    assert spec.data["screening"] == {"backend": "none"}
    assert spec.spec_sha256 == ss.canonical_sha256(checked_in_mapping())
    environment = ss.build_environment(spec, REPO_ROOT)
    assert len(environment.zone_ids) == 8
    assert len(environment.damper_ids) == 8
    assert environment.twin_scenario.data["fault_profiles"] == []
    assert environment.twin_scenario.data["steps"] == 96
    assert environment.identity()["spec_sha256"] == spec.spec_sha256


def test_spec_mapping_must_be_an_object() -> None:
    with pytest.raises(ss.StressSearchSpecError, match="spec must be an object"):
        ss.validate_spec_mapping([checked_in_mapping()])


_NESTED_OBJECTS: tuple[tuple[Any, ...], ...] = (
    (),
    ("population",),
    ("fixture",),
    ("policy",),
    ("search_space",),
    ("search_space", "fault_types", "fan_speed_degradation"),
    ("search_space", "fault_types", "damper_jam"),
    ("objective",),
    ("discovery_tiers", 0),
    ("methods",),
    ("methods", "random_search"),
    ("methods", "cross_entropy"),
    ("budget",),
    ("selection",),
    ("replay",),
    ("screening",),
)


@pytest.mark.parametrize("path", _NESTED_OBJECTS, ids=lambda path: "/".join(map(str, path)) or "top")
def test_spec_rejects_an_unknown_field_at_every_nesting_level(path: tuple[Any, ...]) -> None:
    data = wide_mapping()
    _descend(data, path)["surprise"] = 1
    with pytest.raises(ss.StressSearchSpecError, match=r"unknown .* fields: surprise"):
        ss.validate_spec_mapping(data)


@pytest.mark.parametrize(
    "path",
    [path for path in _NESTED_OBJECTS if path[-1:] not in (("damper_jam",), ("random_search",))],
    ids=lambda path: "/".join(map(str, path)) or "top",
)
def test_spec_rejects_each_missing_field_at_every_nesting_level(path: tuple[Any, ...]) -> None:
    for name in sorted(_descend(wide_mapping(), path)):
        data = wide_mapping()
        del _descend(data, path)[name]
        with pytest.raises(ss.StressSearchSpecError, match=rf"missing .* fields: {name}"):
            ss.validate_spec_mapping(data)


@pytest.mark.parametrize("fault_type", sorted(GENERATOR_FAULT_MULTIPLIER_BANDS))
@pytest.mark.parametrize("side", ["below_band", "above_band"])
def test_spec_rejects_multiplier_range_wider_than_the_governing_band(
    fault_type: str, side: str
) -> None:
    low, high = GENERATOR_FAULT_MULTIPLIER_BANDS[fault_type]
    data = wide_mapping()
    declared = [low - 0.01, high] if side == "below_band" else [low, high + 0.01]
    data["search_space"]["fault_types"][fault_type] = {"multiplier_range": declared}
    with pytest.raises(ss.StressSearchSpecError, match="exceeds the governing band"):
        ss.validate_spec_mapping(data)


@pytest.mark.parametrize("fault_type", sorted(GENERATOR_FAULT_MULTIPLIER_BANDS))
def test_spec_accepts_a_multiplier_range_narrower_than_the_governing_band(
    fault_type: str,
) -> None:
    low, high = GENERATOR_FAULT_MULTIPLIER_BANDS[fault_type]
    data = wide_mapping()
    middle = round((low + high) / 2.0, 4)
    data["search_space"]["fault_types"][fault_type] = {"multiplier_range": [middle, high]}
    spec = ss.validate_spec_mapping(data)
    assert spec.data["search_space"]["fault_types"][fault_type]["multiplier_range"] == [
        middle,
        high,
    ]


_INVALID_SPEC_CASES: tuple[tuple[str, tuple[Any, ...], Any, str], ...] = (
    ("schema_version", ("schema_version",), "aeolus_other_v9", "schema_version"),
    ("empty_spec_id", ("spec_id",), "", "spec_id must be a non-empty string"),
    ("scope_validation", ("population", "scope"), "VALIDATION", "population scope must be"),
    ("scope_blind_final", ("population", "scope"), "BLIND_FINAL", "population scope must be"),
    (
        "protected_access_allowed",
        ("population", "protected_population_access"),
        "allowed",
        "protected population access must be prohibited",
    ),
    (
        "protected_access_read_only",
        ("population", "protected_population_access"),
        "read_only",
        "protected population access must be prohibited",
    ),
    (
        "other_base_fixture",
        ("fixture", "base_scenario_relative_path"),
        "scenarios/habitat_v2_reference.json",
        "only the development base fixture",
    ),
    ("bad_base_digest", ("fixture", "base_scenario_sha256"), "abc", "base_scenario_sha256"),
    ("horizon_too_small", ("fixture", "horizon_steps"), 1, "horizon_steps"),
    ("horizon_bool", ("fixture", "horizon_steps"), True, "horizon_steps"),
    ("horizon_float", ("fixture", "horizon_steps"), 96.0, "horizon_steps"),
    ("dt_changed", ("fixture", "dt_seconds"), 30.0, "dt_seconds must be"),
    (
        "initial_state_rule",
        ("fixture", "initial_achieved_state"),
        "zero",
        "initial_achieved_state must be",
    ),
    ("sensor_seed_negative", ("fixture", "sensor_random_seed"), -1, "non-negative integer"),
    ("sensor_seed_roster_base", ("fixture", "sensor_random_seed"), SENSOR_SEED_BASE, "roster seed range"),
    (
        "sensor_seed_roster_variant",
        ("fixture", "sensor_random_seed"),
        SENSOR_SEED_BASE + 3017,
        "roster seed range",
    ),
    (
        "sensor_seed_roster_last",
        ("fixture", "sensor_random_seed"),
        ss.ROSTER_SENSOR_SEED_CEILING - 1,
        "roster seed range",
    ),
    ("policy_other", ("policy", "policy_id"), "adviser_v1", "only the fixed policy"),
    (
        "contract_path_other",
        ("policy", "hmc_contract_relative_path"),
        "contracts/other.json",
        "only contracts/habitat_v2_hmc_v1.json is supported",
    ),
    ("contract_digest_shape", ("policy", "hmc_contract_sha256"), "Z" * 64, "hmc_contract_sha256"),
    ("min_faults_zero", ("search_space", "min_faults"), 0, "fault count must satisfy"),
    ("max_faults_five", ("search_space", "max_faults"), 5, "fault count must satisfy"),
    ("max_faults_float", ("search_space", "max_faults"), 2.0, "must be integers"),
    ("no_fault_types", ("search_space", "fault_types"), {}, "fault_types must be a non-empty object"),
    ("onset_below", ("search_space", "onset_step_range"), [7, 72], "onset_step_range exceeds"),
    ("onset_above", ("search_space", "onset_step_range"), [8, 73], "onset_step_range exceeds"),
    ("onset_reversed", ("search_space", "onset_step_range"), [20, 10], "low <= high"),
    ("onset_float", ("search_space", "onset_step_range"), [8.0, 72], r"\[low, high\] integers"),
    ("onset_three", ("search_space", "onset_step_range"), [8, 9, 10], r"\[low, high\] integers"),
    (
        "duration_below",
        ("search_space", "duration_steps_range"),
        [15, 64],
        "duration_steps_range exceeds",
    ),
    (
        "duration_above",
        ("search_space", "duration_steps_range"),
        [16, 65],
        "duration_steps_range exceeds",
    ),
    ("latest_end_past_horizon", ("search_space", "latest_end_step"), 96, "latest_end_step"),
    ("empty_search_space", ("search_space", "latest_end_step"), 23, "search space is empty"),
    ("ramp_shape_unknown", ("search_space", "ramp_shapes"), ["step"], "ramp_shapes must be unique"),
    (
        "ramp_shape_duplicate",
        ("search_space", "ramp_shapes"),
        ["sustained", "sustained"],
        "ramp_shapes must be unique",
    ),
    ("ramp_shape_empty", ("search_space", "ramp_shapes"), [], "ramp_shapes must be unique"),
    ("decimals_zero", ("search_space", "multiplier_decimals"), 0, "multiplier_decimals"),
    (
        "multiplier_range_reversed",
        ("search_space", "fault_types", "fan_speed_degradation", "multiplier_range"),
        [0.95, 0.85],
        "low <= high",
    ),
    (
        "multiplier_range_strings",
        ("search_space", "fault_types", "fan_speed_degradation", "multiplier_range"),
        ["0.85", "0.95"],
        "finite numbers",
    ),
    ("objective_unknown", ("objective", "objective_id"), "secret_v9", "objective_id must be one of"),
    ("objective_minimize", ("objective", "direction"), "minimize", "must be 'maximize'"),
    ("objective_units", ("objective", "units"), "ppm", "units do not match"),
    ("objective_baseline", ("objective", "baseline"), "none", "baseline must be 'healthy_twin'"),
    ("tiers_empty", ("discovery_tiers",), [], "discovery_tiers must be a non-empty list"),
    ("tier_metric", ("discovery_tiers", 0, "metric"), "objective", "metric or operator"),
    ("tier_operator", ("discovery_tiers", 0, "operator"), "<", "metric or operator"),
    ("tier_value", ("discovery_tiers", 0, "value"), "0", "finite number"),
    (
        "tier_duplicate_id",
        ("discovery_tiers", 1, "tier_id"),
        "A_threshold_crossing",
        "unique non-empty string",
    ),
    ("cem_population", ("methods", "cross_entropy", "population_size"), 1, "population_size"),
    ("cem_elite", ("methods", "cross_entropy", "elite_count"), 12, "elite_count"),
    ("cem_smoothing", ("methods", "cross_entropy", "smoothing"), 0.0, "smoothing"),
    ("cem_min_sigma", ("methods", "cross_entropy", "min_sigma"), 0.5, "min_sigma"),
    ("budget_sim_too_small", ("budget", "max_simulation_calls"), 1, "max_simulation_calls"),
    ("budget_sim_float", ("budget", "max_simulation_calls"), 120.0, "max_simulation_calls"),
    ("budget_sim_huge", ("budget", "max_simulation_calls"), 100_001, "max_simulation_calls"),
    ("budget_attempts_zero", ("budget", "max_candidate_attempts"), 0, "max_candidate_attempts"),
    ("budget_wall_zero", ("budget", "max_wall_seconds"), 0.0, "max_wall_seconds"),
    ("budget_wall_null", ("budget", "max_wall_seconds"), None, "max_wall_seconds"),
    ("seeds_empty", ("seeds",), [], "seeds must be unique"),
    ("seeds_duplicate", ("seeds",), ["test-s1", "test-s1"], "seeds must be unique"),
    ("seeds_separator", ("seeds",), ["test|s1"], "seeds must be unique"),
    ("seeds_integer", ("seeds",), [1], "seeds must be unique"),
    ("shortlist_too_large", ("selection", "shortlist_size"), 65, "shortlist_size"),
    ("family_cap_zero", ("selection", "max_per_causal_family"), 0, "max_per_causal_family"),
    (
        "minimum_objective_string",
        ("selection", "minimum_objective_exclusive"),
        "0",
        "minimum_objective_exclusive",
    ),
    ("tolerance_negative", ("replay", "metric_absolute_tolerance"), -1e-9, "metric_absolute_tolerance"),
    (
        "trace_flag_not_bool",
        ("replay", "require_identical_trace_sha256"),
        "yes",
        "require_identical_trace_sha256",
    ),
)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [case[1:] for case in _INVALID_SPEC_CASES],
    ids=[case[0] for case in _INVALID_SPEC_CASES],
)
def test_spec_rejects_out_of_contract_values(
    path: tuple[Any, ...], value: Any, message: str
) -> None:
    data = _set(wide_mapping(), path, value)
    with pytest.raises(ss.StressSearchSpecError, match=message):
        ss.validate_spec_mapping(data)


@pytest.mark.parametrize(
    "fault_type",
    ["sensor_bias_drift", "sensor_stuck", "feedback_sensor_bias_drift", "made_up_fault"],
)
def test_spec_rejects_unsupported_fault_types(fault_type: str) -> None:
    data = wide_mapping()
    data["search_space"]["fault_types"][fault_type] = {}
    with pytest.raises(ss.StressSearchSpecError, match="unsupported fault type"):
        ss.validate_spec_mapping(data)


def test_spec_rejects_a_multiplier_declared_for_damper_jam() -> None:
    data = wide_mapping()
    data["search_space"]["fault_types"]["damper_jam"] = {"multiplier_range": [0.5, 0.9]}
    with pytest.raises(ss.StressSearchSpecError, match="unknown fault type damper_jam fields"):
        ss.validate_spec_mapping(data)


@pytest.mark.parametrize("backend", ["warp", "cuda", "gpu", "batched_cpu", "", None, "None"])
def test_spec_refuses_any_screening_backend_other_than_none(backend: Any) -> None:
    data = _set(wide_mapping(), ("screening", "backend"), backend)
    with pytest.raises(ss.StressSearchSpecError, match="accelerated screening is refused"):
        ss.validate_spec_mapping(data)


@pytest.mark.parametrize(
    "path",
    [
        ("budget", "max_wall_seconds"),
        ("discovery_tiers", 0, "value"),
        ("selection", "minimum_objective_exclusive"),
        ("replay", "metric_absolute_tolerance"),
        ("search_space", "fault_types", "fan_speed_degradation", "multiplier_range", 0),
        ("methods", "cross_entropy", "smoothing"),
    ],
    ids=lambda path: "/".join(map(str, path)),
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_spec_rejects_non_finite_numbers(path: tuple[Any, ...], value: float) -> None:
    data = _set(wide_mapping(), path, value)
    with pytest.raises(ss.StressSearchSpecError):
        ss.validate_spec_mapping(data)


def test_build_environment_catches_a_wrong_base_scenario_digest() -> None:
    data = _set(wide_mapping(), ("fixture", "base_scenario_sha256"), "0" * 64)
    spec = ss.validate_spec_mapping(data)
    with pytest.raises(ss.StressSearchSpecError, match="base scenario digest does not match"):
        ss.build_environment(spec, REPO_ROOT)


def test_build_environment_catches_a_wrong_hmc_contract_digest() -> None:
    data = _set(wide_mapping(), ("policy", "hmc_contract_sha256"), "f" * 64)
    spec = ss.validate_spec_mapping(data)
    with pytest.raises(ss.StressSearchSpecError, match="HMC contract digest does not match"):
        ss.build_environment(spec, REPO_ROOT)


def test_load_spec_file_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    marker = '"budget": {'
    assert text.count(marker) == 1
    tampered = text.replace(
        marker, marker + '\n    "max_simulation_calls": 100000,', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ss.StressSearchSpecError, match="duplicate JSON key 'max_simulation_calls'"):
        ss.load_spec_file(path)


def test_load_spec_file_rejects_non_finite_constants_and_bad_bytes(tmp_path: Path) -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert text.count("1800.0") == 1
    for index, token in enumerate(("NaN", "Infinity", "-Infinity")):
        path = tmp_path / f"constant-{index}.json"
        path.write_text(text.replace("1800.0", token), encoding="utf-8")
        with pytest.raises(ss.StressSearchSpecError, match="non-finite JSON constant"):
            ss.load_spec_file(path)
    truncated = tmp_path / "truncated.json"
    truncated.write_text(text[: len(text) // 2], encoding="utf-8")
    with pytest.raises(ss.StressSearchSpecError, match="not strict JSON"):
        ss.load_spec_file(truncated)
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe{}")
    with pytest.raises(ss.StressSearchSpecError, match="not strict JSON"):
        ss.load_spec_file(binary)
    with pytest.raises(ss.StressSearchSpecError, match="cannot read spec"):
        ss.load_spec_file(tmp_path / "absent.json")


def test_spec_digest_changes_with_any_declared_value() -> None:
    reference = make_spec(wide_mapping()).spec_sha256
    assert make_spec(wide_mapping()).spec_sha256 == reference
    for path, value in (
        (("spec_id",), "stress-search-unit-other"),
        (("budget", "max_simulation_calls"), 121),
        (("seeds",), ["test-s1"]),
        (("selection", "shortlist_size"), 4),
    ):
        assert make_spec(_set(wide_mapping(), path, value)).spec_sha256 != reference


@pytest.mark.parametrize(
    "fixture_id",
    [
        "bdm-v1-g0001-a-steady",
        "bdm-v1-stress-search-x",
        "test-stress-unit-wide-v1",
        "Stress-Search-x",
        "stress_search-x",
        "stress-search",
        "stress-search-a/b",
        "stress-search-a\\b",
        "stress-search-a.b",
        "stress-search-..",
        "stress-search-a b",
        "stress-search-a|b",
        "stress-search-a\n",
        " stress-search-a",
        "../stress-search-a",
        "stress-search-" + "x" * 51,
        "",
        7,
        None,
        ["stress-search-a"],
    ],
    ids=repr,
)
def test_spec_rejects_a_fixture_id_outside_the_harness_name_namespace(fixture_id: Any) -> None:
    data = _set(wide_mapping(), ("fixture", "fixture_id"), fixture_id)
    with pytest.raises(ss.StressSearchSpecError, match="fixture_id must start with 'stress-search-'"):
        ss.validate_spec_mapping(data)


def test_spec_accepts_fixture_ids_at_the_name_pattern_boundaries() -> None:
    assert ss.FIXTURE_ID_PREFIX == "stress-search-"
    for fixture_id in ("stress-search-a", "stress-search-" + "x" * 50, "stress-search-A_b-9"):
        spec = ss.validate_spec_mapping(_set(wide_mapping(), ("fixture", "fixture_id"), fixture_id))
        assert spec.data["fixture"]["fixture_id"] == fixture_id
        assert len(fixture_id) <= 64


@pytest.mark.parametrize(
    "seed",
    [
        "a|b",
        "a/b",
        "a\\b",
        "a.b",
        "..",
        ".",
        "a b",
        " a",
        "a\n",
        "-leading-dash",
        "_leading-underscore",
        "x" * 65,
        "",
        "s\u00e9ed",
        "a:b",
        "a*",
    ],
    ids=repr,
)
def test_spec_rejects_path_unsafe_or_separator_bearing_seeds(seed: str) -> None:
    for seeds in ([seed], ["test-s1", seed]):
        data = _set(wide_mapping(), ("seeds",), seeds)
        with pytest.raises(ss.StressSearchSpecError, match="seeds must be unique names"):
            ss.validate_spec_mapping(data)


def test_spec_accepts_seed_names_at_the_pattern_boundaries() -> None:
    seeds = ["a", "0", "A-b_9", "x" * 64]
    assert ss.validate_spec_mapping(_set(wide_mapping(), ("seeds",), seeds)).seeds == tuple(seeds)


@pytest.mark.parametrize(
    ("decimals", "declared"),
    [
        (1, [0.80, 0.98]),
        (1, [0.85, 0.9]),
        (2, [0.805, 0.95]),
        (2, [0.85, 0.955]),
        (4, [0.85001, 0.95]),
        (4, [0.85, 0.94999]),
    ],
    ids=["d1_high", "d1_low", "d2_low", "d2_high", "d4_low", "d4_high"],
)
def test_spec_rejects_multiplier_endpoints_off_the_quantisation_grid(
    decimals: int, declared: list[float]
) -> None:
    data = wide_mapping()
    data["search_space"]["multiplier_decimals"] = decimals
    for declaration in data["search_space"]["fault_types"].values():
        if "multiplier_range" in declaration:
            # Every other declared band sits on the one-decimal grid.
            declaration["multiplier_range"] = [
                round(value, 1) for value in declaration["multiplier_range"]
            ]
    low, high = GENERATOR_FAULT_MULTIPLIER_BANDS["fan_speed_degradation"]
    assert low <= declared[0] <= declared[1] <= high
    data["search_space"]["fault_types"]["fan_speed_degradation"] = {"multiplier_range": declared}
    with pytest.raises(
        ss.StressSearchSpecError,
        match=(
            "fan_speed_degradation multiplier_range endpoints must be representable at "
            f"{decimals} decimals"
        ),
    ):
        ss.validate_spec_mapping(data)
    # The same band moved onto the grid is accepted.
    data["search_space"]["fault_types"]["fan_speed_degradation"] = {
        "multiplier_range": [0.8, 0.9]
    }
    assert ss.validate_spec_mapping(data).data["search_space"]["multiplier_decimals"] == decimals


# ---------------------------------------------------------------------------
# Candidate bounds and impossible combinations
# ---------------------------------------------------------------------------


def fault(fault_type: str, **overrides: Any) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "fan_speed_degradation": {"target": None, "multiplier": 0.9},
        "branch_resistance_increase": {"target": ZONE_A, "multiplier": 2.0},
        "cooling_delivery_degradation": {"target": ZONE_A, "multiplier": 0.8},
        "oxygen_delivery_degradation": {"target": ZONE_A, "multiplier": 0.8},
        "scrubber_effectiveness_degradation": {"target": None, "multiplier": 0.7},
        "condenser_effectiveness_degradation": {"target": None, "multiplier": 0.7},
        "damper_jam": {"target": DAMPER_A, "multiplier": None, "ramp_shape": None},
    }
    result: dict[str, Any] = {
        "type": fault_type,
        "onset_step": 10,
        "duration_steps": 20,
        "ramp_shape": "sustained",
        **defaults.get(fault_type, {"target": None, "multiplier": 0.9}),
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize("fault_type", sorted(ss.SUPPORTED_FAULT_TARGETS))
def test_valid_candidate_is_accepted_unchanged_for_every_fault_type(
    wide_environment: ss.SearchEnvironment, fault_type: str
) -> None:
    candidate = fault(fault_type)
    canonical = ss.validate_candidate_parameters(wide_environment, [copy.deepcopy(candidate)])
    assert canonical == [candidate]
    scenario = ss.build_candidate_scenario(wide_environment, canonical, "e" * 64)
    assert len(scenario.data["fault_profiles"]) == 1
    profile = scenario.data["fault_profiles"][0]
    assert (profile["start_step"], profile["end_step"]) == (10, 30)


def _without(candidate: dict[str, Any], name: str) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != name}


_INVALID_CANDIDATES: tuple[tuple[str, list[Any] | Any, str], ...] = (
    ("multiplier_below_range", [fault("fan_speed_degradation", multiplier=0.5)], "is outside"),
    ("multiplier_above_range", [fault("fan_speed_degradation", multiplier=0.99)], "is outside"),
    ("multiplier_at_one", [fault("fan_speed_degradation", multiplier=1.0)], "is outside"),
    (
        "branch_multiplier_below_range",
        [fault("branch_resistance_increase", multiplier=1.0)],
        "is outside",
    ),
    (
        "branch_multiplier_above_range",
        [fault("branch_resistance_increase", multiplier=3.5)],
        "is outside",
    ),
    ("multiplier_unquantised", [fault("fan_speed_degradation", multiplier=0.90001)], "not quantised"),
    (
        "multiplier_integer",
        [fault("branch_resistance_increase", multiplier=2)],
        "multiplier must be a finite float",
    ),
    (
        "multiplier_bool",
        [fault("branch_resistance_increase", multiplier=True)],
        "multiplier must be a finite float",
    ),
    (
        "multiplier_missing_value",
        [fault("fan_speed_degradation", multiplier=None)],
        "multiplier must be a finite float",
    ),
    (
        "multiplier_nan",
        [fault("fan_speed_degradation", multiplier=float("nan"))],
        "multiplier must be a finite float",
    ),
    (
        "multiplier_infinite",
        [fault("branch_resistance_increase", multiplier=float("inf"))],
        "multiplier must be a finite float",
    ),
    ("multiplier_string", [fault("fan_speed_degradation", multiplier="0.9")], "finite float"),
    ("onset_below_range", [fault("fan_speed_degradation", onset_step=7)], "onset_step 7 is outside"),
    ("onset_above_range", [fault("fan_speed_degradation", onset_step=73)], "onset_step 73 is outside"),
    ("onset_float", [fault("fan_speed_degradation", onset_step=10.0)], "onset_step"),
    ("onset_bool", [fault("fan_speed_degradation", onset_step=True)], "onset_step"),
    (
        "duration_below_range",
        [fault("fan_speed_degradation", duration_steps=15)],
        "duration_steps 15 is outside",
    ),
    (
        "duration_above_range",
        [fault("fan_speed_degradation", duration_steps=65)],
        "duration_steps 65 is outside",
    ),
    (
        "end_past_latest_end_step",
        [fault("fan_speed_degradation", onset_step=72, duration_steps=64)],
        "fault end step 136 exceeds latest_end_step 95",
    ),
    (
        "end_one_past_latest_end_step",
        [fault("fan_speed_degradation", onset_step=72, duration_steps=24)],
        "fault end step 96 exceeds latest_end_step 95",
    ),
    ("fan_with_zone_target", [fault("fan_speed_degradation", target=ZONE_A)], "takes no target"),
    (
        "scrubber_with_zone_target",
        [fault("scrubber_effectiveness_degradation", target=ZONE_A)],
        "takes no target",
    ),
    (
        "branch_without_target",
        [fault("branch_resistance_increase", target=None)],
        "target is not a declared zone",
    ),
    (
        "branch_with_damper_target",
        [fault("branch_resistance_increase", target=DAMPER_A)],
        "target is not a declared zone",
    ),
    (
        "cooling_with_unknown_zone",
        [fault("cooling_delivery_degradation", target="mess_hall")],
        "target is not a declared zone",
    ),
    (
        "oxygen_with_integer_target",
        [fault("oxygen_delivery_degradation", target=0)],
        "target is not a declared zone",
    ),
    ("damper_jam_with_zone_target", [fault("damper_jam", target=ZONE_A)], "not a declared damper"),
    ("damper_jam_without_target", [fault("damper_jam", target=None)], "not a declared damper"),
    (
        "damper_jam_with_multiplier",
        [fault("damper_jam", multiplier=0.5)],
        "takes no multiplier or ramp shape",
    ),
    (
        "damper_jam_with_ramp_shape",
        [fault("damper_jam", ramp_shape="sustained")],
        "takes no multiplier or ramp shape",
    ),
    ("ramp_shape_unknown", [fault("fan_speed_degradation", ramp_shape="step")], "ramp_shape 'step'"),
    ("ramp_shape_missing", [fault("fan_speed_degradation", ramp_shape=None)], "ramp_shape None"),
    ("type_unsupported", [fault("sensor_bias_drift")], "'sensor_bias_drift' is not permitted"),
    ("type_not_a_string", [fault("fan_speed_degradation", type=["x"])], "is not permitted"),
    (
        "target_field_missing",
        [_without(fault("branch_resistance_increase"), "target")],
        "exactly the fields",
    ),
    (
        "multiplier_field_missing",
        [_without(fault("fan_speed_degradation"), "multiplier")],
        "exactly the fields",
    ),
    ("extra_field", [{**fault("fan_speed_degradation"), "slot": 0}], "exactly the fields"),
    ("fault_not_an_object", ["fan_speed_degradation"], "exactly the fields"),
    ("too_few_faults", [], r"fault count 0 is outside \[1, 3\]"),
    (
        "too_many_faults",
        [
            fault("fan_speed_degradation"),
            fault("damper_jam"),
            fault("branch_resistance_increase"),
            fault("scrubber_effectiveness_degradation"),
        ],
        r"fault count 4 is outside \[1, 3\]",
    ),
    ("faults_not_a_list", (fault("fan_speed_degradation"),), "faults must be a list"),
    ("faults_a_mapping", {"0": fault("fan_speed_degradation")}, "faults must be a list"),
)


@pytest.mark.parametrize(
    ("faults", "message"),
    [case[1:] for case in _INVALID_CANDIDATES],
    ids=[case[0] for case in _INVALID_CANDIDATES],
)
def test_candidate_outside_the_declared_space_is_rejected_not_clipped(
    wide_environment: ss.SearchEnvironment, faults: Any, message: str
) -> None:
    before = copy.deepcopy(faults)
    with pytest.raises(ss.StressSearchCandidateError, match=message):
        ss.validate_candidate_parameters(wide_environment, faults)
    # Nothing was coerced into range behind the caller's back.
    assert repr(faults) == repr(before)


def test_candidate_fault_type_must_be_declared_by_the_spec_not_merely_supported() -> None:
    data = wide_mapping()
    del data["search_space"]["fault_types"]["damper_jam"]
    environment = make_environment(data)
    with pytest.raises(ss.StressSearchCandidateError, match="'damper_jam' is not permitted"):
        ss.validate_candidate_parameters(environment, [fault("damper_jam")])


def test_candidate_bounds_follow_a_narrowed_spec() -> None:
    data = wide_mapping()
    data["search_space"]["fault_types"]["fan_speed_degradation"] = {
        "multiplier_range": [0.9, 0.95]
    }
    data["search_space"]["ramp_shapes"] = ["sustained"]
    environment = make_environment(data)
    assert ss.validate_candidate_parameters(environment, [fault("fan_speed_degradation")])
    with pytest.raises(ss.StressSearchCandidateError, match="is outside"):
        ss.validate_candidate_parameters(
            environment, [fault("fan_speed_degradation", multiplier=0.85)]
        )
    with pytest.raises(ss.StressSearchCandidateError, match="ramp_shape 'recovery'"):
        ss.validate_candidate_parameters(
            environment, [fault("fan_speed_degradation", ramp_shape="recovery")]
        )


@pytest.mark.parametrize("fault_type", sorted(ss.SUPPORTED_FAULT_TARGETS))
def test_same_target_overlap_is_rejected_by_the_existing_scenario_validator(
    wide_environment: ss.SearchEnvironment, fault_type: str
) -> None:
    faults = [
        fault(fault_type, onset_step=10, duration_steps=20),
        fault(fault_type, onset_step=29, duration_steps=16),
    ]
    # The harness's own parameter check has no overlap rule of its own ...
    canonical = ss.validate_candidate_parameters(wide_environment, faults)
    # ... the impossible combination is refused by Scenario.from_mapping.
    with pytest.raises(ScenarioValidationError, match="may not overlap"):
        ss.build_candidate_scenario(wide_environment, canonical, "a" * 64)


@pytest.mark.parametrize("fault_type", sorted(ss.SUPPORTED_FAULT_TARGETS))
def test_same_target_back_to_back_faults_are_accepted(
    wide_environment: ss.SearchEnvironment, fault_type: str
) -> None:
    faults = [
        fault(fault_type, onset_step=10, duration_steps=20),
        fault(fault_type, onset_step=30, duration_steps=16),
    ]
    canonical = ss.validate_candidate_parameters(wide_environment, faults)
    scenario = ss.build_candidate_scenario(wide_environment, canonical, "a" * 64)
    assert [
        (profile["start_step"], profile["end_step"])
        for profile in scenario.data["fault_profiles"]
    ] == [(10, 30), (30, 46)]


@pytest.mark.parametrize(
    "faults",
    [
        [
            fault("branch_resistance_increase", target=ZONE_A),
            fault("branch_resistance_increase", target=ZONE_B),
        ],
        [fault("damper_jam", target=DAMPER_A), fault("damper_jam", target=DAMPER_B)],
        [
            fault("cooling_delivery_degradation", target=ZONE_A),
            fault("cooling_delivery_degradation", target=ZONE_B),
        ],
        [
            fault("cooling_delivery_degradation", target=ZONE_A),
            fault("oxygen_delivery_degradation", target=ZONE_A),
            fault("branch_resistance_increase", target=ZONE_A),
        ],
        [
            fault("fan_speed_degradation"),
            fault("scrubber_effectiveness_degradation"),
            fault("condenser_effectiveness_degradation"),
        ],
    ],
    ids=["two_zones_branch", "two_dampers", "two_zones_cooling", "one_zone_three_mechanisms", "three_globals"],
)
def test_overlapping_faults_on_different_targets_or_mechanisms_are_accepted(
    wide_environment: ss.SearchEnvironment, faults: list[dict[str, Any]]
) -> None:
    canonical = ss.validate_candidate_parameters(wide_environment, faults)
    scenario = ss.build_candidate_scenario(wide_environment, canonical, "b" * 64)
    assert len(scenario.data["fault_profiles"]) == len(faults)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [("worsening", (1.0, 0.9)), ("recovery", (0.9, 1.0)), ("sustained", (0.9, 0.9))],
)
def test_ramp_shape_sets_the_scenario_multiplier_endpoints(
    wide_environment: ss.SearchEnvironment, shape: str, expected: tuple[float, float]
) -> None:
    canonical = ss.validate_candidate_parameters(
        wide_environment, [fault("fan_speed_degradation", ramp_shape=shape)]
    )
    profile = ss.build_candidate_scenario(wide_environment, canonical, "c" * 64).data[
        "fault_profiles"
    ][0]
    assert (profile["start_multiplier"], profile["end_multiplier"]) == expected


# ---------------------------------------------------------------------------
# Sampler: properties, determinism and seed separation
# ---------------------------------------------------------------------------


def proposals(
    environment: ss.SearchEnvironment,
    method: str,
    seed: str,
    count: int,
    *,
    feedback: bool = False,
) -> list[list[dict[str, Any]]]:
    sampler = ss.CandidateSampler(environment, method, seed)
    result = []
    for attempt in range(count):
        raw = sampler.propose(attempt)
        result.append(copy.deepcopy(raw))
        if feedback:
            # Synthetic feedback so cross-entropy leaves its uniform start.
            sampler.observe(
                raw, -float(sum(item["onset_step"] + item["duration_steps"] for item in raw))
            )
    return result


@pytest.mark.parametrize("method", ss.METHODS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_every_proposal_respects_declared_bounds_and_quantisation(
    wide_environment: ss.SearchEnvironment, method: str, seed: str
) -> None:
    space = wide_environment.spec.data["search_space"]
    for raw in proposals(wide_environment, method, seed, 300, feedback=True):
        assert space["min_faults"] <= len(raw) <= space["max_faults"]
        assert [item["slot"] for item in raw] == sorted({item["slot"] for item in raw})
        for item in raw:
            assert item["type"] in space["fault_types"]
            assert space["onset_step_range"][0] <= item["onset_step"] <= space["onset_step_range"][1]
            assert (
                space["duration_steps_range"][0]
                <= item["duration_steps"]
                <= space["duration_steps_range"][1]
            )
            assert type(item["onset_step"]) is int and type(item["duration_steps"]) is int
            if item["type"] == "damper_jam":
                assert item["multiplier"] is None and item["ramp_shape"] is None
                assert item["target"] in wide_environment.damper_ids
                continue
            low, high = space["fault_types"][item["type"]]["multiplier_range"]
            assert type(item["multiplier"]) is float
            assert low <= item["multiplier"] <= high
            assert round(item["multiplier"], space["multiplier_decimals"]) == item["multiplier"]
            assert item["ramp_shape"] in space["ramp_shapes"]
            kind = ss.SUPPORTED_FAULT_TARGETS[item["type"]]
            assert (item["target"] is None) == (kind == ss.GLOBAL_TARGET)


@pytest.mark.parametrize("method", ss.METHODS)
def test_valid_proposals_build_scenarios_and_invalid_ones_only_miss_the_end_bound(
    wide_environment: ss.SearchEnvironment, method: str
) -> None:
    built = overlapping = invalid = 0
    for raw in proposals(wide_environment, method, "test-s1", 400, feedback=True):
        parameters = ss.strip_slots(raw)
        try:
            canonical = ss.validate_candidate_parameters(wide_environment, parameters)
        except ss.StressSearchCandidateError as error:
            # The sampler stays inside every declared range; the only raw
            # violation it can produce is a fault that ends too late, and that
            # is rejected rather than shortened.
            assert "exceeds latest_end_step" in str(error)
            invalid += 1
            continue
        identity = ss.episode_identity(wide_environment, canonical)
        try:
            scenario = ss.build_candidate_scenario(
                wide_environment, canonical, identity["episode_id"]
            )
        except ScenarioValidationError as error:
            assert "may not overlap" in str(error)
            assert len(canonical) > 1
            overlapping += 1
            continue
        assert isinstance(scenario, Scenario)
        assert len(scenario.data["fault_profiles"]) == len(canonical)
        assert scenario.data["steps"] == wide_environment.spec.horizon_steps
        for profile in scenario.data["fault_profiles"]:
            assert profile["end_step"] <= wide_environment.spec.data["search_space"]["latest_end_step"]
        built += 1
    assert built > 50
    assert invalid > 0
    assert built + overlapping + invalid == 400


@pytest.mark.parametrize("method", ss.METHODS)
def test_every_valid_single_fault_proposal_builds_a_valid_scenario(method: str) -> None:
    environment = make_environment(single_fault_mapping())
    seen_types: set[str] = set()
    for raw in proposals(environment, method, "test-s2", 300, feedback=True):
        canonical = ss.validate_candidate_parameters(environment, ss.strip_slots(raw))
        assert len(canonical) == 1
        scenario = ss.build_candidate_scenario(
            environment, canonical, ss.episode_identity(environment, canonical)["episode_id"]
        )
        assert scenario.data["steps"] == 32
        seen_types.add(canonical[0]["type"])
    assert seen_types == set(ss.SUPPORTED_FAULT_TARGETS)


def test_sampler_never_proposes_an_unquantised_multiplier_for_an_accepted_spec() -> None:
    data = single_fault_mapping()
    data["search_space"]["multiplier_decimals"] = 1
    data["search_space"]["fault_types"] = {
        "fan_speed_degradation": {"multiplier_range": [0.80, 0.98]}
    }
    # An endpoint that is not on the grid is refused up front ...
    with pytest.raises(ss.StressSearchSpecError, match="representable at 1 decimals"):
        make_spec(data)
    # ... and with on-grid endpoints every proposal is a valid quantised candidate.
    data["search_space"]["fault_types"] = {
        "fan_speed_degradation": {"multiplier_range": [0.8, 0.9]},
        "branch_resistance_increase": {"multiplier_range": [1.3, 3.0]},
    }
    environment = make_environment(data)
    seen: set[float] = set()
    for method in ss.METHODS:
        for raw in proposals(environment, method, "test-s1", 200, feedback=True):
            (candidate,) = ss.validate_candidate_parameters(environment, ss.strip_slots(raw))
            assert round(candidate["multiplier"], 1) == candidate["multiplier"]
            seen.add(candidate["multiplier"])
    assert {0.8, 0.9, 1.3, 3.0} <= seen


@pytest.mark.parametrize("method", ss.METHODS)
def test_same_spec_method_and_seed_give_the_same_proposal_sequence(
    wide_environment: ss.SearchEnvironment, method: str
) -> None:
    first = proposals(wide_environment, method, "test-s1", 120, feedback=True)
    second = proposals(wide_environment, method, "test-s1", 120, feedback=True)
    rebuilt = proposals(make_environment(wide_mapping()), method, "test-s1", 120, feedback=True)
    assert first == second == rebuilt


def test_proposals_are_a_pure_function_of_the_attempt_index_for_random_search(
    wide_environment: ss.SearchEnvironment,
) -> None:
    sampler = ss.CandidateSampler(wide_environment, ss.METHOD_RANDOM, "test-s1")
    forward = [sampler.propose(attempt) for attempt in range(40)]
    backward = [sampler.propose(attempt) for attempt in reversed(range(40))]
    assert forward == backward[::-1]
    sampler.observe(forward[0], 99.0)
    assert sampler.state()["generation_index"] == 0
    assert sampler.propose(7) == forward[7]


def test_different_seeds_methods_and_specs_give_different_sequences(
    wide_environment: ss.SearchEnvironment,
) -> None:
    reference = proposals(wide_environment, ss.METHOD_RANDOM, "test-s1", 60)
    other_seed = proposals(wide_environment, ss.METHOD_RANDOM, "test-s2", 60)
    other_method = proposals(wide_environment, ss.METHOD_CROSS_ENTROPY, "test-s1", 60)
    renamed = _set(wide_mapping(), ("spec_id",), "stress-search-unit-wide-v2")
    other_spec = proposals(make_environment(renamed), ss.METHOD_RANDOM, "test-s1", 60)
    assert reference != other_seed
    assert reference != other_method
    assert reference != other_spec
    assert other_seed != other_method != other_spec
    # Not merely a shifted or partly shared stream.
    for other in (other_seed, other_method, other_spec):
        shared = sum(1 for left, right in zip(reference, other) if left == right)
        assert shared <= 3


def test_cross_entropy_adapts_only_through_recorded_objectives(
    wide_environment: ss.SearchEnvironment,
) -> None:
    population = wide_environment.spec.data["methods"]["cross_entropy"]["population_size"]
    adaptive = ss.CandidateSampler(wide_environment, ss.METHOD_CROSS_ENTROPY, "test-s1")
    unscored = ss.CandidateSampler(wide_environment, ss.METHOD_CROSS_ENTROPY, "test-s1")
    initial = adaptive.state()
    for attempt in range(population):
        raw = adaptive.propose(attempt)
        assert raw == unscored.propose(attempt)
        adaptive.observe(raw, float(raw[0]["onset_step"]))
        unscored.observe(raw, None)
    assert adaptive.state()["generation_index"] == unscored.state()["generation_index"] == 1
    # A generation with no scored member leaves the distribution untouched.
    assert unscored.state()["slots"] == initial["slots"]
    assert adaptive.state()["slots"] != initial["slots"]
    assert adaptive.state()["slots"][0]["onset"]["uniform"] is False
    later_adaptive = [adaptive.propose(attempt) for attempt in range(population, 4 * population)]
    later_unscored = [unscored.propose(attempt) for attempt in range(population, 4 * population)]
    assert later_adaptive != later_unscored


def test_sampler_refuses_an_undeclared_seed_or_method(
    wide_environment: ss.SearchEnvironment,
) -> None:
    with pytest.raises(ss.StressSearchError, match="seed 'issue93-s1' is not declared"):
        ss.CandidateSampler(wide_environment, ss.METHOD_RANDOM, "issue93-s1")
    with pytest.raises(ss.StressSearchError, match="unknown method 'bayesian'"):
        ss.CandidateSampler(wide_environment, "bayesian", "test-s1")


def test_unit_draw_is_in_the_unit_interval_and_stable() -> None:
    assert ss.unit_draw("a", 1) == 0.15123676918210766
    assert ss.unit_draw("test-s1", 0, "onset") == 0.22391579763627686
    draws = [ss.unit_draw("test-s1", index, "probe") for index in range(2000)]
    assert all(0.0 <= value < 1.0 for value in draws)
    assert len(set(draws)) == len(draws)
    assert 0.45 < sum(draws) / len(draws) < 0.55
    assert draws == [ss.unit_draw("test-s1", index, "probe") for index in range(2000)]
    # Part boundaries matter: ("ab", "c") and ("a", "bc") are different keys.
    assert ss.unit_draw("ab", "c") != ss.unit_draw("a", "bc")
    assert ss.unit_draw("test-s1", 1) != ss.unit_draw("test-s2", 1)


# ---------------------------------------------------------------------------
# Identity and deduplication keys
# ---------------------------------------------------------------------------


def test_episode_identity_is_invariant_to_fault_order(
    wide_environment: ss.SearchEnvironment,
) -> None:
    faults = [
        fault("scrubber_effectiveness_degradation"),
        fault("branch_resistance_increase", target=ZONE_B),
        fault("damper_jam"),
    ]
    reference = ss.episode_identity(wide_environment, faults)
    for order in ((0, 2, 1), (1, 0, 2), (2, 1, 0), (2, 0, 1)):
        shuffled = [faults[index] for index in order]
        assert ss.episode_identity(wide_environment, shuffled) == reference
        assert ss.canonical_faults(shuffled) == ss.canonical_faults(faults)
        assert (
            ss.build_candidate_scenario(
                wide_environment, shuffled, reference["episode_id"]
            ).scenario_sha256
            == ss.build_candidate_scenario(
                wide_environment, faults, reference["episode_id"]
            ).scenario_sha256
        )


@pytest.mark.parametrize(
    ("overrides", "same_family"),
    [
        ({"onset_step": 11}, True),
        ({"duration_steps": 21}, True),
        ({"multiplier": 2.0001}, True),
        ({"ramp_shape": "worsening"}, True),
        ({"target": ZONE_B}, False),
        ({"type": "cooling_delivery_degradation", "multiplier": 0.8}, False),
    ],
    ids=["onset", "duration", "multiplier", "ramp_shape", "target", "type"],
)
def test_episode_id_changes_with_every_parameter_and_family_only_with_mechanism(
    wide_environment: ss.SearchEnvironment, overrides: dict[str, Any], same_family: bool
) -> None:
    companion = fault("fan_speed_degradation")
    base = ss.episode_identity(wide_environment, [fault("branch_resistance_increase"), companion])
    changed = ss.episode_identity(
        wide_environment, [fault("branch_resistance_increase", **overrides), companion]
    )
    assert changed["episode_id"] != base["episode_id"]
    assert (changed["causal_family_id"] == base["causal_family_id"]) is same_family
    assert (changed["causal_family_key"] == base["causal_family_key"]) is same_family


def test_causal_family_ignores_magnitude_timing_and_shape_together(
    wide_environment: ss.SearchEnvironment,
) -> None:
    first = ss.episode_identity(
        wide_environment, [fault("damper_jam"), fault("oxygen_delivery_degradation")]
    )
    second = ss.episode_identity(
        wide_environment,
        [
            fault(
                "oxygen_delivery_degradation",
                onset_step=40,
                duration_steps=33,
                multiplier=0.6501,
                ramp_shape="recovery",
            ),
            fault("damper_jam", onset_step=50, duration_steps=16),
        ],
    )
    assert first["episode_id"] != second["episode_id"]
    assert first["causal_family_id"] == second["causal_family_id"]
    extra_mechanism = ss.episode_identity(
        wide_environment,
        [fault("damper_jam"), fault("oxygen_delivery_degradation"), fault("fan_speed_degradation")],
    )
    other_damper = ss.episode_identity(
        wide_environment,
        [fault("damper_jam", target=DAMPER_B), fault("oxygen_delivery_degradation")],
    )
    assert extra_mechanism["causal_family_id"] != first["causal_family_id"]
    assert other_damper["causal_family_id"] != first["causal_family_id"]
    assert first["causal_family_key"]["sensor_failure_bundle"] == "none"


def test_identity_is_bound_to_the_fixture(
    wide_environment: ss.SearchEnvironment, small_environment: ss.SearchEnvironment
) -> None:
    candidate = [fault("fan_speed_degradation", onset_step=8, duration_steps=16)]
    wide = ss.episode_identity(wide_environment, candidate)
    small = ss.episode_identity(small_environment, candidate)
    assert wide["episode_id"] != small["episode_id"]
    assert wide["causal_family_id"] != small["causal_family_id"]
    for identity in (wide, small):
        assert ss._is_sha(identity["episode_id"]) and ss._is_sha(identity["causal_family_id"])


def test_names_ids_seeds_and_nonces_stay_out_of_the_bdm_namespace(
    wide_environment: ss.SearchEnvironment,
) -> None:
    faults = [fault("damper_jam"), fault("fan_speed_degradation")]
    identity = ss.episode_identity(wide_environment, faults)
    scenario = ss.build_candidate_scenario(wide_environment, faults, identity["episode_id"])
    name = scenario.data["name"]
    assert name == f"stress-search-unit-wide-v1-{identity['episode_id'][:16]}"
    assert not name.lower().startswith("bdm")
    assert not wide_environment.twin_scenario.data["name"].lower().startswith("bdm")
    profile_ids = [profile["id"] for profile in scenario.data["fault_profiles"]]
    assert profile_ids == ["ss1-0-damper_jam", "ss1-1-fan_speed_degradation"]
    assert all("bdm" not in profile_id.lower() for profile_id in profile_ids)
    assert len(set(profile_ids)) == len(profile_ids)
    nonce = ss.episode_nonce(identity["episode_id"])
    assert nonce == hashlib.sha256(
        b"stress-search-v1|" + identity["episode_id"].encode("ascii")
    ).digest()
    assert nonce != hashlib.sha256(identity["episode_id"].encode("ascii")).digest()
    assert not ss._RNG_NAMESPACE.lower().startswith("bdm")
    seed = scenario.data["sensor_model"]["random_seed"]
    assert not SENSOR_SEED_BASE <= seed < ss.ROSTER_SENSOR_SEED_CEILING
    # The checked-in spec keeps the same separation.
    checked_in = checked_in_mapping()
    assert not checked_in["fixture"]["fixture_id"].lower().startswith("bdm")
    assert not any(item.lower().startswith("bdm") for item in checked_in["seeds"])


# ---------------------------------------------------------------------------
# Scoring helper used by the fabricated evaluator
# ---------------------------------------------------------------------------


def test_fabricated_rows_use_the_real_truth_layout_and_score_as_no_discovery(
    small_environment: ss.SearchEnvironment,
) -> None:
    twin = fabricated_rows(small_environment.twin_scenario)
    assert twin.shape == (32, 8 * ZONE_FIELD_COUNT + 3) == (32, 51)
    assert np.isfinite(twin).all()
    faults = [fault("damper_jam", onset_step=8, duration_steps=16)]
    scenario = ss.build_candidate_scenario(small_environment, faults, "d" * 64)
    metrics = ss.score_against_twin(small_environment, fabricated_rows(scenario), twin)
    assert 0.0 < metrics["peak_hazard_shift"] < 0.5
    assert metrics["peak_hazard_shift_family"] == "high_co2"
    assert metrics["peak_hazard_shift_zone"] == small_environment.zone_ids[0]
    assert metrics["hazard_exposure_delta"] == 0.0
    for tier in small_environment.spec.data["discovery_tiers"]:
        assert not ss.tier_met(tier, metrics)
    with pytest.raises(ss.StressSearchError, match="different shapes"):
        ss.score_against_twin(small_environment, twin[:-1], twin)
    broken = twin.copy()
    broken[0, 0] = float("nan")
    with pytest.raises(ss.StressSearchError, match="non-finite"):
        ss.score_against_twin(small_environment, broken, twin)


# ---------------------------------------------------------------------------
# Driver bookkeeping with the FABRICATED evaluator (not integration evidence)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ss.METHODS)
def test_simulation_call_budget_is_never_exceeded(tmp_path: Path, method: str) -> None:
    summary, fake = run_fake(tmp_path / "run", small_mapping(), method=method)
    assert summary.end_reason == ss.END_SIMULATION_BUDGET
    assert summary.final is True
    assert summary.simulation_calls_used == 10
    # One baseline call plus nine candidate calls: the budget is a hard cap.
    assert (fake.twin_calls, fake.candidate_calls) == (1, 9)
    records = candidates_of(tmp_path / "run")
    assert len(simulated(records)) == 9
    assert max(record["simulation_calls_used"] for record in records) == 10
    assert all(record["simulation_calls_used"] <= 10 for record in records)
    consumption = manifest_of(tmp_path / "run")["budget_consumption"]
    assert consumption["end_reason"] == ss.END_SIMULATION_BUDGET
    assert consumption["simulation_calls_used"] == 10
    assert consumption["candidate_attempts_used"] == summary.candidate_attempts_used


def test_candidate_attempt_budget_ends_the_run(tmp_path: Path) -> None:
    data = small_mapping()
    data["budget"]["max_candidate_attempts"] = 5
    data["budget"]["max_simulation_calls"] = 50
    summary, fake = run_fake(tmp_path / "run", data)
    assert summary.end_reason == ss.END_ATTEMPT_BUDGET
    assert summary.final is True
    assert summary.candidate_attempts_used == 5
    assert fake.candidate_calls <= 5
    assert summary.simulation_calls_used == 1 + fake.candidate_calls
    assert [record["attempt_index"] for record in candidates_of(tmp_path / "run")] == list(range(5))


def test_every_attempt_has_exactly_one_audit_record(tmp_path: Path) -> None:
    summary, fake = run_fake(tmp_path / "run", mixed_mapping())
    assert summary.end_reason == ss.END_ATTEMPT_BUDGET
    records = candidates_of(tmp_path / "run")
    assert [record["attempt_index"] for record in records] == list(range(80))
    assert summary.candidate_attempts_used == 80
    dispositions = {record["disposition"] for record in records}
    # This spec was chosen so the fixed seed produces every non-failure kind.
    assert dispositions == {
        ss.DISPOSITION_EVALUATED,
        ss.DISPOSITION_INVALID_PARAMETERS,
        ss.DISPOSITION_SCENARIO_VALIDATION,
        ss.DISPOSITION_DUPLICATE,
    }
    for record in records:
        assert type(record["parameters"]) is list and record["parameters"]
        if record["disposition"] == ss.DISPOSITION_INVALID_PARAMETERS:
            assert "exceeds latest_end_step" in record["reason"]
            assert record["episode_id"] is None and record["objective"] is None
        if record["disposition"] == ss.DISPOSITION_SCENARIO_VALIDATION:
            assert "may not overlap" in record["reason"]
            assert record["episode_id"] is not None and record["outcome"] is None
    # Counters are cumulative, monotone and move only for simulated attempts.
    calls = 1
    for index, record in enumerate(records):
        calls += record["disposition"] == ss.DISPOSITION_EVALUATED
        assert record["simulation_calls_used"] == calls
        assert record["candidate_attempts_used"] == index + 1
    assert summary.simulation_calls_used == calls == 1 + fake.candidate_calls
    manifest = manifest_of(tmp_path / "run")
    listed = [case["attempt_index"] for case in manifest["selected_cases"]] + [
        item["attempt_index"] for item in manifest["rejected_contenders"]
    ]
    assert sorted(listed) == list(range(80))
    assert all(item["not_selected_reason"] for item in manifest["rejected_contenders"])
    counted = manifest["budget_consumption"]["dispositions"]
    assert sum(counted.values()) == 80
    for name in ss.DISPOSITIONS:
        assert counted[name] == sum(1 for record in records if record["disposition"] == name)


def test_duplicate_episodes_are_recorded_but_never_re_simulated(tmp_path: Path) -> None:
    summary, fake = run_fake(tmp_path / "run", tiny_mapping())
    records = candidates_of(tmp_path / "run")
    assert len(records) == 60 and summary.end_reason == ss.END_ATTEMPT_BUDGET
    duplicates = [r for r in records if r["disposition"] == ss.DISPOSITION_DUPLICATE]
    # Only eight distinct episodes exist, so at least 52 attempts repeat one.
    assert len(duplicates) >= 52
    assert fake.candidate_calls <= 8
    assert summary.simulation_calls_used == 1 + fake.candidate_calls
    assert len(fake.started) == len(set(fake.started)) == fake.candidate_calls
    assert fake.started == [record["episode_id"] for record in simulated(records)]
    by_attempt = {record["attempt_index"]: record for record in records}
    for record in duplicates:
        first = by_attempt[record["duplicate_of_attempt_index"]]
        assert first["attempt_index"] < record["attempt_index"]
        assert first["disposition"] == ss.DISPOSITION_EVALUATED
        assert first["episode_id"] == record["episode_id"]
        assert first["scenario_sha256"] == record["scenario_sha256"]
        assert first["objective"] == record["objective"]
        assert record["outcome"] is None and record["metrics"] is None
        assert ss.canonical_faults(first["parameters"]) == ss.canonical_faults(
            record["parameters"]
        )
    # A duplicate is never selected and never counted as a distinct episode.
    manifest = manifest_of(tmp_path / "run")
    selected = {case["attempt_index"] for case in manifest["selected_cases"]}
    assert not selected & {record["attempt_index"] for record in duplicates}
    episode_ids = [
        episode for family in manifest["causal_families"] for episode in family["episode_ids"]
    ]
    assert len(episode_ids) == len(set(episode_ids))


def test_fixed_seed_reproduces_the_whole_candidate_sequence_and_selection(
    tmp_path: Path,
) -> None:
    for method in ss.METHODS:
        first, _ = run_fake(tmp_path / f"{method}-a", small_mapping(), method=method)
        second, _ = run_fake(tmp_path / f"{method}-b", small_mapping(), method=method)
        other, _ = run_fake(
            tmp_path / f"{method}-c", small_mapping(), method=method, seed="test-s2"
        )
        assert candidates_of(tmp_path / f"{method}-a") == candidates_of(tmp_path / f"{method}-b")
        assert first.manifest_sha256 == second.manifest_sha256
        assert first.audit_head_sha256 == second.audit_head_sha256
        assert first.manifest_sha256 != other.manifest_sha256
        assert [r["parameters"] for r in candidates_of(tmp_path / f"{method}-a")] != [
            r["parameters"] for r in candidates_of(tmp_path / f"{method}-c")
        ]
    assert [r["parameters"] for r in candidates_of(tmp_path / "random_search-a")] != [
        r["parameters"] for r in candidates_of(tmp_path / "cross_entropy-a")
    ]


def test_wall_deadline_between_episodes_ends_the_run(tmp_path: Path) -> None:
    data = single_fault_mapping()
    data["budget"]["max_wall_seconds"] = 4.0
    data["budget"]["max_simulation_calls"] = 50
    clock = FakeClock()
    fake = FakeEvaluator(clock=clock, seconds_per_episode=1.0)
    summary, _ = run_fake(tmp_path / "run", data, evaluator=fake, clock=clock)
    assert summary.end_reason == ss.END_WALL_BUDGET
    assert summary.final is True
    # Baseline plus three whole candidate episodes fit in four seconds.
    assert (fake.twin_calls, fake.candidate_calls) == (1, 3)
    assert summary.simulation_calls_used == 4
    audit = load_audit(tmp_path / "run")
    assert audit.records("incomplete") == []
    assert len(audit.records("candidate")) == 3
    assert (tmp_path / "run" / ss.MANIFEST_FILE_NAME).is_file()


def test_wall_deadline_mid_episode_records_an_incomplete_attempt(tmp_path: Path) -> None:
    data = single_fault_mapping()
    data["budget"]["max_wall_seconds"] = 4.5
    data["budget"]["max_simulation_calls"] = 50
    clock = FakeClock()
    fake = FakeEvaluator(clock=clock, seconds_per_episode=1.0)
    summary, _ = run_fake(tmp_path / "run", data, evaluator=fake, clock=clock)
    assert summary.end_reason == ss.END_WALL_BUDGET
    assert summary.final is True
    assert fake.candidate_calls == 4 and len(fake.completed) == 3
    # The interrupted episode is not charged to either budget ...
    assert summary.simulation_calls_used == 4
    assert summary.candidate_attempts_used == 3
    audit = load_audit(tmp_path / "run")
    candidates = audit.records("candidate")
    assert [record["attempt_index"] for record in candidates] == [0, 1, 2]
    # ... and is recorded explicitly instead of vanishing.
    (incomplete,) = audit.records("incomplete")
    assert incomplete["attempt_index"] == 3
    assert incomplete["cause"] == ss.INTERRUPT_DEADLINE
    assert incomplete["simulation_calls_used"] == 4
    assert incomplete["candidate_attempts_used"] == 3
    regenerated = ss.CandidateSampler(
        make_environment(data), ss.METHOD_RANDOM, "test-s1"
    ).propose(3)
    assert incomplete["parameters"] == ss.strip_slots(regenerated)
    kinds = [entry["record"]["kind"] for entry in audit.entries]
    assert kinds[-3:] == ["incomplete", "session_end", "run_footer"]
    manifest = manifest_of(tmp_path / "run")
    assert manifest["budget_consumption"]["incomplete_attempts"] == 1
    assert manifest["budget_consumption"]["end_reason"] == ss.END_WALL_BUDGET
    # A wall-time ending is final: the budget cannot be bought back by resuming.
    with pytest.raises(ss.StressSearchAuditError, match="already complete; nothing to resume"):
        resume_fake(tmp_path / "run")
    assert len(load_audit(tmp_path / "run").entries) == len(audit.entries)


def test_cancellation_between_attempts_is_not_final_and_writes_no_manifest(
    tmp_path: Path,
) -> None:
    fake = FakeEvaluator(cancel_at_candidate_call=3, cancel_phase="end")
    summary, _ = run_fake(tmp_path / "run", small_mapping(), evaluator=fake)
    assert summary.end_reason == ss.END_CANCELLED
    assert summary.final is False
    assert summary.manifest_sha256 is None
    assert not (tmp_path / "run" / ss.MANIFEST_FILE_NAME).exists()
    assert not (tmp_path / "run" / ss.CASES_DIRECTORY_NAME).exists()
    assert summary.simulation_calls_used == 4 and fake.candidate_calls == 3
    audit = load_audit(tmp_path / "run")
    assert audit.records("run_footer") == [] and audit.records("incomplete") == []
    (session_end,) = audit.records("session_end")
    assert session_end["end_reason"] == ss.END_CANCELLED and session_end["final"] is False
    assert audit.entries[-1]["record"] == session_end


@pytest.mark.parametrize("method", ss.METHODS)
@pytest.mark.parametrize("phase", ["mid", "end"])
def test_resume_after_cancellation_matches_an_uninterrupted_run(
    tmp_path: Path, method: str, phase: str
) -> None:
    straight, straight_fake = run_fake(tmp_path / "straight", small_mapping(), method=method)
    first_fake = FakeEvaluator(cancel_at_candidate_call=4, cancel_phase=phase)
    first, _ = run_fake(tmp_path / "resumed", small_mapping(), method=method, evaluator=first_fake)
    assert (first.end_reason, first.final) == (ss.END_CANCELLED, False)
    assert not (tmp_path / "resumed" / ss.MANIFEST_FILE_NAME).exists()
    recorded_before = candidates_of(tmp_path / "resumed")
    incomplete_before = load_audit(tmp_path / "resumed").records("incomplete")
    assert len(incomplete_before) == (1 if phase == "mid" else 0)

    # A second interruption, then the finishing session.
    second_fake = FakeEvaluator(cancel_at_candidate_call=2, cancel_phase=phase)
    second, _ = resume_fake(tmp_path / "resumed", evaluator=second_fake)
    assert (second.end_reason, second.final) == (ss.END_CANCELLED, False)
    assert candidates_of(tmp_path / "resumed")[: len(recorded_before)] == recorded_before
    final, third_fake = resume_fake(tmp_path / "resumed")

    assert (final.end_reason, final.final) == (straight.end_reason, True)
    assert final.simulation_calls_used == straight.simulation_calls_used == 10
    assert final.candidate_attempts_used == straight.candidate_attempts_used
    # Same candidates, same order, same recorded results and counters.
    resumed_records = candidates_of(tmp_path / "resumed")
    assert resumed_records == candidates_of(tmp_path / "straight")
    assert [r["attempt_index"] for r in resumed_records] == list(range(len(resumed_records)))
    audit = load_audit(tmp_path / "resumed")
    assert audit.records("run_footer") == load_audit(tmp_path / "straight").records("run_footer")
    assert audit.records("baseline") == load_audit(tmp_path / "straight").records("baseline")
    assert [record["session_index"] for record in audit.records("session_start")] == [0, 1, 2]
    assert [record["end_reason"] for record in audit.records("session_end")] == [
        ss.END_CANCELLED,
        ss.END_CANCELLED,
        straight.end_reason,
    ]
    # Nothing recorded was simulated twice and nothing was skipped: across the
    # three sessions each candidate episode completed exactly once.
    completed = first_fake.completed + second_fake.completed + third_fake.completed
    assert completed == straight_fake.completed
    assert len(completed) == len(set(completed)) == 9
    recorded_once = [record["episode_id"] for record in simulated(resumed_records)]
    assert recorded_once == straight_fake.completed
    # The healthy twin is recomputed per session but charged only once.
    assert (first_fake.twin_calls, second_fake.twin_calls, third_fake.twin_calls) == (1, 1, 1)
    assert len(audit.records("baseline")) == 1
    # Selection and every result-bearing manifest field agree with the
    # uninterrupted run; only session provenance differs.
    resumed_manifest = manifest_of(tmp_path / "resumed")
    straight_manifest = manifest_of(tmp_path / "straight")
    for name in ss._MANIFEST_FIELDS:
        if name in ("budget_consumption", "audit", "manifest_sha256"):
            continue
        assert resumed_manifest[name] == straight_manifest[name], name
    resumed_budget = dict(resumed_manifest["budget_consumption"])
    straight_budget = dict(straight_manifest["budget_consumption"])
    assert resumed_budget.pop("sessions") == 3 and straight_budget.pop("sessions") == 1
    assert resumed_budget.pop("incomplete_attempts") == (2 if phase == "mid" else 0)
    assert straight_budget.pop("incomplete_attempts") == 0
    assert resumed_budget == straight_budget
    # A finished run cannot be resumed again.
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        resume_fake(tmp_path / "resumed")


def test_resume_does_not_reset_budgets_or_wall_time(tmp_path: Path) -> None:
    data = single_fault_mapping()
    data["budget"]["max_wall_seconds"] = 8.0
    data["budget"]["max_simulation_calls"] = 50
    first_clock = FakeClock(start=10.0)
    first_fake = FakeEvaluator(
        clock=first_clock, seconds_per_episode=1.0, cancel_at_candidate_call=4, cancel_phase="end"
    )
    first, _ = run_fake(tmp_path / "run", data, evaluator=first_fake, clock=first_clock)
    assert (first.end_reason, first.simulation_calls_used) == (ss.END_CANCELLED, 5)
    assert ss._prior_wall_seconds(load_audit(tmp_path / "run")) == 5.0

    # A fresh process: its clock has an unrelated origin.
    second_clock = FakeClock(start=777_000.0)
    second_fake = FakeEvaluator(clock=second_clock, seconds_per_episode=1.0)
    second, _ = resume_fake(tmp_path / "run", evaluator=second_fake, clock=second_clock)
    assert second.end_reason == ss.END_WALL_BUDGET
    assert second.final is True
    # Five seconds were already spent, one more went on re-deriving the twin, so
    # only two further episodes fit although this session ran for three seconds.
    assert second_fake.candidate_calls == 2
    assert second_clock.now - 777_000.0 == 3.0
    # Counters continue from the recorded values instead of restarting.
    assert second.simulation_calls_used == 7
    assert second.candidate_attempts_used == 6
    audit = load_audit(tmp_path / "run")
    records = audit.records("candidate")
    assert [record["attempt_index"] for record in records] == list(range(6))
    assert [record["simulation_calls_used"] for record in records] == [2, 3, 4, 5, 6, 7]
    resumed_start = [
        entry for entry in audit.entries if entry["record"]["kind"] == "session_start"
    ][1]
    assert resumed_start["record"]["session_index"] == 1
    assert resumed_start["timing"]["cumulative_wall_seconds"] == 5.0
    cumulative = [
        entry["timing"]["cumulative_wall_seconds"] for entry in audit.entries if entry["timing"]
    ]
    assert cumulative == sorted(cumulative) and cumulative[-1] == 8.0
    assert manifest_of(tmp_path / "run")["budget_consumption"]["sessions"] == 2


def test_resume_cannot_extend_an_exhausted_simulation_budget(tmp_path: Path) -> None:
    data = small_mapping()
    fake = FakeEvaluator(cancel_at_candidate_call=9, cancel_phase="end")
    first, _ = run_fake(tmp_path / "run", data, evaluator=fake)
    # The budget check runs before the cancellation check, so the run is final.
    assert (first.end_reason, first.final) == (ss.END_SIMULATION_BUDGET, True)
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        resume_fake(tmp_path / "run")

    fake = FakeEvaluator(cancel_at_candidate_call=8, cancel_phase="end")
    first, _ = run_fake(tmp_path / "late", data, evaluator=fake)
    assert (first.end_reason, first.simulation_calls_used) == (ss.END_CANCELLED, 9)
    final, resumed_fake = resume_fake(tmp_path / "late")
    assert final.end_reason == ss.END_SIMULATION_BUDGET
    assert final.simulation_calls_used == 10
    assert resumed_fake.candidate_calls == 1


@pytest.mark.parametrize("mode", ["nan_rows", "inf_rows", "nonfinite_status"])
def test_nonfinite_simulator_output_is_recorded_and_counted(tmp_path: Path, mode: str) -> None:
    fake = FakeEvaluator(modes={2: mode})
    summary, _ = run_fake(tmp_path / "run", single_fault_mapping(), evaluator=fake)
    records = candidates_of(tmp_path / "run")
    (bad,) = [r for r in records if r["disposition"] == ss.DISPOSITION_NONFINITE]
    assert bad["attempt_index"] == 1
    assert "non-finite" in bad["reason"]
    assert bad["metrics"] is None and bad["objective"] is None
    assert bad["episode_id"] is not None and bad["scenario_sha256"] is not None
    # It consumed a simulation call like any other episode.
    assert bad["simulation_calls_used"] == records[0]["simulation_calls_used"] + 1
    assert summary.simulation_calls_used == 10 and fake.candidate_calls == 9
    manifest = manifest_of(tmp_path / "run")
    assert manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_NONFINITE] == 1
    assert bad["attempt_index"] not in [c["attempt_index"] for c in manifest["selected_cases"]]
    (contender,) = [c for c in manifest["rejected_contenders"] if c["attempt_index"] == 1]
    assert contender["not_selected_reason"] == f"not eligible: {ss.DISPOSITION_NONFINITE}"
    assert "NaN" not in (tmp_path / "run" / ss.AUDIT_FILE_NAME).read_text(encoding="utf-8")
    assert "Infinity" not in (tmp_path / "run" / ss.MANIFEST_FILE_NAME).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("raise", "RuntimeError: fabricated simulator crash"),
        ("short", "without a full truth trajectory"),
        ("no_rows", "without a full truth trajectory"),
    ],
)
def test_simulator_error_is_recorded_and_the_search_continues(
    tmp_path: Path, mode: str, reason: str
) -> None:
    fake = FakeEvaluator(modes={1: mode, 3: mode})
    summary, _ = run_fake(tmp_path / "run", single_fault_mapping(), evaluator=fake)
    assert summary.end_reason == ss.END_SIMULATION_BUDGET
    records = candidates_of(tmp_path / "run")
    failed = [r for r in records if r["disposition"] == ss.DISPOSITION_SIMULATOR_ERROR]
    assert [record["attempt_index"] for record in failed] == [0, 2]
    for record in failed:
        assert reason in record["reason"]
        assert record["objective"] is None and record["metrics"] is None
    assert len(records) == 9 and summary.simulation_calls_used == 10
    assert [r["disposition"] for r in records].count(ss.DISPOSITION_EVALUATED) == 7
    manifest = manifest_of(tmp_path / "run")
    assert manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_SIMULATOR_ERROR] == 2
    assert not {0, 2} & {case["attempt_index"] for case in manifest["selected_cases"]}


def test_failed_episode_is_not_re_simulated_when_proposed_again(tmp_path: Path) -> None:
    fake = FakeEvaluator(modes={index: "raise" for index in range(1, 9)})
    run_fake(tmp_path / "run", tiny_mapping(), evaluator=fake)
    records = candidates_of(tmp_path / "run")
    assert len(fake.started) == len(set(fake.started)) == fake.candidate_calls
    failed = {
        r["attempt_index"] for r in records if r["disposition"] == ss.DISPOSITION_SIMULATOR_ERROR
    }
    assert len(failed) == fake.candidate_calls >= 4
    repeats = [r for r in records if r["duplicate_of_attempt_index"] in failed]
    assert len(repeats) == 60 - len(failed)
    for record in repeats:
        assert record["disposition"] == ss.DISPOSITION_DUPLICATE
        assert record["objective"] is None


def test_terminal_failure_is_recorded_and_never_selected(tmp_path: Path) -> None:
    fake = FakeEvaluator(modes={1: "terminal", 2: "terminal"})
    summary, _ = run_fake(tmp_path / "run", single_fault_mapping(), evaluator=fake)
    records = candidates_of(tmp_path / "run")
    terminal = [r for r in records if r["disposition"] == ss.DISPOSITION_TERMINAL]
    assert [record["attempt_index"] for record in terminal] == [0, 1]
    for record in terminal:
        assert record["reason"] == "HMC terminal failure: FABRICATED_TERMINAL"
        assert record["outcome"]["terminal_reason"] == "FABRICATED_TERMINAL"
        assert record["outcome"]["committed_steps"] == 5
        assert record["objective"] is None and record["metrics"] is None
    assert summary.simulation_calls_used == 10
    manifest = manifest_of(tmp_path / "run")
    assert manifest["selected_cases"]
    assert not {0, 1} & {case["attempt_index"] for case in manifest["selected_cases"]}
    for attempt in (0, 1):
        (contender,) = [
            c for c in manifest["rejected_contenders"] if c["attempt_index"] == attempt
        ]
        assert contender["not_selected_reason"] == f"not eligible: {ss.DISPOSITION_TERMINAL}"
    assert manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_TERMINAL] == 2
    selected, reasons = ss.select_candidates(make_spec(single_fault_mapping()), records)
    assert all(case["disposition"] == ss.DISPOSITION_EVALUATED for case in selected)
    assert reasons[0] == reasons[1] == "not eligible: TERMINAL_FAILURE"


_COMPLETED_BUT_UNUSABLE_TWINS = ("short", "no_rows", "nan_rows", "inf_rows")


@pytest.mark.parametrize(
    "twin_mode",
    [
        "raise",
        "terminal",
        "nonfinite_status",
        *_COMPLETED_BUT_UNUSABLE_TWINS,
        "nan_metrics",
        "bad_status",
        "negative_steps",
        "not_outcome",
    ],
)
def test_baseline_failure_ends_the_run(tmp_path: Path, twin_mode: str) -> None:
    fake = FakeEvaluator(twin_mode=twin_mode)
    summary, _ = run_fake(tmp_path / "run", small_mapping(), evaluator=fake)
    assert summary.end_reason == ss.END_BASELINE_FAILED
    assert summary.final is True
    assert (fake.twin_calls, fake.candidate_calls) == (1, 0)
    assert summary.simulation_calls_used == 1 and summary.candidate_attempts_used == 0
    audit = load_audit(tmp_path / "run")
    assert audit.records("candidate") == []
    (baseline,) = audit.records("baseline")
    assert (baseline["outcome"]["status"] == ss.OUTCOME_COMPLETED) is (
        twin_mode in _COMPLETED_BUT_UNUSABLE_TWINS
    )
    assert "NaN" not in (tmp_path / "run" / ss.AUDIT_FILE_NAME).read_text(encoding="utf-8")
    manifest = run_manifest_of(tmp_path / "run")
    assert manifest["budget_consumption"]["end_reason"] == ss.END_BASELINE_FAILED
    assert manifest["selected_cases"] == [] and manifest["rejected_contenders"] == []
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        resume_fake(tmp_path / "run")


def test_nonfinite_baseline_rows_end_the_run_as_baseline_failed(tmp_path: Path) -> None:
    fake = FakeEvaluator(twin_mode="nan_rows")
    summary, _ = run_fake(tmp_path / "run", small_mapping(), evaluator=fake)
    assert summary.end_reason == ss.END_BASELINE_FAILED
    assert fake.candidate_calls == 0


def test_run_search_refuses_bad_arguments_and_existing_output(tmp_path: Path) -> None:
    spec = make_spec(small_mapping())
    common: dict[str, Any] = {
        "spec": spec,
        "repo_root": REPO_ROOT,
        "evaluator": FakeEvaluator(),
        "confirm_evaluator": FakeEvaluator(),
        "clock": FakeClock(),
    }
    with pytest.raises(ss.StressSearchError, match="unknown method 'grid'"):
        ss.run_search(output_dir=tmp_path / "a", method="grid", seed="test-s1", **common)
    with pytest.raises(ss.StressSearchError, match="seed 'issue93-s1' is not declared"):
        ss.run_search(
            output_dir=tmp_path / "a", method=ss.METHOD_RANDOM, seed="issue93-s1", **common
        )
    assert not (tmp_path / "a").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite existing output"):
        ss.run_search(output_dir=existing, method=ss.METHOD_RANDOM, seed="test-s1", **common)
    assert list(existing.iterdir()) == []
    run_fake(tmp_path / "done", small_mapping())
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite existing output"):
        run_fake(tmp_path / "done", small_mapping())


def test_fabricated_confirmation_is_never_reported_as_confirmed(tmp_path: Path) -> None:
    run_fake(tmp_path / "run", small_mapping())
    manifest = manifest_of(tmp_path / "run")
    assert 1 <= len(manifest["selected_cases"]) <= 3
    for case in manifest["selected_cases"]:
        # The fake produces no control trace and no authoritative replay, so the
        # harness must not label its output as confirmed.
        assert case["confirmation"]["replay_verified"] is False
        assert case["confirmation"]["status"] == "CONFIRMATION_MISMATCH"
        stored = tmp_path / "run" / case["files"]["scenario.json"]["relative_path"]
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == case["scenario_sha256"] or (
            hashlib.sha256(stored.read_bytes()).hexdigest()
            == case["files"]["scenario.json"]["sha256"]
        )
    assert manifest["budget_consumption"]["confirmation_calls_outside_search_budget"] == 1 + len(
        manifest["selected_cases"]
    )
    assert ss.summarize_run(manifest)["confirmed_cases"] == 0


def test_outputs_carry_the_data_boundary_and_keep_gpu_screening_open(tmp_path: Path) -> None:
    run_fake(tmp_path / "run", small_mapping())
    audit = load_audit(tmp_path / "run")
    header = audit.entries[0]["record"]
    assert header["kind"] == "run_header"
    assert header["data_boundary"] == ss.DATA_BOUNDARY
    assert header["identity"]["spec_sha256"] == make_spec(small_mapping()).spec_sha256
    manifest = manifest_of(tmp_path / "run")
    assert manifest["data_boundary"] == ss.DATA_BOUNDARY
    for phrase in ("Development-only", "biased discoveries", "not IID", "BLIND_FINAL"):
        assert phrase in ss.DATA_BOUNDARY
    assert manifest["gpu_screening"]["enabled"] is False
    assert manifest["gpu_screening"]["status"] == "OPEN_NOT_RUN"
    assert manifest["spec"]["screening"] == {"backend": "none"}
    assert manifest["spec"]["population"]["protected_population_access"] == "prohibited"
    assert manifest["split_rule"] == "every episode of one causal_family_id stays in one partition"
    assert manifest["replay_command"][-4:] == [
        "--manifest",
        "out/<run-directory>/manifest.json",
        "--output",
        "out/<new-replay-directory>",
    ]
    # Wall-clock values stay outside every hashed payload.
    assert "timing" not in json.dumps(manifest)
    for entry in audit.entries:
        assert "timing" not in json.dumps(entry["record"])
        assert "cumulative_wall_seconds" not in json.dumps(entry["record"])
    # Privileged truth rows are scored, never stored.
    spec_bytes = (tmp_path / "run" / ss.SPEC_FILE_NAME).read_bytes()
    assert spec_bytes == ss.canonical_bytes(small_mapping())
    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        ss.AUDIT_FILE_NAME,
        ss.CASES_DIRECTORY_NAME,
        ss.MANIFEST_FILE_NAME,
        ss.LOCK_FILE_NAME,
        ss.SPEC_FILE_NAME,
    ]
    assert ss.LOCK_FILE_NAME == "run.lock"


# ---------------------------------------------------------------------------
# Audit integrity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def finished_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("finished-template") / "run"
    run_fake(run, small_mapping())
    return run


@pytest.fixture(scope="module")
def cancelled_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("cancelled-template") / "run"
    run_fake(
        run, small_mapping(), evaluator=FakeEvaluator(cancel_at_candidate_call=4, cancel_phase="end")
    )
    return run


@pytest.fixture()
def finished_run(finished_template: Path, tmp_path: Path) -> Path:
    """A private copy of one finished fabricated run (random_search, test-s1)."""

    return Path(shutil.copytree(finished_template, tmp_path / "run"))


@pytest.fixture()
def cancelled_run(cancelled_template: Path, tmp_path: Path) -> Path:
    """A private copy of one fabricated run cancelled after four candidates."""

    return Path(shutil.copytree(cancelled_template, tmp_path / "run"))


def audit_lines(run: Path) -> list[bytes]:
    return (run / ss.AUDIT_FILE_NAME).read_bytes().split(b"\n")[:-1]


def write_audit(run: Path, lines: list[bytes], *, trailing: bytes = b"\n") -> Path:
    path = run / ss.AUDIT_FILE_NAME
    path.write_bytes(b"\n".join(lines) + trailing)
    return path


def test_audit_log_round_trips_and_chains_every_record(finished_run: Path) -> None:
    audit = load_audit(finished_run)
    kinds = [entry["record"]["kind"] for entry in audit.entries]
    assert kinds[:3] == ["run_header", "session_start", "baseline"]
    assert kinds[-2:] == ["session_end", "run_footer"]
    previous = "0" * 64
    for sequence, entry in enumerate(audit.entries):
        assert entry["sequence"] == sequence
        assert entry["previous_record_sha256"] == previous
        assert entry["record_sha256"] == ss.canonical_sha256(
            {"sequence": sequence, "previous_record_sha256": previous, "record": entry["record"]}
        )
        previous = entry["record_sha256"]
    assert audit.head_sha256 == previous
    assert manifest_of(finished_run)["audit"]["footer_record_sha256"] == previous
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite existing audit log"):
        ss.AuditLog.create(finished_run / ss.AUDIT_FILE_NAME)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (b'"attempt_index":1,', b'"attempt_index":7,'),
        (b'"disposition":"EVALUATED"', b'"disposition":"EVALUATEE"'),
        (b'"simulation_calls_used":3', b'"simulation_calls_used":2'),
        (b'"sequence":4,', b'"sequence":5,'),
    ],
    ids=["attempt_index", "disposition", "budget_counter", "sequence"],
)
def test_audit_load_rejects_a_flipped_byte(
    finished_run: Path, needle: bytes, replacement: bytes
) -> None:
    lines = audit_lines(finished_run)
    index = next(i for i, line in enumerate(lines) if needle in line)
    lines[index] = lines[index].replace(needle, replacement, 1)
    path = write_audit(finished_run, lines)
    with pytest.raises(ss.StressSearchAuditError, match="digest mismatch|chain is broken"):
        ss.AuditLog.load(path)


@pytest.mark.parametrize("field", ["record_sha256", "previous_record_sha256"])
def test_audit_load_rejects_a_flipped_digest_character(finished_run: Path, field: str) -> None:
    lines = audit_lines(finished_run)
    entry = json.loads(lines[5])
    digest = entry[field]
    entry[field] = ("0" if digest[0] != "0" else "1") + digest[1:]
    lines[5] = ss.canonical_bytes(entry)
    path = write_audit(finished_run, lines)
    with pytest.raises(ss.StressSearchAuditError, match="digest mismatch|chain is broken"):
        ss.AuditLog.load(path)


def test_audit_load_rejects_any_single_flipped_bit_in_a_chained_record(
    finished_run: Path,
) -> None:
    lines = audit_lines(finished_run)
    original = lines[4]
    start = original.index(b'"record":')
    end = original.index(b',"record_sha256"')
    for offset in range(start + len(b'"record":'), end, 7):
        mutated = bytearray(original)
        mutated[offset] ^= 0x01
        path = write_audit(finished_run, [*lines[:4], bytes(mutated), *lines[5:]])
        with pytest.raises(ss.StressSearchAuditError):
            ss.AuditLog.load(path)


@pytest.mark.parametrize("index", [0, 1, 4])
def test_audit_load_rejects_a_deleted_line(finished_run: Path, index: int) -> None:
    lines = audit_lines(finished_run)
    del lines[index]
    path = write_audit(finished_run, lines)
    with pytest.raises(ss.StressSearchAuditError, match="chain is broken at line"):
        ss.AuditLog.load(path)


def test_audit_load_rejects_reordered_and_duplicated_lines(finished_run: Path) -> None:
    lines = audit_lines(finished_run)
    swapped = list(lines)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    with pytest.raises(ss.StressSearchAuditError, match="chain is broken at line 4"):
        ss.AuditLog.load(write_audit(finished_run, swapped))
    repeated = [*lines[:4], lines[3], *lines[4:]]
    with pytest.raises(ss.StressSearchAuditError, match="chain is broken at line 5"):
        ss.AuditLog.load(write_audit(finished_run, repeated))


def test_audit_load_rejects_a_partial_trailing_line(finished_run: Path) -> None:
    lines = audit_lines(finished_run)
    with pytest.raises(ss.StressSearchAuditError, match="ends with a partial record"):
        ss.AuditLog.load(write_audit(finished_run, lines, trailing=b""))
    half = lines[-1][: len(lines[-1]) // 2]
    with pytest.raises(ss.StressSearchAuditError, match="ends with a partial record"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:-1], half], trailing=b""))
    with pytest.raises(ss.StressSearchAuditError, match="not strict JSON"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:-1], half]))
    with pytest.raises(ss.StressSearchAuditError, match="not strict JSON"):
        ss.AuditLog.load(write_audit(finished_run, [*lines, b""]))


def test_audit_load_rejects_non_canonical_and_malformed_lines(finished_run: Path) -> None:
    lines = audit_lines(finished_run)
    entry = json.loads(lines[2])
    spaced = json.dumps(entry, sort_keys=True).encode("utf-8")
    assert json.loads(spaced) == entry and spaced != lines[2]
    with pytest.raises(ss.StressSearchAuditError, match="audit line 3 is not canonical JSON"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:2], spaced, *lines[3:]]))
    windows = lines[2] + b"\r"
    with pytest.raises(ss.StressSearchAuditError, match="audit line 3 is not canonical JSON"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:2], windows, *lines[3:]]))
    extra = ss.canonical_bytes({**entry, "note": "added"})
    with pytest.raises(ss.StressSearchAuditError, match="unknown audit entry fields: note"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:2], extra, *lines[3:]]))
    stripped = ss.canonical_bytes({k: v for k, v in entry.items() if k != "timing"})
    with pytest.raises(ss.StressSearchAuditError, match="missing audit entry fields: timing"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:2], stripped, *lines[3:]]))
    duplicated = lines[2][:-1] + b',"timing":{}}'
    with pytest.raises(ss.StressSearchAuditError, match="duplicate JSON key 'timing'"):
        ss.AuditLog.load(write_audit(finished_run, [*lines[:2], duplicated, *lines[3:]]))
    with pytest.raises(ss.StressSearchAuditError, match="cannot read audit log"):
        ss.AuditLog.load(finished_run / "absent.jsonl")


def test_resume_refuses_a_tampered_audit(cancelled_run: Path) -> None:
    lines = audit_lines(cancelled_run)
    index = next(i for i, line in enumerate(lines) if b'"kind":"candidate"' in line)
    lines[index] = lines[index].replace(
        b'"disposition":"EVALUATED"', b'"disposition":"TERMINAL_FAILURE"', 1
    )
    assert lines != audit_lines(cancelled_run)
    write_audit(cancelled_run, lines)
    with pytest.raises(ss.StressSearchAuditError, match="digest mismatch"):
        resume_fake(cancelled_run)
    assert not (cancelled_run / ss.MANIFEST_FILE_NAME).exists()


def test_resume_refuses_a_run_whose_spec_was_altered(cancelled_run: Path) -> None:
    before = (cancelled_run / ss.AUDIT_FILE_NAME).read_bytes()
    altered = small_mapping()
    altered["budget"]["max_simulation_calls"] = 1000
    (cancelled_run / ss.SPEC_FILE_NAME).write_bytes(ss.canonical_bytes(altered))
    with pytest.raises(ss.StressSearchAuditError, match="run identity does not match"):
        resume_fake(cancelled_run)
    widened = small_mapping()
    widened["budget"]["max_wall_seconds"] = 2.0e6
    (cancelled_run / ss.SPEC_FILE_NAME).write_bytes(ss.canonical_bytes(widened))
    with pytest.raises(ss.StressSearchAuditError, match="run identity does not match"):
        resume_fake(cancelled_run)
    invalid = small_mapping()
    invalid["screening"]["backend"] = "warp"
    (cancelled_run / ss.SPEC_FILE_NAME).write_bytes(ss.canonical_bytes(invalid))
    with pytest.raises(ss.StressSearchSpecError, match="accelerated screening is refused"):
        resume_fake(cancelled_run)
    # Refusals leave the audit untouched and write no manifest.
    assert (cancelled_run / ss.AUDIT_FILE_NAME).read_bytes() == before
    assert not (cancelled_run / ss.MANIFEST_FILE_NAME).exists()
    # Restoring the original spec makes the run resumable again.
    (cancelled_run / ss.SPEC_FILE_NAME).write_bytes(ss.canonical_bytes(small_mapping()))
    final, _ = resume_fake(cancelled_run)
    assert final.final is True


def test_resume_refuses_a_headerless_or_missing_run(cancelled_run: Path, tmp_path: Path) -> None:
    lines = audit_lines(cancelled_run)
    (cancelled_run / ss.AUDIT_FILE_NAME).write_bytes(b"")
    with pytest.raises(ss.StressSearchAuditError, match="audit log has no run header"):
        resume_fake(cancelled_run)
    write_audit(cancelled_run, lines)
    (cancelled_run / ss.AUDIT_FILE_NAME).unlink()
    with pytest.raises(ss.StressSearchAuditError, match="not a run directory"):
        resume_fake(cancelled_run)
    (cancelled_run / ss.SPEC_FILE_NAME).unlink()
    with pytest.raises(ss.StressSearchAuditError, match="not a run directory"):
        resume_fake(cancelled_run)
    with pytest.raises(ss.StressSearchAuditError, match="run directory does not exist"):
        resume_fake(tmp_path / "never-started")
    assert not (tmp_path / "never-started").exists()


def test_resume_refuses_a_baseline_that_no_longer_reproduces(cancelled_run: Path) -> None:
    before = load_audit(cancelled_run)
    with pytest.raises(ss.StressSearchAuditError, match="no longer reproduces the recorded baseline"):
        resume_fake(cancelled_run, evaluator=FakeEvaluator(twin_mode="short"))
    assert not (cancelled_run / ss.MANIFEST_FILE_NAME).exists()
    # The refused session is still closed in the audit, as an aborted one.
    after = load_audit(cancelled_run)
    assert after.entries[: len(before.entries)] == before.entries
    added = [entry["record"] for entry in after.entries[len(before.entries) :]]
    assert [record["kind"] for record in added] == ["session_start", "session_end"]
    assert added[1] == {
        "kind": "session_end",
        "session_index": 1,
        "end_reason": ss.END_ABORTED,
        "final": False,
        "error_type": "StressSearchAuditError",
        "simulation_calls_used": 1,
        "candidate_attempts_used": 0,
    }
    # With the reproducing baseline back the run still finishes.
    final, _ = resume_fake(cancelled_run)
    assert final.final is True and final.simulation_calls_used == 10


def test_resume_of_a_completed_run_raises(finished_run: Path) -> None:
    before = (finished_run / ss.AUDIT_FILE_NAME).read_bytes()
    manifest_before = (finished_run / ss.MANIFEST_FILE_NAME).read_bytes()
    fake = FakeEvaluator()
    with pytest.raises(ss.StressSearchAuditError, match="run is already complete; nothing to resume"):
        resume_fake(finished_run, evaluator=fake)
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    assert (finished_run / ss.AUDIT_FILE_NAME).read_bytes() == before
    assert (finished_run / ss.MANIFEST_FILE_NAME).read_bytes() == manifest_before


def test_unfinished_finalisation_is_completed_without_new_search_work(
    finished_run: Path,
) -> None:
    manifest_bytes = (finished_run / ss.MANIFEST_FILE_NAME).read_bytes()
    audit_bytes = (finished_run / ss.AUDIT_FILE_NAME).read_bytes()
    (finished_run / ss.MANIFEST_FILE_NAME).unlink()
    fake = FakeEvaluator()
    summary, _ = resume_fake(finished_run, evaluator=fake)
    assert summary.final is True and summary.end_reason == ss.END_SIMULATION_BUDGET
    assert summary.simulation_calls_used == 10
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    assert (finished_run / ss.MANIFEST_FILE_NAME).read_bytes() == manifest_bytes
    assert (finished_run / ss.AUDIT_FILE_NAME).read_bytes() == audit_bytes


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def contender(
    attempt: int,
    objective: float | None,
    *,
    family: str = "f0",
    episode: str | None = None,
    disposition: str = ss.DISPOSITION_EVALUATED,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt,
        "disposition": disposition,
        "objective": objective,
        "episode_id": episode or f"e{attempt:03d}",
        "causal_family_id": family,
    }


def selection_spec(**selection: Any) -> ss.SearchSpec:
    data = small_mapping()
    data["selection"].update(selection)
    return make_spec(data)


def test_selection_honours_shortlist_size_and_ranks_by_objective() -> None:
    spec = selection_spec(shortlist_size=2, max_per_causal_family=1)
    pool = [
        contender(0, 0.2, family="a"),
        contender(1, 0.9, family="b"),
        contender(2, 0.5, family="c"),
        contender(3, 0.7, family="d"),
    ]
    selected, reasons = ss.select_candidates(spec, pool)
    assert [case["attempt_index"] for case in selected] == [1, 3]
    assert reasons == {2: "ranked below the shortlist", 0: "ranked below the shortlist"}
    assert pool[1] == contender(1, 0.9, family="b")
    none_selected, all_reasons = ss.select_candidates(selection_spec(shortlist_size=0), pool)
    assert none_selected == []
    assert set(all_reasons) == {0, 1, 2, 3}


def test_selection_honours_max_per_causal_family() -> None:
    pool = [
        contender(0, 0.9, family="a"),
        contender(1, 0.8, family="a"),
        contender(2, 0.7, family="a"),
        contender(3, 0.6, family="b"),
        contender(4, 0.5, family="c"),
    ]
    selected, reasons = ss.select_candidates(
        selection_spec(shortlist_size=3, max_per_causal_family=1), pool
    )
    # Capped family members do not consume shortlist places.
    assert [case["attempt_index"] for case in selected] == [0, 3, 4]
    assert reasons == {
        1: "a higher-ranked member of the same causal family was selected",
        2: "a higher-ranked member of the same causal family was selected",
    }
    selected, reasons = ss.select_candidates(
        selection_spec(shortlist_size=3, max_per_causal_family=2), pool
    )
    assert [case["attempt_index"] for case in selected] == [0, 1, 3]
    assert reasons[2] == "a higher-ranked member of the same causal family was selected"
    assert reasons[4] == "ranked below the shortlist"


def test_selection_honours_the_exclusive_minimum_objective() -> None:
    pool = [
        contender(0, 0.25, family="a"),
        contender(1, 0.2500001, family="b"),
        contender(2, -1.0, family="c"),
        contender(3, 0.0, family="d"),
    ]
    selected, reasons = ss.select_candidates(
        selection_spec(shortlist_size=4, minimum_objective_exclusive=0.25), pool
    )
    assert [case["attempt_index"] for case in selected] == [1]
    assert reasons == {
        0: "objective not above minimum_objective_exclusive",
        2: "objective not above minimum_objective_exclusive",
        3: "objective not above minimum_objective_exclusive",
    }


def test_selection_tie_break_is_deterministic_and_order_independent() -> None:
    spec = selection_spec(shortlist_size=2, max_per_causal_family=1)
    pool = [
        contender(0, 0.5, family="a", episode="cc"),
        contender(1, 0.5, family="b", episode="aa"),
        contender(2, 0.5, family="c", episode="bb"),
        contender(3, 0.5, family="d", episode="dd"),
    ]
    expected = [1, 2]
    for order in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1), (1, 3, 0, 2)):
        selected, reasons = ss.select_candidates(spec, [pool[index] for index in order])
        assert [case["attempt_index"] for case in selected] == expected
        assert reasons == {0: "ranked below the shortlist", 3: "ranked below the shortlist"}


def test_selection_never_picks_an_ineligible_disposition() -> None:
    pool = [
        contender(0, None, disposition=ss.DISPOSITION_INVALID_PARAMETERS),
        contender(1, None, disposition=ss.DISPOSITION_SCENARIO_VALIDATION),
        contender(2, 5.0, family="a", disposition=ss.DISPOSITION_DUPLICATE),
        contender(3, None, disposition=ss.DISPOSITION_TERMINAL),
        contender(4, None, disposition=ss.DISPOSITION_NONFINITE),
        contender(5, None, disposition=ss.DISPOSITION_SIMULATOR_ERROR),
        contender(6, 0.1, family="b"),
    ]
    selected, reasons = ss.select_candidates(selection_spec(shortlist_size=5), pool)
    assert [case["attempt_index"] for case in selected] == [6]
    assert reasons == {
        index: f"not eligible: {pool[index]['disposition']}" for index in range(6)
    }


def test_manifest_selection_matches_the_declared_rule(finished_run: Path) -> None:
    records = candidates_of(finished_run)
    spec = make_spec(small_mapping())
    selected, reasons = ss.select_candidates(spec, records)
    manifest = manifest_of(finished_run)
    assert [case["attempt_index"] for case in manifest["selected_cases"]] == [
        case["attempt_index"] for case in selected
    ]
    assert [case["rank"] for case in manifest["selected_cases"]] == list(range(len(selected)))
    objectives = [case["objective"] for case in manifest["selected_cases"]]
    assert objectives == sorted(objectives, reverse=True)
    families = [case["causal_family_id"] for case in manifest["selected_cases"]]
    assert len(set(families)) == len(families)
    for item in manifest["rejected_contenders"]:
        assert item["not_selected_reason"] == reasons[item["attempt_index"]]


# ---------------------------------------------------------------------------
# Manifest fail-closed loading
# ---------------------------------------------------------------------------


def rehash(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["manifest_sha256"] = ss.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    path.write_bytes(ss.canonical_bytes(dict(manifest)))
    return path


def test_manifest_round_trips_through_the_strict_loader(finished_run: Path, tmp_path: Path) -> None:
    manifest = manifest_of(finished_run)
    assert set(manifest) == set(ss._MANIFEST_FIELDS)
    assert manifest["method"] == ss.METHOD_RANDOM and manifest["seed"] == "test-s1"
    assert manifest["spec"] == make_spec(small_mapping()).to_mapping()
    copy_path = write_manifest(tmp_path / "copy.json", rehash(copy.deepcopy(manifest)))
    assert ss.load_manifest(copy_path) == manifest


def test_manifest_with_altered_bytes_fails_closed(finished_run: Path, tmp_path: Path) -> None:
    raw = (finished_run / ss.MANIFEST_FILE_NAME).read_bytes()
    needle = b'"simulation_calls_used":10'
    assert needle in raw
    altered = tmp_path / "altered.json"
    altered.write_bytes(raw.replace(needle, b'"simulation_calls_used":11', 1))
    with pytest.raises(ss.StressSearchManifestError, match="manifest self-hash mismatch"):
        ss.load_manifest(altered)
    marker = b'"objective":0.'
    position = raw.index(marker) + len(marker)
    flipped = bytearray(raw)
    flipped[position] = ord("9") if flipped[position] != ord("9") else ord("8")
    altered.write_bytes(bytes(flipped))
    # Depending on the digit the edited number may also stop being the shortest
    # float representation; either way the load fails closed.
    with pytest.raises(
        ss.StressSearchManifestError, match="manifest self-hash mismatch|not canonical JSON"
    ):
        ss.load_manifest(altered)
    altered.write_bytes(raw[:-20])
    with pytest.raises(ss.StressSearchManifestError, match="not strict JSON"):
        ss.load_manifest(altered)
    with pytest.raises(ss.StressSearchManifestError, match="cannot read manifest"):
        ss.load_manifest(tmp_path / "absent.json")


def test_manifest_with_an_unknown_field_fails_closed(finished_run: Path, tmp_path: Path) -> None:
    manifest = manifest_of(finished_run)
    manifest["approved_for_training"] = True
    path = write_manifest(tmp_path / "unknown.json", rehash(manifest))
    with pytest.raises(
        ss.StressSearchManifestError, match="unknown manifest fields: approved_for_training"
    ):
        ss.load_manifest(path)
    manifest = manifest_of(finished_run)
    manifest["selected_cases"][0]["label"] = "validated"
    path = write_manifest(tmp_path / "unknown-case.json", rehash(manifest))
    with pytest.raises(ss.StressSearchManifestError, match="unknown selected case fields: label"):
        ss.load_manifest(path)


@pytest.mark.parametrize("name", ss._MANIFEST_FIELDS)
def test_manifest_with_a_missing_field_fails_closed(
    finished_run: Path, tmp_path: Path, name: str
) -> None:
    manifest = manifest_of(finished_run)
    del manifest[name]
    if name != "manifest_sha256":
        rehash(manifest)
    path = write_manifest(tmp_path / "missing.json", manifest)
    with pytest.raises(ss.StressSearchManifestError, match=f"missing manifest fields: {name}"):
        ss.load_manifest(path)


def test_manifest_with_a_missing_case_field_fails_closed(
    finished_run: Path, tmp_path: Path
) -> None:
    for name in ("confirmation", "parameters", "metrics", "files"):
        manifest = manifest_of(finished_run)
        del manifest["selected_cases"][0][name]
        path = write_manifest(tmp_path / f"missing-{name}.json", rehash(manifest))
        with pytest.raises(
            ss.StressSearchManifestError, match=f"missing selected case fields: {name}"
        ):
            ss.load_manifest(path)


@pytest.mark.parametrize("digest", ["0" * 64, "", None, 7], ids=["zeros", "empty", "null", "int"])
def test_manifest_with_a_bad_self_hash_fails_closed(
    finished_run: Path, tmp_path: Path, digest: Any
) -> None:
    manifest = manifest_of(finished_run)
    manifest["manifest_sha256"] = digest
    path = write_manifest(tmp_path / "bad-hash.json", manifest)
    with pytest.raises(ss.StressSearchManifestError, match="manifest self-hash mismatch"):
        ss.load_manifest(path)


def test_manifest_with_non_canonical_formatting_fails_closed(
    finished_run: Path, tmp_path: Path
) -> None:
    manifest = manifest_of(finished_run)
    raw = (finished_run / ss.MANIFEST_FILE_NAME).read_bytes()
    variants = {
        "indented": json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        "spaced": json.dumps(manifest, sort_keys=True).encode("utf-8"),
        "trailing-newline": raw + b"\n",
        "bom": b"\xef\xbb\xbf" + raw,
        "unsorted": json.dumps(
            dict(reversed(list(manifest.items()))), separators=(",", ":")
        ).encode("utf-8"),
    }
    for label, payload in variants.items():
        assert payload != raw
        path = tmp_path / f"{label}.json"
        path.write_bytes(payload)
        with pytest.raises(
            ss.StressSearchManifestError, match="not canonical JSON|not strict JSON"
        ):
            ss.load_manifest(path)
    duplicated = tmp_path / "duplicate.json"
    duplicated.write_bytes(raw[:-1] + b',"seed":"test-s2"}')
    with pytest.raises(ss.StressSearchManifestError, match="duplicate JSON key 'seed'"):
        ss.load_manifest(duplicated)


def test_manifest_with_an_altered_data_boundary_fails_closed(
    finished_run: Path, tmp_path: Path
) -> None:
    manifest = manifest_of(finished_run)
    manifest["data_boundary"] = "Validated held-out test set; safe for final evaluation."
    path = write_manifest(tmp_path / "boundary.json", rehash(manifest))
    with pytest.raises(ss.StressSearchManifestError, match="data boundary statement was altered"):
        ss.load_manifest(path)
    manifest = manifest_of(finished_run)
    manifest["data_boundary"] = ss.DATA_BOUNDARY.replace("not IID samples, ", "")
    path = write_manifest(tmp_path / "boundary-trimmed.json", rehash(manifest))
    with pytest.raises(ss.StressSearchManifestError, match="data boundary statement was altered"):
        ss.load_manifest(path)


def test_manifest_with_wrong_schema_method_or_case_lists_fails_closed(
    finished_run: Path, tmp_path: Path
) -> None:
    cases: tuple[tuple[str, Any, str], ...] = (
        ("schema_version", "aeolus_habitat_v2_stress_search_manifest_v0", "schema_version"),
        ("method", "bayesian", "unknown method"),
        ("selected_cases", {}, "case lists are malformed"),
        ("rejected_contenders", None, "case lists are malformed"),
        ("selected_cases", ["case"], "selected case must be an object"),
    )
    for name, value, message in cases:
        manifest = manifest_of(finished_run)
        manifest[name] = value
        path = write_manifest(tmp_path / "wrong.json", rehash(manifest))
        with pytest.raises(ss.StressSearchManifestError, match=message):
            ss.load_manifest(path)
    array = tmp_path / "array.json"
    array.write_bytes(b"[]")
    with pytest.raises(ss.StressSearchManifestError, match="manifest must be an object"):
        ss.load_manifest(array)


def reseal(run: Path, manifest: dict[str, Any], name: str = ss.MANIFEST_FILE_NAME) -> Path:
    """Write an edited manifest into a run directory with a fresh self-hash."""

    return write_manifest(run / name, rehash(manifest))


def forge_run(run: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    """Rewrite a run's audit records with a consistent hash chain and re-derive the manifest.

    This models the strongest tampering the audit binding cannot see: someone
    who rewrites the whole run directory consistently.  It is used only to
    reach the replay checks that sit behind :func:`ss.load_run_manifest`.
    """

    audit = load_audit(run)
    records = [copy.deepcopy(entry["record"]) for entry in audit.entries]
    mutate(records)
    previous = "0" * 64
    lines = []
    for sequence, (record, entry) in enumerate(zip(records, audit.entries, strict=True)):
        digest = ss.canonical_sha256(
            {"sequence": sequence, "previous_record_sha256": previous, "record": record}
        )
        lines.append(
            ss.canonical_bytes(
                {
                    "sequence": sequence,
                    "previous_record_sha256": previous,
                    "record": record,
                    "record_sha256": digest,
                    "timing": entry["timing"],
                }
            )
        )
        previous = digest
    write_audit(run, lines)
    original = manifest_of(run)
    forged = ss.manifest_core(make_spec(original["spec"]), load_audit(run))
    for case, old in zip(forged["selected_cases"], original["selected_cases"], strict=True):
        case["confirmation"] = old["confirmation"]
        case["files"] = old["files"]
    forged["budget_consumption"]["confirmation_calls_outside_search_budget"] = original[
        "budget_consumption"
    ]["confirmation_calls_outside_search_budget"]
    forged["manifest_sha256"] = None
    reseal(run, forged)
    # The forgery is self-consistent: the audit-bound loader accepts it.
    assert run_manifest_of(run)["selected_cases"]


def top_case_record(run: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    attempt = manifest_of(run)["selected_cases"][0]["attempt_index"]
    (record,) = [
        record
        for record in records
        if record["kind"] == "candidate" and record["attempt_index"] == attempt
    ]
    return record


@pytest.fixture(scope="module")
def faithful_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("faithful-template") / "run"
    run_fake(run, small_mapping(), evaluator=FakeEvaluator(faithful=True))
    return run


@pytest.fixture()
def faithful_run(faithful_template: Path, tmp_path: Path) -> Path:
    """A finished fabricated run whose fake outcomes carry fabricated trace identity."""

    return Path(shutil.copytree(faithful_template, tmp_path / "run"))


def replay_fake(run: Path, evaluator: FakeEvaluator, **options: Any) -> dict[str, Any]:
    return ss.replay_manifest(
        manifest_path=run / ss.MANIFEST_FILE_NAME,
        repo_root=REPO_ROOT,
        evaluator=evaluator,
        **options,
    )


def _edit_identity(manifest: dict[str, Any]) -> None:
    manifest["identity"]["source_sha256"] = "0" * 64


def _edit_spec_budget(manifest: dict[str, Any]) -> None:
    manifest["spec"]["budget"]["max_simulation_calls"] = 11


def _edit_spec_screening(manifest: dict[str, Any]) -> None:
    manifest["spec"]["screening"]["backend"] = "warp"


def _edit_spec_fixture(manifest: dict[str, Any]) -> None:
    manifest["spec"]["fixture"]["base_scenario_sha256"] = "0" * 64


def _edit_seed_undeclared(manifest: dict[str, Any]) -> None:
    manifest["seed"] = "issue93-s1"


def _edit_parameters_out_of_space(manifest: dict[str, Any]) -> None:
    manifest["selected_cases"][0]["parameters"][0]["onset_step"] = 99


def _edit_parameters_in_space(manifest: dict[str, Any]) -> None:
    first = manifest["selected_cases"][0]["parameters"][0]
    first["onset_step"] = 8 if first["onset_step"] != 8 else 9


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_edit_identity, "manifest section 'identity' does not match the audit log"),
        (_edit_spec_budget, "does not bind to its audit: audit header was produced under a different spec"),
        (_edit_spec_screening, "does not bind to its audit: accelerated screening is refused"),
        (_edit_spec_fixture, "does not bind to its audit: audit header was produced under a different spec"),
        (_edit_seed_undeclared, "manifest section 'seed' does not match the audit log"),
        (_edit_parameters_out_of_space, "manifest section 'selected_cases' does not match the audit log"),
        (_edit_parameters_in_space, "manifest section 'selected_cases' does not match the audit log"),
    ],
    ids=["identity", "spec_budget", "spec_screening", "spec_fixture", "seed", "out_of_space", "in_space"],
)
@pytest.mark.parametrize("name", [ss.MANIFEST_FILE_NAME, "edited-copy.json"])
def test_replay_fails_closed_on_an_edited_and_re_sealed_manifest(
    finished_run: Path, tmp_path: Path, edit: Callable[[dict[str, Any]], None], message: str, name: str
) -> None:
    # The audit binding runs before any episode, so no evaluator is ever called.
    fake = FakeEvaluator()
    manifest = manifest_of(finished_run)
    edit(manifest)
    path = reseal(finished_run, manifest, name)
    # On its own the edited document is a well-formed, correctly sealed manifest.
    assert ss.load_manifest(path)["manifest_sha256"] == manifest["manifest_sha256"]
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.load_run_manifest(path)
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.replay_manifest(
            manifest_path=path, repo_root=REPO_ROOT, evaluator=fake, output_dir=tmp_path / "replay"
        )
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    assert not (tmp_path / "replay").exists()


def test_replay_fails_closed_when_the_environment_has_drifted(
    finished_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEvaluator()
    real_identity = ss.collect_source_identity
    real_base = ss.load_base_scenario_data
    with monkeypatch.context() as patch:
        # A source tree whose simulated-path files differ from the run's.
        patch.setattr(
            ss,
            "collect_source_identity",
            lambda root: {**real_identity(root), "source_sha256": "0" * 64},
        )
        with pytest.raises(
            ss.StressSearchManifestError, match="manifest identity does not match this environment"
        ):
            replay_fake(finished_run, fake)
    with monkeypatch.context() as patch:
        # A base fixture that drifted from the digest the spec declares.
        patch.setattr(ss, "load_base_scenario_data", lambda root: (real_base(root)[0], "0" * 64))
        with pytest.raises(
            ss.StressSearchManifestError, match="manifest does not match this environment"
        ):
            replay_fake(finished_run, fake)
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    # Unpatched, the same run directory replays (to FAIL: the evaluator is fabricated).
    assert replay_fake(finished_run, fake)["status"] == ss.REPLAY_STATUS_FAIL


def test_replay_refuses_an_existing_output_directory_before_any_work(
    finished_run: Path, tmp_path: Path
) -> None:
    fake = FakeEvaluator()
    existing = tmp_path / "existing-replay"
    existing.mkdir()
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite existing output"):
        replay_fake(finished_run, fake, output_dir=existing)
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    assert list(existing.iterdir()) == []
    # The output check even precedes loading: a missing manifest is not reached.
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite existing output"):
        ss.replay_manifest(
            manifest_path=tmp_path / "absent.json",
            repo_root=REPO_ROOT,
            output_dir=existing,
            evaluator=fake,
        )


@pytest.mark.parametrize("name", ["scenario.json", "control-trace.json", "actuation.jsonl"])
def test_replay_raises_on_a_tampered_or_missing_stored_case_file(
    faithful_run: Path, tmp_path: Path, name: str
) -> None:
    fake = FakeEvaluator(faithful=True)
    manifest = run_manifest_of(faithful_run)
    case = manifest["selected_cases"][0]
    assert sorted(case["files"]) == ["actuation.jsonl", "control-trace.json", "scenario.json"]
    entry = case["files"][name]
    assert entry["relative_path"] == f"cases/{case['episode_id']}/{name}"
    stored = faithful_run / entry["relative_path"]
    original = stored.read_bytes()
    assert len(original) == entry["byte_length"]
    assert hashlib.sha256(original).hexdigest() == entry["sha256"]
    flipped = bytearray(original)
    flipped[len(flipped) // 2] ^= 0x01
    for tampered in (bytes(flipped), original + b" ", original[:-1], b""):
        stored.write_bytes(tampered)
        with pytest.raises(
            ss.StressSearchManifestError,
            match=f"stored {name} for case .* does not match its recorded digest",
        ):
            replay_fake(faithful_run, fake, output_dir=tmp_path / "replay")
    stored.unlink()
    with pytest.raises(ss.StressSearchManifestError, match=f"stored {name} for case .* is missing"):
        replay_fake(faithful_run, fake, output_dir=tmp_path / "replay")
    assert not (tmp_path / "replay").exists()
    # Restoring the exact bytes makes the run replayable again.
    stored.write_bytes(original)
    assert replay_fake(faithful_run, fake)["status"] == ss.REPLAY_STATUS_PASS


def test_replay_raises_when_a_re_sealed_file_entry_points_at_a_different_scenario(
    finished_run: Path,
) -> None:
    # ``files`` entries are produced by confirmation, not by the audit, so an
    # editor can re-seal them; the regenerated scenario still has to match.
    fake = FakeEvaluator()
    manifest = manifest_of(finished_run)
    entry = manifest["selected_cases"][0]["files"]["scenario.json"]
    stored = finished_run / entry["relative_path"]
    altered = stored.read_bytes().replace(b'"steps":32', b'"steps":31', 1)
    assert altered != stored.read_bytes()
    stored.write_bytes(altered)
    entry["sha256"] = hashlib.sha256(altered).hexdigest()
    entry["byte_length"] = len(altered)
    reseal(finished_run, manifest)
    assert run_manifest_of(finished_run)["manifest_sha256"] == manifest["manifest_sha256"]
    with pytest.raises(
        ss.StressSearchManifestError, match="stored scenario for case .* does not match its parameters"
    ):
        replay_fake(finished_run, fake)


def test_replay_checks_case_identity_even_behind_a_consistently_forged_audit(
    finished_run: Path,
) -> None:
    fake = FakeEvaluator()
    recorded_onset = manifest_of(finished_run)["selected_cases"][0]["parameters"][0]["onset_step"]

    def onset(value: int) -> Callable[[list[dict[str, Any]]], None]:
        def mutate(records: list[dict[str, Any]]) -> None:
            top_case_record(finished_run, records)["parameters"][0]["onset_step"] = value

        return mutate

    # Parameters outside the declared space ...
    forge_run(finished_run, onset(99))
    with pytest.raises(ss.StressSearchManifestError, match="outside the declared search space"):
        replay_fake(finished_run, fake)
    # ... and parameters inside it that are not the recorded episode.
    forge_run(finished_run, onset(9 if recorded_onset == 8 else 8))
    with pytest.raises(ss.StressSearchManifestError, match="identity does not match its parameters"):
        replay_fake(finished_run, fake)
    assert fake.candidate_calls <= 1


def test_fabricated_replay_cannot_report_pass_without_replay_verification(
    finished_run: Path,
) -> None:
    receipt = replay_fake(finished_run, FakeEvaluator())
    assert receipt["status"] == ss.REPLAY_STATUS_FAIL and receipt["reproduced_count"] == 0
    assert receipt["case_count"] == len(manifest_of(finished_run)["selected_cases"]) >= 1
    for case in receipt["cases"]:
        assert case["status"] == "MISMATCH"
        assert case["replay_verified"] is False
        assert case["trace_sha256_identical"] is False
        # The numbers themselves reproduce; the missing authoritative replay decides.
        assert case["metric_comparisons"]
        assert all(item["within_tolerance"] for item in case["metric_comparisons"])
        assert case["stored_scenario_bytes_identical"] is True
        assert case["stored_artifacts_identical"] is True
        assert case["objective_consistent_with_metrics"] is True


def test_replay_status_follows_the_evaluator_it_is_given(faithful_run: Path, tmp_path: Path) -> None:
    # Bookkeeping only: "PASS" below is one fabricated evaluator agreeing with
    # itself.  It shows how the status is derived, not that anything replays.
    manifest = run_manifest_of(faithful_run)
    cases = manifest["selected_cases"]
    assert cases and all(case["confirmation"]["status"] == "CONFIRMED" for case in cases)
    assert ss.summarize_run(manifest)["confirmed_cases"] == len(cases)

    same = FakeEvaluator(faithful=True)
    receipt = replay_fake(faithful_run, same, output_dir=tmp_path / "replay-pass")
    assert receipt["status"] == ss.REPLAY_STATUS_PASS == "PASS"
    assert receipt["case_count"] == receipt["reproduced_count"] == len(cases)
    assert (same.twin_calls, same.candidate_calls) == (1, len(cases))
    assert receipt["manifest_sha256"] == manifest["manifest_sha256"]
    assert [case["episode_id"] for case in receipt["cases"]] == [c["episode_id"] for c in cases]
    for case in receipt["cases"]:
        assert case["status"] == "REPRODUCED"
        assert case["stored_artifacts_identical"] is True
        assert case["objective_consistent_with_metrics"] is True
        assert case["trace_sha256_identical"] is True and case["trace_sha256_required"] is True
    assert receipt["receipt_sha256"] == ss.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in ("receipt_sha256", "runtime_environment")
        }
    )
    written = tmp_path / "replay-pass" / ss.REPLAY_RECEIPT_FILE_NAME
    assert written.read_bytes() == ss.canonical_bytes(receipt)
    assert [item.name for item in written.parent.iterdir()] == [ss.REPLAY_RECEIPT_FILE_NAME]

    # An evaluator whose truth differs from the recorded one: metrics move.
    receipt = replay_fake(faithful_run, FakeEvaluator(faithful=True, gain=1.5))
    assert receipt["status"] == ss.REPLAY_STATUS_FAIL and receipt["reproduced_count"] == 0
    for case in receipt["cases"]:
        assert case["status"] == "MISMATCH"
        assert not all(item["within_tolerance"] for item in case["metric_comparisons"])
        assert case["stored_artifacts_identical"] is True

    # Same numbers, different trace bytes: stored artifacts no longer regenerate.
    receipt = replay_fake(faithful_run, FakeEvaluator(faithful=True, trace_salt="other"))
    assert receipt["status"] == ss.REPLAY_STATUS_FAIL and receipt["reproduced_count"] == 0
    for case in receipt["cases"]:
        assert all(item["within_tolerance"] for item in case["metric_comparisons"])
        assert case["trace_sha256_identical"] is False
        assert case["stored_artifacts_identical"] is False

    # One case that crashes on replay is a mismatch, not an exception.
    receipt = replay_fake(faithful_run, FakeEvaluator(faithful=True, modes={1: "raise"}))
    assert receipt["status"] == ss.REPLAY_STATUS_FAIL
    assert receipt["reproduced_count"] == len(cases) - 1
    assert receipt["cases"][0]["status"] == "MISMATCH"
    assert receipt["cases"][0]["metric_comparisons"] == []
    assert receipt["cases"][0]["outcome"]["status"] == ss.OUTCOME_ERROR

    # A healthy twin that fails on replay ends the replay; it is never a PASS.
    with pytest.raises(ss.StressSearchError, match="healthy twin did not complete"):
        replay_fake(faithful_run, FakeEvaluator(faithful=True, twin_mode="nan_rows"))


def test_replay_requires_the_objective_to_equal_its_declared_metric(faithful_run: Path) -> None:
    # Reachable only behind a consistently forged audit: the recorded objective
    # is nudged away from the metric it is defined as.
    def nudge(records: list[dict[str, Any]]) -> None:
        top_case_record(faithful_run, records)["objective"] += 0.125

    forge_run(faithful_run, nudge)
    case = run_manifest_of(faithful_run)["selected_cases"][0]
    assert case["objective"] == case["metrics"]["peak_hazard_shift"] + 0.125
    receipt = replay_fake(faithful_run, FakeEvaluator(faithful=True))
    assert receipt["status"] == ss.REPLAY_STATUS_FAIL
    first, *others = receipt["cases"]
    assert first["status"] == "MISMATCH"
    assert first["objective_consistent_with_metrics"] is False
    assert all(item["within_tolerance"] for item in first["metric_comparisons"])
    assert all(other["status"] == "REPRODUCED" for other in others)


def _no_cases_shortlist() -> dict[str, Any]:
    data = small_mapping()
    data["selection"]["shortlist_size"] = 0
    return data


def _no_cases_minimum() -> dict[str, Any]:
    data = small_mapping()
    data["selection"]["minimum_objective_exclusive"] = 1.0e9
    return data


@pytest.mark.parametrize(
    ("mapping", "twin_mode", "end_reason"),
    [
        (_no_cases_shortlist, None, ss.END_SIMULATION_BUDGET),
        (_no_cases_minimum, None, ss.END_SIMULATION_BUDGET),
        (small_mapping, "raise", ss.END_BASELINE_FAILED),
    ],
    ids=["shortlist_zero", "nothing_above_minimum", "baseline_failed"],
)
def test_replay_of_a_manifest_without_cases_is_no_cases_never_pass(
    tmp_path: Path, mapping: Callable[[], dict[str, Any]], twin_mode: str | None, end_reason: str
) -> None:
    run = tmp_path / "run"
    summary, _ = run_fake(run, mapping(), evaluator=FakeEvaluator(twin_mode=twin_mode))
    assert (summary.end_reason, summary.final) == (end_reason, True)
    manifest = run_manifest_of(run)
    assert manifest["selected_cases"] == []
    assert manifest["budget_consumption"]["confirmation_calls_outside_search_budget"] == 0
    assert not (run / ss.CASES_DIRECTORY_NAME).exists()
    # Even an evaluator that claims verified replay has nothing to reproduce.
    fake = FakeEvaluator(faithful=True)
    receipt = replay_fake(run, fake, output_dir=tmp_path / "replay")
    assert receipt["status"] == ss.REPLAY_STATUS_NO_CASES == "NO_CASES"
    assert receipt["status"] != ss.REPLAY_STATUS_PASS
    assert (receipt["case_count"], receipt["reproduced_count"], receipt["cases"]) == (0, 0, [])
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    written = json.loads((tmp_path / "replay" / ss.REPLAY_RECEIPT_FILE_NAME).read_bytes())
    assert written["status"] == "NO_CASES"
    assert {ss.REPLAY_STATUS_PASS, ss.REPLAY_STATUS_FAIL, ss.REPLAY_STATUS_NO_CASES} == {
        "PASS",
        "FAIL",
        "NO_CASES",
    }


# ---------------------------------------------------------------------------
# Matched-budget comparison (fabricated manifests; bookkeeping only)
# ---------------------------------------------------------------------------


def comparison_manifests(
    root: Path,
    mapping: Mapping[str, Any],
    *,
    gain: float = 1.0,
    modes: Mapping[int, str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    manifests = {}
    for method in ss.METHODS:
        for seed in mapping["seeds"]:
            output = root / f"{method}-{seed}"
            run_fake(
                output,
                mapping,
                method=method,
                seed=seed,
                evaluator=FakeEvaluator(gain=gain, modes=modes),
            )
            # Loaded the way the CLI comparison loads them: bound to the audit.
            manifests[(method, seed)] = run_manifest_of(output)
    return manifests


@pytest.fixture(scope="module")
def complete_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, str], dict[str, Any]]:
    return comparison_manifests(tmp_path_factory.mktemp("comparison"), small_mapping())


@pytest.fixture()
def complete_manifests(
    complete_template: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return copy.deepcopy(complete_template)


def test_comparison_reports_zero_discoveries_as_a_negative_result(
    complete_manifests: dict[tuple[str, str], dict[str, Any]],
) -> None:
    comparison = ss.compare_runs(list(complete_manifests.values()))
    assert comparison["seeds"] == TEST_SEEDS
    assert comparison["budget"] == small_mapping()["budget"]
    # Every declared (method, seed) run is present, including all-negative ones.
    assert [(run["method"], run["seed"]) for run in comparison["runs"]] == [
        (method, seed) for method in ss.METHODS for seed in TEST_SEEDS
    ]
    tier_ids = [tier["tier_id"] for tier in small_mapping()["discovery_tiers"]]
    for run in comparison["runs"]:
        assert run["simulation_calls_used"] == 10
        assert run["end_reason"] == ss.END_SIMULATION_BUDGET
        assert run["evaluated_candidates"] > 0 and run["best_objective"] > 0.0
        assert run["confirmed_cases"] == 0
        assert [tier["tier_id"] for tier in run["tiers"]] == tier_ids
        for tier in run["tiers"]:
            assert tier["discovered_episodes"] == 0
            assert tier["discovered_causal_families"] == 0
            assert tier["simulation_calls_to_first_discovery"] is None
            assert tier["discovered_families_per_simulation_call"] == 0.0
    for method in ss.METHODS:
        summary = comparison["per_method"][method]
        assert summary["simulation_calls_used"] == {"count": 2, "mean": 10.0, "min": 10.0, "max": 10.0}
        for tier_id in tier_ids:
            tier = summary["tiers"][tier_id]
            assert tier["seeds_with_discovery"] == 0
            assert tier["discovered_causal_families"] == {
                "count": 2,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            assert tier["simulation_calls_to_first_discovery"]["count"] == 0
            assert tier["simulation_calls_to_first_discovery"]["mean"] is None
    assert [item["seed"] for item in comparison["paired_by_seed"]] == TEST_SEEDS
    for item in comparison["paired_by_seed"]:
        assert item["tier_family_difference_cross_entropy_minus_random"] == {
            tier_id: 0 for tier_id in tier_ids
        }
    paired = comparison["paired_best_objective_summary"]
    assert (
        paired["seeds_cross_entropy_higher"] + paired["seeds_random_higher"] + paired["seeds_tied"]
        == 2
    )
    assert comparison["comparison_sha256"] == ss.canonical_sha256(
        {key: value for key, value in comparison.items() if key != "comparison_sha256"}
    )
    # Input order does not change the report.
    assert ss.compare_runs(list(reversed(complete_manifests.values()))) == comparison


def test_comparison_text_makes_no_superiority_claim(
    complete_manifests: dict[tuple[str, str], dict[str, Any]],
) -> None:
    comparison = ss.compare_runs(list(complete_manifests.values()))
    interpretation = comparison["interpretation"]
    assert "No significance test was run" in interpretation
    assert "no claim is made that either method is better" in interpretation
    assert "not independent populations" in interpretation
    assert comparison["data_boundary"] == ss.DATA_BOUNDARY
    report = ss.render_comparison_markdown(comparison)
    assert interpretation in report and ss.DATA_BOUNDARY in report
    for seed in TEST_SEEDS:
        assert f"| {ss.METHOD_RANDOM} | {seed} |" in report
        assert f"| {ss.METHOD_CROSS_ENTROPY} | {seed} |" in report
    assert "0 of 2" in report and "0 (none)" in report
    claims = (
        "outperform",
        "superior",
        "beats",
        "wins",
        "significantly",
        "more effective",
        "is better than",
        "proves",
        "validated",
    )
    for text in (interpretation, report):
        lowered = text.lower().replace(interpretation.lower(), "")
        assert not any(claim in lowered for claim in claims)
        assert "better" not in lowered


def test_comparison_counts_discoveries_when_the_fabricated_signal_is_loud(
    tmp_path: Path,
) -> None:
    # Guards against a comparison that reports zero regardless of its input.
    # The "discoveries" here are produced by an inflated fabricated offset and
    # are not evidence about the plant or about either method.
    manifests = comparison_manifests(tmp_path / "loud", small_mapping(), gain=40.0)
    comparison = ss.compare_runs(list(manifests.values()))
    tier = "B_half_scale_shift"
    for method in ss.METHODS:
        assert comparison["per_method"][method]["tiers"][tier]["seeds_with_discovery"] == 2
    for run in comparison["runs"]:
        (hit,) = [item for item in run["tiers"] if item["tier_id"] == tier]
        assert hit["discovered_causal_families"] >= 1
        assert 2 <= hit["simulation_calls_to_first_discovery"] <= 10
        assert hit["discovered_families_per_simulation_call"] == (
            hit["discovered_causal_families"] / 10
        )
        assert hit["discovered_episodes"] >= hit["discovered_causal_families"]


def test_comparison_fails_closed_when_a_declared_seed_is_missing(
    complete_manifests: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for dropped in complete_manifests:
        partial = [m for key, m in complete_manifests.items() if key != dropped]
        with pytest.raises(ss.StressSearchManifestError, match="needs every declared seed") as info:
            ss.compare_runs(partial)
        assert dropped[0] in str(info.value) and dropped[1] in str(info.value)
    only_first_seed = [m for (_method, seed), m in complete_manifests.items() if seed == "test-s1"]
    with pytest.raises(ss.StressSearchManifestError, match="needs every declared seed"):
        ss.compare_runs(only_first_seed)
    with pytest.raises(ss.StressSearchManifestError, match="at least one manifest"):
        ss.compare_runs([])


@pytest.mark.parametrize("missing", ss.METHODS)
def test_comparison_fails_closed_when_a_method_is_missing(
    complete_manifests: dict[tuple[str, str], dict[str, Any]], missing: str
) -> None:
    partial = [m for (method, _seed), m in complete_manifests.items() if method != missing]
    assert len(partial) == 2
    with pytest.raises(ss.StressSearchManifestError, match="needs every declared seed") as info:
        ss.compare_runs(partial)
    assert missing in str(info.value)


def test_comparison_fails_closed_on_duplicate_runs(
    complete_manifests: dict[tuple[str, str], dict[str, Any]],
) -> None:
    manifests = list(complete_manifests.values())
    with pytest.raises(ss.StressSearchManifestError, match="duplicate run for method and seed"):
        ss.compare_runs([*manifests, manifests[0]])
    with pytest.raises(ss.StressSearchManifestError, match="duplicate run for method and seed"):
        ss.compare_runs([manifests[0], manifests[0], manifests[2], manifests[3]])


def test_comparison_fails_closed_on_mixed_identities(
    complete_manifests: dict[tuple[str, str], dict[str, Any]], tmp_path: Path
) -> None:
    other_budget = small_mapping()
    other_budget["budget"]["max_simulation_calls"] = 12
    run_fake(tmp_path / "other-budget", other_budget, method=ss.METHOD_RANDOM, seed="test-s1")
    renamed = small_mapping()
    renamed["spec_id"] = "stress-search-unit-small-v2"
    run_fake(tmp_path / "renamed", renamed, method=ss.METHOD_RANDOM, seed="test-s1")
    for foreign_dir in ("other-budget", "renamed"):
        foreign = manifest_of(tmp_path / foreign_dir)
        mixed = dict(complete_manifests)
        mixed[(ss.METHOD_RANDOM, "test-s1")] = foreign
        with pytest.raises(ss.StressSearchManifestError, match="do not share one spec"):
            ss.compare_runs(list(mixed.values()))
        with pytest.raises(ss.StressSearchManifestError, match="do not share one spec"):
            ss.compare_runs([*complete_manifests.values(), foreign])
    drifted = copy.deepcopy(complete_manifests[(ss.METHOD_CROSS_ENTROPY, "test-s2")])
    drifted["identity"]["source_sha256"] = "0" * 64
    mixed = dict(complete_manifests)
    mixed[(ss.METHOD_CROSS_ENTROPY, "test-s2")] = drifted
    with pytest.raises(ss.StressSearchManifestError, match="do not share one spec"):
        ss.compare_runs(list(mixed.values()))


# ---------------------------------------------------------------------------
# Write-once outputs
# ---------------------------------------------------------------------------


def test_write_once_accepts_identical_content_and_refuses_different_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "deeper" / "output.json"
    ss.write_once(path, b'{"a":1}')
    assert path.read_bytes() == b'{"a":1}'
    ss.write_once(path, b'{"a":1}')
    assert path.read_bytes() == b'{"a":1}'
    for different in (b'{"a":2}', b'{"a":1}\n', b"", b'{"a":1} '):
        with pytest.raises(
            ss.StressSearchOutputError,
            match="refusing to overwrite existing output output.json with different content",
        ):
            ss.write_once(path, different)
        assert path.read_bytes() == b'{"a":1}'
    assert [item.name for item in path.parent.iterdir()] == ["output.json"]


def test_finalisation_refuses_to_overwrite_a_different_manifest(finished_run: Path) -> None:
    environment, audit, _header = ss.load_run(finished_run, REPO_ROOT)
    original = (finished_run / ss.MANIFEST_FILE_NAME).read_bytes()
    # Re-finalising identical content is accepted ...
    assert (
        ss.finalize_run(
            environment=environment,
            output_dir=finished_run,
            audit=audit,
            confirm_evaluator=FakeEvaluator(),
        )
        == json.loads(original)["manifest_sha256"]
    )
    # ... a pre-existing different manifest is never replaced.
    (finished_run / ss.MANIFEST_FILE_NAME).write_bytes(original.replace(b"test-s1", b"test-s9"))
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite existing output"):
        ss.finalize_run(
            environment=environment,
            output_dir=finished_run,
            audit=audit,
            confirm_evaluator=FakeEvaluator(),
        )


def test_canonical_bytes_reject_non_finite_and_unsupported_values() -> None:
    assert ss.canonical_bytes({"b": 1, "a": [1.5, None, "x"]}) == b'{"a":[1.5,null,"x"],"b":1}'
    for value in (float("nan"), float("inf"), {"a": float("-inf")}, {1, 2}, b"bytes", object()):
        with pytest.raises(ss.StressSearchError, match="not canonical JSON"):
            ss.canonical_bytes(value)


# ---------------------------------------------------------------------------
# Write-once publication: atomic, never partial
# ---------------------------------------------------------------------------


def test_write_once_links_a_complete_temporary_sibling_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out" / "file.json"
    observed: list[tuple[str, bytes, bool]] = []
    real_link = os.link

    def spying_link(source: Any, target: Any) -> None:
        # At publication time the final name does not exist yet and the
        # temporary sibling already holds every byte.
        observed.append((Path(source).name, Path(source).read_bytes(), Path(target).exists()))
        real_link(source, target)

    monkeypatch.setattr(os, "link", spying_link)
    ss.write_once(path, b"complete payload")
    assert observed == [(f".file.json.{os.getpid()}.tmp", b"complete payload", False)]
    assert path.read_bytes() == b"complete payload"
    assert [item.name for item in path.parent.iterdir()] == ["file.json"]
    # A refused overwrite goes through the same path and also cleans up.
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite"):
        ss.write_once(path, b"other payload")
    assert observed[-1] == (f".file.json.{os.getpid()}.tmp", b"other payload", True)
    assert path.read_bytes() == b"complete payload"
    assert [item.name for item in path.parent.iterdir()] == ["file.json"]


@pytest.mark.parametrize("failing", ["fsync", "link"])
def test_write_once_never_leaves_a_partial_final_file_or_a_temporary_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    path = tmp_path / "file.json"

    def broken(*_arguments: Any) -> None:
        raise OSError("fabricated I/O failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, failing, broken)
        with pytest.raises(OSError, match="fabricated I/O failure"):
            ss.write_once(path, b"x" * 4096)
    # The failed publication left nothing behind, complete or partial.
    assert list(tmp_path.iterdir()) == []
    ss.write_once(path, b"first")
    with monkeypatch.context() as patch:
        patch.setattr(os, failing, broken)
        with pytest.raises(OSError, match="fabricated I/O failure"):
            ss.write_once(path, b"second")
    assert path.read_bytes() == b"first"
    assert [item.name for item in tmp_path.iterdir()] == ["file.json"]


# ---------------------------------------------------------------------------
# Single-writer run lock
# ---------------------------------------------------------------------------


def test_run_lock_refuses_a_second_holder_and_is_released_on_exit(tmp_path: Path) -> None:
    first = ss.RunLock(tmp_path)
    second = ss.RunLock(tmp_path)
    with first as held:
        assert held is first
        assert (tmp_path / ss.LOCK_FILE_NAME).is_file()
        for _ in range(2):
            with pytest.raises(
                ss.StressSearchOutputError, match="another process is writing this run directory"
            ):
                second.__enter__()
        with pytest.raises(ss.StressSearchOutputError, match="another process is writing"):
            with ss.RunLock(tmp_path):
                pytest.fail("a second holder entered the lock")
    # Released on a normal exit: another instance can now take it.
    with second:
        with pytest.raises(ss.StressSearchOutputError, match="another process is writing"):
            first.__enter__()
    # Released when the holder exits through an exception, too.
    with pytest.raises(RuntimeError, match="fabricated failure"):
        with first:
            raise RuntimeError("fabricated failure")
    with second:
        pass
    # Exiting a lock that was never taken (or twice) is harmless.
    ss.RunLock(tmp_path).__exit__(None, None, None)
    second.__exit__(None, None, None)
    with first:
        pass


def test_search_holds_the_run_lock_for_the_whole_session(tmp_path: Path) -> None:
    output = tmp_path / "run"
    attempts: list[str] = []

    class ProbingEvaluator(FakeEvaluator):
        def __call__(
            self, scenario: Scenario, episode_id: str, interrupt_check: Callable[[], None] | None
        ) -> ss.EpisodeOutcome:
            try:
                with ss.RunLock(output):
                    attempts.append("entered")
            except ss.StressSearchOutputError as error:
                attempts.append(str(error))
            return super().__call__(scenario, episode_id, interrupt_check)

    summary, fake = run_fake(output, small_mapping(), evaluator=ProbingEvaluator())
    assert summary.final is True
    # Baseline and every candidate episode saw the directory locked.
    assert len(attempts) == fake.twin_calls + fake.candidate_calls == 10
    assert set(attempts) == {
        "another process is writing this run directory; refusing a second writer"
    }
    # The lock is free again once run_search has returned.
    with ss.RunLock(output):
        pass


def test_resume_refuses_a_run_directory_another_writer_holds(cancelled_run: Path) -> None:
    before = (cancelled_run / ss.AUDIT_FILE_NAME).read_bytes()
    fake = FakeEvaluator()
    with ss.RunLock(cancelled_run):
        with pytest.raises(ss.StressSearchOutputError, match="another process is writing"):
            resume_fake(cancelled_run, evaluator=fake)
        assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
        assert (cancelled_run / ss.AUDIT_FILE_NAME).read_bytes() == before
        assert not (cancelled_run / ss.MANIFEST_FILE_NAME).exists()
    final, _ = resume_fake(cancelled_run, evaluator=fake)
    assert final.final is True
    # A failed resume releases the lock as well.
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        resume_fake(cancelled_run)
    with ss.RunLock(cancelled_run):
        pass


# ---------------------------------------------------------------------------
# Audit timing blocks (outside the hash chain, still validated)
# ---------------------------------------------------------------------------

ZERO_TIMING = {"session_elapsed_seconds": 0.0, "cumulative_wall_seconds": 0.0}


@pytest.fixture()
def timed_run(tmp_path: Path) -> Path:
    """A fabricated run cancelled after five seconds of fake-clock time."""

    clock = FakeClock(start=10.0)
    fake = FakeEvaluator(
        clock=clock, seconds_per_episode=1.0, cancel_at_candidate_call=4, cancel_phase="end"
    )
    summary, _ = run_fake(tmp_path / "timed", timed_mapping(), evaluator=fake, clock=clock)
    assert (summary.end_reason, summary.final) == (ss.END_CANCELLED, False)
    return tmp_path / "timed"


def timed_mapping() -> dict[str, Any]:
    data = single_fault_mapping()
    data["budget"]["max_wall_seconds"] = 8.0
    data["budget"]["max_simulation_calls"] = 50
    return data


def retime(run: Path, index: int, timing: Any) -> Path:
    """Replace one audit line's timing block, keeping the line canonical."""

    lines = audit_lines(run)
    entry = json.loads(lines[index])
    entry["timing"] = timing
    lines[index] = ss.canonical_bytes(entry)
    return write_audit(run, lines)


def test_every_audit_line_carries_a_complete_timing_block(timed_run: Path) -> None:
    audit = load_audit(timed_run)
    assert audit.entries[0]["record"]["kind"] == "run_header"
    assert audit.entries[0]["timing"] == ZERO_TIMING
    for entry in audit.entries:
        assert set(entry["timing"]) == {"session_elapsed_seconds", "cumulative_wall_seconds"}
        for value in entry["timing"].values():
            assert type(value) in (int, float) and value >= 0
    cumulative = [entry["timing"]["cumulative_wall_seconds"] for entry in audit.entries]
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == 5.0 == ss._prior_wall_seconds(audit)


_BAD_TIMINGS: tuple[tuple[str, Any, str], ...] = (
    ("emptied", {}, "malformed timing block"),
    ("null", None, "malformed timing block"),
    ("list", [0.0, 0.0], "malformed timing block"),
    ("number", 5.0, "malformed timing block"),
    ("missing_cumulative", {"session_elapsed_seconds": 5.0}, "malformed timing block"),
    ("missing_session", {"cumulative_wall_seconds": 5.0}, "malformed timing block"),
    ("extra_field", {**ZERO_TIMING, "reset": True}, "malformed timing block"),
    (
        "negative_cumulative",
        {"session_elapsed_seconds": 5.0, "cumulative_wall_seconds": -1.0},
        "timing cumulative_wall_seconds must be a finite number >= 0",
    ),
    (
        "negative_session",
        {"session_elapsed_seconds": -0.5, "cumulative_wall_seconds": 5.0},
        "timing session_elapsed_seconds must be a finite number >= 0",
    ),
    (
        "string_cumulative",
        {"session_elapsed_seconds": 5.0, "cumulative_wall_seconds": "5.0"},
        "timing cumulative_wall_seconds must be a finite number >= 0",
    ),
    (
        "null_cumulative",
        {"session_elapsed_seconds": 5.0, "cumulative_wall_seconds": None},
        "timing cumulative_wall_seconds must be a finite number >= 0",
    ),
    (
        "bool_session",
        {"session_elapsed_seconds": True, "cumulative_wall_seconds": 5.0},
        "timing session_elapsed_seconds must be a finite number >= 0",
    ),
    (
        "nested_cumulative",
        {"session_elapsed_seconds": 5.0, "cumulative_wall_seconds": [5.0]},
        "timing cumulative_wall_seconds must be a finite number >= 0",
    ),
    (
        "decreasing_cumulative",
        {"session_elapsed_seconds": 5.0, "cumulative_wall_seconds": 4.0},
        "audit wall time decreases at line",
    ),
    ("reset_to_zero", ZERO_TIMING, "audit wall time decreases at line"),
)


@pytest.mark.parametrize(
    ("timing", "message"),
    [case[1:] for case in _BAD_TIMINGS],
    ids=[case[0] for case in _BAD_TIMINGS],
)
def test_audit_load_rejects_tampered_timing_and_resume_keeps_the_wall_time(
    timed_run: Path, timing: Any, message: str
) -> None:
    intact = (timed_run / ss.AUDIT_FILE_NAME).read_bytes()
    last = len(audit_lines(timed_run)) - 1
    path = retime(timed_run, last, timing)
    with pytest.raises(ss.StressSearchAuditError, match=message):
        ss.AuditLog.load(path)
    # Resume fails closed instead of treating the consumed wall time as zero.
    tampered = path.read_bytes()
    fake = FakeEvaluator()
    with pytest.raises(ss.StressSearchAuditError, match=message):
        resume_fake(timed_run, evaluator=fake)
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    assert path.read_bytes() == tampered
    assert not (timed_run / ss.MANIFEST_FILE_NAME).exists()
    # With the recorded timing restored, the five consumed seconds still count:
    # one more goes on re-deriving the twin, so only two further episodes fit.
    path.write_bytes(intact)
    clock = FakeClock(start=500.0)
    resumed = FakeEvaluator(clock=clock, seconds_per_episode=1.0)
    summary, _ = resume_fake(timed_run, evaluator=resumed, clock=clock)
    assert (summary.end_reason, summary.final) == (ss.END_WALL_BUDGET, True)
    assert resumed.candidate_calls == 2
    assert ss._prior_wall_seconds(load_audit(timed_run)) == 8.0


def test_audit_load_rejects_bad_timing_on_any_line_and_non_finite_tokens(timed_run: Path) -> None:
    intact = audit_lines(timed_run)
    for index in range(len(intact)):
        path = retime(timed_run, index, {})
        with pytest.raises(
            ss.StressSearchAuditError, match=f"audit line {index + 1} has a malformed timing block"
        ):
            ss.AuditLog.load(path)
        write_audit(timed_run, intact)
    # A mid-log decrease is caught at the line where the value drops.
    middle = next(
        index
        for index, line in enumerate(intact)
        if json.loads(line)["timing"]["cumulative_wall_seconds"] >= 2.0
    )
    path = retime(
        timed_run, middle, {"session_elapsed_seconds": 0.0, "cumulative_wall_seconds": 0.5}
    )
    with pytest.raises(
        ss.StressSearchAuditError, match=f"audit wall time decreases at line {middle + 1}"
    ):
        ss.AuditLog.load(path)
    # Non-finite values cannot be written as strict canonical JSON at all.
    needle = b'"cumulative_wall_seconds":5.0'
    assert needle in intact[-1]
    for token, message in (
        (b"NaN", "non-finite JSON constant"),
        (b"Infinity", "non-finite JSON constant"),
        (b"-Infinity", "non-finite JSON constant"),
        (b"1e999", "not canonical JSON"),
    ):
        line = intact[-1].replace(needle, b'"cumulative_wall_seconds":' + token)
        with pytest.raises(ss.StressSearchError, match=message):
            ss.AuditLog.load(write_audit(timed_run, [*intact[:-1], line]))
    write_audit(timed_run, intact)
    assert len(load_audit(timed_run).entries) == len(intact)


def test_audit_append_refuses_a_malformed_timing_block(timed_run: Path) -> None:
    audit = load_audit(timed_run)
    before = (timed_run / ss.AUDIT_FILE_NAME).read_bytes()
    count = len(audit.entries)
    for timing in ({}, {"session_elapsed_seconds": 1.0}, {**ZERO_TIMING, "cumulative_wall_seconds": -1.0}):
        with pytest.raises(ss.StressSearchAuditError, match="timing"):
            audit.append({"kind": "fabricated"}, timing)
    assert len(audit.entries) == count
    assert (timed_run / ss.AUDIT_FILE_NAME).read_bytes() == before


# ---------------------------------------------------------------------------
# Evaluator results that cannot be recorded faithfully (FABRICATED evaluator)
# ---------------------------------------------------------------------------

_REJECTED = "evaluator outcome rejected: "


@pytest.mark.parametrize(
    ("mode", "disposition", "reason"),
    [
        ("nan_metrics", ss.DISPOSITION_NONFINITE, _REJECTED + "outcome is not canonical JSON"),
        ("inf_metrics", ss.DISPOSITION_NONFINITE, _REJECTED + "outcome is not canonical JSON"),
        (
            "unserialisable_metrics",
            ss.DISPOSITION_NONFINITE,
            _REJECTED + "outcome is not canonical JSON",
        ),
        (
            "bad_status",
            ss.DISPOSITION_NONFINITE,
            _REJECTED + "unknown outcome status 'FABRICATED_UNKNOWN_STATUS'",
        ),
        (
            "negative_steps",
            ss.DISPOSITION_NONFINITE,
            _REJECTED + "committed_steps is not a non-negative integer",
        ),
        (
            "bool_steps",
            ss.DISPOSITION_NONFINITE,
            _REJECTED + "committed_steps is not a non-negative integer",
        ),
        (
            "string_steps",
            ss.DISPOSITION_NONFINITE,
            _REJECTED + "committed_steps is not a non-negative integer",
        ),
        ("not_outcome", ss.DISPOSITION_SIMULATOR_ERROR, "evaluator returned no EpisodeOutcome"),
        ("none_outcome", ss.DISPOSITION_SIMULATOR_ERROR, "evaluator returned no EpisodeOutcome"),
    ],
)
def test_unrecordable_evaluator_outcome_is_recorded_and_charged_and_never_escapes(
    tmp_path: Path, mode: str, disposition: str, reason: str
) -> None:
    run = tmp_path / "run"
    fake = FakeEvaluator(modes={2: mode})
    summary, _ = run_fake(run, single_fault_mapping(), evaluator=fake)
    # The run still finishes on its budget ...
    assert (summary.end_reason, summary.final) == (ss.END_SIMULATION_BUDGET, True)
    assert summary.simulation_calls_used == 10 and fake.candidate_calls == 9
    # ... and the audit can be loaded again.
    records = candidates_of(run)
    assert [record["attempt_index"] for record in records] == list(range(9))
    bad = records[1]
    assert bad["disposition"] == disposition
    assert reason in bad["reason"]
    assert bad["outcome"]["status"] == disposition
    assert bad["outcome"]["committed_steps"] == 0
    assert bad["outcome"]["race_metrics"] is None
    assert bad["metrics"] is None and bad["objective"] is None
    # It consumed a simulation call like any other episode.
    assert bad["simulation_calls_used"] == records[0]["simulation_calls_used"] + 1 == 3
    assert [r["disposition"] for r in records].count(ss.DISPOSITION_EVALUATED) == 8
    manifest = run_manifest_of(run)
    assert manifest["budget_consumption"]["dispositions"][disposition] == 1
    assert 1 not in [case["attempt_index"] for case in manifest["selected_cases"]]
    (contender,) = [c for c in manifest["rejected_contenders"] if c["attempt_index"] == 1]
    assert contender["not_selected_reason"] == f"not eligible: {disposition}"
    for name in (ss.AUDIT_FILE_NAME, ss.MANIFEST_FILE_NAME):
        text = (run / name).read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        resume_fake(run)


def test_unrecordable_outcome_survives_a_cancel_and_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    fake = FakeEvaluator(modes={2: "nan_metrics"}, cancel_at_candidate_call=3, cancel_phase="end")
    first, _ = run_fake(run, single_fault_mapping(), evaluator=fake)
    assert (first.end_reason, first.simulation_calls_used) == (ss.END_CANCELLED, 4)
    final, resumed = resume_fake(run)
    assert final.final is True and final.simulation_calls_used == 10
    # The rejected episode was charged once and is not simulated again.
    assert resumed.candidate_calls == 6
    records = candidates_of(run)
    assert [r["disposition"] for r in records].count(ss.DISPOSITION_NONFINITE) == 1
    assert run_manifest_of(run)["budget_consumption"]["sessions"] == 2


# ---------------------------------------------------------------------------
# Interrupted and aborted sessions (FABRICATED evaluator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("twin_mode", ["keyboard_interrupt", "episode_interrupted"])
def test_interrupt_during_the_baseline_is_a_resumable_cancellation(
    tmp_path: Path, twin_mode: str
) -> None:
    run_fake(tmp_path / "straight", small_mapping())
    run = tmp_path / "run"
    fake = FakeEvaluator(twin_mode=twin_mode)
    summary, _ = run_fake(run, small_mapping(), evaluator=fake)
    assert (summary.end_reason, summary.final) == (ss.END_CANCELLED, False)
    assert summary.manifest_sha256 is None
    assert (summary.simulation_calls_used, summary.candidate_attempts_used) == (0, 0)
    assert (fake.twin_calls, fake.candidate_calls) == (1, 0)
    assert not (run / ss.MANIFEST_FILE_NAME).exists()
    audit = load_audit(run)
    # Nothing was recorded for the interrupted baseline.
    assert [entry["record"]["kind"] for entry in audit.entries] == [
        "run_header",
        "session_start",
        "session_end",
    ]
    assert audit.entries[-1]["record"] == {
        "kind": "session_end",
        "session_index": 0,
        "end_reason": ss.END_CANCELLED,
        "final": False,
        "simulation_calls_used": 0,
        "candidate_attempts_used": 0,
    }
    # The next session runs the baseline and the whole search.
    final, resumed = resume_fake(run)
    assert (final.end_reason, final.final) == (ss.END_SIMULATION_BUDGET, True)
    assert (resumed.twin_calls, resumed.candidate_calls) == (1, 9)
    assert final.simulation_calls_used == 10
    assert candidates_of(run) == candidates_of(tmp_path / "straight")
    audit = load_audit(run)
    straight_audit = load_audit(tmp_path / "straight")
    assert audit.records("baseline") == straight_audit.records("baseline")
    assert audit.records("run_footer") == straight_audit.records("run_footer")
    assert [record["end_reason"] for record in audit.records("session_end")] == [
        ss.END_CANCELLED,
        ss.END_SIMULATION_BUDGET,
    ]
    resumed_manifest = run_manifest_of(run)
    straight_manifest = run_manifest_of(tmp_path / "straight")
    assert resumed_manifest["budget_consumption"]["sessions"] == 2
    for name in ("selected_cases", "rejected_contenders", "causal_families", "identity", "spec"):
        assert resumed_manifest[name] == straight_manifest[name], name


def test_keyboard_interrupt_during_a_candidate_is_an_uncharged_incomplete_attempt(
    tmp_path: Path,
) -> None:
    run_fake(tmp_path / "straight", small_mapping())
    run = tmp_path / "run"
    fake = FakeEvaluator(modes={3: "keyboard_interrupt"})
    summary, _ = run_fake(run, small_mapping(), evaluator=fake)
    assert (summary.end_reason, summary.final) == (ss.END_CANCELLED, False)
    assert summary.simulation_calls_used == 3 and fake.candidate_calls == 3
    audit = load_audit(run)
    (incomplete,) = audit.records("incomplete")
    assert incomplete["cause"] == ss.INTERRUPT_CANCELLED
    assert incomplete["simulation_calls_used"] == 3
    assert incomplete["attempt_index"] == summary.candidate_attempts_used
    assert [entry["record"]["kind"] for entry in audit.entries][-2:] == [
        "incomplete",
        "session_end",
    ]
    final, _ = resume_fake(run)
    assert final.final is True and final.simulation_calls_used == 10
    assert candidates_of(run) == candidates_of(tmp_path / "straight")
    assert run_manifest_of(run)["budget_consumption"]["incomplete_attempts"] == 1


def test_an_exception_inside_a_session_leaves_an_aborted_session_end(tmp_path: Path) -> None:
    run_fake(tmp_path / "straight", small_mapping())
    run = tmp_path / "run"
    fake = FakeEvaluator()

    def exploding_cancel_check() -> bool:
        # Fires in the driver loop, between attempts, right after the third
        # candidate episode has been recorded (not inside an episode, where a
        # failure would be a recorded SIMULATOR_ERROR instead).
        if len(fake.completed) == 3 and fake.candidate_calls == 3:
            raise RuntimeError("fabricated driver failure")
        return False

    with pytest.raises(RuntimeError, match="fabricated driver failure"):
        run_fake(run, small_mapping(), evaluator=fake, cancel_requested=exploding_cancel_check)
    assert not (run / ss.MANIFEST_FILE_NAME).exists()
    audit = load_audit(run)
    recorded = audit.records("candidate")
    assert len(simulated(recorded)) == 3
    assert audit.records("run_footer") == []
    assert audit.entries[-1]["record"] == {
        "kind": "session_end",
        "session_index": 0,
        "end_reason": ss.END_ABORTED,
        "final": False,
        "error_type": "RuntimeError",
        "simulation_calls_used": 4,
        "candidate_attempts_used": len(recorded),
    }
    assert ss.END_ABORTED == "ABORTED_ERROR" and ss.END_ABORTED not in ss.FINAL_END_REASONS
    assert len(audit.records("session_start")) == len(audit.records("session_end")) == 1
    # The lock was released and the run is resumable to the same result.
    final, resumed = resume_fake(run)
    assert (final.end_reason, final.final) == (ss.END_SIMULATION_BUDGET, True)
    assert resumed.candidate_calls == 6
    assert candidates_of(run) == candidates_of(tmp_path / "straight")
    audit = load_audit(run)
    assert [record["end_reason"] for record in audit.records("session_end")] == [
        ss.END_ABORTED,
        ss.END_SIMULATION_BUDGET,
    ]
    assert run_manifest_of(run)["budget_consumption"]["sessions"] == 2


# ---------------------------------------------------------------------------
# Cross-entropy: an update without selection is skipped
# ---------------------------------------------------------------------------


def test_cross_entropy_skips_an_update_that_would_apply_no_selection(
    small_environment: ss.SearchEnvironment,
) -> None:
    settings = small_environment.spec.data["methods"]["cross_entropy"]
    population, elite = settings["population_size"], settings["elite_count"]
    assert (population, elite) == (4, 2)
    for scored in range(population + 1):
        sampler = ss.CandidateSampler(small_environment, ss.METHOD_CROSS_ENTROPY, "test-s1")
        untouched = ss.CandidateSampler(small_environment, ss.METHOD_CROSS_ENTROPY, "test-s1")
        initial = sampler.state()
        assert (initial["updates_applied"], initial["updates_skipped_no_selection"]) == (0, 0)
        for attempt in range(population):
            raw = sampler.propose(attempt)
            sampler.observe(raw, float(attempt) if attempt < scored else None)
        state = sampler.state()
        assert state["generation_index"] == 1 and state["pending_in_generation"] == 0
        later = [sampler.propose(attempt) for attempt in range(population, 6 * population)]
        fresh = [untouched.propose(attempt) for attempt in range(population, 6 * population)]
        if scored <= elite:
            # Keeping every scored candidate as "elite" would select nothing.
            assert (state["updates_applied"], state["updates_skipped_no_selection"]) == (0, 1)
            assert state["slots"] == initial["slots"]
            assert later == fresh
        else:
            assert (state["updates_applied"], state["updates_skipped_no_selection"]) == (1, 0)
            assert state["slots"] != initial["slots"]
            assert later != fresh


def test_cross_entropy_update_counters_accumulate_over_generations(
    small_environment: ss.SearchEnvironment,
) -> None:
    sampler = ss.CandidateSampler(small_environment, ss.METHOD_CROSS_ENTROPY, "test-s1")
    scored_per_generation = [2, 4, 0, 3, 1]
    snapshots = [sampler.state()["slots"]]
    for generation, scored in enumerate(scored_per_generation):
        for member in range(4):
            raw = sampler.propose(4 * generation + member)
            sampler.observe(raw, float(member) if member < scored else None)
        snapshots.append(sampler.state()["slots"])
    state = sampler.state()
    assert state["generation_index"] == 5
    assert (state["updates_applied"], state["updates_skipped_no_selection"]) == (2, 3)
    changed = [after != before for before, after in zip(snapshots, snapshots[1:])]
    assert changed == [False, True, False, True, False]
    # Random search never adapts and never counts an update either way.
    random_sampler = ss.CandidateSampler(small_environment, ss.METHOD_RANDOM, "test-s1")
    for attempt in range(12):
        random_sampler.observe(random_sampler.propose(attempt), 1.0)
    random_state = random_sampler.state()
    assert random_state["generation_index"] == 0
    assert (random_state["updates_applied"], random_state["updates_skipped_no_selection"]) == (0, 0)


def test_run_footer_sampler_state_is_reproduced_from_the_recorded_attempts(
    tmp_path: Path,
) -> None:
    mapping = mixed_mapping()
    run_fake(tmp_path / "run", mapping, method=ss.METHOD_CROSS_ENTROPY)
    audit = load_audit(tmp_path / "run")
    (footer,) = audit.records("run_footer")
    recorded = footer["sampler_state"]
    assert recorded["generation_index"] == 80 // 4
    assert (
        recorded["updates_applied"] + recorded["updates_skipped_no_selection"]
        == recorded["generation_index"]
    )
    sampler = ss.CandidateSampler(make_environment(mapping), ss.METHOD_CROSS_ENTROPY, "test-s1")
    scored_per_generation: list[int] = []
    for record in audit.records("candidate"):
        raw = sampler.propose(record["attempt_index"])
        assert ss.strip_slots(raw) == record["parameters"]
        if record["attempt_index"] % 4 == 0:
            scored_per_generation.append(0)
        scored_per_generation[-1] += record["objective"] is not None
        sampler.observe(raw, record["objective"])
    assert json.loads(ss.canonical_bytes(sampler.state())) == recorded
    assert recorded["updates_skipped_no_selection"] == sum(
        1 for scored in scored_per_generation if scored <= 2
    )


# ---------------------------------------------------------------------------
# Manifest bound to the audit beside it
# ---------------------------------------------------------------------------


def test_manifest_core_is_a_pure_function_of_the_spec_and_the_audit(
    finished_run: Path, cancelled_template: Path
) -> None:
    spec = make_spec(small_mapping())
    audit = load_audit(finished_run)
    core = ss.manifest_core(spec, audit)
    manifest = run_manifest_of(finished_run)
    assert manifest == manifest_of(finished_run)
    assert set(core) == set(ss._MANIFEST_FIELDS) - {"manifest_sha256"}
    assert core == ss.manifest_core(spec, load_audit(finished_run))
    for name, value in core.items():
        if name == "budget_consumption":
            assert manifest[name] == {
                **value,
                "confirmation_calls_outside_search_budget": 1 + len(core["selected_cases"]),
            }
        elif name == "selected_cases":
            assert [
                {key: case[key] for key in ss._CASE_CORE_FIELDS} for case in manifest[name]
            ] == value
            assert all(set(case) == set(ss._CASE_CORE_FIELDS) for case in value)
        else:
            assert manifest[name] == value, name
    assert core["gpu_screening"] == ss.GPU_SCREENING_STATUS
    assert core["split_rule"] == ss.SPLIT_RULE
    assert core["gpu_screening"] is not ss.GPU_SCREENING_STATUS
    with pytest.raises(ss.StressSearchAuditError, match="ends with exactly one run footer"):
        ss.manifest_core(spec, load_audit(cancelled_template))
    with pytest.raises(ss.StressSearchAuditError, match="produced under a different spec"):
        ss.manifest_core(make_spec(single_fault_mapping()), audit)
    with pytest.raises(ss.StressSearchAuditError, match="audit log has no run header"):
        ss.manifest_core(spec, ss.AuditLog(finished_run / ss.AUDIT_FILE_NAME, []))


def _case(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["selected_cases"][0]


def _scenario_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    return _case(manifest)["files"]["scenario.json"]


def _set_in(getter: Callable[[dict[str, Any]], dict[str, Any]], key: str, value: Any) -> Any:
    def edit(manifest: dict[str, Any]) -> None:
        getter(manifest)[key] = value

    return edit


def _top(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest


def _budget(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["budget_consumption"]


def _drop_first_case(manifest: dict[str, Any]) -> None:
    del manifest["selected_cases"][0]


def _raise_objective(manifest: dict[str, Any]) -> None:
    _case(manifest)["objective"] += 1.0


def _raise_metric(manifest: dict[str, Any]) -> None:
    _case(manifest)["metrics"]["peak_hazard_shift"] += 1.0


def _raise_objective_and_metric(manifest: dict[str, Any]) -> None:
    _raise_objective(manifest)
    _raise_metric(manifest)


def _remove_one_contender(manifest: dict[str, Any]) -> None:
    manifest["rejected_contenders"].pop()
    manifest["budget_consumption"]["candidate_attempts_used"] -= 1


def _count_one_more_evaluated(manifest: dict[str, Any]) -> None:
    manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_EVALUATED] += 1


def _move_scenario_path(manifest: dict[str, Any]) -> None:
    episode = _case(manifest)["episode_id"]
    _scenario_entry(manifest)["relative_path"] = f"cases/{episode}/../{episode}/scenario.json"


def _escape_scenario_path(manifest: dict[str, Any]) -> None:
    _scenario_entry(manifest)["relative_path"] = "../spec.json"


def _add_unexpected_file(manifest: dict[str, Any]) -> None:
    _case(manifest)["files"]["notes.txt"] = dict(_scenario_entry(manifest))


def _drop_scenario_file(manifest: dict[str, Any]) -> None:
    del _case(manifest)["files"]["scenario.json"]


def _extend_file_entry(manifest: dict[str, Any]) -> None:
    _scenario_entry(manifest)["verified"] = True


def _extend_confirmation(manifest: dict[str, Any]) -> None:
    _case(manifest)["confirmation"]["approved_by"] = "nobody"


def _extend_budget(manifest: dict[str, Any]) -> None:
    manifest["budget_consumption"]["extra_calls"] = 0


def _drop_confirmation_calls(manifest: dict[str, Any]) -> None:
    del manifest["budget_consumption"]["confirmation_calls_outside_search_budget"]


_AUDIT_MISMATCH = "manifest section '{}' does not match the audit log"
_RE_SEALED_EDITS: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
    ("dropped_selected_case", _drop_first_case, _AUDIT_MISMATCH.format("selected_cases")),
    ("dropped_all_cases", _set_in(_top, "selected_cases", []), _AUDIT_MISMATCH.format("selected_cases")),
    ("changed_objective", _raise_objective, _AUDIT_MISMATCH.format("selected_cases")),
    ("changed_metric", _raise_metric, _AUDIT_MISMATCH.format("selected_cases")),
    (
        "changed_objective_and_metric",
        _raise_objective_and_metric,
        _AUDIT_MISMATCH.format("selected_cases"),
    ),
    ("changed_rank", _set_in(_case, "rank", 7), _AUDIT_MISMATCH.format("selected_cases")),
    ("metrics_null", _set_in(_case, "metrics", None), _AUDIT_MISMATCH.format("selected_cases")),
    ("metrics_list", _set_in(_case, "metrics", [1.0]), _AUDIT_MISMATCH.format("selected_cases")),
    (
        "parameters_not_a_list",
        _set_in(_case, "parameters", {"0": "fan_speed_degradation"}),
        _AUDIT_MISMATCH.format("selected_cases"),
    ),
    ("parameters_null", _set_in(_case, "parameters", None), _AUDIT_MISMATCH.format("selected_cases")),
    (
        "search_outcome_null",
        _set_in(_case, "search_outcome", None),
        _AUDIT_MISMATCH.format("selected_cases"),
    ),
    ("changed_method", _set_in(_top, "method", ss.METHOD_CROSS_ENTROPY), _AUDIT_MISMATCH.format("method")),
    ("changed_seed", _set_in(_top, "seed", "test-s2"), _AUDIT_MISMATCH.format("seed")),
    ("seed_null", _set_in(_top, "seed", None), _AUDIT_MISMATCH.format("seed")),
    (
        "emptied_rejected_contenders",
        _set_in(_top, "rejected_contenders", []),
        _AUDIT_MISMATCH.format("rejected_contenders"),
    ),
    (
        "contenders_not_objects",
        _set_in(_top, "rejected_contenders", [1, "two", None]),
        _AUDIT_MISMATCH.format("rejected_contenders"),
    ),
    ("removed_one_contender", _remove_one_contender, "does not match the audit log"),
    (
        "budget_simulation_calls",
        _set_in(_budget, "simulation_calls_used", 9),
        _AUDIT_MISMATCH.format("budget_consumption"),
    ),
    (
        "budget_candidate_attempts",
        _set_in(_budget, "candidate_attempts_used", 1),
        _AUDIT_MISMATCH.format("budget_consumption"),
    ),
    (
        "budget_end_reason",
        _set_in(_budget, "end_reason", ss.END_ATTEMPT_BUDGET),
        _AUDIT_MISMATCH.format("budget_consumption"),
    ),
    ("budget_sessions", _set_in(_budget, "sessions", 7), _AUDIT_MISMATCH.format("budget_consumption")),
    (
        "budget_incomplete_attempts",
        _set_in(_budget, "incomplete_attempts", 3),
        _AUDIT_MISMATCH.format("budget_consumption"),
    ),
    ("budget_dispositions", _count_one_more_evaluated, _AUDIT_MISMATCH.format("budget_consumption")),
    (
        "budget_dispositions_null",
        _set_in(_budget, "dispositions", None),
        _AUDIT_MISMATCH.format("budget_consumption"),
    ),
    ("budget_null", _set_in(_top, "budget_consumption", None), "manifest budget_consumption must be an object"),
    ("budget_list", _set_in(_top, "budget_consumption", [10]), "manifest budget_consumption must be an object"),
    ("budget_extra_field", _extend_budget, "unknown manifest budget_consumption fields: extra_calls"),
    (
        "budget_without_confirmation_calls",
        _drop_confirmation_calls,
        "missing manifest budget_consumption fields: confirmation_calls_outside_search_budget",
    ),
    (
        "confirmation_calls_string",
        _set_in(_budget, "confirmation_calls_outside_search_budget", "2"),
        "confirmation call count must be an integer",
    ),
    (
        "confirmation_calls_bool",
        _set_in(_budget, "confirmation_calls_outside_search_budget", True),
        "confirmation call count must be an integer",
    ),
    ("causal_families_emptied", _set_in(_top, "causal_families", []), _AUDIT_MISMATCH.format("causal_families")),
    ("causal_families_null", _set_in(_top, "causal_families", None), _AUDIT_MISMATCH.format("causal_families")),
    ("identity_null", _set_in(_top, "identity", None), _AUDIT_MISMATCH.format("identity")),
    ("replay_command", _set_in(_top, "replay_command", ["true"]), _AUDIT_MISMATCH.format("replay_command")),
    ("spec_null", _set_in(_top, "spec", None), "does not bind to its audit: spec must be an object"),
    ("spec_list", _set_in(_top, "spec", []), "does not bind to its audit: spec must be an object"),
    ("audit_null", _set_in(_top, "audit", None), "manifest audit binding must be an object"),
    ("audit_string", _set_in(_top, "audit", ss.AUDIT_FILE_NAME), "manifest audit binding must be an object"),
    (
        "audit_other_path",
        _set_in(lambda manifest: manifest["audit"], "relative_path", "../other/audit.jsonl"),
        "manifest audit path is not the run's audit log",
    ),
    (
        "audit_footer_digest",
        _set_in(lambda manifest: manifest["audit"], "footer_record_sha256", "0" * 64),
        _AUDIT_MISMATCH.format("audit"),
    ),
    (
        "gpu_screening_enabled",
        _set_in(lambda manifest: manifest["gpu_screening"], "enabled", True),
        "manifest claims GPU screening; this harness is CPU only",
    ),
    (
        "gpu_screening_status",
        _set_in(lambda manifest: manifest["gpu_screening"], "status", "VALIDATED_ON_DGX"),
        "manifest claims GPU screening",
    ),
    ("gpu_screening_null", _set_in(_top, "gpu_screening", None), "manifest claims GPU screening"),
    ("split_rule", _set_in(_top, "split_rule", "random split"), "manifest split rule was altered"),
    ("files_parent_segment", _move_scenario_path, "case file 'scenario.json' path leaves its case directory"),
    ("files_escape", _escape_scenario_path, "case file 'scenario.json' path leaves its case directory"),
    ("files_unexpected_name", _add_unexpected_file, "unexpected case file 'notes.txt'"),
    ("files_without_scenario", _drop_scenario_file, "case files must include scenario.json"),
    ("files_list", _set_in(_case, "files", []), "case files must include scenario.json"),
    ("files_null", _set_in(_case, "files", None), "case files must include scenario.json"),
    ("file_entry_extra_field", _extend_file_entry, "unknown case file entry fields: verified"),
    (
        "file_entry_not_object",
        lambda manifest: _case(manifest)["files"].__setitem__("scenario.json", "scenario.json"),
        "case file entry must be an object",
    ),
    (
        "file_entry_bad_digest",
        _set_in(_scenario_entry, "sha256", "not-a-digest"),
        "case file 'scenario.json' digest entry is malformed",
    ),
    (
        "file_entry_bad_length",
        _set_in(_scenario_entry, "byte_length", "12"),
        "case file 'scenario.json' digest entry is malformed",
    ),
    ("confirmation_null", _set_in(_case, "confirmation", None), "case confirmation must be an object"),
    ("confirmation_extra_field", _extend_confirmation, "unknown case confirmation fields: approved_by"),
    (
        "confirmation_status",
        _set_in(lambda manifest: _case(manifest)["confirmation"], "status", "APPROVED"),
        "case confirmation status is unknown",
    ),
    ("episode_id_not_a_digest", _set_in(_case, "episode_id", "../../escape"), "case episode_id is not a sha256 digest"),
    ("episode_id_null", _set_in(_case, "episode_id", None), "case episode_id is not a sha256 digest"),
)


@pytest.mark.parametrize(
    ("edit", "message"),
    [case[1:] for case in _RE_SEALED_EDITS],
    ids=[case[0] for case in _RE_SEALED_EDITS],
)
def test_run_manifest_edited_and_re_sealed_fails_closed(
    finished_run: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    manifest = manifest_of(finished_run)
    assert len(manifest["selected_cases"]) >= 1 and len(manifest["rejected_contenders"]) >= 2
    edit(manifest)
    path = reseal(finished_run, manifest)
    # Only the manifest error type may escape, never a raw TypeError or KeyError.
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.load_run_manifest(path)
    fake = FakeEvaluator(faithful=True)
    with pytest.raises(ss.StressSearchManifestError, match=message):
        replay_fake(finished_run, fake)
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)


def test_run_manifest_needs_its_own_intact_audit_log(
    finished_run: Path, cancelled_template: Path, tmp_path: Path
) -> None:
    path = finished_run / ss.MANIFEST_FILE_NAME
    intact = audit_lines(finished_run)
    assert ss.load_run_manifest(path) == ss.load_manifest(path)
    bind = "manifest does not bind to its audit: "

    def refused(message: str) -> None:
        with pytest.raises(ss.StressSearchManifestError, match=message):
            ss.load_run_manifest(path)
        fake = FakeEvaluator(faithful=True)
        with pytest.raises(ss.StressSearchManifestError, match=message):
            replay_fake(finished_run, fake)
        assert (fake.twin_calls, fake.candidate_calls) == (0, 0)

    # Truncated: the footer, or the footer and more, is gone.
    for kept in (len(intact) - 1, len(intact) - 2, 3, 1):
        write_audit(finished_run, intact[:kept])
        refused(bind + "a manifest needs an audit that ends with exactly one run footer")
    (finished_run / ss.AUDIT_FILE_NAME).write_bytes(b"")
    refused(bind + "audit log has no run header")
    # Truncated inside the last record.
    write_audit(finished_run, [*intact[:-1], intact[-1][:40]], trailing=b"")
    refused(bind + "audit log ends with a partial record")
    # A record removed from the middle, or one flipped byte.
    write_audit(finished_run, [*intact[:4], *intact[5:]])
    refused(bind + "audit chain is broken at line 5")
    needle = b'"disposition":"EVALUATED"'
    index = next(i for i, line in enumerate(intact) if needle in line)
    flipped = intact[index].replace(needle, b'"disposition":"EVALUATEE"', 1)
    write_audit(finished_run, [*intact[:index], flipped, *intact[index + 1 :]])
    refused(bind + f"audit record digest mismatch at line {index + 1}")
    # A record appended after the footer.
    write_audit(finished_run, intact)
    audit = load_audit(finished_run)
    audit.append({"kind": "session_start", "session_index": 1, "resumed_after_sequence": 0}, ZERO_TIMING)
    refused(bind + "a manifest needs an audit that ends with exactly one run footer")
    # Another run's (valid, unfinished) audit.
    shutil.copyfile(cancelled_template / ss.AUDIT_FILE_NAME, finished_run / ss.AUDIT_FILE_NAME)
    refused(bind + "a manifest needs an audit that ends with exactly one run footer")
    # No audit at all, in place or after copying the manifest elsewhere.
    (finished_run / ss.AUDIT_FILE_NAME).unlink()
    refused(bind + "cannot read audit log")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copyfile(path, elsewhere / ss.MANIFEST_FILE_NAME)
    assert ss.load_manifest(elsewhere / ss.MANIFEST_FILE_NAME)["method"] == ss.METHOD_RANDOM
    with pytest.raises(ss.StressSearchManifestError, match=bind + "cannot read audit log"):
        ss.load_run_manifest(elsewhere / ss.MANIFEST_FILE_NAME)
    # Restored, the binding holds again.
    write_audit(finished_run, intact)
    assert ss.load_run_manifest(path)["seed"] == "test-s1"


def test_run_manifest_does_not_bind_to_another_runs_finished_audit(
    finished_run: Path, tmp_path: Path
) -> None:
    run_fake(tmp_path / "other-seed", small_mapping(), seed="test-s2")
    run_fake(tmp_path / "other-method", small_mapping(), method=ss.METHOD_CROSS_ENTROPY)
    path = finished_run / ss.MANIFEST_FILE_NAME
    for other, section in (("other-seed", "seed"), ("other-method", "method")):
        shutil.copyfile(tmp_path / other / ss.AUDIT_FILE_NAME, finished_run / ss.AUDIT_FILE_NAME)
        with pytest.raises(ss.StressSearchManifestError, match=_AUDIT_MISMATCH.format(section)):
            ss.load_run_manifest(path)
        # The other run's own manifest is fine beside its own audit.
        assert run_manifest_of(tmp_path / other)[section] != manifest_of(finished_run)[section]


def test_resume_of_a_finished_run_validates_its_manifest_first(finished_run: Path) -> None:
    audit_before = (finished_run / ss.AUDIT_FILE_NAME).read_bytes()
    original = (finished_run / ss.MANIFEST_FILE_NAME).read_bytes()
    manifest = manifest_of(finished_run)
    del manifest["selected_cases"][0]
    reseal(finished_run, manifest)
    fake = FakeEvaluator()
    with pytest.raises(
        ss.StressSearchManifestError, match=_AUDIT_MISMATCH.format("selected_cases")
    ):
        resume_fake(finished_run, evaluator=fake)
    (finished_run / ss.MANIFEST_FILE_NAME).write_bytes(original[:-1])
    with pytest.raises(ss.StressSearchManifestError, match="not strict JSON"):
        resume_fake(finished_run, evaluator=fake)
    assert (fake.twin_calls, fake.candidate_calls) == (0, 0)
    assert (finished_run / ss.AUDIT_FILE_NAME).read_bytes() == audit_before
    # The refusal released the lock and replaced nothing.
    assert (finished_run / ss.MANIFEST_FILE_NAME).read_bytes() == original[:-1]
    (finished_run / ss.MANIFEST_FILE_NAME).write_bytes(original)
    with pytest.raises(ss.StressSearchAuditError, match="run is already complete"):
        resume_fake(finished_run, evaluator=fake)


def test_environment_file_is_written_before_the_manifest_and_stays_unhashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    real_write_once = ss.write_once

    def recording_write_once(path: Path, data: bytes) -> None:
        written.append(path.name)
        real_write_once(path, data)

    monkeypatch.setattr(ss, "write_once", recording_write_once)
    runtime = {"python": "fabricated-runtime", "source_commit": None}
    run = tmp_path / "run"
    summary, _ = run_fake(run, small_mapping(), runtime_environment=runtime)
    assert written[0] == ss.SPEC_FILE_NAME
    assert written[-2:] == [ss.ENVIRONMENT_FILE_NAME, ss.MANIFEST_FILE_NAME]
    environment = json.loads((run / ss.ENVIRONMENT_FILE_NAME).read_bytes())
    assert set(environment) == {
        "runtime_environment",
        "source_identity",
        "wall_seconds_provisional",
        "timing_note",
    }
    assert "manifest_sha256" not in json.dumps(environment)
    assert environment["runtime_environment"] == runtime
    assert environment["source_identity"]["source_sha256"] == (
        run_manifest_of(run)["identity"]["source_sha256"]
    )
    # Unhashed provenance: the manifest is the same with or without it.
    plain, _ = run_fake(tmp_path / "plain", small_mapping())
    assert plain.manifest_sha256 == summary.manifest_sha256
    assert not (tmp_path / "plain" / ss.ENVIRONMENT_FILE_NAME).exists()
    # A crash after environment.json but before manifest.json is repaired by
    # re-running finalisation, which keeps the environment file it finds.
    environment_bytes = (run / ss.ENVIRONMENT_FILE_NAME).read_bytes()
    manifest_bytes = (run / ss.MANIFEST_FILE_NAME).read_bytes()
    (run / ss.MANIFEST_FILE_NAME).unlink()
    repaired, _ = resume_fake(run, runtime_environment={"python": "a-later-runtime"})
    assert repaired.manifest_sha256 == summary.manifest_sha256
    assert (run / ss.MANIFEST_FILE_NAME).read_bytes() == manifest_bytes
    assert (run / ss.ENVIRONMENT_FILE_NAME).read_bytes() == environment_bytes


# ---------------------------------------------------------------------------
# Comparison: deterministic endings only, paired at the common call count
# ---------------------------------------------------------------------------


def test_comparison_refuses_wall_time_limited_and_baseline_failed_runs(
    complete_manifests: dict[tuple[str, str], dict[str, Any]], tmp_path: Path
) -> None:
    assert ss.DETERMINISTIC_END_REASONS == (ss.END_SIMULATION_BUDGET, ss.END_ATTEMPT_BUDGET)
    clock = FakeClock()
    slow = FakeEvaluator(clock=clock, seconds_per_episode=3.0e5)
    wall, _ = run_fake(tmp_path / "wall", small_mapping(), evaluator=slow, clock=clock)
    assert (wall.end_reason, wall.final) == (ss.END_WALL_BUDGET, True)
    failed, _ = run_fake(
        tmp_path / "baseline",
        small_mapping(),
        method=ss.METHOD_CROSS_ENTROPY,
        seed="test-s2",
        evaluator=FakeEvaluator(twin_mode="raise"),
    )
    assert (failed.end_reason, failed.final) == (ss.END_BASELINE_FAILED, True)
    wall_manifest = run_manifest_of(tmp_path / "wall")
    failed_manifest = run_manifest_of(tmp_path / "baseline")
    # Same spec and identity as the complete set: only the ending differs.
    reference = complete_manifests[(ss.METHOD_RANDOM, "test-s1")]
    assert wall_manifest["identity"] == failed_manifest["identity"] == reference["identity"]
    refusal = "comparison refuses runs that did not end on a deterministic budget"
    for replacements in (
        {(ss.METHOD_RANDOM, "test-s1"): wall_manifest},
        {(ss.METHOD_CROSS_ENTROPY, "test-s2"): failed_manifest},
        {
            (ss.METHOD_RANDOM, "test-s1"): wall_manifest,
            (ss.METHOD_CROSS_ENTROPY, "test-s2"): failed_manifest,
        },
    ):
        mixed = {**complete_manifests, **replacements}
        with pytest.raises(ss.StressSearchManifestError, match=refusal) as info:
            ss.compare_runs(list(mixed.values()))
        for method, seed in replacements:
            assert f"('{method}', '{seed}')" in str(info.value)
        for key in set(complete_manifests) - set(replacements):
            assert str(key) not in str(info.value)
    # Leaving the refused run out is no way round it either.
    partial = [m for key, m in complete_manifests.items() if key != (ss.METHOD_RANDOM, "test-s1")]
    with pytest.raises(ss.StressSearchManifestError, match="needs every declared seed"):
        ss.compare_runs(partial)
    assert ss.compare_runs(list(complete_manifests.values()))["runs"]


def test_comparison_run_rows_and_pairs_report_cost_and_matching(
    complete_manifests: dict[tuple[str, str], dict[str, Any]],
) -> None:
    comparison = ss.compare_runs(list(complete_manifests.values()))
    assert comparison["all_seeds_used_equal_simulation_calls"] is True
    assert comparison["discovery_tiers"] == small_mapping()["discovery_tiers"]
    for run in comparison["runs"]:
        manifest = complete_manifests[(run["method"], run["seed"])]
        assert run == ss.summarize_run(manifest)
        assert run["manifest_sha256"] == manifest["manifest_sha256"]
        assert (run["sessions"], run["incomplete_attempts"]) == (1, 0)
        assert run["confirmation_calls_outside_search_budget"] == 1 + run["selected_cases"]
        assert run["terminal_failure_episodes"] == 0
        assert run["objective_metric"] == "peak_hazard_shift"
        assert run["candidate_attempts_used"] == sum(run["dispositions"].values())
    for method in ss.METHODS:
        assert comparison["per_method"][method]["terminal_failure_episodes"] == {
            "count": 2,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    by_key = {(run["method"], run["seed"]): run for run in comparison["runs"]}
    for item in comparison["paired_by_seed"]:
        assert item["simulation_calls_used"] == {method: 10 for method in ss.METHODS}
        assert item["simulation_calls_matched"] is True
        assert item["compared_at_simulation_calls"] == 10
        assert item["best_objective"] == {
            method: by_key[(method, item["seed"])]["best_objective"] for method in ss.METHODS
        }
        assert item["best_objective_cross_entropy_minus_random"] == (
            item["best_objective"][ss.METHOD_CROSS_ENTROPY]
            - item["best_objective"][ss.METHOD_RANDOM]
        )
    report = ss.render_comparison_markdown(comparison)
    assert "| Terminal |" in report and "| Compared at calls | Calls matched |" in report
    assert "| test-s1 | 10 | yes |" in report


def test_summarize_run_truncates_at_a_simulation_call_count(finished_run: Path) -> None:
    manifest = run_manifest_of(finished_run)
    records = candidates_of(finished_run)
    full = ss.summarize_run(manifest)
    assert full == ss.summarize_run(manifest, max_simulation_calls=None)
    assert full == ss.summarize_run(manifest, max_simulation_calls=10)
    assert full == ss.summarize_run(manifest, max_simulation_calls=10_000)
    assert full["simulation_calls_used"] == 10
    assert full["candidate_attempts_used"] == len(records)
    previous_best = None
    for limit in range(1, 10):
        partial = ss.summarize_run(manifest, max_simulation_calls=limit)
        kept = [record for record in records if record["simulation_calls_used"] <= limit]
        evaluated = [r for r in kept if r["disposition"] == ss.DISPOSITION_EVALUATED]
        assert partial["simulation_calls_used"] == limit
        assert partial["candidate_attempts_used"] == len(kept)
        assert partial["evaluated_candidates"] == len(evaluated) == limit - 1
        assert sum(partial["dispositions"].values()) == len(kept)
        assert partial["best_objective"] == max(
            (record["objective"] for record in evaluated), default=None
        )
        if previous_best is not None:
            assert partial["best_objective"] >= previous_best
        previous_best = partial["best_objective"]
        # Run-level provenance is not truncated.
        for name in ("end_reason", "sessions", "selected_cases", "manifest_sha256"):
            assert partial[name] == full[name]
    assert ss.summarize_run(manifest, max_simulation_calls=1)["best_objective"] is None
    assert previous_best is not None and previous_best <= full["best_objective"]


def test_comparison_counts_terminal_failure_episodes(tmp_path: Path) -> None:
    manifests = comparison_manifests(
        tmp_path / "terminal", single_fault_mapping(), modes={1: "terminal", 3: "terminal"}
    )
    comparison = ss.compare_runs(list(manifests.values()))
    for run in comparison["runs"]:
        assert run["terminal_failure_episodes"] == 2
        assert run["dispositions"][ss.DISPOSITION_TERMINAL] == 2
        assert run["evaluated_candidates"] == 7
        # A terminal failure is never a discovery and never a selected case.
        assert run["selected_cases"] <= run["evaluated_candidates"]
    for method in ss.METHODS:
        assert comparison["per_method"][method]["terminal_failure_episodes"] == {
            "count": 2,
            "mean": 2.0,
            "min": 2.0,
            "max": 2.0,
        }
    # Truncating before the second terminal episode counts only the first.
    manifest = manifests[(ss.METHOD_RANDOM, "test-s1")]
    assert ss.summarize_run(manifest, max_simulation_calls=3)["terminal_failure_episodes"] == 1
    report = ss.render_comparison_markdown(comparison)
    assert "| Terminal |" in report


def unmatched_mapping() -> dict[str, Any]:
    """A spec under which the two methods spend different simulation-call counts.

    The space is tiny and full of duplicates and rejections, so for the fixed
    test seeds one method reaches the simulation-call budget while the other
    runs out of candidate attempts first.  Chosen by inspection of this
    fabricated set-up; it carries no meaning about either method.
    """

    data = mixed_mapping()
    data["spec_id"] = "stress-search-unit-unmatched-v1"
    data["budget"]["max_simulation_calls"] = 20
    data["budget"]["max_candidate_attempts"] = 60
    return data


def test_comparison_pairs_runs_at_the_common_simulation_call_count(tmp_path: Path) -> None:
    mapping = unmatched_mapping()
    manifests = comparison_manifests(tmp_path / "unmatched", mapping)
    comparison = ss.compare_runs(list(manifests.values()))
    end_reasons = {run["end_reason"] for run in comparison["runs"]}
    assert end_reasons == {ss.END_SIMULATION_BUDGET, ss.END_ATTEMPT_BUDGET}
    assert comparison["all_seeds_used_equal_simulation_calls"] is False
    unmatched_seeds = 0
    for item in comparison["paired_by_seed"]:
        seed = item["seed"]
        calls = {
            method: manifests[(method, seed)]["budget_consumption"]["simulation_calls_used"]
            for method in ss.METHODS
        }
        common = min(calls.values())
        assert item["simulation_calls_used"] == calls
        assert item["compared_at_simulation_calls"] == common
        assert item["simulation_calls_matched"] is (len(set(calls.values())) == 1)
        unmatched_seeds += not item["simulation_calls_matched"]
        expected: dict[str, float | None] = {}
        for method in ss.METHODS:
            # Recomputed from the audit, independently of summarize_run.
            records = candidates_of(tmp_path / "unmatched" / f"{method}-{seed}")
            within = [
                record["objective"]
                for record in records
                if record["disposition"] == ss.DISPOSITION_EVALUATED
                and record["simulation_calls_used"] <= common
            ]
            expected[method] = max(within, default=None)
            truncated = ss.summarize_run(manifests[(method, seed)], max_simulation_calls=common)
            assert truncated["simulation_calls_used"] == common
            assert truncated["best_objective"] == expected[method]
            if calls[method] > common:
                full = ss.summarize_run(manifests[(method, seed)])
                assert truncated["candidate_attempts_used"] < full["candidate_attempts_used"]
                assert truncated["evaluated_candidates"] < full["evaluated_candidates"]
        assert item["best_objective"] == expected
        assert item["best_objective_cross_entropy_minus_random"] == (
            expected[ss.METHOD_CROSS_ENTROPY] - expected[ss.METHOD_RANDOM]
        )
    assert unmatched_seeds >= 1
    # The per-run rows still report each run's own full consumption.
    for run in comparison["runs"]:
        manifest = manifests[(run["method"], run["seed"])]
        assert run["simulation_calls_used"] == manifest["budget_consumption"]["simulation_calls_used"]
    report = ss.render_comparison_markdown(comparison)
    assert "| no |" in report
