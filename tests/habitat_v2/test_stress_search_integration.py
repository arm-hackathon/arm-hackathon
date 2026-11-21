"""Integration tests for the Issue #93 fault stress-search harness.

Every test in this module runs the REAL authoritative CPU plant under the real
deterministic HMC through the harness's default evaluator.  No test here
injects a fake, stubbed or scripted evaluator, and no score is fabricated: the
only injected callables are the ``clock`` (so no wall-clock value can steer a
run) and ``cancel_requested`` (a call counter, to interrupt a real episode).

One deliberate exception, confined to the "non-finite plant truth" section: two
tests wrap the harness's reference to ``project_true_targets`` so that the
truth projection refuses one chosen call with ``Issue55RaceError``.  The plant
and the HMC still run for real; the wrapper only triggers the harness's
failure path.  The resulting ``NONFINITE_OUTPUT`` records are test-induced and
are not a finding about the plant.

The spec is the checked-in ``scripts/stress_search_spec_v1.json`` narrowed to a
32-step horizon and a tiny budget, with its own ``spec_id`` and seeds, so the
five declared evidence seeds are never consumed.  All output goes to pytest
temporary directories.  These runs are development-only smoke evidence that the
pipeline is wired to the real plant; they say nothing about which search method
is better and are not a validation or test population.  GPU screening is not
exercised (CPU only).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
from types import ModuleType
from typing import Any
import uuid

import numpy as np
import pytest

from aeolus.habitat_v2 import stress_search as ss
from aeolus.habitat_v2.bdm_v1_families import load_base_scenario_data
from aeolus.habitat_v2.forecast_issue55_race import Issue55RaceError
from aeolus.habitat_v2.scenario import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "scripts" / "stress_search_spec_v1.json"
CLI_PATH = REPO_ROOT / "scripts" / "run_fault_stress_search.py"
MODULE_PATH = REPO_ROOT / "src" / "aeolus" / "habitat_v2" / "stress_search.py"

HORIZON_STEPS = 32
MAX_SIMULATION_CALLS = 8
MAX_CANDIDATE_ATTEMPTS = 40
SHORTLIST_SIZE = 2
SEED_ONE = "test-s1"
SEED_TWO = "test-s2"
DECLARED_EVIDENCE_SEEDS = {"issue93-s1", "issue93-s2", "issue93-s3", "issue93-s4", "issue93-s5"}
# One uninterrupted candidate episode consults ``cancel_requested`` once before
# the attempt and once per plant step.
CANCEL_CALLS_PER_EPISODE = HORIZON_STEPS + 1
CASE_FILE_NAMES = ("actuation.jsonl", "control-trace.json", "scenario.json")
ACTUATOR_GROUPS = ("condenser", "cooling", "dampers", "fan", "oxygen", "scrubber")


def _frozen_clock() -> float:
    return 0.0


def _small_spec_mapping() -> dict[str, Any]:
    mapping = copy.deepcopy(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
    mapping["spec_id"] = "test-stress-search-integration-small"
    mapping["fixture"]["horizon_steps"] = HORIZON_STEPS
    mapping["search_space"]["onset_step_range"] = [8, 12]
    mapping["search_space"]["duration_steps_range"] = [16, 18]
    mapping["search_space"]["latest_end_step"] = HORIZON_STEPS - 1
    mapping["methods"]["cross_entropy"]["population_size"] = 3
    mapping["methods"]["cross_entropy"]["elite_count"] = 1
    mapping["budget"]["max_simulation_calls"] = MAX_SIMULATION_CALLS
    mapping["budget"]["max_candidate_attempts"] = MAX_CANDIDATE_ATTEMPTS
    mapping["selection"]["shortlist_size"] = SHORTLIST_SIZE
    mapping["seeds"] = [SEED_ONE, SEED_TWO]
    return mapping


@dataclass(frozen=True)
class RealRun:
    directory: Path
    summary: ss.RunSummary
    manifest: dict[str, Any]


def _real_run(spec: ss.SearchSpec, directory: Path, method: str, seed: str) -> RealRun:
    summary = ss.run_search(
        spec=spec,
        repo_root=REPO_ROOT,
        output_dir=directory,
        method=method,
        seed=seed,
        clock=_frozen_clock,
    )
    return RealRun(directory, summary, ss.load_manifest(directory / ss.MANIFEST_FILE_NAME))


def _audit_entries(directory: Path) -> list[dict[str, Any]]:
    return ss.AuditLog.load(directory / ss.AUDIT_FILE_NAME).entries


def _hashed_chain(directory: Path) -> list[tuple[Any, ...]]:
    """Every audit line without its unhashed ``timing`` block."""

    return [
        (entry["sequence"], entry["previous_record_sha256"], entry["record"], entry["record_sha256"])
        for entry in _audit_entries(directory)
    ]


def _candidate_records(directory: Path) -> list[dict[str, Any]]:
    return ss.AuditLog.load(directory / ss.AUDIT_FILE_NAME).records("candidate")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_run(run: RealRun, destination: Path) -> Path:
    shutil.copytree(run.directory, destination)
    return destination / ss.MANIFEST_FILE_NAME


def _reseal(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Rewrite a tampered manifest with a correct self-hash and canonical bytes."""

    manifest["manifest_sha256"] = ss.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_bytes(ss.canonical_bytes(manifest))
    assert ss.load_manifest(manifest_path)["manifest_sha256"] == manifest["manifest_sha256"]


def _single_fault(fault_type: str, multiplier: float) -> list[dict[str, Any]]:
    return [
        {
            "type": fault_type,
            "target": None,
            "onset_step": 8,
            "duration_steps": 16,
            "ramp_shape": "sustained",
            "multiplier": multiplier,
        }
    ]


class _CancelAfterCalls:
    """Counter-based cancellation: true once consulted more than ``limit`` times."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.limit


# ---------------------------------------------------------------------------
# Shared real runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_spec() -> ss.SearchSpec:
    spec = ss.validate_spec_mapping(_small_spec_mapping())
    assert not DECLARED_EVIDENCE_SEEDS & set(spec.seeds)
    return spec


@pytest.fixture(scope="module")
def environment(small_spec: ss.SearchSpec) -> ss.SearchEnvironment:
    return ss.build_environment(small_spec, REPO_ROOT)


@pytest.fixture(scope="module")
def twin_outcome(environment: ss.SearchEnvironment) -> ss.EpisodeOutcome:
    outcome = ss.run_authoritative_episode(
        environment.twin_scenario, environment.contract, "0" * 64
    )
    assert outcome.status == ss.OUTCOME_COMPLETED
    assert outcome.rows is not None and outcome.rows.shape[0] == HORIZON_STEPS
    return outcome


@pytest.fixture(scope="module")
def run_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("stress-search-real-runs")


@pytest.fixture(scope="module")
def reference_runs(small_spec: ss.SearchSpec, run_root: Path) -> dict[str, RealRun]:
    """One uninterrupted real run per method on seed ``test-s1``."""

    return {
        method: _real_run(small_spec, run_root / f"reference-{method}", method, SEED_ONE)
        for method in ss.METHODS
    }


@pytest.fixture(scope="module")
def repeat_runs(small_spec: ss.SearchSpec, run_root: Path) -> dict[str, RealRun]:
    """The same (spec, method, seed) again, in different directories."""

    return {
        method: _real_run(small_spec, run_root / f"repeat-{method}", method, SEED_ONE)
        for method in ss.METHODS
    }


@pytest.fixture(scope="module")
def other_seed_run(small_spec: ss.SearchSpec, run_root: Path) -> RealRun:
    return _real_run(small_spec, run_root / "other-seed-random", ss.METHOD_RANDOM, SEED_TWO)


@pytest.fixture(scope="module")
def other_seed_cross_entropy_run(small_spec: ss.SearchSpec, run_root: Path) -> RealRun:
    return _real_run(
        small_spec, run_root / "other-seed-cross-entropy", ss.METHOD_CROSS_ENTROPY, SEED_TWO
    )


@pytest.fixture(scope="module")
def cli_module() -> ModuleType:
    module_spec = importlib.util.spec_from_file_location(
        "run_fault_stress_search_under_test", CLI_PATH
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture responsiveness: the trap this harness exists to avoid
# ---------------------------------------------------------------------------


def test_base_fixture_physical_fault_is_a_noop_under_hmc_hold(
    environment: ss.SearchEnvironment,
) -> None:
    base_data, base_sha256 = load_base_scenario_data(REPO_ROOT)
    assert base_sha256 == environment.spec.data["fixture"]["base_scenario_sha256"]
    faulted_data = copy.deepcopy(dict(base_data))
    faulted_data["fault_profiles"] = ss.fault_profiles_for(
        _single_fault("scrubber_effectiveness_degradation", 0.5)
    )
    healthy = ss.run_authoritative_episode(
        Scenario.from_mapping(copy.deepcopy(dict(base_data))), environment.contract, "1" * 64
    )
    faulted = ss.run_authoritative_episode(
        Scenario.from_mapping(faulted_data), environment.contract, "1" * 64
    )
    assert healthy.status == faulted.status == ss.OUTCOME_COMPLETED
    assert healthy.rows is not None and faulted.rows is not None
    assert healthy.committed_steps == faulted.committed_steps == int(base_data["steps"])
    # Under HMC hold the unmodified base fixture never runs the equipment, so a
    # physical fault cannot move privileged truth at all.
    assert float(np.max(np.abs(faulted.rows - healthy.rows))) == 0.0


@pytest.mark.parametrize(
    ("fault_type", "multiplier"),
    [("scrubber_effectiveness_degradation", 0.5), ("fan_speed_degradation", 0.8)],
)
def test_nominal_hold_fixture_fault_moves_privileged_truth(
    environment: ss.SearchEnvironment,
    twin_outcome: ss.EpisodeOutcome,
    fault_type: str,
    multiplier: float,
) -> None:
    canonical = ss.validate_candidate_parameters(environment, _single_fault(fault_type, multiplier))
    identity = ss.episode_identity(environment, canonical)
    scenario = ss.build_candidate_scenario(environment, canonical, identity["episode_id"])
    outcome = ss.run_authoritative_episode(scenario, environment.contract, identity["episode_id"])
    assert outcome.status == ss.OUTCOME_COMPLETED
    assert outcome.committed_steps == HORIZON_STEPS
    assert outcome.rows is not None and twin_outcome.rows is not None
    assert outcome.rows.shape == twin_outcome.rows.shape
    assert float(np.max(np.abs(outcome.rows - twin_outcome.rows))) > 0.0
    # Before the onset step the faulted episode is the healthy twin.
    assert np.array_equal(outcome.rows[:7], twin_outcome.rows[:7])
    metrics = ss.score_against_twin(environment, outcome.rows, twin_outcome.rows)
    assert metrics["peak_hazard_shift"] > 0.0
    assert metrics["peak_hazard_shift_step"] > 8


def test_healthy_twin_scores_zero_against_itself(
    environment: ss.SearchEnvironment, twin_outcome: ss.EpisodeOutcome
) -> None:
    metrics = ss.score_against_twin(environment, twin_outcome.rows, twin_outcome.rows)
    assert metrics["peak_hazard_shift"] == 0.0
    assert metrics["hazard_exposure_delta"] == 0.0
    assert metrics["residual_shift_at_end"] == 0.0


# ---------------------------------------------------------------------------
# Both methods on the real plant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ss.METHODS)
def test_method_completes_real_run_within_budget(
    reference_runs: dict[str, RealRun], small_spec: ss.SearchSpec, method: str
) -> None:
    run = reference_runs[method]
    summary = run.summary
    assert summary.final is True
    assert summary.method == method and summary.seed == SEED_ONE
    assert summary.end_reason == ss.END_SIMULATION_BUDGET
    assert summary.simulation_calls_used == MAX_SIMULATION_CALLS
    assert 1 <= summary.candidate_attempts_used <= MAX_CANDIDATE_ATTEMPTS
    assert (run.directory / ss.MANIFEST_FILE_NAME).is_file()
    assert (run.directory / ss.LOCK_FILE_NAME).is_file()
    assert summary.manifest_sha256 == run.manifest["manifest_sha256"]
    # The manifest binds to the hash-chained audit beside it.
    assert ss.load_run_manifest(run.directory / ss.MANIFEST_FILE_NAME) == run.manifest
    assert run.manifest["gpu_screening"] == ss.GPU_SCREENING_STATUS
    assert run.manifest["split_rule"] == ss.SPLIT_RULE
    assert (run.directory / ss.SPEC_FILE_NAME).read_bytes() == ss.canonical_bytes(
        small_spec.to_mapping()
    )

    consumption = run.manifest["budget_consumption"]
    assert consumption["end_reason"] == summary.end_reason
    assert consumption["simulation_calls_used"] == summary.simulation_calls_used
    assert consumption["candidate_attempts_used"] == summary.candidate_attempts_used
    assert consumption["sessions"] == 1
    assert consumption["incomplete_attempts"] == 0
    assert sum(consumption["dispositions"].values()) == summary.candidate_attempts_used
    assert run.manifest["method"] == method and run.manifest["seed"] == SEED_ONE
    assert run.manifest["data_boundary"] == ss.DATA_BOUNDARY
    assert run.manifest["spec"] == small_spec.to_mapping()


@pytest.mark.parametrize("method", ss.METHODS)
def test_audit_records_every_attempt_from_the_real_simulator(
    reference_runs: dict[str, RealRun], method: str
) -> None:
    run = reference_runs[method]
    entries = _audit_entries(run.directory)
    kinds = [entry["record"]["kind"] for entry in entries]
    assert kinds[:3] == ["run_header", "session_start", "baseline"]
    assert kinds[-2:] == ["session_end", "run_footer"]
    assert set(kinds[3:-2]) == {"candidate"}
    # Every line, the header included, carries a complete timing block; under
    # the frozen test clock no wall time is consumed.
    for entry in entries:
        assert entry["timing"] == {
            "session_elapsed_seconds": 0.0,
            "cumulative_wall_seconds": 0.0,
        }
    sampler_state = entries[-1]["record"]["sampler_state"]
    expected_generations = (
        run.summary.candidate_attempts_used
        // run.manifest["spec"]["methods"]["cross_entropy"]["population_size"]
        if method == ss.METHOD_CROSS_ENTROPY
        else 0
    )
    assert sampler_state["generation_index"] == expected_generations
    assert (
        sampler_state["updates_applied"] + sampler_state["updates_skipped_no_selection"]
        == expected_generations
    )

    baseline = entries[2]["record"]
    assert baseline["outcome"]["status"] == ss.OUTCOME_COMPLETED
    assert baseline["outcome"]["committed_steps"] == HORIZON_STEPS
    assert baseline["simulation_calls_used"] == 1

    candidates = _candidate_records(run.directory)
    assert len(candidates) == run.summary.candidate_attempts_used
    assert [record["attempt_index"] for record in candidates] == list(range(len(candidates)))
    simulated = 1
    evaluated_objectives = []
    for record in candidates:
        assert record["disposition"] in ss.DISPOSITIONS
        if record["disposition"] == ss.DISPOSITION_EVALUATED:
            simulated += 1
            outcome = record["outcome"]
            assert outcome["status"] == ss.OUTCOME_COMPLETED
            assert outcome["committed_steps"] == HORIZON_STEPS
            assert len(outcome["trace_sha256"]) == 64
            assert len(outcome["final_state_sha256"]) == 64
            assert outcome["race_metrics"] is not None
            assert record["objective"] == record["metrics"]["peak_hazard_shift"]
            evaluated_objectives.append(record["objective"])
        else:
            # Rejections and failures are recorded with a reason, never a score.
            assert record["reason"]
            assert record["metrics"] is None
            if record["disposition"] in (
                ss.DISPOSITION_TERMINAL,
                ss.DISPOSITION_NONFINITE,
                ss.DISPOSITION_SIMULATOR_ERROR,
            ):
                simulated += 1
        assert record["simulation_calls_used"] == simulated
        assert record["simulation_calls_used"] <= MAX_SIMULATION_CALLS
    assert simulated == run.summary.simulation_calls_used
    # The real plant responds on this fixture: the search found movement.
    assert evaluated_objectives and max(evaluated_objectives) > 0.0
    simulated_ids = [
        record["episode_id"]
        for record in candidates
        if record["disposition"] == ss.DISPOSITION_EVALUATED
    ]
    assert len(set(simulated_ids)) == len(simulated_ids)

    selected_attempts = {case["attempt_index"] for case in run.manifest["selected_cases"]}
    rejected_attempts = {item["attempt_index"] for item in run.manifest["rejected_contenders"]}
    assert not selected_attempts & rejected_attempts
    assert selected_attempts | rejected_attempts == set(range(len(candidates)))
    for contender in run.manifest["rejected_contenders"]:
        assert contender["not_selected_reason"]


@pytest.mark.parametrize("method", ss.METHODS)
def test_selected_cases_are_confirmed_with_hash_bound_artifacts(
    reference_runs: dict[str, RealRun], method: str
) -> None:
    run = reference_runs[method]
    cases = run.manifest["selected_cases"]
    assert 1 <= len(cases) <= SHORTLIST_SIZE
    assert [case["rank"] for case in cases] == list(range(len(cases)))
    assert [case["objective"] for case in cases] == sorted(
        (case["objective"] for case in cases), reverse=True
    )
    assert len({case["causal_family_id"] for case in cases}) == len(cases)
    assert run.manifest["budget_consumption"]["confirmation_calls_outside_search_budget"] == (
        len(cases) + 1
    )
    for case in cases:
        assert case["objective"] > 0.0
        confirmation = case["confirmation"]
        assert confirmation["status"] == "CONFIRMED"
        assert confirmation["replay_verified"] is True
        assert confirmation["trace_sha256_identical"] is True
        assert confirmation["metric_comparisons"]
        assert all(item["within_tolerance"] for item in confirmation["metric_comparisons"])

        assert sorted(case["files"]) == sorted(CASE_FILE_NAMES)
        payloads = {}
        for name in CASE_FILE_NAMES:
            entry = case["files"][name]
            assert entry["relative_path"] == f"cases/{case['episode_id']}/{name}"
            payload = (run.directory / entry["relative_path"]).read_bytes()
            assert _sha256(payload) == entry["sha256"]
            assert len(payload) == entry["byte_length"]
            payloads[name] = payload
        stored_scenario = Scenario.from_mapping(json.loads(payloads["scenario.json"]))
        assert stored_scenario.canonical_bytes == payloads["scenario.json"]
        assert stored_scenario.scenario_sha256 == case["scenario_sha256"]
        assert stored_scenario.data["steps"] == HORIZON_STEPS
        assert _sha256(payloads["control-trace.json"]) == case["search_outcome"]["trace_sha256"]
        assert _sha256(payloads["control-trace.json"]) == confirmation["outcome"]["trace_sha256"]

        rows = [json.loads(line) for line in payloads["actuation.jsonl"].splitlines()]
        assert [row["application_step"] for row in rows] == list(range(HORIZON_STEPS))
        assert ss.canonical_sha256(rows) == confirmation["outcome"]["actuation_sha256"]
        for row in rows:
            # No adviser exists in this harness: the HMC always receives None.
            assert "adviser_requested_command" in row
            assert row["adviser_requested_command"] is None
            assert type(row["final_command"]) is dict and row["final_command"]
            assert len(row["final_command_sha256"]) == 64
            assert type(row["active_faults"]) is list
            assert sorted(row["plant_actuators"]) == sorted(ACTUATOR_GROUPS)
            for group in ACTUATOR_GROUPS:
                fields = row["plant_actuators"][group]
                for stage in ("requested", "achieved", "effective"):
                    assert any(name.startswith(f"{stage}_") for name in fields), (group, stage)
        assert rows[0]["active_faults"] == []
        active_types = {fault["fault_type"] for row in rows for fault in row["active_faults"]}
        assert active_types == {fault["type"] for fault in case["parameters"]}


def test_selected_case_metrics_match_a_direct_authoritative_episode(
    reference_runs: dict[str, RealRun],
    environment: ss.SearchEnvironment,
    twin_outcome: ss.EpisodeOutcome,
) -> None:
    run = reference_runs[ss.METHOD_RANDOM]
    case = run.manifest["selected_cases"][0]
    canonical = ss.validate_candidate_parameters(
        environment, [dict(fault) for fault in case["parameters"]]
    )
    identity = ss.episode_identity(environment, canonical)
    assert identity["episode_id"] == case["episode_id"]
    assert identity["causal_family_id"] == case["causal_family_id"]
    scenario = ss.build_candidate_scenario(environment, canonical, identity["episode_id"])
    stored = run.directory / case["files"]["scenario.json"]["relative_path"]
    assert stored.read_bytes() == scenario.canonical_bytes
    outcome = ss.run_authoritative_episode(
        scenario, environment.contract, identity["episode_id"], verify_replay=True
    )
    assert outcome.status == ss.OUTCOME_COMPLETED
    assert outcome.replay_verified is True
    assert outcome.trace_sha256 == case["search_outcome"]["trace_sha256"]
    metrics = ss.score_against_twin(environment, outcome.rows, twin_outcome.rows)
    assert metrics == case["metrics"]
    assert metrics["peak_hazard_shift"] == case["objective"]


# ---------------------------------------------------------------------------
# Determinism and seed separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ss.METHODS)
def test_same_spec_method_seed_reproduces_audit_and_manifest(
    reference_runs: dict[str, RealRun], repeat_runs: dict[str, RealRun], method: str
) -> None:
    first, second = reference_runs[method], repeat_runs[method]
    assert first.directory != second.directory
    assert _hashed_chain(first.directory) == _hashed_chain(second.directory)
    assert first.summary.audit_head_sha256 == second.summary.audit_head_sha256
    assert first.summary.manifest_sha256 == second.summary.manifest_sha256
    assert (first.directory / ss.MANIFEST_FILE_NAME).read_bytes() == (
        second.directory / ss.MANIFEST_FILE_NAME
    ).read_bytes()
    for case in first.manifest["selected_cases"]:
        for entry in case["files"].values():
            assert (first.directory / entry["relative_path"]).read_bytes() == (
                second.directory / entry["relative_path"]
            ).read_bytes()


def test_different_seeds_and_methods_give_different_candidate_sequences(
    reference_runs: dict[str, RealRun], other_seed_run: RealRun
) -> None:
    def parameters(run: RealRun) -> list[Any]:
        return [record["parameters"] for record in _candidate_records(run.directory)]

    def episode_ids(run: RealRun) -> set[str]:
        return {
            record["episode_id"]
            for record in _candidate_records(run.directory)
            if record["episode_id"] is not None
        }

    random_one = reference_runs[ss.METHOD_RANDOM]
    adaptive_one = reference_runs[ss.METHOD_CROSS_ENTROPY]
    assert other_seed_run.summary.final is True
    assert parameters(random_one) != parameters(other_seed_run)
    assert parameters(random_one)[0] != parameters(other_seed_run)[0]
    assert not episode_ids(random_one) & episode_ids(other_seed_run)
    assert random_one.summary.manifest_sha256 != other_seed_run.summary.manifest_sha256
    # The method is part of the draw key, so the two methods do not share a
    # candidate stream even on one seed.
    assert parameters(random_one)[0] != parameters(adaptive_one)[0]
    assert not episode_ids(random_one) & episode_ids(adaptive_one)


# ---------------------------------------------------------------------------
# Interrupt and resume on the real plant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ss.METHODS)
def test_interrupt_then_resume_matches_uninterrupted_run(
    reference_runs: dict[str, RealRun],
    small_spec: ss.SearchSpec,
    tmp_path: Path,
    method: str,
) -> None:
    reference = reference_runs[method]
    directory = tmp_path / "interrupted"
    # Fires a few plant steps into a real episode after three full ones.
    cancel = _CancelAfterCalls(3 * CANCEL_CALLS_PER_EPISODE + 10)
    interrupted = ss.run_search(
        spec=small_spec,
        repo_root=REPO_ROOT,
        output_dir=directory,
        method=method,
        seed=SEED_ONE,
        clock=_frozen_clock,
        cancel_requested=cancel,
    )
    assert cancel.calls == cancel.limit + 1
    assert interrupted.final is False
    assert interrupted.end_reason == ss.END_CANCELLED
    assert interrupted.manifest_sha256 is None
    assert not (directory / ss.MANIFEST_FILE_NAME).exists()
    assert not (directory / ss.CASES_DIRECTORY_NAME).exists()
    assert 1 < interrupted.simulation_calls_used < MAX_SIMULATION_CALLS

    before = ss.AuditLog.load(directory / ss.AUDIT_FILE_NAME)
    recorded_before = before.records("candidate")
    assert len(recorded_before) == interrupted.candidate_attempts_used
    assert not before.records("run_footer")
    incomplete = before.records("incomplete")
    assert len(incomplete) == 1
    assert incomplete[0]["cause"] == ss.INTERRUPT_CANCELLED
    # The interrupted episode is explicit, uncounted and not a candidate record.
    assert incomplete[0]["attempt_index"] == len(recorded_before)
    assert incomplete[0]["simulation_calls_used"] == interrupted.simulation_calls_used
    assert incomplete[0]["candidate_attempts_used"] == interrupted.candidate_attempts_used
    session_end = before.records("session_end")
    assert len(session_end) == 1
    assert session_end[0]["final"] is False and session_end[0]["end_reason"] == ss.END_CANCELLED
    bytes_before = (directory / ss.AUDIT_FILE_NAME).read_bytes()

    resumed = ss.resume_search(output_dir=directory, repo_root=REPO_ROOT, clock=_frozen_clock)
    assert resumed.final is True
    assert resumed.end_reason == reference.summary.end_reason
    # Budgets continue from the interrupted counters; they are not reset.
    assert resumed.simulation_calls_used == reference.summary.simulation_calls_used
    assert resumed.simulation_calls_used == MAX_SIMULATION_CALLS
    assert resumed.candidate_attempts_used == reference.summary.candidate_attempts_used

    # The audit is append-only across the resume.
    assert (directory / ss.AUDIT_FILE_NAME).read_bytes().startswith(bytes_before)
    after = ss.AuditLog.load(directory / ss.AUDIT_FILE_NAME)
    candidates = after.records("candidate")
    assert candidates[: len(recorded_before)] == recorded_before
    assert [record["attempt_index"] for record in candidates] == list(range(len(candidates)))
    assert candidates == _candidate_records(reference.directory)
    assert after.records("baseline") == ss.AuditLog.load(
        reference.directory / ss.AUDIT_FILE_NAME
    ).records("baseline")
    assert len(after.records("baseline")) == 1
    assert [record["session_index"] for record in after.records("session_start")] == [0, 1]
    assert after.records("session_start")[1]["resumed_after_sequence"] == len(before.entries) - 1
    footer = after.records("run_footer")
    assert len(footer) == 1
    assert footer[0] == ss.AuditLog.load(reference.directory / ss.AUDIT_FILE_NAME).records(
        "run_footer"
    )[0]

    # The manifest reports the same discoveries.  It differs from the
    # uninterrupted manifest only where it explicitly records the interruption
    # (session count, incomplete attempts and the audit chain position).
    manifest = ss.load_manifest(directory / ss.MANIFEST_FILE_NAME)
    assert manifest["manifest_sha256"] == resumed.manifest_sha256
    assert manifest["budget_consumption"]["sessions"] == 2
    assert manifest["budget_consumption"]["incomplete_attempts"] == 1

    def without_session_provenance(value: dict[str, Any]) -> dict[str, Any]:
        stripped = copy.deepcopy(value)
        del stripped["manifest_sha256"]
        del stripped["audit"]["footer_record_sha256"]
        del stripped["budget_consumption"]["sessions"]
        del stripped["budget_consumption"]["incomplete_attempts"]
        return stripped

    assert without_session_provenance(manifest) == without_session_provenance(reference.manifest)
    for case in manifest["selected_cases"]:
        assert case["confirmation"]["status"] == "CONFIRMED"
        for entry in case["files"].values():
            assert (directory / entry["relative_path"]).read_bytes() == (
                reference.directory / entry["relative_path"]
            ).read_bytes()

    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        ss.resume_search(output_dir=directory, repo_root=REPO_ROOT, clock=_frozen_clock)


def test_resuming_an_uninterrupted_finished_run_is_refused(
    repeat_runs: dict[str, RealRun],
) -> None:
    run = repeat_runs[ss.METHOD_RANDOM]
    audit_before = (run.directory / ss.AUDIT_FILE_NAME).read_bytes()
    manifest_before = (run.directory / ss.MANIFEST_FILE_NAME).read_bytes()
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        ss.resume_search(output_dir=run.directory, repo_root=REPO_ROOT, clock=_frozen_clock)
    assert (run.directory / ss.AUDIT_FILE_NAME).read_bytes() == audit_before
    assert (run.directory / ss.MANIFEST_FILE_NAME).read_bytes() == manifest_before


def test_starting_a_run_in_an_existing_directory_is_refused(
    repeat_runs: dict[str, RealRun], small_spec: ss.SearchSpec
) -> None:
    run = repeat_runs[ss.METHOD_RANDOM]
    audit_before = (run.directory / ss.AUDIT_FILE_NAME).read_bytes()
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite"):
        ss.run_search(
            spec=small_spec,
            repo_root=REPO_ROOT,
            output_dir=run.directory,
            method=ss.METHOD_RANDOM,
            seed=SEED_ONE,
            clock=_frozen_clock,
        )
    assert (run.directory / ss.AUDIT_FILE_NAME).read_bytes() == audit_before


# ---------------------------------------------------------------------------
# Authoritative replay and fail-closed manifests
# ---------------------------------------------------------------------------


def test_replay_reproduces_reported_metrics_and_writes_once(
    reference_runs: dict[str, RealRun], tmp_path: Path
) -> None:
    run = reference_runs[ss.METHOD_RANDOM]
    manifest_path = run.directory / ss.MANIFEST_FILE_NAME
    output = tmp_path / "replay"
    receipt = ss.replay_manifest(
        manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output
    )
    tolerance = run.manifest["spec"]["replay"]["metric_absolute_tolerance"]
    assert receipt["status"] == "PASS"
    assert receipt["manifest_sha256"] == run.manifest["manifest_sha256"]
    assert receipt["metric_absolute_tolerance"] == tolerance
    assert receipt["case_count"] == len(run.manifest["selected_cases"]) >= 1
    assert receipt["reproduced_count"] == receipt["case_count"]
    assert receipt["data_boundary"] == ss.DATA_BOUNDARY
    for case, result in zip(run.manifest["selected_cases"], receipt["cases"], strict=True):
        assert result["episode_id"] == case["episode_id"]
        assert result["status"] == "REPRODUCED"
        assert result["replay_verified"] is True
        assert result["stored_scenario_bytes_identical"] is True
        assert result["stored_artifacts_identical"] is True
        assert result["objective_consistent_with_metrics"] is True
        assert result["trace_sha256_identical"] is True
        assert result["trace_sha256_required"] is True
        assert {item["metric"] for item in result["metric_comparisons"]} == set(case["metrics"])
        for item in result["metric_comparisons"]:
            assert item["within_tolerance"] is True
            assert item["reported"] == case["metrics"][item["metric"]]
            if item["absolute_difference"] is not None:
                assert item["absolute_difference"] <= tolerance

    receipt_path = output / ss.REPLAY_RECEIPT_FILE_NAME
    receipt_bytes = receipt_path.read_bytes()
    assert receipt_bytes == ss.canonical_bytes(receipt)
    assert receipt["receipt_sha256"] == ss.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in ("receipt_sha256", "runtime_environment")
        }
    )
    with pytest.raises(ss.StressSearchOutputError, match="refusing to overwrite"):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output)
    assert receipt_path.read_bytes() == receipt_bytes
    assert sorted(path.name for path in output.iterdir()) == [ss.REPLAY_RECEIPT_FILE_NAME]


def test_replay_rejects_resealed_manifest_with_changed_multiplier(
    reference_runs: dict[str, RealRun], tmp_path: Path
) -> None:
    manifest_path = _copy_run(reference_runs[ss.METHOD_RANDOM], tmp_path / "run")
    manifest = json.loads(manifest_path.read_bytes())
    fault = next(
        fault
        for fault in manifest["selected_cases"][0]["parameters"]
        if fault["multiplier"] is not None
    )
    low, high = manifest["spec"]["search_space"]["fault_types"][fault["type"]]["multiplier_range"]
    # Still a legal, quantised value inside the declared band, in a manifest
    # whose self-hash is correct: the audit recorded different parameters.
    fault["multiplier"] = float(low) if fault["multiplier"] != float(low) else float(high)
    _reseal(manifest_path, manifest)
    output = tmp_path / "replay"
    with pytest.raises(
        ss.StressSearchManifestError,
        match="manifest section 'selected_cases' does not match the audit log",
    ):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output)
    assert not output.exists()


def _drop_first_case(manifest: dict[str, Any]) -> None:
    del manifest["selected_cases"][0]


def _change_reported_metric(manifest: dict[str, Any]) -> None:
    manifest["selected_cases"][0]["metrics"]["peak_hazard_shift"] += 0.25


def _change_objective(manifest: dict[str, Any]) -> None:
    manifest["selected_cases"][0]["objective"] += 0.25


def _relabel_method(manifest: dict[str, Any]) -> None:
    manifest["method"] = next(name for name in ss.METHODS if name != manifest["method"])


def _understate_simulation_calls(manifest: dict[str, Any]) -> None:
    manifest["budget_consumption"]["simulation_calls_used"] -= 1


def _drop_rejected_contender(manifest: dict[str, Any]) -> None:
    del manifest["rejected_contenders"][0]


@pytest.mark.parametrize(
    ("edit", "section"),
    [
        (_drop_first_case, "selected_cases"),
        (_change_reported_metric, "selected_cases"),
        (_change_objective, "selected_cases"),
        (_relabel_method, "method"),
        (_understate_simulation_calls, "budget_consumption"),
        (_drop_rejected_contender, "rejected_contenders"),
    ],
    ids=[
        "dropped-case",
        "changed-metric",
        "changed-objective",
        "relabelled-method",
        "changed-budget-figure",
        "dropped-contender",
    ],
)
def test_resealed_manifest_edit_fails_closed_through_audit_binding(
    reference_runs: dict[str, RealRun], tmp_path: Path, edit: Any, section: str
) -> None:
    run = reference_runs[ss.METHOD_RANDOM]
    manifest_path = _copy_run(run, tmp_path / "run")
    audit_bytes = (manifest_path.parent / ss.AUDIT_FILE_NAME).read_bytes()
    assert ss.load_run_manifest(manifest_path) == run.manifest
    manifest = json.loads(manifest_path.read_bytes())
    edit(manifest)
    # ``_reseal`` proves the edited document is canonical, has a correct
    # self-hash and passes the document-only loader; only the audit catches it.
    _reseal(manifest_path, manifest)
    assert manifest["manifest_sha256"] != run.manifest["manifest_sha256"]
    message = f"manifest section '{section}' does not match the audit log"
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.load_run_manifest(manifest_path)
    output = tmp_path / "replay"
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output)
    # No receipt of any status is written for an unbound manifest.
    assert not output.exists()
    # A finished run whose manifest was edited is not reported as "complete".
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.resume_search(
            output_dir=manifest_path.parent, repo_root=REPO_ROOT, clock=_frozen_clock
        )
    assert (manifest_path.parent / ss.AUDIT_FILE_NAME).read_bytes() == audit_bytes


def test_manifest_copied_beside_another_runs_audit_is_refused(
    reference_runs: dict[str, RealRun], other_seed_run: RealRun, tmp_path: Path
) -> None:
    manifest_path = _copy_run(reference_runs[ss.METHOD_RANDOM], tmp_path / "run")
    # A genuine, untouched manifest from a different real run (other seed).
    manifest_path.write_bytes((other_seed_run.directory / ss.MANIFEST_FILE_NAME).read_bytes())
    assert ss.load_manifest(manifest_path) == other_seed_run.manifest
    with pytest.raises(ss.StressSearchManifestError, match="does not match the audit log"):
        ss.load_run_manifest(manifest_path)
    with pytest.raises(ss.StressSearchManifestError, match="does not match the audit log"):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT)


@pytest.mark.parametrize("name", ["control-trace.json", "actuation.jsonl"])
def test_replay_rejects_tampered_stored_case_artifact(
    reference_runs: dict[str, RealRun], tmp_path: Path, name: str
) -> None:
    manifest_path = _copy_run(reference_runs[ss.METHOD_RANDOM], tmp_path / "run")
    manifest = ss.load_run_manifest(manifest_path)
    case = manifest["selected_cases"][0]
    stored = manifest_path.parent / case["files"][name]["relative_path"]
    original = stored.read_bytes()
    assert _sha256(original) == case["files"][name]["sha256"]
    # Same length, one digit changed: only the recorded digest can notice.
    position = next(index for index, byte in enumerate(original) if chr(byte).isdigit())
    replacement = b"1" if original[position : position + 1] != b"1" else b"2"
    tampered = original[:position] + replacement + original[position + 1 :]
    assert tampered != original and len(tampered) == len(original)
    stored.write_bytes(tampered)
    output = tmp_path / "replay"
    with pytest.raises(
        ss.StressSearchManifestError,
        match=f"stored {name} for case .* does not match its recorded digest",
    ):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output)
    assert not output.exists()
    stored.unlink()
    with pytest.raises(
        ss.StressSearchManifestError, match=f"stored {name} for case .* is missing"
    ):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output)
    assert not output.exists()


def test_replay_rejects_corrupted_stored_scenario(
    reference_runs: dict[str, RealRun], tmp_path: Path
) -> None:
    manifest_path = _copy_run(reference_runs[ss.METHOD_RANDOM], tmp_path / "run")
    manifest = ss.load_manifest(manifest_path)
    relative = manifest["selected_cases"][0]["files"]["scenario.json"]["relative_path"]
    stored = manifest_path.parent / relative
    original = stored.read_bytes()
    corrupted = original.replace(b'"steps":32', b'"steps":31')
    assert corrupted != original
    stored.write_bytes(corrupted)
    with pytest.raises(
        ss.StressSearchManifestError,
        match="stored scenario.json for case .* does not match its recorded digest",
    ):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT)
    stored.unlink()
    with pytest.raises(
        ss.StressSearchManifestError, match="stored scenario.json for case .* is missing"
    ):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT)


def test_replay_rejects_resealed_manifest_with_foreign_source_identity(
    reference_runs: dict[str, RealRun], tmp_path: Path
) -> None:
    manifest_path = _copy_run(reference_runs[ss.METHOD_RANDOM], tmp_path / "run")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["identity"]["source_sha256"] = "f" * 64
    _reseal(manifest_path, manifest)
    output = tmp_path / "replay"
    with pytest.raises(
        ss.StressSearchManifestError,
        match="manifest section 'identity' does not match the audit log",
    ):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT, output_dir=output)
    assert not output.exists()


def test_replay_rejects_manifest_edited_without_resealing(
    reference_runs: dict[str, RealRun], tmp_path: Path
) -> None:
    manifest_path = _copy_run(reference_runs[ss.METHOD_RANDOM], tmp_path / "run")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["selected_cases"][0]["objective"] += 1.0
    manifest_path.write_bytes(ss.canonical_bytes(manifest))
    with pytest.raises(ss.StressSearchManifestError, match="self-hash mismatch"):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT)
    manifest_path.write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    with pytest.raises(ss.StressSearchManifestError, match="not canonical JSON"):
        ss.replay_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Terminal failures are data
# ---------------------------------------------------------------------------


def test_hmc_terminal_failure_is_returned_as_data(environment: ss.SearchEnvironment) -> None:
    data = copy.deepcopy(dict(environment.twin_data))
    data["initial_utility"]["oxygen_store_mol"] = 0.2
    data["name"] = "test-stress-search-terminal-oxygen-store"
    scenario = Scenario.from_mapping(data)
    outcome = ss.run_authoritative_episode(
        scenario, environment.contract, "2" * 64, verify_replay=True, keep_artifacts=True
    )
    assert outcome.status == ss.OUTCOME_TERMINAL
    assert type(outcome.terminal_reason) is str and outcome.terminal_reason
    assert outcome.error is None
    assert 0 < outcome.committed_steps < HORIZON_STEPS
    assert outcome.rows is not None and outcome.rows.shape[0] == outcome.committed_steps
    assert len(outcome.actuation) == outcome.committed_steps
    assert type(outcome.trace_sha256) is str and len(outcome.trace_sha256) == 64
    assert outcome.trace_bytes is not None
    assert _sha256(outcome.trace_bytes) == outcome.trace_sha256
    # A truncated episode never receives completed-episode metrics.
    assert outcome.race_metrics is None


# ---------------------------------------------------------------------------
# Non-finite plant truth on the real path
# ---------------------------------------------------------------------------


class _TruthProjectionRefusal:
    """Wraps the real ``project_true_targets`` and refuses one chosen call.

    TEST-ONLY TRIGGER.  The plant, the HMC and every other projection call are
    real; this exists solely to drive the harness down the path it takes when
    the existing truth projection refuses non-finite plant state.  Nothing
    recorded because of it is a plant result.
    """

    def __init__(self, fail_on_call: int) -> None:
        self._real = ss.project_true_targets
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise Issue55RaceError("test-induced refusal: plant truth declared non-finite")
        return self._real(*args, **kwargs)


def test_truth_projection_refusal_is_returned_as_nonfinite_outcome(
    environment: ss.SearchEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patches only the truth projection, to reach the failure path (see the
    # class docstring).  Call 1 projects the initial state; call ``k + 1``
    # projects the state after committed step ``k``.
    failing_step = 5
    refusal = _TruthProjectionRefusal(fail_on_call=failing_step + 1)
    monkeypatch.setattr(ss, "project_true_targets", refusal)
    outcome = ss.run_authoritative_episode(
        environment.twin_scenario,
        environment.contract,
        "3" * 64,
        verify_replay=True,
        keep_artifacts=True,
    )
    assert refusal.calls == failing_step + 1
    assert outcome.status == ss.OUTCOME_NONFINITE
    assert outcome.committed_steps == failing_step
    assert 0 < outcome.committed_steps < HORIZON_STEPS
    assert f"plant truth is not finite at step {failing_step}" in outcome.error
    assert outcome.terminal_reason is None
    # A truncated, unscoreable episode carries no truth, metrics or trace.
    assert outcome.rows is None
    assert outcome.race_metrics is None
    assert outcome.trace_sha256 is None and outcome.trace_bytes is None
    assert outcome.replay_verified is False
    assert type(outcome.actuation_sha256) is str and len(outcome.actuation_sha256) == 64

    initial = _TruthProjectionRefusal(fail_on_call=1)
    monkeypatch.setattr(ss, "project_true_targets", initial)
    outcome = ss.run_authoritative_episode(
        environment.twin_scenario, environment.contract, "3" * 64
    )
    assert outcome.status == ss.OUTCOME_NONFINITE
    assert outcome.committed_steps == 0
    assert "initial plant truth is not finite" in outcome.error


def test_search_records_nonfinite_candidate_and_still_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _small_spec_mapping()
    mapping["spec_id"] = "test-stress-search-integration-nonfinite"
    mapping["budget"]["max_simulation_calls"] = 3
    spec = ss.validate_spec_mapping(mapping)
    failing_step = 4
    # The healthy twin makes HORIZON_STEPS + 1 projection calls; the refusal
    # lands on step ``failing_step`` of the first simulated candidate.  Patches
    # only the truth projection, to reach the failure path: the recorded
    # NONFINITE_OUTPUT is test-induced, not a plant result.
    refusal = _TruthProjectionRefusal(fail_on_call=(HORIZON_STEPS + 1) + 1 + failing_step)
    monkeypatch.setattr(ss, "project_true_targets", refusal)
    directory = tmp_path / "run"
    summary = ss.run_search(
        spec=spec,
        repo_root=REPO_ROOT,
        output_dir=directory,
        method=ss.METHOD_RANDOM,
        seed=SEED_ONE,
        clock=_frozen_clock,
    )
    assert refusal.calls > refusal.fail_on_call
    assert summary.final is True
    assert summary.end_reason == ss.END_SIMULATION_BUDGET
    assert summary.simulation_calls_used == 3

    candidates = _candidate_records(directory)
    simulated = [record for record in candidates if record["outcome"] is not None]
    assert len(simulated) == 2
    failed = simulated[0]
    assert failed["disposition"] == ss.DISPOSITION_NONFINITE
    assert failed["outcome"]["status"] == ss.OUTCOME_NONFINITE
    assert failed["outcome"]["committed_steps"] == failing_step
    assert f"plant truth is not finite at step {failing_step}" in failed["reason"]
    assert failed["metrics"] is None and failed["objective"] is None
    assert failed["outcome"]["trace_sha256"] is None
    # The failed episode was charged to the simulation budget.
    assert failed["simulation_calls_used"] == 2
    assert simulated[1]["simulation_calls_used"] == 3
    assert simulated[1]["disposition"] != ss.DISPOSITION_NONFINITE

    manifest = ss.load_run_manifest(directory / ss.MANIFEST_FILE_NAME)
    assert manifest["manifest_sha256"] == summary.manifest_sha256
    assert manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_NONFINITE] == 1
    assert failed["episode_id"] not in {case["episode_id"] for case in manifest["selected_cases"]}
    contender = next(
        item
        for item in manifest["rejected_contenders"]
        if item["attempt_index"] == failed["attempt_index"]
    )
    assert contender["disposition"] == ss.DISPOSITION_NONFINITE
    assert contender["not_selected_reason"] == f"not eligible: {ss.DISPOSITION_NONFINITE}"
    # A non-finite episode is never a discovery in the comparison figures.
    figures = ss.summarize_run(manifest)
    assert figures["dispositions"][ss.DISPOSITION_NONFINITE] == 1
    assert figures["evaluated_candidates"] == sum(
        1 for record in candidates if record["disposition"] == ss.DISPOSITION_EVALUATED
    )


# ---------------------------------------------------------------------------
# Single writer
# ---------------------------------------------------------------------------


def test_second_writer_is_refused_while_run_lock_is_held(
    small_spec: ss.SearchSpec, tmp_path: Path
) -> None:
    directory = tmp_path / "run"
    # Stopped at the first cancellation check, right after the real baseline.
    stopped = ss.run_search(
        spec=small_spec,
        repo_root=REPO_ROOT,
        output_dir=directory,
        method=ss.METHOD_RANDOM,
        seed=SEED_ONE,
        clock=_frozen_clock,
        cancel_requested=_CancelAfterCalls(0),
    )
    assert stopped.final is False and stopped.end_reason == ss.END_CANCELLED
    assert stopped.simulation_calls_used == 1 and stopped.candidate_attempts_used == 0
    assert (directory / ss.LOCK_FILE_NAME).is_file()
    audit_before = (directory / ss.AUDIT_FILE_NAME).read_bytes()

    with ss.RunLock(directory):
        with pytest.raises(ss.StressSearchOutputError, match="another process is writing"):
            ss.resume_search(output_dir=directory, repo_root=REPO_ROOT, clock=_frozen_clock)
        with pytest.raises(ss.StressSearchOutputError, match="another process is writing"):
            ss.RunLock(directory).__enter__()
        # The refused writer appended nothing and finalised nothing.
        assert (directory / ss.AUDIT_FILE_NAME).read_bytes() == audit_before
        assert not (directory / ss.MANIFEST_FILE_NAME).exists()

    # Once the holder releases the lock the run is resumable again.
    resumed = ss.resume_search(
        output_dir=directory,
        repo_root=REPO_ROOT,
        clock=_frozen_clock,
        cancel_requested=_CancelAfterCalls(0),
    )
    assert resumed.final is False and resumed.end_reason == ss.END_CANCELLED
    audit = ss.AuditLog.load(directory / ss.AUDIT_FILE_NAME)
    assert (directory / ss.AUDIT_FILE_NAME).read_bytes().startswith(audit_before)
    assert [record["session_index"] for record in audit.records("session_start")] == [0, 1]
    assert len(audit.records("baseline")) == 1


# ---------------------------------------------------------------------------
# Matched-budget comparison on real runs
# ---------------------------------------------------------------------------


def _all_keys(value: Any) -> set[str]:
    if type(value) is dict:
        return set(value) | {name for item in value.values() for name in _all_keys(item)}
    if type(value) is list:
        return {name for item in value for name in _all_keys(item)}
    return set()


def test_comparison_of_real_runs_covers_every_run_and_claims_nothing(
    reference_runs: dict[str, RealRun],
    other_seed_run: RealRun,
    other_seed_cross_entropy_run: RealRun,
    small_spec: ss.SearchSpec,
) -> None:
    real_runs = {
        (ss.METHOD_RANDOM, SEED_ONE): reference_runs[ss.METHOD_RANDOM],
        (ss.METHOD_CROSS_ENTROPY, SEED_ONE): reference_runs[ss.METHOD_CROSS_ENTROPY],
        (ss.METHOD_RANDOM, SEED_TWO): other_seed_run,
        (ss.METHOD_CROSS_ENTROPY, SEED_TWO): other_seed_cross_entropy_run,
    }
    manifests = {
        key: ss.load_run_manifest(run.directory / ss.MANIFEST_FILE_NAME)
        for key, run in real_runs.items()
    }
    comparison = ss.compare_runs(list(manifests.values()))

    assert comparison["data_boundary"] == ss.DATA_BOUNDARY
    assert comparison["seeds"] == [SEED_ONE, SEED_TWO]
    assert comparison["budget"] == small_spec.data["budget"]
    tiers = small_spec.data["discovery_tiers"]
    tier_ids = [tier["tier_id"] for tier in tiers]
    assert comparison["discovery_tiers"] == tiers
    assert comparison["comparison_sha256"] == ss.canonical_sha256(
        {key: value for key, value in comparison.items() if key != "comparison_sha256"}
    )

    # Every (method, seed) appears exactly once; nothing is left out.
    rows = {(row["method"], row["seed"]): row for row in comparison["runs"]}
    assert len(comparison["runs"]) == 4
    assert set(rows) == {(method, seed) for method in ss.METHODS for seed in small_spec.seeds}
    for key, row in rows.items():
        manifest = manifests[key]
        candidates = _candidate_records(real_runs[key].directory)
        evaluated = [
            record for record in candidates if record["disposition"] == ss.DISPOSITION_EVALUATED
        ]
        assert row["manifest_sha256"] == manifest["manifest_sha256"]
        assert row["end_reason"] in ss.DETERMINISTIC_END_REASONS
        assert row["simulation_calls_used"] == MAX_SIMULATION_CALLS
        assert row["candidate_attempts_used"] == len(candidates)
        assert row["sessions"] == 1 and row["incomplete_attempts"] == 0
        assert row["confirmation_calls_outside_search_budget"] == row["selected_cases"] + 1
        assert row["confirmed_cases"] == row["selected_cases"] == len(manifest["selected_cases"])
        assert row["evaluated_candidates"] == len(evaluated)
        assert row["terminal_failure_episodes"] == len(
            {
                record["episode_id"]
                for record in candidates
                if record["disposition"] == ss.DISPOSITION_TERMINAL
            }
        )
        assert row["dispositions"] == manifest["budget_consumption"]["dispositions"]
        assert row["best_objective"] == max(record["objective"] for record in evaluated)
        # Tier counts, recomputed here from the audit's evaluated candidates.
        assert [tier["tier_id"] for tier in row["tiers"]] == tier_ids
        for tier, figures in zip(tiers, row["tiers"], strict=True):
            hits = [record for record in evaluated if ss.tier_met(tier, record["metrics"])]
            families = {record["causal_family_id"] for record in hits}
            assert figures["discovered_episodes"] == len({record["episode_id"] for record in hits})
            assert figures["discovered_causal_families"] == len(families)
            assert figures["simulation_calls_to_first_discovery"] == min(
                (record["simulation_calls_used"] for record in hits), default=None
            )
            assert figures["discovered_families_per_simulation_call"] == (
                len(families) / MAX_SIMULATION_CALLS
            )

    for method in ss.METHODS:
        summary = comparison["per_method"][method]
        assert summary["simulation_calls_used"]["count"] == 2
        assert summary["best_objective"]["count"] == 2
        assert sorted(summary["tiers"]) == sorted(tier_ids)
        for index, tier_id in enumerate(tier_ids):
            assert summary["tiers"][tier_id]["seeds_with_discovery"] == sum(
                1
                for seed in small_spec.seeds
                if rows[(method, seed)]["tiers"][index]["discovered_causal_families"] > 0
            )

    # Paired figures are taken at the common simulation-call count.
    assert comparison["all_seeds_used_equal_simulation_calls"] is True
    assert [item["seed"] for item in comparison["paired_by_seed"]] == [SEED_ONE, SEED_TWO]
    for item in comparison["paired_by_seed"]:
        assert item["simulation_calls_used"] == {
            method: MAX_SIMULATION_CALLS for method in ss.METHODS
        }
        assert item["simulation_calls_matched"] is True
        assert item["compared_at_simulation_calls"] == MAX_SIMULATION_CALLS
        for method in ss.METHODS:
            assert item["best_objective"][method] == rows[(method, item["seed"])]["best_objective"]
        assert item["best_objective_cross_entropy_minus_random"] == (
            item["best_objective"][ss.METHOD_CROSS_ENTROPY]
            - item["best_objective"][ss.METHOD_RANDOM]
        )
        assert sorted(item["tier_family_difference_cross_entropy_minus_random"]) == sorted(tier_ids)
    paired = comparison["paired_best_objective_summary"]
    assert (
        paired["seeds_cross_entropy_higher"] + paired["seeds_random_higher"] + paired["seeds_tied"]
        == 2
    )

    # Descriptive only: no winner, no significance, no superiority claim.
    interpretation = " ".join(comparison["interpretation"].split())
    assert "No significance test was run" in interpretation
    assert "no claim is made that either method is better" in interpretation
    forbidden = ("winner", "superior", "outperform", "significan", "p_value", "better")
    assert not [
        name for name in _all_keys(comparison) if any(word in name.lower() for word in forbidden)
    ]
    report = ss.render_comparison_markdown(comparison)
    assert comparison["interpretation"] in report
    assert ss.DATA_BOUNDARY in report
    for method, seed in rows:
        assert f"| {method} | {seed} |" in report
    report_without_disclaimer = report.replace(comparison["interpretation"], "").lower()
    for word in forbidden:
        assert word not in report_without_disclaimer, word

    # A truncated common count never reports more than the full run did.
    for key, manifest in manifests.items():
        truncated = ss.summarize_run(manifest, max_simulation_calls=MAX_SIMULATION_CALLS // 2)
        assert truncated["simulation_calls_used"] == MAX_SIMULATION_CALLS // 2
        assert truncated["candidate_attempts_used"] <= rows[key]["candidate_attempts_used"]
        assert truncated["evaluated_candidates"] <= rows[key]["evaluated_candidates"]

    # A missing run cannot be silently left out, and a run cannot be counted twice.
    with pytest.raises(ss.StressSearchManifestError, match="needs every declared seed"):
        ss.compare_runs(list(manifests.values())[:3])
    with pytest.raises(ss.StressSearchManifestError, match="duplicate run"):
        ss.compare_runs([*manifests.values(), manifests[(ss.METHOD_RANDOM, SEED_ONE)]])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _run_cli(cli_module: ModuleType, arguments: list[str]) -> int:
    """Call ``main`` and restore the SIGINT handler the CLI may have installed."""

    previous = signal.getsignal(signal.SIGINT)
    try:
        return cli_module.main(arguments)
    finally:
        signal.signal(signal.SIGINT, previous)


def test_cli_search_refuses_output_outside_out_directory(
    cli_module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "not-under-out"
    code = _run_cli(
        cli_module,
        ["search", "--method", ss.METHOD_RANDOM, "--seed", "issue93-s1", "--output", str(output)],
    )
    assert code == 2
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    assert "must live under the ignored out/ directory" in capsys.readouterr().err


def test_cli_search_refuses_undeclared_seed_and_creates_nothing(
    cli_module: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # A unique, never-created name: the refusal must happen before any write.
    output = REPO_ROOT / "out" / f"pytest-stress-search-never-created-{uuid.uuid4().hex}"
    assert not output.exists()
    code = _run_cli(
        cli_module,
        [
            "search",
            "--method",
            ss.METHOD_RANDOM,
            "--seed",
            "test-undeclared-seed",
            "--output",
            str(output),
        ],
    )
    assert code == 2
    assert not output.exists()
    assert "is not declared in the spec" in capsys.readouterr().err


def test_cli_resume_and_replay_refuse_paths_outside_out_directory(
    cli_module: ModuleType,
    reference_runs: dict[str, RealRun],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = reference_runs[ss.METHOD_RANDOM]
    assert _run_cli(cli_module, ["resume", "--run", str(run.directory)]) == 2
    assert "must live under the ignored out/ directory" in capsys.readouterr().err
    output = tmp_path / "replay"
    code = _run_cli(
        cli_module,
        [
            "replay",
            "--manifest",
            str(run.directory / ss.MANIFEST_FILE_NAME),
            "--output",
            str(output),
        ],
    )
    assert code == 2
    assert not output.exists()
    assert "must live under the ignored out/ directory" in capsys.readouterr().err
    # ``compare`` and ``run-all`` (fresh and ``--resume``) refuse the same way.
    assert (
        _run_cli(cli_module, ["compare", "--runs", str(run.directory), "--output", str(output)])
        == 2
    )
    assert "must live under the ignored out/ directory" in capsys.readouterr().err
    for extra in ([], ["--resume"]):
        assert _run_cli(cli_module, ["run-all", "--output", str(output), *extra]) == 2
        assert "must live under the ignored out/ directory" in capsys.readouterr().err
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Boundaries: no roster or custody access, GPU screening stays open
# ---------------------------------------------------------------------------


def test_module_source_never_touches_roster_builders_or_custody() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("build_family", "generate_families", "assign_partitions", "family_custody"):
        assert forbidden not in source, forbidden

    imported_modules: set[str] = set()
    imported_from_families: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
            if node.module == "bdm_v1_families":
                imported_from_families.update(alias.name for alias in node.names)
    assert not [name for name in imported_modules if "custody" in name.lower()]
    assert not [name for name in imported_modules if "corpus" in name.lower()]
    # Only reviewed vocabulary constants, the development base-fixture loader
    # and that loader's error type come from the roster module; no family or
    # partition machinery.
    assert imported_from_families == {
        "BdmV1FamilyError",
        "DT_SECONDS",
        "GENERATOR_FAULT_DURATION_RANGE",
        "GENERATOR_FAULT_MULTIPLIER_BANDS",
        "GENERATOR_FAULT_ONSET_CLASSES",
        "GENERATOR_RAMP_SHAPES",
        "SENSOR_SEED_BASE",
        "load_base_scenario_data",
    }
    # The only mention of protected populations is the refusal statement.
    assert source.count("BLIND_FINAL") == 1
    assert "not a validation, final or BLIND_FINAL population" in " ".join(
        ss.DATA_BOUNDARY.split()
    )


@pytest.mark.parametrize("method", ss.METHODS)
def test_manifest_keeps_gpu_screening_open_and_cpu_only(
    reference_runs: dict[str, RealRun], method: str
) -> None:
    manifest = reference_runs[method].manifest
    assert manifest["gpu_screening"]["status"] == "OPEN_NOT_RUN"
    assert manifest["gpu_screening"]["enabled"] is False
    assert manifest["spec"]["screening"] == {"backend": "none"}
    assert manifest["spec"]["population"] == {
        "scope": "DEVELOPMENT_ONLY",
        "protected_population_access": "prohibited",
    }


def test_accelerated_screening_backend_is_refused_before_any_run(tmp_path: Path) -> None:
    mapping = _small_spec_mapping()
    mapping["screening"]["backend"] = "gpu_batched"
    with pytest.raises(ss.StressSearchSpecError, match="accelerated screening is refused"):
        ss.validate_spec_mapping(mapping)
    assert list(tmp_path.iterdir()) == []
