"""Regression tests for the Issue 93 stress-search hardening fixes.

Scope and honesty boundary
--------------------------
Every test here that drives ``run_search`` / ``resume_search`` uses the
FABRICATED evaluator from ``test_stress_search_unit`` (``FakeEvaluator``) or a
thin wrapper around it.  Nothing in this module runs the plant or the HMC.
These are bookkeeping tests of the driver, the audit log, the manifest loader
and the command-line refusal paths.  They are NOT integration evidence, say
nothing about the physics, and their scores and selections are meaningless as
results.

The command-line tests point the script's ``REPO_ROOT`` at ``tmp_path`` and
exercise refusal and exit-code paths only, so no search is started and nothing
is written outside ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from aeolus.habitat_v2 import stress_search as ss
from aeolus.habitat_v2.scenario import Scenario

from .test_stress_search_unit import (
    REPO_ROOT,
    FakeClock,
    FakeEvaluator,
    audit_lines,
    candidates_of,
    load_audit,
    make_spec,
    reseal,
    resume_fake,
    run_fake,
    run_manifest_of,
    single_fault_mapping,
    small_mapping,
    write_audit,
)

CLI_PATH = REPO_ROOT / "scripts" / "run_fault_stress_search.py"
TRUTH_WIDTH = 51


def sorted_tree(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


# ---------------------------------------------------------------------------
# 1. The abort handler re-synchronises with the audit on disk
# ---------------------------------------------------------------------------


def test_interrupt_inside_an_audit_append_leaves_a_loadable_resumable_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_fake(tmp_path / "straight", small_mapping())
    run = tmp_path / "run"
    real_fsync = os.fsync
    fired: list[int] = []

    def interrupting_fsync(descriptor: int) -> None:
        # The record is durably on disk before the interrupt lands, so the
        # in-memory log is one entry behind the file when the handler runs.
        real_fsync(descriptor)
        audit_path = run / ss.AUDIT_FILE_NAME
        if fired or not audit_path.exists():
            return
        lines = audit_path.read_bytes().split(b"\n")[:-1]
        record = json.loads(lines[-1])["record"] if lines else {}
        if record.get("kind") == "candidate" and record["attempt_index"] == 2:
            fired.append(len(lines))
            raise KeyboardInterrupt

    monkeypatch.setattr(ss.os, "fsync", interrupting_fsync)
    with pytest.raises(KeyboardInterrupt):
        run_fake(run, small_mapping())
    monkeypatch.setattr(ss.os, "fsync", real_fsync)
    assert len(fired) == 1
    assert not (run / ss.MANIFEST_FILE_NAME).exists()

    # The audit still loads: unique consecutive sequence numbers, chain intact.
    audit = load_audit(run)
    sequences = [entry["sequence"] for entry in audit.entries]
    assert sequences == list(range(len(sequences)))
    assert len(set(sequences)) == len(sequences)
    # The interrupted append is on disk exactly once, followed by the abort record.
    assert len(audit.entries) == fired[0] + 1
    recorded = audit.records("candidate")
    assert [record["attempt_index"] for record in recorded] == [0, 1, 2]
    last = audit.entries[-1]["record"]
    assert last["kind"] == "session_end"
    assert last["end_reason"] == ss.END_ABORTED == "ABORTED_ERROR"
    assert last["final"] is False
    assert last["error_type"] == "KeyboardInterrupt"
    assert last["candidate_attempts_used"] == len(recorded)
    assert audit.entries[-1]["previous_record_sha256"] == audit.entries[-2]["record_sha256"]
    assert audit.records("run_footer") == []
    assert len(audit.records("session_start")) == len(audit.records("session_end")) == 1

    # Resume completes the run to the same result as an uninterrupted one.
    final, _ = resume_fake(run)
    assert (final.end_reason, final.final) == (ss.END_SIMULATION_BUDGET, True)
    assert final.manifest_sha256 is not None
    assert candidates_of(run) == candidates_of(tmp_path / "straight")
    manifest = run_manifest_of(run)
    assert manifest["manifest_sha256"] == final.manifest_sha256
    assert manifest["budget_consumption"]["sessions"] == 2
    assert [record["end_reason"] for record in load_audit(run).records("session_end")] == [
        ss.END_ABORTED,
        ss.END_SIMULATION_BUDGET,
    ]


# ---------------------------------------------------------------------------
# 2. Confirmation blocks are bound by load_run_manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def confirmed_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("hardening-confirmed") / "run"
    run_fake(run, small_mapping(), evaluator=FakeEvaluator(faithful=True))
    return run


@pytest.fixture(scope="module")
def mismatch_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("hardening-mismatch") / "run"
    run_fake(run, small_mapping())
    return run


@pytest.fixture()
def confirmed_run(confirmed_template: Path, tmp_path: Path) -> Path:
    """A finished fabricated run whose selected cases are all CONFIRMED."""

    return Path(shutil.copytree(confirmed_template, tmp_path / "run"))


@pytest.fixture()
def mismatch_run(mismatch_template: Path, tmp_path: Path) -> Path:
    """A finished fabricated run whose selected cases are CONFIRMATION_MISMATCH."""

    return Path(shutil.copytree(mismatch_template, tmp_path / "mismatch-run"))


def _block(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["selected_cases"][0]["confirmation"]


def _float_comparison(manifest: dict[str, Any]) -> dict[str, Any]:
    return next(
        item for item in _block(manifest)["metric_comparisons"] if type(item["reproduced"]) is float
    )


def _status_to_mismatch(manifest: dict[str, Any]) -> None:
    _block(manifest)["status"] = "CONFIRMATION_MISMATCH"


def _replay_verified_string(manifest: dict[str, Any]) -> None:
    _block(manifest)["replay_verified"] = "true"


def _replay_verified_integer(manifest: dict[str, Any]) -> None:
    _block(manifest)["replay_verified"] = 1


def _replay_verified_null(manifest: dict[str, Any]) -> None:
    _block(manifest)["replay_verified"] = None


def _pipeline_renamed(manifest: dict[str, Any]) -> None:
    _block(manifest)["pipeline"] = "fabricated_pipeline"


def _trace_flag_flipped(manifest: dict[str, Any]) -> None:
    _block(manifest)["trace_sha256_identical"] = not _block(manifest)["trace_sha256_identical"]


def _trace_flag_integer(manifest: dict[str, Any]) -> None:
    _block(manifest)["trace_sha256_identical"] = 1


def _comparisons_emptied(manifest: dict[str, Any]) -> None:
    _block(manifest)["metric_comparisons"] = []


def _comparisons_not_a_list(manifest: dict[str, Any]) -> None:
    _block(manifest)["metric_comparisons"] = {}


def _reproduced_changed(manifest: dict[str, Any]) -> None:
    _float_comparison(manifest)["reproduced"] += 1.0


def _within_tolerance_changed(manifest: dict[str, Any]) -> None:
    item = _float_comparison(manifest)
    item["within_tolerance"] = not item["within_tolerance"]


def _outcome_field_missing(manifest: dict[str, Any]) -> None:
    del _block(manifest)["outcome"]["trace_sha256"]


def _calls(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["budget_consumption"]


def _calls_one_more(manifest: dict[str, Any]) -> None:
    _calls(manifest)[ss._CONFIRMATION_CALLS_FIELD] += 1


def _calls_zero(manifest: dict[str, Any]) -> None:
    _calls(manifest)[ss._CONFIRMATION_CALLS_FIELD] = 0


def _calls_negative(manifest: dict[str, Any]) -> None:
    _calls(manifest)[ss._CONFIRMATION_CALLS_FIELD] = -1


_CONFIRMATION_EDITS: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
    ("status_to_mismatch", _status_to_mismatch, "status does not follow from its recorded evidence"),
    ("replay_verified_string", _replay_verified_string, "confirmation flags must be booleans"),
    ("replay_verified_integer", _replay_verified_integer, "confirmation flags must be booleans"),
    ("replay_verified_null", _replay_verified_null, "confirmation flags must be booleans"),
    ("pipeline_renamed", _pipeline_renamed, "case confirmation pipeline is unknown"),
    ("trace_flag_flipped", _trace_flag_flipped, "trace identity flag is inconsistent"),
    ("trace_flag_integer", _trace_flag_integer, "confirmation flags must be booleans"),
    ("comparisons_emptied", _comparisons_emptied, "status does not follow from its recorded evidence"),
    ("comparisons_not_a_list", _comparisons_not_a_list, "metric_comparisons must be a list"),
    ("reproduced_changed", _reproduced_changed, "do not restate the case metrics"),
    ("within_tolerance_changed", _within_tolerance_changed, "do not restate the case metrics"),
    (
        "outcome_field_missing",
        _outcome_field_missing,
        "missing confirmation outcome fields: trace_sha256",
    ),
    ("calls_one_more", _calls_one_more, "confirmation call count must be an integer"),
    ("calls_zero", _calls_zero, "confirmation call count must be an integer"),
    ("calls_negative", _calls_negative, "confirmation call count must be an integer"),
)


def test_untouched_manifests_load_and_carry_the_expected_confirmation_status(
    confirmed_run: Path, mismatch_run: Path
) -> None:
    for run, status in ((confirmed_run, "CONFIRMED"), (mismatch_run, "CONFIRMATION_MISMATCH")):
        manifest = run_manifest_of(run)
        cases = manifest["selected_cases"]
        assert cases and all(case["confirmation"]["status"] == status for case in cases)
        assert manifest["budget_consumption"][ss._CONFIRMATION_CALLS_FIELD] == len(cases) + 1
        # Re-sealing an unedited manifest is a no-op that still loads.
        reseal(run, copy.deepcopy(manifest))
        assert run_manifest_of(run) == manifest


@pytest.mark.parametrize(
    ("edit", "message"),
    [case[1:] for case in _CONFIRMATION_EDITS],
    ids=[case[0] for case in _CONFIRMATION_EDITS],
)
def test_run_manifest_with_an_edited_and_re_sealed_confirmation_fails_closed(
    confirmed_run: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    manifest = run_manifest_of(confirmed_run)
    edited = copy.deepcopy(manifest)
    edit(edited)
    # Compared as canonical bytes: in Python ``1 == True``, in JSON it is not.
    assert ss.canonical_bytes(edited) != ss.canonical_bytes(manifest)
    path = reseal(confirmed_run, edited)
    # The edit is correctly re-sealed: only the confirmation binding can catch it.
    sealed = json.loads(path.read_bytes())
    assert sealed["manifest_sha256"] != manifest["manifest_sha256"]
    assert sealed["manifest_sha256"] == ss.canonical_sha256(
        {key: value for key, value in sealed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ss.StressSearchManifestError, match=message):
        ss.load_run_manifest(path)
    # Restoring the original bytes restores a loadable manifest.
    reseal(confirmed_run, copy.deepcopy(manifest))
    assert run_manifest_of(confirmed_run) == manifest


def test_run_manifest_with_a_mismatch_promoted_to_confirmed_fails_closed(
    mismatch_run: Path,
) -> None:
    manifest = run_manifest_of(mismatch_run)
    for index in range(len(manifest["selected_cases"])):
        edited = copy.deepcopy(manifest)
        edited["selected_cases"][index]["confirmation"]["status"] = "CONFIRMED"
        path = reseal(mismatch_run, edited)
        with pytest.raises(
            ss.StressSearchManifestError, match="status does not follow from its recorded evidence"
        ):
            ss.load_run_manifest(path)


# ---------------------------------------------------------------------------
# 3. Outcome rows and error text are sanitised before they are recorded
# ---------------------------------------------------------------------------

_REJECTED = "evaluator outcome rejected: "


def _string_rows(rows: np.ndarray) -> Any:
    return [["not-a-number"] * rows.shape[1] for _ in range(rows.shape[0])]


def _ragged_rows(rows: np.ndarray) -> Any:
    ragged = rows.tolist()
    ragged[1] = ragged[1][:-1]
    return ragged


def _seven_column_rows(rows: np.ndarray) -> Any:
    return rows[:, :7].copy()


def _one_row_short(rows: np.ndarray) -> Any:
    return rows[:-1].copy()


def _one_row_long(rows: np.ndarray) -> Any:
    return np.vstack([rows, rows[-1:]])


_MALFORMED_ROWS: tuple[tuple[str, Callable[[np.ndarray], Any], str], ...] = (
    ("strings", _string_rows, _REJECTED + "truth rows are not numeric"),
    ("ragged", _ragged_rows, _REJECTED + "truth rows are not numeric"),
    (
        "seven_columns",
        _seven_column_rows,
        _REJECTED + f"truth rows have 7 targets, expected {TRUTH_WIDTH}",
    ),
    ("one_row_short", _one_row_short, _REJECTED + "truth rows are not [committed_steps, targets]"),
    ("one_row_long", _one_row_long, _REJECTED + "truth rows are not [committed_steps, targets]"),
)
_MALFORMED_PARAMETERS = [case[1:] for case in _MALFORMED_ROWS]
_MALFORMED_IDS = [case[0] for case in _MALFORMED_ROWS]


class TamperingEvaluator:
    """Fabricated evaluator that corrupts one outcome of a ``FakeEvaluator``.

    Still not a simulator: it only reshapes or replaces fabricated rows, or
    raises, on the chosen call.
    """

    def __init__(
        self,
        *,
        malform: Callable[[np.ndarray], Any] | None = None,
        candidate_call: int | None = None,
        twin: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self.inner = FakeEvaluator()
        self.malform = malform
        self.candidate_call = candidate_call
        self.twin = twin
        self.error = error

    def __call__(
        self, scenario: Scenario, episode_id: str, interrupt_check: Callable[[], None] | None
    ) -> ss.EpisodeOutcome:
        outcome = self.inner(scenario, episode_id, interrupt_check)
        is_twin = not scenario.data["fault_profiles"]
        chosen = self.twin if is_twin else self.inner.candidate_calls == self.candidate_call
        if not chosen:
            return outcome
        if self.error is not None:
            raise self.error
        assert self.malform is not None and outcome.rows is not None
        outcome.rows = self.malform(outcome.rows)
        return outcome


def run_tampered(output_dir: Path, evaluator: TamperingEvaluator) -> ss.RunSummary:
    return ss.run_search(
        spec=make_spec(single_fault_mapping()),
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        method=ss.METHOD_RANDOM,
        seed="test-s1",
        evaluator=evaluator,
        confirm_evaluator=FakeEvaluator(),
        clock=FakeClock(),
    )


def test_fabricated_truth_rows_have_the_width_the_session_expects() -> None:
    environment = ss.build_environment(make_spec(single_fault_mapping()), REPO_ROOT)
    outcome = FakeEvaluator()(environment.twin_scenario, "0" * 64, None)
    assert outcome.rows is not None
    assert outcome.rows.shape == (environment.spec.horizon_steps, TRUTH_WIDTH)


@pytest.mark.parametrize(("malform", "reason"), _MALFORMED_PARAMETERS, ids=_MALFORMED_IDS)
def test_malformed_candidate_rows_are_recorded_as_nonfinite_and_charged(
    tmp_path: Path, malform: Callable[[np.ndarray], Any], reason: str
) -> None:
    run = tmp_path / "run"
    fake = TamperingEvaluator(malform=malform, candidate_call=2)
    summary = run_tampered(run, fake)
    # The run still ends on its simulation budget ...
    assert (summary.end_reason, summary.final) == (ss.END_SIMULATION_BUDGET, True)
    assert summary.simulation_calls_used == 10 and fake.inner.candidate_calls == 9
    records = candidates_of(run)
    assert [record["attempt_index"] for record in records] == list(range(9))
    bad = records[1]
    assert bad["disposition"] == ss.DISPOSITION_NONFINITE == "NONFINITE_OUTPUT"
    assert reason in bad["reason"]
    assert bad["outcome"]["status"] == ss.OUTCOME_NONFINITE
    assert bad["outcome"]["committed_steps"] == 0
    assert bad["metrics"] is None and bad["objective"] is None
    # ... and the malformed episode consumed exactly one simulation call.
    assert bad["simulation_calls_used"] == records[0]["simulation_calls_used"] + 1 == 3
    assert [r["disposition"] for r in records].count(ss.DISPOSITION_EVALUATED) == 8
    manifest = run_manifest_of(run)
    assert manifest["manifest_sha256"] == summary.manifest_sha256
    assert manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_NONFINITE] == 1
    assert 1 not in [case["attempt_index"] for case in manifest["selected_cases"]]
    with pytest.raises(ss.StressSearchAuditError, match="already complete"):
        resume_fake(run)


@pytest.mark.parametrize(("malform", "reason"), _MALFORMED_PARAMETERS, ids=_MALFORMED_IDS)
def test_malformed_healthy_twin_rows_end_the_run_baseline_failed(
    tmp_path: Path, malform: Callable[[np.ndarray], Any], reason: str
) -> None:
    run = tmp_path / "run"
    fake = TamperingEvaluator(malform=malform, twin=True)
    summary = run_tampered(run, fake)
    assert (summary.end_reason, summary.final) == (ss.END_BASELINE_FAILED, True)
    assert summary.simulation_calls_used == 1 and summary.candidate_attempts_used == 0
    assert fake.inner.candidate_calls == 0
    audit = load_audit(run)
    assert audit.records("candidate") == []
    (baseline,) = audit.records("baseline")
    assert baseline["outcome"]["status"] == ss.OUTCOME_NONFINITE
    assert reason in baseline["outcome"]["error"]
    assert baseline["truth_rows_sha256"] is None
    manifest = run_manifest_of(run)
    assert manifest["manifest_sha256"] == summary.manifest_sha256
    assert manifest["selected_cases"] == []


def test_error_text_with_a_lone_surrogate_is_recorded_as_a_simulator_error(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    message = "bad \udc80 path"
    with pytest.raises(UnicodeEncodeError):
        message.encode("utf-8")
    fake = TamperingEvaluator(error=OSError(message), candidate_call=2)
    summary = run_tampered(run, fake)
    assert (summary.end_reason, summary.final) == (ss.END_SIMULATION_BUDGET, True)
    assert summary.simulation_calls_used == 10
    # The audit and the manifest are still strict, loadable UTF-8 JSON.
    records = candidates_of(run)
    bad = records[1]
    assert bad["disposition"] == ss.DISPOSITION_SIMULATOR_ERROR == "SIMULATOR_ERROR"
    assert bad["reason"] == "OSError: bad \\udc80 path"
    assert bad["reason"].isascii()
    assert bad["outcome"]["status"] == ss.OUTCOME_ERROR
    assert bad["outcome"]["error"] == bad["reason"]
    assert bad["simulation_calls_used"] == records[0]["simulation_calls_used"] + 1 == 3
    (run / ss.AUDIT_FILE_NAME).read_bytes().decode("utf-8")
    manifest = run_manifest_of(run)
    assert manifest["budget_consumption"]["dispositions"][ss.DISPOSITION_SIMULATOR_ERROR] == 1


# ---------------------------------------------------------------------------
# 4. resume_search refuses a directory that is not a run directory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("present", [(), (ss.SPEC_FILE_NAME,), (ss.AUDIT_FILE_NAME,)], ids=str)
def test_resume_of_a_non_run_directory_fails_closed_and_creates_nothing(
    tmp_path: Path, present: tuple[str, ...]
) -> None:
    directory = tmp_path / "not-a-run"
    directory.mkdir()
    for name in present:
        content = (
            ss.canonical_bytes(make_spec(small_mapping()).to_mapping())
            if name == ss.SPEC_FILE_NAME
            else b""
        )
        (directory / name).write_bytes(content)
    before = {name: (directory / name).read_bytes() for name in present}
    fake = FakeEvaluator()
    with pytest.raises(
        ss.StressSearchAuditError,
        match="not a run directory: spec.json and audit.jsonl are required",
    ):
        resume_fake(directory, evaluator=fake)
    assert not (directory / ss.LOCK_FILE_NAME).exists()
    assert sorted_tree(directory) == sorted(present)
    assert {name: (directory / name).read_bytes() for name in present} == before
    assert fake.twin_calls == 0 and fake.candidate_calls == 0


def test_resume_of_a_missing_directory_fails_closed_and_creates_nothing(tmp_path: Path) -> None:
    directory = tmp_path / "absent"
    with pytest.raises(ss.StressSearchAuditError, match="run directory does not exist"):
        resume_fake(directory)
    assert not directory.exists()
    assert sorted_tree(tmp_path) == []


# ---------------------------------------------------------------------------
# 5. A number that overflows to infinity is an audit error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", [b"1e999", b"-1e999", b"1E+400"], ids=repr)
def test_audit_load_rejects_an_overflowing_number_as_an_audit_error(
    mismatch_run: Path, token: bytes
) -> None:
    intact = audit_lines(mismatch_run)
    timing_needle = b'"cumulative_wall_seconds":0.0'
    index = next(
        number
        for number, line in enumerate(intact)
        if json.loads(line)["record"]["kind"] == "baseline"
    )
    record_needle = b'"simulation_calls_used":1'
    assert intact[index].count(timing_needle) == 1
    assert intact[index].count(record_needle) == 1
    for needle in (timing_needle, record_needle):
        name = needle.split(b":")[0]
        line = intact[index].replace(needle, name + b":" + token)
        assert line != intact[index]
        path = write_audit(mismatch_run, [*intact[:index], line, *intact[index + 1 :]])
        with pytest.raises(
            ss.StressSearchAuditError, match=f"audit line {index + 1} is not canonical JSON"
        ) as caught:
            ss.AuditLog.load(path)
        assert type(caught.value) is ss.StressSearchAuditError
        # The run manifest beside it no longer binds either.
        with pytest.raises(ss.StressSearchManifestError, match="does not bind to its audit"):
            run_manifest_of(mismatch_run)
    write_audit(mismatch_run, intact)
    assert len(load_audit(mismatch_run).entries) == len(intact)


# ---------------------------------------------------------------------------
# 6. Command line: refusal paths and exit codes (no search is started)
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The CLI script as a module whose repository root is ``tmp_path``."""

    specification = importlib.util.spec_from_file_location(
        "stress_search_cli_under_test", CLI_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a refusal-path test must not start or resume a search")

    monkeypatch.setattr(module, "run_search", refuse)
    monkeypatch.setattr(module, "resume_search", refuse)
    monkeypatch.setattr(module, "collect_runtime_environment", lambda _root: {})
    return module


def first_run_name(cli: ModuleType) -> str:
    spec = ss.load_spec_file(cli.DEFAULT_SPEC)
    return f"{ss.METHODS[0]}-{spec.seeds[0]}"


@pytest.mark.parametrize(
    "present", [(), (ss.SPEC_FILE_NAME,), (ss.AUDIT_FILE_NAME,)], ids=str
)
def test_cli_run_all_resume_refuses_a_per_run_directory_that_is_not_a_started_run(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], present: tuple[str, ...]
) -> None:
    name = first_run_name(cli)
    directory = tmp_path / "out" / "batch" / name
    directory.mkdir(parents=True)
    spec_bytes = ss.canonical_bytes(ss.load_spec_file(cli.DEFAULT_SPEC).to_mapping())
    for file_name in present:
        (directory / file_name).write_bytes(spec_bytes if file_name == ss.SPEC_FILE_NAME else b"")
    before = sorted_tree(tmp_path)
    assert cli.main(["run-all", "--resume", "--output", "out/batch"]) == 2
    error = capsys.readouterr().err
    assert f"error: {name} exists but is not a started run" in error
    assert sorted_tree(tmp_path) == before


def test_cli_run_all_resume_refuses_a_per_run_directory_of_a_different_spec(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    name = first_run_name(cli)
    directory = tmp_path / "out" / "batch" / name
    directory.mkdir(parents=True)
    other = ss.canonical_bytes(make_spec(small_mapping()).to_mapping())
    assert other != ss.canonical_bytes(ss.load_spec_file(cli.DEFAULT_SPEC).to_mapping())
    (directory / ss.SPEC_FILE_NAME).write_bytes(other)
    (directory / ss.AUDIT_FILE_NAME).write_bytes(b"")
    before = sorted_tree(tmp_path)
    assert cli.main(["run-all", "--resume", "--output", "out/batch"]) == 2
    error = capsys.readouterr().err
    assert f"error: {name} was produced by a different spec than the one given" in error
    assert sorted_tree(tmp_path) == before
    assert (directory / ss.SPEC_FILE_NAME).read_bytes() == other


def test_cli_run_all_resume_requires_an_existing_batch_under_out(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run-all", "--resume", "--output", "out/absent"]) == 2
    assert "does not exist" in capsys.readouterr().err
    (tmp_path / "elsewhere").mkdir()
    assert cli.main(["run-all", "--resume", "--output", "elsewhere"]) == 2
    assert "must live under the ignored out/ directory" in capsys.readouterr().err
    assert sorted_tree(tmp_path) == ["elsewhere", "out"]


def replay_arguments(tmp_path: Path) -> list[str]:
    manifest = tmp_path / "out" / "run" / ss.MANIFEST_FILE_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(b"{}")
    return ["replay", "--manifest", "out/run/manifest.json", "--output", "out/replay"]


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        (ss.REPLAY_STATUS_PASS, 0),
        (ss.REPLAY_STATUS_FAIL, 1),
        (ss.REPLAY_STATUS_NO_CASES, 4),
    ],
)
def test_cli_replay_exit_code_follows_the_receipt_status(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    exit_code: int,
) -> None:
    calls: list[dict[str, Any]] = []

    def fabricated_replay(**options: Any) -> dict[str, Any]:
        calls.append(options)
        return {
            "status": status,
            "reproduced_count": 0,
            "case_count": 0,
            "metric_absolute_tolerance": 1.0e-9,
        }

    monkeypatch.setattr(cli, "replay_manifest", fabricated_replay)
    arguments = replay_arguments(tmp_path)
    before = sorted_tree(tmp_path)
    assert (ss.REPLAY_STATUS_PASS, ss.REPLAY_STATUS_FAIL, ss.REPLAY_STATUS_NO_CASES) == (
        "PASS",
        "FAIL",
        "NO_CASES",
    )
    assert cli.main(arguments) == exit_code
    assert f"replay {status}: 0 of 0 selected cases reproduced" in capsys.readouterr().err
    (call,) = calls
    assert call["manifest_path"] == (tmp_path / "out" / "run" / "manifest.json").resolve()
    assert call["output_dir"] == (tmp_path / "out" / "replay").resolve()
    assert call["repo_root"] == tmp_path
    assert sorted_tree(tmp_path) == before


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ENOSPC, "No space left on device"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError("fabricated failure without an errno"),
    ],
    ids=["oserror_errno", "permission_error", "oserror_message_only"],
)
def test_cli_maps_an_os_error_inside_a_handler_to_exit_code_two(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: OSError,
) -> None:
    def failing_replay(**_options: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(cli, "replay_manifest", failing_replay)
    assert cli.main(replay_arguments(tmp_path)) == 2
    printed = capsys.readouterr().err
    assert printed.startswith(f"error: {type(error).__name__}: {error.strerror or error}")
    assert "Traceback" not in printed


def test_cli_maps_a_harness_error_inside_a_handler_to_exit_code_two(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_replay(**_options: Any) -> dict[str, Any]:
        raise ss.StressSearchManifestError("fabricated manifest refusal")

    monkeypatch.setattr(cli, "replay_manifest", failing_replay)
    assert cli.main(replay_arguments(tmp_path)) == 2
    assert capsys.readouterr().err == "error: fabricated manifest refusal\n"
