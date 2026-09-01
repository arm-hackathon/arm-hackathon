#!/usr/bin/env python3
"""Build a replayable Issue #56 V4 development corpus.

This command only collects development-scenario features and counterfactual
HMC labels.  It does not fit, export, quantize, or integrate a learned model.
Every counterfactual trace is written as a separate content-addressed artifact
and independently parsed and replayed after serialization.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast_issue55_race import (
    EPISODE_STEPS,
    FAMILY_COUNT,
    build_family_scenario,
    deterministic_family_ids,
    family_condition_descriptor,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import v2_decision_steps
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    ISSUE56_V3_SCHEMA_VERSION,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
    ISSUE56_V4_CORPUS_SCHEMA_VERSION,
    V4RiskSample,
    collect_v4_family_samples,
    load_v4_samples,
    v4_label_manifest,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_features import v4_feature_manifest
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
    ISSUE56_V4_MODEL_PROTOCOL_ID,
    V4_MODEL_V3_SPLIT_PROTOCOL,
    V4_MODEL_V6_SPLIT_PROTOCOL,
    V4_MODEL_V8_SPLIT_PROTOCOL,
    family_split_for_protocol,
    load_v4_model_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (REPO_ROOT / "out").resolve()
CORPUS_SOURCE_PATHS = (
    Path("contracts/habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1.json"),
    Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v2.json"),
    Path("src/aeolus/habitat_v2/forecast/projection.py"),
    Path("src/aeolus/habitat_v2/forecast_issue55_race.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v2.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v3.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_corpus.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_features.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_model_protocol.py"),
    Path("src/aeolus/habitat_v2/control_trace.py"),
    Path("src/aeolus/habitat_v2/hmc.py"),
    Path("src/aeolus/habitat_v2/physics.py"),
    Path("scripts/build_action_risk_v4_corpus.py"),
    Path("scripts/verify_action_risk_v4_corpus.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


class V4CorpusRunError(RuntimeError):
    """Raised when a V4 corpus cannot be built without weakening its boundary."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object) -> str:
    try:
        return _sha_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise V4CorpusRunError("V4 corpus value is not canonical JSON") from error


def _git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise V4CorpusRunError("V4 corpus source identity cannot be read from Git") from error
    return result.stdout.strip()


def _source_identity() -> dict[str, Any]:
    files = []
    for relative in CORPUS_SOURCE_PATHS:
        path = REPO_ROOT / relative
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise V4CorpusRunError(f"cannot read V4 source file {relative}") from error
        files.append(
            {
                "relative_path": relative.as_posix(),
                "byte_length": len(raw),
                "sha256": _sha_bytes(raw),
            }
        )
    return {
        "schema_version": "issue56-v4-corpus-source-identity-v1",
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(_git_output("status", "--porcelain")),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "source_files": files,
    }


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=2,
    ).encode("utf-8")
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _write_json_atomic(path: Path, value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=2,
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return _sha_bytes(payload)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _write_jsonl_rows(
    handle: Any,
    rows: list[Mapping[str, Any]],
    digest: "hashlib._Hash",
) -> None:
    """Append canonical rows without retaining the complete corpus in memory."""

    for row in rows:
        payload = canonical_json_bytes(row) + b"\n"
        handle.write(payload)
        digest.update(payload)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4CorpusRunError("V4 resume sample contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise V4CorpusRunError(f"non-finite JSON value {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, V4CorpusRunError) as error:
        raise V4CorpusRunError(f"cannot read V4 resume state: {path}") from error
    if type(value) is not dict:
        raise V4CorpusRunError("V4 resume state must be an object")
    return value


def _read_resume_rows(path: Path) -> list[dict[str, Any]]:
    """Read the complete canonical row prefix from an interrupted build."""

    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V4CorpusRunError("V4 resume samples cannot be read") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            break
        try:
            row = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, V4CorpusRunError) as error:
            raise V4CorpusRunError(
                f"V4 resume samples contain malformed JSON at line {line_number}"
            ) from error
        if type(row) is not dict:
            raise V4CorpusRunError("V4 resume sample row is not an object")
        if canonical_json_bytes(row) + b"\n" != line:
            raise V4CorpusRunError("V4 resume sample row is not canonical JSON")
        rows.append(row)
    return rows


def _resume_family_groups(
    rows: list[dict[str, Any]],
    selected_ids: tuple[str, ...],
    rows_per_family: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Return only complete, ordered family prefixes from a partial corpus."""

    if rows_per_family <= 0:
        raise V4CorpusRunError("V4 resume family row count is invalid")
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        base_sample = row.get("base_sample")
        family_id = base_sample.get("family_id") if isinstance(base_sample, dict) else None
        previous_base = groups[-1][0].get("base_sample") if groups else None
        previous_family_id = (
            previous_base.get("family_id") if isinstance(previous_base, dict) else None
        )
        if groups and previous_family_id == family_id:
            groups[-1].append(row)
        else:
            groups.append([row])

    complete: list[list[dict[str, Any]]] = []
    for index, group in enumerate(groups):
        if index >= len(selected_ids):
            raise V4CorpusRunError("V4 resume samples contain an unknown family suffix")
        family_id = selected_ids[index]
        if any(
            not isinstance(row.get("base_sample"), dict)
            or row["base_sample"].get("family_id") != family_id
            for row in group
        ):
            raise V4CorpusRunError("V4 resume samples are not in roster order")
        if len(group) < rows_per_family:
            if index != len(groups) - 1:
                raise V4CorpusRunError("V4 resume samples contain rows after an incomplete family")
            break
        if len(group) != rows_per_family:
            raise V4CorpusRunError("V4 resume family contains too many rows")
        complete.append(group)
    return complete, sum(len(group) for group in complete)


def _resume_state(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    state = _read_json(path)
    required = {
        "schema_version",
        "status",
        "source_identity",
        "source_identity_sha256",
        "corpus_schema_version",
        "preregistration_id",
        "model_protocol_sha256",
        "family_ids",
        "family_split",
        "scenario_manifest",
        "scenario_manifest_sha256",
        "feature_manifest_sha256",
        "label_manifest_sha256",
        "completed_family_ids",
        "sample_count",
        "sample_counts",
        "samples_prefix_sha256",
        "trace_manifest_sha256",
        "manifest_sha256",
        "results_sha256",
    }
    if set(state) != required:
        raise V4CorpusRunError("V4 resume state fields drift")
    if state["schema_version"] != "issue56-v4-corpus-build-state-v1":
        raise V4CorpusRunError("V4 resume state schema drift")
    if state["status"] != "IN_PROGRESS":
        raise V4CorpusRunError("V4 resume state is already finalized")
    for key in (
        "source_identity",
        "source_identity_sha256",
        "corpus_schema_version",
        "preregistration_id",
        "model_protocol_sha256",
        "family_ids",
        "family_split",
        "scenario_manifest",
        "scenario_manifest_sha256",
        "feature_manifest_sha256",
        "label_manifest_sha256",
    ):
        if state[key] != expected[key]:
            raise V4CorpusRunError(f"V4 resume state identity drift: {key}")
    if state["manifest_sha256"] is not None or state["results_sha256"] is not None:
        raise V4CorpusRunError("V4 resume state finalization fields are inconsistent")
    completed = state["completed_family_ids"]
    if type(completed) is not list or completed != expected["family_ids"][: len(completed)]:
        raise V4CorpusRunError("V4 resume family checkpoint is not an ordered prefix")
    if type(state["sample_count"]) is not int or state["sample_count"] < 0:
        raise V4CorpusRunError("V4 resume sample count is malformed")
    if type(state["sample_counts"]) is not dict:
        raise V4CorpusRunError("V4 resume split counts are malformed")
    if set(state["sample_counts"]) != {"TRAIN", "VALIDATION", "EVALUATION"}:
        raise V4CorpusRunError("V4 resume split count fields drift")
    if any(
        type(value) is not int or value < 0 for value in state["sample_counts"].values()
    ):
        raise V4CorpusRunError("V4 resume split counts are malformed")
    if (
        type(state["samples_prefix_sha256"]) is not str
        or type(state["trace_manifest_sha256"]) is not str
    ):
        raise V4CorpusRunError("V4 resume prefix identities are malformed")
    return state


def _write_resume_state(
    output: Path,
    *,
    expected: Mapping[str, Any],
    completed_family_ids: list[str],
    sample_count: int,
    sample_counts: Mapping[str, int],
    samples_prefix_sha256: str,
    trace_manifest_sha256: str,
) -> None:
    state = {
        "schema_version": "issue56-v4-corpus-build-state-v1",
        "status": "IN_PROGRESS",
        "source_identity": expected["source_identity"],
        "source_identity_sha256": expected["source_identity_sha256"],
        "corpus_schema_version": expected["corpus_schema_version"],
        "preregistration_id": expected["preregistration_id"],
        "model_protocol_sha256": expected["model_protocol_sha256"],
        "family_ids": expected["family_ids"],
        "family_split": expected["family_split"],
        "scenario_manifest": expected["scenario_manifest"],
        "scenario_manifest_sha256": expected["scenario_manifest_sha256"],
        "feature_manifest_sha256": expected["feature_manifest_sha256"],
        "label_manifest_sha256": expected["label_manifest_sha256"],
        "completed_family_ids": completed_family_ids,
        "sample_count": sample_count,
        "sample_counts": dict(sample_counts),
        "samples_prefix_sha256": samples_prefix_sha256,
        "trace_manifest_sha256": trace_manifest_sha256,
        "manifest_sha256": None,
        "results_sha256": None,
    }
    _write_json_atomic(output / ".build-state.json", state)


def _remove_unreferenced_traces(output: Path, expected: set[str]) -> None:
    trace_root = output / "counterfactual-traces"
    for path in trace_root.rglob("*"):
        if path.is_file() and path.relative_to(output).as_posix() not in expected:
            path.unlink()


def _resolve_output(path: Path, *, resume: bool) -> Path:
    resolved = path.resolve()
    if OUT_ROOT not in resolved.parents:
        raise V4CorpusRunError("V4 corpus output must be below repository out/")
    if resume:
        if not resolved.is_dir():
            raise V4CorpusRunError("V4 resume output directory must already exist")
        if not (resolved / "counterfactual-traces").is_dir():
            raise V4CorpusRunError("V4 resume trace directory is missing")
        if (resolved / "results.json").exists():
            raise V4CorpusRunError("V4 resume output is already finalized")
        return resolved
    if resolved.exists():
        raise V4CorpusRunError("V4 corpus output directory must be new and write-once")
    resolved.mkdir(parents=True, exist_ok=False)
    (resolved / "counterfactual-traces").mkdir()
    return resolved


def _select_families(
    roster: tuple[str, ...],
    split: Mapping[str, str],
    count: int,
) -> tuple[str, ...]:
    if count == len(roster):
        return roster
    if count < 2 or count % 2:
        raise V4CorpusRunError("V4 smoke family count must be an even number of at least two")
    groups = tuple(roster[index : index + 2] for index in range(0, len(roster), 2))
    if any(len(group) != 2 or split[group[0]] != split[group[1]] for group in groups):
        raise V4CorpusRunError("V4 family split does not preserve paired sensor variants")
    group_count = count // 2
    if group_count < 3:
        raise V4CorpusRunError(
            "V4 smoke corpus must include at least one paired condition group per split"
        )
    selected: list[tuple[str, ...]] = []
    groups_by_label = {
        label: [group for group in groups if split[group[0]] == label]
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
    }
    for label in ("TRAIN", "VALIDATION", "EVALUATION"):
        selected.append(groups_by_label[label][0])
    remaining = [
        group
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
        for group in groups_by_label[label][1:]
    ]
    selected.extend(remaining[: group_count - len(selected)])
    return tuple(family_id for group in selected for family_id in group)


def _write_trace(output: Path, sample: V4RiskSample) -> str:
    path = output / sample.counterfactual_trace_relative_path
    if path.resolve().parent != (output / "counterfactual-traces").resolve():
        raise V4CorpusRunError("V4 trace path escaped the corpus directory")
    if path.exists():
        if path.read_bytes() != sample.counterfactual_trace_bytes:
            raise V4CorpusRunError("V4 trace path has conflicting content")
        return _sha_bytes(sample.counterfactual_trace_bytes)
    path.write_bytes(sample.counterfactual_trace_bytes)
    return _sha_bytes(sample.counterfactual_trace_bytes)


def _write_trace_artifact(output: Path, relative_path: str, content: bytes) -> str:
    path = output / relative_path
    if path.resolve().parent != (output / "counterfactual-traces").resolve():
        raise V4CorpusRunError("V4 trace path escaped the corpus directory")
    if path.exists():
        if path.read_bytes() != content:
            raise V4CorpusRunError("V4 trace path has conflicting content")
        return _sha_bytes(content)
    path.write_bytes(content)
    return _sha_bytes(content)


def build_v4_corpus(
    output_path: Path,
    *,
    families: int,
    allow_dirty_smoke: bool,
    resume: bool = False,
    split_protocol: str = V4_MODEL_V3_SPLIT_PROTOCOL,
) -> dict[str, Any]:
    if not 6 <= families <= FAMILY_COUNT or families % 2:
        raise V4CorpusRunError(f"--families must be an even number between 6 and {FAMILY_COUNT}")
    if split_protocol not in (
        V4_MODEL_V3_SPLIT_PROTOCOL,
        V4_MODEL_V6_SPLIT_PROTOCOL,
        V4_MODEL_V8_SPLIT_PROTOCOL,
    ):
        raise V4CorpusRunError(f"unknown --split-protocol {split_protocol!r}")
    source_identity = _source_identity()
    source_identity_sha256 = _sha(source_identity)
    if source_identity["source_worktree_dirty"] and not (
        allow_dirty_smoke and families < FAMILY_COUNT
    ):
        raise V4CorpusRunError(
            "full V4 corpus generation refuses a dirty source worktree; use a smoke run "
            "with --allow-dirty-smoke while developing"
        )
    output = _resolve_output(output_path, resume=resume)

    bundle = load_forecast_contracts(REPO_ROOT)
    _, model_protocol_sha256 = load_v4_model_protocol(REPO_ROOT)
    feature_manifest = v4_feature_manifest(bundle)
    label_manifest = v4_label_manifest()
    feature_manifest_sha256 = _sha(feature_manifest)
    label_manifest_sha256 = _sha(label_manifest)
    roster = deterministic_family_ids(FAMILY_COUNT)
    split = family_split_for_protocol(split_protocol, roster)
    selected_ids = _select_families(roster, split, families)
    scenarios = {
        family_id: build_family_scenario(bundle.development_scenario, roster.index(family_id))
        for family_id in selected_ids
    }
    scenario_manifest = {
        family_id: scenario.scenario_sha256 for family_id, scenario in scenarios.items()
    }
    expected_resume_identity = {
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
        "corpus_schema_version": ISSUE56_V4_CORPUS_SCHEMA_VERSION,
        "preregistration_id": ISSUE56_V4_MODEL_PROTOCOL_ID,
        "model_protocol_sha256": model_protocol_sha256,
        "family_ids": list(selected_ids),
        "family_split": {family_id: split[family_id] for family_id in selected_ids},
        "scenario_manifest": scenario_manifest,
        "scenario_manifest_sha256": _sha(scenario_manifest),
        "feature_manifest_sha256": feature_manifest_sha256,
        "label_manifest_sha256": label_manifest_sha256,
    }
    manifest = {
        "schema_version": f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.manifest",
        "v3_sample_schema_version": ISSUE56_V3_SCHEMA_VERSION,
        "preregistration_id": ISSUE56_V4_MODEL_PROTOCOL_ID,
        "diagnostic_preregistration_id": (
            "habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1"
        ),
        "model_protocol_sha256": model_protocol_sha256,
        "family_ids": list(selected_ids),
        "family_roster": [family_condition_descriptor(index) for index in range(FAMILY_COUNT)],
        "family_split": {family_id: split[family_id] for family_id in selected_ids},
        "decision_steps": list(v2_decision_steps()),
        "episode_steps": EPISODE_STEPS,
        "counterfactual_trace_bytes_present": True,
        "hold_trace_bytes_present": True,
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
        "hmc_binding_sha256": bundle.binding_sha256,
        "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
        "action_catalogue_sha256": bundle.action_catalogue_sha256,
        "alarm_manifest_sha256": bundle.alarm_manifest_sha256,
        "scenario_manifest": scenario_manifest,
        "scenario_manifest_sha256": _sha(scenario_manifest),
        "feature_manifest": feature_manifest,
        "feature_manifest_sha256": feature_manifest_sha256,
        "label_manifest": label_manifest,
        "label_manifest_sha256": label_manifest_sha256,
        "smoke_only": families < FAMILY_COUNT,
    }
    trace_manifest: dict[str, str] = {}
    samples_path = output / "samples.jsonl"
    samples_digest = hashlib.sha256()
    sample_count = 0
    counts = {label: 0 for label in ("TRAIN", "VALIDATION", "EVALUATION")}
    completed_groups: list[list[dict[str, Any]]] = []
    if resume:
        state = _resume_state(output / ".build-state.json", expected_resume_identity)
        rows = _read_resume_rows(samples_path)
        available_groups, _ = _resume_family_groups(
            rows,
            selected_ids,
            len(v2_decision_steps()) * len(bundle.actions),
        )
        checkpoint_count = len(state["completed_family_ids"])
        if len(available_groups) < checkpoint_count:
            raise V4CorpusRunError("V4 resume samples are shorter than the checkpoint")
        completed_groups = available_groups[:checkpoint_count]
        retained_rows = [row for group in completed_groups for row in group]
        retained_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in retained_rows)
        if _sha_bytes(retained_payload) != state["samples_prefix_sha256"]:
            raise V4CorpusRunError("V4 resume sample prefix identity drift")
        temporary_samples_path = samples_path.with_name(f".{samples_path.name}.tmp")
        temporary_samples_path.write_bytes(retained_payload)
        temporary_samples_path.replace(samples_path)
        for row in retained_rows:
            samples_digest.update(canonical_json_bytes(row) + b"\n")
        for group in completed_groups:
            family_id = group[0]["base_sample"]["family_id"]
            try:
                reloaded_family = load_v4_samples(
                    group,
                    output,
                    bundle,
                    {family_id: scenarios[family_id]},
                )
            except Exception as error:
                raise V4CorpusRunError(
                    f"V4 resume verification failed for family {family_id}"
                ) from error
            sample_count += len(reloaded_family)
            counts[split[family_id]] += len(reloaded_family)
            for sample in reloaded_family:
                trace_manifest[sample.counterfactual_trace_relative_path] = (
                    sample.counterfactual_trace_sha256
                )
                trace_manifest[sample.hold_trace_relative_path] = sample.hold_trace_sha256
        trace_manifest_sha256 = _write_json_atomic(
            output / "trace-manifest.json", trace_manifest
        )
        if trace_manifest_sha256 != state["trace_manifest_sha256"]:
            raise V4CorpusRunError("V4 resume trace manifest identity drift")
        _remove_unreferenced_traces(output, set(trace_manifest))
        if state["sample_count"] != sample_count or state["sample_counts"] != counts:
            raise V4CorpusRunError("V4 resume sample counts drift")
        if state["completed_family_ids"] != list(selected_ids[:checkpoint_count]):
            raise V4CorpusRunError("V4 resume family checkpoint identity drift")
        if len(retained_rows) != sample_count:
            raise V4CorpusRunError("V4 resume sample count differs after verification")
    else:
        trace_manifest_sha256 = _write_json_atomic(output / "trace-manifest.json", {})
        _write_resume_state(
            output,
            expected=expected_resume_identity,
            completed_family_ids=[],
            sample_count=0,
            sample_counts=counts,
            samples_prefix_sha256=samples_digest.hexdigest(),
            trace_manifest_sha256=trace_manifest_sha256,
        )

    with samples_path.open("ab" if resume else "wb") as samples_handle:
        for index, family_id in enumerate(
            selected_ids[len(completed_groups) :],
            start=len(completed_groups) + 1,
        ):
            family_samples = collect_v4_family_samples(
                bundle,
                scenarios[family_id],
                family_id,
                split=split[family_id],
            )
            for sample in family_samples:
                trace_digest = _write_trace_artifact(
                    output,
                    sample.counterfactual_trace_relative_path,
                    sample.counterfactual_trace_bytes,
                )
                if trace_digest != sample.counterfactual_trace_sha256:
                    raise V4CorpusRunError("V4 written trace digest differs from sample")
                trace_manifest[sample.counterfactual_trace_relative_path] = trace_digest
                hold_digest = _write_trace_artifact(
                    output,
                    sample.hold_trace_relative_path,
                    sample.hold_trace_bytes,
                )
                if hold_digest != sample.hold_trace_sha256:
                    raise V4CorpusRunError("V4 written hold trace digest differs from sample")
                trace_manifest[sample.hold_trace_relative_path] = hold_digest

            family_rows = [sample.to_mapping() for sample in family_samples]
            _write_jsonl_rows(samples_handle, family_rows, samples_digest)
            try:
                reloaded_family = load_v4_samples(
                    family_rows,
                    output,
                    bundle,
                    {family_id: scenarios[family_id]},
                )
            except Exception as error:
                raise V4CorpusRunError(
                    f"V4 serialized corpus verification failed for family {family_id}"
                ) from error
            if len(reloaded_family) != len(family_samples):
                raise V4CorpusRunError("V4 serialized family sample count differs")
            sample_count += len(reloaded_family)
            counts[split[family_id]] += len(reloaded_family)
            samples_handle.flush()
            os.fsync(samples_handle.fileno())
            completed_groups.append(family_rows)
            trace_manifest_sha256 = _write_json_atomic(
                output / "trace-manifest.json", trace_manifest
            )
            _write_resume_state(
                output,
                expected=expected_resume_identity,
                completed_family_ids=list(selected_ids[:index]),
                sample_count=sample_count,
                sample_counts=counts,
                samples_prefix_sha256=samples_digest.hexdigest(),
                trace_manifest_sha256=trace_manifest_sha256,
            )
            print(
                f"  family {index}/{len(selected_ids)}: {family_id} ({len(family_samples)} samples)",
                file=sys.stderr,
            )

    samples_sha256 = samples_digest.hexdigest()

    trace_manifest_sha256 = _write_json(output / "trace-manifest.json", trace_manifest)
    manifest.update(
        {
            "samples_sha256": samples_sha256,
            "trace_manifest_sha256": trace_manifest_sha256,
            "sample_count": sample_count,
            "sample_counts": counts,
            "trace_count": len(trace_manifest),
        }
    )
    manifest_sha256 = _write_json(output / "manifest.json", manifest)
    result = {
        "schema_version": f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.result",
        "manifest_sha256": manifest_sha256,
        "samples_sha256": samples_sha256,
        "trace_manifest_sha256": trace_manifest_sha256,
        "family_count": len(selected_ids),
        "sample_count": sample_count,
        "sample_counts": counts,
        "trace_count": len(trace_manifest),
        "independent_trace_replay_verified": True,
        "counterfactual_trace_bytes_present": True,
        "hold_trace_bytes_present": True,
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
        "model_protocol_sha256": model_protocol_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "label_manifest_sha256": label_manifest_sha256,
        "scenario_manifest_sha256": _sha(scenario_manifest),
        "smoke_only": families < FAMILY_COUNT,
        "status": "SMOKE_PATH_ONLY" if families < FAMILY_COUNT else "DEVELOPMENT_CORPUS",
    }
    results_sha256 = _write_json(output / "results.json", result)
    (output / ".build-state.json").unlink(missing_ok=True)
    return {"output": str(output), "results_sha256": results_sha256, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Issue #56 V4 replayable corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=FAMILY_COUNT)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a partial write-once corpus directory after family verification",
    )
    parser.add_argument(
        "--split-protocol",
        default=V4_MODEL_V3_SPLIT_PROTOCOL,
        choices=(
            V4_MODEL_V3_SPLIT_PROTOCOL,
            V4_MODEL_V6_SPLIT_PROTOCOL,
            V4_MODEL_V8_SPLIT_PROTOCOL,
        ),
        help="preregistered family split used to label corpus samples",
    )
    args = parser.parse_args()
    result = build_v4_corpus(
        args.output,
        families=args.families,
        allow_dirty_smoke=args.allow_dirty_smoke,
        resume=args.resume,
        split_protocol=args.split_protocol,
    )
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
