#!/usr/bin/env python3
"""Independently verify a serialized Issue #56 V4 development corpus."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
from pathlib import Path
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
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
    ISSUE56_V4_CORPUS_SCHEMA_VERSION,
    V4_CORPUS_TRACE_DIRECTORY,
    Issue56V4CorpusError,
    load_v4_samples,
    v4_label_manifest,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_features import v4_feature_manifest
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
    ISSUE56_V4_MODEL_PROTOCOL_ID,
    V4_MODEL_V3_SPLIT_PROTOCOL,
    V4_MODEL_V6_SPLIT_PROTOCOL,
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


class V4CorpusVerificationError(RuntimeError):
    """Raised when a serialized V4 corpus fails independent verification."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: object) -> str:
    return _sha_bytes(canonical_json_bytes(value))


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V4CorpusVerificationError(f"cannot strictly read {path}") from error


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise V4CorpusVerificationError(f"cannot read V4 samples: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(
                line,
                object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs),
                parse_constant=lambda value: _reject_json_constant(value),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise V4CorpusVerificationError(
                f"invalid V4 sample JSON at {path}:{line_number}"
            ) from error
        if type(row) is not dict:
            raise V4CorpusVerificationError(f"V4 sample row is not an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise V4CorpusVerificationError("V4 samples.jsonl is empty")
    return rows


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _resolve_corpus(path: Path) -> Path:
    resolved = path.resolve()
    if OUT_ROOT not in resolved.parents:
        raise V4CorpusVerificationError("V4 corpus must be below repository out/")
    if not resolved.is_dir():
        raise V4CorpusVerificationError("V4 corpus must be an existing directory")
    return resolved


def _verify_source_identity(identity: Mapping[str, Any]) -> None:
    _require_keys(
        identity,
        {"schema_version", "source_commit", "source_worktree_dirty", "runtime", "source_files"},
        "V4 source identity",
    )
    if identity["schema_version"] != "issue56-v4-corpus-source-identity-v1":
        raise V4CorpusVerificationError("V4 source identity schema drift")
    source_commit = identity["source_commit"]
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or source_commit != source_commit.lower()
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise V4CorpusVerificationError("V4 source commit identity is malformed")
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise V4CorpusVerificationError("V4 current source commit cannot be read") from error
    if source_commit != current_commit:
        raise V4CorpusVerificationError("V4 source commit identity is stale")
    if type(identity["source_worktree_dirty"]) is not bool:
        raise V4CorpusVerificationError("V4 source dirty flag is malformed")
    if type(identity["runtime"]) is not dict:
        raise V4CorpusVerificationError("V4 source runtime identity is malformed")
    source_files = identity["source_files"]
    if type(source_files) is not list or not source_files:
        raise V4CorpusVerificationError("V4 source file identity is missing")
    expected_paths = {path.as_posix() for path in CORPUS_SOURCE_PATHS}
    seen: set[str] = set()
    for entry in source_files:
        _require_keys(entry, {"relative_path", "byte_length", "sha256"}, "V4 source file")
        relative = entry["relative_path"]
        if type(relative) is not str or not relative or relative in seen:
            raise V4CorpusVerificationError("V4 source file path identity is invalid")
        seen.add(relative)
        source_path = (REPO_ROOT / relative).resolve()
        if REPO_ROOT.resolve() not in source_path.parents:
            raise V4CorpusVerificationError("V4 source file path escaped the repository")
        try:
            raw = source_path.read_bytes()
        except OSError as error:
            raise V4CorpusVerificationError(f"V4 source file is missing: {relative}") from error
        if type(entry["byte_length"]) is not int or entry["byte_length"] != len(raw):
            raise V4CorpusVerificationError(f"V4 source file length drift: {relative}")
        if type(entry["sha256"]) is not str or _sha_bytes(raw) != entry["sha256"]:
            raise V4CorpusVerificationError(f"V4 source file digest drift: {relative}")
    if seen != expected_paths:
        raise V4CorpusVerificationError("V4 source file coverage drift")


def _require_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(mapping) is not dict or set(mapping) != expected:
        raise V4CorpusVerificationError(f"{label} fields drift")


def verify_v4_corpus(
    corpus_path: Path, *, split_protocol: str = V4_MODEL_V3_SPLIT_PROTOCOL
) -> dict[str, Any]:
    if split_protocol not in (V4_MODEL_V3_SPLIT_PROTOCOL, V4_MODEL_V6_SPLIT_PROTOCOL):
        raise V4CorpusVerificationError(f"unknown --split-protocol {split_protocol!r}")
    corpus = _resolve_corpus(corpus_path)
    manifest = _strict_json(corpus / "manifest.json")
    trace_manifest = _strict_json(corpus / "trace-manifest.json")
    result = _strict_json(corpus / "results.json")
    rows = _strict_jsonl(corpus / "samples.jsonl")
    _require_keys(
        manifest,
        {
            "schema_version",
            "v3_sample_schema_version",
            "preregistration_id",
            "family_ids",
            "family_roster",
            "family_split",
            "decision_steps",
            "episode_steps",
            "counterfactual_trace_bytes_present",
            "hold_trace_bytes_present",
            "source_identity",
            "source_identity_sha256",
            "diagnostic_preregistration_id",
            "model_protocol_sha256",
            "hmc_binding_sha256",
            "hmc_contract_sha256",
            "action_catalogue_sha256",
            "alarm_manifest_sha256",
            "scenario_manifest",
            "scenario_manifest_sha256",
            "feature_manifest",
            "feature_manifest_sha256",
            "label_manifest",
            "label_manifest_sha256",
            "smoke_only",
            "samples_sha256",
            "trace_manifest_sha256",
            "sample_count",
            "sample_counts",
            "trace_count",
        },
        "V4 manifest",
    )
    if manifest["schema_version"] != f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.manifest":
        raise V4CorpusVerificationError("V4 manifest schema drift")
    if manifest["v3_sample_schema_version"] != "aeolus_habitat_v2_risk_issue_56_v3_v2":
        raise V4CorpusVerificationError("V4 base sample schema drift")
    if manifest["preregistration_id"] != ISSUE56_V4_MODEL_PROTOCOL_ID:
        raise V4CorpusVerificationError("V4 preregistration identity drift")
    if manifest["diagnostic_preregistration_id"] != (
        "habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1"
    ):
        raise V4CorpusVerificationError("V4 diagnostic preregistration identity drift")
    if manifest["counterfactual_trace_bytes_present"] is not True:
        raise V4CorpusVerificationError("V4 corpus does not retain trace bytes")
    if manifest["hold_trace_bytes_present"] is not True:
        raise V4CorpusVerificationError("V4 corpus does not retain hold trace bytes")
    if manifest["smoke_only"] is not (len(manifest["family_ids"]) < FAMILY_COUNT):
        raise V4CorpusVerificationError("V4 manifest smoke classification drift")
    _verify_source_identity(manifest["source_identity"])
    if manifest["source_identity_sha256"] != _sha_json(manifest["source_identity"]):
        raise V4CorpusVerificationError("V4 source identity digest drift")
    if manifest["decision_steps"] != list(v2_decision_steps()):
        raise V4CorpusVerificationError("V4 decision-step contract drift")
    if manifest["episode_steps"] != EPISODE_STEPS:
        raise V4CorpusVerificationError("V4 episode-step contract drift")

    roster = deterministic_family_ids(FAMILY_COUNT)
    if manifest["family_roster"] != [
        family_condition_descriptor(index) for index in range(FAMILY_COUNT)
    ]:
        raise V4CorpusVerificationError("V4 family roster descriptor drift")
    if type(manifest["family_ids"]) is not list:
        raise V4CorpusVerificationError("V4 family list is malformed")
    family_ids = tuple(manifest["family_ids"])
    if not family_ids or len(set(family_ids)) != len(family_ids):
        raise V4CorpusVerificationError("V4 family list is not unique")
    if any(family_id not in roster for family_id in family_ids):
        raise V4CorpusVerificationError("V4 corpus contains an unknown family")
    selected = set(family_ids)
    for index in range(0, len(roster), 2):
        pair = set(roster[index : index + 2])
        if len(selected & pair) not in {0, 2}:
            raise V4CorpusVerificationError("V4 corpus splits a paired sensor group")
    expected_split = family_split_for_protocol(split_protocol, roster)
    if manifest["family_split"] != {
        family_id: expected_split[family_id] for family_id in family_ids
    }:
        raise V4CorpusVerificationError("V4 family split drift")
    if type(manifest["scenario_manifest"]) is not dict:
        raise V4CorpusVerificationError("V4 scenario manifest is malformed")
    if manifest["scenario_manifest"].keys() != set(family_ids):
        raise V4CorpusVerificationError("V4 scenario manifest coverage drift")
    if manifest["sample_count"] != len(rows):
        raise V4CorpusVerificationError("V4 sample count drift")

    bundle = load_forecast_contracts(REPO_ROOT)
    _, model_protocol_sha256 = load_v4_model_protocol(REPO_ROOT)
    if manifest["model_protocol_sha256"] != model_protocol_sha256:
        raise V4CorpusVerificationError("V4 model protocol digest drift")
    expected_feature_manifest = v4_feature_manifest(bundle)
    if (
        manifest["feature_manifest"] != expected_feature_manifest
        or manifest["feature_manifest_sha256"] != _sha_json(expected_feature_manifest)
    ):
        raise V4CorpusVerificationError("V4 feature manifest drift")
    expected_label_manifest = v4_label_manifest()
    if (
        manifest["label_manifest"] != expected_label_manifest
        or manifest["label_manifest_sha256"] != _sha_json(expected_label_manifest)
    ):
        raise V4CorpusVerificationError("V4 label manifest drift")
    if manifest["hmc_binding_sha256"] != bundle.binding_sha256:
        raise V4CorpusVerificationError("V4 HMC binding drift")
    if manifest["hmc_contract_sha256"] != bundle.hmc_contract.hmc_contract_sha256:
        raise V4CorpusVerificationError("V4 HMC contract drift")
    if manifest["action_catalogue_sha256"] != bundle.action_catalogue_sha256:
        raise V4CorpusVerificationError("V4 action catalogue drift")
    if manifest["alarm_manifest_sha256"] != bundle.alarm_manifest_sha256:
        raise V4CorpusVerificationError("V4 alarm manifest drift")

    scenarios = {
        family_id: build_family_scenario(bundle.development_scenario, roster.index(family_id))
        for family_id in family_ids
    }
    if manifest["scenario_manifest"] != {
        family_id: scenario.scenario_sha256 for family_id, scenario in scenarios.items()
    }:
        raise V4CorpusVerificationError("V4 scenario identity drift")
    if manifest["scenario_manifest_sha256"] != _sha_json(manifest["scenario_manifest"]):
        raise V4CorpusVerificationError("V4 scenario manifest digest drift")

    sample_bytes = (corpus / "samples.jsonl").read_bytes()
    if _sha_bytes(sample_bytes) != manifest["samples_sha256"]:
        raise V4CorpusVerificationError("V4 samples digest drift")
    expected_trace_root = corpus / V4_CORPUS_TRACE_DIRECTORY
    if not expected_trace_root.is_dir():
        raise V4CorpusVerificationError("V4 trace directory is missing")
    if type(trace_manifest) is not dict:
        raise V4CorpusVerificationError("V4 trace manifest is malformed")
    if _sha_bytes((corpus / "trace-manifest.json").read_bytes()) != manifest[
        "trace_manifest_sha256"
    ]:
        raise V4CorpusVerificationError("V4 trace manifest digest drift")

    rows_by_family: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        base_sample = row.get("base_sample")
        family_id = base_sample.get("family_id") if type(base_sample) is dict else None
        rows_by_family.setdefault(family_id, []).append(row)
    if set(rows_by_family) != set(family_ids):
        raise V4CorpusVerificationError("V4 sample family coverage is incomplete")

    expected_trace_manifest: dict[str, str] = {}
    observed_keys: set[tuple[str, int, str]] = set()
    observed_counts = {split: 0 for split in ("TRAIN", "VALIDATION", "EVALUATION")}
    verified_sample_count = 0
    for family_id in family_ids:
        try:
            family_samples = load_v4_samples(
                rows_by_family[family_id],
                corpus,
                bundle,
                {family_id: scenarios[family_id]},
            )
        except Issue56V4CorpusError as error:
            raise V4CorpusVerificationError(
                f"V4 serialized trace replay failed for family {family_id}"
            ) from error
        verified_sample_count += len(family_samples)
        for sample in family_samples:
            key = (sample.family_id, sample.decision_step, sample.action_id)
            if key in observed_keys:
                raise V4CorpusVerificationError("V4 samples contain duplicate decision/action rows")
            observed_keys.add(key)
            if sample.split != manifest["family_split"][sample.family_id]:
                raise V4CorpusVerificationError("V4 sample split identity drift")
            observed_counts[sample.split] += 1
            expected_trace_manifest[sample.counterfactual_trace_relative_path] = (
                sample.counterfactual_trace_sha256
            )
            expected_trace_manifest[sample.hold_trace_relative_path] = sample.hold_trace_sha256
        del family_samples
        gc.collect()
    expected_keys = {
        (family_id, decision_step, action.action_id)
        for family_id in family_ids
        for decision_step in v2_decision_steps()
        for action in bundle.actions
    }
    if observed_keys != expected_keys:
        raise V4CorpusVerificationError("V4 sample Cartesian coverage is incomplete")
    if trace_manifest != expected_trace_manifest:
        raise V4CorpusVerificationError("V4 trace manifest coverage or identity drift")
    actual_trace_files = {
        path.relative_to(corpus).as_posix()
        for path in expected_trace_root.rglob("*")
        if path.is_file()
    }
    if actual_trace_files != set(expected_trace_manifest):
        raise V4CorpusVerificationError("V4 trace directory contains unexpected files")
    if manifest["trace_count"] != len(expected_trace_manifest):
        raise V4CorpusVerificationError("V4 trace count drift")
    if manifest["sample_counts"] != observed_counts:
        raise V4CorpusVerificationError("V4 split sample counts drift")

    _require_keys(
        result,
        {
            "schema_version",
            "manifest_sha256",
            "samples_sha256",
            "trace_manifest_sha256",
            "family_count",
            "sample_count",
            "sample_counts",
            "trace_count",
            "independent_trace_replay_verified",
            "counterfactual_trace_bytes_present",
            "hold_trace_bytes_present",
            "source_identity",
            "source_identity_sha256",
            "model_protocol_sha256",
            "feature_manifest_sha256",
            "label_manifest_sha256",
            "scenario_manifest_sha256",
            "smoke_only",
            "status",
        },
        "V4 result",
    )
    if result["schema_version"] != f"{ISSUE56_V4_CORPUS_SCHEMA_VERSION}.result":
        raise V4CorpusVerificationError("V4 result schema drift")
    if result["manifest_sha256"] != _sha_bytes((corpus / "manifest.json").read_bytes()):
        raise V4CorpusVerificationError("V4 result manifest digest drift")
    if result["samples_sha256"] != manifest["samples_sha256"]:
        raise V4CorpusVerificationError("V4 result sample digest drift")
    if result["trace_manifest_sha256"] != manifest["trace_manifest_sha256"]:
        raise V4CorpusVerificationError("V4 result trace manifest digest drift")
    if result["source_identity"] != manifest["source_identity"]:
        raise V4CorpusVerificationError("V4 result source identity drift")
    if result["family_count"] != len(family_ids) or result["sample_count"] != verified_sample_count:
        raise V4CorpusVerificationError("V4 result counts drift")
    if result["sample_counts"] != observed_counts or result["trace_count"] != len(expected_trace_manifest):
        raise V4CorpusVerificationError("V4 result split or trace counts drift")
    if result["independent_trace_replay_verified"] is not True:
        raise V4CorpusVerificationError("V4 corpus does not report replay verification")
    if result["counterfactual_trace_bytes_present"] is not True:
        raise V4CorpusVerificationError("V4 result does not report retained trace bytes")
    if result["hold_trace_bytes_present"] is not True:
        raise V4CorpusVerificationError("V4 result does not report retained hold traces")
    if (
        result["source_identity_sha256"] != manifest["source_identity_sha256"]
        or result["model_protocol_sha256"] != manifest["model_protocol_sha256"]
        or result["feature_manifest_sha256"] != manifest["feature_manifest_sha256"]
        or result["label_manifest_sha256"] != manifest["label_manifest_sha256"]
        or result["scenario_manifest_sha256"] != manifest["scenario_manifest_sha256"]
    ):
        raise V4CorpusVerificationError("V4 result provenance identity drift")
    expected_smoke = len(family_ids) < FAMILY_COUNT
    if result["smoke_only"] is not expected_smoke:
        raise V4CorpusVerificationError("V4 result smoke classification drift")
    expected_status = "SMOKE_PATH_ONLY" if expected_smoke else "DEVELOPMENT_CORPUS"
    if result["status"] != expected_status:
        raise V4CorpusVerificationError("V4 result status drift")

    return {
        "corpus": str(corpus),
        "split_protocol": split_protocol,
        "family_count": len(family_ids),
        "sample_count": verified_sample_count,
        "trace_count": len(expected_trace_manifest),
        "sample_counts": observed_counts,
        "strict_trace_replay_verified": True,
        "manifest_sha256": result["manifest_sha256"],
        "samples_sha256": result["samples_sha256"],
        "trace_manifest_sha256": result["trace_manifest_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Issue #56 V4 replayable corpus")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--split-protocol",
        default=V4_MODEL_V3_SPLIT_PROTOCOL,
        choices=(V4_MODEL_V3_SPLIT_PROTOCOL, V4_MODEL_V6_SPLIT_PROTOCOL),
        help="preregistered family split the corpus must match",
    )
    args = parser.parse_args()
    result = verify_v4_corpus(args.corpus, split_protocol=args.split_protocol)
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
