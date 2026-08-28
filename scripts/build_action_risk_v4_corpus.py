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
    v3_family_split,
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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _resolve_output(path: Path) -> Path:
    resolved = path.resolve()
    if OUT_ROOT not in resolved.parents:
        raise V4CorpusRunError("V4 corpus output must be below repository out/")
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
) -> dict[str, Any]:
    if not 6 <= families <= FAMILY_COUNT or families % 2:
        raise V4CorpusRunError(f"--families must be an even number between 6 and {FAMILY_COUNT}")
    source_identity = _source_identity()
    source_identity_sha256 = _sha(source_identity)
    if source_identity["source_worktree_dirty"] and not (
        allow_dirty_smoke and families < FAMILY_COUNT
    ):
        raise V4CorpusRunError(
            "full V4 corpus generation refuses a dirty source worktree; use a smoke run "
            "with --allow-dirty-smoke while developing"
        )
    output = _resolve_output(output_path)

    bundle = load_forecast_contracts(REPO_ROOT)
    _, model_protocol_sha256 = load_v4_model_protocol(REPO_ROOT)
    feature_manifest = v4_feature_manifest(bundle)
    label_manifest = v4_label_manifest()
    feature_manifest_sha256 = _sha(feature_manifest)
    label_manifest_sha256 = _sha(label_manifest)
    roster = deterministic_family_ids(FAMILY_COUNT)
    split = v3_family_split(roster)
    selected_ids = _select_families(roster, split, families)
    scenarios = {
        family_id: build_family_scenario(bundle.development_scenario, roster.index(family_id))
        for family_id in selected_ids
    }
    scenario_manifest = {
        family_id: scenario.scenario_sha256 for family_id, scenario in scenarios.items()
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
    samples: list[V4RiskSample] = []
    trace_manifest: dict[str, str] = {}
    for index, family_id in enumerate(selected_ids, start=1):
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
        samples.extend(family_samples)
        print(
            f"  family {index}/{len(selected_ids)}: {family_id} ({len(family_samples)} samples)",
            file=sys.stderr,
        )

    sample_rows = [sample.to_mapping() for sample in samples]
    samples_sha256 = _write_jsonl(output / "samples.jsonl", sample_rows)
    try:
        reloaded_samples = load_v4_samples(sample_rows, output, bundle, scenarios)
    except Exception as error:
        raise V4CorpusRunError("V4 serialized corpus verification failed") from error
    if len(reloaded_samples) != len(samples):
        raise V4CorpusRunError("V4 serialized sample count differs")

    trace_manifest_sha256 = _write_json(output / "trace-manifest.json", trace_manifest)
    counts = {
        label: sum(sample.split == label for sample in reloaded_samples)
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
    }
    manifest.update(
        {
            "samples_sha256": samples_sha256,
            "trace_manifest_sha256": trace_manifest_sha256,
            "sample_count": len(reloaded_samples),
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
        "sample_count": len(reloaded_samples),
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
    return {"output": str(output), "results_sha256": results_sha256, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Issue #56 V4 replayable corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=FAMILY_COUNT)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    result = build_v4_corpus(
        args.output,
        families=args.families,
        allow_dirty_smoke=args.allow_dirty_smoke,
    )
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
