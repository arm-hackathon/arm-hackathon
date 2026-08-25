#!/usr/bin/env python3
"""Run the preregistered Issue #56 V3 development study.

V3 is a forecast-only adviser.  Every comparative arm uses the same 13-decision
window, and every proposal is submitted to the frozen HMC for validation,
arbitration, execution, and strict replay.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model
from aeolus.habitat_v2.forecast_issue55_race import (
    EPISODE_STEPS,
    FAMILY_COUNT,
    build_family_scenario,
    deterministic_family_ids,
    family_condition_descriptor,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    ISSUE56_V3_SCHEMA_VERSION,
    PREREGISTRATION_ID,
    V3RiskModel,
    V3_ARMS,
    V3EpisodeRecord,
    calibration_metrics_v3,
    collect_v3_family_samples,
    load_v3_samples,
    run_v3_episode,
    v2_decision_steps,
    v3_family_split,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = (
    REPO_ROOT / "contracts" / "habitat_v2_forecast_issue_56_v3_preregistration_v2.json"
)
POINT_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
POINT_ARTIFACT_SHA256 = (
    "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
)
MIN_FAMILY_COUNT = 16
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 560057

SOURCE_PATHS = (
    Path("contracts/habitat_v2_forecast_issue_56_v3_preregistration_v2.json"),
    Path("src/aeolus/habitat_v2/forecast/projection.py"),
    Path("src/aeolus/habitat_v2/forecast/live_mlp_demo.py"),
    Path("src/aeolus/habitat_v2/forecast_issue55_race.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v2.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v3.py"),
    Path("src/aeolus/habitat_v2/hmc.py"),
    Path("src/aeolus/habitat_v2/physics.py"),
    Path("src/aeolus/habitat_v2/safety.py"),
    Path("scripts/run_action_risk_v3.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise RuntimeError("V3 source identity cannot be read from Git") from error
    return result.stdout.strip()


def _source_identity() -> dict[str, Any]:
    files = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        raw = path.read_bytes()
        files.append(
            {
                "relative_path": relative.as_posix(),
                "byte_length": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
    return {
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(_git_output("status", "--porcelain")),
        "command_line": list(sys.argv),
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
    return _sha256_bytes(payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _verify_preregistration() -> tuple[str, dict[str, Any]]:
    raw = PREREGISTRATION_PATH.read_bytes().replace(b"\r\n", b"\n")
    value = json.loads(raw)
    if value.get("preregistration_id") != PREREGISTRATION_ID:
        raise RuntimeError("Issue #56 V3 preregistration identity drifted")
    return _sha256_bytes(raw), value


def _validate_output_path(path: Path, *, allow_dirty_smoke: bool, families: int) -> Path:
    output = path.resolve()
    out_root = (REPO_ROOT / "out").resolve()
    if out_root not in output.parents:
        raise RuntimeError("V3 output must be below repository out/")
    if output.exists():
        raise RuntimeError("V3 output directory must be new and write-once")
    identity = _source_identity()
    if identity["source_worktree_dirty"] and not (allow_dirty_smoke and families < FAMILY_COUNT):
        raise RuntimeError(
            "comparative V3 evaluation refuses a dirty source worktree; use a smoke run "
            "with --allow-dirty-smoke while developing"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    (output / "traces").mkdir()
    return output


def _select_families(roster: tuple[str, ...], split: Mapping[str, str], count: int) -> tuple[str, ...]:
    if count == len(roster):
        return roster
    if count % 2:
        raise RuntimeError("V3 smoke family count must preserve paired sensor variants")
    groups = tuple(roster[index : index + 2] for index in range(0, len(roster), 2))
    if any(len(group) != 2 or split[group[0]] != split[group[1]] for group in groups):
        raise RuntimeError("V3 family split does not preserve paired sensor variants")
    group_count = count // 2
    target_counts = {
        "TRAIN": max(1, int(round(group_count * 0.60))),
        "VALIDATION": max(1, int(round(group_count * 0.20))),
        "EVALUATION": max(1, int(round(group_count * 0.20))),
    }
    while sum(target_counts.values()) > group_count:
        label = max(target_counts, key=lambda item: (target_counts[item], item == "TRAIN"))
        if target_counts[label] > 1:
            target_counts[label] -= 1
        else:
            break
    while sum(target_counts.values()) < group_count:
        label = min(target_counts, key=lambda item: (target_counts[item], item))
        target_counts[label] += 1
    selected_by_label = {
        label: [group for group in groups if split[group[0]] == label]
        for label in target_counts
    }
    return tuple(
        family_id
        for label in ("TRAIN", "VALIDATION", "EVALUATION")
        for group in selected_by_label[label][: target_counts[label]]
        for family_id in group
    )


def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    result = np.empty((repetitions, count), dtype=np.int64)
    for replicate in range(repetitions):
        for draw in range(count):
            digest = hashlib.sha256(
                f"issue56-v3-bootstrap-v1|{seed}|{replicate}|{draw}".encode("utf-8")
            ).digest()
            result[replicate, draw] = int.from_bytes(digest[:8], "big") % count
    return result


def _paired_bootstrap(differences: list[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError("V3 bootstrap differences are invalid")
    draws = np.mean(values[_bootstrap_indices(BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, len(values))], axis=1)
    return {
        "point_difference": float(np.mean(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _aggregate(records: list[V3EpisodeRecord]) -> dict[str, Any]:
    by_arm: dict[str, list[V3EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_arm[record.arm].append(record)
    summaries: dict[str, Any] = {}
    for arm, arm_records in sorted(by_arm.items()):
        summaries[arm] = {
            "family_count": len(arm_records),
            "family_means": {
                metric: float(np.mean([float(getattr(record, metric)) for record in arm_records]))
                for metric in (
                    "safety_exposure",
                    "safety_violation_steps",
                    "comfort_deviation",
                    "resource_composite",
                )
            },
            "proposal_count": sum(record.proposal_count for record in arm_records),
            "abstention_count": sum(record.abstention_count for record in arm_records),
            "admitted_proposal_count": sum(record.admitted_proposal_count for record in arm_records),
            "hmc_mismatch_count": sum(record.hmc_mismatch_count for record in arm_records),
            "replay_committed_steps": sum(record.replay_committed_steps for record in arm_records),
        }
    return {"family_count": len({record.family_id for record in records}), "arm_summaries": summaries}


def _candidate_diagnostics(model: V3RiskModel, samples: list[Any]) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("V3 candidate diagnostics require evaluation samples")
    latencies: list[float] = []
    rejected_dangerous = 0
    dangerous = 0
    retained_dangerous = 0
    retained = 0
    for sample in samples:
        started = time.perf_counter_ns()
        prediction = model.predict_features(sample.features_f32)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        actual_dangerous = sample.label.remaining_metric.crossing_event > 0.5
        if actual_dangerous:
            dangerous += 1
            rejected_dangerous += int(prediction.hard_ineligible)
        if not prediction.hard_ineligible:
            retained += 1
            retained_dangerous += int(actual_dangerous)
    recall = rejected_dangerous / max(dangerous, 1)
    false_safe = retained_dangerous / max(retained, 1)
    baseline_false_safe = dangerous / len(samples)
    return {
        "sample_count": len(samples),
        "dangerous_sample_count": dangerous,
        "retained_sample_count": retained,
        "dangerous_event_recall": recall,
        "false_safe_rate": false_safe,
        "unfiltered_reference_recall": 0.0,
        "unfiltered_reference_false_safe_rate": baseline_false_safe,
        "crossing_recall_difference": recall,
        "false_safe_rate_difference": false_safe - baseline_false_safe,
        "inference_latency_p99_ms": float(np.quantile(np.asarray(latencies), 0.99)),
        "diagnostic_support_passed": dangerous > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Issue #56 V3 policy-aligned risk study")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=FAMILY_COUNT)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    if not MIN_FAMILY_COUNT <= args.families <= FAMILY_COUNT:
        raise SystemExit(f"--families must be between {MIN_FAMILY_COUNT} and {FAMILY_COUNT}")
    output = _validate_output_path(
        args.output,
        allow_dirty_smoke=args.allow_dirty_smoke,
        families=args.families,
    )
    preregistration_sha256, preregistration = _verify_preregistration()
    source_identity = _source_identity()
    bundle = load_forecast_contracts(REPO_ROOT)
    point_model = load_live_mlp_model(POINT_ARTIFACT_PATH, expected_sha256=POINT_ARTIFACT_SHA256)
    roster = deterministic_family_ids(FAMILY_COUNT)
    split = v3_family_split(roster)
    selected_ids = _select_families(roster, split, args.families)
    manifest = {
        "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.manifest",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": preregistration_sha256,
        "family_ids": list(selected_ids),
        "family_roster": [family_condition_descriptor(index) for index in range(FAMILY_COUNT)],
        "family_split": {family_id: split[family_id] for family_id in selected_ids},
        "decision_steps": list(v2_decision_steps()),
        "episode_steps": EPISODE_STEPS,
        "source_identity": source_identity,
        "point_artifact_sha256": point_model.artifact_sha256,
    }
    manifest_sha256 = _write_json(output / "manifest.json", manifest)
    print(f"Issue #56 V3 study ({args.families} families)", file=sys.stderr)
    samples: list[Any] = []
    for index, family_id in enumerate(selected_ids):
        family_samples = collect_v3_family_samples(
            bundle,
            build_family_scenario(bundle.development_scenario, roster.index(family_id)),
            family_id,
            split=split[family_id],
        )
        samples.extend(family_samples)
        print(f"  family {index + 1}/{len(selected_ids)}: {family_id} ({len(family_samples)} samples)", file=sys.stderr)
    samples_sha256 = _write_jsonl(output / "samples.jsonl", [sample.to_mapping() for sample in samples])
    samples = list(load_v3_samples([sample.to_mapping() for sample in samples]))
    train = [sample for sample in samples if sample.split == "TRAIN"]
    validation = [sample for sample in samples if sample.split == "VALIDATION"]
    evaluation = [sample for sample in samples if sample.split == "EVALUATION"]
    if not train or not validation or not evaluation:
        raise RuntimeError("V3 selected roster must contain all three splits")
    model = V3RiskModel.fit(train).calibrate(validation)
    model_sha256 = _write_json(output / "model.json", model.to_mapping())
    calibration = {
        "model_metrics": calibration_metrics_v3(model, evaluation),
        "candidate_diagnostics": _candidate_diagnostics(model, evaluation),
    }
    calibration_sha256 = _write_json(output / "calibration.json", calibration)
    validation_groups: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for sample in validation:
        validation_groups[(sample.family_id, sample.decision_step)].append(sample)
    retained = sum(
        any(not model.predict_features(sample.features_f32).hard_ineligible for sample in group)
        for group in validation_groups.values()
    )
    non_vacuity = {
        "decision_count": len(validation_groups),
        "retained_decision_count": retained,
        "coverage": retained / max(len(validation_groups), 1),
        "passed": retained >= 1 and retained / max(len(validation_groups), 1) >= 0.10,
    }
    non_vacuity_sha256 = _write_json(output / "validation-non-vacuity.json", non_vacuity)
    if not non_vacuity["passed"]:
        raise RuntimeError("V3 validation non-vacuity failed; evaluation was not run")
    evaluation_ids = tuple(sorted({sample.family_id for sample in evaluation}))
    records: list[V3EpisodeRecord] = []
    for family_id in evaluation_ids:
        family_index = roster.index(family_id)
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        for arm in (
            "rules_only_common_window",
            "point_model_common_window",
            *V3_ARMS,
        ):
            record = run_v3_episode(
                bundle,
                scenario,
                arm,
                family_id,
                family_index,
                model,
                point_model.predictor,
            )
            records.append(record)
            trace_path = output / "traces" / f"{arm}--{family_id}.json"
            trace_path.write_bytes(record.trace_canonical_bytes)
        print(f"  evaluation family {family_id} complete (4 arms)", file=sys.stderr)
    episodes_sha256 = _write_jsonl(output / "episodes.jsonl", [record.to_mapping() for record in records])
    aggregate = _aggregate(records)
    aggregate_sha256 = _write_json(output / "aggregate.json", aggregate)
    by_pair = {(record.arm, record.family_id): record for record in records}
    paired = {}
    for arm in ("point_model_common_window", *V3_ARMS):
        paired[arm] = {
            metric: _paired_bootstrap(
                [
                    float(getattr(by_pair[(arm, family_id)], metric))
                    - float(getattr(by_pair[("rules_only_common_window", family_id)], metric))
                    for family_id in evaluation_ids
                ]
            )
            for metric in ("safety_exposure", "safety_violation_steps", "comfort_deviation", "resource_composite")
        }
    diagnostics = calibration["candidate_diagnostics"]
    technical = {
        "authority_violation_count": sum(not record.authority_verified for record in records),
        "replay_failure_count": sum(not record.replay_verified for record in records),
        "provenance_violation_count": sum(not record.provenance_verified for record in records),
        "non_finite_metric_count": sum(not record.metrics_finite_verified for record in records),
        "proposal_admission_failure_count": sum(not record.proposal_admission_verified for record in records),
    }
    safety = paired["risk_filtered_point_v3"]["safety_exposure"]
    gates = {
        **technical,
        "validation_non_vacuity_passed": bool(non_vacuity["passed"]),
        "risk_filtered_safety_point_and_ci_passed": (
            safety["point_difference"] <= 0.0 and safety["ci_upper"] <= 0.0
        ),
        "crossing_recall_passed": diagnostics["crossing_recall_difference"] >= -0.02,
        "false_safe_rate_passed": diagnostics["false_safe_rate_difference"] <= 0.01,
        "latency_p99_passed": diagnostics["inference_latency_p99_ms"] <= 250.0,
        "candidate_diagnostic_support_passed": bool(diagnostics["diagnostic_support_passed"]),
    }
    results = {
        "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.results",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": preregistration_sha256,
        "family_count": args.families,
        "evaluation_family_count": len(evaluation_ids),
        "manifest_sha256": manifest_sha256,
        "samples_sha256": samples_sha256,
        "model_sha256": model_sha256,
        "calibration_sha256": calibration_sha256,
        "validation_non_vacuity_sha256": non_vacuity_sha256,
        "episodes_sha256": episodes_sha256,
        "aggregate_sha256": aggregate_sha256,
        "sample_counts": {
            "TRAIN": len(train),
            "VALIDATION": len(validation),
            "EVALUATION": len(evaluation),
            "total": len(samples),
        },
        "source_identity": source_identity,
        "hard_gates": gates,
        "all_gates_passed": all(bool(value) if isinstance(value, bool) else value == 0 for value in gates.values()),
        "non_vacuity": non_vacuity,
        "calibration": calibration,
        "paired_bootstrap_vs_rules": paired,
        "aggregate": aggregate,
        "smoke_only": args.families < FAMILY_COUNT,
        "status": "SMOKE_PATH_ONLY" if args.families < FAMILY_COUNT else "DEVELOPMENT_EVIDENCE",
        "preregistration": preregistration,
    }
    results_sha256 = _write_json(output / "results.json", results)
    print(f"results.json sha256: {results_sha256}", file=sys.stderr)
    print(f"output: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
