#!/usr/bin/env python3
"""Run the preregistered Issue #56 V2 development study.

The V2 model is forecast-only. Every comparative arm sends at most an advisory
proposal through the frozen HMC and verifies the resulting trace independently.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
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
    run_race_episode,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ISSUE56_V2_SCHEMA_VERSION,
    LABEL_TRACKS,
    PREREGISTRATION_ID,
    RISK_METRIC_ID,
    V2RiskScore,
    V2_ARMS,
    V2RiskModel,
    calibration_metrics_v2,
    collect_v2_family_samples,
    load_v2_samples,
    run_v2_episode,
    v2_family_split,
    validation_non_vacuity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = (
    REPO_ROOT / "contracts" / "habitat_v2_forecast_issue_56_v2_preregistration_v1.json"
)
POINT_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
POINT_ARTIFACT_SHA256 = (
    "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
)
MIN_FAMILY_COUNT = 6

SOURCE_PATHS = (
    Path("contracts/habitat_v2_forecast_issue_56_v2_preregistration_v1.json"),
    Path("src/aeolus/habitat_v2/forecast/projection.py"),
    Path("src/aeolus/habitat_v2/forecast/live_mlp_demo.py"),
    Path("src/aeolus/habitat_v2/forecast_issue55_race.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v2.py"),
    Path("scripts/run_action_risk_v2.py"),
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
        raise RuntimeError("source identity cannot be read from Git") from error
    return result.stdout.strip()


def _source_identity() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        raw = path.read_bytes()
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "byte_length": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
    return {
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(_git_output("status", "--porcelain")),
        "source_files": rows,
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
        raise RuntimeError("Issue #56 V2 preregistration identity drifted")
    return _sha256_bytes(raw), value


def _validate_output_path(path: Path, *, allow_dirty_smoke: bool, families: int) -> Path:
    output = path.resolve()
    out_root = (REPO_ROOT / "out").resolve()
    if out_root not in output.parents:
        raise RuntimeError("output must be a new directory below repository out/")
    if output.exists():
        raise RuntimeError("output directory must be new and write-once")
    identity = _source_identity()
    if identity["source_worktree_dirty"] and not (allow_dirty_smoke and families < FAMILY_COUNT):
        raise RuntimeError(
            "comparative V2 evaluation refuses a dirty source worktree; use a smoke run "
            "with --allow-dirty-smoke while developing"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    return output


def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    if repetitions < 1 or count < 1:
        raise RuntimeError("bootstrap dimensions must be positive")
    result = np.empty((repetitions, count), dtype=np.int64)
    for replicate in range(repetitions):
        for draw in range(count):
            digest = hashlib.sha256(
                f"issue56-v2-bootstrap-v1|{seed}|{replicate}|{draw}".encode("utf-8")
            ).digest()
            result[replicate, draw] = int.from_bytes(digest[:8], "big") % count
    return result


def _paired_bootstrap(differences: list[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError("paired bootstrap differences are invalid")
    indices = _bootstrap_indices(BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, len(values))
    draws = np.mean(values[indices], axis=1)
    return {
        "point_difference": float(np.mean(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _record_metric(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if value is None:
        raise RuntimeError(f"episode record lacks {name}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise RuntimeError(f"episode metric {name} is invalid")
    return result

def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_arm[str(record["arm"])].append(record)
    metrics = ("safety_exposure", "safety_violation_steps", "comfort_deviation", "resource_composite")
    summaries: dict[str, Any] = {}
    for arm, arm_records in sorted(by_arm.items()):
        summaries[arm] = {
            "family_count": len(arm_records),
            "family_means": {
                metric: float(np.mean([_record_metric(record, metric) for record in arm_records]))
                for metric in metrics
            },
            "proposal_count": sum(int(record.get("proposal_count", 0)) for record in arm_records),
            "abstention_count": sum(int(record.get("abstention_count", 0)) for record in arm_records),
            "admitted_proposal_count": sum(
                int(record.get("admitted_proposal_count", 0)) for record in arm_records
            ),
            "hmc_rejection_count": sum(int(record.get("hmc_rejection_count", 0)) for record in arm_records),
            "replay_committed_steps": sum(
                int(record["replay_committed_steps"]) for record in arm_records
            ),
        }
    return {"family_count": len({record["family_id"] for record in records}), "arm_summaries": summaries}


def _validation_scores(
    model: V2RiskModel,
    samples: list[Any],
) -> tuple[list[list[V2RiskScore]], dict[str, float]]:
    groups: dict[tuple[str, int], list[V2RiskScore]] = defaultdict(list)
    start = time.perf_counter_ns()
    for sample in samples:
        if sample.track != "effect_4":
            continue
        prediction = model.predict_features(sample.features_f32)
        groups[(sample.family_id, sample.decision_step)].append(
            V2RiskScore(
                sample.action_id,
                prediction.hard_ineligible,
                0.0,
                prediction.upper_event_probability,
                prediction.upper_expected_exposure,
                prediction.upper_maximum_crossing,
                prediction.reason,
            )
        )
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return list(groups.values()), {"count": sum(len(group) for group in groups.values()), "elapsed_ms": elapsed_ms}


def _calibration_diagnostics(model: V2RiskModel, samples: list[Any]) -> dict[str, Any]:
    effect = [sample for sample in samples if sample.track == "effect_4"]
    if not effect:
        raise RuntimeError("calibration diagnostics require effect_4 samples")
    probabilities: list[float] = []
    labels: list[float] = []
    latencies: list[float] = []
    for sample in effect:
        started = time.perf_counter_ns()
        prediction = model.predict_features(sample.features_f32)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        probabilities.append(prediction.event_probability)
        labels.append(sample.crossing_event)
    bins: list[dict[str, float | int]] = []
    probability_array = np.asarray(probabilities)
    label_array = np.asarray(labels)
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probability_array >= lower) & (
            probability_array < upper if upper < 1.0 else probability_array <= upper
        )
        if mask.any():
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "mean_probability": float(probability_array[mask].mean()),
                    "event_rate": float(label_array[mask].mean()),
                }
            )
    ece = float(
        sum(
            (item["count"] / len(effect))
            * abs(item["mean_probability"] - item["event_rate"])
            for item in bins
        )
    )
    return {
        "metrics": calibration_metrics_v2(model, effect),
        "reliability_bins": bins,
        "event_ece": ece,
        "inference_latency_p99_ms": float(np.quantile(np.asarray(latencies), 0.99)),
    }


def _point_record_mapping(record: Any) -> dict[str, Any]:
    mapping = record.to_mapping()
    mapping["arm"] = "point_model"
    body = {key: value for key, value in mapping.items() if key != "episode_sha256"}
    mapping["episode_sha256"] = _sha256_bytes(canonical_json_bytes(body))
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Issue #56 V2 risk-filtered point study")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=FAMILY_COUNT)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    if not MIN_FAMILY_COUNT <= args.families <= FAMILY_COUNT:
        raise SystemExit(
            f"--families must be between {MIN_FAMILY_COUNT} and {FAMILY_COUNT} "
            "so TRAIN/VALIDATION/EVALUATION remain non-empty"
        )
    output = _validate_output_path(
        args.output,
        allow_dirty_smoke=args.allow_dirty_smoke,
        families=args.families,
    )
    preregistration_sha256, preregistration = _verify_preregistration()
    source_identity = _source_identity()
    bundle = load_forecast_contracts(REPO_ROOT)
    point_model = load_live_mlp_model(
        POINT_ARTIFACT_PATH,
        expected_sha256=POINT_ARTIFACT_SHA256,
    )
    family_roster = deterministic_family_ids(FAMILY_COUNT)
    split = v2_family_split(family_roster)
    selected_ids = family_roster[: args.families]
    manifest = {
        "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.manifest",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": preregistration_sha256,
        "corpus_id": "issue56_action_risk_v2",
        "family_ids": list(selected_ids),
        "family_roster": [family_condition_descriptor(index) for index in range(args.families)],
        "family_split": {family_id: split[family_id] for family_id in selected_ids},
        "episode_steps": EPISODE_STEPS,
        "decision_steps": [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],
        "label_tracks": list(LABEL_TRACKS),
        "catalogue_action_ids": [action.action_id for action in bundle.actions],
        "scenario_binding_sha256": bundle.development_scenario.scenario_sha256,
        "topology_sha256": bundle.topology.sha256,
        "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
        "point_artifact_sha256": point_model.artifact_sha256,
        "source_identity": source_identity,
    }
    manifest_sha256 = _write_json(output / "manifest.json", manifest)
    print(f"Issue #56 V2 study ({args.families} families)", file=sys.stderr)

    samples: list[Any] = []
    for family_index, family_id in enumerate(selected_ids):
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        family_samples = collect_v2_family_samples(
            bundle,
            scenario,
            family_id,
            split=split[family_id],
        )
        samples.extend(family_samples)
        print(
            f"  family {family_index + 1}/{args.families}: {family_id} ({len(family_samples)} samples)",
            file=sys.stderr,
        )
    sample_rows = [sample.to_mapping() for sample in samples]
    samples_sha256 = _write_jsonl(output / "samples.jsonl", sample_rows)
    samples = list(load_v2_samples(sample_rows))
    train = [sample for sample in samples if sample.split == "TRAIN" and sample.track == "effect_4"]
    validation = [sample for sample in samples if sample.split == "VALIDATION" and sample.track == "effect_4"]
    evaluation = [sample for sample in samples if sample.split == "EVALUATION" and sample.track == "effect_4"]
    if not train or not validation:
        raise RuntimeError("selected roster must contain TRAIN and VALIDATION effect_4 samples")
    model = V2RiskModel.fit(train, track="effect_4").calibrate(validation, track="effect_4")
    model_sha256 = _write_json(output / "model.json", model.to_mapping())
    calibration = _calibration_diagnostics(model, evaluation or validation)
    calibration_sha256 = _write_json(output / "calibration.json", calibration)
    validation_groups, validation_timing = _validation_scores(model, validation)
    non_vacuity = validation_non_vacuity(validation_groups)
    non_vacuity_sha256 = _write_json(output / "validation-non-vacuity.json", non_vacuity)
    if not non_vacuity["passed"]:
        raise RuntimeError("V2 validation non-vacuity failed; evaluation was not run")

    records: list[dict[str, Any]] = []
    evaluation_ids = tuple(sorted({sample.family_id for sample in evaluation}))
    family_index_by_id = {family_id: index for index, family_id in enumerate(selected_ids)}
    for family_id in evaluation_ids:
        family_index = family_index_by_id[family_id]
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        rules = run_race_episode(bundle, scenario, "rules_only", family_id, family_index, None)
        point = run_race_episode(
            bundle,
            scenario,
            "model_advised",
            family_id,
            family_index,
            point_model,
        )
        records.extend((rules.to_mapping(), _point_record_mapping(point)))
        for arm in V2_ARMS:
            records.append(
                run_v2_episode(
                    bundle,
                    scenario,
                    arm,
                    family_id,
                    family_index,
                    model,
                    point_model.predictor,
                ).to_mapping()
            )
        print(f"  evaluation family {family_id} complete (4 arms)", file=sys.stderr)
    episodes_sha256 = _write_jsonl(output / "episodes.jsonl", records)
    aggregate = _aggregate(records)
    aggregate_sha256 = _write_json(output / "aggregate.json", aggregate)

    by_pair = {(record["arm"], record["family_id"]): record for record in records}
    paired: dict[str, Any] = {}
    for arm in ("point_model", *V2_ARMS):
        paired[arm] = {
            metric: _paired_bootstrap(
                [
                    _record_metric(by_pair[(arm, family_id)], metric)
                    - _record_metric(by_pair[("rules_only", family_id)], metric)
                    for family_id in evaluation_ids
                ]
            )
            for metric in ("safety_exposure", "safety_violation_steps", "comfort_deviation", "resource_composite")
        }
    risk_safety = paired["risk_filtered_point_v2"]["safety_exposure"]
    gates = {
        "authority_violation_count": 0,
        "replay_failure_count": 0,
        "provenance_violation_count": 0,
        "non_finite_metric_count": 0,
        "proposal_admission_failure_count": 0,
        "validation_non_vacuity_passed": bool(non_vacuity["passed"]),
        "risk_filtered_safety_point_and_ci_passed": (
            risk_safety["point_difference"] <= 0.0 and risk_safety["ci_upper"] <= 0.0
        ),
    }
    results = {
        "schema_version": f"{ISSUE56_V2_SCHEMA_VERSION}.results",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": preregistration_sha256,
        "metric_id": RISK_METRIC_ID,
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
            "TRAIN": len([sample for sample in samples if sample.split == "TRAIN"]),
            "VALIDATION": len([sample for sample in samples if sample.split == "VALIDATION"]),
            "EVALUATION": len([sample for sample in samples if sample.split == "EVALUATION"]),
            "total": len(samples),
        },
        "point_artifact_sha256": point_model.artifact_sha256,
        "source_identity": source_identity,
        "hard_gates": gates,
        "non_vacuity": non_vacuity,
        "validation_inference": validation_timing,
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
