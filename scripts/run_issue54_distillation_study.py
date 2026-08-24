#!/usr/bin/env python3
"""Issue #54 distillation study: collect corpus, train students, evaluate.

Runs the frozen forecast pipeline with the development scenario at multiple
anchor steps and varied sensor seeds to generate a distillation corpus for
both MLP and ridge teachers.  Trains progressively smaller students and
evaluates how well they copy the teacher's predictions.

Usage::

    python scripts/run_issue54_distillation_study.py --pilot
    python scripts/run_issue54_distillation_study.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from aeolus.habitat_v2.forecast.baselines import flatten_features
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast.live_demo import load_live_ridge_model
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model
from aeolus.habitat_v2.forecast.pipeline import _proposal
from aeolus.habitat_v2.forecast.projection import (
    project_history_window,
    project_physical_targets,
    project_proposed_action,
)
from aeolus.habitat_v2.forecast_issue52 import extend_scenario_for_issue52
from aeolus.habitat_v2.forecast_issue54_distillation import (
    DistillationSample,
    StudentMlpModel,
    build_corpus_manifest,
    deterministic_family_ids,
    evaluate_student,
    family_split,
    fit_student_ridge,
    make_distillation_sample,
    save_student_mlp,
    train_student_mlp,
    validate_samples,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics import advance_one_step_with_command, initial_state
from aeolus.habitat_v2.scenario import Scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
MLP_ARTIFACT_PATH = REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
RIDGE_ARTIFACT_PATH = REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
MLP_ARTIFACT_SHA = "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
RIDGE_ARTIFACT_SHA = "0de4b5cdb6ec2b47be260a06f924d8eb00f1def16d5ae668b3ab5191251f29df"

MLP_WINDOW = 16
RIDGE_WINDOW = 4
HORIZON = 8
TARGET_COUNT = 51
N_ZONES = 8
CORPUS_ID = "fresh_pipeline"
EXTENDED_STEPS = 48

PILOT_FAMILIES = 6
PILOT_ANCHORS = (16, 24)
FULL_FAMILIES = 32
FULL_ANCHORS = (16, 24, 32)

STUDENT_IDS = ("sanity-2.1m", "medium-500k", "small-100k", "tiny-25k", "linear")

_ZONE_FIELDS = [
    (10.0, 295.15, 250.0, 330.0),
    (1000.0, 101325.0, 50000.0, 150000.0),
    (800.0, 800.0, 300.0, 5000.0),
    (0.005, 0.2095, 0.15, 0.30),
    (0.25, 0.45, 0.0, 1.0),
    (0.1, 0.05, 0.0, 1.0),
]
_RESOURCE_FIELDS = [
    (0.25, 0.75, 0.0, 1.0),
    (0.25, 0.75, 0.0, 1.0),
    (0.25, 0.75, 0.0, 1.0),
]


def build_eval_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scales: list[float] = []
    nominals: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for _ in range(N_ZONES):
        for s, n, lo, hi in _ZONE_FIELDS:
            scales.append(s)
            nominals.append(n)
            lowers.append(lo)
            uppers.append(hi)
    for s, n, lo, hi in _RESOURCE_FIELDS:
        scales.append(s)
        nominals.append(n)
        lowers.append(lo)
        uppers.append(hi)
    return (
        np.array(scales, dtype=np.float64),
        np.array(nominals, dtype=np.float64),
        np.array(lowers, dtype=np.float64),
        np.array(uppers, dtype=np.float64),
    )


def create_family_scenario(base_data: dict, family_index: int) -> Scenario:
    data = json.loads(json.dumps(base_data, allow_nan=False))
    base_seed = int(data.get("sensor_model", {}).get("random_seed", 20260812))
    data["sensor_model"]["random_seed"] = base_seed + family_index * 1000
    data["name"] = f"{data.get('name', 'dev')}-issue54-f{family_index:04d}"
    scenario = Scenario.from_mapping(data)
    return extend_scenario_for_issue52(scenario, minimum_steps=EXTENDED_STEPS)


def run_hmc_and_collect(
    scenario: Scenario,
    bundle: object,
    mlp_teacher: object,
    ridge_teacher: object,
    action: object,
    anchor: int,
    family_id: str,
    split: str,
) -> tuple[DistillationSample, DistillationSample]:
    nonce = hashlib.sha256(
        f"issue54-{family_id}-{anchor}-{action.action_id}".encode("utf-8")
    ).digest()
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[object, object]] = {}
    states: dict[int, object] = {0: shadow}

    mlp_input: np.ndarray | None = None
    ridge_input: np.ndarray | None = None
    mlp_pred: np.ndarray | None = None
    ridge_pred: np.ndarray | None = None

    total_steps = anchor + HORIZON + 1

    for step in range(total_steps):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise RuntimeError(f"HMC terminated at step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        if step > 0:
            snapshots[step] = (snapshot, verification)

        proposal = None
        if step == anchor:
            pairs_mlp = [
                snapshots[s]
                for s in range(anchor - MLP_WINDOW + 1, anchor + 1)
            ]
            mlp_history = project_history_window(
                bundle, pairs_mlp, window_steps=MLP_WINDOW,
            )

            pairs_ridge = [
                snapshots[s]
                for s in range(anchor - RIDGE_WINDOW + 1, anchor + 1)
            ]
            ridge_history = project_history_window(
                bundle, pairs_ridge, window_steps=RIDGE_WINDOW,
            )

            proposed_action = project_proposed_action(bundle, action.command)

            mlp_pred = mlp_teacher.predictor.predict(mlp_history, proposed_action)
            ridge_pred = ridge_teacher.predictor.predict(ridge_history, proposed_action)

            mlp_input = np.concatenate([
                mlp_history.numeric_f32.reshape(-1),
                proposed_action,
                np.ones(1, dtype=np.float32),
            ])
            ridge_input = flatten_features(
                ridge_history, proposed_action, include_action=True,
            )

            proposal = _proposal(
                hmc,
                snapshot.snapshot_sha256,
                step,
                action.command.to_mapping(),
                action.action_id,
            )

        hmc.propose(proposal, handle)
        arbitration = hmc.arbitrate()
        hmc.step()

        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command,
        )
        shadow = shadow_result.state
        states[shadow.step] = shadow

    truth_steps = list(range(anchor + 1, anchor + HORIZON + 1))
    truth = project_physical_targets(
        bundle, [states[s] for s in truth_steps], horizon_steps=HORIZON,
    )

    mlp_sample = make_distillation_sample(
        family_id, split, action.action_id, "mlp",
        mlp_input.astype(np.float32),
        np.asarray(mlp_pred, dtype=np.float32).reshape(-1),
        np.asarray(truth, dtype=np.float32).reshape(-1),
    )
    ridge_sample = make_distillation_sample(
        family_id, split, action.action_id, "ridge",
        ridge_input.astype(np.float32),
        np.asarray(ridge_pred, dtype=np.float32).reshape(-1),
        np.asarray(truth, dtype=np.float32).reshape(-1),
    )

    return mlp_sample, ridge_sample


def collect_corpus(
    bundle: object,
    mlp_teacher: object,
    ridge_teacher: object,
    n_families: int,
    anchor_steps: tuple[int, ...],
) -> tuple[list[DistillationSample], list[DistillationSample], list[str], dict[str, str]]:
    base_data = bundle.development_scenario.data
    family_ids = deterministic_family_ids(n_families, CORPUS_ID)
    split = family_split(family_ids)

    samples_mlp: list[DistillationSample] = []
    samples_ridge: list[DistillationSample] = []

    for family_idx, family_id in enumerate(family_ids):
        scenario = create_family_scenario(base_data, family_idx)
        s = split[family_id]

        for anchor in anchor_steps:
            for action in bundle.actions:
                mlp_s, ridge_s = run_hmc_and_collect(
                    scenario, bundle, mlp_teacher, ridge_teacher,
                    action, anchor, family_id, s,
                )
                samples_mlp.append(mlp_s)
                samples_ridge.append(ridge_s)

        print(
            f"  family {family_idx + 1}/{n_families}: {family_id} ({s}) "
            f"- {len(samples_mlp)} total samples",
            file=sys.stderr,
        )

    return samples_mlp, samples_ridge, list(family_ids), split


def compute_samples_sha256(samples: list[DistillationSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(canonical_json_bytes({
            "family_id": sample.family_id,
            "candidate_id": sample.candidate_id,
            "sha": sample.sample_sha256,
        }))
        digest.update(b"\n")
    return digest.hexdigest()


def train_and_evaluate(
    samples_mlp: list[DistillationSample],
    samples_ridge: list[DistillationSample],
    eval_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
) -> dict:
    scales, nominals, lowers, uppers = eval_arrays
    results: dict[str, dict] = {"mlp": {}, "ridge": {}}

    for teacher_label, samples in [("mlp", samples_mlp), ("ridge", samples_ridge)]:
        train_samples = [s for s in samples if s.split == "TRAIN"]
        val_samples = [s for s in samples if s.split == "VALIDATION"]
        final_samples = [s for s in samples if s.split == "FINAL"]

        print(
            f"\n  {teacher_label}: {len(train_samples)} TRAIN, "
            f"{len(val_samples)} VAL, {len(final_samples)} FINAL",
            file=sys.stderr,
        )

        for student_id in STUDENT_IDS:
            t0 = time.time()

            if student_id == "linear":
                student = fit_student_ridge(train_samples, teacher_label, student_id)
            else:
                student = train_student_mlp(
                    train_samples, val_samples, teacher_label, student_id,
                )

            result = evaluate_student(
                student, final_samples, scales, nominals, lowers, uppers,
            )

            if isinstance(student, StudentMlpModel):
                save_student_mlp(
                    student,
                    output_dir / f"student_{teacher_label}_{student_id}.npz",
                )

            elapsed = time.time() - t0
            results[teacher_label][student_id] = {
                "param_count": result.student_param_count,
                "nmae_teacher": result.nmae_teacher,
                "nmae_student": result.nmae_student,
                "nmae_ratio": result.nmae_ratio,
                "nmae_ratio_lower_ci": result.nmae_ratio_lower_ci,
                "nmae_ratio_upper_ci": result.nmae_ratio_upper_ci,
                "top1_agreement": result.top1_agreement,
                "kendall_tau": result.kendall_tau,
                "safety_exposure_difference": result.safety_exposure_difference,
                "family_count": result.family_count,
                "train_time_s": elapsed,
            }

            print(
                f"    {student_id}: params={result.student_param_count}, "
                f"ratio={result.nmae_ratio:.3f}, "
                f"top1={result.top1_agreement:.3f}, "
                f"tau={result.kendall_tau:.3f}, "
                f"safety_diff={result.safety_exposure_difference:.3f} "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #54 distillation study")
    parser.add_argument(
        "--pilot", action="store_true", help="Run a smaller pilot study",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("out/issue54"),
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    n_families = PILOT_FAMILIES if args.pilot else FULL_FAMILIES
    anchor_steps = PILOT_ANCHORS if args.pilot else FULL_ANCHORS

    print(
        f"Issue #54 distillation study ({'pilot' if args.pilot else 'full'})",
        file=sys.stderr,
    )
    print(
        f"  families: {n_families}, anchors: {anchor_steps}",
        file=sys.stderr,
    )

    print("Loading frozen forecast contracts and teachers...", file=sys.stderr)
    bundle = load_forecast_contracts(REPO_ROOT)
    mlp_teacher = load_live_mlp_model(MLP_ARTIFACT_PATH, expected_sha256=MLP_ARTIFACT_SHA)
    ridge_teacher = load_live_ridge_model(RIDGE_ARTIFACT_PATH, expected_sha256=RIDGE_ARTIFACT_SHA)

    print(
        f"\nCollecting distillation corpus ({n_families} families "
        f"x {len(anchor_steps)} anchors x 4 actions)...",
        file=sys.stderr,
    )
    t0 = time.time()
    samples_mlp, samples_ridge, family_ids, split = collect_corpus(
        bundle, mlp_teacher, ridge_teacher, n_families, anchor_steps,
    )
    collect_time = time.time() - t0
    print(
        f"  Collected {len(samples_mlp)} MLP + {len(samples_ridge)} ridge samples "
        f"in {collect_time:.1f}s",
        file=sys.stderr,
    )

    for teacher_label, samples in [("mlp", samples_mlp), ("ridge", samples_ridge)]:
        samples_sha = compute_samples_sha256(samples)
        manifest = build_corpus_manifest(
            teacher_label, CORPUS_ID, family_ids, split, samples_sha,
        )
        validate_samples(samples, manifest)

        manifest_path = output_dir / f"corpus_manifest_{teacher_label}.json"
        manifest_path.write_text(json.dumps({
            "schema_version": manifest.schema_version,
            "teacher_label": manifest.teacher_label,
            "corpus_id": manifest.corpus_id,
            "family_split": manifest.family_split,
            "family_ids": list(manifest.family_ids),
            "samples_sha256": manifest.samples_sha256,
            "manifest_sha256": manifest.manifest_sha256,
        }, indent=2, sort_keys=True))

        print(
            f"  {teacher_label} manifest: {manifest.manifest_sha256[:16]}...",
            file=sys.stderr,
        )

    for teacher_label, samples in [("mlp", samples_mlp), ("ridge", samples_ridge)]:
        samples_path = output_dir / f"samples_{teacher_label}.jsonl"
        with open(samples_path, "w") as f:
            for sample in samples:
                f.write(json.dumps({
                    "family_id": sample.family_id,
                    "split": sample.split,
                    "candidate_id": sample.candidate_id,
                    "teacher_label": sample.teacher_label,
                    "input_hex": sample.input_f32.tobytes().hex(),
                    "teacher_prediction_hex": sample.teacher_prediction_f32.tobytes().hex(),
                    "ground_truth_hex": sample.ground_truth_f32.tobytes().hex(),
                    "sample_sha256": sample.sample_sha256,
                }, sort_keys=True) + "\n")

    eval_arrays = build_eval_arrays()

    print("\nTraining and evaluating students...", file=sys.stderr)
    students_dir = output_dir / "students"
    students_dir.mkdir(parents=True, exist_ok=True)

    results = train_and_evaluate(
        samples_mlp, samples_ridge, eval_arrays, students_dir,
    )

    results_data = {
        "corpus_id": CORPUS_ID,
        "n_families": n_families,
        "anchor_steps": list(anchor_steps),
        "collect_time_s": collect_time,
        "teachers": results,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_data, indent=2, sort_keys=True))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("Issue #54 distillation study complete", file=sys.stderr)
    print(
        f"  Corpus: {CORPUS_ID}, {n_families} families, "
        f"{len(samples_mlp)} samples/teacher",
        file=sys.stderr,
    )
    print(f"  Results: {results_path}", file=sys.stderr)
    print(f"  Collection time: {collect_time:.1f}s", file=sys.stderr)

    print()
    print(
        f"{'teacher':>8} {'student':>12} {'params':>10} "
        f"{'ratio':>8} {'top1':>6} {'tau':>6} {'safety':>8}",
    )
    for teacher_label in ("mlp", "ridge"):
        for student_id in STUDENT_IDS:
            r = results[teacher_label][student_id]
            print(
                f"{teacher_label:>8} {student_id:>12} {r['param_count']:>10} "
                f"{r['nmae_ratio']:>8.3f} {r['top1_agreement']:>6.3f} "
                f"{r['kendall_tau']:>6.3f} {r['safety_exposure_difference']:>8.4f}",
            )


if __name__ == "__main__":
    main()
