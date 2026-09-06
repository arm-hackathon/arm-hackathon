"""Calibrate Issue #76 interval correction and abstention on CALIBRATION only.

Retrains the five Issue #75 seeds deterministically (asserting their digests
against the bound BDM-v1 receipt), fits conformal offsets and the declared
abstention thresholds on the CALIBRATION partition, reports coverage and
reliability by horizon and stratum plus risk-coverage and useful-opportunity
curves, and writes a write-once receipt. The frozen threshold contract is
committed separately, after this run, before any closed-loop study.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.bdm_v1_corpus import FEATURE_FIELD_NAMES  # noqa: F401
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast_issue74_baselines import delta_labels, trajectory_labels
from aeolus.habitat_v2.forecast_issue75_bdm import (
    DELTA_SLICE,
    forward,
    input_tensor,
    train_model,
)
from aeolus.habitat_v2.forecast_issue76_abstention import (
    ABSTENTION_REASONS,
    CalibrationLayer,
    abstention_guard,
    abstention_reason,
    corrected_delta_interval,
    fit_conformal_offsets,
    fit_thresholds,
    interval_coverage,
    layer_digest,
    risk_coverage_curve,
    trajectory_interval_coverage,
    useful_opportunity_recall,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TCN_PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_tcn_preregistration_v1.json"
GUARD_MAX_ABSTAINED_USEFUL = 0.5
CURVE_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


class CalibrationStudyError(RuntimeError):
    """Raised when the calibration study cannot proceed safely."""


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise CalibrationStudyError("calibration output must live under the ignored out/ directory") from error
    if output.exists():
        raise CalibrationStudyError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def _load_samples(corpus: Path, partition: str) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    shard_dir = corpus / "shards"
    if not shard_dir.is_dir():
        raise CalibrationStudyError(f"corpus shards missing under {corpus}")
    for shard in sorted(shard_dir.glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line:
                continue
            sample = json.loads(line)
            if sample["partition"] == partition:
                samples.append(sample)
    samples.sort(
        key=lambda item: (
            item["family_id"],
            item["decision_step"],
            item["features"]["candidate_action_index"],
        )
    )
    if not samples:
        raise CalibrationStudyError(f"no {partition} samples in corpus {corpus}")
    return tuple(samples)


def run_calibration(corpus: Path, bdm_receipt_path: Path, output: Path) -> dict[str, Any]:
    bdm_receipt_raw = bdm_receipt_path.read_bytes()
    bdm_receipt = json.loads(bdm_receipt_raw)
    prereg = json.loads(TCN_PREREG_PATH.read_bytes())
    if not prereg["freeze"]["frozen_before_final_runs"]:
        raise CalibrationStudyError("BDM-v1 preregistration is not frozen")

    train_samples = _load_samples(corpus, prereg["training"]["fit_partition"])
    cal_samples = _load_samples(corpus, "CALIBRATION")
    train_tensors = np.stack([input_tensor(sample) for sample in train_samples])
    train_trajectories = np.stack([trajectory_labels(sample) for sample in train_samples])
    train_deltas = np.stack([delta_labels(sample) for sample in train_samples])
    cal_tensors = np.stack([input_tensor(sample) for sample in cal_samples])
    cal_trajectories = np.stack([trajectory_labels(sample) for sample in cal_samples])
    cal_deltas = np.stack([delta_labels(sample) for sample in cal_samples])

    training = prereg["training"]
    started = time.perf_counter()
    models = []
    for expected in bdm_receipt["seeds"]:
        model = train_model(
            train_tensors,
            train_trajectories,
            train_deltas,
            cal_tensors,
            cal_deltas,
            seed=expected["seed"],
            learning_rate=float(training["optimizer"]["learning_rate"]),
            batch_size=int(training["batch_size"]),
            epochs_max=int(training["epochs_max"]),
            patience=int(training["early_stopping"]["patience"]),
            betas=tuple(training["optimizer"]["betas"]),
            eps=float(training["optimizer"]["eps"]),
        )
        if model.digest != expected["digest"]:
            raise CalibrationStudyError(
                f"retrained seed {expected['seed']} digest drifted from the bound receipt"
            )
        models.append(model)
    train_seconds = time.perf_counter() - started

    per_seed_outs = [forward(model, cal_tensors)[0] for model in models]
    per_seed_deltas = np.stack([out[:, DELTA_SLICE[0] : DELTA_SLICE[1]] for out in per_seed_outs])
    pooled_out = np.mean(np.stack(per_seed_outs), axis=0)

    delta_offsets, trajectory_offsets = fit_conformal_offsets(
        pooled_out, cal_trajectories, cal_deltas
    )
    thresholds = fit_thresholds(cal_samples, per_seed_deltas, (delta_offsets, trajectory_offsets), pooled_out)
    digest = layer_digest(delta_offsets, trajectory_offsets, thresholds)
    layer = CalibrationLayer(
        delta_offsets,
        trajectory_offsets,
        thresholds["max_staleness"],
        thresholds["min_mask_fraction"],
        thresholds["seed_disagreement"],
        thresholds["interval_width"],
        digest,
    )

    coverage_delta = interval_coverage(layer, pooled_out, cal_deltas)
    coverage_by_horizon = {
        horizon: trajectory_interval_coverage(layer, pooled_out, cal_trajectories, index)
        for index, horizon in enumerate(("4", "16", "32"))
    }
    strata = sorted({sample["stratum"] for sample in cal_samples})
    coverage_by_stratum = {}
    for stratum in strata:
        indices = [
            index
            for index, sample in enumerate(cal_samples)
            if sample["stratum"] == stratum
        ]
        coverage_by_stratum[stratum] = interval_coverage(
            layer, pooled_out[indices], cal_deltas[indices]
        )

    low, high = corrected_delta_interval(layer, pooled_out)
    widths = high[:, 1] - low[:, 1]
    losses = np.abs(pooled_out[:, DELTA_SLICE[0] + 1] - cal_deltas[:, 1])
    reasons_counter = {reason: 0 for reason in ABSTENTION_REASONS}
    abstained_flags: list[bool] = []
    predicted_improving: list[bool] = []
    true_improving: list[bool] = []
    for index, sample in enumerate(cal_samples):
        reason = abstention_reason(sample, per_seed_deltas[:, index], layer, pooled_out[index])
        if reason is not None:
            reasons_counter[reason] += 1
        abstained_flags.append(reason is not None)
        predicted_improving.append(float(pooled_out[index][DELTA_SLICE[0] + 1]) < 0.0)
        true_improving.append(float(cal_deltas[index][1]) < 0.0)
    abstention_rate = sum(abstained_flags) / len(abstained_flags)
    recall, abstained_useful_rate = useful_opportunity_recall(
        predicted_improving, true_improving, abstained_flags
    )
    guard_passed = abstention_guard(abstained_useful_rate, GUARD_MAX_ABSTAINED_USEFUL)
    curve = risk_coverage_curve(losses, widths, CURVE_MULTIPLIERS, layer.interval_width)

    receipt = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_abstention_receipt_v1",
        "bdm_receipt_sha256": hashlib.sha256(bdm_receipt_raw).hexdigest(),
        "tcn_preregistration_sha256": hashlib.sha256(TCN_PREREG_PATH.read_bytes()).hexdigest(),
        "corpus_digest": bdm_receipt["corpus_digest"],
        "calibration_samples": len(cal_samples),
        "retrained_seed_digests": [model.digest for model in models],
        "layer": layer.mapping(),
        "coverage": {
            "delta_interval": coverage_delta,
            "trajectory_by_horizon": coverage_by_horizon,
            "delta_by_stratum": coverage_by_stratum,
            "nominal": 0.8,
        },
        "abstention": {
            "rate": abstention_rate,
            "reasons": reasons_counter,
            "useful_opportunity_recall": recall,
            "abstained_useful_rate": abstained_useful_rate,
            "guard_max_abstained_useful": GUARD_MAX_ABSTAINED_USEFUL,
            "guard_passed": guard_passed,
        },
        "risk_coverage_curve": curve,
        "runtime_seconds": {"retrain_all_seeds": round(train_seconds, 2)},
    }
    (output / "abstention-receipt.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bdm-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _resolve_output(args.output)
    receipt = run_calibration(args.corpus, args.bdm_receipt, output)
    print(json.dumps(receipt["coverage"], indent=1))
    print(json.dumps(receipt["abstention"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
