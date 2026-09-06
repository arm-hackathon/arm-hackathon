"""Train and evaluate the Issue #75 BDM-v1 causal TCN adviser.

Fits the five preregistered seeds on TRAIN corpus samples only, uses DEV
exclusively for early-stopping model choice and evaluation, refits the Issue
#74 baselines to bind the paired comparison, and writes a write-once receipt.
The model proposes or abstains only; nothing here commands or steps the plant.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.bdm_v1_evaluation import bootstrap_ci, comparison_table
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast_issue74_baselines import (
    delta_labels,
    finite_catalogue_regret,
    fit_linear_state_space,
    fit_ridge_baseline,
    predict_sample,
    ranking_correlation,
    trajectory_labels,
    useful_action_precision_recall,
)
from aeolus.habitat_v2.forecast_issue75_bdm import (
    DELTA_SLICE,
    input_tensor,
    parameter_count,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_tcn_preregistration_v1.json"


class BdmV1StudyError(RuntimeError):
    """Raised when the BDM-v1 study cannot proceed under its preregistration."""


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise BdmV1StudyError("study output must live under the ignored out/ directory") from error
    if output.exists():
        raise BdmV1StudyError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def _load_samples(corpus: Path, partition: str) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    shard_dir = corpus / "shards"
    if not shard_dir.is_dir():
        raise BdmV1StudyError(f"corpus shards missing under {corpus}")
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
        raise BdmV1StudyError(f"no {partition} samples in corpus {corpus}")
    return tuple(samples)


def _tensors(samples: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = np.stack([input_tensor(sample) for sample in samples])
    trajectories = np.stack([trajectory_labels(sample) for sample in samples])
    deltas = np.stack([delta_labels(sample) for sample in samples])
    return tensors, trajectories, deltas


def _metric_rows(
    samples: Sequence[Mapping[str, Any]], predictions: np.ndarray
) -> tuple[list[dict[str, Any]], float, float, float]:
    rows: list[dict[str, Any]] = []
    predicted_improving: list[bool] = []
    true_improving: list[bool] = []
    value_errors: list[float] = []
    index = 0
    groups: dict[tuple[str, int], list[int]] = {}
    for position, sample in enumerate(samples):
        groups.setdefault((sample["family_id"], sample["decision_step"]), []).append(position)
    exposure_errors: list[float] = []
    for (_family, _decision), positions in groups.items():
        predicted_deltas = [float(predictions[position][1]) for position in positions]
        true_deltas = [float(delta_labels(samples[position])[1]) for position in positions]
        first = samples[positions[0]]
        value_error = float(
            np.mean(
                [
                    abs(float(predictions[position][k]) - float(delta_labels(samples[position])[k]))
                    for position in positions
                    for k in range(5)
                ]
            )
        )
        rows.append(
            {
                "group_id": str(first["group_id"]),
                "family_id": first["family_id"],
                "decision_step": first["decision_step"],
                "ranking_correlation": ranking_correlation(predicted_deltas, true_deltas),
                "regret": finite_catalogue_regret(predicted_deltas, true_deltas),
                "exposure_error": abs(
                    float(np.mean(predicted_deltas)) - float(np.mean(true_deltas))
                ),
                "action_value_error": value_error,
            }
        )
        for position in positions:
            predicted_improving.append(float(predictions[position][1]) < 0.0)
            true_improving.append(float(delta_labels(samples[position])[1]) < 0.0)
            exposure_errors.append(
                abs(float(predictions[position][1]) - float(delta_labels(samples[position])[1]))
            )
            value_errors.append(
                float(
                    np.mean(
                        [
                            abs(float(predictions[position][k]) - float(delta_labels(samples[position])[k]))
                            for k in range(5)
                        ]
                    )
                )
            )
    precision, recall = useful_action_precision_recall(predicted_improving, true_improving)
    return (
        rows,
        precision,
        recall,
        float(np.mean(exposure_errors)),
    )


def _family_recall(rows: list[dict[str, Any]], samples: Sequence[Mapping[str, Any]], predictions: np.ndarray) -> dict[str, float]:
    recall_by_family: dict[str, list[float]] = {}
    by_family: dict[str, list[int]] = {}
    for position, sample in enumerate(samples):
        by_family.setdefault(sample["family_id"], []).append(position)
    for family, positions in by_family.items():
        predicted = [float(predictions[position][1]) < 0.0 for position in positions]
        truth = [float(delta_labels(samples[position])[1]) < 0.0 for position in positions]
        _precision, recall = useful_action_precision_recall(predicted, truth)
        recall_by_family.setdefault(family, []).append(recall)
    group_recall: dict[str, list[float]] = {}
    family_group = {sample["family_id"]: str(sample["group_id"]) for sample in samples}
    for family, values in recall_by_family.items():
        group_recall.setdefault(family_group[family], []).append(sum(values) / len(values))
    return {group: sum(values) / len(values) for group, values in group_recall.items()}


def run_study(corpus: Path, baselines_receipt: Path, output: Path) -> dict[str, Any]:
    prereg_raw = PREREG_PATH.read_bytes()
    prereg = json.loads(prereg_raw)
    if not prereg["freeze"]["frozen_before_final_runs"]:
        raise BdmV1StudyError("BDM-v1 preregistration is not frozen")
    if parameter_count() != prereg["architecture"]["parameter_count"]:
        raise BdmV1StudyError("module parameter count drifted from the preregistration")
    if parameter_count() > prereg["architecture"]["parameter_budget_cap"]:
        raise BdmV1StudyError("parameter budget cap exceeded")
    bound_receipt = json.loads(baselines_receipt.read_bytes())
    for baseline_id, digest in prereg["evaluation"]["baseline_binding"]["expected_digests"].items():
        if bound_receipt["digests"][baseline_id] != digest:
            raise BdmV1StudyError(f"baselines receipt digest drifted for {baseline_id}")

    train_samples = _load_samples(corpus, prereg["training"]["fit_partition"])
    dev_samples = _load_samples(corpus, prereg["evaluation"]["partition"])
    train_tensors, train_trajectories, train_deltas = _tensors(train_samples)
    dev_tensors, _dev_trajectories, dev_deltas = _tensors(dev_samples)

    training = prereg["training"]
    started = time.perf_counter()
    seeds: list[dict[str, Any]] = []
    dev_predictions: list[np.ndarray] = []
    for seed in training["seeds"]:
        model = train_model(
            train_tensors,
            train_trajectories,
            train_deltas,
            dev_tensors,
            dev_deltas,
            seed=seed,
            learning_rate=float(training["optimizer"]["learning_rate"]),
            batch_size=int(training["batch_size"]),
            epochs_max=int(training["epochs_max"]),
            patience=int(training["early_stopping"]["patience"]),
            betas=tuple(training["optimizer"]["betas"]),
            eps=float(training["optimizer"]["eps"]),
        )
        from aeolus.habitat_v2.forecast_issue75_bdm import predict_deltas

        predictions = predict_deltas(model, dev_tensors)
        dev_predictions.append(predictions)
        seeds.append(
            {
                "seed": seed,
                "digest": model.digest,
                "epochs_run": model.epochs_run,
                "best_epoch": model.best_epoch,
            }
        )
        print(
            f"  {seed}: epochs={model.epochs_run} best={model.best_epoch} digest={model.digest[:12]}",
            flush=True,
        )
    train_seconds = time.perf_counter() - started
    pooled = np.mean(np.stack(dev_predictions), axis=0)

    baseline_models = {
        "action_agnostic_ridge": fit_ridge_baseline(
            train_samples,
            baseline_id="action_agnostic_ridge",
            ridge_lambda=float(prereg["evaluation"]["baseline_binding"]["ridge_lambda"]),
            seed=str(prereg["evaluation"]["baseline_binding"]["seed"]),
        ),
        "action_conditioned_ridge": fit_ridge_baseline(
            train_samples,
            baseline_id="action_conditioned_ridge",
            ridge_lambda=float(prereg["evaluation"]["baseline_binding"]["ridge_lambda"]),
            seed=str(prereg["evaluation"]["baseline_binding"]["seed"]),
        ),
        "controlled_linear_state_space": fit_linear_state_space(
            train_samples,
            ridge_lambda=float(prereg["evaluation"]["baseline_binding"]["ridge_lambda"]),
            seed=str(prereg["evaluation"]["baseline_binding"]["seed"]),
        ),
    }
    for baseline_id, model in baseline_models.items():
        if model.digest != prereg["evaluation"]["baseline_binding"]["expected_digests"][baseline_id]:
            raise BdmV1StudyError(f"refit baseline digest drifted for {baseline_id}")
    baseline_predictions = np.stack(
        [
            np.asarray(
                predict_sample(baseline_models["action_conditioned_ridge"], sample)["delta"][
                    "safety_exposure"
                ]
            )
            for sample in dev_samples
        ]
    )
    baseline_full = np.stack(
        [
            np.asarray(
                [
                    predict_sample(baseline_models["action_conditioned_ridge"], sample)["delta"][name]
                    for name in ("crossing_event", "safety_exposure", "maximum_crossing", "comfort_deviation", "resource_composite")
                ]
            )
            for sample in dev_samples
        ]
    )

    evaluations: dict[str, Any] = {}
    row_sets: dict[str, list[dict[str, Any]]] = {}
    for label, predictions in [("pooled_bdm_v1", pooled)] + [
        (entry["seed"], pred) for entry, pred in zip(seeds, dev_predictions, strict=True)
    ]:
        rows, precision, recall, exposure_error = _metric_rows(dev_samples, predictions)
        row_sets[label] = rows
        evaluations[label] = {
            "mean_ranking_correlation": sum(row["ranking_correlation"] for row in rows) / len(rows),
            "mean_regret": sum(row["regret"] for row in rows) / len(rows),
            "mean_exposure_error": exposure_error,
            "mean_action_value_error": sum(row["action_value_error"] for row in rows) / len(rows),
            "useful_action_precision": precision,
            "useful_action_recall": recall,
        }
    baseline_rows, baseline_precision, baseline_recall, baseline_exposure = _metric_rows(
        dev_samples, baseline_full
    )
    row_sets["action_conditioned_ridge"] = baseline_rows
    evaluations["action_conditioned_ridge"] = {
        "mean_ranking_correlation": sum(row["ranking_correlation"] for row in baseline_rows)
        / len(baseline_rows),
        "mean_regret": sum(row["regret"] for row in baseline_rows) / len(baseline_rows),
        "mean_exposure_error": baseline_exposure,
        "mean_action_value_error": sum(row["action_value_error"] for row in baseline_rows)
        / len(baseline_rows),
        "useful_action_precision": baseline_precision,
        "useful_action_recall": baseline_recall,
    }

    statistics = prereg["evaluation"]["statistics"]
    tables = {
        metric: comparison_table(
            row_sets,
            metric,
            baseline_arm="action_conditioned_ridge",
            seed=str(statistics["seed"]),
            resamples=int(statistics["bootstrap_resamples"]),
            alpha=float(statistics["alpha"]),
        )
        for metric in ("ranking_correlation", "regret", "exposure_error", "action_value_error")
    }

    pooled_group_recall = _family_recall(row_sets["pooled_bdm_v1"], dev_samples, pooled)
    baseline_group_recall = _family_recall(baseline_rows, dev_samples, baseline_full)
    gate: dict[str, Any] = {}
    paired = tables["ranking_correlation"]["arms"]["pooled_bdm_v1"]
    gate["ranking_ci_excludes_zero_beneficial"] = paired["paired_difference_ci_lower"] > 0.0
    regret_pair = tables["regret"]["arms"]["pooled_bdm_v1"]
    gate["regret_ci_excludes_zero_beneficial"] = regret_pair["paired_difference_ci_upper"] < 0.0
    exposure_pair = tables["exposure_error"]["arms"]["pooled_bdm_v1"]
    gate["exposure_error_ci_excludes_zero_beneficial"] = exposure_pair["paired_difference_ci_upper"] < 0.0
    recall_diffs = [
        pooled_group_recall[group] - baseline_group_recall[group]
        for group in sorted(pooled_group_recall)
    ]
    _mean, lo, hi = bootstrap_ci(recall_diffs, seed=str(statistics["seed"]) + "|coverage")
    gate["coverage_ci_excludes_zero_beneficial"] = lo > 0.0
    gate["promotion_gate_passed"] = all(gate.values())

    receipt = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_tcn_receipt_v1",
        "preregistration_sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "corpus_digest": bound_receipt["corpus_digest"],
        "baselines_receipt_sha256": hashlib.sha256(baselines_receipt.read_bytes()).hexdigest(),
        "parameter_count": parameter_count(),
        "seeds": seeds,
        "evaluations": evaluations,
        "tables": tables,
        "promotion_gate": gate,
        "runtime_seconds": {"train_all_seeds": round(train_seconds, 2)},
    }
    (output / "bdm-v1-receipt.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baselines-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _resolve_output(args.output)
    receipt = run_study(args.corpus, args.baselines_receipt, output)
    for label, evaluation in receipt["evaluations"].items():
        print(
            f"{label}: ranking={evaluation['mean_ranking_correlation']:.4f} "
            f"regret={evaluation['mean_regret']:.6f} "
            f"value_error={evaluation['mean_action_value_error']:.6f} "
            f"recall={evaluation['useful_action_recall']:.4f}"
        )
    print("promotion gate:", json.dumps(receipt["promotion_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
