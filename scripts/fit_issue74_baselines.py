"""Fit and evaluate the Issue #74 linear baselines on the BDM-v1 corpus.

Fits the three preregistered baselines on TRAIN samples only, evaluates them
on DEV at independent causal-group level, and writes a write-once receipt
binding corpus, recipe, scaler, seed, and split identities. Baselines never
issue commands or step the plant.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.bdm_v1_corpus import load_partition_samples
from aeolus.habitat_v2.bdm_v1_evaluation import comparison_table
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast_issue74_baselines import (
    BASELINE_IDS,
    RECIPE_INPUT_FIELDS,
    LinearStateSpaceBaseline,
    RidgeBaseline,
    delta_labels,
    finite_catalogue_regret,
    fit_linear_state_space,
    fit_ridge_baseline,
    predict_sample,
    ranking_correlation,
    useful_action_precision_recall,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RIDGE_LAMBDA = 1e-3
SEED = "issue74-baselines-v1"
LEARNING_CURVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
STATISTICS_SEED = "issue74-bootstrap-v1"


class BaselineFitError(RuntimeError):
    """Raised when the baseline study cannot proceed safely."""


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise BaselineFitError("baseline output must live under the ignored out/ directory") from error
    if output.exists():
        raise BaselineFitError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def _family_decision_groups(samples: tuple[dict[str, Any], ...]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault((sample["family_id"], sample["decision_step"]), []).append(sample)
    return groups


def _evaluate(
    baseline: RidgeBaseline | LinearStateSpaceBaseline,
    dev_samples: tuple[dict[str, Any], ...],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    predicted_improving: list[bool] = []
    true_improving: list[bool] = []
    exposure_errors: list[float] = []
    for (_family_id, _decision), group in _family_decision_groups(dev_samples).items():
        predicted_deltas: list[float] = []
        true_deltas: list[float] = []
        for sample in group:
            prediction = predict_sample(baseline, sample)
            predicted = float(prediction["delta"]["safety_exposure"])
            true = float(delta_labels(sample)[1])
            predicted_deltas.append(predicted)
            true_deltas.append(true)
            predicted_improving.append(predicted < 0.0)
            true_improving.append(true < 0.0)
            exposure_errors.append(abs(predicted - true))
        first = group[0]
        rows.append(
            {
                "group_id": _group_of(first),
                "family_id": first["family_id"],
                "decision_step": first["decision_step"],
                "ranking_correlation": ranking_correlation(predicted_deltas, true_deltas),
                "regret": finite_catalogue_regret(predicted_deltas, true_deltas),
                "exposure_error": sum(exposure_errors[-len(group):]) / len(group),
            }
        )
    precision, recall = useful_action_precision_recall(predicted_improving, true_improving)
    return rows, {"useful_action_precision": precision, "useful_action_recall": recall}


def _group_of(sample: Mapping[str, Any]) -> str:
    return str(sample["group_id"])


def run_fit(corpus: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((corpus / "corpus-manifest.json").read_bytes())
    train = load_partition_samples(corpus, "TRAIN")
    dev = load_partition_samples(corpus, "DEV")
    started = time.perf_counter()
    baselines: dict[str, Any] = {}
    for baseline_id in ("action_agnostic_ridge", "action_conditioned_ridge"):
        baselines[baseline_id] = fit_ridge_baseline(
            train, baseline_id=baseline_id, ridge_lambda=RIDGE_LAMBDA, seed=SEED
        )
    baselines["controlled_linear_state_space"] = fit_linear_state_space(
        train, ridge_lambda=RIDGE_LAMBDA, seed=SEED
    )
    fit_seconds = time.perf_counter() - started

    learning_curve: dict[str, list[float]] = {}
    for fraction in LEARNING_CURVE_FRACTIONS:
        prefix = train[: max(1, int(len(train) * fraction))]
        model = fit_ridge_baseline(
            prefix, baseline_id="action_conditioned_ridge", ridge_lambda=RIDGE_LAMBDA, seed=SEED
        )
        rows, _ = _evaluate(model, dev)
        learning_curve[str(fraction)] = [
            sum(row["regret"] for row in rows) / len(rows)
        ]

    evaluate_started = time.perf_counter()
    evaluations: dict[str, Any] = {}
    row_sets: dict[str, list[dict[str, Any]]] = {}
    for baseline_id in BASELINE_IDS:
        rows, binary = _evaluate(baselines[baseline_id], dev)
        row_sets[baseline_id] = rows
        evaluations[baseline_id] = {
            "digest": baselines[baseline_id].digest,
            "mean_ranking_correlation": sum(row["ranking_correlation"] for row in rows) / len(rows),
            "mean_regret": sum(row["regret"] for row in rows) / len(rows),
            "mean_exposure_error": sum(row["exposure_error"] for row in rows) / len(rows),
            **binary,
        }
    evaluate_seconds = time.perf_counter() - evaluate_started

    tables = {
        metric: comparison_table(
            row_sets,
            metric,
            baseline_arm="action_agnostic_ridge",
            seed=STATISTICS_SEED,
        )
        for metric in ("ranking_correlation", "regret", "exposure_error")
    }
    receipt = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_baselines_receipt_v1",
        "corpus_digest": manifest["corpus_digest"],
        "corpus_manifest_sha256": hashlib.sha256(
            (corpus / "corpus-manifest.json").read_bytes()
        ).hexdigest(),
        "recipe": {
            "input_fields": list(RECIPE_INPUT_FIELDS),
            "recipe_sha256": hashlib.sha256(canonical_json_bytes(list(RECIPE_INPUT_FIELDS))).hexdigest(),
        },
        "split": {"fit": "TRAIN", "evaluate": "DEV", "train_samples": len(train), "dev_samples": len(dev)},
        "ridge_lambda": RIDGE_LAMBDA,
        "seed": SEED,
        "digests": {baseline_id: baselines[baseline_id].digest for baseline_id in BASELINE_IDS},
        "learning_curve_mean_regret": learning_curve,
        "evaluations": evaluations,
        "tables": tables,
        "runtime_seconds": {"fit": round(fit_seconds, 2), "evaluate": round(evaluate_seconds, 2)},
    }
    (output / "baselines-receipt.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _resolve_output(args.output)
    receipt = run_fit(args.corpus, output)
    for baseline_id, evaluation in receipt["evaluations"].items():
        print(
            f"{baseline_id}: ranking={evaluation['mean_ranking_correlation']:.4f} "
            f"regret={evaluation['mean_regret']:.6f} exposure_error={evaluation['mean_exposure_error']:.6f} "
            f"precision={evaluation['useful_action_precision']:.4f} recall={evaluation['useful_action_recall']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
