"""Run the Issue #73 guard-attribution ablation study on the DEV partition.

The ablation preregistration must already be frozen; this command refuses to
run otherwise. Screens are fit on TRAIN corpus samples only; every arm then
runs identical fresh DEV exogenous traces behind the HMC. The receipt is
written once under an ignored ``out/`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.bdm_v1_corpus import load_partition_samples
from aeolus.habitat_v2.bdm_v1_evaluation import comparison_table, paired_group_differences
from aeolus.habitat_v2.bdm_v1_families import (
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast_issue73_ablations import (
    ABLATION_ARMS,
    NON_PROPOSING_ARMS,
    fit_linear_screen,
    fit_mlp_screen,
    run_ablation_episode,
    screen_features_from_sample,
)
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_ablation_preregistration_v1.json"
REGISTRY_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json"


class AblationStudyError(RuntimeError):
    """Raised when the study cannot run under its preregistration."""


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise AblationStudyError("study output must live under the ignored out/ directory") from error
    if output.exists():
        raise AblationStudyError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def run_study(corpus: Path, output: Path, max_families: int | None) -> dict[str, Any]:
    prereg_raw = PREREG_PATH.read_bytes()
    prereg = json.loads(prereg_raw)
    if not prereg["freeze"]["frozen_before_study_runs"]:
        raise AblationStudyError("ablation preregistration is not frozen")
    registry = json.loads(REGISTRY_PATH.read_bytes())
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    bundle = load_forecast_contracts(REPO_ROOT)

    train_samples = load_partition_samples(corpus, prereg["partitions"]["screen_fit"])
    features = np.stack(
        [screen_features_from_sample(sample) for sample in train_samples]
    )
    targets = np.asarray(
        [float(sample["action_minus_hold"]["safety_exposure"]) for sample in train_samples],
        dtype=np.float64,
    )
    screen_config = prereg["screen"]
    seed = str(screen_config["seeds"][0])
    ridge = fit_linear_screen(
        features, targets, ridge_lambda=float(screen_config["ridge_lambda"]), seed=seed
    )
    mlp = fit_mlp_screen(
        features,
        targets,
        seed=seed,
        epochs=int(screen_config["mlp"]["epochs"]),
        learning_rate=float(screen_config["mlp"]["learning_rate"]),
    )
    screens = {"c9_learned_screen_without_guard": mlp, "c9_full": mlp}
    print(
        f"screens fit on {len(train_samples)} TRAIN samples "
        f"(ridge {ridge.digest[:12]}, mlp {mlp.digest[:12]})",
        file=sys.stderr,
    )

    config = GeneratorConfig()
    plans = {plan.group_index: plan for plan in assign_partitions(config)}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ABLATION_ARMS}
    records: list[dict[str, Any]] = []
    dev_rows = [row for row in registry["groups"] if row["partition"] == "DEV"]
    families: list[tuple[dict[str, Any], int, str]] = []
    for row in dev_rows:
        for variant_index, variant in enumerate(("a", "b")):
            families.append((row, variant_index, variant))
    if max_families is not None:
        families = families[:max_families]
    for row, variant_index, variant in families:
        definition, scenario = build_family(
            config,
            plans[row["group_index"]],
            variant_index,
            int(row["attempts"][variant]),
            base_data,
            manifest,
        )
        if definition.scenario_sha256 != row["scenario_sha256"][variant]:
            raise AblationStudyError(f"family {definition.family_id} drifted from the registry")
        for arm in ABLATION_ARMS:
            screen = screens.get(arm)
            if arm in NON_PROPOSING_ARMS or arm == "c8_guard_only":
                screen = None
            record = run_ablation_episode(
                bundle,
                scenario,
                arm,
                definition.family_id,
                screen=screen,
                scenario_sha256=definition.scenario_sha256,
            )
            record["group_id"] = row["group_id"]
            record["partition"] = row["partition"]
            records.append(record)
            rows_by_arm[arm].append(record)
            print(
                f"  {definition.family_id} {arm}: exposure={record['safety_exposure']:.6f} "
                f"proposals={record['proposal_count']}",
                file=sys.stderr,
            )

    statistics = prereg["statistics"]
    tables = {
        metric: comparison_table(
            rows_by_arm,
            metric,
            baseline_arm="hold_current_command",
            seed=str(statistics["seed"]),
            resamples=int(statistics["bootstrap_resamples"]),
            alpha=float(statistics["alpha"]),
        )
        for metric in [prereg["metrics"]["primary"], *prereg["metrics"]["secondary"]]
    }
    attribution = {
        difference_name: {
            metric: {
                "group_differences": paired_group_differences(
                    rows_by_arm[arm], rows_by_arm["hold_current_command"], metric
                )
            }
            for metric in [prereg["metrics"]["primary"]]
        }
        for difference_name, arm in (
            ("guard_credit", "c8_guard_only"),
            ("learned_credit", "c9_learned_screen_without_guard"),
            ("hybrid_credit", "c9_full"),
        )
    }
    receipt = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_ablation_receipt_v1",
        "preregistration_sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "registry_sha256": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
        "screen_fit_partition": prereg["partitions"]["screen_fit"],
        "study_partition": prereg["partitions"]["study"],
        "train_sample_count": len(train_samples),
        "ridge_digest": ridge.digest,
        "mlp_digest": mlp.digest,
        "family_count": len(families),
        "record_count": len(records),
        "tables": tables,
        "attribution": attribution,
        "records": records,
    }
    (output / "ablation-receipt.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-families", type=int, default=None)
    args = parser.parse_args()
    output = _resolve_output(args.output)
    receipt = run_study(args.corpus, output, args.max_families)
    primary = receipt["tables"]["safety_exposure"]["arms"]
    for arm, row in primary.items():
        print(
            f"{arm}: mean={row['mean']:.6f} paired_diff={row['paired_difference_mean']:.6f} "
            f"CI=[{row['paired_difference_ci_lower']:.6f}, {row['paired_difference_ci_upper']:.6f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
