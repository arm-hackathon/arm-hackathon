"""Run the Issue #77 six-arm HMC-filtered closed-loop development study.

Refits and digest-binds every upstream component (Issue #73 screens, Issue #74
baseline, Issue #75 seeds, Issue #76 abstention layer), runs all six arms on
identical DEV families behind the HMC, evaluates the preregistered gate order
at causal-group level, and writes a write-once receipt. On gate failure the
candidate packet is withheld and the blind protocol draft stays WITHHELD and
UNAUTHORIZED; the blind population remains sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.bdm_v1_corpus import delta_labels, trajectory_labels
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast_issue73_ablations import (
    fit_linear_screen,
    fit_mlp_screen,
    screen_features_from_sample,
)
from aeolus.habitat_v2.forecast_issue74_baselines import fit_ridge_baseline
from aeolus.habitat_v2.forecast_issue75_bdm import input_tensor, train_model
from aeolus.habitat_v2.forecast_issue76_abstention import CalibrationLayer
from aeolus.habitat_v2.forecast_issue77_closed_loop import (
    evaluate_gates,
    make_calibrated_bdm_policy,
    make_guard_screen_policy,
    make_linear_policy,
    make_null_policy,
    run_closed_loop_episode,
)
from aeolus.habitat_v2.bdm_v1_families import (
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_closed_loop_preregistration_v1.json"
ABLATION_PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_ablation_preregistration_v1.json"
ABSTENTION_CONTRACT_PATH = (
    REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_abstention_thresholds_v1.json"
)
ARM_IDS = (
    "hmc_rules_only",
    "hold_current_command",
    "c8_guard_ridge",
    "c9_guard_mlp",
    "linear_action_conditioned_ridge",
    "calibrated_bdm_v1",
)


class ClosedLoopStudyError(RuntimeError):
    """Raised when the closed-loop study cannot proceed under its preregistration."""


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise ClosedLoopStudyError("study output must live under the ignored out/ directory") from error
    if output.exists():
        raise ClosedLoopStudyError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def _load_samples(corpus: Path, partition: str) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    shard_dir = corpus / "shards"
    if not shard_dir.is_dir():
        raise ClosedLoopStudyError(f"corpus shards missing under {corpus}")
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
        raise ClosedLoopStudyError(f"no {partition} samples in corpus {corpus}")
    return tuple(samples)


def run_study(corpus: Path, output: Path) -> dict[str, Any]:
    prereg_raw = PREREG_PATH.read_bytes()
    prereg = json.loads(prereg_raw)
    if not prereg["freeze"]["frozen_before_final_runs"]:
        raise ClosedLoopStudyError("closed-loop preregistration is not frozen")
    abstention_raw = ABSTENTION_CONTRACT_PATH.read_bytes()
    abstention = json.loads(abstention_raw)
    if hashlib.sha256(abstention_raw).hexdigest() != prereg["bindings"]["abstention_contract_sha256"]:
        raise ClosedLoopStudyError("abstention contract drifted from the preregistration")
    layer_map = abstention["layer"]
    if layer_map["digest"] != prereg["bindings"]["abstention_layer_digest"]:
        raise ClosedLoopStudyError("abstention layer digest drifted from the preregistration")
    layer = CalibrationLayer(
        np.asarray(layer_map["delta_offsets"], dtype=np.float64),
        np.asarray(layer_map["trajectory_offsets"], dtype=np.float64),
        float(layer_map["max_staleness"]),
        float(layer_map["min_mask_fraction"]),
        float(layer_map["seed_disagreement"]),
        float(layer_map["interval_width"]),
        layer_map["digest"],
    )

    train_samples = _load_samples(corpus, "TRAIN")
    ablation_prereg = json.loads(ABLATION_PREREG_PATH.read_bytes())
    screen_config = ablation_prereg["screen"]
    screen_seed = str(screen_config["seeds"][0])
    screen_features = np.stack(
        [screen_features_from_sample(sample) for sample in train_samples]
    )
    screen_targets = np.asarray(
        [float(delta_labels(sample)[1]) for sample in train_samples], dtype=np.float64
    )
    ridge_screen = fit_linear_screen(
        screen_features,
        screen_targets,
        ridge_lambda=float(screen_config["ridge_lambda"]),
        seed=screen_seed,
    )
    mlp_screen = fit_mlp_screen(
        screen_features,
        screen_targets,
        seed=screen_seed,
        epochs=int(screen_config["mlp"]["epochs"]),
        learning_rate=float(screen_config["mlp"]["learning_rate"]),
    )
    if ridge_screen.digest != prereg["bindings"]["screen_digests"]["ridge"]:
        raise ClosedLoopStudyError("ridge screen digest drifted from the preregistration")
    if mlp_screen.digest != prereg["bindings"]["screen_digests"]["mlp"]:
        raise ClosedLoopStudyError("mlp screen digest drifted from the preregistration")
    linear_baseline = fit_ridge_baseline(
        train_samples,
        baseline_id="action_conditioned_ridge",
        ridge_lambda=0.001,
        seed="issue74-baselines-v1",
    )
    if linear_baseline.digest != prereg["bindings"]["linear_conditioned_ridge_digest"]:
        raise ClosedLoopStudyError("linear baseline digest drifted from the preregistration")

    train_tensors = np.stack([input_tensor(sample) for sample in train_samples])
    train_trajectories = np.stack([trajectory_labels(sample) for sample in train_samples])
    train_deltas = np.stack([delta_labels(sample) for sample in train_samples])
    dev_samples = _load_samples(corpus, "DEV")
    dev_tensors = np.stack([input_tensor(sample) for sample in dev_samples])
    dev_deltas = np.stack([delta_labels(sample) for sample in dev_samples])
    models = []
    for seed_name, expected_digest in prereg["bindings"]["bdm_seed_digests"].items():
        model = train_model(
            train_tensors,
            train_trajectories,
            train_deltas,
            dev_tensors,
            dev_deltas,
            seed=seed_name,
            learning_rate=0.001,
            batch_size=256,
            epochs_max=40,
            patience=5,
        )
        if model.digest != expected_digest:
            raise ClosedLoopStudyError(f"seed {seed_name} digest drifted from the preregistration")
        models.append(model)

    bundle = load_forecast_contracts(REPO_ROOT)
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    registry = json.loads(
        (REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json").read_bytes()
    )
    config = GeneratorConfig()
    plans = {plan.group_index: plan for plan in assign_partitions(config)}
    dev_rows = [row for row in registry["groups"] if row["partition"] == "DEV"]

    policies = {
        "hmc_rules_only": make_null_policy(),
        "hold_current_command": make_null_policy(),
        "c8_guard_ridge": make_guard_screen_policy(ridge_screen, include_guard=False),
        "c9_guard_mlp": make_guard_screen_policy(mlp_screen, include_guard=True),
        "linear_action_conditioned_ridge": make_linear_policy(linear_baseline),
        "calibrated_bdm_v1": make_calibrated_bdm_policy(models, layer),
    }
    decision_steps = tuple(prereg["episode"]["decision_steps"])
    records_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_IDS}
    latencies: list[float] = []
    started = time.perf_counter()
    for row in dev_rows:
        for variant_index, variant in enumerate(("a", "b")):
            definition, scenario = build_family(
                config,
                plans[row["group_index"]],
                variant_index,
                int(row["attempts"][variant]),
                base_data,
                manifest,
            )
            if definition.scenario_sha256 != row["scenario_sha256"][variant]:
                raise ClosedLoopStudyError(
                    f"family {definition.family_id} drifted from the registry"
                )
            for arm in ARM_IDS:
                record = run_closed_loop_episode(
                    bundle,
                    scenario,
                    policies[arm],
                    definition.family_id,
                    definition.scenario_sha256,
                    decision_steps,
                )
                record["group_id"] = row["group_id"]
                record["arm"] = arm
                records_by_arm[arm].append(record)
                if arm == "calibrated_bdm_v1":
                    latencies.extend(entry["latency_ms"] for entry in record["lineage"])
                print(
                    f"  {definition.family_id} {arm}: exposure={record['safety_exposure']:.6f} "
                    f"proposals={record['proposal_count']} rejected={record['rejected_proposal_count']}",
                    flush=True,
                )
    runtime_seconds = time.perf_counter() - started

    gates = evaluate_gates(records_by_arm, prereg)
    latency_p99 = float(np.quantile(latencies, 0.99)) if latencies else 0.0
    latency_gate = latency_p99 <= float(prereg["gates"]["model_path_latency_p99_ms_maximum"])

    summary = {
        arm: {
            "mean_safety_exposure": sum(r["safety_exposure"] for r in records) / len(records),
            "mean_comfort_deviation": sum(r["comfort_deviation"] for r in records) / len(records),
            "mean_resource_composite": sum(r["resource_composite"] for r in records) / len(records),
            "total_proposals": sum(r["proposal_count"] for r in records),
            "total_rejected": sum(r["rejected_proposal_count"] for r in records),
            "total_abstentions": sum(r["abstention_count"] for r in records),
        }
        for arm, records in records_by_arm.items()
    }
    receipt = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_closed_loop_receipt_v1",
        "preregistration_sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "abstention_contract_sha256": hashlib.sha256(abstention_raw).hexdigest(),
        "corpus_digest": prereg["bindings"]["corpus_digest"],
        "family_count": len(dev_rows) * 2,
        "record_count": sum(len(records) for records in records_by_arm.values()),
        "arms": summary,
        "gates": gates,
        "latency": {
            "p99_ms": latency_p99,
            "ceiling_ms": float(prereg["gates"]["model_path_latency_p99_ms_maximum"]),
            "gate_passed": latency_gate,
        },
        "candidate_packet": None,
        "candidate_packet_status": "withheld_gate_failed"
        if not gates["all_gates_passed"]
        else "frozen",
        "blind_protocol_draft_status": "WITHHELD_UNAUTHORIZED",
        "blind_population": "sealed",
        "runtime_seconds": round(runtime_seconds, 2),
        "records": records_by_arm,
    }
    (output / "closed-loop-receipt.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _resolve_output(args.output)
    receipt = run_study(args.corpus, output)
    print(json.dumps(receipt["gates"], indent=1))
    print("latency:", json.dumps(receipt["latency"]))
    print("packet:", receipt["candidate_packet_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
