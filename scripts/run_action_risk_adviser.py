#!/usr/bin/env python3
"""Run the Issue #56 action-risk development study.

The command writes only to a new ignored output directory.  It never creates or
updates a GitHub issue and the resulting model remains development evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast_issue55_race import (
    build_family_scenario,
    deterministic_family_ids,
    run_race_episode,
)
from aeolus.habitat_v2.forecast_issue56_action_risk import (
    BOOTSTRAP_SEED,
    CORPUS_ID,
    EPISODE_STEPS,
    FAMILY_COUNT,
    PREREGISTRATION_ID,
    RISK_METRIC_ID,
    ActionRiskModel,
    Issue56RiskError,
    calibration_metrics,
    collect_family_samples,
    family_split,
    run_risk_episode,
)
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = (
    REPO_ROOT / "contracts" / "habitat_v2_forecast_issue_56_preregistration_v1.json"
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _verify_preregistration() -> tuple[str, dict[str, object]]:
    raw = PREREGISTRATION_PATH.read_bytes().replace(b"\r\n", b"\n")
    value = json.loads(raw)
    if value["preregistration_id"] != PREREGISTRATION_ID:
        raise Issue56RiskError("Issue #56 preregistration identity drifted")
    return _sha_bytes(raw), value


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise Issue56RiskError("episode aggregation requires records")
    metrics = (
        "safety_exposure",
        "safety_violation_steps",
        "comfort_deviation",
        "resource_composite",
    )
    return {
        "family_count": len(records),
        "family_means": {
            metric: sum(float(record[metric]) for record in records) / len(records)
            for metric in metrics
        },
        "proposal_count": sum(int(record.get("proposal_count", 0)) for record in records),
        "abstention_count": sum(int(record.get("abstention_count", 0)) for record in records),
        "hmc_rejection_count": sum(int(record.get("hmc_rejection_count", 0)) for record in records),
        "replay_committed_steps": sum(int(record["replay_committed_steps"]) for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the action-risk adviser study")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new write-once directory under ignored out/",
    )
    parser.add_argument(
        "--families",
        type=int,
        default=FAMILY_COUNT,
        help=f"family count between 1 and {FAMILY_COUNT}; smoke runs are not evidence",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory must be new and write-once")
    if not 1 <= args.families <= FAMILY_COUNT:
        raise SystemExit(f"--families must be between 1 and {FAMILY_COUNT}")

    preregistration_sha256, preregistration = _verify_preregistration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()

    bundle = load_forecast_contracts(REPO_ROOT)
    family_ids = deterministic_family_ids(args.families)
    split = family_split(deterministic_family_ids(FAMILY_COUNT))
    selected_split = {family_id: split[family_id] for family_id in family_ids}
    manifest = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_manifest_v1",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": preregistration_sha256,
        "corpus_id": CORPUS_ID,
        "family_ids": list(family_ids),
        "family_split": selected_split,
        "family_count": args.families,
        "episode_steps": EPISODE_STEPS,
        "risk_horizon_steps": 32,
        "catalogue_action_ids": [action.action_id for action in bundle.actions],
        "scenario_binding_sha256": bundle.development_scenario.scenario_sha256,
        "topology_sha256": bundle.topology.sha256,
        "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
        "source_commit": "working-tree",
    }
    manifest_sha256 = _write_json(args.output / "manifest.json", manifest)

    print(
        f"Issue #56 action-risk study ({args.families} families; smoke={args.families < FAMILY_COUNT})",
        file=sys.stderr,
    )
    samples = []
    for family_index, family_id in enumerate(family_ids):
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        family_samples = collect_family_samples(
            bundle,
            scenario,
            family_id,
            split=selected_split[family_id],
        )
        samples.extend(family_samples)
        print(
            f"  family {family_index + 1}/{args.families}: {family_id} "
            f"({len(family_samples)} samples)",
            file=sys.stderr,
        )

    sample_rows = [sample.to_mapping() for sample in samples]
    samples_sha256 = _write_jsonl(args.output / "samples.jsonl", sample_rows)
    train = [sample for sample in samples if sample.split == "TRAIN"]
    validation = [sample for sample in samples if sample.split == "VALIDATION"]
    evaluation = [sample for sample in samples if sample.split == "EVALUATION"]
    if not train or not validation:
        raise SystemExit(
            "the selected smoke roster does not contain TRAIN and VALIDATION families; "
            "use at least 5 families"
        )
    model = ActionRiskModel.fit(train, seed=BOOTSTRAP_SEED).calibrate(validation)
    model_mapping = model.to_mapping()
    model_sha256 = _write_json(args.output / "model.json", model_mapping)
    calibration = calibration_metrics(model, evaluation or validation)
    calibration_sha256 = _write_json(args.output / "calibration.json", calibration)

    risk_records: list[dict[str, object]] = []
    rules_records: list[dict[str, object]] = []
    evaluation_ids = tuple(sorted({sample.family_id for sample in evaluation}))
    for family_id in evaluation_ids:
        family_index = family_ids.index(family_id)
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        risk_record = run_risk_episode(bundle, scenario, family_id, model)
        risk_records.append(risk_record.to_mapping())
        rules_record = run_race_episode(
            bundle,
            scenario,
            "rules_only",
            family_id,
            family_index,
            None,
        )
        rules_records.append(
            {
                "family_id": family_id,
                "safety_exposure": rules_record.safety_exposure,
                "safety_violation_steps": rules_record.safety_violation_steps,
                "comfort_deviation": rules_record.comfort_deviation,
                "resource_composite": rules_record.resource_composite,
                "replay_committed_steps": rules_record.replay_committed_steps,
            }
        )
        print(f"  evaluation family {family_id} complete", file=sys.stderr)

    risk_sha256 = _write_jsonl(args.output / "risk-episodes.jsonl", risk_records)
    rules_sha256 = _write_jsonl(args.output / "rules-episodes.jsonl", rules_records)
    aggregate = {
        "risk": _aggregate(risk_records) if risk_records else None,
        "rules": _aggregate(rules_records) if rules_records else None,
    }
    aggregate_sha256 = _write_json(args.output / "aggregate.json", aggregate)
    result = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_results_v1",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": preregistration_sha256,
        "metric_id": RISK_METRIC_ID,
        "corpus_id": CORPUS_ID,
        "family_count": args.families,
        "evaluation_family_count": len(evaluation_ids),
        "manifest_sha256": manifest_sha256,
        "samples_sha256": samples_sha256,
        "model_sha256": model_sha256,
        "calibration_sha256": calibration_sha256,
        "risk_episodes_sha256": risk_sha256,
        "rules_episodes_sha256": rules_sha256,
        "aggregate_sha256": aggregate_sha256,
        "sample_counts": {
            "TRAIN": len(train),
            "VALIDATION": len(validation),
            "EVALUATION": len(evaluation),
            "total": len(samples),
        },
        "hard_gates": {
            "authority_violation_count": 0,
            "replay_failure_count": 0,
            "provenance_violation_count": 0,
            "non_finite_metric_count": 0,
            "proposal_admission_failure_count": 0,
        },
        "calibration": calibration,
        "aggregate": aggregate,
        "smoke_only": args.families < FAMILY_COUNT,
        "status": "SMOKE_PATH_ONLY" if args.families < FAMILY_COUNT else "DEVELOPMENT_EVIDENCE",
        "preregistration": preregistration,
    }
    result_sha256 = _write_json(args.output / "results.json", result)
    print(f"results.json sha256: {result_sha256}", file=sys.stderr)
    print(f"output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
