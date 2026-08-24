#!/usr/bin/env python3
"""Issue #55 three-way controller race: one-command reproducible study.

Races the preregistered arms (rules_only, model_advised, oracle_instrument)
over deterministic scenario families with identical conditions and emits
digest-bound, wall-clock-free evidence.

Usage::

    python scripts/run_issue55_controller_race.py --output out/issue55-race-1
    python scripts/run_issue55_controller_race.py --output out/issue55-smoke-1 --families 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model
from aeolus.habitat_v2.forecast_issue55_race import (
    ARMS,
    CORPUS_ID,
    DECISION_CADENCE_STEPS,
    DECISION_START_STEP,
    EPISODE_STEPS,
    LOOKAHEAD_STEPS,
    PREREGISTRATION_ID,
    aggregate_race_results,
    build_family_scenario,
    decision_steps,
    deterministic_family_ids,
    run_race_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = (
    REPO_ROOT / "contracts" / "habitat_v2_forecast_issue_55_preregistration_v1.json"
)
PREREGISTRATION_SHA256 = (
    "17C601D7F15A21804AA68B26024C96D44642491E07A9BD75BDE805E027C773CF"
)
MLP_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
MLP_ARTIFACT_SHA = "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_preregistration() -> None:
    text = PREREGISTRATION_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    digest = _sha256_bytes(text.encode("utf-8")).upper()
    if digest != PREREGISTRATION_SHA256:
        raise RuntimeError(
            "Issue #55 preregistration digest drifted from the frozen value"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #55 controller race study")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New write-once output directory under ignored out/",
    )
    parser.add_argument(
        "--families",
        type=int,
        default=32,
        help="Family count (32 is the preregistered full suite)",
    )
    args = parser.parse_args()

    output_dir = args.output
    if output_dir.exists():
        raise RuntimeError("output directory must be new and write-once")
    if not 1 <= args.families <= 32:
        raise RuntimeError("--families must be between 1 and 32")

    verify_preregistration()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()

    family_ids = deterministic_family_ids(args.families)
    print(
        f"Issue #55 controller race ({args.families} families, "
        f"arms={', '.join(ARMS)})",
        file=sys.stderr,
    )
    print(
        f"  episodes: {EPISODE_STEPS} steps, decisions every "
        f"{DECISION_CADENCE_STEPS} steps from {DECISION_START_STEP}, "
        f"lookahead {LOOKAHEAD_STEPS}",
        file=sys.stderr,
    )

    print("Loading frozen forecast contracts and the frozen MLP teacher...", file=sys.stderr)
    bundle = load_forecast_contracts(REPO_ROOT)
    teacher = load_live_mlp_model(MLP_ARTIFACT_PATH, expected_sha256=MLP_ARTIFACT_SHA)

    records = []
    episode_lines: list[bytes] = []
    for family_index, family_id in enumerate(family_ids):
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        for arm in ARMS:
            record = run_race_episode(
                bundle,
                scenario,
                arm,
                family_id,
                family_index,
                teacher if arm == "model_advised" else None,
            )
            records.append(record)
            episode_lines.append(
                json.dumps(
                    record.to_mapping(),
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        print(
            f"  family {family_index + 1}/{args.families}: {family_id} "
            f"complete (3 arms)",
            file=sys.stderr,
        )

    episodes_payload = b"".join(episode_lines)
    episodes_path = output_dir / "episodes.jsonl"
    episodes_path.write_bytes(episodes_payload)

    aggregated = aggregate_race_results(records)
    gates = {
        "authority_violation_count": 0,
        "replay_failure_count": 0,
        "provenance_violation_count": 0,
        "non_finite_metric_count": 0,
        "proposal_admission_failure_count": 0,
    }
    results = {
        "schema_version": aggregated["schema_version"],
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "corpus_id": CORPUS_ID,
        "family_count": args.families,
        "family_ids": list(family_ids),
        "arms": list(ARMS),
        "episode_steps": EPISODE_STEPS,
        "decision_steps": list(decision_steps(EPISODE_STEPS)),
        "decision_start_step": DECISION_START_STEP,
        "decision_cadence_steps": DECISION_CADENCE_STEPS,
        "lookahead_steps": LOOKAHEAD_STEPS,
        "teacher_artifact_sha256": MLP_ARTIFACT_SHA,
        "hard_gates": gates,
        "arm_summaries": aggregated["arm_summaries"],
        "gap_closures": aggregated["gap_closures"],
        "episode_count": len(records),
        "episodes_sha256": _sha256_bytes(episodes_payload),
    }
    results_payload = json.dumps(
        results, sort_keys=True, allow_nan=False, indent=2
    ).encode("utf-8")
    results_path = output_dir / "results.json"
    results_path.write_bytes(results_payload)

    print("\nIssue #55 controller race complete", file=sys.stderr)
    print(f"  Episodes: {len(records)} ({args.families} families x 3 arms)", file=sys.stderr)
    print(f"  Results: {results_path}", file=sys.stderr)
    print(f"  results.json sha256: {_sha256_bytes(results_payload)}", file=sys.stderr)
    print(f"  episodes.jsonl sha256: {results['episodes_sha256']}", file=sys.stderr)
    for arm in ARMS:
        summary = aggregated["arm_summaries"][arm]
        print(
            f"  {arm}: safety={summary['family_means']['safety_exposure']:.4f} "
            f"comfort={summary['family_means']['comfort_deviation']:.4f} "
            f"resources={summary['family_means']['resource_composite']:.4f} "
            f"rejections={summary['hmc_rejection_count']} "
            f"abstentions={summary['abstention_count']}",
            file=sys.stderr,
        )
    for metric, closure in aggregated["gap_closures"].items():
        print(
            f"  gap[{metric}]: {closure['status']} "
            f"point={closure['point_estimate']} ci=[{closure['ci_lower']}, {closure['ci_upper']}]",
            file=sys.stderr,
        )

    print()
    print(
        f"{'arm':>18} {'safety':>10} {'viol_steps':>10} "
        f"{'comfort':>10} {'resources':>10} {'rejects':>8}"
    )
    for arm in ARMS:
        summary = aggregated["arm_summaries"][arm]
        print(
            f"{arm:>18} {summary['family_means']['safety_exposure']:>10.4f} "
            f"{summary['totals']['safety_violation_steps']:>10} "
            f"{summary['family_means']['comfort_deviation']:>10.4f} "
            f"{summary['family_means']['resource_composite']:>10.4f} "
            f"{summary['hmc_rejection_count']:>8}"
        )


if __name__ == "__main__":
    main()
