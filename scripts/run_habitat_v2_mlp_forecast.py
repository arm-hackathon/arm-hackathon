from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aeolus.habitat_v2.forecast.live_mlp_demo import (
    load_live_mlp_model,
    run_live_mlp_forecast_demo,
)

# Frozen identity of artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz
MODEL_SHA256 = "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
TRAINING_RUN_ID = "full-v1-20260818-a"
CLOSED_LOOP_EVIDENCE_PR = "https://github.com/arm-hackathon/arm-hackathon/pull/40"


def _arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the action-aware MLP artifact associated with recorded Historical "
            "V2 development run full-v1-20260818-a at step 16 of the development "
            "scenario: forecast every catalogue action, let HMC execute one "
            "operator-selected action, and compare each forecast against realized "
            "simulator truth. Development evidence only; HMC is the sole actuator "
            "authority."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root
        / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz",
    )
    parser.add_argument(
        "--selected-action",
        default="normal-occupied-v1",
        choices=(
            "normal-occupied-v1",
            "normal-eva_transition-v1",
            "normal-contingency-v1",
            "normal-dormant-v1",
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON receipt instead of the human-readable summary.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = arguments.repo_root.resolve()
    model = load_live_mlp_model(arguments.model, expected_sha256=MODEL_SHA256)
    result = run_live_mlp_forecast_demo(
        root, model, selected_action_id=arguments.selected_action,
    )

    truth = np.asarray(result.truth_f32, dtype=np.float64)
    candidates = []
    for item in result.candidate_forecasts:
        prediction = np.asarray(item.prediction_f32, dtype=np.float64)
        candidates.append(
            {
                "action_id": item.action_id,
                "forecast_mae_vs_realized_truth": float(
                    np.abs(prediction - truth).mean()
                ),
            }
        )
    receipt = {
        "model": {
            "kind": result.model_kind,
            "artifact_sha256": result.model_artifact_sha256,
            "training_run_id": TRAINING_RUN_ID,
            "held_out_normalized_mae": 0.1146,
            "held_out_metric_status": (
                "HISTORICALLY_REPORTED_NOT_INDEPENDENTLY_VERIFIABLE"
            ),
            "closed_loop_evidence": CLOSED_LOOP_EVIDENCE_PR,
        },
        "run": {
            "selected_action_id": result.selected_action_id,
            "forecast_completed_step": result.forecast_completed_step,
            "forecast_history_steps": list(result.forecast_history_steps),
            "truth_steps": list(result.truth_steps),
            "candidate_action_count": len(result.candidate_forecasts),
            "distinct_prediction_count": len(
                {
                    item.prediction_f32.tobytes()
                    for item in result.candidate_forecasts
                }
            ),
            "arbitration_disposition": result.arbitration_disposition,
            "hmc_is_sole_actuator_authority": (
                result.hmc_is_sole_actuator_authority
            ),
            "terminal_status": result.terminal_status,
            "trace_sha256": result.trace_sha256,
            "replay_committed_steps": result.replay_committed_steps,
        },
        "per_candidate_forecast_error": candidates,
        "claim_boundary": (
            "development evidence only; not qualification; not deployment; "
            "model output is advisory and HMC arbitration is the sole authority"
        ),
    }
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    width = max(len(item["action_id"]) for item in candidates)
    print("AEOLUS Habitat V2 - action-aware forecast run")
    print(f"Model: {result.model_kind} (recorded training run {TRAINING_RUN_ID}, "
          "historically reported held-out error 0.1146, lower is better)")
    print()
    print("At step 16 the model forecast the next 8 habitat states for each")
    print("candidate action. Average forecast error against what the")
    print("simulator actually did (lower = predicted reality better):")
    for item in sorted(
        candidates, key=lambda entry: entry["forecast_mae_vs_realized_truth"]
    ):
        marker = "  <- selected by operator" if (
            item["action_id"] == result.selected_action_id
        ) else ""
        print(
            f"  {item['action_id']:<{width}}  "
            f"{item['forecast_mae_vs_realized_truth']:8.2f}{marker}"
        )
    print()
    print(
        f"HMC (deterministic safety controller) reviewed the operator-selected "
        f"action: {result.arbitration_disposition}."
    )
    print(
        f"All {result.replay_committed_steps} steps completed; control trace "
        "replayed bit-for-bit."
    )
    print("The model only advises. HMC is the sole command authority.")
    print()
    print(f"Trace hash: {result.trace_sha256}")
    print("Boundary: development evidence only - not qualification or deployment.")
    print(f"Full closed-loop evidence: {CLOSED_LOOP_EVIDENCE_PR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
