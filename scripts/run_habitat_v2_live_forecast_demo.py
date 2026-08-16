from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.habitat_v2.forecast.live_demo import (
    load_live_ridge_model,
    run_live_forecast_demo,
)
from aeolus.habitat_v2.forecast.live_demo_report import write_live_forecast_report

MODEL_SHA256 = "a6e4ef34fc837bb6539a84e20d015bbd7bbfe4e9fd5a6fc74e3f0217bd978d9a"
SOURCE_FOUNDATION_GIT_COMMIT = "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3"


def _arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the forecast-only Habitat V2 model at step 16, let HMC execute "
            "one operator-selected action, and write a self-contained report."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root
        / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz",
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
        "--output",
        type=Path,
        default=repo_root
        / "out/habitat-v2-live-forecast-demo/artifacts/live-run-v1",
    )
    parser.add_argument(
        "--integration-source-commit",
        help="Full Git commit for a committed CI run; omit for a local dirty run.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = arguments.repo_root.resolve()
    model = load_live_ridge_model(
        arguments.model,
        expected_sha256=MODEL_SHA256,
    )
    result = run_live_forecast_demo(
        root,
        model,
        selected_action_id=arguments.selected_action,
    )
    sources = (
        root / "src/aeolus/habitat_v2/forecast/live_demo.py",
        root / "src/aeolus/habitat_v2/forecast/live_demo_report.py",
        root / "scripts/run_habitat_v2_live_forecast_demo.py",
        root / "scripts/verify_habitat_v2_live_forecast_demo.py",
        root / "tests/habitat_v2/test_forecast_live_demo.py",
        root / ".github/workflows/habitat-v2-live-forecast-arm64.yml",
        root / "artifacts/demo-only/habitat-v2-forecast/README.md",
        root / "artifacts/demo-only/habitat-v2-forecast/training-report.json",
        root / "artifacts/demo-only/habitat-v2-forecast/training-receipt.json",
    )
    artifact = write_live_forecast_report(
        root,
        result,
        arguments.output,
        source_foundation_git_commit=SOURCE_FOUNDATION_GIT_COMMIT,
        integration_source_git_commit=arguments.integration_source_commit,
        source_paths=sources,
    )
    print(
        json.dumps(
            {
                **artifact,
                "model_sha256": result.model_artifact_sha256,
                "forecast_history_steps": list(result.forecast_history_steps),
                "truth_steps": list(result.truth_steps),
                "candidate_action_count": len(result.candidate_forecasts),
                "distinct_prediction_count": len(
                    {
                        item.prediction_f32.tobytes()
                        for item in result.candidate_forecasts
                    }
                ),
                "selected_action_id": result.selected_action_id,
                "arbitration_disposition": result.arbitration_disposition,
                "hmc_is_sole_actuator_authority": (
                    result.hmc_is_sole_actuator_authority
                ),
                "trace_sha256": result.trace_sha256,
                "replay_committed_steps": result.replay_committed_steps,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
