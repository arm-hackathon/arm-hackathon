"""Check whether the live demo telemetry has drifted from training stats.

    uv run --locked --python 3.11 --extra dev python scripts/check_habitat_v2_mlp_drift.py

Rebuilds the demo's 16-step anchor window and scores it against the
training-distribution statistics stored inside the hash-pinned model
artifact. Diagnostic only — HMC remains the sole command authority.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aeolus.habitat_v2.forecast.drift_monitor import (  # noqa: E402
    assess_window,
    build_demo_window,
    load_window_stats,
    render_report,
)
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model  # noqa: E402

ARTIFACT = (
    REPO_ROOT
    / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
ARTIFACT_SHA256 = (
    "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
)


def main() -> int:
    model = load_live_mlp_model(ARTIFACT, expected_sha256=ARTIFACT_SHA256)
    mean, std = load_window_stats(ARTIFACT)
    window = build_demo_window(REPO_ROOT, model)
    report = assess_window(window, mean, std)
    print("AEOLUS drift check — demo anchor window vs training distribution")
    print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
