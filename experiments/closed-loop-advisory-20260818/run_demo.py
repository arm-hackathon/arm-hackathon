"""One-command closed-loop demo: canonical HMC vs forecast-advised HMC.

Runs ONE paired scenario twice with identical scenario, noise seed and reset
nonce: once with the canonical HMC policy alone, once with the development
forecaster advising.  Prints the physical-outcome comparison.

Usage (from the repository root, Python 3.10+ with numpy and torch):

    pip install -e . && pip install torch
    python experiments/closed-loop-advisory-20260818/run_demo.py

Takes roughly one minute.  Development evidence only: not qualification, not
deployment, no learned actuator authority.  HMC arbitration remains the sole
authority in both arms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aeolus_closed_loop import HistoricalAdviser, run_closed_loop

# Fixed demo scenario: one held-out cluster, one fault member, one repetition.
DEMO_SCENARIO = (
    "pilot-v1/contingency/nominal/crew-metabolic-humidity-skew",
    "T01",
    "R01",
)


def main() -> int:
    checkpoint = HERE / "action-aware-mlp-v1.pt"
    if not checkpoint.is_file():
        print(f"missing model checkpoint: {checkpoint}", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is required: pip install torch", file=sys.stderr)
        return 2

    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design

    design = load_approved_pilot_design(REPO_ROOT)
    contracts = load_forecast_contracts(REPO_ROOT)
    adviser = HistoricalAdviser(checkpoint)

    cluster, member, repetition = DEMO_SCENARIO
    print(f"scenario: {cluster} member={member} repetition={repetition}")
    print("running control arm (canonical HMC, no proposals) ...")
    control = run_closed_loop(
        repo_root=REPO_ROOT, design=design, contracts=contracts,
        cluster_id=cluster, member_id=member, repetition_id=repetition,
        adviser=None,
    )
    print("running advised arm (model proposes, HMC arbitrates) ...")
    advised = run_closed_loop(
        repo_root=REPO_ROOT, design=design, contracts=contracts,
        cluster_id=cluster, member_id=member, repetition_id=repetition,
        adviser=adviser,
    )

    summary = {
        "pairing_verified": control["scenario_sha256"] == advised["scenario_sha256"],
        "terminal_status": {
            "control": control["terminal_status"],
            "advised": advised["terminal_status"],
        },
        "integrated_threshold_exceedance_lower_is_better": {
            "control_canonical_hmc": round(control["integrated_exceedance"], 4),
            "advised_model_plus_hmc": round(advised["integrated_exceedance"], 4),
        },
        "steps_above_warning": {
            "control": control["exceedance_steps"],
            "advised": advised["exceedance_steps"],
        },
        "adviser_activity": {
            "proposals_made": advised["proposals_made"],
            "proposals_admitted_valid": advised["proposals_admitted"],
            "hmc_overrides": advised["hmc_overrides"],
        },
        "trace_sha256": {
            "control": control["trace_sha256"],
            "advised": advised["trace_sha256"],
        },
        "claim_boundary": "development evidence only; not qualification; not deployment",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
