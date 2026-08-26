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

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aeolus_closed_loop import HistoricalAdviser, StepRecord, run_closed_loop  # noqa: E402

# Fixed demo scenario: one held-out cluster, one fault member, one repetition.
DEMO_SCENARIO = (
    "pilot-v1/contingency/nominal/crew-metabolic-humidity-skew",
    "T01",
    "R01",
)


def _live_printer(arm: str, delay: float):
    def print_step(record: StepRecord) -> None:
        event = ""
        if record.adviser_abstained_unavailable:
            event = " | adviser ABSTAINED (incomplete sensor evidence)"
        elif record.proposed_candidate is not None:
            if record.validation_outcome != "VALID":
                event = (
                    f" | adviser proposed {record.proposed_candidate} -> "
                    f"HMC {record.validation_outcome}"
                )
            elif record.final_command_sha256 != record.requested_command_sha256:
                event = (
                    f" | adviser proposed {record.proposed_candidate} -> "
                    "HMC OVERRIDDEN (canonical command kept)"
                )
            else:
                event = (
                    f" | adviser proposed {record.proposed_candidate} -> "
                    "HMC ACCEPTED"
                )
        print(
            f"[{arm}] step {record.application_step:3d} | "
            f"CO2 {record.max_co2_ppm:8.1f} ppm | "
            f"T {record.max_temperature_k - 273.15:5.2f} C | "
            f"O2 {record.min_o2_mole_fraction * 100:5.2f}% | "
            f"exceed {record.step_exceedance:7.3f}{event}"
        )
        if delay > 0.0:
            time.sleep(delay)

    return print_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Print every simulation step as it happens (metrics + decisions).",
    )
    parser.add_argument(
        "--live-delay",
        type=float,
        default=0.0,
        help="Seconds to pause between live steps (e.g. 0.1 for recordings).",
    )
    arguments = parser.parse_args()
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
        on_step=_live_printer("control ", arguments.live_delay) if arguments.live else None,
        live_metrics=arguments.live,
    )
    if arguments.live:
        print()
    print("running advised arm (model proposes, HMC arbitrates) ...")
    advised = run_closed_loop(
        repo_root=REPO_ROOT, design=design, contracts=contracts,
        cluster_id=cluster, member_id=member, repetition_id=repetition,
        adviser=adviser,
        on_step=_live_printer("advised ", arguments.live_delay) if arguments.live else None,
        live_metrics=arguments.live,
    )
    if arguments.live:
        print()

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
