"""AEOLUS guided tour: one command, plain language, real artifacts.

Run from a source checkout:

    uv run --locked --python 3.11 --extra dev python scripts/aeolus_tour.py

Welcomes a new visitor, explains the project in plain English, and offers
hands-on options: run the development forecaster live, inspect a checked-in
paired replay artifact step by step, or verify a current demo run independently.
No network access; everything runs from checked-in code and artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_ARTIFACT = (
    REPO_ROOT
    / "artifacts/demo-only/habitat-v2-forecast/paired-live-replay-v1.json"
)
REPLAY_SHA256 = "63c91397b7f33782c96d11de5c33296a76d0a42d68c1b51ac24d44556884d062"

INTRO = """\
======================================================================
 AEOLUS — a simulated space habitat that a learned model helps run
======================================================================

The picture in one paragraph:

  A space habitat's air system has inertia — by the time CO2 crosses a
  safety line, the crew has already been breathing it.  AEOLUS pairs a
  deterministic safety controller (HMC, the only thing allowed to act)
  with a small neural network that forecasts the next 8 steps for each
  possible action, so problems get prevented instead of reacted to.

  The model advises.  The controller decides.  The checked-in live demo
  validates and replays its deterministic control trace; the historical
  campaign is a separate archive with disclosed reproduction gaps.
"""

HOW_IT_WORKS = """\
How it works, in four pieces:

  1. THE PLANT — a deterministic habitat simulator: 8 zones, air flow,
     CO2, oxygen, temperature, humidity.  Same inputs, same outputs,
     every time.  That makes current deterministic trace claims replayable.

  2. HMC (Habitat Management Computer) — the safety controller and the
     ONLY authority over actuators.  Each step it observes, verifies,
     hears proposals, decides, and issues exactly one command.

  3. THE MODEL — an artifact associated with a historical run reported as
     using 23,400 simulated examples.  Given the last 16 steps and one
     candidate action, it predicts the next 8 steps of the habitat's
     atmosphere.  It never touches an actuator.  The historical harness
     was coded to stay silent when required availability evidence was
     missing, but the retained V3 result records zero such cases.

  4. THE HISTORICAL ADVISER LOOP — each step, the model forecasts every
     allowed action; the safest predicted future is PROPOSED to HMC, which
     can accept, override, or reject it.  HMC's command always wins.  The
     current bounded demo instead uses an operator-selected proposal.

  The historical summary records 119 control/advised pairs: across 102
  fault pairs, 78 safer, 24 equal, 0 worse — and HMC modified or replaced
  81 proposals.  Raw V2 runs and training receipts are missing.  This is
  development evidence only: not certified, deployed, or hardware evidence.
"""

ACTIONS = (
    ("normal-occupied-v1", "keep the habitat in normal occupied operation"),
    ("normal-eva_transition-v1", "shift to the spacewalk-transition mode"),
    ("normal-contingency-v1", "shift to contingency mode"),
    ("normal-dormant-v1", "shift to dormant (low-activity) mode"),
)


def _pause(seconds: float = 0.0) -> None:
    if seconds > 0.0:
        time.sleep(seconds)


def _wait_for_enter() -> None:
    try:
        input("\n[press Enter to return to the menu] ")
    except EOFError:
        pass


def _worst_co2(target_row) -> float:
    """Worst-zone CO2 (ppm) from one [51] target row (8 zones x 6 fields)."""
    return float(max(float(target_row[zone * 6 + 2]) for zone in range(8)))


def run_live_forecast() -> None:
    print("\nThe model will look at the habitat at step 16 and forecast the")
    print("next 8 steps for every allowed action.  You pick which action")
    print("the operator proposes; HMC decides whether to accept it.\n")
    for index, (action_id, description) in enumerate(ACTIONS, start=1):
        print(f"  {index}. {action_id:<26} ({description})")
    choice = input("\nChoose an action [1-4, default 1]: ").strip() or "1"
    try:
        action_id = ACTIONS[int(choice) - 1][0]
    except (ValueError, IndexError):
        print("Not a valid choice — using option 1.")
        action_id = ACTIONS[0][0]
    print()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aeolus.habitat_v2.forecast.live_mlp_demo import (
        load_live_mlp_model,
        run_live_mlp_forecast_demo,
    )

    model = load_live_mlp_model(
        REPO_ROOT
        / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz",
        expected_sha256=(
            "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
        ),
    )
    print("Running the habitat to step 16, then asking the model to forecast")
    print("every action...")
    result = run_live_mlp_forecast_demo(
        REPO_ROOT, model, selected_action_id=action_id,
    )
    import numpy as np

    truth = np.asarray(result.truth_f32, dtype=np.float64)
    forecasts = {
        item.action_id: np.asarray(item.prediction_f32, dtype=np.float64)
        for item in result.candidate_forecasts
    }
    short = {action: action.replace("normal-", "").replace("-v1", "") for action, _ in ACTIONS}

    print("\n--- what the model predicts (worst-zone CO2, next 8 steps) ---")
    header = "step  |" + "|".join(f" {short[a]:>12}" for a, _ in ACTIONS)
    print(header)
    for horizon in range(8):
        step_no = 17 + horizon
        cells = "|".join(
            f" {_worst_co2(forecasts[a][horizon]):>12.1f}" for a, _ in ACTIONS
        )
        print(f"{step_no:5d} |{cells}")
        _pause(0.15)

    print(f"\nOperator proposes your action: {action_id}")
    print(f"HMC reviews it -> {result.arbitration_disposition}.")
    print("(HMC can accept, modify, or reject any proposal; it always has")
    print(" the final word.  The model never commands.)")

    chosen = forecasts[action_id]
    print(f"\n--- prediction vs reality for {short[action_id]} "
          "(steps 17-24, worst-zone CO2 ppm) ---")
    total_error = 0.0
    for horizon in range(8):
        predicted = _worst_co2(chosen[horizon])
        actual = _worst_co2(truth[horizon])
        total_error += abs(predicted - actual)
        print(
            f"step {17 + horizon:3d} | predicted {predicted:8.1f} | "
            f"actual {actual:8.1f} | error {abs(predicted - actual):6.1f}"
        )
        _pause(0.15)
    print(f"\nMean per-step error for your action: {total_error / 8:.1f} ppm.")
    print(f"All {result.replay_committed_steps} steps completed; the control")
    print("trace replayed bit-for-bit.  The model advised; HMC commanded.")


def replay_paired_experiment() -> None:
    import hashlib

    raw = REPLAY_ARTIFACT.read_bytes()
    # Git may materialize LF blobs as CRLF on Windows checkouts; the pinned
    # hash covers the canonical LF bytes.
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != REPLAY_SHA256:
        print("The replay artifact does not match its recorded hash — refusing.")
        return
    data = json.loads(raw)
    print("\nThis checked-in replay's metadata records the same habitat, fault")
    print("and noise in two arms: once with HMC alone, once with the model")
    print("advising.  'exceed' measures departure past the recorded warning")
    print("thresholds — 0.000 means no recorded threshold exceedance.\n")
    speed = input("Playback speed — seconds per step [default 0.05]: ").strip()
    try:
        delay = max(0.0, float(speed)) if speed else 0.05
    except ValueError:
        delay = 0.05
    for arm_key, label in (("control", "CONTROL — HMC alone"), ("advised", "ADVISED — model + HMC")):
        arm = data[arm_key]
        print(f"\n--- {label} ---")
        for row in arm["steps"]:
            line = (
                f"step {row['step']:3d} | CO2 {row['co2_ppm']:8.1f} ppm | "
                f"T {row['temperature_c']:5.2f} C | O2 {row['o2_percent']:5.2f}% | "
                f"exceed {row['exceedance']:7.3f}"
            )
            if row["event"]:
                line += f"  | adviser {row['event']}"
            print(line)
            _pause(delay)
        print(
            f"{label.split(' ')[0]} total: exceedance "
            f"{arm['integrated_exceedance']:.2f} over "
            f"{arm['exceedance_steps']} unsafe step(s)."
        )
    control, advised = data["control"], data["advised"]
    print("\nResult: control", f"{control['integrated_exceedance']:.2f}",
          "vs advised", f"{advised['integrated_exceedance']:.2f}",
          "(lower is safer).  This replays a checked-in historical artifact;")
    print("it does not rerun or independently authenticate the full campaign.")


def verify_a_run() -> None:
    print("\nRunning the independently verified forecast receipt...")
    print("(creates a fresh receipt directory, replays the control trace,")
    print("and only prints the report if verification passes)\n")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/run_habitat_v2_forecast_report.py"),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode == 0:
        print("\nVerification passed.  Open the FORECAST_REPORT_URL above in a browser.")


def menu() -> int:
    print(INTRO)
    options = {
        "1": ("Watch the model forecast live (runs the real trained model)", run_live_forecast),
        "2": ("Replay the paired experiment, step by step", replay_paired_experiment),
        "3": ("Verify a run yourself (hash-checked receipt)", verify_a_run),
        "4": ("How does this work? (plain-English tour)", None),
        "5": ("Quit", None),
    }
    while True:
        print("What would you like to do?")
        for key, (label, _) in options.items():
            print(f"  {key}. {label}")
        choice = input("\nChoose [1-5]: ").strip()
        if choice == "4":
            print(HOW_IT_WORKS)
            _wait_for_enter()
            continue
        if choice == "5" or choice.lower() in {"q", "quit", "exit"}:
            print("Thanks for visiting AEOLUS.")
            return 0
        action = options.get(choice)
        if action is None:
            print("Please choose 1-5.")
            continue
        try:
            action[1]()
        except (EOFError, KeyboardInterrupt):
            print("\n(interrupted)")
        _wait_for_enter()


if __name__ == "__main__":
    if not sys.stdin.isatty():
        print(INTRO)
        print(HOW_IT_WORKS)
        print("Run interactively for the hands-on options:")
        print("  python scripts/aeolus_tour.py")
        raise SystemExit(0)
    raise SystemExit(menu())
