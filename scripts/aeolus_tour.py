"""AEOLUS guided tour: one command, plain language, real artifacts.

Run from a source checkout:

    uv run --locked --python 3.11 --extra dev python scripts/aeolus_tour.py

Welcomes a new visitor, explains the project in plain English, and offers
hands-on options: run the trained forecaster live, replay the recorded paired
experiment step by step, or verify a run independently.  No network access;
everything runs from checked-in code and artifacts.
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
REPLAY_SHA256 = "9bbff09dd36ea7f647c542145b631b574a46b55ae2a8a2d9d4b87fc6ef407460"

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

  The model advises.  The controller decides.  Every run replays
  bit-for-bit, so nothing here is a video of a result — it is the result.
"""

HOW_IT_WORKS = """\
How it works, in four pieces:

  1. THE PLANT — a deterministic habitat simulator: 8 zones, air flow,
     CO2, oxygen, temperature, humidity.  Same inputs, same outputs,
     every time.  That is what makes every claim here checkable.

  2. HMC (Habitat Management Computer) — the safety controller and the
     ONLY authority over actuators.  Each step it observes, verifies,
     hears proposals, decides, and issues exactly one command.

  3. THE MODEL — a small network trained on 23,400 simulated examples.
     Given the last 16 steps and one candidate action, it predicts the
     next 8 steps of the habitat's atmosphere.  It never touches an
     actuator, and if any sensor evidence is missing it stays silent.

  4. THE ADVISER LOOP — each step, the model forecasts every allowed
     action; the safest predicted future is PROPOSED to HMC, which can
     accept, override, or reject it.  HMC's command always wins.

  The evidence: 238 paired runs (identical scenarios, with and without
  the model): 78 safer, 24 equal, 0 worse — and HMC overruled the model
  81 times along the way.  Development evidence only: not certified,
  not deployed, not hardware.
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
    result = run_live_mlp_forecast_demo(
        REPO_ROOT, model, selected_action_id=action_id,
    )
    import numpy as np

    truth = np.asarray(result.truth_f32, dtype=np.float64)
    print("Forecast error per candidate action (vs what actually happened,")
    print("lower = predicted reality better):")
    for item in result.candidate_forecasts:
        error = float(
            np.abs(np.asarray(item.prediction_f32, dtype=np.float64) - truth).mean()
        )
        marker = "  <- your choice" if item.action_id == action_id else ""
        print(f"  {item.action_id:<26} {error:8.2f}{marker}")
    print(f"\nHMC reviewed your selected action: {result.arbitration_disposition}.")
    print(f"All {result.replay_committed_steps} steps completed and the control")
    print("trace replayed bit-for-bit.  The model only advised; HMC commanded.")


def replay_paired_experiment() -> None:
    import hashlib

    raw = REPLAY_ARTIFACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPLAY_SHA256:
        print("The replay artifact does not match its recorded hash — refusing.")
        return
    data = json.loads(raw)
    print("\nSame habitat, same fault (a CO2 scrubber quietly degrading), same")
    print("noise.  Run twice: once with HMC alone, once with the model")
    print("advising.  'exceed' measures how far past the safety line the air")
    print("is — 0.000 means safe.\n")
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
          "(lower is safer).  This is a replay of recorded run data;")
    print("the live version needs the experiment branch (see README).")


def verify_a_run() -> None:
    print("\nRunning the independently verified forecast receipt...")
    print("(creates a fresh receipt directory, replays the control trace,")
    print("and only prints the report if verification passes)\n")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/run_habitat_v2_forecast_judge_demo.py"),
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
