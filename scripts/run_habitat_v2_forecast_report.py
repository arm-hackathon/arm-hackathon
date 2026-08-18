"""Create and independently verify a fresh, local-only AEOLUS judge demo receipt."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a deterministic advisory-only forecast receipt and verify its replay."
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--output-parent", type=Path, default=root / "out" / "judge-demo-runs")
    parser.add_argument(
        "--selected-action", default="normal-occupied-v1",
        choices=("normal-occupied-v1", "normal-eva_transition-v1", "normal-contingency-v1", "normal-dormant-v1"),
    )
    return parser.parse_args()


def fresh_output(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("judge-demo-%Y%m%dT%H%M%SZ")
    for number in range(10_000):
        candidate = parent / f"{stem}-{number:04d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a fresh judge-demo output directory")


def main() -> int:
    args = arguments()
    root = args.repo_root.resolve()
    output_parent = args.output_parent.resolve()
    try:
        output_parent.relative_to(root)
    except ValueError as error:
        raise ValueError("output parent must be inside the repository") from error
    output = fresh_output(output_parent)
    run = root / "scripts" / "run_habitat_v2_live_forecast.py"
    verify = root / "scripts" / "verify_habitat_v2_live_forecast_demo.py"
    subprocess.run([sys.executable, str(run), "--repo-root", str(root), "--selected-action", args.selected_action, "--output", str(output)], check=True)
    subprocess.run([sys.executable, str(verify), "--repo-root", str(root), "--report", str(output)], check=True)
    print(f"VERIFIED_RECEIPT={output / 'receipt.json'}")
    print(f"FORECAST_REPORT_URL={(output / 'index.html').as_uri()}")
    print(f"BROWSER_SIMULATOR_URL={(root / 'demo/browser-simulator/index.html').as_uri()}")
    print("BOUNDARY=local simulated forecast; advisory-only; deterministic HMC remains sole command authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
