"""Paired closed-loop runner: control vs model-advised, pre-registered design."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aeolus_closed_loop import HistoricalAdviser, run_closed_loop  # noqa: E402

MEMBERS_V1 = ("T01", "T07")
MEMBERS_V2 = ("T01", "T07", "T12")


def held_out_clusters() -> tuple[str, ...]:
    roster = json.loads((HERE / "held-out-clusters.json").read_text())
    return tuple(roster["clusters"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="one scenario, control arm run twice for determinism")
    parser.add_argument("--roster", choices=("v1", "v2"), default="v2",
                        help="v1: 4 clusters x T01/T07 x R01; v2: preregistration-v2 full roster")
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2

    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design

    design = load_approved_pilot_design(REPO_ROOT)
    contracts = load_forecast_contracts(REPO_ROOT)
    adviser = HistoricalAdviser(HERE / "action-aware-mlp-v1.pt")

    scenarios: list[tuple[str, str, str]] = []
    if args.smoke:
        scenarios = [(held_out_clusters()[0], MEMBERS_V1[0], "R01")]
    elif args.roster == "v1":
        scenarios = [
            (cluster, member, "R01")
            for cluster in held_out_clusters()[:4]
            for member in MEMBERS_V1
        ]
    else:
        scenarios = [
            (cluster, member, repetition)
            for cluster in held_out_clusters()
            for member in MEMBERS_V2
            for repetition in ("R01", "R02")
        ] + [
            (cluster, "HEALTHY", "R01") for cluster in held_out_clusters()
        ]

    results: list[dict] = []
    total = len(scenarios) * 2
    done = 0
    for cluster, member, repetition in scenarios:
        arms = ("control-a", "control-b") if args.smoke else ("control", "advised")
        for arm in arms:
            results.append(run_closed_loop(
                repo_root=REPO_ROOT, design=design, contracts=contracts,
                cluster_id=cluster, member_id=member, repetition_id=repetition,
                adviser=None if arm.startswith("control") else adviser,
            ))
            if args.smoke:
                results[-1]["arm"] = arm
            done += 1
            print(f"[{done}/{total}] {cluster} {member} {repetition} {arm} "
                  f"exceedance={results[-1]['integrated_exceedance']:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": "aeolus_habitat_v2_closed_loop_results_v1",
                   "roster": args.roster if not args.smoke else "smoke",
                   "results": results}, handle, indent=1, sort_keys=True)
    print(json.dumps({
        "runs": len(results),
        "output": str(args.output),
        "by_arm": {arm: sum(1 for r in results if r["arm"] == arm)
                   for arm in {r["arm"] for r in results}},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
