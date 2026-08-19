"""Run the preregistration-v3 ensemble arm on the frozen v2 roster.

Control and single-model arms are NOT re-run: their frozen v2 results are
the paired comparison, with identical deterministic seeds.  ``--smoke``
repeats one v2 control scenario twice and requires the integrated
exceedance to match the frozen v2 record exactly — the determinism guard
preregistered in preregistration-v3.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aeolus_closed_loop import run_closed_loop  # noqa: E402
from ensemble_adviser import EnsembleAdviser  # noqa: E402
from run_paired import MEMBERS_V1, MEMBERS_V2, held_out_clusters  # noqa: E402

V2_RESULTS = Path(
    r"C:/Users/Nxiss/code/aeolus-next-gates-20260818/closed-loop-v1/paired-v2-results.json"
)


def scenarios_v2() -> list[tuple[str, str, str]]:
    return [
        (cluster, member, repetition)
        for cluster in held_out_clusters()
        for member in MEMBERS_V2
        for repetition in ("R01", "R02")
    ] + [(cluster, "HEALTHY", "R01") for cluster in held_out_clusters()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="one scenario, control arm twice; must match the frozen v2 record")
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2

    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design

    design = load_approved_pilot_design(REPO_ROOT)
    contracts = load_forecast_contracts(REPO_ROOT)

    if args.smoke:
        scenario = (held_out_clusters()[0], MEMBERS_V1[0], "R01")
        frozen = json.loads(V2_RESULTS.read_text())
        reference = [
            r["integrated_exceedance"] for r in frozen["results"]
            if r["arm"] == "control" and r["cluster_id"] == scenario[0]
            and r["member_id"] == scenario[1] and r["repetition_id"] == scenario[2]
        ]
        if len(reference) != 1:
            print("frozen v2 reference not found for smoke scenario", file=sys.stderr)
            return 2
        runs = [
            run_closed_loop(repo_root=REPO_ROOT, design=design, contracts=contracts,
                            cluster_id=scenario[0], member_id=scenario[1],
                            repetition_id=scenario[2], adviser=None)
            for _ in range(2)
        ]
        got = [r["integrated_exceedance"] for r in runs]
        ok = got[0] == got[1] == reference[0]
        print(json.dumps({"smoke": "determinism-guard", "scenario": scenario,
                          "frozen_v2": reference[0], "rerun": got, "match": ok}, indent=2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"smoke": True, "match": ok,
                                           "frozen_v2": reference[0], "rerun": got},
                                          indent=1) + "\n")
        return 0 if ok else 1

    adviser = EnsembleAdviser(HERE)
    scenarios = scenarios_v2()
    results: list[dict] = []
    total = len(scenarios)
    for done, (cluster, member, repetition) in enumerate(scenarios, start=1):
        results.append(run_closed_loop(
            repo_root=REPO_ROOT, design=design, contracts=contracts,
            cluster_id=cluster, member_id=member, repetition_id=repetition,
            adviser=adviser,
        ))
        results[-1]["arm"] = "advised_ensemble"
        print(f"[{done}/{total}] {cluster} {member} {repetition} advised_ensemble "
              f"exceedance={results[-1]['integrated_exceedance']:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": "aeolus_habitat_v2_closed_loop_results_v1",
                   "roster": "v2-ensemble-arm", "preregistration": "preregistration-v3.json",
                   "results": results}, handle, indent=1, sort_keys=True)
    print(json.dumps({"runs": len(results), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
