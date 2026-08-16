from __future__ import annotations

from pathlib import Path
import sys

from .config import load_scenario_file
from .runner import run_scenario
from .trace import validate_trace_bytes

USAGE = "Usage: python -m aeolus.habitat_v2 <scenario.json> <trace.jsonl>"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2

    scenario_path = Path(argv[0])
    trace_path = Path(argv[1])
    if trace_path.exists():
        print(f"refusing to overwrite existing trace: {trace_path}", file=sys.stderr)
        return 2

    try:
        scenario = load_scenario_file(scenario_path)
        run = run_scenario(scenario)
        validate_trace_bytes(run.trace_bytes, scenario=scenario)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("xb") as output:
            output.write(run.trace_bytes)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"habitat-v2 run failed: {error}", file=sys.stderr)
        return 2

    final_row = run.rows[-1]
    print(f"run_id={scenario.run_id} steps={scenario.data['steps']} trace={trace_path}")
    for zone_id in sorted(final_row["telemetry"]):
        telemetry = final_row["telemetry"][zone_id]
        print(
            f"{zone_id} co2_ppm={telemetry['co2_ppm']:.6f} "
            f"temperature_k={telemetry['temperature_k']:.6f} "
            f"relative_humidity={telemetry['relative_humidity']:.6f} "
            f"pressure_pa={telemetry['pressure_pa']:.6f}"
        )
    resources = final_row["resource_state"]
    print(
        f"battery_energy_wh={resources['battery_energy_wh']:.6f} "
        f"oxygen_store_mol={resources['oxygen_store_mol']:.6f} "
        f"co2_sorbent_remaining_mol="
        f"{resources['co2_sorbent_remaining_mol']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
