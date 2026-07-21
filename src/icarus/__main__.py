"""Run an explicit scenario JSON file and write its JSONL replay trace.

Usage:
    PYTHONPATH=src python3 -m icarus <scenario.json> <trace-path>

Example:
    PYTHONPATH=src python3 -m icarus scenarios/standard_habitat.json traces/standard_habitat.jsonl
"""

import sys

from icarus.config import load_scenario
from icarus.scenario import run_scenario

USAGE = (
    "Usage: PYTHONPATH=src python3 -m icarus <scenario.json> <trace-path>\n"
    "Example: PYTHONPATH=src python3 -m icarus scenarios/standard_habitat.json "
    "traces/standard_habitat.jsonl"
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    scenario_path, trace_path = argv
    try:
        config = load_scenario(scenario_path)
    except ValueError as exc:
        print(f"invalid scenario: {exc}", file=sys.stderr)
        return 2
    records = run_scenario(config, trace_path=trace_path)
    last = records[-1]
    zone_summary = " ".join(
        f"{zone.id}_co2={last.zones[zone.id]['co2']:.3f}"
        for zone in config.non_processing_zones()
    )
    processing_id = config.processing_zone().id
    print(f"scenario={scenario_path} ticks={len(records)} trace={trace_path}")
    print(
        f"final tick={last.tick} {zone_summary} "
        f"captured_co2={last.zones[processing_id]['captured_co2']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
