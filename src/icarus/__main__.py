"""Run a named scenario and write its JSONL replay trace.

Usage:
    PYTHONPATH=src python3 -m icarus <scenario-name> <trace-path>

Example:
    PYTHONPATH=src python3 -m icarus primary_fan_degradation traces/primary_fan_degradation.jsonl
"""

import sys

from icarus.scenario import SCENARIOS, run_scenario

USAGE = """Usage: PYTHONPATH=src python3 -m icarus <scenario-name> <trace-path>
Example: PYTHONPATH=src python3 -m icarus primary_fan_degradation traces/primary_fan_degradation.jsonl"""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    name, trace_path = argv
    if name not in SCENARIOS:
        print(
            f"unknown scenario {name!r}; choose from: {', '.join(sorted(SCENARIOS))}",
            file=sys.stderr,
        )
        return 2
    records = run_scenario(name, trace_path=trace_path)
    last = records[-1]
    print(f"scenario={name} ticks={len(records)} trace={trace_path}")
    print(
        f"final tick={last.tick} room_a_co2={last.room_a_co2:.3f} "
        f"room_b_co2={last.room_b_co2:.3f} "
        f"main_fan_effectiveness={last.main_fan_effectiveness:.2f} "
        f"airflow={last.airflow:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
