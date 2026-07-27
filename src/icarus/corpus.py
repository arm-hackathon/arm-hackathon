"""Leakage-safe labelled corpus generation from ICARUS scenario replays.

Every feature row is exactly :func:`icarus.trace.model_feature_row` output, so
the corpus cannot contain hidden fault truth (health, fault effectiveness,
seed or noise). Labels come from the scenario's declared fault profiles at
each window's final measured tick, never from telemetry.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from icarus.config import GradualPrimaryFanDegradation, HabitatConfig, load_scenario
from icarus.scenario import run_scenario
from icarus.trace import model_feature_row

CORPUS_VERSION = 1
DEFAULT_WINDOW_TICKS = 10
DEFAULT_STRIDE_TICKS = 5
NOMINAL_LABEL = "nominal"
LABEL_SET = (
    "blocked_path",
    "frozen_sensor",
    "gradual_primary_fan_degradation",
    "nominal",
)

USAGE = (
    "Usage: PYTHONPATH=src python3 -m icarus.corpus <out-dir> "
    "<scenario.json> [scenario.json ...]"
)


def label_for_window(config: HabitatConfig, end_tick: int) -> str:
    """Return the single fault class active at a window's final measured tick.

    A window with no active fault is ``nominal``. More than one active fault
    is rejected: corpus v1 ships single-fault scenarios only, so a multi-fault
    window means the scenario set drifted from that contract.
    """
    active: list[str] = []
    for profile in config.connection_faults():
        if profile.effectiveness_at(end_tick) < 1.0:
            if isinstance(profile, GradualPrimaryFanDegradation):
                active.append("gradual_primary_fan_degradation")
            else:
                active.append("blocked_path")
    for profile in config.sensor_faults():
        if profile.is_frozen_at(end_tick):
            active.append("frozen_sensor")
    if len(active) > 1:
        raise ValueError(
            f"more than one fault active at tick {end_tick}; "
            "corpus v1 supports single-fault scenarios only"
        )
    return active[0] if active else NOMINAL_LABEL


def build_corpus_rows(
    config: HabitatConfig,
    records: list,
    *,
    scenario_name: str,
    window: int = DEFAULT_WINDOW_TICKS,
    stride: int = DEFAULT_STRIDE_TICKS,
) -> list[dict]:
    """Slide a fixed window over one scenario's records and label each span."""
    if window < 1:
        raise ValueError("window must be a positive number of ticks")
    if stride < 1:
        raise ValueError("stride must be a positive number of ticks")
    rows: list[dict] = []
    for window_index, start in enumerate(range(0, len(records) - window + 1, stride)):
        window_records = records[start : start + window]
        end_tick = window_records[-1].tick
        rows.append(
            {
                "scenario": scenario_name,
                "window_index": window_index,
                "start_tick": window_records[0].tick,
                "end_tick": end_tick,
                "label": label_for_window(config, end_tick),
                "features": [model_feature_row(record) for record in window_records],
            }
        )
    return rows


def generate_corpus(
    scenario_paths,
    out_dir,
    *,
    window: int = DEFAULT_WINDOW_TICKS,
    stride: int = DEFAULT_STRIDE_TICKS,
) -> dict:
    """Run each scenario, write corpus.jsonl and manifest.json, return the manifest.

    Scenarios are processed in file-name order and every serialisation is
    canonical, so regenerating from the same inputs is byte-identical.
    """
    paths = sorted((Path(path) for path in scenario_paths), key=lambda path: path.name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    scenario_entries: list[dict] = []
    for path in paths:
        config = load_scenario(path)
        records = run_scenario(config)
        rows = build_corpus_rows(
            config,
            records,
            scenario_name=path.stem,
            window=window,
            stride=stride,
        )
        all_rows.extend(rows)
        scenario_entries.append(
            {
                "name": path.stem,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "ticks": len(records),
                "windows": len(rows),
            }
        )

    label_counts = {label: 0 for label in LABEL_SET}
    for row in all_rows:
        label_counts[row["label"]] += 1

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "window_ticks": window,
        "stride_ticks": stride,
        "label_set": list(LABEL_SET),
        "scenarios": scenario_entries,
        "total_windows": len(all_rows),
        "label_counts": label_counts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(USAGE, file=sys.stderr)
        return 2
    out_dir, *scenario_paths = argv
    try:
        manifest = generate_corpus(scenario_paths, out_dir)
    except ValueError as exc:
        print(f"invalid corpus input: {exc}", file=sys.stderr)
        return 2
    print(
        f"corpus={Path(out_dir) / 'corpus.jsonl'} "
        f"windows={manifest['total_windows']} "
        f"labels={manifest['label_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
