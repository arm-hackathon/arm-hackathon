"""Leakage-safe labelled corpus generation from AEOLUS scenario replays.

Every feature row is exactly :func:`aeolus.trace.model_feature_row` output, so
the corpus cannot contain hidden fault truth (health, fault effectiveness,
seed or noise). Labels come from the scenario's declared fault profiles at
each window's final measured tick, never from telemetry.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from aeolus.config import GradualPrimaryFanDegradation, HabitatConfig, load_scenario
from aeolus.families import load_family_manifest, observable_onset
from aeolus.model_input import build_model_input_contract, model_input_v1
from aeolus.scenario import run_scenario
from aeolus.trace import model_feature_row

CORPUS_VERSION = 1
CORPUS_V2_VERSION = 2
DEFAULT_WINDOW_TICKS = 10
DEFAULT_STRIDE_TICKS = 5
NOMINAL_LABEL = "nominal"
EXCLUDED_TRANSITION_LABEL = "excluded_transition"
LABEL_SET = (
    "blocked_path",
    "frozen_sensor",
    "gradual_primary_fan_degradation",
    "nominal",
)

USAGE = (
    "Usage: PYTHONPATH=src python3 -m aeolus.corpus <out-dir> "
    "<scenario.json> [scenario.json ...]\n"
    "   or: PYTHONPATH=src python3 -m aeolus.corpus --v2 <out-dir> <families.json>"
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


def generate_corpus_v2(
    family_manifest_path,
    out_dir,
    *,
    window: int = DEFAULT_WINDOW_TICKS,
    stride: int = DEFAULT_STRIDE_TICKS,
) -> dict:
    """Generate observable-labelled corpus-v2 rows from strict scenario families."""
    if window < 1:
        raise ValueError("window must be a positive number of ticks")
    if stride < 1:
        raise ValueError("stride must be a positive number of ticks")
    families = load_family_manifest(Path(family_manifest_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    onsets: dict[str, int] = {}
    family_evidence: dict[str, dict[str, object]] = {}
    for family in families.families:
        onset = observable_onset(family, families.contract_metadata)
        onsets[family.family_id] = onset.tick
        family_evidence[family.family_id] = {
            "fault_scenario_sha256": onset.fault_scenario_sha256,
            "observable_onset_tick": onset.tick,
            "reference_scenario_sha256": onset.reference_scenario_sha256,
            "split": family.split,
        }
        contract = build_model_input_contract(load_scenario(family.reference_path))
        for role, path in (("reference", family.reference_path), ("fault", family.fault_path)):
            config = load_scenario(path)
            records = run_scenario(config)
            for index, start in enumerate(range(0, len(records) - window + 1, stride)):
                window_records = records[start : start + window]
                start_tick = window_records[0].tick
                end_tick = window_records[-1].tick
                label = _v2_label(
                    role=role,
                    fault_class=family.fault_class,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    onset_tick=onset.tick,
                )
                all_rows.append(
                    {
                        "family_id": family.family_id,
                        "split": family.split,
                        "scenario_role": role,
                        "window_index": index,
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "observable_onset_tick": onset.tick,
                        "label": label,
                        "model_input_version": families.contract_metadata["model_input_version"],
                        "selector_sha256": families.contract_metadata["selector_sha256"],
                        "topology_sha256": families.contract_metadata["topology_sha256"],
                        "features": [
                            model_input_v1(record, contract).tolist()
                            for record in window_records
                        ],
                    }
                )

    all_rows.sort(key=lambda row: (row["family_id"], row["scenario_role"], row["end_tick"]))
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    label_counts: dict[str, int] = {}
    for row in all_rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
    manifest = {
        "corpus_version": CORPUS_V2_VERSION,
        **families.contract_metadata,
        "family_manifest_sha256": families.manifest_sha256,
        "families": family_evidence,
        "window_ticks": window,
        "stride_ticks": stride,
        "total_windows": len(all_rows),
        "scored_windows": len(all_rows) - label_counts.get(EXCLUDED_TRANSITION_LABEL, 0),
        "excluded_transition_windows": label_counts.get(EXCLUDED_TRANSITION_LABEL, 0),
        "label_counts": label_counts,
        "observable_onsets": onsets,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _v2_label(
    *, role: str, fault_class: str, start_tick: int, end_tick: int, onset_tick: int
) -> str:
    if role == "reference" or end_tick < onset_tick:
        return NOMINAL_LABEL
    if start_tick < onset_tick <= end_tick:
        return EXCLUDED_TRANSITION_LABEL
    return fault_class


def main(argv: list[str]) -> int:
    v2 = argv[:1] == ["--v2"]
    arguments = argv[1:] if v2 else argv
    if len(arguments) < 2 or (v2 and len(arguments) != 2):
        print(USAGE, file=sys.stderr)
        return 2
    out_dir, *scenario_paths = arguments
    try:
        manifest = (
            generate_corpus_v2(scenario_paths[0], out_dir)
            if v2
            else generate_corpus(scenario_paths, out_dir)
        )
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
