#!/usr/bin/env python3
"""Explain the two C4 transient CO2-improvement exceptions."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aeolus.config import load_scenario
from aeolus.families import load_family_manifest
from aeolus.recovery import RecoverySettings
from aeolus.recovery_evidence import RECOVERY_RUN, _arm_metrics
from aeolus.scenario import run_recovery_scenario

FAMILY_IDS = {
    "train-s211-primary-high-reserve-t55-cabin_a-transient-degradation-d060-e750",
    "validation-s601-primary-high-reserve-t55-cabin_a-transient-degradation-d060-e750",
}
SETTINGS = RecoverySettings(
    entry_residual_ratio=0.10,
    entry_isolation_margin=0.05,
    entry_persistence_ticks=2,
    exit_residual_ratio=0.06,
    handback_abort_residual_ratio=0.08,
    handback_abort_persistence_ticks=2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _zone_series(result: Any, zone_id: str, ceiling: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, decision in zip(result.records, result.decisions, strict=True):
        co2 = float(record.plant.zones[zone_id]["co2_concentration"])
        rows.append(
            {
                "tick": int(record.plant.tick),
                "co2": co2,
                "excess": max(0.0, co2 - ceiling),
                "authority_state": decision.state.value,
                "reserve_command": float(decision.reserve_commands[zone_id]),
                "reserve_delivered_airflow": float(
                    record.reserve["system"]["total_delivered_airflow"]
                ),
            }
        )
    return rows


def run(manifest_path: Path, evidence_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    manifest = load_family_manifest(manifest_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_rows = {str(row["family_id"]): row for row in evidence["per_family"]}
    families: list[dict[str, Any]] = []
    for family in manifest.families:
        if family.family_id not in FAMILY_IDS:
            continue
        row = evidence_rows[family.family_id]
        config = load_scenario(family.fault_path)
        target = str(row["paired_metrics"]["target_zone_id"])
        ceiling = RECOVERY_RUN.crew_cabin_co2_concentration_ceiling
        result = run_recovery_scenario(
            config,
            run_id=f"co2-exception:{family.family_id}",
            governed=True,
            run=RECOVERY_RUN,
            settings=SETTINGS,
        )
        metrics = _arm_metrics(result, config=config, run=RECOVERY_RUN)
        off = row["arms"]["fault_reserve_off"]["metrics"]
        off_excess = float(off["integrated_physical_co2_excess"][target])
        governed_excess = float(metrics["integrated_physical_co2_excess"][target])
        series = _zone_series(result, target, ceiling)
        unsafe = [item for item in series if float(item["excess"]) > 0.0]
        first_protect = metrics["states"]["first_protect_tick"]
        around_activation = [
            item
            for item in series
            if first_protect is not None
            and int(first_protect) - 3 <= int(item["tick"]) <= int(first_protect) + 8
        ]
        families.append(
            {
                "family_id": family.family_id,
                "split": family.split,
                "target_zone_id": target,
                "safe_co2_ceiling": ceiling,
                "fault_reserve_off": {
                    "integrated_excess": off_excess,
                    "ticks_above_ceiling": int(off["ticks_above_ceiling"][target]),
                    "maximum_co2": float(off["maximum_physical_co2"][target]),
                },
                "fault_governed": {
                    "integrated_excess": governed_excess,
                    "excess_delta": governed_excess - off_excess,
                    "improvement_fraction": (
                        (off_excess - governed_excess) / off_excess
                        if off_excess > 0.0
                        else None
                    ),
                    "ticks_above_ceiling": int(metrics["ticks_above_ceiling"][target]),
                    "maximum_co2": float(metrics["maximum_physical_co2"][target]),
                    "reserve_delivered_airflow_integral": float(
                        metrics["reserve_delivered_airflow_integral"]
                    ),
                    "first_protect_tick": first_protect,
                    "first_handback_tick": metrics["states"]["first_handback_tick"],
                    "physical_zero_acknowledgement_tick": metrics["states"][
                        "physical_zero_acknowledgement_tick"
                    ],
                    "first_unsafe_tick": unsafe[0]["tick"] if unsafe else None,
                    "last_unsafe_tick": unsafe[-1]["tick"] if unsafe else None,
                },
                "around_activation": around_activation,
            }
        )
    if {item["family_id"] for item in families} != FAMILY_IDS:
        raise RuntimeError("did not resolve both expected families")
    report = {
        "schema_version": 1,
        "analysis_class": "development_debug_only",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "settings": asdict(SETTINGS),
        "settings_sha256": _canonical_sha256(asdict(SETTINGS)),
        "families": families,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: diagnose_recovery_co2_exceptions.py "
            "<families.json> <recovery-evidence.json> <output.json>"
        )
    report = run(
        Path(sys.argv[1]).resolve(),
        Path(sys.argv[2]).resolve(),
        Path(sys.argv[3]).resolve(),
    )
    print(
        json.dumps(
            {
                "output_sha256": _sha256(Path(sys.argv[3]).resolve()),
                "families": len(report["families"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
