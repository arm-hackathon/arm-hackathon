#!/usr/bin/env python3
"""Trace the two late C4 handback failures under a fixed candidate policy."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aeolus.config import load_scenario
from aeolus.families import load_family_manifest
from aeolus.recovery import RecoverySettings
from aeolus.scenario import RECOVERY_RUN, run_recovery_scenario

FAMILY_IDS = (
    "train-s213-primary-high-reserve-t55-lab-transient-blocked-d060-e650",
    "train-s213-primary-high-reserve-t55-lab-transient-degradation-d060-e750",
)
SETTINGS = RecoverySettings(
    entry_residual_ratio=0.10,
    entry_isolation_margin=0.05,
    entry_persistence_ticks=2,
    exit_residual_ratio=0.06,
    handback_abort_residual_ratio=0.09,
)


def _ratio(payload: dict[str, Any]) -> float:
    requested = float(payload["requested_airflow"])
    delivered = float(payload["delivered_airflow"])
    if requested <= 0.0:
        return 0.0
    return max(0.0, (requested - delivered) / requested)


def _analyse(record: Any, config: Any) -> dict[str, Any]:
    payloads = {
        zone.id: record.plant.connections[config.path_to_processing(zone.id).id]
        for zone in config.non_processing_zones()
    }
    residuals = {
        zone_id: _ratio(payload)
        for zone_id, payload in payloads.items()
    }
    candidates = [
        zone_id
        for zone_id, ratio in residuals.items()
        if ratio >= SETTINGS.entry_residual_ratio
        and float(payloads[zone_id]["requested_airflow"])
        >= SETTINGS.minimum_requested_fraction
    ]
    target = candidates[0] if len(candidates) == 1 else None
    runner_up = max(
        (ratio for zone_id, ratio in residuals.items() if zone_id != target),
        default=0.0,
    )
    isolated = (
        target is not None
        and residuals[target] - runner_up >= SETTINGS.entry_isolation_margin
    )
    return {
        "residuals": residuals,
        "entry_candidates": candidates,
        "isolated_target": target if isolated else None,
        "runner_up_residual": runner_up,
    }


def _max_streak(values: Sequence[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def run(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    manifest = load_family_manifest(manifest_path)
    by_id = {family.family_id: family for family in manifest.families}
    report: dict[str, Any] = {
        "status": "development_debug_only",
        "settings": asdict(SETTINGS),
        "settings_sha256": hashlib.sha256(
            json.dumps(asdict(SETTINGS), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "families": [],
    }
    for family_id in FAMILY_IDS:
        family = by_id[family_id]
        config = load_scenario(family.fault_path)
        raw_config = json.loads(Path(family.fault_path).read_text(encoding="utf-8"))
        fault_end_tick = int(raw_config["fault_profiles"][0]["end_tick"])
        result = run_recovery_scenario(
            config,
            run_id=f"handback-diagnostic:{family_id}",
            governed=True,
            run=RECOVERY_RUN,
            settings=SETTINGS,
        )
        post_fault_rows: list[dict[str, Any]] = []
        clear_flags: list[bool] = []
        non_abort_flags: list[bool] = []
        for index, record in enumerate(result.records):
            if record.plant.tick < fault_end_tick:
                continue
            analysis = _analyse(record, config)
            target_ratio = float(analysis["residuals"]["lab"])
            clear = (
                not analysis["entry_candidates"]
                and target_ratio <= SETTINGS.exit_residual_ratio
            )
            abort = bool(analysis["entry_candidates"]) or (
                target_ratio >= SETTINGS.handback_abort_residual_ratio
            )
            clear_flags.append(clear)
            non_abort_flags.append(not abort)
            decision = result.decisions[index]
            post_fault_rows.append(
                {
                    "tick": record.plant.tick,
                    "authority_state": decision.state.value,
                    "authority_target": decision.target_zone_id,
                    "target_residual": target_ratio,
                    "runner_up_residual": float(analysis["runner_up_residual"]),
                    "entry_candidates": analysis["entry_candidates"],
                    "isolated_target": analysis["isolated_target"],
                    "protect_clear": clear,
                    "handback_abort": abort,
                    "commanded_position": float(decision.reserve_commands["lab"]),
                    "total_reserve_delivered_airflow": float(
                        record.reserve["system"]["total_delivered_airflow"]
                    ),
                    "lab_reserve_actual_position": float(
                        record.reserve["actuators"]["lab"]["actual_position"]
                    ),
                }
            )
        report["families"].append(
            {
                "family_id": family_id,
                "fault_class": family.fault_class,
                "fault_end_tick": fault_end_tick,
                "events": [
                    {
                        "decision_tick": event.decision_tick,
                        "observation_tick": event.observation_tick,
                        "from_state": event.from_state.value,
                        "to_state": event.to_state.value,
                        "reason": event.reason,
                        "target_zone_id": event.target_zone_id,
                    }
                    for event in result.events
                ],
                "maximum_post_fault_clear_streak": _max_streak(clear_flags),
                "maximum_post_fault_non_abort_streak": _max_streak(non_abort_flags),
                "post_fault_timeline": post_fault_rows,
            }
        )
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: diagnose_recovery_handback_failures.py <families.json> <output.json>",
            file=sys.stderr,
        )
        return 2
    report = run(Path(args[0]).resolve(), Path(args[1]).resolve())
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "families": [
                    {
                        "family_id": row["family_id"],
                        "events": row["events"],
                        "maximum_post_fault_clear_streak": row[
                            "maximum_post_fault_clear_streak"
                        ],
                    }
                    for row in report["families"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
