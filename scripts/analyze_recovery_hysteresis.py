#!/usr/bin/env python3
"""Compare ordered recovery hysteresis pairs through the real closed loop.

The experiment reuses the preserved C4 family corpus. Candidate selection is
based only on training families. The already-inspected validation split is
reported as development validation, not as a fresh blind set.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aeolus.config import load_scenario
from aeolus.families import load_family_manifest
from aeolus.recovery import AuthorityState, RecoverySettings
from aeolus.recovery_evidence import _arm_metrics
from aeolus.scenario import RECOVERY_RUN, run_recovery_scenario

REPORT_VERSION = "aeolus_recovery_hysteresis_probe_v1"
ENTRY_RESIDUAL_RATIO = 0.10
ENTRY_ISOLATION_MARGIN = 0.05
ENTRY_PERSISTENCE_TICKS = 2
MINIMUM_HYSTERESIS_GAP = 0.02
SOFT_ABORT_PERSISTENCE_TICKS = 2
CANDIDATE_PAIRS = (
    (0.04, 0.08),
    (0.05, 0.08),
    (0.06, 0.08),
    (0.06, 0.09),
)
TRANSIENT_CLASSES = {
    "transient_blocked_path",
    "transient_gradual_primary_fan_degradation",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def _source_provenance() -> dict[str, Any]:
    diff = _git_output("diff", "--binary", "HEAD", binary=True)
    assert isinstance(diff, bytes)
    status = _git_output("status", "--porcelain")
    head = _git_output("rev-parse", "HEAD")
    assert isinstance(status, str) and isinstance(head, str)
    return {
        "commit": head.strip(),
        "worktree_dirty": bool(status.strip()),
        "status_lines": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _improvement(off: float, governed: float) -> float | None:
    if off <= 1e-12:
        return None
    return (off - governed) / off


def _candidate_settings(exit_ratio: float, abort_ratio: float) -> RecoverySettings:
    return RecoverySettings(
        entry_residual_ratio=ENTRY_RESIDUAL_RATIO,
        entry_isolation_margin=ENTRY_ISOLATION_MARGIN,
        entry_persistence_ticks=ENTRY_PERSISTENCE_TICKS,
        exit_residual_ratio=exit_ratio,
        handback_abort_residual_ratio=abort_ratio,
        handback_abort_persistence_ticks=SOFT_ABORT_PERSISTENCE_TICKS,
    )


def _candidate_id(exit_ratio: float, abort_ratio: float) -> str:
    return (
        f"exit-{round(exit_ratio * 100):02d}"
        f"-abort-{round(abort_ratio * 100):02d}"
        f"-soft-p{SOFT_ABORT_PERSISTENCE_TICKS}"
    )


def _healthy_summary(
    reference_rows: dict[str, dict[str, Any]], settings: RecoverySettings
) -> dict[str, Any]:
    false_protect_ids: list[str] = []
    regressed_ids: list[str] = []
    for reference_path, row in sorted(reference_rows.items()):
        config = load_scenario(reference_path)
        result = run_recovery_scenario(
            config,
            run_id=f"hysteresis:{_canonical_sha256(asdict(settings))[:12]}:healthy:{Path(reference_path).stem}",
            governed=True,
            run=RECOVERY_RUN,
            settings=settings,
        )
        metrics = _arm_metrics(result, config=config, run=RECOVERY_RUN)
        if metrics["states"]["first_protect_tick"] is not None:
            false_protect_ids.append(reference_path)
        baseline = row["arms"]["reference_reserve_off"]["metrics"]
        if any(
            float(metrics["integrated_physical_co2_excess"][zone_id])
            > float(baseline["integrated_physical_co2_excess"][zone_id]) + 1e-12
            or int(metrics["ticks_above_ceiling"][zone_id])
            > int(baseline["ticks_above_ceiling"][zone_id])
            for zone_id in baseline["integrated_physical_co2_excess"]
        ):
            regressed_ids.append(reference_path)
    return {
        "unique_reference_scenarios": len(reference_rows),
        "false_protect_count": len(false_protect_ids),
        "false_protect_scenarios": false_protect_ids,
        "physical_non_regression": not regressed_ids,
        "regressed_scenarios": regressed_ids,
    }


def _evaluate_transient(row: dict[str, Any], settings: RecoverySettings) -> dict[str, Any]:
    config = load_scenario(row["fault_scenario_path"])
    family_id = str(row["family_id"])
    expected_target = str(row["paired_metrics"]["target_zone_id"])
    result = run_recovery_scenario(
        config,
        run_id=f"hysteresis:{_canonical_sha256(asdict(settings))[:12]}:{family_id}",
        governed=True,
        run=RECOVERY_RUN,
        settings=settings,
    )
    metrics = _arm_metrics(result, config=config, run=RECOVERY_RUN)
    states = metrics["states"]
    protect_events = [event for event in result.events if event.to_state is AuthorityState.PROTECT]
    recurrence_events = [
        event
        for event in result.events
        if event.from_state is AuthorityState.HANDBACK
        and event.to_state is AuthorityState.PROTECT
    ]
    protect_targets = {
        decision.target_zone_id
        for decision in result.decisions
        if decision.state is AuthorityState.PROTECT
    }
    timeout = any(event.reason == "handback_timeout" for event in result.events)
    reserve_failure = any(
        event.reason == "reserve_delivery_failure" for event in result.events
    )
    final_records = result.records[-5:]
    final_physical_zero = all(
        float(record.reserve["system"]["total_delivered_airflow"]) <= 1e-12
        and all(
            float(actuator["actual_position"]) <= 1e-12
            for actuator in record.reserve["actuators"].values()
        )
        for record in final_records
    )
    off_metrics = row["arms"]["fault_reserve_off"]["metrics"]
    off_excess = float(off_metrics["integrated_physical_co2_excess"][expected_target])
    governed_excess = float(metrics["integrated_physical_co2_excess"][expected_target])
    return {
        "family_id": family_id,
        "split": row["split"],
        "expected_target": expected_target,
        "protected": states["first_protect_tick"] is not None,
        "wrong_target": bool(protect_targets - {expected_target}),
        "protect_targets": sorted(target for target in protect_targets if target is not None),
        "protect_entry_count": len(protect_events),
        "handback_recurrence_count": len(recurrence_events),
        "handback_started": states["first_handback_tick"] is not None,
        "physical_zero_acknowledged": states["physical_zero_acknowledgement_tick"] is not None,
        "handback_timeout": timeout,
        "reserve_failure": reserve_failure,
        "final_physical_zero": final_physical_zero,
        "first_protect_tick": states["first_protect_tick"],
        "first_handback_tick": states["first_handback_tick"],
        "physical_zero_acknowledgement_tick": states[
            "physical_zero_acknowledgement_tick"
        ],
        "reserve_delivered_airflow_integral": float(
            metrics["reserve_delivered_airflow_integral"]
        ),
        "off_integrated_target_excess": off_excess,
        "governed_integrated_target_excess": governed_excess,
        "integrated_excess_improvement_fraction": _improvement(
            off_excess, governed_excess
        ),
    }


def _split_summary(records: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [record for record in records if record["split"] == split]
    defined = [
        float(record["integrated_excess_improvement_fraction"])
        for record in selected
        if record["integrated_excess_improvement_fraction"] is not None
    ]
    return {
        "families": len(selected),
        "protected": sum(bool(record["protected"]) for record in selected),
        "missed_family_ids": [
            record["family_id"] for record in selected if not record["protected"]
        ],
        "wrong_target_count": sum(bool(record["wrong_target"]) for record in selected),
        "wrong_target_family_ids": [
            record["family_id"] for record in selected if record["wrong_target"]
        ],
        "nonpositive_excess_improvement_family_ids": [
            record["family_id"]
            for record in selected
            if record["integrated_excess_improvement_fraction"] is not None
            and float(record["integrated_excess_improvement_fraction"]) <= 0.0
        ],
        "single_protect_episode": sum(
            int(record["protect_entry_count"]) == 1 for record in selected
        ),
        "multiple_protect_episode_family_ids": [
            record["family_id"]
            for record in selected
            if int(record["protect_entry_count"]) > 1
        ],
        "handback_started": sum(bool(record["handback_started"]) for record in selected),
        "no_handback_family_ids": [
            record["family_id"] for record in selected if not record["handback_started"]
        ],
        "physical_zero_acknowledged": sum(
            bool(record["physical_zero_acknowledged"]) for record in selected
        ),
        "no_physical_zero_acknowledgement_family_ids": [
            record["family_id"]
            for record in selected
            if not record["physical_zero_acknowledged"]
        ],
        "final_physical_zero": sum(
            bool(record["final_physical_zero"]) for record in selected
        ),
        "nonzero_final_family_ids": [
            record["family_id"]
            for record in selected
            if not record["final_physical_zero"]
        ],
        "handback_recurrence_count": sum(
            int(record["handback_recurrence_count"]) for record in selected
        ),
        "handback_recurrence_family_ids": [
            record["family_id"]
            for record in selected
            if int(record["handback_recurrence_count"]) > 0
        ],
        "handback_timeout_count": sum(
            bool(record["handback_timeout"]) for record in selected
        ),
        "handback_timeout_family_ids": [
            record["family_id"] for record in selected if record["handback_timeout"]
        ],
        "reserve_failure_count": sum(
            bool(record["reserve_failure"]) for record in selected
        ),
        "reserve_failure_family_ids": [
            record["family_id"] for record in selected if record["reserve_failure"]
        ],
        "median_first_protect_tick": _median(
            [float(record["first_protect_tick"]) for record in selected if record["first_protect_tick"] is not None]
        ),
        "median_physical_zero_acknowledgement_tick": _median(
            [
                float(record["physical_zero_acknowledgement_tick"])
                for record in selected
                if record["physical_zero_acknowledgement_tick"] is not None
            ]
        ),
        "defined_excess_improvement_families": len(defined),
        "positive_excess_improvement_fraction": (
            sum(value > 0.0 for value in defined) / len(defined) if defined else None
        ),
        "median_excess_improvement_fraction": _median(defined),
        "median_reserve_delivered_airflow_integral": _median(
            [float(record["reserve_delivered_airflow_integral"]) for record in selected]
        ),
    }


def _train_safe(candidate: dict[str, Any]) -> bool:
    train = candidate["splits"]["train"]
    healthy = candidate["healthy"]
    gaps = candidate["hysteresis_gaps"]
    total = int(train["families"])
    return (
        float(gaps["entry_to_abort"]) >= MINIMUM_HYSTERESIS_GAP - 1e-12
        and float(gaps["abort_to_exit"]) >= MINIMUM_HYSTERESIS_GAP - 1e-12
        and healthy["false_protect_count"] == 0
        and healthy["physical_non_regression"]
        and train["protected"] == total
        and train["wrong_target_count"] == 0
        and train["single_protect_episode"] == total
        and train["handback_started"] == total
        and train["physical_zero_acknowledged"] == total
        and train["final_physical_zero"] == total
        and train["handback_recurrence_count"] == 0
        and train["handback_timeout_count"] == 0
        and train["reserve_failure_count"] == 0
    )


def _selection_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    train = candidate["splits"]["train"]
    return (
        float(train["positive_excess_improvement_fraction"] or -math.inf),
        float(train["median_excess_improvement_fraction"] or -math.inf),
        -float(train["median_reserve_delivered_airflow_integral"] or math.inf),
        float(candidate["settings"]["exit_residual_ratio"]),
        float(candidate["hysteresis_gaps"]["entry_to_abort"]),
    )


def _augment_manifest_rows(manifest: Any, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {str(row["family_id"]): row for row in evidence["per_family"]}
    rows: list[dict[str, Any]] = []
    for family in manifest.families:
        row = dict(by_id[family.family_id])
        row["reference_scenario_path"] = str(family.reference_path)
        row["fault_scenario_path"] = str(family.fault_path)
        rows.append(row)
    return rows


def run(manifest_path: Path, evidence_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    manifest = load_family_manifest(manifest_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows = _augment_manifest_rows(manifest, evidence)
    transient_rows = [row for row in rows if row["fault_class"] in TRANSIENT_CLASSES]
    reference_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        reference_rows.setdefault(str(row["reference_scenario_path"]), row)

    candidates: list[dict[str, Any]] = []
    for exit_ratio, abort_ratio in CANDIDATE_PAIRS:
        settings = _candidate_settings(exit_ratio, abort_ratio)
        records = [_evaluate_transient(row, settings) for row in transient_rows]
        candidate = {
            "candidate_id": _candidate_id(exit_ratio, abort_ratio),
            "settings": asdict(settings),
            "settings_sha256": _canonical_sha256(asdict(settings)),
            "hysteresis_gaps": {
                "entry_to_abort": ENTRY_RESIDUAL_RATIO - abort_ratio,
                "abort_to_exit": abort_ratio - exit_ratio,
                "minimum_required": MINIMUM_HYSTERESIS_GAP,
            },
            "healthy": _healthy_summary(reference_rows, settings),
            "splits": {
                split: _split_summary(records, split)
                for split in ("train", "validation")
            },
        }
        candidate["train_safe"] = _train_safe(candidate)
        candidates.append(candidate)

    eligible = [candidate for candidate in candidates if candidate["train_safe"]]
    selected = max(eligible, key=_selection_key) if eligible else None
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "status": "development_debug_only",
        "warning": (
            "The validation split has already been inspected during sequential policy research "
            "and is not a fresh blind set."
        ),
        "source": _source_provenance(),
        "inputs": {
            "family_manifest": str(manifest_path.resolve()),
            "family_manifest_file_sha256": _sha256_file(manifest_path),
            "family_manifest_canonical_sha256": manifest.manifest_sha256,
            "reference_evidence": str(evidence_path.resolve()),
            "reference_evidence_file_sha256": _sha256_file(evidence_path),
            "reference_evidence_sha256": evidence["evidence_sha256"],
            "families": len(rows),
            "transient_families": len(transient_rows),
            "unique_reference_scenarios": len(reference_rows),
        },
        "frozen_entry_contract": {
            "entry_residual_ratio": ENTRY_RESIDUAL_RATIO,
            "entry_isolation_margin": ENTRY_ISOLATION_MARGIN,
            "entry_persistence_ticks": ENTRY_PERSISTENCE_TICKS,
            "soft_handback_abort_persistence_ticks": SOFT_ABORT_PERSISTENCE_TICKS,
        },
        "selection_rule": {
            "split": "train_only",
            "hard_gates": [
                "minimum 0.02 entry-to-abort and abort-to-exit gaps",
                "zero healthy protection",
                "healthy physical non-regression",
                "all transient faults protected at the expected target",
                "one protection episode per transient family",
                "handback and physical-zero acknowledgement for every transient family",
                "zero handback recurrence, timeout, reserve failure, or nonzero final state",
            ],
            "ranking_after_hard_gates": [
                "higher positive CO2-excess improvement fraction",
                "higher median CO2-excess improvement",
                "lower median reserve delivery",
                "higher exit threshold within the safe plateau",
                "wider entry-to-abort gap",
            ],
        },
        "candidates": candidates,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_settings": selected["settings"] if selected else None,
        "selected_settings_sha256": selected["settings_sha256"] if selected else None,
        "claims_not_supported": [
            "fresh blind generalization",
            "real spacecraft or building safety",
            "physical hardware behaviour",
            "submission-grade reproducibility before a write-once four-arm rerun",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: analyze_recovery_hysteresis.py <families.json> <reference-evidence.json> <output.json>",
            file=sys.stderr,
        )
        return 2
    report = run(Path(args[0]).resolve(), Path(args[1]).resolve(), Path(args[2]).resolve())
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "selected_candidate_id": report["selected_candidate_id"],
                "selected_settings_sha256": report["selected_settings_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
