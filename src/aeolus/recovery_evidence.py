"""Write-once four-arm evidence runner for deterministic recovery suites."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aeolus.config import load_scenario
from aeolus.families import (
    RECOVERY_COUNTERFACTUAL_ARMS,
    ScenarioFamily,
    load_family_manifest,
)
from aeolus.recovery import AuthorityState, RecoverySettings
from aeolus.scenario import (
    RECOVERY_RUN,
    RecoveryRunResult,
    RunSpec,
    run_recovery_scenario,
)
from aeolus.sweep import SWEEP_V4_VERSION, generate_sweep, load_sweep_spec

RECOVERY_EVIDENCE_VERSION = "aeolus_recovery_evidence_v1"
RECOVERY_ARMS = RECOVERY_COUNTERFACTUAL_ARMS
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EPSILON = 1e-12
USAGE = "Usage: python -m aeolus.recovery_evidence <recovery-sweep.json> <output-dir>"


def run_recovery_evidence(
    sweep_path: str | Path,
    output_dir: str | Path,
    *,
    run: RunSpec = RECOVERY_RUN,
    require_clean_source: bool = True,
    settings: RecoverySettings | None = None,
) -> dict[str, Any]:
    """Run every v4 development or final family in four fixed recovery arms."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"recovery evidence output already exists: {destination}")

    sweep = load_sweep_spec(sweep_path)
    if (
        sweep.schema_version != SWEEP_V4_VERSION
        or sweep.suite_role not in {"development", "final"}
    ):
        raise ValueError(
            "recovery evidence requires an aeolus_sweep_v4 development or final suite"
        )
    selected_settings = settings if settings is not None else RecoverySettings()
    if not isinstance(selected_settings, RecoverySettings):
        raise TypeError("recovery evidence settings are malformed")
    provenance = _source_provenance(require_clean_source=require_clean_source)

    destination.mkdir(parents=True)
    corpus_dir = destination / "corpus"
    generate_sweep(sweep_path, corpus_dir)
    manifest = load_family_manifest(corpus_dir / "families.json")
    if any(family.counterfactual_arms != RECOVERY_ARMS for family in manifest.families):
        raise ValueError("recovery manifest does not declare the required four arms")

    trace_dir = destination / "traces"
    rows = [
        _evaluate_family(
            family,
            trace_dir=trace_dir,
            run=run,
            settings=selected_settings,
        )
        for family in manifest.families
    ]
    final_provenance = _source_provenance(require_clean_source=False)
    if final_provenance != provenance:
        raise ValueError("source provenance changed during recovery evidence generation")
    settings_document = asdict(selected_settings)
    receipt: dict[str, Any] = {
        "evidence_version": RECOVERY_EVIDENCE_VERSION,
        "environment": provenance["environment"],
        "source": provenance["source"],
        "sweep": {
            "schema_version": sweep.schema_version,
            "suite_role": sweep.suite_role,
            "bytes_sha256": _sha256_file(sweep.source_path),
            "canonical_sha256": sweep.sha256,
            "family_manifest_sha256": manifest.manifest_sha256,
        },
        "recovery_settings": settings_document,
        "recovery_settings_sha256": _canonical_sha256(settings_document),
        "arms": list(RECOVERY_ARMS),
        "run_spec": asdict(run),
        "run_spec_sha256": _canonical_sha256(asdict(run)),
        "families_evaluated": len(rows),
        "per_family": rows,
        "gates": _evaluate_gates(
            rows,
            evaluation_split="validation" if sweep.suite_role == "development" else "final",
        ),
    }
    receipt["evidence_sha256"] = _canonical_sha256(receipt)
    _write_json_new(destination / "recovery-evidence.json", receipt)
    return receipt


def reproduce_recovery_evidence(
    sweep_path: str | Path,
    first_output_dir: str | Path,
    second_output_dir: str | Path,
    *,
    run: RunSpec = RECOVERY_RUN,
    require_clean_source: bool = True,
    settings: RecoverySettings | None = None,
) -> dict[str, Any]:
    """Run and compare two relocated recovery-evidence reproductions."""
    first = run_recovery_evidence(
        sweep_path,
        first_output_dir,
        run=run,
        require_clean_source=require_clean_source,
        settings=settings,
    )
    second = run_recovery_evidence(
        sweep_path,
        second_output_dir,
        run=run,
        require_clean_source=require_clean_source,
        settings=settings,
    )
    first_dir = Path(first_output_dir).resolve()
    second_dir = Path(second_output_dir).resolve()
    first_files = _output_file_tree(first_dir)
    second_files = _output_file_tree(second_dir)
    all_paths = sorted(set(first_files).union(second_files))
    mismatched_files = [
        path for path in all_paths if first_files.get(path) != second_files.get(path)
    ]
    return {
        "first_evidence_sha256": first["evidence_sha256"],
        "second_evidence_sha256": second["evidence_sha256"],
        "first_file_count": len(first_files),
        "second_file_count": len(second_files),
        "trace_count": sum(
            path.startswith("traces/") and path.endswith(".jsonl")
            for path in first_files
        ),
        "mismatched_files": mismatched_files,
        "byte_identical": not mismatched_files,
    }


def _output_file_tree(root: Path) -> dict[str, str]:
    """Digest every generated file under its relocation-stable relative path."""
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _evaluate_family(
    family: ScenarioFamily,
    *,
    trace_dir: Path,
    run: RunSpec,
    settings: RecoverySettings,
) -> dict[str, Any]:
    """Run a paired recovery family under each immutable authority arm."""
    reference_config = load_scenario(family.reference_path)
    fault_config = load_scenario(family.fault_path)
    arm_inputs = (
        ("reference_reserve_off", reference_config, False),
        ("reference_governed", reference_config, True),
        ("fault_reserve_off", fault_config, False),
        ("fault_governed", fault_config, True),
    )
    arms: dict[str, dict[str, Any]] = {}
    results: dict[str, RecoveryRunResult] = {}
    for arm_name, config, governed in arm_inputs:
        trace_path = trace_dir / f"{family.family_id}--{arm_name}.jsonl"
        result = run_recovery_scenario(
            config,
            run_id=f"{family.family_id}:{arm_name}",
            governed=governed,
            run=run,
            trace_path=trace_path,
            settings=settings if governed else None,
        )
        results[arm_name] = result
        arms[arm_name] = {
            "scenario_sha256": _sha256_file(
                family.reference_path if arm_name.startswith("reference") else family.fault_path
            ),
            "trace_path": trace_path.name,
            "trace_sha256": _sha256_file(trace_path),
            "metrics": _arm_metrics(result, config=config, run=run),
        }
    paired_metrics = _paired_metrics(
        arms,
        results,
        fault_config=fault_config,
        run=run,
    )
    arms["fault_governed"]["metrics"]["steady_state_restoration_fraction"] = (
        paired_metrics["steady_state_restoration_fraction"]
    )
    return {
        "family_id": family.family_id,
        "counterfactual_group_id": family.counterfactual_group_id,
        "base_condition_id": family.base_condition_id,
        "split": family.split,
        "fault_class": family.fault_class,
        "arms": arms,
        "paired_metrics": paired_metrics,
    }


def _paired_metrics(
    arms: dict[str, dict[str, Any]],
    results: dict[str, RecoveryRunResult],
    *,
    fault_config: Any,
    run: RunSpec,
) -> dict[str, Any]:
    target_zone = _affected_zone(fault_config)
    fault_off = arms["fault_reserve_off"]["metrics"]
    fault_governed = arms["fault_governed"]["metrics"]
    reference_off = arms["reference_reserve_off"]["metrics"]
    reference_governed = arms["reference_governed"]["metrics"]
    off_excess = float(fault_off["integrated_physical_co2_excess"][target_zone])
    governed_excess = float(
        fault_governed["integrated_physical_co2_excess"][target_zone]
    )
    first_handback = fault_governed["states"]["first_handback_tick"]
    acknowledged_zero = fault_governed["states"]["physical_zero_acknowledgement_tick"]
    transient = any(
        type(profile).__name__.startswith("Transient")
        for profile in fault_config.fault_profiles
    )
    return {
        "target_zone_id": target_zone,
        "eligible_physical_airflow_fault": not bool(fault_config.sensor_faults()),
        "integrated_excess_improvement_fraction": _ratio(
            off_excess - governed_excess, off_excess
        ),
        "total_delivered_airflow_delta": (
            float(fault_governed["total_delivered_airflow_integral"])
            - float(fault_off["total_delivered_airflow_integral"])
        ),
        "steady_state_restoration_fraction": _steady_state_restoration(
            results,
            fault_config=fault_config,
            target_zone=target_zone,
        ),
        "preactivation_physical_parity": _preactivation_plant_parity(
            results["fault_reserve_off"],
            results["fault_governed"],
            fault_governed["states"]["first_protect_tick"],
        ),
        "healthy_reference_non_regression": _healthy_non_regression(
            reference_off, reference_governed, run=run
        ),
        "transient_handback_acknowledged": (
            not transient
            or (
                first_handback is not None
                and acknowledged_zero is not None
                and acknowledged_zero - first_handback <= 36
            )
        ),
        "failed_reserve_no_rearm": _failed_reserve_does_not_rearm(
            results["fault_governed"]
        ),
    }


def _affected_zone(config: Any) -> str:
    for profile in config.connection_faults():
        for zone in config.non_processing_zones():
            if profile.connection_id in {
                config.path_to_processing(zone.id).id,
                config.path_from_processing(zone.id).id,
            }:
                return zone.id
    for profile in config.sensor_faults():
        return profile.zone_id
    raise ValueError("recovery family fault does not identify a target zone")


def _preactivation_plant_parity(
    reserve_off: RecoveryRunResult,
    governed: RecoveryRunResult,
    first_protect_tick: int | None,
) -> bool:
    limit = len(governed.records) if first_protect_tick is None else first_protect_tick - 1
    return all(
        left.plant == right.plant
        for left, right in zip(
            reserve_off.records[:limit], governed.records[:limit], strict=True
        )
    )


def _steady_state_restoration(
    results: dict[str, RecoveryRunResult],
    *,
    fault_config: Any,
    target_zone: str,
) -> dict[str, Any]:
    governed = results["fault_governed"]
    last_moving = max(
        (
            index
            for index, record in enumerate(governed.records)
            if any(
                bool(values["moving"])
                for values in record.reserve["actuators"].values()
            )
        ),
        default=-1,
    )
    start = last_moving + 1
    if last_moving < 0 or start >= len(governed.records):
        return {"status": "not_applicable", "value": None}

    reference = results["reference_reserve_off"]
    reserve_off = results["fault_reserve_off"]
    denominator = math.fsum(
        max(
            0.0,
            _zone_delivered(reference.records[index], fault_config, target_zone)
            - _zone_delivered(reserve_off.records[index], fault_config, target_zone),
        )
        for index in range(start, len(governed.records))
    )
    numerator = math.fsum(
        max(
            0.0,
            _zone_delivered(governed.records[index], fault_config, target_zone)
            - _zone_delivered(reserve_off.records[index], fault_config, target_zone),
        )
        for index in range(start, len(governed.records))
    )
    return _ratio(numerator, denominator, cap=1.0)


def _zone_delivered(record: Any, config: Any, zone_id: str) -> float:
    primary = float(
        record.plant.connections[config.path_to_processing(zone_id).id][
            "delivered_airflow"
        ]
    )
    reserve = float(
        record.reserve["connections"][config.reserve_path_to_processing(zone_id).id][
            "delivered_airflow"
        ]
    )
    return primary + reserve


def _healthy_non_regression(
    reserve_off: dict[str, Any], governed: dict[str, Any], *, run: RunSpec
) -> bool:
    del run
    return all(
        float(governed["integrated_physical_co2_excess"][zone_id])
        <= float(reserve_off["integrated_physical_co2_excess"][zone_id]) + _EPSILON
        and int(governed["ticks_above_ceiling"][zone_id])
        <= int(reserve_off["ticks_above_ceiling"][zone_id])
        for zone_id in reserve_off["integrated_physical_co2_excess"]
    )


def _failed_reserve_does_not_rearm(result: RecoveryRunResult) -> bool:
    for index, decision in enumerate(result.decisions):
        if decision.reason == "reserve_delivery_failure":
            return all(
                later.state is not AuthorityState.PROTECT
                for later in result.decisions[index + 1 :]
            )
    return True


def _evaluate_gates(
    rows: Sequence[dict[str, Any]], *, evaluation_split: str
) -> dict[str, Any]:
    arms = [arm for row in rows for arm in row["arms"].values()]
    harmful_physical = [
        row
        for row in rows
        if row["paired_metrics"]["eligible_physical_airflow_fault"]
        and float(
            row["arms"]["fault_reserve_off"]["metrics"][
                "integrated_physical_co2_excess"
            ][row["paired_metrics"]["target_zone_id"]]
        )
        > _EPSILON
    ]
    missed_harmful = [
        row["family_id"]
        for row in harmful_physical
        if row["arms"]["fault_governed"]["metrics"]["states"][
            "first_protect_tick"
        ]
        is None
    ]
    wrong_target = [
        row["family_id"]
        for row in rows
        if set(
            row["arms"]["fault_governed"]["metrics"]["lifecycle"][
                "protect_target_zone_ids"
            ]
        )
        - {row["paired_metrics"]["target_zone_id"]}
    ]
    transients = [row for row in rows if row["fault_class"].startswith("transient_")]
    multiple_or_missing_protect = [
        row["family_id"]
        for row in transients
        if row["arms"]["fault_governed"]["metrics"]["lifecycle"][
            "protect_entry_count"
        ]
        != 1
    ]
    handback_recurrence = [
        row["family_id"]
        for row in transients
        if row["arms"]["fault_governed"]["metrics"]["lifecycle"][
            "handback_recurrence_count"
        ]
        != 0
    ]
    handback_timeout = [
        row["family_id"]
        for row in transients
        if row["arms"]["fault_governed"]["metrics"]["lifecycle"][
            "handback_timeout_count"
        ]
        != 0
    ]
    nonzero_final = [
        row["family_id"]
        for row in transients
        if not row["arms"]["fault_governed"]["metrics"]["lifecycle"][
            "final_physical_zero"
        ]
    ]
    safety: dict[str, Any] = {
        "zero_invariant_violations": _gate(
            all(arm["metrics"]["invariant_violation_count"] == 0 for arm in arms),
            {"families": len(rows)},
        ),
        "reserve_off_zero_delivery": _gate(
            all(
                arm["metrics"]["reserve_delivered_airflow_integral"] <= _EPSILON
                for row in rows
                for name, arm in row["arms"].items()
                if name.endswith("reserve_off")
            ),
            {"families": len(rows)},
        ),
        "healthy_governed_no_protect": _gate(
            all(
                row["arms"]["reference_governed"]["metrics"]["states"][
                    "first_protect_tick"
                ]
                is None
                for row in rows
            ),
            {"families": len(rows)},
        ),
        "frozen_sensor_no_protect": _gate(
            all(
                row["arms"]["fault_governed"]["metrics"]["states"][
                    "first_protect_tick"
                ]
                is None
                for row in rows
                if row["fault_class"] == "frozen_sensor"
            ),
            {"families": sum(row["fault_class"] == "frozen_sensor" for row in rows)},
        ),
        "harmful_physical_fault_protected": _gate(
            not missed_harmful,
            {
                "families": len(harmful_physical),
                "missed_family_ids": missed_harmful,
            },
        ),
        "fault_governed_expected_target_only": _gate(
            not wrong_target,
            {"families": len(rows), "wrong_target_family_ids": wrong_target},
        ),
        "transient_single_protect_episode": _gate(
            not multiple_or_missing_protect,
            {
                "families": len(transients),
                "failed_family_ids": multiple_or_missing_protect,
            },
        ),
        "transient_zero_handback_recurrence": _gate(
            not handback_recurrence,
            {
                "families": len(transients),
                "failed_family_ids": handback_recurrence,
            },
        ),
        "transient_zero_handback_timeout": _gate(
            not handback_timeout,
            {
                "families": len(transients),
                "failed_family_ids": handback_timeout,
            },
        ),
        "transient_final_physical_zero": _gate(
            not nonzero_final,
            {
                "families": len(transients),
                "failed_family_ids": nonzero_final,
            },
        ),
        "preactivation_physical_parity": _gate(
            all(row["paired_metrics"]["preactivation_physical_parity"] for row in rows),
            {"families": len(rows)},
        ),
        "transient_handback_acknowledged": _gate(
            all(
                row["paired_metrics"]["transient_handback_acknowledged"]
                for row in rows
                if row["fault_class"].startswith("transient_")
            ),
            {"families": sum(row["fault_class"].startswith("transient_") for row in rows)},
        ),
        "failed_reserve_no_rearm": _gate(
            all(row["paired_metrics"]["failed_reserve_no_rearm"] for row in rows),
            {"families": len(rows)},
        ),
    }
    safety["passed"] = all(entry["passed"] for entry in safety.values())
    benefit = _benefit_gates(rows, evaluation_split=evaluation_split)
    return {
        "safety": safety,
        "benefit": benefit,
        "duplicate_execution": {"status": "not_run", "passed": None},
    }


def _benefit_gates(
    rows: Sequence[dict[str, Any]], *, evaluation_split: str
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["split"] == evaluation_split
        and row["paired_metrics"]["eligible_physical_airflow_fault"]
    ]
    defined = [
        row
        for row in eligible
        if row["paired_metrics"]["integrated_excess_improvement_fraction"]["status"]
        == "defined"
    ]
    improvements = [
        float(row["paired_metrics"]["integrated_excess_improvement_fraction"]["value"])
        for row in defined
    ]
    delivery_deltas = [
        float(row["paired_metrics"]["total_delivered_airflow_delta"])
        for row in defined
    ]
    median_improvement = _median(improvements)
    median_delivery_delta = _median(delivery_deltas)
    improvement_fraction = (
        sum(value > 0.0 for value in improvements) / len(improvements)
        if improvements
        else None
    )
    gates: dict[str, Any] = {
        "evaluation_split": evaluation_split,
        "physical_reserve_delivery_for_benefit": _gate(
            bool(defined)
            and all(
                row["arms"]["fault_governed"]["metrics"][
                    "reserve_delivered_airflow_integral"
                ]
                > _EPSILON
                for row in defined
            ),
            {"eligible_defined_families": len(defined)},
        ),
        "median_excess_improvement": _gate(
            median_improvement is not None and median_improvement >= 0.05,
            {"median_fraction": median_improvement, "threshold": 0.05},
        ),
        "validation_improvement_fraction": _gate(
            improvement_fraction is not None and improvement_fraction >= 0.60,
            {"fraction": improvement_fraction, "threshold": 0.60},
        ),
        "median_total_delivery_non_regression": _gate(
            median_delivery_delta is not None and median_delivery_delta >= -_EPSILON,
            {"median_delta": median_delivery_delta},
        ),
        "healthy_reference_non_regression": _gate(
            all(row["paired_metrics"]["healthy_reference_non_regression"] for row in rows),
            {"families": len(rows)},
        ),
        "undefined_benefit_denominator_families": {
            "count": len(eligible) - len(defined),
            "status": "reported_separately",
        },
    }
    gates["passed"] = all(
        gates[name]["passed"]
        for name in (
            "physical_reserve_delivery_for_benefit",
            "median_excess_improvement",
            "validation_improvement_fraction",
            "median_total_delivery_non_regression",
            "healthy_reference_non_regression",
        )
    )
    return gates


def _gate(passed: bool, observed: dict[str, Any]) -> dict[str, Any]:
    return {"passed": passed, "observed": observed}


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _arm_metrics(
    result: RecoveryRunResult, *, config: Any, run: RunSpec
) -> dict[str, Any]:
    zone_ids = tuple(zone.id for zone in config.non_processing_zones())
    primary_requested = 0.0
    primary_delivered = 0.0
    reserve_requested = 0.0
    reserve_delivered = 0.0
    reserve_saturation_ticks = 0
    invariant_violations = 0
    integrated_co2 = {zone_id: 0.0 for zone_id in zone_ids}
    integrated_excess = {zone_id: 0.0 for zone_id in zone_ids}
    ticks_above = {zone_id: 0 for zone_id in zone_ids}
    maximum_co2 = {zone_id: 0.0 for zone_id in zone_ids}

    for record in result.records:
        primary_tick_requested = math.fsum(
            float(record.plant.connections[config.path_to_processing(zone_id).id]["requested_airflow"])
            for zone_id in zone_ids
        )
        primary_tick_delivered = math.fsum(
            float(record.plant.connections[config.path_to_processing(zone_id).id]["delivered_airflow"])
            for zone_id in zone_ids
        )
        reserve_tick_requested = float(record.reserve["system"]["total_requested_airflow"])
        reserve_tick_delivered = float(record.reserve["system"]["total_delivered_airflow"])
        primary_requested += primary_tick_requested
        primary_delivered += primary_tick_delivered
        reserve_requested += reserve_tick_requested
        reserve_delivered += reserve_tick_delivered
        reserve_capacity = float(record.reserve["system"]["reserve_airflow_capacity"])
        if (
            reserve_tick_requested >= reserve_capacity - _EPSILON
            and reserve_tick_requested > _EPSILON
        ):
            reserve_saturation_ticks += 1
        if (
            primary_tick_delivered + reserve_tick_delivered
            > config.air_system.shared_airflow_capacity
            + config.air_system.reserve_airflow_capacity
            + _EPSILON
        ):
            invariant_violations += 1
        for zone_id in zone_ids:
            concentration = float(record.plant.zones[zone_id]["co2_concentration"])
            integrated_co2[zone_id] += concentration
            integrated_excess[zone_id] += max(
                0.0, concentration - run.crew_cabin_co2_concentration_ceiling
            )
            ticks_above[zone_id] += int(
                concentration > run.crew_cabin_co2_concentration_ceiling
            )
            maximum_co2[zone_id] = max(maximum_co2[zone_id], concentration)

    states = tuple(decision.state for decision in result.decisions)
    protect_events = [
        event for event in result.events if event.to_state is AuthorityState.PROTECT
    ]
    handback_recurrence_events = [
        event
        for event in result.events
        if event.from_state is AuthorityState.HANDBACK
        and event.to_state is AuthorityState.PROTECT
    ]
    protect_target_zone_ids = sorted(
        {
            decision.target_zone_id
            for decision in result.decisions
            if decision.state is AuthorityState.PROTECT
            and decision.target_zone_id is not None
        }
    )
    final_records = result.records[-5:]
    final_physical_zero = bool(final_records) and all(
        float(record.reserve["system"]["total_delivered_airflow"]) <= _EPSILON
        and all(
            float(actuator["actual_position"]) <= _EPSILON
            for actuator in record.reserve["actuators"].values()
        )
        for record in final_records
    )
    processing_id = config.processing_zone().id
    captured = (
        float(result.records[-1].plant.zones[processing_id]["captured_co2"])
        if result.records
        else 0.0
    )
    shortfall = primary_requested - primary_delivered
    return {
        "primary_requested_airflow_integral": primary_requested,
        "primary_delivered_airflow_integral": primary_delivered,
        "primary_shortfall_integral": shortfall,
        "reserve_requested_airflow_integral": reserve_requested,
        "reserve_delivered_airflow_integral": reserve_delivered,
        "total_delivered_airflow_integral": primary_delivered + reserve_delivered,
        "reserve_shortfall_coverage_fraction": _ratio(reserve_delivered, shortfall, cap=1.0),
        "steady_state_restoration_fraction": {
            "status": "not_applicable",
            "value": None,
        },
        "reserve_saturation_ticks": reserve_saturation_ticks,
        "integrated_physical_co2": integrated_co2,
        "integrated_physical_co2_excess": integrated_excess,
        "ticks_above_ceiling": ticks_above,
        "maximum_physical_co2": maximum_co2,
        "captured_co2_delta": captured,
        "states": {
            "first_degraded_tick": _first_state_tick(states, AuthorityState.DEGRADED),
            "first_protect_tick": _first_state_tick(states, AuthorityState.PROTECT),
            "first_handback_tick": _first_state_tick(states, AuthorityState.HANDBACK),
            "physical_zero_acknowledgement_tick": _physical_zero_ack_tick(result),
        },
        "lifecycle": {
            "protect_target_zone_ids": protect_target_zone_ids,
            "protect_entry_count": len(protect_events),
            "handback_recurrence_count": len(handback_recurrence_events),
            "handback_timeout_count": _decision_reason_episode_count(
                result, "handback_timeout"
            ),
            "reserve_failure_count": _decision_reason_episode_count(
                result, "reserve_delivery_failure"
            ),
            "final_physical_zero": final_physical_zero,
        },
        "invariant_violation_count": invariant_violations,
    }


def _decision_reason_episode_count(result: RecoveryRunResult, reason: str) -> int:
    episodes = 0
    active = False
    for decision in result.decisions:
        matches = decision.reason == reason
        if matches and not active:
            episodes += 1
        active = matches
    return episodes


def _first_state_tick(states: Sequence[AuthorityState], state: AuthorityState) -> int | None:
    for index, observed in enumerate(states, start=1):
        if observed is state:
            return index
    return None


def _physical_zero_ack_tick(result: RecoveryRunResult) -> int | None:
    for index in range(1, len(result.decisions)):
        if (
            result.decisions[index - 1].state is AuthorityState.HANDBACK
            and result.decisions[index].state is AuthorityState.NOMINAL
            and float(result.records[index].reserve["system"]["total_delivered_airflow"])
            <= _EPSILON
        ):
            return result.decisions[index].decision_tick
    return None


def _ratio(numerator: float, denominator: float, *, cap: float | None = None) -> dict[str, Any]:
    if denominator <= _EPSILON:
        return {"status": "undefined_zero_denominator", "value": None}
    value = numerator / denominator
    if cap is not None:
        value = min(cap, value)
    return {"status": "defined", "value": value}


def _source_provenance(*, require_clean_source: bool) -> dict[str, Any]:
    dirty = bool(_git_output("status", "--porcelain"))
    if require_clean_source and dirty:
        raise ValueError("canonical recovery evidence requires a clean source worktree")
    source_files = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256_file(path)
        for path in sorted((REPOSITORY_ROOT / "src" / "aeolus").rglob("*.py"))
    }
    lock_path = REPOSITORY_ROOT / "uv.lock"
    if not lock_path.is_file():
        raise OSError(f"canonical recovery evidence requires lock file: {lock_path}")
    return {
        "environment": {
            "source_commit": _git_output("rev-parse", "HEAD"),
            "source_worktree_dirty": dirty,
            "python_version": platform.python_version(),
            "python_implementation": sys.implementation.name,
            "uv_lock_sha256": _sha256_file(lock_path),
            "runtime_packages": {
                "aeolus": importlib.metadata.version("aeolus"),
                "numpy": importlib.metadata.version("numpy"),
            },
        },
        "source": {
            "files_sha256": source_files,
            "manifest_sha256": _canonical_sha256(source_files),
        },
    }


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_new(path: Path, document: object) -> None:
    if path.exists():
        raise FileExistsError(f"recovery evidence output already exists: {path}")
    path.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        receipt = run_recovery_evidence(argv[0], argv[1])
    except (FileExistsError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"cannot run recovery evidence: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))
