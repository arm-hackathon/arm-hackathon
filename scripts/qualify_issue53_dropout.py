#!/usr/bin/env python3
"""Fit, calibrate, and evaluate the Issue #53 dropout lane.

The runner intentionally keeps collection separate from qualification.  It
loads only content-addressed samples emitted by the collector, fits on TRAIN
``k=0`` samples, calibrates on VALIDATION, freezes the artifact, and evaluates
FINAL exactly once when ``--sealed-final`` is supplied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

import collect_issue53_dropout_dataset as collector
from aeolus.habitat_v2.forecast_issue52 import (
    CandidateCatalogue,
    CandidateSchedule,
    ForecastHistory,
    ForecastTrajectory,
    HORIZON_STEPS,
    ObservationRecord,
    TargetManifest,
    TrainingSample,
    extend_scenario_for_issue52,
    score_trajectory,
)
from aeolus.habitat_v2.forecast_issue52_rollout import (
    build_offline_checkpoint,
    rollout_catalogue,
)
from aeolus.habitat_v2.forecast_issue53_dropout import (
    DropoutAwareLinearForecaster,
    DropoutConfig,
    Issue53ForecastError,
    _masked_slope,
    impute_history_values,
)
from aeolus.habitat_v2.hmc_contract import canonical_json_bytes, load_hmc_contract
from aeolus.habitat_v2.physics import validate_external_command
from aeolus.habitat_v2.scenario import Scenario


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "issue53_dropout"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "issue53_dropout_model.json"
DEFAULT_REPORT = ROOT / "artifacts" / "issue53_dropout_qualification.json"
SCENARIO_PATH = ROOT / "scenarios" / "habitat_v2_actuator_feedback.json"
CONTRACT_PATH = ROOT / "contracts" / "habitat_v2_hmc_v1.json"
HORIZON_START = 8
REQUIRED_K = (0, 1, 3, 6)
BOOTSTRAP_SEED = 530053
BOOTSTRAP_REPETITIONS = 10000
TIMED_RUNS = 1000
ISSUE53_PREREGISTRATION_SHA256 = (
    "a96245f6e717bc83b44438f9d02dbaaa42fa5ded14d3a160fd47a0f4d393d76a"
)
ISSUE52_PREREGISTRATION_SHA256 = (
    "de4744e127d2946a43d623ec90d3289b0a3735c99e62c8ceccd87768e0702a3b"
)
EXPECTED_CANDIDATE_IDS = {
    "candidate_hold",
    "candidate_balanced",
    "candidate_ventilation",
    "candidate_scrubbing",
    "candidate_cooling",
    "candidate_dehumidifying",
    "candidate_oxygen",
    "candidate_resource_preserve",
    "candidate_laboratory",
    "candidate_crew",
    "candidate_low_intervention",
    "candidate_high_protection",
}


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _normalized_sha256(path: Path) -> str:
    """Hash JSON contract bytes using the preregistration's LF convention."""

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_commit(value: object, *, label: str = "source commit") -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} is not a full lowercase git commit")
    return value


def _array_from_payload(
    values_payload: object, mask_payload: object, *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(values_payload, list) or not isinstance(mask_payload, list):
        raise ValueError(f"{label} arrays are malformed")
    if len(values_payload) != len(mask_payload) or not values_payload:
        raise ValueError(f"{label} arrays have inconsistent lengths")
    values = np.empty(len(values_payload), dtype=np.float32)
    mask = np.empty(len(mask_payload), dtype=bool)
    for index, (raw_value, raw_available) in enumerate(
        zip(values_payload, mask_payload)
    ):
        if type(raw_available) is not bool:
            raise ValueError(f"{label} availability is not boolean")
        mask[index] = raw_available
        if raw_available:
            values[index] = np.float32(
                _finite_float(raw_value, label=f"{label}[{index}]")
            )
        elif raw_value is None:
            values[index] = np.nan
        else:
            raise ValueError(f"{label} unavailable value is not null")
    values.setflags(write=False)
    mask.setflags(write=False)
    return values, mask


def _record_from_payload(payload: Mapping[str, object]) -> ObservationRecord:
    values, mask = _array_from_payload(
        payload.get("target_values"), payload.get("available_mask"), label="history"
    )
    command = payload.get("command")
    if not isinstance(command, dict):
        raise ValueError("history command is malformed")
    return ObservationRecord(
        snapshot_sha256=str(payload["snapshot_sha256"]),
        verification_receipt_sha256=str(payload["verification_receipt_sha256"]),
        control_run_id=str(payload["control_run_id"]),
        authority_epoch=str(payload["authority_epoch"]),
        topology_sha256=str(payload["topology_sha256"]),
        hmc_contract_sha256=str(payload["hmc_contract_sha256"]),
        snapshot_schema_sha256=str(payload["snapshot_schema_sha256"]),
        scenario_sha256=str(payload["scenario_sha256"]),
        previous_verification_receipt_sha256=str(
            payload["previous_verification_receipt_sha256"]
        ),
        previous_control_chain_sha256=str(payload["previous_control_chain_sha256"]),
        control_chain_sha256=str(payload["control_chain_sha256"]),
        sequence=int(payload["sequence"]),
        completed_step=int(payload["completed_step"]),
        completed_time_s=_finite_float(
            payload["completed_time_s"], label="completed_time_s"
        ),
        mode=None if payload.get("mode") is None else str(payload["mode"]),
        command=command,
        command_sha256=str(payload["command_sha256"]),
        target_values=values,
        available_mask=mask,
    )


def _schedule_from_payload(
    payload: Mapping[str, object], scenario: Scenario
) -> CandidateSchedule:
    commands_payload = payload.get("commands")
    if not isinstance(commands_payload, list):
        raise ValueError("schedule commands are malformed")
    commands = tuple(
        validate_external_command(scenario, command)
        for command in commands_payload
        if isinstance(command, Mapping)
    )
    if len(commands) != len(commands_payload):
        raise ValueError("schedule contains a malformed command")
    command_digests = payload.get("command_sha256")
    if command_digests != [command.sha256 for command in commands]:
        raise ValueError("schedule command digests are inconsistent")
    applicable_modes = payload.get("applicable_modes")
    if not isinstance(applicable_modes, list):
        raise ValueError("schedule applicable modes are malformed")
    return CandidateSchedule(
        candidate_id=str(payload["candidate_id"]),
        purpose=str(payload["purpose"]),
        commands=commands,
        schedule_sha256=str(payload["schedule_sha256"]),
        applicable_modes=tuple(str(mode) for mode in applicable_modes),
    )


def _sample_from_payload(
    payload: Mapping[str, object], scenario: Scenario
) -> tuple[TrainingSample, np.ndarray]:
    family_id = str(payload["family_id"])
    family_scenario, family_metadata = collector._family_scenario(
        scenario, family_id, decision_step=15
    )
    history_payload = payload.get("history_records")
    if not isinstance(history_payload, list):
        raise ValueError("sample history is malformed")
    if len(history_payload) != 16 or any(
        not isinstance(record, Mapping) for record in history_payload
    ):
        raise ValueError("sample history must contain exactly 16 record objects")
    history = ForecastHistory.from_records(
        tuple(_record_from_payload(record) for record in history_payload)
    )
    if payload.get("scenario_sha256") != family_scenario.scenario_sha256:
        raise ValueError("sample scenario identity does not bind family replay")
    family_manifest = TargetManifest.from_scenario(family_scenario)
    if payload.get("manifest_sha256") != family_manifest.manifest_sha256:
        raise ValueError("sample manifest identity does not bind family replay")
    metadata = payload.get("family_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("sample family metadata is malformed")
    if metadata.get("fault_presence") != family_metadata["fault_presence"]:
        raise ValueError("sample fault metadata does not bind family replay")
    if metadata.get("family_index") != family_metadata["family_index"]:
        raise ValueError("sample family index does not bind family replay")
    if metadata.get("fault_profile_id") != family_metadata["fault_profile_id"]:
        raise ValueError("sample fault profile does not bind family replay")
    if metadata.get("operating_mode") != history.latest_record.mode:
        raise ValueError("sample operating mode does not bind history replay")
    schedule_payload = payload.get("schedule")
    if not isinstance(schedule_payload, Mapping):
        raise ValueError("sample schedule is malformed")
    schedule = _schedule_from_payload(schedule_payload, family_scenario)
    catalogue = CandidateCatalogue.from_scenario(
        family_scenario,
        base_command=history.latest_record.command,
    )
    expected_schedule = {
        candidate.candidate_id: candidate for candidate in catalogue.candidates
    }.get(schedule.candidate_id)
    if expected_schedule is None or expected_schedule.to_mapping() != schedule.to_mapping():
        raise ValueError("sample schedule does not bind the frozen candidate catalogue")
    targets_payload = payload.get("targets")
    if not isinstance(targets_payload, list):
        raise ValueError("sample targets are malformed")
    targets = np.asarray(targets_payload, dtype=np.float32)
    if targets.shape != (HORIZON_STEPS, history.target_values.shape[1]):
        raise ValueError("sample target shape is invalid")
    if not np.isfinite(targets).all():
        raise ValueError("sample targets are non-finite")
    truth_latest_payload = payload.get("truth_latest")
    if (
        not isinstance(truth_latest_payload, list)
        or len(truth_latest_payload) != family_manifest.width
    ):
        raise ValueError("sample latest truth is malformed")
    truth_latest = np.asarray(
        [
            _finite_float(value, label="truth_latest")
            for value in truth_latest_payload
        ],
        dtype=np.float32,
    )
    truth_latest.setflags(write=False)
    return TrainingSample(
        family_id=family_id,
        split=str(payload["split"]),
        scenario_sha256=str(payload["scenario_sha256"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        schedule_sha256=str(payload["schedule_sha256"]),
        history=history,
        schedule=schedule,
        targets=targets,
    ), truth_latest


def _load_samples(
    dataset: Path, scenario: Scenario, *, require_full: bool
) -> tuple[
    DropoutConfig,
    collector.DropoutDatasetManifest,
    tuple[TrainingSample, ...],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], str],
]:
    if _normalized_sha256(CONTRACT_PATH.with_name("habitat_v2_forecast_issue_53_preregistration_v1.json")) != ISSUE53_PREREGISTRATION_SHA256:
        raise ValueError("Issue #53 preregistration digest is not the frozen digest")
    if _normalized_sha256(CONTRACT_PATH.with_name("habitat_v2_forecast_issue_52_preregistration_v1.json")) != ISSUE52_PREREGISTRATION_SHA256:
        raise ValueError("Issue #52 preregistration digest is not the frozen digest")
    collector.validate_dataset(dataset, require_coverage=require_full)
    config = DropoutConfig.from_mapping(
        json.loads((dataset / "dropout_config.json").read_text(encoding="utf-8"))
    )
    manifest = collector.DropoutDatasetManifest.from_mapping(
        json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    )
    expected_families = tuple(
        collector._deterministic_family_ids(
            collector.DEFAULT_FAMILIES_FULL if require_full else len(manifest.family_ids)
        )
    )
    if manifest.family_ids != expected_families:
        raise ValueError("dataset requires the frozen deterministic family roster")
    if manifest.family_split != collector._family_split(list(manifest.family_ids)):
        raise ValueError("dataset family split does not match the preregistered hash split")
    collector_base_scenario = Scenario.from_mapping(
        json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    )
    samples: list[TrainingSample] = []
    truth_latest_by_key: dict[tuple[str, str], np.ndarray] = {}
    rollout_sha_by_key: dict[tuple[str, str], str] = {}
    with (dataset / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("sample_sha256") != collector._sample_digest(payload):
                raise ValueError("sample digest is inconsistent")
            _require_sha256(payload.get("rollout_sha256"), label="sample rollout")
            if payload.get("dropout_config_sha256") != config.config_sha256:
                raise ValueError("sample dropout config does not bind loaded config")
            sample, truth_latest = _sample_from_payload(payload, collector_base_scenario)
            key = (sample.family_id, sample.schedule.candidate_id)
            previous_truth = truth_latest_by_key.setdefault(key, truth_latest)
            if not np.array_equal(previous_truth, truth_latest):
                raise ValueError("sample latest truth differs within a family candidate")
            previous_rollout = rollout_sha_by_key.setdefault(
                key, str(payload["rollout_sha256"])
            )
            if previous_rollout != payload["rollout_sha256"]:
                raise ValueError("sample rollout identity differs within a family candidate")
            samples.append(sample)
    if not samples:
        raise ValueError("qualification dataset contains no samples")
    if {sample.family_id for sample in samples} != set(manifest.family_ids):
        raise ValueError("qualification samples do not cover the manifest families")
    grouped = _group_samples(samples)
    for split, by_k in grouped.items():
        for k, items in by_k.items():
            families: dict[str, set[str]] = defaultdict(set)
            for item in items:
                families[item.family_id].add(item.schedule.candidate_id)
            if any(
                len([item for item in items if item.family_id == family_id]) != 12
                or candidate_ids != EXPECTED_CANDIDATE_IDS
                for family_id, candidate_ids in families.items()
            ):
                raise ValueError("dataset decision group does not contain the exact candidate catalogue")
    return config, manifest, tuple(samples), truth_latest_by_key, rollout_sha_by_key


def _verify_replayed_samples(
    base_scenario: Scenario,
    contract: Any,
    config: DropoutConfig,
    samples: Sequence[TrainingSample],
    truth_latest_by_key: Mapping[tuple[str, str], np.ndarray],
    rollout_sha_by_key: Mapping[tuple[str, str], str],
) -> dict[str, int]:
    """Reconstruct each family once and measure lineage, replay, and authority faults."""

    counts = {
        "replay_failure_count": 0,
        "provenance_or_split_violation_count": 0,
        "authority_violation_count": 0,
        "non_finite_committed_state_count": 0,
    }
    by_family: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        by_family[sample.family_id].append(sample)
    for family_id, family_samples in sorted(by_family.items()):
        try:
            family_scenario, _metadata = collector._family_scenario(
                base_scenario, family_id, decision_step=15
            )
            checkpoint = build_offline_checkpoint(
                family_scenario,
                contract,
                decision_step=15,
                family_id=family_id,
            )
            family_manifest = TargetManifest.from_scenario(family_scenario)
            catalogue = CandidateCatalogue.from_scenario(
                family_scenario,
                base_command=checkpoint.last_final_command,
            )
            rollouts = rollout_catalogue(
                checkpoint, catalogue, manifest=family_manifest
            )
            rollouts_by_id = {rollout.candidate_id: rollout for rollout in rollouts}
            base_history = ForecastHistory.from_records(checkpoint.history_records)
            state_values = [
                checkpoint.state.step,
                *(
                    value
                    for zone in checkpoint.state.zones.values()
                    for value in (
                        zone.co2_mol,
                        zone.o2_mol,
                        zone.water_vapor_mol,
                        zone.inert_mol,
                        zone.temperature_k,
                    )
                ),
                checkpoint.state.utility.co2_sorbent_remaining_mol,
                checkpoint.state.utility.captured_co2_mol,
                checkpoint.state.utility.condensed_water_mol,
                checkpoint.state.utility.oxygen_store_mol,
                checkpoint.state.utility.battery_energy_wh,
            ]
            if not all(math.isfinite(float(value)) for value in state_values):
                counts["non_finite_committed_state_count"] += 1
                continue
        except Exception:
            counts["replay_failure_count"] += 1
            continue

        def record_identity(record: ObservationRecord) -> tuple[object, ...]:
            return (
                record.snapshot_sha256,
                record.verification_receipt_sha256,
                record.control_run_id,
                record.authority_epoch,
                record.topology_sha256,
                record.hmc_contract_sha256,
                record.snapshot_schema_sha256,
                record.scenario_sha256,
                record.previous_verification_receipt_sha256,
                record.previous_control_chain_sha256,
                record.control_chain_sha256,
                record.sequence,
                record.completed_step,
                record.completed_time_s,
                record.mode,
                record.command_sha256,
            )

        for sample in family_samples:
            try:
                candidate = next(
                    item
                    for item in catalogue.candidates
                    if item.candidate_id == sample.schedule.candidate_id
                )
                rollout = rollouts_by_id[candidate.candidate_id]
                k = int(np.sum(~sample.history.available_mask[-1]))
                expected_history = collector.apply_dropout_to_history(
                    base_history,
                    family_manifest,
                    config,
                    family_id=family_id,
                    decision_step=15,
                    latest_missing_count=k,
                )
                if (
                    sample.checkpoint_sha256 != checkpoint.checkpoint_sha256
                    or sample.scenario_sha256 != family_scenario.scenario_sha256
                    or sample.manifest_sha256 != family_manifest.manifest_sha256
                    or sample.schedule.to_mapping() != candidate.to_mapping()
                    or rollout_sha_by_key[(family_id, candidate.candidate_id)]
                    != rollout.rollout_sha256
                    or not np.array_equal(sample.targets, rollout.targets)
                    or not np.array_equal(sample.history.target_values, expected_history.target_values, equal_nan=True)
                    or not np.array_equal(sample.history.available_mask, expected_history.available_mask)
                    or not np.array_equal(
                        truth_latest_by_key[(family_id, candidate.candidate_id)],
                        base_history.latest,
                    )
                ):
                    counts["replay_failure_count"] += 1
                    continue
                if any(
                    record.hmc_contract_sha256 != contract.hmc_contract_sha256
                    or record.snapshot_schema_sha256 != contract.snapshot_schema_sha256
                    or record.topology_sha256 != family_manifest.topology_sha256
                    or record.scenario_sha256 != family_scenario.scenario_sha256
                    for record in sample.history.records
                ):
                    counts["authority_violation_count"] += 1
                    continue
                if any(
                    record_identity(left) != record_identity(right)
                    for left, right in zip(sample.history.records, base_history.records)
                ):
                    counts["provenance_or_split_violation_count"] += 1
            except Exception:
                counts["replay_failure_count"] += 1
    return counts


def _group_samples(
    samples: Sequence[TrainingSample],
) -> dict[str, dict[int, tuple[TrainingSample, ...]]]:
    grouped: dict[str, dict[int, list[TrainingSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        k = int(np.sum(~sample.history.available_mask[-1]))
        grouped[sample.split][k].append(sample)
    return {
        split: {k: tuple(items) for k, items in sorted(by_k.items())}
        for split, by_k in grouped.items()
    }


def _persistence_nmae(
    sample: TrainingSample, manifest: TargetManifest, start: int = HORIZON_START
) -> float:
    latest = impute_history_values(
        sample.history.target_values, sample.history.available_mask, manifest
    )[-1].astype(np.float64)
    truth = sample.targets.astype(np.float64)[start:]
    scales = np.asarray([descriptor.scale for descriptor in manifest.descriptors])
    return float(np.mean(np.abs(truth - latest[None, :]) / scales[None, :]))


def _model_nmae(
    trajectory: Any,
    sample: TrainingSample,
    manifest: TargetManifest,
    start: int = HORIZON_START,
    end: int | None = None,
) -> float | None:
    if trajectory.status != "PREDICTION" or trajectory.mean is None:
        return None
    scales = np.asarray([descriptor.scale for descriptor in manifest.descriptors])
    mean = np.asarray(trajectory.mean, dtype=np.float64)
    targets = sample.targets.astype(np.float64)
    if mean.shape != targets.shape or mean.shape != (HORIZON_STEPS, manifest.width):
        return None
    if not np.isfinite(mean).all() or not np.isfinite(targets).all():
        return None
    return float(np.mean(np.abs(mean[start:end] - targets[start:end]) / scales[None, :]))


def _metric_forecast(model: Any, sample: TrainingSample) -> ForecastTrajectory:
    """Use finite model output for metrics; selective status is accounted separately."""

    if isinstance(model, DropoutAwareLinearForecaster):
        return model.forecast(
            sample.history,
            sample.schedule,
            apply_abstention=False,
        )
    return model.forecast(sample.history, sample.schedule)


def _family_nmae(
    model: Any,
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
) -> dict[str, float | None]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        trajectory = _metric_forecast(model, sample)
        value = _model_nmae(trajectory, sample, manifest)
        if value is not None:
            grouped[sample.family_id].append(value)
        else:
            grouped.setdefault(sample.family_id, [])
    return {
        family_id: (float(np.mean(values)) if values else None)
        for family_id, values in grouped.items()
    }


def _paired_family_nmae(
    primary: Any,
    comparator: Any,
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
) -> tuple[dict[str, float | None], dict[str, float | None], int]:
    """Aggregate both methods over one candidate eligibility mask."""

    grouped: dict[str, dict[str, TrainingSample]] = defaultdict(dict)
    for sample in samples:
        grouped[sample.family_id][sample.schedule.candidate_id] = sample
    primary_values: dict[str, float | None] = {}
    comparator_values: dict[str, float | None] = {}
    ineligible_decisions = 0
    for family_id, candidates in grouped.items():
        paired: list[tuple[float, float]] = []
        for sample in candidates.values():
            primary_value = _model_nmae(_metric_forecast(primary, sample), sample, manifest)
            comparator_value = _model_nmae(
                _metric_forecast(comparator, sample), sample, manifest
            )
            if primary_value is not None and comparator_value is not None:
                paired.append((primary_value, comparator_value))
        if not paired:
            ineligible_decisions += 1
            primary_values[family_id] = None
            comparator_values[family_id] = None
            continue
        primary_values[family_id] = float(np.mean([item[0] for item in paired]))
        comparator_values[family_id] = float(np.mean([item[1] for item in paired]))
    return primary_values, comparator_values, ineligible_decisions


def _decision_statuses(
    model: Any, sample_group: Sequence[TrainingSample]
) -> tuple[str, tuple[ForecastTrajectory, ...]]:
    trajectories = tuple(
        model.forecast(sample.history, sample.schedule) for sample in sample_group
    )
    statuses = {trajectory.status for trajectory in trajectories}
    if "INVALID_OUTPUT" in statuses:
        return "INVALID_OUTPUT", trajectories
    if "ABSTAIN" in statuses:
        return "ABSTAINED", trajectories
    if statuses == {"PREDICTION"} and len(trajectories) == 12:
        return "PREDICTION", trajectories
    return "INVALID_OUTPUT", trajectories


def _status_counts(
    model: Any, samples: Sequence[TrainingSample]
) -> dict[str, int]:
    counts = {"PREDICTION": 0, "ABSTAINED": 0, "INVALID_OUTPUT": 0}
    for families in _decision_groups(samples).values():
        for items_by_family in families.values():
            for items in items_by_family.values():
                status, _ = _decision_statuses(model, items)
                counts[status] += 1
    return counts


def _decision_groups(
    samples: Sequence[TrainingSample],
) -> dict[str, dict[int, dict[str, tuple[TrainingSample, ...]]]]:
    grouped: dict[str, dict[int, dict[str, list[TrainingSample]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for sample in samples:
        k = int(np.sum(~sample.history.available_mask[-1]))
        grouped[sample.split][k][sample.family_id].append(sample)
    return {
        split: {
            k: {family: tuple(items) for family, items in sorted(families.items())}
            for k, families in sorted(by_k.items())
        }
        for split, by_k in grouped.items()
    }


def _complete_candidate_values(
    model: Any,
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
) -> tuple[dict[str, float], int]:
    grouped = _decision_groups(samples)
    values: dict[str, float] = {}
    invalid = 0
    for by_k in grouped.values():
        for families in by_k.values():
            for family_id, items in families.items():
                status, trajectories = _decision_statuses(model, items)
                if status != "PREDICTION":
                    invalid += 1
                    continue
                family_values = [
                    _model_nmae(trajectory, sample, manifest)
                    for trajectory, sample in zip(trajectories, items)
                ]
                if any(value is None for value in family_values):
                    invalid += 1
                    continue
                values[family_id] = float(np.mean([float(value) for value in family_values]))
    return values, invalid


def _paired_decision_values(
    primary: Any,
    comparator: Any,
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
    *,
    start: int = HORIZON_START,
    end: int | None = None,
) -> tuple[dict[str, float | None], dict[str, float | None], dict[str, int]]:
    """Aggregate paired candidate rows after applying one shared eligibility mask."""

    grouped: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.family_id].append(sample)
    primary_values: dict[str, float | None] = {}
    comparator_values: dict[str, float | None] = {}
    counts = {
        "eligible_candidates": 0,
        "abstained_decisions": 0,
        "invalid_decisions": 0,
        "incomplete_decisions": 0,
    }
    for family_id, family_samples in sorted(grouped.items()):
        by_candidate = {sample.schedule.candidate_id: sample for sample in family_samples}
        if len(by_candidate) != 12:
            counts["incomplete_decisions"] += 1
            continue
        paired: list[tuple[float, float]] = []
        decision_abstained = False
        decision_invalid = False
        for candidate_id, sample in sorted(by_candidate.items()):
            selective = primary.forecast(sample.history, sample.schedule)
            if selective.status == "INVALID_OUTPUT":
                decision_invalid = True
            elif selective.status == "ABSTAIN":
                decision_abstained = True
            learned = _metric_forecast(primary, sample)
            control = _metric_forecast(comparator, sample)
            if learned.status == "ABSTAIN" or control.status == "ABSTAIN":
                decision_abstained = True
                continue
            if learned.status == "INVALID_OUTPUT" or control.status == "INVALID_OUTPUT":
                decision_invalid = True
                continue
            learned_value = _model_nmae(
                learned, sample, manifest, start=start, end=end
            )
            control_value = _model_nmae(
                control, sample, manifest, start=start, end=end
            )
            if learned_value is None or control_value is None:
                decision_invalid = True
                continue
            paired.append((learned_value, control_value))
        if decision_invalid:
            counts["invalid_decisions"] += 1
        elif decision_abstained:
            counts["abstained_decisions"] += 1
        if len(paired) != 12:
            decision_invalid = True
        if paired and not decision_invalid:
            counts["eligible_candidates"] += len(paired)
            primary_values[family_id] = float(np.mean([value[0] for value in paired]))
            comparator_values[family_id] = float(np.mean([value[1] for value in paired]))
        else:
            primary_values.pop(family_id, None)
            comparator_values.pop(family_id, None)
    return primary_values, comparator_values, counts


class MaskAwarePersistenceForecaster:
    """Frozen non-neural comparator that imputes observations then persists them."""

    def __init__(self, manifest: TargetManifest) -> None:
        self.manifest = manifest
        self.model_id = "issue53-mask-aware-persistence-v1"

    def forecast(
        self, history: ForecastHistory, schedule: CandidateSchedule
    ) -> ForecastTrajectory:
        if history.target_values.shape[1] != self.manifest.width:
            return ForecastTrajectory(
                "INVALID_OUTPUT", None, None, None, self.model_id, "manifest_width_mismatch"
            )
        latest = impute_history_values(
            history.target_values, history.available_mask, self.manifest
        )[-1].astype(np.float64)
        mean = np.repeat(latest[None, :], HORIZON_STEPS, axis=0)
        scales = np.asarray([descriptor.scale for descriptor in self.manifest.descriptors])
        spread = scales[None, :] * 0.02 * np.sqrt(
            np.arange(1, HORIZON_STEPS + 1, dtype=np.float64)
        )[:, None]
        return ForecastTrajectory(
            "PREDICTION",
            mean.astype(np.float32),
            (mean - spread).astype(np.float32),
            (mean + spread).astype(np.float32),
            self.model_id,
        )


class MaskAwareLinearContinuationForecaster:
    """Frozen non-neural baseline using only masked history continuation."""

    def __init__(self, manifest: TargetManifest) -> None:
        self.manifest = manifest
        self.model_id = "issue53-mask-aware-linear-continuation-v1"

    def forecast(
        self, history: ForecastHistory, schedule: CandidateSchedule
    ) -> ForecastTrajectory:
        if history.target_values.shape[1] != self.manifest.width:
            return ForecastTrajectory(
                "INVALID_OUTPUT", None, None, None, self.model_id, "manifest_width_mismatch"
            )
        try:
            values = impute_history_values(
                history.target_values, history.available_mask, self.manifest
            )
            slope = _masked_slope(
                values, history.available_mask, history.completed_times_s
            ).astype(np.float64)
        except (Issue53ForecastError, ValueError):
            return ForecastTrajectory(
                "INVALID_OUTPUT", None, None, None, self.model_id, "history_features_invalid"
            )
        latest = values[-1].astype(np.float64)
        mean = latest[None, :] + np.arange(
            1, HORIZON_STEPS + 1, dtype=np.float64
        )[:, None] * slope[None, :]
        scales = np.asarray([descriptor.scale for descriptor in self.manifest.descriptors])
        spread = scales[None, :] * 0.02 * np.sqrt(
            np.arange(1, HORIZON_STEPS + 1, dtype=np.float64)
        )[:, None]
        return ForecastTrajectory(
            "PREDICTION",
            mean.astype(np.float32),
            (mean - spread).astype(np.float32),
            (mean + spread).astype(np.float32),
            self.model_id,
        )


def _baseline_nmae(
    baseline: Any, sample: TrainingSample, manifest: TargetManifest
) -> float | None:
    return _model_nmae(_metric_forecast(baseline, sample), sample, manifest)


def _select_baseline(
    candidates: Sequence[Any],
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
) -> tuple[Any, dict[str, float]]:
    if not samples:
        raise ValueError("baseline selection requires VALIDATION k=0 samples")
    scores: dict[str, float] = {}
    for candidate in candidates:
        family_values: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            value = _baseline_nmae(candidate, sample, manifest)
            if value is not None:
                family_values[sample.family_id].append(value)
        if not family_values or any(
            len(values) != 12 for values in family_values.values()
        ):
            raise ValueError(f"baseline {candidate.model_id} produced an invalid k=0 output")
        scores[candidate.model_id] = float(
            np.mean([np.mean(values) for values in family_values.values()])
        )
    selected = min(candidates, key=lambda candidate: (scores[candidate.model_id], candidate.model_id))
    return selected, scores


def _family_means(
    values: Mapping[tuple[str, str], float | None],
) -> dict[str, float | None]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for (family_id, _candidate_id), value in values.items():
        if value is not None and math.isfinite(value):
            grouped[family_id].append(value)
        else:
            grouped.setdefault(family_id, [])
    return {
        family_id: (float(np.mean(items)) if items else None)
        for family_id, items in grouped.items()
    }


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if len(finite) == len(values) and finite else None


def _ratio(
    candidate: Mapping[str, float | None], comparator: Mapping[str, float | None]
) -> tuple[float | None, list[str]]:
    families = sorted(
        family_id
        for family_id in set(candidate) & set(comparator)
        if candidate[family_id] is not None and comparator[family_id] is not None
    )
    left = [candidate[family] for family in families]
    right = [comparator[family] for family in families]
    if not families:
        return None, families
    left_mean = float(np.mean([float(value) for value in left]))
    right_mean = float(np.mean([float(value) for value in right]))
    if right_mean == 0.0:
        return (1.0 if left_mean == 0.0 else None), families
    return left_mean / right_mean, families


def _strict_ratio(
    candidate: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
) -> tuple[float | None, list[str], bool]:
    """Return a ratio and reject non-identical or non-finite family masks."""

    if set(candidate) != set(comparator) or not candidate:
        return None, [], False
    values = [candidate[family_id] for family_id in sorted(candidate)]
    controls = [comparator[family_id] for family_id in sorted(comparator)]
    if any(
        value is None or control is None
        or not math.isfinite(float(value))
        or not math.isfinite(float(control))
        for value, control in zip(values, controls)
    ):
        return None, sorted(candidate), False
    numerator = float(np.mean([float(value) for value in values]))
    denominator = float(np.mean([float(value) for value in controls]))
    if denominator == 0.0:
        return (1.0 if numerator == 0.0 else None), sorted(candidate), numerator == 0.0
    return numerator / denominator, sorted(candidate), True


@lru_cache(maxsize=16)
def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    """Return the registered SHA-256 family bootstrap draws, cached per shape."""

    if repetitions < 1 or count < 1:
        raise ValueError("bootstrap dimensions must be positive")
    indices = np.empty((repetitions, count), dtype=np.int64)
    for replicate in range(repetitions):
        for draw in range(count):
            indices[replicate, draw] = (
                int.from_bytes(
                    hashlib.sha256(
                        f"issue53-bootstrap-v1|{seed}|{replicate}|{draw}".encode(
                            "utf-8"
                        )
                    ).digest()[:8],
                    "big",
                )
                % count
            )
    indices.setflags(write=False)
    return indices


def _bootstrap_ratio(
    candidate: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[float | None, float | None]:
    family_ids = sorted(set(candidate) & set(comparator))
    if set(candidate) != set(comparator) or not family_ids:
        return None, None
    if any(
        candidate[family_id] is None
        or comparator[family_id] is None
        or not math.isfinite(float(candidate[family_id]))
        or not math.isfinite(float(comparator[family_id]))
        for family_id in family_ids
    ):
        return None, None
    left = np.asarray([float(candidate[family_id]) for family_id in family_ids])
    right = np.asarray([float(comparator[family_id]) for family_id in family_ids])
    count = len(family_ids)
    indices = _bootstrap_indices(seed, repetitions, count)
    numerators = np.mean(left[indices], axis=1)
    denominators = np.mean(right[indices], axis=1)
    ratios = np.divide(
        numerators,
        denominators,
        out=np.ones(repetitions, dtype=np.float64),
        where=denominators != 0.0,
    )
    invalid_zero = (denominators == 0.0) & (numerators != 0.0)
    ratios[invalid_zero] = np.inf
    if not np.isfinite(ratios).all():
        return None, None
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _coverage_and_width(
    model: DropoutAwareLinearForecaster,
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
) -> tuple[float | None, float | None, float]:
    family_coverages: dict[str, list[float]] = defaultdict(list)
    family_widths: dict[str, list[float]] = defaultdict(list)
    scales = np.asarray([descriptor.scale for descriptor in manifest.descriptors])
    for sample in samples:
        trajectory = _metric_forecast(model, sample)
        if (
            trajectory.status != "PREDICTION"
            or trajectory.lower is None
            or trajectory.upper is None
        ):
            # An abstention is not a free coverage pass. Keep it in the
            # family denominator as an uncovered decision.
            family_coverages.setdefault(sample.family_id, []).append(0.0)
            continue
        lower = np.asarray(trajectory.lower, dtype=np.float64)
        upper = np.asarray(trajectory.upper, dtype=np.float64)
        truth = sample.targets.astype(np.float64)
        family_coverages[sample.family_id].append(
            float(np.mean((truth >= lower) & (truth <= upper)))
        )
        family_widths[sample.family_id].append(
            float(np.mean((upper - lower) / scales[None, :]))
        )
    coverages = [np.mean(values) for values in family_coverages.values() if values]
    widths = [np.mean(values) for values in family_widths.values() if values]
    return (
        _mean_or_none([float(value) for value in coverages]),
        _mean_or_none([float(value) for value in widths]),
        float(len(samples) * HORIZON_STEPS * manifest.width),
    )


def _abstention_metrics(
    model: DropoutAwareLinearForecaster,
    samples: Sequence[TrainingSample],
    oracle_errors: Sequence[float],
    threshold: float,
) -> dict[str, float | None]:
    if len(samples) != len(oracle_errors) or not samples:
        return {"rate": None, "precision": None, "recall": None, "threshold": threshold}
    family_errors: dict[str, list[float]] = defaultdict(list)
    family_samples: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample, error in zip(samples, oracle_errors):
        family_errors[sample.family_id].append(float(error))
        family_samples[sample.family_id].append(sample)
    family_ids = sorted(family_errors)
    eligible: list[tuple[bool, bool]] = []
    invalid = 0
    for family_id in family_ids:
        status, _ = _decision_statuses(model, family_samples[family_id])
        if status == "INVALID_OUTPUT":
            invalid += 1
            continue
        eligible.append(
            (
                float(np.mean(family_errors[family_id])) > threshold,
                status == "ABSTAINED",
            )
        )
    if not eligible:
        return {
            "rate": None,
            "precision": None,
            "recall": None,
            "threshold": float(threshold),
            "eligible_decisions": 0.0,
            "invalid_decisions": float(invalid),
        }
    high = [item[0] for item in eligible]
    abstained = [item[1] for item in eligible]
    tp = sum(a and h for a, h in zip(abstained, high))
    fp = sum(a and not h for a, h in zip(abstained, high))
    fn = sum(not a and h for a, h in zip(abstained, high))
    return {
        "rate": float(sum(abstained) / len(abstained)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "threshold": float(threshold),
        "eligible_decisions": float(len(eligible)),
        "invalid_decisions": float(invalid),
    }


def _abstention_pr_curve(
    model: DropoutAwareLinearForecaster,
    samples: Sequence[TrainingSample],
    oracle_errors: Sequence[float],
    threshold: float,
) -> list[dict[str, float]]:
    if len(samples) != len(oracle_errors) or not samples:
        return []
    grouped: dict[str, list[tuple[TrainingSample, float]]] = defaultdict(list)
    for sample, error in zip(samples, oracle_errors):
        grouped[sample.family_id].append((sample, float(error)))
    records: list[tuple[float, bool]] = []
    for family_id, items in sorted(grouped.items()):
        status, _ = _decision_statuses(model, [item[0] for item in items])
        if status == "INVALID_OUTPUT":
            continue
        risk = float(
            np.max(
                [model.risk_score(item.history, item.schedule) for item, _ in items]
            )
        )
        records.append(
            (risk, float(np.mean([error for _, error in items])) > threshold)
        )
    if not records:
        return []
    candidates = sorted({0.0, *[risk for risk, _ in records]})
    curve: list[dict[str, float]] = []
    for risk_limit in candidates:
        predicted = [risk > risk_limit for risk, _ in records]
        high = [is_high for _, is_high in records]
        tp = sum(a and h for a, h in zip(predicted, high))
        fp = sum(a and not h for a, h in zip(predicted, high))
        fn = sum(not a and h for a, h in zip(predicted, high))
        curve.append(
            {
                "risk_limit": float(risk_limit),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "rate": float(sum(predicted) / len(predicted)),
            }
        )
    return curve


def _abstention_rate(model: Any, samples: Sequence[TrainingSample]) -> float:
    groups = _decision_groups(samples)
    decisions = [
        items
        for by_k in groups.values()
        for families in by_k.values()
        for items in families.values()
    ]
    if not decisions:
        return math.nan
    return float(
        sum(_decision_statuses(model, items)[0] == "ABSTAINED" for items in decisions)
        / len(decisions)
    )


def _status_counts_by_k(
    model: Any, samples: Sequence[TrainingSample]
) -> dict[str, dict[str, int]]:
    groups = _group_samples(samples)
    result: dict[str, dict[str, int]] = {}
    for by_k in groups.values():
        for k, items in by_k.items():
            counts = _status_counts(model, items)
            existing = result.setdefault(
                str(k), {"PREDICTION": 0, "ABSTAINED": 0, "INVALID_OUTPUT": 0}
            )
            for status, count in counts.items():
                existing[status] += count
    return result


def _abstention_rate_at_k(model: Any, samples: Sequence[TrainingSample], k: int) -> float:
    items = tuple(
        sample for sample in samples
        if int(np.sum(~sample.history.available_mask[-1])) == k
    )
    return _abstention_rate(model, items)


def _crossing_metrics(
    model: DropoutAwareLinearForecaster,
    baseline: Any,
    samples: Sequence[TrainingSample],
    manifest: TargetManifest,
    scenario: Scenario,
    truth_latest_by_key: Mapping[tuple[str, str], np.ndarray],
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, float | None]:
    def events(values: np.ndarray, bound: float, direction: str) -> list[int]:
        result: list[int] = []
        for index in range(1, len(values)):
            source = float(values[index - 1])
            destination = float(values[index])
            if (direction == "LOW" and source > bound and destination <= bound) or (
                direction == "HIGH" and source < bound and destination >= bound
            ):
                result.append(index)
        return result

    def matched(
        truth_events: list[tuple[int, str]],
        predicted_events: list[tuple[int, str]],
    ) -> int:
        remaining = sorted(predicted_events, key=lambda event: (event[0], event[1]))
        result = 0
        for truth_step, truth_direction in sorted(
            truth_events, key=lambda event: (event[0], event[1])
        ):
            candidates = [
                (index, event)
                for index, event in enumerate(remaining)
                if event[1] == truth_direction and abs(event[0] - truth_step) <= 1
            ]
            if not candidates:
                continue
            selected_index, _selected = min(
                candidates,
                key=lambda item: (
                    abs(item[1][0] - truth_step),
                    item[1][0],
                    item[1][1],
                ),
            )
            remaining.pop(selected_index)
            result += 1
        return result

    grouped: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.family_id].append(sample)
    learned_exposure: list[float] = []
    baseline_exposure: list[float] = []
    learned_recall_hits = learned_truth_events = 0
    baseline_recall_hits = baseline_truth_events = 0
    learned_false_opportunities = learned_non_crossing_opportunities = 0
    baseline_false_opportunities = baseline_non_crossing_opportunities = 0
    family_differences: dict[str, list[float]] = defaultdict(list)
    family_learned_exposure: dict[str, list[float]] = defaultdict(list)
    family_baseline_exposure: dict[str, list[float]] = defaultdict(list)
    excluded_abstentions = 0
    excluded_invalid = 0
    incomplete_decisions = 0
    for family_id, family_samples in sorted(grouped.items()):
        by_candidate = {sample.schedule.candidate_id: sample for sample in family_samples}
        if len(by_candidate) != 12 or set(by_candidate) != EXPECTED_CANDIDATE_IDS:
            incomplete_decisions += 1
            continue
        counter_start = (
            learned_recall_hits,
            baseline_recall_hits,
            learned_truth_events,
            baseline_truth_events,
            learned_false_opportunities,
            learned_non_crossing_opportunities,
            baseline_false_opportunities,
            baseline_non_crossing_opportunities,
        )
        family_start = [
            (mapping, len(mapping.get(family_id, ())))
            for mapping in (
                family_learned_exposure,
                family_baseline_exposure,
                family_differences,
            )
        ]
        raw_pairs: list[tuple[TrainingSample, ForecastTrajectory, ForecastTrajectory]] = []
        decision_invalid = False
        for candidate_id in sorted(by_candidate):
            sample = by_candidate[candidate_id]
            learned = _metric_forecast(model, sample)
            control = _metric_forecast(baseline, sample)
            if learned.status != "PREDICTION" or control.status != "PREDICTION":
                decision_invalid = True
                break
            raw_pairs.append((sample, learned, control))
        if decision_invalid:
            excluded_invalid += 1
            continue
        truth_latest = truth_latest_by_key.get((family_id, sorted(by_candidate)[0]))
        if truth_latest is None or truth_latest.shape != (manifest.width,):
            excluded_invalid += 1
            continue
        for sample, learned, control in raw_pairs:
            candidate_truth_latest = truth_latest_by_key.get(
                (family_id, sample.schedule.candidate_id)
            )
            if candidate_truth_latest is None or not np.array_equal(
                candidate_truth_latest, truth_latest
            ):
                decision_invalid = True
                break
            learned_score = score_trajectory(
                manifest, sample.history, sample.schedule, learned, scenario
            )
            baseline_score = score_trajectory(
                manifest, sample.history, sample.schedule, control, scenario
            )
            learned_value = float(learned_score.safety_exposure)
            baseline_value = float(baseline_score.safety_exposure)
            family_learned_exposure[family_id].append(learned_value)
            family_baseline_exposure[family_id].append(baseline_value)
            family_differences[family_id].append(
                float(learned_score.safety_exposure - baseline_score.safety_exposure)
            )
            truth = np.concatenate(
                (truth_latest[None, :], np.asarray(sample.targets, dtype=np.float64)),
                axis=0,
            )
            learned_mean = np.concatenate(
                (truth_latest[None, :], np.asarray(learned.mean, dtype=np.float64)),
                axis=0,
            )
            baseline_mean = np.concatenate(
                (truth_latest[None, :], np.asarray(control.mean, dtype=np.float64)),
                axis=0,
            )
            for descriptor_index, descriptor in enumerate(manifest.descriptors):
                lower = float(
                    descriptor.lower
                    if descriptor.crossing_lower is None
                    else descriptor.crossing_lower
                )
                upper = float(
                    descriptor.upper
                    if descriptor.crossing_upper is None
                    else descriptor.crossing_upper
                )
                truth_events = sorted(
                    [
                        (step, "LOW")
                        for step in events(truth[:, descriptor_index], lower, "LOW")
                    ]
                    + [
                        (step, "HIGH")
                        for step in events(truth[:, descriptor_index], upper, "HIGH")
                    ],
                    key=lambda event: (event[0], event[1]),
                )
                learned_events = sorted(
                    [
                        (step, "LOW")
                        for step in events(learned_mean[:, descriptor_index], lower, "LOW")
                    ]
                    + [
                        (step, "HIGH")
                        for step in events(learned_mean[:, descriptor_index], upper, "HIGH")
                    ],
                    key=lambda event: (event[0], event[1]),
                )
                baseline_events = sorted(
                    [
                        (step, "LOW")
                        for step in events(baseline_mean[:, descriptor_index], lower, "LOW")
                    ]
                    + [
                        (step, "HIGH")
                        for step in events(baseline_mean[:, descriptor_index], upper, "HIGH")
                    ],
                    key=lambda event: (event[0], event[1]),
                )
                learned_recall_hits += matched(truth_events, learned_events)
                baseline_recall_hits += matched(truth_events, baseline_events)
                learned_truth_events += len(truth_events)
                baseline_truth_events += len(truth_events)
                for direction in ("LOW", "HIGH"):
                    truth_direction_events = [
                        event for event in truth_events if event[1] == direction
                    ]
                    if not truth_direction_events:
                        learned_non_crossing_opportunities += 1
                        baseline_non_crossing_opportunities += 1
                        learned_false_opportunities += int(
                            any(event[1] == direction for event in learned_events)
                        )
                        baseline_false_opportunities += int(
                            any(event[1] == direction for event in baseline_events)
                        )
        if decision_invalid:
            (
                learned_recall_hits,
                baseline_recall_hits,
                learned_truth_events,
                baseline_truth_events,
                learned_false_opportunities,
                learned_non_crossing_opportunities,
                baseline_false_opportunities,
                baseline_non_crossing_opportunities,
            ) = counter_start
            for mapping, start_length in family_start:
                if start_length:
                    del mapping[family_id][start_length:]
                else:
                    mapping.pop(family_id, None)
            excluded_invalid += 1

    learned_exposure = [
        float(np.mean(values)) for values in family_learned_exposure.values() if values
    ]
    baseline_exposure = [
        float(np.mean(values)) for values in family_baseline_exposure.values() if values
    ]
    family_values = [
        float(np.mean(family_differences[family_id]))
        for family_id in sorted(family_differences)
        if family_differences[family_id]
    ]
    exposure_upper = None
    if family_values:
        differences = np.asarray(family_values, dtype=np.float64)
        rng_count = len(differences)
        bootstrap_indices = _bootstrap_indices(
            BOOTSTRAP_SEED, bootstrap_repetitions, rng_count
        )
        bootstrap = np.mean(differences[bootstrap_indices], axis=1)
        exposure_upper = float(np.quantile(bootstrap, 0.975))

    learned_recall = (
        float(learned_recall_hits / learned_truth_events)
        if learned_truth_events
        else (1.0 if not learned_false_opportunities else 0.0)
    )
    baseline_recall = (
        float(baseline_recall_hits / baseline_truth_events)
        if baseline_truth_events
        else (1.0 if not baseline_false_opportunities else 0.0)
    )
    return {
        "learned_exposure": float(np.mean(learned_exposure)) if learned_exposure else None,
        "baseline_exposure": float(np.mean(baseline_exposure)) if baseline_exposure else None,
        "exposure_difference": float(np.mean(learned_exposure) - np.mean(baseline_exposure))
        if learned_exposure and baseline_exposure
        else None,
        "exposure_difference_upper_95": exposure_upper,
        "learned_dangerous_crossing_recall": learned_recall,
        "baseline_dangerous_crossing_recall": baseline_recall,
        "dangerous_crossing_recall_difference": learned_recall - baseline_recall,
        "learned_dangerous_crossing_false_rate": (
            float(learned_false_opportunities / learned_non_crossing_opportunities)
            if learned_non_crossing_opportunities
            else 0.0
        ),
        "baseline_dangerous_crossing_false_rate": (
            float(baseline_false_opportunities / baseline_non_crossing_opportunities)
            if baseline_non_crossing_opportunities
            else 0.0
        ),
        "dangerous_crossing_false_rate_difference": (
            float(learned_false_opportunities / learned_non_crossing_opportunities)
            if learned_non_crossing_opportunities
            else 0.0
        )
        - (
            float(baseline_false_opportunities / baseline_non_crossing_opportunities)
            if baseline_non_crossing_opportunities
            else 0.0
        ),
        "eligible_candidates": float(12 * len(learned_exposure)),
        "eligible_families": float(len(learned_exposure)),
        "incomplete_decisions": float(incomplete_decisions),
        "excluded_abstentions": float(excluded_abstentions),
        "excluded_invalid_outputs": float(excluded_invalid),
    }


def _latency_ms(
    model: DropoutAwareLinearForecaster,
    samples: Sequence[TrainingSample],
    *,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    groups: dict[tuple[str, int], list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        k = int(np.sum(~sample.history.available_mask[-1]))
        groups[(sample.family_id, k)].append(sample)
    workload = [groups[key] for key in sorted(groups) if len(groups[key]) == 12]
    if not workload:
        return {"p99_ms": math.nan, "timeout_rate": 1.0, "timed_runs": 0.0}
    # Time one complete 12-candidate decision group, matching the runtime gate.
    workload = [workload[0]]

    def invoke() -> None:
        for items in workload:
            for sample in items:
                model.forecast(sample.history, sample.schedule)

    for _ in range(warmup_runs):
        invoke()
    elapsed: list[float] = []
    for _ in range(timed_runs):
        started = time.perf_counter_ns()
        invoke()
        elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(elapsed)
    index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return {
        "p99_ms": float(ordered[index]),
        "timeout_rate": float(sum(value > 250.0 for value in elapsed) / len(elapsed)),
        "timed_runs": float(len(elapsed)),
    }


def _artifact_mapping(
    model: DropoutAwareLinearForecaster,
    config: DropoutConfig,
    dataset_manifest: collector.DropoutDatasetManifest,
    *,
    source_commit: str,
    comparator_model_id: str,
    hmc_contract_sha256: str,
) -> dict[str, object]:
    if model.coefficients is None:
        raise Issue53ForecastError("qualification requires fitted coefficients")
    return {
        "schema_version": "aeolus_habitat_v2_issue53_dropout_model_artifact_v1",
        "model_id": model.model_id,
        "scenario_sha256": model.scenario.scenario_sha256,
        "manifest_sha256": model.manifest.manifest_sha256,
        "topology_sha256": model.manifest.topology_sha256,
        "dropout_config_sha256": config.config_sha256,
        "parent_artifact_sha256": dataset_manifest.parent_artifact_sha256,
        "dataset_sha256": dataset_manifest.dataset_sha256,
        "samples_sha256": dataset_manifest.samples_sha256,
        "source_commit": source_commit,
        "issue53_preregistration_sha256": ISSUE53_PREREGISTRATION_SHA256,
        "issue52_preregistration_sha256": ISSUE52_PREREGISTRATION_SHA256,
        "hmc_contract_sha256": hmc_contract_sha256,
        "coefficients": np.asarray(model.coefficients, dtype=np.float32).tolist(),
        "per_k_interval_scale": {
            str(k): float(value) for k, value in model.per_k_interval_scale.items()
        },
        "abstention_width_limit": float(model.abstention_width_limit),
        "abstention_min_k": int(model.abstention_min_k),
        "abstention_risk_limit": float(model.abstention_risk_limit),
        "risk_coefficients": None
        if model.risk_coefficients is None
        else np.asarray(model.risk_coefficients, dtype=np.float64).tolist(),
        "risk_center": None
        if model.risk_center is None
        else np.asarray(model.risk_center, dtype=np.float64).tolist(),
        "risk_scale": None
        if model.risk_scale is None
        else np.asarray(model.risk_scale, dtype=np.float64).tolist(),
        "risk_intercept": float(model.risk_intercept),
        "comparator_model_id": comparator_model_id,
    }


def _load_artifact_model(
    path: Path,
    scenario: Scenario,
    manifest: TargetManifest,
    config: DropoutConfig,
    dataset_manifest: collector.DropoutDatasetManifest,
    sample: TrainingSample,
    *,
    source_commit: str,
    hmc_contract_sha256: str,
) -> DropoutAwareLinearForecaster:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("model artifact is not an object")
    artifact_digest = _require_sha256(
        document.get("artifact_sha256"), label="model artifact"
    )
    payload = {
        key: value for key, value in document.items() if key != "artifact_sha256"
    }
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != artifact_digest:
        raise ValueError("model artifact digest is inconsistent")
    for key, expected in (
        ("scenario_sha256", scenario.scenario_sha256),
        ("manifest_sha256", manifest.manifest_sha256),
        ("topology_sha256", manifest.topology_sha256),
        ("dropout_config_sha256", config.config_sha256),
        ("parent_artifact_sha256", dataset_manifest.parent_artifact_sha256),
        ("dataset_sha256", dataset_manifest.dataset_sha256),
        ("samples_sha256", dataset_manifest.samples_sha256),
        ("issue53_preregistration_sha256", ISSUE53_PREREGISTRATION_SHA256),
        ("issue52_preregistration_sha256", ISSUE52_PREREGISTRATION_SHA256),
        ("source_commit", source_commit),
        ("hmc_contract_sha256", hmc_contract_sha256),
    ):
        if document.get(key) != expected:
            raise ValueError(f"model artifact binding mismatch: {key}")
    coefficients = np.asarray(document.get("coefficients"), dtype=np.float32)
    if coefficients.ndim != 2:
        raise ValueError("model artifact coefficients are malformed")
    interval_payload = document.get("per_k_interval_scale")
    if not isinstance(interval_payload, Mapping):
        raise ValueError("model artifact interval scales are malformed")
    interval_scales = {
        int(key): _finite_float(value, label="interval scale")
        for key, value in interval_payload.items()
    }
    if (
        not interval_scales
        or set(interval_scales) - set(REQUIRED_K)
        or any(value <= 0.0 for value in interval_scales.values())
    ):
        raise ValueError("model artifact interval scales are invalid")
    model = DropoutAwareLinearForecaster(
        scenario=scenario,
        manifest=manifest,
        dropout_config=config,
        coefficients=coefficients,
        model_id=str(document["model_id"]),
        per_k_interval_scale=interval_scales,
        abstention_width_limit=_finite_float(
            document["abstention_width_limit"], label="abstention width limit"
        ),
        abstention_min_k=int(document["abstention_min_k"]),
        abstention_risk_limit=_finite_float(
            document["abstention_risk_limit"], label="abstention risk limit"
        ),
        risk_coefficients=None
        if document.get("risk_coefficients") is None
        else np.asarray(document["risk_coefficients"], dtype=np.float64),
        risk_center=None
        if document.get("risk_center") is None
        else np.asarray(document["risk_center"], dtype=np.float64),
        risk_scale=None
        if document.get("risk_scale") is None
        else np.asarray(document["risk_scale"], dtype=np.float64),
        risk_intercept=_finite_float(
            document.get("risk_intercept", 0.0), label="risk intercept"
        ),
    )
    trajectory = model.forecast(sample.history, sample.schedule, apply_abstention=False)
    if trajectory.status != "PREDICTION" or trajectory.mean is None:
        raise ValueError("model artifact cannot produce a finite forecast")
    if np.asarray(trajectory.mean).shape != (HORIZON_STEPS, manifest.width):
        raise ValueError("model artifact forecast shape is invalid")
    return model


def _write_addressed_json(path: Path, payload: Mapping[str, object], key: str) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    document = {**payload, key: digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    written = json.loads(path.read_text(encoding="utf-8"))
    written_payload = {key_name: value for key_name, value in written.items() if key_name != key}
    if written.get(key) != digest or hashlib.sha256(
        canonical_json_bytes(written_payload)
    ).hexdigest() != digest:
        raise ValueError(f"{key} write verification failed")
    return digest


def _create_sealed_lock(path: Path, payload: Mapping[str, object]) -> None:
    """Reserve the one-shot FINAL invocation without allowing an overwrite."""

    lock_path = path.with_suffix(path.suffix + ".sealed.lock")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise ValueError("sealed FINAL has already been invoked for this artifact") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise


def _finalize_sealed_lock(path: Path, payload: Mapping[str, object]) -> str:
    lock_path = path.with_suffix(path.suffix + ".sealed.lock")
    if not lock_path.exists():
        raise ValueError("sealed FINAL reservation is missing")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    document = {**payload, "lock_sha256": digest}
    temporary = lock_path.with_name(f".{lock_path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, lock_path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _source_commit() -> str:
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        return _require_commit(commit)
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("cannot determine a valid source commit")


def _source_dirty() -> bool:
    try:
        import subprocess

        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def qualify(
    *,
    dataset: Path,
    artifact: Path,
    report: Path,
    sealed_final: bool,
    timed_runs: int,
    bootstrap_reps: int,
) -> dict[str, object]:
    source_commit = _source_commit()
    if sealed_final:
        if timed_runs != TIMED_RUNS or bootstrap_reps != BOOTSTRAP_REPETITIONS:
            raise ValueError(
                f"sealed FINAL requires timed-runs={TIMED_RUNS} and "
                f"bootstrap-reps={BOOTSTRAP_REPETITIONS}"
            )
        if _source_dirty():
            raise ValueError("sealed FINAL requires a clean source worktree")
        if artifact.exists():
            raise ValueError("sealed FINAL artifact path already exists")
        _create_sealed_lock(
            artifact,
            {
                "schema_version": "aeolus_habitat_v2_issue53_sealed_final_reservation_v1",
                "status": "RESERVED",
                "source_commit": source_commit,
                "preregistration_sha256": ISSUE53_PREREGISTRATION_SHA256,
            },
        )
    raw_scenario = Scenario.from_mapping(
        json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    )
    scenario = extend_scenario_for_issue52(raw_scenario)
    hmc_contract = load_hmc_contract(CONTRACT_PATH)
    (
        config,
        dataset_manifest,
        samples,
        truth_latest_by_key,
        rollout_sha_by_key,
    ) = _load_samples(
        dataset, scenario, require_full=sealed_final
    )
    replay_metrics = _verify_replayed_samples(
        Scenario.from_mapping(json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))),
        hmc_contract,
        config,
        samples,
        truth_latest_by_key,
        rollout_sha_by_key,
    )
    groups = _group_samples(samples)
    train = groups.get("TRAIN", {}).get(0, ())
    validation = tuple(item for items in groups.get("VALIDATION", {}).values() for item in items)
    final = tuple(item for items in groups.get("FINAL", {}).values() for item in items)
    if not train or not validation or (sealed_final and not final):
        raise ValueError("dataset does not contain the required train/validation/final partitions")
    reference_manifest = TargetManifest.from_scenario(scenario)

    learned = DropoutAwareLinearForecaster.fit_for_scenario(
        scenario,
        reference_manifest,
        train,
        dropout_config=config,
        alpha=1e-4,
        augment_dropout=True,
    )
    baseline_candidates = (
        MaskAwarePersistenceForecaster(reference_manifest),
        MaskAwareLinearContinuationForecaster(reference_manifest),
    )
    validation_k0 = groups.get("VALIDATION", {}).get(0, ())
    baseline, baseline_scores = _select_baseline(
        baseline_candidates, validation_k0, reference_manifest
    )
    risk_train = groups.get("TRAIN", {}).get(3, ())
    if not risk_train:
        raise ValueError("dataset does not contain TRAIN k=3 samples for risk calibration")
    train_oracle = [_baseline_nmae(baseline, sample, reference_manifest) for sample in risk_train]
    if any(value is None for value in train_oracle):
        raise ValueError("selected baseline produced an invalid TRAIN risk target")
    learned = learned.fit_risk_head(risk_train, train_oracle, alpha=0.1)
    validation_oracle_k0 = [
        _baseline_nmae(baseline, sample, reference_manifest) for sample in validation_k0
    ]
    if any(value is None for value in validation_oracle_k0):
        raise ValueError("selected baseline produced an invalid VALIDATION oracle target")
    validation_family_oracle: dict[str, list[float]] = defaultdict(list)
    for sample, error in zip(validation_k0, validation_oracle_k0):
        validation_family_oracle[sample.family_id].append(error)
    oracle_threshold = float(
        np.quantile(
            np.asarray([np.mean(values) for values in validation_family_oracle.values()]),
            0.90,
        )
    )
    validation_oracle = [
        _baseline_nmae(baseline, sample, reference_manifest) for sample in validation
    ]
    if any(value is None for value in validation_oracle):
        raise ValueError("selected baseline produced an invalid abstention oracle target")
    learned = learned.calibrate(
        validation,
        oracle_errors=validation_oracle,
        oracle_high_error_threshold=oracle_threshold,
        abstention_k=3,
    )

    artifact_payload = _artifact_mapping(
        learned,
        config,
        dataset_manifest,
        source_commit=source_commit,
        comparator_model_id=baseline.model_id,
        hmc_contract_sha256=hmc_contract.hmc_contract_sha256,
    )
    artifact_digest = _write_addressed_json(artifact, artifact_payload, "artifact_sha256")
    evaluation = final if sealed_final else validation
    if not evaluation:
        raise ValueError("qualification evaluation partition is empty")
    learned = _load_artifact_model(
        artifact,
        scenario,
        reference_manifest,
        config,
        dataset_manifest,
        evaluation[0],
        source_commit=source_commit,
        hmc_contract_sha256=hmc_contract.hmc_contract_sha256,
    )

    evaluation_groups = _group_samples(evaluation)
    metrics: dict[str, object] = {
        "dataset": {
            "families": len(dataset_manifest.family_ids),
            "samples": len(samples),
            "dataset_sha256": dataset_manifest.dataset_sha256,
            "samples_sha256": dataset_manifest.samples_sha256,
            "dropout_config_sha256": config.config_sha256,
        },
        "artifact_sha256": artifact_digest,
        "artifact_model_id": learned.model_id,
        "source_commit": source_commit,
        "oracle_validation_k0_p90": oracle_threshold,
        "baseline_selection": {
            "selected_model_id": baseline.model_id,
            "validation_k0_nmae": baseline_scores,
        },
        "partitions": {
            split: {str(k): len(items) for k, items in by_k.items()}
            for split, by_k in groups.items()
        },
    }

    per_k: dict[str, object] = {}
    values_by_k: dict[
        int,
        tuple[dict[str, float | None], dict[str, float | None], dict[str, int]],
    ] = {}
    for k in REQUIRED_K:
        items = evaluation_groups.get("FINAL" if sealed_final else "VALIDATION", {}).get(k, ())
        learned_values, baseline_values, counts = _paired_decision_values(
            learned, baseline, items, reference_manifest
        )
        values_by_k[k] = (learned_values, baseline_values, counts)

    learned_families_by_k = {
        k: values[0] for k, values in values_by_k.items()
    }
    baseline_families_by_k = {
        k: values[1] for k, values in values_by_k.items()
    }
    for k in REQUIRED_K:
        items = evaluation_groups.get("FINAL" if sealed_final else "VALIDATION", {}).get(k, ())
        learned_family = learned_families_by_k[k]
        baseline_family = baseline_families_by_k[k]
        own_k0 = learned_families_by_k.get(0, {})
        ratio, paired, ratio_valid = _strict_ratio(learned_family, own_k0)
        baseline_ratio, baseline_paired, baseline_ratio_valid = _strict_ratio(
            learned_family, baseline_family
        )
        ci_low, ci_high = _bootstrap_ratio(
            learned_family,
            own_k0,
            seed=BOOTSTRAP_SEED,
            repetitions=bootstrap_reps,
        )
        coverage, width, cells = _coverage_and_width(learned, items, reference_manifest)
        oracle_errors = [
            _baseline_nmae(baseline, sample, reference_manifest) for sample in items
        ]
        abstention = _abstention_metrics(
            learned, items, oracle_errors, oracle_threshold
        )
        per_k[str(k)] = {
            "nmae_h9_h32": _mean_or_none(
                [value for value in learned_family.values() if value is not None]
            ),
            "learned_nmae_h9_h32": _mean_or_none(
                [value for value in learned_family.values() if value is not None]
            ),
            "baseline_nmae_h9_h32": _mean_or_none(
                [value for value in baseline_family.values() if value is not None]
            ),
            "ratio_vs_own_k0": ratio,
            "ratio_vs_frozen_persistence": baseline_ratio,
            "ratio_valid": ratio_valid,
            "baseline_ratio_valid": baseline_ratio_valid,
            "ratio_ci_95": [ci_low, ci_high],
            "paired_families": len(paired),
            "baseline_paired_families": len(baseline_paired),
            "interval_coverage": coverage,
            "mean_normalized_interval_width": width,
            "target_cells": cells,
            "abstention": abstention,
            "decision_accounting": values_by_k[k][2],
        }
    metrics["per_k"] = per_k

    endpoint_groups = evaluation_groups.get("FINAL" if sealed_final else "VALIDATION", {})
    primary_values, primary_baseline_values, primary_counts = _paired_decision_values(
        learned,
        baseline,
        endpoint_groups.get(0, ()),
        reference_manifest,
        start=HORIZON_START,
        end=HORIZON_STEPS,
    )
    short_values, short_baseline_values, short_counts = _paired_decision_values(
        learned,
        baseline,
        endpoint_groups.get(0, ()),
        reference_manifest,
        start=0,
        end=HORIZON_START,
    )
    primary_ratio, primary_families, primary_valid = _strict_ratio(
        primary_values, primary_baseline_values
    )
    primary_ci = _bootstrap_ratio(
        primary_values,
        primary_baseline_values,
        seed=BOOTSTRAP_SEED,
        repetitions=bootstrap_reps,
    )
    short_ratio, short_families, short_valid = _strict_ratio(
        short_values, short_baseline_values
    )
    short_ci = _bootstrap_ratio(
        short_values,
        short_baseline_values,
        seed=BOOTSTRAP_SEED,
        repetitions=bootstrap_reps,
    )
    metrics["primary_forecast"] = {
        "ratio_h9_h32": primary_ratio,
        "ratio_ci_95": list(primary_ci),
        "paired_families": len(primary_families),
        "valid": primary_valid,
        "decision_accounting": primary_counts,
        "short_horizon_ratio_h1_h8": short_ratio,
        "short_horizon_ratio_ci_95": list(short_ci),
        "short_horizon_paired_families": len(short_families),
        "short_horizon_valid": short_valid,
        "short_horizon_decision_accounting": short_counts,
    }

    if sealed_final:
        k0_learned = {
            family: value
            for family, value in learned_families_by_k[0].items()
            if value is not None
        }
        k1_learned = {
            family: value
            for family, value in learned_families_by_k[1].items()
            if value is not None
        }
        k3_learned = {
            family: value
            for family, value in learned_families_by_k[3].items()
            if value is not None
        }
        k1_ratio, k1_paired = _ratio(k1_learned, k0_learned)
        k3_ratio, k3_paired = _ratio(k3_learned, k0_learned)
        metrics["dropout_degradation"] = {
            "k1_vs_k0": k1_ratio,
            "k1_paired_families": len(k1_paired),
            "k3_vs_k0": k3_ratio,
            "k3_paired_families": len(k3_paired),
        }
    safety = {}
    for k in (0, 1, 3):
        items = evaluation_groups.get("FINAL" if sealed_final else "VALIDATION", {}).get(k, ())
        safety[str(k)] = _crossing_metrics(
            learned,
            baseline,
            items,
            reference_manifest,
            scenario,
            truth_latest_by_key,
            bootstrap_repetitions=bootstrap_reps,
        )
    metrics["safety"] = safety
    latency_items = final if sealed_final else validation
    metrics["latency"] = _latency_ms(
        learned,
        latency_items,
        warmup_runs=20 if sealed_final else min(3, timed_runs),
        timed_runs=timed_runs,
    )
    metrics["status_counts"] = {
        "validation": _status_counts_by_k(learned, validation),
        "evaluation": _status_counts_by_k(learned, evaluation),
    }
    k0_evaluation = evaluation_groups.get("FINAL" if sealed_final else "VALIDATION", {}).get(0, ())
    k3_evaluation = evaluation_groups.get("FINAL" if sealed_final else "VALIDATION", {}).get(3, ())
    k0_baseline_rate = _abstention_rate_at_k(baseline, k0_evaluation, 0)
    k0_learned_rate = _abstention_rate_at_k(learned, k0_evaluation, 0)
    metrics["abstention_rate_delta_k0"] = (
        float(k0_learned_rate - k0_baseline_rate)
        if math.isfinite(k0_learned_rate) and math.isfinite(k0_baseline_rate)
        else None
    )
    metrics["abstention_pr_curve_k3"] = _abstention_pr_curve(
        learned,
        k3_evaluation,
        [
            _baseline_nmae(baseline, sample, reference_manifest)
            for sample in k3_evaluation
        ],
        oracle_threshold,
    )
    metrics.update(replay_metrics)
    metrics["source_worktree_dirty"] = _source_dirty()
    metrics["non_finite_committed_state_count"] = replay_metrics[
        "non_finite_committed_state_count"
    ]

    safety_gate = all(
        value["exposure_difference"] is not None
        and value["exposure_difference"] <= 0.0
        and value["exposure_difference_upper_95"] is not None
        and value["exposure_difference_upper_95"] <= 0.0
        and value["dangerous_crossing_recall_difference"] >= -0.02
        and value["dangerous_crossing_false_rate_difference"] <= 0.01
        for value in safety.values()
    )
    nmae_gate = all(
        per_k[str(k)]["learned_nmae_h9_h32"] is not None
        and per_k[str(k)]["baseline_nmae_h9_h32"] is not None
        for k in (0, 1, 3)
    )
    gates = {
        "dataset_complete": bool(sealed_final),
        "nmae_finite": nmae_gate,
        "k1_coverage": per_k["1"]["interval_coverage"] is not None and per_k["1"]["interval_coverage"] >= 0.85,
        "k3_coverage": per_k["3"]["interval_coverage"] is not None and per_k["3"]["interval_coverage"] >= 0.80,
        "k3_abstention_recall": per_k["3"]["abstention"]["recall"] is not None and per_k["3"]["abstention"]["recall"] >= 0.80,
        "k3_abstention_precision": per_k["3"]["abstention"]["precision"] is not None and per_k["3"]["abstention"]["precision"] >= 0.60,
        "latency": metrics["latency"]["p99_ms"] <= 250.0 and metrics["latency"]["timeout_rate"] == 0.0,
        "invalid_output_timeout": all(
            all(
                counts["INVALID_OUTPUT"] == 0
                for counts in metrics["status_counts"][key].values()
            )
            for key in ("validation", "evaluation")
        ),
        "primary_forecast": (
            metrics["primary_forecast"]["valid"]
            and metrics["primary_forecast"]["ratio_h9_h32"] is not None
            and metrics["primary_forecast"]["ratio_h9_h32"] <= 0.90
            and metrics["primary_forecast"]["ratio_ci_95"][1] is not None
            and metrics["primary_forecast"]["ratio_ci_95"][1] < 0.98
        ),
        "short_horizon": (
            metrics["primary_forecast"]["short_horizon_valid"]
            and metrics["primary_forecast"]["short_horizon_ratio_h1_h8"] is not None
            and metrics["primary_forecast"]["short_horizon_ratio_h1_h8"] <= 1.05
            and metrics["primary_forecast"]["short_horizon_ratio_ci_95"][1] is not None
            and metrics["primary_forecast"]["short_horizon_ratio_ci_95"][1] <= 1.08
        ),
        "k1_degradation": (
            per_k["1"]["ratio_valid"]
            and per_k["1"]["ratio_vs_own_k0"] is not None
            and per_k["1"]["ratio_vs_own_k0"] <= 1.15
            and per_k["1"]["ratio_ci_95"][1] is not None
            and per_k["1"]["ratio_ci_95"][1] <= 1.25
        ),
        "k3_degradation": (
            per_k["3"]["ratio_valid"]
            and per_k["3"]["ratio_vs_own_k0"] is not None
            and per_k["3"]["ratio_vs_own_k0"] <= 1.40
            and per_k["3"]["ratio_ci_95"][1] is not None
            and per_k["3"]["ratio_ci_95"][1] <= 1.60
        ),
        "k0_abstention_rate_delta": (
            metrics["abstention_rate_delta_k0"] is not None
            and metrics["abstention_rate_delta_k0"] <= 0.02
        ),
        "safety": safety_gate,
        "provenance": not metrics["source_worktree_dirty"],
        "authority": metrics["authority_violation_count"] == 0,
        "replay": metrics["replay_failure_count"] == 0,
        "non_finite_committed_state": metrics["non_finite_committed_state_count"] == 0,
    }
    metrics["gates"] = gates
    metrics["status"] = "QUALIFIED" if sealed_final and all(gates.values()) else "NOT QUALIFIED"
    metrics["run_type"] = "SEALED_FINAL" if sealed_final else "PILOT_VALIDATION"
    report_digest = _write_addressed_json(report, metrics, "report_sha256")
    if sealed_final:
        _finalize_sealed_lock(
            artifact,
            {
                "schema_version": "aeolus_habitat_v2_issue53_sealed_final_lock_v1",
                "status": metrics["status"],
                "artifact_sha256": artifact_digest,
                "report_sha256": report_digest,
                "dataset_sha256": dataset_manifest.dataset_sha256,
                "samples_sha256": dataset_manifest.samples_sha256,
                "source_commit": source_commit,
                "hmc_contract_sha256": hmc_contract.hmc_contract_sha256,
                "preregistration_sha256": ISSUE53_PREREGISTRATION_SHA256,
            },
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify the Issue #53 dropout lane")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--sealed-final",
        action="store_true",
        help="require the 384-family corpus and evaluate FINAL once",
    )
    parser.add_argument("--timed-runs", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    if args.timed_runs < 1 or args.bootstrap_reps < 1:
        raise SystemExit("timed-runs and bootstrap-reps must be positive")
    result = qualify(
        dataset=args.dataset,
        artifact=args.artifact,
        report=args.report,
        sealed_final=args.sealed_final,
        timed_runs=args.timed_runs,
        bootstrap_reps=args.bootstrap_reps,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
