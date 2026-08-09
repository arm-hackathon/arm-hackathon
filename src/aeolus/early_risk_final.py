"""One-shot closed-loop final evaluation for the frozen early-risk adviser."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from aeolus.config import load_scenario
from aeolus.early_risk import load_early_risk_artifact
from aeolus.families import (
    FamilyManifest,
    load_family_manifest,
    validate_manifest_disjointness,
)
from aeolus.recovery import RecoverySettings
from aeolus.recovery_evidence import _affected_zone, _arm_metrics
from aeolus.scenario import RECOVERY_RUN, run_recovery_scenario
from aeolus.sweep import SWEEP_V4_VERSION, generate_sweep, load_sweep_spec

FINAL_EVIDENCE_VERSION = "aeolus_early_risk_final_v1"
FROZEN_SWEEP_CANONICAL_SHA256 = (
    "d70da5bcad631b2d29b8f801e6679ffefad6bdeb4dc0bb647efc67a3892d7077"
)
FROZEN_ARTIFACT_BYTES_SHA256 = (
    "2f88fac553f3dba6abd3c6f0a4793aa921fbeeb8682b4de740eca88a490b5139"
)
FROZEN_ARTIFACT_SHA256 = (
    "6eeaf089e8ddb07ce6e5304841b48e1d5fa3e5325c0e5f3b7852193d04740063"
)
FROZEN_ARTIFACT_MODEL_SHA256 = (
    "77910da137fbb51ffe4faa995ff837edd8999474a95fffde664fc444d056701c"
)
FROZEN_FINAL_FAMILY_COMPOSITION = {
    "blocked_path": 24,
    "frozen_sensor": 24,
    "gradual_primary_fan_degradation": 48,
    "transient_blocked_path": 24,
    "transient_gradual_primary_fan_degradation": 24,
}
FROZEN_FORBIDDEN_MANIFESTS = {
    "961aa0dc1ae0bc2fe97f4405181195da7e7bc6fa8345303c2b56f8d58e74646c": {
        "role": "temporal_predictor_development",
        "bytes_sha256": "dc31547e3e85d38b57c07adb917cfc3c2a3fb1e990a4ff758baeb743561873db",
        "family_count": 216,
        "splits": ["train", "validation"],
    },
    "26b46fa0f78e70cc1a2449ef3c84215c39f6afa3a00616ff52b3ca2fb471685c": {
        "role": "deterministic_recovery_final",
        "bytes_sha256": "5c01fd2c994ecee6119620cd3ecdb02843c3c5ba91e5b212a72d7ce8568345bf",
        "family_count": 252,
        "splits": ["final"],
    },
}
_TRANSIENT_FAULT_CLASSES = frozenset(
    {
        "transient_blocked_path",
        "transient_gradual_primary_fan_degradation",
    }
)
_SOURCE_FILES = (
    "src/aeolus/early_risk.py",
    "src/aeolus/early_risk_final.py",
    "src/aeolus/families.py",
    "src/aeolus/recovery.py",
    "src/aeolus/recovery_evidence.py",
    "src/aeolus/scenario.py",
    "src/aeolus/sweep.py",
    "src/aeolus/trace.py",
)
_SUMMARY_FIELDS = frozenset(
    {
        "final_families",
        "unique_healthy_references",
        "harmful_physical_families",
        "harmful_gradual_families",
        "healthy_reference_interventions",
        "frozen_sensor_interventions",
        "wrong_target_interventions",
        "invariant_violations",
        "missed_harmful_physical_families",
        "worsened_harmful_physical_families",
        "transient_repeated_protection_families",
        "transient_handback_recurrence_families",
        "transient_handback_timeout_families",
        "transient_nonzero_final_reserve_families",
        "harmful_gradual_earlier_protection_families",
        "harmful_gradual_positive_excess_reduction_families",
        "median_excess_reduction_fraction_vs_governor_harmful_gradual",
    }
)
_COUNT_FIELDS = _SUMMARY_FIELDS - {
    "median_excess_reduction_fraction_vs_governor_harmful_gradual"
}


@dataclass(frozen=True)
class FinalGateThresholds:
    """Predeclared final admission and physical-benefit thresholds."""

    minimum_final_families: int = 144
    minimum_harmful_gradual_families: int = 8
    minimum_earlier_protection_fraction: float = 0.40
    minimum_positive_excess_reduction_fraction: float = 0.25
    minimum_median_excess_reduction_fraction: float = 0.10


FROZEN_THRESHOLDS = FinalGateThresholds()


def assess_final_summary(
    summary: dict[str, Any], thresholds: FinalGateThresholds
) -> dict[str, Any]:
    """Apply the frozen final gates to a strictly shaped metric summary."""
    if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError("final summary fields do not match the frozen gate contract")
    if not isinstance(thresholds, FinalGateThresholds):
        raise TypeError("final gate thresholds are malformed")
    for field in _COUNT_FIELDS:
        value = summary[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"final summary field {field!r} must be a non-negative integer")
    median_reduction = summary[
        "median_excess_reduction_fraction_vs_governor_harmful_gradual"
    ]
    if median_reduction is not None and (
        not isinstance(median_reduction, (int, float))
        or isinstance(median_reduction, bool)
        or not math.isfinite(float(median_reduction))
    ):
        raise ValueError("final median excess reduction must be finite or null")

    harmful_gradual = summary["harmful_gradual_families"]
    required_earlier = math.ceil(
        thresholds.minimum_earlier_protection_fraction * harmful_gradual
    )
    required_positive = math.ceil(
        thresholds.minimum_positive_excess_reduction_fraction * harmful_gradual
    )
    safety_failures = {
        field: summary[field]
        for field in (
            "healthy_reference_interventions",
            "frozen_sensor_interventions",
            "wrong_target_interventions",
            "invariant_violations",
            "missed_harmful_physical_families",
            "worsened_harmful_physical_families",
            "transient_repeated_protection_families",
            "transient_handback_recurrence_families",
            "transient_handback_timeout_families",
            "transient_nonzero_final_reserve_families",
        )
        if summary[field] != 0
    }
    admission_pass = (
        summary["final_families"] >= thresholds.minimum_final_families
        and summary["unique_healthy_references"] > 0
    )
    safety_pass = admission_pass and not safety_failures
    benefit_failures: list[str] = []
    if harmful_gradual < thresholds.minimum_harmful_gradual_families:
        benefit_failures.append("insufficient_harmful_gradual_families")
    if summary["harmful_gradual_earlier_protection_families"] < required_earlier:
        benefit_failures.append("insufficient_earlier_protection")
    if (
        summary["harmful_gradual_positive_excess_reduction_families"]
        < required_positive
    ):
        benefit_failures.append("insufficient_positive_physical_reduction")
    if median_reduction is None or (
        float(median_reduction)
        < thresholds.minimum_median_excess_reduction_fraction
    ):
        benefit_failures.append("insufficient_median_physical_reduction")
    benefit_pass = not benefit_failures
    verdict = (
        "REJECT_SAFETY"
        if not safety_pass
        else "PASS"
        if benefit_pass
        else "REJECT_BENEFIT"
    )
    return {
        "admission_gate_pass": admission_pass,
        "safety_gate_pass": safety_pass,
        "benefit_gate_pass": benefit_pass,
        "verdict": verdict,
        "required_earlier_protection_families": required_earlier,
        "required_positive_excess_reduction_families": required_positive,
        "safety_failures": safety_failures,
        "benefit_failures": benefit_failures,
    }


def run_final_evaluation(
    *,
    repo: str | Path,
    sweep_spec_path: str | Path,
    corpus_dir: str | Path,
    artifact_path: str | Path,
    forbidden_manifest_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the frozen three-arm final comparison exactly once."""
    selected_thresholds = FROZEN_THRESHOLDS
    repository = Path(repo).resolve()
    sweep_path = Path(sweep_spec_path).resolve()
    corpus = Path(corpus_dir).resolve()
    artifact = Path(artifact_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"final comparison output already exists: {output}")
    source_before = _source_snapshot(repository, require_clean=True)
    sweep = load_sweep_spec(sweep_path)
    _validate_frozen_sweep(sweep)
    manifest_path = corpus / "families.json"
    generation_receipt_path = corpus / "sweep-receipt.json"
    manifest = load_family_manifest(manifest_path)
    families = _require_final_families(manifest)
    corpus_proof = _validate_corpus_against_frozen_spec(sweep_path, corpus)
    generation_receipt = _load_generation_receipt(
        generation_receipt_path,
        expected_document=corpus_proof["generation_receipt"],
    )
    forbidden_receipts = _validate_forbidden_manifests(
        forbidden_manifest_paths, final_manifest=manifest
    )

    predictor, training_receipt, calibration_receipt = load_early_risk_artifact(artifact)
    artifact_document = json.loads(artifact.read_text(encoding="utf-8"))
    artifact_identity = artifact_document["artifact_sha256"]
    predictor.assert_artifact_identity(artifact_identity)
    _validate_frozen_artifact(artifact, predictor)
    source_file_hashes = {
        relative: _sha256_file(repository / relative) for relative in _SOURCE_FILES
    }
    run_lock_path = corpus / ".early-risk-final-run-lock.json"
    run_lock = {
        "schema_version": "aeolus_early_risk_final_run_lock_v1",
        "git_head": source_before["git_head"],
        "sweep_spec_canonical_sha256": sweep.sha256,
        "family_manifest_canonical_sha256": manifest.manifest_sha256,
        "artifact_sha256": artifact_identity,
        "artifact_bytes_sha256": _sha256_file(artifact),
        "source_file_sha256": source_file_hashes,
        "output_path": str(output),
    }
    _claim_final_suite(run_lock_path, run_lock)
    reference_cache: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    settings = RecoverySettings()

    for index, family in enumerate(families, 1):
        print(f"[{index}/{len(families)}] {family.family_id}", flush=True)
        reference_key = family.reference_path.resolve()
        if reference_key not in reference_cache:
            reference_config = load_scenario(family.reference_path)
            reference_governor = _run_arm(
                reference_config,
                run_id=f"reference:{family.reference_path.stem}:governor",
                governed=True,
                settings=settings,
            )
            reference_advisory = _run_arm(
                reference_config,
                run_id=f"reference:{family.reference_path.stem}:advisory",
                governed=True,
                settings=settings,
                predictor=predictor,
                artifact_identity=artifact_identity,
            )
            reference_cache[reference_key] = {
                "governor": reference_governor,
                "advisory": reference_advisory,
                "governor_metrics": _arm_metrics(
                    reference_governor, config=reference_config, run=RECOVERY_RUN
                ),
                "advisory_metrics": _arm_metrics(
                    reference_advisory, config=reference_config, run=RECOVERY_RUN
                ),
            }

        fault_config = load_scenario(family.fault_path)
        target = _affected_zone(fault_config)
        reserve_off = _run_arm(
            fault_config,
            run_id=f"{family.family_id}:off",
            governed=False,
            settings=settings,
        )
        governor = _run_arm(
            fault_config,
            run_id=f"{family.family_id}:governor",
            governed=True,
            settings=settings,
        )
        advisory = _run_arm(
            fault_config,
            run_id=f"{family.family_id}:advisory",
            governed=True,
            settings=settings,
            predictor=predictor,
            artifact_identity=artifact_identity,
        )
        off_metrics = _arm_metrics(reserve_off, config=fault_config, run=RECOVERY_RUN)
        governor_metrics = _arm_metrics(governor, config=fault_config, run=RECOVERY_RUN)
        advisory_metrics = _arm_metrics(advisory, config=fault_config, run=RECOVERY_RUN)
        off_excess = float(off_metrics["integrated_physical_co2_excess"][target])
        governor_excess = float(
            governor_metrics["integrated_physical_co2_excess"][target]
        )
        advisory_excess = float(
            advisory_metrics["integrated_physical_co2_excess"][target]
        )
        governor_first = governor_metrics["states"]["first_protect_tick"]
        advisory_first = advisory_metrics["states"]["first_protect_tick"]
        reduction = governor_excess - advisory_excess
        rows.append(
            {
                "family_id": family.family_id,
                "split": family.split,
                "fault_class": family.fault_class,
                "target_zone_id": target,
                "model_warning_count": _warning_count(advisory),
                "accepted_advisory_observation_count": _accepted_warning_count(advisory),
                "governor_first_protect_tick": governor_first,
                "advisory_governor_first_protect_tick": advisory_first,
                "protection_lead_ticks": (
                    governor_first - advisory_first
                    if governor_first is not None and advisory_first is not None
                    else None
                ),
                "reserve_off_integrated_excess": off_excess,
                "governor_integrated_excess": governor_excess,
                "advisory_governor_integrated_excess": advisory_excess,
                "advisory_excess_reduction_vs_governor": reduction,
                "advisory_excess_reduction_fraction_vs_governor": (
                    reduction / governor_excess if governor_excess > 0.0 else None
                ),
                "governor_protect_targets": governor_metrics["lifecycle"][
                    "protect_target_zone_ids"
                ],
                "advisory_governor_protect_targets": advisory_metrics["lifecycle"][
                    "protect_target_zone_ids"
                ],
                "governor_protect_entry_count": governor_metrics["lifecycle"][
                    "protect_entry_count"
                ],
                "advisory_governor_protect_entry_count": advisory_metrics["lifecycle"][
                    "protect_entry_count"
                ],
                "advisory_governor_handback_recurrence_count": advisory_metrics[
                    "lifecycle"
                ]["handback_recurrence_count"],
                "advisory_governor_handback_timeout_count": advisory_metrics["lifecycle"][
                    "handback_timeout_count"
                ],
                "advisory_governor_invariant_violation_count": advisory_metrics[
                    "invariant_violation_count"
                ],
                "advisory_governor_final_physical_zero": advisory_metrics["lifecycle"][
                    "final_physical_zero"
                ],
            }
        )

    reference_rows = _reference_rows(reference_cache)
    summary, diagnostics = _summarise(rows, reference_rows)
    assessment = assess_final_summary(summary, selected_thresholds)
    source_after = _source_snapshot(repository, require_clean=True)
    if source_after != source_before:
        raise ValueError("source provenance changed during final evaluation")

    report: dict[str, Any] = {
        "schema_version": FINAL_EVIDENCE_VERSION,
        "scope": "one-shot untouched predictor-final families",
        "source": source_before,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "run_spec": asdict(RECOVERY_RUN),
        "recovery_settings": asdict(settings),
        "thresholds": asdict(selected_thresholds),
        "provenance": {
            "sweep_spec_path": str(sweep_path),
            "sweep_spec_bytes_sha256": _sha256_file(sweep_path),
            "sweep_spec_canonical_sha256": sweep.sha256,
            "generation_receipt_path": str(generation_receipt_path),
            "generation_receipt_bytes_sha256": _sha256_file(
                generation_receipt_path
            ),
            "generation_receipt": generation_receipt,
            "corpus_payload_tree_sha256": corpus_proof[
                "corpus_payload_tree_sha256"
            ],
            "corpus_file_count": corpus_proof["corpus_file_count"],
            "run_lock_path": str(run_lock_path),
            "run_lock_bytes_sha256": _sha256_file(run_lock_path),
            "run_lock": run_lock,
            "family_manifest_path": str(manifest_path),
            "family_manifest_bytes_sha256": _sha256_file(manifest_path),
            "family_manifest_canonical_sha256": manifest.manifest_sha256,
            "forbidden_manifests": forbidden_receipts,
            "artifact_path": str(artifact),
            "artifact_bytes_sha256": _sha256_file(artifact),
            "artifact_sha256": artifact_identity,
            "artifact_model_sha256": predictor.artifact_model_sha256,
            "training_receipt": training_receipt,
            "calibration_receipt": calibration_receipt,
            "source_file_sha256": source_file_hashes,
        },
        "metric_polarity": {
            "model_warning": "advisory only; a warning is not an intervention",
            "harmful_fault_intervention": "desired when reserve-off physical excess is positive",
            "healthy_reference_intervention": "undesired reserve activation; zero is good",
            "frozen_sensor_intervention": "undesired physical reserve activation; zero is good",
            "wrong_target_intervention": "undesired reserve activation for the wrong zone; zero is good",
            "physical_excess_reduction": "positive is improvement; negative is worsening",
        },
        "summary": summary,
        "diagnostics": diagnostics,
        "assessment": assessment,
        "healthy_references": reference_rows,
        "per_family": rows,
    }
    report["report_sha256"] = _canonical_sha256(report)
    _write_json_new(output, report)
    return report


def _require_final_families(manifest: FamilyManifest) -> tuple[Any, ...]:
    families = tuple(manifest.families)
    splits = {family.split for family in families}
    if splits != {"final"}:
        raise ValueError("early-risk final manifest must contain only final families")
    if len(families) != FROZEN_THRESHOLDS.minimum_final_families:
        raise ValueError("early-risk final manifest family count is not frozen")
    composition = Counter(family.fault_class for family in families)
    if dict(composition) != FROZEN_FINAL_FAMILY_COMPOSITION:
        raise ValueError("early-risk final manifest composition is not frozen")
    if len({family.reference_path.name for family in families}) != 4:
        raise ValueError("early-risk final manifest reference count is not frozen")
    return families


def _validate_frozen_sweep(sweep: Any) -> None:
    """Require the one predeclared final sweep rather than any valid final suite."""
    if sweep.schema_version != SWEEP_V4_VERSION or sweep.suite_role != "final":
        raise ValueError("early-risk final evaluation requires an aeolus_sweep_v4 final suite")
    if sweep.sha256 != FROZEN_SWEEP_CANONICAL_SHA256:
        raise ValueError("early-risk final sweep does not match the frozen canonical hash")


def _load_generation_receipt(
    path: Path, *, expected_document: dict[str, Any]
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load final sweep generation receipt: {exc}") from None
    if not isinstance(document, dict):
        raise ValueError("final sweep generation receipt must be an object")
    if document != expected_document:
        raise ValueError("final sweep generation receipt is stale or substituted")
    return document


def _validate_corpus_against_frozen_spec(
    sweep_path: Path, corpus: Path
) -> dict[str, Any]:
    """Regenerate the pinned suite and byte-compare every generated payload."""
    if not corpus.is_dir():
        raise ValueError(f"final corpus directory does not exist: {corpus}")
    with TemporaryDirectory(prefix="aeolus-final-preflight-") as temporary:
        expected_root = Path(temporary)
        generate_sweep(sweep_path, expected_root)
        expected_tree = _corpus_file_tree(expected_root)
        actual_tree = _corpus_file_tree(corpus)
        if actual_tree != expected_tree:
            raise ValueError(
                "final corpus payload tree does not match the frozen deterministic sweep"
            )
        generation_receipt = json.loads(
            (expected_root / "sweep-receipt.json").read_text(encoding="utf-8")
        )
    return {
        "corpus_payload_tree_sha256": _canonical_sha256(actual_tree),
        "corpus_file_count": len(actual_tree),
        "generation_receipt": generation_receipt,
    }


def _corpus_file_tree(root: Path) -> dict[str, str]:
    """Hash the complete generated corpus tree, excluding the one-shot lock."""
    ignored = {".early-risk-final-run-lock.json"}
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in ignored
    }


def _validate_forbidden_manifests(
    paths: Sequence[str | Path], *, final_manifest: FamilyManifest
) -> list[dict[str, Any]]:
    """Require the exact distinct development and prior-final identity set."""
    if len(paths) != len(FROZEN_FORBIDDEN_MANIFESTS):
        raise ValueError("final evaluation requires the exact frozen forbidden set")
    receipts: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        manifest = load_family_manifest(path)
        canonical = manifest.manifest_sha256
        if canonical in receipts:
            raise ValueError("final evaluation forbidden manifests must be distinct")
        expected = FROZEN_FORBIDDEN_MANIFESTS.get(canonical)
        actual = {
            "path": str(path),
            "bytes_sha256": _sha256_file(path),
            "canonical_manifest_sha256": canonical,
            "family_count": len(manifest.families),
            "splits": sorted({family.split for family in manifest.families}),
        }
        if expected is None or (
            actual["bytes_sha256"] != expected["bytes_sha256"]
            or actual["family_count"] != expected["family_count"]
            or actual["splits"] != expected["splits"]
        ):
            raise ValueError("forbidden manifest does not match a frozen identity")
        validate_manifest_disjointness(final_manifest, manifest)
        actual["role"] = expected["role"]
        receipts[canonical] = actual
    if set(receipts) != set(FROZEN_FORBIDDEN_MANIFESTS):
        raise ValueError("final evaluation forbidden manifest set is incomplete")
    return [receipts[identity] for identity in sorted(receipts)]


def _validate_frozen_artifact(path: Path, predictor: Any) -> None:
    """Reject any retrained, recalibrated or reserialized candidate artifact."""
    if _sha256_file(path) != FROZEN_ARTIFACT_BYTES_SHA256:
        raise ValueError("early-risk artifact bytes do not match the frozen candidate")
    if predictor.artifact_sha256 != FROZEN_ARTIFACT_SHA256:
        raise ValueError("early-risk artifact identity does not match the frozen candidate")
    if predictor.artifact_model_sha256 != FROZEN_ARTIFACT_MODEL_SHA256:
        raise ValueError("early-risk model payload does not match the frozen candidate")


def _run_arm(
    config,
    *,
    run_id: str,
    governed: bool,
    settings: RecoverySettings,
    predictor=None,
    artifact_identity: str | None = None,
):
    return run_recovery_scenario(
        config,
        run_id=run_id,
        governed=governed,
        run=RECOVERY_RUN,
        settings=settings if governed else None,
        early_risk_predictor=predictor,
        early_risk_artifact_sha256=artifact_identity,
    )


def _warning_count(result) -> int:
    return sum(advisory is not None for advisory in result.advisories)


def _accepted_warning_count(result) -> int:
    return sum(
        decision.reason in {"advisory_unique_concern", "advisory_entry_persistence_met"}
        for decision in result.decisions
    )


def _reference_rows(reference_cache: dict[Path, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path, cached in sorted(reference_cache.items(), key=lambda item: str(item[0])):
        metrics = cached["advisory_metrics"]
        result = cached["advisory"]
        rows.append(
            {
                "scenario_path": path.name,
                "model_warning_count": _warning_count(result),
                "accepted_advisory_observation_count": _accepted_warning_count(result),
                "governor_first_protect_tick": cached["governor_metrics"]["states"][
                    "first_protect_tick"
                ],
                "advisory_governor_first_protect_tick": metrics["states"][
                    "first_protect_tick"
                ],
                "advisory_governor_protect_targets": metrics["lifecycle"][
                    "protect_target_zone_ids"
                ],
                "advisory_governor_invariant_violation_count": metrics[
                    "invariant_violation_count"
                ],
            }
        )
    return rows


def _summarise(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    harmful = [row for row in rows if row["reserve_off_integrated_excess"] > 0.0]
    harmful_gradual = [
        row
        for row in harmful
        if row["fault_class"] == "gradual_primary_fan_degradation"
    ]
    transient = [
        row for row in rows if row["fault_class"] in _TRANSIENT_FAULT_CLASSES
    ]
    healthy_interventions = [
        row
        for row in reference_rows
        if row["advisory_governor_first_protect_tick"] is not None
    ]
    frozen_interventions = [
        row
        for row in rows
        if row["fault_class"] == "frozen_sensor"
        and row["advisory_governor_first_protect_tick"] is not None
    ]
    wrong_target = [
        row
        for row in rows
        if set(row["advisory_governor_protect_targets"]) - {row["target_zone_id"]}
    ]
    missed_harmful = [
        row
        for row in harmful
        if row["advisory_governor_first_protect_tick"] is None
    ]
    worsened_harmful = [
        row
        for row in harmful
        if row["advisory_excess_reduction_vs_governor"] < 0.0
    ]
    earlier_gradual = [
        row
        for row in harmful_gradual
        if row["protection_lead_ticks"] is not None
        and row["protection_lead_ticks"] > 0
    ]
    positive_gradual = [
        row
        for row in harmful_gradual
        if row["advisory_excess_reduction_vs_governor"] > 0.0
    ]
    reduction_fractions = [
        row["advisory_excess_reduction_fraction_vs_governor"]
        for row in harmful_gradual
        if row["advisory_excess_reduction_fraction_vs_governor"] is not None
    ]
    repeated_transient = [
        row for row in transient if row["advisory_governor_protect_entry_count"] > 1
    ]
    recurrence_transient = [
        row
        for row in transient
        if row["advisory_governor_handback_recurrence_count"] > 0
    ]
    timeout_transient = [
        row
        for row in transient
        if row["advisory_governor_handback_timeout_count"] > 0
    ]
    nonzero_transient = [
        row
        for row in transient
        if not row["advisory_governor_final_physical_zero"]
    ]
    summary = {
        "final_families": len(rows),
        "unique_healthy_references": len(reference_rows),
        "harmful_physical_families": len(harmful),
        "harmful_gradual_families": len(harmful_gradual),
        "healthy_reference_interventions": len(healthy_interventions),
        "frozen_sensor_interventions": len(frozen_interventions),
        "wrong_target_interventions": len(wrong_target),
        "invariant_violations": sum(
            row["advisory_governor_invariant_violation_count"] for row in rows
        )
        + sum(
            row["advisory_governor_invariant_violation_count"]
            for row in reference_rows
        ),
        "missed_harmful_physical_families": len(missed_harmful),
        "worsened_harmful_physical_families": len(worsened_harmful),
        "transient_repeated_protection_families": len(repeated_transient),
        "transient_handback_recurrence_families": len(recurrence_transient),
        "transient_handback_timeout_families": len(timeout_transient),
        "transient_nonzero_final_reserve_families": len(nonzero_transient),
        "harmful_gradual_earlier_protection_families": len(earlier_gradual),
        "harmful_gradual_positive_excess_reduction_families": len(
            positive_gradual
        ),
        "median_excess_reduction_fraction_vs_governor_harmful_gradual": (
            statistics.median(reduction_fractions) if reduction_fractions else None
        ),
    }
    diagnostics = {
        "model_warning_windows_fault_arms": sum(
            row["model_warning_count"] for row in rows
        ),
        "model_warning_windows_healthy_references": sum(
            row["model_warning_count"] for row in reference_rows
        ),
        "accepted_advisory_observations_fault_arms": sum(
            row["accepted_advisory_observation_count"] for row in rows
        ),
        "accepted_advisory_observations_healthy_references": sum(
            row["accepted_advisory_observation_count"] for row in reference_rows
        ),
        "median_protection_lead_ticks_when_earlier": (
            statistics.median(row["protection_lead_ticks"] for row in earlier_gradual)
            if earlier_gradual
            else None
        ),
        "healthy_reference_scenarios_with_intervention": [
            row["scenario_path"] for row in healthy_interventions
        ],
        "frozen_sensor_families_with_intervention": [
            row["family_id"] for row in frozen_interventions
        ],
        "wrong_target_family_ids": [row["family_id"] for row in wrong_target],
        "missed_harmful_family_ids": [row["family_id"] for row in missed_harmful],
        "worsened_harmful_family_ids": [
            row["family_id"] for row in worsened_harmful
        ],
        "transient_repeated_protection_family_ids": [
            row["family_id"] for row in repeated_transient
        ],
        "transient_handback_recurrence_family_ids": [
            row["family_id"] for row in recurrence_transient
        ],
        "transient_handback_timeout_family_ids": [
            row["family_id"] for row in timeout_transient
        ],
        "transient_nonzero_final_reserve_family_ids": [
            row["family_id"] for row in nonzero_transient
        ],
        "harmful_gradual_earlier_family_ids": [
            row["family_id"] for row in earlier_gradual
        ],
        "harmful_gradual_positive_reduction_family_ids": [
            row["family_id"] for row in positive_gradual
        ],
    }
    return summary, diagnostics


def _source_snapshot(repo: Path, *, require_clean: bool) -> dict[str, Any]:
    if not repo.is_dir():
        raise ValueError(f"repository directory does not exist: {repo}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if require_clean and status:
        raise ValueError("final evaluation requires a clean source worktree")
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "git_status_porcelain": status.splitlines(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(document: object) -> str:
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_new(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _claim_final_suite(path: Path, document: object) -> None:
    """Irreversibly mark a final corpus consumed before its first simulation."""
    try:
        _write_json_new(path, document)
    except FileExistsError:
        raise FileExistsError(
            f"final corpus has already been claimed for evaluation: {path}"
        ) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--sweep-spec", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--forbidden-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_final_evaluation(
        repo=args.repo,
        sweep_spec_path=args.sweep_spec,
        corpus_dir=args.corpus_dir,
        artifact_path=args.artifact,
        forbidden_manifest_paths=args.forbidden_manifest,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "report_sha256": report["report_sha256"],
                "summary": report["summary"],
                "assessment": report["assessment"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
