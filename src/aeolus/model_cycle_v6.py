"""Frozen V6 development runner for conditional specialist research."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from aeolus.config import load_scenario
from aeolus.corpus_v6 import generate_v6_corpus, validate_v6_corpus
from aeolus.evaluate_v6 import V6EvaluationStream, evaluate_v6
from aeolus.families_v6 import V6FamilyManifest, load_v6_family_manifest
from aeolus.observable_context import build_observable_context_contract, observable_context_v1
from aeolus.scenario import run_scenario
from aeolus.specialists import V6DecisionPolicy
from aeolus.sweep_v6 import V6SweepSpec, generate_v6_sweep, load_v6_sweep_spec
from aeolus.trace import TickRecord
from aeolus.v6_centroid import V6CentroidClassifier

V6_SWEEP_SCHEMA_VERSION = "aeolus_sweep_v6"
V6_REPORT_VERSION = "aeolus_v6_development_evidence_v1"
V6_WINDOW_TICKS = 10
V6_CONFIDENCE_GRID = (0.0, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class V6DevelopmentRequest:
    """A source-controlled, non-deployment V6 execution request."""

    sweep_spec_path: Path
    output_dir: Path
    window_ticks: int = V6_WINDOW_TICKS
    authorize_final_suite: bool = False
    authorize_response_integration: bool = False


class V6CentroidPolicy:
    """Replay a fit-only V6 centroid model using observable TickRecord windows."""

    def __init__(
        self,
        classifier: V6CentroidClassifier,
        reference_config: object,
        *,
        min_confidence: float,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("V6 policy minimum confidence must be in [0, 1]")
        self._classifier = classifier
        self._contract = build_observable_context_contract(reference_config)
        self._min_confidence = min_confidence

    def reset(self) -> None:
        """Centroid inference retains no replay state."""

    def label_window(self, records: Sequence[TickRecord]) -> str:
        """Emit a named class only when calibrated confidence is sufficient."""
        features = [observable_context_v1(record, self._contract).tolist() for record in records]
        probabilities = self._classifier.predict_probabilities([features])[0]
        index = int(np.argmax(probabilities))
        if float(probabilities[index]) < self._min_confidence:
            return "uncertain"
        return self._classifier.class_names[index]


def validate_v6_development_request(request: V6DevelopmentRequest) -> V6SweepSpec:
    """Load and validate the one authoritative V6 sweep allocation."""
    if not isinstance(request, V6DevelopmentRequest):
        raise ValueError("V6 development request is malformed")
    if request.authorize_final_suite:
        raise ValueError("V6 development cannot authorize a final-suite run")
    if request.authorize_response_integration:
        raise ValueError("V6 development cannot authorize response integration")
    if not isinstance(request.sweep_spec_path, Path):
        raise ValueError("V6 sweep path must be a pathlib Path")
    if not isinstance(request.output_dir, Path):
        raise ValueError("V6 output directory must be a pathlib Path")
    if request.output_dir.exists() and any(request.output_dir.iterdir()):
        raise ValueError("V6 output directory must be empty")
    if not isinstance(request.window_ticks, int) or isinstance(request.window_ticks, bool) or request.window_ticks < 2:
        raise ValueError("V6 window_ticks must be an integer of at least two")
    return load_v6_sweep_spec(request.sweep_spec_path)


def run_v6_development(request: V6DevelopmentRequest) -> dict[str, object]:
    """Generate, fit, calibrate, validate, and receipt one V6 development run."""
    spec = validate_v6_development_request(request)
    repository = Path(__file__).resolve().parents[2]
    provenance = _clean_source_provenance(repository)
    output = request.output_dir
    output.mkdir(parents=True, exist_ok=True)

    sweep_dir = output / "sweep"
    corpus_dir = output / "corpus"
    sweep_receipt = generate_v6_sweep(request.sweep_spec_path, sweep_dir)
    families = load_v6_family_manifest(sweep_dir / "families-v6.json", expected_sweep=spec)
    corpus_manifest = generate_v6_corpus(
        families,
        corpus_dir,
        window_ticks=request.window_ticks,
        stride_ticks=1,
    )
    rows = validate_v6_corpus(corpus_dir, expected_families=families)
    split_rows = {
        role: [row for row in rows if row["split"] == role]
        for role in ("fit", "calibration", "validation")
    }
    if any(not split_rows[role] for role in split_rows):
        raise ValueError("V6 runner requires non-empty fit, calibration, and validation rows")
    feature_width = len(split_rows["fit"][0]["features"][0])
    classifier = V6CentroidClassifier.fit(
        split_rows["fit"], window_ticks=request.window_ticks, feature_width=feature_width
    )
    reference_config = load_scenario(families.families[0].reference_path)
    streams = {
        role: _build_streams(families, role, rows)
        for role in ("calibration", "validation")
    }
    baseline = V6DecisionPolicy(reference_config)
    baseline_metrics = {
        role: evaluate_v6(streams[role], baseline, window_ticks=request.window_ticks)
        for role in streams
    }
    calibration = _calibrate_centroid(
        classifier,
        reference_config,
        streams["calibration"],
        baseline_metrics["calibration"],
        request.window_ticks,
    )
    candidate_policy = V6CentroidPolicy(
        classifier, reference_config, min_confidence=float(calibration["selected_min_confidence"])
    )
    candidate_metrics = evaluate_v6(
        streams["validation"], candidate_policy, window_ticks=request.window_ticks
    )
    acceptance = _development_acceptance(candidate_metrics, baseline_metrics["validation"])
    model_path = output / "v6-centroid-model.json"
    model_path.write_text(
        json.dumps(classifier.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "schema_version": V6_REPORT_VERSION,
        "evidence_role": "development_only",
        "source_provenance": provenance,
        "sweep": sweep_receipt,
        "family_manifest_sha256": families.manifest_sha256,
        "corpus_manifest": corpus_manifest,
        "window_ticks": request.window_ticks,
        "rows_by_role": {role: len(role_rows) for role, role_rows in split_rows.items()},
        "families_by_role": dict(families.families_by_role),
        "baseline": {
            "name": "conditional_rule_v6",
            "calibration_metrics": baseline_metrics["calibration"],
            "validation_metrics": baseline_metrics["validation"],
        },
        "candidate": {
            "name": "observable_context_centroid_v1",
            "artifact": {
                "path": model_path.name,
                "sha256": _sha256_file(model_path),
                "bytes": model_path.stat().st_size,
            },
            "calibration": calibration,
            "validation_metrics": candidate_metrics,
            "development_acceptance": acceptance,
        },
        "selected_candidate": "observable_context_centroid_v1" if acceptance["passed"] else None,
        "retained_method": "observable_context_centroid_v1" if acceptance["passed"] else "conditional_rule_v6",
        "development_gate_passed": acceptance["passed"],
        "final_suite_generation_authorized": False,
        "response_layer_integration_authorized": False,
        "optimization_boundary": {
            "hardware_benchmark_claim": False,
            "onnx_export_claim": False,
            "reason": "V6 development result is a simulator-only methodology gate, not deployment evidence",
        },
        "environment": {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    report_path = output / "v6-development-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _build_streams(
    families: V6FamilyManifest, role: str, rows: Sequence[dict[str, object]]
) -> list[V6EvaluationStream]:
    onset_by_family: dict[str, int] = {}
    for row in rows:
        if row["split"] == role and row["scenario_role"] == "fault":
            onset_by_family.setdefault(str(row["family_id"]), int(row["observable_onset_tick"]))
    streams: list[V6EvaluationStream] = []
    for family in families.families:
        if family.role != role:
            continue
        onset = onset_by_family.get(family.family_id)
        if onset is None:
            raise ValueError("V6 corpus has no observable onset for a requested fault family")
        streams.extend(
            (
                V6EvaluationStream(
                    family_id=family.family_id,
                    room_family_id=family.room_family_id,
                    split=role,
                    scenario_role="reference",
                    records=tuple(run_scenario(load_scenario(family.reference_path))),
                ),
                V6EvaluationStream(
                    family_id=family.family_id,
                    room_family_id=family.room_family_id,
                    split=role,
                    scenario_role="fault",
                    records=tuple(run_scenario(load_scenario(family.fault_path))),
                    fault_class=family.fault_class,
                    observable_onset_tick=onset,
                ),
            )
        )
    if not streams:
        raise ValueError(f"V6 runner has no {role} replay streams")
    return streams


def _calibrate_centroid(
    classifier: V6CentroidClassifier,
    reference_config: object,
    streams: Sequence[V6EvaluationStream],
    baseline_metrics: dict[str, object],
    window_ticks: int,
) -> dict[str, object]:
    candidates: list[tuple[float, dict[str, object]]] = []
    for threshold in V6_CONFIDENCE_GRID:
        metrics = evaluate_v6(
            streams,
            V6CentroidPolicy(classifier, reference_config, min_confidence=threshold),
            window_ticks=window_ticks,
        )
        candidates.append((threshold, metrics))
    selected_threshold, selected_metrics = min(
        candidates,
        key=lambda item: (
            -float(item[1]["named_fault_macro_f1"]),
            float(item[1]["healthy_alert_episodes_per_1000_ticks"]),
            float(item[1]["post_onset_uncertainty_fraction"]),
            -item[0],
        ),
    )
    return {
        "selection_role": "calibration",
        "baseline_named_fault_macro_f1": baseline_metrics["named_fault_macro_f1"],
        "confidence_grid": list(V6_CONFIDENCE_GRID),
        "selected_min_confidence": selected_threshold,
        "selected_metrics": selected_metrics,
        "candidates": [
            {"min_confidence": threshold, "metrics": metrics}
            for threshold, metrics in candidates
        ],
    }


def _development_acceptance(
    candidate: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    candidate_macro = float(candidate["named_fault_macro_f1"])
    baseline_macro = float(baseline["named_fault_macro_f1"])
    candidate_alerts = float(candidate["healthy_alert_episodes_per_1000_ticks"])
    baseline_alerts = float(baseline["healthy_alert_episodes_per_1000_ticks"])
    candidate_recall = float(candidate["named_detection_recall"])
    baseline_recall = float(baseline["named_detection_recall"])
    macro_win = candidate_macro > baseline_macro + 1e-12
    healthy_safe = candidate_alerts <= 10.0 and candidate_alerts <= baseline_alerts + 2.0
    detection_non_regression = candidate_recall + 1e-12 >= baseline_recall
    return {
        "criterion": "strict named-fault macro-F1 gain over conditional rules; healthy alert episodes <= 10 per 1,000 ticks and <= 2 above rules; named detection recall non-regression",
        "passed": macro_win and healthy_safe and detection_non_regression,
        "named_fault_macro_f1_gain": candidate_macro - baseline_macro,
        "healthy_alert_episodes_per_1000_ticks_delta": candidate_alerts - baseline_alerts,
        "named_detection_recall_delta": candidate_recall - baseline_recall,
        "macro_win": macro_win,
        "healthy_safe": healthy_safe,
        "detection_non_regression": detection_non_regression,
    }


def _clean_source_provenance(repository: Path) -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot establish V6 source provenance: {exc}") from None
    if status:
        raise ValueError("V6 canonical development run requires a clean Git worktree")
    paths = (
        "scenarios/sweep-v6-development.json",
        "src/aeolus/sweep_v6.py",
        "src/aeolus/families_v6.py",
        "src/aeolus/corpus_v6.py",
        "src/aeolus/evaluate_v6.py",
        "src/aeolus/specialists.py",
        "src/aeolus/v6_centroid.py",
        "src/aeolus/model_cycle_v6.py",
    )
    source_files = {path: _sha256_file(repository / path) for path in paths}
    canonical = json.dumps(source_files, sort_keys=True, separators=(",", ":"))
    return {
        "head_commit": head,
        "worktree_dirty": False,
        "source_files_sha256": source_files,
        "source_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
