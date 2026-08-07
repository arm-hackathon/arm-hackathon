"""Frozen V7 development runner for named-fault escalation research.

V7 is a new lineage after the V6 centroid transfer failure. It keeps the V6
sweep/corpus/evaluation machinery but changes the candidate and the baseline:

- baseline: ``V7EscalatedRulePolicy`` (calibrated named-fault escalation on top
  of the precision-1.0 concern layer);
- candidate: ``V7GatedResidualPolicy`` (residual-vector centroid, only fires
  inside concern windows).

V7 uses a fresh sweep spec whose validation families were never consumed by V6.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict, cast

from aeolus.config import HabitatConfig, load_scenario
from aeolus.corpus_v6 import generate_v6_corpus, validate_v6_corpus
from aeolus.evaluate_v6 import V6EvaluationStream, evaluate_v6
from aeolus.families_v6 import V6FamilyManifest, load_v6_family_manifest
from aeolus.residual_features import ResidualFeatureProjector
from aeolus.scenario import run_scenario
from aeolus.specialists_v7 import (
    V7EscalatedRulePolicy,
    V7EscalationParameters,
    V7GatedResidualPolicy,
    residual_window_vector,
)
from aeolus.sweep_v6 import V6SweepSpec, load_v6_sweep_spec
from aeolus.v7_centroid import V7ResidualCentroid

V7_SWEEP_SCHEMA_VERSION = "aeolus_sweep_v6"
V7_REPORT_VERSION = "aeolus_v7_development_evidence_v1"
V7_WINDOW_TICKS = 10
V7_CONFIDENCE_GRID = (0.0, 0.3, 0.5, 0.7, 0.9)
V7_SETTLED_RESIDUAL_GRID = (0.2, 0.25, 0.3)
V7_SENSOR_TREND_GRID = (0.0025, 0.005, 0.01)
V7_PROXY_GRID = (0.1, 0.15)
V7_SENSOR_DELTA_GRID = (0.02, 0.05)
# A calibration eval is pathological when it is both slow in absolute terms and
# far above the completed-eval median. The absolute floor prevents premature
# skips early in the grid, where the median is small but later combos are
# legitimately heavier (more concern windows trip the centroid).
V7_EVAL_BOUND_MIN_SECONDS = 300.0
V7_EVAL_BOUND_FACTOR = 5.0


@dataclass(frozen=True)
class V7DevelopmentRequest:
    """A source-controlled, non-deployment V7 execution request."""

    sweep_spec_path: Path
    output_dir: Path
    window_ticks: int = V7_WINDOW_TICKS
    authorize_final_suite: bool = False
    authorize_response_integration: bool = False


def validate_v7_development_request(request: V7DevelopmentRequest) -> V6SweepSpec:
    """Load and validate the one authoritative V7 sweep allocation."""
    if not isinstance(request, V7DevelopmentRequest):
        raise ValueError("V7 development request is malformed")
    if request.authorize_final_suite:
        raise ValueError("V7 development cannot authorize a final-suite run")
    if request.authorize_response_integration:
        raise ValueError("V7 development cannot authorize response integration")
    if not isinstance(request.window_ticks, int) or isinstance(request.window_ticks, bool) or request.window_ticks < 1:
        raise ValueError("V7 window_ticks must be a positive integer")
    return load_v6_sweep_spec(request.sweep_spec_path)


def run_v7_development(request: V7DevelopmentRequest) -> dict[str, object]:
    """Generate, fit, calibrate, validate, and receipt one V7 development run."""
    spec = validate_v7_development_request(request)
    repository = Path(__file__).resolve().parents[2]
    provenance = _clean_source_provenance(repository)
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sweep_dir = output / "sweep"
    corpus_dir = output / "corpus"
    sweep_receipt = _generate_sweep(request.sweep_spec_path, sweep_dir)
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
        raise ValueError("V7 runner requires non-empty fit, calibration, and validation rows")
    reference_config = load_scenario(families.families[0].reference_path)

    streams = {
        role: _build_streams(families, role, rows)
        for role in ("calibration", "validation")
    }

    fit_vectors, fit_labels, fit_roles = _fit_vectors(families, split_rows["fit"], request.window_ticks)
    classifier = V7ResidualCentroid.fit(
        fit_vectors, fit_labels, feature_width=len(fit_vectors[0])
    )

    calibration = _calibrate(
        classifier,
        reference_config,
        streams["calibration"],
        request.window_ticks,
        progress_path=output / "calibration-progress.log",
    )

    baseline_policy = V7EscalatedRulePolicy(
        reference_config, _selected_parameters(calibration)
    )
    baseline_metrics = {
        role: evaluate_v6(streams[role], baseline_policy, window_ticks=request.window_ticks)
        for role in streams
    }
    candidate_policy = V7GatedResidualPolicy(
        reference_config,
        classifier,
        min_confidence=float(calibration["selected_min_confidence"]),
        parameters=_selected_parameters(calibration),
    )
    candidate_metrics = evaluate_v6(
        streams["validation"], candidate_policy, window_ticks=request.window_ticks
    )
    acceptance = _development_acceptance(candidate_metrics, baseline_metrics["validation"])
    model_path = output / "v7-residual-centroid.json"
    model_path.write_text(
        json.dumps(classifier.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "schema_version": V7_REPORT_VERSION,
        "evidence_role": "development_only",
        "source_provenance": provenance,
        "sweep": sweep_receipt,
        "family_manifest_sha256": families.manifest_sha256,
        "corpus_manifest": corpus_manifest,
        "window_ticks": request.window_ticks,
        "rows_by_role": {role: len(role_rows) for role, role_rows in split_rows.items()},
        "families_by_role": dict(families.families_by_role),
        "fit_vectors": {"count": len(fit_vectors), "width": len(fit_vectors[0])},
        "baseline": {
            "name": "escalated_rule_v7",
            "parameters": calibration["selected_parameters"],
            "calibration_metrics": baseline_metrics["calibration"],
            "validation_metrics": baseline_metrics["validation"],
        },
        "candidate": {
            "name": "gated_residual_centroid_v1",
            "artifact": {
                "path": model_path.name,
                "sha256": _sha256_file(model_path),
                "bytes": model_path.stat().st_size,
            },
            "calibration": calibration,
            "validation_metrics": candidate_metrics,
            "development_acceptance": acceptance,
        },
        "selected_candidate": "gated_residual_centroid_v1" if acceptance["passed"] else None,
        "retained_method": "gated_residual_centroid_v1" if acceptance["passed"] else "escalated_rule_v7",
        "development_gate_passed": acceptance["passed"],
        "final_suite_generation_authorized": False,
        "response_layer_integration_authorized": False,
        "optimization_boundary": {
            "hardware_benchmark_claim": False,
            "onnx_export_claim": False,
            "reason": "V7 development result is a simulator-only methodology gate, not deployment evidence",
        },
        "environment": {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    report_path = output / "v7-development-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _generate_sweep(spec_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Generate a V6-schema sweep into an empty directory (importable wrapper)."""
    from aeolus.sweep_v6 import generate_v6_sweep

    return generate_v6_sweep(spec_path, output_dir)


def _build_streams(
    families: V6FamilyManifest, role: str, rows: Sequence[dict[str, object]]
) -> list[V6EvaluationStream]:
    """Build paired reference/fault replay streams for one role."""
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
            raise ValueError("V7 corpus has no observable onset for a requested fault family")
        streams.extend(
            (
                V6EvaluationStream(
                    family_id=family.family_id,
                    room_family_id=family.room_family_id,
                    split=role,
                    scenario_role="reference",
                    records=tuple(run_scenario(load_scenario(family.reference_path))),
                    reference_identity=family.reference_scenario_sha256,
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
        raise ValueError(f"V7 runner has no {role} replay streams")
    return streams


def _fit_vectors(
    families: V6FamilyManifest,
    rows: Sequence[dict[str, object]],
    window_ticks: int,
) -> tuple[list[list[float]], list[str], list[str]]:
    """Build labelled residual vectors from fit rows by replaying their scenarios."""
    vectors: list[list[float]] = []
    labels: list[str] = []
    roles: list[str] = []
    projector_cache: dict[str, ResidualFeatureProjector] = {}
    for row in rows:
        label = str(row["label"])
        if label == "excluded_transition":
            continue
        family_id = str(row["family_id"])
        scenario_role = str(row["scenario_role"])
        start_tick = int(row["start_tick"])
        end_tick = int(row["end_tick"])
        family = next(item for item in families.families if item.family_id == family_id)
        scenario_path = family.fault_path if scenario_role == "fault" else family.reference_path
        key = str(scenario_path)
        if key not in projector_cache:
            config = load_scenario(scenario_path)
            projector_cache[key] = ResidualFeatureProjector(config)
        projector = projector_cache[key]
        config = load_scenario(scenario_path)
        records = tuple(run_scenario(load_scenario(scenario_path)))
        window = tuple(record for record in records if start_tick <= record.tick <= end_tick)
        if len(window) != window_ticks:
            raise ValueError("V7 fit vector window tick span is malformed")
        zone_ids = tuple(zone.id for zone in config.non_processing_zones())
        vectors.append(residual_window_vector(projector, window, zone_ids).tolist())
        labels.append(label)
        roles.append(scenario_role)
    if not vectors:
        raise ValueError("V7 runner has no usable fit vectors")
    return vectors, labels, roles


def _selected_parameters(calibration: dict[str, object]) -> V7EscalationParameters:
    """Rebuild immutable escalation parameters from a calibration record."""
    raw = calibration["selected_parameters"]
    if not isinstance(raw, dict):
        raise ValueError("V7 calibration selected_parameters must be a dict")
    return V7EscalationParameters(
        sensor_trend_abs_max=float(raw["sensor_trend_abs_max"]),
        expected_change_proxy=float(raw["expected_change_proxy"]),
        sensor_max_delta=float(raw["sensor_max_delta"]),
        settled_residual_threshold=float(raw["settled_residual_threshold"]),
    )


def _append_progress(progress_path: Path | None, line: str) -> None:
    """Append one flushed progress line so long calibration is observable."""
    if progress_path is None:
        return
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _eval_exceeds_bound(elapsed_seconds: float, completed: Sequence[float]) -> bool:
    """Flag an eval that is both slow in absolute terms and far above the median."""
    if elapsed_seconds < V7_EVAL_BOUND_MIN_SECONDS or not completed:
        return False
    ordered = sorted(completed)
    median_elapsed = ordered[len(ordered) // 2]
    return elapsed_seconds > V7_EVAL_BOUND_FACTOR * median_elapsed


def _progress_line(
    eval_index: int,
    total_evals: int,
    role: str,
    settled: float,
    trend: float,
    proxy: float,
    delta: float,
    confidence: float | None,
    elapsed_list: Sequence[float],
    skipped: bool = False,
) -> str:
    """Render one human-readable calibration progress record."""
    ordered = sorted(elapsed_list)
    median_elapsed = ordered[len(ordered) // 2] if ordered else 0.0
    tag = "SKIPPED " if skipped else ""
    selector = "baseline" if confidence is None else f"confidence={confidence}"
    return (
        f"{tag}eval={eval_index}/{total_evals} role={role} {selector} "
        f"settled={settled} trend={trend} proxy={proxy} delta={delta} "
        f"elapsed_s={elapsed_list[-1]:.1f} median_s={median_elapsed:.1f}"
    )


class _ProgressRecord(TypedDict):
    """One completed calibration eval, rendered by the parent into the log."""

    role: str
    confidence: float | None
    elapsed_s: float
    skipped: bool


_CalibrationCandidate = tuple[
    tuple[float, float, float, float],
    dict[str, object],
    V7EscalationParameters,
    float,
    tuple[float, float, float, float],
]
_ComboResult = tuple[
    tuple[float, float, float, float],
    list[_CalibrationCandidate],
    list[_ProgressRecord],
    int,
]

_WORKER_CLASSIFIER: V7ResidualCentroid | None = None
_WORKER_REFERENCE_CONFIG: HabitatConfig | None = None
_WORKER_STREAMS: Sequence[V6EvaluationStream] | None = None
_WORKER_WINDOW_TICKS: int = V7_WINDOW_TICKS


def _init_worker(
    classifier: V7ResidualCentroid | None,
    reference_config: HabitatConfig | None,
    streams: Sequence[V6EvaluationStream] | None,
    window_ticks: int,
) -> None:
    """Populate the per-process evaluation context for pool workers."""
    global _WORKER_CLASSIFIER, _WORKER_REFERENCE_CONFIG, _WORKER_STREAMS, _WORKER_WINDOW_TICKS
    _WORKER_CLASSIFIER = classifier
    _WORKER_REFERENCE_CONFIG = reference_config
    _WORKER_STREAMS = streams
    _WORKER_WINDOW_TICKS = window_ticks


def _calibrate_combo_task(
    settled: float,
    trend: float,
    proxy: float,
    delta: float,
) -> _ComboResult:
    """Evaluate one escalation parameter combo: 1 baseline + up to 5 confidence evals.

    Runs inside a pool worker, or directly in the parent when the pool is
    unavailable. The pathological-eval bound is applied against the combo's own
    completed evals so one bad setting cannot consume unbounded wall time.
    """
    if _WORKER_CLASSIFIER is None or _WORKER_REFERENCE_CONFIG is None or _WORKER_STREAMS is None:
        raise ValueError("V7 calibration worker context is not initialized")
    parameters = V7EscalationParameters(
        sensor_trend_abs_max=trend,
        expected_change_proxy=proxy,
        sensor_max_delta=delta,
        settled_residual_threshold=settled,
    )
    baseline = V7EscalatedRulePolicy(_WORKER_REFERENCE_CONFIG, parameters)
    started = time.monotonic()
    baseline_metrics = evaluate_v6(_WORKER_STREAMS, baseline, window_ticks=_WORKER_WINDOW_TICKS)
    combo_elapsed = [time.monotonic() - started]
    progress: list[_ProgressRecord] = [
        {"role": "baseline", "confidence": None, "elapsed_s": combo_elapsed[0], "skipped": False}
    ]
    candidates: list[_CalibrationCandidate] = []
    pathological = 0
    for confidence in V7_CONFIDENCE_GRID:
        policy = V7GatedResidualPolicy(
            _WORKER_REFERENCE_CONFIG,
            _WORKER_CLASSIFIER,
            min_confidence=confidence,
            parameters=parameters,
        )
        started = time.monotonic()
        metrics = evaluate_v6(_WORKER_STREAMS, policy, window_ticks=_WORKER_WINDOW_TICKS)
        elapsed = time.monotonic() - started
        combo_elapsed.append(elapsed)
        candidates.append(
            (
                _selection_key(metrics, baseline_metrics),
                metrics,
                parameters,
                confidence,
                (settled, trend, proxy, delta),
            )
        )
        progress.append(
            {
                "role": "candidate",
                "confidence": confidence,
                "elapsed_s": elapsed,
                "skipped": False,
            }
        )
        if _eval_exceeds_bound(elapsed, combo_elapsed):
            pathological += 1
            progress[-1]["skipped"] = True
            break
    return (settled, trend, proxy, delta), candidates, progress, pathological


def _calibration_worker_count(grid_size: int) -> int:
    """Choose pool workers, bounded for Windows spawn startup cost.

    Windows cannot fork, so every worker re-imports numpy and receives the
    evaluation streams over IPC; each worker costs seconds to minutes to boot.
    Default to 12 workers unless the operator overrides with
    ``AEOLUS_CALIBRATION_WORKERS``.
    """
    raw = os.environ.get("AEOLUS_CALIBRATION_WORKERS", "")
    if raw.isdigit() and int(raw) > 0:
        return min(int(raw), grid_size)
    return min(12, os.cpu_count() or 1, grid_size)


def _run_calibration_grid(
    grid: Sequence[tuple[float, float, float, float]],
    classifier: V7ResidualCentroid,
    reference_config: object,
    streams: Sequence[V6EvaluationStream],
    window_ticks: int,
    workers: int,
) -> tuple[list[_ComboResult], bool]:
    """Evaluate every combo, preferring a process pool with sequential fallback.

    The 36 parameter combos are independent, so they map cleanly onto a pool of
    ``workers`` processes. Pool-creation failures (unpicklable init args or
    platform limits) degrade to in-process sequential evaluation; task errors
    always propagate so a failed eval is never silently replaced.
    """
    try:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                classifier,
                cast(HabitatConfig | None, reference_config),
                streams,
                window_ticks,
            ),
        )
    except Exception:
        _init_worker(
            classifier,
            cast(HabitatConfig | None, reference_config),
            streams,
            window_ticks,
        )
        return [(_calibrate_combo_task(*combo)) for combo in grid], False
    futures = [pool.submit(_calibrate_combo_task, *combo) for combo in grid]
    results: list[_ComboResult] = []
    try:
        for future in as_completed(futures):
            results.append(future.result())
    except Exception:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results, True


def _calibrate(
    classifier: V7ResidualCentroid,
    reference_config: object,
    streams: Sequence[V6EvaluationStream],
    window_ticks: int,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Select escalation parameters and centroid confidence on calibration only."""
    grid = [
        (settled, trend, proxy, delta)
        for settled in V7_SETTLED_RESIDUAL_GRID
        for trend in V7_SENSOR_TREND_GRID
        for proxy in V7_PROXY_GRID
        for delta in V7_SENSOR_DELTA_GRID
    ]
    total_evals = len(grid) * (1 + len(V7_CONFIDENCE_GRID))
    workers = _calibration_worker_count(len(grid))
    _append_progress(
        progress_path,
        f"calibration_start grid_combos={len(grid)} total_evals={total_evals} "
        f"workers={workers}",
    )
    combo_results, parallel = _run_calibration_grid(
        grid, classifier, reference_config, streams, window_ticks, workers
    )
    candidates: list[_CalibrationCandidate] = []
    eval_index = 0
    elapsed_list: list[float] = []
    skipped = 0
    for combo, combo_candidates, progress, combo_pathological in combo_results:
        candidates.extend(combo_candidates)
        skipped += combo_pathological
        settled, trend, proxy, delta = combo
        for record in progress:
            eval_index += 1
            elapsed_list.append(float(record["elapsed_s"]))
            _append_progress(
                progress_path,
                _progress_line(
                    eval_index,
                    total_evals,
                    str(record["role"]),
                    settled,
                    trend,
                    proxy,
                    delta,
                    record["confidence"],
                    elapsed_list,
                    skipped=bool(record["skipped"]),
                ),
            )
    if not candidates:
        raise ValueError("V7 calibration produced no candidates")
    # Deterministic selection: rank by the selection key, then by the parameter
    # combo and confidence, so tie-breaks cannot depend on pool completion order.
    _, selected_metrics, selected_parameters, selected_confidence, _ = min(
        candidates, key=lambda item: (item[0], item[4], item[3])
    )
    baseline_metrics = evaluate_v6(
        streams, V7EscalatedRulePolicy(reference_config, selected_parameters), window_ticks=window_ticks
    )
    return {
        "selection_role": "calibration",
        "confidence_grid": list(V7_CONFIDENCE_GRID),
        "settled_residual_grid": list(V7_SETTLED_RESIDUAL_GRID),
        "sensor_trend_grid": list(V7_SENSOR_TREND_GRID),
        "proxy_grid": list(V7_PROXY_GRID),
        "sensor_delta_grid": list(V7_SENSOR_DELTA_GRID),
        "selected_min_confidence": selected_confidence,
        "selected_parameters": {
            "sensor_trend_abs_max": selected_parameters.sensor_trend_abs_max,
            "expected_change_proxy": selected_parameters.expected_change_proxy,
            "sensor_max_delta": selected_parameters.sensor_max_delta,
            "settled_residual_threshold": selected_parameters.settled_residual_threshold,
        },
        "selected_metrics": selected_metrics,
        "baseline_named_fault_macro_f1": baseline_metrics["named_fault_macro_f1"],
        "candidate_count": len(candidates),
        "calibration_evals": eval_index,
        "calibration_pathological_evals": skipped,
        "calibration_parallel": parallel,
    }


def _selection_key(
    metrics: dict[str, object], baseline_metrics: dict[str, object]
) -> tuple[float, float, float, float]:
    """Rank candidates: macro-F1 gain, then alarm burden, then uncertainty."""
    return (
        -float(metrics["named_fault_macro_f1"]),
        float(metrics["healthy_alert_episodes_per_1000_ticks"]),
        float(metrics["post_onset_uncertainty_fraction"]),
        -float(metrics["named_fault_macro_f1"]),
    )


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
        "criterion": "strict named-fault macro-F1 gain over escalated rules; healthy alert episodes <= 10 per 1,000 ticks and <= 2 above rules; named detection recall non-regression",
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
        raise ValueError(f"cannot establish V7 source provenance: {exc}") from None
    if status:
        raise ValueError("V7 canonical development run requires a clean Git worktree")
    paths = (
        "scenarios/sweep-v7-development.json",
        "src/aeolus/sweep_v6.py",
        "src/aeolus/families_v6.py",
        "src/aeolus/corpus_v6.py",
        "src/aeolus/evaluate_v6.py",
        "src/aeolus/specialists.py",
        "src/aeolus/specialists_v7.py",
        "src/aeolus/residual_features.py",
        "src/aeolus/v7_centroid.py",
        "src/aeolus/model_cycle_v7.py",
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
