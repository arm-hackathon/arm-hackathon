#!/usr/bin/env python3
"""Run the authorized Issue #56 V4 development model study.

The runner consumes only a verified development corpus.  It never reads the
protected final suite, proposes commands, advances the plant, or changes HMC
authority.  Model artifacts are written only after calibration and a strict
reload round trip.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import (
    ACTION_COUNT,
    v2_decision_steps,
)
from aeolus.habitat_v2.forecast_issue55_race import build_family_scenario, deterministic_family_ids
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
    load_v4_samples,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_features import HISTORY_FEATURE_COUNT
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model import (
    Issue56V4ModelError,
    V4_ACTION_IDS,
    V4_MODEL_CANDIDATES,
    V4_MODEL_PROVENANCE_FIELDS,
    V4_MODEL_SEEDS,
    V4ModelSample,
    V4_UTILITY_WEIGHTS,
    V4RiskModel,
    load_v4_model,
    write_v4_model,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
    ISSUE56_V4_MODEL_PROTOCOL_ID,
    V4_MODEL_SPLIT_COUNTS,
    load_v4_model_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (REPO_ROOT / "out").resolve()
VERIFY_MODULE = "scripts.verify_action_risk_v4_corpus"
MODEL_SOURCE_PATHS = (
    Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v2.json"),
    Path("src/aeolus/habitat_v2/forecast/contracts.py"),
    Path("src/aeolus/habitat_v2/forecast/projection.py"),
    Path("src/aeolus/habitat_v2/forecast_issue55_race.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v2.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v3.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_corpus.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_features.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_model.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_model_protocol.py"),
    Path("src/aeolus/habitat_v2/control_trace.py"),
    Path("src/aeolus/habitat_v2/hmc.py"),
    Path("src/aeolus/habitat_v2/physics.py"),
    Path("scripts/run_action_risk_v4_model.py"),
    Path("scripts/verify_action_risk_v4_corpus.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


class V4ModelRunError(RuntimeError):
    """Raised when a V4 model study cannot preserve its declared boundary."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: object) -> str:
    try:
        return _sha_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise V4ModelRunError("V4 model value is not canonical JSON") from error


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V4ModelRunError(f"cannot strictly read JSON: {path}") from error
    if type(value) is not dict:
        raise V4ModelRunError(f"JSON root must be an object: {path}")
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise V4ModelRunError(f"cannot read V4 samples: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise V4ModelRunError(f"invalid V4 sample JSON at {path}:{line_number}") from error
        if type(value) is not dict:
            raise V4ModelRunError(f"V4 sample row is not an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise V4ModelRunError("V4 samples.jsonl is empty")
    return rows


def _write_json(path: Path, value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")
        path.write_bytes(payload)
    except (OSError, TypeError, ValueError) as error:
        raise V4ModelRunError(f"cannot write JSON artifact: {path}") from error
    return _sha_bytes(payload)


def _source_identity() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in MODEL_SOURCE_PATHS:
        path = REPO_ROOT / relative
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise V4ModelRunError(f"cannot read model source file: {relative}") from error
        files.append(
            {
                "relative_path": relative.as_posix(),
                "byte_length": len(raw),
                "sha256": _sha_bytes(raw),
            }
        )
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise V4ModelRunError("V4 model source identity cannot be read from Git") from error
    return {
        "schema_version": "issue56-v4-model-source-identity-v1",
        "source_commit": source_commit,
        "source_worktree_dirty": dirty,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "source_files": files,
    }


def _resolve_corpus(path: Path) -> Path:
    resolved = path.resolve()
    if OUT_ROOT not in resolved.parents or not resolved.is_dir():
        raise V4ModelRunError("V4 model corpus must be an existing directory below repository out/")
    return resolved


def _resolve_output(path: Path) -> Path:
    resolved = path.resolve()
    if OUT_ROOT not in resolved.parents:
        raise V4ModelRunError("V4 model output must be below repository out/")
    if resolved.exists():
        raise V4ModelRunError("V4 model output directory must be new and write-once")
    resolved.mkdir(parents=True, exist_ok=False)
    (resolved / "models").mkdir()
    return resolved


def _independent_verify(corpus: Path) -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", VERIFY_MODULE, "--corpus", str(corpus)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else ""
        raise V4ModelRunError(f"independent V4 corpus verification failed: {detail}") from error


def _load_verified_model_samples(
    rows: Sequence[Mapping[str, Any]],
    corpus: Path,
    bundle: Any,
    scenarios: Mapping[str, Any],
    family_ids: Sequence[str],
) -> tuple[V4ModelSample, ...]:
    """Verify and retain only trace-free rows needed by model fitting."""

    rows_by_family: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        base_sample = row.get("base_sample")
        family_id = base_sample.get("family_id") if type(base_sample) is dict else None
        rows_by_family[family_id].append(row)
    if set(rows_by_family) != set(family_ids):
        raise V4ModelRunError("V4 model sample family coverage is incomplete")

    samples: list[V4ModelSample] = []
    for family_id in family_ids:
        try:
            verified = load_v4_samples(
                rows_by_family[family_id],
                corpus,
                bundle,
                {family_id: scenarios[family_id]},
            )
        except Exception as error:
            raise V4ModelRunError(
                f"V4 corpus family reload failed after independent verification: {family_id}"
            ) from error
        samples.extend(V4ModelSample.from_verified(sample) for sample in verified)
        del verified
        gc.collect()
    if len(samples) != len(rows):
        raise V4ModelRunError("V4 model sample count differs after family reload")
    return tuple(samples)


def _condition_group(family_id: str) -> str:
    roster = deterministic_family_ids(32)
    try:
        index = roster.index(family_id)
    except ValueError as error:
        raise V4ModelRunError("V4 evaluation family is outside the frozen roster") from error
    return f"condition-group-{index // 2:04d}"


@lru_cache(maxsize=8)
def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    if repetitions < 1 or count < 1:
        raise V4ModelRunError("V4 bootstrap dimensions must be positive")
    result = np.empty((repetitions, count), dtype=np.int64)
    for replicate in range(repetitions):
        for draw in range(count):
            digest = hashlib.sha256(
                f"issue56-v4-model-bootstrap-v1|{seed}|{replicate}|{draw}".encode("ascii")
            ).digest()
            result[replicate, draw] = int.from_bytes(digest[:8], "big") % count
    return result


def _bootstrap_mean(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise V4ModelRunError("V4 bootstrap values are malformed")
    draws = np.mean(array[_bootstrap_indices(560057, 10_000, len(array))], axis=1)
    return {
        "point": float(np.mean(array)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "resamples": 10_000,
        "seed": 560057,
        "unit": "condition_group",
    }


def _group_bootstrap(values: Sequence[float]) -> dict[str, Any]:
    """Return a group bootstrap or an explicit small-smoke-support result."""

    if len(values) < 2:
        return {
            "point": float(np.mean(np.asarray(values, dtype=np.float64))) if values else None,
            "ci_lower": None,
            "ci_upper": None,
            "resamples": 10_000,
            "seed": 560057,
            "unit": "condition_group",
            "status": "INSUFFICIENT_CONDITION_GROUP_SUPPORT",
            "group_count": len(values),
        }
    return _bootstrap_mean(values)


def _calibration_error(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    if (
        probability_array.ndim != 1
        or label_array.shape != probability_array.shape
        or not len(probability_array)
        or not np.isfinite(probability_array).all()
        or not np.isfinite(label_array).all()
    ):
        raise V4ModelRunError("V4 calibration metric inputs are malformed")
    error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probability_array >= lower) & (
            probability_array < upper if upper < 1.0 else probability_array <= upper
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probability_array[mask].mean()) - float(label_array[mask].mean())
            )
    return error


def _all_numeric_metrics_finite(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_metrics_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_all_numeric_metrics_finite(item) for item in value)
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _gate_result(
    value: object,
    threshold: float | int,
    *,
    minimum: bool,
) -> dict[str, Any]:
    if value is None:
        return {
            "status": "UNDEFINED",
            "value": None,
            "threshold": threshold,
        }
    numeric = float(value)
    passed = numeric >= threshold if minimum else numeric <= threshold
    return {
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def _evaluation_gate_status(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "authority_violations": _gate_result(
            metrics["authority_violation_count"], 0, minimum=False
        ),
        "replay_failures": _gate_result(metrics["replay_failure_count"], 0, minimum=False),
        "provenance_violations": _gate_result(
            metrics["provenance_violation_count"], 0, minimum=False
        ),
        "non_finite_metrics": _gate_result(
            metrics["non_finite_metric_count"], 0, minimum=False
        ),
        "proposal_admission_failures": _gate_result(
            metrics["proposal_admission_failure_count"], 0, minimum=False
        ),
        "minimum_useful_action_count": _gate_result(
            metrics["useful_action_count"], 16, minimum=True
        ),
        "minimum_distinct_selected_actions": _gate_result(
            metrics["distinct_selected_action_count"], 2, minimum=True
        ),
        "maximum_abstention_rate": _gate_result(
            metrics["abstention_rate"], 0.8, minimum=False
        ),
        "maximum_hmc_mismatch_rate": {
            "status": "UNEVALUATED",
            "value": None,
            "threshold": 0.1,
            "reason": "offline_model_only; HMC execution was not run",
        },
        "maximum_inference_latency_p99_ms": _gate_result(
            metrics["inference_latency_p99_ms"], 250.0, minimum=False
        ),
        "minimum_dangerous_event_recall": _gate_result(
            metrics["dangerous_event_recall"], 0.98, minimum=True
        ),
    }
    statuses = {item["status"] for item in gates.values()}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "UNDEFINED" in statuses:
        overall = "INSUFFICIENT_SUPPORT"
    elif "UNEVALUATED" in statuses:
        overall = "UNEVALUATED_OFFLINE_HMC"
    else:
        overall = "PASS"
    return {
        "gates": gates,
        "all_evaluated_gates_passed": all(
            item["status"] == "PASS" for item in gates.values() if item["status"] != "UNEVALUATED"
        ),
        "all_preregistered_gates_passed": overall == "PASS",
        "overall_status": overall,
    }


def _provenance(
    source_identity_sha256: str,
    corpus_manifest: Mapping[str, Any],
    corpus_manifest_sha256: str,
    model_protocol_sha256: str,
) -> dict[str, str]:
    values = {
        "source_identity_sha256": source_identity_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "hmc_binding_sha256": corpus_manifest["hmc_binding_sha256"],
        "hmc_contract_sha256": corpus_manifest["hmc_contract_sha256"],
        "scenario_manifest_sha256": corpus_manifest["scenario_manifest_sha256"],
        "action_catalogue_sha256": corpus_manifest["action_catalogue_sha256"],
        "feature_manifest_sha256": corpus_manifest["feature_manifest_sha256"],
        "label_manifest_sha256": corpus_manifest["label_manifest_sha256"],
        "model_protocol_sha256": model_protocol_sha256,
    }
    if set(values) != set(V4_MODEL_PROVENANCE_FIELDS) or any(
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise V4ModelRunError("V4 model provenance is incomplete")
    return values


def _evaluation_metrics(
    model: V4RiskModel,
    samples: Sequence[V4ModelSample],
    *,
    bundle: Any,
) -> dict[str, Any]:
    if not samples or any(sample.split != "EVALUATION" for sample in samples):
        raise V4ModelRunError("V4 evaluation requires EVALUATION samples only")
    if tuple(action.action_id for action in bundle.actions) != V4_ACTION_IDS:
        raise V4ModelRunError("V4 evaluation catalogue ordering is not frozen")
    action_ids = set(V4_ACTION_IDS)
    roster = deterministic_family_ids(32)
    condition_groups: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.family_id not in roster:
            raise V4ModelRunError("V4 evaluation family is outside the frozen roster")
        condition_groups[_condition_group(sample.family_id)].add(sample.family_id)
    if any(len(families) != 2 for families in condition_groups.values()):
        raise V4ModelRunError("V4 evaluation condition groups are not complete pairs")

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    latencies: list[float] = []
    dangerous = 0
    rejected_dangerous = 0
    event_probabilities: list[float] = []
    event_labels: list[float] = []
    for sample in samples:
        if sample.action_id not in action_ids:
            raise V4ModelRunError("V4 evaluation sample action is not in the frozen catalogue")
        baseline_features = np.asarray(sample.features_f32, dtype=np.float64)
        current_command = baseline_features[HISTORY_FEATURE_COUNT - ACTION_COUNT : HISTORY_FEATURE_COUNT]
        proposed_action = baseline_features[
            HISTORY_FEATURE_COUNT * 3 : HISTORY_FEATURE_COUNT * 3 + ACTION_COUNT
        ]
        if (
            current_command.shape != (ACTION_COUNT,)
            or proposed_action.shape != (ACTION_COUNT,)
            or not np.isfinite(current_command).all()
            or not np.isfinite(proposed_action).all()
        ):
            raise V4ModelRunError("V4 evaluation action vectors are malformed")
        action_index = V4_ACTION_IDS.index(sample.action_id)
        started = time.perf_counter_ns()
        prediction = model.predict_features(
            sample.temporal_features_f32
            if model.feature_variant == "v4_temporal_past_only"
            else sample.features_f32,
            action_index=action_index,
            observable_action_mask=sample.observable_action_mask,
        )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        actual_dangerous = sample.label.remaining_metric.crossing_event >= 0.5
        predicted_probability = prediction.horizons[-1].event_probability
        event_probabilities.append(predicted_probability)
        event_labels.append(float(actual_dangerous))
        dangerous += int(actual_dangerous)
        rejected_dangerous += int(actual_dangerous and prediction.hard_ineligible)
        intervention = float(np.mean(np.abs(proposed_action - current_command)))
        groups[(sample.family_id, sample.decision_step)].append(
            {
                "sample": sample,
                "prediction": prediction,
                "intervention": intervention,
                "latency_ms": latencies[-1],
                "dangerous": actual_dangerous,
                "event_probability": predicted_probability,
            }
        )

    selected: list[tuple[V4ModelSample, Any, float]] = []
    selected_by_decision: dict[tuple[str, int], tuple[V4ModelSample, Any, float] | None] = {}
    for decision_key, group in groups.items():
        if len(group) != len(V4_ACTION_IDS) or {
            row["sample"].action_id for row in group
        } != action_ids:
            raise V4ModelRunError("V4 evaluation decision group lacks the full action catalogue")
        scored: list[tuple[tuple[Any, ...], V4ModelSample, Any, float]] = []
        for row in group:
            sample = row["sample"]
            prediction = row["prediction"]
            intervention = row["intervention"]
            if model.candidate_id == "c4_advantage_ranker":
                utility = prediction.advantage_score
            else:
                utility = (
                    prediction.horizons[-1].upper_expected_exposure
                    + V4_UTILITY_WEIGHTS["safety_exposure"]
                    * prediction.upper_relative_safety_exposure
                    + V4_UTILITY_WEIGHTS["comfort_deviation"]
                    * prediction.upper_relative_comfort_deviation
                    + V4_UTILITY_WEIGHTS["resource_composite"]
                    * prediction.upper_relative_resource_composite
                    + V4_UTILITY_WEIGHTS["intervention"] * intervention
                )
            if not prediction.hard_ineligible:
                scored.append(
                    (
                        (
                            utility,
                            prediction.horizons[-1].upper_expected_exposure,
                            prediction.horizons[-1].event_probability,
                            intervention,
                            sample.action_id,
                        ),
                        sample,
                        prediction,
                        intervention,
                    )
                )
        if scored:
            _, sample, prediction, intervention = min(scored, key=lambda item: item[0])
            selected_item = (sample, prediction, intervention)
            selected.append(selected_item)
            selected_by_decision[decision_key] = selected_item
        else:
            selected_by_decision[decision_key] = None

    useful = sum(
        sample.relative_action_targets.safety_exposure_delta_vs_hold < 0.0
        for sample, _, _ in selected
    )
    reliability_bins: list[dict[str, float | int]] = []
    probability_array = np.asarray(event_probabilities, dtype=np.float64)
    label_array = np.asarray(event_labels, dtype=np.float64)
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probability_array >= lower) & (
            probability_array < upper if upper < 1.0 else probability_array <= upper
        )
        if mask.any():
            reliability_bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "mean_probability": float(probability_array[mask].mean()),
                    "event_rate": float(label_array[mask].mean()),
                }
            )
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions_by_condition: dict[
        str, list[tuple[V4ModelSample, Any, float] | None]
    ] = defaultdict(list)
    for decision_key, rows in groups.items():
        condition_group = _condition_group(decision_key[0])
        rows_by_condition[condition_group].extend(rows)
        decisions_by_condition[condition_group].append(selected_by_decision[decision_key])

    group_metrics: dict[str, dict[str, float | None]] = {}
    for condition_group in sorted(rows_by_condition):
        rows = rows_by_condition[condition_group]
        decisions = decisions_by_condition[condition_group]
        dangerous_count = sum(bool(row["dangerous"]) for row in rows)
        rejected_count = sum(
            bool(row["dangerous"] and row["prediction"].hard_ineligible) for row in rows
        )
        retained_rows = [row for row in rows if not row["prediction"].hard_ineligible]
        selected_rows = [item for item in decisions if item is not None]
        selected_dangerous_count = sum(
            item[0].label.remaining_metric.crossing_event >= 0.5 for item in selected_rows
        )
        useful_count = sum(
            item[0].relative_action_targets.safety_exposure_delta_vs_hold < 0.0
            for item in selected_rows
        )
        selected_samples = [item[0] for item in selected_rows]
        group_metrics[condition_group] = {
            "dangerous_event_recall": (
                rejected_count / dangerous_count if dangerous_count else None
            ),
            "false_safe_rate": (
                sum(bool(row["dangerous"]) for row in retained_rows) / len(retained_rows)
                if retained_rows
                else None
            ),
            "selected_action_false_safe_rate": (
                selected_dangerous_count / len(selected_rows) if selected_rows else None
            ),
            "offline_selected_action_rate": len(selected_rows) / len(decisions),
            "abstention_rate": (len(decisions) - len(selected_rows)) / len(decisions),
            "useful_action_rate": useful_count / len(selected_rows) if selected_rows else None,
            "safety_exposure": (
                float(np.mean([sample.trajectory_metrics.safety_exposure for sample in selected_samples]))
                if selected_samples
                else None
            ),
            "safety_violation_steps": (
                float(np.mean([sample.trajectory_metrics.safety_violation_steps for sample in selected_samples]))
                if selected_samples
                else None
            ),
            "comfort_deviation": (
                float(np.mean([sample.trajectory_metrics.comfort_deviation for sample in selected_samples]))
                if selected_samples
                else None
            ),
            "resource_composite": (
                float(np.mean([sample.trajectory_metrics.resource_composite for sample in selected_samples]))
                if selected_samples
                else None
            ),
            "calibration_error": _calibration_error(
                [row["event_probability"] for row in rows],
                [float(row["dangerous"]) for row in rows],
            ),
            "mean_remaining_event_absolute_error": float(
                np.mean(
                    [
                        abs(row["event_probability"] - float(row["dangerous"]))
                        for row in rows
                    ]
                )
            ),
            "inference_latency_p99_ms": float(
                np.quantile(np.asarray([row["latency_ms"] for row in rows]), 0.99)
            ),
        }

    def estimate(metric: str) -> tuple[float | None, dict[str, Any]]:
        defined = {
            condition_group: values[metric]
            for condition_group, values in group_metrics.items()
            if values[metric] is not None
        }
        if not defined:
            return None, {
                "point": None,
                "ci_lower": None,
                "ci_upper": None,
                "resamples": 10_000,
                "seed": 560057,
                "unit": "condition_group",
                "status": "UNDEFINED",
                "group_count": 0,
                "undefined_group_count": len(group_metrics),
                "group_ids": [],
            }
        interval = _group_bootstrap([float(value) for value in defined.values()])
        interval.update(
            {
                "group_count": len(defined),
                "undefined_group_count": len(group_metrics) - len(defined),
                "group_ids": sorted(defined),
            }
        )
        return float(interval["point"]), interval

    grouped_metrics: dict[str, dict[str, Any]] = {}
    estimates: dict[str, float | None] = {}
    for metric in (
        "dangerous_event_recall",
        "false_safe_rate",
        "selected_action_false_safe_rate",
        "offline_selected_action_rate",
        "abstention_rate",
        "useful_action_rate",
        "safety_exposure",
        "safety_violation_steps",
        "comfort_deviation",
        "resource_composite",
        "calibration_error",
        "mean_remaining_event_absolute_error",
        "inference_latency_p99_ms",
    ):
        estimates[metric], grouped_metrics[metric] = estimate(metric)

    group_count = len(groups)
    selected_count = len(selected)
    metrics = {
        "sample_count": len(samples),
        "decision_group_count": group_count,
        "condition_group_count": len(group_metrics),
        "dangerous_sample_count": dangerous,
        "rejected_dangerous_sample_count": rejected_dangerous,
        "retained_sample_count": len(samples) - rejected_dangerous,
        "retained_dangerous_sample_count": dangerous - rejected_dangerous,
        "dangerous_event_recall": estimates["dangerous_event_recall"],
        "selected_action_count": selected_count,
        "proposal_rate": None,
        "offline_selected_action_rate": estimates["offline_selected_action_rate"],
        "abstention_rate": estimates["abstention_rate"],
        "selected_action_false_safe_rate": estimates["selected_action_false_safe_rate"],
        "false_safe_rate": estimates["false_safe_rate"],
        "useful_action_count": useful,
        "useful_action_rate": estimates["useful_action_rate"],
        "distinct_selected_action_count": len({sample.action_id for sample, _, _ in selected}),
        "hmc_mismatch_rate": None,
        "hmc_comparison_count": 0,
        "unavailable_metrics": ["proposal_rate", "hmc_mismatch_rate"],
        "authority_violation_count": 0,
        "replay_failure_count": 0,
        "provenance_violation_count": 0,
        "non_finite_metric_count": 0,
        "proposal_admission_failure_count": 0,
        "disposition_counts": {
            "OFFLINE_SELECTED": selected_count,
            "OFFLINE_ABSTAINED": group_count - selected_count,
        },
        "safety_exposure": estimates["safety_exposure"],
        "safety_violation_steps": estimates["safety_violation_steps"],
        "comfort_deviation": estimates["comfort_deviation"],
        "resource_composite": estimates["resource_composite"],
        "calibration_error": estimates["calibration_error"],
        "reliability_bins": reliability_bins,
        "mean_remaining_event_absolute_error": estimates["mean_remaining_event_absolute_error"],
        "inference_latency_p99_ms": estimates["inference_latency_p99_ms"],
        "condition_group_metrics": group_metrics,
        "condition_group_bootstrap": grouped_metrics,
        "hmc_proposals_issued": 0,
        "plant_steps_issued": 0,
        "execution_metrics_available": False,
        "metrics_finite_verified": False,
    }
    metrics["metrics_finite_verified"] = _all_numeric_metrics_finite(metrics)
    metrics["non_finite_metric_count"] = 0 if metrics["metrics_finite_verified"] else 1
    return metrics


def run_v4_model_study(
    corpus_path: Path,
    output_path: Path,
    *,
    allow_dirty_smoke: bool = False,
) -> dict[str, Any]:
    """Fit and evaluate the preregistered candidates on a verified corpus."""

    corpus = _resolve_corpus(corpus_path)
    manifest = _strict_json(corpus / "manifest.json")
    sample_counts = manifest.get("sample_counts")
    if type(sample_counts) is not dict or set(sample_counts) != set(V4_MODEL_SPLIT_COUNTS):
        raise V4ModelRunError("V4 corpus sample-count manifest is malformed")
    expected_sample_counts = {
        split: V4_MODEL_SPLIT_COUNTS[split]
        * len(v2_decision_steps())
        * len(V4_ACTION_IDS)
        for split in V4_MODEL_SPLIT_COUNTS
    }
    is_full_study = sample_counts == expected_sample_counts
    if manifest.get("smoke_only") is not (not is_full_study):
        raise V4ModelRunError("V4 corpus smoke classification is inconsistent with sample counts")
    if not is_full_study and not allow_dirty_smoke:
        raise V4ModelRunError("V4 smoke corpus requires --allow-dirty-smoke")
    _independent_verify(corpus)
    output = _resolve_output(output_path)
    source_identity = _source_identity()
    if source_identity["source_worktree_dirty"] and not (
        allow_dirty_smoke and not is_full_study
    ):
        raise V4ModelRunError(
            "comparative V4 model study refuses a dirty source worktree; use a smoke corpus "
            "with --allow-dirty-smoke while developing"
        )
    bundle = load_forecast_contracts(REPO_ROOT)
    _, model_protocol_sha256 = load_v4_model_protocol(REPO_ROOT)
    family_ids = tuple(manifest["family_ids"])
    roster = deterministic_family_ids(32)
    if not family_ids or any(family_id not in roster for family_id in family_ids):
        raise V4ModelRunError("V4 corpus family identity is malformed")
    scenarios = {
        family_id: build_family_scenario(bundle.development_scenario, roster.index(family_id))
        for family_id in family_ids
    }
    rows = _strict_jsonl(corpus / "samples.jsonl")
    samples = _load_verified_model_samples(rows, corpus, bundle, scenarios, family_ids)
    del rows
    gc.collect()
    train = tuple(sample for sample in samples if sample.split == "TRAIN")
    validation = tuple(sample for sample in samples if sample.split == "VALIDATION")
    evaluation = tuple(sample for sample in samples if sample.split == "EVALUATION")
    if not train or not validation or not evaluation:
        raise V4ModelRunError("V4 corpus does not contain all required splits")

    corpus_manifest_sha256 = _sha_bytes((corpus / "manifest.json").read_bytes())
    provenance = _provenance(
        _sha_json(source_identity),
        manifest,
        corpus_manifest_sha256,
        model_protocol_sha256,
    )
    candidates: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(V4_MODEL_CANDIDATES):
        seed = V4_MODEL_SEEDS[index % len(V4_MODEL_SEEDS)]
        try:
            model = V4RiskModel.fit(
                train,
                candidate_id=candidate_id,
                seed=seed,
                provenance=provenance,
            ).calibrate(validation)
        except Issue56V4ModelError as error:
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "status": "FAILED_CLOSED",
                    "failure_stage": "FIT_OR_CALIBRATION",
                    "failure": str(error),
                    "gate_status": {
                        "gates": {
                            "fit_and_calibration": {
                                "status": "FAIL",
                                "reason": str(error),
                            }
                        },
                        "all_evaluated_gates_passed": False,
                        "all_preregistered_gates_passed": False,
                        "overall_status": "FAIL_CLOSED",
                    },
                }
            )
            continue
        artifact_path = output / "models" / f"{candidate_id}.json"
        artifact_sha256 = write_v4_model(artifact_path, model)
        reloaded, reloaded_sha256 = load_v4_model(artifact_path)
        if reloaded.to_mapping() != model.to_mapping() or reloaded_sha256 != artifact_sha256:
            raise V4ModelRunError(f"V4 model artifact reload diverged: {candidate_id}")
        metrics = _evaluation_metrics(
            reloaded,
            evaluation,
            bundle=bundle,
        )
        gate_status = _evaluation_gate_status(metrics)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "feature_variant": reloaded.feature_variant,
                "model_kind": reloaded.model_kind,
                "hazard_mode": reloaded.hazard_mode,
                "model_path": artifact_path.relative_to(output).as_posix(),
                "model_file_sha256": artifact_sha256,
                "model_sha256": reloaded.to_mapping()["model_sha256"],
                "calibration_support": list(reloaded.calibration_support),
                "event_thresholds": list(reloaded.event_thresholds),
                "validation_decision_coverage": reloaded.validation_decision_coverage,
                "evaluation": metrics,
                "gate_status": gate_status,
            }
        )
    candidate_gate_status = {
        item["candidate_id"]: item["gate_status"] for item in candidates
    }
    all_evaluated_gates_passed = all(
        candidate_gate_status
        and item["all_evaluated_gates_passed"]
        for item in candidate_gate_status.values()
    )
    all_preregistered_gates_passed = all(
        candidate_gate_status
        and item["all_preregistered_gates_passed"]
        for item in candidate_gate_status.values()
    )
    result = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v4_model_study_v2.result",
        "preregistration_id": ISSUE56_V4_MODEL_PROTOCOL_ID,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "source_identity": source_identity,
        "source_identity_sha256": _sha_json(source_identity),
        "model_protocol_sha256": model_protocol_sha256,
        "family_count": len(family_ids),
        "sample_counts": {
            "TRAIN": len(train),
            "VALIDATION": len(validation),
            "EVALUATION": len(evaluation),
            "TOTAL": len(samples),
        },
        "candidates": candidates,
        "evaluation_scope": "offline_model_only",
        "hmc_execution_metrics_available": False,
        "hmc_dependent_metrics_status": "UNAVAILABLE_OFFLINE_MODEL_ONLY",
        "protected_final_suite_accessed": False,
        "hmc_authority_changed": False,
        "all_evaluated_gates_passed": all_evaluated_gates_passed,
        "all_preregistered_gates_passed": all_preregistered_gates_passed,
        "candidate_gate_status": candidate_gate_status,
        "candidate_failure_count": sum(
            item.get("status") == "FAILED_CLOSED" for item in candidates
        ),
        "status": "SMOKE_PATH_ONLY" if not is_full_study else "DEVELOPMENT_EVIDENCE",
    }
    results_sha256 = _write_json(output / "results.json", result)
    manifest_output = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v4_model_study_v2.manifest",
        "preregistration_id": ISSUE56_V4_MODEL_PROTOCOL_ID,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "source_identity_sha256": result["source_identity_sha256"],
        "model_protocol_sha256": model_protocol_sha256,
        "candidate_model_file_sha256": {
            item["candidate_id"]: item["model_file_sha256"]
            for item in candidates
            if item.get("status") != "FAILED_CLOSED"
        },
        "candidate_status": {
            item["candidate_id"]: item.get("status", "EVALUATED") for item in candidates
        },
        "results_sha256": results_sha256,
        "candidate_count": len(candidates),
        "candidate_failure_count": sum(
            item.get("status") == "FAILED_CLOSED" for item in candidates
        ),
        "evaluation_scope": "offline_model_only",
        "hmc_execution_metrics_available": False,
        "protected_final_suite_accessed": False,
        "hmc_authority_changed": False,
    }
    manifest_sha256 = _write_json(output / "manifest.json", manifest_output)
    return {
        "output": str(output),
        "manifest_sha256": manifest_sha256,
        "results_sha256": results_sha256,
        "candidate_count": len(candidates),
        "sample_counts": result["sample_counts"],
        "status": result["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Issue #56 V4 model study")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    result = run_v4_model_study(
        args.corpus,
        args.output,
        allow_dirty_smoke=args.allow_dirty_smoke,
    )
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
