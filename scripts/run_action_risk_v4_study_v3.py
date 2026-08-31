#!/usr/bin/env python3
"""Run the Issue #56 V4 protocol revision 3 development study.

Stage A fits and evaluates the preregistered candidates offline against the
frozen, independently verified development corpus using the composite
point-delta selection contract.  Stage B replays the first Stage-A-gate-passing
candidate through the frozen HMC as an advisory arm alongside the frozen V3
arms under identical episode conditions.

The runner never reads the protected final suite and never changes HMC
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
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model
from aeolus.habitat_v2.forecast_issue55_race import (
    FAMILY_COUNT,
    build_family_scenario,
    deterministic_family_ids,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    V3EpisodeRecord,
    V3RiskModel,
    V4_MODEL_ARM,
    collect_v3_family_samples,
    load_v3_samples,
    run_v3_episode,
    v3_family_split,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import load_v4_samples
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model import (
    Issue56V4ModelError,
    V4_ACTION_IDS,
    V4_MODEL_PROVENANCE_FIELDS,
    V4_MODEL_SEEDS,
    V4_THRESHOLD_GRID_EXTENDED,
    V4_COMPOSITE_SELECTION_WEIGHTS,
    V4ModelSample,
    V4RiskModel,
    load_v4_model,
    write_v4_model,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
    ISSUE56_V4_MODEL_PROTOCOL_V5_ID,
    ISSUE56_V4_MODEL_PROTOCOL_V6_ID,
    ISSUE56_V4_MODEL_PROTOCOL_V7_ID,
    V4_MODEL_V3_SPLIT_PROTOCOL,
    V4_MODEL_V3_STAGE_B_ARMS,
    V4_MODEL_V4_CANDIDATE_IDS,
    V4_MODEL_V4_STAGE_B_RULE,
    load_v4_model_protocol_v5,
    load_v4_model_protocol_v6,
    load_v4_model_protocol_v7,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (REPO_ROOT / "out").resolve()
VERIFY_MODULE = "scripts.verify_action_risk_v4_corpus"
POINT_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
POINT_ARTIFACT_SHA256 = (
    "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
)
FROZEN_V3_MODEL_FILE_SHA256 = (
    "e977ccb6b4298c5793838621bd819df50f46926ca2c2b73664ea9da232e4fdb8"
)
PROTOCOL_VERSION_CONTRACTS = {
    "v5": Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v5.json"),
    "v6": Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v6.json"),
    "v7": Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v7.json"),
}
PROTOCOL_VERSION_LOADERS = {
    "v5": (load_v4_model_protocol_v5, ISSUE56_V4_MODEL_PROTOCOL_V5_ID),
    "v6": (load_v4_model_protocol_v6, ISSUE56_V4_MODEL_PROTOCOL_V6_ID),
    "v7": (load_v4_model_protocol_v7, ISSUE56_V4_MODEL_PROTOCOL_V7_ID),
}
STUDY_SOURCE_PATHS_BASE = (
    Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v3.json"),
    Path("contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v2.json"),
    Path("src/aeolus/habitat_v2/forecast/contracts.py"),
    Path("src/aeolus/habitat_v2/forecast/projection.py"),
    Path("src/aeolus/habitat_v2/forecast/live_mlp_demo.py"),
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
    Path("scripts/run_action_risk_v4_study_v3.py"),
    Path("scripts/verify_action_risk_v4_corpus.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _study_source_paths(protocol_version: str) -> tuple[Path, ...]:
    return (PROTOCOL_VERSION_CONTRACTS[protocol_version],) + STUDY_SOURCE_PATHS_BASE


class V4StudyV3Error(RuntimeError):
    """Raised when the V4 protocol v3 study cannot preserve its boundary."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: object) -> str:
    try:
        return _sha_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise V4StudyV3Error("V4 study value is not canonical JSON") from error


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
        raise V4StudyV3Error(f"cannot strictly read JSON: {path}") from error
    if type(value) is not dict:
        raise V4StudyV3Error(f"JSON root must be an object: {path}")
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
        raise V4StudyV3Error(f"cannot read V4 samples: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise V4StudyV3Error(f"invalid V4 sample JSON at {path}:{line_number}") from error
        if type(value) is not dict:
            raise V4StudyV3Error(f"V4 sample row is not an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise V4StudyV3Error("V4 samples.jsonl is empty")
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
        raise V4StudyV3Error(f"cannot write JSON artifact: {path}") from error
    return _sha_bytes(payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _source_identity(protocol_version: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in _study_source_paths(protocol_version):
        path = REPO_ROOT / relative
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise V4StudyV3Error(f"cannot read study source file: {relative}") from error
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
        raise V4StudyV3Error("V4 study source identity cannot be read from Git") from error
    return {
        "schema_version": "issue56-v4-study-v3-source-identity-v1",
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
        raise V4StudyV3Error("V4 study corpus must be an existing directory below repository out/")
    return resolved


def _resolve_output(path: Path) -> Path:
    resolved = path.resolve()
    if OUT_ROOT not in resolved.parents:
        raise V4StudyV3Error("V4 study output must be below repository out/")
    if resolved.exists():
        raise V4StudyV3Error("V4 study output directory must be new and write-once")
    resolved.mkdir(parents=True, exist_ok=False)
    (resolved / "models").mkdir()
    (resolved / "traces").mkdir()
    return resolved


def _independent_verify(corpus: Path, split_protocol: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                VERIFY_MODULE,
                "--corpus",
                str(corpus),
                "--split-protocol",
                split_protocol,
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else ""
        raise V4StudyV3Error(f"independent V4 corpus verification failed: {detail}") from error
    try:
        return json.loads(completed.stderr.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise V4StudyV3Error("independent V4 corpus verification emitted no receipt") from error


def _verify_corpus_binding(
    corpus: Path,
    protocol: Mapping[str, Any],
    *,
    allow_smoke: bool,
) -> tuple[dict[str, Any], str, bool]:
    """Check the corpus against the protocol data contract; return (manifest, digest, smoke)."""

    manifest = _strict_json(corpus / "manifest.json")
    manifest_digest = _sha_bytes((corpus / "manifest.json").read_bytes())
    requirement = protocol["corpus_requirement"]
    smoke = manifest.get("smoke_only") is True
    if smoke:
        if not allow_smoke:
            raise V4StudyV3Error("smoke corpus requires --allow-dirty-smoke")
        return manifest, manifest_digest, True
    if manifest.get("preregistration_id") != requirement["corpus_preregistration_id"]:
        raise V4StudyV3Error("V4 corpus is not bound to the required data-contract protocol")
    if manifest.get("schema_version") != requirement["corpus_schema_version"] + ".manifest":
        raise V4StudyV3Error("V4 corpus schema version differs from the protocol requirement")
    if len(manifest.get("family_ids", [])) != requirement["family_count"]:
        raise V4StudyV3Error("V4 corpus family count differs from the protocol requirement")
    required_counts = {
        key: value for key, value in requirement["sample_counts"].items() if key != "TOTAL"
    }
    if manifest.get("sample_counts") != required_counts:
        raise V4StudyV3Error("V4 corpus sample counts differ from the protocol requirement")
    if manifest.get("trace_count") != requirement["trace_count"]:
        raise V4StudyV3Error("V4 corpus trace count differs from the protocol requirement")
    if (
        manifest.get("counterfactual_trace_bytes_present") is not requirement["counterfactual_trace_bytes_present"]
        or manifest.get("hold_trace_bytes_present") is not requirement["hold_trace_bytes_present"]
    ):
        raise V4StudyV3Error("V4 corpus trace-byte presence differs from the protocol requirement")
    return manifest, manifest_digest, False


def _load_verified_model_samples(
    rows: Sequence[Mapping[str, Any]],
    corpus: Path,
    bundle: Any,
    scenarios: Mapping[str, Any],
    family_ids: Sequence[str],
) -> tuple[V4ModelSample, ...]:
    rows_by_family: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        base_sample = row.get("base_sample")
        family_id = base_sample.get("family_id") if type(base_sample) is dict else None
        rows_by_family[family_id].append(row)
    if set(rows_by_family) != set(family_ids):
        raise V4StudyV3Error("V4 study sample family coverage is incomplete")

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
            raise V4StudyV3Error(
                f"V4 corpus family reload failed after independent verification: {family_id}"
            ) from error
        samples.extend(V4ModelSample.from_verified(sample) for sample in verified)
        del verified
        gc.collect()
    if len(samples) != len(rows):
        raise V4StudyV3Error("V4 study sample count differs after family reload")
    return tuple(samples)


def _condition_group(family_id: str) -> str:
    roster = deterministic_family_ids(32)
    try:
        index = roster.index(family_id)
    except ValueError as error:
        raise V4StudyV3Error("V4 study family is outside the frozen roster") from error
    return f"condition-group-{index // 2:04d}"


@lru_cache(maxsize=8)
def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    if repetitions < 1 or count < 1:
        raise V4StudyV3Error("V4 study bootstrap dimensions must be positive")
    result = np.empty((repetitions, count), dtype=np.int64)
    for replicate in range(repetitions):
        for draw in range(count):
            digest = hashlib.sha256(
                f"issue56-v4-study-v3-bootstrap-v1|{seed}|{replicate}|{draw}".encode("ascii")
            ).digest()
            result[replicate, draw] = int.from_bytes(digest[:8], "big") % count
    return result


def _group_bootstrap(values: Sequence[float], *, seed: int, resamples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise V4StudyV3Error("V4 study bootstrap values are malformed")
    if len(array) < 2:
        return {
            "point": float(np.mean(array)),
            "ci_lower": None,
            "ci_upper": None,
            "resamples": resamples,
            "seed": seed,
            "unit": "condition_group",
            "status": "INSUFFICIENT_CONDITION_GROUP_SUPPORT",
            "group_count": len(array),
        }
    draws = np.mean(array[_bootstrap_indices(seed, resamples, len(array))], axis=1)
    return {
        "point": float(np.mean(array)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "seed": seed,
        "unit": "condition_group",
    }


def _paired_family_bootstrap(
    differences: Sequence[float], *, seed: int, resamples: int
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise V4StudyV3Error("V4 study paired bootstrap differences are invalid")
    draws = np.mean(values[_bootstrap_indices(seed, resamples, len(values))], axis=1)
    return {
        "point_difference": float(np.mean(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


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
        raise V4StudyV3Error("V4 study calibration metric inputs are malformed")
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
        raise V4StudyV3Error("V4 study model provenance is incomplete")
    return values


def _composite_utility(prediction: Any) -> float:
    return float(
        V4_COMPOSITE_SELECTION_WEIGHTS["safety_exposure"]
        * prediction.relative_safety_exposure
        + V4_COMPOSITE_SELECTION_WEIGHTS["comfort_deviation"]
        * prediction.relative_comfort_deviation
        + V4_COMPOSITE_SELECTION_WEIGHTS["resource_composite"]
        * prediction.relative_resource_composite
    )


def _stage_a_metrics(
    model: V4RiskModel,
    samples: Sequence[V4ModelSample],
) -> dict[str, Any]:
    """Evaluate one calibrated candidate with the composite point-delta contract."""

    if not samples or any(sample.split != "EVALUATION" for sample in samples):
        raise V4StudyV3Error("V4 stage A requires EVALUATION samples only")
    action_ids = set(V4_ACTION_IDS)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    latencies: list[float] = []
    dangerous = 0
    rejected_dangerous = 0
    event_probabilities: list[float] = []
    event_labels: list[float] = []
    for sample in samples:
        if sample.action_id not in action_ids:
            raise V4StudyV3Error("V4 stage A sample action is outside the catalogue")
        features = (
            sample.features_f32
            if model.feature_variant == "v3_708_past_only"
            else sample.temporal_features_f32
        )
        action_index = V4_ACTION_IDS.index(sample.action_id)
        started = time.perf_counter_ns()
        prediction = model.predict_features(
            features,
            action_index=action_index,
            observable_action_mask=sample.observable_action_mask,
        )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        actual_dangerous = sample.label.remaining_metric.crossing_event >= 0.5
        event_probabilities.append(prediction.horizons[-1].event_probability)
        event_labels.append(float(actual_dangerous))
        dangerous += int(actual_dangerous)
        rejected_dangerous += int(actual_dangerous and prediction.hard_ineligible)
        groups[(sample.family_id, sample.decision_step)].append(
            {
                "sample": sample,
                "prediction": prediction,
                "dangerous": actual_dangerous,
                "latency_ms": latencies[-1],
            }
        )

    selected: list[V4ModelSample] = []
    selected_by_decision: dict[tuple[str, int], V4ModelSample | None] = {}
    for decision_key, group in groups.items():
        if len(group) != len(V4_ACTION_IDS) or {
            row["sample"].action_id for row in group
        } != action_ids:
            raise V4StudyV3Error("V4 stage A decision group lacks the full action catalogue")
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for row in group:
            prediction = row["prediction"]
            if prediction.hard_ineligible or prediction.relative_safety_exposure >= 0.0:
                continue
            eligible.append((_composite_utility(prediction), row["sample"].action_id, row))
        if eligible:
            _, _, row = min(eligible, key=lambda item: (item[0], item[1]))
            selected.append(row["sample"])
            selected_by_decision[decision_key] = row["sample"]
        else:
            selected_by_decision[decision_key] = None

    useful = sum(
        sample.relative_action_targets.safety_exposure_delta_vs_hold < 0.0
        for sample in selected
    )
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions_by_condition: dict[str, list[V4ModelSample | None]] = defaultdict(list)
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
            item.label.remaining_metric.crossing_event >= 0.5 for item in selected_rows
        )
        useful_count = sum(
            item.relative_action_targets.safety_exposure_delta_vs_hold < 0.0
            for item in selected_rows
        )
        selected_samples = list(selected_rows)
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
                [row["prediction"].horizons[-1].event_probability for row in rows],
                [float(row["dangerous"]) for row in rows],
            ),
            "inference_latency_p99_ms": float(
                np.quantile(np.asarray([row["latency_ms"] for row in rows], dtype=np.float64), 0.99)
            ),
        }

    estimates: dict[str, Any] = {}
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
    ):
        defined = [
            values[metric]
            for values in group_metrics.values()
            if values[metric] is not None
        ]
        estimates[metric] = (
            _group_bootstrap([float(value) for value in defined], seed=560057, resamples=10_000)
            if defined
            else {"point": None, "group_count": 0}
        )

    metrics = {
        "sample_count": len(samples),
        "decision_group_count": len(groups),
        "condition_group_count": len(group_metrics),
        "dangerous_sample_count": dangerous,
        "rejected_dangerous_sample_count": rejected_dangerous,
        "dangerous_event_recall": estimates["dangerous_event_recall"]["point"],
        "selected_action_count": len(selected),
        "offline_selected_action_rate": estimates["offline_selected_action_rate"]["point"],
        "abstention_rate": (len(groups) - len(selected)) / len(groups),
        "selected_action_false_safe_rate": estimates["selected_action_false_safe_rate"]["point"],
        "false_safe_rate": estimates["false_safe_rate"]["point"],
        "useful_action_count": useful,
        "useful_action_rate": estimates["useful_action_rate"]["point"],
        "distinct_selected_action_count": len({sample.action_id for sample in selected}),
        "authority_violation_count": 0,
        "replay_failure_count": 0,
        "provenance_violation_count": 0,
        "non_finite_metric_count": 0,
        "proposal_admission_failure_count": 0,
        "calibration_error": estimates["calibration_error"]["point"],
        "safety_exposure": estimates["safety_exposure"]["point"],
        "comfort_deviation": estimates["comfort_deviation"]["point"],
        "resource_composite": estimates["resource_composite"]["point"],
        "inference_latency_p99_ms": float(np.quantile(np.asarray(latencies), 0.99)),
        "condition_group_metrics": group_metrics,
        "condition_group_bootstrap": estimates,
        "metrics_finite_verified": False,
    }
    metrics["metrics_finite_verified"] = _all_numeric_metrics_finite(metrics)
    metrics["non_finite_metric_count"] = 0 if metrics["metrics_finite_verified"] else 1
    return metrics


STAGE_A_GATE_METRICS = {
    "authority_violations": ("authority_violation_count", False),
    "replay_failures": ("replay_failure_count", False),
    "provenance_violations": ("provenance_violation_count", False),
    "non_finite_metrics": ("non_finite_metric_count", False),
    "proposal_admission_failures": ("proposal_admission_failure_count", False),
    "minimum_useful_action_count": ("useful_action_count", True),
    "minimum_distinct_selected_actions": ("distinct_selected_action_count", True),
    "maximum_abstention_rate": ("abstention_rate", False),
    "maximum_inference_latency_p99_ms": ("inference_latency_p99_ms", False),
    "minimum_dangerous_event_recall": ("dangerous_event_recall", True),
}


def _stage_a_gate_status(
    metrics: Mapping[str, Any], gate_spec: Mapping[str, Any]
) -> dict[str, Any]:
    def gate(value: object, threshold: float | int, *, minimum: bool) -> dict[str, Any]:
        if value is None:
            return {"status": "UNDEFINED", "value": None, "threshold": threshold}
        numeric = float(value)
        passed = numeric >= threshold if minimum else numeric <= threshold
        return {"status": "PASS" if passed else "FAIL", "value": value, "threshold": threshold}

    unknown = set(gate_spec) - set(STAGE_A_GATE_METRICS)
    if unknown:
        raise V4StudyV3Error(f"V4 study stage A gate spec has unknown gates {sorted(unknown)}")
    gates = {
        gate_key: gate(
            metrics[STAGE_A_GATE_METRICS[gate_key][0]],
            gate_spec[gate_key],
            minimum=STAGE_A_GATE_METRICS[gate_key][1],
        )
        for gate_key in sorted(gate_spec)
    }
    statuses = {item["status"] for item in gates.values()}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "UNDEFINED" in statuses:
        overall = "INSUFFICIENT_SUPPORT"
    else:
        overall = "PASS"
    return {
        "gates": gates,
        "all_preregistered_gates_passed": overall == "PASS",
        "overall_status": overall,
    }


def _aggregate_episodes(records: Sequence[V3EpisodeRecord]) -> dict[str, Any]:
    by_arm: dict[str, list[V3EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_arm[record.arm].append(record)
    summaries: dict[str, Any] = {}
    for arm, arm_records in sorted(by_arm.items()):
        summaries[arm] = {
            "family_count": len(arm_records),
            "family_means": {
                metric: float(np.mean([float(getattr(record, metric)) for record in arm_records]))
                for metric in (
                    "safety_exposure",
                    "safety_violation_steps",
                    "comfort_deviation",
                    "resource_composite",
                )
            },
            "proposal_count": sum(record.proposal_count for record in arm_records),
            "abstention_count": sum(record.abstention_count for record in arm_records),
            "admitted_proposal_count": sum(record.admitted_proposal_count for record in arm_records),
            "hmc_mismatch_count": sum(record.hmc_mismatch_count for record in arm_records),
            "replay_committed_steps": sum(record.replay_committed_steps for record in arm_records),
        }
    return {
        "family_count": len({record.family_id for record in records}),
        "arm_summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Issue #56 V4 protocol revision 3 development study"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    parser.add_argument(
        "--protocol-version",
        default="v5",
        choices=tuple(PROTOCOL_VERSION_LOADERS),
        help="authorized V4 model-study protocol revision to run",
    )
    args = parser.parse_args()

    protocol_loader, preregistration_id = PROTOCOL_VERSION_LOADERS[args.protocol_version]
    protocol, protocol_sha256 = protocol_loader(REPO_ROOT)
    split_protocol = protocol["corpus_requirement"].get(
        "split_protocol", V4_MODEL_V3_SPLIT_PROTOCOL
    )
    corpus = _resolve_corpus(args.corpus)
    manifest, corpus_manifest_digest, smoke = _verify_corpus_binding(
        corpus, protocol, allow_smoke=args.allow_dirty_smoke
    )
    source_identity = _source_identity(args.protocol_version)
    if source_identity["source_worktree_dirty"] and not (
        args.allow_dirty_smoke and smoke
    ):
        raise V4StudyV3Error(
            "comparative V4 study refuses a dirty source worktree; use a smoke corpus "
            "with --allow-dirty-smoke while developing"
        )
    verification_receipt = _independent_verify(corpus, split_protocol)
    if verification_receipt.get("strict_trace_replay_verified") is not True:
        raise V4StudyV3Error("independent verification did not confirm strict replay")
    if verification_receipt.get("split_protocol") != split_protocol:
        raise V4StudyV3Error("independent verification split protocol drifted")

    output = _resolve_output(args.output)
    bundle = load_forecast_contracts(REPO_ROOT)
    roster = deterministic_family_ids(FAMILY_COUNT)
    family_split = manifest["family_split"]
    family_ids = tuple(manifest["family_ids"])
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
        raise V4StudyV3Error("V4 study corpus must contain all three splits")

    provenance = _provenance(
        _sha_json(source_identity), manifest, corpus_manifest_digest, protocol_sha256
    )

    print(f"Issue #56 V4 study v4 (stage A: {len(V4_MODEL_V4_CANDIDATE_IDS)} candidates)", file=sys.stderr)
    candidates: list[dict[str, Any]] = []
    gate_status_by_candidate: dict[str, Any] = {}
    for index, candidate_id in enumerate(V4_MODEL_V4_CANDIDATE_IDS):
        seed = V4_MODEL_SEEDS[index % len(V4_MODEL_SEEDS)]
        try:
            model = V4RiskModel.fit(
                train, candidate_id=candidate_id, seed=seed, provenance=provenance
            ).calibrate(validation, threshold_grid=V4_THRESHOLD_GRID_EXTENDED)
        except Issue56V4ModelError as error:
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "status": "FAILED_CLOSED",
                    "failure_stage": "FIT_OR_CALIBRATION",
                    "failure": str(error),
                    "gate_status": {
                        "all_preregistered_gates_passed": False,
                        "overall_status": "FAIL_CLOSED",
                        "gates": {"fit_and_calibration": {"status": "FAIL", "reason": str(error)}},
                    },
                }
            )
            gate_status_by_candidate[candidate_id] = False
            print(f"  {candidate_id}: FAIL_CLOSED ({error})", file=sys.stderr)
            continue
        model_path = output / "models" / f"{candidate_id}.json"
        model_file_sha256 = write_v4_model(model_path, model)
        reloaded, reload_sha256 = load_v4_model(model_path)
        if reloaded.to_mapping() != model.to_mapping() or reload_sha256 != model_file_sha256:
            raise V4StudyV3Error(f"V4 model artifact reload diverged: {candidate_id}")
        metrics = _stage_a_metrics(model, evaluation)
        gate_status = _stage_a_gate_status(
            metrics, protocol["evaluation"]["stage_a_offline"]["gates"]
        )
        gate_status_by_candidate[candidate_id] = gate_status["all_preregistered_gates_passed"]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "status": "EVALUATED",
                "feature_variant": model.feature_variant,
                "model_kind": model.model_kind,
                "hazard_mode": model.hazard_mode,
                "model_path": f"models/{candidate_id}.json",
                "model_file_sha256": model_file_sha256,
                "model_sha256": model.model_sha256,
                "event_thresholds": list(model.event_thresholds),
                "calibration_support": list(model.calibration_support),
                "validation_decision_coverage": model.validation_decision_coverage,
                "evaluation": metrics,
                "gate_status": gate_status,
            }
        )
        print(
            f"  {candidate_id}: {gate_status['overall_status']} "
            f"(useful={metrics['useful_action_count']}, "
            f"distinct={metrics['distinct_selected_action_count']})",
            file=sys.stderr,
        )

    if protocol["evaluation"]["stage_b_hmc_replay"]["stage_b_candidate_rule"] != (
        V4_MODEL_V4_STAGE_B_RULE
    ):
        raise V4StudyV3Error("V4 study stage B candidate rule drifted from the protocol")

    evaluated_by_id = {
        item["candidate_id"]: item for item in candidates if item["status"] == "EVALUATED"
    }

    def _safety_critical_pass(cid: str) -> bool:
        item = evaluated_by_id.get(cid)
        if item is None:
            return False
        metrics = item["evaluation"]
        return bool(
            metrics["dangerous_event_recall"] is not None
            and metrics["dangerous_event_recall"] >= 0.98
            and metrics["authority_violation_count"] == 0
            and metrics["replay_failure_count"] == 0
            and metrics["provenance_violation_count"] == 0
            and metrics["non_finite_metric_count"] == 0
            and metrics["proposal_admission_failure_count"] == 0
        )

    full_passers = [
        cid for cid in V4_MODEL_V4_CANDIDATE_IDS if gate_status_by_candidate.get(cid)
    ]
    if full_passers:
        stage_b_candidate = full_passers[0]
        stage_b_trigger = "stage_a_gate_passer"
    else:
        safety_passers = [cid for cid in V4_MODEL_V4_CANDIDATE_IDS if _safety_critical_pass(cid)]
        if safety_passers:
            stage_b_candidate = max(
                safety_passers,
                key=lambda cid: evaluated_by_id[cid]["evaluation"]["useful_action_count"],
            )
            stage_b_trigger = "best_safety_passing_usefulness"
        else:
            stage_b_candidate = None
            stage_b_trigger = "no_safety_passing_candidate"

    stage_b: dict[str, Any] = {
        "ran": False,
        "candidate_id": stage_b_candidate,
        "candidate_rule": V4_MODEL_V4_STAGE_B_RULE,
        "trigger": stage_b_trigger,
    }
    if stage_b_candidate is not None:
        stage_b = _run_stage_b(
            output,
            bundle,
            roster,
            family_split,
            evaluation,
            stage_b_candidate,
            evaluated_by_id[stage_b_candidate],
            protocol,
            smoke=smoke,
        )
        stage_b["candidate_rule"] = V4_MODEL_V4_STAGE_B_RULE
        stage_b["trigger"] = stage_b_trigger

    results = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v4_study_v4.result",
        "preregistration_id": preregistration_id,
        "model_protocol_sha256": protocol_sha256,
        "split_protocol": split_protocol,
        "corpus_manifest_sha256": corpus_manifest_digest,
        "corpus_verification": verification_receipt,
        "evaluation_scope": "stage_a_offline_plus_stage_b_hmc_replay"
        if stage_b.get("ran")
        else "stage_a_offline_only",
        "hmc_authority_changed": False,
        "protected_final_suite_accessed": False,
        "sample_counts": {
            "TRAIN": len(train),
            "VALIDATION": len(validation),
            "EVALUATION": len(evaluation),
            "TOTAL": len(samples),
        },
        "source_identity": source_identity,
        "source_identity_sha256": _sha_json(source_identity),
        "candidates": candidates,
        "candidate_gate_status": {
            candidate_id: bool(passed)
            for candidate_id, passed in gate_status_by_candidate.items()
        },
        "stage_b": stage_b,
        "outperforms_v3": bool(
            stage_b.get("ran")
            and stage_b.get("all_stage_b_gates_passed")
            and stage_b.get("superiority_over_v3", {}).get("achieved")
        ),
        "smoke_only": smoke,
        "status": "SMOKE_PATH_ONLY" if smoke else "DEVELOPMENT_EVIDENCE",
    }
    manifest_payload = {
        "schema_version": "aeolus_habitat_v2_risk_issue_56_v4_study_v4.manifest",
        "preregistration_id": preregistration_id,
        "model_protocol_sha256": protocol_sha256,
        "split_protocol": split_protocol,
        "corpus_manifest_sha256": corpus_manifest_digest,
        "candidate_ids": list(V4_MODEL_V4_CANDIDATE_IDS),
        "stage_b_candidate": stage_b_candidate,
        "stage_b_arms": list(V4_MODEL_V3_STAGE_B_ARMS),
        "hmc_authority_changed": False,
        "protected_final_suite_accessed": False,
        "smoke_only": smoke,
    }
    results["manifest_sha256"] = _write_json(output / "manifest.json", manifest_payload)
    results_sha256 = _write_json(output / "results.json", results)
    print(f"results.json sha256: {results_sha256}", file=sys.stderr)
    print(f"output: {output}", file=sys.stderr)
    return 0


def _superiority_over_v3(
    spec: Mapping[str, Any],
    safety: Mapping[str, Any],
    admitted_v4: int,
    admitted_v3: int,
    mismatch_v4: int,
) -> dict[str, Any]:
    point_difference = float(safety["point_difference"])
    ci_lower = float(safety["ci_lower"])
    ci_upper = float(safety["ci_upper"])
    superiority: dict[str, Any] = {
        "safety_exposure_paired_point_difference": point_difference,
        "safety_exposure_paired_ci": [ci_lower, ci_upper],
        "admitted_proposal_count_v4": admitted_v4,
        "admitted_proposal_count_v3": admitted_v3,
        "hmc_mismatch_count_v4": mismatch_v4,
    }
    if "admitted_proposal_count_must_exceed_v3" in spec:
        safety_no_worse = bool(
            point_difference <= spec["safety_exposure_paired_point_difference_maximum"]
        )
        more_admitted = bool(admitted_v4 > admitted_v3)
        superiority["safety_no_worse_than_v3"] = safety_no_worse
        superiority["more_admitted_proposals_than_v3"] = more_admitted
        superiority["achieved"] = bool(safety_no_worse and more_admitted)
        return superiority
    admissions_ok = bool(admitted_v4 >= admitted_v3)
    within_maximum = bool(
        point_difference <= spec["safety_exposure_paired_point_difference_maximum"]
    )
    strictly_negative = bool(point_difference < 0.0) if spec[
        "safety_exposure_paired_point_difference_must_be_strictly_negative"
    ] else True
    ci_ok = bool(ci_upper <= spec["safety_exposure_paired_ci_upper_maximum"])
    overrides_ok = bool(mismatch_v4 <= spec["maximum_hmc_mismatch_count"])
    superiority["admissions_at_least_v3"] = admissions_ok
    superiority["safety_point_within_maximum"] = within_maximum
    superiority["safety_point_strictly_negative"] = strictly_negative
    superiority["safety_ci_upper_within_maximum"] = ci_ok
    superiority["zero_hmc_mismatches"] = overrides_ok
    superiority["achieved"] = bool(
        admissions_ok and within_maximum and strictly_negative and ci_ok and overrides_ok
    )
    return superiority


def _run_stage_b(
    output: Path,
    bundle: Any,
    roster: tuple[str, ...],
    family_split: Mapping[str, str],
    evaluation: Sequence[V4ModelSample],
    candidate_id: str,
    candidate_record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    model, _ = load_v4_model(output / "models" / f"{candidate_id}.json")

    print("Issue #56 V4 study v3 (stage B: HMC replay)", file=sys.stderr)
    if smoke:
        v3_samples_by_family: dict[str, list[Any]] = defaultdict(list)
        for family_id in roster:
            if family_id not in family_split:
                continue
            v3_samples_by_family[family_id] = list(
                collect_v3_family_samples(
                    bundle,
                    build_family_scenario(bundle.development_scenario, roster.index(family_id)),
                    family_id,
                    split=family_split[family_id],
                )
            )
        split = dict(family_split)
    else:
        split = v3_family_split(roster)
        v3_samples_by_family = {}
        for family_id in roster:
            v3_samples_by_family[family_id] = list(
                collect_v3_family_samples(
                    bundle,
                    build_family_scenario(bundle.development_scenario, roster.index(family_id)),
                    family_id,
                    split=split[family_id],
                )
            )
    v3_rows = [
        sample.to_mapping()
        for family_samples in v3_samples_by_family.values()
        for sample in family_samples
    ]
    v3_samples = list(load_v3_samples(v3_rows))
    v3_train = [sample for sample in v3_samples if sample.split == "TRAIN"]
    v3_validation = [sample for sample in v3_samples if sample.split == "VALIDATION"]
    if not v3_train or not v3_validation:
        raise V4StudyV3Error("V3 baseline refit lacks TRAIN or VALIDATION support")
    v3_model = V3RiskModel.fit(v3_train).calibrate(v3_validation)
    v3_model_file_sha256 = _write_json(output / "v3-baseline-model.json", v3_model.to_mapping())
    if not smoke and v3_model_file_sha256 != FROZEN_V3_MODEL_FILE_SHA256:
        raise V4StudyV3Error("V3 baseline refit diverged from the frozen V3 model artifact")

    point_model = load_live_mlp_model(POINT_ARTIFACT_PATH, expected_sha256=POINT_ARTIFACT_SHA256)
    evaluation_ids = tuple(sorted({sample.family_id for sample in evaluation}))
    records: list[V3EpisodeRecord] = []
    for family_id in evaluation_ids:
        family_index = roster.index(family_id)
        scenario = build_family_scenario(bundle.development_scenario, family_index)
        for arm in V4_MODEL_V3_STAGE_B_ARMS:
            record = run_v3_episode(
                bundle,
                scenario,
                arm,
                family_id,
                family_index,
                v3_model,
                point_model.predictor,
                v4_model=model if arm == V4_MODEL_ARM else None,
            )
            records.append(record)
            trace_path = output / "traces" / f"{arm}--{family_id}.json"
            trace_path.write_bytes(record.trace_canonical_bytes)
        print(f"  evaluation family {family_id} complete ({len(V4_MODEL_V3_STAGE_B_ARMS)} arms)", file=sys.stderr)

    episodes_sha256 = _write_jsonl(
        output / "episodes.jsonl", [record.to_mapping() for record in records]
    )
    aggregate = _aggregate_episodes(records)
    aggregate_sha256 = _write_json(output / "aggregate.json", aggregate)

    stage_b_spec = protocol["evaluation"]["stage_b_hmc_replay"]
    seed = int(stage_b_spec["bootstrap_seed"])
    resamples = int(stage_b_spec["bootstrap_resamples"])
    by_pair = {(record.arm, record.family_id): record for record in records}
    paired_vs_rules: dict[str, Any] = {}
    for arm in ("point_model_common_window", "risk_filtered_point_v3", V4_MODEL_ARM):
        paired_vs_rules[arm] = {
            metric: _paired_family_bootstrap(
                [
                    float(getattr(by_pair[(arm, family_id)], metric))
                    - float(getattr(by_pair[("rules_only_common_window", family_id)], metric))
                    for family_id in evaluation_ids
                ],
                seed=seed,
                resamples=resamples,
            )
            for metric in (
                "safety_exposure",
                "safety_violation_steps",
                "comfort_deviation",
                "resource_composite",
            )
        }
    paired_vs_v3 = {
        metric: _paired_family_bootstrap(
            [
                float(getattr(by_pair[(V4_MODEL_ARM, family_id)], metric))
                - float(getattr(by_pair[("risk_filtered_point_v3", family_id)], metric))
                for family_id in evaluation_ids
            ],
            seed=seed,
            resamples=resamples,
        )
        for metric in (
            "safety_exposure",
            "safety_violation_steps",
            "comfort_deviation",
            "resource_composite",
        )
    }

    v4_records = [record for record in records if record.arm == V4_MODEL_ARM]
    v3_records = [record for record in records if record.arm == "risk_filtered_point_v3"]
    admitted_v4 = sum(record.admitted_proposal_count for record in v4_records)
    admitted_v3 = sum(record.admitted_proposal_count for record in v3_records)
    mismatch_v4 = sum(record.hmc_mismatch_count for record in v4_records)
    mismatch_rate = mismatch_v4 / admitted_v4 if admitted_v4 else 0.0
    technical = {
        "authority_violation_count": sum(not record.authority_verified for record in records),
        "replay_failure_count": sum(not record.replay_verified for record in records),
        "provenance_violation_count": sum(not record.provenance_verified for record in records),
        "non_finite_metric_count": sum(not record.metrics_finite_verified for record in records),
        "proposal_admission_failure_count": sum(
            not record.proposal_admission_verified for record in records
        ),
    }
    safety_vs_rules = paired_vs_rules[V4_MODEL_ARM]["safety_exposure"]
    gates = {
        **{key: (value == 0) for key, value in technical.items()},
        "safety_vs_rules_only_point_and_ci_upper_nonpositive": bool(
            safety_vs_rules["point_difference"] <= 0.0 and safety_vs_rules["ci_upper"] <= 0.0
        ),
        "maximum_hmc_mismatch_rate": mismatch_rate <= 0.1,
        "maximum_inference_latency_p99_ms": bool(
            candidate_record["evaluation"]["inference_latency_p99_ms"] <= 250.0
        ),
        "hmc_mismatch_rate": mismatch_rate,
        "hmc_mismatch_count": mismatch_v4,
    }
    all_gates_passed = all(
        bool(value) if isinstance(value, bool) else float(value) <= 0.1
        for key, value in gates.items()
        if key not in {"hmc_mismatch_rate", "hmc_mismatch_count"}
    )
    superiority = _superiority_over_v3(
        stage_b_spec["superiority_over_v3"],
        paired_vs_v3["safety_exposure"],
        admitted_v4,
        admitted_v3,
        mismatch_v4,
    )
    return {
        "ran": True,
        "candidate_id": candidate_id,
        "arms": list(V4_MODEL_V3_STAGE_B_ARMS),
        "evaluation_family_ids": list(evaluation_ids),
        "v3_baseline_model_file_sha256": v3_model_file_sha256,
        "v3_baseline_matches_frozen_artifact": (not smoke)
        and v3_model_file_sha256 == FROZEN_V3_MODEL_FILE_SHA256,
        "point_artifact_sha256": point_model.artifact_sha256,
        "episodes_sha256": episodes_sha256,
        "aggregate_sha256": aggregate_sha256,
        "aggregate": aggregate,
        "paired_bootstrap_vs_rules": paired_vs_rules,
        "paired_bootstrap_v4_vs_v3": paired_vs_v3,
        "gates": gates,
        "all_stage_b_gates_passed": bool(all_gates_passed),
        "superiority_over_v3": superiority,
    }


if __name__ == "__main__":
    raise SystemExit(main())
