"""Bounded FP64-to-FP32 optimisation for the demo-only forecast model."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import platform
import time
from typing import Final
import zipfile

import numpy as np

from .live_demo import (
    LiveForecastModel,
    load_live_ridge_model,
    run_live_forecast_demo,
)

_SOURCE_SCHEMA: Final = "aeolus_habitat_v2_forecast_demo_model_v1"
_CANDIDATE_SCHEMA: Final = "aeolus_habitat_v2_forecast_demo_model_fp32_v1"
_RELEASE_TIER: Final = "DEMO_ONLY_PERMANENTLY_EXCLUDED"
_ARRAY_FIELDS: Final = ("feature_mean", "feature_scale", "target_mean", "coef")
_MODEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "release_tier",
        "actuator_authority",
        "alpha",
        "include_action",
        *_ARRAY_FIELDS,
        "window_steps",
        "horizon_steps",
        "input_manifest_sha256",
        "target_manifest_sha256",
    }
)


class ArmOptimizationError(ValueError):
    """The bounded reduced-precision optimisation contract was violated."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _compressed_model_npz(values: dict[str, np.ndarray]) -> bytes:
    """Write canonical NPY members; DEFLATE container bytes are platform-dependent."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(values):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, values[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, payload.getvalue(), compresslevel=9)
    return buffer.getvalue()


def optimise_ridge_fp32(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_sha256: str,
) -> dict[str, object]:
    """Convert only the numeric model arrays to FP32 and record exact reduction."""
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ArmOptimizationError("source and candidate paths must differ")
    try:
        source_raw = source_path.read_bytes()
    except OSError as error:
        raise ArmOptimizationError("source model is unreadable") from error
    source_sha256 = _sha256(source_raw)
    if source_sha256 != expected_source_sha256:
        raise ArmOptimizationError(
            "source model SHA-256 does not match the frozen input"
        )

    try:
        with np.load(io.BytesIO(source_raw), allow_pickle=False) as archive:
            if set(archive.files) != _MODEL_FIELDS:
                raise ArmOptimizationError("source model fields drift")
            values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except ArmOptimizationError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ArmOptimizationError("source model archive is malformed") from error

    if (
        str(values["schema_version"].item()) != _SOURCE_SCHEMA
        or str(values["release_tier"].item()) != _RELEASE_TIER
        or bool(values["actuator_authority"].item()) is not False
        or bool(values["include_action"].item()) is not True
        or any(values[name].dtype != np.float64 for name in _ARRAY_FIELDS)
        or any(not np.isfinite(values[name]).all() for name in _ARRAY_FIELDS)
    ):
        raise ArmOptimizationError("source model contract or precision is invalid")

    source_raw_array_bytes = sum(int(values[name].nbytes) for name in _ARRAY_FIELDS)
    values["schema_version"] = np.asarray(_CANDIDATE_SCHEMA)
    for name in _ARRAY_FIELDS:
        values[name] = values[name].astype(np.float32)
    candidate_raw_array_bytes = sum(int(values[name].nbytes) for name in _ARRAY_FIELDS)
    if candidate_raw_array_bytes * 2 != source_raw_array_bytes:
        raise ArmOptimizationError("FP32 candidate did not halve raw model-array bytes")

    candidate_raw = _compressed_model_npz(values)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(candidate_raw)

    return {
        "schema_version": "aeolus_habitat_v2_forecast_fp32_optimisation_receipt_v1",
        "release_tier": _RELEASE_TIER,
        "actuator_authority": False,
        "source_precision": "float64",
        "candidate_precision": "float32",
        "source_model_sha256": source_sha256,
        "source_model_file_bytes": len(source_raw),
        "source_raw_array_bytes": source_raw_array_bytes,
        "candidate_model_sha256": _sha256(candidate_raw),
        "candidate_model_file_bytes": len(candidate_raw),
        "candidate_raw_array_bytes": candidate_raw_array_bytes,
        "candidate_raw_array_bytes_reduction_fraction": 0.5,
        "quality_gate": "normalised_prediction_drift_lte_1e-4",
        "quality_gate_evaluated": False,
        "native_arm64_benchmark_evaluated": False,
        "qualified_model": False,
        "production_deployed": False,
    }


class _CapturePredictor:
    def __init__(self, predictor: object) -> None:
        self.predictor = predictor
        self.calls: list[tuple[object, np.ndarray]] = []

    def predict(self, history: object, proposed_action_f32: np.ndarray) -> np.ndarray:
        action = np.array(proposed_action_f32, dtype=np.float32, copy=True)
        action.setflags(write=False)
        self.calls.append((history, action))
        return self.predictor.predict(history, proposed_action_f32)


def _model_raw_array_bytes(model: LiveForecastModel) -> int:
    predictor = model.predictor
    return sum(
        int(getattr(predictor, name).nbytes)
        for name in ("feature_mean", "feature_scale", "target_mean", "coef")
    )


def _timing_summary(values: list[int]) -> dict[str, object]:
    durations = np.asarray(values, dtype=np.int64)
    median_ns = int(np.median(durations))
    p95_ns = int(np.percentile(durations, 95))
    return {
        "sample_count": int(durations.size),
        "median_ns": median_ns,
        "p95_ns": p95_ns,
        "minimum_ns": int(durations.min()),
        "maximum_ns": int(durations.max()),
        "predictions_per_second_at_median": float(1_000_000_000 / median_ns),
    }


def benchmark_fp64_vs_fp32(
    repo_root: str | Path,
    source: str | Path,
    candidate: str | Path,
    *,
    expected_source_sha256: str,
    expected_candidate_sha256: str,
    warmup_iterations: int = 20,
    measured_iterations: int = 200,
) -> dict[str, object]:
    """Compare exact FP64 and FP32 models on identical live-demo inputs."""
    if warmup_iterations < 1 or measured_iterations < 4:
        raise ArmOptimizationError("benchmark iteration counts are too small")
    root = Path(repo_root).resolve()
    source_path = Path(source)
    candidate_path = Path(candidate)
    try:
        source_raw = source_path.read_bytes()
        candidate_raw = candidate_path.read_bytes()
    except OSError as error:
        raise ArmOptimizationError("benchmark model artifact is unreadable") from error
    fp64_model = load_live_ridge_model(
        source_raw,
        expected_sha256=expected_source_sha256,
    )
    fp32_model = load_live_ridge_model(
        candidate_raw,
        expected_sha256=expected_candidate_sha256,
    )
    if (
        fp64_model.model_kind != "action_aware_ridge"
        or fp32_model.model_kind != "action_aware_ridge_fp32"
    ):
        raise ArmOptimizationError("benchmark model precision identities drift")

    capture = _CapturePredictor(fp64_model.predictor)
    captured_model = LiveForecastModel(
        predictor=capture,
        model_kind=fp64_model.model_kind,
        artifact_sha256=fp64_model.artifact_sha256,
        actuator_authority=False,
    )
    fp64_result = run_live_forecast_demo(
        root,
        captured_model,
        selected_action_id="normal-occupied-v1",
    )
    fp32_result = run_live_forecast_demo(
        root,
        fp32_model,
        selected_action_id="normal-occupied-v1",
    )
    if (
        len(capture.calls) != 4
        or tuple(item.action_id for item in fp64_result.candidate_forecasts)
        != tuple(item.action_id for item in fp32_result.candidate_forecasts)
        or fp64_result.selected_action_id != fp32_result.selected_action_id
        or not np.array_equal(fp64_result.truth_f32, fp32_result.truth_f32)
        or fp64_result.hmc_is_sole_actuator_authority is not True
        or fp32_result.hmc_is_sole_actuator_authority is not True
    ):
        raise ArmOptimizationError("live comparison workload or authority drifted")

    fp64_predictions = np.stack(
        [item.prediction_f32 for item in fp64_result.candidate_forecasts]
    ).astype(np.float64)
    fp32_predictions = np.stack(
        [item.prediction_f32 for item in fp32_result.candidate_forecasts]
    ).astype(np.float64)
    difference = np.abs(fp32_predictions - fp64_predictions)
    normaliser = np.maximum(np.abs(fp64_predictions), 1.0)
    normalised = difference / normaliser
    maximum_normalised_drift = float(normalised.max(initial=0.0))
    mean_normalised_drift = float(normalised.mean())
    maximum_absolute_drift = float(difference.max(initial=0.0))
    parity_passed = bool(
        np.isfinite(fp32_predictions).all() and maximum_normalised_drift <= 1e-4
    )

    methods = (
        ("fp64", fp64_model.predictor),
        ("fp32", fp32_model.predictor),
    )
    checksum = 0.0
    for iteration in range(warmup_iterations):
        order = methods if iteration % 2 == 0 else tuple(reversed(methods))
        history, action = capture.calls[iteration % len(capture.calls)]
        for _, predictor in order:
            checksum += float(predictor.predict(history, action)[0, 0])

    measurements: dict[str, list[int]] = {"fp64": [], "fp32": []}
    for iteration in range(measured_iterations):
        order = methods if iteration % 2 == 0 else tuple(reversed(methods))
        history, action = capture.calls[iteration % len(capture.calls)]
        for name, predictor in order:
            started_ns = time.perf_counter_ns()
            prediction = predictor.predict(history, action)
            elapsed_ns = time.perf_counter_ns() - started_ns
            if elapsed_ns <= 0 or not np.isfinite(prediction).all():
                raise ArmOptimizationError("benchmark prediction or timing is invalid")
            measurements[name].append(elapsed_ns)
            checksum += float(prediction[0, 0])
    if not np.isfinite(checksum):
        raise ArmOptimizationError("benchmark checksum is non-finite")

    fp64_timing = _timing_summary(measurements["fp64"])
    fp32_timing = _timing_summary(measurements["fp32"])
    fp64_file_bytes = len(source_raw)
    fp32_file_bytes = len(candidate_raw)
    fp64_array_bytes = _model_raw_array_bytes(fp64_model)
    fp32_array_bytes = _model_raw_array_bytes(fp32_model)
    machine = platform.machine().lower()
    native_arm64 = machine in {"aarch64", "arm64"}

    return {
        "schema_version": "aeolus_habitat_v2_fp64_fp32_benchmark_receipt_v1",
        "evidence_role": "demo_only_reduced_precision_benchmark",
        "release_tier": _RELEASE_TIER,
        "environment": {
            "platform_machine": machine,
            "native_arm64": native_arm64,
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "workload": {
            "source": "one real Habitat V2 live-demo history",
            "selected_action_id": "normal-occupied-v1",
            "candidate_action_count": 4,
            "prediction_shape": [8, 51],
            "batch_size": 1,
            "warmup_iterations_per_model": warmup_iterations,
            "measured_iterations_per_model": measured_iterations,
            "interleaved_and_order_alternated": True,
        },
        "models": {
            "fp64": {
                "precision": "float64",
                "sha256": expected_source_sha256,
                "file_bytes": fp64_file_bytes,
                "raw_array_bytes": fp64_array_bytes,
            },
            "fp32": {
                "precision": "float32",
                "sha256": expected_candidate_sha256,
                "file_bytes": fp32_file_bytes,
                "raw_array_bytes": fp32_array_bytes,
            },
        },
        "prediction_parity": {
            "gate": "max_abs_drift_div_max_abs_reference_or_one_lte_1e-4",
            "threshold": 1e-4,
            "maximum_normalised_drift": maximum_normalised_drift,
            "mean_normalised_drift": mean_normalised_drift,
            "maximum_absolute_drift": maximum_absolute_drift,
            "passed": parity_passed,
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "scope": "feature_flattening_plus_standardisation_plus_ridge_matmul_plus_output_cast",
            "fp64": fp64_timing,
            "fp32": fp32_timing,
            "median_speedup_fp64_over_fp32": float(
                fp64_timing["median_ns"] / fp32_timing["median_ns"]
            ),
            "p95_speedup_fp64_over_fp32": float(
                fp64_timing["p95_ns"] / fp32_timing["p95_ns"]
            ),
        },
        "reductions": {
            "raw_array_bytes_fraction_reduced": float(
                1.0 - fp32_array_bytes / fp64_array_bytes
            ),
            "file_bytes_fraction_reduced": float(
                1.0 - fp32_file_bytes / fp64_file_bytes
            ),
        },
        "claims": {
            "reduced_precision_model": True,
            "prediction_parity_gate_passed": parity_passed,
            "raw_array_memory_reduced": fp32_array_bytes < fp64_array_bytes,
            "artifact_file_bytes_reduced": fp32_file_bytes < fp64_file_bytes,
            "native_arm64_timing_measured": native_arm64,
            "median_latency_improved_on_native_arm64": bool(
                native_arm64 and fp32_timing["median_ns"] < fp64_timing["median_ns"]
            ),
            "p95_latency_improved_on_native_arm64": bool(
                native_arm64 and fp32_timing["p95_ns"] < fp64_timing["p95_ns"]
            ),
            "arm_specific_operator_optimisation": False,
            "actuator_authority": False,
            "qualified_model": False,
            "production_deployed": False,
        },
    }
