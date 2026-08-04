"""Reproducible ONNX latency receipts with explicit hardware claim boundaries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from aeolus.baseline import RuleBaseline
from aeolus.detector import FEATURE_WIDTH, WINDOW_TICKS


def benchmark_onnx(
    model_path: str | Path,
    windows: Sequence[Sequence[Sequence[float]]] | NDArray[np.float32],
    *,
    warmup_iterations: int = 50,
    measured_iterations: int = 500,
    batch_size: int = 1,
    intra_op_threads: int = 1,
    declared_arm_target: bool = False,
) -> dict[str, object]:
    """Measure one ONNX model and state exactly what the host can support."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ONNX benchmark requires the 'ml' project extra") from exc
    for name, value in (
        ("warmup_iterations", warmup_iterations),
        ("measured_iterations", measured_iterations),
        ("batch_size", batch_size),
        ("intra_op_threads", intra_op_threads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"benchmark {name} must be a positive integer")
    model = Path(model_path)
    if not model.is_file():
        raise ValueError(f"benchmark model not found: {model}")
    tensor = _validated_windows(windows)
    if len(tensor) < batch_size:
        repeats = math.ceil(batch_size / len(tensor))
        tensor = np.tile(tensor, (repeats, 1, 1))
    batch = np.ascontiguousarray(tensor[:batch_size], dtype=np.float32)
    machine = platform.machine().lower()
    is_arm_host = machine in {"aarch64", "arm64"}
    if declared_arm_target and not is_arm_host:
        raise ValueError("cannot declare an Arm target benchmark on a non-AArch64 host")

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    for _ in range(warmup_iterations):
        session.run(output_names, {input_name: batch})
    timings_ms: list[float] = []
    for _ in range(measured_iterations):
        start = time.perf_counter_ns()
        session.run(output_names, {input_name: batch})
        elapsed_ns = time.perf_counter_ns() - start
        timings_ms.append(elapsed_ns / 1_000_000.0)
    median_ms = float(statistics.median(timings_ms))
    p95_ms = float(np.percentile(np.asarray(timings_ms), 95))
    if median_ms <= 0.0 or p95_ms <= 0.0:
        raise ValueError("benchmark timer resolution produced a non-positive latency")
    arm_claim = bool(declared_arm_target and is_arm_host)
    return {
        "schema_version": "aeolus_onnx_benchmark_v1",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "model_bytes": model.stat().st_size,
        "input_sha256": hashlib.sha256(batch.tobytes(order="C")).hexdigest(),
        "input_shape": list(batch.shape),
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": 1,
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "latency_ms": {
            "median": median_ms,
            "p95": p95_ms,
            "minimum": min(timings_ms),
            "maximum": max(timings_ms),
        },
        "throughput_windows_per_second": batch_size / (median_ms / 1000.0),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_boundary": "process lifetime maximum; not isolated model allocation",
        "arm_performance_claim": arm_claim,
        "claim_scope": (
            "declared_arm_target_measurement" if arm_claim else "local_readiness_only"
        ),
    }


def benchmark_rule(
    rule: RuleBaseline,
    windows: Sequence[Sequence[Sequence[float]]] | NDArray[np.float32],
    *,
    scenario_sha256: str,
    warmup_iterations: int = 50,
    measured_iterations: int = 500,
    declared_arm_target: bool = False,
) -> dict[str, object]:
    """Measure the streaming rule baseline under the same claim boundary."""
    if not isinstance(rule, RuleBaseline):
        raise ValueError("rule benchmark requires a RuleBaseline")
    for name, value in (
        ("warmup_iterations", warmup_iterations),
        ("measured_iterations", measured_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"benchmark {name} must be a positive integer")
    _validate_sha256(scenario_sha256)
    tensor = _validated_windows(windows)
    benchmark_windows = tensor.tolist()
    machine = platform.machine().lower()
    is_arm_host = machine in {"aarch64", "arm64"}
    if declared_arm_target and not is_arm_host:
        raise ValueError("cannot declare an Arm target benchmark on a non-AArch64 host")
    rule.reset()
    for index in range(warmup_iterations):
        rule.label_window(benchmark_windows[index % len(benchmark_windows)])
    rule.reset()
    timings_ms: list[float] = []
    for index in range(measured_iterations):
        window = benchmark_windows[index % len(benchmark_windows)]
        start = time.perf_counter_ns()
        rule.label_window(window)
        timings_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    median_ms = float(statistics.median(timings_ms))
    p95_ms = float(np.percentile(np.asarray(timings_ms), 95))
    if median_ms <= 0.0 or p95_ms <= 0.0:
        raise ValueError("benchmark timer resolution produced a non-positive latency")
    arm_claim = bool(declared_arm_target and is_arm_host)
    benchmark_input = np.ascontiguousarray(tensor, dtype=np.float32)
    return {
        "schema_version": "aeolus_rule_benchmark_v1",
        "method": "calibrated_rules",
        "scenario_sha256": scenario_sha256,
        "rule_parameters": rule.parameters.as_dict(),
        "input_sha256": hashlib.sha256(
            benchmark_input.tobytes(order="C")
        ).hexdigest(),
        "input_shape": list(benchmark_input.shape),
        "batch_size": 1,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "latency_ms": {
            "median": median_ms,
            "p95": p95_ms,
            "minimum": min(timings_ms),
            "maximum": max(timings_ms),
        },
        "throughput_windows_per_second": 1.0 / (median_ms / 1000.0),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_boundary": "process lifetime maximum; not isolated rule allocation",
        "arm_performance_claim": arm_claim,
        "claim_scope": (
            "declared_arm_target_measurement" if arm_claim else "local_readiness_only"
        ),
    }


def _validate_sha256(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("scenario_sha256 must be a lowercase SHA-256 digest")


def _validated_windows(
    windows: Sequence[Sequence[Sequence[float]]] | NDArray[np.float32],
) -> NDArray[np.float32]:
    try:
        tensor = np.asarray(windows)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark windows must be a rectangular numeric array") from exc
    if (
        tensor.ndim != 3
        or tensor.shape[0] < 1
        or tensor.shape[1:] != (WINDOW_TICKS, FEATURE_WIDTH)
    ):
        raise ValueError(
            f"benchmark windows must have shape [N,{WINDOW_TICKS},{FEATURE_WIDTH}]"
        )
    if tensor.dtype == np.bool_ or not np.issubdtype(tensor.dtype, np.number):
        raise ValueError("benchmark windows must be numeric")
    result = tensor.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("benchmark windows must be finite")
    return result
