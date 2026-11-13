"""Benchmark the Issue #91 batched air-network candidate against the reference.

Compares the research-only closed-form kernels (NumPy FP64, and optionally
eager PyTorch FP32/FP64 on CPU or CUDA) with the reviewed scalar bisection
solver and an independent high-precision root, applies the per-variable
tolerances frozen in ``contracts/habitat_v2_air_network_batched_v1.json`` and
writes one write-once receipt under the ignored ``out/`` directory.

A kernel timing is not an end-to-end simulator speedup, a model improvement or
a control benefit. No GPU speedup is assumed: a crossover is reported only when
it was measured, and CUDA results exist only when a CUDA device actually ran.
The script cannot verify what hardware it runs on; hardware and image identity
are operator declarations and are labelled as such.

Exit status: 0 when every gated comparison passed, 1 when the receipt was
written but a gate failed, 2 when the benchmark could not run or could not be
recorded (no receipt is left behind).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import numpy as np

from aeolus.habitat_v2 import air_network as reference_module
from aeolus.habitat_v2 import air_network_analytic as analytic_module
from aeolus.habitat_v2 import air_network_batched_torch as torch_module
from aeolus.habitat_v2.air_network import (
    AirNetworkSpec,
    BranchSpec,
    FanSpec,
    solve_air_network,
)
from aeolus.habitat_v2.air_network_analytic import (
    CONTRACT_RELATIVE_PATH,
    AirNetworkBatch,
    AirNetworkBatchResult,
    air_network_batch_sha256,
    air_network_operating_point_numpy,
    compare_results,
    generate_air_network_cases,
    load_air_network_batched_contract,
    solve_air_network_analytic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "out"
RECEIPT_NAME = "benchmark-receipt.json"
RECEIPT_SCHEMA_VERSION = "aeolus_habitat_v2_air_network_batched_benchmark_receipt_v1"
DEFAULT_BATCH_SIZES = (1, 8, 32, 128, 512, 2048)
TIMING_PROFILE = "fixture_band"
STRESS_PROFILE = "supported_domain_stress"
HIGH_PRECISION_WORLDS = 128
SCRIPT_RELATIVE_PATH = "scripts/benchmark_issue91_air_network_batched.py"
EXECUTED_MODULES = (
    ("src/aeolus/habitat_v2/air_network.py", reference_module),
    ("src/aeolus/habitat_v2/air_network_analytic.py", analytic_module),
    ("src/aeolus/habitat_v2/air_network_batched_torch.py", torch_module),
)
SOURCE_PATHS = (
    *(relative for relative, _ in EXECUTED_MODULES),
    SCRIPT_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH,
)
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_HARDWARE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")
TIMING_CONDITIONS = ("dedicated_host", "shared_host_provisional")


class AirNetworkBenchmarkError(RuntimeError):
    """Raised when the benchmark cannot proceed safely."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _resolve_output(output_path: Path, out_root: Path) -> Path:
    """Containment and write-once checks; does not create anything."""

    root = out_root.resolve()
    output = output_path.resolve()
    try:
        relative = output.relative_to(root)
    except ValueError as error:
        raise AirNetworkBenchmarkError(
            "benchmark output must live under the ignored out/ directory"
        ) from error
    if not relative.parts:
        raise AirNetworkBenchmarkError("benchmark output must be a new directory below out/")
    if output.exists():
        raise AirNetworkBenchmarkError("refusing to overwrite existing output directory")
    ancestor = output.parent
    while ancestor != root:
        if (ancestor / RECEIPT_NAME).exists():
            raise AirNetworkBenchmarkError(
                "refusing to write inside an existing receipt directory"
            )
        ancestor = ancestor.parent
    return output


def _git_output(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "--no-optional-locks", *arguments),
            cwd=repo_root, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_identity(repo_root: Path) -> dict[str, Any]:
    """Bind the receipt to the code that is actually imported and measured.

    Fails unless every measured module was imported from ``repo_root``, so a
    receipt can never name one checkout while an installed wheel or another
    path supplied the kernels. File identities ignore checkout line endings.
    """

    for relative, module in EXECUTED_MODULES:
        if Path(module.__file__).resolve() != (repo_root / relative).resolve():
            raise AirNetworkBenchmarkError(
                f"measured module {module.__name__} was not imported from the "
                f"checkout being recorded ({relative}); run from a source checkout"
            )
    if (repo_root / SCRIPT_RELATIVE_PATH).resolve() != Path(__file__).resolve():
        raise AirNetworkBenchmarkError(
            "benchmark script is not the one in the checkout being recorded"
        )
    files = []
    for relative in SOURCE_PATHS:
        normalised = (repo_root / relative).read_bytes().replace(b"\r\n", b"\n")
        files.append(
            {
                "relative_path": relative,
                "byte_length_lf_normalised": len(normalised),
                "sha256_lf_normalised": hashlib.sha256(normalised).hexdigest(),
                "git_blob_sha1": hashlib.sha1(
                    f"blob {len(normalised)}\0".encode("ascii") + normalised
                ).hexdigest(),
            }
        )
    status = _git_output(repo_root, "status", "--porcelain")
    return {
        "source_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "source_worktree_dirty": None if status is None else bool(status),
        "executed_modules_verified_from_checkout": True,
        "source_files": files,
    }


def _reference_points(
    batch: AirNetworkBatch,
) -> list[tuple[AirNetworkSpec, float, dict[str, float]]]:
    points = []
    for world in range(batch.batch_size):
        branches = tuple(
            BranchSpec(
                zone_id=f"zone_{index:02d}",
                damper_id=f"damper_{index:02d}",
                open_supply_resistance_pa_s2_m6=float(
                    batch.open_supply_resistance_pa_s2_m6[world, index]
                ),
                return_resistance_pa_s2_m6=float(
                    batch.return_resistance_pa_s2_m6[world, index]
                ),
                damper_leak_fraction=float(batch.damper_leak_fraction[world, index]),
            )
            for index in range(batch.branch_count)
        )
        spec = AirNetworkSpec(
            fan=FanSpec(
                component_id="fan",
                rated_free_delivery_m3_s=float(batch.rated_free_delivery_m3_s[world]),
                rated_shutoff_pressure_pa=float(batch.rated_shutoff_pressure_pa[world]),
                total_efficiency=float(batch.total_efficiency[world]),
            ),
            shared_resistance_pa_s2_m6=float(batch.shared_resistance_pa_s2_m6[world]),
            air_density_kg_m3=float(batch.air_density_kg_m3[world]),
            branches=branches,
        )
        dampers = {
            f"damper_{index:02d}": float(batch.damper_position_fraction[world, index])
            for index in range(batch.branch_count)
        }
        points.append((spec, float(batch.fan_speed_fraction[world]), dampers))
    return points


def _solve_reference(
    points: Sequence[tuple[AirNetworkSpec, float, dict[str, float]]],
) -> AirNetworkBatchResult:
    """One reviewed scalar bisection solve per world, packed in batch layout."""

    rows = []
    for spec, speed, dampers in points:
        result = solve_air_network(
            spec, fan_speed_fraction=speed, damper_position_by_id=dampers
        )
        zones = sorted(result.zone_flow_m3_s)
        rows.append(
            (
                result.fan_pressure_rise_pa,
                result.shared_pressure_loss_pa,
                result.branch_pressure_loss_pa[zones[0]],
                result.total_flow_m3_s,
                result.fan_air_power_w,
                result.fan_electrical_power_w,
                result.operating_point_residual_pa,
                [result.zone_flow_m3_s[zone] for zone in zones],
                [result.zone_mass_flow_kg_s[zone] for zone in zones],
            )
        )
    return AirNetworkBatchResult(
        *(np.array(column, dtype=np.float64) for column in zip(*rows))
    )


def _high_precision_root(batch: AirNetworkBatch) -> AirNetworkBatchResult:
    """80-digit decimal bisection of the implicit fan/system equation.

    Solves ``P_fan(Q(p_b), s) - R_shared * Q(p_b)**2 - p_b = 0`` with
    ``q_i(p_b) = sqrt(p_b / R_i)`` directly; it does not use the
    equivalent-resistance closed form that the candidate implements.
    """

    rows = []
    with localcontext() as context:
        context.prec = 80
        for world in range(batch.batch_size):
            speed = Decimal(float(batch.fan_speed_fraction[world]))
            free = Decimal(float(batch.rated_free_delivery_m3_s[world])) * speed
            shutoff = Decimal(float(batch.rated_shutoff_pressure_pa[world])) * speed * speed
            shared = Decimal(float(batch.shared_resistance_pa_s2_m6[world]))
            density = Decimal(float(batch.air_density_kg_m3[world]))
            efficiency = Decimal(float(batch.total_efficiency[world]))
            inverse_roots = []
            for index in range(batch.branch_count):
                leak = Decimal(float(batch.damper_leak_fraction[world, index]))
                position = Decimal(float(batch.damper_position_fraction[world, index]))
                area = leak + (1 - leak) * position
                resistance = Decimal(
                    float(batch.open_supply_resistance_pa_s2_m6[world, index])
                ) / (area * area) + Decimal(
                    float(batch.return_resistance_pa_s2_m6[world, index])
                )
                inverse_roots.append(1 / resistance.sqrt())
            if speed == 0:
                rows.append((0.0,) * 7 + ([0.0] * batch.branch_count,) * 2)
                continue
            low, high = Decimal(0), shutoff
            for _ in range(420):
                middle = (low + high) / 2
                total = middle.sqrt() * sum(inverse_roots)
                fan = shutoff * max(Decimal(0), 1 - (total / free) ** 2)
                if fan - shared * total * total - middle > 0:
                    low = middle
                else:
                    high = middle
            pressure = (low + high) / 2
            flows = [pressure.sqrt() * inverse for inverse in inverse_roots]
            total = sum(flows)
            fan = shutoff * max(Decimal(0), 1 - (total / free) ** 2)
            rows.append(
                (
                    float(fan),
                    float(shared * total * total),
                    float(pressure),
                    float(total),
                    float(fan * total),
                    float(fan * total / efficiency),
                    0.0,
                    [float(flow) for flow in flows],
                    [float(flow * density) for flow in flows],
                )
            )
    return AirNetworkBatchResult(
        *(np.array(column, dtype=np.float64) for column in zip(*rows))
    )


def _take(batch: AirNetworkBatch, count: int) -> AirNetworkBatch:
    return AirNetworkBatch(
        **{
            name: np.ascontiguousarray(getattr(batch, name)[:count])
            for name in batch.__dataclass_fields__
        }
    )


def _summarise(samples_ns: Sequence[int], batch_size: int) -> dict[str, Any]:
    ordered = np.sort(np.array(samples_ns, dtype=np.float64))
    median = float(np.percentile(ordered, 50))
    return {
        "sample_count": int(ordered.size),
        "minimum_ns": float(ordered[0]),
        "median_ns": median,
        "p95_ns": float(np.percentile(ordered, 95)),
        "p99_ns": float(np.percentile(ordered, 99)),
        "maximum_ns": float(ordered[-1]),
        "mean_ns": float(ordered.mean()),
        "worlds_per_second_at_median": batch_size / (median * 1e-9) if median > 0 else None,
    }


def _time_call(
    call: Callable[[], object],
    *,
    warmup_iterations: int,
    measured_iterations: int,
    synchronize: Callable[[], None] | None = None,
) -> list[int]:
    """Warm up, then time each call; device work is synchronized inside the clock."""

    for _ in range(warmup_iterations):
        call()
    samples = []
    for _ in range(measured_iterations):
        if synchronize is not None:
            synchronize()  # nothing from an earlier call may spill into this sample
        start = time.perf_counter_ns()
        call()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter_ns() - start)
    return samples


def _result_bytes(result: AirNetworkBatchResult) -> int:
    return int(
        sum(
            np.asarray(getattr(result, name)).nbytes
            for name in result.__dataclass_fields__
        )
    )


def _all_passed(report: dict[str, dict[str, Any]]) -> bool:
    return bool(report) and all(entry["passed"] is True for entry in report.values())


def _host_peak_bytes(call: Callable[[], object]) -> int:
    """tracemalloc peak attributable to one call, leaving caller tracing intact."""

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return max(int(peak - baseline), 0)


def _torch_comparison_id(precision: str) -> str:
    return "torch_fp32_vs_numpy_fp64" if precision == "fp32" else "torch_fp64_vs_numpy_fp64"


def _torch_measurements(
    batch: AirNetworkBatch,
    numpy_result: AirNetworkBatchResult,
    tolerances: dict[str, Any],
    *,
    device: str,
    precision: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    torch = torch_module.require_torch()
    is_cuda = device != "cpu"
    synchronize = (lambda: torch.cuda.synchronize(device)) if is_cuda else None
    comparison_id = _torch_comparison_id(precision)

    def transfer_inclusive() -> AirNetworkBatchResult:
        return torch_module.solve_air_network_torch(batch, precision=precision, device=device)

    result = transfer_inclusive()
    accuracy = compare_results(
        result, numpy_result, batch, tolerances[comparison_id]["variables"]
    )
    output_bytes = _result_bytes(result)
    del result

    def kernel_only_samples() -> list[int]:
        # Staged tensors live only in this scope, so they are released before
        # the transfer-inclusive timing and the peak-memory measurement.
        staged = torch_module.to_torch_inputs(batch, precision=precision, device=device)

        def kernel_only() -> object:
            with torch.no_grad():
                return torch_module.air_network_operating_point_torch(staged)

        return _time_call(
            kernel_only,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            synchronize=synchronize,
        )

    kernel_samples = kernel_only_samples()
    transfer_samples = _time_call(
        transfer_inclusive,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        synchronize=synchronize,
    )
    if is_cuda:
        torch.cuda.synchronize(device)
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        transfer_inclusive()
        torch.cuda.synchronize(device)
        peak_memory: dict[str, Any] = {
            "peak_device_bytes": max(
                int(torch.cuda.max_memory_allocated(device)) - baseline, 0
            ),
            "method": "torch.cuda.max_memory_allocated above the allocation held before one transfer-inclusive solve",
        }
    else:
        peak_memory = {
            "peak_device_bytes": None,
            "method": "not measured: PyTorch CPU allocations are not visible to tracemalloc",
        }
    return {
        "comparison_id": comparison_id,
        "accuracy_vs_numpy_fp64": accuracy,
        "accuracy_passed": _all_passed(accuracy),
        "kernel_only": _summarise(kernel_samples, batch.batch_size),
        "transfer_inclusive": _summarise(transfer_samples, batch.batch_size),
        "peak_memory": peak_memory,
        "output_bytes": output_bytes,
    }


# Like-for-like baselines: a kernel is compared with a kernel, and a complete
# validated host-to-host solve with a complete validated host-to-host solve.
_CROSSOVER_BASELINE = {"kernel_only": "kernel_only", "transfer_inclusive": "validated"}


def _crossover(
    grid: Sequence[dict[str, Any]], candidate_key: str, route: str
) -> dict[str, Any]:
    """Where, in the tested grid, a Torch route's median beats NumPy FP64's.

    Medians are compared without a noise margin, so a status describes this
    run's measurements; it is not a statistical claim.
    """

    baseline_route = _CROSSOVER_BASELINE[route]
    compared = []
    for entry in sorted(grid, key=lambda item: item["batch_size"]):
        candidate = entry["torch"].get(candidate_key)
        if candidate is None:
            continue
        compared.append(
            {
                "batch_size": entry["batch_size"],
                "candidate_median_ns": candidate[route]["median_ns"],
                "numpy_fp64_median_ns": entry["numpy_fp64"][baseline_route]["median_ns"],
            }
        )
    outcome: dict[str, Any] = {
        "numpy_fp64_baseline_route": baseline_route,
        "first_faster_batch_size": None,
        "faster_batch_sizes": [],
        "compared": compared,
    }
    if not compared:
        return {**outcome, "status": "not_evaluated"}
    flags = [row["candidate_median_ns"] < row["numpy_fp64_median_ns"] for row in compared]
    faster = [row["batch_size"] for row, flag in zip(compared, flags) if flag]
    outcome["faster_batch_sizes"] = faster
    if not faster:
        return {**outcome, "status": "no_crossover_in_tested_grid"}
    if all(flags):
        return {**outcome, "status": "faster_at_every_tested_size"}
    if flags == sorted(flags):  # slower at small sizes, then faster for the rest
        return {
            **outcome,
            "status": "measured_crossover",
            "first_faster_batch_size": faster[0],
        }
    return {**outcome, "status": "mixed_no_clean_crossover"}


def _cuda_receipt_prerequisites(
    *,
    cuda_devices: Sequence[str],
    batch_sizes: Sequence[int],
    precisions: Sequence[str],
    image_digest: str | None,
    hardware_label: str | None,
    source_worktree_dirty: bool | None,
    source_commit: str | None,
    gates_passed: bool,
    timing_conditions: str,
) -> list[str]:
    """Requirements this run misses for use as a hardware receipt (empty = none).

    Meeting them all does not prove the hardware identity: that remains an
    operator declaration to be checked by a reviewer against ``device_name``.
    """

    checks = (
        ("no CUDA device was measured", not cuda_devices),
        (
            "the default batch grid was not fully measured",
            not set(DEFAULT_BATCH_SIZES) <= set(batch_sizes),
        ),
        ("both precisions were not measured", set(precisions) != {"fp32", "fp64"}),
        ("no container image digest was declared", not image_digest),
        ("no hardware label was declared", not hardware_label),
        ("source commit is unknown", not source_commit),
        ("source worktree is dirty or its state is unknown", source_worktree_dirty is not False),
        ("an accuracy gate failed", not gates_passed),
        ("timings were taken on a shared host", timing_conditions != "dedicated_host"),
    )
    return [reason for reason, missing in checks if missing]


def run_benchmark(
    *,
    repo_root: Path = REPO_ROOT,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    branch_count: int = 8,
    seed: int = 91,
    warmup_iterations: int = 20,
    measured_iterations: int = 200,
    reference_iterations: int = 5,
    torch_devices: Sequence[str] = (),
    precisions: Sequence[str] = ("fp32", "fp64"),
    torch_num_threads: int | None = None,
    image_digest: str | None = None,
    hardware_label: str | None = None,
    timing_conditions: str = "shared_host_provisional",
) -> dict[str, Any]:
    """Measure accuracy and latency; return the receipt mapping (not written)."""

    if not batch_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in batch_sizes
    ) or len(set(batch_sizes)) != len(batch_sizes):
        raise AirNetworkBenchmarkError("batch sizes must be distinct positive integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (warmup_iterations, measured_iterations, reference_iterations)
    ) or warmup_iterations < 1 or measured_iterations < 4 or reference_iterations < 4:
        raise AirNetworkBenchmarkError(
            "warmup must be >= 1 and measured and reference iterations >= 4"
        )
    if len(set(torch_devices)) != len(torch_devices) or not set(precisions) <= {"fp32", "fp64"}:
        raise AirNetworkBenchmarkError("torch devices must be distinct and precisions fp32/fp64")
    if len(set(precisions)) != len(precisions) or (torch_devices and not precisions):
        raise AirNetworkBenchmarkError(
            "precisions must be distinct and at least one precision is required with a torch device"
        )
    if torch_num_threads is not None and (
        isinstance(torch_num_threads, bool)
        or not isinstance(torch_num_threads, int)
        or torch_num_threads < 1
        or not torch_devices
    ):
        raise AirNetworkBenchmarkError(
            "torch thread count must be a positive integer and needs a torch device"
        )
    if image_digest is not None and not _IMAGE_DIGEST.fullmatch(image_digest):
        raise AirNetworkBenchmarkError("image digest must look like sha256:<64 hex digits>")
    if hardware_label is not None and not _HARDWARE_LABEL.fullmatch(hardware_label):
        raise AirNetworkBenchmarkError(
            "hardware label must be 1-64 letters, digits, spaces, dots, dashes or underscores"
        )
    if timing_conditions not in TIMING_CONDITIONS:
        raise AirNetworkBenchmarkError(f"timing conditions must be one of {TIMING_CONDITIONS}")

    source = _source_identity(repo_root)
    contract, contract_sha256 = load_air_network_batched_contract(repo_root)
    tolerances = contract["tolerances"]

    previous_threads: int | None = None
    torch_identity: dict[str, Any] = {}
    if torch_devices:
        # Fail before any measurement if the backend or a device is unavailable.
        torch = torch_module.require_torch()
        for device in torch_devices:
            torch_module.require_device(device)
        if torch_num_threads is not None:
            previous_threads = int(torch.get_num_threads())
            torch.set_num_threads(torch_num_threads)
    try:
        for device in torch_devices:
            torch_identity[device] = torch_module.torch_runtime_identity(device)
        grid, stress_accuracy, gates = _measure(
            contract,
            tolerances,
            batch_sizes=batch_sizes,
            branch_count=branch_count,
            seed=seed,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            reference_iterations=reference_iterations,
            torch_devices=torch_devices,
            precisions=precisions,
        )
    finally:
        if previous_threads is not None:
            torch_module.require_torch().set_num_threads(previous_threads)

    cuda_devices = [device for device in torch_devices if device != "cpu"]
    gates_passed = bool(gates) and all(gates)
    missing = _cuda_receipt_prerequisites(
        cuda_devices=cuda_devices,
        batch_sizes=batch_sizes,
        precisions=precisions if torch_devices else (),
        image_digest=image_digest,
        hardware_label=hardware_label,
        source_worktree_dirty=source["source_worktree_dirty"],
        source_commit=source["source_commit"],
        gates_passed=gates_passed,
        timing_conditions=timing_conditions,
    )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "issue": 91,
        "evidence_role": "research_only_kernel_benchmark",
        "release_tier": "DEVELOPMENT_FIXTURE_ONLY",
        "source": source,
        "contract": {
            "relative_path": CONTRACT_RELATIVE_PATH,
            "canonical_sha256": contract_sha256,
            "status": contract["status"],
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine().lower(),
            "logical_cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch": torch_identity or None,
            "torch_num_threads_requested": torch_num_threads,
            "container_image_digest": {
                "value": image_digest,
                "source": "operator_declared_not_verified" if image_digest else "not_provided",
            },
            "hardware_label": {
                "value": hardware_label,
                "source": "operator_declared_not_verified" if hardware_label else "not_provided",
            },
            "timing_conditions": {
                "value": timing_conditions,
                "source": "operator_declared_not_verified",
                "provisional": timing_conditions != "dedicated_host",
            },
        },
        "inputs": {
            "generator": "air_network_analytic.generate_air_network_cases (numpy PCG64)",
            "seed": seed,
            "branch_count": branch_count,
            "timing_case_profile": TIMING_PROFILE,
            "stress_case_profile": STRESS_PROFILE,
            "requested_batch_sizes": list(batch_sizes),
        },
        "timing_method": {
            "clock": "time.perf_counter_ns",
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "scalar_reference_measured_iterations": reference_iterations,
            "scalar_reference_warmup_iterations": 1,
            "cuda_synchronization": "torch.cuda.synchronize immediately before the clock starts and before it stops, for every sample",
            "numpy_validated": "host validation plus closed-form kernel",
            "numpy_kernel_only": "closed-form kernel on an already validated batch",
            "torch_kernel_only": "pre-staged device tensors; excludes validation and host/device transfer",
            "torch_transfer_inclusive": "host FP64 validation, copy to device, kernel, copy back to host arrays",
            "scalar_reference_loop": "one reviewed bisection solve per world plus packing into the batch layout, specs prebuilt",
            "order": "implementations measured sequentially per batch size, not interleaved",
            "latency_scope": "kernel and host/device transfer only; no simulator orchestration or authoritative verification is included",
        },
        "grid": grid,
        "stress_accuracy": stress_accuracy,
        "crossover_vs_numpy_fp64": {
            f"{device}/{precision}/{route}": _crossover(grid, f"{device}/{precision}", route)
            for device in (torch_devices or ("cuda",))
            for precision in precisions
            for route in ("transfer_inclusive", "kernel_only")
        },
        "gate": {
            "description": (
                "every gated per-variable comparison within its frozen tolerance: NumPy FP64 vs "
                "the reviewed solver on the fixture band, NumPy FP64 vs the high-precision root "
                "on a stress sample, and each PyTorch precision vs NumPy FP64 on both profiles"
            ),
            "passed": gates_passed,
            "gated_comparison_count": len(gates),
        },
        "claims": {
            "research_only": True,
            "actuator_authority": False,
            "canonical_replay_or_hash_authority": False,
            "end_to_end_simulator_speedup_claimed": False,
            "torch_backend_evaluated": bool(torch_devices),
            "cuda_device_measured": bool(cuda_devices),
            "hardware_identity_verified_by_script": False,
            "cuda_receipt_prerequisites_met": not missing,
            "cuda_receipt_prerequisites_missing": missing,
            "qualified_model": False,
            "production_deployed": False,
        },
    }


def _measure(
    contract: dict[str, Any],
    tolerances: dict[str, Any],
    *,
    batch_sizes: Sequence[int],
    branch_count: int,
    seed: int,
    warmup_iterations: int,
    measured_iterations: int,
    reference_iterations: int,
    torch_devices: Sequence[str],
    precisions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[bool]]:
    grid = []
    gates: list[bool] = []
    for batch_size in batch_sizes:
        batch = generate_air_network_cases(
            contract["case_profiles"][TIMING_PROFILE],
            seed=seed, batch_size=batch_size, branch_count=branch_count,
        )
        points = _reference_points(batch)
        numpy_result = solve_air_network_analytic(batch)
        accuracy = compare_results(
            numpy_result, _solve_reference(points), batch,
            tolerances["numpy_fp64_vs_scalar_reference"]["variables"],
        )
        gates.append(_all_passed(accuracy))

        reference_samples = _time_call(
            lambda: _solve_reference(points),
            warmup_iterations=1, measured_iterations=reference_iterations,
        )
        validated_samples = _time_call(
            lambda: solve_air_network_analytic(batch),
            warmup_iterations=warmup_iterations, measured_iterations=measured_iterations,
        )
        kernel_samples = _time_call(
            lambda: air_network_operating_point_numpy(batch),
            warmup_iterations=warmup_iterations, measured_iterations=measured_iterations,
        )
        numpy_peak = _host_peak_bytes(lambda: solve_air_network_analytic(batch))

        torch_entries: dict[str, Any] = {}
        for device in torch_devices:
            for precision in precisions:
                entry = _torch_measurements(
                    batch, numpy_result, tolerances, device=device, precision=precision,
                    warmup_iterations=warmup_iterations,
                    measured_iterations=measured_iterations,
                )
                gates.append(entry["accuracy_passed"])
                torch_entries[f"{device}/{precision}"] = entry
        grid.append(
            {
                "batch_size": batch_size,
                "input_sha256": air_network_batch_sha256(batch),
                "scalar_reference_loop": _summarise(reference_samples, batch_size),
                "numpy_fp64": {
                    "comparison_id": "numpy_fp64_vs_scalar_reference",
                    "accuracy_vs_scalar_reference": accuracy,
                    "accuracy_passed": _all_passed(accuracy),
                    "validated": _summarise(validated_samples, batch_size),
                    "kernel_only": _summarise(kernel_samples, batch_size),
                    "peak_host_bytes": numpy_peak,
                    "peak_memory_method": "tracemalloc peak above the pre-call allocation, one validated solve",
                    "output_bytes": _result_bytes(numpy_result),
                },
                "torch": torch_entries,
            }
        )

    # Stress profile. NumPy FP64 is gated against the independent high-precision
    # root on a leading sample; PyTorch is gated against NumPy FP64 on the whole
    # batch. The reviewed solver is reported but not gated here, because its
    # bisection does not converge for some fan-dominated worlds off the band.
    stress_size = max(batch_sizes)
    stress = generate_air_network_cases(
        contract["case_profiles"][STRESS_PROFILE],
        seed=seed, batch_size=stress_size, branch_count=branch_count,
    )
    stress_numpy = solve_air_network_analytic(stress)
    sample = _take(stress, min(stress_size, HIGH_PRECISION_WORLDS))
    high_precision_report = compare_results(
        solve_air_network_analytic(sample), _high_precision_root(sample), sample,
        tolerances["numpy_fp64_vs_high_precision_root"]["variables"],
    )
    gates.append(_all_passed(high_precision_report))
    reference_report = compare_results(
        stress_numpy, _solve_reference(_reference_points(stress)), stress,
        tolerances["numpy_fp64_vs_scalar_reference"]["variables"],
    )
    stress_torch: dict[str, Any] = {}
    for device in torch_devices:
        for precision in precisions:
            comparison_id = _torch_comparison_id(precision)
            report = compare_results(
                torch_module.solve_air_network_torch(stress, precision=precision, device=device),
                stress_numpy, stress, tolerances[comparison_id]["variables"],
            )
            gates.append(_all_passed(report))
            stress_torch[f"{device}/{precision}"] = {
                "comparison_id": comparison_id,
                "accuracy_vs_numpy_fp64": report,
                "accuracy_passed": _all_passed(report),
            }
    stress_accuracy = {
        "batch_size": stress_size,
        "input_sha256": air_network_batch_sha256(stress),
        "numpy_fp64_vs_high_precision_root": {
            "comparison_id": "numpy_fp64_vs_high_precision_root",
            "gated": True,
            "world_count": sample.batch_size,
            "sample": "leading worlds of the stress batch",
            "input_sha256": air_network_batch_sha256(sample),
            "accuracy": high_precision_report,
            "accuracy_passed": _all_passed(high_precision_report),
        },
        "numpy_fp64_vs_scalar_reference": {
            "comparison_id": "numpy_fp64_vs_scalar_reference",
            "gated": False,
            "reason_not_gated": "tolerance is declared for the fixture band only; off the band the reviewed bisection can be unconverged",
            "accuracy": reference_report,
            "accuracy_passed": _all_passed(reference_report),
        },
        "torch": stress_torch,
    }
    return grid, stress_accuracy, gates


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory below out/")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--branch-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=200)
    parser.add_argument("--reference-iterations", type=int, default=5)
    parser.add_argument(
        "--torch-devices", nargs="*", default=[],
        help="PyTorch devices to measure, e.g. cpu cuda; omitted means NumPy only. "
        "A requested but unavailable backend or device is an error, never a fallback.",
    )
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "fp64"), default=["fp32", "fp64"])
    parser.add_argument(
        "--torch-num-threads", type=int, default=None,
        help="set PyTorch intra-op threads while measuring; default leaves PyTorch's own setting",
    )
    parser.add_argument(
        "--image-digest", default=None,
        help="operator-declared container image digest, sha256:<64 hex>; recorded, not verified",
    )
    parser.add_argument(
        "--hardware-label", default=None,
        help="operator-declared hardware label; recorded, not verified",
    )
    parser.add_argument(
        "--timing-conditions", choices=TIMING_CONDITIONS, default="shared_host_provisional",
        help="declare whether other workloads shared the host; shared timings are provisional",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    created: Path | None = None
    try:
        output = _resolve_output(arguments.output, OUT_ROOT)
        # Claim the directory before measuring so a path problem costs nothing.
        output.mkdir(parents=True, exist_ok=False)
        created = output
        receipt = run_benchmark(
            batch_sizes=tuple(arguments.batch_sizes),
            branch_count=arguments.branch_count,
            seed=arguments.seed,
            warmup_iterations=arguments.warmup_iterations,
            measured_iterations=arguments.measured_iterations,
            reference_iterations=arguments.reference_iterations,
            torch_devices=tuple(arguments.torch_devices),
            precisions=tuple(arguments.precisions),
            torch_num_threads=arguments.torch_num_threads,
            image_digest=arguments.image_digest,
            hardware_label=arguments.hardware_label,
            timing_conditions=arguments.timing_conditions,
        )
        raw = _canonical_json_bytes(receipt)
        temporary = output / f".{RECEIPT_NAME}.partial"
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / RECEIPT_NAME)
    except Exception as error:  # noqa: BLE001 - every failure must map to status 2
        if created is not None:
            for leftover in created.iterdir():
                leftover.unlink()
            created.rmdir()
        # OSError text can carry host paths; report only its kind.
        detail = type(error).__name__ if isinstance(error, OSError) else f"{error}"
        print(f"benchmark not run: {detail}", file=sys.stderr)
        return 2
    print(f"receipt: {(output / RECEIPT_NAME).relative_to(OUT_ROOT.resolve().parent).as_posix()}")
    print(f"receipt sha256: {hashlib.sha256(raw).hexdigest()}")
    print(f"gate passed: {receipt['gate']['passed']}")
    return 0 if receipt["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
