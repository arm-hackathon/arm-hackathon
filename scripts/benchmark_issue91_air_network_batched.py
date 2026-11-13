"""Benchmark the Issue #91 batched air-network candidate against the reference.

Compares the research-only closed-form kernels (NumPy FP64, and optionally
eager PyTorch FP32/FP64 on CPU or CUDA) with the reviewed scalar bisection
solver, applies the per-variable tolerances frozen in
``contracts/habitat_v2_air_network_batched_v1.json`` and writes one write-once
receipt under the ignored ``out/`` directory.

A kernel timing is not an end-to-end simulator speedup, a model improvement or
a control benefit. No GPU speedup is assumed: a crossover is reported only when
it was measured, and CUDA results exist only when a CUDA device actually ran.

Exit status: 0 when every gated comparison passed, 1 when the receipt was
written but a gate failed, 2 when the benchmark could not run (nothing written).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import numpy as np

from aeolus.habitat_v2.air_network import (
    AirNetworkSpec,
    BranchSpec,
    FanSpec,
    solve_air_network,
)
from aeolus.habitat_v2.air_network_analytic import (
    CONTRACT_RELATIVE_PATH,
    AirNetworkAnalyticError,
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
SOURCE_PATHS = (
    "src/aeolus/habitat_v2/air_network.py",
    "src/aeolus/habitat_v2/air_network_analytic.py",
    "src/aeolus/habitat_v2/air_network_batched_torch.py",
    "scripts/benchmark_issue91_air_network_batched.py",
    CONTRACT_RELATIVE_PATH,
)


class AirNetworkBenchmarkError(RuntimeError):
    """Raised when the benchmark cannot proceed safely."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _resolve_output(output_path: Path, out_root: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(out_root.resolve())
    except ValueError as error:
        raise AirNetworkBenchmarkError(
            "benchmark output must live under the ignored out/ directory"
        ) from error
    if output == out_root.resolve():
        raise AirNetworkBenchmarkError("benchmark output must be a new directory below out/")
    if output.exists():
        raise AirNetworkBenchmarkError(
            "refusing to overwrite existing output directory"
        )
    return output


def _git_output(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments), cwd=repo_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_identity(repo_root: Path) -> dict[str, Any]:
    """Commit and per-file identity, independent of checkout line endings."""

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
    if synchronize is not None:
        synchronize()
    samples = []
    for _ in range(measured_iterations):
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
    return all(entry["passed"] is True for entry in report.values())


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
    from aeolus.habitat_v2.air_network_batched_torch import (
        air_network_operating_point_torch,
        require_torch,
        solve_air_network_torch,
        to_torch_inputs,
    )

    torch = require_torch()
    is_cuda = device != "cpu"
    synchronize = (lambda: torch.cuda.synchronize(device)) if is_cuda else None
    comparison_id = (
        "torch_fp32_vs_numpy_fp64" if precision == "fp32" else "torch_fp64_vs_numpy_fp64"
    )
    result = solve_air_network_torch(batch, precision=precision, device=device)
    accuracy = compare_results(
        result, numpy_result, batch, tolerances[comparison_id]["variables"]
    )

    staged = to_torch_inputs(batch, precision=precision, device=device)

    def kernel_only() -> object:
        with torch.no_grad():
            return air_network_operating_point_torch(staged)

    kernel_samples = _time_call(
        kernel_only,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        synchronize=synchronize,
    )
    transfer_samples = _time_call(
        lambda: solve_air_network_torch(batch, precision=precision, device=device),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        synchronize=synchronize,
    )
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        solve_air_network_torch(batch, precision=precision, device=device)
        torch.cuda.synchronize(device)
        peak_memory: dict[str, Any] = {
            "peak_device_bytes": int(torch.cuda.max_memory_allocated(device)),
            "method": "torch.cuda.max_memory_allocated after reset_peak_memory_stats, one transfer-inclusive solve",
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
        "output_bytes": _result_bytes(result),
    }


def _crossover(
    grid: Sequence[dict[str, Any]], candidate_key: str, metric: str
) -> dict[str, Any]:
    """Smallest tested batch size at which a Torch route beats validated NumPy."""

    compared = []
    for entry in grid:
        candidate = entry["torch"].get(candidate_key)
        if candidate is None:
            continue
        compared.append(
            {
                "batch_size": entry["batch_size"],
                "candidate_median_ns": candidate[metric]["median_ns"],
                "numpy_fp64_validated_median_ns": entry["numpy_fp64"]["validated"]["median_ns"],
            }
        )
    if not compared:
        return {"status": "not_evaluated", "first_faster_batch_size": None, "compared": []}
    faster = [
        row["batch_size"]
        for row in compared
        if row["candidate_median_ns"] < row["numpy_fp64_validated_median_ns"]
    ]
    return {
        "status": "measured_crossover" if faster else "no_crossover_in_tested_grid",
        "first_faster_batch_size": min(faster) if faster else None,
        "faster_batch_sizes": faster,
        "compared": compared,
    }


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
    image_digest: str | None = None,
    hardware_label: str | None = None,
) -> dict[str, Any]:
    """Measure accuracy and latency; return the receipt mapping (not written)."""

    if not batch_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in batch_sizes
    ) or len(set(batch_sizes)) != len(batch_sizes):
        raise AirNetworkBenchmarkError("batch sizes must be distinct positive integers")
    if warmup_iterations < 1 or measured_iterations < 4 or reference_iterations < 1:
        raise AirNetworkBenchmarkError(
            "warmup must be >= 1, measured iterations >= 4 and reference iterations >= 1"
        )
    if len(set(torch_devices)) != len(torch_devices) or not set(precisions) <= {"fp32", "fp64"}:
        raise AirNetworkBenchmarkError("torch devices must be distinct and precisions fp32/fp64")
    if torch_devices and not precisions:
        raise AirNetworkBenchmarkError("at least one precision is required with a torch device")

    contract, contract_sha256 = load_air_network_batched_contract(repo_root)
    tolerances = contract["tolerances"]
    torch_identity: dict[str, Any] = {}
    if torch_devices:
        # Fail before any measurement if the backend or a device is unavailable.
        from aeolus.habitat_v2.air_network_batched_torch import (
            require_device,
            torch_runtime_identity,
        )

        for device in torch_devices:
            require_device(device)
            torch_identity[device] = torch_runtime_identity(device)

    grid = []
    gates: list[bool] = []
    for batch_size in batch_sizes:
        batch = generate_air_network_cases(
            contract["case_profiles"][TIMING_PROFILE],
            seed=seed, batch_size=batch_size, branch_count=branch_count,
        )
        points = _reference_points(batch)
        reference = _solve_reference(points)
        numpy_result = solve_air_network_analytic(batch)
        accuracy = compare_results(
            numpy_result, reference, batch,
            tolerances["numpy_fp64_vs_scalar_reference"]["variables"],
        )
        gates.append(_all_passed(accuracy))

        reference_samples = _time_call(
            lambda: _solve_reference(points),
            warmup_iterations=1, measured_iterations=max(reference_iterations, 4),
        )
        validated_samples = _time_call(
            lambda: solve_air_network_analytic(batch),
            warmup_iterations=warmup_iterations, measured_iterations=measured_iterations,
        )
        kernel_samples = _time_call(
            lambda: air_network_operating_point_numpy(batch),
            warmup_iterations=warmup_iterations, measured_iterations=measured_iterations,
        )
        tracemalloc.start()
        solve_air_network_analytic(batch)
        _, numpy_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

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
                    "peak_host_bytes": int(numpy_peak),
                    "peak_memory_method": "tracemalloc peak over one validated solve",
                    "output_bytes": _result_bytes(numpy_result),
                },
                "torch": torch_entries,
            }
        )

    # Stress accuracy: Torch is gated; the scalar reference is reported ungated
    # because the bisection is not accurate relative to tiny outputs off the
    # fixture band (the gated FP64 leg there is the high-precision test oracle).
    stress_size = max(batch_sizes)
    stress = generate_air_network_cases(
        contract["case_profiles"][STRESS_PROFILE],
        seed=seed, batch_size=stress_size, branch_count=branch_count,
    )
    stress_numpy = solve_air_network_analytic(stress)
    stress_reference_report = compare_results(
        stress_numpy, _solve_reference(_reference_points(stress)), stress,
        tolerances["numpy_fp64_vs_scalar_reference"]["variables"],
    )
    stress_torch: dict[str, Any] = {}
    for device in torch_devices:
        from aeolus.habitat_v2.air_network_batched_torch import solve_air_network_torch

        for precision in precisions:
            comparison_id = (
                "torch_fp32_vs_numpy_fp64" if precision == "fp32" else "torch_fp64_vs_numpy_fp64"
            )
            report = compare_results(
                solve_air_network_torch(stress, precision=precision, device=device),
                stress_numpy, stress, tolerances[comparison_id]["variables"],
            )
            gates.append(_all_passed(report))
            stress_torch[f"{device}/{precision}"] = {
                "comparison_id": comparison_id,
                "accuracy_vs_numpy_fp64": report,
                "accuracy_passed": _all_passed(report),
            }

    cuda_devices = [device for device in torch_devices if device != "cpu"]
    source = _source_identity(repo_root)
    dgx_missing = [
        reason
        for reason, missing in (
            ("no CUDA device was measured", not cuda_devices),
            ("no container image digest was declared", not image_digest),
            ("no hardware label was declared", not hardware_label),
            ("source worktree is dirty or its state is unknown", source["source_worktree_dirty"] is not False),
            ("an accuracy gate failed", not all(gates)),
        )
        if missing
    ]
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
            "container_image_digest": {
                "value": image_digest,
                "source": "operator_declared_cli_argument" if image_digest else "not_provided",
            },
            "hardware_label": {
                "value": hardware_label,
                "source": "operator_declared_cli_argument" if hardware_label else "not_provided",
            },
        },
        "inputs": {
            "generator": "air_network_analytic.generate_air_network_cases (numpy PCG64)",
            "seed": seed,
            "branch_count": branch_count,
            "timing_case_profile": TIMING_PROFILE,
            "stress_case_profile": STRESS_PROFILE,
            "stress_batch_size": stress_size,
            "stress_input_sha256": air_network_batch_sha256(stress),
            "requested_batch_sizes": list(batch_sizes),
        },
        "timing_method": {
            "clock": "time.perf_counter_ns",
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "scalar_reference_measured_iterations": max(reference_iterations, 4),
            "scalar_reference_warmup_iterations": 1,
            "cuda_synchronization": "torch.cuda.synchronize before the first sample and inside every timed sample",
            "numpy_validated": "host validation plus closed-form kernel",
            "torch_kernel_only": "pre-staged device tensors; excludes validation and host/device transfer",
            "torch_transfer_inclusive": "host FP64 validation, copy to device, kernel, copy back to host arrays",
            "scalar_reference_loop": "one reviewed bisection solve per world plus packing into the batch layout, specs prebuilt",
            "order": "implementations measured sequentially per batch size, not interleaved",
        },
        "grid": grid,
        "stress_accuracy": {
            "scalar_reference_reported_not_gated": stress_reference_report,
            "torch": stress_torch,
        },
        "crossover_vs_numpy_fp64_validated": {
            f"{device}/{precision}/{route}": _crossover(grid, f"{device}/{precision}", route)
            for device in (torch_devices or ("cuda",))
            for precision in precisions
            for route in ("transfer_inclusive", "kernel_only")
        },
        "gate": {
            "description": "every gated per-variable comparison within its frozen tolerance",
            "passed": all(gates),
            "gated_comparison_count": len(gates),
        },
        "claims": {
            "research_only": True,
            "actuator_authority": False,
            "canonical_replay_or_hash_authority": False,
            "end_to_end_simulator_speedup_claimed": False,
            "torch_backend_evaluated": bool(torch_devices),
            "cuda_hardware_evaluated": bool(cuda_devices),
            "dgx_hardware_receipt": not dgx_missing,
            "dgx_hardware_receipt_missing_requirements": dgx_missing,
            "qualified_model": False,
            "production_deployed": False,
        },
    }


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
    parser.add_argument("--image-digest", default=None, help="operator-declared container image digest")
    parser.add_argument("--hardware-label", default=None, help="operator-declared hardware label")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        output = _resolve_output(arguments.output, OUT_ROOT)
        receipt = run_benchmark(
            batch_sizes=tuple(arguments.batch_sizes),
            branch_count=arguments.branch_count,
            seed=arguments.seed,
            warmup_iterations=arguments.warmup_iterations,
            measured_iterations=arguments.measured_iterations,
            reference_iterations=arguments.reference_iterations,
            torch_devices=tuple(arguments.torch_devices),
            precisions=tuple(arguments.precisions),
            image_digest=arguments.image_digest,
            hardware_label=arguments.hardware_label,
        )
    except (AirNetworkBenchmarkError, AirNetworkAnalyticError, RuntimeError) as error:
        # Includes an unavailable PyTorch backend or CUDA device. Nothing is written.
        print(f"benchmark not run: {error}", file=sys.stderr)
        return 2
    raw = _canonical_json_bytes(receipt)
    output.mkdir(parents=True)
    with (output / RECEIPT_NAME).open("xb") as handle:
        handle.write(raw)
    print(f"receipt: {(output / RECEIPT_NAME).relative_to(OUT_ROOT.parent).as_posix()}")
    print(f"receipt sha256: {hashlib.sha256(raw).hexdigest()}")
    print(f"gate passed: {receipt['gate']['passed']}")
    return 0 if receipt["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
