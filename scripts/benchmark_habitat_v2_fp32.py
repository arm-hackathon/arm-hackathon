from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median

from aeolus.habitat_v2.forecast.arm_optimization import (
    benchmark_fp64_vs_fp32,
    optimise_ridge_fp32,
)
from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

SOURCE_MODEL_SHA256 = "0de4b5cdb6ec2b47be260a06f924d8eb00f1def16d5ae668b3ab5191251f29df"
_CONVERSION_FIELDS = frozenset(
    {
        "schema_version",
        "release_tier",
        "actuator_authority",
        "source_precision",
        "candidate_precision",
        "source_model_sha256",
        "source_model_file_bytes",
        "source_raw_array_bytes",
        "candidate_model_sha256",
        "candidate_model_file_bytes",
        "candidate_raw_array_bytes",
        "candidate_raw_array_bytes_reduction_fraction",
        "quality_gate",
        "quality_gate_evaluated",
        "native_arm64_benchmark_evaluated",
        "qualified_model",
        "production_deployed",
    }
)
_CONVERSION_CONSTANTS = {
    "schema_version": "aeolus_habitat_v2_forecast_fp32_optimisation_receipt_v1",
    "release_tier": "DEMO_ONLY_PERMANENTLY_EXCLUDED",
    "actuator_authority": False,
    "source_precision": "float64",
    "candidate_precision": "float32",
    "source_raw_array_bytes": 28_759_024,
    "candidate_raw_array_bytes": 14_379_512,
    "candidate_raw_array_bytes_reduction_fraction": 0.5,
    "quality_gate": "normalised_prediction_drift_lte_1e-4",
    "quality_gate_evaluated": False,
    "native_arm64_benchmark_evaluated": False,
    "qualified_model": False,
    "production_deployed": False,
}


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    parser = argparse.ArgumentParser(
        description=(
            "Convert and benchmark the frozen Habitat V2 FP64 ridge model. "
            "Compressed NPZ bytes are not cross-platform reproducible."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--source", type=Path, default=artifact_root / "action-aware-ridge.npz"
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=artifact_root / "action-aware-ridge-fp32.npz",
    )
    parser.add_argument(
        "--conversion-receipt",
        type=Path,
        default=artifact_root / "fp32-conversion-receipt.json",
    )
    parser.add_argument(
        "--benchmark-receipt",
        type=Path,
        default=root / "out/habitat-v2-fp32-arm/benchmark-receipt.json",
    )
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=200)
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        default=1,
        help="Run independent benchmark executions and report their median distribution.",
    )
    parser.add_argument(
        "--use-existing-candidate",
        action="store_true",
        help=(
            "Benchmark the exact SHA-bound candidate and conversion receipt already "
            "present instead of regenerating platform-dependent compressed bytes."
        ),
    )
    return parser.parse_args()


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_existing_conversion(
    source: Path,
    candidate: Path,
    receipt_path: Path,
) -> tuple[dict[str, object], bytes]:
    try:
        source_raw = source.read_bytes()
        candidate_raw = candidate.read_bytes()
        raw = receipt_path.read_bytes()
        conversion = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("existing conversion evidence is unreadable") from error
    if type(conversion) is not dict or raw != canonical_json_bytes(conversion):
        raise ValueError("existing conversion receipt is not canonical JSON")
    if set(conversion) != _CONVERSION_FIELDS:
        raise ValueError("existing conversion receipt fields drift")
    expected = {
        **_CONVERSION_CONSTANTS,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_model_file_bytes": len(source_raw),
        "candidate_model_sha256": _sha256(candidate_raw),
        "candidate_model_file_bytes": len(candidate_raw),
    }
    if _sha256(source_raw) != SOURCE_MODEL_SHA256 or any(
        conversion[field] != value for field, value in expected.items()
    ):
        raise ValueError(
            "existing conversion receipt semantics or model identity drift"
        )
    return conversion, raw


def _provenance(conversion_raw: bytes, repetitions: int) -> dict[str, object]:
    return {
        "conversion_receipt_sha256": _sha256(conversion_raw),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "github_job": os.environ.get("GITHUB_JOB"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "benchmark_repetitions": repetitions,
    }


def _aggregate_benchmarks(
    runs: list[dict[str, object]], provenance: dict[str, object]
) -> dict[str, object]:
    fp64_medians = [int(run["timing"]["fp64"]["median_ns"]) for run in runs]
    fp32_medians = [int(run["timing"]["fp32"]["median_ns"]) for run in runs]
    speedups = [float(run["timing"]["median_speedup_fp64_over_fp32"]) for run in runs]
    return {
        "schema_version": "aeolus_habitat_v2_fp64_fp32_benchmark_series_v1",
        "evidence_role": "demo_only_reduced_precision_benchmark_series",
        "provenance": provenance,
        "run_count": len(runs),
        "all_prediction_parity_passed": all(
            run["prediction_parity"]["passed"] is True for run in runs
        ),
        "median_distribution": {
            "fp64_median_ns_by_run": fp64_medians,
            "fp32_median_ns_by_run": fp32_medians,
            "speedup_by_run": speedups,
            "fp64_median_of_run_medians_ns": int(median(fp64_medians)),
            "fp32_median_of_run_medians_ns": int(median(fp32_medians)),
            "median_of_run_speedups": float(median(speedups)),
        },
        "runs": runs,
    }


def main() -> int:
    arguments = _arguments()
    if arguments.benchmark_repetitions < 1:
        raise ValueError("benchmark repetitions must be positive")
    if arguments.use_existing_candidate:
        conversion, conversion_raw = _load_existing_conversion(
            arguments.source,
            arguments.candidate,
            arguments.conversion_receipt,
        )
    else:
        conversion = optimise_ridge_fp32(
            arguments.source,
            arguments.candidate,
            expected_source_sha256=SOURCE_MODEL_SHA256,
        )
        _write_canonical(arguments.conversion_receipt, conversion)
        conversion_raw = arguments.conversion_receipt.read_bytes()
    provenance = _provenance(conversion_raw, arguments.benchmark_repetitions)
    runs = []
    for repetition in range(1, arguments.benchmark_repetitions + 1):
        benchmark = benchmark_fp64_vs_fp32(
            arguments.repo_root,
            arguments.source,
            arguments.candidate,
            expected_source_sha256=SOURCE_MODEL_SHA256,
            expected_candidate_sha256=str(conversion["candidate_model_sha256"]),
            warmup_iterations=arguments.warmup_iterations,
            measured_iterations=arguments.measured_iterations,
        )
        benchmark["provenance"] = {**provenance, "repetition_index": repetition}
        runs.append(benchmark)
    receipt = (
        runs[0]
        if arguments.benchmark_repetitions == 1
        else _aggregate_benchmarks(runs, provenance)
    )
    _write_canonical(arguments.benchmark_receipt, receipt)
    return 0 if all(run["prediction_parity"]["passed"] is True for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
