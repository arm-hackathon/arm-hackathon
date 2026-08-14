from __future__ import annotations

import argparse
from pathlib import Path

from aeolus.habitat_v2.forecast.arm_optimization import (
    benchmark_fp64_vs_fp32,
    optimise_ridge_fp32,
)
from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

SOURCE_MODEL_SHA256 = "a6e4ef34fc837bb6539a84e20d015bbd7bbfe4e9fd5a6fc74e3f0217bd978d9a"


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    parser = argparse.ArgumentParser(
        description="Convert and benchmark the frozen Habitat V2 FP64 ridge model."
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
    return parser.parse_args()


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def main() -> int:
    arguments = _arguments()
    conversion = optimise_ridge_fp32(
        arguments.source,
        arguments.candidate,
        expected_source_sha256=SOURCE_MODEL_SHA256,
    )
    _write_canonical(arguments.conversion_receipt, conversion)
    benchmark = benchmark_fp64_vs_fp32(
        arguments.repo_root,
        arguments.source,
        arguments.candidate,
        expected_source_sha256=SOURCE_MODEL_SHA256,
        expected_candidate_sha256=str(conversion["candidate_model_sha256"]),
        warmup_iterations=arguments.warmup_iterations,
        measured_iterations=arguments.measured_iterations,
    )
    _write_canonical(arguments.benchmark_receipt, benchmark)
    return 0 if benchmark["prediction_parity"]["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
