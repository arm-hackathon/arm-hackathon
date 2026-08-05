"""Frozen V5 development protocol for load-preserving nominal counterfactuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeolus.model_cycle_v4 import (
    PROHIBITED_HISTORICAL_SEEDS,
    V4_CALIBRATION_SEEDS,
    V4_FIT_SEEDS,
    V4_VALIDATION_SEEDS,
    run_v4_development as _run_model_cycle,
)
from aeolus.sweep import SWEEP_V5_VERSION


V5_FIT_SEEDS = (1100, 1101, 1102, 1103)
V5_CALIBRATION_SEEDS = (1104, 1105)
V5_VALIDATION_SEEDS = (1300, 1301, 1302, 1303, 1304, 1305)
CANONICAL_V5_DEVELOPMENT_SPEC_SHA256 = (
    "d9ae68eb4ad16e91bc8318d1ee028e51efec35d3fa1352d3f10df88becfd5065"
)
V5_PROHIBITED_HISTORICAL_SEEDS = PROHIBITED_HISTORICAL_SEEDS | frozenset(
    (*V4_FIT_SEEDS, *V4_CALIBRATION_SEEDS, *V4_VALIDATION_SEEDS)
)
V5_SOURCE_PATHS = (
    "uv.lock",
    "scenarios/sweep-v5-development.json",
    "src/aeolus/sweep.py",
    "src/aeolus/corpus.py",
    "src/aeolus/families.py",
    "src/aeolus/config.py",
    "src/aeolus/scenario.py",
    "src/aeolus/plant.py",
    "src/aeolus/actuator.py",
    "src/aeolus/model_input.py",
    "src/aeolus/detector.py",
    "src/aeolus/baseline.py",
    "src/aeolus/control.py",
    "src/aeolus/measurement.py",
    "src/aeolus/trace.py",
    "src/aeolus/error_analysis.py",
    "src/aeolus/temporal_cnn.py",
    "src/aeolus/model_cycle_v4.py",
    "src/aeolus/model_cycle_v5.py",
    "src/aeolus/edge_benchmark.py",
)


def run_v5_development(
    sweep_spec_path: str | Path,
    output_dir: str | Path,
    *,
    mlp_epochs: int = 300,
    cnn_epochs: int = 300,
) -> dict[str, Any]:
    """Run the single-use V5 train/calibration/validation development protocol."""
    return _run_model_cycle(
        sweep_spec_path,
        output_dir,
        mlp_epochs=mlp_epochs,
        cnn_epochs=cnn_epochs,
        protocol_name="v5",
        expected_schema_version=SWEEP_V5_VERSION,
        expected_spec_sha256=CANONICAL_V5_DEVELOPMENT_SPEC_SHA256,
        fit_seeds=V5_FIT_SEEDS,
        calibration_seeds=V5_CALIBRATION_SEEDS,
        validation_seeds=V5_VALIDATION_SEEDS,
        prohibited_seeds=V5_PROHIBITED_HISTORICAL_SEEDS,
        report_schema_version="aeolus_v5_development_evidence_v1",
        report_filename="v5-development-report.json",
        source_paths=V5_SOURCE_PATHS,
    )
