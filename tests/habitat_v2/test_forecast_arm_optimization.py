from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_fp32_optimisation_preserves_contract_and_runtime_precision(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.arm_optimization import optimise_ridge_fp32
    from aeolus.habitat_v2.forecast.live_demo import load_live_ridge_model

    root = _repo_root()
    source = root / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
    destination = tmp_path / "action-aware-ridge-fp32.npz"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    receipt = optimise_ridge_fp32(
        source,
        destination,
        expected_source_sha256=source_sha256,
    )

    with np.load(destination, allow_pickle=False) as archive:
        assert str(archive["schema_version"].item()) == (
            "aeolus_habitat_v2_forecast_demo_model_fp32_v1"
        )
        for field in ("feature_mean", "feature_scale", "target_mean", "coef"):
            assert archive[field].dtype == np.float32

    model = load_live_ridge_model(
        destination,
        expected_sha256=receipt["candidate_model_sha256"],
    )
    predictor = model.predictor

    assert model.model_kind == "action_aware_ridge_fp32"
    assert model.actuator_authority is False
    assert predictor.feature_mean.dtype == np.float32
    assert predictor.feature_scale.dtype == np.float32
    assert predictor.target_mean.dtype == np.float32
    assert predictor.coef.dtype == np.float32
    assert receipt["source_model_sha256"] == source_sha256
    assert receipt["candidate_raw_array_bytes"] * 2 == receipt["source_raw_array_bytes"]
    assert receipt["candidate_raw_array_bytes_reduction_fraction"] == 0.5


def test_fp32_candidate_passes_live_drift_gate_and_emits_comparable_timings(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.arm_optimization import (
        benchmark_fp64_vs_fp32,
        optimise_ridge_fp32,
    )

    root = _repo_root()
    source = root / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
    candidate = tmp_path / "action-aware-ridge-fp32.npz"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    conversion = optimise_ridge_fp32(
        source,
        candidate,
        expected_source_sha256=source_sha256,
    )

    receipt = benchmark_fp64_vs_fp32(
        root,
        source,
        candidate,
        expected_source_sha256=source_sha256,
        expected_candidate_sha256=conversion["candidate_model_sha256"],
        warmup_iterations=2,
        measured_iterations=8,
    )

    assert receipt["prediction_parity"]["gate"] == (
        "max_abs_drift_div_max_abs_reference_or_one_lte_1e-4"
    )
    assert receipt["prediction_parity"]["passed"] is True
    assert receipt["prediction_parity"]["maximum_normalised_drift"] <= 1e-4
    assert receipt["workload"]["candidate_action_count"] == 4
    assert receipt["workload"]["prediction_shape"] == [8, 51]
    assert receipt["models"]["fp64"]["precision"] == "float64"
    assert receipt["models"]["fp32"]["precision"] == "float32"
    assert receipt["models"]["fp32"]["raw_array_bytes"] * 2 == (
        receipt["models"]["fp64"]["raw_array_bytes"]
    )
    assert receipt["timing"]["fp64"]["sample_count"] == 8
    assert receipt["timing"]["fp32"]["sample_count"] == 8
    assert receipt["timing"]["fp64"]["median_ns"] > 0
    assert receipt["timing"]["fp32"]["median_ns"] > 0
    assert receipt["claims"]["actuator_authority"] is False
    assert receipt["claims"]["arm_specific_operator_optimisation"] is False
