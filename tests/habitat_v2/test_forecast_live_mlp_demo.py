"""Tests for the Historical V2 MLP live-demo adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.live_demo import LiveForecastError
from aeolus.habitat_v2.forecast.live_mlp_demo import (
    NumpyMlpPredictor,
    load_live_mlp_model,
    run_live_mlp_forecast_demo,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
ARTIFACT_SHA256 = "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"


def test_mlp_artifact_loads_with_exact_identity() -> None:
    model = load_live_mlp_model(ARTIFACT, expected_sha256=ARTIFACT_SHA256)
    assert model.model_kind == "action_aware_mlp_v1"
    assert model.actuator_authority is False
    assert model.artifact_sha256 == ARTIFACT_SHA256


def test_mlp_artifact_rejects_identity_drift() -> None:
    with pytest.raises(LiveForecastError, match="SHA-256"):
        load_live_mlp_model(ARTIFACT, expected_sha256="0" * 64)


def test_mlp_forward_pass_matches_frozen_reference() -> None:
    """Fixed-seed forward pass must reproduce the recorded reference digest."""
    model = load_live_mlp_model(ARTIFACT, expected_sha256=ARTIFACT_SHA256)
    predictor = model.predictor
    assert type(predictor) is NumpyMlpPredictor
    rng = np.random.default_rng(20260818)

    class _History:
        numeric_f32 = rng.standard_normal((16, 194)).astype(np.float32)

    action = rng.standard_normal(27).astype(np.float32)
    prediction = predictor.predict(_History(), action)
    assert prediction.shape == (8, 51)
    assert prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    # These values were recorded as a Torch-checkpoint comparison during
    # artifact conversion. The original conversion environment is not retained;
    # this test pins the current NumPy output, not the historical comparison.
    # Inputs are out-of-distribution by design, so this is a drift digest rather
    # than a plausibility check.
    assert prediction.mean() == pytest.approx(-132516.125, rel=1e-4)
    assert prediction.std() == pytest.approx(327965.46875, rel=1e-4)


def test_mlp_demo_runs_closed_with_realized_truth() -> None:
    model = load_live_mlp_model(ARTIFACT, expected_sha256=ARTIFACT_SHA256)
    result = run_live_mlp_forecast_demo(
        ROOT, model, selected_action_id="normal-occupied-v1",
    )
    assert result.terminal_status == "COMPLETED"
    assert result.hmc_is_sole_actuator_authority is True
    assert result.actuator_authority is False
    assert len(result.candidate_forecasts) == 4
    assert result.forecast_history_steps == tuple(range(1, 17))
    assert result.truth_steps == tuple(range(17, 25))
    distinct = {
        item.prediction_f32.tobytes() for item in result.candidate_forecasts
    }
    assert len(distinct) == 4  # action conditioning produces distinct forecasts
