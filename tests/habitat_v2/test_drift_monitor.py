"""Tests for the telemetry drift monitor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.drift_monitor import (
    DriftMonitorError,
    assess_window,
    build_demo_window,
    load_window_stats,
    render_report,
)
from aeolus.habitat_v2.forecast.live_mlp_demo import load_live_mlp_model

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz"
)
ARTIFACT_SHA256 = (
    "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
)


def _stats():
    return load_window_stats(ARTIFACT)


def test_demo_window_is_nominal() -> None:
    model = load_live_mlp_model(ARTIFACT, expected_sha256=ARTIFACT_SHA256)
    mean, std = _stats()
    window = build_demo_window(REPO_ROOT, model)
    report = assess_window(window, mean, std)
    assert not report.drifted, render_report(report)
    assert report.max_abs_z < 6.0


def test_shifted_window_is_flagged() -> None:
    mean, std = _stats()
    window = mean.copy()
    window[0, 7] += 50.0 * std[0, 7]  # massive single-feature shift
    window[3, 120] += 20.0 * std[3, 120]
    report = assess_window(window, mean, std)
    assert report.drifted
    assert report.n_over_threshold >= 2
    worst_positions = {(step, feature) for step, feature, _ in report.worst[:2]}
    assert (0, 7) in worst_positions
    assert (3, 120) in worst_positions


def test_wrong_shape_rejected() -> None:
    mean, std = _stats()
    with pytest.raises(DriftMonitorError):
        assess_window(np.zeros((4, 4)), mean, std)


def test_render_report_shows_verdict_and_worst() -> None:
    mean, std = _stats()
    report = assess_window(mean.copy(), mean, std)
    text = render_report(report)
    assert "NOMINAL" in text
    assert "worst positions" in text
    shifted = mean.copy()
    shifted[5, 5] += 30.0 * std[5, 5]
    assert "DRIFT DETECTED" in render_report(assess_window(shifted, mean, std))
