"""Telemetry drift monitoring for the action-aware MLP forecaster.

Compares a live 16-step numeric telemetry window against the training
distribution stored inside the hash-pinned model artifact
(``feature_mean`` / ``feature_std``, first 16*194 positions — the numeric
window portion of the model input).

A window whose statistics have left the training distribution is exactly
the situation where point forecasts deserve less trust; this monitor makes
that visible instead of silent. It is a diagnostic, not an authority: HMC
remains the sole command authority regardless of the verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aeolus.habitat_v2.forecast.live_demo import (
    load_forecast_contracts,
    project_history_window,
)
from aeolus.habitat_v2.forecast.live_mlp_demo import (
    MLP_ANCHOR_STEP,
    MLP_WINDOW_STEPS,
    LiveForecastModel,
    _demo_nonce,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer

WINDOW_FEATURES = 194  # per-step numeric fields in the model window
DEFAULT_Z_THRESHOLD = 6.0


class DriftMonitorError(Exception):
    """Raised on malformed monitor inputs."""


@dataclass(frozen=True)
class DriftReport:
    max_abs_z: float
    mean_abs_z: float
    n_over_threshold: int
    z_threshold: float
    worst: tuple[tuple[int, int, float], ...]  # (step, feature, z) top-k
    drifted: bool


def load_window_stats(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) shaped (16, 194) from the model artifact."""
    arrays = np.load(Path(npz_path))
    mean = np.asarray(arrays["feature_mean"], dtype=np.float64)
    std = np.asarray(arrays["feature_std"], dtype=np.float64)
    count = MLP_WINDOW_STEPS * WINDOW_FEATURES
    if mean.shape[0] < count or std.shape[0] < count:
        raise DriftMonitorError("artifact does not carry full window statistics")
    return mean[:count].reshape(MLP_WINDOW_STEPS, WINDOW_FEATURES), std[:count].reshape(
        MLP_WINDOW_STEPS, WINDOW_FEATURES
    )


def assess_window(
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    top_k: int = 8,
) -> DriftReport:
    """Score one (16, 194) window against training statistics."""
    values = np.asarray(window, dtype=np.float64)
    if values.shape != (MLP_WINDOW_STEPS, WINDOW_FEATURES):
        raise DriftMonitorError(
            f"window must have shape {(MLP_WINDOW_STEPS, WINDOW_FEATURES)}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, (values - mean) / std, 0.0)
    abs_z = np.abs(z)
    flat = np.argsort(abs_z, axis=None)[::-1][:top_k]
    worst = tuple(
        (int(i // WINDOW_FEATURES), int(i % WINDOW_FEATURES), float(z.flat[i]))
        for i in flat
    )
    n_over = int((abs_z > z_threshold).sum())
    return DriftReport(
        max_abs_z=float(abs_z.max()),
        mean_abs_z=float(abs_z.mean()),
        n_over_threshold=n_over,
        z_threshold=z_threshold,
        worst=worst,
        drifted=n_over > 0,
    )


def build_demo_window(repo_root: str | Path, model: LiveForecastModel) -> np.ndarray:
    """Reproduce the demo's 16-step anchor window (steps 1-16)."""
    root = Path(repo_root).resolve()
    bundle = load_forecast_contracts(root)
    action = next(iter(bundle.actions))
    hmc = HabitatManagementComputer.reset(
        bundle.development_scenario,
        bundle.hmc_contract,
        _demo_nonce(model, action.action_id),
    )
    snapshots: dict[int, tuple[object, object]] = {}
    for application_step in range(MLP_ANCHOR_STEP + 1):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise DriftMonitorError("HMC terminated before the anchor step")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        if application_step:
            snapshots[application_step] = (snapshot, verification)
        hmc.propose(None, handle)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise DriftMonitorError("HMC terminated while arbitrating")
        hmc.step()
    pairs = [
        snapshots[step]
        for step in range(MLP_ANCHOR_STEP - MLP_WINDOW_STEPS + 1, MLP_ANCHOR_STEP + 1)
    ]
    history = project_history_window(bundle, pairs, window_steps=MLP_WINDOW_STEPS)
    return np.asarray(history.numeric_f32, dtype=np.float64)


def render_report(report: DriftReport) -> str:
    verdict = "DRIFT DETECTED" if report.drifted else "NOMINAL"
    lines = [
        f"verdict: {verdict}",
        (
            f"max |z|: {report.max_abs_z:.2f}   mean |z|: {report.mean_abs_z:.3f}   "
            f"features beyond {report.z_threshold:.0f} sigma: {report.n_over_threshold}"
        ),
        "worst positions (step, feature, z):",
    ]
    for step, feature, z in report.worst:
        lines.append(f"  step {step:2d}  feature {feature:3d}  z {z:+.2f}")
    if report.drifted:
        lines.append(
            "telemetry statistics have left the training distribution - "
            "forecasts from this window deserve less trust"
        )
    return "\n".join(lines)
