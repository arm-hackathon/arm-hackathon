"""Causal, observable feature extensions for the Issue #56 V4 study.

The feature path consumes only an already verified operational history and a
catalogue command.  It never reads scenario truth, future state, or HMC
arbitration output.  The action mask admits every command that has already
passed the frozen catalogue and external-command contract; HMC applies the
operating-mode and reserve policies during arbitration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .forecast.contracts import ForecastContracts
from .forecast.projection import ForecastHistory, MODE_ORDER
from .forecast_issue56_action_risk_v2 import (
    FEATURE_COUNT,
    HISTORY_WINDOW_STEPS,
    v2_feature_vector,
)


V4_TEMPORAL_FEATURE_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v4_temporal_features_v2"
HISTORY_FEATURE_COUNT = 194
V4_TEMPORAL_EXTRA_FEATURE_COUNT = HISTORY_FEATURE_COUNT * 3
V4_TEMPORAL_FEATURE_COUNT = FEATURE_COUNT + V4_TEMPORAL_EXTRA_FEATURE_COUNT
V4_TEMPORAL_BLOCKS = (
    "v2_past_only_projection",
    "window_slope",
    "recent_delta",
    "recent_volatility",
)


class Issue56V4FeatureError(ValueError):
    """Raised when a V4 feature or compatibility mask is malformed."""


def _history_is_complete(history: ForecastHistory) -> bool:
    if type(history) is not ForecastHistory:
        return False
    if history.numeric_f32.shape != (HISTORY_WINDOW_STEPS, HISTORY_FEATURE_COUNT):
        return False
    if history.numeric_f32.dtype != np.float32 or not np.isfinite(history.numeric_f32).all():
        return False
    if (
        len(history.steps) != HISTORY_WINDOW_STEPS
        or tuple(history.steps) != tuple(
            range(history.steps[0], history.steps[0] + HISTORY_WINDOW_STEPS)
        )
        or len(history.completed_times_s) != HISTORY_WINDOW_STEPS
    ):
        return False
    times = np.asarray(history.completed_times_s, dtype=np.float64)
    return bool(np.isfinite(times).all() and np.all(np.diff(times) > 0.0))


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise Issue56V4FeatureError(f"{label} is malformed or non-finite")
    result = result.astype(np.float32)
    result.setflags(write=False)
    return result


def observable_operating_mode(history: ForecastHistory) -> str:
    """Return the one-hot mode from the latest verified snapshot."""

    if not _history_is_complete(history):
        raise Issue56V4FeatureError("V4 mode requires a complete finite history")
    mode = np.asarray(history.mode_f32[-1], dtype=np.float64)
    if mode.shape != (len(MODE_ORDER),) or not np.all((mode == 0.0) | (mode == 1.0)):
        raise Issue56V4FeatureError("V4 observable mode is not one-hot")
    if int(np.sum(mode)) != 1:
        raise Issue56V4FeatureError("V4 observable mode is ambiguous")
    return MODE_ORDER[int(np.argmax(mode))]


def v4_observable_action_mask(
    bundle: ForecastContracts,
    history: ForecastHistory,
) -> tuple[bool, ...]:
    """Admit all validated catalogue actions for HMC policy arbitration.

    ``source_mode`` describes the catalogue entry; it is not an exclusivity
    constraint.  HMC remains responsible for applying the mode and reserve
    policies to each valid proposal.
    """

    if type(bundle) is not ForecastContracts or not _history_is_complete(history):
        raise Issue56V4FeatureError("V4 action mask requires frozen contracts and history")
    observable_operating_mode(history)
    actions = tuple(bundle.actions)
    if (
        len(actions) != 4
        or len({action.action_id for action in actions}) != len(actions)
        or len({action.source_mode for action in actions}) != len(actions)
        or any(action.source_mode not in MODE_ORDER for action in actions)
    ):
        raise Issue56V4FeatureError("V4 action catalogue identity is malformed")
    return (True,) * len(actions)


def v4_temporal_summary_blocks(
    numeric_f32: np.ndarray,
    completed_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate causal trend summaries from public numeric history only."""

    numeric = np.asarray(numeric_f32, dtype=np.float64)
    times = np.asarray(completed_times_s, dtype=np.float64)
    if (
        numeric.shape != (HISTORY_WINDOW_STEPS, HISTORY_FEATURE_COUNT)
        or times.shape != (HISTORY_WINDOW_STEPS,)
        or not np.isfinite(numeric).all()
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0.0)
    ):
        raise Issue56V4FeatureError("V4 temporal history inputs are malformed")
    centered_times = times - float(np.mean(times))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise Issue56V4FeatureError("V4 temporal time basis is degenerate")
    centered_values = numeric - np.mean(numeric, axis=0, keepdims=True)
    window_slope = (centered_times[:, None] * centered_values).sum(axis=0) / denominator
    recent_delta = numeric[-1] - numeric[-2]
    recent_volatility = np.std(np.diff(numeric, axis=0), axis=0)
    return tuple(
        _finite_vector(block, (HISTORY_FEATURE_COUNT,), "V4 temporal summary")
        for block in (window_slope, recent_delta, recent_volatility)
    )


def v4_temporal_feature_vector(
    history: ForecastHistory,
    action_f32: np.ndarray,
    *,
    decision_step: int | None = None,
    alarm_family_slots: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Build the V4 past-only vector from causal window summaries."""

    if not _history_is_complete(history):
        raise Issue56V4FeatureError("V4 temporal features require a complete finite history")
    if alarm_family_slots is None:
        raise Issue56V4FeatureError("V4 temporal features require alarm binding")
    try:
        baseline = v2_feature_vector(
            history,
            action_f32,
            decision_step=decision_step,
            alarm_family_slots=alarm_family_slots,
        )
    except Exception as error:
        raise Issue56V4FeatureError("V4 baseline feature projection failed") from error

    window_slope, recent_delta, recent_volatility = v4_temporal_summary_blocks(
        history.numeric_f32,
        history.completed_times_s,
    )
    values = np.concatenate(
        (
            np.asarray(baseline, dtype=np.float64),
            np.asarray(window_slope, dtype=np.float64),
            np.asarray(recent_delta, dtype=np.float64),
            np.asarray(recent_volatility, dtype=np.float64),
        )
    )
    return _finite_vector(
        values,
        (V4_TEMPORAL_FEATURE_COUNT,),
        "V4 temporal feature vector",
    )


def v4_feature_manifest(bundle: ForecastContracts) -> dict[str, Any]:
    """Return the hashable declaration used by corpus provenance."""

    if type(bundle) is not ForecastContracts:
        raise Issue56V4FeatureError("V4 feature manifest requires frozen contracts")
    return {
        "schema_version": V4_TEMPORAL_FEATURE_SCHEMA_VERSION,
        "history_window_steps": HISTORY_WINDOW_STEPS,
        "baseline_feature_count": FEATURE_COUNT,
        "temporal_feature_count": V4_TEMPORAL_FEATURE_COUNT,
        "temporal_extra_feature_count": V4_TEMPORAL_EXTRA_FEATURE_COUNT,
        "blocks": list(V4_TEMPORAL_BLOCKS),
        "causal_time_source": "verified_completed_times_s",
        "action_mask": {
            "source": "validated_catalogue_actions",
            "mode_metadata_is_not_exclusive": True,
            "catalogue_sha256": bundle.action_catalogue_sha256,
            "ordering": [action.action_id for action in bundle.actions],
        },
        "prohibited_inputs": [
            "future_measurements",
            "hidden_fault_truth",
            "hmc_arbitration_outcome",
        ],
    }


__all__ = [
    "Issue56V4FeatureError",
    "V4_TEMPORAL_BLOCKS",
    "V4_TEMPORAL_EXTRA_FEATURE_COUNT",
    "V4_TEMPORAL_FEATURE_COUNT",
    "V4_TEMPORAL_FEATURE_SCHEMA_VERSION",
    "observable_operating_mode",
    "v4_feature_manifest",
    "v4_observable_action_mask",
    "v4_temporal_summary_blocks",
    "v4_temporal_feature_vector",
]
