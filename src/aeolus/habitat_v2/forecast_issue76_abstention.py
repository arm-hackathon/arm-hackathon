"""Issue #76 calibrated selective prediction for the BDM-v1 adviser.

Conformal interval correction and a deterministic abstention policy fit on the
CALIBRATION partition only. The layer calibrates the pooled (seed-mean) TCN
quantile heads and abstains with explicit, receipt-visible reasons:

- ``INVALID_INPUT``: non-finite features or predictions (fail closed);
- ``STALE_OBSERVATIONS``: window staleness/mask beyond the calibrated range;
- ``SEED_DISAGREEMENT``: cross-seed delta spread beyond the calibrated range;
- ``WIDE_INTERVAL``: corrected p10-p90 width beyond the calibrated range.

Thresholds are declared CALIBRATION statistics (max/min observed staleness and
mask fraction, 95th percentiles of disagreement and interval width) frozen in
``contracts/habitat_v2_bdm_v1_abstention_thresholds_v1.json`` before any
closed-loop study. Uncertainty here is empirical calibration evidence, never a
formal safety guarantee. The policy proposes or abstains only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .bdm_v1_corpus import FEATURE_FIELD_NAMES
from .forecast.contracts import canonical_json_bytes
from .forecast_issue75_bdm import (
    DELTA_QUANTILE_SLICE,
    DELTA_SLICE,
    TRAJ_QUANTILE_SLICE,
    input_tensor,
)

ISSUE76_ABSTENTION_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_abstention_v1"
HORIZON_KEYS = ("4", "16", "32")
QUANTILE_LOW = 0
QUANTILE_HIGH = 2
NOMINAL_INTERVAL_COVERAGE = 0.8
ABSTENTION_REASONS = (
    "INVALID_INPUT",
    "STALE_OBSERVATIONS",
    "SEED_DISAGREEMENT",
    "WIDE_INTERVAL",
)
RECIPE_INPUT_FIELDS = (
    "observed_value_mask",
    "steps_since_last_valid_observation",
)


class Issue76AbstentionError(ValueError):
    """Raised when calibration inputs or layers are inadmissible."""


def _check_recipe() -> None:
    unknown = set(RECIPE_INPUT_FIELDS) - set(FEATURE_FIELD_NAMES)
    if unknown:
        raise Issue76AbstentionError(f"abstention recipe reads undeclared fields: {sorted(unknown)}")


_check_recipe()


@dataclass(frozen=True, slots=True)
class CalibrationLayer:
    """Conformal offsets plus frozen abstention thresholds."""

    delta_offsets: np.ndarray
    trajectory_offsets: np.ndarray
    max_staleness: float
    min_mask_fraction: float
    seed_disagreement: float
    interval_width: float
    digest: str

    def mapping(self) -> dict[str, Any]:
        return {
            "delta_offsets": self.delta_offsets.tolist(),
            "trajectory_offsets": self.trajectory_offsets.tolist(),
            "max_staleness": self.max_staleness,
            "min_mask_fraction": self.min_mask_fraction,
            "seed_disagreement": self.seed_disagreement,
            "interval_width": self.interval_width,
            "digest": self.digest,
        }


def fit_conformal_offsets(
    pooled_out: np.ndarray,
    trajectory_truths: np.ndarray,
    delta_truths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Additive conformal offsets per quantile slot (3 x targets)."""

    if pooled_out.ndim != 2 or not np.isfinite(pooled_out).all():
        raise Issue76AbstentionError("pooled outputs are malformed or non-finite")
    delta_q = pooled_out[:, DELTA_QUANTILE_SLICE[0] : DELTA_QUANTILE_SLICE[1]].reshape(
        pooled_out.shape[0], 3, 5
    )
    traj_q = pooled_out[:, TRAJ_QUANTILE_SLICE[0] : TRAJ_QUANTILE_SLICE[1]].reshape(
        pooled_out.shape[0], 3, 153
    )
    delta_residuals = delta_truths[:, None, :] - delta_q
    traj_residuals = trajectory_truths[:, None, :] - traj_q
    levels = np.asarray([0.1, 0.5, 0.9])
    delta_quantiles = np.quantile(delta_residuals, levels, axis=0)
    trajectory_quantiles = np.quantile(traj_residuals, levels, axis=0)
    delta_offsets = np.stack([delta_quantiles[p, p, :] for p in range(3)])
    trajectory_offsets = np.stack([trajectory_quantiles[p, p, :] for p in range(3)])
    if not np.isfinite(delta_offsets).all() or not np.isfinite(trajectory_offsets).all():
        raise Issue76AbstentionError("conformal offsets are non-finite")
    return delta_offsets, trajectory_offsets


def corrected_delta_interval(
    layer: CalibrationLayer, pooled_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected (p10, p90) bounds for the five delta targets."""

    delta_q = pooled_out[:, DELTA_QUANTILE_SLICE[0] : DELTA_QUANTILE_SLICE[1]].reshape(
        pooled_out.shape[0], 3, 5
    )
    low = delta_q[:, QUANTILE_LOW, :] + layer.delta_offsets[QUANTILE_LOW]
    high = delta_q[:, QUANTILE_HIGH, :] + layer.delta_offsets[QUANTILE_HIGH]
    return low, high


def corrected_trajectory_interval(
    layer: CalibrationLayer, pooled_out: np.ndarray, horizon_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected (p10, p90) bounds for one horizon's 153 trajectory targets."""

    if not 0 <= horizon_index < len(HORIZON_KEYS):
        raise Issue76AbstentionError("horizon index out of range")
    traj_q = pooled_out[:, TRAJ_QUANTILE_SLICE[0] : TRAJ_QUANTILE_SLICE[1]].reshape(
        pooled_out.shape[0], 3, 153
    )
    columns = slice(horizon_index * 51, (horizon_index + 1) * 51)
    low = traj_q[:, QUANTILE_LOW, columns] + layer.trajectory_offsets[QUANTILE_LOW][columns]
    high = traj_q[:, QUANTILE_HIGH, columns] + layer.trajectory_offsets[QUANTILE_HIGH][columns]
    return low, high


def interval_coverage(
    layer: CalibrationLayer,
    pooled_out: np.ndarray,
    delta_truths: np.ndarray,
) -> float:
    """Marginal coverage: fraction of target values inside corrected bounds."""

    low, high = corrected_delta_interval(layer, pooled_out)
    inside = (delta_truths >= low) & (delta_truths <= high)
    return float(inside.mean())


def trajectory_interval_coverage(
    layer: CalibrationLayer,
    pooled_out: np.ndarray,
    trajectory_truths: np.ndarray,
    horizon_index: int,
) -> float:
    low, high = corrected_trajectory_interval(layer, pooled_out, horizon_index)
    truth = trajectory_truths[:, horizon_index * 51 : (horizon_index + 1) * 51]
    inside = (truth >= low) & (truth <= high)
    return float(inside.mean())


def sample_staleness_stats(sample: Mapping[str, Any]) -> tuple[float, float]:
    """Max staleness and mean observed-mask fraction over the window."""

    features = sample["features"]
    stale = np.asarray(features["steps_since_last_valid_observation"], dtype=np.float64)
    mask = np.asarray(features["observed_value_mask"], dtype=np.float64)
    return float(stale.max()), float(mask.mean())


def fit_thresholds(
    samples: Sequence[Mapping[str, Any]],
    per_seed_deltas: np.ndarray,
    layer_offsets: tuple[np.ndarray, np.ndarray],
    pooled_out: np.ndarray,
) -> dict[str, float]:
    """Declared CALIBRATION statistics used as abstention thresholds."""

    staleness_values = []
    mask_values = []
    for sample in samples:
        max_stale, mask_fraction = sample_staleness_stats(sample)
        staleness_values.append(max_stale)
        mask_values.append(mask_fraction)
    disagreement = np.std(per_seed_deltas[:, :, 1], axis=0)
    layer = CalibrationLayer(
        layer_offsets[0], layer_offsets[1], 0.0, 0.0, 0.0, 0.0, ""
    )
    low, high = corrected_delta_interval(layer, pooled_out)
    widths = high[:, 1] - low[:, 1]
    return {
        "max_staleness": float(np.max(staleness_values)),
        "min_mask_fraction": float(np.min(mask_values)),
        "seed_disagreement": float(np.quantile(disagreement, 0.95)),
        "interval_width": float(np.quantile(widths, 0.95)),
    }


def layer_digest(
    delta_offsets: np.ndarray,
    trajectory_offsets: np.ndarray,
    thresholds: Mapping[str, float],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "delta_offsets": delta_offsets.tolist(),
                "trajectory_offsets": trajectory_offsets.tolist(),
                "thresholds": dict(thresholds),
            }
        )
    ).hexdigest()


def abstention_reason(
    sample: Mapping[str, Any],
    per_seed_deltas: np.ndarray,
    layer: CalibrationLayer,
    pooled_out_row: np.ndarray,
) -> str | None:
    """Deterministic abstention decision with explicit reason ordering."""

    try:
        tensor = input_tensor(sample)
    except Exception:
        return "INVALID_INPUT"
    if not np.isfinite(tensor).all() or not np.isfinite(pooled_out_row).all():
        return "INVALID_INPUT"
    max_stale, mask_fraction = sample_staleness_stats(sample)
    if max_stale > layer.max_staleness or mask_fraction < layer.min_mask_fraction:
        return "STALE_OBSERVATIONS"
    disagreement = float(np.std(per_seed_deltas[:, 1]))
    if disagreement > layer.seed_disagreement:
        return "SEED_DISAGREEMENT"
    low, high = corrected_delta_interval(
        layer, pooled_out_row[None, :]
    )
    width = float(high[0, 1] - low[0, 1])
    if width > layer.interval_width:
        return "WIDE_INTERVAL"
    return None


def predict_with_abstention(
    sample: Mapping[str, Any],
    per_seed_deltas: np.ndarray,
    layer: CalibrationLayer,
    pooled_out_row: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    reason = abstention_reason(sample, per_seed_deltas, layer, pooled_out_row)
    if reason is not None:
        return None, reason
    return pooled_out_row[DELTA_SLICE[0] : DELTA_SLICE[1]], None


def risk_coverage_curve(
    losses: Sequence[float],
    widths: Sequence[float],
    multipliers: Sequence[float],
    base_width_threshold: float,
) -> list[dict[str, float]]:
    """Risk (mean loss on retained) versus coverage over width thresholds."""

    curve = []
    losses_arr = np.asarray([float(value) for value in losses], dtype=np.float64)
    widths_arr = np.asarray([float(value) for value in widths], dtype=np.float64)
    for multiplier in multipliers:
        threshold = base_width_threshold * float(multiplier)
        retained = widths_arr <= threshold
        coverage = float(np.mean(retained))
        risk = float(np.mean(losses_arr[retained])) if retained.any() else 0.0
        curve.append({"multiplier": float(multiplier), "coverage": coverage, "risk": risk})
    return curve


def useful_opportunity_recall(
    predicted_improving: Sequence[bool],
    true_improving: Sequence[bool],
    abstained: Sequence[bool],
) -> tuple[float, float]:
    """Recall over non-abstained samples and the abstained-useful rate."""

    true_flags = [bool(value) for value in true_improving]
    pred_flags = [bool(value) for value in predicted_improving]
    abstain_flags = [bool(value) for value in abstained]
    true_positives = sum(
        1
        for predicted, truth, abstain in zip(pred_flags, true_flags, abstain_flags, strict=True)
        if predicted and truth and not abstain
    )
    true_count = sum(1 for truth in true_flags if truth)
    abstained_useful = sum(
        1 for truth, abstain in zip(true_flags, abstain_flags, strict=True) if truth and abstain
    )
    recall = true_positives / true_count if true_count else 0.0
    abstained_useful_rate = abstained_useful / true_count if true_count else 0.0
    return recall, abstained_useful_rate


def abstention_guard(abstained_useful_rate: float, max_allowed: float) -> bool:
    """Near-total abstention cannot count as success."""

    return abstained_useful_rate <= max_allowed


__all__ = [
    "ABSTENTION_REASONS",
    "CalibrationLayer",
    "HORIZON_KEYS",
    "ISSUE76_ABSTENTION_SCHEMA_VERSION",
    "Issue76AbstentionError",
    "NOMINAL_INTERVAL_COVERAGE",
    "QUANTILE_HIGH",
    "QUANTILE_LOW",
    "RECIPE_INPUT_FIELDS",
    "abstention_guard",
    "abstention_reason",
    "corrected_delta_interval",
    "corrected_trajectory_interval",
    "fit_conformal_offsets",
    "fit_thresholds",
    "interval_coverage",
    "layer_digest",
    "predict_with_abstention",
    "risk_coverage_curve",
    "sample_staleness_stats",
    "trajectory_interval_coverage",
    "useful_opportunity_recall",
]
