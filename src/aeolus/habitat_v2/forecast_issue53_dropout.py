"""Issue #53 dropout-robust lane — deterministic observation-only sensor dropout.

This module stacks on the frozen Issue #52 lane. It does not mutate the
Issue #52 artifact, manifest, or HMC authority. Dropout is observation-only:
truth ``PlantState`` and ``fault_receipt`` never change. The mask is a
deterministic SHA256-derived view over ``ForecastHistory.available_mask``.

Design reference:
 ``docs/plans/2026-08-22-issue-53-missing-sensors-plan.md`` and
 ``contracts/habitat_v2_forecast_issue_53_preregistration_v1.json``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .forecast_issue52 import (
    CADENCE_SECONDS,
    HORIZON_STEPS,
    HISTORY_STEPS,
    CandidateSchedule,
    ForecastHistory,
    ForecastTrajectory,
    Issue52ForecastError,
    TargetManifest,
    TrainingSample,
    _command_vector,
    _readonly,
)
from .hmc_contract import canonical_json_bytes
from .scenario import Scenario


ISSUE53_SCHEMA_VERSION = "aeolus_habitat_v2_forecast_issue_53_v1"
DROPOUT_SCHEMA_VERSION = "aeolus_habitat_v2_dropout_v1"
FORECAST_METRIC_START = 8
MIN_INTERVAL_SCALE = float(np.finfo(np.float32).eps)


class Issue53ContractError(ValueError):
    """Raised when dropout config or derived history violates its contract."""


class Issue53ForecastError(ValueError):
    """Raised when a dropout-aware forecast cannot be produced safely."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(c in "0123456789abcdef" for c in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise Issue53ContractError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Issue53ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Issue53ContractError(f"{label} must be finite")
    return result


# ---------------------------------------------------------------------------
# Dropout config — frozen identity, never mutates truth
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DropoutConfig:
    p_uniform: float = 0.05
    mode: str = "independent"
    burst_min: int = 2
    burst_max: int = 8
    p_burst_onset: float = 0.02
    resource_gauge_dropout: bool = False
    max_missing_per_row: int | None = 6
    seed: int = 530053
    config_sha256: str = field(default="")

    def __post_init__(self) -> None:
        if (
            isinstance(self.p_uniform, bool)
            or not isinstance(self.p_uniform, (int, float))
            or not math.isfinite(float(self.p_uniform))
            or not 0.0 <= float(self.p_uniform) < 1.0
        ):
            raise Issue53ContractError("p_uniform must be in [0,1)")
        if type(self.mode) is not str or self.mode not in {
            "independent",
            "per_zone_head_burst",
            "mixed",
        }:
            raise Issue53ContractError("dropout mode is invalid")
        if (
            type(self.burst_min) is not int
            or type(self.burst_max) is not int
        ):
            raise Issue53ContractError("burst bounds must be integers")
        if self.burst_min < 1 or self.burst_max < self.burst_min:
            raise Issue53ContractError("burst bounds are invalid")
        if (
            isinstance(self.p_burst_onset, bool)
            or not isinstance(self.p_burst_onset, (int, float))
            or not math.isfinite(float(self.p_burst_onset))
            or not 0.0 <= float(self.p_burst_onset) <= 1.0
        ):
            raise Issue53ContractError("p_burst_onset must be in [0,1]")
        if type(self.resource_gauge_dropout) is not bool:
            raise Issue53ContractError("resource_gauge_dropout must be bool")
        if self.max_missing_per_row is not None:
            if type(self.max_missing_per_row) is not int or self.max_missing_per_row < 0:
                raise Issue53ContractError("max_missing_per_row is invalid")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise Issue53ContractError("seed must be int")
        # canonical digest
        payload = {
            "schema_version": DROPOUT_SCHEMA_VERSION,
            "p_uniform": float(self.p_uniform),
            "mode": self.mode,
            "burst_min": int(self.burst_min),
            "burst_max": int(self.burst_max),
            "p_burst_onset": float(self.p_burst_onset),
            "resource_gauge_dropout": bool(self.resource_gauge_dropout),
            "max_missing_per_row": self.max_missing_per_row,
            "seed": int(self.seed),
        }
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if not self.config_sha256:
            object.__setattr__(self, "config_sha256", expected)
        elif self.config_sha256 != expected:
            raise Issue53ContractError("dropout config digest is inconsistent")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DropoutConfig":
        if not isinstance(value, Mapping):
            raise Issue53ContractError("dropout config must be a mapping")
        schema_version = value.get("schema_version", DROPOUT_SCHEMA_VERSION)
        if schema_version != DROPOUT_SCHEMA_VERSION:
            raise Issue53ContractError("dropout config schema is invalid")
        resource_gauge_dropout = value.get("resource_gauge_dropout", False)
        if type(resource_gauge_dropout) is not bool:
            raise Issue53ContractError("resource_gauge_dropout must be bool")
        config_sha256 = value.get("config_sha256", "")
        if type(config_sha256) is not str:
            raise Issue53ContractError("config digest must be a string")
        return cls(
            p_uniform=value.get("p_uniform", 0.05),
            mode=value.get("mode", "independent"),
            burst_min=value.get("burst_min", 2),
            burst_max=value.get("burst_max", 8),
            p_burst_onset=value.get("p_burst_onset", 0.02),
            resource_gauge_dropout=resource_gauge_dropout,
            max_missing_per_row=value.get("max_missing_per_row", 6),
            seed=value.get("seed", 530053),
            config_sha256=config_sha256,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": DROPOUT_SCHEMA_VERSION,
            "p_uniform": float(self.p_uniform),
            "mode": self.mode,
            "burst_min": int(self.burst_min),
            "burst_max": int(self.burst_max),
            "p_burst_onset": float(self.p_burst_onset),
            "resource_gauge_dropout": bool(self.resource_gauge_dropout),
            "max_missing_per_row": self.max_missing_per_row,
            "seed": int(self.seed),
            "config_sha256": self.config_sha256,
        }


# ---------------------------------------------------------------------------
# Deterministic mask sampler — SHA256, no RNG
# ---------------------------------------------------------------------------

def _sampler_u64(seed: int, family_id: str, decision_step: int, step_offset: int, descriptor_id: str) -> int:
    payload = f"{seed}|{family_id}|{decision_step}|{step_offset}|{descriptor_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _should_drop(u64: int, p: float) -> bool:
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    # threshold in [0, 2**64)
    threshold = int(p * (1 << 64))
    return u64 < threshold


def _burst_group(descriptor: object) -> str:
    """Bind a burst to one zone's environmental sensor head."""

    descriptor_id = str(descriptor.descriptor_id)
    return descriptor_id.split("/", 1)[0] if "/" in descriptor_id else descriptor_id


def _burst_active(
    config: DropoutConfig,
    *,
    family_id: str,
    decision_step: int,
    step_offset: int,
    group_id: str,
) -> bool:
    for onset in range(HISTORY_STEPS):
        onset_u64 = _sampler_u64(
            config.seed, family_id, decision_step, onset, f"{group_id}:burst-onset"
        )
        if not _should_drop(onset_u64, config.p_burst_onset):
            continue
        duration_u64 = _sampler_u64(
            config.seed, family_id, decision_step, onset, f"{group_id}:burst-duration"
        )
        duration = config.burst_min + duration_u64 % (
            config.burst_max - config.burst_min + 1
        )
        if onset <= step_offset < onset + duration:
            return True
    return False


def dropout_mask_for_history(
    history: ForecastHistory,
    manifest: TargetManifest,
    config: DropoutConfig,
    *,
    family_id: str,
    decision_step: int,
    latest_missing_count: int | None = None,
) -> np.ndarray:
    """Deterministic boolean mask: True = keep (AVAILABLE), False = drop."""
    if type(history) is not ForecastHistory or type(manifest) is not TargetManifest or type(config) is not DropoutConfig:
        raise Issue53ContractError("dropout mask requires exact contract types")
    if history.target_values.shape[1] != manifest.width:
        raise Issue53ContractError("history width does not bind manifest")
    if latest_missing_count is not None and (
        not isinstance(latest_missing_count, int)
        or isinstance(latest_missing_count, bool)
        or latest_missing_count < 0
    ):
        raise Issue53ContractError("latest_missing_count is invalid")
    # Native missing (pre-existing UNAVAILABLE) stays dropped — dropout is additive
    base_available = history.available_mask.copy()
    # Resource gauges are the last 3 descriptors when resource_gauge_dropout is False
    gauge_indices: set[int] = set()
    if not config.resource_gauge_dropout:
        gauge_ids = {"battery_state_of_charge", "oxygen_store_fraction", "sorbent_remaining_fraction"}
        for idx, desc in enumerate(manifest.descriptors):
            if desc.descriptor_id in gauge_ids:
                gauge_indices.add(idx)

    mask = np.ones_like(base_available, dtype=bool)
    for t in range(HISTORY_STEPS):
        for c in range(manifest.width):
            if not base_available[t, c]:
                mask[t, c] = False
                continue
            if c in gauge_indices:
                # Never drop gauges unless explicitly enabled.
                continue
            u64 = _sampler_u64(config.seed, family_id, decision_step, t, manifest.descriptors[c].descriptor_id)
            independent = config.mode in {"independent", "mixed"} and _should_drop(
                u64, config.p_uniform
            )
            burst = config.mode in {"per_zone_head_burst", "mixed"} and _burst_active(
                config,
                family_id=family_id,
                decision_step=decision_step,
                step_offset=t,
                group_id=_burst_group(manifest.descriptors[c]),
            )
            if independent or burst:
                mask[t, c] = False

    if latest_missing_count is not None:
        native_missing = int(np.sum(~base_available[-1]))
        if latest_missing_count < native_missing:
            raise Issue53ContractError(
                "latest_missing_count cannot remove native unavailable channels"
            )
        eligible = [
            c
            for c in range(manifest.width)
            if base_available[-1, c] and c not in gauge_indices
        ]
        required_dropout = latest_missing_count - native_missing
        if required_dropout > len(eligible):
            raise Issue53ContractError(
                "latest_missing_count exceeds eligible observable channels"
            )
        if config.max_missing_per_row is not None and required_dropout > config.max_missing_per_row:
            raise Issue53ContractError(
                "latest_missing_count exceeds max_missing_per_row"
            )
        mask[-1, :] = base_available[-1, :]
        eligible.sort(
            key=lambda c: (
                _sampler_u64(
                    config.seed,
                    family_id,
                    decision_step,
                    HISTORY_STEPS - 1,
                    f"{manifest.descriptors[c].descriptor_id}:target-k",
                ),
                manifest.descriptors[c].descriptor_id,
            )
        )
        for c in eligible[:required_dropout]:
            mask[-1, c] = False

    # Optional cap — deterministically retain at most this many new drops per row.
    if config.max_missing_per_row is not None:
        cap = int(config.max_missing_per_row)
        for t in range(HISTORY_STEPS):
            dropped = int(np.sum(~mask[t] & base_available[t]))
            # count only dropout-induced drops, ignore pre-existing unavailable
            if dropped > cap:
                dropout_cols = [c for c in range(manifest.width) if not mask[t, c] and base_available[t, c]]
                dropout_cols.sort(
                    key=lambda c: (
                        _sampler_u64(
                            config.seed,
                            family_id,
                            decision_step,
                            t,
                            f"{manifest.descriptors[c].descriptor_id}:cap",
                        ),
                        manifest.descriptors[c].descriptor_id,
                    )
                )
                mask[t, :] = base_available[t, :]
                for c in dropout_cols[:cap]:
                    mask[t, c] = False

    mask.setflags(write=False)
    return mask


def apply_dropout_to_history(
    history: ForecastHistory,
    manifest: TargetManifest,
    config: DropoutConfig,
    *,
    family_id: str,
    decision_step: int,
    latest_missing_count: int | None = None,
) -> ForecastHistory:
    """Return a new ForecastHistory with dropout mask applied (observation-only)."""
    mask = dropout_mask_for_history(
        history,
        manifest,
        config,
        family_id=family_id,
        decision_step=decision_step,
        latest_missing_count=latest_missing_count,
    )
    # New target_values: NaN where newly masked
    new_values = history.target_values.astype(np.float32).copy()
    new_values[~mask] = np.nan
    # time array unchanged — dropout never leaks future availability
    new_values.setflags(write=False)
    mask.setflags(write=False)
    times = history.completed_times_s.copy()
    times.setflags(write=False)
    return ForecastHistory(history.records, new_values, mask, times)


# ---------------------------------------------------------------------------
# Mask-aware feature construction
# ---------------------------------------------------------------------------

def _masked_slope(target_values: np.ndarray, available_mask: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Slope per column using only available rows — mask-aware variant of ForecastHistory.slope."""
    width = target_values.shape[1]
    result = np.zeros(width, dtype=np.float32)
    x = times.astype(np.float64)
    for col in range(width):
        m = available_mask[:, col]
        if int(m.sum()) < 2:
            continue
        xm = x[m]
        ym = target_values[m, col].astype(np.float64)
        # filter any remaining non-finite (should not happen after imputation)
        finite = np.isfinite(ym) & np.isfinite(xm)
        if int(finite.sum()) < 2:
            continue
        xm = xm[finite]
        ym = ym[finite]
        centered = xm - xm.mean()
        denom = float(np.dot(centered, centered))
        if denom > 0.0:
            result[col] = np.float32(np.dot(centered, ym - ym.mean()) / denom)
    result.setflags(write=False)
    return result


def impute_history_values(
    target_values: np.ndarray,
    available_mask: np.ndarray,
    manifest: TargetManifest,
) -> np.ndarray:
    """Deterministic forward-fill per column, fallback to descriptor nominal."""
    values = target_values.astype(np.float64).copy()
    width = manifest.width
    for col in range(width):
        nominal = float(manifest.descriptors[col].nominal)
        last = nominal
        for t in range(HISTORY_STEPS):
            if available_mask[t, col] and math.isfinite(float(values[t, col])):
                last = float(values[t, col])
            else:
                values[t, col] = last
        # if first rows were missing, they are now nominal — ensures NaN-free
    # final NaN check
    if not np.isfinite(values).all():
        raise Issue53ForecastError("imputed history still contains non-finite values")
    result = values.astype(np.float32)
    result.setflags(write=False)
    return result


def _masked_feature_matrix(
    history: ForecastHistory,
    schedule: CandidateSchedule,
    scenario: Scenario,
    manifest: TargetManifest,
) -> np.ndarray:
    """Mask-aware replacement for _feature_matrix at forecast_issue52.py:1394."""
    # Do NOT raise on unavailable latest row — impute instead
    if history.target_values.shape[1] != manifest.width:
        raise Issue53ForecastError("history width does not bind manifest")
    imputed = impute_history_values(history.target_values, history.available_mask, manifest)
    # mask as float
    mask_float = history.available_mask.astype(np.float64)
    # time-since-observed per column
    age = np.zeros((HISTORY_STEPS, manifest.width), dtype=np.float64)
    for col in range(manifest.width):
        last_seen = -1
        for t in range(HISTORY_STEPS):
            if history.available_mask[t, col]:
                last_seen = t
            age[t, col] = float(t - last_seen) if last_seen >= 0 else float(HISTORY_STEPS)
    # latest imputed + masked slope
    latest = imputed[-1].astype(np.float64)
    slope_age = _masked_slope(imputed, history.available_mask, history.completed_times_s).astype(np.float64) / CADENCE_SECONDS
    # schedule action vector (same as Issue 52)
    rows: list[np.ndarray] = []
    for horizon, command in enumerate(schedule.commands):
        action = _command_vector(scenario, command.to_mapping()).astype(np.float64)
        # history summary: imputed latest, slope, mask density, age
        mask_density = float(np.mean(history.available_mask[-1].astype(np.float64)))
        rows.append(
            np.concatenate(
                (
                    [1.0],
                    latest,
                    slope_age,
                    mask_float[-1],  # per-channel keep indicator for latest row
                    age[-1] / float(HISTORY_STEPS),
                    [mask_density],
                    action,
                    [horizon / float(HORIZON_STEPS - 1)],
                )
            )
        )
    mat = np.stack(rows)
    if not np.isfinite(mat).all():
        raise Issue53ForecastError("masked feature matrix contains non-finite values")
    return mat


def _observation_uncertainty_factor(history: ForecastHistory) -> float:
    """Return a truth-independent multiplier for irregular observation age."""

    latest_age: list[float] = []
    for column in range(history.available_mask.shape[1]):
        available_rows = np.flatnonzero(history.available_mask[:, column])
        latest_age.append(
            float(HISTORY_STEPS)
            if len(available_rows) == 0
            else float(HISTORY_STEPS - 1 - available_rows[-1])
        )
    mean_age = float(np.mean(latest_age)) if latest_age else 0.0
    missing_density = float(np.mean(~history.available_mask))
    return 1.0 + 0.10 * mean_age + 0.50 * missing_density


def _missingness_risk_score(
    history: ForecastHistory,
    coefficients: np.ndarray | None,
    width: int,
) -> float:
    """Score missing latest channels by their learned observation sensitivity."""

    if coefficients is None or width <= 0:
        return 0.0
    missing = ~history.available_mask[-1]
    if not np.any(missing):
        return 0.0
    # Latest value, slope, mask, and observation-age blocks are the only
    # observation features that change when a channel is unavailable.
    blocks = np.stack(
        (
            coefficients[1 : 1 + width],
            coefficients[1 + width : 1 + 2 * width],
            coefficients[1 + 2 * width : 1 + 3 * width],
            coefficients[1 + 3 * width : 1 + 4 * width],
        ),
        axis=0,
    )
    sensitivity = np.sqrt(np.sum(np.square(blocks.astype(np.float64)), axis=(0, 2)))
    normalizer = max(float(np.median(sensitivity)), 1e-9)
    return float(np.sum(sensitivity[missing]) / normalizer)


def _risk_feature_vector(
    history: ForecastHistory,
    schedule: CandidateSchedule,
    scenario: Scenario,
    manifest: TargetManifest,
    coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Build an observation-only risk feature vector for selective prediction."""

    features = _masked_feature_matrix(history, schedule, scenario, manifest)
    action_width = 3 + len(scenario.data["air_network"]["branches"]) + 2 * len(
        scenario.data["zones"]
    )
    expected_feature_width = 4 * manifest.width + action_width + 3
    if features.shape[1] != expected_feature_width:
        raise Issue53ForecastError("risk feature matrix width is invalid")
    action_start = 1 + 4 * manifest.width + 1
    action_end = features.shape[1] - 1
    imputed = impute_history_values(
        history.target_values, history.available_mask, manifest
    )[-1].astype(np.float64)
    scales = np.asarray(
        [descriptor.scale for descriptor in manifest.descriptors], dtype=np.float64
    )
    drift = 0.0
    if coefficients is not None:
        prediction = features @ np.asarray(coefficients, dtype=np.float64)
        drift = float(
            np.mean(
                np.abs(prediction[FORECAST_METRIC_START:] - imputed[None, :])
                / scales[None, :]
            )
        )
    return np.concatenate(
        (
            features[0],
            np.mean(features[:, action_start:action_end], axis=0),
            np.asarray(
                [
                    float(history.latest_record.mode == mode)
                    for mode in ("occupied", "eva_transition", "contingency", "dormant")
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    _observation_uncertainty_factor(history),
                    _missingness_risk_score(
                        history, coefficients, manifest.width
                    ),
                    drift,
                ],
                dtype=np.float64,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Dropout-aware linear forecaster — development evidence only
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DropoutAwareLinearForecaster:
    scenario: Scenario
    manifest: TargetManifest
    dropout_config: DropoutConfig
    coefficients: np.ndarray | None = field(default=None, compare=False)
    model_id: str = "issue53-dropout-linear-v1"
    # per-k interval scale: conformal residual quantile per k bin
    per_k_interval_scale: Mapping[int, float] = field(default_factory=dict, compare=False)
    abstention_width_limit: float = field(default=8.0, compare=False)
    abstention_min_k: int = field(default=3, compare=False)
    abstention_risk_limit: float = field(default=1e9, compare=False)
    risk_coefficients: np.ndarray | None = field(default=None, compare=False)
    risk_center: np.ndarray | None = field(default=None, compare=False)
    risk_scale: np.ndarray | None = field(default=None, compare=False)
    risk_intercept: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if type(self.scenario) is not Scenario or type(self.manifest) is not TargetManifest or type(self.dropout_config) is not DropoutConfig:
            raise Issue53ForecastError("dropout forecaster requires exact scenario, manifest, and dropout config")
        if self.manifest.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue53ForecastError("dropout forecaster manifest does not bind scenario")
        if type(self.model_id) is not str or not self.model_id:
            raise Issue53ForecastError("dropout forecaster model identity is invalid")
        if (
            isinstance(self.abstention_width_limit, bool)
            or not isinstance(self.abstention_width_limit, (int, float))
            or not math.isfinite(float(self.abstention_width_limit))
            or float(self.abstention_width_limit) <= 0.0
        ):
            raise Issue53ForecastError("dropout abstention width limit is invalid")
        if (
            type(self.abstention_min_k) is not int
            or self.abstention_min_k < 0
            or self.abstention_min_k > 6
        ):
            raise Issue53ForecastError("dropout abstention missingness level is invalid")
        if (
            isinstance(self.abstention_risk_limit, bool)
            or not isinstance(self.abstention_risk_limit, (int, float))
            or not math.isfinite(float(self.abstention_risk_limit))
            or float(self.abstention_risk_limit) < 0.0
        ):
            raise Issue53ForecastError("dropout abstention risk limit is invalid")
        if (
            isinstance(self.risk_intercept, bool)
            or not isinstance(self.risk_intercept, (int, float))
            or not math.isfinite(float(self.risk_intercept))
            or float(self.risk_intercept) < 0.0
        ):
            raise Issue53ForecastError("dropout risk intercept is invalid")
        if self.coefficients is not None:
            vals = np.asarray(self.coefficients, dtype=np.float32)
            action_width = 3 + len(self.scenario.data["air_network"]["branches"]) + 2 * len(
                self.scenario.data["zones"]
            )
            expected_features = 4 * self.manifest.width + action_width + 3
            if (
                vals.ndim != 2
                or vals.shape[1] != self.manifest.width
                or vals.shape[0] != expected_features
                or not np.isfinite(vals).all()
            ):
                raise Issue53ForecastError("dropout forecaster coefficients are invalid")
            object.__setattr__(self, "coefficients", _readonly(vals))
        for name in ("risk_coefficients", "risk_center", "risk_scale"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise Issue53ForecastError(f"{name} are invalid")
                object.__setattr__(self, name, _readonly(array))
        risk_arrays = (self.risk_coefficients, self.risk_center, self.risk_scale)
        if any(value is not None for value in risk_arrays):
            if any(value is None for value in risk_arrays):
                raise Issue53ForecastError("risk head parameters are incomplete")
            lengths = {int(np.asarray(value).shape[0]) for value in risk_arrays if value is not None}
            action_width = 3 + len(self.scenario.data["air_network"]["branches"]) + 2 * len(
                self.scenario.data["zones"]
            )
            expected_risk_width = 4 * self.manifest.width + 2 * action_width + 10
            if (
                len(lengths) != 1
                or lengths != {expected_risk_width}
                or not np.all(np.asarray(self.risk_scale) > 0.0)
            ):
                raise Issue53ForecastError("risk head parameter shapes are invalid")
        # freeze per_k scale
        frozen = MappingProxyType(dict(self.per_k_interval_scale))
        object.__setattr__(self, "per_k_interval_scale", frozen)

    @classmethod
    def fit(
        cls,
        samples: Sequence[TrainingSample],
        *,
        scenario: Scenario,
        manifest: TargetManifest | None = None,
        dropout_config: DropoutConfig | None = None,
        alpha: float = 1e-6,
    ) -> "DropoutAwareLinearForecaster":
        bound_manifest = TargetManifest.from_scenario(scenario) if manifest is None else manifest
        config = DropoutConfig() if dropout_config is None else dropout_config
        return cls.fit_for_scenario(scenario, bound_manifest, samples, dropout_config=config, alpha=alpha)

    @classmethod
    def fit_for_scenario(
        cls,
        scenario: Scenario,
        manifest: TargetManifest,
        samples: Sequence[TrainingSample],
        *,
        dropout_config: DropoutConfig,
        alpha: float = 1e-6,
        augment_dropout: bool = True,
    ) -> "DropoutAwareLinearForecaster":
        items = tuple(samples)
        if len(items) < 3 or len({item.family_id for item in items}) < 2:
            raise Issue53ForecastError("fitting requires at least three samples and two families")
        if any(item.split != "TRAIN" for item in items):
            raise Issue53ForecastError("fitting accepts TRAIN samples only")
        if type(scenario) is not Scenario or type(manifest) is not TargetManifest or type(dropout_config) is not DropoutConfig:
            raise Issue53ForecastError("fit inputs are not exact contract types")
        if manifest.scenario_sha256 != scenario.scenario_sha256:
            raise Issue53ForecastError("fit manifest does not bind reference scenario")
        if any(
            item.history.target_values.shape[1] != manifest.width
            or any(
                record.topology_sha256 != manifest.topology_sha256
                for record in item.history.records
            )
            for item in items
        ):
            raise Issue53ForecastError("fit samples do not bind reference topology")
        try:
            for item in items:
                for command in item.schedule.commands:
                    _command_vector(scenario, command.to_mapping())
        except (Issue52ForecastError, KeyError, TypeError, ValueError) as error:
            raise Issue53ForecastError("fit samples contain incompatible commands") from error
        if not math.isfinite(float(alpha)) or alpha <= 0.0:
            raise Issue53ForecastError("fit regularization must be positive and finite")
        # Augment each sample's history with deterministic dropout (same family/decision binding)
        augmented: list[TrainingSample] = []
        for item in items:
            # Derive family_id and decision_step from checkpoint if possible, else fallback
            family_id = str(item.family_id)
            # decision_step approximated as history latest completed_step if checkpoint not directly available
            # TrainingSample stores history with completed_times — use last record step
            try:
                decision_step = int(item.history.records[-1].completed_step)
            except Exception:
                decision_step = 15
            masked_history = (
                apply_dropout_to_history(
                    item.history,
                    manifest,
                    dropout_config,
                    family_id=family_id,
                    decision_step=decision_step,
                )
                if augment_dropout
                else item.history
            )
            augmented.append(
                TrainingSample(
                    family_id=item.family_id,
                    split=item.split,
                    scenario_sha256=item.scenario_sha256,
                    manifest_sha256=item.manifest_sha256,
                    checkpoint_sha256=item.checkpoint_sha256,
                    schedule_sha256=item.schedule_sha256,
                    history=masked_history,
                    schedule=item.schedule,
                    targets=item.targets,
                )
            )
        # feature / target construction — targets remain complete (dropout never masks labels)
        features = np.concatenate([_masked_feature_matrix(s.history, s.schedule, scenario, manifest) for s in augmented], axis=0).astype(np.float64)
        targets = np.concatenate([s.targets for s in augmented], axis=0).astype(np.float64)
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise Issue53ForecastError("training data contains non-finite values")
        gram = features.T @ features
        reg = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
        try:
            coeffs = np.linalg.solve(gram + reg, features.T @ targets)
        except np.linalg.LinAlgError as e:
            raise Issue53ForecastError("dropout-aware fit is singular") from e
        coeffs = coeffs.astype(np.float32)
        coeffs.setflags(write=False)
        model_id = "issue53-dropout-linear-" + hashlib.sha256(coeffs.tobytes()).hexdigest()[:16]
        # Calibration is deliberately separate and must consume VALIDATION data.
        return cls(scenario, manifest, dropout_config, coeffs, model_id, {})

    def calibrate(
        self,
        samples: Sequence[TrainingSample],
        *,
        quantile: float = 0.90,
        oracle_errors: Sequence[float] | None = None,
        oracle_high_error_threshold: float | None = None,
        abstention_k: int | None = None,
    ) -> "DropoutAwareLinearForecaster":
        """Calibrate normalized interval scales on VALIDATION only."""

        if (
            isinstance(quantile, bool)
            or not isinstance(quantile, (int, float))
            or not math.isfinite(float(quantile))
            or not 0.0 < float(quantile) < 1.0
        ):
            raise Issue53ForecastError("calibration quantile must be in (0,1)")
        items = tuple(samples)
        if not items or any(item.split != "VALIDATION" for item in items):
            raise Issue53ForecastError("interval calibration accepts VALIDATION samples only")
        if self.coefficients is None:
            raise Issue53ForecastError("calibration requires fitted coefficients")
        scales = np.asarray(
            [descriptor.scale for descriptor in self.manifest.descriptors],
            dtype=np.float64,
        )
        residuals: dict[int, list[float]] = {}
        for item in items:
            if item.history.target_values.shape[1] != self.manifest.width or any(
                record.topology_sha256 != self.manifest.topology_sha256
                for record in item.history.records
            ):
                raise Issue53ForecastError("calibration sample does not bind model topology")
            try:
                for command in item.schedule.commands:
                    _command_vector(self.scenario, command.to_mapping())
            except (Issue52ForecastError, KeyError, TypeError, ValueError) as error:
                raise Issue53ForecastError(
                    "calibration sample contains incompatible commands"
                ) from error
            features = _masked_feature_matrix(
                item.history, item.schedule, self.scenario, self.manifest
            )
            predicted = features @ self.coefficients.astype(np.float64)
            normalized = np.abs(predicted - item.targets.astype(np.float64)) / scales[None, :]
            k = int(np.sum(~item.history.available_mask[-1]))
            residuals.setdefault(k, []).extend(normalized.ravel().tolist())
        calibrated = {
            k: max(MIN_INTERVAL_SCALE, float(np.quantile(values, quantile)))
            for k, values in residuals.items()
            if values
        }
        previous_scale: float | None = None
        for k in sorted(calibrated):
            if previous_scale is not None:
                calibrated[k] = max(calibrated[k], previous_scale)
            previous_scale = calibrated[k]
        abstention_limit = self.abstention_width_limit
        abstention_min_k = self.abstention_min_k
        abstention_risk_limit = self.abstention_risk_limit
        risk_coefficients = self.risk_coefficients
        risk_center = self.risk_center
        risk_scale = self.risk_scale
        risk_intercept = self.risk_intercept
        if oracle_errors is not None:
            if not isinstance(oracle_errors, Sequence) or isinstance(
                oracle_errors, (str, bytes)
            ):
                raise Issue53ForecastError("calibration oracle errors must be a sequence")
            errors = [float(value) for value in oracle_errors]
            if len(errors) != len(items) or not all(math.isfinite(value) for value in errors):
                raise Issue53ForecastError("calibration oracle errors are invalid")
            if oracle_high_error_threshold is None:
                oracle_threshold = float(np.quantile(np.asarray(errors), 0.90))
            else:
                oracle_threshold = float(oracle_high_error_threshold)
                if not math.isfinite(oracle_threshold) or oracle_threshold < 0.0:
                    raise Issue53ForecastError(
                        "calibration oracle threshold is invalid"
                    )
            widths_by_family: dict[str, list[float]] = {}
            errors_by_family: dict[str, list[float]] = {}
            risks_by_family: dict[str, list[float]] = {}
            for item, error in zip(items, errors):
                if abstention_k is not None and int(
                    np.sum(~item.history.available_mask[-1])
                ) != abstention_k:
                    continue
                k = int(np.sum(~item.history.available_mask[-1]))
                width = (
                    2.0
                    * max(
                        float(
                            calibrated.get(
                                k,
                                calibrated.get(
                                    max(calibrated, default=0), MIN_INTERVAL_SCALE
                                ),
                            )
                        ),
                        MIN_INTERVAL_SCALE,
                    )
                    * math.sqrt(HORIZON_STEPS)
                )
                widths_by_family.setdefault(item.family_id, []).append(width)
                errors_by_family.setdefault(item.family_id, []).append(error)
                risks_by_family.setdefault(item.family_id, []).append(
                    self.risk_score(item.history, item.schedule)
                )
            widths = [float(max(widths_by_family[family_id])) for family_id in sorted(widths_by_family)]
            selected_errors = [
                float(np.mean(errors_by_family[family_id]))
                for family_id in sorted(errors_by_family)
            ]
            selected_risks = [
                float(max(risks_by_family[family_id]))
                for family_id in sorted(risks_by_family)
            ]
            if not widths or len(widths) != len(selected_errors):
                raise Issue53ForecastError(
                    "calibration abstention bin has no aligned validation samples"
                )
            if abstention_k is not None:
                if type(abstention_k) is not int or not 0 <= abstention_k <= 6:
                    raise Issue53ForecastError("calibration abstention bin is invalid")
                abstention_min_k = abstention_k
            if abstention_k is not None:
                abstention_limit = max(widths) + 1.0
            risk_candidates = sorted(
                {0.0, *selected_risks, max(selected_risks, default=0.0) + 1.0}
            )
            candidate_metrics: list[tuple[float, float, float, float, float]] = []
            for risk_limit in risk_candidates:
                predicted = [risk > risk_limit for risk in selected_risks]
                high = [error > oracle_threshold for error in selected_errors]
                tp = sum(is_high and is_abstain for is_high, is_abstain in zip(high, predicted))
                fp = sum(not is_high and is_abstain for is_high, is_abstain in zip(high, predicted))
                fn = sum(is_high and not is_abstain for is_high, is_abstain in zip(high, predicted))
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
                candidate_metrics.append(
                    (
                        risk_limit,
                        precision,
                        recall,
                        f1,
                        sum(predicted) / len(predicted),
                    )
                )
            passing = [
                item
                for item in candidate_metrics
                if item[1] >= 0.60 and item[2] >= 0.80
            ]
            selected = min(
                passing or candidate_metrics,
                key=lambda item: (
                    -item[3],
                    -item[2],
                    -item[1],
                    item[4],
                    item[0],
                ),
            )
            abstention_risk_limit = selected[0]
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "intervals": calibrated,
                    "abstention_width_limit": abstention_limit,
                    "abstention_min_k": abstention_min_k,
                    "abstention_risk_limit": abstention_risk_limit,
                    "risk_coefficients": None
                    if risk_coefficients is None
                    else risk_coefficients.tolist(),
                    "risk_center": None
                    if risk_center is None
                    else risk_center.tolist(),
                    "risk_scale": None if risk_scale is None else risk_scale.tolist(),
                    "risk_intercept": float(risk_intercept),
                }
            )
        ).hexdigest()[:16]
        return replace(
            self,
            per_k_interval_scale=calibrated,
            model_id=f"{self.model_id}-cal-{digest}",
            abstention_width_limit=abstention_limit,
            abstention_min_k=abstention_min_k,
            abstention_risk_limit=abstention_risk_limit,
            risk_coefficients=risk_coefficients,
            risk_center=risk_center,
            risk_scale=risk_scale,
            risk_intercept=risk_intercept,
        )

    def fit_risk_head(
        self,
        samples: Sequence[TrainingSample],
        oracle_errors: Sequence[float],
        *,
        alpha: float = 1.0,
    ) -> "DropoutAwareLinearForecaster":
        """Fit an observation-only error predictor on TRAIN family decisions."""

        items = tuple(samples)
        errors = np.asarray(oracle_errors, dtype=np.float64)
        if not items or len(items) != len(errors) or not np.isfinite(errors).all():
            raise Issue53ForecastError("risk head training inputs are invalid")
        if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise Issue53ForecastError("risk head regularization must be positive and finite")
        if any(item.split != "TRAIN" for item in items):
            raise Issue53ForecastError("risk head accepts TRAIN samples only")
        grouped: dict[str, list[tuple[np.ndarray, float]]] = {}
        for item, error in zip(items, errors):
            grouped.setdefault(item.family_id, []).append(
                (
                    _risk_feature_vector(
                        item.history,
                        item.schedule,
                        self.scenario,
                        self.manifest,
                        self.coefficients,
                    ),
                    float(error),
                )
            )
        features = np.stack(
            [
                np.mean([feature for feature, _ in grouped[family_id]], axis=0)
                for family_id in sorted(grouped)
            ]
        )
        if features.ndim != 2 or features.shape[1] == 0:
            raise Issue53ForecastError("risk head feature matrix is invalid")
        errors = np.asarray(
            [
                np.mean([error for _, error in grouped[family_id]])
                for family_id in sorted(grouped)
            ],
            dtype=np.float64,
        )
        center = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1.0
        normalized = (features - center) / scale
        target = float(errors.mean())
        gram = normalized.T @ normalized + float(alpha) * np.eye(normalized.shape[1])
        coefficients = np.linalg.solve(gram, normalized.T @ (errors - target))
        return replace(
            self,
            risk_coefficients=coefficients,
            risk_center=center,
            risk_scale=scale,
            risk_intercept=target,
        )

    def risk_score(self, history: ForecastHistory, schedule: CandidateSchedule) -> float:
        if (
            self.risk_coefficients is None
            or self.risk_center is None
            or self.risk_scale is None
        ):
            return _missingness_risk_score(history, self.coefficients, self.manifest.width)
        features = _risk_feature_vector(
            history,
            schedule,
            self.scenario,
            self.manifest,
            self.coefficients,
        )
        normalized = (features - self.risk_center) / self.risk_scale
        return max(0.0, float(self.risk_intercept + normalized @ self.risk_coefficients))

    def forecast(
        self,
        history: ForecastHistory,
        schedule: CandidateSchedule,
        *,
        apply_abstention: bool = True,
    ) -> ForecastTrajectory:
        if history.target_values.shape[1] != self.manifest.width:
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "manifest_width_mismatch")
        # Mode applicability same as Issue 52
        if schedule.applicable_modes and history.latest_record.mode is not None and history.latest_record.mode not in schedule.applicable_modes:
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "candidate_mode_inapplicable")
        try:
            feats = _masked_feature_matrix(history, schedule, self.scenario, self.manifest)
        except (Issue53ForecastError, Issue52ForecastError):
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "forecast_features_invalid")
        if self.coefficients is None:
            # heuristic fallback with imputation
            latest = impute_history_values(history.target_values, history.available_mask, self.manifest)[-1].astype(np.float64)
            slope = _masked_slope(impute_history_values(history.target_values, history.available_mask, self.manifest), history.available_mask, history.completed_times_s).astype(np.float64) / CADENCE_SECONDS
            # simple heuristic: latest + slope * horizon
            width = self.manifest.width
            branch_count = len(self.scenario.data["air_network"]["branches"])
            zone_count = len(self.scenario.data["zones"])
            previous_action = _command_vector(self.scenario, history.latest_record.command).astype(np.float64)
            mean = np.empty((HORIZON_STEPS, width), dtype=np.float64)
            for horizon, command in enumerate(schedule.commands):
                action = _command_vector(self.scenario, command.to_mapping()).astype(np.float64)
                delta = action - previous_action
                fan_delta = delta[0]
                damper_delta = float(np.mean(delta[1 : 1 + branch_count])) if branch_count else 0.0
                scrubber_delta = delta[1 + branch_count] if len(delta) > 1 + branch_count else 0.0
                condenser_delta = delta[2 + branch_count] if len(delta) > 2 + branch_count else 0.0
                cooling_start = 3 + branch_count
                cooling_delta = float(np.mean(delta[cooling_start : cooling_start + zone_count])) / 1000.0 if zone_count else 0.0
                oxygen_delta = float(np.sum(delta[cooling_start + zone_count :])) if len(delta) > cooling_start + zone_count else 0.0
                row = latest + slope * float(horizon + 1)
                for idx, desc in enumerate(self.manifest.descriptors):
                    if desc.descriptor_id.endswith("/co2_ppm"):
                        eff = -90.0 * scrubber_delta - 20.0 * fan_delta - 10.0 * damper_delta
                    elif desc.descriptor_id.endswith("/temperature_k"):
                        eff = -18.0 * cooling_delta - 2.0 * fan_delta
                    elif desc.descriptor_id.endswith("/relative_humidity"):
                        eff = -0.22 * condenser_delta - 0.04 * cooling_delta
                    elif desc.descriptor_id == "battery_state_of_charge":
                        eff = -0.03 * (abs(fan_delta) + abs(scrubber_delta) + abs(condenser_delta) + abs(cooling_delta))
                    elif desc.descriptor_id == "oxygen_store_fraction":
                        eff = -8.0 * oxygen_delta
                    else:
                        eff = -0.02 * max(0.0, scrubber_delta)
                    row[idx] += eff * float(horizon + 1) / HORIZON_STEPS
                mean[horizon] = row
        else:
            mean = feats @ self.coefficients.astype(np.float64)
        if not np.isfinite(mean).all():
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "forecast_non_finite")
        # per-k uncertainty — look up k = missing on latest row (normalized scale)
        k = int(np.sum(~history.available_mask[-1]))
        base_norm = float(
            self.per_k_interval_scale.get(
                k,
                self.per_k_interval_scale.get(max(self.per_k_interval_scale, default=0), MIN_INTERVAL_SCALE),
            )
        )
        # Keep only a numerical positive floor; calibration supplies the width.
        base_norm = max(base_norm, MIN_INTERVAL_SCALE)
        lower = np.empty_like(mean)
        upper = np.empty_like(mean)
        for idx, desc in enumerate(self.manifest.descriptors):
            width = float(desc.scale) * base_norm
            # Horizon growth is calibrated through the validation residual scale.
            spread = width * np.sqrt(np.arange(1, HORIZON_STEPS + 1, dtype=np.float64))
            lower[:, idx] = mean[:, idx] - spread
            upper[:, idx] = mean[:, idx] + spread
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "interval_invalid")
        # Doubt-driven abstention — not hard refusal on any missing
        # Compute normalized width; abstain only if interval too wide or k exceeds calibrated capacity
        scales = np.asarray([d.scale for d in self.manifest.descriptors], dtype=np.float64)
        norm_width = float(np.max((upper - lower) / scales[None, :]))
        risk_score = (
            self.risk_score(history, schedule)
            if apply_abstention and k >= self.abstention_min_k
            else 0.0
        )
        if (
            apply_abstention
            and k >= self.abstention_min_k
            and risk_score > self.abstention_risk_limit
        ):
            return ForecastTrajectory(
                "ABSTAIN",
                None,
                None,
                None,
                self.model_id,
                "missingness_risk_limit",
            )
        if (
            apply_abstention
            and k >= self.abstention_min_k
            and norm_width > self.abstention_width_limit
        ):
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "uncertainty_limit")
        # If k is large and we have no calibration for that k, be conservative
        max_calibrated_k = max(self.per_k_interval_scale.keys(), default=0)
        if apply_abstention and k > max_calibrated_k + 2 and k >= 5:
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "uncalibrated_missing_level")
        return ForecastTrajectory("PREDICTION", _readonly(mean), _readonly(lower), _readonly(upper), self.model_id)


# ---------------------------------------------------------------------------
# Dataset & measurement helpers — honest per-k reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DropoutDatasetManifest:
    dropout_config_sha256: str
    parent_artifact_sha256: str | None
    family_ids: tuple[str, ...]
    family_split: Mapping[str, str]
    dataset_sha256: str
    samples_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DropoutDatasetManifest":
        if not isinstance(value, Mapping):
            raise Issue53ContractError("dataset manifest must be a mapping")
        if value.get("schema_version") != f"{ISSUE53_SCHEMA_VERSION}.dropout_dataset_manifest":
            raise Issue53ContractError("dataset manifest schema is invalid")
        family_ids = value.get("family_ids")
        family_split = value.get("family_split")
        if not isinstance(family_ids, list) or not isinstance(family_split, Mapping):
            raise Issue53ContractError("dataset manifest family fields are invalid")
        parent_artifact_sha256 = value.get("parent_artifact_sha256")
        samples_sha256 = value.get("samples_sha256")
        if parent_artifact_sha256 is not None and type(parent_artifact_sha256) is not str:
            raise Issue53ContractError("dataset parent artifact identity is invalid")
        if samples_sha256 is not None and type(samples_sha256) is not str:
            raise Issue53ContractError("dataset samples identity is invalid")
        for key in ("dropout_config_sha256", "dataset_sha256"):
            if type(value.get(key)) is not str:
                raise Issue53ContractError(f"dataset {key} identity is invalid")
        return cls(
            dropout_config_sha256=value["dropout_config_sha256"],
            parent_artifact_sha256=parent_artifact_sha256,
            family_ids=tuple(family_ids),
            family_split=dict(family_split),
            dataset_sha256=value["dataset_sha256"],
            samples_sha256=samples_sha256,
        )

    def __post_init__(self) -> None:
        _require_sha256(self.dropout_config_sha256, label="dataset dropout config")
        if self.parent_artifact_sha256 is not None:
            _require_sha256(self.parent_artifact_sha256, label="dataset parent artifact")
        if type(self.family_ids) is not tuple:
            raise Issue53ContractError("dataset family IDs must be sorted")
        if any(type(family_id) is not str or not family_id for family_id in self.family_ids):
            raise Issue53ContractError("dataset family IDs are invalid")
        if tuple(sorted(self.family_ids)) != self.family_ids:
            raise Issue53ContractError("dataset family IDs must be sorted")
        if len(set(self.family_ids)) != len(self.family_ids):
            raise Issue53ContractError("dataset family IDs must be unique")
        if not isinstance(self.family_split, Mapping):
            raise Issue53ContractError("dataset split must be a mapping")
        if set(self.family_split) != set(self.family_ids):
            raise Issue53ContractError("dataset split must cover every family")
        if any(
            type(family_id) is not str or type(split) is not str
            for family_id, split in self.family_split.items()
        ):
            raise Issue53ContractError("dataset split identities are invalid")
        if any(
            split not in {"TRAIN", "VALIDATION", "FINAL"}
            for split in self.family_split.values()
        ):
            raise Issue53ContractError("dataset split contains an invalid partition")
        object.__setattr__(
            self,
            "family_split",
            MappingProxyType(dict(sorted(self.family_split.items()))),
        )
        _require_sha256(self.dataset_sha256, label="dataset identity")
        if self.samples_sha256 is not None:
            _require_sha256(self.samples_sha256, label="dataset samples identity")
        payload = {
            "schema_version": f"{ISSUE53_SCHEMA_VERSION}.dropout_dataset_manifest",
            "dropout_config_sha256": self.dropout_config_sha256,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "family_ids": list(self.family_ids),
            "family_split": dict(sorted(self.family_split.items())),
            "samples_sha256": self.samples_sha256,
        }
        if self.dataset_sha256 != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
            raise Issue53ContractError("dataset manifest digest is inconsistent")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE53_SCHEMA_VERSION}.dropout_dataset_manifest",
            "dropout_config_sha256": self.dropout_config_sha256,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "family_ids": list(self.family_ids),
            "family_split": dict(self.family_split),
            "dataset_sha256": self.dataset_sha256,
            "samples_sha256": self.samples_sha256,
        }


def build_dropout_dataset_manifest(
    dropout_config: DropoutConfig,
    family_ids: Sequence[str],
    family_split: Mapping[str, str],
    *,
    parent_artifact_sha256: str | None = None,
    samples_sha256: str | None = None,
) -> DropoutDatasetManifest:
    if type(dropout_config) is not DropoutConfig:
        raise Issue53ContractError("dataset manifest requires DropoutConfig")
    if not isinstance(family_ids, Sequence) or isinstance(family_ids, (str, bytes)):
        raise Issue53ContractError("dataset family IDs must be a sequence")
    ids = tuple(family_ids)
    if any(type(family_id) is not str or not family_id for family_id in ids):
        raise Issue53ContractError("dataset family IDs are invalid")
    if len(set(ids)) != len(ids):
        raise Issue53ContractError("dataset family IDs must be unique")
    if not isinstance(family_split, Mapping):
        raise Issue53ContractError("dataset split must be a mapping")
    split = dict(family_split)
    if set(split) != set(ids):
        raise Issue53ContractError("dataset split must cover every family")
    if any(
        type(family_id) is not str or type(partition) is not str
        for family_id, partition in split.items()
    ) or any(partition not in {"TRAIN", "VALIDATION", "FINAL"} for partition in split.values()):
        raise Issue53ContractError("dataset split contains an invalid partition")
    payload = {
        "schema_version": f"{ISSUE53_SCHEMA_VERSION}.dropout_dataset_manifest",
        "dropout_config_sha256": dropout_config.config_sha256,
        "parent_artifact_sha256": parent_artifact_sha256,
        "family_ids": sorted(ids),
        "family_split": dict(sorted(split.items())),
        "samples_sha256": samples_sha256,
    }
    sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return DropoutDatasetManifest(
        dropout_config_sha256=dropout_config.config_sha256,
        parent_artifact_sha256=parent_artifact_sha256,
        family_ids=tuple(sorted(ids)),
        family_split=MappingProxyType(dict(sorted(split.items()))),
        dataset_sha256=sha,
        samples_sha256=samples_sha256,
    )


def _nmae_for_k(
    predictions: np.ndarray,
    truths: np.ndarray,
    manifest: TargetManifest,
) -> float:
    scales = np.asarray([d.scale for d in manifest.descriptors], dtype=np.float64)
    err = np.abs(predictions.astype(np.float64) - truths.astype(np.float64)) / scales[None, :]
    return float(np.mean(err))


def evaluate_per_k(
    forecaster: DropoutAwareLinearForecaster,
    samples_by_k: Mapping[int, Sequence[TrainingSample]],
    manifest: TargetManifest,
) -> dict[int, dict[str, float]]:
    """Honest per-k NMAE and ratio vs k=0 for the same forecaster."""
    result: dict[int, dict[str, float]] = {}
    # compute k=0 baseline
    k0_samples = samples_by_k.get(0, ())
    if k0_samples:
        # use forecaster on dropout-masked k0 (which is mostly identity) vs truth
        nmaes_k0: list[float] = []
        for s in k0_samples:
            traj = forecaster.forecast(s.history, s.schedule)
            if traj.status != "PREDICTION" or traj.mean is None:
                continue
            nmaes_k0.append(_nmae_for_k(traj.mean, s.targets, manifest))
        baseline = float(np.mean(nmaes_k0)) if nmaes_k0 else math.nan
    else:
        baseline = math.nan
    for k, samples in sorted(samples_by_k.items()):
        nmaes: list[float] = []
        for s in samples:
            traj = forecaster.forecast(s.history, s.schedule)
            if traj.status != "PREDICTION" or traj.mean is None:
                continue
            nmaes.append(_nmae_for_k(traj.mean, s.targets, manifest))
        mean = float(np.mean(nmaes)) if nmaes else math.nan
        ratio = float(mean / baseline) if math.isfinite(mean) and math.isfinite(baseline) and baseline > 0 else math.nan
        result[k] = {"nmae": mean, "ratio_vs_k0": ratio, "count": float(len(nmaes))}
    return result


def interval_coverage_at_k(
    forecaster: DropoutAwareLinearForecaster,
    samples: Sequence[TrainingSample],
) -> float:
    hits = 0
    total = 0
    for s in samples:
        traj = forecaster.forecast(s.history, s.schedule)
        if traj.status != "PREDICTION" or traj.lower is None or traj.upper is None:
            continue
        lower = traj.lower.astype(np.float64)
        upper = traj.upper.astype(np.float64)
        truth = s.targets.astype(np.float64)
        hits += int(np.sum((truth >= lower) & (truth <= upper)))
        total += int(truth.size)
    return float(hits / total) if total else math.nan


def abstention_pr(
    forecaster: DropoutAwareLinearForecaster,
    samples: Sequence[TrainingSample],
    *,
    oracle_errors: Sequence[float],
    high_error_quantile: float = 0.9,
) -> dict[str, float]:
    """Precision/recall of ABSTAIN vs externally supplied oracle errors."""
    if (
        isinstance(high_error_quantile, bool)
        or not isinstance(high_error_quantile, (int, float))
        or not math.isfinite(float(high_error_quantile))
        or not 0.0 < float(high_error_quantile) < 1.0
    ):
        raise Issue53ForecastError("high-error quantile must be in (0,1)")
    items = tuple(samples)
    if not isinstance(oracle_errors, Sequence) or isinstance(oracle_errors, (str, bytes)):
        raise Issue53ForecastError("oracle_errors must be a numeric sequence")
    errors = [float(value) for value in oracle_errors]
    if len(errors) != len(items):
        raise Issue53ForecastError("oracle_errors must align with samples")
    if not all(math.isfinite(value) for value in errors):
        raise Issue53ForecastError("oracle_errors must be finite")
    statuses: list[str] = []
    for s in items:
        traj = forecaster.forecast(s.history, s.schedule)
        statuses.append(traj.status)
    thresh = float(np.quantile(np.asarray(errors), high_error_quantile))
    # oracle high-error = NMAE > thresh
    tp = fp = fn = tn = 0
    for err, status in zip(errors, statuses):
        is_high = math.isfinite(err) and err > thresh
        is_abstain = status == "ABSTAIN"
        if is_high and is_abstain:
            tp += 1
        elif not is_high and is_abstain:
            fp += 1
        elif is_high and not is_abstain:
            fn += 1
        else:
            tn += 1
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "threshold": thresh, "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


__all__ = [
    "ISSUE53_SCHEMA_VERSION",
    "DROPOUT_SCHEMA_VERSION",
    "DropoutConfig",
    "Issue53ContractError",
    "Issue53ForecastError",
    "apply_dropout_to_history",
    "dropout_mask_for_history",
    "impute_history_values",
    "DropoutAwareLinearForecaster",
    "DropoutDatasetManifest",
    "build_dropout_dataset_manifest",
    "evaluate_per_k",
    "interval_coverage_at_k",
    "abstention_pr",
]
