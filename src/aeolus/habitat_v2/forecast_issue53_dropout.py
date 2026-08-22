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
from dataclasses import dataclass, field
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
        if not math.isfinite(self.p_uniform) or not 0.0 <= self.p_uniform < 1.0:
            raise Issue53ContractError("p_uniform must be in [0,1)")
        if self.mode not in {"independent", "per_zone_head_burst", "mixed"}:
            raise Issue53ContractError("dropout mode is invalid")
        if not isinstance(self.burst_min, int) or not isinstance(self.burst_max, int):
            raise Issue53ContractError("burst bounds must be integers")
        if self.burst_min < 1 or self.burst_max < self.burst_min:
            raise Issue53ContractError("burst bounds are invalid")
        if not math.isfinite(self.p_burst_onset) or not 0.0 <= self.p_burst_onset <= 1.0:
            raise Issue53ContractError("p_burst_onset must be in [0,1]")
        if type(self.resource_gauge_dropout) is not bool:
            raise Issue53ContractError("resource_gauge_dropout must be bool")
        if self.max_missing_per_row is not None:
            if not isinstance(self.max_missing_per_row, int) or self.max_missing_per_row < 0:
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
        return cls(
            p_uniform=float(value.get("p_uniform", 0.05)),
            mode=str(value.get("mode", "independent")),
            burst_min=int(value.get("burst_min", 2)),
            burst_max=int(value.get("burst_max", 8)),
            p_burst_onset=float(value.get("p_burst_onset", 0.02)),
            resource_gauge_dropout=bool(value.get("resource_gauge_dropout", False)),
            max_missing_per_row=value.get("max_missing_per_row", 6),
            seed=int(value.get("seed", 530053)),
            config_sha256=str(value.get("config_sha256", "")),
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
    payload = f"{seed}\x00{family_id}\x00{decision_step}\x00{step_offset}\x00{descriptor_id}".encode("utf-8")
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


def dropout_mask_for_history(
    history: ForecastHistory,
    manifest: TargetManifest,
    config: DropoutConfig,
    *,
    family_id: str,
    decision_step: int,
) -> np.ndarray:
    """Deterministic boolean mask: True = keep (AVAILABLE), False = drop."""
    if type(history) is not ForecastHistory or type(manifest) is not TargetManifest or type(config) is not DropoutConfig:
        raise Issue53ContractError("dropout mask requires exact contract types")
    if history.target_values.shape[1] != manifest.width:
        raise Issue53ContractError("history width does not bind manifest")
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
            if c in gauge_indices:
                # never drop gauges unless explicitly enabled
                continue
            if not base_available[t, c]:
                mask[t, c] = False
                continue
            u64 = _sampler_u64(config.seed, family_id, decision_step, t, manifest.descriptors[c].descriptor_id)
            if _should_drop(u64, config.p_uniform):
                mask[t, c] = False
            # burst mode omitted in this slice — deterministic via same sampler seeded per burst
            # burst is observable via correlated drops on same zone/head; left for future amendment

    # Optional cap — if more than max_missing_per_row dropped on a row, keep lexicographically smallest descriptors
    if config.max_missing_per_row is not None:
        cap = int(config.max_missing_per_row)
        for t in range(HISTORY_STEPS):
            dropped = int(np.sum(~mask[t] & base_available[t]))
            # count only dropout-induced drops, ignore pre-existing unavailable
            if dropped > cap:
                # revert excess drops: keep the lexicographically earliest descriptor_ids
                dropout_cols = [c for c in range(manifest.width) if not mask[t, c] and base_available[t, c]]
                dropout_cols.sort(key=lambda c: manifest.descriptors[c].descriptor_id)
                # keep exactly `cap` dropped, revert the tail tail
                # We dropped `dropped` items; we need to keep `cap` of them → revert `dropped-cap` earliest
                # Simpler: re-enable all then drop only first `cap` lexicographically — ensures cap respected
                mask[t, :] = base_available[t, :]
                # re-drop cap smallest
                for c in sorted(dropout_cols)[:cap]:
                    mask[t, c] = False
                # remaining columns stay available
                # For rows where we re-enabled, need to recompute from base_available
                # but we just set mask[t]=base_available[t] then dropped cap, so correct
                pass

    mask.setflags(write=False)
    return mask


def apply_dropout_to_history(
    history: ForecastHistory,
    manifest: TargetManifest,
    config: DropoutConfig,
    *,
    family_id: str,
    decision_step: int,
) -> ForecastHistory:
    """Return a new ForecastHistory with dropout mask applied (observation-only)."""
    mask = dropout_mask_for_history(history, manifest, config, family_id=family_id, decision_step=decision_step)
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

    def __post_init__(self) -> None:
        if type(self.scenario) is not Scenario or type(self.manifest) is not TargetManifest or type(self.dropout_config) is not DropoutConfig:
            raise Issue53ForecastError("dropout forecaster requires exact scenario, manifest, and dropout config")
        if self.manifest.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue53ForecastError("dropout forecaster manifest does not bind scenario")
        if type(self.model_id) is not str or not self.model_id:
            raise Issue53ForecastError("dropout forecaster model identity is invalid")
        if self.coefficients is not None:
            vals = np.asarray(self.coefficients, dtype=np.float32)
            if vals.ndim != 2 or not np.isfinite(vals).all():
                raise Issue53ForecastError("dropout forecaster coefficients are invalid")
            object.__setattr__(self, "coefficients", _readonly(vals))
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
    ) -> "DropoutAwareLinearForecaster":
        items = tuple(samples)
        if len(items) < 3 or len({item.family_id for item in items}) < 2:
            raise Issue53ForecastError("fitting requires at least three samples and two families")
        if any(item.split != "TRAIN" for item in items):
            raise Issue53ForecastError("fitting accepts TRAIN samples only")
        if type(scenario) is not Scenario or type(manifest) is not TargetManifest or type(dropout_config) is not DropoutConfig:
            raise Issue53ForecastError("fit inputs are not exact contract types")
        if manifest.scenario_sha256 != scenario.scenario_sha256:
            raise Issue53ForecastError("fit manifest does not bind scenario")
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
            masked_history = apply_dropout_to_history(item.history, manifest, dropout_config, family_id=family_id, decision_step=decision_step)
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
        # per-k interval scale: conformal residuals binned by k (missing count on latest row)
        # For initial fit, calibrate on augmented TRAIN residuals grouped by k
        per_k_scale: dict[int, float] = {}
        # simple residual scale per k: mean abs residual * 1.5 (approx 90% under Gaussian) — refined on VALIDATION
        for k in range(0, 7):
            subset = [s for s in augmented if int(np.sum(~s.history.available_mask[-1])) == k]
            if not subset:
                continue
            feats = np.concatenate([_masked_feature_matrix(s.history, s.schedule, scenario, manifest) for s in subset], axis=0).astype(np.float64)
            preds = feats @ coeffs.astype(np.float64)
            truth = np.concatenate([s.targets for s in subset], axis=0).astype(np.float64)
            scales = np.asarray([d.scale for d in manifest.descriptors], dtype=np.float64)
            norm_resid = np.abs(preds - truth) / scales[None, :]
            per_k_scale[k] = float(np.quantile(norm_resid, 0.90)) if norm_resid.size else 0.02
        # ensure monotone non-decreasing with k
        sorted_ks = sorted(per_k_scale)
        for i in range(1, len(sorted_ks)):
            prev, cur = sorted_ks[i - 1], sorted_ks[i]
            if per_k_scale[cur] < per_k_scale[prev]:
                per_k_scale[cur] = per_k_scale[prev]
        return cls(scenario, manifest, dropout_config, coeffs, model_id, per_k_scale)

    def forecast(self, history: ForecastHistory, schedule: CandidateSchedule) -> ForecastTrajectory:
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
        base_norm = float(self.per_k_interval_scale.get(k, self.per_k_interval_scale.get(max(self.per_k_interval_scale, default=0), 0.02)))
        # ensure base_norm is at least 0.02 (fallback)
        base_norm = max(base_norm, 0.02)
        lower = np.empty_like(mean)
        upper = np.empty_like(mean)
        for idx, desc in enumerate(self.manifest.descriptors):
            width = float(desc.scale) * base_norm
            # monotone widening with horizon + k
            spread = width * np.sqrt(np.arange(1, HORIZON_STEPS + 1, dtype=np.float64)) * (1.0 + 0.15 * k)
            lower[:, idx] = mean[:, idx] - spread
            upper[:, idx] = mean[:, idx] + spread
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "interval_invalid")
        # Doubt-driven abstention — not hard refusal on any missing
        # Compute normalized width; abstain only if interval too wide or k exceeds calibrated capacity
        scales = np.asarray([d.scale for d in self.manifest.descriptors], dtype=np.float64)
        norm_width = float(np.max((upper - lower) / scales[None, :]))
        if norm_width > 8.0:  # wider than Issue 52's 4.0 — dropout lane tolerates wider but still bounded
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "uncertainty_limit")
        # If k is large and we have no calibration for that k, be conservative
        max_calibrated_k = max(self.per_k_interval_scale.keys(), default=0)
        if k > max_calibrated_k + 2 and k >= 5:
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

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE53_SCHEMA_VERSION}.dropout_dataset_manifest",
            "dropout_config_sha256": self.dropout_config_sha256,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "family_ids": list(self.family_ids),
            "family_split": dict(self.family_split),
            "dataset_sha256": self.dataset_sha256,
        }


def build_dropout_dataset_manifest(
    dropout_config: DropoutConfig,
    family_ids: Sequence[str],
    family_split: Mapping[str, str],
    *,
    parent_artifact_sha256: str | None = None,
) -> DropoutDatasetManifest:
    if type(dropout_config) is not DropoutConfig:
        raise Issue53ContractError("dataset manifest requires DropoutConfig")
    payload = {
        "schema_version": f"{ISSUE53_SCHEMA_VERSION}.dropout_dataset_manifest",
        "dropout_config_sha256": dropout_config.config_sha256,
        "parent_artifact_sha256": parent_artifact_sha256,
        "family_ids": sorted(family_ids),
        "family_split": dict(sorted(family_split.items())),
    }
    sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return DropoutDatasetManifest(
        dropout_config_sha256=dropout_config.config_sha256,
        parent_artifact_sha256=parent_artifact_sha256,
        family_ids=tuple(sorted(family_ids)),
        family_split=MappingProxyType(dict(sorted(family_split.items()))),
        dataset_sha256=sha,
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
    manifest: TargetManifest,
    *,
    high_error_quantile: float = 0.9,
) -> dict[str, float]:
    """Precision/recall of ABSTAIN vs oracle high-error decisions."""
    # Determine high-error threshold from k=0 samples or own samples
    errors: list[float] = []
    statuses: list[str] = []
    for s in samples:
        traj = forecaster.forecast(s.history, s.schedule)
        statuses.append(traj.status)
        if traj.status == "PREDICTION" and traj.mean is not None:
            errors.append(_nmae_for_k(traj.mean, s.targets, manifest))
        else:
            errors.append(math.nan)
    finite = [e for e in errors if math.isfinite(e)]
    if not finite:
        return {"precision": math.nan, "recall": math.nan, "threshold": math.nan}
    thresh = float(np.quantile(np.asarray(finite), high_error_quantile))
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
