"""Issue #73 guard-attribution ablations on the frozen BDM-v1 roster.

Development-only, fresh-family study that separates the deterministic O2
excess guard from the learned/statistical screen of the V4 ``c9`` hybrid.
The guard semantics mirror the Issue #56 v10 candidate context rules
(deterministic observable O2-excess override to dormant, abstention under
critical health); the statistical screens are refit here on the current
TRAIN corpus so every arm runs on identical fresh exogenous traces.
Historical V4/V10 artifacts and results are untouched.

Arms (preregistered in
``contracts/habitat_v2_bdm_v1_ablation_preregistration_v1.json``):

- ``hmc_rules_only`` and ``hold_current_command``: zero proposals;
- ``c8_guard_only``: deterministic context guard only, no learned component;
- ``c9_learned_screen_without_guard``: statistical screen only;
- ``c9_full``: guard override plus statistical screen (v10 c9 semantics).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .bdm_v1_families import DECISION_STEPS, EPISODE_STEPS
from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast_issue55_race import (
    compute_race_metrics,
    project_true_targets,
    scenario_zone_order,
    target_bounds,
)
from .hmc import HabitatManagementComputer
from .physics import (
    advance_one_step_with_command,
    command_from_achieved_state,
    initial_state,
    validate_external_command,
)
from .scenario import Scenario

ISSUE73_ABLATION_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_ablation_v1"
ABLATION_ARMS = (
    "hmc_rules_only",
    "hold_current_command",
    "c8_guard_only",
    "c9_learned_screen_without_guard",
    "c9_full",
)
NON_PROPOSING_ARMS = ("hmc_rules_only", "hold_current_command")
DORMANT_ACTION_ID = "normal-dormant-v1"
O2_EXCESS_THRESHOLD = 0.285
SCREEN_FEATURE_DIM = 33
MLP_HIDDEN_DIM = 16


class Issue73AblationError(ValueError):
    """Raised when ablation inputs, arms, or episodes are inadmissible."""


@dataclass(frozen=True, slots=True)
class AblationContext:
    """Deterministic observable decision context (no hidden truth)."""

    nominal_o2_excess: bool
    critical_health: bool
    min_o2_mole_fraction: float
    mode_one_hot: tuple[float, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "nominal_o2_excess": self.nominal_o2_excess,
            "critical_health": self.critical_health,
            "min_o2_mole_fraction": self.min_o2_mole_fraction,
            "mode_one_hot": list(self.mode_one_hot),
        }


def observable_context(
    snapshot: Mapping[str, Any],
    zone_ids: Sequence[str],
    mode_order: Sequence[str],
) -> AblationContext:
    """Reduce one verified snapshot to the declared guard context."""

    primaries = {
        sample["descriptor_id"]: sample
        for sample in snapshot["primary_telemetry"]["samples"]
    }
    o2_values: list[float] = []
    for zone in zone_ids:
        sample = primaries[f"{zone}/o2_mole_fraction"]
        if sample["availability"] == "AVAILABLE":
            o2_values.append(float(sample["value"]))
    if not o2_values:
        return AblationContext(False, True, 0.0, tuple(0.0 for _ in mode_order))
    min_o2 = min(o2_values)
    _scales, _nominals, lowers, _uppers = target_bounds()
    o2_lower = float(min(lowers[index] for index in range(len(lowers)) if index % 6 == 3))
    mode = str(snapshot["completed_operating_mode"])
    one_hot = tuple(1.0 if name == mode else 0.0 for name in mode_order)
    return AblationContext(
        nominal_o2_excess=min_o2 >= O2_EXCESS_THRESHOLD,
        critical_health=min_o2 < o2_lower,
        min_o2_mole_fraction=min_o2,
        mode_one_hot=one_hot,
    )


def screen_feature_vector(
    o2_last: Sequence[float],
    co2_last: Sequence[float],
    temperature_last: Sequence[float],
    mode_one_hot: Sequence[float],
    action_index: int,
) -> np.ndarray:
    """Assemble the declared 33-dim screen feature vector."""

    if not (len(o2_last) == len(co2_last) == len(temperature_last) == 8):
        raise Issue73AblationError("screen features require eight zones")
    if len(mode_one_hot) != 4 or any(value not in (0.0, 1.0) for value in mode_one_hot) or sum(mode_one_hot) != 1:
        raise Issue73AblationError("screen features require a one-hot operating mode")
    if not 0 <= action_index < 4:
        raise Issue73AblationError("screen features require a catalogue action index")
    action_one_hot = [1.0 if index == action_index else 0.0 for index in range(4)]
    values = [
        *o2_last,
        *co2_last,
        *temperature_last,
        float(min(o2_last)),
        *mode_one_hot,
        *action_one_hot,
    ]
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (SCREEN_FEATURE_DIM,) or not np.isfinite(vector).all():
        raise Issue73AblationError("screen feature vector is malformed")
    return vector


def screen_features_from_sample(sample: Mapping[str, Any]) -> np.ndarray:
    features = sample["features"]
    return screen_feature_vector(
        [float(value) for value in features["zone_o2_mole_fraction"][-1]],
        [float(value) for value in features["zone_co2_ppm"][-1]],
        [float(value) for value in features["zone_temperature_k"][-1]],
        [float(value) for value in features["operating_mode_one_hot"][-1]],
        int(features["candidate_action_index"]),
    )


def _last_zone_values(snapshot: Mapping[str, Any], zone_ids: Sequence[str], channel: str) -> list[float]:
    primaries = {
        sample["descriptor_id"]: sample
        for sample in snapshot["primary_telemetry"]["samples"]
    }
    values: list[float] = []
    for zone in zone_ids:
        sample = primaries[f"{zone}/{channel}"]
        if sample["availability"] == "AVAILABLE":
            values.append(float(sample["value"]))
        else:
            values.append(0.0)
    return values


def screen_features_from_snapshot(
    snapshot: Mapping[str, Any],
    zone_ids: Sequence[str],
    mode_order: Sequence[str],
    action_index: int,
) -> np.ndarray:
    mode = str(snapshot["completed_operating_mode"])
    one_hot = [1.0 if name == mode else 0.0 for name in mode_order]
    return screen_feature_vector(
        _last_zone_values(snapshot, zone_ids, "o2_mole_fraction"),
        _last_zone_values(snapshot, zone_ids, "co2_ppm"),
        _last_zone_values(snapshot, zone_ids, "temperature_k"),
        one_hot,
        action_index,
    )


@dataclass(frozen=True, slots=True)
class LinearScreen:
    """Ridge screen on standardized features (closed form, deterministic)."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    ridge_lambda: float
    digest: str

    def predict(self, features: np.ndarray) -> float:
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return float(np.dot(self.weights, normalized) + self.bias)


@dataclass(frozen=True, slots=True)
class MlpScreen:
    """Small shared hidden-layer screen trained with deterministic Adam."""

    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    epochs: int
    digest: str

    def predict(self, features: np.ndarray) -> float:
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        hidden = np.tanh(normalized @ self.w1 + self.b1)
        return float(hidden @ self.w2 + self.b2)


def _standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def fit_linear_screen(
    features: np.ndarray, targets: np.ndarray, *, ridge_lambda: float, seed: str
) -> LinearScreen:
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise Issue73AblationError("ridge screen inputs are misaligned")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise Issue73AblationError("ridge screen inputs are non-finite")
    mean, scale = _standardizer(features)
    normalized = (features - mean) / scale
    augmented = np.concatenate([normalized, np.ones((normalized.shape[0], 1))], axis=1)
    gram = augmented.T @ augmented
    gram += ridge_lambda * np.eye(augmented.shape[1])
    gram[-1, -1] -= ridge_lambda
    solution = np.linalg.solve(gram, augmented.T @ targets)
    weights = solution[:-1]
    bias = float(solution[-1])
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": "ridge",
                "ridge_lambda": ridge_lambda,
                "seed": seed,
                "weights": [float(value) for value in weights],
                "bias": bias,
            }
        )
    ).hexdigest()
    return LinearScreen(mean, scale, weights, bias, ridge_lambda, digest)


def fit_mlp_screen(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: str,
    epochs: int,
    learning_rate: float,
) -> MlpScreen:
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise Issue73AblationError("mlp screen inputs are misaligned")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise Issue73AblationError("mlp screen inputs are non-finite")
    mean, scale = _standardizer(features)
    normalized = (features - mean) / scale
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    )
    w1 = rng.normal(0.0, 0.2, (SCREEN_FEATURE_DIM, MLP_HIDDEN_DIM))
    b1 = np.zeros(MLP_HIDDEN_DIM)
    w2 = rng.normal(0.0, 0.2, MLP_HIDDEN_DIM)
    b2 = 0.0
    m1 = np.zeros_like(w1)
    v1 = np.zeros_like(w1)
    m2 = np.zeros_like(w2)
    v2 = np.zeros_like(w2)
    mb = np.zeros_like(b1)
    vb = np.zeros_like(b1)
    mb2 = 0.0
    vb2 = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        hidden_pre = normalized @ w1 + b1
        hidden = np.tanh(hidden_pre)
        prediction = hidden @ w2 + b2
        error = prediction - targets
        grad_w2 = hidden.T @ error / len(targets)
        grad_b2 = float(error.mean())
        grad_hidden = np.outer(error, w2) * (1.0 - hidden**2)
        grad_w1 = normalized.T @ grad_hidden / len(targets)
        grad_b1 = grad_hidden.mean(axis=0)
        m1 = beta1 * m1 + (1 - beta1) * grad_w1
        v1 = beta2 * v1 + (1 - beta2) * grad_w1**2
        m2 = beta1 * m2 + (1 - beta1) * grad_w2
        v2 = beta2 * v2 + (1 - beta2) * grad_w2**2
        mb = beta1 * mb + (1 - beta1) * grad_b1
        vb = beta2 * vb + (1 - beta2) * grad_b1**2
        mb2 = beta1 * mb2 + (1 - beta1) * grad_b2
        vb2 = beta2 * vb2 + (1 - beta2) * grad_b2**2
        correction = 1.0 - beta1**epoch
        w1 -= learning_rate * (m1 / correction) / (np.sqrt(v1 / (1.0 - beta2**epoch)) + eps)
        b1 -= learning_rate * (mb / correction) / (np.sqrt(vb / (1.0 - beta2**epoch)) + eps)
        w2 -= learning_rate * (m2 / correction) / (np.sqrt(v2 / (1.0 - beta2**epoch)) + eps)
        b2 -= learning_rate * (mb2 / correction) / (np.sqrt(vb2 / (1.0 - beta2**epoch)) + eps)
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": "mlp",
                "seed": seed,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "w1": [[float(value) for value in row] for row in w1],
                "w2": [float(value) for value in w2],
            }
        )
    ).hexdigest()
    return MlpScreen(mean, scale, w1, b1, w2, b2, epochs, digest)


def guard_proposal(context: AblationContext, current_action_id: str) -> str | None:
    """Deterministic v10-style guard: dormant on nominal excess, else abstain."""

    if context.critical_health:
        return None
    if context.nominal_o2_excess and current_action_id != DORMANT_ACTION_ID:
        return DORMANT_ACTION_ID
    return None


def learned_proposal(
    screen: LinearScreen | MlpScreen,
    snapshot: Mapping[str, Any],
    zone_ids: Sequence[str],
    mode_order: Sequence[str],
    action_ids: Sequence[str],
    context: AblationContext,
    current_action_id: str,
    *,
    include_guard: bool,
) -> tuple[str | None, tuple[float, ...]]:
    """Statistical screen policy; returns chosen action id and predictions."""

    if len(action_ids) != 4:
        raise Issue73AblationError("learned policy requires the four-action catalogue")
    predictions = tuple(
        screen.predict(
            screen_features_from_snapshot(snapshot, zone_ids, mode_order, index)
        )
        for index in range(4)
    )
    if include_guard:
        guarded = guard_proposal(context, current_action_id)
        if guarded is not None:
            return guarded, predictions
    if context.critical_health:
        return None, predictions
    if current_action_id == DORMANT_ACTION_ID:
        return None, predictions
    improving = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if prediction < 0.0
    ]
    if not improving:
        return None, predictions
    index = min(improving, key=lambda item: (item[1], item[0]))[0]
    return action_ids[index], predictions


def run_ablation_episode(
    bundle: ForecastContracts,
    scenario: Scenario,
    arm: str,
    family_id: str,
    *,
    screen: LinearScreen | MlpScreen | None = None,
    decision_steps: Sequence[int] = DECISION_STEPS,
    scenario_sha256: str = "",
) -> dict[str, Any]:
    """Run one 96-step HMC episode under one ablation arm."""

    if arm not in ABLATION_ARMS:
        raise Issue73AblationError(f"unknown ablation arm {arm!r}")
    if arm in ("c9_learned_screen_without_guard", "c9_full") and screen is None:
        raise Issue73AblationError(f"arm {arm!r} requires a fitted screen")
    if arm not in ("c9_learned_screen_without_guard", "c9_full") and screen is not None:
        raise Issue73AblationError(f"arm {arm!r} must not carry a screen")
    zone_ids = scenario_zone_order(scenario)
    actions = tuple(bundle.actions)
    mode_order = ("occupied", "eva_transition", "contingency", "dormant")
    decisions = set(int(step) for step in decision_steps)
    nonce = hashlib.sha256(b"issue73-ablation|" + family_id.encode() + b"|" + arm.encode()).digest()
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    shadow = initial_state(scenario)
    initial_row = project_true_targets(scenario, zone_ids, shadow)
    states = {0: shadow}
    proposal_count = 0
    rejected_proposal_count = 0
    abstention_count = 0
    hmc_rejection_count = 0
    decision_actions: list[str | None] = []
    last_command: dict[str, Any] | None = None
    current_action_id = "normal-occupied-v1"
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue73AblationError(f"HMC terminated at step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snap_map = snapshot.to_mapping()
        proposal = None
        if step in decisions:
            if arm == "c8_guard_only":
                context = observable_context(snap_map, zone_ids, mode_order)
                chosen = guard_proposal(context, current_action_id)
            elif arm in ("c9_learned_screen_without_guard", "c9_full"):
                context = observable_context(snap_map, zone_ids, mode_order)
                chosen, _predictions = learned_proposal(
                    screen,
                    snap_map,
                    zone_ids,
                    mode_order,
                    [action.action_id for action in actions],
                    context,
                    current_action_id,
                    include_guard=arm == "c9_full",
                )
            else:
                chosen = None
            if chosen is not None:
                action = next(item for item in actions if item.action_id == chosen)
                proposal = {
                    "source_id": action.action_id,
                    "proposed_command": action.command.to_mapping(),
                }
        receipt = hmc.propose(proposal, handle)
        receipt_mapping = receipt.to_mapping()
        if step in decisions:
            if proposal is None:
                abstention_count += 1
                decision_actions.append(None)
            elif (
                receipt_mapping["attempt_class"],
                receipt_mapping["validation_outcome"],
            ) == ("CANONICAL_PROPOSAL", "VALID"):
                proposal_count += 1
                decision_actions.append(proposal["source_id"])
            else:
                rejected_proposal_count += 1
                abstention_count += 1
                decision_actions.append(None)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue73AblationError(f"HMC terminated while arbitrating step {step}")
        if proposal is not None and step in decisions and decision_actions[-1] == proposal["source_id"]:
            proposed_sha = validate_external_command(
                scenario, proposal["proposed_command"]
            ).sha256
            if arbitration.final_command_sha256 != proposed_sha:
                hmc_rejection_count += 1
        last_command = dict(arbitration.final_command)
        if snap_map["completed_operating_mode"] is not None:
            current_action_id = _mode_action_id(snap_map)
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise Issue73AblationError(f"HMC terminated while stepping step {step}")
        result = advance_one_step_with_command(scenario, shadow, last_command)
        if result.state.step != step + 1:
            raise Issue73AblationError("shadow state drifted from the HMC step")
        if (
            hashlib.sha256(canonical_json_bytes(result.receipt)).hexdigest()
            != step_receipt.plant_receipt_digest
        ):
            raise Issue73AblationError("shadow receipt diverges from the HMC receipt")
        shadow = result.state
        states[shadow.step] = shadow
    if len(decision_actions) != len(decisions):
        raise Issue73AblationError("decision bookkeeping drifted")
    metrics = compute_race_metrics(
        scenario, zone_ids, initial_row, [states[key] for key in range(1, EPISODE_STEPS + 1)]
    )
    record = {
        "schema_version": ISSUE73_ABLATION_SCHEMA_VERSION,
        "arm": arm,
        "family_id": family_id,
        "scenario_sha256": scenario_sha256,
        "decision_steps": sorted(decisions),
        "proposal_count": proposal_count,
        "rejected_proposal_count": rejected_proposal_count,
        "abstention_count": abstention_count,
        "hmc_rejection_count": hmc_rejection_count,
        "decision_actions": decision_actions,
        "safety_exposure": float(metrics["safety_exposure"]),
        "safety_violation_steps": int(metrics["safety_violation_steps"]),
        "comfort_deviation": float(metrics["comfort_deviation"]),
        "resource_composite": float(metrics["resource_composite"]),
    }
    record["episode_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in record.items() if key != "episode_sha256"})
    ).hexdigest()
    return record


def _mode_action_id(snapshot: Mapping[str, Any]) -> str:
    mode = str(snapshot["completed_operating_mode"])
    return {
        "occupied": "normal-occupied-v1",
        "eva_transition": "normal-eva_transition-v1",
        "contingency": "normal-contingency-v1",
        "dormant": "normal-dormant-v1",
    }[mode]


__all__ = [
    "ABLATION_ARMS",
    "AblationContext",
    "DORMANT_ACTION_ID",
    "ISSUE73_ABLATION_SCHEMA_VERSION",
    "Issue73AblationError",
    "LinearScreen",
    "MlpScreen",
    "NON_PROPOSING_ARMS",
    "O2_EXCESS_THRESHOLD",
    "fit_linear_screen",
    "fit_mlp_screen",
    "guard_proposal",
    "learned_proposal",
    "observable_context",
    "run_ablation_episode",
    "screen_features_from_sample",
    "screen_features_from_snapshot",
    "screen_feature_vector",
]
