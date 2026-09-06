"""Issue #74 linear baselines for the BDM-v1 research line.

Three reproducible linear baselines establish whether temporal neural
modelling adds decision value beyond simple causal dynamics and action
conditioning:

- ``action_agnostic_ridge``: ridge on observable context features only;
- ``action_conditioned_ridge``: ridge on context features plus the candidate
  action encoding;
- ``controlled_linear_state_space``: per-horizon linear transition maps from
  the checkpoint observed state and the candidate command vector.

All features derive exclusively from Issue #70 declared corpus fields (the
recipe is exported as ``RECIPE_INPUT_FIELDS`` and validated fail-closed).
All models are propose-only mathematics: nothing in this module imports the
HMC, steps the plant, or issues a final command.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

from .bdm_v1_corpus import FEATURE_FIELD_NAMES
from .forecast.contracts import canonical_json_bytes

ISSUE74_BASELINE_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_baselines_v1"
CONTEXT_DIM = 45
ACTION_DIM = 31
STATE_DIM = 43
COMMAND_DIM = 27
LABEL_HORIZON_KEYS = ("4", "16", "32")
DECISION_LABEL_NAMES = (
    "crossing_event",
    "safety_exposure",
    "maximum_crossing",
    "comfort_deviation",
    "resource_composite",
)
BASELINE_IDS = (
    "action_agnostic_ridge",
    "action_conditioned_ridge",
    "controlled_linear_state_space",
)
RECIPE_INPUT_FIELDS = (
    "zone_o2_mole_fraction",
    "zone_co2_ppm",
    "zone_temperature_k",
    "zone_pressure_pa",
    "zone_relative_humidity",
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_fraction",
    "achieved_actuator_state",
    "operating_mode_one_hot",
    "candidate_action_command_vector",
    "candidate_action_index",
)


class Issue74BaselineError(ValueError):
    """Raised when baseline inputs, fits, or predictions are inadmissible."""


def _check_recipe_against_contract() -> None:
    unknown = set(RECIPE_INPUT_FIELDS) - set(FEATURE_FIELD_NAMES)
    if unknown:
        raise Issue74BaselineError(f"recipe reads undeclared corpus fields: {sorted(unknown)}")


_check_recipe_against_contract()


def _slope(values: Sequence[float]) -> float:
    series = np.asarray([float(value) for value in values], dtype=np.float64)
    times = np.arange(len(series), dtype=np.float64)
    centered_times = times - times.mean()
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 0.0:
        return 0.0
    centered_values = series - series.mean()
    return float(np.dot(centered_times, centered_values) / denominator)


def context_features(sample: Mapping[str, Any]) -> np.ndarray:
    """45-dim observable context vector from declared corpus fields only."""

    features = sample["features"]
    o2_last = [float(value) for value in features["zone_o2_mole_fraction"][-1]]
    co2_last = [float(value) for value in features["zone_co2_ppm"][-1]]
    temp_last = [float(value) for value in features["zone_temperature_k"][-1]]
    o2_matrix = features["zone_o2_mole_fraction"]
    o2_slopes = [
        _slope([row[zone] for row in o2_matrix]) for zone in range(len(o2_matrix[0]))
    ]
    gauges = [
        float(features["battery_state_of_charge"][-1]),
        float(features["oxygen_store_fraction"][-1]),
        float(features["sorbent_fraction"][-1]),
    ]
    achieved = features["achieved_actuator_state"][-1]
    achieved_summary = [
        float(achieved[0]),
        float(np.mean(achieved[1:9])),
        float(np.sum(achieved[11:19])),
        float(np.sum(achieved[19:27])),
    ]
    mode = [float(value) for value in features["operating_mode_one_hot"][-1]]
    vector = np.asarray(
        [
            *o2_last,
            *co2_last,
            *temp_last,
            *o2_slopes,
            min(o2_last),
            max(co2_last),
            *gauges,
            *achieved_summary,
            *mode,
        ],
        dtype=np.float64,
    )
    if vector.shape != (CONTEXT_DIM,) or not np.isfinite(vector).all():
        raise Issue74BaselineError("context feature vector is malformed or non-finite")
    return vector


def action_features(sample: Mapping[str, Any]) -> np.ndarray:
    """31-dim candidate action encoding (one-hot plus command vector)."""

    features = sample["features"]
    index = int(features["candidate_action_index"])
    one_hot = [1.0 if position == index else 0.0 for position in range(4)]
    command = [float(value) for value in features["candidate_action_command_vector"]]
    vector = np.asarray([*one_hot, *command], dtype=np.float64)
    if vector.shape != (ACTION_DIM,) or not np.isfinite(vector).all():
        raise Issue74BaselineError("action feature vector is malformed or non-finite")
    return vector


def state_features(sample: Mapping[str, Any]) -> np.ndarray:
    """43-dim checkpoint observed state (last env rows for 5 sensed channels plus gauges)."""

    features = sample["features"]
    last = [
        float(value)
        for channel in (
            "zone_temperature_k",
            "zone_pressure_pa",
            "zone_co2_ppm",
            "zone_o2_mole_fraction",
            "zone_relative_humidity",
        )
        for value in features[channel][-1]
    ]
    gauges = [
        float(features["battery_state_of_charge"][-1]),
        float(features["oxygen_store_fraction"][-1]),
        float(features["sorbent_fraction"][-1]),
    ]
    vector = np.asarray([*last, *gauges], dtype=np.float64)
    if vector.shape != (STATE_DIM,) or not np.isfinite(vector).all():
        raise Issue74BaselineError("state feature vector is malformed or non-finite")
    return vector


def trajectory_labels(sample: Mapping[str, Any]) -> np.ndarray:
    """153-dim true trajectory label (horizons 4/16/32 concatenated)."""

    targets = sample["labels"]["trajectory_targets"]
    values = [float(value) for key in LABEL_HORIZON_KEYS for value in targets[key]]
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (153,) or not np.isfinite(vector).all():
        raise Issue74BaselineError("trajectory labels are malformed or non-finite")
    return vector


def delta_labels(sample: Mapping[str, Any]) -> np.ndarray:
    """5-dim true action-minus-hold decision targets."""

    deltas = sample["action_minus_hold"]
    vector = np.asarray([float(deltas[name]) for name in DECISION_LABEL_NAMES], dtype=np.float64)
    if vector.shape != (5,) or not np.isfinite(vector).all():
        raise Issue74BaselineError("decision delta labels are malformed or non-finite")
    return vector


@dataclass(frozen=True, slots=True)
class RidgeBaseline:
    """Closed-form ridge with standardized inputs; trajectory and delta heads."""

    baseline_id: str
    mean: np.ndarray
    scale: np.ndarray
    trajectory_weights: np.ndarray
    trajectory_bias: np.ndarray
    delta_weights: np.ndarray
    delta_bias: np.ndarray
    ridge_lambda: float
    digest: str

    def _normalized(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.scale

    def predict_trajectory(self, features: np.ndarray) -> np.ndarray:
        return self._normalized(features) @ self.trajectory_weights + self.trajectory_bias

    def predict_delta(self, features: np.ndarray) -> np.ndarray:
        return self._normalized(features) @ self.delta_weights + self.delta_bias


@dataclass(frozen=True, slots=True)
class LinearStateSpaceBaseline:
    """Per-horizon linear transition maps from observed state plus command.

    Decision deltas come from a linear delta head on the same controlled
    input (the linearization of the controlled dynamics), because the hold
    command is not part of the corpus sample record.
    """

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    delta_weights: np.ndarray
    delta_bias: np.ndarray
    ridge_lambda: float
    digest: str

    def _features(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(state, dtype=np.float64), np.asarray(command, dtype=np.float64)]
        )

    def predict_trajectory(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        features = self._features(state, command)
        rows = [
            features @ weights + bias
            for weights, bias in zip(self.weights, self.biases, strict=True)
        ]
        return np.concatenate(rows)

    def predict_delta(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        return self._features(state, command) @ self.delta_weights + self.delta_bias


def _standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def _ridge_solve(features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    augmented = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    gram = augmented.T @ augmented
    gram += ridge_lambda * np.eye(augmented.shape[1])
    gram[-1, -1] -= ridge_lambda
    solution = np.linalg.solve(gram, augmented.T @ targets)
    return solution[:-1], solution[-1]


def fit_ridge_baseline(
    samples: Sequence[Mapping[str, Any]],
    *,
    baseline_id: str,
    ridge_lambda: float,
    seed: str,
) -> RidgeBaseline:
    if baseline_id not in ("action_agnostic_ridge", "action_conditioned_ridge"):
        raise Issue74BaselineError(f"{baseline_id!r} is not a ridge baseline")
    if not samples:
        raise Issue74BaselineError("ridge fit requires samples")
    contexts = np.stack([context_features(sample) for sample in samples])
    if baseline_id == "action_conditioned_ridge":
        features = np.concatenate(
            [contexts, np.stack([action_features(sample) for sample in samples])], axis=1
        )
    else:
        features = contexts
    trajectories = np.stack([trajectory_labels(sample) for sample in samples])
    deltas = np.stack([delta_labels(sample) for sample in samples])
    if not np.isfinite(features).all():
        raise Issue74BaselineError("ridge features are non-finite")
    mean, scale = _standardizer(features)
    normalized = (features - mean) / scale
    trajectory_weights, trajectory_bias = _ridge_solve(normalized, trajectories, ridge_lambda)
    delta_weights, delta_bias = _ridge_solve(normalized, deltas, ridge_lambda)
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "baseline_id": baseline_id,
                "seed": seed,
                "ridge_lambda": ridge_lambda,
                "trajectory_weights": [[float(value) for value in row] for row in trajectory_weights],
                "delta_weights": [[float(value) for value in row] for row in delta_weights],
            }
        )
    ).hexdigest()
    return RidgeBaseline(
        baseline_id,
        mean,
        scale,
        trajectory_weights,
        trajectory_bias,
        delta_weights,
        delta_bias,
        ridge_lambda,
        digest,
    )


def fit_linear_state_space(
    samples: Sequence[Mapping[str, Any]],
    *,
    ridge_lambda: float,
    seed: str,
) -> LinearStateSpaceBaseline:
    if not samples:
        raise Issue74BaselineError("state-space fit requires samples")
    states = np.stack([state_features(sample) for sample in samples])
    commands = np.stack(
        [
            np.asarray(sample["features"]["candidate_action_command_vector"], dtype=np.float64)
            for sample in samples
        ]
    )
    augmented = np.concatenate([states, commands, np.ones((states.shape[0], 1))], axis=1)
    if not np.isfinite(augmented).all():
        raise Issue74BaselineError("state-space inputs are non-finite")
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for key in LABEL_HORIZON_KEYS:
        targets = np.stack(
            [np.asarray(sample["labels"]["trajectory_targets"][key], dtype=np.float64) for sample in samples]
        )
        gram = augmented.T @ augmented
        gram += ridge_lambda * np.eye(augmented.shape[1])
        gram[-1, -1] -= ridge_lambda
        solution = np.linalg.solve(gram, augmented.T @ targets)
        weights.append(solution[:-1])
        biases.append(solution[-1])
    deltas = np.stack([delta_labels(sample) for sample in samples])
    gram = augmented.T @ augmented
    gram += ridge_lambda * np.eye(augmented.shape[1])
    gram[-1, -1] -= ridge_lambda
    delta_solution = np.linalg.solve(gram, augmented.T @ deltas)
    delta_weights, delta_bias = delta_solution[:-1], delta_solution[-1]
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "baseline_id": "controlled_linear_state_space",
                "seed": seed,
                "ridge_lambda": ridge_lambda,
                "weights": [
                    [[float(value) for value in row] for row in matrix]
                    for matrix in weights
                ],
                "delta_weights": [[float(value) for value in row] for row in delta_weights],
            }
        )
    ).hexdigest()
    return LinearStateSpaceBaseline(
        tuple(weights), tuple(biases), delta_weights, delta_bias, ridge_lambda, digest
    )


def predict_sample(
    baseline: RidgeBaseline | LinearStateSpaceBaseline,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict trajectory and action-minus-hold decision targets for one sample."""

    if isinstance(baseline, RidgeBaseline):
        features = context_features(sample)
        if baseline.baseline_id == "action_conditioned_ridge":
            features = np.concatenate([features, action_features(sample)])
        trajectory = baseline.predict_trajectory(features)
        delta = baseline.predict_delta(features)
    else:
        state = state_features(sample)
        command = np.asarray(
            sample["features"]["candidate_action_command_vector"], dtype=np.float64
        )
        trajectory = baseline.predict_trajectory(state, command)
        delta = baseline.predict_delta(state, command)
    if not np.isfinite(trajectory).all() or not np.isfinite(delta).all():
        raise Issue74BaselineError("baseline prediction is non-finite")
    return {
        "trajectory": [float(value) for value in trajectory],
        "delta": {name: float(value) for name, value in zip(DECISION_LABEL_NAMES, delta, strict=True)},
    }


def ranking_correlation(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Spearman correlation of candidate rankings (ties averaged)."""

    if len(predicted) != len(true) or len(predicted) < 2:
        raise Issue74BaselineError("ranking correlation requires aligned pairs")

    def ranks(values: Sequence[float]) -> list[float]:
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(ordered):
            end = position
            while end + 1 < len(ordered) and values[ordered[end + 1]] == values[ordered[position]]:
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[ordered[index]] = average
            position = end + 1
        return result

    predicted_ranks = ranks([float(value) for value in predicted])
    true_ranks = ranks([float(value) for value in true])
    predicted_centered = np.asarray(predicted_ranks) - np.mean(predicted_ranks)
    true_centered = np.asarray(true_ranks) - np.mean(true_ranks)
    denominator = float(np.sqrt(np.dot(predicted_centered, predicted_centered) * np.dot(true_centered, true_centered)))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(predicted_centered, true_centered) / denominator)


def finite_catalogue_regret(predicted: Sequence[float], true: Sequence[float]) -> float:
    """True delta of the predicted-best candidate minus the best true delta."""

    if len(predicted) != len(true) or not predicted:
        raise Issue74BaselineError("regret requires aligned non-empty pairs")
    chosen = int(np.argmin([float(value) for value in predicted]))
    return float(true[chosen]) - float(min(float(value) for value in true))


def useful_action_precision_recall(
    predicted_improving: Sequence[bool], true_improving: Sequence[bool]
) -> tuple[float, float]:
    predicted_flags = [bool(value) for value in predicted_improving]
    true_flags = [bool(value) for value in true_improving]
    true_positives = sum(1 for predicted, true in zip(predicted_flags, true_flags, strict=True) if predicted and true)
    predicted_positives = sum(1 for predicted in predicted_flags if predicted)
    true_positives_count = sum(1 for true in true_flags if true)
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / true_positives_count if true_positives_count else 0.0
    return precision, recall


__all__ = [
    "ACTION_DIM",
    "BASELINE_IDS",
    "CONTEXT_DIM",
    "DECISION_LABEL_NAMES",
    "ISSUE74_BASELINE_SCHEMA_VERSION",
    "Issue74BaselineError",
    "LABEL_HORIZON_KEYS",
    "LinearStateSpaceBaseline",
    "RECIPE_INPUT_FIELDS",
    "RidgeBaseline",
    "STATE_DIM",
    "action_features",
    "context_features",
    "delta_labels",
    "finite_catalogue_regret",
    "fit_linear_state_space",
    "fit_ridge_baseline",
    "predict_sample",
    "ranking_correlation",
    "state_features",
    "trajectory_labels",
    "useful_action_precision_recall",
]
