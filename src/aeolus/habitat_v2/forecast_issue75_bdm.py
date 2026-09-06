"""Issue #75 BDM-v1 causal temporal-convolutional adviser.

A compact, advisory-only causal dilated TCN in pure NumPy that predicts
candidate-action consequences and declared quantiles from the Issue #72
corpus windows. Architecture, loss, optimizer, seeds, and stopping rules are
preregistered in ``contracts/habitat_v2_bdm_v1_tcn_preregistration_v1.json``
and mirrored by the constants below; the training runner cross-checks both.

The model proposes or abstains only. Nothing in this module imports the HMC
or steps the plant; there is no ONNX, quantisation, or release path here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .bdm_v1_corpus import FEATURE_FIELD_NAMES
from .forecast.contracts import canonical_json_bytes

ISSUE75_BDM_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_tcn_v1"
WINDOW_STEPS = 16
SEQ_CHANNELS = 209
CANDIDATE_DIM = 31
INPUT_DIM = 240
CHANNELS = 32
KERNEL_SIZE = 3
DILATIONS = (1, 2, 4, 8)
RESIDUAL_BLOCKS = 3
HEAD_DIM = 632
QUANTILES = (0.1, 0.5, 0.9)
TRAJ_SLICE = (0, 153)
TRAJ_QUANTILE_SLICE = (153, 612)
DELTA_SLICE = (612, 617)
DELTA_QUANTILE_SLICE = (617, 632)
PARAMETER_BUDGET_CAP = 100_000

ENV_SEQUENCE_FIELDS = (
    "zone_o2_mole_fraction",
    "zone_co2_ppm",
    "zone_temperature_k",
    "zone_pressure_pa",
    "zone_relative_humidity",
    "zone_branch_airflow_m3_s",
)
RECIPE_INPUT_FIELDS = (
    *ENV_SEQUENCE_FIELDS,
    "observed_value_mask",
    "steps_since_last_valid_observation",
    "requested_command_vector",
    "achieved_actuator_state",
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_fraction",
    "operating_mode_one_hot",
    "prior_proposal_dispositions",
    "candidate_action_command_vector",
    "candidate_action_index",
)


class Issue75BdmError(ValueError):
    """Raised when BDM-v1 inputs, architecture, or training are inadmissible."""


def _check_recipe_against_contract() -> None:
    unknown = set(RECIPE_INPUT_FIELDS) - set(FEATURE_FIELD_NAMES)
    if unknown:
        raise Issue75BdmError(f"BDM-v1 recipe reads undeclared corpus fields: {sorted(unknown)}")


_check_recipe_against_contract()


def parameter_count() -> int:
    """Exact trainable parameter count of the preregistered architecture."""

    first = KERNEL_SIZE * INPUT_DIM * CHANNELS + CHANNELS
    blocks = RESIDUAL_BLOCKS * (KERNEL_SIZE * CHANNELS * CHANNELS + CHANNELS)
    head = CHANNELS * HEAD_DIM + HEAD_DIM
    return first + blocks + head


def sequence_tensor(sample: Mapping[str, Any]) -> np.ndarray:
    """[16, 209] observable sequence from declared corpus fields only."""

    features = sample["features"]
    blocks: list[np.ndarray] = []
    for field in ENV_SEQUENCE_FIELDS:
        matrix = np.asarray(features[field], dtype=np.float64)
        blocks.append(matrix)
    blocks.append(
        np.asarray(features["observed_value_mask"], dtype=np.float64)
        .transpose(0, 2, 1)
        .reshape(WINDOW_STEPS, -1)
    )
    blocks.append(
        np.asarray(features["steps_since_last_valid_observation"], dtype=np.float64)
        .transpose(0, 2, 1)
        .reshape(WINDOW_STEPS, -1)
    )
    for field in ("requested_command_vector", "achieved_actuator_state"):
        blocks.append(np.asarray(features[field], dtype=np.float64))
    gauges = np.stack(
        [
            np.asarray(features[name], dtype=np.float64)
            for name in ("battery_state_of_charge", "oxygen_store_fraction", "sorbent_fraction")
        ],
        axis=1,
    )
    blocks.append(gauges)
    blocks.append(np.asarray(features["operating_mode_one_hot"], dtype=np.float64))
    blocks.append(np.asarray(features["prior_proposal_dispositions"], dtype=np.float64))
    tensor = np.concatenate(blocks, axis=1)
    if tensor.shape != (WINDOW_STEPS, SEQ_CHANNELS) or not np.isfinite(tensor).all():
        raise Issue75BdmError("BDM-v1 sequence tensor is malformed or non-finite")
    return tensor


def candidate_encoding(sample: Mapping[str, Any]) -> np.ndarray:
    features = sample["features"]
    index = int(features["candidate_action_index"])
    one_hot = np.zeros(4, dtype=np.float64)
    one_hot[index] = 1.0
    command = np.asarray(features["candidate_action_command_vector"], dtype=np.float64)
    encoding = np.concatenate([one_hot, command])
    if encoding.shape != (CANDIDATE_DIM,) or not np.isfinite(encoding).all():
        raise Issue75BdmError("BDM-v1 candidate encoding is malformed or non-finite")
    return encoding


def input_tensor(sample: Mapping[str, Any]) -> np.ndarray:
    sequence = sequence_tensor(sample)
    encoding = candidate_encoding(sample)
    tensor = np.concatenate(
        [sequence, np.broadcast_to(encoding, (WINDOW_STEPS, CANDIDATE_DIM))], axis=1
    )
    if tensor.shape != (WINDOW_STEPS, INPUT_DIM):
        raise Issue75BdmError("BDM-v1 input tensor has the wrong shape")
    return tensor


@dataclass(frozen=True, slots=True)
class BdmV1Tcn:
    """Frozen weights plus standardizer for one trained seed."""

    mean: np.ndarray
    scale: np.ndarray
    first_weights: np.ndarray
    first_bias: np.ndarray
    block_weights: tuple[np.ndarray, ...]
    block_biases: tuple[np.ndarray, ...]
    head_weights: np.ndarray
    head_bias: np.ndarray
    seed: str
    epochs_run: int
    best_epoch: int
    digest: str

    def parameters(self) -> tuple[np.ndarray, ...]:
        return (
            self.first_weights,
            self.first_bias,
            *self.block_weights,
            *self.block_biases,
            self.head_weights,
            self.head_bias,
        )


def _conv_forward(h: np.ndarray, weights: np.ndarray, bias: np.ndarray, dilation: int) -> tuple[np.ndarray, np.ndarray]:
    batch, length, _ = h.shape
    pre = np.zeros((batch, length, weights.shape[2]), dtype=np.float64)
    for tap in range(KERNEL_SIZE):
        offset = dilation * tap
        if offset == 0:
            pre += h @ weights[tap]
        else:
            pre[:, offset:, :] += h[:, : length - offset, :] @ weights[tap]
    pre += bias
    return pre, np.maximum(pre, 0.0)


def forward(model: BdmV1Tcn, tensors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Batched causal TCN forward pass; returns heads and the backward cache."""

    if tensors.ndim != 3 or tensors.shape[1:] != (WINDOW_STEPS, INPUT_DIM):
        raise Issue75BdmError("forward expects [batch, 16, 240] tensors")
    if not np.isfinite(tensors).all():
        raise Issue75BdmError("forward received non-finite inputs")
    x = (tensors - model.mean) / model.scale
    pre1, h1 = _conv_forward(x, model.first_weights, model.first_bias, DILATIONS[0])
    cache: dict[str, Any] = {"x": x, "pre1": pre1, "h": [h1], "pres": []}
    current = h1
    for index, dilation in enumerate(DILATIONS[1:], start=0):
        pre, activated = _conv_forward(
            current, model.block_weights[index], model.block_biases[index], dilation
        )
        current = activated + current
        cache["pres"].append(pre)
        cache["h"].append(current)
    features = current[:, -1, :]
    out = features @ model.head_weights + model.head_bias
    cache["features"] = features
    cache["out"] = out
    return out, cache


def _conv_backward(
    grad_out: np.ndarray,
    h_in: np.ndarray,
    weights: np.ndarray,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, length, _ = h_in.shape
    grad_weights = np.zeros_like(weights)
    grad_bias = grad_out.sum(axis=(0, 1))
    grad_h = np.zeros_like(h_in)
    for tap in range(KERNEL_SIZE):
        offset = dilation * tap
        if offset == 0:
            grad_weights[tap] = np.einsum("btc,btd->cd", h_in, grad_out)
            grad_h += np.einsum("btd,cd->btc", grad_out, weights[tap])
        else:
            grad_weights[tap] = np.einsum("btc,btd->cd", h_in[:, : length - offset], grad_out[:, offset:])
            grad_h[:, : length - offset] += np.einsum("btd,cd->btc", grad_out[:, offset:], weights[tap])
    return grad_weights, grad_bias, grad_h


def backward(model: BdmV1Tcn, cache: dict[str, Any], grad_out: np.ndarray) -> tuple[np.ndarray, ...]:
    """Gradient of the loss w.r.t. every parameter array, cache-consistent."""

    grad_head_weights = cache["features"].T @ grad_out
    grad_head_bias = grad_out.sum(axis=0)
    grad_features = grad_out @ model.head_weights.T
    grad_current = np.zeros_like(cache["h"][-1])
    grad_current[:, -1, :] = grad_features
    grad_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(RESIDUAL_BLOCKS - 1, -1, -1):
        pre = cache["pres"][index]
        grad_pre = grad_current * (pre > 0.0)
        h_in = cache["h"][index]
        grad_weights, grad_bias, grad_h = _conv_backward(
            grad_pre, h_in, model.block_weights[index], DILATIONS[index + 1]
        )
        grad_blocks.append((grad_weights, grad_bias))
        grad_current = grad_h + grad_current
    grad_pre1 = grad_current * (cache["pre1"] > 0.0)
    grad_first_weights, grad_first_bias, _ = _conv_backward(
        grad_pre1, cache["x"], model.first_weights, DILATIONS[0]
    )
    reversed_blocks = tuple(reversed(grad_blocks))
    return (
        grad_first_weights,
        grad_first_bias,
        *[array for pair in reversed_blocks for array in pair],
        grad_head_weights,
        grad_head_bias,
    )


def losses_and_output_grad(
    out: np.ndarray, trajectories: np.ndarray, deltas: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """Preregistered mixed loss and its gradient w.r.t. the head outputs."""

    batch = out.shape[0]
    traj_pred = out[:, TRAJ_SLICE[0] : TRAJ_SLICE[1]]
    delta_pred = out[:, DELTA_SLICE[0] : DELTA_SLICE[1]]
    traj_q = out[:, TRAJ_QUANTILE_SLICE[0] : TRAJ_QUANTILE_SLICE[1]].reshape(
        batch, len(QUANTILES), 153
    )
    delta_q = out[:, DELTA_QUANTILE_SLICE[0] : DELTA_QUANTILE_SLICE[1]].reshape(
        batch, len(QUANTILES), 5
    )
    mse = float(np.mean((traj_pred - trajectories) ** 2))
    err = delta_pred - deltas
    huber = float(np.mean(np.where(np.abs(err) <= 1.0, 0.5 * err**2, np.abs(err) - 0.5)))
    probs = np.asarray(QUANTILES, dtype=np.float64)[None, :, None]
    diff_traj = trajectories[:, None, :] - traj_q
    diff_delta = deltas[:, None, :] - delta_q
    loss_traj = np.maximum(probs * diff_traj, (probs - 1.0) * diff_traj)
    loss_delta = np.maximum(probs * diff_delta, (probs - 1.0) * diff_delta)
    pin = float((loss_traj.sum() + loss_delta.sum()) / (loss_traj.size + loss_delta.size))
    total = mse + huber + 0.5 * pin

    grad = np.zeros_like(out)
    grad[:, TRAJ_SLICE[0] : TRAJ_SLICE[1]] = 2.0 * (traj_pred - trajectories) / traj_pred.size
    grad[:, DELTA_SLICE[0] : DELTA_SLICE[1]] = (
        np.where(np.abs(err) <= 1.0, err, np.sign(err)) / delta_pred.size
    )
    quantile_size = loss_traj.size + loss_delta.size
    grad[:, TRAJ_QUANTILE_SLICE[0] : TRAJ_QUANTILE_SLICE[1]] = (
        0.5
        * np.where(diff_traj >= 0.0, -probs, 1.0 - probs).reshape(batch, -1)
        / quantile_size
    )
    grad[:, DELTA_QUANTILE_SLICE[0] : DELTA_QUANTILE_SLICE[1]] = (
        0.5
        * np.where(diff_delta >= 0.0, -probs, 1.0 - probs).reshape(batch, -1)
        / quantile_size
    )
    return {
        "trajectory_mse": mse,
        "delta_huber": huber,
        "quantile_pinball": pin,
        "total": total,
    }, grad


def _init_weights(seed: str) -> dict[str, Any]:
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
    )
    first = rng.normal(0.0, np.sqrt(2.0 / (INPUT_DIM * KERNEL_SIZE)), (KERNEL_SIZE, INPUT_DIM, CHANNELS))
    blocks = tuple(
        rng.normal(0.0, np.sqrt(2.0 / (CHANNELS * KERNEL_SIZE)), (KERNEL_SIZE, CHANNELS, CHANNELS))
        for _ in range(RESIDUAL_BLOCKS)
    )
    block_biases = tuple(np.zeros(CHANNELS) for _ in range(RESIDUAL_BLOCKS))
    head = rng.normal(0.0, 0.01, (CHANNELS, HEAD_DIM))
    return {
        "first_weights": first,
        "first_bias": np.zeros(CHANNELS),
        "block_weights": blocks,
        "block_biases": block_biases,
        "head_weights": head,
        "head_bias": np.zeros(HEAD_DIM),
        "rng": rng,
    }


def model_digest(params: Sequence[np.ndarray], seed: str, epochs_run: int, best_epoch: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "seed": seed,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "weights": [matrix.tolist() for matrix in params],
            }
        )
    ).hexdigest()


PARAMETER_ORDER: tuple[str, ...] = (
    "first_weights",
    "first_bias",
    "block_weights[0]",
    "block_biases[0]",
    "block_weights[1]",
    "block_biases[1]",
    "block_weights[2]",
    "block_biases[2]",
    "head_weights",
    "head_bias",
)


def _assemble(
    mean: np.ndarray, scale: np.ndarray, params: Sequence[np.ndarray], seed: str
) -> BdmV1Tcn:
    return BdmV1Tcn(
        mean,
        scale,
        params[0],
        params[1],
        (params[2], params[4], params[6]),
        (params[3], params[5], params[7]),
        params[8],
        params[9],
        seed,
        0,
        0,
        "",
    )


def train_model(
    train_tensors: np.ndarray,
    train_trajectories: np.ndarray,
    train_deltas: np.ndarray,
    dev_tensors: np.ndarray,
    dev_deltas: np.ndarray,
    *,
    seed: str,
    learning_rate: float,
    batch_size: int,
    epochs_max: int,
    patience: int,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> BdmV1Tcn:
    """Adam training with DEV delta-MSE early stopping (model choice on DEV)."""

    if not (train_tensors.shape[0] == train_trajectories.shape[0] == train_deltas.shape[0]):
        raise Issue75BdmError("train tensors and labels are misaligned")
    if train_tensors.shape[1:] != (WINDOW_STEPS, INPUT_DIM):
        raise Issue75BdmError("train tensors must be [n, 16, 240]")
    mean = train_tensors.mean(axis=(0, 1))
    scale = train_tensors.std(axis=(0, 1))
    scale = np.where(scale > 1e-8, scale, 1.0)
    init = _init_weights(seed)
    rng = init.pop("rng")
    params: list[np.ndarray] = [
        init["first_weights"].copy(),
        init["first_bias"].copy(),
        init["block_weights"][0].copy(),
        init["block_biases"][0].copy(),
        init["block_weights"][1].copy(),
        init["block_biases"][1].copy(),
        init["block_weights"][2].copy(),
        init["block_biases"][2].copy(),
        init["head_weights"].copy(),
        init["head_bias"].copy(),
    ]
    moments = [np.zeros_like(array) for array in params]
    velocities = [np.zeros_like(array) for array in params]
    best_dev = np.inf
    best_params = [array.copy() for array in params]
    best_epoch = 0
    waited = 0
    epochs_run = 0
    for epoch in range(1, epochs_max + 1):
        epochs_run = epoch
        order = rng.permutation(train_tensors.shape[0])
        for start in range(0, train_tensors.shape[0], batch_size):
            indices = order[start : start + batch_size]
            model = _assemble(mean, scale, params, seed)
            out, cache = forward(model, train_tensors[indices])
            _losses, grad_out = losses_and_output_grad(
                out, train_trajectories[indices], train_deltas[indices]
            )
            grads = backward(model, cache, grad_out)
            if len(grads) != len(params):
                raise Issue75BdmError("gradient/parameter order drifted")
            for index, grad in enumerate(grads):
                moments[index] = betas[0] * moments[index] + (1 - betas[0]) * grad
                velocities[index] = betas[1] * velocities[index] + (1 - betas[1]) * grad**2
                mhat = moments[index] / (1 - betas[0] ** epoch)
                vhat = velocities[index] / (1 - betas[1] ** epoch)
                params[index] = params[index] - learning_rate * mhat / (np.sqrt(vhat) + eps)
        model = _assemble(mean, scale, params, seed)
        dev_out, _ = forward(model, dev_tensors)
        dev_mse = float(
            np.mean((dev_out[:, DELTA_SLICE[0] : DELTA_SLICE[1]] - dev_deltas) ** 2)
        )
        if dev_mse < best_dev - 1e-9:
            best_dev = dev_mse
            best_epoch = epoch
            waited = 0
            best_params = [array.copy() for array in params]
        else:
            waited += 1
            if waited >= patience:
                break
    digest = model_digest(best_params, seed, epochs_run, best_epoch)
    return BdmV1Tcn(
        mean,
        scale,
        best_params[0],
        best_params[1],
        (best_params[2], best_params[4], best_params[6]),
        (best_params[3], best_params[5], best_params[7]),
        best_params[8],
        best_params[9],
        seed,
        epochs_run,
        best_epoch,
        digest,
    )


def predict_deltas(model: BdmV1Tcn, tensors: np.ndarray) -> np.ndarray:
    out, _ = forward(model, tensors)
    return out[:, DELTA_SLICE[0] : DELTA_SLICE[1]]


__all__ = [
    "BdmV1Tcn",
    "CHANNELS",
    "DELTA_QUANTILE_SLICE",
    "DELTA_SLICE",
    "DILATIONS",
    "HEAD_DIM",
    "INPUT_DIM",
    "ISSUE75_BDM_SCHEMA_VERSION",
    "Issue75BdmError",
    "PARAMETER_BUDGET_CAP",
    "QUANTILES",
    "RECIPE_INPUT_FIELDS",
    "SEQ_CHANNELS",
    "TRAJ_QUANTILE_SLICE",
    "TRAJ_SLICE",
    "WINDOW_STEPS",
    "backward",
    "candidate_encoding",
    "forward",
    "input_tensor",
    "losses_and_output_grad",
    "model_digest",
    "parameter_count",
    "predict_deltas",
    "sequence_tensor",
    "train_model",
]
