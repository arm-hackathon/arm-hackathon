from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.bdm_v1_corpus import FEATURE_FIELD_NAMES
from aeolus.habitat_v2.forecast_issue75_bdm import (
    DELTA_SLICE,
    HEAD_DIM,
    INPUT_DIM,
    PARAMETER_BUDGET_CAP,
    SEQ_CHANNELS,
    WINDOW_STEPS,
    Issue75BdmError,
    _assemble,
    _init_weights,
    backward,
    candidate_encoding,
    forward,
    input_tensor,
    losses_and_output_grad,
    parameter_count,
    sequence_tensor,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_tcn_preregistration_v1.json"


def _make_sample(rng: np.random.Generator, action_index: int) -> dict:
    env = rng.normal(0.21, 0.01, (16, 8))
    return {
        "features": {
            "zone_o2_mole_fraction": env.tolist(),
            "zone_co2_ppm": rng.normal(500.0, 10.0, (16, 8)).tolist(),
            "zone_temperature_k": rng.normal(295.0, 0.5, (16, 8)).tolist(),
            "zone_pressure_pa": rng.normal(101.0, 0.2, (16, 8)).tolist(),
            "zone_relative_humidity": rng.normal(0.4, 0.01, (16, 8)).tolist(),
            "zone_branch_airflow_m3_s": rng.normal(0.05, 0.005, (16, 8)).tolist(),
            "observed_value_mask": [[[True] * 6 for _ in range(8)] for _ in range(16)],
            "steps_since_last_valid_observation": [[[0] * 6 for _ in range(8)] for _ in range(16)],
            "requested_command_vector": rng.normal(0.5, 0.05, (16, 27)).tolist(),
            "achieved_actuator_state": rng.normal(0.5, 0.05, (16, 27)).tolist(),
            "battery_state_of_charge": [0.9] * 16,
            "oxygen_store_fraction": [0.8] * 16,
            "sorbent_fraction": [0.7] * 16,
            "operating_mode_one_hot": [[1.0, 0.0, 0.0, 0.0] for _ in range(16)],
            "prior_proposal_dispositions": [[1.0, 0.0, 0.0, 0.0] for _ in range(16)],
            "topology_configuration_descriptor": {"zone_order": [str(i) for i in range(8)]},
            "candidate_action_command_vector": rng.normal(0.5, 0.05, 27).tolist(),
            "candidate_action_index": action_index,
            "declared_known_future_schedule": [],
        },
        "labels": {
            "decision_targets": {},
            "trajectory_targets": {
                key: rng.normal(0.0, 0.1, 51).tolist() for key in ("4", "16", "32")
            },
        },
        "action_minus_hold": {
            "crossing_event": 0.0,
            "safety_exposure": 0.1 * (action_index - 1.5),
            "maximum_crossing": 0.0,
            "comfort_deviation": 0.0,
            "resource_composite": 0.0,
        },
    }


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(5)
    return tuple(_make_sample(rng, index) for index in range(4))


@pytest.fixture(scope="module")
def tiny_model():
    init = _init_weights("test-seed")
    rng = init.pop("rng")
    del rng
    params = [
        init["first_weights"],
        init["first_bias"],
        init["block_weights"][0],
        init["block_biases"][0],
        init["block_weights"][1],
        init["block_biases"][1],
        init["block_weights"][2],
        init["block_biases"][2],
        init["head_weights"],
        init["head_bias"],
    ]
    mean = np.zeros(INPUT_DIM)
    scale = np.ones(INPUT_DIM)
    return _assemble(mean, scale, params, "test-seed")


def test_recipe_reads_only_declared_fields() -> None:
    from aeolus.habitat_v2 import forecast_issue75_bdm as module

    assert set(module.RECIPE_INPUT_FIELDS) <= set(FEATURE_FIELD_NAMES)


def test_parameter_count_matches_preregistration_and_cap() -> None:
    prereg = json.loads(PREREG_PATH.read_bytes())
    assert parameter_count() == 53240
    assert parameter_count() == prereg["architecture"]["parameter_count"]
    assert parameter_count() <= PARAMETER_BUDGET_CAP <= 100_000
    assert prereg["architecture"]["parameter_budget_cap"] == PARAMETER_BUDGET_CAP


def test_tensor_shapes(samples) -> None:
    assert sequence_tensor(samples[0]).shape == (WINDOW_STEPS, SEQ_CHANNELS)
    assert candidate_encoding(samples[0]).shape == (31,)
    assert input_tensor(samples[0]).shape == (WINDOW_STEPS, INPUT_DIM)


def test_causality_no_lookahead(tiny_model, samples) -> None:
    tensors = np.stack([input_tensor(sample) for sample in samples])
    _out, cache = forward(tiny_model, tensors)
    tampered = tensors.copy()
    cut = 9
    tampered[:, cut + 1 :, :] = 0.0
    _out2, cache2 = forward(tiny_model, tampered)
    for layer in range(len(cache["h"])):
        original = cache["h"][layer][:, : cut + 1, :]
        modified = cache2["h"][layer][:, : cut + 1, :]
        assert np.allclose(original, modified, atol=1e-12)
    assert not np.allclose(cache["h"][-1][:, cut + 1 :, :], cache2["h"][-1][:, cut + 1 :, :])


def test_missingness_and_action_sensitivity(tiny_model, samples) -> None:
    import copy

    base = input_tensor(samples[0])
    masked = copy.deepcopy(samples[0])
    masked["features"]["observed_value_mask"][15][0][0] = False
    masked["features"]["steps_since_last_valid_observation"][15][0][0] = 3
    out_base, _ = forward(tiny_model, base[None])
    out_masked, _ = forward(tiny_model, input_tensor(masked)[None])
    assert not np.allclose(out_base, out_masked)
    other = copy.deepcopy(samples[0])
    other["features"]["candidate_action_index"] = (
        samples[0]["features"]["candidate_action_index"] + 1
    ) % 4
    out_other, _ = forward(tiny_model, input_tensor(other)[None])
    assert not np.allclose(out_base, out_other)


def test_forward_rejects_nonfinite(tiny_model, samples) -> None:
    tensors = np.stack([input_tensor(sample) for sample in samples])
    tensors[0, 0, 0] = float("nan")
    with pytest.raises(Issue75BdmError, match="non-finite"):
        forward(tiny_model, tensors)


def test_loss_matches_independent_recomputation() -> None:
    rng = np.random.default_rng(3)
    out = rng.normal(0.0, 0.5, (4, HEAD_DIM))
    traj = rng.normal(0.0, 0.5, (4, 153))
    deltas = rng.normal(0.0, 0.5, (4, 5))
    losses, grad = losses_and_output_grad(out, traj, deltas)
    probs = np.asarray([0.1, 0.5, 0.9])
    traj_q = out[:, 153:612].reshape(4, 3, 153)
    delta_q = out[:, 617:632].reshape(4, 3, 5)
    diff_t = traj[:, None, :] - traj_q
    diff_d = deltas[:, None, :] - delta_q
    loss_t = np.maximum(probs[None, :, None] * diff_t, (probs[None, :, None] - 1) * diff_t)
    loss_d = np.maximum(probs[None, :, None] * diff_d, (probs[None, :, None] - 1) * diff_d)
    expected_pin = (loss_t.sum() + loss_d.sum()) / (loss_t.size + loss_d.size)
    err = out[:, DELTA_SLICE[0] : DELTA_SLICE[1]] - deltas
    expected_huber = float(np.mean(np.where(np.abs(err) <= 1, 0.5 * err**2, np.abs(err) - 0.5)))
    expected_mse = float(np.mean((out[:, :153] - traj) ** 2))
    assert losses["quantile_pinball"] == pytest.approx(expected_pin)
    assert losses["delta_huber"] == pytest.approx(expected_huber)
    assert losses["trajectory_mse"] == pytest.approx(expected_mse)
    assert losses["total"] == pytest.approx(expected_mse + expected_huber + 0.5 * expected_pin)
    assert grad.shape == out.shape and np.isfinite(grad).all()


def test_backward_matches_finite_differences(tiny_model, samples) -> None:
    tensors = np.stack([input_tensor(sample) for sample in samples])[:2]
    rng = np.random.default_rng(9)
    traj = rng.normal(0.0, 0.3, (2, 153))
    deltas = rng.normal(0.0, 0.3, (2, 5))
    out, cache = forward(tiny_model, tensors)
    _losses, grad_out = losses_and_output_grad(out, traj, deltas)
    grads = backward(tiny_model, cache, grad_out)
    params = tiny_model.parameters()
    for param_index in (0, 1, 8, 9):
        flat = params[param_index].reshape(-1)
        for element in (0, flat.size // 2, flat.size - 1):
            eps = 1e-6
            original = flat[element]
            flat[element] = original + eps
            out_plus, _ = forward(tiny_model, tensors)
            loss_plus = losses_and_output_grad(out_plus, traj, deltas)[0]["total"]
            flat[element] = original - eps
            out_minus, _ = forward(tiny_model, tensors)
            loss_minus = losses_and_output_grad(out_minus, traj, deltas)[0]["total"]
            flat[element] = original
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[param_index].reshape(-1)[element]
            assert abs(numeric - analytic) < 1e-6, (param_index, element, numeric, analytic)


def test_training_is_deterministic() -> None:
    rng = np.random.default_rng(21)
    train = rng.normal(0.0, 1.0, (8, WINDOW_STEPS, INPUT_DIM))
    traj = rng.normal(0.0, 1.0, (8, 153))
    deltas = rng.normal(0.0, 1.0, (8, 5))
    first = train_model(
        train, traj, deltas, train, deltas,
        seed="det", learning_rate=1e-3, batch_size=4, epochs_max=2, patience=2,
    )
    second = train_model(
        train, traj, deltas, train, deltas,
        seed="det", learning_rate=1e-3, batch_size=4, epochs_max=2, patience=2,
    )
    assert first.digest == second.digest
    assert first.digest != ""
    assert len(first.digest) == 64


def test_model_has_no_plant_or_hmc_authority() -> None:
    from aeolus.habitat_v2 import forecast_issue75_bdm as module

    source = inspect.getsource(module)
    assert "HabitatManagementComputer" not in source
    assert "advance_one_step_with_command" not in source
    assert "from .hmc" not in source
    assert "from .physics" not in source


def test_head_layout_is_preregistered() -> None:
    from aeolus.habitat_v2 import forecast_issue75_bdm as module

    prereg = json.loads(PREREG_PATH.read_bytes())
    heads = prereg["architecture"]["heads"]
    assert module.TRAJ_SLICE == (0, heads["trajectory"])
    delta_start = heads["trajectory"] + heads["trajectory_quantiles"]
    assert module.DELTA_SLICE == (delta_start, delta_start + heads["delta"])
    assert module.HEAD_DIM == heads["total"]
    assert module.CHANNELS == prereg["architecture"]["channels"]
