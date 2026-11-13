from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from aeolus.habitat_v2.bdm_v1_corpus import FEATURE_FIELD_NAMES
from aeolus.habitat_v2.forecast_issue74_baselines import (
    ACTION_DIM,
    BASELINE_IDS,
    CONTEXT_DIM,
    RECIPE_INPUT_FIELDS,
    STATE_DIM,
    Issue74BaselineError,
    action_features,
    context_features,
    delta_labels,
    finite_catalogue_regret,
    fit_linear_state_space,
    fit_ridge_baseline,
    predict_sample,
    ranking_correlation,
    state_features,
    trajectory_labels,
    useful_action_precision_recall,
)


def _make_sample(rng: np.random.Generator, action_index: int, weight: float) -> dict:
    env = rng.normal(0.21, 0.01, (16, 8)).tolist()
    co2 = rng.normal(500.0, 10.0, (16, 8)).tolist()
    temp = rng.normal(295.0, 0.5, (16, 8)).tolist()
    pressure = rng.normal(101.0, 0.2, (16, 8)).tolist()
    humidity = rng.normal(0.4, 0.01, (16, 8)).tolist()
    achieved = rng.normal(0.5, 0.05, (16, 27)).tolist()
    command = rng.normal(0.5, 0.05, 27).tolist()
    mode = [1.0, 0.0, 0.0, 0.0]
    trajectory = {
        key: (rng.normal(0.0, 0.1, 51) + weight * action_index).tolist()
        for key in ("4", "16", "32")
    }
    delta = [0.0, weight * (action_index - 1.5), 0.0, 0.0, 0.0]
    return {
        "features": {
            "zone_o2_mole_fraction": env,
            "zone_co2_ppm": co2,
            "zone_temperature_k": temp,
            "zone_pressure_pa": pressure,
            "zone_relative_humidity": humidity,
            "zone_branch_airflow_m3_s": env,
            "observed_value_mask": [[True] * 8 for _ in range(16)],
            "steps_since_last_valid_observation": [[0] * 8 for _ in range(16)],
            "requested_command_vector": achieved,
            "achieved_actuator_state": achieved,
            "battery_state_of_charge": [0.9] * 16,
            "oxygen_store_fraction": [0.8] * 16,
            "sorbent_fraction": [0.7] * 16,
            "operating_mode_one_hot": [mode for _ in range(16)],
            "prior_proposal_dispositions": [[1.0, 0.0, 0.0, 0.0] for _ in range(16)],
            "topology_configuration_descriptor": {"zone_order": [str(i) for i in range(8)]},
            "candidate_action_command_vector": command,
            "candidate_action_index": action_index,
            "declared_known_future_schedule": [],
        },
        "labels": {"decision_targets": {}, "trajectory_targets": trajectory},
        "action_minus_hold": {
            "crossing_event": delta[0],
            "safety_exposure": delta[1],
            "maximum_crossing": delta[2],
            "comfort_deviation": delta[3],
            "resource_composite": delta[4],
        },
        "group_id": "g",
        "family_id": "f",
        "decision_step": 16,
    }


@pytest.fixture(scope="module")
def synthetic_samples():
    rng = np.random.default_rng(11)
    return tuple(
        _make_sample(rng, action_index, 0.4)
        for _ in range(40)
        for action_index in range(4)
    )


def test_recipe_reads_only_declared_fields() -> None:
    assert set(RECIPE_INPUT_FIELDS) <= set(FEATURE_FIELD_NAMES)


def test_feature_vector_shapes(synthetic_samples) -> None:
    sample = synthetic_samples[0]
    assert context_features(sample).shape == (CONTEXT_DIM,)
    assert action_features(sample).shape == (ACTION_DIM,)
    assert state_features(sample).shape == (STATE_DIM,)
    assert trajectory_labels(sample).shape == (153,)
    assert delta_labels(sample).shape == (5,)


def test_non_finite_inputs_fail_closed(synthetic_samples) -> None:
    import copy

    bad = copy.deepcopy(synthetic_samples[0])
    bad["features"]["zone_o2_mole_fraction"][-1][0] = float("nan")
    with pytest.raises(Issue74BaselineError, match="non-finite"):
        context_features(bad)
    bad2 = copy.deepcopy(synthetic_samples[0])
    bad2["action_minus_hold"]["safety_exposure"] = float("inf")
    with pytest.raises(Issue74BaselineError, match="non-finite"):
        delta_labels(bad2)


def test_ridge_fit_is_deterministic_and_recovers_action_signal(synthetic_samples) -> None:
    first = fit_ridge_baseline(
        synthetic_samples, baseline_id="action_conditioned_ridge", ridge_lambda=1e-3, seed="s"
    )
    second = fit_ridge_baseline(
        synthetic_samples, baseline_id="action_conditioned_ridge", ridge_lambda=1e-3, seed="s"
    )
    assert first.digest == second.digest
    predictions = [
        predict_sample(first, sample)["delta"]["safety_exposure"]
        for sample in synthetic_samples[:4]
    ]
    assert predictions == sorted(predictions) or predictions == sorted(predictions, reverse=True)
    agnostic = fit_ridge_baseline(
        synthetic_samples, baseline_id="action_agnostic_ridge", ridge_lambda=1e-3, seed="s"
    )
    assert agnostic.baseline_id == "action_agnostic_ridge"
    with pytest.raises(Issue74BaselineError, match="not a ridge baseline"):
        fit_ridge_baseline(synthetic_samples, baseline_id="controlled_linear_state_space", ridge_lambda=1e-3, seed="s")


def test_state_space_fit_is_deterministic(synthetic_samples) -> None:
    first = fit_linear_state_space(synthetic_samples, ridge_lambda=1e-3, seed="s")
    second = fit_linear_state_space(synthetic_samples, ridge_lambda=1e-3, seed="s")
    assert first.digest == second.digest
    sample = synthetic_samples[0]
    prediction = predict_sample(first, sample)
    assert len(prediction["trajectory"]) == 153
    assert set(prediction["delta"]) == {
        "crossing_event",
        "safety_exposure",
        "maximum_crossing",
        "comfort_deviation",
        "resource_composite",
    }


def test_baselines_never_touch_the_plant() -> None:
    from aeolus.habitat_v2 import forecast_issue74_baselines as module

    source = inspect.getsource(module)
    assert "HabitatManagementComputer" not in source
    assert "advance_one_step_with_command" not in source
    assert "from .hmc" not in source
    assert "from .physics" not in source


def test_ranking_correlation_and_regret() -> None:
    assert ranking_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert ranking_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert ranking_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert finite_catalogue_regret([0.1, 0.2, 0.3], [0.5, 0.1, 0.2]) == pytest.approx(0.4)
    assert finite_catalogue_regret([0.1, 0.2], [0.1, 0.2]) == pytest.approx(0.0)
    with pytest.raises(Issue74BaselineError, match="aligned"):
        ranking_correlation([1.0], [1.0])


def test_precision_recall() -> None:
    precision, recall = useful_action_precision_recall(
        [True, True, False, False], [True, False, True, False]
    )
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert useful_action_precision_recall([False, False], [True, True]) == (0.0, 0.0)


def test_baseline_ids_are_complete() -> None:
    assert BASELINE_IDS == (
        "action_agnostic_ridge",
        "action_conditioned_ridge",
        "controlled_linear_state_space",
    )
    assert math.isfinite(CONTEXT_DIM + ACTION_DIM + STATE_DIM)
