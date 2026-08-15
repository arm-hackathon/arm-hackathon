from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_compact_history_uses_only_causal_target_estimators() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot_baselines import compact_target_history
    from aeolus.habitat_v2.forecast.projection import forecast_layout

    layout = forecast_layout(load_forecast_contracts(ROOT))
    numeric = np.zeros((4, 194), dtype=np.float32)
    numeric[:, 0] = (10.0, 20.0, 30.0, 40.0)
    numeric[:, 40] = (14.0, 24.0, 34.0, 44.0)
    numeric[:, 130] = (1.0, 2.0, 3.0, 4.0)
    numeric[:, 164] = (0.1, 0.2, 0.3, 0.4)

    result = compact_target_history(numeric, layout)

    assert result.dtype == np.float32
    assert result.shape == (4, 51)
    assert np.array_equal(
        result[:, 0], np.array((12.0, 22.0, 32.0, 42.0), dtype=np.float32)
    )
    assert np.array_equal(
        result[:, 5], np.array((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
    )
    assert np.array_equal(
        result[:, 48], np.array((0.1, 0.2, 0.3, 0.4), dtype=np.float32)
    )
    assert np.all(result[:, 1:5] == 0.0)


@pytest.mark.parametrize(
    "history",
    (
        np.zeros((4, 193), dtype=np.float32),
        np.zeros((4, 194), dtype=np.float64),
        np.full((4, 194), np.nan, dtype=np.float32),
    ),
)
def test_compact_history_rejects_unbound_or_nonfinite_numeric_history(
    history: np.ndarray,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot_baselines import (
        PilotBaselineError,
        compact_target_history,
    )
    from aeolus.habitat_v2.forecast.projection import forecast_layout

    layout = forecast_layout(load_forecast_contracts(ROOT))
    with pytest.raises(PilotBaselineError):
        compact_target_history(history, layout)


def test_packet_examples_slice_maximum_tensors_without_physics_rerun() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot_baselines import packet_examples
    from aeolus.habitat_v2.forecast.projection import forecast_layout

    layout = forecast_layout(load_forecast_contracts(ROOT))
    histories = np.zeros((5, 16, 194), dtype=np.float32)
    histories[:, :, 0] = 10.0
    histories[:, :, 40] = 14.0
    actions = np.arange(5 * 27, dtype=np.float32).reshape(5, 27)
    targets = np.arange(5 * 8 * 51, dtype=np.float32).reshape(5, 8, 51)

    examples = packet_examples(
        continuation_ids=np.asarray([f"sample-{index}" for index in range(5)]),
        cluster_ids=np.asarray(["cluster-a"] * 5),
        action_present=np.asarray([False, True, True, True, True]),
        history_numeric_f32=histories,
        proposed_action_f32=actions,
        targets_f32=targets,
        layout=layout,
        window_steps=4,
        horizon_steps=2,
    )

    assert len(examples) == 5
    assert examples[0].sample_id == "sample-0"
    assert examples[0].cluster_id == "cluster-a"
    assert examples[0].action_present is False
    assert examples[0].history_f32.shape == (4, 51)
    assert np.all(examples[0].history_f32[:, 0] == 12.0)
    assert np.array_equal(examples[4].action_f32, actions[4])
    assert np.array_equal(examples[3].targets_f32, targets[3, :2])


def test_compact_ridge_features_blind_actions_and_fit_a_known_relation() -> None:
    from aeolus.habitat_v2.forecast.pilot_baselines import (
        PilotExample,
        compact_feature_matrix,
        fit_compact_ridge,
    )

    examples = []
    for index in range(12):
        history = np.zeros((4, 51), dtype=np.float32)
        history[:, 0] = float(index)
        action = np.zeros(27, dtype=np.float32)
        action[0] = float(index % 3)
        target = np.zeros((2, 51), dtype=np.float32)
        target[:, 0] = 2.0 * history[-1, 0] + 3.0 * action[0]
        examples.append(
            PilotExample(
                sample_id=f"sample-{index}",
                cluster_id=f"cluster-{index // 2}",
                action_present=True,
                history_f32=history,
                action_f32=action,
                targets_f32=target,
            )
        )

    aware = compact_feature_matrix(tuple(examples), include_action=True)
    blind = compact_feature_matrix(tuple(examples), include_action=False)
    assert aware.shape == (12, 4 * 51 + 27)
    assert blind.shape == (12, 4 * 51)

    model = fit_compact_ridge(tuple(examples), include_action=True, alpha=0.001)
    prediction = model.predict(examples[5].history_f32, examples[5].action_f32)
    assert prediction.shape == (2, 51)
    assert prediction[0, 0] == pytest.approx(examples[5].targets_f32[0, 0], abs=0.01)

    changed_actions = tuple(
        PilotExample(
            sample_id=item.sample_id,
            cluster_id=item.cluster_id,
            action_present=item.action_present,
            history_f32=item.history_f32,
            action_f32=np.full(27, 99.0, dtype=np.float32),
            targets_f32=item.targets_f32,
        )
        for item in examples
    )
    assert np.array_equal(
        compact_feature_matrix(tuple(examples), include_action=False),
        compact_feature_matrix(changed_actions, include_action=False),
    )
