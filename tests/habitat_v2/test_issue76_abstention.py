from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast_issue76_abstention import (
    ABSTENTION_REASONS,
    CalibrationLayer,
    Issue76AbstentionError,
    abstention_guard,
    abstention_reason,
    corrected_delta_interval,
    fit_conformal_offsets,
    interval_coverage,
    layer_digest,
    predict_with_abstention,
    risk_coverage_curve,
    sample_staleness_stats,
    trajectory_interval_coverage,
    useful_opportunity_recall,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample(stale: int = 0, mask: bool = True) -> dict:
    value = 0.21 if mask else 0.0
    return {
        "features": {
            "observed_value_mask": [[[mask] * 6 for _ in range(8)] for _ in range(16)],
            "steps_since_last_valid_observation": [
                [[stale] * 6 for _ in range(8)] for _ in range(16)
            ],
            "zone_o2_mole_fraction": [[value] * 8 for _ in range(16)],
            "zone_co2_ppm": [[500.0] * 8 for _ in range(16)],
            "zone_temperature_k": [[295.0] * 8 for _ in range(16)],
            "zone_pressure_pa": [[101.0] * 8 for _ in range(16)],
            "zone_relative_humidity": [[0.4] * 8 for _ in range(16)],
            "zone_branch_airflow_m3_s": [[0.05] * 8 for _ in range(16)],
            "requested_command_vector": [[0.5] * 27 for _ in range(16)],
            "achieved_actuator_state": [[0.5] * 27 for _ in range(16)],
            "battery_state_of_charge": [0.9] * 16,
            "oxygen_store_fraction": [0.8] * 16,
            "sorbent_fraction": [0.7] * 16,
            "operating_mode_one_hot": [[1.0, 0.0, 0.0, 0.0] for _ in range(16)],
            "prior_proposal_dispositions": [[1.0, 0.0, 0.0, 0.0] for _ in range(16)],
            "candidate_action_command_vector": [0.5] * 27,
            "candidate_action_index": 0,
        }
    }


def _layer() -> CalibrationLayer:
    return CalibrationLayer(
        delta_offsets=np.zeros((3, 5)),
        trajectory_offsets=np.zeros((3, 153)),
        max_staleness=0.0,
        min_mask_fraction=1.0,
        seed_disagreement=0.1,
        interval_width=10.0,
        digest="x" * 64,
    )


def _pooled_row() -> np.ndarray:
    row = np.zeros(632)
    row[617:632] = np.tile([-1.0, 0.0, 1.0], 5)
    return row


def test_conformal_offsets_and_coverage_synthetic() -> None:
    rng = np.random.default_rng(4)
    n = 600
    truth_delta = rng.normal(0.0, 1.0, (n, 5))
    truth_traj = rng.normal(0.0, 1.0, (n, 153))
    pooled = np.zeros((n, 632))
    noise = rng.normal(0.0, 1.0, (n, 5))
    for slot, level in enumerate((0.1, 0.5, 0.9)):
        q = np.quantile(noise, level, axis=0)
        pooled[:, 617 + slot * 5 : 617 + (slot + 1) * 5] = q
    traj_noise = rng.normal(0.0, 1.0, (n, 153))
    for slot, level in enumerate((0.1, 0.5, 0.9)):
        q = np.quantile(traj_noise, level, axis=0)
        pooled[:, 153 + slot * 153 : 153 + (slot + 1) * 153] = q
    truth_delta = noise
    truth_traj = traj_noise
    delta_offsets, traj_offsets = fit_conformal_offsets(pooled, truth_traj, truth_delta)
    assert delta_offsets.shape == (3, 5)
    assert traj_offsets.shape == (3, 153)
    layer = CalibrationLayer(
        delta_offsets, traj_offsets, 0.0, 1.0, 1.0, 1e9, layer_digest(delta_offsets, traj_offsets, {})
    )
    coverage = interval_coverage(layer, pooled, truth_delta)
    assert 0.75 <= coverage <= 0.85
    for index in range(3):
        traj_coverage = trajectory_interval_coverage(layer, pooled, truth_traj, index)
        assert 0.75 <= traj_coverage <= 0.85


def test_abstention_reason_priority_and_triggers() -> None:
    layer = _layer()
    row = _pooled_row()
    seeds = np.zeros((5, 5))
    healthy = _sample()
    assert abstention_reason(healthy, seeds, layer, row) is None
    stale = _sample(stale=3)
    assert abstention_reason(stale, seeds, layer, row) == "STALE_OBSERVATIONS"
    unmasked = _sample(mask=False)
    assert abstention_reason(unmasked, seeds, layer, row) == "STALE_OBSERVATIONS"
    disagreeing = seeds.copy()
    disagreeing[:, 1] = np.linspace(0.0, 1.0, 5)
    assert abstention_reason(healthy, disagreeing, layer, row) == "SEED_DISAGREEMENT"
    narrow = _layer()
    narrow_interval = CalibrationLayer(
        narrow.delta_offsets,
        narrow.trajectory_offsets,
        narrow.max_staleness,
        narrow.min_mask_fraction,
        narrow.seed_disagreement,
        0.001,
        narrow.digest,
    )
    assert abstention_reason(healthy, seeds, narrow_interval, row) == "WIDE_INTERVAL"
    bad = _sample()
    bad["features"]["zone_o2_mole_fraction"][0][0] = float("nan")
    assert abstention_reason(bad, seeds, layer, row) == "INVALID_INPUT"


def test_predict_with_abstention_returns_none_and_reason() -> None:
    layer = _layer()
    row = _pooled_row()
    seeds = np.zeros((5, 5))
    prediction, reason = predict_with_abstention(_sample(stale=2), seeds, layer, row)
    assert prediction is None
    assert reason == "STALE_OBSERVATIONS"
    prediction, reason = predict_with_abstention(_sample(), seeds, layer, row)
    assert reason is None
    assert prediction is not None and prediction.shape == (5,)


def test_staleness_stats() -> None:
    max_stale, mask_fraction = sample_staleness_stats(_sample(stale=4))
    assert max_stale == 4.0
    assert mask_fraction == 1.0
    max_stale, mask_fraction = sample_staleness_stats(_sample(mask=False))
    assert mask_fraction == 0.0


def test_risk_coverage_curve_and_recall_and_guard() -> None:
    losses = [0.1, 0.2, 0.3, 0.4]
    widths = [1.0, 2.0, 3.0, 4.0]
    curve = risk_coverage_curve(losses, widths, (0.5, 1.0, 2.0), 2.0)
    assert curve[0]["coverage"] == 0.25
    assert curve[1]["coverage"] == 0.5
    assert curve[2]["coverage"] == 1.0
    assert curve[1]["risk"] == pytest.approx(0.15)
    recall, abstained_useful = useful_opportunity_recall(
        [True, True, False], [True, True, True], [False, True, False]
    )
    assert recall == pytest.approx(1 / 3)
    assert abstained_useful == pytest.approx(1 / 3)
    assert abstention_guard(0.4, 0.5) is True
    assert abstention_guard(0.6, 0.5) is False


def test_reason_vocabulary_is_declared() -> None:
    assert ABSTENTION_REASONS == (
        "INVALID_INPUT",
        "STALE_OBSERVATIONS",
        "SEED_DISAGREEMENT",
        "WIDE_INTERVAL",
    )


def test_module_has_no_plant_or_hmc_authority() -> None:
    from aeolus.habitat_v2 import forecast_issue76_abstention as module

    source = inspect.getsource(module)
    assert "HabitatManagementComputer" not in source
    assert "advance_one_step_with_command" not in source
    assert "from .hmc" not in source
    assert "from .physics" not in source


def test_malformed_pooled_outputs_rejected() -> None:
    bad = np.full((4, 632), float("nan"))
    with pytest.raises(Issue76AbstentionError, match="non-finite"):
        fit_conformal_offsets(bad, np.zeros((4, 153)), np.zeros((4, 5)))
