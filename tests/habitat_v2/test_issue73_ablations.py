from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.bdm_v1_evaluation import (
    BdmV1EvaluationError,
    bootstrap_ci,
    comparison_table,
    group_means,
    paired_group_differences,
)
from aeolus.habitat_v2.bdm_v1_families import (
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast_issue73_ablations import (
    ABLATION_ARMS,
    AblationContext,
    Issue73AblationError,
    fit_linear_screen,
    fit_mlp_screen,
    guard_proposal,
    learned_proposal,
    run_ablation_episode,
    screen_feature_vector,
)
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_ablation_preregistration_v1.json"


def _rows(groups: dict[str, list[float]], arm: str) -> list[dict[str, object]]:
    rows = []
    for group, values in groups.items():
        for value in values:
            rows.append({"group_id": group, "arm": arm, "safety_exposure": value})
    return rows


def test_group_means_and_paired_differences() -> None:
    rows_a = _rows({"g1": [1.0, 3.0], "g2": [2.0]}, "a")
    rows_b = _rows({"g1": [0.0, 1.0], "g2": [4.0]}, "b")
    assert group_means(rows_a, "safety_exposure") == {"g1": 2.0, "g2": 2.0}
    diffs = paired_group_differences(rows_a, rows_b, "safety_exposure")
    assert diffs == {"g1": 1.5, "g2": -2.0}
    with pytest.raises(BdmV1EvaluationError, match="identical group sets"):
        paired_group_differences(rows_a, _rows({"g1": [0.0]}, "b"), "safety_exposure")


def test_bootstrap_ci_is_deterministic_and_sane() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    first = bootstrap_ci(values, seed="seed-a")
    second = bootstrap_ci(values, seed="seed-a")
    assert first == second
    mean, low, high = first
    assert mean == pytest.approx(3.5)
    assert low <= mean <= high
    with pytest.raises(BdmV1EvaluationError, match="at least two"):
        bootstrap_ci([1.0], seed="seed-a")
    with pytest.raises(BdmV1EvaluationError, match="100 resamples"):
        bootstrap_ci(values, seed="seed-a", resamples=10)


def test_comparison_table_baseline_and_coverage() -> None:
    arms = {
        "hold": _rows({"g1": [1.0, 1.0], "g2": [1.0, 1.0]}, "hold"),
        "arm": _rows({"g1": [0.0, 0.0], "g2": [0.0, 0.0]}, "arm"),
    }
    table = comparison_table(arms, "safety_exposure", baseline_arm="hold", seed="s")
    assert table["arms"]["arm"]["paired_difference_mean"] == pytest.approx(-1.0)
    with pytest.raises(BdmV1EvaluationError, match="baseline arm"):
        comparison_table(arms, "safety_exposure", baseline_arm="missing", seed="s")
    arms["partial"] = _rows({"g1": [0.5, 0.5]}, "partial")
    with pytest.raises(BdmV1EvaluationError, match="baseline groups"):
        comparison_table(arms, "safety_exposure", baseline_arm="hold", seed="s")


def test_guard_proposal_table() -> None:
    excess = AblationContext(True, False, 0.30, (1.0, 0.0, 0.0, 0.0))
    nominal = AblationContext(False, False, 0.22, (1.0, 0.0, 0.0, 0.0))
    critical = AblationContext(False, True, 0.10, (1.0, 0.0, 0.0, 0.0))
    assert guard_proposal(excess, "normal-occupied-v1") == "normal-dormant-v1"
    assert guard_proposal(excess, "normal-dormant-v1") is None
    assert guard_proposal(nominal, "normal-occupied-v1") is None
    assert guard_proposal(critical, "normal-occupied-v1") is None


def test_screen_feature_vector_guards() -> None:
    vector = screen_feature_vector([0.2] * 8, [500.0] * 8, [295.0] * 8, (1.0, 0.0, 0.0, 0.0), 2)
    assert vector.shape == (33,)
    with pytest.raises(Issue73AblationError, match="eight zones"):
        screen_feature_vector([0.2] * 7, [500.0] * 8, [295.0] * 8, (1.0, 0.0, 0.0, 0.0), 0)
    with pytest.raises(Issue73AblationError, match="one-hot"):
        screen_feature_vector([0.2] * 8, [500.0] * 8, [295.0] * 8, (0.5, 0.5, 0.0, 0.0), 0)
    with pytest.raises(Issue73AblationError, match="catalogue action index"):
        screen_feature_vector([0.2] * 8, [500.0] * 8, [295.0] * 8, (1.0, 0.0, 0.0, 0.0), 4)


def _synthetic_screen_data():
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (200, 33))
    truth = np.asarray([0.5, -0.25] + [0.0] * 31)
    targets = features @ truth + 0.01 * rng.normal(0.0, 1.0, 200)
    return features, targets, truth


def test_linear_screen_fit_is_deterministic_and_sign_correct() -> None:
    features, targets, truth = _synthetic_screen_data()
    first = fit_linear_screen(features, targets, ridge_lambda=1e-3, seed="s")
    second = fit_linear_screen(features, targets, ridge_lambda=1e-3, seed="s")
    assert first.digest == second.digest
    probe = np.zeros(33)
    probe[0] = 1.0
    positive = first.predict(probe)
    probe[0] = 0.0
    probe[1] = 1.0
    negative = first.predict(probe)
    assert positive > 0.0 > negative
    assert abs(first.weights[0] - truth[0]) < 0.2


def test_mlp_screen_fit_is_deterministic_and_improves() -> None:
    features, targets, _truth = _synthetic_screen_data()
    first = fit_mlp_screen(features, targets, seed="s", epochs=40, learning_rate=3e-3)
    second = fit_mlp_screen(features, targets, seed="s", epochs=40, learning_rate=3e-3)
    assert first.digest == second.digest
    predictions = np.asarray([first.predict(row) for row in features])
    residual = float(np.mean((predictions - targets) ** 2))
    assert residual < float(np.var(targets))


def test_learned_proposal_rules() -> None:
    features, targets, _truth = _synthetic_screen_data()
    screen = fit_linear_screen(features, targets, ridge_lambda=1e-3, seed="s")
    snapshot = {
        "primary_telemetry": {
            "samples": [
                {
                    "descriptor_id": f"zone{zone}/{channel}",
                    "availability": "AVAILABLE",
                    "value": 0.21,
                    "unavailable_reason": None,
                    "unit": "x",
                }
                for zone in range(8)
                for channel in ("o2_mole_fraction", "co2_ppm", "temperature_k")
            ]
        },
        "completed_operating_mode": "occupied",
    }
    zone_ids = [f"zone{zone}" for zone in range(8)]
    mode_order = ("occupied", "eva_transition", "contingency", "dormant")
    action_ids = [
        "normal-occupied-v1",
        "normal-eva_transition-v1",
        "normal-contingency-v1",
        "normal-dormant-v1",
    ]
    nominal = AblationContext(False, False, 0.21, (1.0, 0.0, 0.0, 0.0))
    chosen, predictions = learned_proposal(
        screen, snapshot, zone_ids, mode_order, action_ids, nominal, "normal-occupied-v1",
        include_guard=False,
    )
    assert chosen is None or chosen in action_ids
    assert len(predictions) == 4
    critical = AblationContext(False, True, 0.10, (1.0, 0.0, 0.0, 0.0))
    assert learned_proposal(
        screen, snapshot, zone_ids, mode_order, action_ids, critical, "normal-occupied-v1",
        include_guard=False,
    )[0] is None
    dormant_now = learned_proposal(
        screen, snapshot, zone_ids, mode_order, action_ids, nominal, "normal-dormant-v1",
        include_guard=False,
    )[0]
    assert dormant_now is None
    excess = AblationContext(True, False, 0.30, (1.0, 0.0, 0.0, 0.0))
    guarded = learned_proposal(
        screen, snapshot, zone_ids, mode_order, action_ids, excess, "normal-occupied-v1",
        include_guard=True,
    )[0]
    assert guarded == "normal-dormant-v1"


def test_preregistration_is_frozen_and_complete() -> None:
    prereg = json.loads(PREREG_PATH.read_bytes())
    assert prereg["freeze"]["frozen_before_study_runs"] is True
    assert tuple(item["arm_id"] for item in prereg["arms"]) == ABLATION_ARMS
    assert prereg["partitions"]["study"] == "DEV"
    assert prereg["partitions"]["blind_access"] == "prohibited"
    assert prereg["statistics"]["bootstrap_resamples"] == 10000


def test_episode_runner_smoke_two_arms() -> None:
    bundle = load_forecast_contracts(REPO_ROOT)
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    config = GeneratorConfig(
        groups_per_partition={"TRAIN": 1, "DEV": 0, "CALIBRATION": 0, "BLIND_FINAL": 0}
    )
    plans = assign_partitions(config)
    definition, scenario = build_family(config, plans[0], 0, 0, base_data, manifest)
    records = {}
    for arm in ("hmc_rules_only", "c8_guard_only"):
        record = run_ablation_episode(
            bundle,
            scenario,
            arm,
            definition.family_id,
            decision_steps=(16,),
            scenario_sha256=definition.scenario_sha256,
        )
        records[arm] = record
    assert records["hmc_rules_only"]["scenario_sha256"] == definition.scenario_sha256
    assert records["c8_guard_only"]["scenario_sha256"] == definition.scenario_sha256
    assert records["hmc_rules_only"]["proposal_count"] == 0
    assert records["hmc_rules_only"]["abstention_count"] == 1
    assert records["c8_guard_only"]["proposal_count"] >= 0
    for record in records.values():
        assert record["safety_exposure"] >= 0.0
        assert len(record["episode_sha256"]) == 64
    with pytest.raises(Issue73AblationError, match="requires a fitted screen"):
        run_ablation_episode(bundle, scenario, "c9_full", definition.family_id)
