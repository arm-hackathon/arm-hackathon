from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.bdm_v1_corpus import WINDOW_STEPS
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast_issue73_ablations import LinearScreen
from aeolus.habitat_v2.forecast_issue75_bdm import _assemble, _init_weights
from aeolus.habitat_v2.forecast_issue76_abstention import CalibrationLayer
from aeolus.habitat_v2.forecast_issue77_closed_loop import (
    evaluate_gates,
    make_calibrated_bdm_policy,
    make_guard_screen_policy,
    make_null_policy,
    reconstruct_sample_features,
    run_closed_loop_episode,
)
from aeolus.habitat_v2.bdm_v1_families import (
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "out" / "bdm-v1-corpus-v1"
ENV_CHANNELS = (
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
)


@pytest.fixture(scope="module")
def bundle():
    return load_forecast_contracts(REPO_ROOT)


def _zero_screen() -> LinearScreen:
    return LinearScreen(
        mean=np.zeros(33),
        scale=np.ones(33),
        weights=np.zeros(33),
        bias=0.0,
        ridge_lambda=1e-3,
        digest="0" * 64,
    )


def _dummy_models():
    init = _init_weights("issue77-test")
    init.pop("rng")
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
    return [_assemble(np.zeros(240), np.ones(240), params, "issue77-test")]


def _layer() -> CalibrationLayer:
    return CalibrationLayer(
        delta_offsets=np.zeros((3, 5)),
        trajectory_offsets=np.zeros((3, 153)),
        max_staleness=0.0,
        min_mask_fraction=1.0,
        seed_disagreement=1.0,
        interval_width=1e9,
        digest="1" * 64,
    )


def _synthetic_snapshot(bundle, step: int, o2: float, unavailable: bool = False) -> dict:
    zone_ids = tuple(bundle.topology.zone_ids)
    primary = []
    for zone in zone_ids:
        for channel in ENV_CHANNELS:
            value = o2 if channel == "o2_mole_fraction" else 1.0
            if unavailable and channel == "o2_mole_fraction" and zone == zone_ids[0]:
                primary.append(
                    {
                        "descriptor_id": f"{zone}/{channel}",
                        "availability": "UNAVAILABLE",
                        "value": None,
                        "unavailable_reason": "MISSING",
                        "unit": "mole_fraction",
                    }
                )
            else:
                primary.append(
                    {
                        "descriptor_id": f"{zone}/{channel}",
                        "availability": "AVAILABLE",
                        "value": value,
                        "unavailable_reason": None,
                        "unit": "x",
                    }
                )
    feedback = []
    for zone in zone_ids:
        feedback.append(
            {
                "descriptor_id": f"branch_airflow_m3_s/{zone}",
                "availability": "AVAILABLE",
                "value": 0.05,
                "unavailable_reason": None,
                "unit": "m3_s",
            }
        )
    achieved_names = (
        ["fan_speed_fraction"]
        + [f"damper_position_by_id/{pair[1]}" for pair in bundle.topology.branch_pairs]
        + ["scrubber_capture_rate_mol_s", "condenser_removal_rate_mol_s"]
        + [f"cooling_delivery_w/{zone}" for zone in zone_ids]
        + [f"oxygen_delivery_mol_s/{zone}" for zone in zone_ids]
    )
    for name in achieved_names:
        feedback.append(
            {
                "descriptor_id": name,
                "availability": "AVAILABLE",
                "value": 0.5,
                "unavailable_reason": None,
                "unit": "x",
            }
        )
    gauges = [
        {
            "descriptor_id": name,
            "availability": "AVAILABLE",
            "value": 0.8,
            "unavailable_reason": None,
            "unit": "fraction",
        }
        for name in ("battery_state_of_charge", "oxygen_store_fraction", "sorbent_remaining_fraction")
    ]
    return {
        "completed_step": step,
        "completed_operating_mode": "occupied",
        "observable_topology_sha256": "2" * 64,
        "primary_telemetry": {"samples": primary},
        "operational_feedback": {"samples": feedback},
        "operational_resource_gauges": {"samples": gauges},
        "command_reference": {
            "command": bundle.actions[0].command.to_mapping(),
            "command_reference_kind": "TEST",
            "source_kind": "test",
        },
        "proposal_disposition": "NO_PROPOSAL",
    }


def _window(bundle, o2: float, unavailable: bool = False) -> list[dict]:
    return [
        _synthetic_snapshot(bundle, step, o2, unavailable)
        for step in range(1, WINDOW_STEPS + 1)
    ]


def test_null_policy_abstains() -> None:
    policy = make_null_policy()
    assert policy({}, "normal-occupied-v1") == (None, None, 0.0)


def test_guard_screen_policy_proposes_dormant_on_excess(bundle) -> None:
    policy = make_guard_screen_policy(_zero_screen(), include_guard=False)
    snapshot = _synthetic_snapshot(bundle, 16, 0.30)
    enriched = dict(snapshot)
    enriched["__zone_ids"] = tuple(bundle.topology.zone_ids)
    enriched["__actions"] = tuple(bundle.actions)
    chosen, reason, _latency = policy(enriched, "normal-occupied-v1")
    assert chosen == "normal-dormant-v1"
    assert reason is None
    nominal = dict(_synthetic_snapshot(bundle, 16, 0.21))
    nominal["__zone_ids"] = tuple(bundle.topology.zone_ids)
    nominal["__actions"] = tuple(bundle.actions)
    chosen, _reason, _latency = policy(nominal, "normal-occupied-v1")
    assert chosen is None


def test_calibrated_policy_abstains_on_stale_observation(bundle) -> None:
    policy = make_calibrated_bdm_policy(_dummy_models(), _layer())
    window = _window(bundle, 0.21, unavailable=True)
    snapshot = window[-1]
    enriched = dict(snapshot)
    enriched["__zone_ids"] = tuple(bundle.topology.zone_ids)
    enriched["__actions"] = tuple(bundle.actions)
    enriched["__bundle"] = bundle
    enriched["__episode_snapshots"] = window
    chosen, reason, latency = policy(enriched, "normal-occupied-v1")
    assert chosen is None
    assert reason == "STALE_OBSERVATIONS"
    assert latency >= 0.0


def test_reconstruct_matches_corpus_features(bundle) -> None:
    if not CORPUS.is_dir():
        pytest.skip("local corpus artifact not present")
    target = None
    for shard in sorted((CORPUS / "shards").glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line:
                continue
            sample = json.loads(line)
            if sample["partition"] == "DEV" and sample["decision_step"] == 16 and sample["features"]["candidate_action_index"] == 0:
                target = sample
                break
        if target:
            break
    assert target is not None
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    config = GeneratorConfig()
    plans = {plan.group_index: plan for plan in assign_partitions(config)}
    family_index = int(target["family_id"][8:12])
    variant_index = 0 if target["family_id"].endswith("-a") else 1
    definition, scenario = build_family(
        config, plans[family_index // 2], variant_index, 0, base_data, manifest
    )
    assert definition.family_id == target["family_id"]
    nonce = b"reconstruct-test" + b"0" * 16
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    snapshots = []
    for step in range(17):
        snapshot, verification = hmc.observe()
        handle = hmc.verify_snapshot(snapshot, verification)
        hmc.propose(None, handle)
        hmc.arbitrate()
        hmc.step()
        mapping = snapshot.to_mapping()
        mapping["proposal_disposition"] = "NO_PROPOSAL"
        snapshots.append(mapping)
    command_vector = [
        float(value)
        for value in target["features"]["candidate_action_command_vector"]
    ]
    features = reconstruct_sample_features(
        bundle, snapshots, 16, 0, command_vector
    )
    for key, expected in target["features"].items():
        produced = features[key]
        if isinstance(expected, list) and expected and isinstance(expected[0], (int, float)) and not isinstance(expected[0], bool):
            assert np.allclose(np.asarray(produced, dtype=np.float64), np.asarray(expected, dtype=np.float64), atol=1e-6), key
        else:
            assert produced == expected, key


def test_evaluate_gates_synthetic_failure_and_pass() -> None:
    prereg = {
        "gates": {
            "alpha": 0.05,
            "statistics_seed": "test",
            "bootstrap_resamples": 200,
            "non_inferiority_margin": 0.0,
        }
    }

    def record(family, group, exposure, proposals=0):
        return {
            "family_id": family,
            "group_id": group,
            "safety_exposure": exposure,
            "safety_violation_steps": 0,
            "comfort_deviation": 0.0,
            "resource_composite": 0.0,
            "proposal_count": proposals,
        }

    failing = {
        "calibrated_bdm_v1": [record("f1", "g1", 0.0), record("f2", "g2", 1.0)],
        "hold_current_command": [record("f1", "g1", 0.0), record("f2", "g2", 0.0)],
        "linear_action_conditioned_ridge": [record("f1", "g1", 0.0), record("f2", "g2", 0.0)],
    }
    gates = evaluate_gates(failing, prereg)
    assert gates["components"]["per_family_safety_non_inferiority"] is False
    assert gates["all_gates_passed"] is False

    passing = {
        "calibrated_bdm_v1": [record("f1", "g1", 0.0, 1), record("f2", "g2", 0.0, 1)],
        "hold_current_command": [record("f1", "g1", 1.0), record("f2", "g2", 1.0)],
        "linear_action_conditioned_ridge": [record("f1", "g1", 1.0), record("f2", "g2", 1.0)],
    }
    gates = evaluate_gates(passing, prereg)
    assert gates["components"]["zero_hard_safety_violations_all_arms"] is True
    assert gates["components"]["per_family_safety_non_inferiority"] is True
    assert gates["components"]["paired_safety_benefit_group_ci"] is True
    assert gates["components"]["ranking_and_regret_vs_linear"] is True
    assert gates["components"]["resource_comfort_group_ci_after_safety"] == "evaluated"
    assert gates["all_gates_passed"] is True


def test_lineage_completeness_smoke(bundle) -> None:
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    config = GeneratorConfig(
        groups_per_partition={"TRAIN": 1, "DEV": 0, "CALIBRATION": 0, "BLIND_FINAL": 0}
    )
    plans = assign_partitions(config)
    definition, scenario = build_family(config, plans[0], 0, 0, base_data, manifest)
    record = run_closed_loop_episode(
        bundle, scenario, make_null_policy(), definition.family_id, definition.scenario_sha256, (16,)
    )
    assert len(record["lineage"]) == 1
    assert len(record["episode_sha256"]) == 64
    assert record["abstention_count"] == 1
    assert record["safety_exposure"] >= 0.0
