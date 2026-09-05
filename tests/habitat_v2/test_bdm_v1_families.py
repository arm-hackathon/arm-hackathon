from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.bdm_v1_benchmark_contract import validate_group_disjointness
from aeolus.habitat_v2.bdm_v1_families import (
    BDM_V1_PARTITIONS_ORDERED,
    DECISION_STEPS,
    EPISODE_STEPS,
    GENERATOR_FAULT_MULTIPLIER_BANDS,
    GENERATOR_INITIAL_O2_BANDS,
    GENERATOR_O2_EXCESS_THRESHOLD,
    GENERATOR_RAMP_SHAPES,
    GENERATOR_SENSOR_BIAS_BANDS,
    SENSOR_DEFECT_TYPES,
    STRATA_ORDERED,
    TEMPLATES,
    BdmV1FamilyError,
    FamilyDefinition,
    GeneratorConfig,
    assign_partitions,
    build_family,
    build_family_data,
    build_matched_pair,
    family_partition_groups,
    generate_families,
    load_base_scenario_data,
)
from aeolus.habitat_v2.physics_provenance import (
    load_physics_provenance_manifest,
    parameter_by_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SMALL_CONFIG = GeneratorConfig(
    groups_per_partition={"TRAIN": 12, "DEV": 6, "CALIBRATION": 3, "BLIND_FINAL": 3}
)
ALL_TEMPLATE_CONFIG = GeneratorConfig(
    groups_per_partition={"TRAIN": 12, "DEV": 12, "CALIBRATION": 0, "BLIND_FINAL": 0}
)


@pytest.fixture(scope="module")
def base_data() -> dict:
    data, digest = load_base_scenario_data(REPO_ROOT)
    assert len(digest) == 64
    return data


@pytest.fixture(scope="module")
def manifest() -> dict:
    loaded, digest = load_physics_provenance_manifest(REPO_ROOT)
    assert len(digest) == 64
    return loaded


@pytest.fixture(scope="module")
def small_families(base_data, manifest) -> tuple[FamilyDefinition, ...]:
    return generate_families(SMALL_CONFIG, base_data, manifest)


def test_decision_steps_follow_house_convention() -> None:
    assert DECISION_STEPS == tuple(range(16, 65, 4))
    assert len(DECISION_STEPS) == 13
    assert EPISODE_STEPS == 96


def test_regeneration_is_byte_identical(base_data, manifest) -> None:
    plans = assign_partitions(SMALL_CONFIG)
    plan = plans[3]
    first, scenario_first = build_family(SMALL_CONFIG, plan, 0, 0, base_data, manifest)
    second, scenario_second = build_family(SMALL_CONFIG, plan, 0, 0, base_data, manifest)
    assert first.scenario_sha256 == second.scenario_sha256
    assert json.dumps(scenario_first.data, sort_keys=True) == json.dumps(
        scenario_second.data, sort_keys=True
    )
    assert first.to_mapping() == second.to_mapping()


def test_seed_and_attempt_change_draws(base_data, manifest) -> None:
    plans = assign_partitions(SMALL_CONFIG)
    plan = plans[3]
    baseline, _ = build_family(SMALL_CONFIG, plan, 0, 0, base_data, manifest)
    reseeded_config = GeneratorConfig(
        seed="another-seed",
        groups_per_partition=dict(SMALL_CONFIG.groups_per_partition),
    )
    reseeded_plans = assign_partitions(reseeded_config)
    reseeded_plan = next(p for p in reseeded_plans if p.group_index == plan.group_index)
    reseeded, _ = build_family(reseeded_config, reseeded_plan, 0, 0, base_data, manifest)
    assert reseeded.scenario_sha256 != baseline.scenario_sha256

    attempt_one, _ = build_family(SMALL_CONFIG, plan, 0, 1, base_data, manifest)
    assert attempt_one.scenario_sha256 != baseline.scenario_sha256
    assert attempt_one.parameter_draws != baseline.parameter_draws


def test_paired_variants_share_causal_state(base_data, manifest) -> None:
    plans = assign_partitions(SMALL_CONFIG)
    plan = plans[5]
    variant_a, scenario_a = build_family(SMALL_CONFIG, plan, 0, 0, base_data, manifest)
    variant_b, scenario_b = build_family(SMALL_CONFIG, plan, 1, 0, base_data, manifest)
    assert variant_a.group_id == variant_b.group_id
    assert variant_a.group_key == variant_b.group_key
    assert variant_a.parameter_draws == variant_b.parameter_draws
    assert variant_a.sensor_seed != variant_b.sensor_seed
    assert variant_a.scenario_sha256 != variant_b.scenario_sha256
    data_a = scenario_a.data
    data_b = scenario_b.data
    differing = {
        key for key in data_a if json.dumps(data_a[key], sort_keys=True) != json.dumps(data_b[key], sort_keys=True)
    }
    assert differing == {"sensor_model", "name"}


def test_matched_pair_diff_is_declared_treatment_only(base_data, manifest) -> None:
    plans = assign_partitions(SMALL_CONFIG)
    plan = next(p for p in plans if p.template.fault_kinds)
    fault_data, healthy_data, treatment_paths = build_matched_pair(
        SMALL_CONFIG, plan, 0, 0, base_data, manifest
    )
    assert treatment_paths == ("fault_profiles",)
    differing = {
        key
        for key in fault_data
        if json.dumps(fault_data[key], sort_keys=True)
        != json.dumps(healthy_data[key], sort_keys=True)
    }
    assert differing == set(treatment_paths)
    assert fault_data["fault_profiles"]
    assert healthy_data["fault_profiles"] == []


def test_draws_stay_inside_declared_bands(small_families, manifest) -> None:
    for family in small_families:
        for pid, value in family.parameter_draws.items():
            record = parameter_by_id(manifest, pid)
            band = record["valid_range"]
            if band is not None:
                assert band[0] <= value <= band[1], (family.family_id, pid)
            distribution = record["uncertainty_distribution"]
            assert distribution is not None
            if distribution["kind"] == "uniform_relative":
                low = record["value"] * (1.0 + distribution["low"])
                high = record["value"] * (1.0 + distribution["high"])
            else:
                low, high = distribution["low"], distribution["high"]
            assert low - 1e-9 <= value <= high + 1e-9, (family.family_id, pid)


def test_small_partition_deal_is_exact_and_disjoint(small_families) -> None:
    counts = {partition: 0 for partition in BDM_V1_PARTITIONS_ORDERED}
    for family in small_families:
        counts[family.partition] += 1
    assert counts == {"TRAIN": 24, "DEV": 12, "CALIBRATION": 6, "BLIND_FINAL": 6}
    groups = family_partition_groups(small_families)
    validate_group_disjointness(groups)
    assert sum(len(ids) for ids in groups.values()) == 24


def test_full_sizing_deal_covers_every_stratum() -> None:
    config = GeneratorConfig()
    plans = assign_partitions(config)
    assert len(plans) == 80
    counts = {partition: 0 for partition in BDM_V1_PARTITIONS_ORDERED}
    for plan in plans:
        counts[plan.partition] += 1
    assert counts == {"TRAIN": 40, "DEV": 16, "CALIBRATION": 12, "BLIND_FINAL": 12}
    coverage = {
        partition: {plan.template.stratum for plan in plans if plan.partition == partition}
        for partition in BDM_V1_PARTITIONS_ORDERED
    }
    for partition, strata in coverage.items():
        assert strata == set(STRATA_ORDERED), partition
    train_templates = {plan.template.template_id for plan in plans if plan.partition == "TRAIN"}
    assert len(train_templates) == len(TEMPLATES)


def test_template_stratum_and_opportunity_vocabulary(base_data, manifest) -> None:
    families = generate_families(ALL_TEMPLATE_CONFIG, base_data, manifest)
    assert len(families) == 48
    seen_templates = {family.template_id for family in families}
    assert seen_templates == {template.template_id for template in TEMPLATES}
    allowed_opportunities = {
        "no_opportunity",
        "o2_excess_from_start",
        "fault_early",
        "fault_in_window",
        "fault_late",
    }
    for family in families:
        assert family.stratum in STRATA_ORDERED
        assert family.group_key["action_opportunity"] in allowed_opportunities
        assert family.group_key["causal_scenario_template"] == family.template_id
        if family.template_id == "t01_healthy_steady":
            assert family.group_key["action_opportunity"] == "no_opportunity"
        if family.template_id == "t02_o2_regime_stress":
            assert family.group_key["action_opportunity"] == "o2_excess_from_start"


def test_fault_windows_shapes_and_bands(small_families) -> None:
    saw_shape = set()
    for family in small_families:
        for mechanism in family.generator_draws["fault_mechanisms"]:
            assert mechanism["ramp_shape"] in GENERATOR_RAMP_SHAPES
            saw_shape.add(mechanism["ramp_shape"])
        for profile in family.generator_draws["fault_profiles"]:
            start = int(profile["start_step"])
            end = int(profile["end_step"])
            assert 8 <= start < end <= EPISODE_STEPS - 1
            if "start_multiplier" in profile:
                low, high = GENERATOR_FAULT_MULTIPLIER_BANDS[profile["type"]]
                for key in ("start_multiplier", "end_multiplier"):
                    value = float(profile[key])
                    assert min(1.0, low) - 1e-9 <= value <= max(1.0, high) + 1e-9
    assert saw_shape == set(GENERATOR_RAMP_SHAPES) or len(saw_shape) >= 2


def test_sensor_defect_vocabulary_and_windows(small_families) -> None:
    for family in small_families:
        profiles = family.generator_draws["sensor_defect_profiles"]
        bundle = family.group_key["sensor_failure_bundle"]
        if bundle == "none":
            assert profiles == []
            continue
        for profile in profiles:
            assert profile["type"] in SENSOR_DEFECT_TYPES
            assert 0 <= int(profile["start_step"]) < int(profile["end_step"]) <= EPISODE_STEPS - 1
            if profile["type"] == "sensor_bias_drift":
                low, high = GENERATOR_SENSOR_BIAS_BANDS[profile["channel"]]
                for key in ("start_bias", "end_bias"):
                    assert abs(float(profile[key])) <= high + 1e-12
                    assert abs(float(profile[key])) >= 0.0
                assert low > 0.0


def test_initial_oxygen_regimes(base_data, manifest) -> None:
    plans = assign_partitions(ALL_TEMPLATE_CONFIG)
    for plan in plans:
        definition, _ = build_family(ALL_TEMPLATE_CONFIG, plan, 0, 0, base_data, manifest)
        o2 = float(definition.generator_draws["initial_o2_mole_fraction"])
        regime = plan.template.o2_regime
        low, high = GENERATOR_INITIAL_O2_BANDS[regime]
        assert low - 1e-12 <= o2 <= high + 1e-12
        if regime == "high":
            assert o2 >= GENERATOR_O2_EXCESS_THRESHOLD
        if regime == "low":
            assert o2 < GENERATOR_O2_EXCESS_THRESHOLD


def test_noise_amplitudes_stay_hmc_pinned(base_data, manifest) -> None:
    plans = assign_partitions(SMALL_CONFIG)
    for plan in plans[:6]:
        data, _, _ = build_family_data(SMALL_CONFIG, plan, 0, 0, base_data, manifest)
        assert data["sensor_model"]["primary_noise_amplitude"] == base_data[
            "sensor_model"
        ]["primary_noise_amplitude"]
        assert data["sensor_model"]["secondary_noise_amplitude"] == base_data[
            "sensor_model"
        ]["secondary_noise_amplitude"]


def test_transition_schedule_is_explicit(base_data, manifest) -> None:
    plans = assign_partitions(ALL_TEMPLATE_CONFIG)
    plan = next(p for p in plans if p.template.template_id == "t12_schedule_transition")
    data, group_key, draws = build_family_data(
        ALL_TEMPLATE_CONFIG, plan, 0, 0, base_data, manifest
    )
    assert group_key["operating_schedule"] == "transition"
    segments = data["timeline"]
    assert len(segments) == 3
    cursor = 0
    modes = []
    for segment in segments:
        assert int(segment["start_step"]) == cursor
        assert int(segment["end_step"]) == cursor + 32
        modes.append(segment["operating_mode"])
        cursor = int(segment["end_step"])
    assert cursor == EPISODE_STEPS
    assert modes == ["occupied", "eva_transition", "contingency"]
    occupied_loads = segments[0]["loads"]["crew_quarters_a"]
    eva_loads = segments[1]["loads"]["crew_quarters_a"]
    contingency_loads = segments[2]["loads"]["crew_quarters_a"]
    assert eva_loads["o2_consumption_mol_s"] < occupied_loads["o2_consumption_mol_s"]
    assert contingency_loads["o2_consumption_mol_s"] > occupied_loads["o2_consumption_mol_s"]


def test_low_reserve_regime_draws_below_nominal(base_data, manifest) -> None:
    plans = assign_partitions(ALL_TEMPLATE_CONFIG)
    plan = next(p for p in plans if p.template.template_id == "t11_low_reserve_depletion")
    definition, _ = build_family(ALL_TEMPLATE_CONFIG, plan, 0, 0, base_data, manifest)
    draws = definition.parameter_draws
    nominal_battery = float(base_data["initial_utility"]["battery_energy_wh"])
    nominal_store = float(base_data["initial_utility"]["oxygen_store_mol"])
    nominal_sorbent = float(base_data["initial_utility"]["co2_sorbent_remaining_mol"])
    assert draws["initial_battery_energy_wh"] <= nominal_battery
    assert draws["initial_oxygen_store_mol"] <= nominal_store
    assert draws["initial_co2_sorbent_mol"] <= nominal_sorbent


def test_family_definition_mapping(base_data, manifest) -> None:
    plans = assign_partitions(SMALL_CONFIG)
    definition, _ = build_family(SMALL_CONFIG, plans[0], 0, 0, base_data, manifest)
    mapping = definition.to_mapping()
    assert set(mapping) >= {
        "family_id",
        "group_id",
        "partition",
        "sensor_variant",
        "template_id",
        "stratum",
        "attempt",
        "group_key",
        "parameter_draws",
        "generator_draws",
        "sensor_seed",
        "scenario_sha256",
    }
    json.dumps(mapping, sort_keys=True)
    assert len(mapping["scenario_sha256"]) == 64
    rebuilt, _ = build_family(SMALL_CONFIG, plans[0], 0, 0, base_data, manifest)
    assert rebuilt.to_mapping() == mapping


def test_matched_pair_requires_fault_template(base_data, manifest) -> None:
    plans = assign_partitions(ALL_TEMPLATE_CONFIG)
    healthy_plan = next(p for p in plans if not p.template.fault_kinds)
    with pytest.raises(BdmV1FamilyError, match="matched pair"):
        build_matched_pair(ALL_TEMPLATE_CONFIG, healthy_plan, 0, 0, base_data, manifest)


def test_hmc_reset_accepts_every_template(base_data, manifest) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.hmc import HabitatManagementComputer

    bundle = load_forecast_contracts(REPO_ROOT)
    plans = assign_partitions(ALL_TEMPLATE_CONFIG)
    seen = set()
    for plan in plans:
        if plan.template.template_id in seen:
            continue
        seen.add(plan.template.template_id)
        definition, scenario = build_family(
            ALL_TEMPLATE_CONFIG, plan, 0, 0, base_data, manifest
        )
        nonce = hashlib.sha256(
            b"bdm-v1-test-nonce|" + definition.family_id.encode("utf-8")
        ).digest()
        HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    assert seen == {template.template_id for template in TEMPLATES}
