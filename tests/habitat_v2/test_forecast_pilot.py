from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_approved_design_enumerates_closed_matched_continuation_plan() -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )

    design = load_approved_pilot_design(ROOT)
    plan = tuple(iter_pilot_continuations(design))

    assert len(design.clusters) == 60
    assert design.operating_modes == (
        "occupied",
        "eva_transition",
        "contingency",
        "dormant",
    )
    assert design.load_regimes == ("LOW", "NOMINAL", "HIGH")
    assert design.anchor_completed_steps == (16, 40, 64)
    assert len(plan) == 23_400
    assert len({item.continuation_id for item in plan}) == 23_400
    assert Counter(item.variant for item in plan) == {
        "MATCHED_CONTROL": 4_680,
        "ACTION_PROPOSAL": 18_720,
    }
    assert set(Counter(item.pair_id for item in plan).values()) == {5}

    controls = {
        item.continuation_id for item in plan if item.variant == "MATCHED_CONTROL"
    }
    assert len(controls) == 4_680
    assert all(item.matched_control_id in controls for item in plan)
    assert all(
        item.action_id == "NO_PROPOSAL"
        for item in plan
        if item.variant == "MATCHED_CONTROL"
    )
    assert all(
        item.action_id in design.action_ids
        for item in plan
        if item.variant == "ACTION_PROPOSAL"
    )
    assert len({item.noise_seed for item in plan}) == 120
    assert len({item.hmc_reset_nonce_hex for item in plan}) == 360

    required_mode_actions = {
        (mode, action)
        for mode in design.operating_modes
        for action in design.action_ids
    }
    actual_mode_actions = {
        (item.operating_mode, item.action_id)
        for item in plan
        if item.variant == "ACTION_PROPOSAL"
    }
    assert actual_mode_actions == required_mode_actions

    assignment_rows = [
        item
        for item in plan
        if item.variant == "MATCHED_CONTROL"
        and item.repetition_id == "R01"
        and item.anchor_completed_step == 16
        and item.member_id != "HEALTHY"
    ]
    assignment_counts = Counter(
        (item.member_id, item.treatment_duration) for item in assignment_rows
    )
    assert set(assignment_counts.values()) == {30}
    assert {duration for _, duration in assignment_counts} == {
        "TRANSIENT",
        "PERSISTENT",
    }


def test_approved_design_rejects_self_consistent_byte_substitution(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        PilotContractError,
        load_approved_pilot_design,
    )

    roster_source = (
        ROOT
        / "docs/plans/2026-08-14-habitat-v2-forecast-timing-pilot-roster-proposal-v1.json"
    )
    profile_source = (
        ROOT
        / "docs/plans/2026-08-14-habitat-v2-forecast-pilot-profile-action-proposal-v1.json"
    )
    roster_path = tmp_path / "roster.json"
    profile_path = tmp_path / "profile.json"
    shutil.copy2(profile_source, profile_path)
    roster = json.loads(roster_source.read_text(encoding="utf-8"))
    roster["clusters"][0]["operating_mode"] = "dormant"
    body = dict(roster)
    body.pop("proposal_sha256")
    roster["proposal_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    roster_path.write_text(
        json.dumps(roster, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PilotContractError, match="expected roster identity"):
        load_approved_pilot_design(
            ROOT,
            roster_path=roster_path,
            profile_path=profile_path,
        )


def test_canonical_lineage_rejects_pilot_namespace_and_ancestor() -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        PilotContractError,
        load_approved_pilot_design,
        validate_canonical_pilot_exclusion,
    )

    design = load_approved_pilot_design(ROOT)
    with pytest.raises(PilotContractError, match="pilot lineage"):
        validate_canonical_pilot_exclusion(
            design,
            candidate_cluster_ids=("pilot-v1/occupied/low/new-derived-alias",),
        )
    with pytest.raises(PilotContractError, match="pilot lineage"):
        validate_canonical_pilot_exclusion(
            design,
            candidate_cluster_ids=("canonical-v1/occupied/example",),
            ancestor_cluster_ids=(design.clusters[0].cluster_id,),
        )

    validate_canonical_pilot_exclusion(
        design,
        candidate_cluster_ids=("canonical-v1/occupied/example",),
        ancestor_cluster_ids=("canonical-source-v1/independent",),
    )


def test_resource_preflight_requires_exact_independent_identity(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        PilotContractError,
        load_resource_preflight,
    )

    body: dict[str, object] = {
        "schema_version": "aeolus_habitat_v2_forecast_pilot_resource_preflight_v1",
        "roster_sha256": "9514a25548d95047f3e707d1f2b27c76c3b09378653ecd270cdc9ae2845b06d1",
        "profile_action_sha256": "535cde8c397b115d5dd0b46c257462527f1e3eedfa3fb8560f02e45520854141",
        "planned_hmc_runs": 23_400,
        "benchmark_hmc_runs": 25,
        "measured_wall_time_seconds": 10.0,
        "measured_peak_rss_bytes": 100_000_000,
        "measured_artifact_bytes": 1_000_000,
        "projected_wall_time_seconds": 9_360.0,
        "projected_peak_rss_bytes": 100_000_000,
        "projected_artifact_bytes": 936_000_000,
        "runtime_within_ceiling": True,
        "memory_within_ceiling": True,
        "disk_reserve_preserved": True,
        "verdict": "PASS",
    }
    body["preflight_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    raw = canonical(body) + b"\n"
    expected_semantic = body["preflight_sha256"]
    expected_raw = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "preflight.json"
    path.write_bytes(raw)

    receipt = load_resource_preflight(
        path,
        expected_preflight_sha256=expected_semantic,
        expected_preflight_bytes_sha256=expected_raw,
    )
    assert receipt.verdict == "PASS"
    assert receipt.planned_hmc_runs == 23_400

    substitute = dict(body)
    substitute["benchmark_hmc_runs"] = 24
    substitute.pop("preflight_sha256")
    substitute["preflight_sha256"] = hashlib.sha256(canonical(substitute)).hexdigest()
    path.write_bytes(canonical(substitute) + b"\n")
    with pytest.raises(PilotContractError, match="expected preflight identity"):
        load_resource_preflight(
            path,
            expected_preflight_sha256=expected_semantic,
            expected_preflight_bytes_sha256=expected_raw,
        )


def test_materialize_all_clusters_and_treatment_mechanisms() -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        load_approved_pilot_design,
        materialize_pilot_scenario,
    )

    design = load_approved_pilot_design(ROOT)
    scenario_hashes: set[str] = set()
    for cluster in design.clusters:
        scenario = materialize_pilot_scenario(
            ROOT,
            design,
            cluster_id=cluster.cluster_id,
            member_id="HEALTHY",
            repetition_id="R01",
        )
        scenario.validate_contract_identities()
        assert scenario.data["steps"] == 72
        assert len(scenario.data["timeline"]) == 1
        assert scenario.data["timeline"][0]["operating_mode"] == cluster.operating_mode
        assert scenario.data["timeline"][0]["start_step"] == 0
        assert scenario.data["timeline"][0]["end_step"] == 72
        assert scenario.data["fault_profiles"] == []
        scenario_hashes.add(scenario.scenario_sha256)

    first_cluster = design.clusters[0]
    for treatment_id in design.treatment_ids:
        scenario = materialize_pilot_scenario(
            ROOT,
            design,
            cluster_id=first_cluster.cluster_id,
            member_id=treatment_id,
            repetition_id="R01",
        )
        scenario.validate_contract_identities()
        expected_profiles = 2 if treatment_id == "T12" else 1
        assert len(scenario.data["fault_profiles"]) == expected_profiles
        assert all(
            profile["id"].startswith(f"{first_cluster.cluster_id}.{treatment_id}.P")
            for profile in scenario.data["fault_profiles"]
        )
        scenario_hashes.add(scenario.scenario_sha256)

    assert len(scenario_hashes) == 72


def test_materializer_is_deterministic_and_balances_treatment_intervals() -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        load_approved_pilot_design,
        materialize_pilot_scenario,
    )

    design = load_approved_pilot_design(ROOT)
    intervals = Counter()
    for cluster in design.clusters:
        first = materialize_pilot_scenario(
            ROOT,
            design,
            cluster_id=cluster.cluster_id,
            member_id="T01",
            repetition_id="R02",
        )
        second = materialize_pilot_scenario(
            ROOT,
            design,
            cluster_id=cluster.cluster_id,
            member_id="T01",
            repetition_id="R02",
        )
        assert first.canonical_bytes == second.canonical_bytes
        loads = first.data["timeline"][0]["loads"]
        assert all(value >= 0.0 for zone in loads.values() for value in zone.values())
        profile = first.data["fault_profiles"][0]
        intervals[(profile["start_step"], profile["end_step"])] += 1
    assert intervals == {(25, 49): 30, (25, 73): 30}


def test_reduced_resource_and_pressure_profiles_preserve_claim_boundaries() -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        load_approved_pilot_design,
        materialize_pilot_scenario,
    )

    design = load_approved_pilot_design(ROOT)
    by_role = {cluster.semantic_profile_role: cluster for cluster in design.clusters}
    reduced = materialize_pilot_scenario(
        ROOT,
        design,
        cluster_id=by_role["reduced-resource-inventory"].cluster_id,
        member_id="HEALTHY",
        repetition_id="R01",
    )
    assert reduced.data["initial_utility"]["battery_energy_wh"] == 9_000.0
    assert reduced.data["initial_utility"]["oxygen_store_mol"] == 375.0
    assert reduced.data["initial_utility"]["co2_sorbent_remaining_mol"] == 1_125.0

    pressure = materialize_pilot_scenario(
        ROOT,
        design,
        cluster_id=by_role["pressure-inventory-skew"].cluster_id,
        member_id="HEALTHY",
        repetition_id="R01",
    )
    weighted_pressure = sum(
        zone["volume_m3"] * zone["initial"]["pressure_pa"]
        for zone in pressure.data["zones"]
    )
    assert weighted_pressure == 12_096_000.0


def test_reduced_resource_sensitivity_rehearsal_is_hmc_replayed_and_excluded(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        PilotContractError,
        load_approved_pilot_design,
        load_sensitivity_rehearsal_receipt,
        require_sensitivity_rehearsal_pass,
        run_reduced_resource_sensitivity_rehearsal,
        validate_sensitivity_rehearsal_receipt,
    )

    design = load_approved_pilot_design(ROOT)
    receipt = run_reduced_resource_sensitivity_rehearsal(ROOT, design)
    validate_sensitivity_rehearsal_receipt(receipt, design)
    assert require_sensitivity_rehearsal_pass(receipt, design) is receipt

    assert receipt["verdict"] == "PASS"
    assert receipt["permanently_excluded"] is True
    assert receipt["pilot_generation_authorized"] is False
    assert receipt["model_training_authorized"] is False
    assert receipt["hmc_runs"] == 26
    assert receipt["strict_replays"] == 26
    assert len(receipt["strata"]) == 12
    assert all(row["qualifying_channel_count"] >= 2 for row in receipt["strata"])
    assert all(row["verdict"] == "PASS" for row in receipt["strata"])
    assert receipt["determinism_probe"]["verdict"] == "PASS"
    assert receipt["terminal_closure"]["pilot_scenario_steps"] == 72
    assert receipt["terminal_closure"]["post_evaluation_step_count"] == 1

    receipt_path = tmp_path / "receipt.json"
    receipt_bytes = canonical(receipt)
    receipt_path.write_bytes(receipt_bytes)
    loaded = load_sensitivity_rehearsal_receipt(
        receipt_path,
        design,
        expected_receipt_sha256=receipt["receipt_sha256"],
        expected_receipt_bytes_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
    )
    assert loaded == receipt

    receipt_path.write_bytes(receipt_bytes + b" ")
    with pytest.raises(PilotContractError, match="byte identity"):
        load_sensitivity_rehearsal_receipt(
            receipt_path,
            design,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_receipt_bytes_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        )

    tampered = json.loads(json.dumps(receipt))
    tampered["strata"][0]["qualifying_channel_count"] = 0
    with pytest.raises(PilotContractError, match="self-hash"):
        validate_sensitivity_rehearsal_receipt(tampered, design)
