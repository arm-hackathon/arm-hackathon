from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _passing_preflight():
    from aeolus.habitat_v2.forecast.pilot import PilotResourcePreflight

    return PilotResourcePreflight(
        preflight_sha256="a" * 64,
        preflight_bytes_sha256="b" * 64,
        planned_hmc_runs=23_400,
        benchmark_hmc_runs=2,
        measured_wall_time_seconds=1.0,
        measured_peak_rss_bytes=1,
        measured_artifact_bytes=1,
        projected_wall_time_seconds=1.0,
        projected_peak_rss_bytes=1,
        projected_artifact_bytes=1,
        verdict="PASS",
        schema_version="aeolus_habitat_v2_forecast_pilot_resource_preflight_v2",
        v2_binding_sha256="c" * 64,
    )


def test_qualification_split_is_exact_deterministic_and_closed() -> None:
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.qualification_split import fit_cal_cluster_ids

    design = load_approved_pilot_design(ROOT)
    first = fit_cal_cluster_ids(ROOT, design)
    second = fit_cal_cluster_ids(ROOT, design)

    assert first == second
    assert len(first.fit_cluster_ids) == 36
    assert len(first.cal_cluster_ids) == 12
    assert len(first.validation_cluster_ids) == 12
    assert len(first.authorized_cluster_ids) == 48
    assert first.authorized_cluster_ids.isdisjoint(first.validation_cluster_ids)
    assert first.expected_packet_count == 3_744
    assert first.expected_example_count == 18_720
    assert (
        first.authorized_cluster_ids | frozenset(first.validation_cluster_ids)
        == frozenset(cluster.cluster_id for cluster in design.clusters)
    )


def test_qualification_split_rejects_roster_identity_drift() -> None:
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.qualification_split import (
        QualificationSplitError,
        fit_cal_cluster_ids,
    )

    design = load_approved_pilot_design(ROOT)
    drifted_cluster = replace(design.clusters[0], cluster_id="pilot-v1/drifted/cluster/id")
    drifted = replace(design, clusters=(drifted_cluster, *design.clusters[1:]))

    with pytest.raises(QualificationSplitError, match="identities drift"):
        fit_cal_cluster_ids(ROOT, drifted)


def test_campaign_filters_validation_before_executor(monkeypatch, tmp_path: Path) -> None:
    import aeolus.habitat_v2.forecast.pilot as pilot_module
    import aeolus.habitat_v2.forecast.pilot_campaign as campaign_module
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )
    from aeolus.habitat_v2.forecast.qualification_split import fit_cal_cluster_ids

    design = load_approved_pilot_design(ROOT)
    split = fit_cal_cluster_ids(ROOT, design)
    continuations = tuple(iter_pilot_continuations(design))

    def group_for(cluster_id: str):
        first = next(item for item in continuations if item.cluster_id == cluster_id)
        return tuple(item for item in continuations if item.pair_id == first.pair_id)

    validation_group = group_for(split.validation_cluster_ids[0])
    allowed_group = group_for(split.fit_cluster_ids[0])
    assert len(validation_group) == len(allowed_group) == 5

    monkeypatch.setattr(
        pilot_module,
        "iter_pilot_continuations",
        lambda _design: iter((*validation_group, *allowed_group)),
    )
    executed: list[str] = []

    def fake_execute(payload):
        _, target, group = payload
        executed.extend(item.cluster_id for item in group)
        destination = Path(target) / group[0].pair_id
        destination.mkdir(parents=True, exist_ok=False)
        return {"pair_id": group[0].pair_id}

    monkeypatch.setattr(campaign_module, "_execute_and_stage_pair", fake_execute)
    manifest = campaign_module.run_pilot_campaign(
        ROOT,
        design,
        load_forecast_contracts(ROOT),
        preflight=_passing_preflight(),
        output_root=tmp_path / "campaign",
        allowed_cluster_ids=split.authorized_cluster_ids,
    )

    assert set(executed) == {split.fit_cluster_ids[0]}
    assert set(executed).isdisjoint(split.validation_cluster_ids)
    assert manifest["pairs_completed"] == 1
    assert manifest["allowed_cluster_ids"] == sorted(split.authorized_cluster_ids)


def test_campaign_rejects_unknown_cluster_before_execution(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.pilot_campaign import (
        PilotCampaignError,
        run_pilot_campaign,
    )

    design = load_approved_pilot_design(ROOT)
    with pytest.raises(PilotCampaignError, match="unknown roster IDs"):
        run_pilot_campaign(
            ROOT,
            design,
            load_forecast_contracts(ROOT),
            preflight=_passing_preflight(),
            output_root=tmp_path / "campaign",
            allowed_cluster_ids=frozenset({"pilot-v1/not/approved/cluster"}),
        )


def test_qualification_split_exact_per_split_packet_and_example_totals() -> None:
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.qualification_split import fit_cal_cluster_ids

    design = load_approved_pilot_design(ROOT)
    split = fit_cal_cluster_ids(ROOT, design)

    packets_per_cluster = 2 * 13 * 3  # 2 reps x 13 members x 3 anchors
    examples_per_packet = 5

    fit_packets = len(split.fit_cluster_ids) * packets_per_cluster
    fit_examples = fit_packets * examples_per_packet
    cal_packets = len(split.cal_cluster_ids) * packets_per_cluster
    cal_examples = cal_packets * examples_per_packet

    assert len(split.fit_cluster_ids) == 36
    assert fit_packets == 2_808
    assert fit_examples == 14_040
    assert len(split.cal_cluster_ids) == 12
    assert cal_packets == 936
    assert cal_examples == 4_680
    assert len(split.authorized_cluster_ids) == 48
    assert split.expected_packet_count == 3_744
    assert split.expected_example_count == 18_720
    assert fit_examples + cal_examples == 18_720
    assert len(split.validation_cluster_ids) * packets_per_cluster * examples_per_packet == 4_680
