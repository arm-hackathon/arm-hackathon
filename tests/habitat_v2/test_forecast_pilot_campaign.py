from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_project_run_views_derives_all_nine_timing_views() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )
    from aeolus.habitat_v2.forecast.pilot_campaign import project_run_views
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_control_continuation,
    )

    design = load_approved_pilot_design(ROOT)
    contracts = load_forecast_contracts(ROOT)
    continuation = next(iter_pilot_continuations(design))
    bundle = run_pilot_control_continuation(ROOT, design, continuation)

    views = project_run_views(contracts, continuation, bundle)

    assert len(views) == 9
    assert [(view.window_steps, view.horizon_steps) for view in views] == [
        (4, 2),
        (4, 4),
        (4, 8),
        (8, 2),
        (8, 4),
        (8, 8),
        (16, 2),
        (16, 4),
        (16, 8),
    ]
    anchor = continuation.anchor_completed_step
    for view in views:
        assert view.history.numeric_f32.shape == (view.window_steps, 194)
        assert view.targets_f32.shape == (view.horizon_steps, 51)
        assert view.history.completed_times_s[-1] == anchor * 60.0
        assert view.action_f32 is None
    # All views derive from the same witness: identical shared history rows.
    widest = next(view for view in views if view.window_steps == 16)
    narrow = next(view for view in views if view.window_steps == 4)
    assert (
        widest.history.numeric_f32[-4:].tobytes()
        == narrow.history.numeric_f32.tobytes()
    )



def test_project_run_views_binds_requested_action_vector() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )
    from aeolus.habitat_v2.forecast.pilot_campaign import project_run_views
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_action_continuation,
    )

    design = load_approved_pilot_design(ROOT)
    contracts = load_forecast_contracts(ROOT)
    continuation = next(
        item
        for item in iter_pilot_continuations(design)
        if item.variant == "ACTION_PROPOSAL"
    )
    bundle = run_pilot_action_continuation(ROOT, design, continuation)

    views = project_run_views(contracts, continuation, bundle)

    assert len(views) == 9
    vectors = {view.action_f32.tobytes() for view in views}
    assert len(vectors) == 1
    for view in views:
        assert view.action_f32 is not None
        assert view.action_f32.shape == (27,)
        assert view.targets_f32.shape == (view.horizon_steps, 51)



def test_run_pilot_pair_executes_control_once_and_binds_evidence(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )
    from aeolus.habitat_v2.forecast.pilot_campaign import run_pilot_pair
    from aeolus.habitat_v2.forecast.pilot_custody import (
        stage_pair_packet,
        validate_pilot_pair,
    )

    design = load_approved_pilot_design(ROOT)
    contracts = load_forecast_contracts(ROOT)
    continuations = iter_pilot_continuations(design)
    control = next(continuations)
    actions = tuple(next(continuations) for _ in range(len(design.action_ids)))

    evidence = run_pilot_pair(ROOT, design, contracts, (control, *actions))

    assert evidence.pair_id == control.pair_id
    assert len(evidence.records) == 5
    controls = [
        record
        for record in evidence.records
        if record["variant"] == "MATCHED_CONTROL"
    ]
    assert len(controls) == 1
    validate_pilot_pair(design, evidence.records)
    assert len(evidence.views) == 5
    for record, views in zip(evidence.records, evidence.views):
        assert len(views) == 9
        if record["variant"] == "MATCHED_CONTROL":
            assert all(view.action_f32 is None for view in views)
        else:
            assert all(view.action_f32 is not None for view in views)
    # The four action runs must share the single executed control's identity.
    control_run_ids = {record["control_run_id"] for record in evidence.records}
    assert len(control_run_ids) == 1
    manifest = stage_pair_packet(tmp_path / "packet", design, evidence.records)
    assert manifest["record_count"] == 5



def _synthetic_pass_preflight():
    from aeolus.habitat_v2.forecast.pilot import PilotResourcePreflight

    return PilotResourcePreflight(
        preflight_sha256="a" * 64,
        preflight_bytes_sha256="b" * 64,
        planned_hmc_runs=23_400,
        benchmark_hmc_runs=2,
        measured_wall_time_seconds=20.0,
        measured_peak_rss_bytes=600_000_000,
        measured_artifact_bytes=420_000,
        projected_wall_time_seconds=234_000.0,
        projected_peak_rss_bytes=600_000_000,
        projected_artifact_bytes=4_914_000_000,
        verdict="PASS",
    )


def test_campaign_refuses_without_passing_preflight(tmp_path: Path) -> None:
    import pytest

    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.pilot_campaign import (
        PilotCampaignError,
        run_pilot_campaign,
    )

    design = load_approved_pilot_design(ROOT)
    contracts = load_forecast_contracts(ROOT)
    with pytest.raises(PilotCampaignError, match="preflight"):
        run_pilot_campaign(
            ROOT,
            design,
            contracts,
            preflight=None,
            output_root=tmp_path / "campaign",
            pair_limit=1,
        )
    failed = _synthetic_pass_preflight()
    object.__setattr__(failed, "verdict", "FAIL_RESOURCE_CEILING")
    with pytest.raises(PilotCampaignError, match="preflight"):
        run_pilot_campaign(
            ROOT,
            design,
            contracts,
            preflight=failed,
            output_root=tmp_path / "campaign",
            pair_limit=1,
        )


def test_campaign_executes_bounded_pairs_and_writes_manifest(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.pilot_campaign import run_pilot_campaign

    design = load_approved_pilot_design(ROOT)
    contracts = load_forecast_contracts(ROOT)
    manifest = run_pilot_campaign(
        ROOT,
        design,
        contracts,
        preflight=_synthetic_pass_preflight(),
        output_root=tmp_path / "campaign",
        pair_limit=1,
    )

    assert manifest["pairs_completed"] == 1
    assert manifest["hmc_runs_executed"] == 5
    assert manifest["planned_hmc_runs"] == 23_400
    assert manifest["preflight_sha256"] == "a" * 64
    pair_dirs = [entry for entry in (tmp_path / "campaign").iterdir() if entry.is_dir()]
    assert len(pair_dirs) == 1
    assert (pair_dirs[0] / "records.jsonl").exists()
    assert (pair_dirs[0] / "manifest.json").exists()
