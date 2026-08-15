"""D2 pilot campaign: projection views and batch pair evidence.

Derives all nine ``(W, H)`` timing views from one replay-verified run bundle
without re-executing physics, then binds run evidence and projections into
pair evidence ready for custody staging.  The campaign entry point stays
closed until an independently pinned resource preflight authorizes it; this
module performs no generation campaign by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ForecastContracts
from .pilot import PilotContinuation
from .pilot_execution import PilotActionRunBundle, PilotControlRunBundle
from .projection import (
    ForecastHistory,
    project_history_window,
    project_physical_targets,
    project_proposed_action,
)

WINDOW_GRID: tuple[int, ...] = (4, 8, 16)
HORIZON_GRID: tuple[int, ...] = (2, 4, 8)


class PilotCampaignError(ValueError):
    """Pilot campaign evidence is outside its frozen contract."""


@dataclass(frozen=True, slots=True)
class RunView:
    """One timing view derived from a run's maximum witness."""

    window_steps: int
    horizon_steps: int
    history: ForecastHistory
    targets_f32: np.ndarray
    action_f32: np.ndarray | None


def project_run_views(
    contracts: ForecastContracts,
    continuation: PilotContinuation,
    bundle: PilotControlRunBundle | PilotActionRunBundle,
) -> tuple[RunView, ...]:
    """Slice all nine timing views from one executed run; physics runs once."""
    if type(contracts) is not ForecastContracts:
        raise PilotCampaignError("view projection requires the exact contracts bundle")
    if type(bundle) not in (PilotControlRunBundle, PilotActionRunBundle):
        raise PilotCampaignError("view projection requires an executed run bundle")
    anchor = continuation.anchor_completed_step
    if anchor != bundle.anchor_completed_step:
        raise PilotCampaignError("run bundle anchor drifts from continuation")

    action_f32: np.ndarray | None = None
    if continuation.variant == "ACTION_PROPOSAL":
        if type(bundle) is not PilotActionRunBundle:
            raise PilotCampaignError("action view requires an action run bundle")
        command = bundle.anchor.get("proposed_command")
        if command is None:
            raise PilotCampaignError("action run bundle lacks its proposed command")
        action_f32 = project_proposed_action(contracts, command)

    views: list[RunView] = []
    for window in WINDOW_GRID:
        first_step = anchor - window + 1
        if first_step < 1:
            raise PilotCampaignError("window exceeds the executed history")
        try:
            pairs = [bundle.snapshots[step] for step in range(first_step, anchor + 1)]
        except KeyError as error:
            raise PilotCampaignError("run bundle lacks the required snapshots") from error
        history = project_history_window(contracts, pairs, window_steps=window)
        for horizon in HORIZON_GRID:
            last_step = anchor + horizon
            try:
                states = [bundle.states[step] for step in range(anchor + 1, last_step + 1)]
            except KeyError as error:
                raise PilotCampaignError(
                    "run bundle lacks the required shadow states"
                ) from error
            targets = project_physical_targets(
                contracts, states, horizon_steps=horizon
            )
            views.append(
                RunView(
                    window_steps=window,
                    horizon_steps=horizon,
                    history=history,
                    targets_f32=targets,
                    action_f32=action_f32,
                )
            )
    return tuple(views)



@dataclass(frozen=True, slots=True)
class PilotPairEvidence:
    """One executed matched pair: five records plus their nine views each."""

    pair_id: str
    records: tuple[dict[str, Any], ...]
    views: tuple[tuple[RunView, ...], ...]


def run_pilot_pair(
    repo_root: str | Path,
    design: Any,
    contracts: ForecastContracts,
    pair_continuations: tuple[PilotContinuation, ...],
) -> PilotPairEvidence:
    """Execute one control and four actions as a matched, reusable pair."""
    from .pilot import PilotDesign
    from .pilot_custody import build_run_record, validate_pilot_pair
    from .pilot_execution import (
        run_pilot_action_continuation,
        run_pilot_control_continuation,
    )

    if type(design) is not PilotDesign or type(contracts) is not ForecastContracts:
        raise PilotCampaignError("pair execution requires exact plan and contracts")
    items = tuple(pair_continuations)
    controls = [item for item in items if item.variant == "MATCHED_CONTROL"]
    actions = [item for item in items if item.variant == "ACTION_PROPOSAL"]
    if len(controls) != 1 or len(actions) != len(design.action_ids):
        raise PilotCampaignError(
            "pair execution requires exactly one control and four actions"
        )
    control = controls[0]
    if any(item.pair_id != control.pair_id for item in items):
        raise PilotCampaignError("pair continuations do not share one pair identity")

    root = Path(repo_root).resolve()
    control_bundle = run_pilot_control_continuation(root, design, control)
    action_bundles = tuple(
        run_pilot_action_continuation(root, design, continuation)
        for continuation in actions
    )
    ordered_continuations = (control, *actions)
    ordered_bundles = (control_bundle, *action_bundles)
    records = tuple(
        build_run_record(design, continuation, bundle)
        for continuation, bundle in zip(ordered_continuations, ordered_bundles)
    )
    validate_pilot_pair(design, records)
    views = tuple(
        project_run_views(contracts, continuation, bundle)
        for continuation, bundle in zip(ordered_continuations, ordered_bundles)
    )
    return PilotPairEvidence(
        pair_id=control.pair_id,
        records=records,
        views=views,
    )


def stage_pair_training_packet(
    destination: str | Path,
    records: tuple[dict[str, Any], ...],
    views: tuple[tuple[RunView, ...], ...],
) -> dict[str, Any]:
    """Persist one compact, trainable maximum-context packet for a matched pair."""
    path = Path(destination)
    if path.suffix != ".npz" or not path.parent.is_dir() or path.exists():
        raise PilotCampaignError("training packet destination must be a new .npz")
    if len(records) != 5 or len(views) != len(records):
        raise PilotCampaignError("training packet requires one complete matched pair")

    maximum_views: list[RunView] = []
    for item in views:
        selected = [
            view
            for view in item
            if view.window_steps == 16 and view.horizon_steps == 8
        ]
        if len(selected) != 1:
            raise PilotCampaignError("pair lacks an exact maximum training view")
        maximum_views.append(selected[0])

    histories = np.stack(
        [view.history.numeric_f32 for view in maximum_views], axis=0
    ).astype(np.float32, copy=False)
    targets = np.stack([view.targets_f32 for view in maximum_views], axis=0).astype(
        np.float32, copy=False
    )
    action_present = np.asarray(
        [view.action_f32 is not None for view in maximum_views], dtype=np.bool_
    )
    actions = np.stack(
        [
            view.action_f32
            if view.action_f32 is not None
            else np.zeros(27, dtype=np.float32)
            for view in maximum_views
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if (
        histories.shape != (5, 16, 194)
        or targets.shape != (5, 8, 51)
        or actions.shape != (5, 27)
        or not np.isfinite(histories).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(actions).all()
    ):
        raise PilotCampaignError("training tensors drift from the frozen layout")

    try:
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray(
                    "aeolus_habitat_v2_forecast_training_pair_v1"
                ),
                pair_id=np.asarray(records[0]["pair_id"]),
                continuation_ids=np.asarray(
                    [record["continuation_id"] for record in records]
                ),
                cluster_ids=np.asarray([record["cluster_id"] for record in records]),
                action_ids=np.asarray([record["action_id"] for record in records]),
                action_present=action_present,
                history_numeric_f32=histories,
                proposed_action_f32=actions,
                targets_f32=targets,
            )
    except OSError as error:
        raise PilotCampaignError("training packet cannot be written") from error
    raw = path.read_bytes()
    return {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
        "sample_count": len(records),
    }


def _execute_and_stage_pair(
    payload: tuple[str, str, tuple[PilotContinuation, ...]],
) -> dict[str, Any]:
    """Execute one isolated pair worker and write only its own packet directory."""
    from .contracts import load_forecast_contracts
    from .pilot import load_approved_pilot_design
    from .pilot_custody import stage_pair_packet

    root_text, target_text, group = payload
    root = Path(root_text)
    target = Path(target_text)
    design = load_approved_pilot_design(root)
    evidence = run_pilot_pair(root, design, load_forecast_contracts(root), group)
    pair_dir = target / evidence.pair_id
    pair_manifest = stage_pair_packet(pair_dir, design, evidence.records)
    training_packet = stage_pair_training_packet(
        pair_dir / "training.npz", evidence.records, evidence.views
    )
    return {
        "pair_id": evidence.pair_id,
        "manifest_sha256": pair_manifest["manifest_sha256"],
        "training_packet_sha256": training_packet["sha256"],
        "training_packet_byte_length": training_packet["byte_length"],
    }


def run_pilot_campaign(
    repo_root: str | Path,
    design: Any,
    contracts: ForecastContracts,
    *,
    preflight: Any,
    output_root: str | Path,
    pair_limit: int | None = None,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Execute the bounded pilot campaign under a passing preflight receipt.

    The preflight must be a loaded ``PilotResourcePreflight`` (which already
    proves an independently pinned PASS).  This function opens no authority by
    itself; it executes the frozen plan exactly as benchmarked.
    """
    from concurrent.futures import ProcessPoolExecutor
    from itertools import islice

    from .contracts import canonical_json_bytes
    from .pilot import (
        APPROVED_PROFILE_ACTION_SHA256,
        APPROVED_ROSTER_SHA256,
        PilotDesign,
        PilotResourcePreflight,
        iter_pilot_continuations,
    )

    if type(design) is not PilotDesign or type(contracts) is not ForecastContracts:
        raise PilotCampaignError("campaign requires exact plan and contracts")
    if (
        type(preflight) is not PilotResourcePreflight
        or preflight.verdict != "PASS"
        or preflight.planned_hmc_runs != 23_400
    ):
        raise PilotCampaignError(
            "campaign requires a passing, pinned resource preflight"
        )
    if pair_limit is not None and (type(pair_limit) is not int or pair_limit < 1):
        raise PilotCampaignError("pair limit must be a positive integer")
    if type(worker_count) is not int or worker_count < 1:
        raise PilotCampaignError("worker count must be a positive integer")
    root = Path(repo_root).resolve()
    target = Path(output_root).resolve()
    if target.exists():
        raise PilotCampaignError("campaign destination already exists")

    def pair_groups():
        continuations = iter(iter_pilot_continuations(design))
        width = 1 + len(design.action_ids)
        while True:
            group = tuple(islice(continuations, width))
            if not group:
                return
            if len(group) != width:
                raise PilotCampaignError("pilot plan ends with a partial pair")
            yield group

    groups = pair_groups()
    if pair_limit is not None:
        groups = islice(groups, pair_limit)
    payloads = ((str(root), str(target), group) for group in groups)
    if worker_count == 1:
        pair_manifests = [_execute_and_stage_pair(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pair_manifests = list(executor.map(_execute_and_stage_pair, payloads))

    width = 1 + len(design.action_ids)
    pairs_completed = len(pair_manifests)
    runs_executed = pairs_completed * width

    manifest: dict[str, Any] = {
        "schema_version": "aeolus_habitat_v2_forecast_pilot_campaign_manifest_v1",
        "roster_sha256": APPROVED_ROSTER_SHA256,
        "profile_action_sha256": APPROVED_PROFILE_ACTION_SHA256,
        "preflight_sha256": preflight.preflight_sha256,
        "planned_hmc_runs": preflight.planned_hmc_runs,
        "worker_count": worker_count,
        "pairs_completed": pairs_completed,
        "hmc_runs_executed": runs_executed,
        "pair_manifests": pair_manifests,
    }
    manifest["campaign_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (target / "campaign-manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest
