"""D2 pilot campaign: projection views and batch pair evidence.

Derives all nine ``(W, H)`` timing views from one replay-verified run bundle
without re-executing physics, then binds run evidence and projections into
pair evidence ready for custody staging.  The campaign entry point stays
closed until an independently pinned resource preflight authorizes it; this
module performs no generation campaign by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
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



def run_pilot_campaign(
    repo_root: str | Path,
    design: Any,
    contracts: ForecastContracts,
    *,
    preflight: Any,
    output_root: str | Path,
    pair_limit: int | None = None,
) -> dict[str, Any]:
    """Execute the bounded pilot campaign under a passing preflight receipt.

    The preflight must be a loaded ``PilotResourcePreflight`` (which already
    proves an independently pinned PASS).  This function opens no authority by
    itself; it executes the frozen plan exactly as benchmarked.
    """
    from hashlib import sha256
    from itertools import islice

    from .contracts import canonical_json_bytes
    from .pilot import (
        APPROVED_PROFILE_ACTION_SHA256,
        APPROVED_ROSTER_SHA256,
        PilotDesign,
        PilotResourcePreflight,
        iter_pilot_continuations,
    )
    from .pilot_custody import stage_pair_packet

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
    root = Path(repo_root).resolve()
    target = Path(output_root)
    if target.exists():
        raise PilotCampaignError("campaign destination already exists")

    continuations = iter(iter_pilot_continuations(design))
    width = 1 + len(design.action_ids)
    pairs_completed = 0
    runs_executed = 0
    pair_manifests: list[dict[str, Any]] = []
    while True:
        group = tuple(islice(continuations, width))
        if not group:
            break
        if len(group) != width:
            raise PilotCampaignError("pilot plan ends with a partial pair")
        evidence = run_pilot_pair(root, design, contracts, group)
        pair_dir = target / evidence.pair_id
        pair_manifest = stage_pair_packet(pair_dir, design, evidence.records)
        pair_manifests.append(
            {
                "pair_id": evidence.pair_id,
                "manifest_sha256": pair_manifest["manifest_sha256"],
            }
        )
        pairs_completed += 1
        runs_executed += width
        if pair_limit is not None and pairs_completed >= pair_limit:
            break

    manifest: dict[str, Any] = {
        "schema_version": "aeolus_habitat_v2_forecast_pilot_campaign_manifest_v1",
        "roster_sha256": APPROVED_ROSTER_SHA256,
        "profile_action_sha256": APPROVED_PROFILE_ACTION_SHA256,
        "preflight_sha256": preflight.preflight_sha256,
        "planned_hmc_runs": preflight.planned_hmc_runs,
        "pairs_completed": pairs_completed,
        "hmc_runs_executed": runs_executed,
        "pair_manifests": pair_manifests,
    }
    manifest["campaign_manifest_sha256"] = sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (target / "campaign-manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest
