"""D2 pilot campaign: projection views and batch pair evidence.

Derives all nine ``(W, H)`` timing views from one replay-verified run bundle
without re-executing physics, then binds run evidence and projections into
pair evidence ready for custody staging.  The campaign entry point stays
closed until an independently pinned resource preflight authorizes it; this
module performs no generation campaign by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def _availability_row_sha256(
    continuation_id: str, availability: np.ndarray
) -> str:
    return hashlib.sha256(
        continuation_id.encode("utf-8")
        + b"\0"
        + np.ascontiguousarray(availability, dtype=np.bool_).tobytes()
    ).hexdigest()


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
    availability = np.stack(
        [view.history.status_f32[:, :, 0] == 1.0 for view in maximum_views],
        axis=0,
    )
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
    continuation_ids = [record["continuation_id"] for record in records]
    availability_row_sha256 = np.asarray(
        [
            _availability_row_sha256(continuation_id, row)
            for continuation_id, row in zip(
                continuation_ids, availability, strict=True
            )
        ]
    )
    if (
        histories.shape != (5, 16, 194)
        or availability.shape != (5, 16, 167)
        or availability.dtype != np.bool_
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
                    "aeolus_habitat_v2_forecast_training_pair_v2"
                ),
                pair_id=np.asarray(records[0]["pair_id"]),
                continuation_ids=np.asarray(continuation_ids),
                cluster_ids=np.asarray([record["cluster_id"] for record in records]),
                action_ids=np.asarray([record["action_id"] for record in records]),
                action_present=action_present,
                history_numeric_f32=histories,
                operational_available_bool=availability,
                operational_available_sha256=availability_row_sha256,
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


def _load_validated_staged_pair(
    destination: Path,
    design: Any,
) -> dict[str, Any]:
    """Validate and describe one existing pair before a campaign resume."""
    from .contracts import canonical_json_bytes
    from .pilot import (
        APPROVED_PROFILE_ACTION_SHA256,
        APPROVED_ROSTER_SHA256,
        PilotDesign,
    )
    from .pilot_custody import PAIR_MANIFEST_SCHEMA, validate_pilot_pair

    if type(design) is not PilotDesign or not destination.is_dir():
        raise PilotCampaignError("resume packet has an invalid destination or design")
    required_files = {"manifest.json", "records.jsonl", "training.npz"}
    if {item.name for item in destination.iterdir()} != required_files:
        raise PilotCampaignError("resume packet has unexpected or missing files")
    try:
        raw_records = (destination / "records.jsonl").read_bytes()
        records = [json.loads(line) for line in raw_records.splitlines()]
        validate_pilot_pair(design, records)
        canonical_records = b"".join(
            canonical_json_bytes(record) + b"\n" for record in records
        )
        if raw_records != canonical_records:
            raise PilotCampaignError("resume packet records are not canonical")
        manifest_path = destination / "manifest.json"
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
        declared_manifest_sha256 = manifest.pop("manifest_sha256")
        if (
            raw_manifest != canonical_json_bytes({**manifest, "manifest_sha256": declared_manifest_sha256})
            or declared_manifest_sha256 != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
            or manifest
            != {
                "schema_version": PAIR_MANIFEST_SCHEMA,
                "pair_id": destination.name,
                "record_count": 5,
                "records_sha256": hashlib.sha256(raw_records).hexdigest(),
                "roster_sha256": APPROVED_ROSTER_SHA256,
                "profile_action_sha256": APPROVED_PROFILE_ACTION_SHA256,
            }
        ):
            raise PilotCampaignError("resume packet manifest integrity drifts")
        training_path = destination / "training.npz"
        raw_training = training_path.read_bytes()
        with np.load(training_path, allow_pickle=False) as packet:
            required_arrays = {
                "schema_version",
                "pair_id",
                "continuation_ids",
                "cluster_ids",
                "action_ids",
                "action_present",
                "history_numeric_f32",
                "operational_available_bool",
                "operational_available_sha256",
                "proposed_action_f32",
                "targets_f32",
            }
            if (
                set(packet.files) != required_arrays
                or str(packet["schema_version"].item())
                != "aeolus_habitat_v2_forecast_training_pair_v2"
                or str(packet["pair_id"].item()) != destination.name
                or packet["history_numeric_f32"].shape != (5, 16, 194)
                or packet["operational_available_bool"].shape != (5, 16, 167)
                or packet["operational_available_bool"].dtype != np.bool_
                or packet["proposed_action_f32"].shape != (5, 27)
                or packet["targets_f32"].shape != (5, 8, 51)
                or packet["action_present"].shape != (5,)
                or not np.isfinite(packet["history_numeric_f32"]).all()
                or not np.isfinite(packet["proposed_action_f32"]).all()
                or not np.isfinite(packet["targets_f32"]).all()
                or packet["continuation_ids"].tolist()
                != [record["continuation_id"] for record in records]
                or packet["cluster_ids"].tolist()
                != [record["cluster_id"] for record in records]
                or packet["action_ids"].tolist()
                != [record["action_id"] for record in records]
            ):
                raise PilotCampaignError("resume packet training tensor integrity drifts")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, PilotCampaignError):
            raise
        raise PilotCampaignError("resume packet cannot be validated") from error
    return {
        "pair_id": destination.name,
        "manifest_sha256": declared_manifest_sha256,
        "training_packet_sha256": hashlib.sha256(raw_training).hexdigest(),
        "training_packet_byte_length": len(raw_training),
    }


def run_pilot_campaign(
    repo_root: str | Path,
    design: Any,
    contracts: ForecastContracts,
    *,
    preflight: Any,
    output_root: str | Path,
    allowed_cluster_ids: frozenset[str] | None = None,
    pair_limit: int | None = None,
    worker_count: int = 1,
    resume: bool = False,
    health_check: Callable[[str], None] | None = None,
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
    if (
        preflight.schema_version
        != "aeolus_habitat_v2_forecast_pilot_resource_preflight_v2"
        or type(preflight.v2_binding_sha256) is not str
        or len(preflight.v2_binding_sha256) != 64
        or any(character not in "0123456789abcdef" for character in preflight.v2_binding_sha256)
    ):
        raise PilotCampaignError("campaign requires a loaded v2 preflight binding")
    # This is a qualification-only actuator boundary.  The roster is derived
    # from the signed split here; callers cannot nominate any cluster, notably
    # not validation/final IDs, before grouping or materialization.
    from .qualification_split import build_qualification_split, load_qualified_protocol
    root = Path(repo_root).resolve()
    sealed = build_qualification_split(design, load_qualified_protocol(root))
    derived_cluster_ids = sealed.authorized_cluster_ids
    if allowed_cluster_ids is not None:
        # Compatibility input only: it proves no authority.  Any caller-selected
        # variation is rejected before continuation grouping or materialization.
        validation = allowed_cluster_ids & frozenset(sealed.validation_cluster_ids)
        if validation:
            raise PilotCampaignError("validation/final cluster IDs are forbidden before planning")
        if allowed_cluster_ids != derived_cluster_ids:
            raise PilotCampaignError("campaign received unknown roster IDs; sealed split is derived internally")
    allowed_cluster_ids = derived_cluster_ids
    approved_cluster_ids = frozenset(cluster.cluster_id for cluster in design.clusters)
    if not allowed_cluster_ids or allowed_cluster_ids - approved_cluster_ids or allowed_cluster_ids & frozenset(sealed.validation_cluster_ids):
        raise PilotCampaignError("sealed qualification roster is invalid or includes locked validation IDs")
    if pair_limit is not None and (type(pair_limit) is not int or pair_limit < 1):
        raise PilotCampaignError("pair limit must be a positive integer")
    if type(worker_count) is not int or worker_count < 1:
        raise PilotCampaignError("worker count must be a positive integer")
    if type(resume) is not bool:
        raise PilotCampaignError("resume flag must be a boolean")
    if health_check is not None and not callable(health_check):
        raise PilotCampaignError("health check must be callable")
    target = Path(output_root).resolve()

    def pair_groups():
        # Filter the plan before grouping, scenario materialization, or HMC
        # dispatch. Excluded clusters never reach an execution runner.
        continuations = (
            item
            for item in iter_pilot_continuations(design)
            if item.cluster_id in allowed_cluster_ids
        )
        width = 1 + len(design.action_ids)
        while True:
            group = tuple(islice(continuations, width))
            if not group:
                return
            if len(group) != width:
                raise PilotCampaignError("pilot plan ends with a partial pair")
            if any(item.cluster_id not in allowed_cluster_ids for item in group):
                raise PilotCampaignError("excluded cluster reached pair group")
            yield group

    groups = tuple(pair_groups())
    if pair_limit is not None:
        groups = groups[:pair_limit]
    if not groups:
        raise PilotCampaignError("allowed_cluster_ids selected no pilot pairs")
    expected_pair_ids = {group[0].pair_id for group in groups}
    existing_by_id: dict[str, dict[str, Any]] = {}
    if target.exists():
        if not resume:
            raise PilotCampaignError("campaign destination already exists")
        if (target / "campaign-manifest.json").exists():
            raise PilotCampaignError("cannot resume a campaign with a final manifest")
        for entry in target.iterdir():
            if not entry.is_dir():
                raise PilotCampaignError("resume destination has an unexpected root file")
            packet = _load_validated_staged_pair(entry, design)
            pair_id = packet["pair_id"]
            if pair_id not in expected_pair_ids or pair_id in existing_by_id:
                raise PilotCampaignError("resume destination contains an unexpected pair")
            existing_by_id[pair_id] = packet
    elif resume:
        raise PilotCampaignError("resume destination does not exist")

    missing_groups = [group for group in groups if group[0].pair_id not in existing_by_id]
    payloads = ((str(root), str(target), group) for group in missing_groups)
    if worker_count == 1:
        new_pair_manifests = []
        for payload, group in zip(payloads, missing_groups, strict=True):
            if health_check:
                health_check(f"before-pair-{group[0].pair_id}")
            new_pair_manifests.append(_execute_and_stage_pair(payload))
            if health_check:
                health_check(f"after-pair-{group[0].pair_id}")
    else:
        # A process pool cannot safely share a live guard callback. Check before
        # dispatch and after every completed pair so a breach stops new work.
        if health_check:
            health_check("before-parallel-pair-dispatch")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            new_pair_manifests = []
            for group, manifest in zip(missing_groups, executor.map(_execute_and_stage_pair, payloads), strict=True):
                new_pair_manifests.append(manifest)
                if health_check:
                    health_check(f"after-pair-{group[0].pair_id}")
    all_pair_manifests = {
        **existing_by_id,
        **{item["pair_id"]: item for item in new_pair_manifests},
    }
    pair_manifests = [all_pair_manifests[group[0].pair_id] for group in groups]

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
        "allowed_cluster_ids": sorted(allowed_cluster_ids),
        "pairs_completed": pairs_completed,
        "hmc_runs_executed": runs_executed,
        "pair_manifests": pair_manifests,
    }
    manifest["campaign_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (target / "campaign-manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest
