from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from aeolus.habitat_v2.control_trace import parse_control_trace, replay_control_trace
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics import advance_one_step_with_command, initial_state

from .contracts import ForecastContracts, load_forecast_contracts
from .corpus import (
    CorpusValidationError,
    RELEASE_TIER,
    assign_cluster_splits,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    record_identity,
    validate_jsonl_records,
    validate_lineage,
    validate_record,
    validate_relative_packet_path,
)
from .projection import (
    forecast_layout,
    project_history_window,
    project_physical_targets,
    project_proposed_action,
)


FINAL_HMC_COMMIT_SHA = "3bc5da3d716212cac6524b088a963b6abf47a0ef"
PATH_PREFIX = "development-fixture-only/"
WINDOW_STEPS = 4
HORIZON_STEPS = 2
ANCHOR_STEP = 16
_SPLIT_KEY = b"aeolus-forecast-d1-development-fixture-split-key-v1"
_SPLIT_POLICY_SHA256 = hashlib.sha256(
    b"aeolus-forecast-d1-development-split-policy-v1"
).hexdigest()
_SPLIT_KEY_ID = "development-fixture-fixed-key-v1"


class ForecastPipelineError(ValueError):
    """The closed development fixture cannot be generated or verified."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _nonce(action_id: str) -> bytes:
    return hashlib.sha256(
        b"aeolus-forecast-d1-nonce-v1\0" + action_id.encode("utf-8")
    ).digest()


def _proposal(
    hmc: HabitatManagementComputer,
    snapshot_sha256: str,
    step: int,
    command: dict[str, Any],
    action_id: str,
) -> dict[str, Any]:
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": action_id,
        "source_type": "forecast_data_generator",
        "completed_observation_step": step,
        "observation_snapshot_sha256": snapshot_sha256,
        "requested_application_step": step,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": command,
        "confidence": None,
    }
    return {**body, "proposal_sha256": _sha256(canonical_json_bytes(body))}


def _record(specification: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    record = {
        "schema_version": specification["schema_version"],
        "release_tier": RELEASE_TIER,
        **values,
    }
    identifier = specification.get("id_field")
    if type(identifier) is str:
        record[identifier] = record_identity(record, specification)
    record["record_sha256"] = _sha256(canonical_json_bytes(record))
    return validate_record(record, specification)


def _manifest(
    bundle: ForecastContracts,
    table_artifacts: list[dict[str, Any]],
    trace_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    specification = dict(bundle.record_contract["records"]["manifest"])
    value = {
        "schema_version": specification["schema_version"],
        "release_tier": RELEASE_TIER,
        "record_contract_sha256": bundle.development_record_contract_sha256,
        "hmc_binding_sha256": bundle.binding_sha256,
        "alarm_manifest_sha256": bundle.alarm_manifest_sha256,
        "action_catalogue_sha256": bundle.action_catalogue_sha256,
        "development_profile_sha256": bundle.development_profile_sha256,
        "scenario_sha256": bundle.development_scenario.scenario_sha256,
        "window_steps": WINDOW_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "table_artifacts": table_artifacts,
        "trace_artifacts": trace_artifacts,
        "packet_root": "development-fixture-only",
        "gate_status": "STOP_UNDERPOWERED",
        "selection_allowed": False,
        "final_set_present": False,
        "sealed": True,
    }
    value["manifest_sha256"] = _sha256(canonical_json_bytes(value))
    return validate_record(value, specification, self_hash_field="manifest_sha256")


def _array(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _execute_run(bundle: ForecastContracts, action: Any) -> dict[str, Any]:
    nonce = _nonce(action.action_id)
    hmc = HabitatManagementComputer.reset(
        bundle.development_scenario, bundle.hmc_contract, nonce
    )
    shadow = initial_state(bundle.development_scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    states: dict[int, Any] = {0: shadow}
    witnesses: list[dict[str, Any]] = []
    anchor: dict[str, Any] | None = None
    for application_step in range(int(bundle.development_scenario.data["steps"])):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise ForecastPipelineError(
                "HMC entered terminal state while observing fixture"
            )
        snapshot, verification = observed
        if (
            verification.completed_step != application_step
            or snapshot.snapshot_sha256 != verification.snapshot_sha256
        ):
            raise ForecastPipelineError(
                "issued snapshot and verification receipt drift"
            )
        handle = hmc.verify_snapshot(snapshot, verification)
        if application_step:
            snapshots[application_step] = (snapshot, verification)
        proposal = None
        if application_step == ANCHOR_STEP:
            proposal = _proposal(
                hmc,
                snapshot.snapshot_sha256,
                application_step,
                action.command.to_mapping(),
                action.action_id,
            )
        proposal_receipt = hmc.propose(proposal, handle)
        proposal_mapping = proposal_receipt.to_mapping()
        if application_step == ANCHOR_STEP:
            if (
                proposal_mapping["attempt_class"],
                proposal_mapping["validation_outcome"],
            ) != ("CANONICAL_PROPOSAL", "VALID"):
                raise ForecastPipelineError(
                    "anchor canonical proposal was not admitted"
                )
        elif proposal_mapping["validation_outcome"] != "NO_PROPOSAL":
            raise ForecastPipelineError(
                "fixture requires NO_PROPOSAL outside the anchor"
            )
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise ForecastPipelineError(
                "HMC entered terminal state while arbitrating fixture"
            )
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise ForecastPipelineError(
                "HMC entered terminal state while stepping fixture"
            )
        shadow_result = advance_one_step_with_command(
            bundle.development_scenario, shadow, arbitration.final_command
        )
        shadow_digest = _sha256(canonical_json_bytes(shadow_result.receipt))
        if shadow_digest != step_receipt.plant_receipt_digest:
            raise ForecastPipelineError(
                "shadow plant receipt diverges from final HMC receipt"
            )
        shadow = shadow_result.state
        states[shadow.step] = shadow
        witnesses.append(
            {
                "application_step": application_step,
                "final_command_sha256": arbitration.final_command_sha256,
                "hmc_plant_receipt_digest": step_receipt.plant_receipt_digest,
                "shadow_plant_receipt_digest": shadow_digest,
                "shadow_state_sha256": _state_sha256(shadow),
            }
        )
        if application_step == ANCHOR_STEP:
            anchor = {
                "snapshot": snapshot,
                "verification": verification,
                "proposal_receipt": proposal_receipt,
                "arbitration": arbitration,
                "step_receipt": step_receipt,
                "shadow_digest": shadow_digest,
            }
    if anchor is None or not set(range(1, ANCHOR_STEP + 1)).issubset(snapshots):
        raise ForecastPipelineError(
            "fixture does not contain the required completed history"
        )
    trace = hmc.export_control_trace(FINAL_HMC_COMMIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes,
        scenario=bundle.development_scenario,
        contract=bundle.hmc_contract,
    )
    replay = replay_control_trace(
        trace.canonical_bytes,
        scenario=bundle.development_scenario,
        contract=bundle.hmc_contract,
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != 24
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise ForecastPipelineError("fixture trace fails strict completed replay")
    return {
        "nonce": nonce,
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "snapshots": snapshots,
        "states": states,
        "witnesses": witnesses,
        "anchor": anchor,
        "trace": trace,
        "trace_footer": dict(parsed.footer),
    }


def _state_sha256(state: Any) -> str:
    # The public replay trace footer is the final-state identity authority.  Per-step
    # witness state commitments are deterministic and never read HMC private state.
    zones = {
        zone: {
            "co2_mol": float(value.co2_mol),
            "o2_mol": float(value.o2_mol),
            "water_vapor_mol": float(value.water_vapor_mol),
            "inert_mol": float(value.inert_mol),
            "temperature_k": float(value.temperature_k),
        }
        for zone, value in sorted(state.zones.items())
    }
    utility = state.utility
    return _sha256(
        canonical_json_bytes(
            {
                "step": state.step,
                "zones": zones,
                "battery_energy_wh": float(utility.battery_energy_wh),
                "oxygen_store_mol": float(utility.oxygen_store_mol),
                "co2_sorbent_remaining_mol": float(utility.co2_sorbent_remaining_mol),
            }
        )
    )


def _safe_destination(caller_root: Path, destination_name: str) -> Path:
    if (
        type(destination_name) is not str
        or not destination_name
        or Path(destination_name).name != destination_name
    ):
        raise ForecastPipelineError("destination must be a new simple sibling name")
    root = caller_root.resolve()
    if not root.is_dir():
        raise ForecastPipelineError("explicit caller root must exist")
    destination = (root / destination_name).resolve()
    if destination.parent != root or destination.exists():
        raise ForecastPipelineError(
            "destination must be a new child of explicit caller root"
        )
    return destination


def _write_packet(
    staging: Path,
    bundle: ForecastContracts,
    tables: dict[str, list[dict[str, Any]]],
    traces: dict[str, bytes],
) -> dict[str, str]:
    root = staging / "development-fixture-only"
    root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    artifacts: list[dict[str, Any]] = []
    for name, rows in tables.items():
        specification = bundle.record_contract["records"][name]
        relative = specification["path"]
        validate_relative_packet_path(relative)
        raw = canonical_jsonl_bytes(rows)
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[relative] = _sha256(raw)
        artifacts.append(
            {
                "relative_path": relative,
                "row_count": len(rows),
                "byte_length": len(raw),
                "sha256": hashes[relative],
            }
        )
    trace_artifacts: list[dict[str, Any]] = []
    for record_id, raw in traces.items():
        relative = f"development-fixture-only/traces/{record_id}.json"
        validate_relative_packet_path(relative)
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[relative] = _sha256(raw)
        trace_artifacts.append(
            {
                "control_trace_record_id": record_id,
                "relative_path": relative,
                "byte_length": len(raw),
                "sha256": hashes[relative],
            }
        )
    manifest = _manifest(
        bundle,
        sorted(artifacts, key=lambda item: item["relative_path"]),
        sorted(trace_artifacts, key=lambda item: item["relative_path"]),
    )
    manifest_path = bundle.record_contract["records"]["manifest"]["path"]
    raw_manifest = canonical_json_bytes(manifest)
    (staging / manifest_path).write_bytes(raw_manifest)
    hashes[manifest_path] = _sha256(raw_manifest)
    return hashes


def generate_development_fixture(
    repo_root: str | Path,
    caller_root: str | Path,
    destination_name: str = "habitat-v2-forecast-d1-development",
) -> dict[str, Any]:
    """Build, validate, then atomically publish one new deterministic D1 packet."""
    bundle = load_forecast_contracts(repo_root)
    destination = _safe_destination(Path(caller_root), destination_name)
    staging = destination.parent / f".{destination.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ForecastPipelineError("new sibling staging directory already exists")
    staging.mkdir()
    try:
        cluster_spec = dict(bundle.record_contract["records"]["family_clusters"])
        cluster = _record(
            cluster_spec,
            {
                "stratum": "constant-occupied/no-treatment",
                "generator_contract_sha256": bundle.development_record_contract_sha256,
                "development_profile_sha256": bundle.development_profile_sha256,
            },
        )
        split_spec = dict(bundle.record_contract["records"]["split_assignments"])
        split_values = assign_cluster_splits(
            [cluster],
            split_key=_SPLIT_KEY,
            split_policy_sha256=_SPLIT_POLICY_SHA256,
            split_key_id=_SPLIT_KEY_ID,
            development_only=True,
        )[0]
        split = _record(split_spec, split_values)
        family_spec = dict(bundle.record_contract["records"]["families"])
        family_values = {
            "family_cluster_id": cluster["family_cluster_id"],
            "family_role": "HEALTHY",
            "treatment_kind": "NONE",
        }
        family = _record(family_spec, {**family_values, "scenario_member_ids": []})
        member_spec = dict(bundle.record_contract["records"]["scenario_members"])
        member = _record(
            member_spec,
            {
                "family_id": family["family_id"],
                "scenario_path": "scenarios/habitat_v2_forecast_development.json",
                "scenario_sha256": bundle.development_scenario.scenario_sha256,
                "plant_run_id": bundle.development_scenario.run_id,
                "profile_manifest_sha256": bundle.development_profile_sha256,
            },
        )
        family["scenario_member_ids"] = [member["scenario_member_id"]]
        family["record_sha256"] = _sha256(
            canonical_json_bytes(
                {key: value for key, value in family.items() if key != "record_sha256"}
            )
        )
        validate_record(family, family_spec)
        tables: dict[str, list[dict[str, Any]]] = {
            "family_clusters": [cluster],
            "families": [family],
            "scenario_members": [member],
            "split_assignments": [split],
            "control_runs": [],
            "control_traces": [],
            "replay_witnesses": [],
            "samples": [],
        }
        traces: dict[str, bytes] = {}
        layout = forecast_layout(bundle)
        for action in bundle.actions:
            run = _execute_run(bundle, action)
            trace_footer = run["trace_footer"]
            trace_spec = dict(bundle.record_contract["records"]["control_traces"])
            trace_record = _record(
                trace_spec,
                {
                    "control_run_id": run["control_run_id"],
                    "relative_trace_path": "",  # completed after identity creation
                    "byte_length": len(run["trace"].canonical_bytes),
                    "control_trace_sha256": _sha256(run["trace"].canonical_bytes),
                    "control_trace_footer_sha256": trace_footer[
                        "control_trace_footer_sha256"
                    ],
                    "final_state_sha256": trace_footer["final_state_sha256"],
                },
            )
            trace_record["relative_trace_path"] = (
                f"development-fixture-only/traces/{trace_record['control_trace_record_id']}.json"
            )
            # The relative artifact location is not a stable trace identity, but binds its self hash.
            trace_record["record_sha256"] = _sha256(
                canonical_json_bytes(
                    {
                        key: value
                        for key, value in trace_record.items()
                        if key != "record_sha256"
                    }
                )
            )
            validate_record(trace_record, trace_spec)
            witnesses_hash = _sha256(canonical_json_bytes(run["witnesses"]))
            witness_spec = dict(bundle.record_contract["records"]["replay_witnesses"])
            witness = _record(
                witness_spec,
                {
                    "control_run_id": run["control_run_id"],
                    "control_trace_record_id": trace_record["control_trace_record_id"],
                    "transition_count": len(run["witnesses"]),
                    "step_witnesses": run["witnesses"],
                    "step_witnesses_sha256": witnesses_hash,
                    "final_state_sha256": trace_footer["final_state_sha256"],
                },
            )
            anchor = run["anchor"]
            control_spec = dict(bundle.record_contract["records"]["control_runs"])
            control = _record(
                control_spec,
                {
                    "scenario_member_id": member["scenario_member_id"],
                    "split_assignment_id": split["split_assignment_id"],
                    "action_id": action.action_id,
                    "action_command_sha256": action.command_sha256,
                    "anchor_completed_step": ANCHOR_STEP,
                    "nonce_commitment_sha256": _sha256(run["nonce"]),
                    "control_run_id": run["control_run_id"],
                    "authority_epoch": run["authority_epoch"],
                    "hmc_binding_sha256": bundle.binding_sha256,
                    "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
                    "observable_topology_sha256": bundle.topology.sha256,
                    "terminal_status": "COMPLETED",
                    "control_trace_record_id": trace_record["control_trace_record_id"],
                    "replay_witness_id": witness["replay_witness_id"],
                    "proposal_receipt_sha256": anchor[
                        "proposal_receipt"
                    ].proposal_receipt_sha256,
                    "arbitration_receipt_sha256": anchor[
                        "arbitration"
                    ].arbitration_receipt_sha256,
                    "arbitration_disposition": anchor["arbitration"].to_mapping()[
                        "disposition"
                    ],
                    "final_command_sha256": anchor["arbitration"].final_command_sha256,
                    "anchor_step_receipt_sha256": anchor[
                        "step_receipt"
                    ].step_receipt_sha256,
                },
            )
            pairs = [
                run["snapshots"][step]
                for step in range(ANCHOR_STEP - WINDOW_STEPS + 1, ANCHOR_STEP + 1)
            ]
            history = project_history_window(bundle, pairs, window_steps=WINDOW_STEPS)
            targets = project_physical_targets(
                bundle,
                [
                    run["states"][step]
                    for step in range(ANCHOR_STEP + 1, ANCHOR_STEP + HORIZON_STEPS + 1)
                ],
                horizon_steps=HORIZON_STEPS,
            )
            sample_spec = dict(bundle.record_contract["records"]["samples"])
            sample = _record(
                sample_spec,
                {
                    "family_cluster_id": cluster["family_cluster_id"],
                    "family_id": family["family_id"],
                    "scenario_member_id": member["scenario_member_id"],
                    "split_assignment_id": split["split_assignment_id"],
                    "split_label": "DEVELOPMENT",
                    "control_run_record_id": control["control_run_record_id"],
                    "replay_witness_id": witness["replay_witness_id"],
                    "action_id": action.action_id,
                    "action_command_sha256": action.command_sha256,
                    "anchor_snapshot_sha256": anchor["snapshot"].snapshot_sha256,
                    "anchor_completed_step": ANCHOR_STEP,
                    "window_steps": WINDOW_STEPS,
                    "horizon_steps": HORIZON_STEPS,
                    "history_steps": list(history.steps),
                    "target_steps": [17, 18],
                    "history_completed_times_s": list(history.completed_times_s),
                    "target_completed_times_s": [1020.0, 1080.0],
                    "input_contract_sha256": layout.input_manifest_sha256,
                    "target_contract_sha256": layout.target_manifest_sha256,
                    "input_tensors": {
                        "history_numeric": _array(history.numeric_f32),
                        "history_availability": _array(history.status_f32),
                        "history_mode_one_hot": _array(history.mode_f32),
                        "history_health_one_hot": _array(history.health_f32),
                        "history_alarm_lifecycle_one_hot": _array(
                            history.alarm_lifecycle_f32
                        ),
                        "history_final_command": _array(history.numeric_f32[:, 167:]),
                        "proposed_action": _array(
                            project_proposed_action(bundle, action.command)
                        ),
                    },
                    "target_truth": _array(targets),
                    "evaluator_only_provenance": {
                        "proposal_receipt_sha256": anchor[
                            "proposal_receipt"
                        ].proposal_receipt_sha256,
                        "requested_command_sha256": action.command_sha256,
                        "arbitration_receipt_sha256": anchor[
                            "arbitration"
                        ].arbitration_receipt_sha256,
                        "arbitration_disposition": anchor["arbitration"].to_mapping()[
                            "disposition"
                        ],
                        "final_command_sha256": anchor[
                            "arbitration"
                        ].final_command_sha256,
                        "anchor_step_receipt_sha256": anchor[
                            "step_receipt"
                        ].step_receipt_sha256,
                        "plant_receipt_digest": anchor["shadow_digest"],
                    },
                },
            )
            tables["control_runs"].append(control)
            tables["control_traces"].append(trace_record)
            tables["replay_witnesses"].append(witness)
            tables["samples"].append(sample)
            traces[trace_record["control_trace_record_id"]] = run[
                "trace"
            ].canonical_bytes
        validate_lineage(tables)
        hashes = _write_packet(staging, bundle, tables, traces)
        validate_development_packet(staging, bundle)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "release_tier": RELEASE_TIER,
        "packet_root": str(destination),
        "sample_count": 4,
        "control_run_count": 4,
        "shadow_receipt_matches": 96,
        "strict_trace_replays": 4,
        "file_sha256": hashes,
    }


def _expect_fields(
    actual: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    for field, value in expected.items():
        if actual.get(field) != value:
            raise ForecastPipelineError(f"{label}.{field} drifts from frozen evidence")


def _validate_fixture_semantics(
    bundle: ForecastContracts,
    tables: dict[str, list[dict[str, Any]]],
    trace_bytes: dict[str, bytes],
) -> None:
    expected_counts = {
        "family_clusters": 1,
        "families": 1,
        "scenario_members": 1,
        "split_assignments": 1,
        "control_runs": len(bundle.actions),
        "control_traces": len(bundle.actions),
        "replay_witnesses": len(bundle.actions),
        "samples": len(bundle.actions),
    }
    for name, expected in expected_counts.items():
        if len(tables[name]) != expected:
            raise ForecastPipelineError(
                f"{name} row count drifts from the frozen fixture"
            )

    cluster = tables["family_clusters"][0]
    family = tables["families"][0]
    member = tables["scenario_members"][0]
    split = tables["split_assignments"][0]
    _expect_fields(
        cluster,
        {
            "stratum": "constant-occupied/no-treatment",
            "generator_contract_sha256": bundle.development_record_contract_sha256,
            "development_profile_sha256": bundle.development_profile_sha256,
        },
        label="family_cluster",
    )
    _expect_fields(
        family,
        {
            "family_cluster_id": cluster["family_cluster_id"],
            "family_role": "HEALTHY",
            "treatment_kind": "NONE",
            "scenario_member_ids": [member["scenario_member_id"]],
        },
        label="family",
    )
    _expect_fields(
        member,
        {
            "family_id": family["family_id"],
            "scenario_path": "scenarios/habitat_v2_forecast_development.json",
            "scenario_sha256": bundle.development_scenario.scenario_sha256,
            "plant_run_id": bundle.development_scenario.run_id,
            "profile_manifest_sha256": bundle.development_profile_sha256,
        },
        label="scenario_member",
    )
    expected_split = assign_cluster_splits(
        [cluster],
        split_key=_SPLIT_KEY,
        split_policy_sha256=_SPLIT_POLICY_SHA256,
        split_key_id=_SPLIT_KEY_ID,
        development_only=True,
    )[0]
    _expect_fields(split, expected_split, label="split_assignment")

    control_by_action = {row["action_id"]: row for row in tables["control_runs"]}
    sample_by_action = {row["action_id"]: row for row in tables["samples"]}
    trace_by_id = {
        row["control_trace_record_id"]: row for row in tables["control_traces"]
    }
    witness_by_id = {
        row["replay_witness_id"]: row for row in tables["replay_witnesses"]
    }
    action_ids = {action.action_id for action in bundle.actions}
    if set(control_by_action) != action_ids or set(sample_by_action) != action_ids:
        raise ForecastPipelineError("action-conditioned fixture coverage drifts")

    layout = forecast_layout(bundle)
    for action in bundle.actions:
        run = _execute_run(bundle, action)
        anchor = run["anchor"]
        control = control_by_action[action.action_id]
        trace = trace_by_id[control["control_trace_record_id"]]
        witness = witness_by_id[control["replay_witness_id"]]
        sample = sample_by_action[action.action_id]
        expected_trace_bytes = run["trace"].canonical_bytes
        if trace_bytes.get(trace["control_trace_record_id"]) != expected_trace_bytes:
            raise ForecastPipelineError(
                "control trace bytes drift from deterministic replay"
            )
        _expect_fields(
            trace,
            {
                "control_run_id": run["control_run_id"],
                "relative_trace_path": (
                    "development-fixture-only/traces/"
                    f"{trace['control_trace_record_id']}.json"
                ),
                "byte_length": len(expected_trace_bytes),
                "control_trace_sha256": _sha256(expected_trace_bytes),
                "control_trace_footer_sha256": run["trace_footer"][
                    "control_trace_footer_sha256"
                ],
                "final_state_sha256": run["trace_footer"]["final_state_sha256"],
            },
            label="control_trace",
        )
        _expect_fields(
            witness,
            {
                "control_run_id": run["control_run_id"],
                "control_trace_record_id": trace["control_trace_record_id"],
                "transition_count": len(run["witnesses"]),
                "step_witnesses": run["witnesses"],
                "step_witnesses_sha256": _sha256(
                    canonical_json_bytes(run["witnesses"])
                ),
                "final_state_sha256": run["trace_footer"]["final_state_sha256"],
            },
            label="replay_witness",
        )
        _expect_fields(
            control,
            {
                "scenario_member_id": member["scenario_member_id"],
                "split_assignment_id": split["split_assignment_id"],
                "action_command_sha256": action.command_sha256,
                "anchor_completed_step": ANCHOR_STEP,
                "nonce_commitment_sha256": _sha256(run["nonce"]),
                "control_run_id": run["control_run_id"],
                "authority_epoch": run["authority_epoch"],
                "hmc_binding_sha256": bundle.binding_sha256,
                "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
                "observable_topology_sha256": bundle.topology.sha256,
                "terminal_status": "COMPLETED",
                "control_trace_record_id": trace["control_trace_record_id"],
                "replay_witness_id": witness["replay_witness_id"],
                "proposal_receipt_sha256": anchor[
                    "proposal_receipt"
                ].proposal_receipt_sha256,
                "arbitration_receipt_sha256": anchor[
                    "arbitration"
                ].arbitration_receipt_sha256,
                "arbitration_disposition": anchor["arbitration"].to_mapping()[
                    "disposition"
                ],
                "final_command_sha256": anchor["arbitration"].final_command_sha256,
                "anchor_step_receipt_sha256": anchor[
                    "step_receipt"
                ].step_receipt_sha256,
            },
            label="control_run",
        )

        pairs = [
            run["snapshots"][step]
            for step in range(ANCHOR_STEP - WINDOW_STEPS + 1, ANCHOR_STEP + 1)
        ]
        history = project_history_window(bundle, pairs, window_steps=WINDOW_STEPS)
        target_states = [
            run["states"][step]
            for step in range(ANCHOR_STEP + 1, ANCHOR_STEP + HORIZON_STEPS + 1)
        ]
        targets = project_physical_targets(
            bundle, target_states, horizon_steps=HORIZON_STEPS
        )
        expected_tensors = {
            "history_numeric": _array(history.numeric_f32),
            "history_availability": _array(history.status_f32),
            "history_mode_one_hot": _array(history.mode_f32),
            "history_health_one_hot": _array(history.health_f32),
            "history_alarm_lifecycle_one_hot": _array(history.alarm_lifecycle_f32),
            "history_final_command": _array(history.numeric_f32[:, 167:]),
            "proposed_action": _array(project_proposed_action(bundle, action.command)),
        }
        expected_provenance = {
            "proposal_receipt_sha256": anchor[
                "proposal_receipt"
            ].proposal_receipt_sha256,
            "requested_command_sha256": action.command_sha256,
            "arbitration_receipt_sha256": anchor[
                "arbitration"
            ].arbitration_receipt_sha256,
            "arbitration_disposition": anchor["arbitration"].to_mapping()[
                "disposition"
            ],
            "final_command_sha256": anchor["arbitration"].final_command_sha256,
            "anchor_step_receipt_sha256": anchor["step_receipt"].step_receipt_sha256,
            "plant_receipt_digest": anchor["shadow_digest"],
        }
        _expect_fields(
            sample,
            {
                "family_cluster_id": cluster["family_cluster_id"],
                "family_id": family["family_id"],
                "scenario_member_id": member["scenario_member_id"],
                "split_assignment_id": split["split_assignment_id"],
                "split_label": "DEVELOPMENT",
                "control_run_record_id": control["control_run_record_id"],
                "replay_witness_id": witness["replay_witness_id"],
                "action_command_sha256": action.command_sha256,
                "anchor_snapshot_sha256": anchor["snapshot"].snapshot_sha256,
                "anchor_completed_step": ANCHOR_STEP,
                "window_steps": WINDOW_STEPS,
                "horizon_steps": HORIZON_STEPS,
                "history_steps": list(history.steps),
                "target_steps": [17, 18],
                "history_completed_times_s": list(history.completed_times_s),
                "target_completed_times_s": [1020.0, 1080.0],
                "input_contract_sha256": layout.input_manifest_sha256,
                "target_contract_sha256": layout.target_manifest_sha256,
                "input_tensors": expected_tensors,
                "target_truth": _array(targets),
                "evaluator_only_provenance": expected_provenance,
            },
            label="sample",
        )


def validate_development_packet(
    packet_root: str | Path, bundle: ForecastContracts | None = None
) -> dict[str, Any]:
    root = Path(packet_root).resolve()
    if bundle is None:
        raise ForecastPipelineError(
            "validation requires the frozen forecast contract bundle"
        )
    specs = bundle.record_contract["records"]
    nested = bundle.record_contract["nested_contracts"]
    tables: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] | None = None
    try:
        for name, specification in specs.items():
            relative = validate_relative_packet_path(specification["path"])
            full_path = root / relative
            if name == "manifest":
                raw = full_path.read_bytes()
                try:
                    value = json.loads(raw.decode("utf-8"))
                except Exception as error:
                    raise ForecastPipelineError("manifest cannot be decoded") from error
                if canonical_json_bytes(value) != raw:
                    raise ForecastPipelineError("manifest bytes are not canonical")
                manifest = validate_record(
                    value,
                    specification,
                    self_hash_field="manifest_sha256",
                    nested_contracts=nested,
                )
            else:
                tables[name] = validate_jsonl_records(
                    full_path.read_bytes(),
                    specification,
                    nested_contracts=nested,
                )
        validate_lineage(tables)
    except CorpusValidationError as error:
        raise ForecastPipelineError(str(error)) from error
    except (OSError, KeyError, TypeError) as error:
        raise ForecastPipelineError("packet structure cannot be verified") from error
    if manifest is None:
        raise ForecastPipelineError("manifest is missing")

    _expect_fields(
        manifest,
        {
            "record_contract_sha256": bundle.development_record_contract_sha256,
            "hmc_binding_sha256": bundle.binding_sha256,
            "alarm_manifest_sha256": bundle.alarm_manifest_sha256,
            "action_catalogue_sha256": bundle.action_catalogue_sha256,
            "development_profile_sha256": bundle.development_profile_sha256,
            "scenario_sha256": bundle.development_scenario.scenario_sha256,
            "window_steps": WINDOW_STEPS,
            "horizon_steps": HORIZON_STEPS,
            "packet_root": "development-fixture-only",
            "gate_status": "STOP_UNDERPOWERED",
            "selection_allowed": False,
            "final_set_present": False,
            "sealed": True,
        },
        label="manifest",
    )

    expected_table_paths = {
        specification["path"]
        for name, specification in specs.items()
        if name != "manifest"
    }
    artifact_by_path = {
        artifact["relative_path"]: artifact for artifact in manifest["table_artifacts"]
    }
    if (
        len(artifact_by_path) != len(manifest["table_artifacts"])
        or set(artifact_by_path) != expected_table_paths
    ):
        raise ForecastPipelineError("manifest table artifact set is not closed")
    for name, specification in specs.items():
        if name == "manifest":
            continue
        relative = specification["path"]
        artifact = artifact_by_path[relative]
        raw = (root / relative).read_bytes()
        if (
            artifact["row_count"] != len(tables[name])
            or artifact["byte_length"] != len(raw)
            or artifact["sha256"] != _sha256(raw)
        ):
            raise ForecastPipelineError("table artifact hash/length/count drifts")

    trace_by_id = {
        row["control_trace_record_id"]: row for row in tables["control_traces"]
    }
    trace_artifact_by_id = {
        artifact["control_trace_record_id"]: artifact
        for artifact in manifest["trace_artifacts"]
    }
    if len(trace_artifact_by_id) != len(manifest["trace_artifacts"]) or set(
        trace_artifact_by_id
    ) != set(trace_by_id):
        raise ForecastPipelineError("manifest trace artifact set is not closed")
    trace_bytes: dict[str, bytes] = {}
    for record_id, row in trace_by_id.items():
        artifact = trace_artifact_by_id[record_id]
        relative = validate_relative_packet_path(artifact["relative_path"])
        raw = (root / relative).read_bytes()
        if (
            row["relative_trace_path"] != relative
            or artifact["byte_length"] != len(raw)
            or artifact["sha256"] != _sha256(raw)
            or row["control_trace_sha256"] != _sha256(raw)
        ):
            raise ForecastPipelineError("trace artifact binding drifts")
        parsed = parse_control_trace(
            raw,
            scenario=bundle.development_scenario,
            contract=bundle.hmc_contract,
        )
        replay = replay_control_trace(
            raw,
            scenario=bundle.development_scenario,
            contract=bundle.hmc_contract,
        )
        if (
            parsed.footer["terminal_status"] != "COMPLETED"
            or replay.committed_step_count != 24
            or row["final_state_sha256"] != replay.final_state_sha256
        ):
            raise ForecastPipelineError("trace completion or replay boundary fails")
        trace_bytes[record_id] = raw

    expected_files = {specification["path"] for specification in specs.values()}
    expected_files.update(row["relative_trace_path"] for row in trace_by_id.values())
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ForecastPipelineError("packet file set is not closed")

    _validate_fixture_semantics(bundle, tables, trace_bytes)
    return {
        "release_tier": RELEASE_TIER,
        "sample_count": len(tables["samples"]),
        "control_run_count": len(tables["control_runs"]),
        "shadow_receipt_matches": sum(
            row["transition_count"] for row in tables["replay_witnesses"]
        ),
        "strict_trace_replays": len(trace_by_id),
    }
