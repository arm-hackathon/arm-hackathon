from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.control_trace import parse_control_trace, replay_control_trace
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast.pipeline import (
    generate_development_fixture,
    validate_development_packet,
)
from aeolus.habitat_v2.forecast.projection import forecast_layout
from aeolus.habitat_v2.forecast.timing import (
    emit_baseline_gate_receipt,
    emit_timing_receipt,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import canonical_json_bytes, load_hmc_contract
from aeolus.habitat_v2.physics import (
    advance_one_step_with_command,
    initial_state,
    validate_external_command,
)
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.telemetry import derive_observable_topology

RELEASE_TIER = "DEVELOPMENT_FIXTURE_ONLY"
FINAL_HMC_COMMIT_SHA = "3bc5da3d716212cac6524b088a963b6abf47a0ef"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_self_hash(
    mapping: dict[str, Any],
    *,
    field: str,
    label: str,
) -> None:
    if mapping.get("release_tier") != RELEASE_TIER:
        raise ValueError(f"{label} release tier mismatch")
    claimed = mapping.get(field)
    body = dict(mapping)
    body.pop(field, None)
    if _canonical_sha256(body) != claimed:
        raise ValueError(f"{label} self-hash mismatch")


def _verify_source_manifest(root: Path, binding: dict[str, Any]) -> None:
    commit_sha = binding["final_hmc_commit_sha"]
    if commit_sha != FINAL_HMC_COMMIT_SHA:
        raise ValueError("final HMC commit mismatch")
    for entry in binding["hmc_source_files"]:
        path = entry["path"]
        raw = subprocess.check_output(
            ["git", "show", f"{commit_sha}:{path}"],
            cwd=root,
        )
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"source SHA-256 mismatch: {path}")
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{commit_sha}:{path}"],
            cwd=root,
            text=True,
        ).strip()
        if blob != entry["git_blob_sha1"]:
            raise ValueError(f"Git blob mismatch: {path}")


def verify_frozen_inputs(root: Path) -> dict[str, Any]:
    contract_root = root / "contracts"
    paths = {
        "binding": (
            contract_root / "habitat_v2_forecast_hmc_binding_v2.json",
            "binding_sha256",
        ),
        "alarms": (
            contract_root / "habitat_v2_forecast_alarm_manifest_v1.json",
            "alarm_manifest_sha256",
        ),
        "actions": (
            contract_root / "habitat_v2_forecast_action_catalogue_v1.json",
            "catalogue_sha256",
        ),
        "profile": (
            contract_root / "habitat_v2_forecast_development_profile_v1.json",
            "profile_manifest_sha256",
        ),
        "records": (
            contract_root / "habitat_v2_forecast_development_records_v1.json",
            "record_contract_sha256",
        ),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for label, (path, field) in paths.items():
        mapping = _load_json(path)
        _verify_self_hash(mapping, field=field, label=label)
        loaded[label] = mapping

    binding = loaded["binding"]
    _verify_source_manifest(root, binding)

    profile = loaded["profile"]
    source = _load_json(root / profile["source_scenario_path"])
    fixture = _load_json(root / profile["fixture_scenario_path"])
    if _canonical_sha256(source) != profile["source_scenario_sha256"]:
        raise ValueError("source scenario identity mismatch")
    if _canonical_sha256(fixture) != profile["fixture_scenario_sha256"]:
        raise ValueError("development scenario identity mismatch")
    changed = {
        key for key in set(source) | set(fixture) if source.get(key) != fixture.get(key)
    }
    if changed != {"name", "steps", "fault_profiles", "timeline"}:
        raise ValueError("development scenario changed outside the allowlist")

    scenario = Scenario.from_mapping(fixture)
    hmc_contract = load_hmc_contract(contract_root / "habitat_v2_hmc_v1.json")
    if hmc_contract.hmc_contract_sha256 != binding["hmc_contract_sha256"]:
        raise ValueError("HMC contract identity mismatch")
    topology = derive_observable_topology(scenario)
    if topology.sha256 != binding["observable_topology_sha256"]:
        raise ValueError("observable topology identity mismatch")

    records = loaded["records"]
    if len(records["records"]) != 9:
        raise ValueError("development record type count mismatch")
    for label, specification in records["records"].items():
        if not specification["path"].startswith("development-fixture-only/"):
            raise ValueError(f"record path is not development-only: {label}")
        fields = specification["required_fields"]
        if len(fields) != len(set(fields)):
            raise ValueError(f"duplicate required record field: {label}")

    alarms = loaded["alarms"]
    actions = loaded["actions"]
    if len(alarms["alarm_slots"]) != 287:
        raise ValueError("alarm-slot count mismatch")
    if len(actions["actions"]) != 4:
        raise ValueError("action count mismatch")
    for action in actions["actions"]:
        command = validate_external_command(scenario, action["command"])
        if command.sha256 != action["command_sha256"]:
            raise ValueError(f"action command identity mismatch: {action['action_id']}")

    return {
        "binding_sha256": binding["binding_sha256"],
        "source_file_count": len(binding["hmc_source_files"]),
        "alarm_manifest_sha256": alarms["alarm_manifest_sha256"],
        "alarm_slot_count": len(alarms["alarm_slots"]),
        "action_catalogue_sha256": actions["catalogue_sha256"],
        "action_count": len(actions["actions"]),
        "development_profile_sha256": profile["profile_manifest_sha256"],
        "record_contract_sha256": records["record_contract_sha256"],
        "record_type_count": len(records["records"]),
        "scenario_sha256": scenario.scenario_sha256,
    }


def _proposal_mapping(
    *,
    hmc: HabitatManagementComputer,
    snapshot_sha256: str,
    completed_step: int,
    action: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": action["action_id"],
        "source_type": "forecast_data_generator",
        "completed_observation_step": completed_step,
        "observation_snapshot_sha256": snapshot_sha256,
        "requested_application_step": completed_step,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": action["command"],
        "confidence": None,
    }
    return {**body, "proposal_sha256": _canonical_sha256(body)}


def verify_development_hmc(root: Path) -> dict[str, Any]:
    binding = _load_json(root / "contracts" / "habitat_v2_forecast_hmc_binding_v2.json")
    actions = _load_json(
        root / "contracts" / "habitat_v2_forecast_action_catalogue_v1.json"
    )
    scenario = Scenario.from_mapping(
        _load_json(root / "scenarios" / "habitat_v2_forecast_development.json")
    )
    contract = load_hmc_contract(root / "contracts" / "habitat_v2_hmc_v1.json")
    steps = int(scenario.data["steps"])
    matched_receipts = 0
    trace_sha256: list[str] = []

    for index, action in enumerate(actions["actions"]):
        hmc = HabitatManagementComputer.reset(
            scenario,
            contract,
            bytes([index + 1]) * 32,
        )
        shadow_state = initial_state(scenario)
        for _ in range(steps):
            observed = hmc.observe()
            if type(observed) is not tuple:
                raise ValueError("HMC entered terminal failure during observation")
            snapshot, verification = observed
            handle = hmc.verify_snapshot(snapshot, verification)
            proposal = None
            if verification.completed_step == 16:
                proposal = _proposal_mapping(
                    hmc=hmc,
                    snapshot_sha256=snapshot.snapshot_sha256,
                    completed_step=verification.completed_step,
                    action=action,
                )
            proposal_receipt = hmc.propose(proposal, handle)
            arbitration = hmc.arbitrate()
            if not hasattr(arbitration, "final_command"):
                raise ValueError("HMC entered terminal failure during arbitration")
            if verification.completed_step == 16:
                proposal_evidence = proposal_receipt.to_mapping()
                if (
                    proposal_evidence["attempt_class"] != "CANONICAL_PROPOSAL"
                    or proposal_evidence["validation_outcome"] != "VALID"
                ):
                    raise ValueError("anchor proposal was not admitted")
            step_receipt = hmc.step()
            if not hasattr(step_receipt, "plant_receipt_digest"):
                raise ValueError("HMC entered terminal failure during stepping")
            shadow_result = advance_one_step_with_command(
                scenario,
                shadow_state,
                arbitration.final_command,
            )
            shadow_receipt_digest = hashlib.sha256(
                canonical_json_bytes(shadow_result.receipt)
            ).hexdigest()
            if shadow_receipt_digest != step_receipt.plant_receipt_digest:
                raise ValueError("shadow and HMC plant receipts diverged")
            shadow_state = shadow_result.state
            matched_receipts += 1

        trace_bytes = hmc.export_control_trace(
            binding["final_hmc_commit_sha"]
        ).canonical_bytes
        parsed = parse_control_trace(
            trace_bytes,
            scenario=scenario,
            contract=contract,
        )
        replayed = replay_control_trace(
            trace_bytes,
            scenario=scenario,
            contract=contract,
        )
        if parsed.footer["terminal_status"] != "COMPLETED":
            raise ValueError("development trace is incomplete")
        if replayed.committed_step_count != steps:
            raise ValueError("development replay step count mismatch")
        trace_sha256.append(hashlib.sha256(trace_bytes).hexdigest())

    return {
        "action_runs": len(actions["actions"]),
        "transitions_per_run": steps,
        "shadow_receipts_matched": matched_receipts,
        "strict_trace_replays": len(trace_sha256),
        "trace_sha256": trace_sha256,
    }


def verify_development_packet_and_stop_receipts(root: Path) -> dict[str, Any]:
    bundle = load_forecast_contracts(root)
    layout = forecast_layout(bundle)
    with tempfile.TemporaryDirectory(prefix="aeolus-forecast-d1-") as temporary:
        output_root = Path(temporary)
        first = generate_development_fixture(root, output_root, "packet-a")
        second = generate_development_fixture(root, output_root, "packet-b")
        if first["file_sha256"] != second["file_sha256"]:
            raise ValueError("development packet generation is not byte-identical")
        validation = validate_development_packet(output_root / "packet-a", bundle)

    timing_evidence = {
        "development_sample_count": validation["sample_count"],
        "development_family_cluster_count": 1,
        "train_family_cluster_count": 0,
        "validation_family_cluster_count": 0,
        "candidate_window_steps": 4,
        "candidate_horizon_steps": 2,
        "selection_performed": False,
    }
    baseline_evidence = {
        "development_sample_count": validation["sample_count"],
        "train_family_cluster_count": 0,
        "validation_family_cluster_count": 0,
        "fitted_baseline_count": 0,
        "action_information_comparison_supported": False,
        "reason": "development fixture is not training or validation evidence",
    }
    timing_receipt = emit_timing_receipt(
        4,
        2,
        timing_evidence=timing_evidence,
        input_manifest_sha256=layout.input_manifest_sha256,
        target_manifest_sha256=layout.target_manifest_sha256,
    )
    baseline_receipt = emit_baseline_gate_receipt(
        baseline_evidence=baseline_evidence,
        input_manifest_sha256=layout.input_manifest_sha256,
        target_manifest_sha256=layout.target_manifest_sha256,
    )
    if (
        timing_receipt.outcome != "STOP_UNDERPOWERED"
        or baseline_receipt.outcome != "STOP_UNDERPOWERED"
    ):
        raise ValueError("D1 stop receipt outcome drift")
    return {
        "byte_identical_generation_count": 2,
        "closed_file_count": len(first["file_sha256"]),
        "manifest_file_sha256": first["file_sha256"][
            "development-fixture-only/manifest.json"
        ],
        "validation": validation,
        "timing_receipt": asdict(timing_receipt),
        "baseline_receipt": asdict(baseline_receipt),
    }


def verify_markdown_links(root: Path) -> int:
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.md"],
        cwd=root,
        text=True,
    ).splitlines()
    missing: list[tuple[str, str]] = []
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for relative_path in sorted(set(tracked)):
        path = root / relative_path
        for match in link_pattern.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1).split("#", 1)[0]
            if (
                target
                and not re.match(r"^[a-z]+://", target)
                and not (path.parent / target).resolve().exists()
            ):
                missing.append((relative_path, target))
    if missing:
        raise ValueError(f"missing Markdown links: {missing}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Habitat V2 Forecast D1 development foundation."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    root = arguments.repo_root.resolve()
    result = {
        "schema_version": "aeolus_habitat_v2_forecast_d1_verification_v1",
        "release_tier": RELEASE_TIER,
        "status": "PASS",
        "selection_allowed": False,
        "gate_status": "STOP_UNDERPOWERED",
        "frozen_inputs": verify_frozen_inputs(root),
        "hmc_execution": verify_development_hmc(root),
        "development_packet": verify_development_packet_and_stop_receipts(root),
        "markdown_missing_links": verify_markdown_links(root),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
