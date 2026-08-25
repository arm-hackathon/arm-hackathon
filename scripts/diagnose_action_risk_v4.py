#!/usr/bin/env python3
"""Recompute V4 diagnostics from a frozen Issue #56 V3 receipt.

The adapter is intentionally analysis-only.  It does not train, tune, export,
or run a new learned model.  It independently validates serialized V3 episode
digests and strictly replays every serialized episode trace before emitting the
separate candidate and executed-command metrics required by the V4 draft.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.forecast_issue55_race import (
    build_family_scenario,
    deterministic_family_ids,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    EPISODE_STEPS,
    ISSUE56_V3_SCHEMA_VERSION,
    V3RiskModel,
    load_v3_samples,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_diagnostics import (
    ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION,
    V4CandidateObservation,
    V4ExecutedObservation,
    candidate_screening_metrics,
    equal_weight_group_mean,
    executed_action_metrics,
    observation_manifest_sha256,
    provenance_manifest_sha256,
    validate_condition_groups,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import (
    FEATURE_COUNT,
    HISTORY_WINDOW_STEPS,
    v2_decision_steps,
)
from aeolus.habitat_v2.control_trace import parse_control_trace, replay_control_trace


REPO_ROOT = Path(__file__).resolve().parents[1]
POINT_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts" / "demo-only" / "habitat-v2-forecast" / "action-aware-mlp-v1.npz"
)
POINT_ARTIFACT_SHA256 = (
    "a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd"
)
V4_SOURCE_PATHS = (
    Path("contracts/habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1.json"),
    Path("scripts/diagnose_action_risk_v4.py"),
    Path("src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_diagnostics.py"),
)
V3_EPISODE_ARMS = {
    "rules_only_common_window",
    "point_model_common_window",
    "risk_only_v3",
    "risk_filtered_point_v3",
}


class V4DiagnosticRunError(RuntimeError):
    """Raised when a historical V3 receipt cannot be audited safely."""


def _sha(value: object) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError) as error:
        raise V4DiagnosticRunError("diagnostic value is not canonical JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_identity() -> dict[str, Any]:
    files = []
    for relative in V4_SOURCE_PATHS:
        path = REPO_ROOT / relative
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise V4DiagnosticRunError(f"V4 source identity cannot read {relative}") from error
        files.append(
            {
                "relative_path": relative.as_posix(),
                "byte_length": len(raw),
                "sha256": _sha_bytes(raw),
            }
        )
    return {"schema_version": "issue56-v4-source-identity-v1", "source_files": files}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4DiagnosticRunError(f"cannot read JSON object {path}") from error
    if type(value) is not dict:
        raise V4DiagnosticRunError(f"JSON value must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise V4DiagnosticRunError(f"cannot read JSONL file {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise V4DiagnosticRunError(f"invalid JSONL row {path}:{line_number}") from error
        if type(value) is not dict:
            raise V4DiagnosticRunError(f"JSONL row is not an object {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise V4DiagnosticRunError(f"JSONL file is empty: {path}")
    return rows


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=2,
    ).encode("utf-8")
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _resolve_input(path: Path) -> Path:
    resolved = path.resolve()
    out_root = (REPO_ROOT / "out").resolve()
    if out_root not in resolved.parents:
        raise V4DiagnosticRunError("V3 input must be below repository out/")
    if not resolved.is_dir():
        raise V4DiagnosticRunError("V3 input must be an existing directory")
    return resolved


def _resolve_output(path: Path) -> Path:
    resolved = path.resolve()
    out_root = (REPO_ROOT / "out").resolve()
    if out_root not in resolved.parents:
        raise V4DiagnosticRunError("V4 output must be below repository out/")
    if resolved.exists():
        raise V4DiagnosticRunError("V4 output directory must be new and write-once")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _condition_group(family_id: str, family_index: Mapping[str, int]) -> str:
    try:
        index = family_index[family_id]
    except KeyError as error:
        raise V4DiagnosticRunError(f"unknown V3 family identity: {family_id}") from error
    return f"condition-group-{index // 2:04d}"


def _make_candidate_observation(
    family_id: str,
    decision_step: int,
    action_id: str,
    model_rejected: bool,
    dangerous: bool,
    family_index: Mapping[str, int],
) -> V4CandidateObservation:
    body = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.candidate",
        "condition_group_id": _condition_group(family_id, family_index),
        "family_id": family_id,
        "decision_step": decision_step,
        "action_id": action_id,
        "model_rejected": model_rejected,
        "dangerous": dangerous,
    }
    return V4CandidateObservation(
        body["condition_group_id"],
        family_id,
        decision_step,
        action_id,
        model_rejected,
        dangerous,
        _sha(body),
    )


def _make_executed_observation(
    episode: Mapping[str, Any],
    decision: Mapping[str, Any],
    family_index: Mapping[str, int],
) -> V4ExecutedObservation:
    family_id = str(episode["family_id"])
    selected = decision["selected_action_id"]
    requested = decision["requested_command_sha256"]
    body = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.executed",
        "condition_group_id": _condition_group(family_id, family_index),
        "family_id": family_id,
        "arm": episode["arm"],
        "decision_step": decision["decision_step"],
        "selected_action_id": selected,
        "actual_dangerous": bool(decision["remaining_metric"]["crossing_event"] > 0.5),
        "requested_command_sha256": requested,
        "final_command_sha256": decision["final_command_sha256"],
        "executed_command_sha256": decision["executed_command_sha256"],
        "disposition": decision["disposition"],
    }
    return V4ExecutedObservation(
        body["condition_group_id"],
        family_id,
        str(episode["arm"]),
        int(decision["decision_step"]),
        selected,
        body["actual_dangerous"],
        requested,
        decision["final_command_sha256"],
        decision["executed_command_sha256"],
        decision["disposition"],
        _sha(body),
    )


def _verify_decision_digest(decision: Mapping[str, Any]) -> None:
    if type(decision) is not dict or "decision_sha256" not in decision:
        raise V4DiagnosticRunError("V3 decision digest is missing")
    body = dict(decision)
    digest = body.pop("decision_sha256")
    if type(digest) is not str or digest != _sha(body):
        raise V4DiagnosticRunError("V3 decision digest does not match serialized body")
    if decision["final_command_sha256"] != decision["executed_command_sha256"]:
        raise V4DiagnosticRunError("V3 final and executed command identities differ")
    selected = decision["selected_action_id"]
    if selected is None:
        if decision["requested_command_sha256"] is not None:
            raise V4DiagnosticRunError("V3 abstention carries a requested command")
        if decision["disposition"] != "ABSTAINED_TO_HOLD":
            raise V4DiagnosticRunError("V3 abstention disposition is inconsistent")
    elif decision["requested_command_sha256"] is None:
        raise V4DiagnosticRunError("V3 selected action lacks a requested command")


def _verify_episode(
    episode: Mapping[str, Any],
    run_dir: Path,
    bundle: Any,
    family_index: Mapping[str, int],
) -> dict[str, Any]:
    if type(episode) is not dict:
        raise V4DiagnosticRunError("V3 episode must be an object")
    arm = episode.get("arm")
    family_id = episode.get("family_id")
    if arm not in V3_EPISODE_ARMS or family_id not in family_index:
        raise V4DiagnosticRunError("V3 episode arm or family identity is invalid")
    if episode.get("schema_version") != f"{ISSUE56_V3_SCHEMA_VERSION}.episode":
        raise V4DiagnosticRunError(f"V3 episode schema drifted: {arm}/{family_id}")
    if tuple(episode.get("decision_steps", ())) != v2_decision_steps():
        raise V4DiagnosticRunError(f"V3 decision schedule drifted: {arm}/{family_id}")
    body = dict(episode)
    digest = body.pop("episode_sha256", None)
    if type(digest) is not str or digest != _sha(body):
        raise V4DiagnosticRunError(f"V3 episode digest is invalid: {arm}/{family_id}")
    decisions = episode.get("decisions")
    if type(decisions) is not list or len(decisions) != len(v2_decision_steps()):
        raise V4DiagnosticRunError(f"V3 decision count is invalid: {arm}/{family_id}")
    if tuple(item.get("decision_step") for item in decisions) != v2_decision_steps():
        raise V4DiagnosticRunError(f"V3 decision ordering drifted: {arm}/{family_id}")
    for field in (
        "authority_verified",
        "provenance_verified",
        "replay_verified",
        "metrics_finite_verified",
        "proposal_admission_verified",
    ):
        if episode.get(field) is not True:
            raise V4DiagnosticRunError(f"V3 episode verification flag failed: {field}")
    proposal_count = sum(item["selected_action_id"] is not None for item in decisions)
    if episode.get("proposal_count") != proposal_count:
        raise V4DiagnosticRunError(f"V3 proposal count is inconsistent: {arm}/{family_id}")
    if episode.get("abstention_count") != len(decisions) - proposal_count:
        raise V4DiagnosticRunError(f"V3 abstention count is inconsistent: {arm}/{family_id}")
    if episode.get("admitted_proposal_count") != proposal_count:
        raise V4DiagnosticRunError(f"V3 admission count is inconsistent: {arm}/{family_id}")
    for decision in decisions:
        _verify_decision_digest(decision)

    scenario = build_family_scenario(bundle.development_scenario, family_index[family_id])
    if episode["scenario_sha256"] != scenario.scenario_sha256:
        raise V4DiagnosticRunError(f"V3 scenario identity drifted: {family_id}")
    trace_path = run_dir / "traces" / f"{arm}--{family_id}.json"
    try:
        trace_bytes = trace_path.read_bytes()
    except OSError as error:
        raise V4DiagnosticRunError(f"V3 trace is missing: {trace_path}") from error
    if _sha_bytes(trace_bytes) != episode["trace_sha256"]:
        raise V4DiagnosticRunError(f"V3 trace digest is invalid: {trace_path}")
    parsed = parse_control_trace(
        trace_bytes,
        scenario=scenario,
        contract=bundle.hmc_contract,
    )
    replay = replay_control_trace(
        trace_bytes,
        scenario=scenario,
        contract=bundle.hmc_contract,
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != EPISODE_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
        or replay.final_state_sha256 != episode["replay_final_state_sha256"]
    ):
        raise V4DiagnosticRunError(f"V3 strict replay failed: {trace_path}")
    if episode["replay_committed_steps"] != replay.committed_step_count:
        raise V4DiagnosticRunError(f"V3 replay step receipt is inconsistent: {trace_path}")
    return {
        "arm": arm,
        "family_id": family_id,
        "decision_count": len(decisions),
        "trace_sha256": episode["trace_sha256"],
        "replay_committed_steps": replay.committed_step_count,
        "replay_final_state_sha256": replay.final_state_sha256,
    }


def _group_executed_metrics(
    observations: Sequence[V4ExecutedObservation],
) -> dict[str, Any]:
    by_arm_group: dict[tuple[str, str], list[V4ExecutedObservation]] = defaultdict(list)
    for observation in observations:
        by_arm_group[(observation.arm, observation.condition_group_id)].append(observation)
    result: dict[str, Any] = {}
    for arm in sorted({observation.arm for observation in observations}):
        groups = {
            group: executed_action_metrics(rows)
            for (group_arm, group), rows in sorted(by_arm_group.items())
            if group_arm == arm
        }
        aggregate: dict[str, Any] = {}
        for metric in ("proposal_rate", "abstention_rate", "hmc_mismatch_rate"):
            values = {
                group: (metrics[metric],)
                for group, metrics in groups.items()
                if metrics[metric] is not None
            }
            if len(values) == len(groups):
                aggregate[metric] = equal_weight_group_mean(values)
            else:
                aggregate[metric] = {
                    "undefined_group_count": len(groups) - len(values),
                    "defined_group_count": len(values),
                }
        selected_values = {
            group: (metrics["selected_action_false_safe_rate"],)
            for group, metrics in groups.items()
            if metrics["selected_action_false_safe_rate"] is not None
        }
        aggregate["selected_action_false_safe_rate"] = (
            equal_weight_group_mean(selected_values)
            if selected_values
            else {"defined_group_count": 0, "undefined_group_count": len(groups)}
        )
        result[arm] = {
            "overall_decisions": executed_action_metrics(
                [item for (group_arm, _), rows in by_arm_group.items() if group_arm == arm for item in rows]
            ),
            "condition_groups": groups,
            "equal_weight": aggregate,
        }
    return result


def diagnose_v3_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = _resolve_input(run_dir)
    output_dir = _resolve_output(output_dir)
    input_manifest = _read_json(run_dir / "manifest.json")
    if input_manifest.get("family_ids") != list(deterministic_family_ids(32)):
        raise V4DiagnosticRunError("V4 diagnostic adapter requires the full 32-family V3 run")
    model = V3RiskModel.from_mapping(_read_json(run_dir / "model.json"))
    sample_rows = _read_jsonl(run_dir / "samples.jsonl")
    samples = load_v3_samples(sample_rows)
    evaluation_samples = tuple(sample for sample in samples if sample.split == "EVALUATION")
    if not evaluation_samples:
        raise V4DiagnosticRunError("V3 receipt has no EVALUATION samples")
    episodes = _read_jsonl(run_dir / "episodes.jsonl")
    bundle = load_forecast_contracts(REPO_ROOT)
    roster = deterministic_family_ids(32)
    family_index = {family_id: index for index, family_id in enumerate(roster)}
    actual_point_artifact_sha256 = _sha_bytes(POINT_ARTIFACT_PATH.read_bytes())
    if actual_point_artifact_sha256 != POINT_ARTIFACT_SHA256:
        raise V4DiagnosticRunError("current point artifact identity does not match the frozen artifact")
    if input_manifest.get("point_artifact_sha256") != actual_point_artifact_sha256:
        raise V4DiagnosticRunError("V3 receipt point artifact identity does not match the artifact")

    candidate_observations = tuple(
        _make_candidate_observation(
            sample.family_id,
            sample.decision_step,
            sample.action_id,
            model.predict_features(sample.features_f32).hard_ineligible,
            sample.label.remaining_metric.crossing_event > 0.5,
            family_index,
        )
        for sample in evaluation_samples
    )
    replay_receipts = tuple(
        _verify_episode(episode, run_dir, bundle, family_index) for episode in episodes
    )
    expected_evaluation_families = {sample.family_id for sample in evaluation_samples}
    observed_episode_keys = {(receipt["arm"], receipt["family_id"]) for receipt in replay_receipts}
    expected_episode_keys = {
        (arm, family_id)
        for family_id in expected_evaluation_families
        for arm in V3_EPISODE_ARMS
    }
    if observed_episode_keys != expected_episode_keys:
        raise V4DiagnosticRunError("V3 episode receipt is incomplete or has duplicate arms")
    executed_observations = tuple(
        _make_executed_observation(episode, decision, family_index)
        for episode in episodes
        for decision in episode["decisions"]
    )
    validate_condition_groups(candidate_observations)
    validate_condition_groups(executed_observations)

    samples_sha256 = _sha_bytes((run_dir / "samples.jsonl").read_bytes())
    risk_model_sha256 = _sha_bytes((run_dir / "model.json").read_bytes())
    scenario_manifest = {
        "schema_version": "issue56-v4-scenario-manifest-v1",
        "family_scenarios": {
            family_id: build_family_scenario(
                bundle.development_scenario,
                index,
            ).scenario_sha256
            for family_id, index in family_index.items()
        },
    }
    feature_manifest = {
        "schema_version": "issue56-v4-feature-manifest-legacy-v3-v1",
        "source_path": "src/aeolus/habitat_v2/forecast_issue56_action_risk_v2.py",
        "feature_count": FEATURE_COUNT,
        "history_window_steps": HISTORY_WINDOW_STEPS,
        "feature_contract_status": "legacy_v3_baseline_descriptor_only",
    }
    label_manifest = {
        "schema_version": "issue56-v4-label-manifest-v1",
        "source_samples_sha256": samples_sha256,
        "sample_count": len(evaluation_samples),
        "sample_sha256s": sorted(sample.sample_sha256 for sample in evaluation_samples),
        "counterfactual_trace_bytes_present": False,
    }
    v3_source_identity_sha256 = _sha(input_manifest["source_identity"])
    source_identity = _source_identity()
    source_identity_sha256 = _sha(source_identity)
    scenario_manifest_sha256 = _write_json(output_dir / "scenario-manifest.json", scenario_manifest)
    feature_manifest_sha256 = _write_json(output_dir / "feature-manifest.json", feature_manifest)
    label_manifest_sha256 = _write_json(output_dir / "label-manifest.json", label_manifest)
    provenance = {
        "source_identity_sha256": source_identity_sha256,
        "hmc_binding_sha256": bundle.binding_sha256,
        "hmc_contract_sha256": bundle.hmc_contract.hmc_contract_sha256,
        "scenario_sha256": scenario_manifest_sha256,
        "action_catalogue_sha256": bundle.action_catalogue_sha256,
        "alarm_manifest_sha256": bundle.alarm_manifest_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "label_manifest_sha256": label_manifest_sha256,
        "risk_model_sha256": risk_model_sha256,
        "point_artifact_sha256": input_manifest["point_artifact_sha256"],
    }
    provenance_sha256 = provenance_manifest_sha256(provenance)
    candidate_rows = [item.to_mapping() for item in candidate_observations]
    executed_rows = [item.to_mapping() for item in executed_observations]
    candidate_sha256 = _write_jsonl(output_dir / "candidate-observations.jsonl", candidate_rows)
    executed_sha256 = _write_jsonl(output_dir / "executed-observations.jsonl", executed_rows)
    diagnostics = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.report",
        "input_v3_run": str(run_dir.relative_to(REPO_ROOT)),
        "candidate_source_split": "EVALUATION",
        "candidate_sample_count": len(evaluation_samples),
        "candidate_screening": candidate_screening_metrics(candidate_observations),
        "executed_by_arm": _group_executed_metrics(executed_observations),
        "observation_manifests": {
            "candidate_observations_sha256": candidate_sha256,
            "candidate_rows_content_sha256": observation_manifest_sha256(candidate_observations),
            "executed_observations_sha256": executed_sha256,
            "executed_rows_content_sha256": observation_manifest_sha256(executed_observations),
        },
        "strict_episode_replay": {
            "episode_count": len(replay_receipts),
            "completed_episode_count": len(replay_receipts),
            "receipts": list(replay_receipts),
        },
        "counterfactual_label_replay": {
            "verified": False,
            "reason": "historical V3 samples retain label trace hashes but not trace bytes",
        },
        "ready_for_v4_model_study": False,
        "readiness_blockers": [
            "V4 protocol remains pending authorization",
            "historical V3 sample artifacts do not contain counterfactual trace bytes",
            "no V4 learned model is implemented or trained",
        ],
    }
    diagnostics_sha256 = _write_json(output_dir / "diagnostics.json", diagnostics)
    manifest = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.manifest",
        "input_v3_manifest_sha256": _sha_bytes((run_dir / "manifest.json").read_bytes()),
        "input_v3_source_identity_sha256": v3_source_identity_sha256,
        "input_v3_risk_model_sha256": risk_model_sha256,
        "input_v3_point_artifact_sha256": input_manifest.get("point_artifact_sha256"),
        "source_identity_sha256": source_identity_sha256,
        "source_identity": source_identity,
        "provenance": provenance,
        "provenance_manifest_sha256": provenance_sha256,
        "candidate_observations_sha256": candidate_sha256,
        "executed_observations_sha256": executed_sha256,
        "diagnostics_sha256": diagnostics_sha256,
        "strict_episode_replay": True,
        "counterfactual_label_replay": False,
        "ready_for_v4_model_study": False,
    }
    manifest_sha256 = _write_json(output_dir / "manifest.json", manifest)
    return {
        "output": str(output_dir),
        "manifest_sha256": manifest_sha256,
        "diagnostics_sha256": diagnostics_sha256,
        "candidate_count": len(candidate_observations),
        "executed_decision_count": len(executed_observations),
        "episode_count": len(replay_receipts),
        "ready_for_v4_model_study": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a V3 receipt with V4 diagnostics")
    parser.add_argument("--v3-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose_v3_run(args.v3_run, args.output)
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
