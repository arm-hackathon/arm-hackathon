from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes
from aeolus.habitat_v2.forecast.live_demo import (
    DEMO_RELEASE_TIER,
    LiveForecastError,
    load_live_ridge_model,
    run_live_forecast_demo,
)
from aeolus.habitat_v2.forecast.projection import forecast_layout

MODEL_SHA256 = "a6e4ef34fc837bb6539a84e20d015bbd7bbfe4e9fd5a6fc74e3f0217bd978d9a"
SOURCE_FOUNDATION_GIT_COMMIT = "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3"
_PAYLOAD_PATTERN = re.compile(rb"atob\('([A-Za-z0-9+/=]+)'\)")
_EXPECTED_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "release_tier",
        "source_foundation_git_commit",
        "integration_source_committed",
        "integration_source_git_commit",
        "qualification_evidence",
        "actuator_authority",
        "model_artifact_sha256",
        "control_trace_sha256",
        "source_files",
        "artifacts",
    }
)
_EXPECTED_ARTIFACT_PATHS = frozenset({"index.html", "live-run.json"})
_EXPECTED_SOURCE_PATHS = frozenset(
    {
        ".github/workflows/habitat-v2-live-forecast-arm64.yml",
        "artifacts/demo-only/habitat-v2-forecast/README.md",
        "artifacts/demo-only/habitat-v2-forecast/training-receipt.json",
        "artifacts/demo-only/habitat-v2-forecast/training-report.json",
        "scripts/run_habitat_v2_live_forecast.py",
        "scripts/verify_habitat_v2_live_forecast_demo.py",
        "src/aeolus/habitat_v2/forecast/live_demo.py",
        "src/aeolus/habitat_v2/forecast/live_demo_report.py",
        "tests/habitat_v2/test_forecast_live_demo.py",
    }
)
_EXPECTED_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "release_tier",
        "claims",
        "model",
        "timeline",
        "authority",
        "target_descriptors",
        "candidate_forecasts",
        "simulator_truth",
        "replay_evidence",
    }
)
_EXPECTED_MODEL_FIELDS = frozenset(
    {
        "kind",
        "artifact_sha256",
        "input_manifest_sha256",
        "target_manifest_sha256",
    }
)
_EXPECTED_TIMELINE_FIELDS = frozenset(
    {
        "forecast_completed_step",
        "forecast_completed_time_s",
        "history_steps",
        "truth_steps",
    }
)
_EXPECTED_AUTHORITY_FIELDS = frozenset(
    {
        "selection_source",
        "selected_action_id",
        "selected_command_sha256",
        "arbitration_disposition",
        "final_command_sha256",
        "actuator_authority",
    }
)
_EXPECTED_CANDIDATE_FIELDS = frozenset(
    {
        "action_id",
        "command_sha256",
        "proposed_action_f32",
        "prediction_f32",
    }
)
_EXPECTED_TRUTH_FIELDS = frozenset({"selected_action_id", "truth_f32"})
_EXPECTED_REPLAY_FIELDS = frozenset(
    {
        "control_run_id",
        "terminal_status",
        "trace_sha256",
        "trace_footer_sha256",
        "replay_final_state_sha256",
        "replay_committed_steps",
    }
)


class VerificationError(ValueError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_mapping(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise VerificationError(f"{label} fields drift")
    return value


def _finite_f32(
    value: object,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise VerificationError(f"{label} is malformed") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise VerificationError(f"{label} shape or values drift")
    return array


def _load_canonical_json_bytes(
    raw: bytes,
    *,
    label: str,
) -> tuple[dict[str, object], bytes]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise VerificationError(f"cannot load {label}") from error
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise VerificationError(f"{label} is not a canonical JSON object")
    return value, raw


def _load_canonical_json(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot load {path.name}") from error
    return _load_canonical_json_bytes(raw, label=path.name)


def _verify_bound_file(root: Path, item: object) -> tuple[str, bytes]:
    if type(item) is not dict or set(item) != {
        "relative_path",
        "byte_length",
        "sha256",
    }:
        raise VerificationError("receipt file entry is malformed")
    path = (root / str(item["relative_path"])).resolve()
    try:
        path.relative_to(root)
        raw = path.read_bytes()
    except (OSError, ValueError) as error:
        raise VerificationError("receipt file entry escapes its root") from error
    if len(raw) != item["byte_length"] or _sha256(raw) != item["sha256"]:
        raise VerificationError(f"receipt identity mismatch for {item['relative_path']}")
    return str(item["relative_path"]), raw


def _verify_file_set(
    root: Path,
    items: object,
    expected_paths: frozenset[str],
    *,
    label: str,
) -> dict[str, bytes]:
    if type(items) is not list:
        raise VerificationError(f"receipt {label} must be a list")
    bound_files = [_verify_bound_file(root, item) for item in items]
    paths = [relative_path for relative_path, _ in bound_files]
    if len(paths) != len(set(paths)) or frozenset(paths) != expected_paths:
        raise VerificationError(f"receipt {label} set drift")
    return dict(bound_files)


def verify_report(
    repo_root: Path,
    report: Path,
    model_path: Path,
    *,
    expected_integration_commit: str | None = None,
) -> dict[str, object]:
    root = repo_root.resolve()
    report_root = report.resolve()
    try:
        report_root.relative_to(root)
    except ValueError as error:
        raise VerificationError("report must be inside the repository") from error

    receipt, receipt_raw = _load_canonical_json(report_root / "receipt.json")
    if set(receipt) != _EXPECTED_RECEIPT_FIELDS:
        raise VerificationError("receipt fields drift")
    if (
        receipt.get("schema_version") != "aeolus_habitat_v2_live_forecast_receipt_v2"
        or receipt.get("source_foundation_git_commit")
        != SOURCE_FOUNDATION_GIT_COMMIT
    ):
        raise VerificationError("receipt schema drift")
    if (
        receipt.get("release_tier") != DEMO_RELEASE_TIER
        or receipt.get("qualification_evidence") is not False
        or receipt.get("actuator_authority") is not False
    ):
        raise VerificationError("receipt claim boundary drift")
    integration_committed = receipt.get("integration_source_committed")
    integration_commit = receipt.get("integration_source_git_commit")
    if integration_committed is False:
        if integration_commit is not None:
            raise VerificationError("uncommitted receipt names an integration commit")
    elif integration_committed is True:
        if (
            type(integration_commit) is not str
            or len(integration_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in integration_commit
            )
        ):
            raise VerificationError("committed receipt lacks a full integration commit")
    else:
        raise VerificationError("integration commit state is malformed")
    if expected_integration_commit is not None:
        if (
            type(expected_integration_commit) is not str
            or len(expected_integration_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in expected_integration_commit
            )
        ):
            raise VerificationError("expected integration commit is malformed")
        if integration_committed is not True or integration_commit != expected_integration_commit:
            raise VerificationError("integration commit does not match the CI checkout")
    artifact_bytes = _verify_file_set(
        report_root,
        receipt["artifacts"],
        _EXPECTED_ARTIFACT_PATHS,
        label="artifact",
    )
    _verify_file_set(
        root,
        receipt["source_files"],
        _EXPECTED_SOURCE_PATHS,
        label="source",
    )

    payload, payload_raw = _load_canonical_json_bytes(
        artifact_bytes["live-run.json"],
        label="live-run.json",
    )
    html_raw = artifact_bytes["index.html"]
    match = _PAYLOAD_PATTERN.search(html_raw)
    if match is None or base64.b64decode(match.group(1), validate=True) != payload_raw:
        raise VerificationError("HTML does not embed the exact live-run payload")

    if (
        set(payload) != _EXPECTED_PAYLOAD_FIELDS
        or payload.get("schema_version")
        != "aeolus_habitat_v2_live_forecast_report_v1"
        or payload.get("release_tier") != DEMO_RELEASE_TIER
    ):
        raise VerificationError("live-run schema drift")

    claims = payload["claims"]
    expected_claims = {
        "forecast_only_local_prototype": True,
        "model_predictions_computed_before_future_steps": True,
        "simulator_generated_truth": True,
        "browser_executes_model_inference": False,
        "hmc_is_sole_actuator_authority": True,
        "model_actuator_authority": False,
        "d2_qualified": False,
        "production_deployed": False,
        "physical_habitat_validated": False,
    }
    if claims != expected_claims:
        raise VerificationError("live-run claim boundary drift")

    model_evidence = _exact_mapping(
        payload["model"],
        _EXPECTED_MODEL_FIELDS,
        label="live-run model",
    )
    timeline = _exact_mapping(
        payload["timeline"],
        _EXPECTED_TIMELINE_FIELDS,
        label="live-run timeline",
    )
    authority = _exact_mapping(
        payload["authority"],
        _EXPECTED_AUTHORITY_FIELDS,
        label="live-run authority",
    )
    simulator_truth = _exact_mapping(
        payload["simulator_truth"],
        _EXPECTED_TRUTH_FIELDS,
        label="live-run simulator truth",
    )
    replay = _exact_mapping(
        payload["replay_evidence"],
        _EXPECTED_REPLAY_FIELDS,
        label="live-run replay",
    )
    selected_action = authority["selected_action_id"]
    if type(selected_action) is not str:
        raise VerificationError("live-run selected action is malformed")
    if authority["actuator_authority"] != "deterministic_hmc_only":
        raise VerificationError("live-run HMC authority drift")

    bundle = load_forecast_contracts(root)
    layout = forecast_layout(bundle)
    expected_descriptors = [dict(item) for item in layout.target_descriptors]
    if payload["target_descriptors"] != expected_descriptors:
        raise VerificationError("live-run target descriptors drift")

    model = load_live_ridge_model(model_path, expected_sha256=MODEL_SHA256)
    try:
        fresh = run_live_forecast_demo(
            root,
            model,
            selected_action_id=selected_action,
        )
    except LiveForecastError as error:
        raise VerificationError("fresh deterministic live run failed") from error

    expected_model_evidence = {
        "kind": fresh.model_kind,
        "artifact_sha256": fresh.model_artifact_sha256,
        "input_manifest_sha256": layout.input_manifest_sha256,
        "target_manifest_sha256": layout.target_manifest_sha256,
    }
    if model_evidence != expected_model_evidence:
        raise VerificationError("fresh model identity differs from report")
    expected_timeline = {
        "forecast_completed_step": fresh.forecast_completed_step,
        "forecast_completed_time_s": fresh.forecast_completed_time_s,
        "history_steps": list(fresh.forecast_history_steps),
        "truth_steps": list(fresh.truth_steps),
    }
    if timeline != expected_timeline:
        raise VerificationError("fresh causal timeline differs from report")
    expected_authority = {
        "selection_source": fresh.selection_source,
        "selected_action_id": fresh.selected_action_id,
        "selected_command_sha256": fresh.selected_command_sha256,
        "arbitration_disposition": fresh.arbitration_disposition,
        "final_command_sha256": fresh.final_command_sha256,
        "actuator_authority": "deterministic_hmc_only",
    }
    if authority != expected_authority:
        raise VerificationError("fresh authority identity differs from report")
    expected_replay = {
        "control_run_id": fresh.control_run_id,
        "terminal_status": fresh.terminal_status,
        "trace_sha256": fresh.trace_sha256,
        "trace_footer_sha256": fresh.trace_footer_sha256,
        "replay_final_state_sha256": fresh.replay_final_state_sha256,
        "replay_committed_steps": fresh.replay_committed_steps,
    }
    if replay != expected_replay:
        raise VerificationError("fresh HMC replay identity differs from report")
    if (
        receipt["model_artifact_sha256"] != fresh.model_artifact_sha256
        or receipt["control_trace_sha256"] != fresh.trace_sha256
    ):
        raise VerificationError("receipt-to-fresh-run identity drift")

    candidates = payload["candidate_forecasts"]
    if type(candidates) is not list or len(candidates) != len(fresh.candidate_forecasts):
        raise VerificationError("live-run candidate count drift")
    stored_predictions: dict[str, np.ndarray] = {}
    for index, (candidate, fresh_candidate) in enumerate(
        zip(candidates, fresh.candidate_forecasts, strict=True)
    ):
        candidate_mapping = _exact_mapping(
            candidate,
            _EXPECTED_CANDIDATE_FIELDS,
            label=f"live-run candidate {index}",
        )
        action_id = candidate_mapping["action_id"]
        if type(action_id) is not str:
            raise VerificationError("live-run candidate action is malformed")
        prediction = _finite_f32(
            candidate_mapping["prediction_f32"],
            shape=(8, 51),
            label=f"prediction for {action_id}",
        )
        action = _finite_f32(
            candidate_mapping["proposed_action_f32"],
            shape=(27,),
            label=f"proposed action for {action_id}",
        )
        if (
            action_id != fresh_candidate.action_id
            or candidate_mapping["command_sha256"]
            != fresh_candidate.command_sha256
            or not np.array_equal(action, fresh_candidate.proposed_action_f32)
        ):
            raise VerificationError("candidate action identity differs from fresh catalogue")
        if not np.array_equal(prediction, fresh_candidate.prediction_f32):
            raise VerificationError("fresh model prediction bytes differ from report")
        stored_predictions[action_id] = prediction
    if len(stored_predictions) != len(candidates):
        raise VerificationError("live-run candidate action IDs are not unique")
    if len({value.tobytes() for value in stored_predictions.values()}) != len(candidates):
        raise VerificationError("live-run forecasts are not action-conditioned")

    if simulator_truth["selected_action_id"] != fresh.selected_action_id:
        raise VerificationError("simulator truth action differs from fresh run")
    stored_truth = _finite_f32(
        simulator_truth["truth_f32"],
        shape=(8, 51),
        label="simulator truth",
    )
    if not np.array_equal(fresh.truth_f32, stored_truth):
        raise VerificationError("fresh simulator truth differs from report")
    maximum_prediction_drift = max(
        float(
            np.max(
                np.abs(
                    fresh_candidate.prediction_f32
                    - stored_predictions[fresh_candidate.action_id]
                )
            )
        )
        for fresh_candidate in fresh.candidate_forecasts
    )
    if maximum_prediction_drift != 0.0:
        raise VerificationError("fresh model prediction bytes differ from report")

    return {
        "verified": True,
        "release_tier": DEMO_RELEASE_TIER,
        "model_sha256": MODEL_SHA256,
        "receipt_sha256": _sha256(receipt_raw),
        "live_run_sha256": _sha256(payload_raw),
        "index_sha256": _sha256(html_raw),
        "trace_sha256": fresh.trace_sha256,
        "replay_committed_steps": fresh.replay_committed_steps,
        "candidate_action_count": len(fresh.candidate_forecasts),
        "distinct_prediction_count": len(
            {item.prediction_f32.tobytes() for item in fresh.candidate_forecasts}
        ),
        "maximum_prediction_drift": maximum_prediction_drift,
        "hmc_is_sole_actuator_authority": fresh.hmc_is_sole_actuator_authority,
        "browser_executes_model_inference": False,
        "d2_qualified": False,
    }


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify the Habitat V2 live forecast report")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "out/habitat-v2-live-forecast-demo/artifacts/live-run-v11",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root
        / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz",
    )
    parser.add_argument("--expected-integration-commit")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        result = verify_report(
            arguments.repo_root,
            arguments.report,
            arguments.model,
            expected_integration_commit=arguments.expected_integration_commit,
        )
    except VerificationError as error:
        print(json.dumps({"verified": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
