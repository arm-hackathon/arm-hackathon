from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pytest


class RecordingForecaster:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], bytes]] = []

    def predict(self, history: object, proposed_action_f32: np.ndarray) -> np.ndarray:
        steps = tuple(getattr(history, "steps"))
        action_bytes = proposed_action_f32.tobytes()
        self.calls.append((steps, action_bytes))
        prediction = np.full((8, 51), float(len(self.calls)), dtype=np.float32)
        prediction.setflags(write=False)
        return prediction


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_live_forecast_is_computed_before_future_and_hmc_keeps_authority() -> None:
    from aeolus.habitat_v2.forecast.live_demo import (
        DEMO_RELEASE_TIER,
        LiveForecastModel,
        run_live_forecast_demo,
    )

    recorder = RecordingForecaster()
    model = LiveForecastModel(
        predictor=recorder,
        model_kind="test-recording-forecaster",
        artifact_sha256="1" * 64,
        actuator_authority=False,
    )

    result = run_live_forecast_demo(
        _repo_root(),
        model,
        selected_action_id="normal-occupied-v1",
    )

    assert result.release_tier == DEMO_RELEASE_TIER
    assert result.actuator_authority is False
    assert result.hmc_is_sole_actuator_authority is True
    assert result.selection_source == "operator_selected_demo_action"
    assert result.forecast_completed_step == 16
    assert result.forecast_history_steps == (13, 14, 15, 16)
    assert result.truth_steps == (17, 18, 19, 20, 21, 22, 23, 24)
    assert result.replay_committed_steps == 24
    assert result.terminal_status == "COMPLETED"
    assert result.selected_action_id == "normal-occupied-v1"
    assert len(result.candidate_forecasts) == 4
    assert tuple(item.action_id for item in result.candidate_forecasts) == (
        "normal-occupied-v1",
        "normal-eva_transition-v1",
        "normal-contingency-v1",
        "normal-dormant-v1",
    )
    assert len(recorder.calls) == 4
    assert all(steps == (13, 14, 15, 16) for steps, _ in recorder.calls)
    assert len({action_bytes for _, action_bytes in recorder.calls}) == 4
    assert all(item.prediction_f32.shape == (8, 51) for item in result.candidate_forecasts)
    assert result.truth_f32.shape == (8, 51)
    assert result.arbitration_disposition in {"ACCEPTED", "MODIFIED", "REJECTED"}
    assert len(result.final_command_sha256) == 64
    assert len(result.trace_sha256) == 64
    assert len(result.replay_final_state_sha256) == 64


def test_model_loader_rejects_artifact_that_claims_actuator_authority(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.live_demo import (
        LiveForecastError,
        load_live_ridge_model,
    )

    path = tmp_path / "authority-drift.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray("aeolus_habitat_v2_forecast_demo_model_v1"),
        release_tier=np.asarray("DEMO_ONLY_PERMANENTLY_EXCLUDED"),
        actuator_authority=np.asarray(True),
        alpha=np.asarray(1.0, dtype=np.float64),
        include_action=np.asarray(True),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_scale=np.ones(1, dtype=np.float64),
        target_mean=np.zeros(408, dtype=np.float64),
        coef=np.zeros((1, 408), dtype=np.float64),
        window_steps=np.asarray(4, dtype=np.int64),
        horizon_steps=np.asarray(8, dtype=np.int64),
        input_manifest_sha256=np.asarray(
            "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
        ),
        target_manifest_sha256=np.asarray(
            "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"
        ),
    )
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(LiveForecastError, match="authority"):
        load_live_ridge_model(path, expected_sha256=expected_sha256)


def test_model_loader_deserializes_the_exact_hashed_bytes_when_path_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.live_demo import load_live_ridge_model

    source = (
        _repo_root()
        / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
    )
    original_raw = source.read_bytes()
    with np.load(io.BytesIO(original_raw), allow_pickle=False) as archive:
        original_arrays = {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }
    original_coefficient = original_arrays["coef"].copy()
    replacement_arrays = {
        name: value.copy() for name, value in original_arrays.items()
    }
    replacement_arrays["coef"][0, 0] += 1.0
    replacement_buffer = io.BytesIO()
    np.savez_compressed(replacement_buffer, **replacement_arrays)
    replacement_raw = replacement_buffer.getvalue()

    path = tmp_path / "replaceable-model.npz"
    path.write_bytes(original_raw)
    expected_sha256 = hashlib.sha256(original_raw).hexdigest()
    real_read_bytes = Path.read_bytes
    replaced = False

    def read_then_replace(candidate: Path) -> bytes:
        nonlocal replaced
        raw = real_read_bytes(candidate)
        if candidate == path and not replaced:
            path.write_bytes(replacement_raw)
            replaced = True
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    model = load_live_ridge_model(path, expected_sha256=expected_sha256)
    loaded_coefficient = getattr(model.predictor, "coef")

    assert replaced is True
    assert model.artifact_sha256 == expected_sha256
    np.testing.assert_array_equal(loaded_coefficient, original_coefficient)
    assert not np.array_equal(
        loaded_coefficient,
        replacement_arrays["coef"],
    )


def test_report_preserves_live_inference_and_authority_claim_boundaries(
    tmp_path: Path,
) -> None:
    import json

    from aeolus.habitat_v2.forecast.live_demo import (
        LiveForecastModel,
        run_live_forecast_demo,
    )
    from aeolus.habitat_v2.forecast.live_demo_report import (
        write_live_forecast_report,
    )

    model = LiveForecastModel(
        predictor=RecordingForecaster(),
        model_kind="test-recording-forecaster",
        artifact_sha256="2" * 64,
        actuator_authority=False,
    )
    result = run_live_forecast_demo(
        _repo_root(),
        model,
        selected_action_id="normal-occupied-v1",
    )
    output = tmp_path / "live-report"
    written = write_live_forecast_report(
        _repo_root(),
        result,
        output,
        source_foundation_git_commit="c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3",
    )

    payload = json.loads((output / "live-run.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert payload["claims"] == {
        "browser_executes_model_inference": False,
        "d2_qualified": False,
        "forecast_only_local_prototype": True,
        "hmc_is_sole_actuator_authority": True,
        "model_actuator_authority": False,
        "model_predictions_computed_before_future_steps": True,
        "physical_habitat_validated": False,
        "production_deployed": False,
        "simulator_generated_truth": True,
    }
    assert payload["timeline"]["history_steps"] == [13, 14, 15, 16]
    assert payload["timeline"]["truth_steps"] == [17, 18, 19, 20, 21, 22, 23, 24]
    assert payload["authority"]["actuator_authority"] == "deterministic_hmc_only"
    assert receipt["schema_version"] == "aeolus_habitat_v2_live_forecast_receipt_v2"
    assert receipt["integration_source_committed"] is False
    assert receipt["integration_source_git_commit"] is None
    assert receipt["qualification_evidence"] is False
    assert receipt["actuator_authority"] is False
    assert written["live_run_sha256"] == hashlib.sha256(
        (output / "live-run.json").read_bytes()
    ).hexdigest()
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "does not execute model inference in the browser" in html
    assert "Learned model: no actuator authority" in html
    assert "innerHTML" not in html
    assert "textContent=value" in html


def _real_live_result(root: Path) -> object:
    from aeolus.habitat_v2.forecast.live_demo import (
        load_live_ridge_model,
        run_live_forecast_demo,
    )

    model_path = (
        root / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
    )
    model_sha256 = "a6e4ef34fc837bb6539a84e20d015bbd7bbfe4e9fd5a6fc74e3f0217bd978d9a"
    model = load_live_ridge_model(model_path, expected_sha256=model_sha256)
    return run_live_forecast_demo(
        root,
        model,
        selected_action_id="normal-occupied-v1",
    )


def _load_report_verifier(root: Path) -> object:
    spec = importlib.util.spec_from_file_location(
        "aeolus_live_forecast_report_verifier",
        root / "scripts/verify_habitat_v2_live_forecast_demo.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_real_report(root: Path, verifier: object, report: Path) -> None:
    from aeolus.habitat_v2.forecast.live_demo_report import write_live_forecast_report

    source_paths = tuple(
        root / path for path in sorted(verifier._EXPECTED_SOURCE_PATHS)
    )
    write_live_forecast_report(
        root,
        _real_live_result(root),
        report,
        source_foundation_git_commit=(
            "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3"
        ),
        source_paths=source_paths,
    )


def test_report_verifier_rejects_rehashed_candidate_action_forgery() -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    root = _repo_root()
    verifier = _load_report_verifier(root)
    output_parent = root / "out"
    output_parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="live-forecast-verifier-test-",
        dir=output_parent,
    ) as raw_output:
        report = Path(raw_output) / "report"
        _write_real_report(root, verifier, report)

        payload_path = report / "live-run.json"
        payload = json.loads(payload_path.read_bytes())
        payload["candidate_forecasts"][0]["proposed_action_f32"][0] += 1.0
        payload_raw = canonical_json_bytes(payload)
        payload_path.write_bytes(payload_raw)

        html_path = report / "index.html"
        html_raw = html_path.read_bytes()
        replacement = b"atob('" + base64.b64encode(payload_raw) + b"')"
        html_raw, replacement_count = re.subn(
            rb"atob\('[A-Za-z0-9+/=]+'\)",
            lambda _: replacement,
            html_raw,
            count=1,
        )
        assert replacement_count == 1
        html_path.write_bytes(html_raw)

        receipt_path = report / "receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        for item in receipt["artifacts"]:
            artifact_raw = (report / item["relative_path"]).read_bytes()
            item["byte_length"] = len(artifact_raw)
            item["sha256"] = hashlib.sha256(artifact_raw).hexdigest()
        receipt_path.write_bytes(canonical_json_bytes(receipt))

        with pytest.raises(verifier.VerificationError, match="candidate action"):
            verifier.verify_report(
                root,
                report,
                root
                / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz",
            )


def test_report_verifier_uses_same_payload_bytes_for_receipt_and_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    root = _repo_root()
    verifier = _load_report_verifier(root)
    output_parent = root / "out"
    output_parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="live-forecast-verifier-race-test-",
        dir=output_parent,
    ) as raw_output:
        report = Path(raw_output) / "report"
        _write_real_report(root, verifier, report)
        payload_path = report / "live-run.json"
        genuine_payload_raw = payload_path.read_bytes()
        malicious_payload = json.loads(genuine_payload_raw)
        malicious_payload["candidate_forecasts"][0]["proposed_action_f32"][0] += 1.0
        malicious_payload_raw = canonical_json_bytes(malicious_payload)

        receipt_path = report / "receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        payload_entry = next(
            item
            for item in receipt["artifacts"]
            if item["relative_path"] == "live-run.json"
        )
        payload_entry["byte_length"] = len(malicious_payload_raw)
        payload_entry["sha256"] = hashlib.sha256(malicious_payload_raw).hexdigest()
        receipt_path.write_bytes(canonical_json_bytes(receipt))

        real_read_bytes = Path.read_bytes
        swapped = False

        def read_then_replace(candidate: Path) -> bytes:
            nonlocal swapped
            raw = real_read_bytes(candidate)
            if candidate == payload_path and not swapped:
                payload_path.write_bytes(malicious_payload_raw)
                swapped = True
            return raw

        monkeypatch.setattr(Path, "read_bytes", read_then_replace)
        with pytest.raises(verifier.VerificationError, match="receipt identity mismatch"):
            verifier.verify_report(
                root,
                report,
                root
                / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz",
            )
        assert swapped is True
