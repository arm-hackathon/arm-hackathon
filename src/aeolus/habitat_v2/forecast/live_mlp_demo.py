"""Live forecast demo adapter for the Historical V2 development MLP.

Loads the action-aware MLP trained on the Historical V2 pilot archive
(training run ``full-v1-20260818-a``, held-out normalized MAE 0.1146) as a
pure-NumPy artifact and runs it through the same bounded live-demo lifecycle
as the ridge demo: forecast every catalogue action at the anchor step, let the
operator select one, and let deterministic HMC arbitrate and execute it.

The MLP consumes the 16-step numeric history window (16 x 194 = 3104
features) plus the 27-field proposed action and one presence flag.  This is
development evidence only: not qualification, not deployment, and the model
never holds actuator authority.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..control_trace import parse_control_trace, replay_control_trace
from ..hmc import HabitatManagementComputer
from ..physics import advance_one_step_with_command, initial_state
from .contracts import canonical_json_bytes, load_forecast_contracts
from .live_demo import (
    CandidateForecast,
    LiveForecastError,
    LiveForecastModel,
    LiveForecastResult,
    _demo_nonce,
    _is_sha256,
    _readonly_f32,
    _sha256,
)
from .pipeline import FINAL_HMC_COMMIT_SHA, _proposal
from .projection import (
    project_history_window,
    project_physical_targets,
    project_proposed_action,
)

MLP_WINDOW_STEPS = 16
MLP_HORIZON_STEPS = 8
MLP_ANCHOR_STEP = 16
TARGET_COUNT = 51
ACTION_COUNT = 27
FEATURE_COUNT = 3104 + ACTION_COUNT + 1
MODEL_SCHEMA = "aeolus_habitat_v2_forecast_live_mlp_v1"

_NPZ_FIELDS = frozenset(
    {
        "metadata_json",
        "w0", "b0", "w1", "b1", "w2", "b2", "w3", "b3",
        "feature_mean", "feature_std", "target_mean", "target_std",
    }
)


def _gelu_exact(value: np.ndarray) -> np.ndarray:
    """Exact-error-function GELU matching torch.nn.GELU at float32."""
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (0.5 * value * (1.0 + erf(value / math.sqrt(2.0)))).astype(np.float32)


class NumpyMlpPredictor:
    """Pure-NumPy forward pass for the frozen action-aware MLP."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray],
                 feature_mean: np.ndarray, feature_std: np.ndarray,
                 target_mean: np.ndarray, target_std: np.ndarray) -> None:
        self._weights = weights
        self._biases = biases
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self._target_mean = target_mean
        self._target_std = target_std

    def predict(self, history: Any, proposed_action_f32: np.ndarray) -> np.ndarray:
        numeric = np.asarray(history.numeric_f32, dtype=np.float32)
        if numeric.shape != (MLP_WINDOW_STEPS, 194):
            raise LiveForecastError("MLP requires the exact 16-step numeric window")
        action = np.asarray(proposed_action_f32, dtype=np.float32).reshape(ACTION_COUNT)
        features = np.concatenate(
            [numeric.reshape(-1), action, np.ones(1, dtype=np.float32)]
        )
        hidden = (features - self._feature_mean) / self._feature_std
        for index, (weight, bias) in enumerate(zip(self._weights, self._biases, strict=True)):
            hidden = hidden @ weight.T + bias
            if index < len(self._weights) - 1:
                hidden = _gelu_exact(hidden)
        prediction = hidden.reshape(MLP_HORIZON_STEPS, TARGET_COUNT) * self._target_std + self._target_mean
        result = np.asarray(prediction, dtype=np.float32)
        result.setflags(write=False)
        return result


def load_live_mlp_model(
    source: str | Path | bytes,
    *,
    expected_sha256: str | None = None,
) -> LiveForecastModel:
    """Load the exact frozen MLP artifact and reject identity/authority drift."""
    if type(source) is bytes:
        raw = source
    else:
        try:
            raw = Path(source).read_bytes()
        except OSError as error:
            raise LiveForecastError("live MLP artifact is unreadable") from error
    actual_sha256 = _sha256(raw)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise LiveForecastError("expected model identity must be SHA-256")
        if actual_sha256 != expected_sha256:
            raise LiveForecastError("live MLP artifact SHA-256 does not match receipt")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as value:
            if set(value.files) != set(_NPZ_FIELDS):
                raise LiveForecastError("live MLP artifact fields drift")
            metadata = json.loads(str(value["metadata_json"].item()))
            arrays = {name: np.asarray(value[name]).copy() for name in _NPZ_FIELDS if name != "metadata_json"}
    except LiveForecastError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise LiveForecastError("live MLP artifact archive is malformed") from error

    weights = [arrays[f"w{index}"] for index in range(4)]
    biases = [arrays[f"b{index}"] for index in range(4)]
    feature_mean, feature_std = arrays["feature_mean"], arrays["feature_std"]
    target_mean, target_std = arrays["target_mean"], arrays["target_std"]
    expected_shapes = [(512, FEATURE_COUNT), (512, 512), (256, 512), (MLP_HORIZON_STEPS * TARGET_COUNT, 256)]
    if (
        metadata.get("schema_version") != MODEL_SCHEMA
        or metadata.get("release_tier") != "DEVELOPMENT_EVIDENCE_ONLY"
        or metadata.get("actuator_authority") is not False
        or metadata.get("window_steps") != MLP_WINDOW_STEPS
        or metadata.get("horizon_steps") != MLP_HORIZON_STEPS
        or not _is_sha256(str(metadata.get("source_checkpoint_sha256")))
        or [tuple(weight.shape) for weight in weights] != expected_shapes
        or [bias.shape for bias in biases] != [(512,), (512,), (256,), (MLP_HORIZON_STEPS * TARGET_COUNT,)]
        or feature_mean.shape != (FEATURE_COUNT,)
        or feature_std.shape != (FEATURE_COUNT,)
        or (feature_std <= 0.0).any()
        or target_mean.shape != (MLP_HORIZON_STEPS, TARGET_COUNT)
        or target_std.shape != (MLP_HORIZON_STEPS, TARGET_COUNT)
        or any(array.dtype != np.float32 for array in arrays.values())
        or any(not np.isfinite(array).all() for array in arrays.values())
    ):
        raise LiveForecastError("live MLP artifact shape or contract identity is invalid")
    for array in arrays.values():
        array.setflags(write=False)
    predictor = NumpyMlpPredictor(weights, biases, feature_mean, feature_std, target_mean, target_std)
    return LiveForecastModel(
        predictor=predictor,
        model_kind="action_aware_mlp_v1",
        artifact_sha256=actual_sha256,
        actuator_authority=False,
    )


def run_live_mlp_forecast_demo(
    repo_root: str | Path,
    model: LiveForecastModel,
    *,
    selected_action_id: str,
) -> LiveForecastResult:
    """Forecast all catalogue actions at step 16 with a 16-step MLP window.

    Mirrors ``run_live_forecast_demo`` exactly except for the wider history
    window the MLP was trained on; HMC remains the sole command authority.
    """
    if type(model) is not LiveForecastModel or model.actuator_authority is not False:
        raise LiveForecastError("live run requires an exact forecast-only model wrapper")
    root = Path(repo_root).resolve()
    bundle = load_forecast_contracts(root)
    actions = tuple(bundle.actions)
    action_by_id = {action.action_id: action for action in actions}
    if selected_action_id not in action_by_id:
        raise LiveForecastError("selected demo action is outside the frozen catalogue")
    selected_action = action_by_id[selected_action_id]

    hmc = HabitatManagementComputer.reset(
        bundle.development_scenario,
        bundle.hmc_contract,
        _demo_nonce(model, selected_action_id),
    )
    shadow = initial_state(bundle.development_scenario)
    snapshots: dict[int, tuple[object, object]] = {}
    states: dict[int, object] = {0: shadow}
    candidate_forecasts: tuple[CandidateForecast, ...] | None = None
    forecast_history = None
    anchor_snapshot = None
    anchor_arbitration = None

    steps = int(bundle.development_scenario.data["steps"])
    if steps < MLP_ANCHOR_STEP + MLP_HORIZON_STEPS:
        raise LiveForecastError("development scenario is shorter than the live horizon")

    for application_step in range(steps):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise LiveForecastError("HMC terminated while observing the live demo")
        snapshot, verification = observed
        if (
            verification.completed_step != application_step
            or snapshot.snapshot_sha256 != verification.snapshot_sha256
        ):
            raise LiveForecastError("live demo snapshot verification drift")
        handle = hmc.verify_snapshot(snapshot, verification)
        if application_step:
            snapshots[application_step] = (snapshot, verification)

        proposal = None
        if application_step == MLP_ANCHOR_STEP:
            pairs = [
                snapshots[step]
                for step in range(MLP_ANCHOR_STEP - MLP_WINDOW_STEPS + 1, MLP_ANCHOR_STEP + 1)
            ]
            forecast_history = project_history_window(
                bundle, pairs, window_steps=MLP_WINDOW_STEPS,
            )
            forecasts: list[CandidateForecast] = []
            for action in actions:
                proposed_action = project_proposed_action(bundle, action.command)
                try:
                    raw_prediction = model.predictor.predict(forecast_history, proposed_action)
                except Exception as error:
                    raise LiveForecastError(
                        f"forecast model failed for action {action.action_id}"
                    ) from error
                prediction = _readonly_f32(
                    raw_prediction,
                    shape=(MLP_HORIZON_STEPS, TARGET_COUNT),
                    label=f"prediction for {action.action_id}",
                )
                forecasts.append(
                    CandidateForecast(
                        action_id=action.action_id,
                        command_sha256=action.command_sha256,
                        proposed_action_f32=proposed_action,
                        prediction_f32=prediction,
                    )
                )
            candidate_forecasts = tuple(forecasts)
            anchor_snapshot = snapshot
            proposal = _proposal(
                hmc,
                snapshot.snapshot_sha256,
                application_step,
                selected_action.command.to_mapping(),
                selected_action.action_id,
            )

        proposal_receipt = hmc.propose(proposal, handle)
        proposal_mapping = proposal_receipt.to_mapping()
        if application_step == MLP_ANCHOR_STEP:
            if (
                proposal_mapping["attempt_class"],
                proposal_mapping["validation_outcome"],
            ) != ("CANONICAL_PROPOSAL", "VALID"):
                raise LiveForecastError("selected live-demo proposal was not admitted")
        elif proposal_mapping["validation_outcome"] != "NO_PROPOSAL":
            raise LiveForecastError("live demo issued a proposal outside the anchor")

        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise LiveForecastError("HMC terminated while arbitrating the live demo")
        if application_step == MLP_ANCHOR_STEP:
            anchor_arbitration = arbitration
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise LiveForecastError("HMC terminated while stepping the live demo")
        shadow_result = advance_one_step_with_command(
            bundle.development_scenario, shadow, arbitration.final_command,
        )
        if _sha256(canonical_json_bytes(shadow_result.receipt)) != step_receipt.plant_receipt_digest:
            raise LiveForecastError("live-demo shadow plant diverges from HMC")
        shadow = shadow_result.state
        states[shadow.step] = shadow

    if (
        candidate_forecasts is None
        or forecast_history is None
        or anchor_snapshot is None
        or anchor_arbitration is None
    ):
        raise LiveForecastError("live demo never reached the forecast anchor")

    truth_steps = tuple(range(MLP_ANCHOR_STEP + 1, MLP_ANCHOR_STEP + MLP_HORIZON_STEPS + 1))
    truth = project_physical_targets(
        bundle, [states[step] for step in truth_steps], horizon_steps=MLP_HORIZON_STEPS,
    )
    truth = _readonly_f32(truth, shape=(MLP_HORIZON_STEPS, TARGET_COUNT), label="live simulator truth")

    trace = hmc.export_control_trace(FINAL_HMC_COMMIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes, scenario=bundle.development_scenario, contract=bundle.hmc_contract,
    )
    replay = replay_control_trace(
        trace.canonical_bytes, scenario=bundle.development_scenario, contract=bundle.hmc_contract,
    )
    terminal_status = str(parsed.footer["terminal_status"])
    if (
        terminal_status != "COMPLETED"
        or replay.committed_step_count != steps
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise LiveForecastError("live-demo control trace does not replay to closure")

    arbitration_mapping = anchor_arbitration.to_mapping()
    return LiveForecastResult(
        release_tier="DEVELOPMENT_EVIDENCE_ONLY",
        actuator_authority=False,
        hmc_is_sole_actuator_authority=True,
        selection_source="operator_selected_demo_action",
        model_kind=model.model_kind,
        model_artifact_sha256=model.artifact_sha256,
        control_run_id=hmc.control_run_id,
        selected_action_id=selected_action.action_id,
        selected_command_sha256=selected_action.command_sha256,
        forecast_completed_step=MLP_ANCHOR_STEP,
        forecast_completed_time_s=float(anchor_snapshot.to_mapping()["completed_time_s"]),
        forecast_history_steps=forecast_history.steps,
        truth_steps=truth_steps,
        candidate_forecasts=candidate_forecasts,
        truth_f32=truth,
        arbitration_disposition=str(arbitration_mapping["disposition"]),
        final_command_sha256=anchor_arbitration.final_command_sha256,
        terminal_status=terminal_status,
        trace_sha256=_sha256(trace.canonical_bytes),
        trace_footer_sha256=str(parsed.footer["control_trace_footer_sha256"]),
        replay_final_state_sha256=replay.final_state_sha256,
        replay_committed_steps=replay.committed_step_count,
    )
