"""Local forecast-only integration demo for Habitat V2.

The model sees only issued operational history and catalogue actions. It produces
counterfactual forecasts before the future simulator steps exist. The selected
demo action is supplied by the caller, and deterministic HMC remains the sole
command and actuator authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from ..control_trace import parse_control_trace, replay_control_trace
from ..hmc import HabitatManagementComputer
from ..physics import advance_one_step_with_command, initial_state
from .baselines import (
    ACTION_COUNT,
    HORIZON_CANDIDATES,
    INPUT_MANIFEST_SHA256,
    TARGET_COUNT,
    TARGET_MANIFEST_SHA256,
    WINDOW_CANDIDATES,
    DirectRidgeModel,
)
from .contracts import load_forecast_contracts
from .corpus import canonical_json_bytes
from .pipeline import FINAL_HMC_COMMIT_SHA, _proposal
from .projection import (
    ForecastHistory,
    project_history_window,
    project_physical_targets,
    project_proposed_action,
)

DEMO_RELEASE_TIER: Final = "DEMO_ONLY_PERMANENTLY_EXCLUDED"
FORECAST_ANCHOR_STEP: Final = 16
FORECAST_WINDOW_STEPS: Final = 4
FORECAST_HORIZON_STEPS: Final = 8
_MODEL_SCHEMA: Final = "aeolus_habitat_v2_forecast_demo_model_v1"
_MODEL_SCHEMA_FP32: Final = "aeolus_habitat_v2_forecast_demo_model_fp32_v1"
_MODEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "release_tier",
        "actuator_authority",
        "alpha",
        "include_action",
        "feature_mean",
        "feature_scale",
        "target_mean",
        "coef",
        "window_steps",
        "horizon_steps",
        "input_manifest_sha256",
        "target_manifest_sha256",
    }
)


class LiveForecastError(ValueError):
    """The local live-forecast demo boundary is malformed."""


class ForecastPredictor(Protocol):
    def predict(
        self, history: ForecastHistory, proposed_action_f32: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class LiveForecastModel:
    """A forecast-only model with an immutable no-authority declaration."""

    predictor: ForecastPredictor
    model_kind: str
    artifact_sha256: str
    actuator_authority: bool = False

    def __post_init__(self) -> None:
        if not callable(getattr(self.predictor, "predict", None)):
            raise LiveForecastError("live model must expose a predict method")
        if type(self.model_kind) is not str or not self.model_kind:
            raise LiveForecastError("live model kind must be a non-empty string")
        if not _is_sha256(self.artifact_sha256):
            raise LiveForecastError("live model artifact identity must be SHA-256")
        if self.actuator_authority is not False:
            raise LiveForecastError("live model cannot claim actuator authority")


@dataclass(frozen=True, slots=True)
class CandidateForecast:
    action_id: str
    command_sha256: str
    proposed_action_f32: np.ndarray
    prediction_f32: np.ndarray


@dataclass(frozen=True, slots=True)
class LiveForecastResult:
    release_tier: str
    actuator_authority: bool
    hmc_is_sole_actuator_authority: bool
    selection_source: str
    model_kind: str
    model_artifact_sha256: str
    control_run_id: str
    selected_action_id: str
    selected_command_sha256: str
    forecast_completed_step: int
    forecast_completed_time_s: float
    forecast_history_steps: tuple[int, ...]
    truth_steps: tuple[int, ...]
    candidate_forecasts: tuple[CandidateForecast, ...]
    truth_f32: np.ndarray
    arbitration_disposition: str
    final_command_sha256: str
    terminal_status: str
    trace_sha256: str
    trace_footer_sha256: str
    replay_final_state_sha256: str
    replay_committed_steps: int


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _readonly_f32(value: object, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.shape != shape
        or not np.isfinite(array).all()
    ):
        raise LiveForecastError(f"{label} must be finite float32 with shape {shape}")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def load_live_ridge_model(
    source: str | Path | bytes,
    *,
    expected_sha256: str | None = None,
) -> LiveForecastModel:
    """Load the exact action-aware demo winner and reject identity or authority drift."""
    if type(source) is bytes:
        raw = source
    else:
        try:
            raw = Path(source).read_bytes()
        except OSError as error:
            raise LiveForecastError("live ridge artifact is unreadable") from error
    actual_sha256 = _sha256(raw)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise LiveForecastError("expected model identity must be SHA-256")
        if actual_sha256 != expected_sha256:
            raise LiveForecastError(
                "live ridge artifact SHA-256 does not match receipt"
            )

    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as value:
            if set(value.files) != _MODEL_FIELDS:
                raise LiveForecastError("live ridge artifact fields drift")
            schema_version = str(value["schema_version"].item())
            release_tier = str(value["release_tier"].item())
            actuator_authority = bool(value["actuator_authority"].item())
            alpha = float(value["alpha"].item())
            include_action = bool(value["include_action"].item())
            window_steps = int(value["window_steps"].item())
            horizon_steps = int(value["horizon_steps"].item())
            input_sha256 = str(value["input_manifest_sha256"].item())
            target_sha256 = str(value["target_manifest_sha256"].item())
            arrays = tuple(
                np.asarray(value[name]).copy()
                for name in ("feature_mean", "feature_scale", "target_mean", "coef")
            )
    except LiveForecastError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise LiveForecastError("live ridge artifact archive is malformed") from error

    if actuator_authority:
        raise LiveForecastError("live ridge artifact claims actuator authority")
    feature_mean, feature_scale, target_mean, coefficient = arrays
    feature_count = feature_mean.shape[0] if feature_mean.ndim == 1 else -1
    target_count = FORECAST_HORIZON_STEPS * TARGET_COUNT
    expected_feature_count = (
        FORECAST_WINDOW_STEPS * (194 + 167 * 5 + 4 + 4 + 287 * 4) + ACTION_COUNT
    )
    if (
        schema_version not in {_MODEL_SCHEMA, _MODEL_SCHEMA_FP32}
        or release_tier != DEMO_RELEASE_TIER
        or not np.isfinite(alpha)
        or alpha <= 0.0
        or include_action is not True
        or window_steps != FORECAST_WINDOW_STEPS
        or window_steps not in WINDOW_CANDIDATES
        or horizon_steps != FORECAST_HORIZON_STEPS
        or horizon_steps not in HORIZON_CANDIDATES
        or input_sha256 != INPUT_MANIFEST_SHA256
        or target_sha256 != TARGET_MANIFEST_SHA256
        or feature_count != expected_feature_count
        or feature_scale.shape != (feature_count,)
        or target_mean.shape != (target_count,)
        or coefficient.shape != (feature_count, target_count)
        or (feature_scale <= 0.0).any()
        or any(not np.isfinite(array).all() for array in arrays)
    ):
        raise LiveForecastError(
            "live ridge artifact shape or contract identity is invalid"
        )
    expected_dtype = (
        np.dtype(np.float32)
        if schema_version == _MODEL_SCHEMA_FP32
        else np.dtype(np.float64)
    )
    if any(array.dtype != expected_dtype for array in arrays):
        raise LiveForecastError("live ridge artifact precision contract is invalid")
    for array in arrays:
        array.setflags(write=False)
    predictor = DirectRidgeModel(
        alpha=alpha,
        include_action=True,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        coef=coefficient,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        input_manifest_sha256=input_sha256,
        target_manifest_sha256=target_sha256,
    )
    return LiveForecastModel(
        predictor=predictor,
        model_kind=(
            "action_aware_ridge_fp32"
            if schema_version == _MODEL_SCHEMA_FP32
            else "action_aware_ridge"
        ),
        artifact_sha256=actual_sha256,
        actuator_authority=False,
    )


def _demo_nonce(model: LiveForecastModel, selected_action_id: str) -> bytes:
    return hashlib.sha256(
        b"aeolus-habitat-v2-live-forecast-demo-v1\0"
        + bytes.fromhex(model.artifact_sha256)
        + b"\0"
        + selected_action_id.encode("ascii")
    ).digest()


def run_live_forecast_demo(
    repo_root: str | Path,
    model: LiveForecastModel,
    *,
    selected_action_id: str,
) -> LiveForecastResult:
    """Forecast all catalogue actions at step 16, then let HMC execute one caller choice."""
    if type(model) is not LiveForecastModel or model.actuator_authority is not False:
        raise LiveForecastError(
            "live run requires an exact forecast-only model wrapper"
        )
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
    forecast_history: ForecastHistory | None = None
    anchor_snapshot: object | None = None
    anchor_arbitration: object | None = None

    steps = int(bundle.development_scenario.data["steps"])
    if steps < FORECAST_ANCHOR_STEP + FORECAST_HORIZON_STEPS:
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
        if application_step == FORECAST_ANCHOR_STEP:
            pairs = [
                snapshots[step]
                for step in range(
                    FORECAST_ANCHOR_STEP - FORECAST_WINDOW_STEPS + 1,
                    FORECAST_ANCHOR_STEP + 1,
                )
            ]
            forecast_history = project_history_window(
                bundle,
                pairs,
                window_steps=FORECAST_WINDOW_STEPS,
            )
            forecasts: list[CandidateForecast] = []
            for action in actions:
                proposed_action = project_proposed_action(bundle, action.command)
                try:
                    raw_prediction = model.predictor.predict(
                        forecast_history,
                        proposed_action,
                    )
                except Exception as error:
                    raise LiveForecastError(
                        f"forecast model failed for action {action.action_id}"
                    ) from error
                prediction = _readonly_f32(
                    raw_prediction,
                    shape=(FORECAST_HORIZON_STEPS, TARGET_COUNT),
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
        if application_step == FORECAST_ANCHOR_STEP:
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
        if application_step == FORECAST_ANCHOR_STEP:
            anchor_arbitration = arbitration
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise LiveForecastError("HMC terminated while stepping the live demo")
        shadow_result = advance_one_step_with_command(
            bundle.development_scenario,
            shadow,
            arbitration.final_command,
        )
        if (
            _sha256(canonical_json_bytes(shadow_result.receipt))
            != step_receipt.plant_receipt_digest
        ):
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

    truth_steps = tuple(
        range(
            FORECAST_ANCHOR_STEP + 1,
            FORECAST_ANCHOR_STEP + FORECAST_HORIZON_STEPS + 1,
        )
    )
    truth = project_physical_targets(
        bundle,
        [states[step] for step in truth_steps],
        horizon_steps=FORECAST_HORIZON_STEPS,
    )
    truth = _readonly_f32(
        truth,
        shape=(FORECAST_HORIZON_STEPS, TARGET_COUNT),
        label="live simulator truth",
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
    terminal_status = str(parsed.footer["terminal_status"])
    if (
        terminal_status != "COMPLETED"
        or replay.committed_step_count != steps
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise LiveForecastError("live-demo control trace does not replay to closure")

    arbitration_mapping = anchor_arbitration.to_mapping()
    return LiveForecastResult(
        release_tier=DEMO_RELEASE_TIER,
        actuator_authority=False,
        hmc_is_sole_actuator_authority=True,
        selection_source="operator_selected_demo_action",
        model_kind=model.model_kind,
        model_artifact_sha256=model.artifact_sha256,
        control_run_id=hmc.control_run_id,
        selected_action_id=selected_action.action_id,
        selected_command_sha256=selected_action.command_sha256,
        forecast_completed_step=FORECAST_ANCHOR_STEP,
        forecast_completed_time_s=float(
            anchor_snapshot.to_mapping()["completed_time_s"]
        ),
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
