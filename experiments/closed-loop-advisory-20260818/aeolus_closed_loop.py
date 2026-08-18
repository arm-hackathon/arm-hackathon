"""AEOLUS Habitat V2 closed-loop advisory harness (development evidence only).

Runs the real HMC lifecycle (``reset -> observe -> verify -> propose ->
arbitrate -> step``) with an optional forecast adviser.  The adviser scores
the frozen action catalogue with the historical development model and submits
at most one canonical proposal per step.  HMC arbitration remains the sole
authority: it may accept, modify or ignore any proposal.  Nothing in this
module touches the sealed repositories; it imports them read-only.

This is NOT qualification evidence.  The historical model was trained without
operational availability masks, so closed-loop results here are development
signal only.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Everything resolves relative to the repository; no absolute paths.

# ---------------------------------------------------------------------------
# Pre-registered risk functional (preregistration-v1, frozen before outcomes)
# ---------------------------------------------------------------------------
# Target layout per zone (projection.TARGET_FIELDS): temperature_k,
# pressure_pa, co2_ppm, o2_mole_fraction, relative_humidity,
# branch_airflow_m3_s; then battery/oxygen-store/sorbent fractions.
# Thresholds are the HMC contract health_policy environmental/resource values.

RULES: tuple[dict[str, Any], ...] = (
    {"name": "high_co2", "field": "co2_ppm", "direction": "high",
     "warning": 2500.0, "critical": 5000.0},
    {"name": "high_humidity", "field": "relative_humidity", "direction": "high",
     "warning": 0.65, "critical": 0.75},
    {"name": "high_temperature", "field": "temperature_k", "direction": "high",
     "warning": 300.0, "critical": 303.0},
    {"name": "low_oxygen", "field": "o2_mole_fraction", "direction": "low",
     "warning": 0.285, "critical": 0.27},
    {"name": "low_temperature", "field": "temperature_k", "direction": "low",
     "warning": 291.0, "critical": 288.0},
)

TARGET_FIELDS = (
    "temperature_k", "pressure_pa", "co2_ppm", "o2_mole_fraction",
    "relative_humidity", "branch_airflow_m3_s",
)
ZONES = 8


def rule_exceedance(value: float, rule: dict[str, Any]) -> float:
    """0 below warning, 1.0 at critical, >1 beyond critical; deterministic."""
    margin = abs(float(rule["critical"]) - float(rule["warning"]))
    if rule["direction"] == "high":
        return max(0.0, (float(value) - float(rule["warning"])) / margin)
    return max(0.0, (float(rule["warning"]) - float(value)) / margin)


def trajectory_risk(targets: np.ndarray) -> float:
    """Pre-registered scalar risk of one predicted/realized [H, 51] target row set."""
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 51:
        raise ValueError("risk requires a [horizon, 51] target array")
    total = 0.0
    for zone in range(ZONES):
        base = zone * len(TARGET_FIELDS)
        for rule in RULES:
            column = base + TARGET_FIELDS.index(rule["field"])
            for step in range(values.shape[0]):
                total += rule_exceedance(values[step, column], rule)
    for gauge in range(48, 51):  # battery, oxygen store, sorbent fractions
        for step in range(values.shape[0]):
            total += rule_exceedance(
                values[step, gauge],
                {"direction": "low", "warning": 0.2, "critical": 0.1},
            )
    return float(total)


# ---------------------------------------------------------------------------
# Historical model adviser
# ---------------------------------------------------------------------------

class HistoricalAdviser:
    """Loads the frozen action-aware checkpoint and ranks catalogue actions."""

    def __init__(self, checkpoint: Path) -> None:
        import torch

        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        layers = [int(blob["input_dim"]), *blob["hidden_layers"], 408]
        modules: list[Any] = []
        for index in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[index], layers[index + 1]))
            if index < len(layers) - 2:
                modules.append(torch.nn.GELU())
        model = torch.nn.Sequential(*modules)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        self._torch = torch
        self.model = model
        self.feature_mean = np.asarray(blob["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(blob["feature_std"], dtype=np.float32)
        self.target_mean = np.asarray(blob["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(blob["target_std"], dtype=np.float32)
        if self.feature_mean.shape != (3132,) or self.target_mean.shape != (8, 51):
            raise ValueError("checkpoint normalizer shapes drift from the frozen layout")

    def predict(self, history_f32: np.ndarray, action_f32: np.ndarray | None) -> np.ndarray:
        """Predict denormalized [8, 51] targets for one candidate action."""
        history = np.asarray(history_f32, dtype=np.float32).reshape(-1)
        if history.shape != (3104,):
            raise ValueError("history must flatten to 3104 features")
        if action_f32 is None:
            action = np.zeros(27, dtype=np.float32)
            present = np.zeros(1, dtype=np.float32)
        else:
            action = np.asarray(action_f32, dtype=np.float32).reshape(27)
            present = np.ones(1, dtype=np.float32)
        features = (np.concatenate([history, action, present]) - self.feature_mean) / self.feature_std
        with self._torch.no_grad():
            raw = self.model(self._torch.from_numpy(features.astype(np.float32))).numpy()
        return (raw.reshape(8, 51) * self.target_std + self.target_mean)

    def choose(
        self,
        history_f32: np.ndarray,
        candidates: tuple[tuple[str, np.ndarray | None], ...],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Argmin predicted risk; ties resolve to the earliest candidate.

        Candidate order is fixed by the preregistration: NO_PROPOSAL first,
        then the four catalogue actions in frozen design order.
        """
        scored: list[dict[str, Any]] = []
        for candidate_id, action in candidates:
            risk = trajectory_risk(self.predict(history_f32, action))
            scored.append({"candidate_id": candidate_id, "predicted_risk": risk})
        best = min(range(len(scored)), key=lambda index: scored[index]["predicted_risk"])
        return scored[best]["candidate_id"], scored


# ---------------------------------------------------------------------------
# Closed-loop runner (adapts pilot_execution._execute_lifecycle)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepRecord:
    application_step: int
    proposed_candidate: str | None
    validation_outcome: str
    final_command_sha256: str
    requested_command_sha256: str | None


def run_closed_loop(
    *,
    repo_root: Path,
    design: Any,
    contracts: Any,
    cluster_id: str,
    member_id: str,
    repetition_id: str,
    adviser: HistoricalAdviser | None,
    on_step: Callable[[StepRecord], None] | None = None,
) -> dict[str, Any]:
    """Run one full HMC lifecycle; adviser proposes from step 16 onward.

    With ``adviser=None`` this is the canonical HMC control arm (no proposals).
    Shadow physics and strict trace replay close every run exactly as in the
    sealed pilot execution path.
    """
    from aeolus.habitat_v2.control_trace import (
        parse_control_trace,
        replay_control_trace,
    )
    from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
    from aeolus.habitat_v2.forecast.pilot import (
        _HMC_IMPLEMENTATION_GIT_SHA,
        _noise_seed,
        _reset_nonce,
        materialize_pilot_scenario,
    )
    from aeolus.habitat_v2.forecast.pipeline import _proposal
    from aeolus.habitat_v2.forecast.projection import (
        project_history_window,
        project_physical_targets,
        project_proposed_action,
    )
    from aeolus.habitat_v2.hmc import HabitatManagementComputer
    from aeolus.habitat_v2.physics import (
        advance_one_step_with_command,
        initial_state,
        validate_external_command,
    )

    def _sha256(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    scenario = materialize_pilot_scenario(
        repo_root, design,
        cluster_id=cluster_id, member_id=member_id, repetition_id=repetition_id,
    )
    # Identical nonce/noise across arms: fixed documented anchor argument.
    nonce_anchor = int(design.anchor_completed_steps[0])
    nonce = _reset_nonce(design, cluster_id, repetition_id, nonce_anchor)
    noise_seed = _noise_seed(design, cluster_id, repetition_id)

    catalogue: list[tuple[str, np.ndarray | None, Any]] = [("NO_PROPOSAL", None, None)]
    for action in contracts.actions:
        vector = project_proposed_action(contracts, action.command.to_mapping())
        canonical = validate_external_command(scenario, action.command.to_mapping())
        catalogue.append((action.action_id, vector, canonical.sha256))

    total_steps = int(scenario.data["steps"])
    hmc = HabitatManagementComputer.reset(scenario, contracts.hmc_contract, nonce)
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    states: dict[int, Any] = {0: shadow}
    step_records: list[StepRecord] = []

    for application_step in range(total_steps):
        observed = hmc.observe()
        if type(observed) is not tuple or len(observed) != 2:
            raise RuntimeError("HMC entered terminal state during observation")
        snapshot, verification = observed
        if (
            verification.completed_step != application_step
            or snapshot.snapshot_sha256 != verification.snapshot_sha256
        ):
            raise RuntimeError("issued snapshot and verification drift")
        handle = hmc.verify_snapshot(snapshot, verification)
        if application_step:
            snapshots[application_step] = (snapshot, verification)

        proposal_value = None
        proposed_candidate: str | None = None
        requested_sha: str | None = None
        if adviser is not None and application_step >= 16:
            pairs = [
                snapshots[step]
                for step in range(application_step - 15, application_step + 1)
            ]
            history = project_history_window(contracts, pairs, window_steps=16)
            candidates = tuple((item[0], item[1]) for item in catalogue)
            chosen, _ = adviser.choose(history.numeric_f32, candidates)
            if chosen != "NO_PROPOSAL":
                entry = next(item for item in catalogue if item[0] == chosen)
                action = next(a for a in contracts.actions if a.action_id == chosen)
                proposal_value = _proposal(
                    hmc, snapshot.snapshot_sha256, application_step,
                    action.command.to_mapping(), action.action_id,
                )
                proposed_candidate = chosen
                requested_sha = entry[2]

        proposal_receipt = hmc.propose(proposal_value, handle)
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise RuntimeError("HMC entered terminal state during arbitration")
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise RuntimeError("HMC entered terminal state during step")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        shadow_digest = _sha256(canonical_json_bytes(shadow_result.receipt))
        if shadow_digest != step_receipt.plant_receipt_digest:
            raise RuntimeError("shadow plant receipt diverges from HMC receipt")
        shadow = shadow_result.state
        states[shadow.step] = shadow
        record = StepRecord(
            application_step=application_step,
            proposed_candidate=proposed_candidate,
            validation_outcome=proposal_receipt.to_mapping()["validation_outcome"],
            final_command_sha256=arbitration.final_command_sha256,
            requested_command_sha256=requested_sha,
        )
        step_records.append(record)
        if on_step is not None:
            on_step(record)

    trace = hmc.export_control_trace(_HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=contracts.hmc_contract,
    )
    replay = replay_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=contracts.hmc_contract,
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != total_steps
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise RuntimeError("closed-loop trace fails strict completed replay")

    realized = np.stack(
        [
            project_physical_targets(contracts, [states[step]], horizon_steps=1)[0]
            for step in range(1, total_steps + 1)
        ]
    )
    per_step_risk = np.asarray(
        [trajectory_risk(realized[step : step + 1]) for step in range(total_steps)]
    )
    proposals = [r for r in step_records if r.proposed_candidate is not None]
    admitted = [r for r in proposals if r.validation_outcome == "VALID"]
    overrides = [
        r for r in admitted if r.final_command_sha256 != r.requested_command_sha256
    ]
    return {
        "cluster_id": cluster_id,
        "member_id": member_id,
        "repetition_id": repetition_id,
        "arm": "advised" if adviser is not None else "control",
        "scenario_sha256": scenario.scenario_sha256,
        "noise_seed": noise_seed,
        "total_steps": total_steps,
        "trace_sha256": _sha256(trace.canonical_bytes),
        "final_state_sha256": str(parsed.footer["final_state_sha256"]),
        "terminal_status": parsed.footer["terminal_status"],
        "integrated_exceedance": float(per_step_risk.sum()),
        "peak_step_exceedance": float(per_step_risk.max()),
        "exceedance_steps": int((per_step_risk > 0).sum()),
        "per_step_exceedance": per_step_risk.tolist(),
        "proposals_made": len(proposals),
        "proposals_admitted": len(admitted),
        "hmc_overrides": len(overrides),
        "battery_wh_consumed": float(
            states[0].utility.battery_energy_wh - shadow.utility.battery_energy_wh
        ),
        "oxygen_mol_consumed": float(
            states[0].utility.oxygen_store_mol - shadow.utility.oxygen_store_mol
        ),
        "sorbent_mol_consumed": float(
            states[0].utility.co2_sorbent_remaining_mol
            - shadow.utility.co2_sorbent_remaining_mol
        ),
        "step_records": [r.__dict__ for r in step_records],
    }
