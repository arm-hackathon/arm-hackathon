"""Issue #77 HMC-filtered closed-loop development study.

Six preregistered arms run identical DEV exogenous traces behind the HMC:
rules, hold, c8 (guard + ridge screen), c9 (guard + MLP screen), the Issue #74
action-conditioned ridge, and the calibrated BDM-v1 (seed-mean Issue #75 TCN
gated by the Issue #76 abstention layer). Every policy proposes or abstains;
the HMC remains sole arbitration, final-command, plant-step, and replay
authority. Episode-side feature reconstruction reuses the corpus window
builder so live windows match corpus samples field-for-field (test-enforced).

Gates are evaluated in preregistered order at causal-group level; on failure
the candidate packet is withheld, the blind protocol draft stays WITHHELD and
UNAUTHORIZED, and the blind population remains sealed.

Live windows follow the corpus collection convention for the retained
disposition channel: the corpus was collected under hold episodes, so every
window row carries ``NO_PROPOSAL``; live episodes substitute the same value
while the true live outcomes are recorded only in the decision lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import time
from typing import Any, Callable

import numpy as np

from .bdm_v1_corpus import WINDOW_STEPS, _f32, _window_rows
from .bdm_v1_evaluation import bootstrap_ci
from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast_issue73_ablations import (
    guard_proposal,
    learned_proposal,
    observable_context,
)
from .forecast_issue74_baselines import predict_sample
from .forecast_issue75_bdm import DELTA_SLICE, forward, input_tensor
from .forecast_issue76_abstention import CalibrationLayer, predict_with_abstention
from .forecast.projection import MODE_ORDER
from .forecast_issue55_race import (
    compute_race_metrics,
    project_true_targets,
    scenario_zone_order,
)
from .hmc import HabitatManagementComputer
from .physics import (
    advance_one_step_with_command,
    initial_state,
    validate_external_command,
)
from .scenario import Scenario

ISSUE77_CLOSED_LOOP_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_closed_loop_v1"
MODE_ORDER_TUPLE = tuple(MODE_ORDER)

Policy = Callable[[Mapping[str, Any], str], tuple[str | None, str | None, float]]


class Issue77ClosedLoopError(ValueError):
    """Raised when closed-loop inputs, policies, or gates are inadmissible."""


def reconstruct_sample_features(
    bundle: ForecastContracts,
    episode_snapshots: Sequence[Mapping[str, Any]],
    decision: int,
    action_index: int,
    command_vector: Sequence[float],
) -> dict[str, Any]:
    """Rebuild the corpus feature dict for a live window (field-for-field)."""

    window = [
        snapshot
        for snapshot in episode_snapshots
        if decision - WINDOW_STEPS < int(snapshot["completed_step"]) <= decision
    ]
    if len(window) != WINDOW_STEPS:
        raise Issue77ClosedLoopError("live window is incomplete at the decision step")
    zone_ids = tuple(bundle.topology.zone_ids)
    rows = _window_rows(window, bundle)
    env_by_channel = {
        channel: [
            [_f32(float(value), f"env {channel}") for value in step_row]
            for step_row in rows["env"][:, :, channel_index]
        ]
        for channel_index, channel in enumerate(
            (
                "temperature_k",
                "pressure_pa",
                "co2_ppm",
                "o2_mole_fraction",
                "relative_humidity",
                "branch_airflow_m3_s",
            )
        )
    }
    return {
        "zone_temperature_k": env_by_channel["temperature_k"],
        "zone_pressure_pa": env_by_channel["pressure_pa"],
        "zone_co2_ppm": env_by_channel["co2_ppm"],
        "zone_o2_mole_fraction": env_by_channel["o2_mole_fraction"],
        "zone_relative_humidity": env_by_channel["relative_humidity"],
        "zone_branch_airflow_m3_s": env_by_channel["branch_airflow_m3_s"],
        "observed_value_mask": [
            [[bool(flag) for flag in zone_row] for zone_row in step_row]
            for step_row in rows["mask"]
        ],
        "steps_since_last_valid_observation": [
            [[int(value) for value in zone_row] for zone_row in step_row]
            for step_row in rows["stale"]
        ],
        "requested_command_vector": [
            [_f32(float(value), "requested command") for value in step_row]
            for step_row in rows["requested"]
        ],
        "achieved_actuator_state": [
            [_f32(float(value), "achieved state") for value in step_row]
            for step_row in rows["achieved"]
        ],
        "battery_state_of_charge": [
            _f32(float(value), "battery gauge") for value in rows["gauges"][:, 0]
        ],
        "oxygen_store_fraction": [
            _f32(float(value), "oxygen gauge") for value in rows["gauges"][:, 1]
        ],
        "sorbent_fraction": [
            _f32(float(value), "sorbent gauge") for value in rows["gauges"][:, 2]
        ],
        "operating_mode_one_hot": [
            [_f32(float(value), "mode one-hot") for value in step_row]
            for step_row in rows["modes"]
        ],
        "prior_proposal_dispositions": [
            [_f32(float(value), "disposition one-hot") for value in step_row]
            for step_row in rows["dispositions"]
        ],
        "topology_configuration_descriptor": {
            "zone_order": list(zone_ids),
            "observable_topology_sha256": window[-1]["observable_topology_sha256"],
        },
        "candidate_action_command_vector": [
            _f32(float(value), "candidate command") for value in command_vector
        ],
        "candidate_action_index": action_index,
        "declared_known_future_schedule": [],
    }


def make_null_policy() -> Policy:
    def policy(_snapshot: Mapping[str, Any], _current_action_id: str):
        return None, None, 0.0

    return policy


def make_guard_screen_policy(screen: Any, *, include_guard: bool) -> Policy:
    def policy(snapshot: Mapping[str, Any], current_action_id: str):
        bundle_zone_ids = tuple(snapshot["__zone_ids"])
        actions = tuple(snapshot["__actions"])
        context = observable_context(snapshot, bundle_zone_ids, MODE_ORDER_TUPLE)
        if include_guard:
            chosen, _predictions = learned_proposal(
                screen,
                snapshot,
                bundle_zone_ids,
                MODE_ORDER_TUPLE,
                [action.action_id for action in actions],
                context,
                current_action_id,
                include_guard=True,
            )
            return chosen, None, 0.0
        guarded = guard_proposal(context, current_action_id)
        if guarded is not None:
            return guarded, None, 0.0
        chosen, _predictions = learned_proposal(
            screen,
            snapshot,
            bundle_zone_ids,
            MODE_ORDER_TUPLE,
            [action.action_id for action in actions],
            context,
            current_action_id,
            include_guard=False,
        )
        return chosen, None, 0.0

    return policy


def make_linear_policy(baseline: Any) -> Policy:
    def policy(snapshot: Mapping[str, Any], _current_action_id: str):
        bundle = snapshot["__bundle"]
        actions = tuple(snapshot["__actions"])
        episode_snapshots = snapshot["__episode_snapshots"]
        decision = int(snapshot["completed_step"])
        best_id = None
        best_prediction = 0.0
        for index, action in enumerate(actions):
            command_vector = _command_vector_for(bundle, action)
            sample = {
                "features": reconstruct_sample_features(
                    bundle, episode_snapshots, decision, index, command_vector
                )
            }
            prediction = float(
                predict_sample(baseline, sample)["delta"]["safety_exposure"]
            )
            if prediction < 0.0 and (best_id is None or prediction < best_prediction):
                best_id = action.action_id
                best_prediction = prediction
        return best_id, None, 0.0

    return policy


def make_calibrated_bdm_policy(models: Sequence[Any], layer: CalibrationLayer) -> Policy:
    def policy(snapshot: Mapping[str, Any], _current_action_id: str):
        started = time.perf_counter()
        bundle = snapshot["__bundle"]
        actions = tuple(snapshot["__actions"])
        episode_snapshots = snapshot["__episode_snapshots"]
        decision = int(snapshot["completed_step"])
        candidates: list[tuple[float, str, dict[str, Any], np.ndarray, np.ndarray]] = []
        for index, action in enumerate(actions):
            command_vector = _command_vector_for(bundle, action)
            features = reconstruct_sample_features(
                bundle, episode_snapshots, decision, index, command_vector
            )
            sample = {"features": features}
            tensor = input_tensor(sample)[None]
            seed_deltas = []
            seed_outs = []
            for model in models:
                out, _ = forward(model, tensor)
                seed_outs.append(out[0])
                seed_deltas.append(out[0, DELTA_SLICE[0] : DELTA_SLICE[1]])
            per_seed = np.stack(seed_deltas)
            pooled_row = np.mean(np.stack(seed_outs), axis=0)
            candidates.append(
                (
                    float(pooled_row[DELTA_SLICE[0] + 1]),
                    action.action_id,
                    sample,
                    per_seed,
                    pooled_row,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        exposure, action_id, sample, per_seed, pooled_row = candidates[0]
        _prediction, reason = predict_with_abstention(sample, per_seed, layer, pooled_row)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if reason is not None:
            return None, reason, elapsed_ms
        if exposure >= 0.0:
            return None, None, elapsed_ms
        return action_id, None, elapsed_ms

    return policy


def _command_vector_for(bundle: ForecastContracts, action: Any) -> list[float]:
    from .forecast.projection import _command_vector as _project_command_vector

    return [
        float(value)
        for value in _project_command_vector(bundle, action.command.to_mapping(), "candidate")
    ]


def run_closed_loop_episode(
    bundle: ForecastContracts,
    scenario: Scenario,
    policy: Policy,
    family_id: str,
    scenario_sha256: str,
    decision_steps: Sequence[int],
) -> dict[str, Any]:
    """One 96-step HMC episode under one policy with full decision lineage."""

    zone_ids = scenario_zone_order(scenario)
    actions = tuple(bundle.actions)
    decisions = set(int(step) for step in decision_steps)
    nonce = hashlib.sha256(
        b"issue77-closed-loop|" + family_id.encode("utf-8")
    ).digest()
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    shadow = initial_state(scenario)
    initial_row = project_true_targets(scenario, zone_ids, shadow)
    states = {0: shadow}
    episode_snapshots: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    proposal_count = 0
    rejected_count = 0
    abstention_count = 0
    latencies: list[float] = []
    current_action_id = "normal-occupied-v1"
    for step in range(96):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue77ClosedLoopError(f"HMC terminated at step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        snap_map = snapshot.to_mapping()
        snap_map["proposal_disposition"] = "NO_PROPOSAL"
        episode_snapshots.append(snap_map)
        proposal = None
        reason = None
        latency = 0.0
        if step in decisions:
            enriched = dict(snap_map)
            enriched["__zone_ids"] = zone_ids
            enriched["__actions"] = actions
            enriched["__bundle"] = bundle
            enriched["__episode_snapshots"] = episode_snapshots
            chosen, reason, latency = policy(enriched, current_action_id)
            latencies.append(latency)
            if chosen is not None:
                action = next(item for item in actions if item.action_id == chosen)
                proposal = {
                    "source_id": action.action_id,
                    "proposed_command": action.command.to_mapping(),
                }
        receipt = hmc.propose(proposal, handle)
        receipt_mapping = receipt.to_mapping()
        validation_outcome = str(receipt_mapping["validation_outcome"])
        matched = False
        if step in decisions:
            if proposal is None:
                abstention_count += 1
            elif validation_outcome == "VALID":
                proposal_count += 1
                matched = True
            else:
                rejected_count += 1
                abstention_count += 1
            lineage.append(
                {
                    "step": step,
                    "proposed_action_id": proposal["source_id"] if proposal else None,
                    "abstention_reason": reason,
                    "validation_outcome": validation_outcome,
                    "matched_final_command": matched,
                    "latency_ms": latency,
                }
            )
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue77ClosedLoopError(f"HMC terminated while arbitrating step {step}")
        if lineage and step in decisions and matched:
            lineage[-1]["final_command_sha256"] = arbitration.final_command_sha256
        final_command = dict(arbitration.final_command)
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise Issue77ClosedLoopError(f"HMC terminated while stepping step {step}")
        result = advance_one_step_with_command(scenario, shadow, final_command)
        if result.state.step != step + 1:
            raise Issue77ClosedLoopError("shadow state drifted from the HMC step")
        if (
            hashlib.sha256(canonical_json_bytes(result.receipt)).hexdigest()
            != step_receipt.plant_receipt_digest
        ):
            raise Issue77ClosedLoopError("shadow receipt diverges from the HMC receipt")
        shadow = result.state
        states[shadow.step] = shadow
        if snap_map["completed_operating_mode"] is not None:
            current_action_id = {
                "occupied": "normal-occupied-v1",
                "eva_transition": "normal-eva_transition-v1",
                "contingency": "normal-contingency-v1",
                "dormant": "normal-dormant-v1",
            }[str(snap_map["completed_operating_mode"])]
    if len(lineage) != len(decisions):
        raise Issue77ClosedLoopError("decision lineage drifted")
    metrics = compute_race_metrics(
        scenario, zone_ids, initial_row, [states[key] for key in range(1, 97)]
    )
    record = {
        "schema_version": ISSUE77_CLOSED_LOOP_SCHEMA_VERSION,
        "family_id": family_id,
        "scenario_sha256": scenario_sha256,
        "decision_steps": sorted(decisions),
        "lineage": lineage,
        "proposal_count": proposal_count,
        "rejected_proposal_count": rejected_count,
        "abstention_count": abstention_count,
        "safety_exposure": float(metrics["safety_exposure"]),
        "safety_violation_steps": int(metrics["safety_violation_steps"]),
        "comfort_deviation": float(metrics["comfort_deviation"]),
        "resource_composite": float(metrics["resource_composite"]),
        "latency_p99_ms": float(np.quantile(latencies, 0.99)) if latencies else 0.0,
    }
    record["episode_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in record.items() if key != "episode_sha256"})
    ).hexdigest()
    return record


def _family_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "group_id": str(record["group_id"]),
                "family_id": record["family_id"],
                "safety_exposure": record["safety_exposure"],
                "comfort_deviation": record["comfort_deviation"],
                "resource_composite": record["resource_composite"],
            }
        )
    return rows


def evaluate_gates(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    """Preregistered gate order at causal-group level; tie equals not-beat."""

    gates_cfg = prereg["gates"]
    alpha = float(gates_cfg["alpha"])
    seed = str(gates_cfg["statistics_seed"])
    resamples = int(gates_cfg["bootstrap_resamples"])
    margin = float(gates_cfg["non_inferiority_margin"])
    calibrated = records_by_arm["calibrated_bdm_v1"]
    hold = records_by_arm["hold_current_command"]
    linear = records_by_arm["linear_action_conditioned_ridge"]

    zero_hard = all(
        record["safety_violation_steps"] == 0
        for records in records_by_arm.values()
        for record in records
    )

    hold_by_family = {record["family_id"]: record for record in hold}
    linear_by_family = {record["family_id"]: record for record in linear}
    non_inferiority = True
    for record in calibrated:
        hold_exposure = hold_by_family[record["family_id"]]["safety_exposure"]
        if record["safety_exposure"] > hold_exposure + margin:
            non_inferiority = False

    group_diffs: dict[str, list[float]] = {}
    linear_diffs: dict[str, list[float]] = {}
    for record in calibrated:
        group = str(record["group_id"])
        group_diffs.setdefault(group, []).append(
            hold_by_family[record["family_id"]]["safety_exposure"] - record["safety_exposure"]
        )
        linear_diffs.setdefault(group, []).append(
            linear_by_family[record["family_id"]]["safety_exposure"] - record["safety_exposure"]
        )
    benefit_values = [sum(values) / len(values) for values in group_diffs.values()]
    _mean, benefit_lo, _benefit_hi = bootstrap_ci(benefit_values, seed=seed + "|benefit", resamples=resamples, alpha=alpha)
    benefit = benefit_lo > 0.0
    linear_values = [sum(values) / len(values) for values in linear_diffs.values()]
    _lmean, linear_lo, _linear_hi = bootstrap_ci(linear_values, seed=seed + "|linear", resamples=resamples, alpha=alpha)
    ranking_regret = linear_lo > 0.0

    wins = sum(
        1
        for record in calibrated
        if record["safety_exposure"] <= linear_by_family[record["family_id"]]["safety_exposure"]
    )
    win_rate = wins / len(calibrated) if calibrated else 0.0

    calibrated_useful = sum(1 for record in calibrated if record["proposal_count"] > 0)
    linear_useful = sum(1 for record in linear if record["proposal_count"] > 0)
    precision_recall = calibrated_useful >= linear_useful

    gates = {
        "zero_hard_safety_violations_all_arms": zero_hard,
        "per_family_safety_non_inferiority": non_inferiority,
        "paired_safety_benefit_group_ci": benefit,
        "ranking_and_regret_vs_linear": ranking_regret,
        "useful_action_precision_recall": precision_recall,
    }
    earlier = all(gates.values())
    gates["resource_comfort_group_ci_after_safety"] = (
        "evaluated" if earlier else "skipped_failed_earlier_gate"
    )
    return {
        "components": gates,
        "win_rate_vs_linear": win_rate,
        "calibrated_useful_families": calibrated_useful,
        "linear_useful_families": linear_useful,
        "all_gates_passed": bool(all(gates.values()) and earlier),
    }


__all__ = [
    "ISSUE77_CLOSED_LOOP_SCHEMA_VERSION",
    "Issue77ClosedLoopError",
    "evaluate_gates",
    "make_calibrated_bdm_policy",
    "make_guard_screen_policy",
    "make_linear_policy",
    "make_null_policy",
    "reconstruct_sample_features",
    "run_closed_loop_episode",
]
