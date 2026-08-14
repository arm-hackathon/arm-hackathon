"""Fail-closed D2 pilot design, continuation-plan and exclusion contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping

from ..control_trace import parse_control_trace, replay_control_trace
from ..hmc import HabitatManagementComputer
from ..scenario import Scenario, ScenarioValidationError
from .contracts import load_forecast_contracts
from .corpus import canonical_json_bytes


APPROVED_ROSTER_SHA256: Final = (
    "9514a25548d95047f3e707d1f2b27c76c3b09378653ecd270cdc9ae2845b06d1"
)
APPROVED_ROSTER_BYTES_SHA256: Final = (
    "357ad3286cf80ee1b582096b8251f076e4d111b84f78c51c273a9a89d4921528"
)
APPROVED_PROFILE_ACTION_SHA256: Final = (
    "535cde8c397b115d5dd0b46c257462527f1e3eedfa3fb8560f02e45520854141"
)
APPROVED_PROFILE_ACTION_BYTES_SHA256: Final = (
    "b6403d9f0763c8c522185c428095e472e8e70acc555a15a32912f4eb606a71a5"
)
_ROSTER_RELATIVE: Final = Path(
    "docs/plans/2026-08-14-habitat-v2-forecast-timing-pilot-roster-proposal-v1.json"
)
_PROFILE_RELATIVE: Final = Path(
    "docs/plans/2026-08-14-habitat-v2-forecast-pilot-profile-action-proposal-v1.json"
)
_EXPECTED_MODES: Final = (
    "occupied",
    "eva_transition",
    "contingency",
    "dormant",
)
_EXPECTED_LOADS: Final = ("LOW", "NOMINAL", "HIGH")
_EXPECTED_ROLES: Final = (
    "balanced-initial-state",
    "thermal-air-processing-skew",
    "crew-metabolic-humidity-skew",
    "pressure-inventory-skew",
    "reduced-resource-inventory",
)
_EXPECTED_ACTIONS: Final = (
    "normal-occupied-v1",
    "normal-eva_transition-v1",
    "normal-contingency-v1",
    "normal-dormant-v1",
)
_EXPECTED_TREATMENTS: Final = tuple(f"T{index:02d}" for index in range(1, 13))
_EXPECTED_REPETITIONS: Final = ("R01", "R02")
_EXPECTED_ANCHORS: Final = (16, 40, 64)
_ROSTER_FIELDS: Final = frozenset(
    {
        "schema_version",
        "ratification_status",
        "d1_candidate_sha",
        "namespace",
        "shape",
        "clusters",
        "action_ids",
        "treatment_roles",
        "canonical_corpus_exclusion",
        "unresolved_before_ratification",
        "proposal_sha256",
    }
)
_PROFILE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "ratification_status",
        "authority",
        "foundation",
        "identity_dag",
        "counts",
        "scenario_constructor",
        "load_regime_fields",
        "load_regime_exclusions",
        "numeric_arithmetic",
        "load_regimes",
        "semantic_cluster_profiles",
        "treatments",
        "treatment_schedule",
        "treatment_profile_materialization",
        "actions",
        "noise_repetitions",
        "hmc_reset_nonce",
        "required_pre_generation_checks",
        "permissions",
        "human_ratification_items",
        "human_ratification_resolution",
        "proposal_sha256",
    }
)


class PilotContractError(ValueError):
    """Approved pilot bytes, lineage or resource evidence are not closed."""


@dataclass(frozen=True, slots=True)
class PilotCluster:
    cluster_index: int
    cluster_id: str
    operating_mode: str
    load_regime: str
    semantic_profile_role: str


@dataclass(frozen=True, slots=True)
class PilotDesign:
    roster_sha256: str
    roster_bytes_sha256: str
    profile_action_sha256: str
    profile_action_bytes_sha256: str
    namespace_prefix: str
    clusters: tuple[PilotCluster, ...]
    operating_modes: tuple[str, ...]
    load_regimes: tuple[str, ...]
    semantic_profile_roles: tuple[str, ...]
    action_ids: tuple[str, ...]
    treatment_ids: tuple[str, ...]
    repetition_ids: tuple[str, ...]
    anchor_completed_steps: tuple[int, ...]
    transient_treatment_interval: tuple[int, int]
    persistent_treatment_interval: tuple[int, int]
    forbidden_cluster_ids: frozenset[str]
    noise_domain: str
    reset_nonce_domain: str


@dataclass(frozen=True, slots=True)
class PilotContinuation:
    continuation_id: str
    pair_id: str
    matched_control_id: str
    cluster_id: str
    operating_mode: str
    load_regime: str
    semantic_profile_role: str
    repetition_id: str
    member_id: str
    treatment_duration: str
    treatment_interval: tuple[int, int] | None
    anchor_completed_step: int
    variant: str
    action_id: str
    noise_seed: int
    hmc_reset_nonce_hex: str


@dataclass(frozen=True, slots=True)
class PilotResourcePreflight:
    preflight_sha256: str
    preflight_bytes_sha256: str
    planned_hmc_runs: int
    benchmark_hmc_runs: int
    measured_wall_time_seconds: float
    measured_peak_rss_bytes: int
    measured_artifact_bytes: int
    projected_wall_time_seconds: float
    projected_peak_rss_bytes: int
    projected_artifact_bytes: int
    verdict: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_constant(value: str) -> None:
    raise PilotContractError(f"non-finite JSON constant {value!r} is forbidden")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotContractError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotContractError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise PilotContractError(f"{label} must be one JSON object")
    try:
        canonical_json_bytes(value)
    except ValueError as error:
        raise PilotContractError(f"{label} is not finite canonical JSON") from error
    return value


def _load_bound_json(
    path: Path,
    *,
    label: str,
    self_hash_field: str,
    expected_semantic_sha256: str,
    expected_bytes_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not _is_sha256(expected_semantic_sha256) or not _is_sha256(
        expected_bytes_sha256
    ):
        raise PilotContractError(f"expected {label} identity is malformed")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PilotContractError(f"{label} bytes cannot be read") from error
    raw_sha256 = _sha256(raw)
    if raw_sha256 != expected_bytes_sha256:
        raise PilotContractError(f"bytes differ from expected {label} identity")
    value = _strict_json(raw, label=label)
    declared = value.get(self_hash_field)
    body = dict(value)
    body.pop(self_hash_field, None)
    semantic_sha256 = _sha256(canonical_json_bytes(body))
    if declared != semantic_sha256 or semantic_sha256 != expected_semantic_sha256:
        raise PilotContractError(
            f"semantic bytes differ from expected {label} identity"
        )
    return value, raw_sha256


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise PilotContractError(
            f"{label} has unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def _require_positive_number(value: Any, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value <= 0:
        raise PilotContractError(f"{label} must be a positive finite number")
    return float(value)


def _validate_roster(value: dict[str, Any]) -> tuple[PilotCluster, ...]:
    _require_exact_fields(value, _ROSTER_FIELDS, label="roster")
    if (
        value["schema_version"]
        != "aeolus_habitat_v2_forecast_timing_pilot_roster_proposal_v1"
        or value["ratification_status"] != "APPROVED"
        or value["d1_candidate_sha"] != "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3"
        or value["namespace"] != "pilot-v1"
        or value["unresolved_before_ratification"] != []
    ):
        raise PilotContractError("roster approval or foundation identity drifts")
    shape = value["shape"]
    if type(shape) is not dict or (
        tuple(shape.get("operating_modes", ())) != _EXPECTED_MODES
        or tuple(shape.get("load_regimes", ())) != _EXPECTED_LOADS
        or tuple(shape.get("semantic_profile_roles", ())) != _EXPECTED_ROLES
        or tuple(shape.get("history_candidates", ())) != (4, 8, 16)
        or tuple(shape.get("horizon_candidates", ())) != (2, 4, 8)
        or tuple(shape.get("anchor_completed_steps", ())) != _EXPECTED_ANCHORS
        or shape.get("cluster_count") != 60
        or shape.get("scenario_steps") != 72
        or shape.get("treatment_onset_completed_step") != 25
        or tuple(shape.get("transient_treatment_interval", ())) != (25, 49)
        or tuple(shape.get("persistent_treatment_interval", ())) != (25, 73)
    ):
        raise PilotContractError("roster shape drifts from the approved design")
    if (
        tuple(value["action_ids"]) != _EXPECTED_ACTIONS
        or len(value["treatment_roles"]) != 13
    ):
        raise PilotContractError("roster action or treatment coverage drifts")

    rows = value["clusters"]
    if type(rows) is not list or len(rows) != 60:
        raise PilotContractError("roster must contain exactly 60 clusters")
    clusters: list[PilotCluster] = []
    expected_rows = [
        (mode, load, role)
        for mode in _EXPECTED_MODES
        for load in _EXPECTED_LOADS
        for role in _EXPECTED_ROLES
    ]
    for index, (row, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        if type(row) is not dict or set(row) != {
            "cluster_id",
            "operating_mode",
            "load_regime",
            "semantic_profile_role",
            "canonical_corpus_eligible",
        }:
            raise PilotContractError("roster cluster schema is not closed")
        mode, load, role = expected
        cluster_id = f"pilot-v1/{mode}/{load.lower()}/{role}"
        if row != {
            "cluster_id": cluster_id,
            "operating_mode": mode,
            "load_regime": load,
            "semantic_profile_role": role,
            "canonical_corpus_eligible": False,
        }:
            raise PilotContractError("roster cluster order or semantics drift")
        clusters.append(PilotCluster(index, cluster_id, mode, load, role))
    exclusion = value["canonical_corpus_exclusion"]
    if type(exclusion) is not dict or exclusion != {
        "forbid_namespace_prefix": "pilot-v1/",
        "forbid_exact_cluster_ids": [cluster.cluster_id for cluster in clusters],
        "derived_family_exclusion_required": True,
    }:
        raise PilotContractError("canonical pilot exclusion contract drifts")
    return tuple(clusters)


def _validate_profile(
    value: dict[str, Any], repo_root: Path, clusters: tuple[PilotCluster, ...]
) -> None:
    _require_exact_fields(value, _PROFILE_FIELDS, label="profile/action packet")
    if (
        value["schema_version"]
        != "aeolus_habitat_v2_forecast_pilot_profile_action_proposal_v1"
        or value["ratification_status"] != "APPROVED"
        or value["human_ratification_resolution"]
        != "APPROVED_BY_PROJECT_OWNER_2026-08-14"
    ):
        raise PilotContractError("profile/action approval identity drifts")
    foundation = value["foundation"]
    if foundation.get(
        "d1_candidate_git_sha"
    ) != "c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3" or foundation.get(
        "roster_proposal"
    ) != {
        "semantic_sha256": APPROVED_ROSTER_SHA256,
        "raw_bytes_sha256": APPROVED_ROSTER_BYTES_SHA256,
    }:
        raise PilotContractError("profile/action roster binding drifts")
    try:
        bundle = load_forecast_contracts(repo_root)
    except Exception as error:
        raise PilotContractError(
            "frozen D1 forecast contracts cannot be loaded"
        ) from error
    if (
        foundation.get("action_catalogue", {}).get("catalogue_sha256")
        != bundle.action_catalogue_sha256
        or foundation.get("hmc_binding", {}).get("binding_sha256")
        != bundle.binding_sha256
        or foundation.get("observable_topology_sha256") != bundle.topology.sha256
    ):
        raise PilotContractError("profile/action foundation contract drifts")
    source = foundation.get("source_scenario")
    if type(source) is not dict:
        raise PilotContractError("source scenario binding is missing")
    scenario_path = repo_root / str(source.get("path", ""))
    try:
        scenario_raw = scenario_path.read_bytes()
        scenario_value = json.loads(scenario_raw)
        from aeolus.habitat_v2.scenario import Scenario

        scenario = Scenario.from_mapping(scenario_value)
    except Exception as error:
        raise PilotContractError("bound source scenario cannot be verified") from error
    if (
        _sha256(scenario_raw) != source.get("raw_bytes_sha256")
        or scenario.scenario_sha256 != source.get("scenario_sha256")
        or scenario.scenario_sha256
        != "a9ee8eecdb4a952ef95347edcabb7dad614280eb496877cc9cddf8a5c9f77de7"
    ):
        raise PilotContractError("source scenario identity drifts")

    if value["counts"] != {
        "clusters": 60,
        "repetitions": 2,
        "members_per_family": 13,
        "anchors": 3,
        "proposal_actions": 4,
        "proposal_hmc_runs": 18_720,
        "matched_control_hmc_runs": 4_680,
        "total_hmc_runs": 23_400,
        "timing_views_if_all_nine_pairs_materialized": 168_480,
    }:
        raise PilotContractError("profile/action continuation arithmetic drifts")
    constructor = value["scenario_constructor"]
    if (
        constructor.get("schema_version") != "aeolus_habitat_v2_scenario_v5"
        or constructor.get("steps") != 72
        or constructor.get("dt_seconds") != 60.0
        or constructor.get("fixed_topology") is not True
    ):
        raise PilotContractError("scenario-constructor boundary drifts")
    if tuple(item.get("roster_load_regime") for item in value["load_regimes"]) != (
        _EXPECTED_LOADS
    ):
        raise PilotContractError("load-regime mapping drifts")
    profiles = value["semantic_cluster_profiles"]
    if tuple(item.get("roster_role") for item in profiles) != _EXPECTED_ROLES:
        raise PilotContractError("semantic profile mapping drifts")
    pressure = next(
        (item for item in profiles if item.get("id") == "pressure_inventory_skew"),
        None,
    )
    reduced = next(
        (item for item in profiles if item.get("id") == "reduced_resource_inventory"),
        None,
    )
    if (
        not isinstance(pressure, dict)
        or "cannot support a pressure-driven airflow claim"
        not in pressure.get("scientific_claim_boundary", "")
        or not isinstance(reduced, dict)
        or reduced.get("sensitivity_rehearsal_gate", {}).get("permanently_excluded")
        is not True
        or reduced.get("sensitivity_rehearsal_gate", {}).get("failure")
        != "REPLACE_PROFILE_BEFORE_PILOT"
    ):
        raise PilotContractError("corrected profile claim boundaries drift")
    treatments = value["treatments"]
    if tuple(item.get("id") for item in treatments) != _EXPECTED_TREATMENTS:
        raise PilotContractError("treatment identity order drifts")
    if treatments[-1].get("class") != "interaction_ood_stress":
        raise PilotContractError("T12 claim boundary drifts")
    schedule = value["treatment_schedule"]
    if (
        tuple(schedule.get("transient_half_open_steps", ())) != (25, 49)
        or tuple(schedule.get("persistent_half_open_steps", ())) != (25, 73)
        or schedule.get("counts")
        != {
            "assignments": 720,
            "transient": 360,
            "persistent": 360,
            "per_treatment_transient": 30,
            "per_treatment_persistent": 30,
        }
    ):
        raise PilotContractError("treatment schedule drifts")
    if value["actions"].get("proposal_action_ids") != list(_EXPECTED_ACTIONS):
        raise PilotContractError("proposal action catalogue drifts")
    if tuple(value["noise_repetitions"].get("ids", ())) != _EXPECTED_REPETITIONS:
        raise PilotContractError("noise repetition contract drifts")
    if value["permissions"] != {
        "learned_actuator_authority_allowed": False,
        "model_training_allowed": False,
        "pilot_generation_allowed": False,
        "publication_allowed": False,
        "scenario_generation_allowed": False,
        "validation_access_allowed": False,
    }:
        raise PilotContractError("profile/action packet grants forbidden authority")
    if len(clusters) != value["counts"]["clusters"]:
        raise PilotContractError("profile/action packet does not bind roster count")


def load_approved_pilot_design(
    repo_root: str | Path,
    *,
    roster_path: str | Path | None = None,
    profile_path: str | Path | None = None,
) -> PilotDesign:
    """Load only the exact approved roster and profile/action bytes."""
    root = Path(repo_root).resolve()
    roster_file = (
        Path(roster_path) if roster_path is not None else root / _ROSTER_RELATIVE
    )
    profile_file = (
        Path(profile_path) if profile_path is not None else root / _PROFILE_RELATIVE
    )
    roster, roster_raw_sha = _load_bound_json(
        roster_file,
        label="roster",
        self_hash_field="proposal_sha256",
        expected_semantic_sha256=APPROVED_ROSTER_SHA256,
        expected_bytes_sha256=APPROVED_ROSTER_BYTES_SHA256,
    )
    clusters = _validate_roster(roster)
    profile, profile_raw_sha = _load_bound_json(
        profile_file,
        label="profile/action packet",
        self_hash_field="proposal_sha256",
        expected_semantic_sha256=APPROVED_PROFILE_ACTION_SHA256,
        expected_bytes_sha256=APPROVED_PROFILE_ACTION_BYTES_SHA256,
    )
    _validate_profile(profile, root, clusters)
    return PilotDesign(
        roster_sha256=APPROVED_ROSTER_SHA256,
        roster_bytes_sha256=roster_raw_sha,
        profile_action_sha256=APPROVED_PROFILE_ACTION_SHA256,
        profile_action_bytes_sha256=profile_raw_sha,
        namespace_prefix="pilot-v1/",
        clusters=clusters,
        operating_modes=_EXPECTED_MODES,
        load_regimes=_EXPECTED_LOADS,
        semantic_profile_roles=_EXPECTED_ROLES,
        action_ids=_EXPECTED_ACTIONS,
        treatment_ids=_EXPECTED_TREATMENTS,
        repetition_ids=_EXPECTED_REPETITIONS,
        anchor_completed_steps=_EXPECTED_ANCHORS,
        transient_treatment_interval=(25, 49),
        persistent_treatment_interval=(25, 73),
        forbidden_cluster_ids=frozenset(cluster.cluster_id for cluster in clusters),
        noise_domain=profile["noise_repetitions"]["domain"],
        reset_nonce_domain=profile["hmc_reset_nonce"]["domain"],
    )


def _noise_seed(design: PilotDesign, cluster_id: str, repetition_id: str) -> int:
    digest = hashlib.sha256(
        design.noise_domain.encode("utf-8")
        + b"\0"
        + bytes.fromhex(design.roster_sha256)
        + b"\0"
        + cluster_id.encode("utf-8")
        + b"\0"
        + repetition_id.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _reset_nonce(
    design: PilotDesign, cluster_id: str, repetition_id: str, anchor: int
) -> bytes:
    return hashlib.sha256(
        design.reset_nonce_domain.encode("utf-8")
        + b"\0"
        + bytes.fromhex(design.roster_sha256)
        + b"\0"
        + cluster_id.encode("utf-8")
        + b"\0"
        + repetition_id.encode("ascii")
        + b"\0"
        + str(anchor).encode("ascii")
    ).digest()


def _identity(domain: bytes, value: Mapping[str, Any]) -> str:
    return _sha256(domain + b"\0" + canonical_json_bytes(dict(value)))


def iter_pilot_continuations(design: PilotDesign):  # type: ignore[no-untyped-def]
    """Yield the frozen logical plan only; this function performs no simulation."""
    members = ("HEALTHY",) + design.treatment_ids
    for cluster in design.clusters:
        for repetition_id in design.repetition_ids:
            noise_seed = _noise_seed(design, cluster.cluster_id, repetition_id)
            for member_id in members:
                if member_id == "HEALTHY":
                    treatment_duration = "NONE"
                    treatment_interval = None
                else:
                    treatment_index = int(member_id[1:]) - 1
                    transient = (cluster.cluster_index + treatment_index) % 2 == 0
                    treatment_duration = "TRANSIENT" if transient else "PERSISTENT"
                    treatment_interval = (
                        design.transient_treatment_interval
                        if transient
                        else design.persistent_treatment_interval
                    )
                for anchor in design.anchor_completed_steps:
                    nonce_hex = _reset_nonce(
                        design, cluster.cluster_id, repetition_id, anchor
                    ).hex()
                    pair_body = {
                        "cluster_id": cluster.cluster_id,
                        "repetition_id": repetition_id,
                        "member_id": member_id,
                        "anchor_completed_step": anchor,
                    }
                    pair_id = _identity(
                        b"aeolus-habitat-v2-forecast-pilot-pair-v1", pair_body
                    )
                    control_id = _identity(
                        b"aeolus-habitat-v2-forecast-pilot-continuation-v1",
                        {"pair_id": pair_id, "action_id": "NO_PROPOSAL"},
                    )
                    common = {
                        "pair_id": pair_id,
                        "matched_control_id": control_id,
                        "cluster_id": cluster.cluster_id,
                        "operating_mode": cluster.operating_mode,
                        "load_regime": cluster.load_regime,
                        "semantic_profile_role": cluster.semantic_profile_role,
                        "repetition_id": repetition_id,
                        "member_id": member_id,
                        "treatment_duration": treatment_duration,
                        "treatment_interval": treatment_interval,
                        "anchor_completed_step": anchor,
                        "noise_seed": noise_seed,
                        "hmc_reset_nonce_hex": nonce_hex,
                    }
                    yield PilotContinuation(
                        continuation_id=control_id,
                        variant="MATCHED_CONTROL",
                        action_id="NO_PROPOSAL",
                        **common,
                    )
                    for action_id in design.action_ids:
                        continuation_id = _identity(
                            b"aeolus-habitat-v2-forecast-pilot-continuation-v1",
                            {"pair_id": pair_id, "action_id": action_id},
                        )
                        yield PilotContinuation(
                            continuation_id=continuation_id,
                            variant="ACTION_PROPOSAL",
                            action_id=action_id,
                            **common,
                        )


def validate_canonical_pilot_exclusion(
    design: PilotDesign,
    *,
    candidate_cluster_ids: tuple[str, ...],
    ancestor_cluster_ids: tuple[str, ...] = (),
) -> None:
    """Reject exact, namespace or inherited pilot lineage from canonical data."""
    values = candidate_cluster_ids + ancestor_cluster_ids
    if any(type(value) is not str or not value for value in values):
        raise PilotContractError("canonical cluster lineage identity is malformed")
    if any(
        value.startswith(design.namespace_prefix)
        or value in design.forbidden_cluster_ids
        for value in values
    ):
        raise PilotContractError("canonical candidate contains forbidden pilot lineage")


def load_resource_preflight(
    path: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_preflight_bytes_sha256: str,
) -> PilotResourcePreflight:
    """Load only independently pinned, canonical passing resource evidence."""
    value, raw_sha = _load_bound_json(
        Path(path),
        label="preflight",
        self_hash_field="preflight_sha256",
        expected_semantic_sha256=expected_preflight_sha256,
        expected_bytes_sha256=expected_preflight_bytes_sha256,
    )
    expected_fields = frozenset(
        {
            "schema_version",
            "roster_sha256",
            "profile_action_sha256",
            "planned_hmc_runs",
            "benchmark_hmc_runs",
            "measured_wall_time_seconds",
            "measured_peak_rss_bytes",
            "measured_artifact_bytes",
            "projected_wall_time_seconds",
            "projected_peak_rss_bytes",
            "projected_artifact_bytes",
            "runtime_within_ceiling",
            "memory_within_ceiling",
            "disk_reserve_preserved",
            "verdict",
            "preflight_sha256",
        }
    )
    _require_exact_fields(value, expected_fields, label="preflight")
    if (
        value["schema_version"]
        != "aeolus_habitat_v2_forecast_pilot_resource_preflight_v1"
        or value["roster_sha256"] != APPROVED_ROSTER_SHA256
        or value["profile_action_sha256"] != APPROVED_PROFILE_ACTION_SHA256
        or value["planned_hmc_runs"] != 23_400
        or value["verdict"] != "PASS"
        or any(
            value[field] is not True
            for field in (
                "runtime_within_ceiling",
                "memory_within_ceiling",
                "disk_reserve_preserved",
            )
        )
    ):
        raise PilotContractError("resource preflight does not authorize the plan")
    benchmark_runs = value["benchmark_hmc_runs"]
    if type(benchmark_runs) is not int or benchmark_runs <= 0:
        raise PilotContractError("preflight benchmark count must be positive")
    measured_wall = _require_positive_number(
        value["measured_wall_time_seconds"], label="measured wall time"
    )
    projected_wall = _require_positive_number(
        value["projected_wall_time_seconds"], label="projected wall time"
    )
    integer_fields = (
        "measured_peak_rss_bytes",
        "measured_artifact_bytes",
        "projected_peak_rss_bytes",
        "projected_artifact_bytes",
    )
    if any(
        type(value[field]) is not int or value[field] <= 0 for field in integer_fields
    ):
        raise PilotContractError("preflight byte counts must be positive integers")
    return PilotResourcePreflight(
        preflight_sha256=value["preflight_sha256"],
        preflight_bytes_sha256=raw_sha,
        planned_hmc_runs=value["planned_hmc_runs"],
        benchmark_hmc_runs=benchmark_runs,
        measured_wall_time_seconds=measured_wall,
        measured_peak_rss_bytes=value["measured_peak_rss_bytes"],
        measured_artifact_bytes=value["measured_artifact_bytes"],
        projected_wall_time_seconds=projected_wall,
        projected_peak_rss_bytes=value["projected_peak_rss_bytes"],
        projected_artifact_bytes=value["projected_artifact_bytes"],
        verdict=value["verdict"],
    )


_QUANTUM: Final = Decimal("1e-12")
_SOURCE_SCENARIO_SHA256: Final = (
    "a9ee8eecdb4a952ef95347edcabb7dad614280eb496877cc9cddf8a5c9f77de7"
)
_SOURCE_SCENARIO_BYTES_SHA256: Final = (
    "45ccb82f3720d71d061afd3cbda5afe328146ce9b870bee9e04c3fabdc99c727"
)


def _decimal(value: object, *, label: str) -> Decimal:
    if type(value) not in (int, float, str):
        raise PilotContractError(f"{label} is not a supported decimal value")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise PilotContractError(f"{label} is not a decimal") from error
    if not result.is_finite():
        raise PilotContractError(f"{label} must be finite")
    return result


def _quantize(value: Decimal, *, label: str) -> Decimal:
    try:
        result = value.quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as error:
        raise PilotContractError(f"{label} cannot be quantized") from error
    if not result.is_finite():
        raise PilotContractError(f"{label} is nonfinite after quantization")
    return result


def _correct_decimal_total(
    values: dict[str, Decimal],
    coefficients: Mapping[str, Decimal],
    *,
    target: Decimal,
    label: str,
) -> None:
    current = sum((values[key] * coefficients[key] for key in values), Decimal(0))
    drift = current - target
    if drift == 0:
        return
    for key in sorted(values):
        coefficient = coefficients[key]
        if coefficient == 0:
            continue
        correction = -drift / coefficient
        units = correction / _QUANTUM
        if units != units.to_integral_value():
            continue
        candidate = _quantize(values[key] + correction, label=f"{label}.{key}")
        if candidate < 0:
            continue
        values[key] = candidate
        corrected = sum(
            (values[item] * coefficients[item] for item in values), Decimal(0)
        )
        if corrected != target:
            raise PilotContractError(f"{label} residual correction is inexact")
        return
    raise PilotContractError(f"{label} has no exact quantized residual correction")


def _load_materialization_inputs(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Scenario]:
    profile, _ = _load_bound_json(
        repo_root / _PROFILE_RELATIVE,
        label="profile",
        self_hash_field="proposal_sha256",
        expected_semantic_sha256=APPROVED_PROFILE_ACTION_SHA256,
        expected_bytes_sha256=APPROVED_PROFILE_ACTION_BYTES_SHA256,
    )
    binding = profile["foundation"]["source_scenario"]
    if binding != {
        "path": "scenarios/habitat_v2_actuator_feedback.json",
        "raw_bytes_sha256": _SOURCE_SCENARIO_BYTES_SHA256,
        "scenario_sha256": _SOURCE_SCENARIO_SHA256,
    }:
        raise PilotContractError("bound source scenario identity drifts")
    source_path = (repo_root / binding["path"]).resolve()
    try:
        raw = source_path.read_bytes()
    except OSError as error:
        raise PilotContractError("bound source scenario cannot be read") from error
    if hashlib.sha256(raw).hexdigest() != _SOURCE_SCENARIO_BYTES_SHA256:
        raise PilotContractError("bound source scenario byte identity drifts")
    source_mapping = _strict_json(raw, label="bound source scenario")
    try:
        source = Scenario.from_mapping(source_mapping)
    except ScenarioValidationError as error:
        raise PilotContractError("bound source scenario is invalid") from error
    if source.scenario_sha256 != _SOURCE_SCENARIO_SHA256:
        raise PilotContractError("bound source scenario semantic identity drifts")
    return profile, source_mapping, source


def _apply_load_transformations(
    timeline_entry: dict[str, Any],
    *,
    multiplier: Decimal,
    profile: Mapping[str, Any],
    load_fields: tuple[str, ...],
) -> None:
    loads = timeline_entry["loads"]
    if not isinstance(loads, dict) or not loads:
        raise PilotContractError("source mode has no load mapping")
    zone_ids = tuple(sorted(loads))
    coefficients = {zone_id: Decimal(1) for zone_id in zone_ids}
    for field in load_fields:
        values: dict[str, Decimal] = {}
        for zone_id in zone_ids:
            zone_load = loads[zone_id]
            if not isinstance(zone_load, dict) or field not in zone_load:
                raise PilotContractError(f"load field {field} is missing")
            value = _quantize(
                _decimal(zone_load[field], label=f"loads.{zone_id}.{field}")
                * multiplier,
                label=f"loads.{zone_id}.{field}",
            )
            if value < 0:
                raise PilotContractError("load multiplier creates a negative load")
            values[zone_id] = value
        for zone_id, value in values.items():
            loads[zone_id][field] = float(value)

    for redistribution in profile["load_redistributions"]:
        field = redistribution["field"]
        if field not in load_fields:
            raise PilotContractError("profile names an unsupported load field")
        weights = redistribution["raw_weight_by_zone"]
        if set(weights) != set(zone_ids):
            raise PilotContractError("load redistribution zone set drifts")
        current = {
            zone_id: _decimal(loads[zone_id][field], label=f"loads.{zone_id}.{field}")
            for zone_id in zone_ids
        }
        target = sum(current.values(), Decimal(0))
        if target == 0:
            continue
        weighted_total = sum(
            (
                current[zone_id] * _decimal(weights[zone_id], label=f"weight.{zone_id}")
                for zone_id in zone_ids
            ),
            Decimal(0),
        )
        if weighted_total <= 0:
            raise PilotContractError("load redistribution has no positive support")
        transformed = {
            zone_id: _quantize(
                current[zone_id]
                * _decimal(weights[zone_id], label=f"weight.{zone_id}")
                * target
                / weighted_total,
                label=f"redistributed.{field}.{zone_id}",
            )
            for zone_id in zone_ids
        }
        _correct_decimal_total(
            transformed,
            coefficients,
            target=target,
            label=f"redistributed.{field}",
        )
        for zone_id, value in transformed.items():
            if value < 0:
                raise PilotContractError("load redistribution creates a negative load")
            loads[zone_id][field] = float(value)


def _apply_initial_transformations(
    data: dict[str, Any], profile: Mapping[str, Any]
) -> None:
    zones = data["zones"]
    if not isinstance(zones, list) or not zones:
        raise PilotContractError("source scenario has no zones")
    by_zone = {zone["id"]: zone for zone in zones}
    if len(by_zone) != len(zones):
        raise PilotContractError("source scenario has duplicate zones")
    volumes = {
        zone_id: _decimal(zone["volume_m3"], label=f"volume.{zone_id}")
        for zone_id, zone in by_zone.items()
    }

    for transform in profile["initial_transforms"]:
        method = transform["method"]
        field = transform["field"]
        if method == "multiply":
            if not field.startswith("initial_utility."):
                raise PilotContractError("multiply transform names an unsupported path")
            utility_field = field.removeprefix("initial_utility.")
            utility = data["initial_utility"]
            if utility_field not in utility:
                raise PilotContractError("multiply transform names a missing utility")
            value = _quantize(
                _decimal(utility[utility_field], label=field)
                * _decimal(transform["factor"], label=f"{field}.factor"),
                label=field,
            )
            if value < 0:
                raise PilotContractError("utility transform creates a negative value")
            utility[utility_field] = float(value)
            continue

        if any(field not in zone["initial"] for zone in zones):
            raise PilotContractError(
                "initial transform names an unsupported zone field"
            )
        original = {
            zone_id: _decimal(zone["initial"][field], label=f"{zone_id}.{field}")
            for zone_id, zone in by_zone.items()
        }
        target = sum(
            (original[zone_id] * volumes[zone_id] for zone_id in by_zone),
            Decimal(0),
        )
        if method == "volume_weighted_mean_preserving_additive_offsets":
            offsets = transform["raw_offset_by_zone"]
            if set(offsets) != set(by_zone):
                raise PilotContractError("initial offset zone set drifts")
            parsed_offsets = {
                zone_id: _decimal(offsets[zone_id], label=f"offset.{zone_id}")
                for zone_id in by_zone
            }
            total_volume = sum(volumes.values(), Decimal(0))
            mean_offset = (
                sum(
                    (parsed_offsets[zone_id] * volumes[zone_id] for zone_id in by_zone),
                    Decimal(0),
                )
                / total_volume
            )
            transformed = {
                zone_id: _quantize(
                    original[zone_id] + parsed_offsets[zone_id] - mean_offset,
                    label=f"initial.{field}.{zone_id}",
                )
                for zone_id in by_zone
            }
        elif method == "exact_additive_offsets":
            offsets = transform["offset_by_zone"]
            if set(offsets) != set(by_zone):
                raise PilotContractError("exact offset zone set drifts")
            parsed_offsets = {
                zone_id: _decimal(offsets[zone_id], label=f"offset.{zone_id}")
                for zone_id in by_zone
            }
            expected_sum = _decimal(
                transform["volume_weighted_sum_pa_m3"],
                label="volume_weighted_sum_pa_m3",
            )
            actual_sum = sum(
                (parsed_offsets[zone_id] * volumes[zone_id] for zone_id in by_zone),
                Decimal(0),
            )
            if actual_sum != expected_sum:
                raise PilotContractError("exact offsets violate their declared total")
            transformed = {
                zone_id: _quantize(
                    original[zone_id] + parsed_offsets[zone_id],
                    label=f"initial.{field}.{zone_id}",
                )
                for zone_id in by_zone
            }
        else:
            raise PilotContractError("initial transform method is unsupported")
        _correct_decimal_total(
            transformed,
            volumes,
            target=target,
            label=f"initial.{field}",
        )
        for zone_id, value in transformed.items():
            if value < 0:
                raise PilotContractError("initial transform creates a negative value")
            by_zone[zone_id]["initial"][field] = float(value)


def materialize_pilot_scenario(
    repo_root: str | Path,
    design: PilotDesign,
    *,
    cluster_id: str,
    member_id: str,
    repetition_id: str,
) -> Scenario:
    """Materialize one exact 72-step D2 scenario without executing the HMC."""
    if type(design) is not PilotDesign:
        raise PilotContractError("materialization requires the exact pilot design")
    if repetition_id not in design.repetition_ids:
        raise PilotContractError("materialization repetition is unsupported")
    if member_id != "HEALTHY" and member_id not in design.treatment_ids:
        raise PilotContractError("materialization member is unsupported")
    cluster_by_id = {cluster.cluster_id: cluster for cluster in design.clusters}
    if cluster_id not in cluster_by_id:
        raise PilotContractError(
            "materialization cluster is outside the approved roster"
        )
    cluster = cluster_by_id[cluster_id]

    root = Path(repo_root).resolve()
    profile_packet, source_mapping, source = _load_materialization_inputs(root)
    residual_rule = profile_packet["numeric_arithmetic"].get("residual_correction")
    if (
        type(residual_rule) is not str
        or "integer number of 1e-12 units" not in residual_rule
    ):
        raise PilotContractError("profile residual-correction rule is not frozen")
    profiles = {
        item["roster_role"]: item
        for item in profile_packet["semantic_cluster_profiles"]
    }
    if set(profiles) != set(_EXPECTED_ROLES):
        raise PilotContractError("semantic profile role map drifts")
    profile = profiles[cluster.semantic_profile_role]
    regime_by_role = {
        item["roster_load_regime"]: item for item in profile_packet["load_regimes"]
    }
    if set(regime_by_role) != set(_EXPECTED_LOADS):
        raise PilotContractError("load regime role map drifts")
    multiplier = _decimal(
        regime_by_role[cluster.load_regime]["all_zone_load_fields_multiplier"],
        label="load regime multiplier",
    )

    data = deepcopy(source_mapping)
    matching_timeline = [
        item
        for item in source_mapping["timeline"]
        if item["operating_mode"] == cluster.operating_mode
    ]
    if len(matching_timeline) != 1:
        raise PilotContractError("source mode selection is not unique")
    timeline_entry = deepcopy(matching_timeline[0])
    timeline_entry["start_step"] = 0
    timeline_entry["end_step"] = 72
    _apply_load_transformations(
        timeline_entry,
        multiplier=multiplier,
        profile=profile,
        load_fields=tuple(profile_packet["load_regime_fields"]),
    )
    data["timeline"] = [timeline_entry]
    data["steps"] = 72
    _apply_initial_transformations(data, profile)

    if member_id == "HEALTHY":
        data["fault_profiles"] = []
    else:
        treatment_by_id = {item["id"]: item for item in profile_packet["treatments"]}
        if set(treatment_by_id) != set(design.treatment_ids):
            raise PilotContractError("treatment profile map drifts")
        treatment_index = int(member_id[1:]) - 1
        transient = (cluster.cluster_index + treatment_index) % 2 == 0
        start_step, end_step = (
            design.transient_treatment_interval
            if transient
            else design.persistent_treatment_interval
        )
        materialized_faults = []
        for ordinal, base in enumerate(treatment_by_id[member_id]["profiles"], start=1):
            fault = deepcopy(base)
            fault["id"] = f"{cluster_id}.{member_id}.P{ordinal:02d}"
            fault["start_step"] = start_step
            fault["end_step"] = end_step
            materialized_faults.append(fault)
        data["fault_profiles"] = materialized_faults

    data["sensor_model"]["random_seed"] = _noise_seed(
        design,
        cluster_id=cluster_id,
        repetition_id=repetition_id,
    )
    try:
        scenario = Scenario.from_mapping(data)
    except ScenarioValidationError as error:
        raise PilotContractError("materialized pilot scenario is invalid") from error
    if canonical_json_bytes(data["air_network"]) != canonical_json_bytes(
        source.data["air_network"]
    ):
        raise PilotContractError("materialization changes the fixed air topology")
    return scenario


_REHEARSAL_SCHEMA: Final = "aeolus_habitat_v2_forecast_profile_sensitivity_rehearsal_v1"
_REHEARSAL_NAMESPACE: Final = "pilot-profile-sensitivity-rehearsal-v1"
_HMC_IMPLEMENTATION_GIT_SHA: Final = "79d6a718e0d44122a763bb72f9c8ed929f39fd23"
_HMC_CONTRACT_SHA256: Final = (
    "9f4d269ad8d073d6370f5239d8a78f2541db3001097a460447a8feb84fee2414"
)
_REHEARSAL_CHANNELS: Final = (
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_remaining_fraction",
)
_REHEARSAL_ROWS: Final = tuple(
    step for anchor in (16, 40, 64) for step in range(anchor, anchor + 9)
)
_REHEARSAL_THRESHOLD: Final = 0.02
_REHEARSAL_EXECUTION_STEPS: Final = 73


def _rehearsal_seed(design: PilotDesign, mode: str, load: str) -> int:
    digest = hashlib.sha256(
        _REHEARSAL_NAMESPACE.encode("ascii")
        + b"\0sensor-seed\0"
        + bytes.fromhex(design.roster_sha256)
        + b"\0"
        + mode.encode("ascii")
        + b"\0"
        + load.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _rehearsal_nonce(design: PilotDesign, mode: str, load: str) -> bytes:
    return hashlib.sha256(
        _REHEARSAL_NAMESPACE.encode("ascii")
        + b"\0hmc-reset\0"
        + bytes.fromhex(design.roster_sha256)
        + b"\0"
        + mode.encode("ascii")
        + b"\0"
        + load.encode("ascii")
    ).digest()


def _to_rehearsal_scenario(scenario: Scenario, *, sensor_seed: int) -> Scenario:
    data = _strict_json(scenario.canonical_bytes, label="materialized pilot scenario")
    if data["steps"] != 72 or len(data["timeline"]) != 1:
        raise PilotContractError("rehearsal source is not one frozen 72-step mode")
    data["steps"] = _REHEARSAL_EXECUTION_STEPS
    data["timeline"][0]["end_step"] = _REHEARSAL_EXECUTION_STEPS
    data["sensor_model"]["random_seed"] = sensor_seed
    try:
        return Scenario.from_mapping(data)
    except ScenarioValidationError as error:
        raise PilotContractError("rehearsal scenario is invalid") from error


def _resource_gauges(snapshot: Mapping[str, Any]) -> dict[str, float]:
    wrapper = snapshot.get("operational_resource_gauges")
    if (
        type(wrapper) is not dict
        or set(wrapper) != {"source_kind", "samples"}
        or wrapper["source_kind"] != "operational_resource_gauge"
        or type(wrapper["samples"]) is not list
    ):
        raise PilotContractError("HMC snapshot resource-gauge wrapper drifts")
    values: dict[str, float] = {}
    for sample in wrapper["samples"]:
        if (
            type(sample) is not dict
            or set(sample)
            != {
                "descriptor_id",
                "unit",
                "availability",
                "value",
                "unavailable_reason",
            }
            or sample["descriptor_id"] not in _REHEARSAL_CHANNELS
            or sample["unit"] != "fraction"
            or sample["availability"] != "AVAILABLE"
            or sample["unavailable_reason"] is not None
            or type(sample["value"]) not in (int, float)
            or not math.isfinite(float(sample["value"]))
        ):
            raise PilotContractError("HMC snapshot resource gauge is inadmissible")
        channel = sample["descriptor_id"]
        if channel in values:
            raise PilotContractError("HMC snapshot duplicates a resource gauge")
        value = float(sample["value"])
        if not 0.0 <= value <= 1.0:
            raise PilotContractError(
                "HMC snapshot resource gauge leaves fraction range"
            )
        values[channel] = value
    if tuple(values) != _REHEARSAL_CHANNELS:
        raise PilotContractError("HMC snapshot resource-gauge order drifts")
    return values


def _execute_no_proposal_rehearsal(
    scenario: Scenario,
    *,
    hmc_contract: Any,
    nonce: bytes,
) -> dict[str, Any]:
    hmc = HabitatManagementComputer.reset(scenario, hmc_contract, nonce)
    rows: dict[int, dict[str, float]] = {}
    for application_step in range(_REHEARSAL_EXECUTION_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple or len(observed) != 2:
            raise PilotContractError("HMC rehearsal terminates before observation")
        snapshot, verification = observed
        mapping = snapshot.to_mapping()
        if (
            mapping["completed_step"] != application_step
            or snapshot.snapshot_sha256 != verification.snapshot_sha256
        ):
            raise PilotContractError("HMC rehearsal snapshot identity drifts")
        if application_step in _REHEARSAL_ROWS:
            rows[application_step] = _resource_gauges(mapping)
        handle = hmc.verify_snapshot(snapshot, verification)
        proposal = hmc.propose(None, handle).to_mapping()
        if proposal["validation_outcome"] != "NO_PROPOSAL":
            raise PilotContractError("rehearsal admits a proposal")
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise PilotContractError("HMC rehearsal arbitration terminates")
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise PilotContractError("HMC rehearsal step terminates")
    if tuple(rows) != _REHEARSAL_ROWS:
        raise PilotContractError("HMC rehearsal does not expose every required row")
    trace = hmc.export_control_trace(_HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes,
        scenario=scenario,
        contract=hmc_contract,
    )
    replay = replay_control_trace(
        trace.canonical_bytes,
        scenario=scenario,
        contract=hmc_contract,
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != _REHEARSAL_EXECUTION_STEPS
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise PilotContractError("HMC rehearsal trace fails strict replay")
    return {
        "rows": rows,
        "trace_bytes": trace.canonical_bytes,
        "trace_sha256": hashlib.sha256(trace.canonical_bytes).hexdigest(),
        "control_run_id": hmc.control_run_id,
        "replay_final_state_sha256": replay.final_state_sha256,
    }


def _channel_separation(
    nominal: Mapping[int, Mapping[str, float]],
    reduced: Mapping[int, Mapping[str, float]],
    channel: str,
) -> dict[str, Any]:
    separations = {
        step: abs(float(nominal[step][channel]) - float(reduced[step][channel]))
        for step in _REHEARSAL_ROWS
    }
    max_step = min(
        step
        for step, value in separations.items()
        if value == max(separations.values())
    )
    maximum = separations[max_step]
    return {
        "channel": channel,
        "max_absolute_fractional_separation": maximum,
        "max_completed_step": max_step,
        "nominal_value_at_max": float(nominal[max_step][channel]),
        "reduced_value_at_max": float(reduced[max_step][channel]),
        "qualifies": maximum >= _REHEARSAL_THRESHOLD,
    }


def run_reduced_resource_sensitivity_rehearsal(
    repo_root: str | Path,
    design: PilotDesign,
) -> dict[str, Any]:
    """Run the permanently excluded HMC sensitivity rehearsal in memory."""
    if type(design) is not PilotDesign:
        raise PilotContractError("rehearsal requires the exact pilot design")
    root = Path(repo_root).resolve()
    profile_packet, _, _ = _load_materialization_inputs(root)
    reduced_profile = next(
        item
        for item in profile_packet["semantic_cluster_profiles"]
        if item["id"] == "reduced_resource_inventory"
    )
    gate = reduced_profile["sensitivity_rehearsal_gate"]
    if gate != {
        "namespace": _REHEARSAL_NAMESPACE,
        "permanently_excluded": True,
        "comparison": (
            "healthy NO_PROPOSAL reduced-resource-inventory versus matched nominal profile"
        ),
        "required_in_each_mode_load_stratum": True,
        "rows": "anchors 16, 40 and 64 plus each eight-step continuation",
        "minimum_public_resource_channels": 2,
        "minimum_max_absolute_fractional_separation": "0.02",
        "failure": "REPLACE_PROFILE_BEFORE_PILOT",
    }:
        raise PilotContractError("reduced-resource rehearsal gate drifts")
    bundle = load_forecast_contracts(root)
    if bundle.hmc_contract.hmc_contract_sha256 != _HMC_CONTRACT_SHA256:
        raise PilotContractError("rehearsal HMC contract identity drifts")

    by_key = {
        (
            cluster.operating_mode,
            cluster.load_regime,
            cluster.semantic_profile_role,
        ): cluster
        for cluster in design.clusters
    }
    if len(by_key) != 60:
        raise PilotContractError("rehearsal cluster profile join is not unique")
    strata: list[dict[str, Any]] = []
    first_runs: (
        tuple[dict[str, Any], dict[str, Any], Scenario, Scenario, bytes] | None
    ) = None
    for mode in design.operating_modes:
        for load in design.load_regimes:
            nominal_cluster = by_key[(mode, load, "balanced-initial-state")]
            reduced_cluster = by_key[(mode, load, "reduced-resource-inventory")]
            seed = _rehearsal_seed(design, mode, load)
            nonce = _rehearsal_nonce(design, mode, load)
            nominal = _to_rehearsal_scenario(
                materialize_pilot_scenario(
                    root,
                    design,
                    cluster_id=nominal_cluster.cluster_id,
                    member_id="HEALTHY",
                    repetition_id="R01",
                ),
                sensor_seed=seed,
            )
            reduced = _to_rehearsal_scenario(
                materialize_pilot_scenario(
                    root,
                    design,
                    cluster_id=reduced_cluster.cluster_id,
                    member_id="HEALTHY",
                    repetition_id="R01",
                ),
                sensor_seed=seed,
            )
            nominal_run = _execute_no_proposal_rehearsal(
                nominal,
                hmc_contract=bundle.hmc_contract,
                nonce=nonce,
            )
            reduced_run = _execute_no_proposal_rehearsal(
                reduced,
                hmc_contract=bundle.hmc_contract,
                nonce=nonce,
            )
            if first_runs is None:
                first_runs = (nominal_run, reduced_run, nominal, reduced, nonce)
            channels = [
                _channel_separation(nominal_run["rows"], reduced_run["rows"], channel)
                for channel in _REHEARSAL_CHANNELS
            ]
            qualifying = sum(item["qualifies"] for item in channels)
            strata.append(
                {
                    "operating_mode": mode,
                    "load_regime": load,
                    "nominal_cluster_id": nominal_cluster.cluster_id,
                    "reduced_cluster_id": reduced_cluster.cluster_id,
                    "matched_sensor_seed": seed,
                    "matched_reset_nonce_sha256": hashlib.sha256(nonce).hexdigest(),
                    "nominal_scenario_sha256": nominal.scenario_sha256,
                    "reduced_scenario_sha256": reduced.scenario_sha256,
                    "nominal_control_run_id": nominal_run["control_run_id"],
                    "reduced_control_run_id": reduced_run["control_run_id"],
                    "nominal_trace_sha256": nominal_run["trace_sha256"],
                    "reduced_trace_sha256": reduced_run["trace_sha256"],
                    "nominal_replay_final_state_sha256": nominal_run[
                        "replay_final_state_sha256"
                    ],
                    "reduced_replay_final_state_sha256": reduced_run[
                        "replay_final_state_sha256"
                    ],
                    "channels": channels,
                    "qualifying_channel_count": qualifying,
                    "verdict": "PASS"
                    if qualifying >= 2
                    else "REPLACE_PROFILE_BEFORE_PILOT",
                }
            )
    if first_runs is None:
        raise PilotContractError("rehearsal has no strata")
    (
        first_nominal,
        first_reduced,
        first_nominal_scenario,
        first_reduced_scenario,
        nonce,
    ) = first_runs
    repeated_nominal = _execute_no_proposal_rehearsal(
        first_nominal_scenario,
        hmc_contract=bundle.hmc_contract,
        nonce=nonce,
    )
    repeated_reduced = _execute_no_proposal_rehearsal(
        first_reduced_scenario,
        hmc_contract=bundle.hmc_contract,
        nonce=nonce,
    )
    deterministic = (
        first_nominal["trace_bytes"] == repeated_nominal["trace_bytes"]
        and first_reduced["trace_bytes"] == repeated_reduced["trace_bytes"]
    )
    body: dict[str, Any] = {
        "schema_version": _REHEARSAL_SCHEMA,
        "namespace": _REHEARSAL_NAMESPACE,
        "permanently_excluded": True,
        "roster_sha256": design.roster_sha256,
        "profile_action_sha256": design.profile_action_sha256,
        "source_scenario_sha256": _SOURCE_SCENARIO_SHA256,
        "hmc_contract_sha256": _HMC_CONTRACT_SHA256,
        "hmc_implementation_git_sha": _HMC_IMPLEMENTATION_GIT_SHA,
        "comparison": (
            "healthy NO_PROPOSAL reduced-resource-inventory versus matched "
            "balanced-initial-state"
        ),
        "matched_fields": [
            "operating_mode",
            "load_regime",
            "sensor_seed",
            "reset_nonce",
        ],
        "execution_steps": _REHEARSAL_EXECUTION_STEPS,
        "terminal_closure": {
            "pilot_scenario_steps": 72,
            "evaluation_completed_step_max": 72,
            "post_evaluation_step_count": 1,
            "rationale": (
                "One permanently excluded post-evaluation transition closes the "
                "HMC trace for strict replay; it contributes no sensitivity row."
            ),
        },
        "evaluated_completed_steps": list(_REHEARSAL_ROWS),
        "minimum_max_absolute_fractional_separation": _REHEARSAL_THRESHOLD,
        "minimum_public_resource_channels": 2,
        "public_resource_channels": list(_REHEARSAL_CHANNELS),
        "hmc_runs": 26,
        "strict_replays": 26,
        "strata": strata,
        "determinism_probe": {
            "operating_mode": design.operating_modes[0],
            "load_regime": design.load_regimes[0],
            "nominal_first_trace_sha256": first_nominal["trace_sha256"],
            "nominal_repeat_trace_sha256": repeated_nominal["trace_sha256"],
            "reduced_first_trace_sha256": first_reduced["trace_sha256"],
            "reduced_repeat_trace_sha256": repeated_reduced["trace_sha256"],
            "verdict": "PASS" if deterministic else "FAIL_NONDETERMINISTIC",
        },
        "verdict": (
            "PASS"
            if deterministic and all(item["verdict"] == "PASS" for item in strata)
            else "REPLACE_PROFILE_BEFORE_PILOT"
        ),
        "pilot_generation_authorized": False,
        "model_training_authorized": False,
    }
    body["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    validate_sensitivity_rehearsal_receipt(body, design)
    return body


def validate_sensitivity_rehearsal_receipt(
    receipt: Mapping[str, Any], design: PilotDesign
) -> None:
    """Validate a rehearsal receipt without treating it as generation authority."""
    fields = {
        "schema_version",
        "namespace",
        "permanently_excluded",
        "roster_sha256",
        "profile_action_sha256",
        "source_scenario_sha256",
        "hmc_contract_sha256",
        "hmc_implementation_git_sha",
        "comparison",
        "matched_fields",
        "execution_steps",
        "terminal_closure",
        "evaluated_completed_steps",
        "minimum_max_absolute_fractional_separation",
        "minimum_public_resource_channels",
        "public_resource_channels",
        "hmc_runs",
        "strict_replays",
        "strata",
        "determinism_probe",
        "verdict",
        "pilot_generation_authorized",
        "model_training_authorized",
        "receipt_sha256",
    }
    _require_exact_fields(
        receipt, frozenset(fields), label="sensitivity rehearsal receipt"
    )
    if (
        receipt["schema_version"] != _REHEARSAL_SCHEMA
        or receipt["namespace"] != _REHEARSAL_NAMESPACE
        or receipt["permanently_excluded"] is not True
        or receipt["roster_sha256"] != design.roster_sha256
        or receipt["profile_action_sha256"] != design.profile_action_sha256
        or receipt["source_scenario_sha256"] != _SOURCE_SCENARIO_SHA256
        or receipt["hmc_contract_sha256"] != _HMC_CONTRACT_SHA256
        or receipt["hmc_implementation_git_sha"] != _HMC_IMPLEMENTATION_GIT_SHA
        or receipt["comparison"]
        != (
            "healthy NO_PROPOSAL reduced-resource-inventory versus matched "
            "balanced-initial-state"
        )
        or receipt["matched_fields"]
        != ["operating_mode", "load_regime", "sensor_seed", "reset_nonce"]
        or receipt["execution_steps"] != _REHEARSAL_EXECUTION_STEPS
        or receipt["terminal_closure"]
        != {
            "pilot_scenario_steps": 72,
            "evaluation_completed_step_max": 72,
            "post_evaluation_step_count": 1,
            "rationale": (
                "One permanently excluded post-evaluation transition closes the "
                "HMC trace for strict replay; it contributes no sensitivity row."
            ),
        }
        or receipt["evaluated_completed_steps"] != list(_REHEARSAL_ROWS)
        or receipt["minimum_max_absolute_fractional_separation"] != _REHEARSAL_THRESHOLD
        or receipt["minimum_public_resource_channels"] != 2
        or receipt["public_resource_channels"] != list(_REHEARSAL_CHANNELS)
        or receipt["hmc_runs"] != 26
        or receipt["strict_replays"] != 26
        or receipt["pilot_generation_authorized"] is not False
        or receipt["model_training_authorized"] is not False
    ):
        raise PilotContractError("sensitivity rehearsal receipt header drifts")
    body = dict(receipt)
    declared = body.pop("receipt_sha256")
    if (
        not _is_sha256(declared)
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != declared
    ):
        raise PilotContractError("sensitivity rehearsal receipt self-hash drifts")
    strata = receipt["strata"]
    if type(strata) is not list or len(strata) != 12:
        raise PilotContractError("sensitivity rehearsal stratum count drifts")
    expected_strata = {
        (mode, load) for mode in design.operating_modes for load in design.load_regimes
    }
    design_by_key = {
        (
            cluster.operating_mode,
            cluster.load_regime,
            cluster.semantic_profile_role,
        ): cluster
        for cluster in design.clusters
    }
    if len(design_by_key) != 60:
        raise PilotContractError("sensitivity rehearsal design join is not unique")
    observed_strata: set[tuple[str, str]] = set()
    for row in strata:
        if type(row) is not dict:
            raise PilotContractError("sensitivity rehearsal stratum is not an object")
        _require_exact_fields(
            row,
            frozenset(
                {
                    "operating_mode",
                    "load_regime",
                    "nominal_cluster_id",
                    "reduced_cluster_id",
                    "matched_sensor_seed",
                    "matched_reset_nonce_sha256",
                    "nominal_scenario_sha256",
                    "reduced_scenario_sha256",
                    "nominal_control_run_id",
                    "reduced_control_run_id",
                    "nominal_trace_sha256",
                    "reduced_trace_sha256",
                    "nominal_replay_final_state_sha256",
                    "reduced_replay_final_state_sha256",
                    "channels",
                    "qualifying_channel_count",
                    "verdict",
                }
            ),
            label="sensitivity rehearsal stratum",
        )
        key = (row["operating_mode"], row["load_regime"])
        if key not in expected_strata or key in observed_strata:
            raise PilotContractError("sensitivity rehearsal stratum identity drifts")
        observed_strata.add(key)
        mode, load = key
        expected_nominal = design_by_key[(mode, load, "balanced-initial-state")]
        expected_reduced = design_by_key[(mode, load, "reduced-resource-inventory")]
        expected_nonce_sha256 = hashlib.sha256(
            _rehearsal_nonce(design, mode, load)
        ).hexdigest()
        if (
            row["nominal_cluster_id"] != expected_nominal.cluster_id
            or row["reduced_cluster_id"] != expected_reduced.cluster_id
            or row["matched_sensor_seed"] != _rehearsal_seed(design, mode, load)
            or row["matched_reset_nonce_sha256"] != expected_nonce_sha256
        ):
            raise PilotContractError("sensitivity rehearsal matched identity drifts")
        for sha_field in (
            "matched_reset_nonce_sha256",
            "nominal_scenario_sha256",
            "reduced_scenario_sha256",
            "nominal_control_run_id",
            "reduced_control_run_id",
            "nominal_trace_sha256",
            "reduced_trace_sha256",
            "nominal_replay_final_state_sha256",
            "reduced_replay_final_state_sha256",
        ):
            if not _is_sha256(row[sha_field]):
                raise PilotContractError("sensitivity rehearsal identity is malformed")
        channels = row["channels"]
        if type(channels) is not list or [
            item.get("channel") for item in channels
        ] != list(_REHEARSAL_CHANNELS):
            raise PilotContractError("sensitivity rehearsal channel set drifts")
        qualifying = 0
        for channel in channels:
            _require_exact_fields(
                channel,
                frozenset(
                    {
                        "channel",
                        "max_absolute_fractional_separation",
                        "max_completed_step",
                        "nominal_value_at_max",
                        "reduced_value_at_max",
                        "qualifies",
                    }
                ),
                label="sensitivity rehearsal channel",
            )
            values = (
                channel["max_absolute_fractional_separation"],
                channel["nominal_value_at_max"],
                channel["reduced_value_at_max"],
            )
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values
            ):
                raise PilotContractError("sensitivity rehearsal channel is nonfinite")
            if channel["max_completed_step"] not in _REHEARSAL_ROWS:
                raise PilotContractError("sensitivity rehearsal maximum row drifts")
            separation = abs(
                float(channel["nominal_value_at_max"])
                - float(channel["reduced_value_at_max"])
            )
            if not math.isclose(
                separation,
                float(channel["max_absolute_fractional_separation"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise PilotContractError(
                    "sensitivity rehearsal separation is incoherent"
                )
            expected_qualifies = separation >= _REHEARSAL_THRESHOLD
            if channel["qualifies"] is not expected_qualifies:
                raise PilotContractError(
                    "sensitivity rehearsal threshold result drifts"
                )
            qualifying += expected_qualifies
        if row["qualifying_channel_count"] != qualifying or row["verdict"] != (
            "PASS" if qualifying >= 2 else "REPLACE_PROFILE_BEFORE_PILOT"
        ):
            raise PilotContractError("sensitivity rehearsal stratum verdict drifts")
    if observed_strata != expected_strata:
        raise PilotContractError("sensitivity rehearsal misses a stratum")
    probe = receipt["determinism_probe"]
    if type(probe) is not dict:
        raise PilotContractError("sensitivity rehearsal determinism probe is invalid")
    _require_exact_fields(
        probe,
        frozenset(
            {
                "operating_mode",
                "load_regime",
                "nominal_first_trace_sha256",
                "nominal_repeat_trace_sha256",
                "reduced_first_trace_sha256",
                "reduced_repeat_trace_sha256",
                "verdict",
            }
        ),
        label="sensitivity rehearsal determinism probe",
    )
    deterministic = (
        probe["nominal_first_trace_sha256"] == probe["nominal_repeat_trace_sha256"]
        and probe["reduced_first_trace_sha256"] == probe["reduced_repeat_trace_sha256"]
    )
    if (
        probe["operating_mode"] != design.operating_modes[0]
        or probe["load_regime"] != design.load_regimes[0]
        or any(
            not _is_sha256(probe[field])
            for field in (
                "nominal_first_trace_sha256",
                "nominal_repeat_trace_sha256",
                "reduced_first_trace_sha256",
                "reduced_repeat_trace_sha256",
            )
        )
    ):
        raise PilotContractError("sensitivity rehearsal determinism identity drifts")
    if probe["verdict"] != ("PASS" if deterministic else "FAIL_NONDETERMINISTIC"):
        raise PilotContractError("sensitivity rehearsal determinism verdict drifts")
    expected_verdict = (
        "PASS"
        if deterministic and all(row["verdict"] == "PASS" for row in strata)
        else "REPLACE_PROFILE_BEFORE_PILOT"
    )
    if receipt["verdict"] != expected_verdict:
        raise PilotContractError("sensitivity rehearsal overall verdict drifts")


def load_sensitivity_rehearsal_receipt(
    path: str | Path,
    design: PilotDesign,
    *,
    expected_receipt_sha256: str,
    expected_receipt_bytes_sha256: str,
) -> dict[str, Any]:
    """Load only externally pinned canonical rehearsal evidence."""
    if not _is_sha256(expected_receipt_sha256) or not _is_sha256(
        expected_receipt_bytes_sha256
    ):
        raise PilotContractError("expected sensitivity receipt identity is malformed")
    candidate = Path(path).resolve()
    try:
        raw = candidate.read_bytes()
    except OSError as error:
        raise PilotContractError(
            "sensitivity rehearsal receipt cannot be read"
        ) from error
    if hashlib.sha256(raw).hexdigest() != expected_receipt_bytes_sha256:
        raise PilotContractError("sensitivity rehearsal receipt byte identity drifts")
    value = _strict_json(raw, label="sensitivity rehearsal receipt")
    if raw != canonical_json_bytes(value):
        raise PilotContractError(
            "sensitivity rehearsal receipt bytes are not canonical"
        )
    if value.get("receipt_sha256") != expected_receipt_sha256:
        raise PilotContractError("sensitivity rehearsal receipt identity drifts")
    validate_sensitivity_rehearsal_receipt(value, design)
    require_sensitivity_rehearsal_pass(value, design)
    return value


def require_sensitivity_rehearsal_pass(
    receipt: Mapping[str, Any], design: PilotDesign
) -> Mapping[str, Any]:
    validate_sensitivity_rehearsal_receipt(receipt, design)
    if receipt["verdict"] != "PASS":
        raise PilotContractError("REPLACE_PROFILE_BEFORE_PILOT")
    return receipt
