"""Fail-closed authorized AEOLUS V2 FIT+CAL generation and training launcher.

This launcher is deliberately incapable of enumerating, materializing, or
scoring validation/final data.  HMC remains the sole operational authority;
all learned outputs are advisory qualification artifacts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.forecast.pilot import (
    PilotResourcePreflight,
    load_approved_pilot_design,
)
from aeolus.habitat_v2.forecast.pilot_baselines import compact_target_history
from aeolus.habitat_v2.forecast.pilot_benchmark import V2_RESOURCE_CEILINGS, build_v2_source_manifest
from aeolus.habitat_v2.forecast.qualified_runtime_guard import (
    QualifiedRuntimeGuard, QualifiedRuntimeGuardError, QualifiedRuntimeLimits,
)
from aeolus.habitat_v2.forecast.pilot_campaign import run_pilot_campaign
from aeolus.habitat_v2.forecast.projection import forecast_layout
from aeolus.habitat_v2.forecast.qualification_split import (
    build_qualification_split,
    load_qualified_protocol,
)

PROTOCOL_PATH = Path("docs/plans/2026-08-16-habitat-v2-qualified-model-protocol-v1.json")
AMENDMENT_PATH = Path("docs/plans/2026-08-16-habitat-v2-qualified-model-protocol-v2-amendment-v1.json")
PREFLIGHT_RELATIVE = Path("out/helios-qual-v2/v2-resource-preflight-real.json")
PREFLIGHT_SHA256 = "590bdca9a09bcfe5b527ea295d50f4fefc45dd65250e92eca57af5cc36b85420"
PREFLIGHT_BYTES_SHA256 = "01c0e18141deca4a4e94124a08f60bf45b3a51b3ddacfb87a746ef1fc9e17147"
SOURCE_PREFLIGHT_RELATIVE = Path("out/helios-qual-v2/v2-protocol-preflight-qualified-w1-final3.json")

FIT_CLUSTERS = 36
CAL_CLUSTERS = 12
AUTHORIZED_CLUSTERS = 48
PACKETS_PER_CLUSTER = 78
EXAMPLES_PER_PACKET = 5
FIT_PACKETS = 2808
CAL_PACKETS = 936
AUTHORIZED_PACKETS = 3744
FIT_EXAMPLES = 14040
CAL_EXAMPLES = 4680
AUTHORIZED_EXAMPLES = 18720


class QualifiedLaunchError(RuntimeError):
    """The frozen protocol cannot be executed safely."""


@dataclass(frozen=True)
class CorpusArrays:
    history: np.ndarray
    available: np.ndarray
    action: np.ndarray
    action_present: np.ndarray
    targets: np.ndarray
    clusters: tuple[str, ...]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualifiedLaunchError(f"cannot read required JSON: {path}") from error
    if type(value) is not dict:
        raise QualifiedLaunchError(f"required JSON must be an object: {path}")
    return value


def _self_hashed_json(path: Path, field: str) -> dict[str, Any]:
    value = _read_json(path)
    declared = value.pop(field, None)
    if type(declared) is not str or declared != _sha_bytes(canonical_json_bytes(value)):
        raise QualifiedLaunchError(f"self-hash mismatch: {path}")
    value[field] = declared
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one immutable receipt; never replace an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(value))
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise QualifiedLaunchError(f"refusing to overwrite immutable receipt: {path}") from error


def _verify_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _self_hashed_json(root / PROTOCOL_PATH, "protocol_sha256")
    amendment = _self_hashed_json(root / AMENDMENT_PATH, "self_sha256")
    if protocol.get("schema_version") != "aeolus_habitat_v2_qualified_model_protocol_v1":
        raise QualifiedLaunchError("qualified protocol schema drifts")
    if amendment.get("schema_version") != "aeolus_habitat_v2_qualified_model_protocol_amendment_v1":
        raise QualifiedLaunchError("qualified protocol amendment schema drifts")
    if amendment.get("supersedes_semantics", {}).get("protocol_sha256") != protocol["protocol_sha256"]:
        raise QualifiedLaunchError("amendment is not bound to the frozen protocol")
    optimization = protocol.get("optimization")
    primary = protocol.get("model_candidates", {}).get("primary")
    blind = protocol.get("model_candidates", {}).get("action_blind")
    expected_optimization = {
        "batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.0001,
        "max_epochs": 80, "early_stop_patience": 12, "seed": 20260816,
    }
    if not isinstance(optimization, dict) or any(optimization.get(k) != v for k, v in expected_optimization.items()):
        raise QualifiedLaunchError("frozen optimizer settings drift")
    for candidate, input_dim in ((primary, 5804), (blind, 5776)):
        if not isinstance(candidate, dict) or candidate.get("input_dim") != input_dim or candidate.get("hidden_layers") != [512, 512, 256] or candidate.get("activation") != "GELU":
            raise QualifiedLaunchError("frozen MLP architecture drifts")
    if primary.get("dropout") != 0.0 or protocol.get("baselines") != ["action-aware (primary)", "action-blind", "persistence"]:
        raise QualifiedLaunchError("frozen candidate/baseline policy drifts")
    if protocol.get("locked_out") is None or "validation_access" not in protocol["locked_out"]:
        raise QualifiedLaunchError("validation lockout is not explicit")
    return protocol, amendment


def _assert_source_provenance(root: Path, protocol: Mapping[str, Any], amendment: Mapping[str, Any]) -> str:
    """Recheck the immutable source-manifest preflight before any HMC dispatch."""
    preflight_path = root / SOURCE_PREFLIGHT_RELATIVE
    receipt = _self_hashed_json(preflight_path, "preflight_sha256")
    if receipt.get("schema_version") != "aeolus_habitat_v2_qualified_protocol_preflight_v1" or receipt.get("verdict") != "PASS":
        raise QualifiedLaunchError("source preflight is not a passing qualified receipt")
    if receipt.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise QualifiedLaunchError("source preflight protocol binding drifts")
    if amendment.get("supersedes_semantics", {}).get("protocol_sha256") != protocol["protocol_sha256"]:
        raise QualifiedLaunchError("protocol amendment binding drifts")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualifiedLaunchError("cannot resolve source commit") from error
    sealed_commit = receipt.get("source_commit")
    if type(sealed_commit) is not str:
        raise QualifiedLaunchError("source preflight has no sealed source commit")
    if commit != sealed_commit:
        raise QualifiedLaunchError("current HEAD is not the exact sealed source commit")
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualifiedLaunchError("cannot verify clean sealed source worktree") from error
    if dirty:
        raise QualifiedLaunchError("sealed launcher requires an exactly clean committed worktree")
    if receipt.get("source_manifest") != build_v2_source_manifest(root):
        raise QualifiedLaunchError("current source manifest differs from sealed preflight")
    return commit


def _load_pinned_resource_preflight(root: Path) -> PilotResourcePreflight:
    """Load the byte-pinned benchmark receipt without accepting a substituted one.

    The resource benchmark predates the qualified source-manifest amendment, so
    its historical source manifest is intentionally not treated as permission
    for today's sources.  `_assert_source_provenance` above provides that
    current-source permission; this receipt remains a byte-identical resource
    measurement only.
    """
    path = root / PREFLIGHT_RELATIVE
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise QualifiedLaunchError("pinned resource preflight cannot be read") from error
    if _sha_bytes(raw) != PREFLIGHT_BYTES_SHA256:
        raise QualifiedLaunchError("pinned resource preflight byte hash drifts")
    value = _self_hashed_json(path, "preflight_sha256")
    required = {
        "schema_version", "preflight_sha256", "source_manifest_sha256", "planned_hmc_runs",
        "benchmark_hmc_runs", "measured_wall_time_seconds", "measured_peak_rss_bytes",
        "measured_artifact_bytes", "projected_wall_time_seconds", "projected_peak_rss_bytes",
        "projected_artifact_bytes", "verdict", "runtime_within_ceiling", "memory_within_ceiling",
        "disk_reserve_preserved",
    }
    if value.get("preflight_sha256") != PREFLIGHT_SHA256 or not required <= set(value):
        raise QualifiedLaunchError("pinned resource preflight semantic identity drifts")
    if value["schema_version"] != "aeolus_habitat_v2_forecast_pilot_resource_preflight_v2" or value["planned_hmc_runs"] != 23_400 or value["benchmark_hmc_runs"] != 2 or value["verdict"] != "PASS":
        raise QualifiedLaunchError("pinned resource preflight cannot authorize V2 generation")
    if any(value[key] is not True for key in ("runtime_within_ceiling", "memory_within_ceiling", "disk_reserve_preserved")):
        raise QualifiedLaunchError("pinned resource preflight resource verdict drifts")
    numeric = ("measured_wall_time_seconds", "measured_peak_rss_bytes", "measured_artifact_bytes", "projected_wall_time_seconds", "projected_peak_rss_bytes", "projected_artifact_bytes")
    if any(not isinstance(value[key], (int, float)) or isinstance(value[key], bool) or value[key] <= 0 for key in numeric):
        raise QualifiedLaunchError("pinned resource preflight measurements are invalid")
    return PilotResourcePreflight(
        preflight_sha256=value["preflight_sha256"], preflight_bytes_sha256=PREFLIGHT_BYTES_SHA256,
        planned_hmc_runs=value["planned_hmc_runs"], benchmark_hmc_runs=value["benchmark_hmc_runs"],
        measured_wall_time_seconds=float(value["measured_wall_time_seconds"]), measured_peak_rss_bytes=value["measured_peak_rss_bytes"],
        measured_artifact_bytes=value["measured_artifact_bytes"], projected_wall_time_seconds=float(value["projected_wall_time_seconds"]),
        projected_peak_rss_bytes=value["projected_peak_rss_bytes"], projected_artifact_bytes=value["projected_artifact_bytes"],
        verdict=value["verdict"], schema_version=value["schema_version"], v2_binding_sha256=value["source_manifest_sha256"],
    )


def _start_or_resume_run(run_root: Path, *, protocol: Mapping[str, Any], source_commit: str) -> None:
    marker = run_root / "launcher-start.json"
    complete = run_root / "launcher-complete.json"
    expected = {
        "schema_version": "aeolus_habitat_v2_fitcal_launcher_start_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": source_commit,
        "validation_accessed": False,
    }
    if complete.exists():
        raise QualifiedLaunchError("run is complete; refusing to overwrite complete outputs")
    if marker.exists():
        if _read_json(marker) != expected:
            raise QualifiedLaunchError("existing run marker does not match frozen launch identity")
        return
    if run_root.exists() and any(run_root.iterdir()):
        raise QualifiedLaunchError("nonempty run root has no immutable launch marker")
    run_root.mkdir(parents=True, exist_ok=True)
    _write_new_json(marker, expected)


def _verify_campaign_manifest(corpus: Path, allowed_cluster_ids: frozenset[str]) -> None:
    manifest_path = corpus / "campaign-manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    declared = manifest.pop("campaign_manifest_sha256", None)
    if type(declared) is not str or declared != _sha_bytes(canonical_json_bytes(manifest)):
        raise QualifiedLaunchError("campaign manifest self-hash mismatch")
    expected = {
        "pairs_completed": AUTHORIZED_PACKETS,
        "hmc_runs_executed": AUTHORIZED_EXAMPLES,
        "allowed_cluster_ids": sorted(allowed_cluster_ids),
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise QualifiedLaunchError("campaign manifest does not prove exact authorized corpus")


def _generate_or_resume_corpus(
    root: Path,
    corpus: Path,
    *,
    design: Any,
    contracts: Any,
    preflight: PilotResourcePreflight,
    allowed_cluster_ids: frozenset[str],
    runner: Callable[..., Mapping[str, Any]] = run_pilot_campaign,
    health_check: Callable[[str], None] | None = None,
) -> None:
    """Resume validated pair directories only; bad partial artifacts are retained and fail."""
    if (corpus / "campaign-manifest.json").exists():
        _verify_campaign_manifest(corpus, allowed_cluster_ids)
        return
    # The campaign's resume validator verifies every existing pair before HMC.
    # It never deletes a malformed directory; validation failure is terminal.
    if health_check: health_check("before-corpus-generation")
    runner(root, design, contracts, preflight=preflight, output_root=corpus,
           pair_limit=None, worker_count=1, resume=corpus.exists())
    if health_check: health_check("after-corpus-generation")
    _verify_campaign_manifest(corpus, allowed_cluster_ids)


def _verify_exact_custody(root: Path, corpus: Path, split: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Read only authorized FIT/CAL packet paths and prove exact custody totals."""
    from scripts.qual_v2_prepare import (
        verify_campaign,  # imported only after generation is closed
    )

    if not (corpus / "campaign-manifest.json").exists():
        raise QualifiedLaunchError("custody cannot begin before a final campaign manifest")
    receipt = verify_campaign(root, corpus, forbid_validation=True)
    counts = {
        "fit": (FIT_CLUSTERS, FIT_PACKETS, FIT_EXAMPLES),
        "cal": (CAL_CLUSTERS, CAL_PACKETS, CAL_EXAMPLES),
        "validation": (0, 0, 0),
    }
    if receipt.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise QualifiedLaunchError("custody protocol binding drifts")
    for name, (clusters, packets, examples) in counts.items():
        item = receipt.get("splits", {}).get(name, {})
        if len(item.get("clusters", [])) != clusters or item.get("packets") != packets or item.get("examples") != examples:
            raise QualifiedLaunchError(f"custody count drift for {name}")
    if receipt.get("packet_count") != AUTHORIZED_PACKETS or receipt.get("example_count") != AUTHORIZED_EXAMPLES:
        raise QualifiedLaunchError("custody total drift")
    expected_clusters = set(split.authorized_cluster_ids)
    actual_clusters = set(receipt["splits"]["fit"]["clusters"]) | set(receipt["splits"]["cal"]["clusters"])
    if actual_clusters != expected_clusters:
        raise QualifiedLaunchError("custody cluster allowlist drift")
    return receipt


def _load_authorized_arrays(corpus: Path, allowed_cluster_ids: frozenset[str], custody: Mapping[str, Any]) -> CorpusArrays:
    packets = [Path(path) for split in ("fit", "cal") for path in custody["splits"][split]["packet_paths"]]
    expected_hashes = [digest for split in ("fit", "cal") for digest in custody["splits"][split]["packet_sha256s"]]
    if len(packets) != AUTHORIZED_PACKETS or len(expected_hashes) != len(packets):
        raise QualifiedLaunchError("training requires exact authorized packet count")
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = []
    for path, expected_hash in zip(packets, expected_hashes, strict=True):
        if path.parent.parent != corpus or _sha_bytes(path.read_bytes()) != expected_hash:
            raise QualifiedLaunchError("custody ledger packet path/hash drift")
        with np.load(path, allow_pickle=False) as packet:
            cluster_ids = packet["cluster_ids"].tolist()
            if set(cluster_ids) - allowed_cluster_ids:
                raise QualifiedLaunchError("validation or unknown cluster packet refused")
            rows.append((packet["history_numeric_f32"].copy(), packet["operational_available_bool"].copy(), packet["proposed_action_f32"].copy(), packet["action_present"].copy(), packet["targets_f32"].copy(), cluster_ids))
    return CorpusArrays(
        history=np.concatenate([r[0] for r in rows]), available=np.concatenate([r[1] for r in rows]),
        action=np.concatenate([r[2] for r in rows]), action_present=np.concatenate([r[3] for r in rows]),
        targets=np.concatenate([r[4] for r in rows]), clusters=tuple(cluster for r in rows for cluster in r[5]),
    )


def _features(arrays: CorpusArrays, layout: Any, *, action_aware: bool) -> np.ndarray:
    histories = arrays.history.copy()
    histories[:, :, :167] *= arrays.available.astype(np.float32)
    base = np.concatenate((histories.reshape(len(histories), -1), arrays.available.astype(np.float32).reshape(len(histories), -1)), axis=1)
    if action_aware:
        base = np.concatenate((base, arrays.action, arrays.action_present.astype(np.float32).reshape(-1, 1)), axis=1)
    expected = 5804 if action_aware else 5776
    if base.shape != (len(histories), expected) or not np.isfinite(base).all():
        raise QualifiedLaunchError("frozen MLP feature construction drift")
    return base.astype(np.float32, copy=False)


def _persistence_predictions(arrays: CorpusArrays, layout: Any) -> np.ndarray:
    values = np.stack([compact_target_history(history, layout, operational_available_bool=available)[-1] for history, available in zip(arrays.history, arrays.available, strict=True)])
    return np.repeat(values[:, None, :], 8, axis=1).astype(np.float32)


def _metrics(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (8, 51):
        raise QualifiedLaunchError("metric tensors have wrong shape")
    if scale.shape != (51,) or not np.isfinite(scale).all() or np.any(scale < 1e-6):
        raise QualifiedLaunchError("FIT target scale is invalid")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    per_target_mae = np.abs(error).mean(axis=(0, 1))
    return {
        "aggregate_normalized_mae": float(np.mean(per_target_mae / scale)),
        "aggregate_rmse": float(np.sqrt(np.mean(error * error))),
        "per_target_mae": per_target_mae.tolist(),
    }


def _cal_gate(action_aware: Mapping[str, Any], persistence: Mapping[str, Any]) -> bool:
    return float(action_aware["aggregate_normalized_mae"]) < float(persistence["aggregate_normalized_mae"])


def _seed_deterministic(seed: int) -> Any:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True, warn_only=False)
    except RuntimeError as error: raise QualifiedLaunchError("deterministic torch mode unavailable") from error
    return torch


def _train_candidate(
    *, name: str, fit_x: np.ndarray, fit_y: np.ndarray, cal_x: np.ndarray, cal_y: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray, seed: int,
) -> tuple[dict[str, Any], bytes]:
    torch = _seed_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(torch.nn.Linear(fit_x.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 408)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x = torch.from_numpy(fit_x); y = torch.from_numpy(((fit_y - target_mean) / target_scale).reshape(len(fit_y), -1).astype(np.float32))
    cal_xt = torch.from_numpy(cal_x).to(device)
    best_metric = float("inf"); best_epoch = -1; best_state: dict[str, Any] | None = None; stale = 0
    for epoch in range(80):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        for index in torch.randperm(len(x), generator=generator).split(128):
            prediction = model(x[index].to(device)); loss = torch.nn.functional.mse_loss(prediction, y[index].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            normalized = model(cal_xt).cpu().numpy().reshape(-1, 8, 51)
        metric = _metrics(normalized * target_scale + target_mean, cal_y, target_scale)["aggregate_normalized_mae"]
        if metric < best_metric:
            best_metric = metric; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12: break
    if best_state is None: raise QualifiedLaunchError("candidate produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): prediction = model(cal_xt).cpu().numpy().reshape(-1, 8, 51) * target_scale + target_mean
    metrics = _metrics(prediction, cal_y, target_scale)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save({"model": name, "epoch": best_epoch, "state_dict": best_state, "target_mean": target_mean, "target_scale": target_scale}, handle.name)
        checkpoint = Path(handle.name).read_bytes()
    return {"name": name, "selected_epoch": best_epoch, "early_stop_metric": best_metric, "cal_metrics": metrics}, checkpoint


def _determinism_probe(**kwargs: Any) -> dict[str, Any]:
    one, one_bytes = _train_candidate(**kwargs)
    two, two_bytes = _train_candidate(**kwargs)
    return {"measured": True, "receipt_equal": one == two, "checkpoint_sha256_equal": _sha_bytes(one_bytes) == _sha_bytes(two_bytes), "first": one, "second": two}


def _train_and_record(run_root: Path, corpus: Path, split: Any, contracts: Any, protocol: Mapping[str, Any], custody: Mapping[str, Any]) -> dict[str, Any]:
    arrays = _load_authorized_arrays(corpus, split.authorized_cluster_ids, custody)
    labels = np.asarray([cluster in split.fit_cluster_ids for cluster in arrays.clusters])
    if int(labels.sum()) != FIT_EXAMPLES or int((~labels).sum()) != CAL_EXAMPLES:
        raise QualifiedLaunchError("FIT/CAL example assignment drift")
    layout = forecast_layout(contracts)
    fit, cal = CorpusArrays(*(getattr(arrays, field)[labels] if field != "clusters" else tuple(c for c, keep in zip(arrays.clusters, labels, strict=True) if keep) for field in ("history", "available", "action", "action_present", "targets", "clusters"))), CorpusArrays(*(getattr(arrays, field)[~labels] if field != "clusters" else tuple(c for c, keep in zip(arrays.clusters, ~labels, strict=True) if keep) for field in ("history", "available", "action", "action_present", "targets", "clusters")))
    mean = fit.targets.mean(axis=(0, 1), dtype=np.float64).astype(np.float32); scale = np.maximum(fit.targets.std(axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)
    aware_args = {
        "name": "action-aware-mlp-v1", "fit_x": _features(fit, layout, action_aware=True),
        "fit_y": fit.targets, "cal_x": _features(cal, layout, action_aware=True),
        "cal_y": cal.targets, "target_mean": mean, "target_scale": scale, "seed": 20260816,
    }
    blind_args = {**aware_args, "name": "action-blind-mlp-v1", "fit_x": _features(fit, layout, action_aware=False), "cal_x": _features(cal, layout, action_aware=False)}
    aware, aware_checkpoint = _train_candidate(**aware_args)
    blind, blind_checkpoint = _train_candidate(**blind_args)
    persistence = _metrics(_persistence_predictions(cal, layout), cal.targets, scale)
    probe = _determinism_probe(**aware_args)
    if not probe["receipt_equal"] or not probe["checkpoint_sha256_equal"]: raise QualifiedLaunchError("measured determinism probe failed")
    checkpoint_hashes: dict[str, str] = {}
    for name, raw in (("action-aware-selected.pt", aware_checkpoint), ("action-blind-selected.pt", blind_checkpoint)):
        path = run_root / name
        try:
            with path.open("xb") as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        except FileExistsError as error: raise QualifiedLaunchError(f"refusing to overwrite checkpoint: {path}") from error
        checkpoint_hashes[name] = _sha_bytes(raw)
    result = {"schema_version": "aeolus_habitat_v2_fitcal_training_receipt_v2", "protocol_sha256": protocol["protocol_sha256"], "validation_accessed": False, "fit_target_mean": mean.tolist(), "fit_target_scale": scale.tolist(), "checkpoint_sha256": checkpoint_hashes, "models": {"action_aware": aware, "action_blind": blind, "persistence": persistence}, "determinism_probe": probe, "cal_gate_action_aware_strictly_below_persistence": _cal_gate(aware["cal_metrics"], persistence)}
    result["training_receipt_sha256"] = _sha_bytes(canonical_json_bytes(result))
    _write_new_json(run_root / "training-receipt.json", result)
    if not result["cal_gate_action_aware_strictly_below_persistence"]: raise QualifiedLaunchError("CAL gate failed: action-aware normalized MAE is not strictly below persistence")
    return result


def run(root: Path, run_root: Path, *, dry_run: bool = False) -> None:
    root = root.resolve(); run_root = run_root.resolve()
    protocol, amendment = _verify_protocol(root)
    source_commit = _assert_source_provenance(root, protocol, amendment)
    design = load_approved_pilot_design(root); contracts = load_forecast_contracts(root); split = build_qualification_split(design, load_qualified_protocol(root))
    if (len(split.fit_cluster_ids), len(split.cal_cluster_ids), len(split.validation_cluster_ids)) != (FIT_CLUSTERS, CAL_CLUSTERS, 12) or len(split.authorized_cluster_ids) != AUTHORIZED_CLUSTERS:
        raise QualifiedLaunchError("frozen cluster split cardinality drifts")
    if dry_run:
        print(json.dumps({"verdict": "PASS", "validation_accessed": False, "authorized_clusters": AUTHORIZED_CLUSTERS, "authorized_packets": AUTHORIZED_PACKETS, "authorized_examples": AUTHORIZED_EXAMPLES}, sort_keys=True)); return
    _start_or_resume_run(run_root, protocol=protocol, source_commit=source_commit)
    preflight = _load_pinned_resource_preflight(root)
    limits = protocol["resource_limits"]
    try:
        guard = QualifiedRuntimeGuard(run_root, QualifiedRuntimeLimits(preflight.projected_wall_time_seconds, int(float(limits["abort_free_ram_gb"]) * 1024**3), int(float(limits["abort_vram_free_gb"]) * 1024**3), int(limits["abort_gpu_temperature_c"]), V2_RESOURCE_CEILINGS.disk_reserve_bytes))
        guard.__enter__()
    except QualifiedRuntimeGuardError as error: raise QualifiedLaunchError(str(error)) from error
    corpus = run_root / "corpus"
    _generate_or_resume_corpus(root, corpus, design=design, contracts=contracts, preflight=preflight, allowed_cluster_ids=split.authorized_cluster_ids, health_check=guard.check)
    guard.check("before-custody")
    custody = _verify_exact_custody(root, corpus, split, protocol)
    custody_path = run_root / "custody-receipt.json"
    if custody_path.exists():
        if _read_json(custody_path) != custody: raise QualifiedLaunchError("existing custody receipt does not match corpus")
    else: _write_new_json(custody_path, custody)
    training_path = run_root / "training-receipt.json"
    checkpoint_paths = (run_root / "action-aware-selected.pt", run_root / "action-blind-selected.pt")
    if training_path.exists():
        training = _read_json(training_path)
        if (
            training.get("schema_version") != "aeolus_habitat_v2_fitcal_training_receipt_v2"
            or training.get("protocol_sha256") != protocol["protocol_sha256"]
            or training.get("validation_accessed") is not False
            or training.get("cal_gate_action_aware_strictly_below_persistence") is not True
            or training.get("training_receipt_sha256") != _sha_bytes(canonical_json_bytes({k: v for k, v in training.items() if k != "training_receipt_sha256"}))
            or training.get("checkpoint_sha256") != {path.name: _sha_bytes(path.read_bytes()) for path in checkpoint_paths if path.is_file()}
            or not all(path.is_file() for path in checkpoint_paths)
        ):
            raise QualifiedLaunchError("partial or invalid existing training output is preserved and rejected")
    else:
        if any(path.exists() for path in checkpoint_paths):
            raise QualifiedLaunchError("partial training checkpoint is preserved and rejected")
        training = _train_and_record(run_root, corpus, split, contracts, protocol, custody)
    _write_new_json(run_root / "launcher-complete.json", {"schema_version": "aeolus_habitat_v2_fitcal_launcher_complete_v1", "protocol_sha256": protocol["protocol_sha256"], "validation_accessed": False, "cal_gate_passed": True, "training_receipt_sha256": training["training_receipt_sha256"]})
    guard.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="verify sealed protocol/provenance only; no output, HMC, corpus, or training")
    args = parser.parse_args(argv)
    try: run(args.root, args.run_root, dry_run=args.dry_run)
    except QualifiedLaunchError as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr); return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
