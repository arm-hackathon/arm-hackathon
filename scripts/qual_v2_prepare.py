"""Qualification V2 protocol/corpus preflight and deterministic split/custody tools.
Does not read validation packets. HMC stays sole actuator/safety authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.forecast.pilot import (
    load_approved_pilot_design,
)
from aeolus.habitat_v2.forecast.pilot_benchmark import (
    build_v2_contract_identities,
    build_v2_runtime_identity,
    build_v2_source_manifest,
)
from aeolus.habitat_v2.forecast.pilot_campaign import _availability_row_sha256
from aeolus.habitat_v2.forecast.qualification_split import fit_cal_cluster_ids

PROTOCOL_REL = Path("docs/plans/2026-08-16-habitat-v2-qualified-model-protocol-v1.json")
V2_SCHEMA = "aeolus_habitat_v2_forecast_training_pair_v2"

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha_file(p: Path) -> str: return sha_bytes(p.read_bytes())
def load_protocol(root: Path) -> dict[str, Any]:
    d = json.loads((root / PROTOCOL_REL).read_text(encoding="utf-8"))
    declared = d.pop("protocol_sha256", None)
    if declared != sha_bytes(canonical_json_bytes(d)):
        raise ValueError("protocol self-hash mismatch")
    d["protocol_sha256"] = declared
    return d

def verify_packet(p: Path) -> dict[str, Any]:
    raw = p.read_bytes(); z = np.load(p, allow_pickle=False)
    required = {"schema_version","pair_id","continuation_ids","cluster_ids","action_ids","action_present","history_numeric_f32","operational_available_bool","operational_available_sha256","proposed_action_f32","targets_f32"}
    if set(z.files) != required: raise ValueError(f"{p}: fields drift {set(z.files)^required}")
    if str(z["schema_version"].item()) != V2_SCHEMA: raise ValueError(f"{p}: schema")
    h,a,t = z["history_numeric_f32"], z["operational_available_bool"], z["targets_f32"]
    if h.shape != (5,16,194) or h.dtype != np.float32 or a.shape != (5,16,167) or a.dtype != np.bool_ or t.shape != (5,8,51) or t.dtype != np.float32: raise ValueError(f"{p}: tensor shape/dtype")
    if not (np.isfinite(h).all() and np.isfinite(t).all() and np.isfinite(z["proposed_action_f32"]).all()): raise ValueError(f"{p}: nonfinite")
    ids=z["continuation_ids"].tolist(); hashes=z["operational_available_sha256"].tolist()
    if len(ids)!=5 or len(set(ids))!=5 or any(_availability_row_sha256(i,row) != q for i,row,q in zip(ids,a,hashes,strict=True)): raise ValueError(f"{p}: availability hash")
    clusters=z["cluster_ids"].tolist()
    if len(set(clusters)) != 1: raise ValueError(f"{p}: mixed cluster")
    return {"path":str(p),"sha256":sha_bytes(raw),"cluster_id":clusters[0],"examples":5,"available_true":int(a.sum()),"available_total":int(a.size)}

def verify_campaign(root: Path, corpus: Path, *, forbid_validation: bool = True) -> dict[str, Any]:
    """Cryptographically traverse campaign -> pair evidence -> NPZ custody.

    Discovery by glob is deliberately forbidden: every accepted byte must be
    named by the self-hashed campaign ledger and every directory/file is exact.
    """
    if forbid_validation is not True:
        raise ValueError("qualification custody is permanently validation-closed")
    from aeolus.habitat_v2.forecast.pilot import APPROVED_PROFILE_ACTION_SHA256, APPROVED_ROSTER_SHA256, load_approved_pilot_design
    from aeolus.habitat_v2.forecast.pilot_campaign import _load_validated_staged_pair
    proto = load_protocol(root)
    design = load_approved_pilot_design(root)
    campaign_path = corpus / "campaign-manifest.json"
    raw_campaign = campaign_path.read_bytes()
    campaign = json.loads(raw_campaign)
    declared = campaign.pop("campaign_manifest_sha256", None)
    if type(declared) is not str or declared != sha_bytes(canonical_json_bytes(campaign)) or raw_campaign != canonical_json_bytes({**campaign, "campaign_manifest_sha256": declared}):
        raise ValueError("campaign manifest self-hash/canonical bytes drift")
    required = {"schema_version", "roster_sha256", "profile_action_sha256", "preflight_sha256", "planned_hmc_runs", "worker_count", "allowed_cluster_ids", "pairs_completed", "hmc_runs_executed", "pair_manifests"}
    if set(campaign) != required or campaign["schema_version"] != "aeolus_habitat_v2_forecast_pilot_campaign_manifest_v1":
        raise ValueError("campaign manifest schema drift")
    split = fit_cal_cluster_ids(root, design)
    allowed = sorted(split.authorized_cluster_ids)
    if campaign["roster_sha256"] != APPROVED_ROSTER_SHA256 or campaign["profile_action_sha256"] != APPROVED_PROFILE_ACTION_SHA256 or campaign["allowed_cluster_ids"] != allowed:
        raise ValueError("campaign roster/profile/qualification split drift")
    entries = campaign["pair_manifests"]
    if type(entries) is not list or campaign["pairs_completed"] != len(entries) or campaign["hmc_runs_executed"] != len(entries) * 5:
        raise ValueError("campaign pair/run cardinality drift")
    pair_ids = [item.get("pair_id") for item in entries if type(item) is dict]
    if len(pair_ids) != len(entries) or len(set(pair_ids)) != len(pair_ids) or any(type(x) is not str for x in pair_ids):
        raise ValueError("campaign pair ledger identity drift")
    actual_root = {item.name for item in corpus.iterdir()}
    if actual_root != {"campaign-manifest.json", *pair_ids}:
        raise ValueError("campaign has unledgered or missing root artifacts")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        pair_id = entry["pair_id"]
        validated = _load_validated_staged_pair(corpus / pair_id, design)
        if entry != validated:
            raise ValueError("campaign pair ledger does not bind staged evidence")
        packet = verify_packet(corpus / pair_id / "training.npz")
        if packet["sha256"] != entry["training_packet_sha256"] or packet["cluster_id"] not in split.authorized_cluster_ids:
            raise ValueError("ledger packet hash or authorization drift")
        rows.append(packet)
    assignment = {r["cluster_id"]: split.split_for(r["cluster_id"]) for r in rows}
    if any(v == "validation" for v in assignment.values()):
        raise ValueError("validation packet access refused")
    by = {name: [] for name in ("fit", "cal", "validation")}
    for row in rows: by[assignment[row["cluster_id"]]].append(row)
    receipt = {"schema_version":"aeolus_habitat_v2_qualified_corpus_custody_v2", "protocol_sha256":proto["protocol_sha256"], "campaign_manifest_sha256":declared, "campaign_manifest_bytes_sha256":sha_bytes(raw_campaign), "corpus":str(corpus), "packet_count":len(rows), "example_count":sum(r["examples"] for r in rows), "availability_true":sum(r["available_true"] for r in rows), "availability_total":sum(r["available_total"] for r in rows), "splits":{k:{"packets":len(v),"examples":sum(x["examples"] for x in v),"clusters":sorted({x["cluster_id"] for x in v}),"packet_paths":[x["path"] for x in v],"packet_sha256s":[x["sha256"] for x in v]} for k,v in by.items()}}
    receipt["custody_sha256"] = sha_bytes(canonical_json_bytes(receipt))
    return receipt

def cmd_preflight(a: argparse.Namespace) -> None:
    root=Path(a.root).resolve(); output=Path(a.output).resolve()
    if output.exists(): raise ValueError("new preflight output must not exist")
    proto=load_protocol(root); design=load_approved_pilot_design(root); load_forecast_contracts(root)
    # Validate all immutable source bindings before any HMC execution. The benchmark itself is intentionally separate.
    receipt={"schema_version":"aeolus_habitat_v2_qualified_protocol_preflight_v1","protocol_sha256":proto["protocol_sha256"],"source_commit":a.source_commit,"source_manifest":build_v2_source_manifest(root),"contract_identities":build_v2_contract_identities(root),"runtime_identity":build_v2_runtime_identity(),"design":{"clusters":len(design.clusters),"continuations":sum(1 for _ in __import__("aeolus.habitat_v2.forecast.pilot",fromlist=["iter_pilot_continuations"]).iter_pilot_continuations(design))},"disk_free_bytes":shutil.disk_usage(root).free,"checks":{"protocol":True,"design":True,"contracts":True,"new_output":True,"no_validation_open":True},"verdict":"PASS"}
    receipt["preflight_sha256"]=sha_bytes(canonical_json_bytes(receipt)); output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(canonical_json_bytes(receipt)); print(json.dumps(receipt,indent=2))

def cmd_custody(a: argparse.Namespace) -> None:
    m=verify_campaign(Path(a.root).resolve(),Path(a.corpus).resolve(),forbid_validation=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(canonical_json_bytes(m)); print(json.dumps({k:m[k] for k in ("packet_count","example_count","availability_true","availability_total","custody_sha256","splits")},indent=2))

p=argparse.ArgumentParser(); sub=p.add_subparsers(required=True)
x=sub.add_parser("preflight"); x.add_argument("--root",required=True);x.add_argument("--output",required=True);x.add_argument("--source-commit",required=True);x.set_defaults(fn=cmd_preflight)
x=sub.add_parser("custody");x.add_argument("--root",required=True);x.add_argument("--corpus",required=True);x.add_argument("--output",required=True);x.set_defaults(fn=cmd_custody)
a=p.parse_args();a.fn(a)
