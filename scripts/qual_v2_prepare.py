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

def verify_campaign(root: Path, corpus: Path, *, forbid_validation: bool) -> dict[str, Any]:
    proto=load_protocol(root); packets=sorted(corpus.glob("*/training.npz"))
    if not packets: raise ValueError("no packets")
    rows=[verify_packet(p) for p in packets]
    split=fit_cal_cluster_ids(root,load_approved_pilot_design(root))
    assignment={r["cluster_id"]:split.split_for(r["cluster_id"]) for r in rows}
    if forbid_validation and any(v=="validation" for v in assignment.values()): raise ValueError("validation packet access refused")
    # A cluster may have 78 packets (2 reps x 3 anchors x 13 pairs) but only one split.
    by={s:[] for s in ("fit","cal","validation")}
    for r in rows: by[assignment[r["cluster_id"]]].append(r)
    manifest={"schema_version":"aeolus_habitat_v2_qualified_corpus_custody_v1","protocol_sha256":proto["protocol_sha256"],"corpus":str(corpus),"packet_count":len(rows),"example_count":sum(r["examples"] for r in rows),"availability_true":sum(r["available_true"] for r in rows),"availability_total":sum(r["available_total"] for r in rows),"splits":{k:{"packets":len(v),"examples":sum(x["examples"] for x in v),"clusters":sorted({x["cluster_id"] for x in v}),"packet_sha256s":[x["sha256"] for x in v]} for k,v in by.items()}}
    manifest["custody_sha256"]=sha_bytes(canonical_json_bytes(manifest))
    return manifest

def cmd_preflight(a: argparse.Namespace) -> None:
    root=Path(a.root).resolve(); output=Path(a.output).resolve()
    if output.exists(): raise ValueError("new preflight output must not exist")
    proto=load_protocol(root); design=load_approved_pilot_design(root); load_forecast_contracts(root)
    # Validate all immutable source bindings before any HMC execution. The benchmark itself is intentionally separate.
    receipt={"schema_version":"aeolus_habitat_v2_qualified_protocol_preflight_v1","protocol_sha256":proto["protocol_sha256"],"source_commit":a.source_commit,"source_manifest":build_v2_source_manifest(root),"contract_identities":build_v2_contract_identities(root),"runtime_identity":build_v2_runtime_identity(),"design":{"clusters":len(design.clusters),"continuations":sum(1 for _ in __import__("aeolus.habitat_v2.forecast.pilot",fromlist=["iter_pilot_continuations"]).iter_pilot_continuations(design))},"disk_free_bytes":shutil.disk_usage(root).free,"checks":{"protocol":True,"design":True,"contracts":True,"new_output":True,"no_validation_open":True},"verdict":"PASS"}
    receipt["preflight_sha256"]=sha_bytes(canonical_json_bytes(receipt)); output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(canonical_json_bytes(receipt)); print(json.dumps(receipt,indent=2))

def cmd_custody(a: argparse.Namespace) -> None:
    m=verify_campaign(Path(a.root).resolve(),Path(a.corpus).resolve(),forbid_validation=not a.allow_validation)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(canonical_json_bytes(m)); print(json.dumps({k:m[k] for k in ("packet_count","example_count","availability_true","availability_total","custody_sha256","splits")},indent=2))

p=argparse.ArgumentParser(); sub=p.add_subparsers(required=True)
x=sub.add_parser("preflight"); x.add_argument("--root",required=True);x.add_argument("--output",required=True);x.add_argument("--source-commit",required=True);x.set_defaults(fn=cmd_preflight)
x=sub.add_parser("custody");x.add_argument("--root",required=True);x.add_argument("--corpus",required=True);x.add_argument("--output",required=True);x.add_argument("--allow-validation",action="store_true");x.set_defaults(fn=cmd_custody)
a=p.parse_args();a.fn(a)
