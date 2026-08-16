"""Fail-closed authorized AEOLUS V2 FIT+CAL generation, custody, and training launcher.
Validation and final packets are never enumerated or read. HMC remains sole authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MIN_RAM = int(3.5 * 1024**3); MIN_VRAM_MIB = 4096; MAX_TEMP = 82
ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "qual_v2_prepare.py"

def canonical(v: Any) -> bytes: return json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def fail(msg: str) -> None: raise RuntimeError("FAIL-CLOSED: " + msg)
def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if p.returncode: fail(f"command failed ({' '.join(args)}): {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout

def resources() -> dict[str, Any]:
    try:
        import psutil
        free_ram = psutil.virtual_memory().available
    except (ImportError, OSError) as e: fail(f"cannot probe RAM: {e}")
    q = run("nvidia-smi", "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits").strip().splitlines()
    if len(q) != 1: fail("expected exactly one GPU")
    vram, temp = (int(x.strip()) for x in q[0].split(","))
    apps = run("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader").strip()
    # Never kill applications. Only block named AEOLUS/training compute workloads.
    competing = [x for x in apps.splitlines() if x and ("aeolus" in x.lower() or "train" in x.lower())]
    return {"free_ram_bytes": free_ram, "free_vram_mib": vram, "gpu_temp_c": temp, "competing_compute": competing}

def require_resources() -> dict[str, Any]:
    r=resources()
    if r["free_ram_bytes"] < MIN_RAM: fail(f"free RAM {r['free_ram_bytes']} < {MIN_RAM}")
    if r["free_vram_mib"] < MIN_VRAM_MIB: fail(f"free VRAM {r['free_vram_mib']} MiB < {MIN_VRAM_MIB}")
    if r["gpu_temp_c"] >= MAX_TEMP: fail(f"GPU temperature {r['gpu_temp_c']} >= {MAX_TEMP}")
    if r["competing_compute"]: fail(f"competing AEOLUS/training compute: {r['competing_compute']}")
    return r

def load_preflight(path: Path, head: str) -> dict[str, Any]:
    if not path.is_file(): fail("preflight receipt is missing")
    d=json.loads(path.read_text())
    declared=d.pop("preflight_sha256", None)
    if declared != hashlib.sha256(canonical(d)).hexdigest(): fail("preflight semantic hash mismatch")
    d["preflight_sha256"]=declared
    if d.get("source_commit") != head or d.get("verdict") != "PASS": fail("preflight commit/verdict mismatch")
    from aeolus.habitat_v2.forecast.pilot_benchmark import build_v2_source_manifest
    if d.get("source_manifest") != build_v2_source_manifest(ROOT): fail("preflight source manifest drift")
    return d

def gate(preflight: Path) -> tuple[str, dict[str, Any], Any]:
    head=run("git", "rev-parse", "HEAD").strip()
    dirty=run("git", "status", "--porcelain=v1").strip()
    if dirty: fail(f"repository snapshot dirty: {dirty}")
    p=load_preflight(preflight,head)
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
    from aeolus.habitat_v2.forecast.qualification_split import fit_cal_cluster_ids
    split=fit_cal_cluster_ids(ROOT,load_approved_pilot_design(ROOT))
    if (len(split.fit_cluster_ids),len(split.cal_cluster_ids),len(split.validation_cluster_ids),split.expected_packet_count,split.expected_example_count)!=(36,12,12,3744,18720): fail("split cardinality contract drift")
    return head,p,split

def verify_custody(corpus: Path, receipt: Path, split: Any) -> dict[str, Any]:
    run(sys.executable, str(PREPARE), "custody", "--root", str(ROOT), "--corpus", str(corpus), "--output", str(receipt))
    d=json.loads(receipt.read_text())
    s=d["splits"]
    if (d["packet_count"],d["example_count"],s["fit"]["packets"],s["fit"]["examples"],s["cal"]["packets"],s["cal"]["examples"],s["validation"]["packets"],s["validation"]["examples"]) != (3744,18720,2808,14040,936,4680,0,0): fail("custody exact-count or validation-seal failure")
    if set(s["fit"]["clusters"]) != set(split.fit_cluster_ids) or set(s["cal"]["clusters"]) != set(split.cal_cluster_ids): fail("custody cluster assignment drift")
    return d

def train(corpus: Path, out: Path, split: Any) -> dict[str, Any]:
    # Literal frozen architecture/optimizer; only FIT learns, CAL selects epoch. No validation paths.
    import torch
    torch.manual_seed(20260816); np.random.seed(20260816); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    files=sorted(corpus.glob("*/training.npz")); fit=[]; cal=[]
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            cluster=str(z["cluster_ids"][0]); dest=fit if cluster in split.fit_cluster_ids else cal if cluster in split.cal_cluster_ids else None
            if dest is None: fail("validation/unknown packet encountered during training")
            dest.append((f,cluster))
    def load(rows):
        xs=[]; ys=[]
        for f,_ in rows:
            with np.load(f,allow_pickle=False) as z:
                h=z['history_numeric_f32']; a=z['operational_available_bool'].astype(np.float32); h=h.copy(); h[:,:,:167]*=a
                x=np.concatenate((h.reshape(5,-1),a.reshape(5,-1),z['proposed_action_f32'],z['action_present'].astype(np.float32).reshape(5,1)),axis=1)
                xs.append(x); ys.append(z['targets_f32'].reshape(5,-1))
        return np.concatenate(xs).astype(np.float32),np.concatenate(ys).astype(np.float32)
    xfit,yfit=load(fit); xcal,ycal=load(cal)
    mean=yfit.mean(0); scale=np.maximum(yfit.std(0),1e-6); yfitn=(yfit-mean)/scale; ycaln=(ycal-mean)/scale
    dev='cuda'; model=torch.nn.Sequential(torch.nn.Linear(5804,512),torch.nn.GELU(),torch.nn.Linear(512,512),torch.nn.GELU(),torch.nn.Linear(512,256),torch.nn.GELU(),torch.nn.Linear(256,408)).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=float('inf'); wait=0; best_state=None; history=[]
    for epoch in range(80):
        model.train(); order=np.random.permutation(len(xfit))
        for i in range(0,len(order),128):
            ix=order[i:i+128]; pred=model(torch.from_numpy(xfit[ix]).to(dev)); loss=torch.nn.functional.mse_loss(pred,torch.from_numpy(yfitn[ix]).to(dev)); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): metric=float(torch.mean(torch.abs(model(torch.from_numpy(xcal).to(dev))-torch.from_numpy(ycaln).to(dev))).cpu())
        history.append(metric)
        if metric < best: best=metric; wait=0; best_state={k:v.cpu() for k,v in model.state_dict().items()}
        else: wait+=1
        if wait >= 12: break
    if best_state is None: fail("training yielded no checkpoint")
    ck=out/'checkpoint.pt'; torch.save({'state_dict':best_state,'target_mean':mean,'target_scale':scale,'best_cal_normalized_mae':best,'epochs':len(history),'protocol':'action-aware-mlp-v1'},ck)
    return {'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'best_cal_normalized_mae':best,'epochs':len(history),'fit_examples':len(yfit),'cal_examples':len(ycal)}

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',required=True); ap.add_argument('--resource-preflight',required=True); ap.add_argument('--run-id'); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    head,p,split=gate(Path(a.preflight).resolve()); resource_preflight=Path(a.resource_preflight).resolve(); before=require_resources()
    rid=a.run_id or datetime.now(timezone.utc).strftime('fitcal-%Y%m%dT%H%M%SZ'); out=ROOT/'out'/'helios-qual-v2'/rid
    if out.exists(): fail(f"refuse overwrite existing run output {out}")
    out.mkdir(parents=True); receipt={'schema_version':'aeolus_habitat_v2_fitcal_launcher_v1','run_id':rid,'source_commit':head,'preflight_path':str(Path(a.preflight).resolve()),'preflight_sha256':p['preflight_sha256'],'preflight_bytes_sha256':sha(Path(a.preflight).resolve()),'resource_preflight_path':str(resource_preflight),'resource_preflight_bytes_sha256':sha(resource_preflight),'resources_before':before,'validation':'SEALED'}
    (out/'launcher-start.json').write_bytes(canonical(receipt))
    if a.dry_run: print(json.dumps(receipt,indent=2)); return
    require_resources() # immediate pre-compute re-probe
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot import (
        load_approved_pilot_design,
        load_resource_preflight,
    )
    from aeolus.habitat_v2.forecast.pilot_campaign import run_pilot_campaign
    pre=load_resource_preflight(resource_preflight,repo_root=ROOT,expected_preflight_sha256=json.loads(resource_preflight.read_text())['preflight_sha256'],expected_preflight_bytes_sha256=sha(resource_preflight))
    manifest=run_pilot_campaign(ROOT,load_approved_pilot_design(ROOT),load_forecast_contracts(ROOT),preflight=pre,output_root=out/'corpus',allowed_cluster_ids=split.authorized_cluster_ids,pair_limit=None,worker_count=1,resume=False)
    custody=verify_custody(out/'corpus',out/'custody.json',split); training=train(out/'corpus',out,split)
    receipt.update({'campaign_manifest_sha256':hashlib.sha256(canonical(manifest)).hexdigest(),'custody_sha256':custody['custody_sha256'],'training':training,'completed_at':datetime.now(timezone.utc).isoformat()}); (out/'launcher-complete.json').write_bytes(canonical(receipt)); print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
