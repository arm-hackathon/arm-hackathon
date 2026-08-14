"""Self-contained report and receipt for the local Habitat V2 live forecast demo."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import load_forecast_contracts
from .corpus import canonical_json_bytes
from .live_demo import DEMO_RELEASE_TIER, LiveForecastResult
from .projection import forecast_layout


class LiveForecastReportError(ValueError):
    """The live forecast report cannot be represented safely."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array(value: np.ndarray) -> list[object]:
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise LiveForecastReportError("report arrays must be finite float32")
    return value.tolist()


def build_live_forecast_payload(
    repo_root: str | Path,
    result: LiveForecastResult,
) -> dict[str, Any]:
    """Build the JSON-safe, claim-bounded report payload."""
    if type(result) is not LiveForecastResult:
        raise LiveForecastReportError("report requires an exact live forecast result")
    if (
        result.release_tier != DEMO_RELEASE_TIER
        or result.actuator_authority
        or not result.hmc_is_sole_actuator_authority
    ):
        raise LiveForecastReportError("report authority boundary drift")
    bundle = load_forecast_contracts(Path(repo_root).resolve())
    layout = forecast_layout(bundle)
    descriptors = [dict(item) for item in layout.target_descriptors]
    if len(descriptors) != 51:
        raise LiveForecastReportError("report target descriptor count drift")
    return {
        "schema_version": "aeolus_habitat_v2_live_forecast_report_v1",
        "release_tier": DEMO_RELEASE_TIER,
        "claims": {
            "forecast_only_local_prototype": True,
            "model_predictions_computed_before_future_steps": True,
            "simulator_generated_truth": True,
            "browser_executes_model_inference": False,
            "hmc_is_sole_actuator_authority": True,
            "model_actuator_authority": False,
            "d2_qualified": False,
            "production_deployed": False,
            "physical_habitat_validated": False,
        },
        "model": {
            "kind": result.model_kind,
            "artifact_sha256": result.model_artifact_sha256,
            "input_manifest_sha256": layout.input_manifest_sha256,
            "target_manifest_sha256": layout.target_manifest_sha256,
        },
        "timeline": {
            "forecast_completed_step": result.forecast_completed_step,
            "forecast_completed_time_s": result.forecast_completed_time_s,
            "history_steps": list(result.forecast_history_steps),
            "truth_steps": list(result.truth_steps),
        },
        "authority": {
            "selection_source": result.selection_source,
            "selected_action_id": result.selected_action_id,
            "selected_command_sha256": result.selected_command_sha256,
            "arbitration_disposition": result.arbitration_disposition,
            "final_command_sha256": result.final_command_sha256,
            "actuator_authority": "deterministic_hmc_only",
        },
        "target_descriptors": descriptors,
        "candidate_forecasts": [
            {
                "action_id": item.action_id,
                "command_sha256": item.command_sha256,
                "proposed_action_f32": _array(item.proposed_action_f32),
                "prediction_f32": _array(item.prediction_f32),
            }
            for item in result.candidate_forecasts
        ],
        "simulator_truth": {
            "selected_action_id": result.selected_action_id,
            "truth_f32": _array(result.truth_f32),
        },
        "replay_evidence": {
            "control_run_id": result.control_run_id,
            "terminal_status": result.terminal_status,
            "trace_sha256": result.trace_sha256,
            "trace_footer_sha256": result.trace_footer_sha256,
            "replay_final_state_sha256": result.replay_final_state_sha256,
            "replay_committed_steps": result.replay_committed_steps,
        },
    }


def write_live_forecast_report(
    repo_root: str | Path,
    result: LiveForecastResult,
    destination: str | Path,
    *,
    source_foundation_git_commit: str,
    integration_source_git_commit: str | None = None,
    source_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Write a new report directory and a receipt binding every emitted artifact."""
    if (
        type(source_foundation_git_commit) is not str
        or len(source_foundation_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_foundation_git_commit)
    ):
        raise LiveForecastReportError("source foundation must be a full Git commit")
    if integration_source_git_commit is not None and (
        type(integration_source_git_commit) is not str
        or len(integration_source_git_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in integration_source_git_commit
        )
    ):
        raise LiveForecastReportError("integration source must be a full Git commit")
    root = Path(repo_root).resolve()
    output = Path(destination).resolve()
    if output.exists():
        raise LiveForecastReportError("report destination must not already exist")
    output.mkdir(parents=True)

    payload = build_live_forecast_payload(root, result)
    payload_raw = canonical_json_bytes(payload)
    payload_path = output / "live-run.json"
    payload_path.write_bytes(payload_raw)

    encoded = base64.b64encode(payload_raw).decode("ascii")
    html_raw = _HTML.replace("__PAYLOAD_BASE64__", encoded).encode("utf-8")
    html_path = output / "index.html"
    html_path.write_bytes(html_raw)

    sources: list[dict[str, object]] = []
    for raw_path in source_paths:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
        except (OSError, ValueError) as error:
            raise LiveForecastReportError("report source path is outside the repository") from error
        sources.append(
            {
                "relative_path": relative,
                "byte_length": len(content),
                "sha256": _sha256(content),
            }
        )
    sources.sort(key=lambda item: str(item["relative_path"]))

    receipt = {
        "schema_version": "aeolus_habitat_v2_live_forecast_receipt_v2",
        "release_tier": DEMO_RELEASE_TIER,
        "source_foundation_git_commit": source_foundation_git_commit,
        "integration_source_committed": integration_source_git_commit is not None,
        "integration_source_git_commit": integration_source_git_commit,
        "qualification_evidence": False,
        "actuator_authority": False,
        "model_artifact_sha256": result.model_artifact_sha256,
        "control_trace_sha256": result.trace_sha256,
        "source_files": sources,
        "artifacts": [
            {
                "relative_path": "index.html",
                "byte_length": len(html_raw),
                "sha256": _sha256(html_raw),
            },
            {
                "relative_path": "live-run.json",
                "byte_length": len(payload_raw),
                "sha256": _sha256(payload_raw),
            },
        ],
    }
    receipt_raw = canonical_json_bytes(receipt)
    receipt_path = output / "receipt.json"
    receipt_path.write_bytes(receipt_raw)
    return {
        "output_directory": str(output),
        "index_html": str(html_path),
        "live_run_json": str(payload_path),
        "receipt_json": str(receipt_path),
        "index_sha256": _sha256(html_raw),
        "live_run_sha256": _sha256(payload_raw),
        "receipt_sha256": _sha256(receipt_raw),
    }


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AEOLUS Habitat V2 Live Forecast</title>
<style>
:root{--bg:#070b12;--panel:#0d1420;--panel2:#111b2a;--line:#21324a;--text:#e8f0f8;--muted:#91a2b7;--cyan:#55d8ff;--amber:#ffbd59;--green:#70e3a1;--red:#ff7c87}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#122039 0,#070b12 38%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.5}.shell{max-width:1180px;margin:auto;padding:34px 22px 60px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.18em;font-weight:800}.hero{display:grid;grid-template-columns:1.6fr 1fr;gap:18px;align-items:end;margin:8px 0 22px}.hero h1{font-size:clamp(36px,6vw,68px);line-height:.98;margin:8px 0 14px;letter-spacing:-.045em}.hero p{color:var(--muted);max-width:720px;font-size:17px}.badge{border:1px solid #246c53;background:#0d2b23;color:var(--green);padding:10px 14px;border-radius:999px;font-weight:800;display:inline-flex;gap:9px;align-items:center}.dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 16px var(--green)}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.panel{background:linear-gradient(180deg,rgba(17,27,42,.96),rgba(10,16,26,.96));border:1px solid var(--line);border-radius:16px}.card{padding:17px}.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}.card .v{font-size:24px;font-weight:850;margin-top:5px}.panel{padding:22px;margin-top:14px}.panel h2{margin:0 0 5px;font-size:20px}.sub{color:var(--muted);margin:0 0 18px}.timeline{display:grid;grid-template-columns:1fr 52px 1fr 52px 1fr;align-items:center;gap:8px}.stage{background:#0a111d;border:1px solid var(--line);border-radius:13px;padding:14px;min-height:100px}.stage strong{display:block;color:var(--cyan);margin-bottom:5px}.arrow{text-align:center;color:var(--amber);font-size:26px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}.control label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}.control select{width:100%;background:#080e17;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px}.chart-wrap{background:#080e17;border:1px solid var(--line);border-radius:12px;padding:10px}canvas{width:100%;height:350px;display:block}.legend{display:flex;gap:18px;color:var(--muted);font-size:13px;margin:9px 3px}.swatch{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle}.warning{border-left:3px solid var(--amber);background:#2c2212;padding:12px 14px;color:#ffd89d;border-radius:8px;margin-top:12px}.authority{display:grid;grid-template-columns:1fr 1fr;gap:12px}.authority .good,.authority .off{padding:16px;border-radius:12px}.good{background:#0c2a21;border:1px solid #205c49}.off{background:#211519;border:1px solid #5e2a35}.hashes{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;color:#a8bdd5;word-break:break-all}.table{width:100%;border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums}.table th,.table td{text-align:right;padding:8px;border-bottom:1px solid #18263a}.table th:first-child,.table td:first-child{text-align:left}.table th{color:var(--muted);font-size:12px}.footer{color:var(--muted);font-size:12px;margin-top:20px}@media(max-width:800px){.hero,.authority{grid-template-columns:1fr}.grid{grid-template-columns:1fr 1fr}.timeline{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}.controls{grid-template-columns:1fr}}@media(max-width:480px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main class="shell">
<div class="eyebrow">AEOLUS / HABITAT V2 / FORECAST-ONLY PROTOTYPE</div>
<section class="hero"><div><h1>Forecast before the future existed.</h1><p>At completed step 16, the saved action-aware ridge receives only the issued history from steps 13 to 16. It forecasts all four catalogue actions. HMC then arbitrates the operator-selected proposal and the simulator creates truth for steps 17 to 24.</p></div><div><span class="badge"><span class="dot"></span>DETERMINISTIC LIVE RUN COMPLETE</span></div></section>
<section class="grid"><div class="card"><div class="k">Forecast anchor</div><div class="v" id="anchor">Step 16</div></div><div class="card"><div class="k">Candidate actions</div><div class="v" id="actions">4</div></div><div class="card"><div class="k">Future horizon</div><div class="v">8 minutes</div></div><div class="card"><div class="k">HMC disposition</div><div class="v" id="disposition">ACCEPTED</div></div></section>
<section class="panel"><h2>Causal run order</h2><p class="sub">The model cannot read the future target states because they do not exist when inference runs.</p><div class="timeline"><div class="stage"><strong>1. Observe</strong>HMC issues verified operational snapshots for completed steps 13 to 16.</div><div class="arrow">→</div><div class="stage"><strong>2. Forecast</strong>The model evaluates four complete candidate commands using one shared history.</div><div class="arrow">→</div><div class="stage"><strong>3. Execute and compare</strong>HMC owns the final command. The simulator advances and produces steps 17 to 24.</div></div></section>
<section class="panel"><h2>Action-conditioned forecast explorer</h2><p class="sub">Choose a candidate action and one of the 51 physical output channels.</p><div class="controls"><div class="control"><label for="action">Candidate action</label><select id="action"></select></div><div class="control"><label for="signal">Physical signal</label><select id="signal"></select></div></div><div class="chart-wrap"><canvas id="chart"></canvas></div><div class="legend"><span><i class="swatch" style="background:var(--cyan)"></i>Model forecast</span><span><i class="swatch" style="background:var(--amber)"></i>Simulator truth for executed action</span></div><div class="warning" id="comparison"></div><table class="table"><thead><tr><th>Future minute</th><th>Forecast</th><th>Simulator truth</th><th>Absolute error</th></tr></thead><tbody id="rows"></tbody></table></section>
<section class="panel"><h2>Authority boundary</h2><p class="sub">Forecasting and command authority remain separate by construction.</p><div class="authority"><div class="good"><strong>Deterministic HMC: sole authority</strong><br>Validates the proposal, applies policy, chooses the final command and advances the plant.</div><div class="off"><strong>Learned model: no actuator authority</strong><br>Produces counterfactual trajectories only. It does not select, modify or issue a command.</div></div></section>
<section class="panel"><h2>Replay and identity evidence</h2><div class="hashes" id="hashes"></div></section>
<p class="footer">This HTML replays predictions generated during the live Python run. It does not execute model inference in the browser. The model is trained on simulator data, is permanently excluded from D2 qualification evidence, and is not validated on a physical habitat.</p>
</main><script>
const payload=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__PAYLOAD_BASE64__'),c=>c.charCodeAt(0))));
const actionSelect=document.getElementById('action');const signalSelect=document.getElementById('signal');
payload.candidate_forecasts.forEach(x=>{const o=document.createElement('option');o.value=x.action_id;o.textContent=x.action_id+(x.action_id===payload.authority.selected_action_id?' (executed)':' (counterfactual)');actionSelect.appendChild(o)});
payload.target_descriptors.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=x.descriptor_id+' ['+x.unit+']';signalSelect.appendChild(o)});
const co2=payload.target_descriptors.findIndex(x=>x.descriptor_id.endsWith('/co2_ppm'));signalSelect.value=co2<0?0:co2;actionSelect.value=payload.authority.selected_action_id;
document.getElementById('anchor').textContent='Step '+payload.timeline.forecast_completed_step;document.getElementById('actions').textContent=payload.candidate_forecasts.length;document.getElementById('disposition').textContent=payload.authority.arbitration_disposition;
function fmt(x){const a=Math.abs(x);if((a!==0&&a<.001)||a>=1e6)return x.toExponential(4);return x.toLocaleString(undefined,{maximumFractionDigits:5})}
function line(ctx,values,x,y,color){ctx.beginPath();values.forEach((v,i)=>{const px=x(i),py=y(v);if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)});ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke()}
function render(){const action=payload.candidate_forecasts.find(x=>x.action_id===actionSelect.value);const index=Number(signalSelect.value);const pred=action.prediction_f32.map(row=>row[index]);const comparable=action.action_id===payload.authority.selected_action_id;const truth=payload.simulator_truth.truth_f32.map(row=>row[index]);const values=comparable?pred.concat(truth):pred;let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.12;lo-=pad;hi+=pad;const canvas=document.getElementById('chart');const dpr=window.devicePixelRatio||1;const width=canvas.clientWidth,height=350;canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);const left=70,right=20,top=24,bottom=45;const x=i=>left+i*(width-left-right)/7;const y=v=>top+(hi-v)*(height-top-bottom)/(hi-lo);ctx.strokeStyle='#1d2b40';ctx.lineWidth=1;ctx.fillStyle='#91a2b7';ctx.font='12px system-ui';for(let i=0;i<5;i++){const py=top+i*(height-top-bottom)/4;ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(width-right,py);ctx.stroke();ctx.fillText(fmt(hi-i*(hi-lo)/4),4,py+4)}for(let i=0;i<8;i++){ctx.fillText(String(i+1),x(i)-3,height-18)}line(ctx,pred,x,y,'#55d8ff');if(comparable)line(ctx,truth,x,y,'#ffbd59');const descriptor=payload.target_descriptors[index];ctx.fillStyle='#e8f0f8';ctx.font='600 13px system-ui';ctx.fillText(descriptor.descriptor_id+' ['+descriptor.unit+']',left,15);document.getElementById('comparison').textContent=comparable?'Causal comparison enabled: simulator truth belongs to this executed action.':'Counterfactual only: the simulator executed '+payload.authority.selected_action_id+', so its truth is not presented as evidence for this unexecuted action.';const rows=document.getElementById('rows');rows.replaceChildren();for(let i=0;i<8;i++){const tr=document.createElement('tr');const t=comparable?fmt(truth[i]):'not observed';const e=comparable?fmt(Math.abs(pred[i]-truth[i])):'not applicable';[String(i+1),fmt(pred[i]),t,e].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.appendChild(td)});rows.appendChild(tr)}}
actionSelect.addEventListener('change',render);signalSelect.addEventListener('change',render);window.addEventListener('resize',render);const hashes=document.getElementById('hashes');[['Model artifact',payload.model.artifact_sha256],['Control trace',payload.replay_evidence.trace_sha256],['Replay final state',payload.replay_evidence.replay_final_state_sha256],['Committed replay steps',payload.replay_evidence.replay_committed_steps]].forEach((item,i)=>{if(i)hashes.append(document.createElement('br'),document.createElement('br'));const label=document.createElement('b');label.textContent=item[0];hashes.append(label,document.createElement('br'),document.createTextNode(String(item[1])))});render();
</script></body></html>'''
