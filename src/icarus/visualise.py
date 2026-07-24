"""Create a standalone HTML visualisation from an ICARUS JSONL trace.

Usage:
    PYTHONPATH=src python -m icarus.visualise <trace.jsonl> <report.html>

The report embeds its data and uses no external JavaScript or CSS, so it can
be opened directly in a browser and shared as a single file.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path
from typing import Any


USAGE = (
    "Usage: PYTHONPATH=src python -m icarus.visualise <trace.jsonl> <report.html>\n"
    "Example: PYTHONPATH=src python -m icarus.visualise "
    "traces/standard_habitat.jsonl out/standard_habitat.html"
)


def load_trace(path) -> list[dict[str, Any]]:
    """Load and validate the fields needed by the visualiser."""
    trace_path = Path(path)
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise ValueError(f"trace file not found: {trace_path}") from None
    except OSError as exc:
        raise ValueError(f"cannot read trace file {trace_path}: {exc}") from None

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"trace line {line_number} is not valid JSON: {exc.msg}"
            ) from None
        _validate_row(row, line_number)
        expected_tick = len(rows) + 1
        if row["tick"] != expected_tick:
            raise ValueError(
                f"trace line {line_number} expected tick {expected_tick}, "
                f"got {row['tick']}"
            )
        if rows:
            _validate_schema(row, rows[0], line_number)
        rows.append(row)

    if not rows:
        raise ValueError("trace contains no records")
    return rows


def _finite_number(value: Any, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{description} must be finite")


def _validate_schema(
    row: dict[str, Any], first_row: dict[str, Any], line_number: int
) -> None:
    for field in ("zones", "connections", "actuators"):
        if set(row[field]) != set(first_row[field]):
            raise ValueError(f"trace line {line_number} {field} do not match line 1")


def _validate_row(row: Any, line_number: int) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"trace line {line_number} must be a JSON object")
    for field in ("tick", "zones", "connections", "actuators", "system"):
        if field not in row:
            raise ValueError(f"trace line {line_number} is missing {field!r}")
    tick = row["tick"]
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 1:
        raise ValueError(f"trace line {line_number} tick must be a positive integer")
    if not isinstance(row["zones"], dict):
        raise ValueError(f"trace line {line_number} 'zones' must be an object")
    if not isinstance(row["connections"], dict):
        raise ValueError(f"trace line {line_number} 'connections' must be an object")
    if not isinstance(row["actuators"], dict):
        raise ValueError(f"trace line {line_number} 'actuators' must be an object")
    if not isinstance(row["system"], dict):
        raise ValueError(f"trace line {line_number} 'system' must be an object")

    for zone_id, readings in row["zones"].items():
        if not isinstance(readings, dict):
            raise ValueError(
                f"trace line {line_number} zone {zone_id!r} must be an object"
            )
        for field in (
            "co2_mass",
            "co2_concentration",
            "sensor_co2_concentration",
            "source_co2_mass",
            "occupancy_multiplier",
        ):
            if field not in readings:
                raise ValueError(
                    f"trace line {line_number} zone {zone_id!r} is missing {field!r}"
                )
            _finite_number(
                readings[field],
                f"trace line {line_number} zone {zone_id!r} {field}",
            )
        if "captured_co2" in readings:
            _finite_number(
                readings["captured_co2"],
                f"trace line {line_number} zone {zone_id!r} captured_co2",
            )

    for connection_id, readings in row["connections"].items():
        if not isinstance(readings, dict):
            raise ValueError(
                f"trace line {line_number} connection {connection_id!r} "
                "must be an object"
            )
        for field in ("requested_airflow", "airflow", "health"):
            if field not in readings:
                raise ValueError(
                    f"trace line {line_number} connection {connection_id!r} "
                    f"is missing {field!r}"
                )
            _finite_number(
                readings[field],
                f"trace line {line_number} connection {connection_id!r} {field}",
            )
        if readings["requested_airflow"] < 0.0:
            raise ValueError(
                f"trace line {line_number} connection {connection_id!r} "
                "requested_airflow must not be negative"
            )
        if readings["airflow"] < 0.0:
            raise ValueError(
                f"trace line {line_number} connection {connection_id!r} "
                "airflow must not be negative"
            )
        if not 0.0 <= readings["health"] <= 1.0:
            raise ValueError(
                f"trace line {line_number} connection {connection_id!r} "
                "health must be in 0.0..1.0"
            )

    actuator_fields = (
        "setpoint",
        "actual_position",
        "tracking_residual",
        "moving",
        "movement_seconds",
        "power",
        "direction",
    )
    for actuator_id, readings in row["actuators"].items():
        if not isinstance(readings, dict):
            raise ValueError(
                f"trace line {line_number} actuator {actuator_id!r} "
                "must be an object"
            )
        for field in actuator_fields:
            if field not in readings:
                raise ValueError(
                    f"trace line {line_number} actuator {actuator_id!r} "
                    f"is missing {field!r}"
                )
            _finite_number(
                readings[field],
                f"trace line {line_number} actuator {actuator_id!r} {field}",
            )

    for field in (
        "shared_airflow_capacity",
        "total_requested_airflow",
        "total_actual_airflow",
        "capacity_scale",
    ):
        if field not in row["system"]:
            raise ValueError(
                f"trace line {line_number} system is missing {field!r}"
            )
        _finite_number(
            row["system"][field], f"trace line {line_number} system {field}"
        )


def write_visualisation(trace_path, output_path) -> Path:
    """Write a self-contained timeline report and return its path."""
    rows = load_trace(trace_path)
    payload = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    html = _HTML.replace("__TRACE_DATA__", encoded)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    trace_path, output_path = argv
    try:
        destination = write_visualisation(trace_path, output_path)
    except ValueError as exc:
        print(f"cannot visualise trace: {exc}", file=sys.stderr)
        return 2
    print(f"visualised trace={trace_path} report={destination}")
    return 0


_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICARUS Trace Visualiser</title>
  <style>
    :root { color-scheme: dark; --bg:#080d18; --panel:#111a2b; --line:#26344d;
      --text:#ecf3ff; --muted:#91a1ba; --accent:#63e6be; }
    * { box-sizing: border-box; }
    body { margin:0; background:radial-gradient(circle at top,#13233a 0,var(--bg) 42%);
      color:var(--text); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { width:min(1280px,calc(100% - 32px)); margin:0 auto; padding:40px 0 64px; }
    header { display:flex; justify-content:space-between; gap:24px; align-items:end; margin-bottom:24px; }
    h1 { margin:0; font-size:clamp(28px,5vw,48px); letter-spacing:-.04em; }
    header p { color:var(--muted); margin:6px 0 0; }
    .badge { color:var(--accent); border:1px solid #2f856f; border-radius:999px;
      padding:7px 12px; white-space:nowrap; }
    .summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:20px; }
    .metric,.chart { background:color-mix(in srgb,var(--panel) 92%,transparent);
      border:1px solid var(--line); border-radius:14px; box-shadow:0 14px 40px #0004; }
    .metric { padding:16px; }
    .metric span { color:var(--muted); display:block; font-size:12px; text-transform:uppercase;
      letter-spacing:.08em; }
    .metric strong { display:block; margin-top:5px; font-size:24px; }
    .charts { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .chart { padding:18px; min-width:0; position:relative; }
    .chart.wide { grid-column:1/-1; }
    h2 { font-size:16px; margin:0 0 4px; }
    .subtitle { color:var(--muted); margin:0 0 12px; font-size:12px; }
    svg { width:100%; height:auto; display:block; overflow:visible; }
    .grid { stroke:var(--line); stroke-width:1; }
    .axis-label { fill:var(--muted); font-size:11px; }
    .series { fill:none; stroke-width:2.2; vector-effect:non-scaling-stroke; }
    .legend { display:flex; flex-wrap:wrap; gap:8px 14px; margin-top:10px; color:var(--muted);
      font-size:12px; }
    .legend i { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:5px; }
    .hover-line { stroke:#dce8ff; stroke-width:1; stroke-dasharray:4 4; opacity:0; }
    .hit-area { fill:transparent; cursor:crosshair; }
    .tooltip { display:none; position:absolute; pointer-events:none; z-index:2; min-width:145px;
      background:#050914ed; border:1px solid #3a4a65; border-radius:9px; padding:9px 11px;
      box-shadow:0 8px 28px #0008; font-size:12px; }
    .tooltip strong { display:block; margin-bottom:4px; }
    .tooltip div { color:var(--muted); white-space:nowrap; }
    footer { color:var(--muted); margin-top:20px; text-align:center; font-size:12px; }
    @media (max-width:800px) { .summary { grid-template-columns:1fr 1fr; }
      .charts { grid-template-columns:1fr; } .chart.wide { grid-column:auto; }
      header { align-items:start; flex-direction:column; } }
  </style>
</head>
<body>
<main>
  <header><div><h1>ICARUS telemetry</h1><p>Deterministic habitat simulation trace</p></div>
    <div class="badge">Standalone replay</div></header>
  <section class="summary" id="summary"></section>
  <section class="charts">
    <article class="chart wide"><h2>Occupancy profile</h2><p class="subtitle">Scheduled multiplier applied to each zone's baseline source</p><div id="occupancy"></div></article>
    <article class="chart wide"><h2>CO₂ generated per tick</h2><p class="subtitle">Occupancy demand plus seeded, correlated variation</p><div id="source"></div></article>
    <article class="chart wide"><h2>CO₂ concentration</h2><p class="subtitle">Room sensor values used to calculate actuator demand</p><div id="sensor"></div></article>
    <article class="chart wide"><h2>CO₂ after shared processing</h2><p class="subtitle">Final zone concentration after mixed return airflow</p><div id="co2"></div></article>
    <article class="chart wide"><h2>Requested and allocated airflow</h2><p class="subtitle">Local demand competes for shared fan capacity</p><div id="airflow"></div></article>
    <article class="chart wide"><h2>Shared fan capacity</h2><p class="subtitle">Total demand, allocation and system limit</p><div id="capacity"></div></article>
    <article class="chart wide"><h2>Actuator setpoint and actual position</h2><p class="subtitle">Rate-limited movement exposes normal tracking delay</p><div id="position"></div></article>
    <article class="chart"><h2>Tracking residual</h2><p class="subtitle">Setpoint minus actual actuator position</p><div id="residual"></div></article>
    <article class="chart"><h2>Actuator power</h2><p class="subtitle">Abstract power used while moving or holding</p><div id="power"></div></article>
    <article class="chart"><h2>Connection health</h2><p class="subtitle">1.0 is fully healthy; 0.0 is unavailable</p><div id="health"></div></article>
    <article class="chart"><h2>Captured CO₂</h2><p class="subtitle">Cumulative amount retained by processing</p><div id="captured"></div></article>
  </section>
  <footer>Generated locally by ICARUS. Values are abstract simulation units.</footer>
</main>
<script>
const rows=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__TRACE_DATA__'),c=>c.charCodeAt(0))));
const colours=['#63e6be','#74c0fc','#ff8787','#ffd43b','#b197fc','#ffa94d','#66d9e8','#f783ac','#a9e34b','#ced4da'];
const ticks=rows.map(r=>Number(r.tick));
const zoneIds=[...new Set(rows.flatMap(r=>Object.keys(r.zones)))];
const connectionIds=[...new Set(rows.flatMap(r=>Object.keys(r.connections)))];
const actuatorIds=[...new Set(rows.flatMap(r=>Object.keys(r.actuators)))];
const fmt=n=>Number.isFinite(n)?Number(n).toLocaleString(undefined,{maximumFractionDigits:3}):'—';
const capturedEntries=zoneIds.filter(id=>rows.some(r=>r.zones[id]?.captured_co2!==undefined));
const finalCaptured=capturedEntries.reduce((sum,id)=>sum+Number(rows.at(-1).zones[id]?.captured_co2||0),0);
document.getElementById('summary').innerHTML=[['Ticks',rows.length],['Zones',zoneIds.length],
  ['Actuators',actuatorIds.length],['Connections',connectionIds.length],
  ['Shared capacity',fmt(rows[0].system.shared_airflow_capacity)],['Final captured CO₂',fmt(finalCaptured)]]
  .map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');

function seriesFor(ids, getter){return ids.map((id,i)=>({name:id,colour:colours[i%colours.length],values:rows.map(r=>getter(r,id))}));}
function renderChart(targetId,series,unit,range){
  const target=document.getElementById(targetId), W=1000,H=330,m={l:64,r:18,t:16,b:38};
  const values=series.flatMap(s=>s.values).filter(Number.isFinite);
  if(!values.length){target.textContent='No data in this trace.';return;}
  let ymin=range?.[0]??Math.min(...values), ymax=range?.[1]??Math.max(...values);
  if(ymin===ymax){const pad=Math.max(Math.abs(ymin)*.1,.5);ymin-=pad;ymax+=pad;}
  else if(!range){const pad=(ymax-ymin)*.08;ymin=Math.max(0,ymin-pad);ymax+=pad;}
  const x=i=>m.l+(rows.length===1?0:(i/(rows.length-1))*(W-m.l-m.r));
  const y=v=>m.t+(ymax-v)/(ymax-ymin)*(H-m.t-m.b);
  const ns='http://www.w3.org/2000/svg', svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('role','img');
  svg.setAttribute('aria-label',`${unit} timeline`);
  for(let i=0;i<=5;i++){
    const yy=m.t+i*(H-m.t-m.b)/5, val=ymax-i*(ymax-ymin)/5;
    const line=document.createElementNS(ns,'line'); line.setAttribute('class','grid');
    line.setAttribute('x1',m.l);line.setAttribute('x2',W-m.r);line.setAttribute('y1',yy);line.setAttribute('y2',yy);svg.append(line);
    const label=document.createElementNS(ns,'text');label.setAttribute('class','axis-label');label.setAttribute('x',m.l-8);
    label.setAttribute('y',yy+4);label.setAttribute('text-anchor','end');label.textContent=fmt(val);svg.append(label);
  }
  const xLabels=Math.min(6,rows.length);
  for(let i=0;i<xLabels;i++){
    const index=xLabels===1?0:Math.round(i*(rows.length-1)/(xLabels-1));
    const label=document.createElementNS(ns,'text');label.setAttribute('class','axis-label');label.setAttribute('x',x(index));
    label.setAttribute('y',H-12);label.setAttribute('text-anchor',i===0?'start':i===xLabels-1?'end':'middle');
    label.textContent=`tick ${ticks[index]}`;svg.append(label);
  }
  for(const s of series){
    const path=document.createElementNS(ns,'path');path.setAttribute('class','series');path.setAttribute('stroke',s.colour);
    path.setAttribute('d',s.values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' '));svg.append(path);
  }
  const hover=document.createElementNS(ns,'line');hover.setAttribute('class','hover-line');hover.setAttribute('y1',m.t);
  hover.setAttribute('y2',H-m.b);svg.append(hover);
  const hit=document.createElementNS(ns,'rect');hit.setAttribute('class','hit-area');hit.setAttribute('x',m.l);hit.setAttribute('y',m.t);
  hit.setAttribute('width',W-m.l-m.r);hit.setAttribute('height',H-m.t-m.b);svg.append(hit);target.append(svg);
  const legend=document.createElement('div');legend.className='legend';
  for(const s of series){const item=document.createElement('span'),dot=document.createElement('i');dot.style.background=s.colour;
    item.append(dot,document.createTextNode(s.name));legend.append(item);}target.append(legend);
  const tooltip=document.createElement('div');tooltip.className='tooltip';target.parentElement.append(tooltip);
  hit.addEventListener('mousemove',event=>{const rect=svg.getBoundingClientRect(),px=(event.clientX-rect.left)/rect.width*W;
    const index=Math.max(0,Math.min(rows.length-1,Math.round((px-m.l)/(W-m.l-m.r)*(rows.length-1))));
    hover.setAttribute('x1',x(index));hover.setAttribute('x2',x(index));hover.style.opacity=1;
    tooltip.replaceChildren();const heading=document.createElement('strong');heading.textContent=`Tick ${ticks[index]}`;tooltip.append(heading);
    for(const s of series){const value=document.createElement('div');value.textContent=`${s.name}: ${fmt(s.values[index])}`;tooltip.append(value);}
    tooltip.style.display='block';tooltip.style.left=`${Math.min(event.offsetX+14,target.parentElement.clientWidth-180)}px`;
    tooltip.style.top=`${Math.max(45,event.offsetY-10)}px`;});
  hit.addEventListener('mouseleave',()=>{hover.style.opacity=0;tooltip.style.display='none';});
}
renderChart('occupancy',seriesFor(zoneIds,(r,id)=>Number(r.zones[id]?.occupancy_multiplier??NaN)),'occupancy multiplier');
renderChart('source',seriesFor(zoneIds,(r,id)=>Number(r.zones[id]?.source_co2_mass??NaN)),'generated CO₂ mass');
renderChart('sensor',seriesFor(zoneIds,(r,id)=>Number(r.zones[id]?.sensor_co2_concentration??NaN)),'CO₂ concentration');
renderChart('co2',seriesFor(zoneIds,(r,id)=>Number(r.zones[id]?.co2_concentration??NaN)),'processed CO₂ concentration');
const airflowSeries=connectionIds.flatMap((id,i)=>[
  {name:`${id} requested`,colour:colours[(i*2)%colours.length],values:rows.map(r=>Number(r.connections[id]?.requested_airflow??NaN))},
  {name:`${id} allocated`,colour:colours[(i*2+1)%colours.length],values:rows.map(r=>Number(r.connections[id]?.airflow??NaN))}
]);
renderChart('airflow',airflowSeries,'airflow');
renderChart('capacity',[
  {name:'requested',colour:colours[0],values:rows.map(r=>Number(r.system.total_requested_airflow))},
  {name:'allocated',colour:colours[1],values:rows.map(r=>Number(r.system.total_actual_airflow))},
  {name:'capacity',colour:colours[2],values:rows.map(r=>Number(r.system.shared_airflow_capacity))}
],'shared airflow');
const positionSeries=actuatorIds.flatMap((id,i)=>[
  {name:`${id} setpoint`,colour:colours[(i*2)%colours.length],values:rows.map(r=>Number(r.actuators[id]?.setpoint??NaN))},
  {name:`${id} actual`,colour:colours[(i*2+1)%colours.length],values:rows.map(r=>Number(r.actuators[id]?.actual_position??NaN))}
]);
renderChart('position',positionSeries,'normalised position',[0,1]);
renderChart('residual',seriesFor(actuatorIds,(r,id)=>Number(r.actuators[id]?.tracking_residual??NaN)),'tracking residual');
renderChart('power',seriesFor(actuatorIds,(r,id)=>Number(r.actuators[id]?.power??NaN)),'power');
renderChart('health',seriesFor(connectionIds,(r,id)=>Number(r.connections[id]?.health??NaN)),'health',[0,1]);
renderChart('captured',seriesFor(capturedEntries,(r,id)=>Number(r.zones[id]?.captured_co2??0)),'captured CO₂');
</script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
