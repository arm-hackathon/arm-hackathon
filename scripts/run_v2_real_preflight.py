from __future__ import annotations
import shutil
from pathlib import Path
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast.pilot import iter_pilot_continuations, load_approved_pilot_design
from aeolus.habitat_v2.forecast.pilot_benchmark import V2_RESOURCE_CEILINGS, build_v2_preflight_receipt, measure_continuations, write_preflight_receipt
root=Path('.').resolve()
design=load_approved_pilot_design(root)
c=load_forecast_contracts(root)
it=iter_pilot_continuations(design)
control=next(it)
action=next(x for x in it if x.variant=='ACTION_PROPOSAL')
measured=measure_continuations(root, design, (control,action))
receipt=build_v2_preflight_receipt(root,design,c,measurements=measured,ceilings=V2_RESOURCE_CEILINGS,free_disk_bytes=shutil.disk_usage(root).free)
out=root/'out'/'helios-qual-v2'/'v2-resource-preflight-real.json'
out.parent.mkdir(parents=True,exist_ok=True)
write_preflight_receipt(out,receipt)
print(receipt)
