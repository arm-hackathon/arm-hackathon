from pathlib import Path
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design, load_resource_preflight
from aeolus.habitat_v2.forecast.pilot_campaign import run_pilot_campaign
root=Path('.').resolve()
design=load_approved_pilot_design(root)
c=load_forecast_contracts(root)
p=load_resource_preflight(root/'out'/'helios-qual-v2'/'v2-resource-preflight-real.json',repo_root=root,expected_preflight_sha256='590bdca9a09bcfe5b527ea295d50f4fefc45dd65250e92eca57af5cc36b85420',expected_preflight_bytes_sha256='01c0e18141deca4a4e94124a08f60bf45b3a51b3ddacfb87a746ef1fc9e17147')
out=root/'out'/'helios-qual-v2'/'corpus-fit-cal-20260816'
print(run_pilot_campaign(root,design,c,preflight=p,output_root=out,pair_limit=3600,worker_count=1,resume=out.exists()))
