# Historical closed-loop advisory evidence index

This index records the 2026-08-18/19 single-model and ensemble advisory
campaigns that were developed on the former Habitat V2 forecast stack. The
campaign is **historical development evidence only**. It is not qualification,
deployment evidence, a hardware claim, or a current-`main` runnable feature.
The deterministic HMC remained the sole actuator authority in the recorded
campaign, and it remains the sole actuator authority on current `main`.

## Immutable source identities

- Base campaign lineage: commit
  [`c053926`](https://github.com/arm-hackathon/arm-hackathon/commit/c0539263c220e5b82eddc93a725e616beb57af80).
- Ensemble V3 addition: commit
  [`c9e7d33`](https://github.com/arm-hackathon/arm-hackathon/commit/c9e7d3337a1301f4d96be5211317974c1f0b0b5a)
  and PR [#50](https://github.com/arm-hackathon/arm-hackathon/pull/50).
- Complete 22-file historical subtree after formatting-only repairs: commit
  [`9aea769`](https://github.com/arm-hackathon/arm-hackathon/commit/9aea7691de9c31d7fe9eebcfc3d3dc1c23601597)
  and PR [#59](https://github.com/arm-hackathon/arm-hackathon/pull/59).
- Browse the exact
  [historical subtree](https://github.com/arm-hackathon/arm-hackathon/tree/9aea7691de9c31d7fe9eebcfc3d3dc1c23601597/experiments/closed-loop-advisory-20260818).

PR #59 makes the old stacked branch pass its repository CI. That CI result is
not evidence that the experiment runs: pytest and the wheel exclude
`experiments/`, compileall did not include it, and no ensemble test existed.

## Recorded findings and static audit

The checked historical records are internally coherent:

- the compact V1/V2 summary contains 238 completed runs (119 control and 119
  advised) and recomputes to 78 better, 24 equal, and 0 worse across the 102
  fault pairs used for the headline comparison;
- the full V3 file contains 119 unique completed 72-step ensemble runs;
- all 119 V3 recorded outcomes match their paired V2 single-model outcomes;
- proposal/override totals reconcile to 793/81 for V2 and 1,248/138 for V3;
  and
- scenario, trace, and final-state identities have valid 64-character hash
  forms.

This is a static consistency audit, not a campaign rerun. In particular, the
reported ensemble normalized MAE of `0.1049` cannot be independently checked
from the archive because the cited `ensemble_eval.json` is absent.

## Exact archived-file manifest

The table gives the SHA-256 of the exact bytes in the `9aea769` historical
subtree. The 22 ordinary Git blobs total 46,497,672 bytes; the five Torch
checkpoints account for 42,276,020 bytes.

| Relative path | Bytes | SHA-256 |
| --- | ---: | --- |
| `action-aware-mlp-v1.pt` | 8,455,204 | `873cb77bb82a06b4c862a13275b55133c3ef26c969d3055a799c80dcd98854a6` |
| `aeolus_closed_loop.py` | 17,120 | `58e91e9975c70e2e7d0bd7fdfe37467adf212ae1547b78f3b30ecd0c04b6ede3` |
| `CLOSED_LOOP_REPORT.md` | 5,075 | `931321feeaddf0ae080fef6768df01c1d9c36cda760051c1621859d833be0396` |
| `CLOSED_LOOP_REPORT_V2.md` | 3,114 | `2b8be43998f40997970b929fd06c5fdce58c573f613621d72da66d9447484bc2` |
| `CLOSED_LOOP_REPORT_V3.md` | 4,461 | `fcab0e64b5b6295d0410c9c7593783b1ff3fdcb21d1c330dbb29f6efa699747e` |
| `ensemble/seed-20260819.pt` | 8,455,204 | `9bdfd27fe5f96f8f161d8619c014e84b7e60cd797ed266badd285b7e3bd8a45b` |
| `ensemble/seed-20260820.pt` | 8,455,204 | `303c32ed1dba3ecf6d02b93243279528d357450dfb3d95e041798d42e63d5e12` |
| `ensemble/seed-20260821.pt` | 8,455,204 | `618dfdcac35c8d3562ea878665e210efd9a4dd5a38198cb3a5e96289590ac2f9` |
| `ensemble/seed-20260822.pt` | 8,455,204 | `5049473489f7c6f3d24c2cdda75fac173185883253c3b1f7c6b7fc8122cebbf6` |
| `ensemble_adviser.py` | 3,317 | `03ecf16761c7c2fde5192eaf73f5a810dfae641c6dc1d793466b44efa0f676f1` |
| `held-out-clusters.json` | 1,149 | `ae8f8ad1f5722bacc4c0330e8e33a2efdaf4b085e41025d3ac7251de3b26a887` |
| `paired-v3-ensemble-results.json` | 3,957,440 | `94380cb15a4d9832bf4c385954de2dcbb79bedfc54faa4e56acdf72e042bd79b` |
| `preregistration.json` | 2,802 | `619a07c9b0efeee4328b2aeddb93e912a6ac433ce30e967c0d7c1c9825398187` |
| `preregistration-v2.json` | 2,392 | `a5e4ba5a989927372a29c61e053e9041549b6d0d96e089e91a8df7784f54f4a2` |
| `preregistration-v3.json` | 3,259 | `bf686d9801a5a3634063c717616d6b3386355561612126a3d871269b9af386b9` |
| `README.md` | 2,191 | `57ddc2d4670de74c6531507e932e00d1c5635250632595e9ec3290a00f4917c9` |
| `results-summary.json` | 200,336 | `cc689638064f7ec9b487bf8771f0c7a5719a57b19897368cabb16a8abcef1ed6` |
| `run_demo.py` | 5,950 | `2f99d2b1c619ec64c70b1e704df4a9270edc258d175551d8d0b855eda2c79749` |
| `run_ensemble_paired.py` | 4,531 | `e90aed6828b02652c9e232d4ca744adfa367d97d8a91eeea2f8cf2ae52b74348` |
| `run_paired.py` | 3,707 | `88f5634c552f00592445cc10eda3b7d172486f2ce2470e349d48146658cba850` |
| `smoke-v3.json` | 137 | `bb51e888aff800a3710f6e6ae330cbeb419680f9d9114bbb4196a4ec01b49def` |
| `SYSTEM.md` | 4,671 | `744c6b94beb892ec829b9928e4f67bdc04d81368b30260598487a8f4bc7806bd` |

The V1 and V2 preregistrations also carry canonical self-hashes
`a0c0839da3539aaf6108648060d5a0f0cba1ed0d9506b71d33bdca1a168e265c`
and `0b7e3e7be5413d0093ff51d6065b314adbcaffeb25442107b384e9e4f7795712`,
respectively. These are canonical-record identities, not the raw file hashes
shown in the table.

## Known custody and reproduction gaps

The archive is not self-contained:

- `paired-v1-results.json` and `paired-v2-results.json` are absent. Their
  recorded expected SHA-256 values are
  `6140d1f0790b4555aa709e2d00a11755ac78edb6ccd73e820a1a30aaff65f7b7`
  and `8378e7f6feadab3e47f8815d54c9b97366074a8e7dcf2acf1eba44c461b5c456`.
- `smoke-control-determinism.json`, `ensemble_eval.json`, the training report,
  `artifact_hashes.json`, the training code/archive, raw control traces, and an
  execution-environment receipt are absent.
- the V1/V2 preregistrations and their results first entered Git together in
  `c053926`; the V3 preregistration and result entered together in `c9e7d33`.
  Git history therefore does not independently establish pre-outcome freezing.
- `run_ensemble_paired.py` contains a contributor-local absolute path to the
  missing V2 result, and Torch was neither declared nor locked.
- the historical runners depend on the removed private
  `aeolus.habitat_v2.forecast.pilot` module and on old reviewed-HMC/scenario
  byte identities. Current `main` deliberately uses a different HMC binding
  and implementation, so old outcomes must not be relabelled as current-main
  evidence.

Current `main` carries the dependency-free base NumPy model and its supported
bounded demo. It does not carry or support the five-member historical ensemble.
A runnable current-main ensemble would be a new study requiring a separately
approved preregistration, current contract/HMC bindings, portable inputs, a
pinned runtime, direct tests, and newly generated development results. The old
campaign commands must not be run against current `main` as a shortcut.
