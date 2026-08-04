# V4 development outcome

Date: 2026-08-03

Evidence role: `development_only`

Decision: retain `rule_baseline`. No learned candidate passed the frozen development gate. Do not connect the bounded response layer and do not treat any learned artifact as deployment-selected.

## Frozen evidence contract

- Canonical specification SHA-256: `5a631af5e646535b2d35bebe726fd89b36f99feba4c558b29a7afe98bfe309ea`
- Generated family-manifest SHA-256: `4039e5dcbf1174f850d83cd48f39dbd4d071ea80b5c0bbea0c50faec1f5d018a`
- Generated corpus-manifest SHA-256: `c4ba6b7b7baa032e5ecce95733968c7cd1b9e04c5685bc7cb16e10176695b7eb`
- Fit seeds: `700–703` (`240` families)
- Train-internal calibration seeds: `704–705` (`120` families)
- Single-use development-validation seeds: `900–905` (`360` families)
- Rolling calibration rows: `13,764`
- Rolling validation rows: `41,292`
- Shared healthy references: deduplicated by canonical trace SHA-256
- Causal stride: one simulator tick

The gate required a learned candidate to beat calibrated rules in seed-cluster mean macro-F1 while meeting all of these safety conditions:

- no more than `10.0` false-alert episodes per 1,000 eligible healthy ticks;
- no more than `2.0` episodes per 1,000 above rules;
- nominal false-alarm-rate regression no greater than `0.01`;
- no fault-class recall regression greater than `0.02`;
- ONNX parity within the declared tolerance;
- ONNX operators contained by the declared allowlist;
- strict finite JSON and non-empty artifact checks;
- independent byte-for-byte reproduction before any integration authorization.

## Model-versus-rules table

| Method | Weighting / policy | Threshold | Persistence | Cluster macro-F1 | Window macro-F1 | Nominal FAR | False-alert episodes / 1,000 | Blocked recall | Frozen recall | Degradation recall | Median latency (ticks) | Parameters | ONNX bytes | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Calibrated rules | frozen grid-selected rules | — | 3 ticks | 0.604553 | 0.605062 | 11.1165% | 62.312 | 0.896798 | 0.078834 | 0.576670 | 4 | — | — | comparator; retained fallback |
| Temporal MLP, raw | balanced | — | — | 0.683436 | 0.683544 | 25.1673% | 102.102 | 0.700231 | 0.803208 | 0.552277 | 9 | 2,244 | 14,009 | fail |
| Temporal MLP, gated | balanced | 0.5 | 1 tick | 0.676443 | 0.676450 | 20.9512% | 95.345 | 0.690201 | 0.781690 | 0.501820 | 9 | 2,244 | 14,009 | fail |
| Causal TCN, balanced | equal class totals | 0.5 | 1 tick | **0.704253** | **0.703932** | 23.6276% | 121.622 | 0.785397 | 0.748826 | 0.570178 | 8 | 416 | 3,570 | fail |
| Causal TCN, square-root | square-root inverse frequency | 0.5 | 1 tick | 0.620431 | 0.620602 | **7.5695%** | 75.075 | 0.590471 | 0.493153 | 0.435133 | 11 | 416 | 3,574 | fail |

The balanced TCN is the quality-ranked learned candidate, but it is ineligible. It produced `162` false-alert episodes across `1,332` eligible healthy ticks, alerted on all `12/12` healthy streams and all `6/6` validation seed clusters, exceeded the absolute episode ceiling by more than twelvefold, regressed nominal FAR by `0.125111`, and regressed blocked-path recall by `0.111400` relative to rules.

The square-root weighting reduced nominal FAR and episode burden but still alerted on every healthy stream and seed cluster. In this run, the lower alarm burden coincided with lower recall for every fault class and still failed the absolute episode gate.

The v4 rules themselves also exceed the aspirational absolute episode ceiling (`62.312/1,000`). They are retained because no learned candidate passed the predeclared replacement gate, not because v4 established that rules are safe enough for physical deployment.

## Failure classification

Overall working diagnosis: **mixture**. Within this deterministic run, the results are most consistent with training/objective trade-offs plus incomplete nominal-versus-fault separation. Calibration is secondary; the TCN/raw-window bundle improves quality but does not solve safety; insufficient observability is not established as the sole cause.

### Training and weighting: material contributor

Holding the TCN architecture and gate fixed, square-root inverse-frequency weighting changed:

- window macro-F1 by `-0.083330`;
- nominal FAR by `-0.160580`;
- false-alert episodes by `-46.547` per 1,000 eligible ticks.

This is a large controlled movement for one deterministic initialization and six validation seed clusters. Equal-total weighting coincided with higher minority-fault recall and more nominal alarms; restoring nominal weight lowered alarms but reduced every fault recall enough to fail the gate. Weighting controls the observed trade-off but does not expose a safe operating point in this run.

### Calibration and decision policy: secondary, not the root fix

Holding the MLP weights fixed, the train-internal calibrated gate changed:

- window macro-F1 by `-0.007094`;
- nominal FAR by `-0.042160`;
- false-alert episodes by `-6.757` per 1,000 eligible ticks.

The gate provides a modest alarm reduction but remains far outside the safety ceiling. All frozen gate searches selected threshold `0.5`, persistence `1`; under the predeclared ranking, stricter settings lost more fault quality than their alarm reduction justified. Each learned model evaluated `20` train-internal threshold/persistence settings, and zero settings were safety-eligible. The minimum-episode settings demonstrate the trade-off:

- balanced TCN at threshold `0.9`, persistence `5`: `6.757` episodes per 1,000, but cluster macro-F1 `0.354863`;
- square-root TCN at threshold `0.6`, persistence `5`: zero episodes, but cluster macro-F1 `0.442469`;
- MLP at threshold `0.9`, persistence `5`: `2.252` episodes per 1,000, but cluster macro-F1 `0.328240`.

V3 used raw argmax; the v4 MLP ablation shows that analogous post-processing alone did not repair the v4 learned boundary.

### Architecture/representation bundle: useful but insufficient

Holding balanced weighting and operational gating fixed, the TCN/raw-window bundle improved window macro-F1 over the gated temporal-summary MLP by `0.027482`, but nominal FAR increased by `0.026763` and alert episodes increased from `95.345` to `121.622` per 1,000. Architecture and representation changed together, so this experiment cannot attribute the gain to either one independently.

### Observability: class-dependent limitation, not sole explanation

The calibrated rules obtain `0.896798` blocked-path recall, while the balanced TCN obtains `0.748826` frozen-sensor recall. Within this generated development corpus, those complementary results indicate that the supplied telemetry contains usable signal for both fault types. They do not establish global observability.

However, all candidates and rules alert on every healthy validation stream, and the square-root TCN's seed-level FAR ranges from `1.0184%` to `15.1691%`. These results are consistent with seed-dependent overlap or model instability, but this experiment does not distinguish representation, distribution, and observability causes. The evaluation contains six validation seed clusters and 12 deduplicated healthy streams; its 360 family IDs and 41,292 rolling rows are not independent samples.

## Historical v3 forensic result

Evidence role: `historical_forensic_only`. This report was regenerated from the frozen v3 final manifest and detector and was not consumed by any v4 fit, calibration, ranking, or gate.

- Final manifest SHA-256: `474704de6c6c5930fe4825e0c5238b5f04b3e6eadee487fcfe4fdd032b5d7112`
- Detector SHA-256: `c3b416c77e8b63eca558166cb02ac522af950495e2e8b51837cf2678b1c34344`
- Scored rows: `8,000`
- Excluded transition rows: `280`
- Total MLP errors: `2,896`
- Nominal false alarms: `2,055`

Of the `2,055` v3 nominal false alarms:

- `1,482` (`72.1168%`) were predicted as frozen sensor;
- `573` (`27.8832%`) were predicted as gradual degradation.

The largest concentration was nominal → frozen sensor in the `primary-high` profile (`1,083` windows). The next was nominal → gradual degradation in `primary-low` (`504` windows). This supports a systematic decision-boundary error pattern rather than random noise; weighting as the v3 cause remains inferential because the historical report contains no v3 weighting ablation.

## Local optimisation receipt

Evidence role: `local_readiness_only`.

Host/runtime boundary:

- Linux `6.8.0-136-generic`, `x86_64`;
- Python `3.13.13`;
- ONNX Runtime `CPUExecutionProvider`;
- batch size `1`;
- ONNX Runtime intra-op threads `1`;
- `100` warm-up and `1,000` measured iterations;
- `256` deterministic benchmark windows;
- input selection: first 256 canonical validation rows;
- corpus manifest: `c4ba6b7b7baa032e5ecce95733968c7cd1b9e04c5685bc7cb16e10176695b7eb`;
- complete benchmark-input SHA-256: `c2a0fb7cec4a85081e9e34fad688401398279f4701e3cc943a9fec9ab20eec4b`;
- process-lifetime maximum RSS only, not isolated method memory.

| Method entry point | Median ms | p95 ms | Windows/s |
|---|---:|---:|---:|
| Balanced TCN ONNX kernel | 0.007707 | 0.007822 | 129,752.17 |
| Square-root TCN ONNX kernel | 0.007772 | 0.007892 | 128,667.01 |
| MLP ONNX kernel | 0.019620 | 0.020002 | 50,968.40 |
| Calibrated Python rules | 0.053866 | 0.059792 | 18,564.59 |

This comparison is not an algorithm-only speed comparison: ONNX uses compiled native kernels, rules execute Python, and the learned entries exclude application-side alert-state bookkeeping. It establishes that the artifacts run locally at low batch-1 kernel latency. It does **not** establish Arm performance, NEON use, INT8 acceleration, isolated memory, energy efficiency, production latency, or deployment suitability.

No INT8 artifact was generated because no learned candidate passed the quality/safety gate. Quantising a rejected model would optimise the wrong thing with admirable efficiency.

## Raw local receipts

Authoritative generated artifacts remain ignored working outputs rather than committed corpora:

- `out/v4-development-canonical-2026-08-03-reviewed-a/v4-development-report.json`
  - schema: `aeolus_v4_development_evidence_v2`
  - SHA-256: `40d641e8a6ae54d9e6940e980171a45a54de1ad61d4fa2b0aa2f6f2a637f1aed`
  - bytes: `351,563`
- `out/v4-development-canonical-2026-08-03-reviewed-a/historical-v3-forensic-error-report.json`
  - SHA-256: `6686deab2ecc1b1f5d2de8156f4381d5a0fedd34df5a88c13c225e4e7a59cb7a`
  - bytes: `179,105`
- `out/v4-development-canonical-2026-08-03-reviewed-a/local-optimisation-receipt.json`
  - schema: `aeolus_local_optimisation_comparison_v2`
  - SHA-256: `f9a035e0db8b86e277404ae2cadd0045520814f9b5ade67ec97bc031cad62230`
  - bytes: `5,711`
- `docs/evidence/v4-reproduction-verification.json`
  - SHA-256: `cd5677445d3234a6534e5d143d24cdbf539b707002ebe0d9ab51857b98ea15c0`
  - bytes: `3,179`

The corrected canonical run is bound to source-manifest SHA-256 `7674c62be16b5ebbafbecb41595de75caab6c74227d338cde809d323783670a2`.

## Independent recomputation

Corrected runs A and B both exited `0`. Byte comparison succeeded for the sweep receipt, family manifest, corpus manifest, complete corpus JSONL, all three model JSON files, all three ONNX files, and the complete v2 development report. The committed reproduction receipt lists the hash and byte size of every compared file.

Both reports have SHA-256:

```text
40d641e8a6ae54d9e6940e980171a45a54de1ad61d4fa2b0aa2f6f2a637f1aed
```

Exact runner template, executed once with each fresh output path:

```bash
uv run --extra dev python -c "from aeolus.model_cycle_v4 import run_v4_development; run_v4_development('scenarios/sweep-v4-development.json',OUTPUT_PATH,mlp_epochs=300,cnn_epochs=300)"
```

The development runner records `independent_reproduction_verified=false` because a run cannot certify its own independent reproduction. `docs/evidence/v4-reproduction-verification.json` closes that external freeze condition. It does not convert any rejected learned candidate into an accepted one: every candidate independently fails the predictive quality/safety gate, `selected_candidate` is null, `retained_method` is `rule_baseline`, and both integration authorization flags remain false.

Final repository verification:

```text
348 passed in 57.05s
Ruff: all checks passed
uv lock --check: resolved 21 packages
git diff --check: clean
gitleaks: 597.54 MB scanned; no leaks found
```
