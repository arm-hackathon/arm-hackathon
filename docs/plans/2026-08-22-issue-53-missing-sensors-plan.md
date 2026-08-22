# Issue #53 Full Implementation, Training, and Qualification Plan

## 1. Document control

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/53 — *Teach the model to work with missing or broken sensors*
- Design date: 2026-08-22
- Branch for this plan: `design/issue-53-missing-sensors` (created from `design/issue-52-long-horizon-actions@261f50f`)
- Stacking chain: `ben/habitat-v2-hmc-v1@843a5c1` ← `design/issue-52-long-horizon-actions@261f50f` (PR #60 ready, `CI` + `Running Copilot Code Review` green at `32536162704`/`32534915249`) ← this lane
- Required approver when Ben returns: Ben (`bbeennyy860-cyber`) — but **Ben is offline this cycle, so this plan is published without an approval gate and must not claim `BEN_SIGN_OFF=true`**
- Short note: `docs/plans/2026-08-22-issue-53-missing-sensors-design.md`
- Machine-readable preregistration: `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json` (new; byte-frozen on plan publish)
- Preregistration SHA-256: `6DFA3E084F1585FB696511C54AB676356406496DDF7A639C6D10721A0D3F41B3`
- This-lane status: `IMPLEMENTATION_AUDITED_NOT_QUALIFIED_BEN_OFFLINE`
- `BEN_SIGN_OFF=false`
- `CODE_AUTHORISED=false` (the implementation and audit evidence on this branch do not authorize corpus generation, training, experiments, or deployment; see §20)
- `DATA_GENERATION_AUTHORISED=false`
- `TRAINING_AUTHORISED=false`
- `EXPERIMENTS_AUTHORISED=false`
- `DEPLOYMENT_AUTHORISED=false`
- `METRIC_AMENDMENT_APPROVED=false`

This appendix is normative. The short note is the plain-English entry point. The original plan publication was planning work only. Implementation later landed on this lane without opening implementation, corpus-generation, training, experiment, or deployment authorization. The current qualification runbook is in `docs/evidence/issue-53-dropout-card.md` and `docs/evidence/issue-53-measurements.md`.

## 2. Verified baseline and scope correction

The audited Habitat V2 code post-Issue #52 contains a bounded 16-observation, 32-step, 12-candidate forecast-and-rank lane (`src/aeolus/habitat_v2/forecast_issue52.py:1`, `CandidateCatalogue:1080`, `ForecastHistory:845`). Its current and correct safety posture is to **abstain on any incomplete latest observation**:

- `ForecastHistory` stores `available_mask: np.ndarray` at `src/aeolus/habitat_v2/forecast_issue52.py:576` and freezes it at `622`.
- `PersistenceForecaster` at `src/aeolus/habitat_v2/forecast_issue52.py:1552` and `ActionConditionedLinearForecaster` at `1414` gate with `if not np.all(history.available_mask[-1]): raise ABSTAIN`.
- `ObservationRecord.from_snapshot` at `630` and `target_from_snapshot` at `480` already surface `ChannelSample` availability from `src/aeolus/habitat_v2/instrumentation.py:79` (`AVAILABLE` vs `UNAVAILABLE` with `MISSING|NON_FINITE|MALFORMED|DEPENDENCY_UNAVAILABLE`).

That is safe — HMC at `src/aeolus/habitat_v2/hmc.py:770` still decides `ACCEPT / POLICY_MODIFY / EMERGENCY_OVERRIDE / REJECT_TO_HOLD` — but Issue #53's complaint is accurate: *one broken sensor switches the learned lane off completely*.

Issue #53 is therefore **not a constant change and not a bug-fix**. It is a bounded dropout-robust lane that reuses the Issue #52 contracts verbatim and adds one orthogonal dimension: observation dropout.

**Frozen evidence rule:** The Issue #52 model, its `contracts/habitat_v2_forecast_issue_52_preregistration_v1.json` (`E0A24B2F...1ADA7E61F`), its rolled-out traces, and its `RolloutCheckpoint` digests at `src/aeolus/habitat_v2/forecast_issue52_rollout.py:418` remain byte-identical. This lane adds *new* manifests, *new* datasets, and a *new* artifact with `parent_artifact_sha256` binding the frozen predecessor. The new model is *development evidence only*; the rule-based HMC controller remains in charge of every action, always.

## 3. Goals and non-goals

Goals (exact acceptance language from #53):

1.  Produce a **new training dataset** where sensors randomly drop out, via a deterministic background collection of about **33 hours** wall clock (quiet, isolated runner, no interactive training).
2.  Train a **new model** that keeps forecasting when sensors are missing and emits calibrated uncertainty, abstaining only *when it should*.
3.  Publish **honest per-k measurements**: degradation at **1 sensor missing** and **3 sensors missing** (plus full sweep `0…6`), and a calibrated answer to *when it correctly gives up*.
4.  Publish a **written record of what it cannot do** alongside what it can (capability/limitation card, rollback).

Non-goals (blocked this lane):

1.  No HMC authority change, no new emergency template, no capability minting by learned code.
2.  No online or production reinforcement learning.
3.  No tuning on the sealed FINAL split.
4.  No safety-limit, reserve-limit, emergency-threshold, or `HMCContract` change.
5.  No inter-process model service in V1.
6.  No deployment or “closed-issue” claim through this plan.
7.  No mutation of the frozen Issue #52 lane.

## 4. Exact causal and temporal contract with dropout

Reuse Issue #52 §4 verbatim and add one dropout clause:

Let the latest completed, HMC-issued and runtime-verified state be `S_s` where `snapshot.completed_step == s`.

- History is `S_{s-15} … S_s` (16 rows, `CADENCE_SECONDS=60.0` at `src/aeolus/habitat_v2/forecast_issue52.py:51`).
- Candidate schedule element `k` is `A_{s+k}` for `k in [0,31]`; only `A_s` may be proposed (`requested_application_step=s`).
- Dropout is **observation-only**: the plant truth and `fault_receipt` truth sequence `truth_telemetry` at `docs/plans/2026-08-12-habitat-v2-fault-sensor-contract.md:184` are never masked. Dropout masks `ChannelSample` *after* `instrument_v5_operational_measurement` (`src/aeolus/habitat_v2/instrumentation.py:724`) and before `ObservationRecord` (`forecast_issue52.py:557`). History record at `S_{s-t}` sees only dropouts with `step ≤ s-t`; future availability at `S_{s+k+1}` is never revealed to history. Target labels at `S_{s+k+1}` remain *always* available via primary truth projection — missing *inputs* never create missing *labels* for metric aggregation.

## 5. Runtime history contract with mask

Reuse Issue #52 §5 (`VerifiedHistoryBuffer` at `forecast_issue52.py:721`) with one extension:

The buffer continues to verify each snapshot via `hmc.verify_snapshot` (`forecast_issue52.py:767`), freeze the control-chain identity (`current_control_chain_sha256` at `forecast_issue52.py:773`), and clear on identity change/gap/reversal (`forecast_issue52.py:798`). It now stores, per record:

- `target_values: float32[W]` where `W = 3*zone_count+3` (`TargetManifest.width` at `forecast_issue52.py:441`, V5 width 9),
- `available_mask: bool[W]` (already at `557`/`576`), which **is** the dropout mask for this lane.

Runtime still projects via `target_from_snapshot` at `480`: `AVAILABLE` → finite `value`, `UNAVAILABLE` → `NaN` + `available_mask=false` (`474`). No change to HMC-owned `SensorMemory` at `src/aeolus/habitat_v2/hmc.py:1320`.

## 6. Observation and target schemas with missingness

### 6.1 History input

The Issue #52 history manifest (`TargetManifest.from_scenario` at `368`) and its derivation from `derive_observable_topology` at `src/aeolus/habitat_v2/telemetry.py:65` are reused unchanged. The new forecaster consumes `float32[16,W] + bool[16,W]`.

New feature construction (`_feature_matrix_masked` in the new `src/aeolus/habitat_v2/forecast_issue53_dropout.py`, mirroring `_feature_matrix` at `forecast_issue52.py:1394`):

1.  Impute NaNs deterministically: forward-fill per channel from last-available row; if never available, fill `TargetDescriptor.nominal` at `269`.
2.  Append mask channels: `bool[W]` as `float32` plus `time_since_observed: float32[W]` (steps since last `AVAILABLE`, capped at 16).
3.  Result width `F_history_dropout = W (imputed values) + W (mask) + W (age) + W (slope, mask-aware)` — frozen in `TargetManifest`-derived `observation_manifest` with its own digest. Unknown fields, duplicate descriptors, or digest mismatch fail closed as before (`forecast_issue52.py:296`).

### 6.2 Forecast target

Identical to Issue #52 §6.2: per ordered habitable zone `co2_ppm/temperature_k/relative_humidity` from `primary_telemetry`, plus three global gauges `battery_state_of_charge/oxygen_store_fraction/sorbent_remaining_fraction` from `operational_resource_gauges` at `forecast_issue52.py:296`. Secondary telemetry and `primary_minus_secondary` remain diagnostics only. Every horizon’s target row must be fully `AVAILABLE`; dropout never makes labels unavailable.

## 7. Dropout corpus generation (≈33h background)

### 7.1 Approach

Reuse the offline kernel at `forecast_issue52_rollout.py:160` (`RolloutCheckpoint`/`RolloutResult`/`build_offline_checkpoint`), which cannot mint HMC handles. Two complementary modes:

**A — Mask-derived views (cheap, runs without simulator):** Take every *complete* decision group already valid under Issue #52 §10 and apply a deterministic mask sampler `mask = SHA256(seed||family_id||decision_step||step_offset||descriptor_id) < p`. No plant re-execution; truth/hidden truth at `forecast_issue52_rollout.py:219` untouched. This covers the sweep efficiently.

**B — Native instrumented replay (validation subset):** Re-run `instrument_v5_operational_measurement` with injected dropout masks and validate that mode A’s NaN/mask matches native `ChannelSample` availability for a sampled subset. Byte-identical replay required.

### 7.2 Dropout config (frozen before any collection)

A canonical JSON `dropout_config` with `schema_version: aeolus_habitat_v2_dropout_v1`:

- `mode ∈ {independent, per_zone_head_burst, mixed}` — independent per-sample Bernoulli vs correlated burst per `(zone, sensor_head)`.
- `p_uniform: float in [0,1)` — base per-sample rate; `p=0.05` for pilot, plus evaluated points `p=0` (baseline) and stress points matching `k=1` and `k=3` marginal rates.
- `burst_length: {min: int, max: int}` and `p_burst_onset: float` for burst mode.
- `max_missing_per_row: int | null` — hard cap to keep evaluation tractable.
- `resource_gauge_dropout: bool` — whether the three global gauges may drop (default `false` — they are the always-available anchors; see §12).
- `seeds: tuple[int, ...]` — deterministic seeds distinct from Issue #52 `520032…`.
- `candidate: dropout_config_sha256 = SHA256(canonical_json_bytes(config))`.

The config digest binds every `TrainingSample`, `RolloutCheckpoint`, and `DatasetManifest`. Whole families never split across `TRAIN/VALIDATION/FINAL`.

### 7.3 Cost and cap

Same caps as Issue #52 §11: ≤384 families, ≤2,000,000 candidate transitions. The ~33h estimate assumes the current V5 per-transition cost observed in `build_offline_checkpoint` replay; pilot (§12) refines it and publication records actual `transition_time` and `storage` as for Issue #52. If the full dropout sweep exceeds the cap, stop and publish a negative pilot result (no threshold lowering after seeing data).

## 8. Candidate catalogue

Reuse the Issue #52 catalogue verbatim: `CandidateCatalogue.from_scenario` at `forecast_issue52.py:1088` with `CATALOGUE_SIZE=12`, `HORIZON_STEPS=32`, `candidate_hold` plus 11 bounded variants. Same `STATICALLY_VALID / ROLLOUT_FEASIBLE / RUNTIME_FIRST_STEP_FEASIBLE` separation (§7 → `forecast_issue52.py:1202`). Catalogue SHA-256 binding is unchanged.

## 9. Distinct runtime outcomes with per-k reporting

Reuse Issue #52 §8 outcomes (`OUTCOMES` at `forecast_issue52.py:104`: `SELECTED_HOLD, SELECTED_CANDIDATE, ABSTAINED, WARMUP_NO_PROPOSAL, INVALID_OUTPUT, TIMEOUT_NO_PROPOSAL, HMC_REJECTED_TO_HOLD, HMC_EMERGENCY_OVERRIDDEN, DISABLED`) and add **per-k accounted variants**: each outcome is counted separately for `k=0, k=1, k=3, k≥4`. Timeouts and invalid outputs never contribute to forecast accuracy metrics.

## 10. Offline kernel integration with dropout

`RolloutCheckpoint` at `forecast_issue52_rollout.py:259` gains one field `dropout_config_sha256` and includes it in `_checkpoint_digest` at `418` and `clone` at `347`. `RolloutResult` at `529` and `TrainingSample` at `792` similarly carry the dropout binding. `build_offline_checkpoint` still advances each candidate via `advance_one_step_with_command` at `660` and validates via `validate_external_step_result` at `663`; dropout never changes feasibility — a rollout that was `ROLLOUT_FEASIBLE` remains feasible, its *input view* is masked.

## 11. Pilot, power, coverage, and split

After plan publication (no Ben gate), run a deterministic pilot of ≤32 families / ≤12k transitions (same bound as Issue #52 §11) but evaluated at `k ∈ {0,1,3}`:

- Estimate per-family NMAE variance and paired correlations at each `k`.
- Measure interval coverage vs `k`.
- Estimate abstention precision/recall vs a fixed high-error oracle.
- Same power math as Issue #52: paired log-ratio power with `effect = -log(0.90)=0.105…`, `ddof=1`, zero-SD fallback to 30 FINAL families, identical SHA-256 family split (`SHA256("issue53-split-v1"||family_id)`, 70/15/15, largest-remainder). Family-disjoint invariant preserved.
- Coverage cells are the Issue #52 Cartesian product (4 modes × 2 fault-present flags) — dropout stress is reported *within* each cell, not as new split dimension.

Ranking power remains blocked until any ranking metric amendment (Issue #52 §14) is separately approved.

## 12. Baseline-first model development (mask-aware)

Implement before any learned dropout model:

1.  Frozen Issue #52 `Persistence` and `ActionConditionedLinear` applied with forward-fill imputation (dropout-agnostic baseline).
2.  Mask-aware linear continuation (slope computed on masked rows, `ForecastHistory.slope` at `899` → masked variant).
3.  Regularized linear action-conditioned model trained on mask-augmented features.
4.  Small temporal MLP only if linear fails VALIDATION gates.

All receive identical groups/inputs/targets/candidates/splits/metrics. Best non-neural baseline frozen on VALIDATION primary NMAE before learned FINAL evaluation. Observability and action-identifiability gates as before, now also evaluated at `k=3`.

## 13. Learned forecast model and training (dropout-aware)

Same interface as Issue #52 §13 plus mask:

- History `float32[16,W]` imputed + `bool[16,W]` mask (+ age) + candidate schedule `float32[32,F_action]` + optional known-future schedule → `float32[32,W]` mean + two interval bounds.
- Same autoregressive → direct → temporal candidate order; smallest passing VALIDATION wins.
- Normalizers fit on TRAIN `k=0` only.
- Seeds `530053,530054,530055`; weighted Huber; horizons 1–8 vs 9–32 reported separately; checkpoint selection on VALIDATION NMAE; intervals calibrated on VALIDATION via conformal residuals **binned by `k`** so the model learns to say *how unsure* it is when `k` rises. Thresholds frozen before FINAL.
- Artifact binds `observation_manifest_sha256`, `dropout_config_sha256`, `parent_artifact_sha256`, and all Issue #52 binding digests; mismatch fails loading.

## 14. Deterministic ranking

Unchanged from Issue #52 §14. Ranking remains deterministic code over predicted trajectories. Any change to true-score or predicted-score formulas is a separate commit-bound amendment requiring Ben’s approval when online.

## 15. Advisory integration and HMC authority (frozen)

Identical to Issue #52 §15. The dropout source remains feature-disabled by default, in-process, with no plant handle. HMC remains sole proposal/arbitration/preflight/capability/plant-step/replay authority (`src/aeolus/habitat_v2/hmc.py`). Any `hmc.py`/`physics.py`/`actuators.py` diff is isolated for Ben’s separate safety review when he returns. **The new model never drives actuation directly.**

## 16. Preregistered metrics and gates (per-k, frozen)

The `habitat_v2_forecast_issue_53_preregistration_v1.json` is authoritative. Summary — all gates are paired family-bootstrap (§16 of Issue #52: 10k resamples, SHA-256 counter sampler, type-7 quantiles, Holm within families):

- Primary forecast (held from Issue #52): NMAE horizons 9–32, learned/best-baseline at `k=0` ≤0.90 point & <0.98 upper; horizons 1–8 non-inferior.
- **Dropout degradation** (new, co-primary):
  * `k=1` NMAE vs own `k=0` ≤1.15 point, ≤1.25 upper.
  * `k=3` NMAE vs own `k=0` ≤1.40 point, ≤1.60 upper.
  * Full sweep `k=0…6` reported (no gate beyond `k=3` but non-monotonic degradation fails review).
- **Interval coverage** (new, co-primary): empirical 90% interval coverage at `k=1` ≥85%, at `k=3` ≥80%; monotone non-increasing with `k` beyond sampling noise.
- **Abstention quality** (new, co-primary): recall on oracle high-error decisions at `k=3` ≥0.80, precision ≥0.60, with PR curve reported; abstention rate at `k=0` must not exceed frozen baseline +2pp.
- **Safety non-regression** (hard gate, held): total safety-bound exposure mean diff ≤0 & upper ≤0, dangerous-crossing recall diff ≥−0.02, false-crossing diff ≤+0.01 — evaluated separately at each `k` and must hold at `k=3`.
- **Authority/replay/provenance/split-leakage** exactly zero at every `k` (hard gate, held).

Ratio edge cases frozen as Issue #52 §16.

## 17. Required tests

Extend Issue #52 §17 with dropout-specific contract tests:

- Dropout sampler determinism and byte-identical replay.
- No leakage: future mask not in history, truth `PlantState` byte-identical pre/post mask.
- Imputation is NaN-free, deterministic, and mask-augmented features are invariant under mask permutation.
- Calibration binned by `k` is monotone (higher `k` → non-narrower intervals).
- Per-k abstention monotonicity and per-k metric reporting.

Plus full existing Issue #52 suite must still pass.

## 18. Evidence, failure handling, and rollback

Every decision trace now records `dropout_config_sha256`, `dropout_mask_sha256`, and per-k interval/abstention evidence alongside the Issue #52 trace at §18. Rollback disables the dropout lane and its artifact reference, falling back to the frozen Issue #52 abstaining lane without weakening HMC. Traces remain.

## 19. Phased execution (Ben offline — no approval gate)

Current execution state: the plan, preregistration, implementation, and audit runbook exist; the lane remains `NOT QUALIFIED`, and all authorization flags remain false. The eight-step qualification checklist in `docs/evidence/issue-53-dropout-card.md` is the current gate order.

**Phase 0 — Contracts+plan:** freeze observation/dropout/dataset/artifact trace schemas; publish plan+preregistration as DRAFT.

**Phase 1 — Pilot (≤32 families):** estimate variance at `k=0/1/3`, tune abstention thresholds, freeze ranking/abstention formulas. Publish pilot report.

**Phase 2 — 33h corpus:** implement mask-derived + native validation corpus generation, deterministic validators, leakage checks, and publish `dataset_manifest_sha256`. Runs quietly on isolated runner; no training during collection.

**Phase 3 — Baselines+observability gate:** compare all §12 baselines, confirm action identifiability at `k=3`.

**Phase 4 — Model+calibration:** train in increasing complexity, select & calibrate on VALIDATION per-k.

**Phase 5 — One sealed FINAL:** compute honest per-k degradation and abstention PR; produce `docs/evidence/issue-53-dropout-card.md` (capability vs limitation) and `docs/evidence/issue-53-measurements.md` (1-vs-3 table).

**Phase 6 — Review when Ben returns:** present exact diff, artifacts, manifests, and the limitation card. No deployment.

## 20. Stop conditions

Stop and publish a negative result if: dataset exceeds 384-family/2M-transition caps, native vs mask-derived mismatch, monotone coverage violation, no gain over mask-aware linear baseline at `k=3`, confidence miscalibration persisting after binned conformal fix, or any safety/replay gate failure. Do not lower `1-vs-3` thresholds after seeing pilot data.

Precedence: authority, safety, leakage/provenance, replay/determinism, systems budget, forecast quality at `k=3`, then ranking utility.

## 21. Approval record (Ben offline)

No approval is sought this cycle. When Ben returns, record:

- Approver: `Ben (bbeennyy860-cyber)`
- Approval link / timestamp / approved commit: `PENDING`
- `BEN_SIGN_OFF=false` until filled, all `*_AUTHORISED` remain false until that commit-bound approval is recorded.
