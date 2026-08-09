# AEOLUS Hackathon Working Model Product Requirements Document

**Status:** Active execution contract  
**Owner:** Benedict Anokye-Davies  
**Deadline:** 2026-08-09 09:00 BST  
**Execution stop:** 2026-08-09 08:30 BST  
**Report window:** 08:30–09:00 BST  
**Last updated:** 2026-08-09 01:10 BST
**Repository:** `arm-hackathon/arm-hackathon`  
**Working branch:** `ben/independent-recovery`  
**Repository root:** the active `ben/independent-recovery` working tree

## 1. Purpose

Deliver an evidence-backed AEOLUS hackathon model in which:

```text
observable primary-airflow degradation
→ compact learned adviser diagnoses the fault from causal telemetry
→ deterministic authority verifies target, persistence, topology, and bounds
→ an independent reserve airflow path is activated
→ physical airflow and CO₂ outcomes are compared with reserve-off counterfactuals
→ authority returns to reserve-off only after physical zero acknowledgement
```

The learned component advises. It never writes plant state or owns actuator commands. The deterministic authority and physics engine remain the only action boundary.

A negative learned-model result is acceptable. A fabricated advantage, validation-tuned claim, or unsafe integration is not.

## 2. Deadline deliverable

By 09:00 BST, produce one of these two honest outcomes.

### Outcome A — validation-qualified working adviser

All deterministic recovery gates pass, the learned adviser passes the frozen development/validation criteria, ONNX parity passes, the package builds and smoke-installs, and the exact local commits and evidence paths are reported.

### Outcome B — reproducible negative result

The deterministic recovery layer is either accepted or its exact blocker is recorded; the learned adviser attempt is executed as far as its prerequisites permit; failed validation is preserved without post-hoc threshold tuning; and all completed code, tests, artifacts, and package status are reported precisely.

The final held-out suite is not authorised by this overnight run. It remains a separate human gate after development acceptance.

## 3. Non-goals and prohibited actions

Tonight does not include:

- pushing any branch;
- changing PR #17 or PR #18 remotely;
- opening another PR;
- requesting a reviewer;
- merging or deploying;
- provisioning Azure, Arm cloud, or any paid resource;
- touching real environmental control equipment;
- rerunning or inspecting the frozen protocol-v3 final suite to make model decisions;
- tuning against recovery final-suite output;
- claiming physical, safety-critical, production, Arm64, INT8, or deployment evidence;
- adding dashboards or unrelated hackathon features.

All generated traces and corpora remain under ignored `out/` paths. No secrets may be read or printed.

## 4. Product definition of “working”

A working development model requires all of the following.

1. A validated schema-v10 habitat has one primary loop and one physically independent reserve loop per non-processing zone.
2. Reserve commands default to exactly zero and cannot be written by the legacy governor or the learned adviser.
3. The reserve path changes physical airflow and CO₂ only through the shared plant step.
4. The authority consumes completed-tick observable telemetry only and applies commands no earlier than the following tick.
5. `NOMINAL`, `DEGRADED`, `PROTECT`, and `HANDBACK` transitions are deterministic, bounded, and traceable.
6. Invalid identity, sequence, topology, selector, digest, numeric data, or unavailable advice fails closed.
7. Persistent reserve-delivery failure latches and cannot rearm in the same authority epoch.
8. Handback reaches physical command, position, and delivered-flow zero within the declared bound.
9. Four counterfactual arms are generated for every recovery family.
10. Healthy governed references do not activate reserve authority or regress physical outcomes.
11. Eligible fault-governed arms demonstrate measured benefit over the exact fault reserve-off arms, or the development gate records a negative result.
12. A compact adviser is trained from leakage-safe development telemetry, evaluated on family-held-out validation families, exported to FP32 ONNX, and compared with the frozen rule baseline.
13. The exact source can be built into wheel and sdist artifacts and installed into a clean environment.

## 5. Verified starting state

### 5.1 Live GitHub stack

| PR | Base | Head | State | Verified head |
|---|---|---|---|---|
| #17 `feat: freeze validation-only evaluation policy` | `alex/ai-2` | `ben/ai2-evidence-policy` | open draft | `4706020527c48936a26a440165450f9cb8d9d26e` |
| #18 `Enhance recovery response and evidence integrity` | `ben/ai2-evidence-policy` | `yarofix2` | open draft | `5a77d7a7a46459e6bceb311ea9057b10feeda881` |

PR #17 has no open review threads. Its frozen result remains negative: the temporal MLP loses to calibrated rules. This work must not alter that result.

PR #18 is retained as bounded-response provenance and is not merge-ready recovery.

### 5.2 Local branch

Starting HEAD:

```text
5afae5bb187ae815c7b804640a7d27da753fdfa5
```

Local commits above `origin/yarofix2`:

```text
6142f70 chore: bind bounded-response evidence provenance
a0ec493 feat: add recovery topology and counterfactual sweeps
7864279 feat: model independent reserve airflow
5afae5b feat: add versioned recovery trace boundary
```

Starting divergence: zero behind, four commits ahead of `origin/yarofix2`.

Starting dirty files:

```text
M  src/aeolus/scenario.py
M  src/aeolus/trace.py
?? src/aeolus/recovery.py
?? tests/test_recovery.py
?? tests/test_recovery_scenario.py
```

Preserved combined dirty-patch SHA-256:

```text
b6f59cf70e7d263d18277e3a05d9ce9920fddd3f5ee95bbd7b13635ea414b971
```

Last verified dirty-worktree receipt before this PRD:

```text
430 passed in 52.59s
ruff: clean
git diff --check: clean
```

This is a regression baseline, not approval. The authority slice still requires immutable-SHA review.

## 6. Immutable review protocol

**No SHA, no review.**

1. Finish one bounded code slice.
2. Run the slice tests, full suite, Ruff, and `git diff --check` as applicable.
3. Commit locally with the configured author:
   `Benedict Anokye-Davies <bbeennyy860@gmail.com>`.
4. Record the full checkpoint SHA in this PRD.
5. Create a detached review worktree at that SHA.
6. Dispatch at most one review batch for that slice.
7. The review request must name `BASE_SHA..REVIEW_SHA`; the verdict must echo `REVIEW_SHA`.
8. Do not modify the reviewed files while the review is active.
9. Any fix creates a new checkpoint SHA. The old verdict remains historical and cannot approve the replacement.
10. Delete the detached worktree after recording the verdict.

A late report for a superseded SHA is ignored. Reviews are never dispatched against the live dirty worktree.

## 7. Frozen safety and methodology invariants

### 7.1 Causality

- Decision tick `t+1` may consume only completed observation tick `t` or earlier.
- No fault schedule, effectiveness, connection health, seed, latent measurement state, or evaluator state enters adviser or authority inputs.
- Warm-up data must obey the same boundary.

### 7.2 Authority

- Only the deterministic recovery supervisor owns non-zero reserve commands.
- `NOMINAL` and `DEGRADED` commands are exactly zero.
- `PROTECT` may command one unambiguous target only.
- Command values are finite and within `0.0..1.0`.
- Per-tick reserve command movement is at most `0.1`.
- Healthy references and frozen-sensor faults never receive reserve authority.
- Target flapping, ambiguity, dropout, hash drift, or malformed advice cannot grant authority.

### 7.3 Failure and handback

- Reserve-delivery failure detection remains active in every state with a non-zero reserve command, including `HANDBACK`.
- Failure latches once per authority epoch.
- A failed epoch cannot re-enter `PROTECT` without reset.
- `HANDBACK` must source-enforce the 36-tick maximum, not rely only on current actuator timing.
- Completion requires five fresh ticks with reserve command, physical position, and delivered reserve airflow at zero.
- Fault recurrence during ordinary handback returns to `PROTECT`; a latched reserve failure cannot.

### 7.4 Trace and artifact integrity

- Every reason belongs to one fixed enum and every emitted reason is accepted by both the application and trace gates.
- Run identity, authority epoch, tick, sequence, selector hash, topology hash, and applied-command digest are validated.
- Canonical JSON and manifest hashes bind exact bytes.
- Canonical evidence requires a clean source checkpoint.
- Existing output paths are never overwritten.
- Duplicate executions must reproduce byte-identical traces and matching manifests.

## 8. PR #18 inherited correctness fixes

Prepare one isolated local commit that can later be cherry-picked onto `yarofix2` after push approval.

| ID | Live review issue | Required regression |
|---|---|---|
| PR18-1 | Alternative governor warm-up assumes `settings.window_ticks` | Governor with no settings and malformed window values falls back safely to `run.warmup_ticks` |
| PR18-2 | Frozen hold can jump from pre-rate-limit base command | Frozen transition preserves last commanded setpoint and never exceeds `max_command_delta` |
| PR18-3 | Custom governor factory receipt records default settings | Receipt binds evaluated settings or explicitly marks settings unavailable; misleading defaults rejected |
| PR18-4 | `onset_tick=None` supported at runtime but annotated `int` | Signature is `int | None`; focused test covers `None` |
| PR18-5 | Two Ruff-unused imports at exact PR head | Ruff clean after isolated fix commit |

Do not rewrite the historical 129-family response evidence to imply improvement. If fixes change canonical evidence bytes, preserve the old receipt as historical and generate a separately named development receipt with explicit provenance.

## 9. Deterministic recovery requirements

### 9.1 Topology and plant

- Schema version is exactly 10 for recovery scenarios.
- Every non-processing zone has exactly one primary outbound/return pair and one reserve outbound/return pair.
- Reserve loop IDs, capacities, zones, and processing endpoint are validated structurally.
- Primary and reserve actuator states are separate.
- Primary and reserve requested/delivered airflow are separately metered.
- Reserve allocation has its own declared capacity and does not conjure primary capacity.
- Total plant conservation and non-negative-flow invariants remain true.
- Schema-v9 scenarios retain their inherited bytes and behaviour.

### 9.2 Authority states

| State | Reserve owner | Required behaviour |
|---|---|---|
| `NOMINAL` | reserve off | zero commands; observe only |
| `DEGRADED` | reserve off | accumulate same-target evidence; zero commands |
| `PROTECT` | deterministic supervisor | one target; bounded ramp; monitor delivery and recovery |
| `HANDBACK` | deterministic supervisor until physical zero | bounded ramp-down; recurrence handling; zero acknowledgement |

Default frozen thresholds live only in `RecoverySettings`. Any changed threshold requires a new policy version and complete development rerun.

### 9.3 Required authority tests

- malformed settings and hysteresis ordering;
- topology/capacity mismatch;
- run, epoch, tick, sequence, selector, topology, digest, key-set, bound, and finite-number rejection;
- one-tick causal activation;
- same-target persistence;
- flapping and ambiguity rejection;
- nominal/degraded/protect dropout behaviour;
- slew-bounded increase;
- persistent-fault hold;
- clear persistence;
- recurrence during handback;
- physical zero acknowledgement;
- reserve failure while protecting and while handing back;
- explicit 36-tick handback source bound;
- failure-latch no-rearm;
- reset and deterministic event replay;
- fixed reason enum coverage across every reachable transition.

### 9.4 Required integration tests

- healthy replay is byte-stable and reserve-off;
- persistent eligible fault activates reserve only in governed arm;
- pre-activation plant records are identical between off and governed arms;
- frozen sensor never activates reserve;
- transient fault reaches `PROTECT`, `HANDBACK`, and acknowledged physical zero;
- failed reserve path latches and shuts down within bound;
- application gate validates every emitted decision, not only cold start;
- trace writer refuses malformed authority records.

## 10. Four-arm recovery evidence

### 10.1 Family arms

Every counterfactual recovery family produces exactly:

1. `reference_reserve_off`
2. `reference_governed`
3. `fault_reserve_off`
4. `fault_governed`

All four arms share the same base condition, operating profile, target, seed, source multiplier, telemetry parameters, run settings, and reference/fault scenario pair. Only the declared fault profile and authority arm may differ.

Each family stores `base_condition_id`, `counterfactual_group_id`, split, role, scenario hashes, arm trace hashes, and policy/source hashes.

### 10.2 Frozen suites

Development suite:

- schema `aeolus_sweep_v4`;
- train seeds `211..216`;
- validation seeds `601..603`;
- targets `cabin_a`, `cabin_b`, `lab`;
- blocked, gradual, transient blocked, and transient gradual profiles;
- two declared operating profiles;
- no final-family identity overlap.

Final suite:

- remains unopened for decision-making tonight;
- expected seeds are `1201..1203`; the current `1301..1303` mismatch must be corrected before any authorised final generation;
- output is write-once and requires explicit human approval.

### 10.3 Per-arm physical metrics

Compute from trace plus evaluator-only physical state, never hidden truth in model input:

- primary requested-airflow integral;
- primary delivered-airflow integral;
- primary shortfall integral;
- reserve requested-airflow integral;
- reserve delivered-airflow integral;
- total delivered-airflow integral;
- reserve shortfall-coverage fraction;
- steady-state restoration fraction after reserve actuator movement stops;
- reserve saturation ticks;
- integrated physical CO₂ concentration per zone;
- integrated physical CO₂ excess above the declared ceiling per zone;
- ticks above ceiling per zone;
- maximum physical CO₂ concentration per zone;
- captured-CO₂ delta;
- first `DEGRADED`, `PROTECT`, and `HANDBACK` ticks;
- physical-zero acknowledgement tick;
- invariant-violation count.

Every ratio metric has an explicit `defined`, `not_applicable`, or `undefined_zero_denominator` status. Zero denominators are never silently converted to zero or one.

### 10.4 Frozen development gates

All safety gates are mandatory:

- zero invariant violations in all arms;
- zero reserve delivery in every reserve-off arm;
- zero `PROTECT` entries in healthy governed references;
- zero `PROTECT` entries for frozen-sensor families;
- no pre-activation physical divergence between paired off/governed fault arms;
- all transient governed arms reach acknowledged physical zero within 36 ticks of handback entry;
- no failed-reserve epoch rearms;
- duplicate execution produces byte-identical traces and evidence documents.

Benefit is evaluated only on eligible physical airflow faults:

- median integrated physical CO₂ excess must improve by at least 5% versus `fault_reserve_off`;
- at least 60% of eligible validation families must improve physical CO₂ excess;
- median total delivered-airflow integral must not regress;
- no healthy reference may regress CO₂ excess or ticks-above-ceiling beyond floating-point equality tolerance;
- any family with an undefined benefit denominator is reported separately and excluded from percentage aggregation, never counted as a win.

These thresholds are frozen before the first development evidence result. If they fail, record a negative result. Do not tune them after inspection.

### 10.5 Stress and falsification

After the canonical development run, attempt to falsify the weakest claim with:

- reserve-path health zero during `PROTECT`;
- fault recurrence during handback;
- observation dropout in every authority state;
- target flapping and two-zone ambiguity;
- zero-demand and denominator-zero families;
- high-noise/high-drift telemetry within schema bounds;
- reserve saturation;
- malformed hashes and digests;
- duplicate-run relocation to a different output directory;
- clean-checkout reproduction of the canonical command.

Stress output cannot change the frozen development policy or thresholds.

## 11. Learned adviser lane

This lane opens only if the deterministic safety gates and canonical development evidence complete without a blocker.

### 11.1 Adviser role

The adviser consumes exact causal `float32[10,24]` `model_input_v1` windows and emits:

- predicted diagnosis class;
- confidence;
- class probabilities;
- observation tick;
- selector hash;
- topology hash;
- model artifact hash.

It does not emit actuator commands. Target-zone selection, persistence, command ownership, slew limits, reserve-delivery checks, and handback remain deterministic.

### 11.2 Leakage boundary

Forbidden adviser inputs include:

- fault schedule or type from scenario configuration;
- fault start/end/effectiveness;
- connection health;
- random seed;
- latent CO₂ state not present in `model_input_v1`;
- reserve policy state or future commands;
- evaluator-only physical state;
- final-suite data.

Training and validation split by family/base condition, never by correlated windows. `base_condition_id` and `counterfactual_group_id` cannot cross splits.

### 11.3 Bounded candidates

Use only the existing reproducible candidate families unless a source-level blocker requires less:

1. class-balanced softmax detector;
2. compact `temporal_summary_v1` MLP (`135 → 16 ReLU → 4 softmax`).

No architecture sweep beyond these candidates tonight. Candidate selection, thresholding, and any confidence persistence use train/validation only.

### 11.4 Adviser integration

- Airflow-fault advice (`gradual_primary_fan_degradation` or `blocked_path`) may support deterministic entry only when the independently computed residual target is unambiguous and persistent.
- `frozen_sensor`, low confidence, invalid advice, hash mismatch, stale tick, unavailable model, or disagreement with deterministic target cannot grant authority.
- Malformed/unavailable advice during active protection initiates bounded handback; it never freezes a non-zero command indefinitely.
- The reserve failure latch dominates all adviser output.
- Every tick records whether advice was accepted, rejected, unavailable, or not applicable and the fixed reason.

### 11.5 Model evidence and acceptance

Produce:

- canonical detector JSON;
- FP32 ONNX;
- ONNX parity receipt over validation rows;
- training/selection receipt;
- family-held-out validation metrics;
- calibrated rule-baseline comparison;
- integrated adviser/governor safety comparison if the model qualifies.

Reuse the frozen protocol-v3 preference logic:

- learned macro-F1 must not be below calibrated rules;
- nominal false-alarm rate must not exceed calibrated rules;
- a latency claim requires at least 20% lower median causal detection latency;
- ONNX parity must pass;
- all deterministic safety gates must remain exact.

If the learned candidate fails, freeze `preferred_method=rule_baseline` and `ai_advantage_demonstrated=false`. Do not tune from validation failures during this run.

## 12. Release and reproducibility

Release work occurs only after the deterministic implementation is internally consistent.

1. Set project version to `0.2.0`.
2. Update `CHANGELOG.md`, `README.md`, `PLAN.md`, `PROBLEM.md`, bounded-response documentation, telemetry/simulation contracts, and a recovery-protocol acceptance document.
3. Keep claims calibrated: simulation, development validation, and negative results are named explicitly.
4. Refresh `uv.lock` with Python 3.11.
5. Run:

```text
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
git diff --check
python -m compileall -q src tests
```

6. Build wheel and sdist with `uv build`.
7. Create a clean temporary environment.
8. Install the exact wheel without editable source.
9. Verify package version/imports and run one deterministic recovery replay using repository scenario input.
10. Hash wheel, sdist, policy, model, ONNX, manifests, reports, and canonical traces.
11. Record the exact source SHA and dirty status for every artifact.

Untested packaging is reported as untested. No plausible-looking command output may be substituted.

## 13. Local commit plan

Expected bounded checkpoints:

| Checkpoint | Intended scope | Review required | Current state / SHA |
|---|---|---|---|
| C0 | This PRD only | no code review | complete — `c5c53cb571643254f8a7300d518a4e1ae20901a1` |
| C1 | Authority core, runner, trace enum, focused tests | one immutable compliance/quality batch | staged for local checkpoint — code patch `7083da43f8eecd44621393e90029330f2dec2098077f20f1b8a32d637d55b911`; full gates green; unreviewed |
| C2 | Findings-only correction for C1 | one fresh delta review if C1 had blockers | not started |
| C3 | Four inherited PR #18 fixes and regressions | one immutable review | not started |
| C4 | Four-arm evidence runner, metrics, gates, docs/tests | one immutable review | not started |
| C5 | Adviser training/integration/artifacts, or frozen negative result | one immutable review | not started |
| C6 | Version/docs/package closeout | final immutable review | not started |

Checkpoint names may be combined only if the files and acceptance boundary are inseparable. Every actual SHA replaces the placeholder in the execution ledger.

## 14. Execution schedule and stop rules

This is a priority order, not permission to lower gates.

| Timebox ending | Target |
|---|---|
| 01:30 | PRD checkpoint; authority slice frozen, tested, and under immutable review |
| 03:15 | authority findings resolved; inherited PR #18 blockers fixed |
| 05:30 | four-arm development evidence implemented and canonical run started/completed |
| 06:45 | stress/falsification complete; adviser lane accepted or blocked |
| 07:45 | adviser attempt and ONNX parity complete, or reproducible negative result frozen |
| 08:30 | release/package verification complete or exact blocker frozen; implementation stops |
| 09:00 | final report delivered |

Hard stop rules:

- Stop adviser work if deterministic safety/evidence gates fail.
- Stop model selection after the one frozen candidate comparison; no rolling architecture search.
- Stop a fix loop after two evidence-backed attempts at the same root cause and record the blocker.
- Stop all implementation at 08:30 BST.
- Never extend the deadline silently.

## 15. Execution ledger

Append one row after each material step. Receipts must be actual outputs.

| BST time | Task | State | Source SHA / patch hash | Verification receipt | Finding / next gate |
|---|---|---|---|---|---|
| 00:53 | Live-state reconstruction | complete | HEAD `5afae5b`; dirty patch `b6f59c…` | remote fetched; PR #17/#18 inspected; 430-test prior baseline | Write and freeze PRD |
| 00:56 | PRD checkpoint | complete | `c5c53cb571643254f8a7300d518a4e1ae20901a1` | one-file commit; 574 lines; SHA-256 `65078986…70a65` | Freeze and verify authority slice |
| 00:56 | Unattended-run guardrails | complete | profile configuration | 2500 primary turns; hard loop stop on; two review workers maximum per turn; worker iterations remain 50 | Start bounded execution |
| 00:58–01:01 | Target-worktree reconciliation and regression rerun | complete | HEAD `c5c53cb571643254f8a7300d518a4e1ae20901a1`; pre-existing authority slice dirty | `uv run --locked --python 3.11 --extra dev python -m pytest -q` → `430 passed in 57.66s`; Ruff → `All checks passed!`; authority-only diff check clean | Current recomposed dirty patch does not reproduce recorded `b6f59c…`; source provenance retained as a discrepancy. Repair C1 gaps test-first before checkpoint. |
| 01:03 | C1 handback delivery-failure latch | complete | dirty slice on `c5c53cb…` | focused RED: `handback_ramp` instead of `reserve_delivery_failure`; focused GREEN: `1 passed in 0.07s`; diff check clean | Persistent delivery failure is now evaluated while `HANDBACK` still commands reserve flow. Test hard 36-tick source bound next. |
| 01:04 | C1 handback source bound | complete | dirty slice on `c5c53cb…` | focused RED: state remained `HANDBACK` at dwell 36; focused GREEN: `1 passed in 0.07s` | At elapsed handback tick 36, fail to latched reserve-off `DEGRADED` without faking physical-zero acknowledgement. Freeze the exact threshold and test write-once traces next. |
| 01:04 | C1 frozen handback threshold | complete | dirty slice on `c5c53cb…` | focused RED: `maximum_handback_ticks=35` did not raise; focused GREEN: `7 passed in 0.07s` | `RecoverySettings` now rejects a non-36 handback bound. Test write-once trace paths next. |
| 01:05 | C1 write-once trace paths | complete | dirty slice on `c5c53cb…` | focused RED: two writers did not raise; focused GREEN: `2 passed in 0.03s` | `TraceWriter` and `RecoveryTraceWriter` use atomic exclusive creation and preserve existing bytes. Audit remaining C1 authority/runner invariants before freeze. |
| 01:06 | C1 protect-state observation dropout | complete | dirty slice on `c5c53cb…` | focused RED: non-zero `PROTECT` decision retained after dropout; focused GREEN: `2 passed in 0.07s` | Dropout enters bounded `HANDBACK`; repeated dropout ramps down and cannot restore reserve authority. Audit topology-order and trace cross-field boundaries before freeze. |
| 01:08 | C1 file-order topology binding | complete | dirty slice on `c5c53cb…` | focused RED: `recovery observation zone topology is invalid`; focused GREEN: `1 passed in 0.07s` | Runner observations now use validated scenario file order, matching selector and supervisor topology hashes. Freeze C1 after full focused verification. |
| 01:09 | C1 full pre-freeze verification | complete | staged code patch `7083da43f8eecd44621393e90029330f2dec2098077f20f1b8a32d637d55b911` | locked full suite `436 passed in 56.70s`; Ruff `All checks passed!`; `compileall -q src` and `git diff --check` exit 0 | Stage PRD receipt, commit the bounded C1 slice locally, then review only that immutable SHA. |

## 16. Morning report contract

The report must state, in this order:

1. outcome A or B;
2. exact branch, HEAD, base, ahead/behind, and dirty files;
3. commit table and reviewed SHA verdicts;
4. deterministic test, lint, compile, and replay receipts;
5. four-arm evidence counts and physical outcome metrics;
6. stress failures and weakest remaining claim;
7. adviser architecture, artifact hashes, validation comparison, and ONNX parity—or exact reason the lane did not open;
8. package filenames, hashes, clean-install result, and version;
9. PR #17/#18 fix mapping and the safe stacked integration route;
10. explicit list of actions not taken: push, merge, deploy, cloud, final suite.

## 17. Acceptance checklist

### Process

- [x] Live remote and PR state reconstructed
- [x] Tool budget raised to 2500 primary execution turns; review workers retained at 50
- [x] PRD created before resumed implementation
- [x] PRD checkpointed locally at `c5c53cb571643254f8a7300d518a4e1ae20901a1`
- [x] Execution ledger updated after every material step
- [x] No mutable-worktree review dispatched

### Authority and plant

- [x] Recovery topology and independent reserve plant implemented locally
- [x] Versioned recovery trace boundary implemented locally
- [ ] Authority/runner slice checkpointed
- [ ] Authority/runner immutable review passed
- [ ] Review blockers fixed and reverified

### PR #18

- [ ] Alternative-governor warm-up fixed
- [ ] Frozen-command slew bound fixed
- [ ] Custom-factory receipt fixed
- [ ] Optional onset annotation fixed
- [ ] Exact-head Ruff debt fixed

### Evidence

- [ ] Four-arm runner implemented
- [ ] Metrics have explicit denominator status
- [ ] Canonical development run complete
- [ ] Duplicate reproduction byte-identical
- [ ] Frozen safety gates pass
- [ ] Frozen benefit gates pass or negative result recorded
- [ ] Stress/falsification complete

### Adviser

- [ ] Deterministic prerequisites pass
- [ ] Leakage-safe recovery development corpus built
- [ ] Bounded candidates trained
- [ ] Family-held-out validation complete
- [ ] FP32 ONNX exported
- [ ] ONNX parity passed
- [ ] Adviser integrated without command ownership, or negative result frozen

### Release

- [ ] Version `0.2.0`
- [ ] Documentation reconciled
- [ ] Full suite green
- [ ] Ruff clean
- [ ] Compile and diff checks clean
- [ ] Wheel and sdist built
- [ ] Clean wheel install and replay passed
- [ ] Artifact hashes recorded
- [ ] No push, merge, deploy, cloud, or final-suite action
