# AEOLUS Hackathon Working Model Product Requirements Document

**Status:** Active execution contract  
**Owner:** Benedict Anokye-Davies  
**Deadline:** 2026-08-09 09:00 BST  
**Execution stop:** 2026-08-09 08:30 BST  
**Report window:** 08:30–09:00 BST  
**Last updated:** 2026-08-09 03:02 BST
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
| C1 | Authority core, runner, trace enum, focused tests | one immutable compliance/quality batch | checkpoint `b4c6ddfe79c24b591f18497b9d9ba1b35298396d`; base `c5c53cb571643254f8a7300d518a4e1ae20901a1`; exact detached checkout reverified; no SHA-bound reviewer verdict returned by the 01:31 cutoff, so this checkpoint is unreviewed |
| C2 | Findings-only correction for C1 | one fresh delta review if C1 had blockers | not applicable — no actual C1 finding exists; absence of a reviewer verdict is not a finding |
| C3 | Four inherited PR #18 fixes and regressions | one immutable review | checkpoint `88321f1d7bda00d215d81a535eaeafc9fa72b5c0`; base `b4c6ddfe79c24b591f18497b9d9ba1b35298396d`; detached checkout reverified clean; no SHA-bound review verdict returned by the 01:58 cutoff, so this checkpoint is unreviewed |
| C4 | Four-arm evidence runner, metrics, gates, docs/tests | one immutable review | checkpoint `74154956d64309f067ada7593e2ef8786d140b4e`; base `88321f1d7bda00d215d81a535eaeafc9fa72b5c0`; canonical/duplicate/clean-checkout evidence and stress matrix complete with frozen negative gates; exact detached review had no SHA-bound verdict by the single 02:40 cutoff, so C4 is unreviewed |
| C5 | Adviser training/integration/artifacts, or frozen negative result | one immutable review | closed — C4 frozen safety and physical-delivery gates failed; no corpus/model/ONNX/integration run permitted |
| C6 | Version/docs/package closeout | final immutable review | checkpoint `3776af035b07590db53587212762a9ff3304acfb`; exact-commit packages installed and replayed, but the clean locked suite exposed one uncontrolled dirty-worktree test fixture (`1 failed, 453 passed`), so C6 is superseded and not release-qualified |
| C7 | Clean-source fixture correction | focused regression plus exact-clean full gate | checkpoint `20d7d90caaafe6565bc731d3494a79e8574d626a`; focused RED/GREEN complete; exact clean C7 locked suite `454 passed`; Ruff, compile, diff, and lock gates passed |
| C8 | Final receipt reconciliation and rebuilt packages | one immutable final review | in progress — reconcile the C6 blocker/C7 fix, freeze a clean documentation checkpoint, then rebuild and smoke-test exact-C8 distributions |

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
| 01:11 | C1 local immutable checkpoint | complete | BASE `c5c53cb571643254f8a7300d518a4e1ae20901a1`; REVIEW `b4c6ddfe79c24b591f18497b9d9ba1b35298396d` | clean primary worktree; author and committer both `Benedict Anokye-Davies <bbeennyy860@gmail.com>`; detached review worktree at REVIEW SHA and `diff --check` clean | Dispatch exactly one read-only compliance/quality review batch. Do not modify reviewed files while it runs. |
| 01:31 | C1 immutable review outcome | unreviewed | REVIEW `b4c6ddfe79c24b591f18497b9d9ba1b35298396d` | no returned SHA-bound verdict after the 20-minute review window; detached SHA reverified with `436 passed in 57.76s`, Ruff/compile/diff clean, and clean status | No approval inferred and no second C1 review batch dispatched; continue with this exact unreviewed state. |
| 01:16–01:23 | C3 PR #18 fixes 2–5 | complete | dirty C3 slice on `b4c6ddfe…` | frozen slew RED `Δ=0.85` vs cap `0.05`, then `2 passed`; custom receipt RED `0.1` vs `0.03`, then `2 passed`; optional-onset annotation RED `int`, then `1 passed`; Ruff clean | Response/evidence fixes are isolated; warm-up boundary remains C3-1. |
| 01:32 | C3 alternative-governor warm-up | complete | dirty C3 slice on `b4c6ddfe…` | missing/malformed settings RED `.FFF` (crash or under-seeding); GREEN `4 passed in 0.07s` | invalid/missing `window_ticks` now falls back to `run.warmup_ticks`. |
| 01:33 | C3 full verification | complete | staged C3 code patch `e12bd09adc95c6b0f7d52ddd01d6941880f067e1c53b3f35a26d69ce4eba86d1` | locked suite `443 passed in 57.66s`; Ruff `All checks passed!`; compileall and diff check exit 0 | Stage PRD receipt and commit exact C3 checkpoint. |
| 01:35 | C3 local immutable checkpoint | complete | BASE `b4c6ddfe79c24b591f18497b9d9ba1b35298396d`; REVIEW `88321f1d7bda00d215d81a535eaeafc9fa72b5c0` | clean C3 commit; Ben author/committer; detached review worktree is clean | No C3 review has been dispatched yet. |
| 01:38 | C3 immutable review dispatch | in progress | REVIEW `88321f1d7bda00d215d81a535eaeafc9fa72b5c0` | detached worktree is clean before dispatch | Require a returned verdict echoing REVIEW SHA; do not modify C3 files while it runs. |
| 01:58 | C3 immutable review outcome | unreviewed | REVIEW `88321f1d7bda00d215d81a535eaeafc9fa72b5c0` | no SHA-bound verdict returned in the 20-minute review window; detached checkout still clean | No approval inferred and no second C3 review batch dispatched. |
| 01:39–01:57 | C4 four-arm runner and falsification contracts | in progress | dirty C4 slice on `88321f1…` | arm-name RED/GREEN `3 passed`; final-seed RED/GREEN `1 passed`; recovery evidence/gate/duplicate/saturation/noise/denominator tests `22 passed in 6.10s`; cross-suite pre-freeze `454 passed in 62.74s`; targeted Ruff/compile/diff clean | Four-arm development runner is implemented; freeze exact C4 source before canonical evidence. |
| 02:01 | C4 full pre-freeze verification | complete | staged C4 code patch `19870c21fcae46cd67029a362c9403497b747c001aec4a79f99425e4c206cd77` | locked full suite `454 passed in 63.59s`; Ruff/compile/diff and `uv lock --check` passed | Commit exact C4 source, then run canonical development evidence from the clean SHA. |
| 02:02 | C4 local checkpoint | complete | BASE `88321f1d7bda00d215d81a535eaeafc9fa72b5c0`; SOURCE `74154956d64309f067ada7593e2ef8786d140b4e` | clean source status before canonical execution; Ben author/committer | Canonical evidence must run only from this SHA. |
| 02:02–02:07 | C4 canonical development run A | negative | SOURCE `74154956d64309f067ada7593e2ef8786d140b4e`; `out/overnight/recovery-development-c4-a` | 756 families/3,024 traces; receipt SHA `1cbb9d428824f57c500b4a1ac3859b4ea6ef0a0dd4e70012b2e6c35d230a1730` self-hash verified; source clean | safety false: transient acknowledgement; benefit false: physical-delivery gate. Model lane closes without tuning. |
| 02:08 | C4 stress/falsification matrix | complete | SOURCE `74154956d64309f067ada7593e2ef8786d140b4e` | 11 targeted stress tests passed in 3.42s: delivery failure, recurrence, dropout, ambiguity, malformed authority, saturation, high noise/drift, denominator zero | Tested fail-closed paths are sound; does not repair canonical transient outcome failure. |
| 02:08–02:20 | C4 duplicate and clean-checkout reproduction | complete | SOURCE `74154956d64309f067ada7593e2ef8786d140b4e` | A/B comparison `aadbcd25faffe70a30311187e999c2d8c16c5a61d68a8e8df34606ea6f653343`: 3,801 files and 3,024 traces identical; clean-checkout comparison `2264c85fc85dd63ee99f853523af0d2dec3c67b861f67e61db05d7c9cb0ef733`: same evidence SHA/files, both source-clean | Deterministic reproduction confirmed; retain negative gate outcome. |
| 02:22–02:40 | C4 immutable review window | unreviewed | BASE `88321f1d7bda00d215d81a535eaeafc9fa72b5c0`; REVIEW `74154956d64309f067ada7593e2ef8786d140b4e` | detached review worktree was clean at REVIEW SHA; no returned verdict echoed REVIEW SHA by the single bounded cutoff | No approval inferred; no second C4 review batch will be dispatched, and any late response is stale. |
| 02:28–02:37 | C6 source reconciliation and locked verification | complete | dirty C6 documentation/version source on `74154956…` | `uv lock --offline` refreshed `aeolus` `0.1.0 → 0.2.0`; docs link/fact checks passed; locked full suite `454 passed in 64.65s`; Ruff, compileall, `git diff --check`, and `uv lock --check` passed | Commit C6 locally, then build and smoke-test only from the exact clean C6 SHA. |
| 02:40–02:57 | C6 clean-source package cycle | blocked | `3776af035b07590db53587212762a9ff3304acfb` | wheel `7656eee5…66aa5` and sdist `cadf953a…0c1a` built; fresh wheel/sdist installs and 180-/120-record replays passed; exact clean locked suite failed `1 failed, 453 passed in 66.62s` | Test expected a dirty source without controlling repository state. C6 artifacts are evidence only, not release-qualified. |
| 02:58–03:00 | C7 clean-source fixture correction | complete | patch `934abd8e…e23053`; checkpoint `20d7d90caaafe6565bc731d3494a79e8574d626a` | focused clean-checkout RED `1 failed`; controlled-status GREEN `1 passed`; pre-checkpoint full suite `454 passed in 63.61s`; Ruff, compile, diff, and lock gates passed | Freeze test-only correction, then prove the full gate from the exact clean SHA. |
| 03:01–03:02 | C7 exact-clean verification | complete | `20d7d90caaafe6565bc731d3494a79e8574d626a`; zero dirty lines | locked full suite `454 passed in 62.80s`; Ruff, compile, diff, and lock gates passed | Reconcile receipts as C8 and rebuild exact-C8 packages. |

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
- [x] Authority/runner slice checkpointed at `b4c6ddfe79c24b591f18497b9d9ba1b35298396d`
- [x] Authority/runner bounded review window completed; no SHA-bound verdict, therefore unreviewed
- [x] C2 not applicable — no actual C1 finding exists to correct or re-review

### PR #18

- [x] Alternative-governor warm-up fixed
- [x] Frozen-command slew bound fixed
- [x] Custom-factory receipt fixed
- [x] Optional onset annotation fixed
- [x] Exact-head Ruff debt fixed

### Evidence

- [x] Four-arm runner implemented
- [x] Metrics have explicit denominator status
- [x] Canonical development run complete (756 families / 3,024 traces)
- [x] Duplicate reproduction byte-identical (3,801 files / 3,024 traces)
- [x] Frozen safety gates evaluated; negative result recorded (transient physical-zero acknowledgement)
- [x] Frozen benefit gates evaluated; negative result recorded (physical reserve delivery)
- [x] Stress/falsification complete (11 targeted tests)
- [x] C4 bounded immutable review window completed; no SHA-bound verdict, therefore unreviewed

### Adviser — C5 closed

- [x] Deterministic prerequisites evaluated and required gates failed
- [x] Negative result frozen; adviser lane closed before corpus/training
- [x] No recovery corpus, model, ONNX, tuning, or integration artifact produced
- [x] No final-suite input inspected or evaluated

### Release

- [x] Version `0.2.0`
- [x] Documentation reconciled to Outcome B
- [x] Exact clean C7 locked full suite green (`454 passed in 62.80s`)
- [x] Ruff clean
- [x] Compile and diff checks clean
- [x] C6 wheel and sdist built and smoke-installed; superseded by the C7 test correction and not release-qualified
- [ ] Wheel and sdist rebuilt from exact clean C8 SHA
- [ ] Clean C8 wheel install and deterministic recovery replay passed
- [ ] Exact C8 artifact hashes recorded in ignored closeout receipt
- [x] No push, merge, deploy, cloud, or final-suite action
