# Habitat V2 Forecast D2 Qualification PRD

Status: EXACT POLICY APPROVED. PILOT GENERATION BLOCKED ON RUNNER, CUSTODY AND RESOURCE PREFLIGHT

## 1. Plain-English contract

### What are we making?

A fail-closed qualification layer that decides whether the Habitat V2 forecasting problem has a defensible history length, prediction horizon, action signal and deterministic baseline margin before any learned model is trained.

### What happens?

1. A separately frozen statistical-policy artifact defines every support threshold, statistic, confidence procedure, multiplicity correction, comparator rule and stop precedence.
2. A semantically disjoint 60-cluster public pilot evaluates all nine supported history/horizon pairs using the same maximum witnesses.
3. If timing is defensible, a deterministic canonical campaign is generated with whole-cluster TRAIN, VALIDATION and withheld FINAL assignments.
4. Deterministic baselines are fitted on TRAIN only and evaluated on VALIDATION once.
5. The gate either freezes evidence-derived candidate margins or stops without training.

### How do we know it worked?

- unapproved, malformed, substituted, non-finite, duplicate-key, raw-byte-hash
  drifted or semantic-self-hash-drifted policy artifacts are rejected before
  pilot data is consumed.
- no timing pair can be selected without the exact approved policy identity and all 60 declared pilot clusters.
- pilot families are cryptographically excluded from canonical TRAIN, VALIDATION and FINAL families.
- every fitted value is derived from TRAIN clusters only.
- validation access is one-use and receipt-bound.
- every outcome is one closed stop/proceed enum with explicit reason evidence.
- complete replay, split, package and source identities are preserved.
- no learned-model training is performed by this slice.

## 2. Frozen foundation

- repository: `arm-hackathon/arm-hackathon`
- local parent branch: `ben/habitat-v2-forecast-data-foundation`
c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3
- deterministic HMC foundation: `79d6a718e0d44122a763bb72f9c8ed929f39fd23`
- active remote Habitat integration base at planning time: `origin/alex/ai-2` at the same `79d6a718...`
- D2 branch: `ben/habitat-v2-forecast-qualification`
- D2 worktree: `C:/Users/Nxiss/code/aeolus-forecast-qualification`
- interpreter: `C:/Users/Nxiss/code/aeolus-forecast-data-foundation/.venv/Scripts/python.exe`
- package version at start: `0.8.0`

The D1 candidate remains a separate evidence subject. D2 does not amend, reset or rewrite it.

## 3. Decision options

### Option A: train a learned model from the four-sample D1 fixture

Rejected. Four samples prove plumbing only and cannot support timing, action-information or baseline-headroom claims.

### Option B: generate the proposed 74,880-sample campaign immediately

Rejected. This would spend the canonical family roster before selecting defensible `(W,H)` timing and before freezing the statistical policy that interprets the results.

### Option C: policy boundary -> disjoint timing pilot -> canonical corpus -> baseline gate

Chosen. This is the smallest reversible sequence that preserves scientific and final-set custody.

## 4. Current execution gate

Approved now:

1. add this D2 PRD.
2. define a strict closed schema for the qualification policy.
3. implement canonical loading, validation, immutability and self-hash verification using RED-GREEN TDD.
4. require explicit `APPROVED` ratification plus exact expected raw-byte and
   semantic policy SHA-256 identities before any selector can run.
5. draft a proposed policy and a proposed 60-cluster profile/action packet for human review.
6. run focused D1 regressions after each D2 slice.

Blocked until Ben/team ratification:

- freezing numerical support thresholds.
- freezing fixed-topology scenario profiles, load levels, treatment values, source-prefix/noise policies or proposal actions.
- generating the 60-cluster timing pilot.
- selecting `W` or `H`.
- generating canonical TRAIN, VALIDATION or FINAL data.
- opening validation.
- training or selecting a learned model.
- any push, PR, merge, tag, release, deployment or submission.

## 4.1 Campaign resource preflight

D1's direct-ridge implementation is a correct small-fixture reference, not yet a
campaign-qualified solver. Its flattened maximum input has 34,987 float values
per sample. The proposed canonical TRAIN split has 52,416 samples, so the
current dual formulation would allocate a 52,416 x 52,416 float64 Gram matrix:
approximately 20.47 GiB before feature, target, coefficient and Python/runtime
overhead. The planning host has 15.36 GiB physical RAM.

Therefore pilot generation is additionally blocked until a deterministic
resource preflight proves the selected baseline implementation and packet
representation fit within a frozen memory/runtime budget. The implementation
must not discover this by exhausting host memory during the real pilot.

The following options were reviewed because they alter the comparator:

1. retain the full flattened baseline and implement a deterministic chunked or
   iterative solver.
2. add a compact autoregressive direct-ridge baseline over the causal 51-target
   history estimates plus the 27-value proposed action (843 features at W=16),
   while retaining persistence and linear baselines.
3. predeclare another deterministic dimensionality reduction with exact bytes,
   seed and collision/approximation semantics.

Ben approved option 2 in principle on 2026-08-14 with a fairness safeguard: the
compact representation must be evaluated against the existing full-contract
ridge on an identical manageable subset before its exact feature contract is
frozen. A materially weaker compact comparator cannot be used to establish
learned-model headroom. Reducing cluster support merely to make the current
dense solver fit is not an acceptable resource workaround.

Deterministic gzip storage was also approved on 2026-08-14. Compressed artifacts
must bind both uncompressed canonical-byte SHA-256 and compressed-byte SHA-256,
use gzip level 9 with `mtime=0` and no platform-varying filename metadata, and
reproduce identical compressed bytes across fresh generations.

## 5. Policy artifact requirements

Planned artifact:

`contracts/habitat_v2_forecast_qualification_policy_v1.json`

The top-level object must be closed and contain at least these independently validated sections:

- schema and ratification identity.
- D1 candidate, HMC binding, input manifest, target manifest and evaluator identities.
- supported `W={4,8,16}` and `H={2,4,8}` grids.
- exact 60-cluster pilot manifest identity and canonical-corpus exclusion rule.
- timing support thresholds.
- action-effect statistic and support rule.
- nested whole-cluster fold construction.
- paired confidence/interval procedure.
- critical-cell universe and multiplicity correction.
- history/horizon selection rule and exact tie breaks.
- baseline comparator selection.
- deterministic whole-cluster bootstrap procedure and seed derivation.
- evidence-derived comparator-margin rule.
- validation access rule.
- closed stop precedence.
- prohibited claims and downstream permissions.
- canonical semantic policy self-hash and independently pinned raw artifact
  SHA-256.

The artifact must reject duplicate JSON keys, unknown or missing fields, non-finite values, booleans where integers are required, unordered or duplicate sets, malformed hashes, unsupported enums, non-monotonic quantiles and internally inconsistent thresholds.

A caller-supplied `APPROVED` string is not authority by itself. Runtime
qualification requires both the exact expected canonical semantic policy
SHA-256 and the exact expected raw artifact SHA-256 pinned by the calling
qualification entry point. Both identities are computed from the same read.

## 6. Existing design requirements retained

- observation cadence: `60.0 s`.
- timing grid: `W in {4,8,16}`, `H in {2,4,8}`.
- timing pilot: exactly 60 semantically disjoint public clusters.
- the same maximum witnesses are sliced for all nine timing pairs.
- nested whole-cluster evaluation.
- zero replay/identity failures and complete targets.
- measurable paired action effect versus no-proposal continuation.
- action-aware ridge must reliably outperform action-blinded ridge.
- no corrected critical-cell regression versus persistence/linear baselines.
- choose the shortest defensible history for each horizon.
- choose the largest viable horizon with exact shorter-history tie break.
- TRAIN-only preprocessing and fitted parameters.
- TRAIN-only target scales, with explicit support coverage. A zero-range TRAIN
  target cannot be silently dropped when a held-out cluster varies or crosses a
  declared safety envelope.
- validation opened once after all candidate bytes are frozen.
- 10,000 deterministic whole-cluster bootstrap resamples for baseline margins.
- one-sided paired 95% improvement evidence.
- Holm-corrected critical-cell familywise alpha `0.05`.
- deterministic repeat-evaluation drift is a technical failure.
- equality with a declared safety-envelope boundary is harmful.
- abstention is a no-crossing prediction for confusion accounting and also
  reduces coverage.
- proceed evidence is not permission to train.

## 7. Decisions that require a reviewed proposal

The following are not silently invented by implementation:

1. minimum independent-cluster support for mandatory regression cells.
2. minimum positive and negative cluster support for mandatory envelope-crossing cells.
3. exact action-effect statistic and what counts as measurable support.
4. nested fold count and deterministic fold assignment.
5. exact bootstrap interval and quantile implementation.
6. critical-cell definitions and unsupported-cell handling.
7. Holm hypothesis family construction.
8. stop precedence when more than one stop applies.
9. exact roster of 60 pilot clusters.
10. initial state/profile, load and topology-rotation roster.
11. sensor/noise/source-prefix policies.
12. twelve treatment profiles and values.
13. four normal complete actions.
14. final-set custodian mechanism.
15. campaign-capable direct-ridge representation and solver.
16. peak-memory and runtime ceilings for the timing pilot and canonical fit.
17. target-scale support, held-out out-of-support handling and the minimum
    mandatory target/group coverage.

Each proposal must distinguish values already frozen by source contracts from team-owned experimental choices.

The complete design was approved by Ben on 2026-08-14. The current downstream
machine-readable policy is
`docs/plans/2026-08-14-habitat-v2-forecast-statistical-policy-proposal-v1.json`
with semantic SHA-256
`91e662707c3b4d139cb5bf78f01ef411d4609b12452d8592f30822eaa6e7eced`
and raw-byte SHA-256
`170f8aeaaf8fb938eecb32106c365bab673133f04515e6fada9b6d4a8d07457b`.
Its artifact status is `APPROVED`. The authority boundary admits only these exact
semantic and raw bytes after closed nested-schema validation. Its permissions still
set pilot, scenario and canonical generation, model training, validation access and
publication to false. Policy approval therefore freezes the design without opening
an execution or training path.

The review also found two evaluator discrepancies. They were corrected locally
through RED-GREEN regressions: envelope equality is now harmful, and abstention
now contributes a no-crossing prediction to FN/TN accounting while reducing
coverage. Current ridge alpha selection is still not the proposed balanced
nested-fold procedure. The nested policy semantics and compiled exact identity are
now closed. The local constructor, matched-control plan, permanent exclusion and
resource-preflight loader are implemented. The full runner, output custody and
measured runtime/resource benchmark remain separate gates.

## 8. Planned 60-cluster pilot

The pilot is development-visible evidence, not publication and not canonical training data.

Required invariant shape:

- exactly 60 unique family clusters.
- semantically disjoint from all future canonical clusters.
- all four operating modes and all three declared load regimes covered.
- maximum history/future witness generation once per anchor/action example.
- all nine timing views derived without rerunning or changing physical examples.
- 72 transitions per scenario.
- treatment onset at completed step 25.
- transient treatment active on `[25,49)`.
- persistent treatment active on `[25,73)`.
- anchors at completed steps 16, 40 and 64.
- no `[a-W+1,a+H]` window crosses an operating-mode or treatment boundary.
- one proposal at the anchor followed by `NO_PROPOSAL` continuation.
- one matched `NO_PROPOSAL` continuation per member/repetition/anchor, reused
  across the four action contrasts.
- final HMC outputs and strict control-trace replay remain the execution witness.

The 18,720 proposal runs therefore require 4,680 matched control runs, for
23,400 HMC runs before timing views are sliced. The roster and numeric profile designs are approved. The local deadline slice now
materializes all 60 clusters and 12 treatment mechanisms, enumerates all 23,400
planned HMC continuations, proves the 4,680 matched controls and enforces permanent
pilot exclusion at the plan boundary. This is not a claim that those 23,400 runs
have executed.

The approved numeric packet is
`docs/plans/2026-08-14-habitat-v2-forecast-pilot-profile-action-proposal-v1.json`
with semantic SHA-256
`535cde8c397b115d5dd0b46c257462527f1e3eedfa3fb8560f02e45520854141`
and raw-byte SHA-256
`b6403d9f0763c8c522185c428095e472e8e70acc555a15a32912f4eb606a71a5`.
Its `APPROVED` status freezes design bytes but grants no generation permission.
The implementation audit additionally froze a deterministic `1e-12`-quantum
residual correction after per-zone quantisation, because direct quantisation alone
left volume-weighted drift up to `7.2e-11`. This preserves exact aggregate totals
without changing scientific stress values. The reduced-resource sensitivity gate
has now passed across all 12 mode/load strata using 26 HMC runs and 26 strict
replays, with at least two qualifying public resource channels in every stratum.
The permanently excluded receipt is
`out/habitat-v2-forecast-d2-profile-sensitivity-rehearsal/receipt.json`, semantic SHA-256
`1eaf50a83cfb271066912edc65c149b55a3b07a2ebad64746fe0c259a4af71bf`.
Its raw-byte SHA-256 is
`f42832719acdc7e9f925acafacaaeedd441889fe026c67287c852287ac688a7d`.
It explicitly authorizes neither pilot generation nor model training.

## 9. Canonical campaign after timing qualification

Recommended design, not yet authorised for generation:

```text
4 operating modes
x 3 load regimes
x 20 semantic clusters per stratum
= 240 clusters

x 2 noise repetitions
x 13 members (healthy + 12 treatments)
x 3 anchors
x 4 normal complete proposed actions
= 74,880 samples
```

Predetermined keyed assignment before simulation or outcome inspection:

- TRAIN: 168 clusters / 52,416 samples.
- VALIDATION: 36 clusters / 11,232 samples.
- FINAL: 36 clusters / 11,232 samples.

Pilot identities are forbidden from every canonical split. Healthy/treatment siblings, repetitions, actions, anchors and timing views inherit one whole-cluster assignment.

## 10. Closed gate outcomes

Timing gate:

- `SELECTED`
- `STOP_NO_DEFENSIBLE_TIMING`
- `STOP_POLICY_UNRATIFIED`
- `STOP_EVIDENCE_INVALID`
- `STOP_UNDERPOWERED`

Baseline gate:

- `PROCEED_TO_EXPERIMENT_FREEZE`
- `STOP_NO_ACTION_INFORMATION`
- `STOP_NO_DEFENSIBLE_HEADROOM`
- `STOP_UNDERPOWERED`
- `STOP_EVIDENCE_INVALID`
- `STOP_VALIDATION_ACCESS_INVALID`

The ratified policy must freeze exact stop precedence. No generic `PASS`, truthy value or caller-selected outcome is accepted.

## 11. TDD slice order

### Slice 1: policy authority boundary

RED:

- loading a policy without exact expected semantic and raw-byte SHA-256 values
  must fail.
- a self-consistent `APPROVED` substitute with recomputed self-hash must still fail when its bytes differ from the expected policy.
- a `DRAFT_FOR_REVIEW` policy cannot authorize timing selection.

GREEN:

- strict parser.
- closed immutable policy object.
- canonical self-hash and independently expected byte identity.
- explicit ratification check.

### Slice 2: semantic closure

RED/GREEN one invariant at a time for grid order, support thresholds, confidence procedure, fold construction, correction, stop precedence and prohibited permissions.

### Slice 3: pilot manifest authority

After roster approval, require exact cluster count, stratum coverage, canonical exclusion commitments and phase-safe anchors before any simulation executes.

### Slice 3a: resource qualification

Benchmark the approved ridge representation/solver and canonical packet
representation on a deterministic bounded subset. Record peak resident memory,
wall time, sample throughput and exact output hashes. Refuse the real 60-cluster
pilot when the frozen resource ceiling is exceeded.

### Slice 4: timing evidence

Generate maximum witnesses once, derive all nine views, evaluate nested whole clusters and emit a non-overwritable receipt.

### Slice 5: canonical corpus and baseline gate

Only after timing selection, generate write-once TRAIN/VALIDATION evidence, freeze baseline margins and retain FINAL custody.

## 12. Verification ladder

For each semantic source change:

1. show the focused RED failure.
2. make the narrowest GREEN implementation.
3. rerun the focused test.
4. run D1 forecast regressions.
5. run the full repository suite at semantic closure.
6. run Ruff format/check, compilation and `git diff --check`.
7. generate evidence twice in fresh destinations where generation is in scope.
8. compare exact canonical bytes and manifests.
9. inspect the actual diff and package contents.
10. freeze one local candidate for one bounded read-only review.

No source-tree pass is upgraded into package or publication evidence without a fresh installed-artifact smoke test.

## 13. Non-goals

D2 policy-boundary work does not:

- train a neural network or any learned candidate.
- select a learned architecture.
- expose evaluator-only targets to adapters.
- modify deterministic HMC authority.
- weaken replay, split or leakage checks.
- open or generate a real withheld FINAL corpus.
- claim native Arm execution, latency, power or thermal evidence.
- modify `main` or `origin/alex/ai-2`.
- push, open a PR, merge, tag, release, deploy or submit.

## 14. Version and publication

If eventually published as shipped package behaviour, the D2 qualification capability has intended version impact `minor`. The exact bump is deferred to the repository's release decision. The current local planning/policy slice does not change `0.8.0` and creates no release artifact.
