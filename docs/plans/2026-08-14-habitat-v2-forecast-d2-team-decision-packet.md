# Habitat V2 Forecast D2 Team Decision Packet

Status: DESIGN AND EXACT POLICY APPROVED. NOTHING IN THIS FILE AUTHORIZES PILOT GENERATION

This packet separates scientific choices from implementation facts so review can
focus on decisions that genuinely alter the experiment. Its verified repository
foundation is D1 candidate `c01dec538a73ce7baaf1ee460fff4ab5f3bbfda3`.
It does not authorize pilot generation, validation access, learned-model
training, deployment or actuator authority. Deterministic HMC remains the sole
actuator authority.

## Decision 1: campaign-capable ridge comparator

### Demonstrated fact

At the maximum timing pair `(W=16,H=8)`:

- the D1 full contract flattening contains 34,987 input features per sample.
- the proposed TRAIN split contains 52,416 samples.
- a float64 design matrix alone is approximately 13.66 GiB.
- the current dual Gram matrix alone is approximately 20.47 GiB.
- the planning host has 15.36 GiB physical RAM.
- closing applications cannot make the current dense dual implementation fit.

Local arithmetic receipt:
`out/habitat-v2-forecast-d2-resource-preflight/ridge-resource-estimate.json`
with self-hash
`eec36ca3269f22f12f877c54b95096c6dd9e76a3ec02d3e36ba690fad7f528d2`.
It performs allocation arithmetic only and is not a peak-RSS benchmark.

### Options

A. Keep all 34,987 flattened features and build a deterministic iterative or
chunked solver.

B. Add a compact autoregressive direct ridge using only causal public estimates
of the 51 evaluator targets over the selected history plus the same 27-value
proposed action. At `W=16`, this is `16*51+27 = 843` features. Use a deterministic
primal closed-form solve when features are fewer than samples.

C. Freeze a deterministic projection/hashing scheme for the full model-facing
contract, then fit ridge on the projection.

### Recommendation

**B: compact autoregressive direct ridge.**

Why:

- it is an interpretable and strong forecasting baseline rather than a second
  high-dimensional model.
- every input remains causal and already available through the frozen public
  projection.
- action-aware and action-blinded forms remain exactly paired.
- the primal matrix at maximum timing is only `843 x 843`.
- no random projection, collision policy or iterative convergence tolerance is
  introduced.
- the learned candidate still receives the full frozen model-facing contract.

Persistence, linear extrapolation and the D1 full-flatten reference remain
unchanged. The compact ridge is additive D2 behavior.

Team decision: `APPROVED IN PRINCIPLE BY BEN, 2026-08-14`

Fairness safeguard: the compact representation is additive and its exact
feature set is not frozen until it is evaluated against the existing
full-contract ridge on an identical manageable subset. If the compact form is
materially weaker because it omits predictive public information, it cannot be
the comparator used to claim learned-model headroom.

## Decision 2: deterministic campaign storage

### Demonstrated fact

Measured from the byte-exact D1 candidate packet:

- `samples.jsonl`: 210,099 raw bytes -> 31,729 deterministic gzip bytes
  (`15.10%`).
- representative 24-step control traces: approximately 202.8 KB raw -> 20.3 KB
  deterministic gzip (`10.00%`).
- `replay_witnesses.jsonl`: 40,484 raw bytes -> 6,047 deterministic gzip bytes
  (`14.94%`).

The real-byte smoke artifacts and self-hashed receipt are under
`out/habitat-v2-forecast-d2-storage-smoke/`. Receipt self-hash:
`5c5155bc4c4aefa89508c9da3ded363565cf1a09322b7a399c4f1102bcb3a6d7`.

The C: drive has 57.96 GiB free at review-packet creation time. Naively storing
uncompressed 72-step traces and maximum timing tensors can exceed practical
local custody limits.

### Recommendation

Store canonical JSON/JSONL bytes using deterministic gzip with:

- compression level 9.
- gzip `mtime=0`.
- no platform-varying original filename metadata.
- SHA-256 of the uncompressed canonical bytes.
- SHA-256 of the compressed artifact bytes.
- uncompressed byte length and compressed byte length.
- decompression followed by existing strict canonical parser/replay checks.
- two fresh generations required to produce identical compressed bytes.

This changes storage representation only. It does not change records, target
truth, replay semantics, HMC output or source custody.

Team decision: `APPROVED BY BEN, 2026-08-14`

## Decision 3: pilot stratum shape

### Proposed invariant

Use exactly five public pilot clusters in each mode/load stratum:

```text
4 modes x 3 load regimes x 5 semantic profile clusters = 60 clusters
```

Modes already implemented:

- `occupied`
- `eva_transition`
- `contingency`
- `dormant`

Proposed load regimes:

- `LOW`
- `NOMINAL`
- `HIGH`

Ratified semantic profile roles within every stratum:

1. balanced nominal initial state.
2. thermal/air-processing skew.
3. crew metabolic and humidity skew.
4. pressure-inventory skew.
5. reduced resource-inventory initial state.

Every pilot cluster receives a `pilot-v1` semantic namespace that is forbidden
from canonical TRAIN, VALIDATION and FINAL family derivation.

The approved 60-cluster identity contract is
`docs/plans/2026-08-14-habitat-v2-forecast-timing-pilot-roster-proposal-v1.json`
with semantic SHA-256
`9514a25548d95047f3e707d1f2b27c76c3b09378653ecd270cdc9ae2845b06d1`
and raw-byte SHA-256
`357ad3286cf80ee1b582096b8251f076e4d111b84f78c51c273a9a89d4921528`.
All 60 entries have `canonical_corpus_eligible=false`. Approval freezes the
roster design but does not authorize generation.

Repository audit result: topology rotations are representable only by changing the
scenario topology and therefore the frozen observable-topology identity
`b0246a9dc8f847c3236068c8e1eeeddb31809a680e6133eaf038ea197d6e10e6`.
The ratified design retains one fixed topology and obtains five independent
clusters through declared initial-state and load-profile values only.

The exact approved numeric packet is:

`docs/plans/2026-08-14-habitat-v2-forecast-pilot-profile-action-proposal-v1.json`

- semantic proposal SHA-256:
  `535cde8c397b115d5dd0b46c257462527f1e3eedfa3fb8560f02e45520854141`
- raw-byte SHA-256:
  `b6403d9f0763c8c522185c428095e472e8e70acc555a15a32912f4eb606a71a5`
- ratification status: `APPROVED`

It freezes exact `0.85/1.00/1.15` load multipliers and five fixed public
profiles: nominal, thermal spatial skew, metabolic/latent spatial skew,
pressure-inventory skew and reduced resource inventory. The pressure profile
supports no pressure-driven airflow claim because zone pressure does not enter
the V5 air-network solver. The reduced-resource profile supports inventory
diversity only. Its permanently excluded sensitivity rehearsal passed across all
12 mode/load strata with at least two qualifying public resource channels in each
stratum. The self-hashed receipt is
`out/habitat-v2-forecast-d2-profile-sensitivity-rehearsal/receipt.json`
with semantic SHA-256
`1eaf50a83cfb271066912edc65c149b55a3b07a2ebad64746fe0c259a4af71bf`
and raw-byte SHA-256
`f42832719acdc7e9f925acafacaaeedd441889fe026c67287c852287ac688a7d`.
This rehearsal authorizes neither pilot generation nor model training. Spatial load transformations preserve each
mode/regime total. The implementation audit found that direct 12-decimal
quantisation could exceed the stated `1e-12` conservation tolerance. The packet
therefore freezes a deterministic whole-quantum residual correction that preserves
both per-value quantisation and the exact aggregate total. No scientific stress
value or threshold changed.

The local deadline slice now materializes the approved 60-cluster roster and all
12 treatment mechanisms, enumerates matched controls, validates permanent
exclusion, loads the resource-preflight boundary and executes the excluded
sensitivity rehearsal. The complete 23,400-run runner, output custody and measured
runtime benchmark remain separate gates.
Every numeric load regime and semantic profile carries its exact roster key, so
the join from each of the 60 roster rows is explicit rather than positional or
name-inferred.

Identity direction is acyclic: the profile packet binds the roster and
foundation, the later executable statistical policy may bind the approved
profile packet, and the final freeze manifest pins all raw bytes. The profile
packet does not bind the later policy.

Team decision: `APPROVED BY BEN ON 2026-08-14`; execution remains blocked on the
listed implementation gates.

## Decision 4: treatment roster

The V5 engine currently supports these relevant fault mechanisms:

Physical delivery/effectiveness:

- fan speed degradation.
- branch resistance increase.
- damper jam.
- scrubber capture/effectiveness degradation.
- condenser removal/effectiveness degradation.
- zone cooling delivery/effectiveness degradation.
- zone oxygen delivery/effectiveness degradation.

Observable instrumentation:

- primary/secondary environmental sensor bias drift.
- primary/secondary environmental sensor stuck.
- operational-feedback sensor bias drift.
- operational-feedback sensor stuck.

Proposed 12 treatment roles plus one healthy sibling:

1. fan degradation.
2. branch blockage.
3. damper jam.
4. scrubber capture degradation.
5. condenser removal degradation.
6. cooling delivery degradation.
7. oxygen delivery degradation.
8. primary environmental sensor bias.
9. secondary environmental sensor stuck.
10. battery/resource feedback bias.
11. branch-airflow feedback stuck.
12. one declared compound physical-plus-observation treatment.

The exact values in the approved profile/action packet are ratified as
deterministic simulator stress levels. They are not hardware-calibrated failure
rates.

Both transient and persistent treatments begin at completed step 25. Transient
members clear before completed step 49, while persistent members continue through
completed step 72. Anchor 16 is a common pre-treatment window, anchor 40 measures
both treatment kinds, and anchor 64 separates transient recovery from persistent
exposure. No maximum timing window crosses a treatment boundary.

### Bounded mechanism smoke, not scientific approval

One fixed occupied source profile was used to test these exact candidate values
in memory for 72 steps:

1. fan multiplier `0.75 -> 0.55`.
2. laboratory branch resistance multiplier `2.0 -> 4.0`.
3. jam `airlock_suitport_supply_damper` at its preceding achieved position.
4. scrubber capture multiplier `0.75 -> 0.50`.
5. condenser removal multiplier `0.75 -> 0.50`.
6. air-processing-bay cooling multiplier `0.70 -> 0.50`.
7. laboratory oxygen-delivery multiplier `0.75 -> 0.50`.
8. common-galley primary CO2 bias `+8 -> +24 ppm`.
9. equipment-power-bay secondary temperature sensor stuck.
10. battery-state-of-charge feedback bias `+0.01 -> +0.02`.
11. laboratory branch-airflow feedback stuck.
12. air-processing-bay cooling multiplier `0.70 -> 0.50` combined with
    cooling-delivery feedback bias `+30 -> +100 W`.

All 12 mappings parsed, ran twice byte-identically, passed strict replay and
obeyed the proposed half-open boundaries. This proves only that the mechanisms
are executable. It does not establish suitable severity, action information,
crossing support or multi-profile stability.

The compound member is not an identifiable main-effect treatment. If retained,
it must be labelled as a declared interaction/OOD stress member with both
constituent profiles preserved and no causal attribution to either constituent.

The declared 13-member arithmetic cannot include both a transient and a
persistent copy of every one of the 12 treatments. The current recommendation
is a frozen balanced assignment matrix rather than doubling the corpus:

- each cluster contains six transient and six persistent treatment members.
- each treatment role is transient in exactly 30 pilot clusters and persistent
  in exactly 30 pilot clusters.
- assignment is fixed before simulation. With zero-based indices in the frozen
  roster, a member is transient exactly when
  `(cluster_index + treatment_index) mod 2 == 0`. It is persistent otherwise.
- both noise repetitions inherit the same interval kind so they remain genuine
  repetitions.

This preserves the 18,720-sample pilot arithmetic while preventing treatment
kind from being globally confounded with interval duration.

No implementation may infer them from this list.

Team decision: `APPROVED BY BEN ON 2026-08-14`.

## Decision 5: four proposed actions

### Existing frozen D1 catalogue

- `normal-occupied-v1`
- `normal-eva_transition-v1`
- `normal-contingency-v1`
- `normal-dormant-v1`

### Recommendation

Use these same four complete normal actions for the public timing pilot unless a
scenario audit demonstrates an HMC-domain or support defect. Each action is
applied at the anchor and followed by `NO_PROPOSAL`. A matched all-`NO_PROPOSAL`
continuation is generated once per scenario-member/noise/anchor and reused for
paired action-effect evidence.

The declared action-run count is 18,720. Matched controls add 4,680 HMC runs,
making 23,400 HMC runs before timing views are sliced. The two repetitions must
use two frozen `sensor_model.random_seed` values per cluster. Each repetition's
seed and HMC reset nonce are reused across healthy/treatment siblings and its
paired proposal/control variants. The approved profile/action packet freezes exact
SHA-256 domain-separated derivations. It proves 120 noise-seed and 360 reset-nonce
identities are unique under the approved roster.

Team decision: `APPROVED BY BEN ON 2026-08-14`. HMC remains the sole authority,
and all four one-shot proposals are tested in every source mode against one
byte-matched reusable `NO_PROPOSAL` control.

## Decision 6: statistical thresholds and stop precedence

### Review and ratification state

The earlier read-only reviews returned `REVISE_BEFORE_FREEZE`. Ben then reviewed
every material design choice and approved the complete design on 2026-08-14.
The current machine-readable downstream policy is:

`docs/plans/2026-08-14-habitat-v2-forecast-statistical-policy-proposal-v1.json`

- semantic policy SHA-256:
  `91e662707c3b4d139cb5bf78f01ef411d4609b12452d8592f30822eaa6e7eced`
- raw-byte SHA-256:
  `170f8aeaaf8fb938eecb32106c365bab673133f04515e6fada9b6d4a8d07457b`
- artifact status: `APPROVED`
- upstream roster/profile statuses: `APPROVED`

The exact policy bytes are now compiled into the authority boundary and pass closed
nested-schema validation. Approval freezes the experimental design only. The policy
itself keeps pilot, scenario and canonical generation, model training, validation
access and publication false, so it cannot authorize campaign execution.

### Already fixed by the accepted design

- timing grid `W={4,8,16}`, `H={2,4,8}`.
- 60 independent public pilot clusters.
- same maximum witnesses sliced for all nine timing pairs.
- 10,000 deterministic whole-cluster bootstrap resamples.
- one-sided paired 95% improvement evidence.
- Holm familywise alpha `0.05` for critical cells.
- zero replay/identity failures and complete target truth.
- shortest defensible history per horizon.
- largest viable horizon.
- no learned training from a proceed result without a separate experiment
  freeze.

### Reconciled recommendation requiring ratification

1. Require all 60 clusters and all five clusters from every mode/load stratum
   for aggregate regression, action-pair and coverage evidence. Missing evidence
   is not silently dropped.
2. Require at least 24 independent clusters with new harmful crossings and 24
   independent clusters with non-harm opportunities. Every stratum must
   contribute at least one cluster to each polarity.
3. Use five balanced outer folds. Every outer fold tests one cluster from each
   stratum and trains on 48 clusters. Use four balanced inner folds within each
   outer training set, with 36 inner-training and 12 inner-validation clusters.
4. Fit every scale, preprocessing value and ridge alpha from the applicable
   training clusters only.
5. Use 10,000 deterministic stratified whole-cluster bootstrap resamples with
   NumPy `PCG64DXSM` and `quantile(method="linear")`.
6. Define one 99-claim timing Holm family at familywise alpha `0.05`. It contains
   36 action-effect claims, nine action-aware versus blinded claims, 18 ridge
   non-inferiority claims, 27 coverage/FNR/FPR claims and nine history-equivalence
   claims.
7. Keep per-target, per-offset, per-envelope and regime/profile breakdowns
   descriptive. Sixty independent clusters cannot support an honest corrected
   inferential claim for every granular cell.
8. Require every normal action to exceed its matched no-proposal noise floor by
   a normalized material effect of `0.02` with affirmative corrected evidence.
9. Require action-aware ridge to improve over action-blinded ridge by more than
   `0.005` normalized MAE with affirmative corrected evidence.
10. Permit at most `0.005` normalized-MAE loss for ridge non-inferiority and at
    most `0.02` loss/increase for coverage, FNR and FPR. Failure to prove
    non-inferiority is not proof of safety.
11. Treat equality with an envelope boundary as harmful. Treat abstention as no
    crossing prediction, which is FN for positive truth and TN for negative
    truth, while also reducing coverage.
12. Use this exact timing precedence:
    `STOP_POLICY_UNRATIFIED`, `STOP_EVIDENCE_INVALID`, `STOP_UNDERPOWERED`,
    `STOP_NO_DEFENSIBLE_TIMING`, `SELECTED`.
13. Use this exact baseline precedence:
    `STOP_EVIDENCE_INVALID`, `STOP_VALIDATION_ACCESS_INVALID`,
    `STOP_UNDERPOWERED`, `STOP_NO_ACTION_INFORMATION`,
    `STOP_NO_DEFENSIBLE_HEADROOM`, `PROCEED_TO_EXPERIMENT_FREEZE`.
14. Refuse a failed resource preflight before generation without emitting a
    scientific timing or baseline outcome.

### Implementation disposition after review

Resolved locally with RED-GREEN regressions:

1. Envelope equality now uses inclusive harmful boundaries. Both low and high
   equality cases pass.
2. Abstention now contributes an all-false crossing prediction, producing FN for
   positive truth and TN for negative truth while coverage remains zero.
3. The nested policy schema, source bindings and exact semantic/raw identities are
   validated, and only the approved policy semantic identity is compiled.

Still blocked:

4. Current ridge alpha selection uses leave-one-cluster-out rather than the
   proposed balanced nested folds.
5. The current fixture runner does not create matched no-proposal controls or
   enforce pilot exclusion.
6. The measured resource benchmark, complete D2 custody packet and campaign runner
   have not opened the generation permissions.

### Human choices resolved

Ben approved on 2026-08-14:

- the `24/24` crossing support threshold.
- the `0.02` action materiality threshold.
- the `0.005` ridge advantage/non-inferiority margin.
- the `0.02` coverage/FNR/FPR margin.
- one global 99-claim timing Holm family.
- the narrow inferential universe with granular cells descriptive.
- all four actions as separately mandatory.
- the exact stop precedence above.

Team decision: `DESIGN AND EXACT POLICY APPROVED BY BEN ON 2026-08-14`; campaign
execution remains blocked on the implementation and resource gates listed above.

## Decision 7: resource ceilings

### Recommendation

Before the real pilot:

1. run an approved deterministic subset benchmark.
2. record peak process RSS, wall time, throughput and packet bytes.
3. project the full 60-cluster campaign using the measured upper bound.
4. refuse generation when projected disk use would leave less than a frozen
   safety reserve.
5. refuse a baseline fit when the deterministic allocation estimate exceeds its
   frozen process-memory budget.
6. optionally close approved nonessential applications immediately before the
   benchmark, never user applications blindly.

Current large processes include Discord, the active Hermes Python process,
Brave, Steam helpers and overlays. No process has been stopped.

Exact memory, runtime and disk reserve numbers remain pending the bounded
benchmark.

Team decision: `APPROVED DIRECTION`; the benchmark must freeze exact ceilings
before generation.

## Decision 8: final-set custody

A real FINAL corpus must not be visible to the development loop. The 240-cluster
assignment is committed before outcomes, but FINAL materialization and the split
key/custodian mechanism need an identified human or isolated process outside the
candidate-training environment.

Team decision: `APPROVED CUSTODY BOUNDARY`. No FINAL generation is authorised,
and the custodian mechanism remains an implementation gate.

## Current decision bundle

Approved by Ben on 2026-08-14:

1. the learned component is an action-conditioned forecaster only; deterministic
   HMC remains the sole actuator authority.
2. compact target-history-plus-action primal ridge direction, subject to the
   stated fairness comparison before its exact feature contract is frozen.
3. deterministic gzip storage with raw and compressed hashes.
4. 60 permanently excluded clusters from four modes, three load regimes and five
   approved semantic profiles on one fixed topology.
5. the corrected pressure-inventory and reduced-resource claim boundaries, with
   the permanently excluded reduced-resource sensitivity rehearsal now passed in
   all 12 strata without opening pilot-generation or training authority.
6. the 72-step schedule, anchors 16/40/64, onset 25, balanced transient/persistent
   assignment and exact twelve treatment severities.
7. all four frozen complete normal actions in every mode, plus one matched
   no-proposal control per member/repetition/anchor.
8. the complete statistical design, margins, 99-claim Holm family and stop
   precedence.
9. benchmark-before-generation resource policy.
10. FINAL custody outside the development environment.

Approval freezes the design and exact policy bytes only. The policy permissions do
not authorize pilot generation, training, validation/FINAL access, publication,
deployment or learned actuator authority.
