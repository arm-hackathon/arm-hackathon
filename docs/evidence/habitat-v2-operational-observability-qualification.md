# Habitat V2 operational observability qualification evidence

**Candidate scope:** local, bounded V5 operational-observability qualification at
package version `0.8.0`. This is a software-contract result, not a physical,
hardware, flight, deployment, or learned-model qualification.

## Contract boundary

- Projected rows contain only completed `step`, `time_s`, `mode`, primary and
  secondary telemetry, their declared disagreement, commanded/actual action,
  and operational feedback.
- Fault receipts, truth telemetry, residuals, effective values, realised loads,
  schedules, seeds, labels, and resource state are excluded before evaluation.
- Every projection is bound to an explicit fixture ID plus its validated
  scenario SHA-256, run ID, and source-trace SHA-256.
- Pair manifests require explicit treatment IDs and bind both scenario hashes and
  run IDs, treatment-profile hash, structural baseline, actuator-feedback
  revision/config, feature manifest, and decision/tolerance contract. A trace
  whose scenario or run identity does not match that manifest is rejected before
  scoring. Reports repeat both validated run IDs and source-trace hashes so their
  causal lineage remains auditable without reopening simulator truth.
- The immutable ordered feature manifest SHA-256 is
  `ea9920963a3a3d50533ac4b20912fbc331a6e45a8ef2d84a958316805b60e9e4`.
  Each descriptor explicitly says whether it is compared, asserted equal, or
  deliberately unscored.

## Decision semantics

The fixed persistence window is two contiguous completed rows. Divergence uses
strict `abs(healthy - treatment) > tolerance`. The report keeps separate answers:

1. abnormality detection;
2. subsystem localisation; and
3. exact identification, deliberately `UNKNOWN`.

For a window beginning at treatment step 2, the first divergence is step 2 but
the decision is made at completed step 3. Detection latency is therefore one
completed step (60 seconds), not zero. The pair manifest also binds the shared
half-open end at step 6. Each report separately records baseline, treatment,
recovery and post-recovery phase bounds, divergent-row counts and persistent
concern; in all declared fixtures compared-feature divergence is absent at
steps 6 and 7, so clearance is decidable at step 7, and the final two rows are
a stable post-recovery tail.

## Qualified fixtures

Fixtures use 10 transitions (11 rows) with treatment `[2, 6)`, two clearance /
recovery transitions, and a stable two-transition post-recovery tail. The
operational trace hashes below are source trace SHA-256 values.

| Fixture | Outcome | Localisation | First / decision / clearance | Pair manifest SHA-256 | Report SHA-256 | Trace SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `fan_degradation` | `SUBSYSTEM_LOCALISED` | `air_network` | 2 / 3 / 7 | `ede845fd499253adcef36cb76e56b9b6fe2791933e9e2759e010a446dcef751f` | `f0508c1eee0055fe0b441d5cb16497c3ecb25aab6b5813817b1e808fbcb42650` | `b778d9a69c4ea208091d9cf372a5afde3999b22f00bfb02658967a84a4426f90` |
| `branch_resistance` | `SUBSYSTEM_LOCALISED` | `air_network` | 2 / 3 / 7 | `9f35a3c99685fd7b7517e871f452d060c3365f883c2e69278d5f1b7a362ca4f9` | `c102cb2e879e44558ebfc8c68e23916e7f74df8517200b0821a5ca2f0a9fd304` | `bdf4746f2495d9699c59ef97e972843e6febd670257a201a293f64649caf8ecb` |
| `damper_jam` | `SUBSYSTEM_LOCALISED` | `air_network` | 2 / 3 / 7 | `8771415a621789f4e5bbf5538e78163df438ac8359b7d56a3fe2210e881df124` | `d4b2e381185aa3f0ef9ca33532fab0b315a6620baddc3a241bcb7ac870751208` | `f053f7c5bc18b9fb4e1a8a34161dec114dc9954fcf52486b2a696f3dec38a9be` |
| `primary_sensor_drift` | `SUBSYSTEM_LOCALISED` | `instrumentation` | 2 / 3 / 7 | `4c7a776f740df25a09a1fe977dbec2a0fe9892139ef82aabfc55d828dd90e5af` | `dca80c3fd898d240d4d0d20f310fc2d6e358ac43bbe315c67140fbd902bb3642` | `be09938d748214a2e8225e5aa40539166fdbb5286065af4a5437698d8b3c02cb` |
| `primary_sensor_stuck` | `SUBSYSTEM_LOCALISED` | `instrumentation` | 2 / 3 / 7 | `acb63deb35c4e91d907084d25796f034780f1a620fee41749e3ad178dd6567c8` | `b6f5cab5f00ffac40741f302df73018b4db88b7cb069e340761fa29e630d8b27` | `4425ea8feb6ffe939b7ec4493dabcee165ab927baaac423e0b14e42f9bc4dd8c` |
| `ambiguous` | `UNKNOWN` | `UNKNOWN` | 2 / 3 / 7 | `ac5b10d522e999d7f02a98ba3a4ae7bde8b1b118b5676f1ea71cfb34c056cc02` | `49053e217b888a10b33c0062b13d739fa832cc5e5bf3c74e32738131079efa8f` | `29a32f69e5e8859ba76932b0e06dfbcba9e1448790b5f3bb9d66006f153ffe60` |

The shared healthy nominal source trace SHA-256 is
`e873afef35fc2f36da8840064d5df68b1eaf0e201200d67d7e5702a214c93fac`.

## Hard negative and aggregate polarity

`healthy_elevated` is evaluated independently against the explicitly narrower
primary-telemetry operational boundary (not by self-comparison). It produced
`NO_OBSERVABLE_CONCERN`, `false_concern=false`, result SHA-256
`a2c8576515c6eb80d291ef358dbcd525cd9cdbda8362b02453e94bd355d3cb41`,
source trace SHA-256
`e53e0ff986a08aecea6ff8728499c3fbf83dcd5bdec5a0c6162238b7b4c22d44`, and
boundary-contract SHA-256
`95c88e97971072031c870958e3540d0d041ed3478fd5ea7a4ad2adfcc796c6eb`.

The six treatment fixtures yield:

- harmful concern coverage: **6/6**;
- healthy false concerns: **0/1**;
- eligible localisation coverage: **5/5**;
- exact-identification coverage: **0/0** (not claimed by contract);
- detected latency / null non-detection: **6 / 0**;
- ambiguous abnormality abstention: **1/1**;
- overclaims: **0/6**.

Aggregate artifact SHA-256:
`a30fe2b5c42d618f7575ba1e89b9977fcbaf486b6ef5a6fb2badbe28ffafee6b`.
It binds the ordered qualification-case/report grading manifest
`32a68e86043ee4de770f4155619d181b0d6cbbd5a018ae874ca4ebc30dab34b0`
and hard-negative result manifest
`442ed9b59b75e5d603d23fda26d9aee172933946ade3189c63683fd16445cd76`.

The complete external qualification packet, containing the six manifests,
operational-provenance bindings, canonical reports, hard-negative result and
aggregate metrics, has SHA-256
`1afed658237fd62404094eac2d50a78b8db9ad19f9b612add9ff37d1b0e3866b`.
It is stored outside the source tree at
`C:/Users/Nxiss/AppData/Local/hermes/cache/aeolus-observability-evidence/qualification-packet.json`.

## Verification receipt

The final freeze reruns focused qualification, full project and Habitat V2
suites, Ruff, `compileall`, locked dependency and diff checks, package build,
isolated installed-package qualification smoke, and duplicate canonical report
byte comparison. The local commit and final package artifact hashes are the
source of the final freeze identity; no publication is part of this scope.
