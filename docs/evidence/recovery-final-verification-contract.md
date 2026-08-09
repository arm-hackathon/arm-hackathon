# Deterministic recovery final-verification contract

Status: frozen before the blind final run  
Version impact if proposed as a code PR: `patch`

## Purpose

This contract defines the one-time blind decision gate for the deterministic AEOLUS recovery-policy candidate. The final suite must not be used for threshold selection or post-result tuning.

## Frozen candidate

The production `RecoverySettings` defaults are the candidate:

- entry residual ratio: `0.10`
- entry isolation margin: `0.05`
- entry persistence: `2` fresh observations
- exit residual ratio: `0.06`
- handback abort residual ratio: `0.08`
- soft handback abort persistence: `2` fresh observations
- minimum protect dwell: `10` ticks
- recovery-clear persistence: `10` ticks
- reserve command delta: `0.10` per tick
- reserve zero confirmation: existing physical-zero acknowledgement contract

Expected canonical settings SHA-256: `e6defb9478019869f62a12b1f7de3934776f0512e6c3692454b054545b4afe75`.

The final receipt must record this exact settings hash. Any mismatch invalidates the run.

## Write-once protocol

1. Freeze and commit the source and tests.
2. Require a clean worktree.
3. Run the canonical four-arm evidence generator against the untouched `aeolus_sweep_v4` suite whose role is `final`.
4. Evaluate every family under exactly these arms:
   - reference reserve off
   - reference governed
   - fault reserve off
   - fault governed
5. Preserve the generated family manifest, all traces, source hashes, environment receipt, settings hash, run-spec hash and final evidence hash.
6. Do not overwrite the output directory.
7. Do not alter thresholds or rerun the same final suite to turn a failure into a pass. Any later repair must return to development evidence and use a newly designated untouched evaluation corpus.

## Safety gates

All canonical receipt safety gates must pass:

- zero invariant violations
- reserve-off arms deliver no reserve airflow
- healthy governed arms never enter protection
- frozen-sensor faults never receive reserve authority
- every physically harmful airflow fault enters protection
- protection targets only the expected zone
- physical plant state matches its reserve-off counterfactual before activation
- every transient family has exactly one protection episode
- every transient family begins handback and acknowledges physical zero within the bounded window
- zero handback recurrence
- zero handback timeout
- final reserve delivery and actuator positions are physically zero
- failed reserve cannot silently re-arm

Rate limiting, fresh-observation causality and zero-confirmation semantics must also remain covered by the green automated tests at the frozen source commit.

## Benefit gates

All canonical receipt benefit gates must pass on the `final` split:

- every family with defined physical benefit receives actual reserve airflow
- median integrated physical CO2-excess improvement is at least `0.05`
- at least `0.60` of defined final physical-airflow families have positive integrated CO2-excess improvement
- median total delivered airflow does not regress
- healthy reference physical outcomes do not regress

Undefined zero-denominator families are reported separately and cannot be converted into fabricated improvements.

## Known development limitation

One train family and its matching development-validation family had zero integrated CO2-excess improvement. Their unsafe interval ended at tick `81`, while protection began later at ticks `88` and `84`. Reserve airflow was delivered and handback completed, but the action could not erase historical excess that had already accumulated.

This is classified as a detection-latency limitation. The frozen candidate is not tuned further against those known families before the final run.

## Decision rule

### Safety and benefit both pass

Accept deterministic recovery as the safety authority. Do not train a learned model to replace recovery entry, zone selection, reserve command or handback. A model may still be developed for a separately demonstrated diagnostic or forecasting gap, with the deterministic supervisor retaining authority.

### Safety passes, benefit fails because protection is systematically late

Reject the physical-victory claim. The smallest justified learned experiment is a bounded temporal early-warning or risk-forecasting component. It must demonstrate earlier observable warning on whole-run development families without healthy false intervention. It remains advisory to the deterministic governor.

### Safety fails because healthy and harmful behaviour are not separable by the frozen observable contract

Reject the candidate. First distinguish a policy defect, simulation/evidence defect and genuine feature overlap. Train a learned component only if paired observable-onset evidence proves that fixed rules cannot separate the cases without violating the safety gates.

### Safety fails because of implementation, lifecycle or provenance defects

Reject the candidate and repair the deterministic system. This does not justify model training.

## Claims excluded

Even a pass does not establish real spacecraft safety, real building safety, hardware behaviour, universal indoor-air-quality control, or an AI recovery victory. It establishes reproducible closed-loop deterministic recovery in the preserved AEOLUS final simulation protocol.
