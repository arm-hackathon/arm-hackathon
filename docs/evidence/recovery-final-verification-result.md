# Deterministic recovery blind-final verification result

Status: **PASS**
Blind run completed: 2026-08-09
Frozen evaluated source commit: `d1d39d04d5c2bb2c8a7d7c32eb2a77faa518df26`
Branch: `ben/recovery-policy-candidate`

## Decision

The deterministic recovery candidate passes the predeclared blind-final safety and physical-benefit contract.

AEOLUS therefore retains deterministic recovery authority. The final evidence does not justify training a model to replace or override this policy. Arm export and optimisation remain blocked because this is a deterministic closed-loop victory, not a reproducible learned-model victory.

## Final corpus

- Suite role: `final`
- Scenario families: **252**
- Four-arm traces: **1,008**
- Harmful physical airflow families: **79**
- Transient physical fault families: **72**
- Frozen-sensor families: **36**

Fault-family counts:

- blocked path: 72
- gradual primary-fan degradation: 72
- transient blocked path: 36
- transient gradual primary-fan degradation: 36
- frozen sensor: 36

## Safety result

All predeclared safety gates passed.

- Harmful physical faults protected: **79/79**
- Missed harmful physical faults: **0**
- Healthy governed protection entries: **0**
- Frozen-sensor protection activations: **0**
- Wrong-zone actions: **0**
- Transient families with exactly one protection episode: **72/72**
- Transient families with zero handback recurrence: **72/72**
- Transient families with zero handback timeout: **72/72**
- Transient families ending with acknowledged physical zero: **72/72**
- Invariant violations: **0**

## Physical-benefit result

All predeclared benefit gates passed.

- Eligible families with a defined integrated-excess denominator: **79**
- Families with positive integrated CO2-excess reduction: **72/79** (**91.139%**)
- Median integrated CO2-excess reduction: **80.396%**
- Minimum reduction: **0%**
- Maximum reduction: **100%**
- Median total-delivery delta: **+226.430 abstract airflow units**
- Healthy-reference non-regression: **252/252**

Seven gradual-degradation families produced exactly zero integrated-excess improvement. None were worsened. They remain a documented detection-latency limitation and a possible bounded target for future earlier-risk prediction, but they do not justify learned recovery authority under the frozen contract.

## Frozen policy settings

```text
entry shortfall ratio                 0.10
entry isolation margin                0.05
entry persistence                     2 ticks
exit residual ratio                   0.06
soft handback-abort residual ratio     0.08
soft handback-abort persistence        2 ticks
```

Full settings SHA-256:

`e6defb9478019869f62a12b1f7de3934776f0512e6c3692454b054545b4afe75`

## Evidence and provenance

Local evidence directory:

`C:/Users/Nxiss/state/aeolus-research/recovery-final-blind-d1d39d0/`

Artifacts:

- `recovery-evidence.json`
- `final-verdict-summary.json`
- `final-run-provenance.json`
- `families.json`
- `generated/`
- `traces/`

Hashes:

- Receipt file SHA-256: `1923e8c8ad1cc476f8872046af48002e6c1f7828b8568db70232599a30c58459`
- Receipt canonical self-hash: `dd777caf37856e114723f1738a5072e2796051d7faa0e945d88127cd5a92a69a`
- Final-verdict summary SHA-256: `9fc5ea6d96c44584b8aa65f2a4b7a6da8ac9953697b8e606c457b6ff4dc6fe38`
- Final-run provenance SHA-256: `7829d6cd6172ac09f9c15c669e6b3747e5a01e833f4a6cb25de792d0d8d79476`
- Source-file manifest SHA-256: `e35aa9dd2346597bc545d0da7c27321edbce83a0d217fd01f195525a62d45695`
- Final family manifest SHA-256: `26b46fa0f78e70cc1a2449ef3c84215c39f6afa3a00616ff52b3ca2fb471685c`

The receipt's canonical self-hash was recomputed after the run and matched exactly. The tracked final specification and the preserved recovered copies differ only in JSON whitespace/line endings; their parsed JSON documents are equal and have the same canonical semantic hash.

## Verification before freeze

The exact candidate bytes were verified before the final run:

- focused recovery tests: **52 passed**
- complete locked Python 3.11 suite: **470 passed**
- `compileall`: passed
- `git diff --check`: passed
- one bounded independent read-only review was performed; its concrete lifecycle-counter finding was corrected and regression-tested before the candidate commit

## Next-step gate

Do not train a learned recovery authority from this result.

A future learned component is justified only for a bounded advisory task with its own evidence contract, for example predicting the seven zero-benefit latency cases earlier while preserving deterministic actuator authority and a strict abstention path. Arm export and optimisation may begin only after such a learned component demonstrates reproducible closed-loop value on a separate untouched corpus.
