# Recovery development gate C4 — Outcome B acceptance record

## Status

**Closed as a reproducible negative development result.**

This record covers only the C4 deterministic recovery-development gate. It does
not accept a recovery controller, qualify an adviser, or authorise a final-suite,
hardware, cloud, deployment, or real-control action.

| Field | Value |
|---|---|
| Source commit | `74154956d64309f067ada7593e2ef8786d140b4e` |
| Base commit | `88321f1d7bda00d215d81a535eaeafc9fa72b5c0` |
| Canonical output | `out/overnight/recovery-development-c4-a` |
| Suite role | `development` |
| Families | 756 |
| Counterfactual arms per family | 4 |
| Traces | 3,024 |
| Evidence version | `aeolus_recovery_evidence_v1` |
| Canonical evidence self-hash | `1cbb9d428824f57c500b4a1ac3859b4ea6ef0a0dd4e70012b2e6c35d230a1730` |
| Source worktree at execution | clean |
| Python | CPython 3.11.15 |

## Frozen four-arm contract

Each recovery family consists of the same validated base condition under exactly
these arms:

```text
reference_reserve_off
reference_governed
fault_reserve_off
fault_governed
```

The reference/fault pair must share non-fault configuration. The runner writes
new traces only, binds scenarios, source, sweep, settings, and trace hashes into
a receipt, and refuses an output directory that already exists.

A schema-v10 scenario has disjoint primary and reserve connection IDs and one
paired reserve path per non-processing zone. `run_recovery_scenario()` applies
the authority decision only on the next tick after an observable completed-tick
record. The deterministic supervisor is the sole active reserve-command owner;
a learned component has no command ownership.

## Measured gate outcome

### Safety: false

The following safety subgates passed across their declared populations:

- zero invariant violations (756 families);
- zero reserve delivery in reserve-off arms (756);
- no healthy governed `PROTECT` transition (756);
- no frozen-sensor `PROTECT` transition (108);
- preactivation physical parity (756); and
- no failed-reserve rearm (756).

The required transient handback acknowledgement subgate failed for the 216
transient families. It requires a handback state and a physical-zero
acknowledgement within the frozen 36-tick bound. A command becoming zero is not
sufficient evidence of physical zero.

### Benefit: false

The following benefit subgates were individually true:

- median integrated-excess improvement: `0.5051472489053273` against a `0.05`
  threshold;
- validation improvement fraction: `0.6794871794871795` against a `0.60`
  threshold;
- median total-delivery delta: `206.03607489553315`; and
- healthy-reference non-regression across 756 families.

However, the mandatory `physical_reserve_delivery_for_benefit` subgate failed.
There were 78 eligible, defined validation families, and it required positive
fault-governed reserve delivery for every one. Another 138 eligible families had
an undefined benefit denominator and are reported separately rather than being
silently counted as successes.

Therefore the aggregate benefit result is false. It would be methodologically
wrong to call the positive aggregate submetrics recovery efficacy.

## Reproduction and falsification receipts

- Canonical development run A: 756 families, 3,024 traces, source-clean receipt
  self-hash verified.
- Targeted stress matrix: 11 tests passed in 3.42 seconds. Coverage includes
  reserve delivery failure, recurrence, dropout, ambiguity, malformed authority,
  saturation, high noise/drift, and denominator-zero handling.
- Duplicate A/B reproduction: comparison receipt
  `aadbcd25faffe70a30311187e999c2d8c16c5a61d68a8e8df34606ea6f653343`;
  3,801 files and all 3,024 traces were byte-identical. The local receipt is
  `out/overnight/recovery-development-c4-reproduction.json` and explicitly
  records `all_files_byte_identical: true`.
- Clean-checkout reproduction: comparison receipt
  `2264c85fc85dd63ee99f853523af0d2dec3c67b861f67e61db05d7c9cb0ef733`;
  same evidence identity, file count, and source-clean state. The local receipt
  is `out/overnight/recovery-development-c4-clean-checkout-reproduction.json`.

These receipts establish deterministic reproduction of the negative outcome.
They do not change a required failed gate into acceptance.

The current duplicate helper compares every generated file, including corpus,
manifest, receipt, and trace files. It records both file counts and the exact
relative paths of any mismatch. Current provenance additionally binds the lock
file and runtime package versions and rejects a source change during generation.
These are forward-looking hardening changes; the C4 receipts above remain bound
to source `74154956d64309f067ada7593e2ef8786d140b4e`.

## Post-freeze integration package

Version `0.2.1` packages the source after the dependent bounded-response fixes
were integrated. It is a distinct package identity and does not replace the
source-pinned C4/C11 artifacts, rerun either evidence gate, or change this
record's negative safety and benefit outcome.

## C5 decision

C5 is closed by the C4 safety and physical-delivery failures. No recovery adviser
corpus, training, tuning, ONNX export, integration, or final-suite work was run.
No post-hoc threshold or model change is licensed by this record.

## Single deterministic recovery replay

This small replay exercises the installed recovery API against the checked-in
schema-v10 scenario without generating the C4 corpus. Run it from outside a
source checkout after installing the package, substituting absolute paths:

```bash
python -I -c '
from pathlib import Path
from aeolus.config import load_scenario
from aeolus.scenario import run_recovery_scenario

scenario = Path("/absolute/path/to/recovery_habitat.json")
trace = Path("/absolute/path/to/new-recovery.jsonl")
result = run_recovery_scenario(
    load_scenario(scenario),
    run_id="package-smoke",
    governed=True,
    trace_path=trace,
)
assert len(result.records) == 180
assert trace.exists() and trace.stat().st_size > 0
print(f"records={len(result.records)} trace={trace}")
'
```

The replay proves packaging and deterministic trace generation only. It is not a
benefit, safety, final-suite, model, hardware, or deployment evaluation.

## What this record proves

- the C4 source can generate a hash-bound four-arm development corpus;
- the documented failed safety and benefit gates are reproducible;
- source/receipt provenance and duplicate reproduction are bound to the C4
  source commit; and
- the correct project decision is to preserve the negative result and close C5.

## What this record does not prove

- safe or effective physical recovery;
- accepted authority handback;
- model qualification, AI advantage, or a final result;
- Arm, INT8, cloud, deployment, or physical-system performance.
