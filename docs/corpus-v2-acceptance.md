# Gate 2 acceptance receipt: corpus-v2 contract

**Status:** accepted on 2026-07-29; contract boundary hardened on 2026-07-31.

Gate 2 establishes an auditable boundary for later corpus generation and model
evaluation. It does not establish classifier quality, model generalisation, or
Arm64 performance.

## Contract accepted

- The independent split unit is a **scenario family**, never an overlapping
  window. A reference or fault replay may not appear in more than one split,
  and an exact reference/fault pair may appear in only one family. Identity is
  canonical scenario JSON content, not a mutable filename or path.
- The corpus-v2 CLI requires a caller-supplied, previously recorded
  family-manifest SHA-256 and rejects a supplied manifest that does not match;
  a self-recomputed manifest hash is traceability, not an authority boundary.
- Every family binds a fault-free reference and exactly one faulty scenario to
  the frozen `model_input_v1` selector/topology contract.
- A fault becomes label-eligible at the first equal-tick difference between
  paired `model_input_v1` float32 vectors, not at hidden simulator injection
  time.
- A window that straddles observable onset is labelled `excluded_transition`.
  It updates a stateful detector's history but is excluded from training,
  accuracy, confusion matrices, class support, and scored totals.
- Corpus-v2 evaluation validates an exact row schema and rejects missing or
  unexpected fields; duplicate or malformed row identities; incomplete
  reference/fault streams or window inventories for any manifest-declared family;
  non-finite or non-`float32[24]`-compatible feature vectors, or feature
  windows whose values differ from the recomputed validated replay; and every
  row whose family, split, role, observable onset or derived label disagrees
  with immutable evidence recomputed from the validated family manifest. It
  then scores only an explicitly selected split.

## Fixture evidence

The generated contract fixture contains three test-only families, each with a
healthy reference and a faulty replay. It is deliberately small and must not be
presented as a performance or generalisation result.

| Property | Measured value |
|---|---:|
| Total windows | 138 |
| Scored windows | 134 |
| Transition-excluded windows | 4 |
| Family-manifest SHA-256 | `828880e3257036ff2897a6cc2668c25b87734f8c57004ed36e62b2b6d66f6541` |
| Generated-manifest SHA-256 | `fa3175b2964d37bc8e30d51202be780c43163a6cfa0874f7ba48c06f0d90355c` (canonical JSON with this field omitted) |
| Families by split | train=0, validation=0, test=3 |
| Model input | `model_input_v1`, `float32[24]` |

| Family | Hidden injection tick | First observable tick |
|---|---:|---:|
| Primary-fan degradation | 20 | 21 |
| Blocked path | 30 | 30 |
| Frozen sensor | 30 | 31 |

The rule baseline scores all 134 eligible fixture windows correctly, with
observable-onset detection latency of 10 / 9 / 9 ticks for blocked path, frozen
sensor, and primary-fan degradation respectively. This **100% fixture result is
not a performance claim**: the fixture contains only three hand-designed,
deterministic families and is intended to prove the label/evaluation contract.
A scenario sweep with held-out families is required before comparing any learned
model with the baseline.

## Verification receipt

Run from repository root on branch `ben/corpus-v2-contract`:

```text
uv run --extra dev python -m pytest -q  -> 264 passed
uv run ruff check .                     -> All checks passed!
git diff --check origin/main            -> clean
```

The generated fixture artifact remains untracked under
`out/corpus-v2-contract/`.

## Gate boundary

Gate 2 authorises the next slice: generate a varied scenario sweep and retain
family-held-out train/validation/test partitions. It does not authorise model
quality claims, ONNX export, quantisation claims, Arm benchmarks, or autonomous
actuator control.

## Subsequent experimental status

Branch `alex/ai-2` subsequently used this boundary for the historical schema-v9
840-family IID/stress experiment. Protocol v3 supersedes that work for current
selection and final evidence: it uses separate development and fresh final
family suites with a frozen policy. This historical Gate-2 acceptance receipt
remains unchanged; it does not establish model quality, ONNX parity, INT8, Arm
benchmark or autonomous-control evidence. See
[`protocol-v3-acceptance.md`](protocol-v3-acceptance.md) for current evidence.
