# Gate 2 acceptance receipt: corpus-v2 contract

**Status:** accepted on 2026-07-29.

Gate 2 establishes an auditable boundary for later corpus generation and model
evaluation. It does not establish classifier quality, model generalisation, or
Arm64 performance.

## Contract accepted

- The independent split unit is a **scenario family**, never an overlapping
  window or replay from that family. A reference or fault replay may not appear
  in more than one split.
- Every family binds a fault-free reference and exactly one faulty scenario to
  the frozen `model_input_v1` selector/topology contract.
- A fault becomes label-eligible at the first equal-tick difference between
  paired `model_input_v1` float32 vectors, not at hidden simulator injection
  time.
- A window that straddles observable onset is labelled `excluded_transition`.
  It updates a stateful detector's history but is excluded from training,
  accuracy, confusion matrices, class support, and scored totals.
- Corpus-v2 evaluation rejects rows with missing or mismatched model-input
  contract metadata, non-finite or non-`float32[24]`-compatible feature vectors,
  and scores only an explicitly selected split.

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
uv run --extra dev python -m pytest -q  -> 252 passed in 3.43s
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
