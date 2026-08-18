# Corpus Datasheet — Historical V2 Pilot Archive

Training/evaluation corpus behind `action_aware_mlp_v1` (see
`MODEL_CARD.md`). Manifest SHA-256
`7c35a2da3a9f902c2994b8b29332f8fafed40b81c49204c3546a75d2b3f76659`.

## Motivation

Provide a frozen, provenance-bound dataset for learning atmosphere dynamics
in the Habitat V2 simulation, sufficient to test whether an action-aware
forecaster adds measurable closed-loop value under a deterministic safety
controller — without any hidden simulator state leaking into model inputs.

## Composition

- **4,680 simulation packets**, each a complete seeded Habitat V2 run with
  operational telemetry projections.
- **60 scenario clusters** spanning operating modes (nominal, EVA
  transition, contingency, dormant), fault families (scrubber degradation,
  fan faults, sensor bias/drift/stuck, damper jams, power sag), treatment
  members and repetitions of the pilot design roster.
- **23,400 windowed examples**: each example is 16 steps of telemetry plus
  a candidate action encoding (input 3,132) mapped to the next 8 steps of
  51 per-zone physical targets (output 408).
- Telemetry projections exclude hidden truth: no fault labels, seeds,
  schedules, internal noise/bias state, or future-derived values.

## Collection process

Deterministic Habitat V2 runner, seeded scenarios from the pilot design
roster. Identical inputs reproduce identical packets bit-for-bit; the
manifest hash binds the archive.

## Split discipline

Cluster-level outer holdout: 17 of 60 clusters (6,630 examples) were
excluded from all training, selection, and tuning, and used once for the
reported generalization metrics. Cluster-level (not row-level) splitting
prevents near-duplicate leakage between train and evaluation.

## Uses and known biases

- Suitable: advisory forecasting research inside this simulator; paired
  closed-loop evaluation against the deterministic HMC baseline.
- Not suitable: any physical-world claim; any claim about sensors the
  corpus never degrades.
- **No availability masks.** Every example has complete telemetry. Models
  trained on this corpus cannot handle missing sensors — the reason the
  abstention guard exists. An availability-aware corpus (with explicit
  masks and unavailable-value evidence) is planned future work; its
  collection was paused incomplete (~1,856/3,744 packets) and nothing in
  this repo should be read as claims about it.
- Action coverage is the 4-mode catalogue plus no-proposal; continuous or
  novel actions are out of distribution.

## Maintenance

The archive is immutable. Corrections or extensions land as new versioned
archives with their own manifests and datasheets, never as edits.
