# Corpus Datasheet — Historical V2 Pilot Archive

Training/evaluation corpus behind `action_aware_mlp_v1` (see
`MODEL_CARD.md`). Manifest SHA-256
`7c35a2da3a9f902c2994b8b29332f8fafed40b81c49204c3546a75d2b3f76659`.

This datasheet preserves author-recorded historical metadata. The corpus bytes,
manifest, generator/training code, and execution receipts are not retained, so
current `main` cannot independently verify the composition, split chronology,
or training/evaluation claims below. See the [historical evidence
index](docs/evidence/closed-loop-advisory-historical-index.md).

## Motivation

The recorded design intent was to provide a frozen, provenance-bound dataset
for learning atmosphere dynamics in the Habitat V2 simulation and testing an
action-aware forecaster under a deterministic safety controller, without hidden
simulator state leaking into model inputs. The retained repository does not
independently establish that intended freeze or provenance boundary.

## Composition

- **4,680 simulation packets**, each a complete seeded Habitat V2 run with
  operational telemetry projections.
- **60 scenario clusters** spanning operating modes (nominal, EVA
  transition, contingency, dormant), fault families (scrubber degradation,
  fan faults, sensor bias/drift/stuck, damper jams, power sag), treatment
  members and repetitions of the pilot design roster.
- **23,400 windowed examples**: each example is 16 steps of telemetry plus
  a candidate action encoding (input 3,132) mapped to the next 8 steps of
  51 targets: 48 per-zone values (6 channels × 8 zones) plus 3 global resource
  fractions (output 408 = 8 × 51).
- Telemetry projections exclude hidden truth: no fault labels, seeds,
  schedules, internal noise/bias state, or future-derived values.

## Collection process

Historical materials describe a deterministic Habitat V2 runner and seeded
pilot-roster scenarios. The absent corpus, generator, manifest bytes, and
environment receipt prevent a current fresh checkout from reproducing the
packets or checking them against the recorded manifest hash.

## Split discipline

Historical materials report a cluster-level outer holdout: 17 of 60 clusters
(6,630 examples) excluded from training, selection, and tuning and used once
for the reported generalization metrics. Cluster-level rather than row-level
splitting was intended to prevent near-duplicate leakage. The absent corpus,
split manifest, training code, and receipts prevent independent verification of
that exclusion and once-only use.

## Uses and known biases

- Suitable: advisory forecasting research inside this simulator; paired
  closed-loop evaluation against the deterministic HMC baseline.
- Not suitable: any physical-world claim; any claim about sensors the
  corpus never degrades.
- **No availability masks.** Every example has complete telemetry. Models
  trained on this corpus are not evidenced or supported for missing-sensor
  inputs; the historical harness therefore abstains. At the time of this
  archive, an availability-aware collection was paused incomplete
  (~1,856/3,744 packets). Issue #53 later produced and qualified a separate
  forecast-only successor for its bounded independent-dropout contract; that
  result neither reconstructs this archive nor extends to correlated/mixed
  dropout or actuator authority.
- Action coverage is the 4-mode catalogue plus no-proposal; continuous or
  novel actions are out of distribution.

## Maintenance

The recorded design treats the archive as immutable and requires corrections
or extensions to use new versioned archives. Current `main` preserves the
recorded identifier and this datasheet, not the archive bytes needed to verify
immutability.
