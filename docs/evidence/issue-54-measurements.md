# Issue #54 Corrected Measurements - How Small Is Safe?

Status: **CORRECTED FULL DEVELOPMENT EVIDENCE ONLY** - not qualified and not
deployable. Students remain forecast-only advisory models; HMC is still the sole
proposal, arbitration, preflight, capability, plant-step, and replay authority.

The original v1 tables remain historical provenance and are superseded by this
record. The corrections, protocol choices, and limitations are documented in
`docs/evidence/issue-54-distillation-audit.md`. The frozen preregistration was
not modified.

## Run Identity

- Preregistration:
  `contracts/habitat_v2_forecast_issue_54_preregistration_v1.json`
- Preregistration SHA-256, LF-normalized:
  `E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`
- Protocol addendum:
  `contracts/habitat_v2_forecast_issue_54_distillation_addendum_v1.json`
- Addendum ID: `habitat_v2_forecast_issue_54_distillation_addendum_v1`
- Addendum-declared code commit: `a3802d7`
- Corpus option: `fresh_pipeline`
- Corpus: 32 whole families, anchors 16/24/32, four Track A `normal-*`
  candidates per anchor = 384 samples per teacher and 96 total decisions.
- Family split: 19 TRAIN / 7 VALIDATION / 6 FINAL; 18 FINAL decisions.
- Collection time: 569.7 seconds.
- Target layout: Track A `float32[8,51]`, 408 outputs.
- NMAE scale: TRAIN-only per-horizon/target `P95 truth - P05 truth`, shape
  `8x51`, digest
  `25c9aa1dc0a441c45ca2d13e81d9ee9d725f1de2e759ecdd854e3f10f19ffc83`.
- MLP manifest: `6ba2d3ec6118c7a23f44d1ab70460a54f2f03287b1a121f63be18414f7ab8946`.
- Ridge manifest: `5769871fc5ce1682b8ac86c986f265e6c845bce56508db820a9730d892228406`.
- MLP samples digest: `aa944440767465eb7b595c4343105417fd6ba02378204d1b43036e4eff7ae5b9`.
- Ridge samples digest: `558da30f049008ebee9ce025c9edb9a6b6ff6ca16951bff2f1777e8d3c75eeb1`.
- Results JSON digest: `44abf8167d2fe6116a71d835a68714dd5ec074f8cbda5dc49fe506bec2149252`.
- Raw corpus, traces, and student artifacts remain in ignored
  `out/issue54-full-evidence-1/`; the digests above identify the run without
  committing generated outputs.

## Declared Evaluation Rules

- Decision identity is `(family_id, anchor_step)`, serialized as
  `family_id|anchor=NNNN`; family remains the bootstrap unit.
- The declared ranking metric is
  `issue54-simplified-nominal-point-bound-v1`. It is approved by the addendum
  for the 51-target point lane and is not claimed to be the incompatible frozen
  Issue #52 interval-bearing `score_trajectory` contract.
- MLP teacher targets use per-output TRAIN standardization for optimization and
  are de-standardized before evaluation.
- MLP seeds `540054`, `540055`, and `540056` are all reported. The deterministic
  linear student is reported once with seed `null`; no FINAL-based seed or model
  selection is performed.
- Bootstrap uses 10,000 replicates, seed `540054`, and resamples whole families.
- The primary NMAE gate is point ratio `<= 1.5` and upper 95% CI `< 2.0`.
- Safety-margin closeness is reported against the `<= 0.5` bound; no student has
  actuator authority.

## Hard Gates

| Gate | Result |
| --- | --- |
| Authority violations | 0 |
| Provenance / split violations | 0 |
| HMC replay failures | 0 |
| Non-finite result metrics | 0 |
| Missing or duplicate decision-candidate rows | 0 |

The collector parsed and replayed every HMC trace, and an independent post-run
check verified 384 samples per teacher, 96 decisions, 18 FINAL decisions, 24
student artifact files, all recorded artifact SHA-256 values, both manifests,
both sample digests, the addendum ID, and finite result metrics.

## Results - MLP Teacher

Each row is a separately reported student/seed result. `CI` is the bootstrap
95% interval for student NMAE divided by teacher NMAE. Safety is mean absolute
safety-exposure difference.

| Student | Seed | Params | Ratio | CI | Top-1 | Tau-b | Safety |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| linear | null | 1,278,264 | 0.881 | 0.788-1.008 | 0.889 | 0.963 | 0.000163 |
| sanity-2.1m | 540054 | 2,102,936 | 0.886 | 0.782-1.031 | 0.833 | 0.889 | 0.000128 |
| sanity-2.1m | 540055 | 2,102,936 | 0.933 | 0.835-1.069 | 0.778 | 0.667 | 0.000118 |
| sanity-2.1m | 540056 | 2,102,936 | 0.891 | 0.762-1.074 | 0.778 | 0.556 | 0.000128 |
| medium-500k | 540054 | 496,148 | 1.127 | 0.986-1.308 | 0.667 | 0.815 | 0.000419 |
| medium-500k | 540055 | 496,148 | 1.007 | 0.841-1.206 | 0.389 | 0.463 | 0.000441 |
| medium-500k | 540056 | 496,148 | 1.157 | 1.005-1.332 | 0.389 | 0.500 | 0.000500 |
| small-100k | 540054 | 99,556 | 1.068 | 0.946-1.254 | 0.389 | 0.296 | 0.000443 |
| small-100k | 540055 | 99,556 | 1.018 | 0.922-1.163 | 0.556 | 0.593 | 0.000285 |
| small-100k | 540056 | 99,556 | 1.186 | 1.061-1.373 | 0.556 | 0.426 | 0.000699 |
| tiny-25k | 540054 | 25,195 | 1.148 | 1.064-1.257 | 0.056 | -0.204 | 0.000331 |
| tiny-25k | 540055 | 25,195 | 1.224 | 1.115-1.360 | 0.167 | -0.296 | 0.000587 |
| tiny-25k | 540056 | 25,195 | 1.239 | 1.084-1.436 | 0.056 | -0.333 | 0.000549 |

All MLP rows pass the primary accuracy gate in this corrected full run. That
does not make the smaller models safe for action ranking: the tiny MLP has
negative Kendall tau-b and approximately random top-1 agreement.

## Results - Ridge Teacher

| Student | Seed | Params | Ratio | CI | Top-1 | Tau-b | Safety |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| linear | null | 3,577,344 | 0.937 | 0.881-0.996 | 1.000 | 1.000 | 0.000036 |
| sanity-2.1m | 540054 | 1,835,608 | 1.171 | 1.105-1.246 | 0.889 | 0.963 | 0.000047 |
| sanity-2.1m | 540055 | 1,835,608 | 1.169 | 1.101-1.250 | 0.667 | 0.889 | 0.000048 |
| sanity-2.1m | 540056 | 1,835,608 | 1.163 | 1.095-1.238 | 0.778 | 0.926 | 0.000053 |
| medium-500k | 540054 | 459,208 | 1.299 | 1.249-1.346 | 0.722 | 0.907 | 0.000064 |
| medium-500k | 540055 | 459,208 | 1.260 | 1.201-1.335 | 0.944 | 0.981 | 0.000051 |
| medium-500k | 540056 | 459,208 | 1.293 | 1.254-1.336 | 0.833 | 0.944 | 0.000066 |
| small-100k | 540054 | 92,168 | 1.764 | 1.719-1.817 | 0.444 | 0.630 | 0.000073 |
| small-100k | 540055 | 92,168 | 1.738 | 1.667-1.820 | 0.889 | 0.796 | 0.000075 |
| small-100k | 540056 | 92,168 | 1.723 | 1.677-1.779 | 0.444 | 0.556 | 0.000071 |
| tiny-25k | 540054 | 18,760 | 2.269 | 2.203-2.357 | 0.111 | -0.167 | 0.000068 |
| tiny-25k | 540055 | 18,760 | 2.274 | 2.196-2.391 | 0.167 | -0.037 | 0.000070 |
| tiny-25k | 540056 | 18,760 | 2.324 | 2.233-2.448 | 0.278 | -0.056 | 0.000071 |

Ridge-teacher students through `medium-500k` pass the primary accuracy gate.
The `small-100k` and `tiny-25k` ridge students fail it. Their safety exposure
differences remain numerically small on this benign development fixture, which
does not establish safety under untested conditions.

## Interpretation

- The linear student is the strongest accuracy and ranking baseline for both
  teachers: ratios `0.881` and `0.937`, with Kendall tau-b `0.963` and `1.000`.
- The matching-size MLP students pass the corrected accuracy gate on both
  teachers, but seed sensitivity affects ranking, especially against the MLP
  teacher.
- MLP students down to `tiny-25k` pass the accuracy gate on this larger corpus,
  but the tiny model's ranking is not usable: tau-b is `-0.204` to `-0.333`.
- Ridge students at `medium-500k` retain strong ranking agreement, while ridge
  `small-100k` and `tiny-25k` fail the accuracy gate and tiny ranking collapses.
- All safety differences are below `0.001`, far below the numerical `0.5` gate,
  because this development fixture stays well inside the configured bounds.

These are simulation development findings, not deployment or hardware safety
claims. No tested student may propose, arbitrate, preflight, or execute a plant
action.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_issue54_distillation_study.py --output out/issue54-full-evidence-N
```

Use a new ignored output directory for every run. The addendum must remain
committed before any rerun, and generated outputs must not be hand-edited or
committed.
