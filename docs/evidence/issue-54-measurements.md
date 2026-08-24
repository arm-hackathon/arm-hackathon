# Issue #54 Honest Measurements - How Small Is Safe?

Status: **DEVELOPMENT EVIDENCE ONLY** - research study, not qualified.
Deployment of any student remains blocked: HMC is still the sole proposal,
arbitration, preflight, capability, plant-step, and replay authority.

The Issue #54 preregistration remains authoritative:
`contracts/habitat_v2_forecast_issue_54_preregistration_v1.json`
(preregistration SHA-256, LF-normalized:
`E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`).

## Run Identity

* Teachers: MLP `action-aware-mlp-v1.npz`
  (SHA-256 `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd`,
  3132 -> 512 -> 512 -> 256 -> 408, GELU, window 16); ridge
  `action-aware-ridge.npz`
  (SHA-256 `0de4b5cdb6ec2b47be260a06f924d8eb00f1def16d5ae668b3ab5191251f29df`,
  8767 -> 408, window 4). Both `actuator_authority=false`, frozen.
* HMC binding v2 SHA-256: `aa37ae6394031241d317bcfcc31ea2e3da0b4701ddd96f04dfbc90cf3142e63d`
* Corpus option measured: `fresh_pipeline` (the other two preregistered options
  were assessed as incompatible with the frozen contract as checked in; see the
  distillation card).
* Corpus: 32 whole families (sensor-seed variations of the frozen development
  scenario extended to 48 steps), anchors 16/24/32, 4 catalogue actions per
  anchor = 384 samples per teacher.
* Family split: 19 TRAIN / 7 VALIDATION / 6 FINAL, whole-family isolated.
* MLP corpus manifest SHA-256: `405cb457c077310bfdf7ff260253130ee63dff39c6071cb286461f52191f231d`
* MLP samples SHA-256: `940546929cb82ca689ce08f2141fde82b4299a7d9462dea9d5784ecc343b3edf`
* Ridge corpus manifest SHA-256: `47407af5b0ab826c0dd7324277d3b4e71509deaab0268cd7935ec8ee056cfe4c`
* Ridge samples SHA-256: `566785eac02c98392792b0da3572ae2f63cf71683f6d251fa8001d72b342e2d3`
* Command: `python scripts/run_issue54_distillation_study.py`
  (raw outputs in ignored `out/issue54/`; corpus and manifests are reproducible).
* Trainer: Adam (lr 0.001, l2 1e-4), 200 epochs, GELU exact-erf, MSE distillation
  loss on per-output standardized teacher targets (disclosed deviation from raw
  MSE; see card), seed 540054. Linear student: closed-form ridge with
  cluster-leave-one-out alpha selection.
* Bootstrap: 10,000 replicates, seed 540054, resampling unit = family.

## Hard Gates

| Gate | Result |
| --- | --- |
| Authority violations | 0 |
| Provenance / split violations | 0 |
| Replay failures | 0 |

## Results - NMAE Ratio vs Teacher (primary)

Eligible rows: candidates with all 8 finite targets, identical eligibility mask
for teacher and student. Aggregation: equal-weight family mean of
`NMAE_student / NMAE_teacher` paired by family. Gate: point <= 1.5 and
upper 95% CI < 2.0.

### MLP teacher (teacher family-mean NMAE 0.0157)

| Student | Params | Ratio | 95% CI | Gate |
| --- | ---: | ---: | --- | --- |
| linear | 1,278,264 | 0.753 | 0.699 - 0.823 | PASS |
| sanity-2.1m | 2,102,936 | 1.549 | 1.420 - 1.718 | FAIL (point > 1.5) |
| medium-500k | 496,148 | 3.441 | 3.181 - 3.733 | FAIL |
| small-100k | 99,556 | 3.678 | 3.260 - 4.318 | FAIL |
| tiny-25k | 25,195 | 4.503 | 3.966 - 5.144 | FAIL |

### Ridge teacher (teacher family-mean NMAE 0.0157)

| Student | Params | Ratio | 95% CI | Gate |
| --- | ---: | ---: | --- | --- |
| linear | 3,577,344 | 0.962 | 0.924 - 0.991 | PASS |
| sanity-2.1m | 1,835,608 | 1.409 | 1.370 - 1.443 | PASS |
| medium-500k | 459,208 | 1.637 | 1.590 - 1.680 | FAIL |
| small-100k | 92,168 | 2.394 | 2.344 - 2.445 | FAIL |
| tiny-25k | 18,760 | 3.315 | 3.281 - 3.348 | FAIL |

## Results - Action-Ranking Agreement (co-primary)

Top-1 agreement: fraction of decisions where the student's top-ranked candidate
equals the teacher's. Kendall tau-b: mean over decisions of full-ranking
correlation. 6 FINAL decisions per teacher.

### MLP teacher

| Student | Top-1 | Kendall tau-b |
| --- | ---: | ---: |
| linear | 0.500 | 0.783 |
| sanity-2.1m | 0.500 | 0.449 |
| medium-500k | 0.333 | 0.354 |
| small-100k | 0.167 | 0.162 |
| tiny-25k | 0.000 | 0.066 |

### Ridge teacher

| Student | Top-1 | Kendall tau-b |
| --- | ---: | ---: |
| linear | 0.500 | 0.904 |
| sanity-2.1m | 0.333 | 0.778 |
| medium-500k | 0.167 | 0.717 |
| small-100k | 0.167 | 0.434 |
| tiny-25k | 0.000 | -0.025 |

## Results - Safety-Margin Closeness (co-primary)

Mean absolute difference of normalized safety exposure between student and
teacher per candidate per decision. Gate: <= 0.5.

| Teacher | Student | Mean abs safety-exposure diff |
| --- | --- | ---: |
| MLP | linear | 0.00016 |
| MLP | sanity-2.1m | 0.00013 |
| MLP | medium-500k | 0.00042 |
| MLP | small-100k | 0.00044 |
| MLP | tiny-25k | 0.00033 |
| Ridge | linear | 0.00004 |
| Ridge | sanity-2.1m | 0.00005 |
| Ridge | medium-500k | 0.00006 |
| Ridge | small-100k | 0.00007 |
| Ridge | tiny-25k | 0.00007 |

All students pass the safety-margin gate by a wide margin on this corpus
(both teacher and student trajectories stay inside bounds).

## Best Student Per Param-Count Class

Per the preregistration selection rule ("report all three options and record the
best distillation agreement per param count" - only `fresh_pipeline` was
measurable):

| Class | Best student | Params | Best ratio (teacher) |
| --- | --- | ---: | --- |
| linear (ridge) | linear | 1,278,264 (MLP) / 3,577,344 (ridge) | 0.753 (MLP) / 0.962 (ridge) |
| ~2.1M | sanity-2.1m | 2,102,936 (MLP) / 1,835,608 (ridge) | 1.549 (MLP) / 1.409 (ridge) |
| ~500K | medium-500k | 496,148 (MLP) / 459,208 (ridge) | 3.441 (MLP) / 1.637 (ridge) |
| ~100K | small-100k | 99,556 (MLP) / 92,168 (ridge) | 3.678 (MLP) / 2.394 (ridge) |
| ~25K | tiny-25k | 25,195 (MLP) / 18,760 (ridge) | 4.503 (MLP) / 3.315 (ridge) |

## Honest Negative Results

1. **All MLP students except none meet the primary gate.** Even `sanity-2.1m`,
   which reproduces the MLP teacher's exact hidden widths (512,512,256), fails
   the point gate on the MLP teacher (1.549 > 1.5) with this corpus and trainer.
2. **Tiny students cannot rank.** `tiny-25k` Kendall tau is ~0.07 (MLP) and
   -0.025 (ridge) - effectively random action rankings. These students are NOT
   safe for action ranking.
3. **Without target standardization the MLP trainer emits garbage.** Raw MSE on
   mixed-SI-unit targets (span 0.02-72,000) produced NMAE ratios in the
   thousands in pilot runs; the measured results above require the disclosed
   per-output target standardization, mirroring the teacher's own training.
4. **Two preregistered corpus options were not measurable as checked in:**
   `committed_corpus` uses the horizon-32 / 12-candidate Issue #52 corpus that
   does not convert to the Issue #54 horizon-8 / 4-action contract without
   re-projection, and `synthetic_varied_seed` would require bypassing the frozen
   contract bundle validation. Only `fresh_pipeline` was measured.

## Retained vs Lost Capability

* Retained: the linear student preserves teacher accuracy (ratio 0.75-0.96),
  useful ranking agreement (tau 0.78-0.90), and safety-margin closeness on the
  measured corpus.
* Lost: MLP students at any tested size below the teacher fail the accuracy
  gate; ranking agreement degrades monotonically with size; the smallest
  students lose ranking information entirely.

## Reproduction

```bash
# pilot (6 families, 2 anchors) - ~1 minute
python scripts/run_issue54_distillation_study.py --pilot

# full run (32 families, 3 anchors) - ~3-4 minutes
python scripts/run_issue54_distillation_study.py
```

All outputs are written below ignored `out/issue54/` (corpus manifests,
samples JSONL, student NPZ artifacts, `results.json`). Raw outputs are not
committed; manifests and digests above make the corpus identity reproducible.
