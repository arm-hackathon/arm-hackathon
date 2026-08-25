# Issue #56 Action-Risk Adviser Measurements

Date: 2026-08-25
Branch: `research/action-risk-adviser`
Status: **DEVELOPMENT EVIDENCE ONLY - NOT QUALIFIED OR DEPLOYABLE**

The preregistration is
`contracts/habitat_v2_forecast_issue_56_preregistration_v1.json`. The run was
performed with `scripts/run_action_risk_adviser.py --output
out/action-risk-full-1 --families 32`. Generated output remains ignored under
`out/` and is not checked in.

## Run Identity

- Preregistration SHA-256, LF-normalized:
  `bd175dbf139cb26340202b7c4b0141ce5170e0b61f9c35ab638e66af11dd448f`
- Corpus: 32 complete Issue #55 v2 family identities, with 19 TRAIN, 7
  VALIDATION, and 6 EVALUATION families.
- Samples: 1,664 complete action-conditioned samples: 988 TRAIN, 364
  VALIDATION, and 312 EVALUATION.
- Each family supplied 13 decision steps and four catalogue actions, or 52
  samples.
- Manifest SHA-256: `a8be3d3894ef1cdfea192d7f4645961e3f5c8011a52f7b38a85b5f59e42105f9`
- Samples SHA-256: `f026027b2548d939e8010d04444f8605efe6b6134c0e777a01e562452690b80e`
- Model SHA-256: `411df4f9b10454099ee5f5c9f4732b8cc085ade9b7bd56a4ccd4fcb85d0af435`
- Calibration SHA-256: `fc41f24e17ccc09efd378e58b55389cfa47fd5020056edc3ca241db4e77cbb5a`
- Results SHA-256: `afbeae5fd986cd7d4018fac58fb6afac091a8e55cb6578e1bb7167683db27802`

## Risk Calibration

The model was fitted on TRAIN families and calibrated on VALIDATION families.
The following development report was calculated on EVALUATION samples:

| Metric | Value |
|---|---:|
| Samples | 312 |
| Calibrated upper-exposure coverage | 0.9775641026 |
| Crossing-event Brier score | 0.3868604807 |
| Mean absolute exposure error | 2.4555164378 |

The risk model selected no candidate on any EVALUATION decision. Every risk
episode therefore abstained at all 13 decisions and followed the HMC default
hold policy. This is a conservative fail-closed result, not a demonstrated
utility improvement.

## Paired EVALUATION Episodes

The six EVALUATION families were replayed under both `rules_only` and the risk
adviser. The existing rules arm is the comparator; no existing point-model arm
was changed by this branch.

| Arm | Safety exposure mean | Violation steps mean | Comfort mean | Resource mean | Proposals | Abstentions | HMC rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules_only | 0.0002900759 | 24.3333 | 150.6089 | 0.0121726 | 0 | 0 | 0 |
| calibrated risk | 0.0002900759 | 24.3333 | 150.6089 | 0.0121726 | 0 | 78 | 0 |

The equality is expected because the risk adviser abstained for all candidates
under the preregistered calibrated hard gates. All six risk traces and all six
rules traces committed 96 steps and passed strict replay. Authority,
provenance, proposal-admission, and non-finite hard-gate counts were zero.

## Interpretation

- The feature, label, training, validation-calibration, HMC proposal, shadow
  replay, and strict trace-replay paths executed deterministically.
- The calibrated upper-exposure report covered 97.7564% of EVALUATION samples.
- The current model is too conservative to issue useful proposals on this
  corpus. It did not improve comfort, safety, or resources over rules-only.
- The correct conclusion is an honest negative development result: the lane
  preserves the safety boundary but does not qualify the model for deployment.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python \
  scripts/run_action_risk_adviser.py \
  --output out/action-risk-full-1 \
  --families 32
```

Use a new ignored output directory for each run. The command refuses to reuse an
existing output directory.
