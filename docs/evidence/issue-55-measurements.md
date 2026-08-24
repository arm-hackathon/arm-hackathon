# Issue #55 Controller Race — Measurements

- Study: three-way controller race (rules vs model-advised vs perfect-foresight instrument)
- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Preregistration: `contracts/habitat_v2_forecast_issue_55_preregistration_v1.json`
- Preregistration SHA-256 (LF-normalized): `17C601D7F15A21804AA68B26024C96D44642491E07A9BD75BDE805E027C773CF`
- Branch: `research/issue-55-controller-race` (from `origin/main@099a141`)
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY` — simulation development evidence only

## Run identity

- Command: `uv run --locked --python 3.11 --extra dev python scripts/run_issue55_controller_race.py --output out/issue55-race-1`
- Suite: 32 sensor-seed variant families of the frozen development scenario, extended to 96 steps
- Episodes: 96 (32 families x 3 arms), 18 preregistered decisions per episode (steps 16..84, cadence 4, lookahead 8)
- Teacher: frozen `action-aware-mlp-v1.npz` (SHA-256 `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd`)
- Output digests:
  - `results.json` SHA-256: `7112cf228583b9ecb4e2c071eb5c52c5428288850275171ae84560984091273e`
  - `episodes.jsonl` SHA-256: `b8477e2b33373ec09dfd4f6f3761f91dd57e66a8754358e3c2643db3b4606a74`
- Post-run validation: all 96 episode digests recomputed, rosters/counts/gates verified (`scripts`-run independent check; validation code is temporary and not committed)
- Hard gates: authority violations 0, replay failures 0, provenance violations 0, non-finite metrics 0, proposal admission failures 0
- HMC rejections: 0 across all arms (every admitted proposal was adopted by arbitration)

## Results table (family means; violation steps are totals over 32 x 95 steps)

| Arm | Safety exposure (mean) | Violation steps (total) | Comfort deviation (mean) | Resource composite (mean) | Proposals | Abstentions |
| --- | --- | --- | --- | --- | --- | --- |
| `rules_only` | 0.000870 | 2336 | 164.4376 | 0.000000 | 0 | 0 |
| `model_advised` | 104.2883 | 2180 | 139.7123 | 0.013800 | 32 | 544 |
| `oracle_instrument` | 0.002357 | 544 | 80.2534 | 0.004400 | 576 | 0 |

## Headline gap closures (bootstrap 95% CI over 32 families, 10k resamples, seed 550055)

| Metric | Closure point | 95% CI | Reading |
| --- | --- | --- | --- |
| `comfort_deviation` | **0.294** | [0.186, 0.417] | The model arm closes ~29% of the rules-to-oracle comfort gap |
| `resource_composite` | 3.10 | [2.73, 3.39] | The model arm consumes ~3x the oracle's extra resource budget (overshoots the ceiling's use) |
| `safety_exposure` | 70126 | [59207, 80053] | Formally positive but practically degenerate: the rules-to-oracle safety gap is only 0.0015; the model arm's exposure is ~5 orders of magnitude above both |

The preregistration anticipated a degenerate/noisy safety closure; the honest
safety statement is directional: the model arm is dramatically worse than rules
alone on safety exposure whenever it acts, while the oracle stays within
0.0015 of rules-only exposure.

## Action-level findings

- The oracle proposes `normal-dormant-v1` 512/576 times — including during
  occupied segments — because stopping the O2-driving command pulls the plant
  off the 0.30 O2 bound (violation steps 2336 -> 544) and halves comfort
  deviation, at a small resource cost. The ceiling in this fixture comes
  mostly from a cross-mode choice that rule-following would never make.
- The model arm abstains at 544/576 decisions (94.4%): the frozen teacher
  accurately predicts zone O2 sitting at the 0.30 bound, so the
  predicted-crossing hard gate (frozen Issue #52 semantics) correctly defers
  to rules alone.
- Its 32 admitted proposals (exactly one per family, steps 16-28) split into
  29 `normal-occupied-v1` proposals (episode safety exposure 78-150: the
  actuator regime switches and the consequences unfold beyond the 8-step
  prediction horizon) and 3 `normal-dormant-v1` proposals (episode safety
  0.000286, comfort 69.72-69.74 - the best episodes in the study on both
  metrics).

## Limitations

- The "oracle" is a greedy 8-step perfect-foresight instrument over the
  preregistered composite score. It is a measuring instrument and a strong
  baseline, not a proof of global optimality: the model arm's own three
  dormant-proposal episodes beat it on both safety and comfort.
- The suite varies only the sensor seed, and the true plant is
  observation-independent, so `rules_only` and `oracle_instrument` true
  trajectories are identical across all 32 families; the effective sample for
  those two arms is one scenario. Family variation only exercises the model
  arm's observation stream. The preregistered family bootstrap therefore
  reflects model-arm variability, not scenario variability.
- The advisory ranking metric is a declared point-prediction heuristic; it is
  not frozen Issue #52 `score_trajectory` compliance with uncertainty bands.
- All numbers are simulator development evidence. Nothing here qualifies,
  certifies, or deploys anything, and the oracle must never appear in any demo
  as a controller.
