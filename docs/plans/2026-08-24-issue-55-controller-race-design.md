# Issue #55 Controller Race вЂ” Design Note

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`
- Preregistration: `contracts/habitat_v2_forecast_issue_55_preregistration_v1.json`
- Preregistration SHA-256 (LF-normalized): `17C601D7F15A21804AA68B26024C96D44642491E07A9BD75BDE805E027C773CF`
- Base commit: `099a141022c39b815589ef6879e66ae38b8e957e`
- Normative plan: `docs/plans/2026-08-24-issue-55-controller-race-plan.md`

## Question

How much of the gap between the rule-based controller alone and a perfect-foresight
ceiling does the frozen forecast model close, on safety, comfort, and resource use?

## Three arms

1. `rules_only` вЂ” the HMC runs its default policy; no proposals are ever issued.
2. `model_advised` вЂ” at each decision step the frozen action-aware MLP teacher
   predicts 8-step trajectories for all four Track A catalogue actions from the
   verified 16-step history; the predicted point trajectories are ranked with the
   preregistered `issue55-advisory-point-ranking-v1` metric; the best action is
   proposed through the standard advisory proposal path.
3. `oracle_instrument` вЂ” at each decision step each catalogue action is evaluated
   by advancing the **true plant** 8 steps with that command repeated
   (`issue55-oracle-lookahead-v1`); the best true outcome is proposed. The oracle
   "cheats" by using the deterministic plant simulator as its forecast. It is a
   measuring instrument for the ceiling only and must never appear in any demo,
   report surface, or runtime advisor role.

In all three arms the HMC arbitrates every proposal and may reject it; the HMC
remains the sole actuator authority. All arms experience identical scenarios and
identical sensor streams (same episode nonce per family).

## Why this is a fair race

- Same 32 scenario families, same 96-step episodes, same decision steps
  (16, 20, ..., 84 в†’ 18 decisions), same lookahead (8 steps).
- The episode nonce binds to the family only, not the arm, so all three arms see
  byte-identical observation streams up to the first divergence caused by their
  own proposals.
- The advisory arm uses the same frozen teacher artifact (SHA-pinned) that the
  live demos use; no retraining, no tuning.
- The oracle differs from the model arm only in the source of its 8-step
  trajectory prediction: the true simulator instead of the learned model.

## Metrics

Computed on true plant states (the shadow plant is digest-cross-checked against
every HMC step receipt, as in Issue #54):

- `safety_exposure` вЂ” normalized bound-crossing mass on the preregistered
  51-target Track A bounds table (identical values to frozen Issue #52 scoring
  bounds and Issue #54 evaluation arrays).
- `safety_violation_steps` вЂ” steps with any crossing.
- `comfort_deviation` вЂ” mean normalized deviation from temperature/CO2/humidity
  nominals over occupied-mode steps.
- `resource_composite` вЂ” total normalized battery + oxygen + sorbent consumption.
- Headline: per-metric gap closure `(rules в€’ model) / (rules в€’ oracle)` with a
  bootstrap 95% CI over families; `DEGENERATE_GAP` when the denominator ~ 0.

## Smoke-run observations (development validation only, not evidence)

Two two-family smoke runs (`out/issue55-smoke-1`, `out/issue55-smoke-2`) were
used to validate the pipeline before any full-protocol run. Findings:

- The true plant sits at the zone O₂ bound (0.30) under rules alone — the
  teacher's ~0.3000 O₂ predictions are accurate, so the predicted-crossing
  hard gate correctly abstains at ~94 percent of advisory decisions.
- The rare admitted proposals were already mode-matched; their large safety
  exposure comes from beyond-8-step consequences of switching the actuator
  regime, which the 8-step prediction cannot see. This is an honest expected
  finding of the race, not a protocol flaw.
- The oracle legitimately proposes cross-mode actions (the dormant command
  during occupied segments) because true 8-step outcomes favor them; a
  mode restriction on the model arm alone would bias the race. No protocol
  amendment was made: the preregistration is unchanged from its original
  frozen digest.

## What this does not prove

- Nothing here qualifies, certifies, or deploys anything. The oracle is not a
  controller. The model arm's advisory metric is a declared point-ranking
  heuristic, not frozen Issue #52 `score_trajectory` compliance with uncertainty.
- The fixture is benign; small absolute safety differences are expected and are
  reported honestly, including a possible degenerate safety gap.
