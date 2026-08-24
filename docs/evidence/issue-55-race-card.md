# Issue #55 Controller Race — Capability and Limitation Card

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Measurements: `docs/evidence/issue-55-measurements.md`
- Preregistration: `contracts/habitat_v2_forecast_issue_55_preregistration_v1.json`
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`

## What was raced

Three controllers over identical 96-step episodes of 32 sensor-seed variant
families, with the Habitat Management Controller (HMC) as the sole actuator
authority in every arm:

1. `rules_only` — HMC default policy, no proposals.
2. `model_advised` — the frozen action-aware MLP teacher ranks the four
   catalogue actions per decision with the preregistered point-ranking metric;
   the best eligible action is proposed; the HMC may reject.
3. `oracle_instrument` — a perfect-foresight measuring instrument that
   evaluates each action by advancing the true plant 8 steps and proposes the
   best true outcome. It is the ceiling probe of this study and is not a
   controller.

## What the model advisory can do (in this fixture)

- Abstain safely: it defers to rules alone at 94.4% of decisions because it
  accurately predicts the plant sitting at the zone O2 safety bound.
- When its single per-family proposal happened to be the dormant command
  (3 of 32 families), the episode was the best in the study on both safety
  (0.000286) and comfort (69.72) — better than the oracle instrument itself.
- Close a measurable fraction of the comfort gap: 29.4% (95% CI 18.6-41.7)
  of the rules-to-oracle comfort-deviation gap.

## What the model advisory cannot do (in this fixture)

- See beyond its 8-step prediction horizon: 29 of its 32 proposals switched
  the actuator regime into states whose harm unfolds after the horizon,
  driving safety exposure to 78-150 per episode versus 0.000870 for rules
  alone — roughly five orders of magnitude worse.
- Beat the ceiling's resource discipline: it consumes ~3x the oracle's extra
  resource budget while achieving less.
- Act as a ranker under the hard gate: the predicted-crossing gate plus
  accurate at-bound O2 predictions reduce it to a near-pure abstainer.

## What the oracle instrument revealed

- The dominant ceiling strategy is a cross-mode one: propose the dormant
  command (512/576 decisions) to stop O2 injection at the bound. This cuts
  bound-crossing steps by 77% (2336 -> 544) and halves comfort deviation for
  a small resource cost.
- A greedy 8-step perfect-foresight policy is not a proven global optimum —
  the model arm's three dormant episodes beat it on both safety and comfort —
  so the instrument bounds typical achievable performance rather than proving
  a limit.

## Boundaries

- The oracle is a measuring instrument only. It must never appear in any demo,
  report surface, or runtime advisor role, and it is not shipped anywhere.
- The HMC kept final say in every arm; zero rejections occurred and zero
  authority violations were observed.
- All results are deterministic simulation development evidence on one benign
  development fixture. They do not establish qualification, certification,
  hardware behavior, or deployment readiness of any controller or model.
- The suite varies only sensor seeds, so rules-only and oracle true
  trajectories are identical across families; see the measurements document
  for the full limitation list.
