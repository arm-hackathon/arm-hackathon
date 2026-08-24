# Issue #55 Controller Race - v2 Capability And Limitation Card

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Measurements: `docs/evidence/issue-55-measurements.md`
- Preregistration: `contracts/habitat_v2_forecast_issue_55_preregistration_v2.json`
- Preregistration SHA-256 (LF-normalized): `9041108536E64561ADCEAA434344CDCB6FEAB967F1BD9FB0F47C03FA713FB22E`
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`

The v2 protocol supersedes the historical v1 Issue #55 study. This card is a
concise reading of the v2 development run; the measurements document contains
the full roster and all 96 per-arm/per-family records.

## What Was Raced

Three arms ran over identical paired 96-step episodes for 32 fixed families.
The families cover `4 operating conditions x 4 physical plant conditions x 2
sensor conditions`. The Habitat Management Controller (HMC) remained the sole
proposal-arbitration, command, plant-step and replay authority in every arm.

1. `rules_only` - HMC default policy with no proposals.
2. `model_advised` - the frozen action-aware MLP ranks four catalogue commands using the declared point-prediction advisory metric; eligible proposals enter HMC arbitration.
3. `oracle_instrument` - a true-state, future-schedule measuring instrument scores each catalogue command over the remaining episode when repeated unchanged; the lowest finite-schedule score is proposed.

The oracle is not a controller. It is exact only over the declared four-action
constant-command schedule, not a global optimum over arbitrary action sequences.

## What The Run Shows

- The model improves mean comfort relative to rules: `98.258700076` versus `111.122949941`; the declared comfort gap closure is `0.569478` with bootstrap 95% CI `[0.236158, 0.893338]`.
- The model does not preserve safety: mean normalized safety exposure is `15.678536800` versus `0.000217557` for rules and `0.000047684` for the oracle arm. Its safety violation-step total is lower than rules (`425` versus `584`), but its violations are much more severe.
- The model does not improve resource use: its mean resource composite is `0.010069491342`, above rules at `0.007738955319` and the oracle arm at `0.008458953351`. The declared resource closure point `3.236864` is therefore not a beneficial closure.
- The model proposed at only 8 of 576 advisory decision opportunities and abstained at 568 (98.6%). All 8 proposals were admitted and applied unchanged by HMC.
- The oracle proposed at all 576 decision opportunities. HMC changed the final command for 432 proposals (75.0%), so only 144 oracle proposals were applied unchanged. This is why the oracle remains advisory-only even with true-state foresight.

## What The HMC Boundary Shows

- All 96 records completed 96 committed HMC replay steps.
- Authority violations, replay failures, provenance violations, non-finite metrics and proposal-admission failures were all zero.
- The oracle's 432 HMC final-command differences are counted by the preregistered `hmc_rejection_count`; they are distinct from proposal-admission failures.
- No model or oracle output acquired actuator authority, and the oracle was not added to a demo or runtime advisor surface.

## What The Model Cannot Do In This Fixture

- Its 8-step prediction horizon does not expose consequences that unfold later in the 96-step episode. A small number of admitted model proposals substantially worsened normalized safety exposure.
- Its advisory ranking is a declared point-prediction heuristic with a hard predicted-bound eligibility gate. It is not frozen Issue #52 `score_trajectory` compliance with calibrated uncertainty bands.
- Its high abstention rate is not a general safety guarantee. It is the observed result of this teacher, this gate, this fixed catalogue and this development roster.

## Metric And Suite Limitations

- Comfort deviation is defined only over occupied-mode steps. Families 16 through 31 are `eva_transition` or `contingency` families with no declared occupied-mode comfort rows, so their comfort deviation is zero by metric definition. The overall comfort means include these 16 zero-row family records and are not all-mode comfort scores.
- The 32-family matrix is more varied than v1, but it remains one fixed deterministic development suite. It does not establish behavior on untested scenarios, other plants or physical systems.
- The full-horizon oracle is a finite measuring instrument, not a proof of global optimality. Its result must not be represented as a deployable controller or a universal ceiling.
- The output digests are `results.json` `d99f6f748433b1fbb47cffa5e4c7f103b517c2900dc2580bb72d3e0913ab59b3` and `episodes.jsonl` `4165a9be11c5755df83fa0ee8160d2eb318c615374b75b5418a5485b8cee7265` under ignored local output `out/issue55-race-v2-1/`.
- These are deterministic simulator development results only. They do not establish qualification, certification, hardware behavior, deployment readiness or real-world safety for any controller or model.
