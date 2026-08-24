# Issue #55 Controller Race - Revised Design Note

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`
- Protocol: `contracts/habitat_v2_forecast_issue_55_preregistration_v2.json`
- Protocol SHA-256 (LF-normalized): `9041108536E64561ADCEAA434344CDCB6FEAB967F1BD9FB0F47C03FA713FB22E`
- Base commit for the revised protocol: `25a1486157f0bf556c5097b9ab392d8e1e184e02`
- Normative plan: `docs/plans/2026-08-24-issue-55-controller-race-plan.md`
- The v2 protocol supersedes v1. The v1 run remains historical development output and is not the final Issue #55 result.

## Question

How much of the gap between the rule-based controller and a declared finite
full-horizon perfect-foresight schedule baseline does the frozen forecast model
close across varied operating and physical plant conditions?

## Three arms

1. `rules_only` - the HMC runs its default policy; no proposals are issued.
2. `model_advised` - at each preregistered decision step the frozen action-aware
   MLP teacher predicts the 8-step trajectories for the four catalogue actions
   from the verified 16-step history. The declared point-ranking metric selects
   an advisory proposal.
3. `oracle_instrument` - at each decision step the true plant is advanced for
   the full remaining episode for each of the four catalogue commands, with the
   command repeated unchanged. The lowest declared true-plant score is proposed.

The oracle uses future plant state, timeline loads, operating modes and declared
fault schedules only as an evidence instrument. It is exact only over the
declared four-action constant-command schedule, not a global optimum over all
possible action sequences. It is never a runtime or demo controller.

In every arm the HMC validates, arbitrates and applies commands. The HMC remains
the sole actuator authority and may reject or modify any advisory proposal.

## Revised family suite

The 32-family roster is a fixed `4 x 4 x 2` matrix:

- four operating conditions: nominal occupied, high-load occupied,
  low-load/eva-transition, and hot-humid/contingency;
- four plant conditions: nominal, fan degradation, laboratory branch resistance,
  and equipment cooling-delivery degradation;
- two deterministic sensor seeds per operating/plant combination.

The family index mapping and exact numeric values are frozen in the v2
preregistration. Physical faults run from emitted steps 32 through 79. The
family nonce depends only on the family ID, so all arms are paired by family.

## Fairness and causal boundaries

- The model teacher and its artifact digest are unchanged from the v1 study.
- The model horizon remains 8 steps and the model decision cadence remains
  steps 16, 20, ..., 84.
- Each arm starts from the same family scenario and family-bound HMC nonce.
- True plant metrics are computed from shadow states cross-checked against HMC
  plant receipts at every step.
- Forecast and oracle outputs enter only through standard advisory proposals;
  no authority or physics code is changed.

## Metrics

- `safety_exposure`: normalized bound-crossing mass over the 51 true-plant targets.
- `safety_violation_steps`: steps with any true-target bound crossing.
- `comfort_deviation`: normalized temperature, CO2 and humidity deviation on
  occupied-mode true states.
- `resource_composite`: normalized battery, oxygen and sorbent consumption.
- Headline gap closure: `(mean_rules - mean_model) / (mean_rules - mean_oracle)`
  over equal-weight family means, with the preregistered family bootstrap CI.

The oracle score is `full_remaining_safety + full_remaining_occupied_comfort +
0.1 * final_resource_composite` for each finite constant-command candidate.

## Evidence boundary

The v1 evidence documents and outputs are superseded because v1 used an 8-step
greedy oracle and sensor-seed-only families. The v2 run must publish the full
per-arm/per-family table, family roster, hard gates, proposal statistics and
bootstrap intervals. All numbers remain deterministic simulator development
evidence only; they do not establish qualification, certification, hardware
behavior or deployment readiness.
