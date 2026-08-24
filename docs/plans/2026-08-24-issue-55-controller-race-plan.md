# Issue #55 Controller Race - Revised Normative Plan

## 1. Document control

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Branch: `research/issue-55-controller-race`
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`
- Protocol: `contracts/habitat_v2_forecast_issue_55_preregistration_v2.json`
- Protocol SHA-256 (LF-normalized): `9041108536E64561ADCEAA434344CDCB6FEAB967F1BD9FB0F47C03FA713FB22E`
- Design note: `docs/plans/2026-08-24-issue-55-controller-race-design.md`
- The v2 protocol was declared before the revised smoke or full run. No external
  sign-off is a prerequisite for this local development study.

## 2. Scope and non-goals

Race `rules_only`, `model_advised` and `oracle_instrument` over the fixed 32-family
development roster. Measure true-plant safety, comfort, resource use and the
model-to-oracle gap closure. Preserve all HMC, physics, safety, forecast-artifact
and runtime/demo boundaries. Do not use final-suite or validation data.

The oracle is evidence-only. Its finite schedule is exact only over the four
catalogue commands repeated to the end of the current episode; it is not a claim
of global optimality.

## 3. Fixed protocol

- 32 families: `4 operating conditions x 4 physical plant conditions x 2 sensor
  conditions`, with exact values and index mapping in the preregistration.
- 96-step episodes, family-bound nonce, and decisions at steps 16 through 84
  every 4 steps.
- Model predictions use the frozen action-aware MLP, verified 16-step history and
  8-step prediction horizon.
- Oracle scores each catalogue action by true-plant rollout from the current
  state through the final episode step, repeating that action unchanged.
- HMC remains the only command authority and plant-step authority in all arms.

## 4. Implementation units

1. `src/aeolus/habitat_v2/forecast_issue55_race.py`
   - deterministic family roster and condition descriptors;
   - true target projection and metrics;
   - unchanged frozen-teacher advisory ranking;
   - full-remaining-episode finite oracle scoring;
   - strict HMC proposal, shadow-receipt, trace and replay checks.
2. `tests/habitat_v2/test_forecast_issue55_race.py`
   - v2 protocol digest and constant bindings;
   - family roster uniqueness and condition-axis coverage;
   - full-horizon oracle score shape and feasibility;
   - deterministic episode, authority and replay invariants.
3. `scripts/run_issue55_controller_race.py`
   - write-once smoke/full runner;
   - v2 protocol digest verification;
   - family roster, per-arm records, summaries and bootstrap intervals.
4. `contracts/habitat_v2_forecast_issue_55_preregistration_v2.json`
   - commit-bound revised protocol and supersession record.

## 5. Execution phases

- Phase 0: declare the v2 protocol and update the design/plan documents.
- Phase 1: implement and pass focused tests.
- Phase 2: run a small smoke study in a new ignored `out/` directory, including
  at least one physical-fault family.
- Phase 3: run all 32 families and all three arms in a new ignored directory.
- Phase 4: independently validate output counts, roster identities, episode
  digests, hard gates, HMC trace replay and family summaries.
- Phase 5: replace the Issue #55 evidence documents with the v2 measurements,
  roster and limitations.
- Phase 6: run the applicable locked verification gates and commit locally.

No push, pull request, merge, deployment or external publication is part of this
local execution.

## 6. Stop conditions

Stop and publish the failure if any hard gate trips, a trace fails strict replay,
a proposal is malformed, a family scenario fails closed-schema validation, a
metric is non-finite, or a validator must be loosened to complete the run.

Precedence is authority and replay/provenance, then metric integrity, then the
comparative outcome.

## 7. Required tests and records

- v2 preregistration digest is frozen and bound by both runner and tests.
- Family IDs and condition descriptors are deterministic and unique.
- Every family differs by declared operating, physical-plant or sensor condition.
- Full-horizon oracle candidates either complete the remaining episode or are
  excluded, with deterministic tie-breaking.
- Same family scenario and nonce produce identical episode and trace digests.
- Every advisory proposal enters HMC arbitration and every episode closes through
  strict trace parsing and replay.
- Evidence reports the complete family roster and states the finite oracle scope
  and simulation-only boundary.
