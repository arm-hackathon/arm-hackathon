# Issue #55 Controller Race вЂ” Normative Plan

## 1. Document control

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Branch: `research/issue-55-controller-race` (created from `origin/main@099a141`)
- Preregistration: `contracts/habitat_v2_forecast_issue_55_preregistration_v1.json`
  (byte-frozen before any run)
- Preregistration SHA-256 (LF-normalized): `17C601D7F15A21804AA68B26024C96D44642491E07A9BD75BDE805E027C773CF`
- Design note: `docs/plans/2026-08-24-issue-55-controller-race-design.md`
- This-lane status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`
- Sign-off status: preregistration committed before any run; repository-owner
  review of this PR is the recorded sign-off request. Ben's explicit sign-off is
  acknowledged as pending and is not claimed.

## 2. Scope and non-goals

Race three controllers over one preregistered scenario suite with identical
conditions and measure the fraction of the rules-to-oracle gap that the frozen
forecast model closes. Non-goals follow the preregistration verbatim: no
authority change, no frozen-surface mutation, no oracle in any demo, no
deployment claim, no post-hoc metric tuning.

## 3. Fixed protocol (mirrors the preregistration)

- Suite: 32 families; family `i` = development scenario with
  `sensor_model.random_seed + 1000*i`, renamed, extended to 96 steps via
  `extend_scenario_for_issue52`.
- Episode nonce: `SHA256("issue55-episode-nonce-v1|" + family_id)` вЂ” identical
  across arms.
- Decision steps: 16, 20, ..., 84 (18 per episode); lookahead 8 steps.
- Arms: `rules_only`, `model_advised`, `oracle_instrument`.
- Target bounds: the preregistered 51-column Track A table (values identical to
  the frozen Issue #52 scoring bounds and the Issue #54 evaluation arrays).

## 4. Implementation units

1. `src/aeolus/habitat_v2/forecast_issue55_race.py`
   - family/nonce/decision-step derivation (pure, deterministic);
   - true-plant target projection onto the 51-column layout;
   - metric functions (safety exposure, violation steps, comfort, resource);
   - advisory point ranking (`issue55-advisory-point-ranking-v1`);
   - oracle lookahead selection (`issue55-oracle-lookahead-v1`);
   - episode runner with the Issue #54 collector's discipline: canonical
     proposals only, per-step shadow digest cross-check, strict trace parse +
     replay, digest-bound episode records, no wall-clock.
2. `tests/habitat_v2/test_forecast_issue55_race.py`
   - preregistration digest binding; decision-step and nonce derivation;
   - metric correctness on synthetic states; ranking correctness incl. hard
     gate and tie-breaks; oracle selection on synthetic scores;
   - one real short `rules_only` episode (replay closure, authority invariants,
     determinism of the episode digest); one real short `model_advised` episode
     (proposal admitted, HMC final-say invariant).
3. `scripts/run_issue55_controller_race.py`
   - one command; write-once `out/` directory; loads frozen contracts and the
     SHA-pinned teacher; runs all 3 arms x 32 families; emits canonical JSON
     records, manifests, per-arm summaries, gap closure with bootstrap CI;
     refuses wall-clock in digest-bearing output.

## 5. Execution phases

- Phase 0: preregistration + docs (this commit).
- Phase 1: module + tests + runner; focused suite green.
- Phase 2: small smoke run (2 families) into `out/issue55-smoke-1` to validate
  the pipeline; smoke output is development validation only and is not evidence.
- Phase 3: full preregistered run into `out/issue55-race-1`; independent
  post-run validation of counts, digests, and replay invariants.
- Phase 4: evidence docs (`issue-55-measurements.md`, `issue-55-race-card.md`)
  with the results table, gap-closure headline + CI, rejections, limitations.
- Phase 5: full verification gate, commit, push, PR with `Closes #55`.

## 6. Stop conditions

Stop and publish the failure if: any hard gate trips; any episode fails strict
replay; any proposal is malformed (attempt class `REJECTED_INPUT`); metrics are
non-finite; or the pipeline cannot complete without loosening a validator.

Precedence: authority, replay/provenance, then metric integrity, then the
race outcome itself.

## 7. Required tests

- Preregistration digest is frozen.
- `decision_steps(96)` == 16..84 step 4; eligibility rule respected.
- Nonce is family-bound and arm-independent.
- Safety exposure is zero inside bounds and positive on crossings; comfort is
  zero at nominals; resource composite floors at zero.
- Advisory ranking: perfect predictions rank the true best first; predicted
  hard crossing makes a candidate ineligible; tie-break prefers hold then id;
  mode-inapplicable actions are ineligible for the advisory arm.
- Oracle selection: argmin of the preregistered lookahead score with
  infeasible exclusion and empty-rule fallback.
- Episode determinism: same scenario + nonce в†’ identical episode digest.
- Authority invariants: proposals only at decision steps; arbitration always
  yields the final command; episode records bind trace and replay digests.
