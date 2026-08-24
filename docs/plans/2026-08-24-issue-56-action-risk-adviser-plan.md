# Issue #56 Implementation And Evaluation Plan

## Document control

- Branch: `research/issue-56-action-risk-adviser`
- Base: `origin/main` at `4ae41b7cc13fadb2d94552dab17dfae80dffe294`
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`
- Contract: `contracts/habitat_v2_forecast_issue_56_preregistration_v1.json`
- Design: `docs/plans/2026-08-24-issue-56-action-risk-adviser-design.md`

## Execution sequence

1. Freeze the preregistration before writing model code or running a comparative
   campaign.
2. Implement fail-closed feature projection, offline counterfactual labels,
   train-only fitting, validation-only calibration, and deterministic ranking.
3. Add tests for input provenance, shape/finite checks, train/validation split
   isolation, calibration, hard gates, deterministic tie-breaking, HMC proposal
   binding, and strict replay.
4. Run a bounded smoke corpus in a new ignored output directory. The smoke run is
   a path check and is not comparative evidence.
5. Run the declared development corpus, retaining TRAIN, VALIDATION, and
   EVALUATION family identities and all candidate labels. Do not access unrelated
   final or validation inputs.
6. Independently verify corpus digests, source identities, model artifact bytes,
   calibration provenance, proposal receipts, control traces, and replay closure.
7. Publish measured results, including failed gates, without post-result threshold
   or metric changes.

## Stop conditions

Stop if a feature uses hidden truth, a counterfactual cannot be replayed, a label is
non-finite, a split is crossed during fitting/calibration, a proposal bypasses
HMC, a trace fails replay, an output is overwritten, or a validator must be
weakened. A utility gain never compensates for a failed hard authority or safety
gate.

## Required evidence

- Frozen contract digest and source commit.
- Family roster and split manifest.
- Per-sample content hashes and candidate label counts.
- Train-only normalizers and coefficients.
- Validation-only calibration residual quantiles.
- Evaluation safety, comfort, resource, risk-calibration, abstention, and latency
  measurements.
- Zero authority, replay, provenance, and non-finite violations.
- Explicit capability and limitation card stating that the model remains advisory
  and the experiment is simulator-only.
