# BDM-v1 one-shot blind confirmation protocol — DRAFT (WITHHELD, UNAUTHORIZED)

Status: **WITHHELD and UNAUTHORIZED.** This document is a structural draft
written as part of Issue #77's failure path. It is not an active protocol,
confers no authorization, and must not be executed. The BLIND_FINAL partition
remains sealed; no blind outcome exists anywhere in this repository.

## Why this draft exists and why it is withheld

Issue #77 preregisters that a hash-bound candidate packet and a proposed
one-shot blind protocol follow **only if all development gates pass**. The
closed-loop development study failed its gates (the calibrated BDM-v1 arm
inherits the Issue #75 null: no admitted useful interventions, no safety
benefit over HMC hold at causal-group level). Therefore:

- no candidate packet is frozen;
- this draft stays WITHHELD and UNAUTHORIZED;
- a future redirected protocol (new features, labels, or operating regime)
  may reuse this structure only after its own development gates pass, a
  candidate packet is frozen, and repository-owner approval is recorded in
  writing.

## Structural outline for future use (inactive)

1. **Entry conditions.** All Issue #77-order development gates pass on a
   frozen candidate; candidate packet binds source, corpus, split, scaler,
   weights, thresholds, scorer, action catalogue, and protocol identities by
   SHA-256; owner approval recorded; blind pilot power justification already
   frozen in the Issue #72 custody registry.
2. **One-shot rule.** Exactly one preregistered evaluation run over
   BLIND_FINAL families; no tuning, no threshold adjustment, no second pass;
   any deviation voids the run and reseals the partition.
3. **Arms.** Frozen candidate versus the preregistered reference arms
   (HMC rules, hold, frozen linear baseline) on identical exogenous traces;
   HMC retains sole arbitration, final-command, plant-step, and replay
   authority; the candidate proposes or abstains only.
4. **Statistics.** Causal-group unit; preregistered gate order (zero hard
   violations, per-family non-inferiority, paired benefit, ranking/regret,
   useful-action precision/recall, resource/comfort after safety); group-level
   bootstrap CIs; tie equals not-beat; negative outcome published unchanged.
5. **Lineage.** Per-decision records (proposal, abstention reason, HMC
   disposition, final/achieved command, outcome, replay digest, model-path
   latency) written once and hash-bound into a confirmation receipt.
6. **Aftermath.** Pass: confirmation receipt published, roadmap gate review
   opens. Fail: negative published, blind stays sealed forever for this
   candidate, no runtime or release issue is authorized.

## Non-goals (unchanged)

No real-habitat, hardware-control, deployment, or production-safety claim; no
qualification or certification claim; no ONNX/quantisation/Arm optimisation or
release work; no online recalibration or model self-update; no re-opening of
the blind for any reason other than a future authorized one-shot run.
