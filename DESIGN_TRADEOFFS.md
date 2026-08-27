# Design Tradeoffs

Every engineering decision in AEOLUS traded something away. This file records
what we chose, what we gave up, why, and what it cost — with the measured
evidence for each. Nothing here is hidden; the costs are part of the design.

Development evidence only. Nothing in this document is a qualification,
deployment, or safety claim.

## 1. Advisory-only model authority

**Decision:** the historical adviser uses model forecasts to propose actions;
the deterministic Habitat Management Computer (HMC) reviews every proposal and
remains the sole actuator authority.

**Gave up:** any performance the model could have gained by commanding
actuators directly.

**Why:** a learned model can be confidently wrong in ways that are hard to
audit. A deterministic controller can be reviewed, tested, and replayed
exactly. We chose a system designed so every command crosses a reviewable
deterministic authority boundary.

**Measured observation:** HMC modified or replaced 81 of 793 historical V2
proposals and 138 of 1,248 V3 proposals. These counts show that arbitration was
exercised. The archive lacks the counterfactual evidence needed to decide
whether any individual override was unnecessary.

**Evidence:** [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

## 2. Safety over resources in the original campaign

**Decision:** in the 2026-08-18 paired campaign, the original model-advised
controller spent consumables aggressively to keep air-quality metrics inside
safety limits.

**Gave up:** resource frugality. The rule-based baseline is cheaper; the
advised arm was safer on that recorded 102-fault-pair roster.

**Measured cost:** across the 102 fault pairs, median advised-minus-control
deltas were approximately +757 Wh battery, +1.97 mol oxygen, and +6.04 mol
sorbent. The historical report states that resource floors did not break, but
the raw V2 step records are absent; in a scarcity regime this tradeoff would
need active management.

**Why:** the project's thesis is that safety exceedances are the primary harm;
resources are the budget you spend to avoid them. We state the price openly
rather than presenting the safety win as free.

**Evidence:** [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

**Later finding:** this tradeoff is not a general property of learned advice.
In the Issue #55 controller race, the point-model arm improved mean comfort but
worsened mean normalized safety exposure to `15.678536800`, versus
`0.000217557` for rules-only, while also using more resources. HMC authority
prevented a learned bypass, but did not prevent admitted proposals from causing
harm over the episode. See
`docs/evidence/issue-55-race-card.md`.

## 3. Pure-NumPy inference over optimized runtimes

**Decision:** the current MLP demo artifact is pure NumPy (8.4 MB, no
deep-learning framework at runtime) with a byte-pinned NumPy identity. It is
associated with the historical PyTorch checkpoint; the original conversion
receipt is absent.

**Gave up:** potential peak throughput. An optimized Torch/ONNX runtime with
fused kernels could offer higher throughput on larger hardware.

**Why:** the project targets Arm-class efficiency and auditability. The NumPy
forward pass is readable end-to-end without a deep-learning runtime, and the
native Arm64 workflow demonstrates that the pinned artifact executes on the
target architecture. NumPy remains a runtime dependency.

**Measured result:** 192.7 microseconds median per full 8-step forecast on
native Arm64 (Arm Neoverse-N2, 1000 reps) — roughly 5,000 single forecasts per
second. This is forward-pass latency, not end-to-end control-loop latency.

**Evidence:** Arm64 evidence workflow (`.github/workflows/habitat-v2-live-forecast-arm64.yml`),
PRs #56 and #57.

## 4. A small MLP over sequence architectures

**Decision:** the forecaster is a 2.1M-parameter MLP (512-512-256 hidden
layers) over a 16-step window, not an LSTM, transformer, or state-space model.

**Gave up:** long-horizon expressiveness. The forecast horizon is capped at 8
steps; the fixed 16-step window and lack of recurrent state limit access to
longer histories.

**Why:** the MLP structure is comparatively compact and inspectable. A larger
architecture would be a claim we could not yet evaluate with the recorded
scenario roster: the V3 ensemble campaign did not distinguish model-quality
differences in its top-level threshold-exceedance and terminal-status fields on
that roster.

**Evidence:** `MODEL_CARD.md` and the [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

## 5. The simpler single model over the reported-lower-error ensemble

**Decision:** the public MLP demo command uses the single base MLP (recorded
held-out normalized MAE 0.1146), not the historical five-member ensemble. Its
historical report gives 0.1049, but the cited evaluation file is absent, so
that ensemble figure is not independently checkable from the archive.

**Gave up:** the reported lower ensemble error figure.

**Why:** in the recorded V3 campaign, the ensemble-with-penalty arm matched the
V2 single-model arm on top-level threshold-exceedance and terminal-status
fields across all 102 fault scenarios (78 better / 24 equal / 0 worse versus
control). The archive did not emit penalty-firing statistics, so proposal-count
changes cannot be attributed to the penalty. Publishing the reported error as
the headline would conflate forecast accuracy with closed-loop benefit. The
ensemble is filed as historical research instead.

**Evidence:** [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

## 6. Determinism over stochastic richness

**Decision:** active scenarios use seeded, byte-replayable determinism, and
active artifacts are hash-bound where their controlling contracts require it.

**Gave up:** the realism of richer stochastic simulation.

**Why:** current claims should carry an exact replay or verification path. The
historical 2026-08-18/19 campaign does not meet that current standard: its
committed bytes are hash-identified and selected top-level fields reconcile,
but several raw inputs and its execution-environment receipt are absent.

**Evidence:** replay verification commands in `README.md`; [historical evidence
gaps](docs/evidence/closed-loop-advisory-historical-index.md#known-custody-and-reproduction-gaps).

## 7. Float32 over quantization

**Decision:** the current MLP demo artifact uses float32 arrays throughout.

**Gave up:** a smaller file and (on some hardware) faster inference from 8-bit
quantization.

**Why:** retaining float32 avoids introducing untested quantization error. The
artifact bytes are hash-pinned and the native Arm64 workflow executed them
successfully; the available evidence does not establish cross-architecture
bit equality. Quantization remains untested and would require a separate gate
proving no safety-relevant degradation. Issue #54 instead tested knowledge
distillation: smaller students often retained forecast accuracy, but the tiny
MLP lost action-ranking quality even while passing the accuracy gate.

**Evidence:** `MODEL_CARD.md`;
`docs/evidence/issue-54-distillation-card.md`.

## 8. Conservative abstention for the original model

**Decision:** if a required sensor goes silent, `action_aware_mlp_v1` abstains
and hands control back to HMC rather than forecasting outside its
complete-telemetry training domain.

**Gave up:** availability of that model — one broken required sensor disables
its advisory output entirely.

**Why:** imputing unsupported inputs into this artifact would create an
unmeasured extrapolation. Abstention avoids this missing-input extrapolation; it
does not establish broader in-distribution status. A separate Issue #53
dropout-aware forecast lane now exists and passed its frozen
independent-dropout gates. That evidence does not cover correlated or mixed
dropout, resource-gauge dropout, adversarial channel loss, deployment, or
actuator authority.

**Evidence:** `SAFETY_CASE.md`; [archived authority and availability
surfaces](docs/evidence/closed-loop-advisory-historical-index.md#archived-authority-and-availability-surfaces);
`docs/evidence/issue-53-dropout-card.md`.

## 9. Honest scoping over marketing

**Decision:** the current forecast demo artifacts carry explicit non-authority
tiers (`DEMO_ONLY_PERMANENTLY_EXCLUDED` or `DEVELOPMENT_EVIDENCE_ONLY`), and
limitations are documented next to results.

**Gave up:** stronger-sounding claims.

**Why:** this is a simulation research project, not a qualified control
system. The credibility of the evidence is the product.

**Evidence:** `SAFETY_CASE.md`, `MODEL_CARD.md`, `CORPUS_DATASHEET.md`.

---

*The unifying rule: at every fork we chose safety and verifiability over raw
capability — and documented the cost instead of hiding it.*
