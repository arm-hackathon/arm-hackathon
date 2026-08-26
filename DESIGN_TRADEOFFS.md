# Design Tradeoffs

Every engineering decision in AEOLUS traded something away. This file records
what we chose, what we gave up, why, and what it cost — with the measured
evidence for each. Nothing here is hidden; the costs are part of the design.

Development evidence only. Nothing in this document is a qualification,
deployment, or safety claim.

## 1. Advisory-only model authority

**Decision:** the trained model proposes actions; the deterministic Habitat
Management Computer (HMC) reviews every proposal and remains the sole actuator
authority.

**Gave up:** any performance the model could have gained by commanding
actuators directly.

**Why:** a learned model can be confidently wrong in ways that are hard to
audit. A deterministic controller can be reviewed, tested, and replayed
exactly. We chose a system where every physical decision is auditable over one
where the model has full freedom.

**Measured cost:** in the paired closed-loop campaign, HMC overrode 81 of 793
model proposals (and 138 of 1,248 in the ensemble variant). Some overrides
rejected proposals that would have been fine. We accept that cost deliberately.

**Evidence:** [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

## 2. Safety over resources in the original campaign

**Decision:** in the 2026-08-18 paired campaign, the original model-advised
controller spent consumables aggressively to keep air-quality metrics inside
safety limits.

**Gave up:** resource frugality. The rule-based baseline is cheaper; the
advised arm was safer on that frozen 102-fault-pair roster.

**Measured cost:** median per-scenario deltas of approximately +757 Wh
battery, +1.97 mol oxygen, and +6.04 mol sorbent versus the canonical HMC.
Resource floors never broke in the campaign, but in a scarcity regime this
tradeoff would need active management.

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

**Decision:** the deployed demo model is a pure-NumPy artifact (8.4 MB, no
deep-learning framework at runtime), converted from the PyTorch checkpoint with
a byte-pinned identity check.

**Gave up:** peak throughput. An optimized torch/ONNX runtime with fused
kernels would be faster on large hardware.

**Why:** the project targets Arm-class efficiency and auditability. A NumPy
artifact runs identically on a laptop and a native Arm64 server with zero
dependency risk, and its forward pass is readable end-to-end by a reviewer.

**Measured result:** 192.7 microseconds median per full 8-step forecast on
native Arm64 (Arm Neoverse-N2, 1000 reps) — roughly 5,000 forecasts per
second, about 1000x faster than the habitat's control cadence requires. We
profiled for further gains (activation-function replacement), found none that
were both accurate and faster in pure NumPy, and reverted rather than ship a
gratuitous change.

**Evidence:** Arm64 evidence workflow (`.github/workflows/habitat-v2-live-forecast-arm64.yml`),
PRs #56 and #57.

## 4. A small MLP over sequence architectures

**Decision:** the forecaster is a 2.1M-parameter MLP (512-512-256 hidden
layers) over a 16-step window, not an LSTM, transformer, or state-space model.

**Gave up:** long-horizon expressiveness. The forecast horizon is capped at 8
steps, partly because an MLP cannot carry long temporal dependencies.

**Why:** a small MLP trains in hours on one GPU, can be fully understood by a
reviewer, and is honest about what it is. A larger architecture would be a
claim we could not yet evaluate with our scenario roster (the recorded V3
ensemble campaign did not distinguish model quality differences in outcomes on
that roster).

**Evidence:** `MODEL_CARD.md` and the [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

## 5. The simpler single model over the measured-better ensemble

**Decision:** the shipped demo and public commands use the single base model
(held-out normalized MAE 0.1146), not the five-member ensemble (0.1049).

**Gave up:** a measurably better accuracy number.

**Why:** in the pre-registered v3 paired campaign, the ensemble with a
disagreement penalty changed zero realized outcomes across 102 fault scenarios
(78 better / 24 equal / 0 worse vs control, identical to the single model).
Publishing the bigger number as the headline would conflate forecast accuracy
with closed-loop benefit. The ensemble is filed as a research study instead.

**Evidence:** [historical closed-loop advisory evidence
index](docs/evidence/closed-loop-advisory-historical-index.md#recorded-findings-and-static-audit).

## 6. Determinism over stochastic richness

**Decision:** active scenarios use seeded, byte-replayable determinism, and
active artifacts are hash-bound where their controlling contracts require it.

**Gave up:** the realism of richer stochastic simulation.

**Why:** current claims should carry an exact replay or verification path. The
historical 2026-08-18/19 campaign does not meet that current standard: its
committed records are hash-identified and internally coherent, but several raw
inputs and its execution-environment receipt are absent.

**Evidence:** replay verification commands in `README.md`; [historical evidence
gaps](docs/evidence/closed-loop-advisory-historical-index.md#known-custody-and-reproduction-gaps).

## 7. Float32 over quantization

**Decision:** the inference artifact is float32 throughout.

**Gave up:** a smaller file and (on some hardware) faster inference from 8-bit
quantization.

**Why:** exactness. The artifact is hash-pinned and its outputs are reproduced
bit-for-bit across machines. Quantization remains untested and would require a
separate gate proving no safety-relevant degradation. Issue #54 instead tested
knowledge distillation: smaller students often retained forecast accuracy, but
the tiny MLP lost action-ranking quality even while passing the accuracy gate.

**Evidence:** `MODEL_CARD.md`;
`docs/evidence/issue-54-distillation-card.md`.

## 8. Conservative abstention for the original model

**Decision:** if a required sensor goes silent, `action_aware_mlp_v1` abstains
and hands control back to HMC rather than forecasting outside its
complete-telemetry training domain.

**Gave up:** availability of that model — one broken required sensor disables
its advisory output entirely.

**Why:** imputing unsupported inputs into this artifact would create an
unmeasured extrapolation. Abstention keeps its proposals inside the measured
domain. A separate Issue #53 dropout-aware forecast lane now exists and passed
its frozen independent-dropout gates. That evidence does not cover correlated
or mixed dropout, resource-gauge dropout, adversarial channel loss, deployment,
or actuator authority.

**Evidence:** `SAFETY_CASE.md`; `docs/evidence/issue-53-dropout-card.md`;
abstention guard tests (`tests/habitat_v2/` forecast tests).

## 9. Honest scoping over marketing

**Decision:** every artifact carries its evidence tier (`DEMO_ONLY`,
development evidence only); limitations are documented next to results.

**Gave up:** stronger-sounding claims.

**Why:** this is a simulation research project, not a qualified control
system. The credibility of the evidence is the product.

**Evidence:** `SAFETY_CASE.md`, `MODEL_CARD.md`, `CORPUS_DATASHEET.md`.

---

*The unifying rule: at every fork we chose safety and verifiability over raw
capability — and documented the cost instead of hiding it.*
