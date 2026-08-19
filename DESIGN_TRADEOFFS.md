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

**Evidence:** `experiments/closed-loop-advisory-20260818/CLOSED_LOOP_REPORT_V2.md`,
`CLOSED_LOOP_REPORT_V3.md`.

## 2. Safety over resources

**Decision:** the model-advised controller spends consumables aggressively to
keep air-quality metrics inside safety limits.

**Gave up:** resource frugality. The rule-based baseline is cheaper; the
advised controller is safer.

**Measured cost:** median per-scenario deltas of approximately +757 Wh
battery, +1.97 mol oxygen, and +6.04 mol sorbent versus the canonical HMC.
Resource floors never broke in the campaign, but in a scarcity regime this
tradeoff would need active management.

**Why:** the project's thesis is that safety exceedances are the primary harm;
resources are the budget you spend to avoid them. We state the price openly
rather than presenting the safety win as free.

**Evidence:** `experiments/closed-loop-advisory-20260818/CLOSED_LOOP_REPORT_V2.md`.

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
claim we could not yet evaluate with our scenario roster (the v3 ensemble
campaign proved the roster already cannot distinguish model quality
differences in outcomes).

**Evidence:** `MODEL_CARD.md`, `experiments/closed-loop-advisory-20260818/CLOSED_LOOP_REPORT_V3.md`.

## 5. The simpler single model over the measured-better ensemble

**Decision:** the shipped demo and public commands use the single base model
(held-out normalized MAE 0.1146), not the five-member ensemble (0.1049).

**Gave up:** a measurably better accuracy number.

**Why:** in the pre-registered v3 paired campaign, the ensemble with a
disagreement penalty changed zero realized outcomes across 102 fault scenarios
(78 better / 24 equal / 0 worse vs control, identical to the single model).
Publishing the bigger number as the headline would conflate forecast accuracy
with closed-loop benefit. The ensemble is filed as a research study instead.

**Evidence:** `experiments/closed-loop-advisory-20260818/CLOSED_LOOP_REPORT_V3.md`,
`preregistration-v3.json`.

## 6. Determinism over stochastic richness

**Decision:** scenarios run with seeded, byte-replayable determinism; every
artifact is hash-chained.

**Gave up:** the realism of richer stochastic simulation.

**Why:** every claim in this repository can be independently re-run and
verified byte-for-byte. For a project whose credibility rests on its evidence,
reproducibility outranks realism.

**Evidence:** replay verification commands in `README.md`; paired-experiment
determinism guard in `run_ensemble_paired.py` (control-arm reruns matched the
frozen v2 record to 16 decimal places).

## 7. Float32 over quantization

**Decision:** the inference artifact is float32 throughout.

**Gave up:** a smaller file and (on some hardware) faster inference from 8-bit
quantization.

**Why:** exactness. The artifact is hash-pinned and its outputs are reproduced
bit-for-bit across machines. Quantization is filed as future work under the
compression research study, gated on proving no safety-relevant degradation.

**Evidence:** `MODEL_CARD.md`; issue #54 tracks the compression study.

## 8. Conservative abstention over model uptime

**Decision:** if a required sensor goes silent, the adviser abstains and hands
control back to HMC rather than forecasting from incomplete telemetry.

**Gave up:** model availability — one broken sensor disables the model
entirely.

**Why:** forecasting from missing inputs means guessing with confidence
scores that no longer mean anything. Abstention keeps every proposal inside
the model's validated domain. The availability-aware model (issue #53) is the
planned fix, gated on a proper masked-telemetry corpus.

**Evidence:** `SAFETY_CASE.md`; abstention guard tests
(`tests/habitat_v2/` forecast tests).

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
