# Safety Case — Learned Advisory in AEOLUS

Claims–argument–evidence summary for the learned advisory path, in the
style of assurance-case practice (cf. UL 4600). Scope: the Habitat V2
research simulation only. This document asserts nothing about hardware,
production control, or qualification.

## Top-level claim

**Within the Habitat V2 simulation envelope, adding the learned adviser
cannot silently degrade safety relative to canonical HMC, and every safety
claim about it is independently re-checkable from the repository.**

### C1 — The model never commands

- **Argument:** All actuator authority flows through the deterministic HMC.
  The adviser emits proposals; HMC validates each against fixed safety
  rules and may accept, override, or reject. No code path executes model
  output directly.
- **Evidence:** `experiments/closed-loop-advisory-20260818/aeolus_closed_loop.py`
  (arbitration on every advised step); 81 recorded HMC overrides of 793
  proposals in the paired campaign; `SYSTEM.md` authority boundaries;
  enforced `DEMO_ONLY_PERMANENTLY_EXCLUDED` release tier in artifacts.
- **Residual risk:** future code could add a bypass path — mitigated by
  contract tests and code review, not by proof.

### C2 — Degraded inputs hand control back to the deterministic controller

- **Argument:** The forecaster was trained on complete telemetry only, so
  forecasting from missing sensors would be unsupported extrapolation. The
  harness therefore refuses to propose whenever any required telemetry is
  unavailable; HMC continues alone.
- **Evidence:** abstention guard in `aeolus_closed_loop.py` (merged PR #41),
  `adviser_abstentions_unavailable` counters in step records, unit-style
  guard verification, smoke parity on complete telemetry.
- **Residual risk:** partial-but-present *corruption* (wrong values, not
  missing ones) is not yet detected — see C5 and the drift-monitor roadmap.

### C3 — Advice measurably helps, under pre-registered scoring

- **Argument:** Benefit is claimed only on outcomes frozen before results
  were seen, on scenarios the model never trained on, against the strongest
  baseline (canonical HMC itself), with identical noise/seeds per pair.
- **Evidence:** frozen preregistrations (`preregistration-v2.json`, hashes
  recorded pre-run); 238-run results — 78 safer / 24 equal / 0 worse across
  102 fault pairs; 72 advised runs at zero exceedance; demo scenario
  19.94 → 0.0 with identical traces up to the intervention step.
- **Residual risk:** simulator-only evidence; ensemble/uncertainty and
  multi-seed sensitivity are open work.

### C4 — The evidence itself is intact and replayable

- **Argument:** Hash-chained control traces are validated by re-executing
  the deterministic policy and plant, not by hash checks alone — an
  internally consistent forgery is rejected. Demo artifacts are hash-pinned
  and loaders refuse modified bytes.
- **Evidence:** replay/validation tooling in `src/aeolus/habitat_v2/`;
  adversarial forgery tests in the suite; tour replay artifact hash check;
  fresh-clone reproduction of headline numbers.
- **Residual risk:** none identified within the simulation scope.

### C5 — Known limitations are part of the case, not footnotes

- One healthy EVA-transition pair scored 0.038 vs control 8.35 — a 99.5%
  reduction, not literal zero; reported as such.
- Advised runs cost more consumables (median +757 Wh, +1.97 mol O2,
  +6.04 mol sorbent); safety is bought with resources.
- The model has no uncertainty estimates and no learned abstention skill;
  both are scoped future work with explicit non-claims today.
- CI carries 7 documented pre-existing failures on the qualification
  branch (base-branch debt, unrelated to the advisory path).

## Verdict discipline

Any change that weakens a sub-claim (new bypass path, weakened guard,
edited pinned artifact, weakened baseline) invalidates this case and must
be recorded here before merge — the same "record the result rather than
weaken the baseline" rule the evaluation follows.
