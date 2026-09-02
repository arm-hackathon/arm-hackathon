# Safety Case — Learned Advisory in AEOLUS

Claims–argument–evidence summary for the learned advisory path, in the
style of assurance-case practice (cf. UL 4600). Scope: the Habitat V2
research simulation only. This document asserts nothing about hardware,
production control, or qualification.

## Top-level claim

**Within the Habitat V2 simulation envelope, learned advice remains subordinate
to deterministic HMC authority, with evidence identities and replay paths
recorded for independent checking. This authority boundary does not guarantee
that an HMC-admitted proposal is beneficial.**

### C1 — The model never commands

- **Argument:** All actuator authority flows through the deterministic HMC.
  The adviser emits proposals; HMC validates each against fixed safety
  rules and may accept, override, or reject. No code path executes model
  output directly.
- **Evidence:** current authority implementation and tests in
  `src/aeolus/habitat_v2/hmc.py`,
  `src/aeolus/habitat_v2/forecast/live_demo.py`, and
  `src/aeolus/habitat_v2/forecast/live_mlp_demo.py`, plus
  `tests/habitat_v2/test_forecast_live_mlp_demo.py`; the [historical evidence
  index](docs/evidence/closed-loop-advisory-historical-index.md) records the
  former arbitration harness, authority document, and 81 historical overrides
  of 793 proposals; the current loaders enforce non-authority release tiers
  (`DEVELOPMENT_EVIDENCE_ONLY` for the MLP and
  `DEMO_ONLY_PERMANENTLY_EXCLUDED` for the ridge demo).
- **Residual risk:** future code could add a bypass path — mitigated by
  contract tests and code review, not by proof.

### C2 — Each model follows its declared degraded-input contract

- **Argument:** `action_aware_mlp_v1` was trained on complete telemetry only,
  so its harness refuses to propose whenever any required telemetry is
  unavailable; HMC continues alone. The separate Issue #53 successor carries
  missingness as explicit input and may forecast only within its frozen,
  independently dropped observation contract. It remains forecast-only.
- **Evidence:** the [archived authority and availability
  surfaces](docs/evidence/closed-loop-advisory-historical-index.md#archived-authority-and-availability-surfaces)
  identify the historical guard, proposal path, counter, and their limitations.
  PR #41 records the change, but the retained V3 result contains zero
  unavailable-input abstentions and does not itself exercise the guard.
  `docs/evidence/issue-53-dropout-card.md` separately records the Issue #53
  successor's preregistered forecast-lane gates.
- **Residual risk:** Issue #53 does not qualify correlated or mixed dropout,
  resource-gauge dropout, adversarial channel loss, other out-of-distribution
  missingness, or partial-but-present corruption.

### C3 — Benefit and harm are reported under declared scoring

- **Argument:** the original archive labels its scoring plan frozen before
  outcomes and records held-out scenarios, canonical-HMC comparison, and paired
  seeds/noise. The plan and result first entered Git together, and the raw V2
  runs and training archive are absent. Repository history therefore does not
  independently establish freeze chronology, training exclusion, or every
  paired-run detail. The recorded V3 freeze time also conflicts with its commit
  chronology, as detailed in the evidence index. These are recorded historical
  claims, not current-main rerun results, and are not generalized to other
  controller fixtures or learned advisory lanes.
- **Evidence:** the [historical evidence
  index](docs/evidence/closed-loop-advisory-historical-index.md) records the
  preregistration, custody limitation, and 238-run result: 78 safer / 24 equal /
  0 worse across 102 fault pairs. The later Issue #55
  controller race is the required counter-evidence: HMC admitted and applied
  all eight point-model proposals, while mean normalized safety exposure was
  `15.678536800` versus `0.000217557` for rules-only. The Issue #56 V3
  risk-filtered lane passed its bounded six-family safety gate but made only two
  proposals and abstained 76 times. See
  `docs/evidence/issue-55-race-card.md` and
  `docs/evidence/issue-56-action-risk-v3-support-revision.md`.
- **Residual risk:** all results are fixed-roster simulator evidence. HMC
  authority prevents a learned bypass; it does not prove non-degradation after
  HMC admits a proposal, broad useful action selection, or generalization.

### C4 — Current replay paths and historical gaps are explicit

- **Argument:** Current-main hash-chained control traces are validated by
  re-executing the deterministic policy and plant, not by hash checks alone.
  Current demo artifacts are hash-pinned and loaders refuse modified bytes.
- **Evidence:** replay/validation tooling in `src/aeolus/habitat_v2/`;
  adversarial forgery tests in the suite; and the tour replay artifact hash
  check.
- **Residual risk:** the historical 2026-08-18/19 campaign is not reproducible
  from a fresh current-main checkout. Raw V1/V2 results, training/environment
  receipts, and old runner dependencies are missing; see the [historical
  evidence index](docs/evidence/closed-loop-advisory-historical-index.md#known-custody-and-reproduction-gaps).

### C5 — Known limitations are part of the case, not footnotes

- One healthy EVA-transition pair scored 0.038 vs control 8.35 — a 99.5%
  reduction, not literal zero; reported as such.
- Across the 102 fault pairs, the advised-minus-control medians were +757 Wh,
  +1.97 mol O2, and +6.04 mol sorbent; safety is bought with resources.
- The original model has no calibrated uncertainty and uses a rule-based
  missing-input guard. Issue #53 adds bounded missingness calibration and
  abstention; Issue #56 adds development-only action-risk lanes. Neither result
  changes the original model or authorizes deployment.
- Issue #54 shows why forecast error is insufficient by itself: its tiny MLP
  students passed the declared accuracy gate while losing action-ranking
  quality.

## Verdict discipline

Any change that weakens a sub-claim (new bypass path, weakened guard,
edited pinned artifact, weakened baseline) invalidates this case and must
be recorded here before merge — the same "record the result rather than
weaken the baseline" rule the evaluation follows.

## Related work and references

The architectural patterns in this case are not novel inventions; they are
deliberate implementations of established safety and evaluation practice.

**Authority boundary (C1).** The HMC-plus-adviser split is a run-time
assurance / safety-filter architecture: a complex, unverified component is
supervised by a simpler verified controller with a backup policy.
Hobbs, Mote, Abate, Coogan & Feron, "Run Time Assurance for Safety-Critical
Systems: An Introduction to Safety Filtering Approaches for Complex Control
Systems," *IEEE Control Systems Magazine*, 2023
(<https://arxiv.org/abs/2110.03506>). Domain precedent for advisory-only
fault management on life support: Pachura, Suleiman & Mendler, "ARGES: an
Expert System for Fault Diagnosis Within Space-Based ECLS Systems," NASA,
1988 (<https://ntrs.nasa.gov/citations/19880019996>).

**Forecast-and-propose mechanism (C3).** Scoring candidate actions by
simulated future outcomes is model-predictive control with learned
dynamics; the adviser is a greedy, safety-gated instance over the fixed
action catalogue. Nagabandi, Kahn, Fearing & Levine, "Neural Network
Dynamics for Model-Based Deep Reinforcement Learning with Model-Free
Fine-Tuning," *ICRA*, 2018 (<https://arxiv.org/abs/1708.02596>); Chua,
Calandra, McAllister & Levine, "Deep Reinforcement Learning in a Handful
of Trials using Probabilistic Dynamics Models" (PETS), *NeurIPS*, 2018
(<https://arxiv.org/abs/1805.12114>) — PETS's probabilistic ensembles remain
relevant context for the separate uncertainty and action-risk research lanes;
no qualification follows from that similarity.

**Abstention (C2).** The original guard is rule-based selective
classification. The Issue #53 successor represents missingness explicitly and
reports measured abstention rate, precision and recall alongside interval
coverage under its bounded contract.
Geifman & El-Yaniv, "Selective Classification
for Deep Neural Networks," *NeurIPS*, 2017
(<https://arxiv.org/abs/1705.08500>); Geifman & El-Yaniv, "SelectiveNet:
A Deep Neural Network with an Integrated Reject Option," *ICML*, 2019
(<https://arxiv.org/abs/1901.09192>).

**Missing-sensor forecasting (C2/C5).** The Issue #53 lane treats missingness
as evidence through masks, observation age, and mask-aware slopes rather than
silently imputing it away. Its independent-dropout evidence does not qualify
the broader missingness patterns listed in C2. Cao et al., "BRITS: Bidirectional
Recurrent Imputation for Time Series," *NeurIPS*, 2018
(<https://arxiv.org/abs/1805.10572>); Che et al., "Recurrent Neural
Networks for Multivariate Time Series with Missing Values" (GRU-D),
*Scientific Reports* 8:6085, 2018
(<https://doi.org/10.1038/s41598-018-24271-9>).

**Documentation practice.** `MODEL_CARD.md` and `CORPUS_DATASHEET.md`
follow Mitchell et al., "Model Cards for Model Reporting," *FAT\**, 2019
(<https://arxiv.org/abs/1810.03993>) and Gebru et al., "Datasheets for
Datasets," *CACM* 64(12), 2021 (<https://arxiv.org/abs/1803.09010>).
Safety-roadmap framing follows EASA, "Artificial Intelligence Roadmap 1.0,"
2020 (bounded operational domain, traceability, human oversight, and a
clear line between simulation evidence and deployment assurance).

**Evaluation methodology (C3/C4).** Frozen preregistrations follow the
registered-reports pattern (Chambers, "Registered Reports: A New Publishing
Initiative at Cortex," *Cortex* 49(3), 2013,
<https://doi.org/10.1016/j.cortex.2012.12.016>) and the NeurIPS
reproducibility program (Pineau et al., "Improving Reproducibility in
Machine Learning Research," *JMLR* 22(164), 2021,
<https://www.jmlr.org/papers/v22/20-303.html>). Current-main replay paths and
independently re-executed trace validation are aligned with ACM artifact review
and badging, v1.1; the historical campaign's disclosed gaps prevent making the
same claim for that archive
(<https://www.acm.org/publications/policies/artifact-review-and-badging-current>).

**Domain grounding.** Katipamula & Brambley, "Methods for Fault Detection,
Diagnostics, and Prognostics for Building Systems — A Review, Part I,"
*HVAC&R Research* 11(1), 2005
(<https://doi.org/10.1080/10789669.2005.10391123>); Lance & Malin, "An
Expert Systems Approach to Automated Fault Diagnostics" (CS-1 FIXER,
regenerative ECLSS), SAE 851380, 1985
(<https://ntrs.nasa.gov/citations/19860038824>).
