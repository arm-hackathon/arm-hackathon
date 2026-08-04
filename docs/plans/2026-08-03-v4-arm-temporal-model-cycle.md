# V4 Arm-Compatible Temporal Model Cycle Implementation Plan

> **For Hermes:** Execute task-by-task with strict RED-GREEN-REFACTOR discipline. Do not use the frozen v3 final suite for v4 fitting, calibration, candidate selection, or acceptance.

**Goal:** Diagnose the frozen v3 MLP failure, train and calibrate bounded temporal candidates on fresh v4 development families, and accept a candidate only if it beats the calibrated rules on validation without an unacceptable nominal false-alarm or fault-recall regression.

**Architecture:** Preserve protocol v3 unchanged as historical evidence. Add an isolated v4 development module that compares the existing temporal MLP with a width-4, three-layer causal dilated NumPy TCN and deterministic confidence/persistence gates. Export candidate probabilities through standard ONNX operators and record deployment receipts; the deterministic gate remains explicit post-processing. No final suite or response-layer integration is authorised unless the v4 development gate passes.

**Pre-result audit amendment (2026-08-03):** Before any v4 development result was retained, independent architecture and methodology audits identified insufficient receptive field in the initial one-layer CNN and leakage from using validation for epoch, rule, and gate calibration. This plan therefore fixes four train seeds for fitting, two train seeds for internal calibration, six separately clustered validation seeds for one-time candidate evaluation, stride-one operational evaluation, and SHA-256 deduplication of shared healthy references. No validation metric informed this amendment.

**Post-result protocol correction (2026-08-03):** Independent review found that transition windows updated causal state but were excluded from first-detection latency. The corrected contract counts any causal detection at or after observable onset for latency while continuing to exclude transition windows from classification and false-alert metrics. This correction does not change fitting, calibration, safety thresholds, or safety criteria; the corrected run recomputes latency and any latency-dependent tie-breaks from fresh output paths.

**Tech stack:** Python 3.11, NumPy, ONNX opset 17, ONNX Runtime CPUExecutionProvider, pytest, Ruff.

---

## Frozen constraints

- `docs/protocol-v3-acceptance.md` and its recorded metrics are immutable historical evidence.
- V3 final rows may be used only by an explicitly marked forensic command. Its output cannot be accepted by v4 training/selection APIs.
- V4 development contains only fresh `train` and `validation` families with canonical identities disjoint from v3.
- All model inputs remain exact observable `float32[10,24]` windows under `model_input_v1`.
- Model fitting uses train seed clusters `700..703`; epoch selection, rule calibration, and gate calibration use train-only clusters `704..705`; candidate evaluation uses validation clusters `900..905` once after candidates are fixed.
- Candidate selection requires a strict seed-cluster mean macro-F1 win over identically evaluated calibrated rules.
- False-alert episodes must be `<= 10` per 1,000 eligible healthy ticks and no more than `2` per 1,000 above rules; nominal false-alarm regression must be no more than `0.01`; every fault recall must be within `0.02` of rules or better.
- No latency-only escape clause for v4 development acceptance.
- No Yaroslav response-layer integration unless `development_gate_passed=true`.
- No Arm performance claim from local x86 measurements.
- INT8 calibration inputs, if produced after a development pass, come from training rows only.

## Predeclared candidates

1. `temporal_mlp_balanced_raw` — current MLP and raw argmax; diagnostic reference.
2. `temporal_mlp_balanced_gated` — same MLP with train-internal-calibration-selected fault threshold/persistence.
3. `temporal_cnn_balanced_gated` — width-4 causal dilated TCN with equal-total class weighting.
4. `temporal_cnn_sqrt_gated` — same TCN with square-root inverse-frequency weighting.

No additional candidate may be introduced after v4 validation metrics are inspected without starting a new protocol version.

## Predeclared alert-gate grid

- fault probability threshold: `0.50, 0.60, 0.70, 0.80, 0.90`
- persistence windows: `1, 2, 3, 5`
- decision: output `nominal` below threshold; otherwise choose the highest-probability fault class; emit a fault only after the same candidate fault persists for the configured number of causal windows
- reset at every canonical stream boundary
- excluded-transition rows update causal state and may establish first detection at or after observable onset, but remain absent from confusion-matrix, class-recall, and false-alert metrics

Selection order within one base model:

1. candidates satisfying the absolute/relative false-alert episode bounds, false-alarm regression `<= 0.01`, and every fault recall delta `>= -0.02` against calibrated rules;
2. seed-cluster mean macro-F1 descending;
3. window macro-F1 descending;
4. false-alert episodes ascending;
5. causal median latency ascending;
6. threshold descending (more conservative);
7. persistence ascending.

If no gate candidate satisfies the constraints, retain the best diagnostic candidate but mark it ineligible.

## V4 development acceptance rule

A learned candidate passes only when all are true on fresh validation families:

- model seed-cluster mean macro-F1 is strictly greater than calibrated-rule seed-cluster mean macro-F1;
- false-alert episodes are `<= 10` per 1,000 eligible healthy ticks and no more than `2` per 1,000 above rules;
- model nominal false-alarm rate minus rule rate is `<= 0.01`;
- every fault-class recall delta is `>= -0.02`;
- Python/FP32 ONNX maximum absolute probability error is `<= 1e-5`;
- model and gate artifacts are strict, finite, hash-bound, and reproducible;
- the selected ONNX graph uses the declared standard operator allowlist.

## Task 1: Freeze a forensic error-report contract

**Files:**
- Create: `src/aeolus/error_analysis.py`
- Create: `tests/test_error_analysis.py`

**Steps:**
1. Write a failing test for an API that accepts explicitly supplied rows, model predictions, and family evidence and returns errors grouped by true class, predicted class, family, scenario role, and operating-profile identity.
2. Require the report format to contain `evidence_role: historical_forensic_only` and source manifest/model hashes.
3. Add a failing test proving v4 development rejects this report object as an input.
4. Run focused tests and capture RED.
5. Implement the minimal strict report builder and JSON writer.
6. Run focused tests to GREEN.

## Task 2: Add a fresh v4 development sweep

**Files:**
- Create: `scenarios/sweep-v4-development.json`
- Modify: `src/aeolus/sweep.py`
- Modify: `tests/test_sweep.py`

**Steps:**
1. Write failing tests for `aeolus_sweep_v4` accepting exactly role `development` with `train` and `validation`, and rejecting `final`.
2. Require fresh train seeds `700..705` and validation seeds `900..905`; retain the same fault classes and target coverage while using predeclared v4 operating-profile identities.
3. Assert exact generated family counts and canonical disjointness from generated v3 development/final manifests.
4. Implement only the schema support needed for v4 development.
5. Generate the canonical v4 development suite into ignored output and verify the receipt.

## Task 3: Add deterministic alert-gate evaluation

**Files:**
- Create: `src/aeolus/model_cycle_v4.py`
- Create: `tests/test_model_cycle_v4.py`

**Steps:**
1. Write a failing test for nominal fallback below the fault threshold.
2. Implement minimal threshold behaviour.
3. Write a failing test for same-class causal persistence and stream reset.
4. Implement minimal persistence state.
5. Write a failing test proving excluded-transition rows update gate state but do not enter metrics.
6. Implement ordered stream evaluation.
7. Write failing tests for the fixed gate grid, finite values, deterministic tie-breaking, and false-alarm/recall eligibility constraints.
8. Implement gate calibration using only train-internal seed clusters `704..705`; reserve validation for fixed-policy evaluation.

## Task 4: Add the compact temporal Conv1D model

**Files:**
- Create: `src/aeolus/temporal_cnn.py`
- Create: `tests/test_temporal_cnn.py`

**Model:**
- input: `float32[N,10,24]`
- train-only per-channel normalisation
- causal Conv1D `24 → 4`, kernel 3, dilation 1, left pad 2; ReLU
- causal Conv1D `4 → 4`, kernel 3, dilation 2, left pad 4; ReLU
- causal Conv1D `4 → 4`, kernel 3, dilation 4, left pad 8; ReLU
- Conv1D `4 → 4`, kernel 1; take the final causal timestep
- softmax probabilities
- receptive field: 15 ticks, covering the full 10-tick input
- learned parameters: 416
- deterministic Adam training and fixed initialization seed

**Steps:**
1. Write a failing forward-shape/probability test.
2. Implement the minimal detector forward path.
3. Write a failing deterministic-training/round-trip test.
4. Implement full-batch deterministic training with precomputed convolution patches and Adam.
5. Add the equal-total and square-root inverse-frequency weighting modes as exactly two declared options.
6. Write malformed/non-finite/shape tests before strict JSON persistence.
7. Implement strict save/load and parameter-count receipts.

## Task 5: Add FP32 ONNX export and deployment receipts

**Files:**
- Modify: `src/aeolus/temporal_cnn.py`
- Modify: `tests/test_temporal_cnn.py`
- Create: `src/aeolus/edge_benchmark.py`
- Create: `tests/test_edge_benchmark.py`

**Steps:**
1. Write a failing ONNX structure test requiring only `Sub`, `Div`, `Transpose`, `Conv`, `Relu`, `Gather`, and `Softmax` plus graph metadata; assert dilations and one-sided causal padding.
2. Implement opset-17 FP32 export with fixed `[N,10,24]` input semantics.
3. Write parity-bound tests and run ONNX Runtime parity on validation rows.
4. Write failing benchmark-receipt schema tests: model hash/bytes, runtime, platform, CPU, threads, warmups, iterations, batch size, median/p95 latency, throughput, and explicit `hardware_claim=false` unless architecture is AArch64.
5. Implement a deterministic-input local benchmark runner. Local results are readiness evidence only.

## Task 6: Implement the v4 development runner

**Files:**
- Modify: `src/aeolus/model_cycle_v4.py`
- Modify: `tests/test_model_cycle_v4.py`

**Steps:**
1. Write a failing integration test that accepts only a v4 development manifest containing exactly train/validation.
2. Train the current MLP and both Conv1D weighting variants.
3. Fit on train clusters `700..703`; select epochs, calibrate rules, and gate each learned candidate on train-only clusters `704..705`; evaluate fixed candidates once on validation clusters `900..905`.
4. Build stride-one operational windows from trusted traces and count each shared healthy reference once by canonical SHA-256.
5. Emit every candidate result, including losers.
6. Select using the predeclared gate and development acceptance rule.
7. If no candidate passes, emit `development_gate_passed=false`, persist diagnostics, and do not produce a response-layer artifact.
8. If one passes, save its base model, gate policy, FP32 ONNX, hashes, ONNX parity, parameter count, artifact bytes, and operator inventory.
9. Reject existing output paths and non-finite evidence.

## Task 7: Run the canonical development experiment

**Output:** ignored directory under `out/v4-development-<run-id>/`.

**Steps:**
1. Generate fresh v4 development scenarios and corpus.
2. Run the v4 development runner once from a new output path.
3. Inspect candidate metrics and error diagnostics.
4. Attempt to falsify the selected result by replaying stream order, checking family counts, checking manifest disjointness from v3, and recomputing artifact hashes.
5. Run the local FP32 benchmark only as x86 readiness evidence.
6. Produce a tracked acceptance record only after the canonical result is known.

## Task 8: Conditional INT8 readiness

**Files:**
- Modify only if the development gate passes: `src/aeolus/edge_benchmark.py`, `tests/test_edge_benchmark.py`, and the v4 acceptance record.

**Steps:**
1. Add static INT8 calibration using representative training-only windows.
2. Verify INT8 model validity and output quality on validation.
3. Record local size/quality/parity evidence without claiming Arm speed.
4. If the development gate fails, record `INT8 deferred: learned candidate did not clear quality gate` rather than optimising a rejected model.

## Task 9: Documentation and closeout

**Files:**
- Create: `docs/v4-development-acceptance.md`
- Modify: `PLAN.md`, `README.md`, `PROBLEM.md` only to reflect measured results.

**Steps:**
1. State what failed in v3 and what each v4 ablation tested.
2. Record exact family counts, hashes, metrics, operator inventory, parameter count, artifact bytes, commands, and local-platform boundary.
3. State whether the development gate passed.
4. If failed, prohibit final-suite generation and Yaroslav response integration.
5. If passed, authorise a separately reviewed v4 final protocol; do not generate it in this branch.
6. Run full pytest, Ruff, `uv lock --check`, ONNX checker, JSON parsing, diff hygiene, and independent methodology/code review.
7. Commit locally under Benedict Anokye-Davies; do not push or create a PR without approval.
