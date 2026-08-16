# Habitat V2 FP32 Arm benchmark handoff

Date: 2026-08-14

## Status

This branch is a work-in-progress optimisation candidate based on immutable submitted-demo commit:

```text
422b3931052db1f935344b04052bdfb60a70ff10
```

It is not the submitted candidate, a completed optimisation, qualification evidence, a release or a deployment.

## Goal

Compare the existing FP64 action-aware ridge model with an FP32 version using identical real live-demo inputs, then benchmark both on the same native Arm64 runner.

## Implemented

- FP32 conversion preserves the existing forecast, authority and release-tier contracts.
- The model loader accepts either the original FP64 schema or the explicit FP32 schema and preserves the stored runtime dtype.
- Ridge inference uses the model's declared FP64 or FP32 dtype rather than silently converting FP32 back to FP64.
- The converter records exact source and candidate hashes, compressed file bytes and raw numeric-array bytes.
- Regeneration guarantees canonical uncompressed `.npy` payloads and equal arrays,
  not cross-platform identity of the DEFLATE-compressed `.npz` container.
- The committed FP32 `.npz` is frozen benchmark evidence bound by its recorded
  byte length and SHA-256; native runs benchmark those exact bytes without rewriting them.
- The FP32 candidate halves raw model-array bytes by construction.
- A real live-demo comparison checks all four action-conditioned forecasts against the predeclared numerical gate.
- The benchmark alternates FP64-first and FP32-first execution order over the same captured live-demo history and actions.
- The receipt separates size, memory, prediction drift, median latency, p95 latency and native-architecture claims.

## Predeclared quality gate

```text
maximum over all outputs of:
abs(fp32 - fp64) / max(abs(fp64), 1.0)
<= 1e-4
```

The gate was declared before native Arm results are available.

## Current files

- `src/aeolus/habitat_v2/forecast/arm_optimization.py`
- `src/aeolus/habitat_v2/forecast/baselines.py`
- `src/aeolus/habitat_v2/forecast/live_demo.py`
- `tests/habitat_v2/test_forecast_arm_optimization.py`

## Verification completed locally

```text
56 forecast tests passed
Ruff passed
py_compile passed
git diff --check passed
```

The local host is AMD64. Local timings are development diagnostics only and are not Arm evidence.

## Missing work

1. Add a small reproducible CLI that:
   - converts the frozen FP64 artifact to FP32
   - writes the conversion receipt
   - runs `benchmark_fp64_vs_fp32`
   - writes canonical JSON
   - exits non-zero if prediction parity fails
2. Generate and commit the exact FP32 candidate artifact and static conversion receipt.
3. Add a separate workflow for branch `ben/habitat-v2-fp32-arm-benchmark` on `ubuntu-24.04-arm`.
4. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before Python starts.
5. Run the same FP64 and FP32 artifact bytes in one job.
6. Upload the candidate model, conversion receipt, benchmark receipt, `uname`, `lscpu` and a SHA-256 manifest.
7. Inspect actual Arm metrics before writing any Devpost claim.

## Required claim boundaries

Allowed only after a successful native receipt:

- FP64 to FP32 reduced-precision optimisation
- exact raw-array and compressed-file size reductions
- measured FP64 and FP32 latency on the named Arm64 runner
- prediction drift under the frozen threshold, if it passes

Do not claim without separate evidence:

- INT8 optimisation
- Arm-specific operators or kernels
- NEON use
- Azure, Cobalt, Graviton or Performix deployment
- physical board, NPU, energy or thermal evidence
- qualification or production deployment
- learned actuator authority

Deterministic HMC remains the sole command and actuator authority.

## Ownership

Alex owns the next optimisation implementation step. Avoid concurrent edits to the files listed above. Ben owns the Devpost wording. Atlas reviews the exact candidate and native receipt before any result is added to the entry.
