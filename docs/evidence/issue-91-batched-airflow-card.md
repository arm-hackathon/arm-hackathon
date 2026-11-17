# Issue #91: reference-checked batched airflow backend

Date: 2026-11-13
Issue: [#91](https://github.com/arm-hackathon/arm-hackathon/issues/91) (simulation, research-only)
Measured source commit: `6fe5b11817280718a3b90878674394277aa7dd56` (clean worktree in every run)
Contract: `contracts/habitat_v2_air_network_batched_v1.json`, canonical SHA-256
`7cc46b05839bee4e9024f611af39766a8647d5c7803b0be47dae666b11269e01`
Derivation and layout: [`docs/research/2026-11-13-issue-91-air-network-closed-form.md`](../research/2026-11-13-issue-91-air-network-closed-form.md)
Version impact: `minor` (declared; numeric bump, changelog, lock and source-binding changes deferred to the release owner)

**Status: CPU evidence complete; the CUDA/DGX acceptance item is open.** No CUDA
device was available, so no GPU measurement exists and none is claimed.

## What was built

- `src/aeolus/habitat_v2/air_network_analytic.py`: closed-form FP64 NumPy
  operating point for the existing quadratic fan, shared resistance and
  parallel branches; batch layout; supported-domain validation; deterministic
  synthetic case generator; per-variable comparison; frozen-contract loader.
- `src/aeolus/habitat_v2/air_network_batched_torch.py`: optional eager PyTorch
  FP32/FP64 backend with the same algebra. PyTorch is imported lazily and is
  not a declared dependency.
- `contracts/habitat_v2_air_network_batched_v1.json`: layout, supported domain,
  case profiles with their ranges, and per-variable tolerances. Its digest is
  pinned in code.
- `scripts/benchmark_issue91_air_network_batched.py`: benchmark command that
  writes one write-once receipt.
- Three test files under `tests/habitat_v2/test_issue91_air_network_*.py`.

`air_network.py`, the reviewed source packages, HMC, the plant and replay are
unchanged. The reviewed solver stays authoritative.

## Receipts

Receipts are under ignored `out/` and are not committed.

| Run | Receipt | SHA-256 |
| --- | --- | --- |
| Windows 10, AMD64, NumPy only | `out/issue91-air-network-batched-v2/windows-numpy/benchmark-receipt.json` | `b554b5b6b333ca1c25256cae9af2319dc301b41f3bf8db53c9dce33eb4fb61af` |
| Linux (WSL2), x86_64, PyTorch CPU, default 14 threads | `out/issue91-air-network-batched-v2/linux-wsl2-torch-cpu-default-threads/benchmark-receipt.json` | `c44307b9e5a49a65a24176928ec05e46d3d6670840c2d99a9eb5ba922aa1e450` |
| Linux (WSL2), x86_64, PyTorch CPU, 1 thread | `out/issue91-air-network-batched-v2/linux-wsl2-torch-cpu-1-thread/benchmark-receipt.json` | `24ed5c365320eae15a920bc4575ec7e0fb67c9079c19571cd32a31e5f958d3cc` |

Environment: Python 3.11, NumPy 2.4.6, PyTorch 2.15.0+cpu (Linux runs only),
14 logical CPUs. All three runs used seed 91, eight branches, 20 warm-up and
200 measured iterations, and the full grid 1, 8, 32, 128, 512, 2048. Input
identities are the same in all three receipts.

PyTorch could not be loaded natively on the Windows host: an Application
Control policy blocks its DLLs. That policy was left alone, and the PyTorch
runs were made in WSL2 on the same machine, from a fresh clone of the same
commit.

Every timing below is **provisional**. The host was shared with other
workloads, each number is a single median, and implementations were measured
one after another. They are development-machine measurements, not hardware
claims.

## Accuracy

Every gated comparison passed in all three runs (7 gates NumPy-only, 21 with
PyTorch). The table gives the worst observed error as a fraction of the frozen
allowance, over all variables and batch sizes.

| Comparison | Case profile | Worst error / allowed | Worst variable |
| --- | --- | --- | --- |
| NumPy FP64 vs reviewed bisection solver | fixture band, 6 batch sizes | 0.0010 | residual |
| NumPy FP64 vs 80-digit root | stress, 128 worlds | 0.0066 | air power |
| PyTorch FP64 (CPU) vs NumPy FP64 | fixture band | 0.0082 | air power |
| PyTorch FP64 (CPU) vs NumPy FP64 | stress, 2048 worlds | 0.011 | branch pressure |
| PyTorch FP32 (CPU) vs NumPy FP64 | fixture band | 0.043 | air power |
| PyTorch FP32 (CPU) vs NumPy FP64 | stress, 2048 worlds | 0.052 | branch pressure |

In absolute terms: NumPy FP64 agrees with the reviewed solver to about `1e-15`
of each variable's physical scale on the fixture band, and PyTorch FP32 agrees
with NumPy FP64 to about `5e-7` relative.

The tolerances were declared from error analysis before any run and turned out
20 to 1000 times looser than the observed error. They were not tightened
afterwards.

The stress-profile comparison against the reviewed solver is reported but not
gated. On this sample it also passed (worst 0.0027 of the allowance). It is
left ungated because the reviewed bisection is known to be unconverged in some
fan-dominated corners of the supported domain; see Limitations.

Beyond the benchmark, the test suite checks the candidate against every solver
call the plant makes while running all twelve checked-in air-network scenarios
(360 calls, including faulted, slewing and fan-off states), and against direct
substitution into the physical laws.

## Latency and throughput

Medians in microseconds per call. "Validated" includes supported-domain
validation; "kernel" excludes it. PyTorch "transfer-inclusive" covers host
validation, copy to tensors, the kernel and copy back.

Linux (WSL2), PyTorch limited to 1 thread:

| Batch | Scalar reference loop | NumPy validated | NumPy kernel | Torch FP32 kernel | Torch FP32 transfer-inclusive | Torch FP64 kernel | Torch FP64 transfer-inclusive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 105.6 | 37.4 | 20.2 | 52.6 | 190.7 | 42.7 | 105.9 |
| 8 | 847.5 | 57.4 | 16.4 | 50.0 | 151.7 | 52.2 | 109.1 |
| 32 | 4,203 | 41.3 | 17.0 | 54.6 | 169.6 | 45.4 | 104.9 |
| 128 | 16,284 | 77.6 | 23.9 | 56.0 | 135.1 | 49.4 | 142.4 |
| 512 | 63,310 | 94.7 | 57.8 | 81.5 | 172.5 | 67.0 | 228.1 |
| 2048 | 267,588 | 232.0 | 186.7 | 117.1 | 367.0 | 140.9 | 279.7 |

Linux (WSL2), PyTorch at its default 14 threads:

| Batch | NumPy validated | NumPy kernel | Torch FP32 kernel | Torch FP32 transfer-inclusive | Torch FP64 kernel | Torch FP64 transfer-inclusive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40.9 | 16.3 | 53.4 | 132.1 | 73.8 | 123.3 |
| 8 | 54.2 | 15.5 | 53.6 | 196.4 | 45.6 | 167.6 |
| 32 | 58.1 | 21.1 | 75.1 | 138.7 | 83.1 | 168.8 |
| 128 | 77.1 | 37.6 | 60.3 | 179.7 | 54.9 | 171.4 |
| 512 | 97.4 | 69.3 | 15,524 | 16,049 | 15,071 | 15,519 |
| 2048 | 319.1 | 247.1 | 15,933 | 16,039 | 14,748 | 14,865 |

Windows, NumPy only:

| Batch | Scalar reference loop | NumPy validated (p99) | NumPy kernel | Worlds per second | Peak host bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 122.8 | 42.1 (52.3) | 16.8 | 2.4e4 | 4,064 |
| 8 | 961.3 | 41.0 (89.6) | 17.1 | 2.0e5 | 7,824 |
| 32 | 3,964 | 42.8 (161.9) | 19.5 | 7.5e5 | 20,496 |
| 128 | 15,817 | 55.2 (86.5) | 26.4 | 2.3e6 | 71,184 |
| 512 | 61,428 | 91.7 (120.6) | 61.4 | 5.6e6 | 273,936 |
| 2048 | 248,393 | 218.6 (1,363) | 174.9 | 9.4e6 | 1,019,408 |

Full distributions (minimum, median, p95, p99, maximum, mean) for every cell
are in the receipts.

What the measurements show:

- **NumPy closed form vs the scalar loop.** At batch 2048 the validated NumPy
  solve took about 0.22 to 0.32 ms against about 250 to 280 ms for 2048 scalar
  bisection solves, roughly a thousand times less. At batch 1 it is about
  three times less. This is a kernel comparison only.
- **PyTorch on CPU, transfer-inclusive: no crossover.** It was slower than
  validated NumPy at every tested batch size, in both thread settings and both
  precisions.
- **PyTorch on CPU, kernel only, 1 thread: one crossover at 2048.** The FP32
  kernel (117 µs) and the FP64 kernel (141 µs) were faster than the NumPy
  kernel (187 µs) at batch 2048 only. This is one median on a shared host and
  should be treated as weak.
- **Unsuccessful configuration: default threading.** With 14 intra-op threads
  the PyTorch CPU kernel took about 15 ms at batch 512 and 2048, some 60 to
  220 times slower than NumPy, against about 0.06 ms at batch 128. Limiting
  PyTorch to one thread removed the effect. The cause was not investigated
  further; thread oversubscription inside WSL2 is the likely explanation.
- **CUDA: not evaluated.** Every `cuda/*` crossover entry is `not_evaluated`.

Peak memory: NumPy host peaks are in the table (about 1 MB at batch 2048).
PyTorch CPU peak memory was **not measured**; the receipts record it as null
with the reason. CUDA peak memory is implemented but has never run.

## Limitations

- **No CUDA or DGX evidence.** The CUDA code paths (device transfer,
  synchronisation, peak device memory) have never executed. CUDA tests are
  skipped with an explicit reason and are not counted as passes.
- **Agreement with the reviewed solver is declared for the fixture band
  only.** At one fan-dominated corner of the supported domain at full speed,
  the reviewed solver's 100-iteration bisection is unconverged: its fan
  pressure is off the exact root by `2.75e-11` of shutoff pressure, 27 times
  the fixture-band tolerance, while the candidate matches the 80-digit root.
  Across the wider domain the candidate is checked against that independent
  root instead.
- **Tiny speeds differ by design, and agreement is never bitwise.** The
  candidate keeps flow linear in speed for any positive speed. The reviewed
  solver returns zero flow once the squared speed underflows (near `1e-170`)
  and raises `ZeroDivisionError` at `5e-324`. Only a speed of exactly zero
  gives exact zero outputs from both.
- **FP32 underflow.** FP32 pressure and power are not resolved below `1e-37`
  in their units, which for fixture-scale plants means speeds below about
  `1e-20` (pressure) and `1e-13` (power).
- **The supported domain is narrower than the reviewed solver's.** Inputs
  outside the declared box are rejected, not solved.
- **The contract was revised once before these results**, after independent
  review: a `1e-300` absolute term was added to the FP64 comparisons for
  subnormal pressure scales, and two descriptions were corrected. Earlier
  exploratory receipts from before that revision are not reported.

## Checklist to evidence

| Issue checklist item | Evidence | State |
| --- | --- | --- |
| Algebra, units and supported domain documented; code compared with the current scalar solver and independent physical checks | Derivation record; `test_candidate_matches_reviewed_scalar_solver_on_fixture_band`, `test_every_solver_call_of_the_checked_in_scenario_runs_is_reproduced`, `test_candidate_matches_independent_high_precision_root_on_stress_domain`, `test_outputs_satisfy_the_physical_laws_by_direct_substitution`, affinity, monotonicity and equivalent-branch tests | Met |
| Deterministic generator records seed and ranges; per-variable absolute and relative tolerances fixed before results | Contract `case_profiles` and `tolerances`; digest pinned in `air_network_analytic.CONTRACT_CANONICAL_SHA256`; generator bytes pinned for both profiles | Met, with the one pre-result revision noted above |
| CPU tests run without Torch or GPU; optional-backend tests fail clearly for unavailable CUDA and distinguish skipped from passed | Locked environment without PyTorch: 100 passed, 40 skipped, each skip naming PyTorch. With PyTorch CPU: 135 passed, 5 skipped, each naming CUDA. `test_cuda_request_without_cuda_fails_clearly_and_never_falls_back` | Met |
| Measured CPU and CUDA grid at 1, 8, 32, 128, 512, 2048 with warm-up, synchronisation, latency distribution, throughput, peak memory and transfer-inclusive time; measured crossover or none | Three CPU receipts above | CPU met, except PyTorch CPU peak memory (not measured). **CUDA open** |
| Real DGX receipt binding source SHA, image digest, Python/Torch/CUDA versions, precision, input identity, tolerances, max errors and timing method | The receipt schema carries all of these and refuses to run unless the measured modules are the recorded checkout | **Open: hardware unavailable** |
| Outputs write-once and reproducible; existing numerical, replay and contract tests and installed-wheel smoke remain green | Write-once and failure-path tests; accuracy and input identities identical across the three receipts. Full locked suite at `6fe5b11`: 1429 passed, 44 skipped (baseline `main`: 1329 passed, 4 skipped; the 40 added skips are the PyTorch tests, absent from the locked environment). `ruff check .`, `compileall`, `uv lock --check` and `git diff --check` pass. The observability packet rebuilds byte-identically (`1afed658...866b`) and the installed-wheel smoke test passes; the wheel ships both new modules and importing them does not import PyTorch | Met |

## Boundaries and non-claims

Research-only kernel. It has no actuator authority and no replay or hash
authority, and it does not change the authoritative solver, HMC, plant or any
trace. It adds no physics and is not an arbitrary duct-network solver. A
kernel timing here is not an end-to-end simulator speedup, a model improvement
or a control benefit. Nothing here is a hardware, qualification or deployment
claim. No protected final or BLIND_FINAL data was read or produced.

Ben's separate scalar analytic diagnostic was looked for on the remote on
2026-11-13 and not found, so nothing was reused from it. The only prior
analytic material is the single-branch oracle already on `main`, which this
work agrees with.

## Reproduction

```bash
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest -q \
  tests/habitat_v2/test_issue91_air_network_analytic.py \
  tests/habitat_v2/test_issue91_air_network_batched_torch.py \
  tests/habitat_v2/test_issue91_air_network_benchmark.py

# NumPy only (no PyTorch needed); choose a new directory each time
uv run --locked --python 3.11 --extra dev python \
  scripts/benchmark_issue91_air_network_batched.py \
  --output out/issue91-air-network-batched-<new>
```

With PyTorch installed separately in the environment that runs the script:

```bash
python scripts/benchmark_issue91_air_network_batched.py \
  --output out/issue91-air-network-batched-<new> \
  --torch-devices cpu --torch-num-threads 1
```

For the open hardware item, on the DGX from a clean checkout of the commit
under test:

```bash
python scripts/benchmark_issue91_air_network_batched.py \
  --output out/issue91-air-network-batched-<new> \
  --torch-devices cpu cuda \
  --image-digest sha256:<digest of the container image> \
  --hardware-label "<hardware name>" \
  --timing-conditions dedicated_host
```

The command exits 2 without writing anything if PyTorch or CUDA is missing. The
image digest and hardware label are recorded as operator declarations; the
script cannot verify them, so a reviewer should check them against the
recorded device name.
