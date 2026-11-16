# NVIDIA DGX Spark execution record

## What this establishes

On 13 November 2026, AEOLUS 0.8.0 was installed from a wheel and executed on
an NVIDIA DGX Spark. The Habitat V2 compound-fault simulation ran on its Arm64
CPU. A separate synthetic neural-network probe ran on the NVIDIA GB10 GPU.
This is historical software-execution evidence, not learned-model qualification,
physical-system validation, or permission to operate equipment.

Source: [`4e1d5aced1214875c72c611cadd0677218d77196`](https://github.com/arm-hackathon/arm-hackathon/commit/4e1d5aced1214875c72c611cadd0677218d77196).
The successful recorded attempt is `preflight-v3`. Later revisions require their
own execution checks; this record is not a rolling compatibility guarantee.

## Recorded results

| Scope | Measurement | Result | Evidence |
| --- | --- | --- | --- |
| AEOLUS CPU simulation | Installed wheel | `aeolus-0.8.0-py3-none-any.whl`; import resolved inside the isolated package environment, not the source tree | `container-preflight.log` |
| AEOLUS CPU simulation | Compound-fault scenario | 4 steps; 5 states including initial; all 5 invariant statuses passed | `installed-compound-faults.jsonl` |
| AEOLUS CPU simulation | Fault receipt | Five expected faults present after the first step; none active in the final state | Trace and source-pinned installed-wheel verifier |
| Forecast setup | Frozen contract loading | `FORECAST_D1_CONTRACT_OK` | `container-preflight.log` |
| Synthetic GPU probe | Network / precision | 19,080 parameters; FP32 | `runtime.json`, `runtime_probe.py.txt` |
| Synthetic GPU probe | Optimisation | 37 Adam updates: 5 warm-up and 32 timed; all parameter tensors changed; finite loss, gradients and parameters enforced | Runtime and probe source |
| Synthetic GPU probe | Timed interval | 0.08368740099831484 seconds; 83.69 ms rounded | `runtime.json` |
| Synthetic GPU probe | Throughput | 48,944.045951223634 synthetic examples/s; about 48,944 rounded | `runtime.json` |
| Synthetic GPU probe | CUDA allocator peaks | 29,739,520 bytes allocated (28.36 MiB rounded); 46,137,344 bytes reserved (44 MiB) | `runtime.json` |
| Synthetic forward check | CPU/GPU parity | Passed with `rtol=0.0001`, `atol=0.00001` | `runtime.json` |
| Synthetic CPU resume | Fresh-process continuation | Recorded exact model and optimiser-state equality at update 12 | `runtime.json`, probe source and log; checkpoint binaries omitted |
| Whole preflight | Container completion | Exit code 0; `AEOLUS_SPARK_PREFLIGHT_CORE_OK` | `container-status.json`, log |

The installed-wheel verifier is available at the recorded source commit as
[`scripts/verify_installed_habitat_v2.py`](https://github.com/arm-hackathon/arm-hackathon/blob/4e1d5aced1214875c72c611cadd0677218d77196/scripts/verify_installed_habitat_v2.py).
It checks installed-package identity, validates the trace against the scenario,
and checks fault activation and clearance. Passing internal invariants is not
external validation of the physics, and four steps are not a long-duration study.
Contract loading does not mean that a forecaster was evaluated on the GPU.

## Environment and provenance

- Architecture: `aarch64`; GPU: `NVIDIA GB10`; compute capability: `12.1`.
- Python: `3.12.3`; PyTorch: `2.10.0a0+b558c986e8.nv25.11`; CUDA build: `13.0`;
  NumPy: `2.2.6`.
- The NVIDIA PyTorch image is pinned by index, Arm64 manifest and config digests
  in [image-receipt.json](image-receipt.json). Those digests identify the recorded
  environment; this is not a promise that the image will remain downloadable.
- Installation used an additive virtual environment with local wheels. The
  container's image-level PyTorch/CUDA stack was unchanged. The recorded
  container had no network, a four-CPU limit and a 16 GiB memory limit.
- [input-verification.json](input-verification.json) records the source commit,
  tree, wheel and script hashes, and 27 reviewed-source file checks. Its original
  staging inventory is preserved. The `canonical_archive` entry identifies the
  LF-canonical replacement source archive used for `preflight-v3`, not the older
  archive in the initial inventory.
- [provenance.json](provenance.json) maps the retained original files to their
  export hashes, identifies omitted binaries and explains the redacted receipt.
  All six original exported files were hash-checked before packaging this subset.

The original trace, runtime receipt, input receipt and log are byte-for-byte
copies. The probe and runner are also byte-exact copies of the inputs named in
the remote receipt, stored with `.txt` suffixes as historical source rather than
new supported entry points. The log retains a PyTorch TF32 deprecation warning;
the run still completed successfully. No errors or trace rows were edited out.

The host endpoint and account identity are not published. `container-status.json`
is explicitly derived from the original receipt, with container/user identifiers
omitted and its duplicated image object replaced by a link. Generic `/workspace`
container paths remain in the original log and source; these are not host paths.

## Interpreting the numbers

The GPU network is a small two-convolution/ReLU network with average pooling and
an eight-output linear layer, not an AEOLUS learned controller or the proposed
causal TCN. Its input shape is `[128, 32, 64]`. The probe repeatedly reuses one
random synthetic batch and target, with Adam at learning rate `0.001`.

Five updates warm up the probe. The next 32 updates process 4,096 repeated
synthetic example presentations. The timer is bracketed by CUDA synchronisation;
finite-value checks remain inside the update loop. Throughput is
`32 * 128 / timed_seconds`. Data loading and host-to-device input transfer are
outside that interval. This is one short timing sample, with no repeat-run
uncertainty estimate, sustained-load study, energy measurement or comparison
against another device. Allocated/reserved memory is PyTorch allocator usage,
not total device or system memory consumption.

The CPU/GPU check compares an untrained forward pass, not training trajectories.
The checkpoint probe is separate CPU work: save after six updates, continue to
12, and compare a fresh process resumed from six against that reference. It uses
fixed synthetic batches, not an AEOLUS data pipeline or full sampler/data-cursor
resume. The final synthetic loss in `runtime.json` is a diagnostic value, not a
prediction-quality result. BF16 capability was detected but not benchmarked.

No AEOLUS learned model was trained or qualified in this run. No GPU-accelerated
simulator, end-to-end AEOLUS training throughput, deployment readiness or physical
performance is established. This record does not alter any research promotion
gate or reinterpret a negative model result.

## Verify the retained evidence

From the repository root, with GNU `sha256sum` available (including Git Bash):

```bash
(cd docs/evidence/dgx-spark-2026-11-13 && sha256sum -c SHA256SUMS)
```

The bundle's `.gitattributes` disables line-ending conversion so the recorded
bytes survive Windows/Linux checkouts. Checksums verify byte integrity against
the checked-in manifest, not independent attestation that a machine executed the
run. The receipts and trace are recorded execution evidence, not a new rerun.

To inspect the recorded invariant results without rerunning the simulator, use
Python 3 from the repository root:

```bash
python -c 'import json; from pathlib import Path; rows = [json.loads(line) for line in Path("docs/evidence/dgx-spark-2026-11-13/installed-compound-faults.jsonl").read_text().splitlines()]; assert [row["step"] for row in rows] == list(range(5)); assert all(row["invariant_status"]["passed"] is True for row in rows); print("5 recorded states; all recorded invariant statuses passed")'
```

This inspects recorded statuses, not a new simulation or independent physical
validation. Use the checksum check first. It does not repeat the GPU probe.

### Exact-replay portability limitation

The original installed-wheel verifier performed exact deterministic trace replay
inside the Spark run. During evidence packaging, the same validator did **not**
accept these bytes on Windows/AMD64 with Python `3.11.15` and NumPy `2.4.6`:
`TraceValidationError: trace does not match deterministic replay`. A fresh local
simulation completed and its parsed output differed in 134 numeric values, with
no other differences. The Spark used a different architecture, operating system,
Python and NumPy version; this check does not isolate which difference caused
the numerical drift.

Do not claim bit-exact replay across those environments. The original trace was
not rewritten and the validator was not weakened. File-integrity verification
and inspection of recorded results remain distinct from replay verification.

The synthetic `.pt` checkpoint and `.npz` expected-state binaries, wheel, source
archive and container image are not bundled. Their available recorded identities
remain in the receipts. Consequently this small bundle does not independently
replay the checkpoint test or reproduce the whole original container run.
A fresh hardware run would need those inputs or a separately recorded rebuild,
the pinned compatible environment and authorised hardware access. No new Spark
access was used to prepare this record.
