# Canonical experiment evidence environment

AEOLUS supports Python `>=3.10` as a library. That compatibility statement does
not mean every interpreter and dependency resolution must emit byte-identical
learned artifacts. The canonical evidence workflow is intentionally narrower:

- CPython 3.11;
- the committed `uv.lock` without resolution changes;
- a clean Git worktree at the recorded source commit;
- new, empty output and artifact directories under ignored `out/`.

Run a canonical experiment with a new run identifier:

```bash
uv sync --locked --python 3.11 --extra ml
PYTHONPATH=src uv run --locked --python 3.11 --extra ml \
  python -m aeolus.experiment \
  scenarios/sweep-v2.json \
  out/evidence-py311-<run-id> \
  out/evidence-py311-<run-id>-artifacts
```

The runner rejects non-empty output or artifact directories. It writes
`experiment-receipt.json` beside the generated sweep/corpus evidence. The
receipt binds the output to:

- source commit and whether the source worktree was dirty;
- SHA-256 of `uv.lock`;
- Python implementation and version;
- NumPy, ONNX and ONNX Runtime versions;
- the actual ONNX IR version and imported opsets;
- sweep/corpus hashes and SHA-256 values for all generated artifacts.

## Promotion policy

Generated files under `out/` are candidates, not canonical repository evidence.
Before replacing any tracked file under `artifacts/`:

1. verify `source_worktree_dirty` is `false` in the candidate receipt;
2. verify each receipt hash against the candidate artifact bytes;
3. inspect metric differences and classify them as operational, numerical-only,
   exporter/protobuf-only, or unexplained;
4. copy the three reviewed artifacts and their receipt into `artifacts/` in a
   dedicated evidence-only commit;
5. record the command and receipt SHA-256 in the commit/review notes.

A same-environment byte match proves deterministic regeneration for that
recorded environment. A mismatch across different Python or exporter versions
requires semantic comparison before any claim of equivalence. Do not edit ONNX
IR metadata merely to imitate an older file hash.

## ONNX acceptance

The experiment measures Python-versus-ONNX raw-softmax-score error on the first
512 scored rows of the current IID test partition (or all rows if fewer are
available). An export is rejected when its maximum absolute error exceeds
`1e-5`; the metrics artifact records both the observed error and acceptance
bound, and the experiment receipt binds that metrics artifact by SHA-256. This
is exporter parity evidence, not Arm hardware evidence or a deployment-safety
claim.
