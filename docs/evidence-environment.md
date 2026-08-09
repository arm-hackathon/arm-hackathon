# Canonical evidence environment

AEOLUS supports Python `>=3.10` as a library. That compatibility statement does
not mean every interpreter and dependency resolution must emit byte-identical
learned artifacts. The canonical protocol-v3 evidence workflow is intentionally
narrower:

- CPython 3.11;
- the committed `uv.lock` without resolution changes;
- a clean Git worktree at the recorded source commit; and
- new, empty output paths under ignored `out/`.

The current canonical procedure is the staged development-selection →
final-evaluation path in [protocol v3 acceptance](protocol-v3-acceptance.md).
It does not use the legacy one-command `sweep-v2` runner as final evidence.

## Evidence boundaries

The development command creates a detector JSON artifact, an ONNX artifact and
a strict policy. The policy binds development-manifest, detector and contract
metadata and records validation-only candidate selection, rule calibration,
model/rule comparison and ONNX parity.

The final command requires the expected SHA-256 for the policy, detector JSON,
detector ONNX, development manifest and final manifest. It refuses to write to
an existing report. Before evaluating final rows, it replays development
candidate training/selection, ONNX parity and rule calibration, then verifies
the saved validation comparison. A final report records the policy, detector
JSON and detector ONNX digests alongside final metrics and the frozen policy
outcome.

This is deliberately stricter than a policy file plus its own checksum. A
self-supplied hash proves only which bytes were supplied, not whether the bytes
truthfully describe selection or calibration.

## ONNX acceptance

ONNX parity is checked against validation rows, and export is rejected when its
maximum absolute probability error exceeds `1e-5`. The final evaluator repeats
and enforces this parity check before it writes a report.

Parity is exporter-equivalence evidence only. It is not a hardware benchmark,
wall-clock inference-latency result, deployment-safety result or evidence that
an ONNX runtime has been selected for an Arm device.

## Promotion policy

Generated files under `out/` are candidates, not canonical repository evidence.
Before promoting a generated receipt or artifact, record the source commit,
clean-worktree state, lock-file hash, dependency/exporter versions, command,
all artifact digests and reproduced final report. Keep any historical v2
artifact clearly labelled as historical and non-comparable with protocol-v3
final evidence.

A same-environment byte match proves deterministic regeneration for that
recorded environment only. A mismatch across Python or exporter versions
requires semantic comparison before any claim of equivalence. Do not edit ONNX
IR metadata merely to imitate an older file hash.
