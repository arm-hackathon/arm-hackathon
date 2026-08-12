# Habitat V2 sensor clamp-order correction

Date: 2026-08-12
Branch: `ben/habitat-v2-fault-sensors`
Rejected candidate: `940608b93be0340c4b64c735db1640dd81ead58d`
Corrected base: `6cbb8a400bdc2bd2f0628d7c540435c767bb67c0`
Version impact: `patch` within the still-unpublished `0.6.0` minor candidate
Publication status: local and unpushed

## Independent review finding

The rejected candidate clamped each healthy truth-plus-noise sensor sample and
then applied additive bias followed by a second clamp. The frozen V4 contract
requires raw truth plus deterministic noise, then the active sensor fault, then
one final clamp. The two orders differ when noise crosses a channel boundary and
the bias moves the raw value back toward the valid range.

## RED evidence

The focused two-boundary regression failed on the rejected implementation:

```text
lower boundary: observed 1000.0, required 0.0
upper boundary: observed 999000.0, required 1000000.0
2 failed
```

## Correction

- Healthy sensor projection retains raw truth-plus-noise values.
- Additive bias is applied to that raw value without an intermediate clamp.
- Primary and secondary completed observations are clamped once after all
  active sensor faults have been applied.
- A stuck sensor still selects its previous completed final observation, which
  is unchanged by the idempotent final clamp.

## Focused GREEN evidence

```text
bias lower/upper clamp boundaries, ordinary bias, stuck memory and compound replay:
5 passed in 0.16s
complete fault/sensor and CLI boundary:
34 passed in 0.76s
version contract:
1 passed in 0.03s
full repository suite:
630 passed in 117.72s
Ruff 0.14.10: All checks passed!
compileall, uv lock --check and git diff --check: exit 0
```

The replacement commit identity, full-suite result, package hashes, isolated
installed-run evidence and correction rereview verdict are recorded only after
the replacement bytes are frozen. No push, PR, merge, tag or release occurred.