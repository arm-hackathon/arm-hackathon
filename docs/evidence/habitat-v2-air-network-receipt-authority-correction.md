# Habitat V2 air-network receipt-authority correction

Date: 2026-08-12
Branch: `ben/habitat-v2-receipt-authority-fix`
Rejected candidate: `5df56c0327a1557a99cfca085ce620f042cccb88`
Version impact: `patch` within the still-unreleased `0.5.0` minor candidate
Publication status at correction time: local and unpushed
Integration update: merged through PR #26 into `alex/ai-2` at
`ba025817920158ab59c6f68b4bf61f31aac301c9` on 2026-08-12

> Historical evidence note: the RED/GREEN counts and publication statements
> below describe the correction candidate before its merge. They are retained as
> time-bound evidence, not as the current branch status.

## Review finding

The standalone scenario-v3 accounting validator accepted a fabricated but
internally self-consistent zero-flow network and electrical receipt. It checked
network algebra, density, efficiency and electrical fan-load consistency, but a
scenario alone could not establish the slew-limited actuator state for an
arbitrary step.

## Correction

- Scenario-v3 receipt validation now requires the trusted pre-step plant state.
- Validation reruns the canonical deterministic physical transition from that
  state instead of duplicating fan, damper or operating-point equations.
- Every network receipt field is compared with the canonical recomputation under
  declared numeric tolerances.
- The electrical fan load is separately compared with the recomputed step.
- A missing scenario-v3 air-network receipt now fails closed.
- Scenario-v1 and scenario-v2 receipts retain their existing behavior because
  they do not carry an air-network receipt.

## RED evidence

Before the correction, the focused regression command produced five expected
failures, all `DID NOT RAISE AccountingInvariantError`:

```text
test_network_accounting_rejects_coherent_alternative_operating_point
test_network_accounting_rejects_forged_actuator_receipt[requested_fan_speed_fraction]
test_network_accounting_rejects_forged_actuator_receipt[actual_fan_speed_fraction]
test_network_accounting_rejects_forged_actuator_receipt[requested_damper_position_by_id]
test_network_accounting_rejects_forged_actuator_receipt[actual_damper_position_by_id]
```

A separate omission regression also failed before its correction with
`DID NOT RAISE AccountingInvariantError`.

## Final source verification

```text
focused exploit family: 6 passed in 0.07s
Habitat V2 suite: 130 passed in 0.82s
repository suite: 601 passed in 126.24s
Ruff 0.14.10: All checks passed!
compileall: exit 0
git diff --check: exit 0
uv lock --check: resolved 21 packages
```

The replacement commit, package hashes, installed-boundary smoke result and
repeated reference-trace hash are recorded after candidate freeze in the external
archive receipt. At the time of candidate verification, no push, PR, merge,
tag or release had occurred. The integration update above supersedes that
historical publication state.
