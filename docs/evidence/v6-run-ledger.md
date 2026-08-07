# V6 run ledger

This ledger is append-only. Each row points to a generated `run-record.json`, a human-readable run record, and the relevant immutable receipts. Failed and invalidated runs remain listed.

| Run ID | Parent | Class | Candidate/change | Data role | Result | Decision | Receipt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v6-conditional-specialists-canonical-2026-08-07-a` | — | development | `observable_context_centroid_v1` vs `conditional_rule_v6` | fit 72 / calibration 36 / validation 36 families | Gate FAILED: candidate macro-F1 0.0 on validation (100% abstention); baseline macro-F1 0.0 by design (concerns only) | Retain `conditional_rule_v6`; v7 escalation required | `out/v6-conditional-specialists-canonical-2026-08-07-a/v6-development-report.json` |

See [V6 experiment records](../contracts/v6-experiment-records.md) for the required record schema, split-consumption rules, diagnosis checklist, and handoff format.
