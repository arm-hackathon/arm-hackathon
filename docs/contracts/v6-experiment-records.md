# V6 experiment records, decisions, and handoffs

V6 is an evidence-led model-development protocol. A training command is not an experiment record; a metric table without source/data/policy identity is not evidence.

## Objective and comparator

The goal is not to make a learned candidate look better than an arbitrary historical number. A candidate may be retained only when it:

1. beats the declared **V6 conditional-rule baseline** on the frozen comparison policy; and
2. satisfies the unchanged V5 operational gates, including the ceiling of **10 healthy false-alert episodes per 1,000 healthy ticks**, permitted baseline delta, and fault-recall constraints.

The V6 rule baseline is a development comparator, not a production-qualified fallback. If every candidate fails, the report must say that the observables/policy remain insufficient for safe operational use.

## Run classes

| Class | Permitted data | Permitted decisions |
| --- | --- | --- |
| `design` | no generated candidate result required | state a falsifiable hypothesis and predeclare change |
| `fit` | fit room families only | fit model parameters and training-only normalisation |
| `calibration` | calibration room family only | select predeclared thresholds/persistence from the complete feasibility grid |
| `validation` | withheld validation room family, once per frozen candidate | evaluate only; no model, threshold, feature, or policy selection |
| `reproduction` | same frozen source and protocol in a new output directory | check deterministic/semantic repeatability; no selection |
| `forensic` | retired evidence only | diagnose historical failure; never select a candidate |

Inspecting a validation result consumes it for that candidate lineage. Any change to model architecture, feature projection, learning objective, fit data, calibration grid, threshold, or policy semantics creates a revised lineage. That lineage must return to fit/calibration-only work and use a newly predeclared untouched validation family before it can claim improvement.

## Required record for every executable run

Each `out/v6-.../` run must write a machine-readable `run-record.json` and a human-readable `run-record.md` with these fields:

```text
run_id, parent_run_id, run_class, status
started_at_utc, finished_at_utc, exact_command, exit_status
source_commit, worktree_dirty, lockfile_sha256, Python/package versions
V6 sweep digest, generated-family manifest digest, corpus digest
observable-context version/selector/topology hashes
room-family allocation and seed/run clusters
candidate ID, artifact hashes, immutable parameter record
baseline ID and immutable parameter record
change summary, falsifiable hypothesis, expected failure mode
complete calibration feasibility grid and its selection order
metrics by role, room family, seed/run cluster, profile, class, stream, and episode
validation-consumed flag and validation-family identities
acceptance-gate values, pass/fail reason, retained method, authorization flags
output paths and SHA-256 receipts
```

A missing, malformed, or inconsistent record invalidates the run for model-selection claims.

## Diagnostic fields: underperformance and overperformance

Every completed fit/calibration/validation record must answer these separately.

### Underperformance

- Which operational gate failed, by how much, and on what aggregation unit?
- Is the failure low fault recall, too many healthy episodes, excessive nominal uncertainty, or all of them?
- Which class, profile, room family, and seed/run clusters dominate the error?
- Is the candidate worse than V6 rules because of false concerns, missed concerns, or bad concern-to-class escalation?
- Does the residual table support one declared causal hypothesis, contradict it, or leave it unresolved?

### Overperformance / suspicious performance

- Is fit performance materially better than calibration performance?
- Is calibration performance materially better than frozen validation performance?
- Does success disappear by room family, profile, or seed/run cluster?
- Are results driven by correlated stride-one windows rather than independent family/stream evidence?
- Did any candidate or threshold consume validation evidence during selection?
- Do artifact, source, family, corpus, policy, and model hashes exactly bind the result?

A high metric is not accepted until these questions are answered. In particular, a candidate that improves overlapping-window F1 while increasing healthy episode burden is an operational regression.

## Handoff record

Every run record ends with this bounded handoff:

```md
## Handoff

State
- Current source commit and branch state:
- Run class/status:
- Candidate/policy frozen for this run:
- Evidence paths and receipt hashes:

Decision
- What changed:
- Why this was the next permitted change:
- What the result supports or falsifies:
- Retained method and authorization state:

Next
- Exact next authorised action:
- Data allowed for that action:
- Inputs that are now consumed/retired:
- Known risks and unverified assumptions:
```

No handoff may say “tune until better” without naming the permitted split, predeclared change, and acceptance comparison.

## Human-readable run table

`docs/evidence/v6-run-ledger.md` indexes every run with:

| Run ID | Parent | Class | Candidate/change | Data role | Result | Decision | Receipt |
| --- | --- | --- | --- | --- | --- | --- | --- |

The ledger records failures as first-class results. A negative run is not deleted, renamed, or overwritten.

## Current state

No V6 model-training run has occurred. Existing V6 commits establish the source protocol, observable context, room-family sweep, residual features, and conditional concern baseline only. The next permitted executable evidence is the stateful evaluator and V6 manifest/corpus implementation; only then can a fit/calibration run be recorded.
