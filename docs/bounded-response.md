# Bounded recovery response

## Status

Development-stage, evidence-backed design on `yarofix2`. The
governor is a deterministic, causal, observable-only decision maker that emits
bounded per-zone actuator commands with structured rationale. It runs in
parallel with — never instead of — the deterministic baseline controller.

## Problem

The frozen v3 evaluation documents detection only. PROBLEM.md identifies a
simulation demonstration of *fault detection and bounded response* as the
remaining gap. The conservation model requires that only the physics engine
changes state and that the controller emits bounded commands with action
bounds and rationale.

## Design constraints

1. **Causality** — the governor decides commands for the next tick from
   completed-tick observations only. It never sees fault effectiveness,
   health, seeds or schedules.
2. **Observable-only input** — it consumes exactly the `model_input_v1`
   `float32[24]` vector per completed tick; the same features the detector
   uses.
3. **Bounds** — every command stays in `0.0..1.0`; per-zone commands move by
   at most `max_command_delta` per tick.
4. **Determinism** — identical scenario plus identical settings yields
   identical commands and rationale.
5. **Auditability** — every threshold is a declared constant in
   `ResponseSettings`; every tick records a per-zone rationale object, never
   free text.

## Policy

1. **Proportional demand** — each zone starts from the same bounded
   proportional command the baseline controller would issue for its latest
   sensor reading.
2. **Frozen-sensor hold** — a zone whose reading is flat across its window is
   held at its last good command instead of chasing a stale reading.
3. **Degraded-loop spare-capacity release** — a loop with a severe, isolated,
   sustained delivery residual is treated as degraded. Its commanded demand is
   released back to shared capacity only while that zone still has spare
   comfort (reading at or below the comfort threshold). A zone that actually
   needs air keeps its full proportional command, so the governor never
   under-drives a hot zone.
4. **Rate and energy bounds** — commands move by at most `max_command_delta`
   per tick and remain in `0.0..1.0`.

Rationale reasons: `nominal`, `frozen_hold`, `degraded_spare_release`,
`bounded_rate`.

## Constants (defaults)

| Constant | Value | Meaning |
|---|---|---|
| `window_ticks` | 10 | causal observation window |
| `max_command_delta` | 0.1 | per-tick command movement bound |
| `degraded_residual_threshold` | 0.4 | severe residual-to-request ratio |
| `degradation_isolation_margin` | 0.2 | required margin over other loops |
| `degradation_persistence_ticks` | 3 | sustained tail before acting |
| `min_requested_fraction` | 0.05 | ratio denominator floor vs loop max |
| `frozen_normalized_range` | 0.02 | flatness detection scale |
| `frozen_persistence_ticks` | 10 | flat window before holding |

## Evidence-driven iteration

The policy was tuned against measured outcome, not intuition. Every policy
below was evaluated over the same `scenarios/sweep-response.json` corpus
(129 families: blocked, gradual degradation and frozen sensors across two
operating profiles):

| Policy | time-above parity | healthy-reference outcome |
|---|---|---|
| v1 boost + priority cut + arbitration | worse on 51/129 (mean +2.5 ticks) | 60/129 overruns |
| v2 unconditional throttle | worse on 64/129 (mean +8.8 ticks) | 60/129 overruns |
| v3 spare-capacity release + parity baseline | 117/129 exact, 128/129 within margin (1 at +2 ticks) | 0 beyond margin |

Measured findings that shaped the final policy:

- Boosting a degraded loop wastes shared capacity the loop cannot turn into
  airflow; the hub then starves healthy loops (cabin_a time-above rose from 8
  to 19 ticks in one blocked family).
- Throttling a degraded loop that actually needs air under-delivers it; only
  a loop with spare comfort can be throttled without loss.
- The baseline controller is same-tick (it reads the post-ventilation
  concentration of the tick it commands). A causal governor is therefore one
  tick behind by construction. The evidence reports this explicitly as a
  causality margin instead of pretending it away.

## Evidence

Reproduce with:

```bash
PYTHONPATH=src uv run python -m aeolus.response_evidence \
  scenarios/sweep-response.json out/response-evidence
```

The receipt (`out/response-evidence/response-evidence.json`) binds the result to
more than the canonical sweep and family-manifest identities:

- `environment` follows the experiment receipt convention and records the
  Python/platform versions, `uv.lock` hash, source commit and the actual
  `source_worktree_dirty` Git-status flag.
- `source.files_sha256` records exact bytes for every repository `src/aeolus`
  module, with a deterministic `source.manifest_sha256` over that map.
- `config` records exact `pyproject.toml`, `uv.lock` and base-scenario bytes,
  plus canonical hashes of the fixed run and response settings.
- `sweep` records the exact sweep-spec bytes, its canonical JSON hash and
  exact hashes for generated scenario files and their deterministic manifest.

Only repository-relative logical names occur in these manifests. Output
locations are not included in receipt values or canonical hashes, so moving
an output directory cannot change the receipt. A dirty worktree is recorded as
`true`; it is never reported as clean merely because the generated metrics are
reproducible. The tracked receipt in `artifacts/response-evidence.json` is the
frozen historical artifact and is regenerated separately when its provenance
is intentionally refreshed.

Metrics per family and controller: time above the `0.30` crew-cabin ceiling,
max excursion, response latency from onset to first non-nominal action on the
affected loop, cumulative actuator energy, and invariant violations (delivered
airflow above shared capacity).

## Scope

This is simulation evidence on declared synthetic operating profiles. It is
not hardware, INT8, wall-clock or deployment evidence. The governor never
changes plant state directly and never overrides the physics invariant.
