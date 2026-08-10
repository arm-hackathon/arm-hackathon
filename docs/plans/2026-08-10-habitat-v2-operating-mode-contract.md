# Habitat V2 Phase 2 Slice 1: Versioned Operating-Mode Contract

Date: 2026-08-10
Status: approved for isolated local implementation
Repository: `C:\Users\Nxiss\code\aeolus-habitat-v2-operating-modes`
Branch: `ben/habitat-v2-operating-modes`
Base: `8712ea011eb82ae13457df5fb43e0c7d9f3ea05e`
Publication authority: local implementation and verification only. No push, PR, merge or external package publication.
Version impact: minor, candidate `0.3.0` to local stacked candidate `0.4.0`.

## What are we making?

A second, explicitly versioned Habitat V2 scenario and trace contract that records which operating mode applied during each simulated interval while leaving the accepted `aeolus_habitat_v2_scenario_v1` behaviour and bytes unchanged.

Allowed operating modes:

- `occupied`
- `eva_transition`
- `contingency`
- `dormant`

## What happens?

1. Existing scenario-v1 files remain closed-schema and continue to reject `operating_mode`.
2. New scenario-v2 files require one valid `operating_mode` on every timeline segment.
3. The mode is operational context only. It must not silently choose or alter loads, commands, physics constants or safety thresholds. Those remain explicit scenario fields.
4. Each scenario object carries its actual scenario-schema, trace-schema and equation-contract identities. Lineage and `run_id` are derived from those per-scenario identities rather than assumed module-wide globals.
5. Scenario v2 selects trace v2. Trace-v2 rows add `applied_operating_mode`:
   - `null` on the initial-state row because no interval has yet executed;
   - the exact segment mode on every post-step row for the interval that produced that row.
6. Trace-v2 loading checks the closed row schema and exact deterministic replay, so a forged or mistimed mode fails closed.
7. A checked-in mode-reference scenario exercises all four modes with explicit loads and commands.

## How do we know it worked?

### Preservation gates

- The existing checked-in scenario-v1 loads unchanged.
- Its canonical scenario SHA-256 remains `8bd3586ace18d008417122b127258b90ad255e622fe329eb70a580d38ed7b48d`.
- Its run ID remains `e0ff08d2e00a06bfabf82ddfca43ca67d19f9ade9c778f186a10d31f92c64c75`.
- Its deterministic trace SHA-256 remains `a94b098cf8707cde6383319be913032de53053d033fe6a7d2f0a07efad6260fb`.
- Scenario-v1 does not gain new fields or widened parsing.

### New-contract gates

- Scenario-v2 rejects a missing, unknown, non-string or unsupported mode.
- Changing only a valid mode changes scenario digest and run ID.
- Changing only the mode does not alter physical telemetry, state, actions, loads or accounting receipts once lineage and the mode-context field are excluded.
- Trace-v2 initial row uses `applied_operating_mode: null`.
- Every post-step row uses the mode of the timeline interval that produced it.
- Forging a valid-looking mode in a trace is rejected.
- The new checked-in scenario runs through the CLI, writes a non-empty trace, and validates against its source scenario.
- Same scenario bytes produce byte-identical trace bytes.

### Repository gates

- Each production change follows an observed RED test.
- Focused Habitat V2 suite passes.
- Complete repository suite passes.
- Ruff check and format check pass.
- Compileall passes.
- `git diff --check` passes.
- Wheel and source distribution build.
- Clean external wheel install can run and validate the new reference scenario.
- Original Habitat V2 worktree at `8712ea0` remains unchanged.

## TDD sequence

1. RED then GREEN: scenario-v1 rejects the new field while scenario-v2 requires and validates it.
2. RED then GREEN: scenario objects expose per-scenario contract identities and derive the correct lineage.
3. RED then GREEN: trace-v2 emits interval-aligned `applied_operating_mode` while trace-v1 bytes remain unchanged.
4. RED then GREEN: forged, mistimed or wrongly typed mode values fail loading.
5. RED then GREEN: checked-in four-mode scenario runs and replays byte-identically.
6. Refactor only while focused tests remain green.

## Non-goals

- No scrubber, filter, leak, thermal, sensor or power fault yet.
- No automatic mode presets or hidden mode-dependent physics.
- No emergency-action catalogue.
- No candidate-plan catalogue.
- No model, dataset, training, inference, Arm optimisation or frontend work.
- No edits to accepted V1 deterministic evidence.
- No push, PR, merge or release.

## Review boundary

Freeze one clean local commit and one immutable diff. Then run one bounded independent review focused on:

- v1 byte preservation;
- schema-version isolation;
- interval alignment at mode transitions;
- forged-mode rejection;
- lineage derived from the parsed scenario rather than caller defaults.

Only finding-specific corrections are allowed before targeted and full retesting.
