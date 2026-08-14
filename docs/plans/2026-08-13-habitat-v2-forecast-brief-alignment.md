# Habitat V2 forecast design: brief-alignment assessment

**Status:** Alex's historical review artifact, now integrated as design input for the bounded D1 execution PRD

**Original branch:** `alex/habitat-v2-forecast-design-review`

**Review base:** `ec8a6e07cddcd97915398e5a84d348b30d850c86`

**Companion design:** [Habitat V2 forecast data/evaluation design](2026-08-13-habitat-v2-forecast-data-evaluation-design.md)

## Scope and publication state

This assessment compares the completed local design work with the original
Habitat V2 forecast data/evaluation brief. It does not approve the design on
the team's behalf.

- The work was performed from a separate checkout based on the requested
  canonical commit.
- The active HMC branch was not modified.
- No model training, corpus generation, baseline execution, UI work or direct
  control design was performed.
- At the original handoff, no pull request, merge, tag, release or deployment
  was created. Alex subsequently pushed commit
  `f21787771a3abd278625aacef2f2bad757b37523`.
- The exact authored change is preserved in integration history. Current-state
  corrections and implementation are separate follow-up work.

## Alignment summary

| Brief requirement | Status | Assessment |
|---|---|---|
| Use a separate checkout at `ec8a6e0` | Met | The design checkout was detached at the requested base before this local design branch was created. |
| Inspect `ed3fd5c` and current HMC interfaces read-only | Met | Observability, HMC, trace, health, safety, proposal and physics contracts were inspected without changing their branches. |
| One recommended contract | Met | The companion document gives one design, not an architecture menu. |
| Exact model-input order, units and types | Met, with provisional HMC binding | The operational field order, expansion rules, masks, modes, health, alarm encoding and complete action shape are specified. HMC-derived manifests must be rebound to final HMC bytes. |
| Exact cadence, history and horizon | Partially instantiated | Cadence is fixed at 60 seconds. History and horizon are selected by a closed, pre-training timing procedure rather than guessed before evidence. |
| Exact forecast targets and crossings | Met | The 51 physical targets, units, step semantics, missing-data rule and harmful-positive crossings are specified. |
| Corpus identities, canonicalisation and lineage | Met | Closed JSONL records, identity hierarchy, source/contract bundle and trace-to-sample checks are defined. |
| Whole-family splits and leakage controls | Met | A deterministic cluster-level split and explicit cross-boundary rejection tests are specified. |
| Persistence, extrapolation and fitted baseline | Met | All three are specified, including train-only fitting and an action-blinded ridge diagnostic. |
| Unreachable physics oracle | Met | It is explicitly evaluator-only and cannot be selected as a production comparator. |
| Metrics and fail-closed evaluator | Met | Error, crossing, abstention, invalid output, runtime, memory, size and identity behavior are defined. |
| Evidence-derived admission margins | Met | The baseline-qualification procedure derives and freezes margins before learned training. |
| Files inspected, rejected alternatives, tests and dependencies | Met | All are listed in the companion design. |
| No implementation/training/publication | Met | Only local Markdown design artifacts were created. |

## Positives

- It follows the requested advisory, action-conditioned forecasting role and
  does not return to historical fault-name classification.
- It keeps HMC as the sole deterministic authority and treats the proposed
  action as a complete, topology-valid command.
- It uses completed observations only and makes the action/target timing
  causal and explicit.
- It separates operational model inputs from evaluator-only physical truth.
- It accounts for the actual HMC snapshot, health, alarm, proposal, receipt and
  trace interfaces rather than designing against an imagined API.
- It defines exact ordered fields, units, scalar types, status masks and tensor
  projections for the inspected eight-zone topology.
- It adds the replay-witness layer required because HMC control traces commit
  snapshot identities but do not contain all snapshot bodies and intermediate
  physical targets needed by the corpus.
- It splits whole semantic families before windows, keeping repetitions,
  healthy/treatment siblings, actions and anchors together.
- It gives deterministic generation, duplicate-generation, lineage and
  fail-closed artifact-substitution rules.
- It defines strong deterministic comparators and an explicit action-blinded
  fitted-baseline diagnostic.
- It derives later candidate margins from baseline evidence rather than
  inventing an accuracy percentage.
- It clearly distinguishes recommendations from final team-owned bindings.

## Negatives and limitations

- The result is documentation, not executable contracts, parsers, corpus or
  evaluator code.
- History length and forecast horizon are not yet numerical constants. The
  design deliberately makes them outputs of a predeclared timing pilot, so the
  final tensor dimensions cannot be instantiated until that receipt exists.
- At the original design handoff, the HMC was still moving. The reviewed final
  foundation is now `79d6a718e0d44122a763bb72f9c8ed929f39fd23`, tree
  `91cea3b4c2334a4ece140bd1bf7144353f52ec0d`. D1 derives the remaining
  manifests from those exact bytes.
- The HMC emergency safe-action catalogue is not a suitable normal forecast-action
  catalogue. At the original handoff, four complete normal actions and their
  hashes were absent. D1 now freezes the four production-validated operating-mode
  commands in `contracts/habitat_v2_forecast_action_catalogue_v1.json`.
- The recommended scenario/profile roster has not been ratified or persisted
  as a separate canonical profile packet.
- The 287 alarm slots were reproduced against the reviewed final foundation and
  materialised in `contracts/habitat_v2_forecast_alarm_manifest_v1.json`. The
  exact ordered slot list is independently hashed.
- The fixed eight-zone scope is intentionally narrow and does not generalise to
  arbitrary topology shapes.
- The final split can hide exact realisations, seeds, traces and targets, but a
  public finite scenario roster may reveal broad held-out strata by
  elimination. The honest claim is therefore **distribution-transparent,
  realisation-blind**, unless the roster design changes.
- No baseline evidence exists yet, so no comparator, admission margin or
  proceed/stop result has been selected.
- At the original handoff, no independent review or team approval had been
  recorded for the recommendation.

## Verdict

The review/design brief is satisfied as a local proposed contract.

The final HMC bytes are now available. The bounded D1 implementation owns the
normal forecast-action catalogue and executable data/evaluation foundation.
The full canonical corpus still depends on the large scenario/profile packet
and final-set custody arrangement. Those remain explicit later gates, not
claims hidden by this review.
