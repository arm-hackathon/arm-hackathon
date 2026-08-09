# Temporal early-risk predictor development contract

Status: frozen for the first development experiment

Version impact for a future code PR: **minor**

## Research question

Can a compact learned predictor identify a uniquely targeted, physically harmful CO2 trajectory early enough for the existing deterministic recovery governor to reduce physical CO2 exposure without introducing healthy or wrong-zone interventions?

This experiment does not give a learned model actuator authority. The deterministic recovery supervisor remains the only reserve-command owner.

## Scope

The first slice covers gradual primary-fan degradation and difficult non-actionable lookalikes:

- persistent gradual degradation that becomes physically harmful;
- persistent gradual degradation that remains physically safe;
- transient gradual degradation that clears without becoming harmful;
- paired healthy references under varied demand, capacity, noise, bias and drift;
- frozen-sensor and abrupt-fault families as safety confounders during governed evaluation, not positive early-risk training targets.

Arm export, quantisation and benchmark optimisation are outside this experiment.

## Data boundary

### Model inputs

One inference uses the last 10 completed `model_input_v1` telemetry vectors. The predictor may see only values available to a deployed system, including measured CO2, actuator state, requested airflow, delivered airflow and airflow residuals.

It must not receive:

- fault type;
- fault target;
- injected fault start or end tick;
- effectiveness or health configuration;
- random seed;
- future telemetry;
- physical simulator state that is absent from `model_input_v1`;
- final-suite identity or membership.

### Training labels

A row from a persistent gradual-degradation fault is positive for exactly one crew cabin when the reserve-off physical counterfactual for that family will cross the declared crew-cabin CO2 ceiling within the next 12 completed ticks while the current tick remains below the ceiling. The label is `risk:cabin_a` or `risk:cabin_b`.

Lab faults remain covered by the deterministic governor but are not positive targets for this crew-cabin early-risk experiment. Healthy references, persistent gradual faults that remain physically safe, transient faults, abrupt faults and frozen-sensor faults are negative lookalikes labelled `no_early_risk` while they remain scorable.

All other scorable rows use `no_early_risk`. Rows are excluded when:

- fewer than 10 completed ticks are available;
- more than one zone becomes unsafe in the forecast horizon;
- the current tick is already physically unsafe;
- the row belongs to a final split;
- the trace or contract is malformed.

Labels may use future physical simulator truth during corpus construction. That truth is never serialized in model-facing features.

## Corpus isolation

The experiment must generate a fresh development sweep with train and validation splits only. Scenario identities, random seeds and parameter combinations must be disjoint by split and disjoint from every opened recovery final family by canonical scenario identity.

The opened seven zero-improvement final families may motivate this research question, but their scenarios, traces and parameter identities are prohibited from training, calibration and development scoring.

A separate future blind suite will be generated only after the model, abstention thresholds and governor acceptance policy are frozen.

## Predictor output

The compact predictor returns:

- a target zone or no target;
- risk probability;
- top-two probability margin;
- an explicit abstention decision;
- the completed observation tick and model-input contract hashes.

It does not return actuator commands, reserve strength or handback instructions.

Model calibration may allow bounded advisory warning windows because a warning cannot command recovery. Development admission requires at most 5% healthy-reference warning windows, at most 5% negative-fault lookalike warning windows and at least 50% recall for each crew-cabin target. Closed-loop intervention gates remain stricter: zero healthy and zero wrong-zone interventions.

## Deterministic acceptance gate

The existing recovery settings and default no-advisory behaviour remain unchanged.

An advisory can influence entry only when all of the following are true:

1. the artifact and telemetry contract hashes match;
2. the advisory is fresh for the completed observation tick;
3. the predictor does not abstain;
4. probability and margin meet thresholds frozen using validation only;
5. the advised zone exists in the current topology;
6. the same zone is the unique current airflow-shortfall leader;
7. its measured residual ratio is at least the frozen advisory evidence floor;
8. its lead over every other zone meets the existing 5 percentage-point isolation margin;
9. the same accepted target persists for the existing two-tick entry persistence;
10. reserve hardware has not latched failed.

Any malformed, stale, ambiguous, low-confidence or physically unsupported advisory is refused with reserve commands at zero. The deterministic governor still selects the bounded command from current measured shortfall, owns slew limiting, monitors reserve delivery and controls handback.

While protection was entered from advisory evidence, a continuing accepted advisory for the latched target prevents immediate clear/handback. The normal physical clear, dwell, reserve-failure and handback gates remain mandatory.

## Development selection order

Validation calibration uses the following lexicographic order:

1. zero healthy-reference interventions;
2. zero wrong-zone interventions;
3. zero frozen-sensor interventions;
4. no increase in missed harmful families;
5. no repeated protection episodes, handback recurrence, timeout or invariant violation;
6. greater median warning lead time;
7. lower integrated physical CO2 excess;
8. lower recovery time;
9. smaller serialized model.

A warning metric alone is not evidence of value.

## Development acceptance and stop rules

The predictor-plus-governor candidate is development-eligible only if, against governor-alone on validation families, it:

- preserves all safety gates;
- reduces median integrated physical CO2 excess by at least 5% among harmful gradual-degradation families;
- improves at least one harmful family and worsens none by more than the declared numerical tolerance;
- produces zero healthy-reference and wrong-zone interventions;
- remains byte-reproducible from the same source, settings and manifest.

If no calibrated candidate satisfies those gates, record a negative result and stop. Do not weaken the gates, train on final families or proceed to Arm optimisation.
