# V6 stateful evaluation semantics

`aeolus.evaluate_v6` evaluates a policy over complete ordered V6 replay streams. It is separate from the historical V1/V2 vector evaluators because a V6 policy consumes a causal window of observable `TickRecord` values through the V6 observable-context/residual contract.

## Stream contract

Each `V6EvaluationStream` contains:

- one non-empty `family_id` and `room_family_id`;
- exactly one V6 split: `fit`, `calibration`, or `validation`;
- exactly one role: `reference` or `fault`;
- contiguous `TickRecord.tick` values beginning at one;
- for faults only, a supported named fault class and trusted observable-onset tick.

Evaluation rejects duplicate streams and rejects a family lacking either reference or fault role. Fault class/onset remain evaluation metadata; they are not an input to the policy.

## Causal replay and reset

For each stream, the evaluator calls `policy.reset()` and then calls `policy.label_window()` on every complete causal window in tick order. The current V6 baseline is stateless, but reset is mandatory to prevent a future stateful policy carrying persistence across streams.

The allowed policy output set is deliberately closed:

```text
nominal
uncertain
sensor_health_concern
physical_flow_concern
frozen_sensor
blocked_path
gradual_primary_fan_degradation
```

An unknown label invalidates the run rather than becoming an accidental alert category.

## Healthy operational burden

On reference streams, `nominal` and `uncertain` are non-alert states. Every transition from either state into any concern/named-fault state begins one healthy operational-alert episode. Contiguous alert windows are one episode until the policy returns to `nominal` or `uncertain`.

The primary rate is:

```text
1,000 * healthy_alert_episode_count / healthy_eligible_ticks
```

`healthy_eligible_ticks` is the number of complete causal windows replayed on healthy reference streams. The report separately records alerting healthy streams and the fraction of healthy windows marked `uncertain`.

## Fault detection and abstention

For fault streams, every causal window ending at or after the trusted observable onset contributes to post-onset operational metrics. A detection requires the policy to emit the expected **named** fault class. A concern-only decision or `uncertain` does not establish named recall or latency.

The report records:

- named-detection count and fault-stream recall;
- first named-detection latency per expected class;
- post-onset uncertain windows/fraction;
- post-onset unresolved fault streams;
- supervised named-fault confusion, macro F1, and class-specific recall;
- sensor-health and physical-flow binary precision/recall on the same supervised windows.

Onset-spanning windows are excluded from supervised named/specialist metrics while still replayed for policy state and operational latency. Specialist precision/recall are marked **window diagnostics**: overlapping windows are correlated and cannot replace family/stream-level operational evidence.

This prevents abstention from being counted as safe silence on healthy data and as competence after a fault begins.

## Scope boundary

This evaluator establishes replay semantics only. It does not yet make V6 corpus/manifests, choose thresholds, train a model, bind a result to a generated family-manifest digest, or determine an operational winner. Those bindings are enforced by the next V6 corpus/runner layer before any canonical training run is allowed.
