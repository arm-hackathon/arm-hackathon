# V6 conditional specialist policy

The first V6 policy is a conservative conditional-rule baseline. It does not alter historical `RuleBaseline` behaviour and does not train or export a learned model.

## Inputs and boundary

Both specialists consume only `ResidualFeatureProjector` output. That projector validates each tick against `observable_context_v1`; the specialists therefore do not receive labels, fault type/onset, connection effectiveness, simulator seed, occupancy, or hidden plant mass.

## Sensor-health specialist

A sensor-health concern requires both:

1. measured maximum sensor delta at or below `sensor_max_delta`; and
2. `expected_change_proxy` at or above its threshold.

The proxy is independently observable local/sibling actuator movement or capacity-normalised delivery change. A flat sensor in a settled system is not enough. The assessment returns a score, the corroborating zone, and one of:

```text
flat_sensor_with_corroboration
no_expected_change
sensor_not_flat
```

## Physical-flow specialist

A physical-flow concern requires all of:

1. normalized residual at or above the threshold;
2. fully isolated residual according to the residual-contract convention;
3. residual persistence at or above the threshold;
4. transient proxy at or below the threshold.

Capacity-scale contention and actuator movement contribute to the transient proxy. High residual during shared-capacity/transient operation therefore produces `shared_capacity_transient`, not a local physical concern.

## Decision policy

| Sensor assessment | Physical assessment | Output |
| --- | --- | --- |
| no concern | no concern | `nominal` |
| concern | no concern | `sensor_health_concern` |
| no concern | concern | `physical_flow_concern` |
| concern | concern | `uncertain` |

`uncertain` is deliberately non-operational and does not become a named class. It has no ground-truth input path.

The policy exposes `reset()` for replay stream boundaries. The current rules are stateless, but the reset contract is already tested so later learned/stateful specialists cannot silently carry evidence across streams.

## Deliberate non-claim

This first policy does not emit `frozen_sensor`, `blocked_path`, or `gradual_primary_fan_degradation`. It has only earned the right to report observable concerns. Named-fault escalation requires V6 corpus identity, calibration-only threshold selection, and a stateful evaluator that accounts for `uncertain` as unresolved after fault onset.
