# V6 residual feature contract

`ResidualFeatureProjector` produces transparent causal evidence from a window of validated `TickRecord` values under `observable_context_v1`. It is for V6 diagnostic specialists, not a replacement generic feature vector.

## Prohibited inputs

The projector never reads or accepts:

- fault type, fault onset, effectiveness, connection health, labels, seed, or scenario source multiplier;
- `co2_mass`, `source_co2_mass`, or `occupancy_multiplier`;
- any value outside the V6 observable-context projection.

Every tick is first checked against the topology-bound observable-context contract. A malformed trace, unknown graph identity, non-finite value, unordered tick sequence, or window shorter than two ticks is rejected.

## Sensor-health evidence per zone

| Feature | Physical reading |
| --- | --- |
| Sensor slope, range, maximum delta | What the measured CO₂ reading did over the causal window |
| Setpoint/actual span, tracking mean, moving fraction | Whether the local controller visibly reconfigured or failed to track |
| Outbound request/delivery/residual change | Whether the loop attached to that zone changed its observable flow state |
| Maximum sibling actuator span | Independent system reconfiguration that can corroborate a response opportunity |
| `expected_change_proxy` | `max(local actual-position span, capacity-normalised delivery change, sibling actual-position span)` |

The expected-change proxy is dimensionless and observable-only. It means **a response opportunity was visible**, not “the simulator says CO₂ should have changed by X.” A flat sensor in a settled system therefore has zero proxy; a flat sensor while independent actuator/flow changes occur can have a high proxy.

## Physical-flow evidence per outbound loop

| Feature | Physical reading |
| --- | --- |
| Request, delivery, residual | Current observable flow shortfall |
| Normalized residual | `residual / request`; defined as `0.0` when request is zero |
| Residual slope, maximum jump, persistence | Whether a flow shortfall is growing, abrupt, or sustained |
| Isolation ratio | Local normalized residual relative to other outbound loops; `0.0` when the local residual is zero, `1.0` when it is positive and all peers are zero |
| Capacity headroom/contention | Current spare shared flow capacity and `1 - capacity_scale` |
| Transient/settled proxy | `max(capacity_contention, local actuator moving fraction)` and its complement |

This projection intentionally separates a local delivery deficit from legitimate shared-capacity contention. It does not diagnose a fault. The subsequent specialist policy must require corroboration and persistence before issuing an operational concern.

## Executable confounds

`tests/test_residual_features.py` proves:

1. exact ten-tick slope/range/residual calculations;
2. zero request has no divide-by-zero or synthetic normalized residual;
3. settled flat sensor does not gain expected-change evidence;
4. flat sensor during observable actuator reconfiguration retains high expected-change evidence;
5. shared capacity contention is visible even with zero local residual;
6. changing hidden mass/source/occupancy truth cannot affect either feature record.
