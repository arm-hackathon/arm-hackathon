# V6 development boundary

**Status:** frozen before V6 implementation.

## Purpose

V6 investigates conditional specialist detection after the V5 development-only negative result. It is not a final evaluation, response-integration, deployment, or hardware-validation protocol.

## Retired evidence

V3, V4, and V5 development/final evidence is historical. It may be used for forensic hypothesis generation only. It must not be used for V6 fit, calibration, candidate selection, threshold selection, or acceptance claims.

## Fixed V6 seed roles

| Role | Seeds |
| --- | --- |
| Fit | 2100–2103 |
| Internal calibration | 2104–2105 |
| Development validation | 2300–2305 |

The roles are disjoint. These identifiers are distinct from retired V3–V5 seed clusters. A future successor protocol must treat this entire V6 allocation as retired.

## Authorization boundary

The V6 runner must reject any request that:

- uses a schema other than `aeolus_sweep_v6`;
- declares a role other than `development`;
- reuses retired seed identifiers;
- changes the predeclared V6 seed allocation;
- writes into a non-empty output directory;
- asks to generate a final suite;
- asks to authorize response-layer integration.

## Observability decision

`occupancy_multiplier` may appear in historical V5 alert-forensic context because it is present in the simulator trace. It is **not** approved as a V6 candidate-model feature unless the team separately verifies that an equivalent operational workload/occupancy signal exists in the intended real system.

V6 candidate features may use only declared observable telemetry. They must exclude simulator truth, including fault labels/timing, connection health/effectiveness, source mass, RNG state, and noise state.
