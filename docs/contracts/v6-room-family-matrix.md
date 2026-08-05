# V6-A room-family matrix

**Evidence role:** V6 development source data. This matrix is not a final suite and does not authorize a response-layer action.

## Fixed topology

Every family preserves the existing four-zone hub:

- non-processing zones: `cabin_a`, `cabin_b`, `lab`;
- processing zone: `processing`;
- one outbound and one return connection per non-processing zone.

All four bases therefore share `observable_context_v1` selector digest:

```text
88819b63f2f52c7c2c8d849436d07151cbeb59e2a9e16165dedd762b15d7c8ad
```

Room physics—not graph identity—is varied.

| Room family | Role | Seeds | Declared purpose |
| --- | --- | --- | --- |
| `room-balanced` | fit | 2100–2101 | Balanced current-style hub baseline |
| `room-volume-asymmetric` | fit | 2110–2111 | Different zone volumes, processing volume, and per-loop capacities |
| `room-capacity-constrained` | calibration | 2120–2121 | Legitimate contention-prone shared-capacity regime |
| `room-transition-heavy` | validation | 2300–2301 | Different response times and rapid staggered nominal load transitions |

Each family has one predeclared operating profile, onset ticks 25 and 70, all three target zones, one gradual severity (`30` ticks to `0.75` effectiveness), and one blocked severity (`0.65` effectiveness). This produces 144 paired fault families: 72 fit, 36 calibration, and 36 validation.

## Pairing rule

For every generated fault scenario, the generator deep-copies its reference and changes only `fault_profiles`. That preserves seed, base room configuration, operating profile, telemetry regime, capacity, and occupancy schedule.

Each manifest row includes:

- room-family ID and development role;
- base-scenario canonical SHA-256;
- reference and fault scenario filenames;
- fault class;
- V6 observable-context receipt.

A room-family ID or seed cluster cannot appear in multiple roles. Missing one of fit, calibration, or validation is rejected. There is no random parameter sampling outside the checked-in sweep source.

## Contract-generation receipt

Generated without model training into:

```text
out/v6-room-family-contract-2026-08-05-a/
```

```text
sweep_spec_sha256:       6036f592e1c2c82ddbbb12584ead617e86342c39dd6200a04d3a99910a4ab04a
family_manifest_sha256:  8d84fc53c3ad9ad3d444c44ef9e2eff6709f43cefa98dfb200e471c9320493f5
generated_scenario_files: 152
```

This receipt demonstrates configuration generation only. V6 selection remains disabled until the residual feature, specialist policy, corpus, and stateful evaluator contracts are implemented and pass their development gates.
