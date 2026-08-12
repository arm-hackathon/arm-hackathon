# Habitat V2 air-network research

Date: 2026-08-12
Status: implementation input
Scope: reduced-order multizone ventilation, fan, duct, damper, telemetry, and electrical receipts

## Research question

What is the smallest physically coherent air-network model that is complex enough to generate useful multiroom control data, remains deterministic and judge-runnable on CPU-only machines, and does not claim CFD or flight validation?

## Sources

1. NIST, **CONTAM**: <https://www.nist.gov/services-resources/software/contam>
2. NISTIR 7251, **CONTAM 2.4 User Guide and Program Documentation**: <https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7251.pdf>
3. US Department of Energy and AMCA, **Improving Fan System Performance: A Sourcebook for Industry**: <https://www.energy.gov/sites/default/files/2014/05/f16/fan_sourcebook.pdf>
4. NASA NTRS 20210020015, **Lunar Surface Habitat Thermal Control System Design**: <https://ntrs.nasa.gov/citations/20210020015>
5. NASA NTRS 20210019410, **Regenerative ECLSS for Exploration**: <https://ntrs.nasa.gov/citations/20210019410>
6. ONNX Runtime, **Model quantization**: <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
7. ONNX Runtime, **ACL Execution Provider**: <https://onnxruntime.ai/docs/execution-providers/community-maintained/ACL-ExecutionProvider.html>

## Findings that constrain implementation

### Multizone modelling

NIST describes CONTAM as a multizone indoor-air-quality and ventilation program that calculates airflow and relative pressures between zones, contaminant transport, filtration, deposition, source generation, and occupant exposure. Its manual gives the volumetric power-law form `Q = C * (delta_p ** n)`. AEOLUS uses the bounded turbulent special case `n = 0.5`, equivalent to `delta_p = R * Q**2`. This is an engineering network model, not CFD.

CONTAM performs its core calculations in mass-flow units. AEOLUS must therefore keep volumetric and mass flow distinct. Trace v3 reports measured flow in m³/s and density-derived flow in kg/s using the scenario's declared air density.

The first AEOLUS air-network contract deliberately supports one closed recirculation loop with parallel room branches. It does not reproduce CONTAM, exterior infiltration, wind pressure, buoyancy-driven door flow, or arbitrary nonlinear networks.

### Fan and system curves

The DOE and AMCA sourcebook treats fan performance as the intersection of fan and system behaviour. It records the fan affinity relationships:

- airflow varies approximately with rotational speed
- pressure varies approximately with rotational speed squared
- fan power is determined from airflow, pressure and efficiency

The AEOLUS fan uses a declared quadratic pressure-flow curve at rated speed and applies the affinity relationships to speed commands. A deterministic bisection solver finds the intersection between the fan curve and the total system resistance.

### Duct and damper pressure loss

For a reduced-order turbulent branch, pressure loss is represented as:

`delta_p_pa = resistance_pa_s2_m6 * q_m3_s * abs(q_m3_s)`

This is the lumped form of geometry and local-loss effects. Every resistance is an explicit scenario assumption. The implementation does not infer a certified duct coefficient from a visual mesh.

A damper changes the branch's effective flow area. The first contract uses a documented notional mapping:

`area_fraction = leak_fraction + (1 - leak_fraction) * actual_position`

`effective_resistance = open_resistance / area_fraction**2`

The mapping is deterministic and monotonic but is not presented as a manufacturer-specific damper curve.

### Fan electrical power

Air power is:

`air_power_w = fan_pressure_rise_pa * total_flow_m3_s`

Electrical power is:

`electrical_power_w = air_power_w / total_efficiency`

The total efficiency parameter includes fan, motor and drive losses for the reference model. Later evidence may replace it with a measured or tabulated curve. The current value remains a labelled notional assumption.

### Habitat geometry and values

NASA sources provide system context and representative habitat concepts but not a sufficiently authoritative room-by-room public floor plan for this project. The AEOLUS interior, room dimensions, device positions and duct routing are therefore explicitly **notional**. They are chosen to make geometry, physics IDs and viewer objects agree. They are not claimed to reproduce a specific NASA vehicle.

Existing Habitat V2 thermodynamic assumptions remain governed by `docs/provenance/habitat-v2-numerical-ledger.md`. New network coefficients are recorded in the scenario and trace lineage rather than hidden in code.

## Reference topology

The submission world uses eight zones:

1. `laboratory`
2. `common_galley`
3. `crew_quarters_a`
4. `crew_quarters_b`
5. `hygiene_medical`
6. `airlock_suitport`
7. `equipment_power_bay`
8. `air_processing_bay`

The loop contains:

- return plenum
- variable-speed supply fan in the air-processing bay
- supply trunk
- filter/scrubber pressure-loss component
- one supply branch and motorised damper per zone
- one supply diffuser per zone
- one return grille and return branch per zone
- return trunk

Each zone branch is a parallel path between common supply and return plenums. Shared trunk and filter losses depend on total flow. This supports a deterministic scalar operating-point solve while still creating real competition between rooms for available pressure and airflow.

## Placement contract

Every physical object carries a stable ID and placement metadata in habitat coordinates measured in metres:

- zones: centre and size
- fan: point and orientation
- dampers: point, orientation and controlled branch ID
- supply diffusers and return grilles: point, orientation and zone ID
- duct runs: ordered polyline points and cross-section metadata
- sensors: point, measured physical quantity and attached zone or component ID

The physics engine does not derive resistance from these coordinates in this slice. Geometry and resistance are both explicit and checked for referential consistency. This avoids pretending that illustrative Three.js geometry is a CFD mesh.

## Model-facing consequences

The richer world creates a useful future action-conditioned forecasting problem because:

- one fan command affects all rooms
- each damper command redistributes a limited pressure and flow budget
- room volumes and loads differ
- shared filter resistance couples every branch
- actuator slew creates delayed effects
- physical faults alter actual behaviour without changing requested commands
- sensor faults alter observed telemetry without altering physical truth

The deterministic rule baseline remains strong. Learned methods must be evaluated on held-out whole scenarios and closed-loop physical outcomes.

## Future optimization implications

No model architecture, quantisation scheme, runtime execution provider or
training procedure is selected by this deterministic-world slice. Those choices
must be made with Ben during the later model-development phase and must be
evaluated against the frozen world, corpus and closed-loop contracts.

Future optimization priorities are:

1. preserve safety and closed-loop outcome gates
2. preserve forecast quality on held-out scenario families
3. reduce p95 and p99 inference latency and jitter
4. reduce sustained CPU and energy proxy at the declared control frequency
5. reduce peak RSS, model size and startup time
6. improve candidate-action batch throughput as a secondary objective

Any future Arm-specific execution-provider experiment must remain optional. The
judge-run path must stay portable and CPU-only, and no Arm or model-performance
claim exists until native measurements are captured from a frozen candidate.

## Explicit non-claims

- not CFD
- not a certified HVAC design
- not a validated spacecraft digital twin
- not flight software
- not a manufacturer-specific fan or damper model
- not evidence that cloud Arm hardware is literally onboard a habitat
- not real-world model validation
