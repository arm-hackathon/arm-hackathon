# AEOLUS Habitat V2 Numerical Provenance Ledger

Date: 2026-08-12
Status: implemented deterministic V1-V4 analogue. Public sources support requirements and architecture, not an exact Artemis habitat.

## Classification rules

Every numerical value or equation used by Habitat V2 must be labelled as one of:

1. **Physical constant**: a broadly accepted physical constant with unit and source.
2. **Public requirement/range**: quoted from a public primary source.
3. **Physics-derived**: calculated from declared inputs and equations.
4. **Engineering assumption**: a plausible value selected for the notional analogue, not attributed to a flight system.
5. **Stress-test range**: an intentionally challenging value used to evaluate robustness, not a nominal design claim.

An engineering assumption or stress-test range must never be presented as NASA, ESA, Artemis, Gateway or flight-qualified data.

## Primary sources

### NASA-STD-3001 Volume 2 Revision F

Source:
https://standards.nasa.gov/standard/NASA/NASA-STD-3001_VOL_2

Public PDF:
https://standards.nasa.gov/system/files/tmp/NASA-STD-3001%20Vol%202%20Rev%20F.pdf

Relevant requirements and rationale:

- Section 6.2.1.1 requires at least 30% inert diluent gas when the balance is oxygen.
- Section 6.2.1.2 and Table 6.2-1 provide inspired oxygen partial-pressure exposure ranges. The indefinite normoxia target is 145–155 mmHg PIO2.
- Section 6.2.1.3, requirement V2 6004, limits nominal habitat one-hour average CO2 partial pressure to no more than 3 mmHg.
- Section 6.2.2.1, requirement V2 6006, gives an indefinite human-exposure total-pressure range of 34.5 kPa < pressure ≤ 103 kPa.
- Section 6.2.4.1 requires control of pressure, humidity, temperature, ventilation and oxygen partial pressure.
- Section 6.2.5 requires per-compartment recording and display of pressure, humidity, temperature, oxygen partial pressure and carbon-dioxide partial pressure.
- Section 6.2.6 requires alerting when atmospheric parameters leave safe limits.
- Section 6.2.7 addresses atmospheric mixing and ventilation, including re-establishing temperature and humidity after configuration changes.
- Requirement V2 7041 states that environmental control must accommodate activity-dependent oxygen consumption and additional heat, carbon dioxide and perspiration.

Use in AEOLUS:

- supports tracked variables and safety-reference ranges;
- supports per-zone monitoring and temporal trend analysis;
- supports occupancy/activity as exogenous loads;
- does not provide AEOLUS room dimensions, duct topology, component curves, fault distributions or controller software.

### NASA Regenerative Life Support Systems for Exploration Habitats

Source:
https://ntrs.nasa.gov/citations/20220006727

Public PDF:
https://ntrs.nasa.gov/api/citations/20220006727/downloads/ICES-2022-196.pdf

Relevant public architecture evidence:

- sustained lunar habitats increase the importance of reliability, maintainability, mass and power;
- lower pressure, low gravity, contingency protocols and extended uncrewed/dormant periods affect life-support requirements;
- limited water and oxygen consumables motivate recovery from waste products;
- the notional surface-habitat discussion uses two resident crew with intermittent accommodation for four;
- baseline functional capabilities include water processing, oxygen generation from water, carbon-dioxide removal while retaining humidity for condensate collection and high-pressure oxygen storage;
- power, thermal load, logistics and subsystem integration are coupled design considerations.

Use in AEOLUS:

- supports shared consumable inventories, power coupling, operating modes and processing functions;
- supports a two-occupied-zone analogue as a bounded abstraction;
- does not supply final flight equipment capacities or exact distributions.

### NASA Conceptual Thermal Control System Design for a Lunar Surface Habitat

Source:
https://ntrs.nasa.gov/citations/20210020015

Relevant public architecture evidence:

- conceptual surface habitat supports 30–60 day habitability for up to four crew;
- thermal-control design uses partitioned internal service loops and an external loop;
- infrequent eclipses may last up to about 100 hours in the cited concept;
- dormant and operational thermal/power behaviour differ;
- radiator geometry, energy storage, temperature excursions and electrical-power-system growth are coupled concerns.

Use in AEOLUS:

- supports a separate utility thermal sink, power-constrained cooling and future operating-mode scenarios;
- does not justify an exact radiator, coolant loop or lunar external-temperature model in the first slice.

### NASA Lunar Life Support Overview

Source:
https://ntrs.nasa.gov/citations/20210019410

Relevant public architecture themes include atmosphere revitalisation, water processing, waste management and particle measurement.

### ESA MELiSSA Closed Loop Concept

Source:
https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Melissa/Closed_Loop_Concept

Relevant public architecture evidence:

- closed-loop life support links recovery of food, water and oxygen from organic waste, carbon dioxide and minerals;
- waste processing, air revitalisation and water purification are coupled cycles.

Use in AEOLUS:

- supports treating consumables and waste streams as explicit inventories;
- does not justify implementing a biological closed loop in the initial plant.

## Physics-derived rules planned for V2

### Ideal gas relation

`n = pV / (RT)`

Used to derive initial gas moles and post-step pressure from declared volume, temperature and species inventories.

Classification: physics-derived.

### Partial pressure and concentration

`p_species = mole_fraction_species * p_total`

`co2_ppm = 1e6 * n_co2 / n_total`

Classification: physics-derived.

### Well-mixed exchange

`exchange_fraction = 1 - exp(-q * dt / V)`

Used as a bounded first-order well-mixed-zone approximation.

Classification: grey-box engineering equation. It is not CFD and must be described as a lumped approximation.

### Lumped thermal balance

`delta_T = net_heat_J / thermal_capacity_J_per_K`

Recirculation heat exchange uses a declared air density and specific heat. Cross-zone exchange must sum to zero before external sources and sinks.

Classification: physics-derived balance with engineering-assumption coefficients.

### Relative humidity and saturation

Water-vapour partial pressure is calculated from moles and temperature. Saturation vapour pressure uses one declared approximation over the supported temperature range. Values outside that range fail validation. Supersaturation condenses into the explicit water inventory.

Classification: physics-derived using a documented approximation.

## Initial reference-scenario values

These values remain provisional until the exact scenario is implemented and tested. They are intentionally not attributed to a flight design.

- total pressure around 72 kPa: **engineering assumption**, selected inside NASA’s indefinite exposure range;
- oxygen dry-gas fraction around 0.30: **engineering assumption**, selected so inspired oxygen partial pressure is near the NASA indefinite normoxia target while retaining well above 30% diluent;
- initial CO2 around 800 ppm: **engineering assumption**, below the public 3 mmHg one-hour average ceiling;
- initial relative humidity around 45%: **engineering assumption**;
- initial temperature around 295 K: **engineering assumption**;
- occupied-zone volumes: **engineering assumptions**;
- crew metabolic CO2, O2, water and heat rates: **engineering assumptions until separately sourced**;
- fan, scrubber, condenser, cooling, oxygen-store and battery capacities: **engineering assumptions**;
- actuator time constants and sensor imperfections: **engineering assumptions or stress-test ranges**;
- fault timing, severity and combinations: **stress-test ranges**.

The scenario-v4 checked-in compound fault values are stress-test inputs for
software and evaluation behavior. Fan effectiveness multipliers, branch
resistance multipliers, jam timing, sensor-noise amplitudes, sensor-bias values
and stuck-sensor intervals are not measured component reliability data and are
not attributed to a NASA, ESA, Artemis, Gateway or flight design.

## Explicitly unsupported claims

Habitat V2 must not be described as:

- a NASA, Artemis or Gateway digital twin;
- flight-qualified or validated against operational lunar-habitat telemetry;
- a CFD model;
- a complete ECLSS simulation;
- evidence that a learned model is superior;
- Arm-optimised until exact target measurements exist.

The accurate description is:

> a deterministic, source-grounded, notional lunar-habitat environmental analogue with explicit engineering assumptions and replayable multivariable dynamics.
