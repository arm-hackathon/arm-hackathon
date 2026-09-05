# Habitat V2 Parameter Provenance, Reference Checks, And Sensitivity

Date: 2026-09-05
Issue: [#71](https://github.com/arm-hackathon/arm-hackathon/issues/71)
Branch: `mesh/physics-provenance`
Status: **DEVELOPMENT EVIDENCE — SUPPORTS SCENARIO FAMILY GENERATOR V2 (ISSUE #72)**

## What Was Built

1. **Machine-readable provenance manifest**
   `contracts/habitat_v2_physics_provenance_v1.json`: 107 parameter records
   promoted from `docs/provenance/habitat-v2-numerical-ledger.md`, each with
   value, unit, valid range, classification (physical constant, public
   requirement, physics-derived, engineering assumption, stress-test range),
   citation where required, source path, generator variability, declared
   uncertainty band, and affected systems and metrics. 46 parameters are
   flagged    generator-variable with explicit uniform bands for the family
   generator.
   Manifest SHA-256:
   `404fabe89c61784167ab1f865fb5130cf6eb186d4d3ec1150728f636bc753c21`,
   validated fail-closed by `src/aeolus/habitat_v2/physics_provenance.py`.
2. **Independent numerical reference checks**
   `tests/habitat_v2/test_physics_reference_oracles.py`: 13 oracle tests over
   the seven declared domains — species/ideal-gas conservation, well-mixed
   exchange, scrubber capture limits, humidity/condensation equilibrium,
   lumped thermal balance, fan/system-curve operating point (closed-form
   single-branch and an independently written 300-iteration bisection for the
   two-branch case), actuator slew with shared-capacity projection, and
   electrical/resource depletion including the infeasibility path. Every
   expected value is re-derived inside the test from hand-entered constants
   (CODATA gas constant, Murphy & Koop 2005 coefficients) with float64
   arithmetic; the production helpers are called only to obtain the values
   under test, never to produce expectations.
3. **Bounded sensitivity analysis**
   `scripts/run_habitat_v2_parameter_sensitivity.py`: one-at-a-time
   perturbation of every generator-variable scenario parameter to its declared
   band edges on the frozen development fixture (race family 0, 96-step
   no-proposal hold replay), with an inline crossing metric computed against
   the declared evaluation bounds rather than a production scorer. Receipt:
   `out/habitat-v2-parameter-sensitivity/sensitivity-receipt.json`
   (write-once), SHA-256
   `94201faa7184401686e910dc01d50ab46643146dc56fa84fea46a9b90b6d7861`.

## Tolerances And Rationale

Production physics computes in float64; oracles recompute independently in
float64. Agreement is asserted at 1e-9 absolute / 1e-12 relative, which covers
summation-order differences only. Float32 telemetry rounding is excluded from
oracle comparisons and remains checked at the trace boundary by the existing
deterministic replay validators. The declared budgets live in the manifest's
`conservation_tolerances` block.

## Sensitivity Results

Baseline (family 0, hold policy): total safety exposure `8.893e-04`, 74
violation steps, eventful. Family 3 (fan-degradation, sensor-b variant) has an
identical hold-policy exposure, confirming that under the no-proposal regime
the fault multiplier does not move the O2-injection-dominated exposure.

Reversal-flagged parameters (≥25% relative exposure change or event-presence
flip):

| Parameter | Direction | Perturbed | Exposure | Relative |
|---|---|---:|---:|---:|
| `initial_relative_humidity` | −1 | 0.36 | 1.140e-03 | **+28.2%** |
| `initial_relative_humidity` | +1 | 0.54 | 6.39e-04 | **−28.2%** |
| `occupied_zone_water_vapor_generation_mol_s` | −1 | 8.4e-04 | 1.116e-03 | **+25.5%** |
| `zone_volume_m3` | +1 | 50 | 1.152e-03 | **+29.5%** |

Near-threshold movers (not flagged): `occupied_zone_water_vapor_generation`
+1 direction (−21.5%), `initial_temperature_k` (−14.8% / +16.1%).

Insensitive at this operating point (0.0% exposure change across both band
edges): air density, base load, battery capacity/energy, branch and shared
resistances, condenser capacity, cooling COP and capacity, damper leak, all
three fan-curve parameters, initial CO2/sorbent/O2-store/pressure, metabolic
CO2 and O2 rates, sensible heat, scrubber capacity and rate, oxygen-injection
capacity, thermal capacity, sink temperature (−2.7%/+2.7% only), and passive
conductance (−2.7% only).

## Interpretation For Issue #72

- Exposure under the hold regime is dominated by O2-fraction and dilution
  dynamics: water inventory (initial humidity, vapour generation) and zone
  volume materially move decision-relevant exposure and MUST be varied across
  generated families.
- Electrical, network, and capacity parameters do not move safety exposure at
  this operating point; they remain decision-relevant for resource metrics
  (battery/sorbent/store deltas recorded in the receipt) and must still be
  varied, but exposure-based strata should not expect separation from them.
- Fault multipliers and operating-condition offsets are already systematically
  varied by the Issue #55 family design (16 condition groups); this sweep
  deliberately does not duplicate that axis.
- The one plausible-assumption reversal risk identified: a generator that
  sampled humidity or volume only at nominal values would systematically
  understate exposure spread by roughly ±28%; the declared bands in the
  manifest prevent that.

## Verification

- `uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_physics_provenance.py tests/habitat_v2/test_physics_reference_oracles.py` — 21 passed
- Full locked suite, ruff, compileall, `uv lock --check`, `git diff --check` — see PR verification log
- Existing deterministic replay and HMC authority tests remain green (no production behaviour changed; this issue adds a manifest, a loader, tests, a script, and documentation only)

## Claim Boundaries

This is development evidence. Nothing here claims that Habitat V2 is a NASA
digital twin, CFD model, flight design, or physically validated habitat; the
classifications in the manifest are the authoritative statement of what each
number is. No model training, blind evaluation, or control-authority change
occurred.
