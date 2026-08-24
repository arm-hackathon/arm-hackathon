# Issue #55 Controller Race - v2 Measurements

- Study: three-way controller race (`rules_only` vs `model_advised` vs `oracle_instrument`)
- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/55
- Preregistration: `contracts/habitat_v2_forecast_issue_55_preregistration_v2.json`
- Preregistration SHA-256 (LF-normalized): `9041108536E64561ADCEAA434344CDCB6FEAB967F1BD9FB0F47C03FA713FB22E`
- Protocol base commit: `25a1486157f0bf556c5097b9ab392d8e1e184e02`
- Branch: `research/issue-55-controller-race`
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY` - deterministic simulator development evidence only

The v2 protocol and run supersede the historical v1 Issue #55 study. v1 used
an 8-step greedy oracle and sensor-seed-only families. The v2 result below uses
the declared full-remaining-episode finite oracle schedule and the fixed
operating-condition/plant-condition/sensor-condition roster. The v1 output is
historical provenance, not the final Issue #55 evidence.

## Run Identity

- Command: `uv run --locked --python 3.11 --extra dev python scripts/run_issue55_controller_race.py --output out/issue55-race-v2-1`
- Corpus: `issue55_race_v2`
- Suite: 32 families in a fixed `4 operating x 4 physical plant x 2 sensor` matrix
- Episodes: 96 (32 families x 3 arms), 96 steps per episode
- Decisions: 18 per episode at steps `16, 20, ..., 84` (cadence 4, model horizon 8)
- Teacher: frozen `action-aware-mlp-v1.npz` (SHA-256 `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd`)
- HMC: the sole proposal-arbitration, command, plant-step, and replay authority in every arm
- Output digests:
  - `results.json` SHA-256: `d99f6f748433b1fbb47cffa5e4c7f103b517c2900dc2580bb72d3e0913ab59b3`
  - `episodes.jsonl` SHA-256: `4165a9be11c5755df83fa0ee8160d2eb318c615374b75b5418a5485b8cee7265`
- Generated outputs remain in the ignored local directory `out/issue55-race-v2-1/` and are not committed.

## Validation

- 96 records were present: one record for each of 32 families and 3 arms.
- All 32 family IDs were unique and each family had all three arms.
- Every record committed and replayed 96 HMC plant steps.
- Results and episode digests were recomputed independently and matched the recorded values.
- Hard gates were all zero: authority violations, replay failures, provenance violations, non-finite metrics, and proposal admission failures.

## Arm Summaries

Means are equal-weight means over the 32 family records. Violation-step totals
are sums over those records. `hmc_rejection_count` is the preregistered count
of decision steps where the final HMC command SHA-256 differed from the
proposed command SHA-256; it records HMC modification/rejection, not a
proposal-admission failure.

| Arm | Safety violation steps (total) | Safety exposure (mean) | Comfort deviation (mean) | Resource composite (mean) | Proposals | Admitted | Abstentions | HMC final-command differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rules_only` | 584 | 0.000217556954 | 111.122949941 | 0.007738955319 | 0 | 0 | 0 | 0 |
| `model_advised` | 425 | 15.678536800 | 98.258700076 | 0.010069491342 | 8 | 8 | 568 | 0 |
| `oracle_instrument` | 128 | 0.000047684 | 88.533414776 | 0.008458953351 | 576 | 576 | 0 | 432 |

The oracle submitted all 576 proposals, but HMC changed the final command for
432 of them; only 144 oracle proposals were applied unchanged. The model
submitted 8 proposals, all of which were admitted and applied unchanged. The
zero proposal-admission-failure gate therefore does not mean that every
proposal became the final command.

## Gap Closure

The preregistered point estimate is
`(mean_rules - mean_model) / (mean_rules - mean_oracle)` over equal-weight
family means. Intervals are 10,000 family bootstrap resamples with seed
`550055`.

| Metric | Point estimate | Bootstrap 95% CI | Reading |
| --- | ---: | --- | --- |
| `comfort_deviation` | 0.569478 | [0.236158, 0.893338] | The model closes 56.9% of the declared rules-to-oracle comfort gap. |
| `resource_composite` | 3.236864 | [1.483982, 5.011287] | This is not beneficial closure: lower resource use is better, and the model mean is above both rules and oracle. |
| `safety_exposure` | -92294.227440 | [-166273.118013, -23027.234192] | The model worsens safety exposure dramatically; the negative value is the honest result of the declared closure formula. |

The safety result is the central negative finding. The model improves mean
comfort relative to rules (`111.122949941` to `98.258700076`) but has mean
safety exposure `15.678536800`, versus `0.000217557` for rules and `0.000047684`
for the oracle arm. Safety violation-step count alone is not a sufficient
severity measure here: the model has fewer violating steps than rules but much
larger normalized exposure on those steps.

The resource result also requires directional care. The model mean
`0.010069491342` is above the oracle mean `0.008458953351` and the rules mean
`0.007738955319`; the declared algebra returns `3.236864`, but that number is
not a successful resource-gap closure.

## Decision Accounting

- `rules_only` issued no proposals.
- `model_advised` proposed at 8 of 576 advisory decision opportunities and abstained at 568 (98.6%). All 8 proposals were admitted by HMC, and none was modified at arbitration.
- `oracle_instrument` proposed at all 576 decision opportunities. HMC modified 432 final commands (75.0%) while admitting every proposal canonically.
- No arm produced a proposal outside its preregistered decision steps.

The model's abstentions are a measured consequence of its hard predicted-bound
eligibility gate and point-prediction advisory contract. They are not evidence
that the model is safe outside this fixture. The oracle's 432 final-command
differences are evidence that the oracle is advisory-only even when it has
access to true plant state and future schedule information.

## Family Roster

Physical fault schedules, operating-condition numeric values, resource factors
and family-ID construction are frozen in the v2 preregistration. The physical
faults run on emitted steps 32 through 79.

| # | Family ID | Operating condition | Mode | Plant condition | Sensor condition | Fault |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `issue55f458c17c345edc522` | nominal_occupied | occupied | nominal_plant | sensor_seed_a | none |
| 1 | `issue55fdcf992246a11c7a4` | nominal_occupied | occupied | nominal_plant | sensor_seed_b | none |
| 2 | `issue55f6522ff641bafe164` | nominal_occupied | occupied | fan_degradation | sensor_seed_a | fan_speed_degradation |
| 3 | `issue55f60c41aa30de7271f` | nominal_occupied | occupied | fan_degradation | sensor_seed_b | fan_speed_degradation |
| 4 | `issue55f53b1e256dd7611de` | nominal_occupied | occupied | laboratory_resistance | sensor_seed_a | branch_resistance_increase |
| 5 | `issue55f31c163f8f4ab0939` | nominal_occupied | occupied | laboratory_resistance | sensor_seed_b | branch_resistance_increase |
| 6 | `issue55f44ee4d86f5843e9d` | nominal_occupied | occupied | equipment_cooling_loss | sensor_seed_a | cooling_delivery_degradation |
| 7 | `issue55fd344b804108fbba4` | nominal_occupied | occupied | equipment_cooling_loss | sensor_seed_b | cooling_delivery_degradation |
| 8 | `issue55f5a25d9a8c8dc7f15` | high_load_occupied | occupied | nominal_plant | sensor_seed_a | none |
| 9 | `issue55f735ec1cf9a0a11c0` | high_load_occupied | occupied | nominal_plant | sensor_seed_b | none |
| 10 | `issue55f71375dcd30cb63be` | high_load_occupied | occupied | fan_degradation | sensor_seed_a | fan_speed_degradation |
| 11 | `issue55fb2059e04239daec5` | high_load_occupied | occupied | fan_degradation | sensor_seed_b | fan_speed_degradation |
| 12 | `issue55ff3655b7aa04d2c55` | high_load_occupied | occupied | laboratory_resistance | sensor_seed_a | branch_resistance_increase |
| 13 | `issue55f0861572fb0b8cff7` | high_load_occupied | occupied | laboratory_resistance | sensor_seed_b | branch_resistance_increase |
| 14 | `issue55fc358e71bbbb68dce` | high_load_occupied | occupied | equipment_cooling_loss | sensor_seed_a | cooling_delivery_degradation |
| 15 | `issue55f3f7179adb03793b9` | high_load_occupied | occupied | equipment_cooling_loss | sensor_seed_b | cooling_delivery_degradation |
| 16 | `issue55fc58c7ad8528df366` | eva_transition | eva_transition | nominal_plant | sensor_seed_a | none |
| 17 | `issue55f884d4648a316246b` | eva_transition | eva_transition | nominal_plant | sensor_seed_b | none |
| 18 | `issue55fe269e68a2ae18773` | eva_transition | eva_transition | fan_degradation | sensor_seed_a | fan_speed_degradation |
| 19 | `issue55ff9328b223503a254` | eva_transition | eva_transition | fan_degradation | sensor_seed_b | fan_speed_degradation |
| 20 | `issue55f2c66befdfdcab19e` | eva_transition | eva_transition | laboratory_resistance | sensor_seed_a | branch_resistance_increase |
| 21 | `issue55f6f7ed37ef54dd779` | eva_transition | eva_transition | laboratory_resistance | sensor_seed_b | branch_resistance_increase |
| 22 | `issue55ffb8872374310adfd` | eva_transition | eva_transition | equipment_cooling_loss | sensor_seed_a | cooling_delivery_degradation |
| 23 | `issue55f367a60c2b1dbbd5a` | eva_transition | eva_transition | equipment_cooling_loss | sensor_seed_b | cooling_delivery_degradation |
| 24 | `issue55feee7f0c73a29f6ac` | contingency | contingency | nominal_plant | sensor_seed_a | none |
| 25 | `issue55f2666f27a0083f15f` | contingency | contingency | nominal_plant | sensor_seed_b | none |
| 26 | `issue55f6ed0ce47696bbf56` | contingency | contingency | fan_degradation | sensor_seed_a | fan_speed_degradation |
| 27 | `issue55fadd6138c8e0894ab` | contingency | contingency | fan_degradation | sensor_seed_b | fan_speed_degradation |
| 28 | `issue55f1e264c07f4be8aef` | contingency | contingency | laboratory_resistance | sensor_seed_a | branch_resistance_increase |
| 29 | `issue55f67bac2dbd1099349` | contingency | contingency | laboratory_resistance | sensor_seed_b | branch_resistance_increase |
| 30 | `issue55fc9d1b10c8988a591` | contingency | contingency | equipment_cooling_loss | sensor_seed_a | cooling_delivery_degradation |
| 31 | `issue55f4db0b2116395a76d` | contingency | contingency | equipment_cooling_loss | sensor_seed_b | cooling_delivery_degradation |

## Per-Family Results

Each row is one recorded family/arm episode. `HMC diff` is the recorded
`hmc_rejection_count`; `prop` and `abst` are proposal and abstention counts.

| # | Arm | Safety violation steps | Safety exposure | Comfort deviation | Resource composite | Prop | Abst | HMC diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `rules_only` | 73 | 0.000870 | 164.435184 | 0.000000 | 0 | 0 | 0 |
| 0 | `model_advised` | 69 | 78.237185 | 131.204642 | 0.014328 | 1 | 17 | 0 |
| 0 | `oracle_instrument` | 16 | 0.000191 | 72.234872 | 0.002880 | 18 | 0 | 0 |
| 1 | `rules_only` | 73 | 0.000870 | 164.435184 | 0.000000 | 0 | 0 | 0 |
| 1 | `model_advised` | 32 | 0.000381 | 70.735568 | 0.002304 | 1 | 17 | 0 |
| 1 | `oracle_instrument` | 16 | 0.000191 | 72.234872 | 0.002880 | 18 | 0 | 0 |
| 2 | `rules_only` | 73 | 0.000870 | 164.270153 | 0.000000 | 0 | 0 | 0 |
| 2 | `model_advised` | 24 | 0.000286 | 71.811278 | 0.002592 | 1 | 17 | 0 |
| 2 | `oracle_instrument` | 16 | 0.000191 | 74.204932 | 0.002880 | 18 | 0 | 0 |
| 3 | `rules_only` | 73 | 0.000870 | 164.270153 | 0.000000 | 0 | 0 | 0 |
| 3 | `model_advised` | 74 | 122.467041 | 151.417275 | 0.016060 | 1 | 17 | 0 |
| 3 | `oracle_instrument` | 16 | 0.000191 | 74.204932 | 0.002880 | 18 | 0 | 0 |
| 4 | `rules_only` | 73 | 0.000870 | 164.003919 | 0.000000 | 0 | 0 | 0 |
| 4 | `model_advised` | 28 | 0.000334 | 74.731090 | 0.002448 | 1 | 17 | 0 |
| 4 | `oracle_instrument` | 16 | 0.000191 | 77.036807 | 0.002880 | 18 | 0 | 0 |
| 5 | `rules_only` | 73 | 0.000870 | 164.003919 | 0.000000 | 0 | 0 | 0 |
| 5 | `model_advised` | 36 | 0.000429 | 77.204560 | 0.002160 | 1 | 17 | 0 |
| 5 | `oracle_instrument` | 16 | 0.000191 | 77.036807 | 0.002880 | 18 | 0 | 0 |
| 6 | `rules_only` | 73 | 0.000870 | 164.435184 | 0.000000 | 0 | 0 | 0 |
| 6 | `model_advised` | 81 | 150.503761 | 162.764235 | 0.017343 | 1 | 17 | 0 |
| 6 | `oracle_instrument` | 16 | 0.000191 | 72.235266 | 0.002880 | 18 | 0 | 0 |
| 7 | `rules_only` | 73 | 0.000870 | 164.435184 | 0.000000 | 0 | 0 | 0 |
| 7 | `model_advised` | 81 | 150.503761 | 162.764235 | 0.017343 | 1 | 17 | 0 |
| 7 | `oracle_instrument` | 16 | 0.000191 | 72.235266 | 0.002880 | 18 | 0 | 0 |
| 8 | `rules_only` | 0 | 0.000000 | 287.733381 | 0.033696 | 0 | 0 | 0 |
| 8 | `model_advised` | 0 | 0.000000 | 287.733381 | 0.033696 | 0 | 18 | 0 |
| 8 | `oracle_instrument` | 0 | 0.000000 | 287.733381 | 0.033696 | 18 | 0 | 18 |
| 9 | `rules_only` | 0 | 0.000000 | 287.782817 | 0.033816 | 0 | 0 | 0 |
| 9 | `model_advised` | 0 | 0.000000 | 287.782817 | 0.033816 | 0 | 18 | 0 |
| 9 | `oracle_instrument` | 0 | 0.000000 | 287.782817 | 0.033816 | 18 | 0 | 18 |
| 10 | `rules_only` | 0 | 0.000000 | 287.481161 | 0.035596 | 0 | 0 | 0 |
| 10 | `model_advised` | 0 | 0.000000 | 287.481161 | 0.035596 | 0 | 18 | 0 |
| 10 | `oracle_instrument` | 0 | 0.000000 | 287.481161 | 0.035596 | 18 | 0 | 18 |
| 11 | `rules_only` | 0 | 0.000000 | 287.431642 | 0.035469 | 0 | 0 | 0 |
| 11 | `model_advised` | 0 | 0.000000 | 287.431642 | 0.035469 | 0 | 18 | 0 |
| 11 | `oracle_instrument` | 0 | 0.000000 | 287.431642 | 0.035469 | 18 | 0 | 18 |
| 12 | `rules_only` | 0 | 0.000000 | 228.112660 | 0.000000 | 0 | 0 | 0 |
| 12 | `model_advised` | 0 | 0.000000 | 228.112660 | 0.000000 | 0 | 18 | 0 |
| 12 | `oracle_instrument` | 0 | 0.000000 | 228.112660 | 0.000000 | 18 | 0 | 18 |
| 13 | `rules_only` | 0 | 0.000000 | 287.582173 | 0.034056 | 0 | 0 | 0 |
| 13 | `model_advised` | 0 | 0.000000 | 287.582173 | 0.034056 | 0 | 18 | 0 |
| 13 | `oracle_instrument` | 0 | 0.000000 | 287.582173 | 0.034056 | 18 | 0 | 18 |
| 14 | `rules_only` | 0 | 0.000000 | 287.788304 | 0.037573 | 0 | 0 | 0 |
| 14 | `model_advised` | 0 | 0.000000 | 287.788304 | 0.037573 | 0 | 18 | 0 |
| 14 | `oracle_instrument` | 0 | 0.000000 | 287.788304 | 0.037573 | 18 | 0 | 18 |
| 15 | `rules_only` | 0 | 0.000000 | 287.733381 | 0.037440 | 0 | 0 | 0 |
| 15 | `model_advised` | 0 | 0.000000 | 287.733381 | 0.037440 | 0 | 18 | 0 |
| 15 | `oracle_instrument` | 0 | 0.000000 | 287.733381 | 0.037440 | 18 | 0 | 18 |
| 16 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 16 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 16 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 17 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 17 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 17 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 18 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 18 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 18 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 19 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 19 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 19 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 20 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 20 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 20 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 21 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 21 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 21 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 22 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 22 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 22 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 23 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 23 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 23 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 24 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 24 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 24 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 25 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 25 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 25 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 26 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 26 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 26 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 27 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 27 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 27 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 28 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 28 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 28 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 29 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 29 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 29 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 30 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 30 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 30 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |
| 31 | `rules_only` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 31 | `model_advised` | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 18 | 0 |
| 31 | `oracle_instrument` | 0 | 0.000000 | 0.000000 | 0.000000 | 18 | 0 | 18 |

## Limitations And Boundaries

- The oracle is a measuring instrument, not a controller. It evaluates the four frozen catalogue commands by repeating each command through the remaining episode. That is exact only over this finite constant-command schedule, not a global optimum over arbitrary action sequences. Its reported arm still passes through HMC, which modified 432 selected commands.
- The comfort metric is defined only over occupied-mode steps. Families 16 through 31 are `eva_transition` or `contingency` families with no declared occupied-mode comfort rows, so their reported comfort deviation is zero by metric definition, not evidence of zero comfort error. The overall comfort mean therefore includes 16 zero-row family records and must not be read as a single all-mode comfort score.
- The roster has four operating conditions, four physical plant conditions and two sensor seeds, but it is still one fixed development suite. It does not establish behavior on untested scenarios or physical systems.
- The model advisory uses the declared point-prediction ranking and hard eligibility gate. It is not frozen Issue #52 `score_trajectory` compliance with calibrated uncertainty bands.
- All outputs are deterministic simulation development evidence only. They do not qualify, certify, deploy or validate hardware, and no result establishes deployment readiness for a controller or model.
- No final-suite or validation data was used. No thresholds, frozen teacher artifact, HMC policy, physics, safety contract or runtime/demo surface was changed for this study.

## Reproduction

Use a new ignored output directory for every run. Do not hand-edit or commit
generated output.

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_issue55_controller_race.py --output out/issue55-race-v2-N
```
