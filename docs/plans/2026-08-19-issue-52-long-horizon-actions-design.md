# Issue #52: Longer Forecasts and a Wider Action Menu

Status: proposed, awaiting Ben sign-off
Issue: https://github.com/arm-hackathon/arm-hackathon/issues/52
Design base: `843a5c1485de841462cbb47e486c2185099b71a2`
Normative appendix: `2026-08-19-issue-52-long-horizon-actions-plan.md`
Preregistration: `contracts/habitat_v2_forecast_issue_52_preregistration_v1.json`
Preregistration SHA-256: `E0A24B2FD9309ED551DCD6D4FB98EFF1FDDA6B364DE2DBE73584CCF1ADA7E61F`

## Short design note for Ben

The issue says that today's model looks eight steps ahead and chooses among four actions. The audited Habitat V2 code does not yet contain that planner. Its existing four-output model is a fault classifier, and those four outputs are fault types rather than actions. This proposal therefore builds the missing forecast-and-rank stage instead of changing two constants.

The new stage will look at the latest 16 one-minute observations, compare 12 approved 32-minute action schedules, and forecast 32 future one-minute states for each schedule. The 12 schedules are finite, versioned data. The model cannot invent actuator values. Deterministic code scores the forecasts, and only the first command of the selected schedule may be submitted as a proposal. The other 31 commands are planning context only; the system replans after the next verified snapshot.

The forecast target is topology-derived, not a hard-coded width of 51. For each ordered habitable zone it predicts CO2, temperature, and relative humidity from the snapshot's primary telemetry head, plus the three unique operational resource gauges for battery, oxygen reserve, and sorbent reserve. The width is therefore `3 * zone_count + 3`; the audited two-zone V5 topology has width 9. Resource gauges are included once. A missing, duplicate, or non-finite required target in any candidate invalidates the entire 12-candidate decision; the family is quarantined only if no complete decision remains. Secondary telemetry remains an input or diagnostic, not an alternative label source.

The timing contract is exact. If the current HMC-issued and runtime-verified snapshot represents state `S_s`, the first candidate command is `A_s`, its valid proposal has `requested_application_step=s`, and the first forecast target is `S_{s+1}`. `A_s` moves the plant from `S_s` to `S_{s+1}`. This avoids the future-step rejection that would result from labelling the first command `s+1`.

A new in-process `VerifiedHistoryBuffer`, outside the HMC authority core, will accept a snapshot only after the exact snapshot and receipt pass `hmc.verify_snapshot`. It retains normalized observation records plus their identities, requires 16 consecutive sequences with one cadence and one run/epoch/topology/schema identity, and clears itself on any identity change, gap, duplicate, reversal, or verification failure. During the first 15 observations it returns no learned proposal. The design does not claim cryptographic authentication and does not add an inter-process transport.

Training data will come from an offline-only counterfactual rollout kernel. It forks a fully specified simulator checkpoint, including `PlantState`, sensor memory, health tracker, exogenous timeline, fault schedule, scenario identity, and random seed, then advances each of the 12 schedules for 32 transitions using the public deterministic physics seam. Labels use primary environmental telemetry and the unique resource gauges from the runtime instrumentation path; hidden physical truth is retained only for evaluator diagnostics and cannot enter model inputs or targets. The offline kernel cannot issue HMC capabilities and is not imported by the runtime planner.

Checked-in Habitat V2 scenarios are currently at most 10 steps long. The data phase therefore adds long-horizon V5 scenario families with `dt_seconds=60`. A decision at completed step `s` is eligible only when 16 consecutive observations ending at `S_s` exist and all 32 transitions `A_s` through `A_{s+31}` fit. No truncated group is accepted. A minimally eligible scenario has 47 configured transitions when history includes `S_0` through `S_15` and the decision is at `s=15`; families with multiple decisions must be longer.

Candidate status is explicit:

- `STATICALLY_VALID`: every command has the required shape and static bounds.
- `ROLLOUT_FEASIBLE`: all 32 offline transitions complete for a particular checkpoint.
- `RUNTIME_FIRST_STEP_FEASIBLE`: HMC preflight accepts `A_s` for the current state.

Future infeasibility makes that candidate ranking-ineligible and is never silently removed from its ranking group. Its unavailable tail is masked from forecast training and metrics, and every forecaster uses the same complete-trajectory eligibility mask. The catalogue includes an intentional `candidate_hold`. Selecting it is a valid learned proposal. Abstention, malformed output, timeout, HMC rejection to hold, and HMC emergency override are separate outcomes.

HMC remains the only final-command and plant-step authority. It may accept a proposal, policy-modify or clamp it, emergency-override it, or reject it to safe hold. The model receives no plant handle, capability token, or safety bypass. HMC still validates identity and bounds, performs operating-mode and reserve policy, runs physical preflight, binds the final command to a capability, calls `HMC.step()`, and verifies causal replay.

## Expected improvement and frozen decision rule

The primary forecast metric is normalized MAE over horizons 9-32 across all target channels. The learned model must improve its point estimate by at least 10% over the best frozen non-neural baseline, and the upper bound of the paired 95% family-bootstrap ratio must be below 0.98. Horizons 1-8 must be non-inferior within 5%.

The proposed primary ranking metric is normalized regret against offline oracle-trajectory ranking. It remains inactive until a separate commit-bound amendment freezes the exact true and predicted score formulas, units, normalizers, weights, hard-infeasibility value, operational metric formulas, and manifest digests, and Ben approves that commit. Only then may the 12-candidate arm be tested for at least 10% point-estimate regret improvement over the frozen four-candidate ablation with an upper 95% ratio below 0.98.

Hard release gates require zero authority violations, zero replay/provenance failures, no increase in total safety-bound exposure, no more than 2 percentage points loss in dangerous-crossing recall, no more than 1 percentage point increase in false crossings, and, after the amendment activates their exact formulas, no more than 5% regression in wear, reserve use, or healthy false interventions. Runtime inference for all 12 candidates must have p99 at or below 250 ms and zero deadline misses over 1,000 timed runs after 20 warm-ups. The initial qualification host is Windows 11 Pro build 26200 on an AMD Ryzen 5 4600H, x64, 12 logical processors, 16,505,847,808 bytes RAM, and Python 3.14.0, with no competing benchmark workload. Statistics use the preregistered deterministic SHA-256 family-bootstrap sampler for 10,000 resamples and the registered Holm procedure.

The family count is not guessed in advance. Initial approval permits a maximum 32-family raw feasibility pilot. Forecast power uses paired family log-ratios from the frozen deterministic baseline pair. Ranking power remains blocked until the metric amendment is approved. The amendment records the exact manifests, formulas, power result, final roster, and deterministic 70/15/15 split. The roster is capped at 384 families and 2,000,000 candidate transitions.

## Delivery sequence

1. Ben approves the exact three-file package and its Git commit.
2. Implement the approved contracts and run only the bounded raw feasibility pilot.
3. Commit the exact manifests, scoring formulas, operational metric formulas, power result, roster, and split as a metric amendment.
4. Obtain Ben's separate approval of that amendment before comparative fitting, model training, or experiments.
5. Establish persistence, recent-delta, linear, and autoregressive baselines under the frozen amendment.
6. Train the smallest action-conditioned forecast model that can pass the gates.
7. Qualify forecast quality before enabling deterministic candidate ranking.
8. Integrate a feature-disabled-by-default advisory source and run authority, fail-closed, replay, latency, and end-to-end demo tests.
9. Open the sealed final split once after all choices are frozen.
10. Present the implementation and any safety-core diff to Ben for a separate review. Deployment remains separately blocked.

## Approval requested

Ben can approve by replying with the exact local commit and this statement:

> I, Ben (`bbeennyy860-cyber`), approve the Issue #52 design package at the identified commit. I authorize contract implementation and the bounded raw feasibility pilot under the attached HMC authority boundary. Comparative fitting, ranking power calculation, model training, experiments, and deployment remain blocked until I approve the required commit-bound metric amendment. Any safety-core change requires my separate review before merge.

Until that approval is recorded: `BEN_SIGN_OFF=false`, `CODE_AUTHORISED=false`, `DATA_GENERATION_AUTHORISED=false`, `TRAINING_AUTHORISED=false`, `EXPERIMENTS_AUTHORISED=false`, `DEPLOYMENT_AUTHORISED=false`, and `METRIC_AMENDMENT_APPROVED=false`.
