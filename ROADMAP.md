# AEOLUS research roadmap

- **Status:** proposed direction and development programme for team review.
- **Updated:** 2026-09-06.
- **Source baseline:** `main` at `25a8a9ac194472d05b13983d2f8078f3e0b67f21`, package version `0.8.0`.
- **Authority invariant:** learned components may advise; the deterministic Habitat Management Computer (HMC) remains the sole final-command, plant-step and replay authority.

## Direction

AEOLUS aims to become a habitat-focused environmental-intelligence research
platform that learns from observation histories, predicts the consequences of
interventions, plans under uncertainty, and demonstrates when its recommendations
improve outcomes against competent conventional methods.

The project includes the simulation world, learned models, planning, evaluation,
reproducible infrastructure and a research interface. Visualization supports
inspection of real evidence; it is not the central research result.

**Read the [detailed development programme](docs/plans/aeolus-development-programme.md)**
for the proposed team split, architecture, scientific questions, phased delivery,
24 work packages, dependencies, acceptance criteria and GitHub issue template.
That document is the current detailed planning source; this roadmap is its
short entry point, not a second task catalogue.

## What exists and what remains candidate work

The merged foundation is a deterministic eight-zone, reduced-order environmental
analogue. It includes gas inventories, temperature, pressure, airflow, resources,
requested and achieved actuation, physical/sensor faults, HMC arbitration and
replay. Physics provenance and independent numerical-reference checks already
exist; future work should extend them rather than describe them as missing.

AEOLUS is not CFD, a complete ECLSS model, a NASA/Artemis/Gateway digital twin,
or flight-validated software. Its physical parameters and scope include explicit
engineering assumptions. Simulation success does not authorize physical control.

The [open research stack #84–89](docs/plans/aeolus-development-programme.md#1-starting-point)
adds family generation, corpus tooling, attribution, linear baselines, the BDM-v1
TCN, calibration and a closed-loop development study. Those implementations are
not all merged into main. Their source and evidence must be reviewed before
choosing the next foundation.

### The latest negative results change the next step

- The [BDM-v1 evidence card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-75-bdm-v1-card.md)
  reports that the trained causal TCN failed promotion and did not beat the
  action-conditioned ridge baseline.
- The [closed-loop evidence card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-77-closed-loop-card.md)
  reports no admitted proposals across the compared arms, universal abstention
  for the calibrated BDM, and zero measured safety exposure even under hold.
  The candidate is withheld; the published record says BLIND_FINAL remains sealed.

These are reported development results, not newly reproduced measurements or
general safety findings. Preserve them unchanged. First diagnose model numerics,
action opportunity and the proposal-to-actuation path. Do not simply train a
larger network to improve a safety score already at its minimum.

Historical model evidence also remains bounded: the original action-aware MLP
has incomplete full-campaign custody; forecast-only missingness and distillation
results do not establish closed-loop utility; the V4/V10 development result does
not alone isolate learned credit. See [MODEL_CARD.md](MODEL_CARD.md) and the
[historical advisory index](docs/evidence/closed-loop-advisory-historical-index.md).

## Proposed responsibility split

Subject to agreement by all three, and without changing existing issue owners:

- **Ben — models and state estimation:** causal features, hidden-state inference,
  training, action-conditioned prediction, uncertainty and model diagnostics.
- **Alex — simulation world and data:** physical mechanisms, configurations,
  sensors, faults, numerical verification and scenario/corpus generation.
- **Yaroslav — planning and integration:** conventional and learned planning,
  HMC adapters, action/outcome tracing and the complete research experience.

Benchmark design, independent evaluation, public claims and major architecture
choices are shared. Each lane owns its component tests and documentation;
Yaroslav is not the owner of every miscellaneous infrastructure task. Existing
contributors retain credit and current assignments until explicitly changed.

## Research loop and first milestone

```text
observe -> infer state -> predict consequences -> rank -> constrain -> act -> verify
```

The first milestone is a useful, non-degenerate decision on a small justified
problem using the current layout. Show that different permissible **achieved**
actions have different outcomes, permitted observations contain actionable
information, a competent conventional method can exploit some opportunities,
and some cases correctly require no action or abstention.

Two separate successor tracks are proposed:

- **Recovery:** reduce harmful exposure or improve recovery after justified
  disturbances.
- **Efficient operation:** improve resources, comfort or wear while preserving
  predeclared hard constraints and safety non-inferiority.

The new protocol must declare endpoints, units, margins, baselines, split groups,
sample-size rationale and budgets before candidate selection. This does not
retroactively change BDM-v1's failed gate.

## Evidence-gated sequence

1. **Converge and diagnose:** review the current stack, preserve evidence, identify
   model failures and decision/authority bottlenecks, and choose a pinned base.
2. **Establish opportunity:** build a useful conventional end-to-end decision,
   with physically verified mechanisms and negative/ambiguous controls.
3. **Train and compare:** establish reliable training, serious baselines and
   controlled temporal/hybrid model experiments; model size follows evidence.
4. **Plan and attribute:** evaluate HMC-filtered outcomes and isolate learned
   benefit from deterministic guards, model choice and planner design.
5. **Generalize and falsify:** test new layouts, mechanisms and assumptions;
   investigate failures rather than hiding them in aggregate scores.
6. **Confirm and release:** independently evaluate a frozen candidate on untouched
   evidence, then deliver installable artifacts and tested reproduction paths.
7. **Optional external validation:** pursue safe measurements or a domain
   collaboration under separately agreed permissions and scope.

These are gates, not deadline promises. Graph models, larger architectures,
optional accelerator training, bounded MPC and 3D replay are conditional tools,
not all mandatory prerequisites for the first meaningful result. Platform
maintenance may proceed before model confirmation; packaging or hardware work
cannot substitute for it.

## Non-negotiable evidence boundaries

- Only causal, operationally available information enters models and realistic
  baselines; privileged simulator truth is separately labelled diagnostic data.
- Keep related scenario variants and counterfactual actions in one split group.
  Keep final confirmation out of iterative tuning.
- Distinguish proposals, HMC decisions, final commands and achieved actions.
- Retain strong conventional comparisons, failed attempts, uncertainty and
  worst-family outcomes.
- Missing, stale, invalid or unsuitable advice abstains/fails closed; HMC retains
  all command authority.
- Report model quality, planning benefit, runtime speed, Arm-specific evidence,
  package readiness, publication and deployment separately.
- No physical actuation, broad real-world safety claim, online self-modification
  or blind-population access is authorized by this plan.

## Work tracking and next action

Existing issues #70–71 are closed and #72–77 remain open at the source snapshot.
The newer plan does not recreate or reassign those tasks. Proposed work-package
IDs are not newly created GitHub issues.

Review the [ownership split](docs/plans/aeolus-development-programme.md#3-three-person-responsibility-split),
[success criteria](docs/plans/aeolus-development-programme.md#5-scientific-questions-and-success-criteria)
and [first execution wave](docs/plans/aeolus-development-programme.md#8-first-execution-wave).
Then convert only the next agreed wave using the
[issue template and dependency rules](docs/plans/aeolus-development-programme.md#11-turning-the-plan-into-github-issues).

The [earlier BDM-v1-first roadmap](https://github.com/arm-hackathon/arm-hackathon/blob/61341d3fa9fcdd17524d19dedac38c350666f244/ROADMAP.md)
is preserved in Git history. Its sequence is superseded for new planning by the
reported results above; its historical protocol is not silently rewritten.
The [earlier working-model PRD](docs/plans/2026-08-09-aeolus-working-model-prd.md)
also remains a historical execution record, not the current task list.

Publishing this documentation does not start implementation, create new issues,
launch a development fleet, train models, merge the research stack or deploy
anything. Those actions follow explicit team decisions and bounded scope.
