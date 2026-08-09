# Bounded response and recovery boundary

## Status

This document separates two development-stage mechanisms:

1. the historical bounded-response governor on `yarofix2`; and
2. the schema-v10 reserve recovery authority evaluated by C4.

Neither is an accepted physical recovery controller. C4 closed as **Outcome B**:
its reproducible development run failed the transient physical-zero handback
acknowledgement gate and the physical-reserve-delivery benefit gate. See
[`recovery-protocol-acceptance.md`](recovery-protocol-acceptance.md).

## Historical bounded-response governor

The earlier `BoundedRecoveryGovernor` is a deterministic, causal,
observable-only development controller. It emits bounded primary per-zone
commands with structured reasons and runs alongside the baseline controller; it
does not directly mutate plant state.

Its historical response-evidence harness is retained as development context. It
is not C4 recovery evidence and does not establish independent reserve delivery,
physical handback, a qualified model, or a final result.

The policy constraints remain useful as source-level boundaries:

1. decisions use completed-tick observable data only;
2. hidden fault state, health, schedules, seeds, and internal noise do not enter
   the decision input;
3. commands are finite, bounded, and rate-limited; and
4. the physics engine is the only component that changes plant state.

## C4 deterministic reserve authority

Schema-v10 recovery scenarios add a disjoint primary/reserve topology. The
`DeterministicRecoverySupervisor` is the only component that can own the
reserve command channel in `PROTECT` or `HANDBACK`; `NOMINAL` and `DEGRADED`
use an explicit reserve-off owner.

The authority checks identity, topology, selector and command digests, sequence,
finite observations, bounded commands, persistence, recurrence, and reserve
delivery failure. Its state transitions are recorded separately from the legacy
plant projection in strict versioned recovery traces.

The frozen settings include a 36-tick maximum handback duration. Completion
requires physical command, actuator position, and delivered-flow acknowledgement
rather than a command-zero assertion alone.

## C4 evidence boundary

The C4 runner evaluates each development family under four exact arms:

```text
reference_reserve_off
reference_governed
fault_reserve_off
fault_governed
```

It records paired safety and benefit predicates. A recovery claim requires the
mechanism-to-outcome chain to hold: reserve authority command, positive observed
reserve delivery where required, and predeclared target-zone benefit against the
fault reserve-off arm.

C4 did not satisfy that chain. Its positive aggregate improvement submetrics do
not override the failed mandatory physical-delivery predicate, and its passing
safety subgates do not override the failed transient handback acknowledgement.

## Reproduction boundary

The C4 corpus command is intentionally strict and writes only to a new ignored
output directory:

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.recovery_evidence \
  scenarios/sweep-recovery-development.json /absolute/new-output-directory
```

It is a historical reproduction command, not permission to tune, rerun a final
suite, train an adviser, or treat deterministic output as acceptance.

## Scope

All evidence here is abstract simulation evidence. It makes no claim about
physical airflow hardware, real CO₂ safety limits, hardware-in-the-loop tests,
Arm performance, INT8 quantisation, deployment, or autonomous control.
