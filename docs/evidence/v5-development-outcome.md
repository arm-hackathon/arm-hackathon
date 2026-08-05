# V5 development outcome: load-preserving nominal counterfactuals

**Status:** development-only negative result. No learned detector is eligible for integration. Rules remain the retained baseline.

## Question

V3/V4 false alerts clustered around healthy telemetry that looked unusually flat or high-load. V5 tests whether more realistic **timing** of normal occupancy/load reduces that confusion without changing total declared occupancy load per zone.

## Protocol

- Schema: `aeolus_sweep_v5`; development-only.
- New normal profiles: `v5-staggered-load` and `v5-lab-peak-transition`, plus low/high baseline profiles.
- Every zone retains its base occupancy-period boundaries and duration-weighted total declared load. V5 alters only validated positive period multipliers whose weighted sum is conserved.
- Fault scenarios are deep copies of their shaped reference and differ only in `fault_profiles`.
- Fresh family clusters: fit `1100-1103`, internal calibration `1104-1105`, validation `1300-1305`.
- Validation families: 720. V5 is tested disjoint from V3 development, V3 final, and V4. V3 final was not used.
- Candidate set and safety policy are unchanged from V4.

## Canonical receipt

Command:

```sh
PYTHONPATH=src uv run --locked --python 3.11 --extra dev python -c   "from aeolus.model_cycle_v5 import run_v5_development;   run_v5_development('scenarios/sweep-v5-development.json', '<empty-output-dir>')"
```

- Source commit: `9e81011c93961a645a75d4cd7d61b2ef4ab6c9c2` (clean worktree)
- Sweep SHA-256: `d9ae68eb4ad16e91bc8318d1ee028e51efec35d3fa1352d3f10df88becfd5065`
- Family manifest SHA-256: `dd62ba90245f3288cade8cbfd95731e16a1121682118a068899d1bc381a78022`
- Source manifest SHA-256: `8d855d6e352fc8b18544ccf28f74c9660abd8f74d207057d4ed613508c2fb451`
- Report SHA-256: `f5b8aa520e7e1d3da99a0530a9309ea91130082487c0255b8e405a60a1fe52bd`
- Output: `out/v5-nominal-counterfactuals-canonical-2026-08-05-c/v5-development-report.json`

## Result

The development gate failed. No learned candidate was selected; `rule_baseline` remains retained.

| Method | Macro F1 | Nominal FAR | False-alert episodes / 1,000 healthy ticks | Healthy streams alerted |
| --- | ---: | ---: | ---: | ---: |
| Rules baseline | 0.6045 | 9.95% | 63.81 | 100% (8/8) |
| Balanced gated CNN | 0.6632 | 21.79% | 121.25 | 100% (8/8) |
| Square-root gated CNN | 0.5647 | 10.06% | 57.43 | 100% (8/8) |
| Balanced gated MLP | 0.6172 | 31.00% | 109.61 | 100% (8/8) |
| Balanced raw MLP | 0.6180 | 41.75% | 94.97 | 100% (8/8) |

The diagnostic learned winner was the balanced gated CNN, but it fails the operational gate: `121.25` episodes/1,000 healthy ticks exceeds the ceiling of `10`; it is `57.43` above the rules baseline, where at most `2` is allowed. Its nominal FAR is `+11.84` percentage points versus rules, beyond the allowed `+1` point. The square-root CNN lowers episodes below rules but still produces `57.43`, not `<= 10`, and loses too much blocked-path and gradual-degradation recall.

All learned artifacts passed ONNX parity, operator allowlist, and strict-artifact checks. Independent reproduction is not verified by this single canonical run, so no candidate could be eligible even absent the operational failures.

## Interpretation

V5 answered the intended question: preserving total nominal load while changing its timing does **not** make the current learned detectors operationally safe. The failure is not rescued by moving the existing threshold/persistence grid: lower alert burden trades directly into unacceptable fault-recall loss. This is a negative but useful result. It rejects a brittle learned-detector claim rather than hiding it behind correlated window metrics.

## Boundaries and next step

This is deterministic simulator development evidence, not real-habitat validation. Do not create or inspect a V5 final suite from this result. The next research step is error analysis by operating profile and predicted class, then a predeclared alternative architecture or specialist/rule strategy. No response-layer integration is authorized.
