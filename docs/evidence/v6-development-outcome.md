# V6 development outcome

Date: 2026-08-07

Evidence role: `development_only`

Decision: retain `conditional_rule_v6`. No learned candidate passed the frozen development gate. Do not connect the response layer; do not treat any learned artifact as deployment-selected.

## Frozen evidence contract

- Canonical sweep spec SHA-256: `6036f592e1c2c82ddbbb12584ead617e86342c39dd6200a04d3a99910a4ab04a`
- Generated family manifest SHA-256: `89afbde7101a9a7e71f3c304890c6048c65eb850af8e9a0abb5a3977c040c348`
- Corpus manifest SHA-256: `85495da73178b3ae0ef8f6c9a136c72b04c596a78096ab71a159e9b826f5d8d4`
- Source manifest SHA-256: `64e30d3f801d04a402ff785c066a220122ecc87d84bbb0629e7827692a3d7a7a`
- Head commit: `417d498126dc8459b04a5a7cf842598888de1391`, clean worktree
- Fit room families: `room-balanced` (seeds 2100–2101), `room-volume-asymmetric` (2110–2111)
- Calibration room family: `room-capacity-constrained` (2120–2121)
- Validation room family: `room-transition-heavy` (2300–2301)
- Windows: fit 15,984 / calibration 7,992 / validation 7,992; total 31,968
- Causal stride: one simulator tick; window: 10 ticks

The gate required a learned candidate to beat the conditional rules in named-fault macro-F1 while meeting all of these safety conditions:

- no more than `10.0` false-alert episodes per 1,000 eligible healthy ticks;
- no more than `2.0` episodes per 1,000 above rules;
- named detection recall non-regression;
- strict finite JSON and non-empty artifact checks.

## Model-versus-rules table (validation)

| Method | Named macro-F1 | Healthy episodes / 1,000 | Healthy streams alerted | Named recall | Post-onset uncertainty | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conditional_rule_v6` (baseline) | 0.0 (by design: concerns only) | 0.0 | 0/2 | 0.0 | 0.0 | comparator; retained fallback |
| `observable_context_centroid_v1` (threshold 0.5) | 0.0 | 0.0 | 0/2 | 0.0 | 1.0 (all windows abstained) | fail |

## Failure classification

The v6 result is a **transfer/abstention failure**, not the v4/v5 false-positive disease.

### 1. Rules are blind by declaration

`conditional_rule_v6` emits `sensor_health_concern`, `physical_flow_concern`, `uncertain`, or `nominal` — never `blocked_path`, `frozen_sensor`, or `gradual_primary_fan_degradation`. Named macro-F1 is therefore 0.0 by construction. The physical-flow specialist is nevertheless a strong concern detector on validation: precision `1.0`, recall `0.603` (924 true positives, 0 false positives, 608 false negatives over 1,532 physical-flow windows). The sensor-health specialist detects nothing (recall 0.0) — see defect 3.

### 2. Centroid cannot transfer across room families

The centroid consumes raw `observable_context_v1` (46 dims/tick). On the held-out validation room family, every window's maximum softmax probability fell below the calibrated `0.5` abstention threshold: `post_onset_uncertainty_fraction = 1.0`, `healthy_uncertainty_fraction = 1.0`. The candidate is not unsafe — it is blind. It made zero named detections on validation and therefore gained zero macro-F1 over a baseline that itself never names faults.

On calibration (same room family distribution as fit), the centroid fired but collapsed: all classes mapped to `blocked_path` (nominal 1,596, frozen 762, degradation 762 all named blocked), giving named macro-F1 `0.1105` with `8.33` healthy episodes per 1,000. Calibration selection chose threshold `0.5` on a grid `(0.0, 0.5, 0.75, 0.9)`; the identity of calibration and fit room-family distributions is exactly why the calibration metrics are optimistic and the validation metrics are zero.

### 3. Frozen-sensor specialist is structurally blind to noise

Empirically: the freeze is applied at the latent level (fault trace flat ~0.238–0.241 vs reference rising 0.24→0.31 over ticks 25–40), but measurement noise is added after the freeze, so observed readings jitter. Across 1,032 post-onset frozen windows: `sensor_max_delta` median `0.0080` (max `0.0165`); the specialist's flatness threshold is `0.0015`, below the noise floor. Result: `sensor_not_flat` on all 110 probed windows; frozen recall structurally 0.0.

The distributions overlap on the current sensor-health features: frozen vs reference slope median `-0.0002` vs `-0.0012`, range `0.0279` vs `0.0375`, proxy `0.0724` vs `0.0853`. Healthy traces also saturate flat late in a run, so absolute flatness cannot separate frozen from settled healthy. The v6 sweep's `transition-heavy` room family makes this harder: the reference itself goes flat.

## What v7 must change

1. **Named-fault escalation with calibration-only thresholds.** Escalate `physical_flow_concern` → `blocked_path` or `gradual_primary_fan_degradation` via a calibrated discriminator, and a calibrated sensor-trend/expected-change test for `frozen_sensor`. The concern layer already has precision 1.0; the missing piece is naming.
2. **A transfer-robust learned candidate.** Replace raw-context nearest-centroid with a candidate trained on residual/trend features that are normalized across room families, gated by the specialist concern layer so healthy silence is preserved.
3. **No fresh final suite and no response integration** until a candidate beats the escalated baseline under the frozen gate with an independent reproduction.

## Deliberate non-claims

- No Arm, ONNX, INT8, hardware, or wall-clock claim.
- No deployment or response-layer integration authorization.
- The v6 validation families are consumed by the v6 centroid lineage; a revised lineage must use newly predeclared untouched validation families.
