# V5 historical healthy-alert forensics — 2026-08-05

**Evidence role:** `historical_forensic_only`. These findings generate V6 hypotheses. They are not V6 fit, calibration, threshold-selection, or acceptance evidence.

## Replayed source

| Item | Receipt |
| --- | --- |
| Frozen V5 source commit | `9e81011c93961a645a75d4cd7d61b2ef4ab6c9c2` |
| V5 source-manifest digest | `8d855d6e352fc8b18544ccf28f74c9660abd8f74d207057d4ed613508c2fb451` |
| V5 family-manifest digest | `dd62ba90245f3288cade8cbfd95731e16a1121682118a068899d1bc381a78022` |
| V5 development report | `out/v5-nominal-counterfactuals-canonical-2026-08-05-c/v5-development-report.json` |
| Forensic bundle | `out/v5-historical-forensics-2026-08-05-b/v5-healthy-alert-forensics.json` |
| Forensic bundle digest | `8257b77f4ba61b6e7c3a6291a9cb0769f9cf5074c605ff103fd3d654181138de` |

The replay deduplicated healthy reference streams exactly as the V5 rolling evaluator did: **48 streams**, **111 stride-one ten-tick windows per stream**, and **5,328 healthy windows per method**.

## Recorded policy replay

| Method | Healthy-alert episodes | Method-report digest |
| --- | ---: | --- |
| Calibrated rule baseline | 347 | `de7feb36d6a474f0affb03411a14fb18ebccc2a9907efd9471cf909fec7655fb` |
| Balanced temporal MLP, raw argmax | 514 | `47e374bb46c395343edf871144f47c6932c7a3d1c76decb22ca95f2487b5be6b` |
| Balanced temporal MLP, recorded gate | 529 | `d4960facbaec435f19fb39593e5625cd9abdc7d514e1ca34610dc5a82ab9bf9e` |
| Balanced temporal TCN, recorded gate | 644 | `70326485882b6e600c34a4f5c4f59fafaa414f55f73478c34be8bc619e499e05` |
| Sqrt-weighted temporal TCN, recorded gate | 274 | `fcb120951b880b42ae46df7f548ed2bc6fdef9e1b53a868de0be402b902a7704` |

The recorded one-window, 0.5-confidence gate did not reduce MLP healthy-alert episodes: raw MLP produced 514 and gated MLP produced 529. It is not a safety mechanism merely because it has the word `gate` in it.

## Frozen-sensor finding

The V5 calibrated rule parameters use `frozen_normalized_range = 0.005` and controller scale `0.30`, giving a physical three-tick tail-range threshold of `0.0015`.

- Rule replay produced 335 healthy episodes containing `frozen_sensor`.
- 332 of those 335 episode-start windows had a minimum non-processing-zone three-tick sensor-tail range at or below `0.0015`.
- All 335 episode-start windows had at least one actuator moving during the ten-tick window.

**Conclusion:** [solid] a test for actuator movement is not sufficient corroboration for a frozen-sensor alert. The simulator has legitimate short-tail sensor stability while control is moving. V6 must model expected sensor response conditional on demand/settling state, and it must prove that relationship on new room families. It must not promote simulator occupancy to a candidate feature without a separate operational observability decision.

The learned methods have a separate problem: only 55 of 416 balanced-TCN frozen episodes, 16 of 83 sqrt-TCN frozen episodes, 53 of 341 gated-MLP frozen episodes, and 54 of 356 raw-MLP frozen episodes begin at or below the rule threshold. [solid] Their frozen predictions are not reducible to the rule’s short-tail flatness criterion; a generic class predictor has learned a broader, unsafe association.

## Physical-flow finding

- Rules produced 12 healthy episodes containing `blocked_path` or `gradual_primary_fan_degradation`; none began under shared-capacity constraint (`capacity_scale_min < 1`).
- Learned physical-alert episodes under shared-capacity constraint were 45/244 (raw MLP), 52/208 (gated MLP), 61/282 (balanced TCN), and 58/192 (sqrt TCN).

**Conclusion:** [solid] shared-capacity contention is a real learned-model confound but is not the explanation for all physical false alerts. V6 must preserve explicit shared-capacity state in forensic evaluation and test isolation/persistence against new room physics; it cannot solve the problem by a one-line capacity exception.

## V6 design consequences

1. Keep an observable-context layer separate from candidate-model input. Occupancy remains forensic-only pending operational confirmation.
2. Replace unconditional tail-flatness with a sensor-health specialist whose score includes a predeclared expected-response/settling context and corroboration requirement.
3. Replace generic physical labels with a physical-flow specialist conditioned on requested/delivered flow, residual isolation, capacity state, and transient versus settled operation.
4. Treat `uncertain` as no alert but a fault-window miss in evaluation. This prevents silence from being scored as competence.
5. Do not use any number in this document to tune V6 on the V5 corpus.
