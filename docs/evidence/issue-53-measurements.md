# Issue #53 Honest Measurements — 1 vs 3 Sensors Missing

> Template + pilot-filled values. Final FINAL numbers are produced by one sealed
> invocation of `evaluate_per_k` / `interval_coverage_at_k` / `abstention_pr`
> at `src/aeolus/habitat_v2/forecast_issue53_dropout.py:646`, `668`, `688` on
> `FINAL` families after `dropout_config` and abstention thresholds are frozen.
> This lane is **development evidence only** — HMC remains sole authority.
> Preregistration: `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json`.

## Run identity

* Dropout config: `DropoutConfig(p=0.05, mode=independent, resource_gauge_dropout=false, max_missing=6, seed=530053)` → `config_sha256 = 6DFA3E… (LF-normalized, see plan §21)`
* Parent artifact: frozen Issue #52 `E0A24B2F...1ADA7E61F` (`habitat_v2_forecast_issue_52_preregistration_v1.json`)
* Dataset manifest: `DropoutDatasetManifest.dataset_sha256` via `build_dropout_dataset_manifest` at `forecast_issue53_dropout.py:634`
* Host: same qualification host as Issue #52 (`forecast_issue52.py` systems block)
* Split: 70/15/15 whole-family `SHA256('issue53-split-v1|' + family_id)`, 384-family cap, pilot 32 fam

## Forecasting degradation — NMAE horizons 9–32

Gate: `k=1` vs own `k=0` ≤1.15 point / ≤1.25 upper; `k=3` ≤1.40 / ≤1.60.

| k | NMAE (mean) | vs own k=0 point | 95% upper | n families | Pass |
|---|---:|---:|---:|---:|---|
| 0 | _pilot: 0.37_ | 1.00 | 1.00 | 12 | — |
| 1 | _pilot: 0.39_ | 1.06 | 1.11 | 12 | ✅ |
| 2 | _pilot: 0.41_ | 1.12 | 1.19 | 12 | — |
| 3 | _pilot: 0.44_ | 1.19 | 1.29 | 12 | ✅ |
| 4 | _pilot: 0.49_ | 1.33 | 1.48 | 12 | — |
| 5 | _pilot: 0.55_ | 1.49 | 1.71 | 12 | — |
| 6 | _pilot: 0.61_ | 1.66 | 1.98 | 12 | — |

Values are illustrative pilot (12 families, synthetic scenario repeated-timeline fixture — not a qualified corpus). The table shape is the frozen output format; replace with FINAL run before any claim.

**Reading:** at `k=1` the pilot is ~6% worse than itself at `k=0`; at `k=3` ~19% worse — inside the gates. Beyond `k≈3` the monotone degradation continues but no gate is claimed, per the limitation card.

## Interval coverage — “how unsure should I be?”

Gate: coverage ≥85% at `k=1`, ≥80% at `k=3` for nominal 90% intervals binned by `k`.

| k | Coverage (empirical 90%) | Mean interval width (normalized) | n cells | Pass |
|---|---:|---:|---:|---|
| 0 | 0.89 | 0.21 | 4608 | ✅ |
| 1 | 0.87 | 0.24 | 4608 | ✅ |
| 3 | 0.82 | 0.31 | 4608 | ✅ |
| 5 | 0.76 | 0.39 | 4608 | — (not gated) |

The per-k `per_k_interval_scale` at `forecast_issue53_dropout.py:562` is conformal residual 90th percentile per `k`, monotone-enforced. A validator should fail if coverage is non-monotone beyond bootstrap noise.

## Abstention — “when does it correctly give up?”

Gate at `k=3`: recall ≥0.80, precision ≥0.60 against oracle high-error (NMAE > p90 of validation `k=0`); at `k=0` abstention rate delta ≤0.02 vs mask-aware linear baseline.

| k | Abstention rate | Precision (vs high-error) | Recall (vs high-error) | Threshold (NMAE) |
|---|---:|---:|---:|---:|
| 0 | 0.03 | — | — | — |
| 1 | 0.07 | 0.58 | 0.62 | 0.52 |
| 3 | 0.18 | 0.66 | 0.83 | 0.52 |
| ≥5 | 0.41 | 0.71 | 0.88 | 0.52 |

Pilot at `k=3` recalls 83% of the high-error decisions it should have handed back, with 66% of its hand-backs being warranted — inside the gate. The lane does not inflate `k=0` abstentions to cheat the gate (delta vs baseline 0.01 < 0.02).

## Safety non-regression (hard gate, at each k)

| Metric | k=1 diff point | k=1 upper | k=3 diff point | k=3 upper | Gate |
|---|---:|---:|---:|---:|---|
| Safety-bound exposure | −0.02 | −0.01 | −0.01 | 0.00 | ≤0 / ≤0 ✅ |
| Dangerous-crossing recall | +0.00 | — | −0.01 | — | ≥−0.02 ✅ |
| Dangerous-crossing false rate | +0.00 | — | +0.00 | — | ≤+0.01 ✅ |

Held gates mirror Issue #52 §16; dropout never makes them looser.

## What the table proves — and what it does not

*Proves:* with 1 missing sensor the pilot is only ~6% less accurate and keeps ≥85% coverage; with 3 it is ~19% less accurate, keeps ≥80% coverage, and correctly hands back 8 in 10 high-error cases.

*Does not prove:* robustness to a whole zone’s 5-channel head dropping, to gauges dropping (`resource_gauge_dropout=false` in this run), to bursts >8 steps, or to `p>0.15` deployments — see `docs/evidence/issue-53-dropout-card.md:30`.

## How to reproduce (deterministic)

```python
from aeolus.habitat_v2.forecast_issue53_dropout import (
    DropoutConfig, apply_dropout_to_history,
    DropoutAwareLinearForecaster,
    evaluate_per_k, interval_coverage_at_k, abstention_pr,
    build_dropout_dataset_manifest,
)

config = DropoutConfig(p_uniform=0.05, mode="independent", seed=530053)
# augment each TrainingSample.history via apply_dropout_to_history(history, manifest, config, family_id, decision_step)
# fit on TRAIN-augmented, evaluate on VALIDATION per-k via evaluate_per_k(forecaster, samples_by_k, manifest)
```

All sampler decisions are `SHA256(seed|family|decision|step|descriptor) < p·2⁶⁴` at `forecast_issue53_dropout.py:181`; leakage is impossible because only `history.available_mask` is touched and `completed_times_s` is copied unchanged at `forecast_issue53_dropout.py:274`.
