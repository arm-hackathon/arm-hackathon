# System: Forecast-Advised HMC (How It Works and Who Does What)

**Status: development evidence only. Not qualification. Not deployment. No learned actuator authority.**

## The one-paragraph version

AEOLUS Habitat V2 simulates a spacecraft habitat's air system (8 zones:
temperature, pressure, CO2, oxygen, humidity, airflow). A deterministic
controller — the **Habitat Management Computer (HMC)** — decides fan speed,
dampers, scrubber, condenser, cooling and oxygen injection every step. We
trained a small neural network that **predicts the next 8 steps** of the
habitat for each candidate action, and integrated it as an **adviser**: it
recommends, HMC decides. HMC can and does override the model (81 overrides in
793 proposals during the experiment campaign).

## The agents/components and their authority

1. **The simulator (plant)** — deterministic physics. Given the same scenario,
   seed and commands, it always produces the same future. This is what makes
   the evidence replayable.
2. **HMC (the authority)** — the only component allowed to command actuators.
   Every step it: observes telemetry, verifies its own snapshot, accepts or
   rejects proposals, arbitrates, and issues the final command. Safety rules
   live here, not in the model.
3. **The forecast adviser (the model)** — a read-only adviser. Each step from
   step 16 onward it receives the last 16 observations, predicts the next 8
   steps five times (once per candidate action), scores each prediction
   against HMC's own frozen safety thresholds, and proposes the lowest-risk
   action. If nothing looks risky it proposes nothing (in dormant scenarios it
   stayed silent throughout).
4. **The experiment harness** — runs both arms (HMC alone vs HMC + adviser)
   on identical scenarios and seeds, checks shadow physics against HMC
   receipts every step, and replays every trace. Results are only comparable
   because pairing is exact.

## What the model actually is

- A multi-layer perceptron (MLP): 3,132 inputs → 512 → 512 → 256 → 408
  outputs, GELU activations, ~1.9M parameters.
- Inputs: the last 16 timesteps of habitat telemetry (flattened, 3,104
  numbers) + the candidate action being considered (27 numbers) + one flag
  saying "an action is proposed".
- Outputs: predicted habitat state for the next 8 steps — 51 values per step
  (6 environmental channels × 8 zones + 3 resource gauges).
- Trained on 23,400 simulator examples from the Historical V2 archive;
  evaluated on 6,630 examples from 17 clusters never seen in training
  (normalized MAE 0.1146 vs 0.2880 for the same model without action input —
  the action information cuts error by 60%).
- Checkpoint: `action-aware-mlp-v1.pt` (8 MB), hash-bound to the frozen
  training run `full-v1-20260818-a`.

## What the model does NOT do

- It never touches actuators. HMC validates, caps, modifies or rejects every
  proposal.
- It does not handle missing/broken sensors — it was trained without
  availability masks. That is the explicit next qualification tier.
- It is not qualified, certified, or deployed anywhere.

## Reproduce

```bash
pip install -e ".[closed-loop]"
python experiments/closed-loop-advisory-20260818/run_demo.py
```

Expected: control exceedance 19.9406 vs advised 0.0 on the demo scenario, in
about one minute. Full campaign evidence: `CLOSED_LOOP_REPORT_V2.md` and
`results-summary.json` (hash-bound to the raw local evidence).

## Evidence discipline (for reviewers)

- Scoring rules, roster and success criteria were frozen in self-hashed
  preregistrations before outcomes were seen.
- Raw result files are not committed (repo convention); `results-summary.json`
  carries per-run metrics plus SHA-256 of the full local evidence.
- Scenario/contract JSON files are stored with exact CRLF bytes because the
  sealed pipeline binds hashes to those bytes; `.gitattributes` (`* -text`)
  preserves them across clones.
