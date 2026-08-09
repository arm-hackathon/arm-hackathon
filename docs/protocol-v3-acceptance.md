# Protocol v3 acceptance record

Gate: validation-only decision policy and untouched final evaluation
Status: accepted

## Implemented

Protocol v3 retires the inspected v2 test and stress partitions as decision
inputs. It separates the current evidence into:

- a deterministic development suite with 360 train and 120 validation scenario
  families;
- a separately generated final suite with 180 fresh scenario families;
- a strict policy artifact selected only from development data; and
- a final evaluator that applies the policy without reselection.

A family is the independent replay unit: its healthy reference and fault replay
share a family and identical declared non-fault configuration, while differing
in fault profile, and are assigned together. Splits use disjoint canonical
scenario identities, not only different labels or shuffled windows.

The policy stores the selected learned candidate, validation candidate receipt,
rule-calibration receipt, validation model/rule comparison, ONNX parity receipt,
contract metadata, and detector-JSON hash. Before a final report can be created,
the evaluator verifies the supplied policy, detector JSON, detector ONNX,
development manifest and final manifest hashes; rebuilds both learned candidates
from development train/validation rows; replays candidate selection, ONNX parity
and the rule calibration grid; re-evaluates the validation comparison; and
rejects an existing report path. It then evaluates only the final rows and
copies the already-frozen policy outcome into the report.

## Measured evidence

A canonical isolated run produced:

| Final-suite evidence | Temporal MLP | Calibrated rules |
|---|---:|---:|
| Macro-F1 | 0.5754744477098027 | 0.642588422763726 |
| Nominal false-alarm rate | 38.5698% | 0.5631% |
| Nominal false alarms | 2,055 | 30 |
| Overall median detection latency | 9 ticks | 10 ticks |
| Scored windows | 8,000 | 8,000 |

The frozen decision is `preferred_method=rule_baseline` and
`ai_advantage_demonstrated=false`.

The policy's advantage criterion accepts a learned detector only for either a
large quality improvement without material false-alarm or fault-recall
regression, or at least a 20% latency reduction while preserving quality,
per-fault recall and false alarms. The final temporal MLP reduced the measured
median detection latency by 11.1%, not 20%, while losing macro-F1 and adding a
38.0 percentage-point nominal false-alarm regression. The rule baseline remains
preferred.

The canonical run's binding values were:

```text
Development manifest: 4d15d46d46d6a0f339a2e69126fbbb4f990dba9f263b890eeacfbbe9a63630b2
Final manifest:       474704de6c6c5930fe4825e0c5238b5f04b3e6eadee487fcfe4fdd032b5d7112
Policy:               7a16f6887379fd6656a0e2bf1223fde3dfc618302c08786b04b02d4338abfc4c
Detector JSON:        c3b416c77e8b63eca558166cb02ac522af950495e2e8b51837cf2678b1c34344
Detector ONNX:        c065becb4f9948d322f9c6204cc5ab72b9fcb58725dc65eb7c1ed6b25e3b4c70
```

## Reproduction

Use CPython 3.11 and the locked ML environment. Start from a clean worktree and
a new ignored output directory. The commands deliberately expose each protocol
stage; a one-shot runner would make it too easy to hide a reselection step.

```bash
uv sync --locked --python 3.11 --extra ml

RUN=out/v3-<run-id>
PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -m aeolus.sweep \
  scenarios/sweep-v3-development.json "$RUN/development-sweep"
PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -m aeolus.corpus \
  --v2 "$RUN/development-corpus" "$RUN/development-sweep/families.json"

DEV_HASH=$(PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -c \
  "from pathlib import Path; from aeolus.families import load_family_manifest; print(load_family_manifest(Path('$RUN/development-sweep/families.json')).manifest_sha256)")

PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -m aeolus.experiment select \
  "$RUN/development-corpus/corpus.jsonl" "$RUN/development-sweep/families.json" "$DEV_HASH" \
  "$RUN/detector.json" "$RUN/detector.onnx" "$RUN/policy.json"

PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -m aeolus.experiment build-final \
  scenarios/sweep-v3-final.json "$RUN/final"

FINAL_HASH=$(PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -c \
  "from pathlib import Path; from aeolus.families import load_family_manifest; print(load_family_manifest(Path('$RUN/final/sweep/families.json')).manifest_sha256)")
POLICY_HASH=$(sha256sum "$RUN/policy.json" | cut -d' ' -f1)
DETECTOR_HASH=$(sha256sum "$RUN/detector.json" | cut -d' ' -f1)
ONNX_HASH=$(sha256sum "$RUN/detector.onnx" | cut -d' ' -f1)

PYTHONPATH=src uv run --locked --python 3.11 --extra ml python -m aeolus.experiment final-evaluate \
  "$RUN/final/corpus/corpus.jsonl" "$RUN/final/sweep/families.json" "$FINAL_HASH" \
  "$RUN/development-corpus/corpus.jsonl" "$RUN/development-sweep/families.json" "$DEV_HASH" \
  "$RUN/policy.json" "$POLICY_HASH" "$RUN/detector.json" "$DETECTOR_HASH" \
  "$RUN/detector.onnx" "$ONNX_HASH" "$RUN/final-report.json"
```

The final command must be run once per report path. It fails rather than
silently overwriting a final report.

## What this proves

- The declared simulator, corpus, selection rule, rule calibration, model
  export and final evaluation can be reproduced deterministically under this
  locked environment.
- The final suite was not used to select the candidate, tune rule parameters or
  determine the preferred method.
- Under the declared synthetic distribution and final family set, the rule
  baseline outperformed the selected temporal MLP under the frozen policy.
- Python and ONNX agreement is verified on validation rows before the final
  report is accepted.

## What this does not prove

- It does not estimate uncertainty over 8,000 independent deployments. The
  8,000 scored windows are correlated observations within 180 replay families;
  no confidence intervals or independent-window claims are reported.
- Detection latency is simulator ticks from observable onset to the first
  correct causal label. It is not wall-clock inference latency, throughput,
  scheduling latency, alert burden, real-time performance or Arm performance.
- The final suite is a fresh held-out sample of the same declared synthetic
  operating profiles. It is not an OOD stress test, physical validation,
  hardware-in-the-loop result, safety case or deployment trial.
- No INT8 model, Arm benchmark, production controller or autonomous actuator
  command is implemented or claimed.

## Next authorised step

Develop a new candidate only from a new development protocol. Keep this final
suite frozen. A later protocol needs predeclared external or shifted operating
profiles if it wants to claim OOD robustness, and a declared Arm device plus
raw benchmark evidence if it wants to claim runtime performance.
