# Project AEOLUS Plan

## End goal

Build an Arm-powered orchestration layer for otherwise independent ventilation
actuators. The system must detect a local equipment fault, understand its
effect on the wider habitat, and coordinate a bounded response using healthy
equipment before environmental conditions become unsafe.

The final demonstration must show:

```text
isolated actuator degrades
→ system-wide AI identifies the fault and affected zones
→ deterministic governor activates healthy redundant capacity
→ airflow recovers and CO₂ exposure is reduced
```

AI provides diagnosis and confidence. Deterministic safety logic retains
control of every actuator command.

## Status — Gates 0–2 accepted; fair `alex/ai-2` experiment implemented

Landed: the deterministic schema-v9 simulator with separated requested/delivered
airflow and explicit residuals; five scenarios (nominal, healthy high-demand,
gradual degradation, blocked path, frozen sensor); the telemetry allowlist and
model-feature projection; the HTML visualiser; the leakage-safe labelled window
corpus; and the streaming rule baseline with its evaluation harness (111/115
windows on corpus v1, latencies 10/5/10 ticks).

The experimental branch adds deterministic measurement noise, bias and drift;
controller-facing imperfect CO2 sensing; an 840-family train/validation/IID
test/OOD-stress sweep; validation-selected softmax and temporal-MLP candidates
over `float32[10,24]`; validation-calibrated robust rules; stride-one causal
latency; and FP32 ONNX export. The locked IID result does not demonstrate an AI
advantage: temporal-MLP macro-F1 is 0.5765 versus 0.6410 for calibrated rules,
with substantially more false alarms. The model's 9-tick median latency versus
11 ticks for rules is below the fixed 20% latency-win threshold. Stress
evidence also favours rules, so the rule remains preferred.

Gate 0 accepts the R2 semantic contract. Gate 1 adds graph-derived
outbound/return loop pairing, the exact topology-bound `model_input_v1`
float32[24] selector, canonical selector/topology hashes, and fail-closed
artifact metadata validation.

Gate 2 is accepted. Its strict, topology-bound family manifest binds each
healthy/fault pair to one split and rejects pairs that differ outside
`fault_profiles`. Corpus v2 persists the frozen model-input contract and labels
windows from the first divergent paired `model_input_v1` tensor. Windows that
straddle that observable onset update stateful detectors but are excluded from
training and scored metrics. The three current families are contract fixtures,
not a training corpus; the scenario sweep is the next prerequisite for model
work.

Not started: INT8 quantisation, the safety governor and redundant fan, Arm64
benchmarks, and deployment reproducibility packaging.

## Core objectives

1. **System awareness:** combine telemetry from independent actuators, sensors
   and the ventilation topology into one coherent system state.
2. **Fault diagnosis:** distinguish normal demand from fan degradation,
   blockage and invalid sensor data.
3. **Safe orchestration:** coordinate healthy actuators without exceeding
   declared command limits; hand control back when evidence is invalid.
4. **Measurable benefit:** prove that orchestration improves airflow recovery
   and environmental outcomes over isolated local control.
5. **Arm optimisation:** demonstrate reproducible, efficient local inference on
   Arm64 hardware.

## What must be implemented

### 1. Actuator and plant model

- Preserve local CO₂-driven controllers for each zone. (done)
- Separate commanded output, actual output, health, airflow and power. (done:
  command vs measured position vs requested/delivered airflow are separate;
  health stays hidden by design — see §3)
- Add shared ducts or capacity constraints so actuator behaviour has
  system-wide consequences. (mechanism done and unit-tested; a contention
  scenario arrives with the governor slice — current habitats never saturate
  the shared bay)
- Add a healthy redundant fan or alternative airflow path. (pending — changes
  scenario geometry; lands with the governor slice)

### 2. Deterministic scenarios

- Nominal operation. (done: `standard_habitat`, `high_demand_healthy`)
- Gradual primary-fan degradation. (done: `primary_fan_degradation`)
- Blocked airflow path. (done: `blocked_path`)
- Invalid or frozen sensor. (done: `frozen_sensor`, with a lab demand step so
  the held reading diverges from reality — a frozen sensor at saturated
  steady state is otherwise invisible)
- Shared-capacity contention. (pending — governor slice)
- Identical fault runs with orchestration enabled and disabled. (pending —
  requires the governor)

### 3. Telemetry and topology

- Record CO₂, actuator command and position, requested/delivered/residual
  airflow and power. (done) Health and fault truth are deliberately NOT
  recorded: they stay behind the telemetry boundary so model-facing data
  contains only what honest deployment telemetry would show. Sensor validity
  is handled as a fault class (`frozen_sensor`) and later as a governor
  `HAND_BACK` condition, not as a telemetry field.
- Airflow is recorded as metered duct flow. Declaring flow sensing is a
  deliberate abstraction, kept because the requested/delivered residual is
  the primary fault observable; a deployment-facing version would need
  declared flow sensors or inferred flow.
- Derive command-tracking residuals and rolling trends. (residuals done;
  rolling trends belong to the model lane)
- Map each actuator to the ducts and zones it affects. (done — the validated
  scenario graph is the topology)
- Keep every run seeded, replayable and testable. (done — byte-identical
  replay under test)

### 4. AI diagnosis

- Generate labelled telemetry windows from the simulator. (done for corpus v1;
  for the corpus-v2 contract fixture, each row is schema-validated and bound to
  immutable family evidence for its split, role, onset and label, plus replayed
  model-input traces; generated manifests persist frozen Gate-1 metadata, family
  split counts and a canonical integrity hash.)
- Train a compact temporal classifier and compare it with the rule baseline.
  (done experimentally with IID and separately reported OOD stress families;
  split by family, never by window; current locked result favours the rule)
- Choose the architecture with quantisation in mind: prefer operations that
  survive INT8 cleanly over exotic layers, so quantisation is a design
  input, not an afterthought.
- Export FP32 and INT8 ONNX models. (FP32 done; INT8 pending)
- Report fault class, confidence, detection latency and false alarms. (done for
  FP32 and rule comparison)

### 5. Safety governor

- Require persistent model confidence before intervening.
- Select healthy redundant capacity using topology and current demand.
- Cap autonomous commands at 80%.
- Produce `HAND_BACK` for invalid telemetry, invalid inference or insufficient
  healthy capacity.
- Record the reason for every decision.

### 6. Evidence, optimisation story and presentation

The judges weight technical implementation most heavily, and the challenge
brief demands an explicit optimisation narrative: baseline, technical
changes, measured improvement on Arm, and why it matters. The plan for that:

- **Baseline vs optimised:** the FP32 ONNX classifier is the declared
  baseline; the INT8 quantised model is the optimisation. Report model size,
  latency, throughput, memory and detection-quality delta between them on a
  declared Arm64 target.
- **Declared Arm64 target (later):** Azure Arm64 (Ampere Altra, Dps v5 series,
  Ubuntu) remains the intended benchmark class. No Azure resource is provisioned
  during Gates 0-2 or without Ben's explicit approval. After a frozen model and
  benchmark harness exist, record the selected VM size, CPU, OS and runtime
  versions beside raw results so the numbers are reproducible.
- **Reusable artifacts, not claims:** a reusable benchmark runner script,
  raw benchmark results committed to the repo, and migration/optimisation
  notes another developer could follow. Tooling and lessons are scored.
- **README optimisation section:** baseline, changes made, results, and why
  they matter, written for a judge who reads the repo before the pitch.
- Plot fault injection, detection, commands, airflow and CO₂ recovery together.
- Measure recovery time and CO₂ exposure with and without orchestration.
- Provide one-command reproduction and a demonstration video under three
  minutes.

## Completion criteria

The project is complete when the nominal run produces no intervention, each
declared fault is detected and classified, invalid data causes `HAND_BACK`, the
backup command remains within its bound, and the orchestrated run measurably
outperforms the same untreated fault. All claims must be backed by replay
traces, automated tests and Arm benchmark evidence.

## Scope guard

Do not prioritise a dashboard, database, cloud service, detailed fluid
dynamics, physical sensors or general building-management features until the
closed-loop orchestration demonstration and Arm measurements are complete.
