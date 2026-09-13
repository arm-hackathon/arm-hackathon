# AEOLUS · Habitat simulation and environmental decision support

Explore how a simulated habitat responds to changing conditions, equipment faults
and proposed interventions—and test whether a model helps make better decisions.

AEOLUS stands for **Airflow and Environmental Observation Laboratory for
User-defined Scenarios**.

## Why this exists

A controller does not get to see the full physical state of its environment. It
receives sensor readings that can be noisy, delayed or wrong, and must work within
limits on equipment and resources. A forecast may look accurate without helping
the controller choose a useful action.

AEOLUS provides a repeatable simulated environment for investigating that gap:
what can we infer from the available observations, what would happen under a
proposed intervention, and does the achieved action improve the outcome compared
with a conventional controller?

**Research simulation only.** This is not a spacecraft, life-support system,
building controller, calibrated CFD model or flight-validated digital twin. It
must not control physical equipment. Simulation results do not establish physical
performance, certification or deployment readiness.

## What works, and what is still research

- **Simulation foundation:** Habitat Plant V2 models an eight-zone environment
  with gas inventories, temperature, pressure, airflow, resource use, equipment
  responses and physical/sensor faults. Its reduced-order equations and physical
  parameters are explicit research assumptions, not a complete habitat model.
- **Deterministic control and replay:** the Habitat Management Computer (HMC)
  checks proposed actions, owns final commands and plant steps, and supports
  replay of recorded execution. Models advise; they do not command the plant.
- **Runnable demonstrations:** a guided terminal tour, a verified local forecast
  report, and a trained NumPy forecaster demonstrate bounded parts of the system.
  A separate browser fixture explorer has no command authority.
- **Learned decision support:** several model studies and evidence records exist,
  but they cover different contracts and experiments—not one generally validated
  autonomous controller. The newer BDM-v1 research stack is separate from merged
  `main`; it reports a failed model-promotion gate and no admitted closed-loop
  proposals. See [research direction](#research-direction) and
  [results and limitations](#results-and-limitations).

The repository also retains an older abstract-unit simulator and its recovery
experiments. **Start with Habitat Plant V2 for the current research direction.**
The legacy simulator is a separate lineage, not an alternative implementation of
V2's physical model.

## Start here

Use a source checkout of `main`, [uv](https://docs.astral.sh/uv/) and Python 3.11
for the locked development workflow. Package metadata permits Python 3.10 or
newer; the documented workflow uses 3.11.

```bash
git clone https://github.com/arm-hackathon/arm-hackathon.git
cd arm-hackathon
uv sync --locked --python 3.11 --extra dev
```

### 1. Take the guided tour

```bash
uv run --locked --python 3.11 --extra dev python scripts/aeolus_tour.py
```

The interactive tour explains the architecture and offers a trained-forecaster
run, a recorded paired-experiment replay and independent receipt verification.
Recorded experiments are historical evidence, not new training runs.

### 2. Run the simulation directly

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_actuator_feedback.json \
  out/habitat-v2-actuator-feedback.jsonl
```

This checked-in scenario includes equipment feedback and faults. The command
validates the scenario, runs the deterministic plant and writes a validated JSONL
trace: one JSON object per line. Generated traces belong under ignored `out/`
paths. The runner refuses to overwrite an existing output; choose a new filename
for a subsequent run.

For a smaller starting example, use
[`scenarios/habitat_v2_reference.json`](scenarios/habitat_v2_reference.json).
Other checked-in examples cover
[operating modes](scenarios/habitat_v2_operating_modes.json),
the [air network](scenarios/habitat_v2_air_network.json) and
[compound faults](scenarios/habitat_v2_compound_faults.json).
Scenario versions are separate contracts; newer fields do not silently change
older scenarios.

### 3. Inspect a verified forecast report

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_habitat_v2_forecast_report.py
```

The command generates a fresh local forecast receipt, independently verifies a
deterministic HMC replay, then prints the local `file:` URL for an HTML report.
The report embeds generated data; it does not perform model inference in the
browser. Replay verification checks the recorded software execution—it does not
qualify a learned model or validate the physics against hardware.

To run the historical trained MLP forecaster instead:

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_habitat_v2_mlp_forecast.py
```

It forecasts catalogue actions and compares predictions with simulated outcomes
for an operator-selected action under HMC authority. See its
[model card](MODEL_CARD.md) before interpreting the results.

For a separate offline fixture explorer, open
[`demo/browser-simulator/index.html`](demo/browser-simulator/index.html) locally.
It makes no network requests and retains `actionAuthority: "none"`; it is not a
browser implementation of the Python model or controller.

## How the system fits together

There are two different meanings of **model** here:

- The **physical simulation model** defines how the habitat changes: gas,
  energy, airflow, equipment and resource accounting.
- A **forecasting model** uses permitted observations and a candidate action to
  predict what may happen. It does not get hidden fault labels or future truth.

The core relationship is:

```text
Scenario: layout, initial conditions, loads and fault schedules
                            |
                            v
                    Simulated plant
                            |
                      Sensor readings
                            |
                 Optional forecast/adviser
                            |
                     Proposed action
                            |
                   Deterministic HMC
                            |
                 Allowed final command
                            |
                    Next plant state
                            |
                Recorded trace and replay
```

The HMC can operate without a learned adviser. In the advisory path, requested,
accepted and achieved actions are distinct: a proposal can be rejected or
modified, and equipment may not achieve the requested setting. Evaluation must
check the resulting trajectory, not count a prediction or proposal as a useful
intervention. Evaluator-only simulator truth remains separate from model inputs.

### Read the code in this order

| Question | Starting point |
| --- | --- |
| What world are we simulating? | [`scenarios/`](scenarios/) and [`scenario.py`](src/aeolus/habitat_v2/scenario.py) |
| How does its state change? | [`state.py`](src/aeolus/habitat_v2/state.py), [`physics.py`](src/aeolus/habitat_v2/physics.py), [`air_network.py`](src/aeolus/habitat_v2/air_network.py) |
| What can the controller observe? | [`telemetry.py`](src/aeolus/habitat_v2/telemetry.py) and [`instrumentation.py`](src/aeolus/habitat_v2/instrumentation.py) |
| Who decides what is allowed? | [`hmc.py`](src/aeolus/habitat_v2/hmc.py), [`proposal.py`](src/aeolus/habitat_v2/proposal.py) and [`safety.py`](src/aeolus/habitat_v2/safety.py) |
| Where do forecasts fit? | [`forecast/`](src/aeolus/habitat_v2/forecast/) and [model documentation](MODEL_CARD.md) |
| How is a run executed and checked? | [`runner.py`](src/aeolus/habitat_v2/runner.py), [`trace.py`](src/aeolus/habitat_v2/trace.py) and [`tests/habitat_v2/`](tests/habitat_v2/) |

The wider repository separates [active contracts](contracts/),
[research and evidence documentation](docs/), [reproduction scripts](scripts/),
[historical artifacts](artifacts/) and ignored generated output (`out/`).
See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions and
[AGENTS.md](AGENTS.md) for source, snapshot and protected-data boundaries.

## Results and limitations

Results belong to their named experiment and source version. They are not
interchangeable claims about the current system.

- **Legacy deterministic recovery:** a frozen policy passed its one-time blind
  simulation safety and physical-benefit gates. This was deterministic recovery,
  not evidence of learned-control advantage. The
  [final verification record](docs/evidence/recovery-final-verification-result.md)
  preserves the measured results, source identity and limitations; the
  [recovery acceptance record](docs/recovery-protocol-acceptance.md) retains
  earlier negative development evidence.
- **Historical action-aware MLP:** the supported NumPy demo is runnable, but the
  full historical training and paired closed-loop campaign cannot be completely
  reconstructed from current `main`. Consult the
  [model card](MODEL_CARD.md), [corpus datasheet](CORPUS_DATASHEET.md) and
  [historical evidence index](docs/evidence/closed-loop-advisory-historical-index.md)
  for the reported findings and missing provenance.
- **Forecast and action-risk studies:** the
  [dropout capability card](docs/evidence/issue-53-dropout-card.md) covers a bounded
  forecast-only contract. The [Issue #56 final protocol](docs/issue-56-v4-model-final-protocol.md)
  records a favourable V4-versus-V3 development comparison, but does not by itself
  isolate the learned component's contribution. Neither result gives a model
  actuator authority.
- **Operational observability:** the
  [qualification record](docs/evidence/habitat-v2-operational-observability-qualification.md)
  documents a bounded software contract and its reproducible packet—not hardware
  qualification.
- **Newer BDM-v1 work:** the [roadmap](ROADMAP.md#the-latest-negative-results-change-the-next-step)
  links the reported negative development results and unmerged research stack.
  These are not new measurements from this README or evidence of a successful
  integrated learned controller.

For the safety argument and engineering decisions, see
[SAFETY_CASE.md](SAFETY_CASE.md) and [DESIGN_TRADEOFFS.md](DESIGN_TRADEOFFS.md).
Historical Arm benchmark measurements concern specific artifacts and runners;
they do not establish physical-board deployment, energy performance or model
qualification. Version history lives in [CHANGELOG.md](CHANGELOG.md), not in the
getting-started path.

## Research direction

The goal is environmental decision support under incomplete observations:

```text
observe -> infer state -> predict consequences -> rank -> constrain -> act -> verify
```

This is the intended research loop, not a claim that every stage is complete.
The immediate proposed milestone is a small, meaningful decision problem:
different permissible achieved actions must produce different outcomes, the
available observations must support a useful choice, and a competent conventional
method must provide a baseline. Some cases should correctly require no action or
abstention. A larger neural network is not a substitute for that evidence.

The [research roadmap](ROADMAP.md) is the short planning entry point. The
[detailed development programme](docs/plans/aeolus-development-programme.md)
contains the proposed architecture, team split, dependencies and acceptance
criteria, subject to team review. Follow
[GitHub issues](https://github.com/arm-hackathon/arm-hackathon/issues) and
[open pull requests](https://github.com/arm-hackathon/arm-hackathon/pulls) for work
in progress; an open research branch is not a capability merged into `main`.
[PLAN.md](PLAN.md) is historical context, not the current roadmap.

## Development checks

For changes to Python behaviour, use the locked source-checkout checks:

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
uv run --locked --python 3.11 --extra dev python -m compileall -q src tests scripts
uv lock --check
git diff --check
```

For contract-specific checks, packaging, installed-module usage and historical
reproduction commands, follow [CONTRIBUTING.md](CONTRIBUTING.md),
[AGENTS.md](AGENTS.md) and the linked evidence records. Historical source-pinned
experiments must not be rerun as if they were current defaults. Protected
validation/final data requires separate explicit authorisation.

## License

[MIT](LICENSE).
