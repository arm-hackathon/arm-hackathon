# Corpus v2 Observable-Label Contract Implementation Plan

> **For Hermes:** Use `subagent-driven-development` only if execution is delegated. Implement each task in strict TDD order: one failing behavioural test, confirm RED, minimum implementation, confirm GREEN, then refactor while green.

**Goal:** Replace corpus-v1 labels based on hidden fault schedules with a topology-bound, family-split corpus-v2 contract whose labels and detection latency begin only when a fault changes the model's frozen `model_input_v1` observable surface.

**Architecture:** A strict scenario-family manifest binds one healthy reference and one single-fault scenario to a train/validation/test split and the existing Gate-1 selector/topology hashes. The observable-onset resolver replays both scenarios and compares only the corresponding `float32[24]` `model_input_v1` tensors. Corpus generation persists that evidence and labels windows as nominal, fault-class, or `excluded_transition`; evaluation scores only the first two categories and measures latency from observable onset.

**Tech Stack:** Python 3.11, dataclasses, standard-library JSON/SHA-256, NumPy, pytest, Ruff, existing `aeolus.config`, `aeolus.scenario`, and `aeolus.model_input` contracts.

---

## Decisions already accepted

1. **Independence unit:** scenario family, not window. All variants/replicas derived from a family share exactly one split; an exact reference/fault pair may appear only once. No random window split is allowed.
2. **Label evidence:** the offline label builder may compare a fault replay to its paired healthy reference. The trained/inference model receives only `model_input_v1`; it never receives the reference replay, fault profile, start tick, effectiveness, target, seed, or source-noise state.
3. **Observable onset:** first measured tick where the fault and reference `model_input_v1` vectors differ. The resolver must compare exactly the Gate-1 `float32[24]` tensors.
4. **Boundary windows:** a fixed-width window that spans the observable-onset boundary is retained as `excluded_transition`, but excluded from supervised training/scored accuracy. It remains available for audit and latency work.
5. **Fault scope:** corpus v2 supports one fault profile per fault scenario. Multi-fault scenarios fail closed.
6. **Current evidence check:** on the existing deterministic paired scenarios the expected observable onsets are degradation=21, blocked path=30, frozen sensor=31. Their hidden profile starts are 20, 30, and 30 respectively; the frozen-sensor one-tick difference proves why schedule truth cannot define the score boundary.

## Non-goals

- Generating the large parameter sweep or corpus-v2 training volume.
- Training a classifier, exporting ONNX, quantisation, inference, governor logic, backup capacity, Azure provisioning, or Arm claims.
- Modifying the frozen Gate-1 `model_input_v1` selector or telemetry allowlist.
- Adding magic onset thresholds. If future family replicas intentionally vary reference noise, this exact-pair design must reject them until a separately specified effect-size/persistence contract is accepted.

## Proposed family manifest

Create committed `scenarios/families.json`. It is strict JSON with no unknown fields and canonical JSON hashing.

```json
{
  "schema_version": "aeolus_family_manifest_v1",
  "model_input_version": "model_input_v1",
  "selector_sha256": "<64 lowercase hex characters>",
  "topology_sha256": "<64 lowercase hex characters>",
  "families": [
    {
      "family_id": "degradation-primary-fan-v1",
      "split": "train",
      "fault_class": "gradual_primary_fan_degradation",
      "reference_scenario": "high_demand_healthy.json",
      "fault_scenario": "primary_fan_degradation.json"
    }
  ]
}
```

The manifest binds a single validated topology and model-input contract. Each pair must be schema-v7, use equivalent topology/selector hashes, share identical non-fault scenario configuration, use a fault-free reference, and contain exactly one fault in the paired scenario matching `fault_class`.

## Corpus-v2 row and manifest additions

A corpus-v2 row retains `window_index`, ticks, and `features`, adding:

```json
{
  "family_id": "degradation-primary-fan-v1",
  "split": "train",
  "label": "nominal | gradual_primary_fan_degradation | blocked_path | frozen_sensor | excluded_transition",
  "observable_onset_tick": 21,
  "model_input_version": "model_input_v1",
  "selector_sha256": "...",
  "topology_sha256": "..."
}
```

The generated corpus manifest must include `manifest_sha256`: SHA-256 of its canonical compact UTF-8 JSON with the `manifest_sha256` field omitted. It must also include source scenario SHA-256 values, family counts by split, label counts, excluded-transition counts, and frozen Gate-1 metadata. Evaluation derives each complete window inventory from trusted replay lengths rather than accepting a row-provided count. Generated files remain under ignored `out/`; no generated JSONL is committed.

## Task 1: Add strict family-manifest contracts

**Objective:** Define the unit of independence and bind each pair to the frozen model-input contract before any corpus-v2 label is produced.

**Files:**
- Create: `src/aeolus/families.py`
- Create: `tests/test_families.py`
- Create: `scenarios/families.json`
- Modify: `PLAN.md`
- Modify: `docs/telemetry-contract.md`

**Step 1: Write failing tests**

```python
def test_load_family_manifest_is_canonical_and_topology_bound():
    manifest = load_family_manifest(FAMILY_MANIFEST_PATH)
    assert manifest.manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert manifest.contract == model_artifact_metadata(
        build_model_input_contract(load_scenario(HIGH_DEMAND_PATH))
    )


def test_manifest_rejects_duplicate_family_id_and_cross_split_reuse():
    with pytest.raises(ValueError, match="duplicate family_id"):
        parse_family_manifest(duplicate_id_document)
```

Add one behaviour-focused test each for unknown fields, missing/stale Gate-1 metadata, non-hex hashes, incompatible reference/fault topology, fault-bearing reference, a multi-fault paired scenario, mismatched class, and non-fault configuration drift.

**Step 2: Verify RED**

Run:

```bash
uv run --extra dev python -m pytest tests/test_families.py -q
```

Expected: FAIL because the family-manifest API does not yet exist.

**Step 3: Implement minimum API**

```python
@dataclass(frozen=True)
class ScenarioFamily:
    family_id: str
    split: Literal["train", "validation", "test"]
    fault_class: str
    reference_path: Path
    fault_path: Path

@dataclass(frozen=True)
class FamilyManifest:
    families: tuple[ScenarioFamily, ...]
    contract_metadata: Mapping[str, str]
    canonical_json: str
    manifest_sha256: str

def load_family_manifest(path: Path) -> FamilyManifest: ...
def parse_family_manifest(document: object, *, base_dir: Path) -> FamilyManifest: ...
```

Use strict field sets, sorted compact UTF-8 JSON, and SHA-256 exactly as Gate 1 does. Keep manifest parsing and validation in `families.py`; do not duplicate parsing in corpus or evaluation code.

**Step 4: Verify GREEN**

Run the focused test command again. Then run:

```bash
uv run --extra dev python -m pytest tests/test_families.py tests/test_model_input.py -q
uv run ruff check .
```

**Step 5: Commit checkpoint**

```bash
git add src/aeolus/families.py tests/test_families.py scenarios/families.json PLAN.md docs/telemetry-contract.md
git commit -m "feat: add corpus v2 family manifest contract"
```

## Task 2: Resolve observable onset from the frozen model input

**Objective:** Compute an auditable, observable fault onset without reading hidden schedule truth.

**Files:**
- Modify: `src/aeolus/families.py`
- Modify: `tests/test_families.py`

**Step 1: Write failing tests**

```python
@pytest.mark.parametrize(
    ("family_id", "expected_tick"),
    (
        ("degradation-primary-fan-v1", 21),
        ("blocked-path-v1", 30),
        ("frozen-sensor-v1", 31),
    ),
)
def test_observable_onset_compares_only_model_input_v1(family_id, expected_tick):
    family = load_family_manifest(FAMILY_MANIFEST_PATH).family(family_id)
    assert observable_onset(family).tick == expected_tick
```

Add tests for equal reference/fault traces, different trace lengths, non-finite input, incompatible selector/topology hashes, and an assertion that changing raw trace fields outside `model_input_v1` cannot advance onset.

**Step 2: Verify RED**

```bash
uv run --extra dev python -m pytest tests/test_families.py -q
```

Expected: FAIL because `observable_onset` does not exist.

**Step 3: Implement minimum API**

```python
@dataclass(frozen=True)
class ObservableOnset:
    family_id: str
    tick: int
    model_input_version: str
    selector_sha256: str
    topology_sha256: str
    reference_scenario_sha256: str
    fault_scenario_sha256: str

def observable_onset(family: ScenarioFamily) -> ObservableOnset:
    """Return the first paired tick whose model_input_v1 tensor differs."""
```

Replay the validated pair, build one `ModelInputContract`, call `model_input_v1` for each same-tick pair, and use `numpy.array_equal`. Do not inspect `fault_profiles` after manifest validation. If no difference occurs, fail closed with a message naming the family.

**Step 4: Verify GREEN**

```bash
uv run --extra dev python -m pytest tests/test_families.py tests/test_model_input.py -q
uv run ruff check .
```

**Step 5: Commit checkpoint**

```bash
git add src/aeolus/families.py tests/test_families.py
git commit -m "feat: define observable corpus onset"
```

## Task 3: Generate corpus-v2 rows from families

**Objective:** Replace schedule-truth labels with manifest-bound observable labels while preserving the feature leakage boundary.

**Files:**
- Modify: `src/aeolus/corpus.py`
- Modify: `tests/test_corpus.py`
- Modify: `docs/simulation-rules.md`

**Step 1: Write failing tests**

```python
def test_corpus_v2_marks_onset_straddling_window_excluded(tmp_path):
    manifest = generate_corpus_v2(FAMILY_MANIFEST_PATH, tmp_path)
    rows = read_jsonl(tmp_path / "corpus.jsonl")
    assert any(row["label"] == "excluded_transition" for row in rows)
    assert all(
        row["selector_sha256"] == manifest["selector_sha256"] for row in rows
    )
```

Add tests that pre-onset end ticks are nominal, post-onset windows receive the paired family class, every row carries family/split/onset/contract metadata, feature dictionaries are still exactly `model_feature_row` values, generated output is byte-identical across runs, malformed family manifests fail, and no forbidden truth field occurs in features or row metadata.

**Step 2: Verify RED**

```bash
uv run --extra dev python -m pytest tests/test_corpus.py -q
```

Expected: FAIL because corpus-v2 generation does not exist.

**Step 3: Implement minimum API**

```python
CORPUS_V2_VERSION = 2
EXCLUDED_TRANSITION_LABEL = "excluded_transition"

def generate_corpus_v2(manifest_path: Path, out_dir: Path, *, window: int = 10, stride: int = 5) -> dict: ...
def label_v2_window(*, start_tick: int, end_tick: int, onset_tick: int, fault_class: str) -> str: ...
```

A window is excluded only when `start_tick < onset_tick <= end_tick`; windows ending before onset are nominal, and windows starting at or after onset receive `fault_class`. Persist only family-level metadata needed to audit split and contract compatibility. Do not persist hidden schedule or target data into row features.

**Step 4: Verify GREEN**

```bash
uv run --extra dev python -m pytest tests/test_corpus.py tests/test_families.py -q
uv run ruff check .
```

**Step 5: Commit checkpoint**

```bash
git add src/aeolus/corpus.py tests/test_corpus.py docs/simulation-rules.md
git commit -m "feat: generate observable-labelled corpus v2"
```

## Task 4: Make evaluation score only observable evidence

**Objective:** Score corpus-v2 predictions without hidden fault starts and report excluded-transition evidence separately.

**Files:**
- Modify: `src/aeolus/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `README.md`
- Modify: `docs/simulation-rules.md`

**Step 1: Write failing tests**

```python
def test_evaluate_v2_excludes_transition_rows_from_accuracy_and_confusion():
    result = evaluate_v2(rows_with_one_transition, fake_labeller)
    assert result["scored_total"] == 2
    assert result["excluded_transition_total"] == 1
    assert result["accuracy"] == 1.0


def test_evaluate_v2_latency_starts_at_persisted_observable_onset():
    assert result["detection_latency_ticks"]["frozen_sensor"] == expected_latency
```

Add tests that stateful labellers reset at each scenario/family boundary, latency never rebuilds `fault_start_tick` from scenario profiles, malformed or mixed contract metadata fails closed, and corpus-v1 evaluation remains explicitly versioned historical output if retained.

**Step 2: Verify RED**

```bash
uv run --extra dev python -m pytest tests/test_evaluate.py -q
```

Expected: FAIL because corpus-v2 scoring does not exist.

**Step 3: Implement minimum API**

```python
def evaluate_v2(rows: list[dict], labeller: WindowLabeller) -> dict:
    """Score only nominal/fault rows and measure latency from observable onset."""
```

Exclude `excluded_transition` rows before totals, accuracy, per-class support, and confusion. Count them in a separate metric. Calculate first correct fault-class prediction minus the row/family's persisted observable onset. Validate every row shares one exact selector/topology contract before evaluating.

**Step 4: Verify GREEN**

```bash
uv run --extra dev python -m pytest tests/test_evaluate.py tests/test_corpus.py tests/test_families.py -q
uv run ruff check .
```

**Step 5: Commit checkpoint**

```bash
git add src/aeolus/evaluate.py tests/test_evaluate.py README.md docs/simulation-rules.md
git commit -m "feat: score corpus v2 from observable onset"
```

## Task 5: Final proof and Gate-2 acceptance record

**Objective:** Demonstrate the contract is deterministic, leakage-safe, and ready to constrain the future scenario sweep.

**Files:**
- Modify: `PLAN.md`
- Modify: `README.md`
- Modify: `docs/telemetry-contract.md`

**Step 1: Generate an ignored evidence artifact**

```bash
rm -rf out/corpus-v2-contract
PYTHONPATH=src uv run python -m aeolus.corpus \
  --v2 out/corpus-v2-contract scenarios/families.json
PYTHONPATH=src uv run python -m aeolus.evaluate \
  --v2 out/corpus-v2-contract/corpus.jsonl scenarios/families.json \
  --expected-family-manifest-sha256 828880e3257036ff2897a6cc2668c25b87734f8c57004ed36e62b2b6d66f6541 \
  --split test
```

Use the actual final v2 CLI/API documented by the preceding implementation. Generated JSONL stays under `out/` and remains untracked.

**Step 2: Verify exact contract properties**

```bash
uv run --extra dev python -m pytest -q
uv run ruff check .
git diff --check origin/main
git status --short
```

Record exact pass count, lint result, branch HEAD, manifest SHA-256, per-family observable onset values, corpus label counts, and transition-exclusion counts in the PR body later. Do not request review until Ben approves that body.

**Step 3: Commit documentation checkpoint**

```bash
git add PLAN.md README.md docs/telemetry-contract.md
git commit -m "docs: record corpus v2 contract evidence"
```

## Acceptance criteria

- Family split membership is strict, canonical, versioned, and auditable.
- No single family can appear across train/validation/test.
- Corpus-v2 labels and latency do not use declared profile start ticks.
- All present labels can be reproduced from paired `model_input_v1` values and persisted evidence metadata.
- `excluded_transition` rows are never included in trained/scored accuracy support.
- Any stale/missing selector or topology contract fails closed.
- Existing Gate-1 model-input, telemetry, replay, and rule-baseline tests stay green.
- The full test suite, Ruff, and `git diff --check` are clean.

## Review gate after this plan

Before starting Task 1, inspect the `families.json` example together and confirm that its current intended family-level splits are sensible. Before moving from Gate 2 to the scenario sweep, inspect the generated v2 manifest and verify observable onsets 21/30/31 from actual receipts. The scenario sweep begins only after these contracts are accepted.
