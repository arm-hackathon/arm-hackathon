# Habitat V2 air-network candidate receipt

Date: 2026-08-12
Branch: `ben/habitat-v2-air-network`
Base commit: `087b1e9a97bba7260876386363a3e76662b9d4f5b`
Version impact: `minor`
Package version: `0.5.0`
Publication status: local candidate only

## Scope

This candidate adds scenario-v3 and trace-v3 for a reduced-order multizone
recirculating air network. A variable-speed fan, shared pressure losses and one
motorised damper per zone determine the fan/system operating point and derived
per-zone airflow. It preserves scenario-v1 and scenario-v2 identities and frozen
reference replay bytes.

The checked-in reference habitat has eight notional zones and covers all four
operating-mode labels. Dimensions, resistances, loads and schedules are declared
research assumptions. This is not CFD, a NASA floor plan, a certified digital
twin, flight software or hardware validation.

## Verification

Repository suite:

```text
uv run --locked --python 3.11 --extra dev python -m pytest -q
595 passed in 118.86s
```

Static checks:

```text
uvx ruff@0.14.10 check .
All checks passed!

uv run --locked --python 3.11 --extra dev python -m compileall -q src tests
exit 0

git diff --check
exit 0
```

Package build:

```text
uv build
Successfully built dist/aeolus-0.5.0.tar.gz
Successfully built dist/aeolus-0.5.0-py3-none-any.whl
```

The build emitted the existing setuptools warning that the TOML-table form of
`project.license` is deprecated. It did not fail the build and is deferred to a
separate packaging-maintenance slice.

Artifact hashes:

- wheel: `6610f6bb7f67ad445b90630444204e10cd9fb8a89aadd306714e1844e1be5e95`
- source distribution: `9b77e023308ebe0d2aedc30bc0dc9d505b0099db8e584e09f54bd9d2c16ec882`

The wheel was installed into a clean Python 3.11 environment outside the source
checkout. `python -I` confirmed package metadata and runtime version `0.5.0`,
imported `aeolus.habitat_v2.air_network`, and ran the checked-in eight-zone
scenario twice. The emitted traces were byte-identical.

Installed trace receipt:

- rows: 5
- zones: 8
- final step: 4
- trace SHA-256: `dd3b3a579f5eaa8b08b0ffa5a230f5ef833f39233dcefc07a12e2ad4d6b3bd8d`
- maximum total airflow: `0.5407453977342523 m³/s`
- maximum fan electrical power: `273.8724855662156 W`
- maximum absolute species residual: `9.094947017729282e-13 mol`
- maximum absolute zone thermal residual: `2.0942752598784864e-07 J`
- maximum absolute system thermal residual: `6.511545507237315e-07 J`
- maximum absolute electrical residual: `1.4210854715202004e-14 Wh`
- maximum absolute fan/system residual: `1.4210854715202004e-13 Pa`
- maximum absolute fixed-density supply-return mass residual: `0.0 kg/s`

Final resource state:

- battery: `12130.558975377451 Wh`
- oxygen store: `499.802 mol`
- CO₂ sorbent: `1499.3472 mol`
- captured CO₂: `0.6527999999999999 mol`
- condensed water: `1.8143999999999998 mol`

Retained artifacts are outside the repository worktree under:

`C:/Users/Nxiss/AppData/Local/hermes/cache/aeolus-0.5.0-air-network/`

## Deferred

- deterministic fault primitives
- truth-versus-observed sensor telemetry and disagreement
- scenario families and provenance-bound corpus generation
- learned-model implementation or training
- browser viewer
- native Arm64 measurements
- remote push, PR, merge or release
