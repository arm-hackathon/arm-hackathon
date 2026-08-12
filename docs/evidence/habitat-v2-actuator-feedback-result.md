# Habitat V2 actuator-feedback verification result

Date: 2026-08-12
Status: exact candidate commit pending
Base: `42e85e23bf065203b3d5b68d4b7398300bbb807a`
Version impact: minor
Resulting version: `0.7.0`

## Implemented boundary

The V5 slice adds:

- closed V5 scenario and trace dispatch with actuator-feedback identity
- requested, achieved, fault-effective and measured actuator layers
- rate-limited achieved cooling and oxygen state
- fan, scrubber, condenser, cooling and oxygen effectiveness semantics
- deterministic operational instrumentation and feedback-sensor bias/stuck faults
- external current-command stepping with a canonical command digest
- an eight-zone checked-in V5 scenario
- trace validation and deterministic replay binding for the actuator receipt and operational feedback

The measured fan-speed channel now follows the fault-effective physical fan response rather than copying the pre-fault achieved setpoint. The achieved value remains separate in `actual_action` and the actuator receipt.

## Verification

All commands used Python 3.11 through the locked uv environment unless stated otherwise.

- focused actuator-feedback suite: `27 passed`
- Habitat V2 suite: `186 passed`
- full repository suite: `657 passed`
- Ruff: passed
- `compileall` for `src` and `tests`: passed
- `uv lock --check`: passed
- `git diff --check`: passed
- package version test: passed

## Frozen legacy replay check

Canonical traces regenerated from the base snapshot and the candidate match byte-for-byte:

- V1 reference: `a94b098cf8707cde6383319be913032de53053d033fe6a7d2f0a07efad6260fb`
- V2 operating modes: `3264c0c1dad48fc9136ffe0470886c5de51ec48d1cd4ec5bc9265d2a01f07133`
- V3 air network: `dd3b3a579f5eaa8b08b0ffa5a230f5ef833f39233dcefc07a12e2ad4d6b3bd8d`
- V4 compound faults: `7151a62b5db6c001d4131d1711c53a63f2fc3d57444b46c823f1c1bda70e0ded`

## V5 replay

The checked-in V5 scenario replayed twice with identical bytes:

- rows: `5`
- final step: `4`
- zones: `8`
- trace SHA-256: `bda0989c01d8446c1607f8207256f349ce33490b091f043ce54598f69db4fd54`
- active-fault counts: `0,6,7,5,0`
- actuator receipt sections: fan, dampers, scrubber, condenser, cooling and oxygen

At completed step 1, the fan layers were:

- requested: `0.72`
- achieved: `0.60`
- fault-effective: `0.45`
- measured after deterministic feedback noise: `0.44966259176259604`

## Package evidence

The `0.7.0` wheel and source distribution built successfully. A clean Python 3.11 virtual environment installed the wheel from local bytes, imported `aeolus` from `site-packages`, reported version `0.7.0`, replayed the checked-in V5 scenario and validated the resulting trace.

The exact package and installed-trace checksums for the immutable reviewed candidate are stored with the external evidence archive rather than embedded here, so rebuilding nondeterministic archive containers does not silently stale this tracked document.

Evidence directory: `C:/Users/Nxiss/AppData/Local/hermes/cache/aeolus-actuator-feedback-evidence/`

## Claims boundary

This evidence qualifies deterministic software behaviour on local x86 Windows/Python 3.11 only. It is not native Arm64 evidence, hardware validation, CFD, flight software or certification. No HMC, learned model, training, corpus generation, quantisation, tag, release or merge was performed.
