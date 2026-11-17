# Habitat V2 air network: closed-form operating point and batched layout

Date: 2026-11-13
Issue: [#91](https://github.com/arm-hackathon/arm-hackathon/issues/91)
Status: **research-only numerical candidate; not an authoritative solver**
Machine-readable contract: [`contracts/habitat_v2_air_network_batched_v1.json`](../../contracts/habitat_v2_air_network_batched_v1.json)
Modules: `src/aeolus/habitat_v2/air_network_analytic.py` (NumPy FP64),
`src/aeolus/habitat_v2/air_network_batched_torch.py` (optional PyTorch FP32/FP64)
Version impact: `minor` (declared; numeric bump, changelog, lock and
source-binding changes are left to the release owner)

This record documents the algebra, units, supported domain and batch layout of
the optional batched air-network kernel. It covers the existing model only:
one quadratic fan, one shared quadratic resistance and N parallel quadratic
branches, as specified in
[`2026-08-12-habitat-v2-air-network-contract.md`](../plans/2026-08-12-habitat-v2-air-network-contract.md).
It is not an arbitrary duct-network solver and it adds no physics.

`src/aeolus/habitat_v2/air_network.py` remains the authoritative solver. It is
byte-unchanged, and the plant, HMC, receipts and replay keep using its
100-iteration bisection. The kernel described here agrees with it only to the
declared tolerances, so its outputs must never be written into canonical
traces, receipts or hashes.

## Model being solved

For fan speed fraction `s` in `(0, 1]` the contract defines:

| Law | Equation | Units |
| --- | --- | --- |
| Damper area | `a_i = leak_i + (1 - leak_i) * position_i` | dimensionless |
| Branch resistance | `R_i = R_open_i / a_i^2 + R_return_i` | Pa s² m⁻⁶ |
| Branch flow | `q_i = sqrt(p_b / R_i)` | m³ s⁻¹ |
| Total flow | `Q = sum_i q_i` | m³ s⁻¹ |
| Shared loss | `p_s = R_shared * Q^2` | Pa |
| Fan curve | `P_fan = P0 * s^2 * max(0, 1 - (Q / (Q0 * s))^2)` | Pa |
| Operating point | `P_fan = p_s + p_b` | Pa |
| Air power | `W_air = P_fan * Q` | W |
| Electrical power | `W_el = W_air / efficiency` | W |
| Mass flow | `m_i = density * q_i` | kg s⁻¹ |

`P0` is the rated shutoff pressure, `Q0` the rated free delivery and `p_b` the
single pressure drop common to all parallel branches. At `s = 0` every output
is exactly zero.

## Closed form

All branches see the same pressure, so `q_i = sqrt(p_b) * g_i` with
`g_i = R_i^(-1/2)`. Summing gives `Q = sqrt(p_b) * G` where `G = sum_i g_i`,
so the parallel branches act as one resistance:

```
p_b = R_eq * Q^2          R_eq = 1 / G^2
```

Away from the clamp the fan curve is itself quadratic in `Q`:

```
P_fan = P0 * s^2 - k * Q^2          k = P0 / Q0^2
```

The fan-curve slope `k` has the units of a resistance and does not depend on
speed, because shutoff pressure scales with `s^2` and free delivery with `s`.
Setting fan pressure equal to system loss:

```
P0 * s^2 - k * Q^2 = (R_shared + R_eq) * Q^2
Q = s * sqrt(P0 / R_tot)             R_tot = k + R_shared + R_eq
```

and therefore:

```
q_i   = Q * g_i / G
p_b   = s^2 * P0 * R_eq / R_tot
p_s   = s^2 * P0 * R_shared / R_tot
P_fan = s^2 * P0 * (R_shared + R_eq) / R_tot
```

Three consequences are used by the implementation.

1. **In exact arithmetic the clamp is never active at the operating point.**
   `R_tot > k` because `R_eq > 0`, so `Q^2 = s^2 P0 / R_tot < s^2 P0 / k =
   (Q0 s)^2`. The root lies strictly below free delivery, which is why the
   quadratic form of the fan curve is the right one to solve. In floating
   point this holds only up to rounding: when `R_shared + R_eq` is below about
   one rounding of `k`, `R_tot` rounds to `k` and the computed flow can sit
   one rounding above free delivery. The fan-side residual therefore keeps the
   contract's `max(0, ...)` clamp.
2. **Speed scaling is exact.** Flow is linear in `s`, pressure quadratic and
   power cubic (the fan affinity laws). The kernel computes unit-speed
   quantities once and multiplies by `s`; it never divides by `s`, so tiny
   positive speeds need no clamp or special case. `s = 0` gives exact zeros
   through the same expressions.
3. **Every operation is well conditioned.** The closed form uses products,
   quotients, square roots and sums of non-negative terms only. Each output
   carries a few roundings relative to its own magnitude. The one exception is
   the residual, discussed below.

### Residual

`operating_point_residual_pa` is reported as the clamped fan curve evaluated at
the solved flow, `P0 s^2 max(0, 1 - (Q / (Q0 s))^2)`, minus `p_s + p_b`. The fan-curve
side is computed as `1 - (unit flow / Q0)^2`, again without dividing by speed.
It is a difference of nearly equal pressures, so it is a rounding residual
bounded relative to the shutoff pressure `P0 s^2`, not relative to its own
size. It is not a convergence measure: there is no iteration to converge.

The reported `fan_pressure_rise_pa` is the system-side value
`s^2 P0 (R_shared + R_eq) / R_tot`. The reviewed solver instead evaluates the
fan curve at its bisected flow. The two differ by the residual.

## Batch layout

The layout is the interface that later batched work builds on. It is frozen in
the contract's `layout` block and checked against the module at load time.

| Field | Shape | Unit |
| --- | --- | --- |
| `rated_free_delivery_m3_s` | `[B]` | m³ s⁻¹ |
| `rated_shutoff_pressure_pa` | `[B]` | Pa |
| `total_efficiency` | `[B]` | dimensionless |
| `shared_resistance_pa_s2_m6` | `[B]` | Pa s² m⁻⁶ |
| `air_density_kg_m3` | `[B]` | kg m⁻³ |
| `fan_speed_fraction` | `[B]` | dimensionless |
| `open_supply_resistance_pa_s2_m6` | `[B, N]` | Pa s² m⁻⁶ |
| `return_resistance_pa_s2_m6` | `[B, N]` | Pa s² m⁻⁶ |
| `damper_leak_fraction` | `[B, N]` | dimensionless |
| `damper_position_fraction` | `[B, N]` | dimensionless |

- Axis 0 indexes `B` independent worlds. No value is shared between worlds:
  parameters, states and commands are all per world, so a batch can mix
  different plants and different candidate actions.
- Axis 1 indexes `N` parallel branches in ascending `zone_id` order, the key
  order of the reviewed solver's result mappings. `1 <= N <= 16`, and `N` is
  fixed within a batch (no padding or masking).
- Host arrays are `float64`. Any other dtype is rejected.
- `shared_resistance_pa_s2_m6` is the already-summed trunk and filter
  resistance, and `open_supply_resistance_pa_s2_m6` already includes any
  branch-resistance fault multiplier, exactly as `physics._air_network_spec`
  builds them. `fan_speed_fraction` is the effective speed after any fan fault
  multiplier. Actuator slew, jams and fault schedules are not part of this
  kernel.

Outputs are `[B]` for `fan_pressure_rise_pa`, `shared_pressure_loss_pa`,
`branch_pressure_loss_pa`, `total_flow_m3_s`, `fan_air_power_w`,
`fan_electrical_power_w` and `operating_point_residual_pa`, and `[B, N]` for
`zone_flow_m3_s` and `zone_mass_flow_kg_s`. `branch_pressure_loss_pa` is one
value per world (the reviewed solver repeats it per zone). The reviewed
solver's `mass_balance_residual_kg_s` is identically zero and is not carried.

`batch_from_operating_points` packs reviewed-solver inputs
(`AirNetworkSpec`, speed, damper mapping) into this layout, and
`solve_air_network_analytic_scalar` returns the reviewed solver's
`AirNetworkResult` type for a single operating point.

## Supported domain

The kernel validates every input and fails closed with
`AirNetworkAnalyticError`. The supported domain is a closed box, deliberately a
strict subset of what the reviewed solver accepts (any positive finite value):

| Field | Supported range |
| --- | --- |
| `rated_free_delivery_m3_s` | `[1e-4, 1e4]` |
| `rated_shutoff_pressure_pa` | `[1e-2, 1e7]` |
| `total_efficiency` | `[1e-3, 1]` |
| `shared_resistance_pa_s2_m6` | `[0, 1e9]` |
| `air_density_kg_m3` | `[1e-3, 1e3]` |
| `fan_speed_fraction` | `[0, 1]` |
| `open_supply_resistance_pa_s2_m6` | `[1e-3, 1e9]` |
| `return_resistance_pa_s2_m6` | `[1e-3, 1e9]` |
| `damper_leak_fraction` | `[1e-5, 1]` |
| `damper_position_fraction` | `[0, 1]` |

Inside the box no intermediate overflows or divides by zero in FP64 or FP32:
the largest branch resistance is `1e9 / 1e-10 + 1e9 ≈ 1e19` and the largest
fan-curve slope is `1e7 / 1e-8 = 1e15`, both far below the FP32 maximum.
Nonfinite values, values outside the box, wrong dtypes, wrong shapes, `N`
outside `1..16` and mismatched world counts are all rejected, with the field
name and the first offending world in the message.

The box is a numerical envelope, not a statement that such plants are
physically meaningful. The checked-in fixtures occupy a small part of it.

## Numerical limitations, stated plainly

**Agreement is to tolerance, never bitwise, and tiny speeds differ by design.**
The candidate and the reviewed solver are different algorithms. Their outputs
agree within the declared tolerances and are not bit-identical. At tiny
positive speeds they deliberately differ: the candidate keeps flow linear in
speed, while the reviewed solver returns zero flow once `s^2` underflows. Only
a speed of exactly zero (`0.0` or `-0.0`) guarantees exact `+0.0` outputs from
both. Downstream code that branches on exact zero flow must test the commanded
speed, not rely on the two solvers agreeing bit for bit.

**Candidate, FP64.** Flow stays linear in speed for every positive `s`.
Pressure and power underflow gradually once their true value drops below the
smallest normal FP64 number; for fixture-scale plants that is near `s ≈ 1e-155`
for pressure, reaching exact zero near `s ≈ 1e-163`. This is the correctly
rounded result, not an error, but relative accuracy is lost in the subnormal
range. Scaling is ordered as `(s * x) * s` so that a representable pressure is
not degraded by a subnormal intermediate `s * s`.

**Candidate, FP32.** Inputs are validated in FP64 on the host and then cast, so
an FP32 tensor can differ from the validated value by one rounding. For
fixture-scale plants pressure loses relative accuracy below about `s ≈ 1e-20`
and reaches zero near `s ≈ 1e-24`; power does so near `1e-13` and `1e-16`.
FP64 still resolves all of these. The FP32 tolerance therefore carries an
absolute term of `1e-37` (in each variable's unit): outputs smaller than that
are not resolved in FP32.

**Reviewed bisection solver.** These are properties of the reference, recorded
by `test_reference_underflow_limitations_are_reported_not_hidden`, and are not
changed by this work:

- When `s^2` underflows (`s ≈ 1e-170`), the bisection bracket `[0, P0 s^2]`
  collapses and the solver returns zero flow for a running fan.
- When `Q0 * s` underflows to zero (`s = 5e-324`), it raises
  `ZeroDivisionError`.
- When `a_i^2` underflows (leak fraction near `1e-200` with a closed damper),
  it raises `ZeroDivisionError`. Such leak fractions are outside the
  candidate's supported domain and are rejected.

- **Unconverged bisection off the fixture band.** Where the branch pressure is
  below roughly `1e-18` of shutoff pressure (a fan-dominated plant with very
  open branches), 100 halvings of `[0, P0 s^2]` cannot resolve it. At one
  supported-domain corner at full speed the reviewed solver's fan pressure is
  off the exact root by `2.75e-11` of shutoff pressure and its own residual is
  `2.75e-4 Pa`, while the candidate matches the 80-digit root. This is a
  convergence limit, not an underflow, and is pinned by
  `test_unconverged_reference_bisection_off_the_fixture_band_is_reported`.

None of these is reachable from checked-in scenarios, where speeds, leaks and
resistances are of order one to `1e5`. They are the reason agreement with the
reviewed solver is declared for the fixture band only, and why the wider
supported domain is checked against an independent high-precision root
instead.

## How agreement is checked

Tolerances are declared per variable in the contract, in the form

```
abs(candidate - reference) <= absolute
                              + relative_to_reference * abs(reference)
                              + relative_to_scale * scale
```

where `scale` is the variable's physical scale in that world: `P0 s^2` for
pressures and the residual, `Q0 s` for flows, their product for power. A
nonfinite candidate always fails.

The contract's canonical SHA-256 is pinned in `air_network_analytic.py`, and
the loader refuses a contract that does not match it. Changing any tolerance,
range or wording therefore needs a deliberate edit of the pin, visible in
review. The loader also rejects duplicate keys, non-JSON constants, unknown or
missing fields and wrong types.

The contract was first committed before any benchmark was run. It was revised
once, after independent review and before any result was reported: the FP64
comparisons gained a `1e-300` absolute term to cover gradual underflow at
subnormal pressure scale, and two descriptions were corrected. No tolerance was
loosened in response to a measured result, and the earlier exploratory receipts
are not reported.

| Comparison | Case profile | Per-variable tolerance |
| --- | --- | --- |
| NumPy FP64 vs reviewed bisection solver | `fixture_band` | `1e-12` of scale plus `1e-300` absolute, every variable |
| NumPy FP64 vs independent 80-digit decimal root | `supported_domain_stress` | `1e-13` relative plus `1e-300` absolute; residual `1e-12` of scale |
| PyTorch FP64 vs NumPy FP64 | `supported_domain_stress` | `1e-13` relative plus `1e-300` absolute; residual `1e-12` of scale |
| PyTorch FP32 vs NumPy FP64 | `supported_domain_stress` | `1e-5` relative plus `1e-37` absolute; residual `1e-5` of scale |

The comparison against the reviewed solver is declared relative to scale
because bisection resolves branch pressure to rounding of the shutoff
pressure, not to rounding of each output's own magnitude.

Three legs that share no code with the candidate back the comparison:

1. **The reviewed solver itself**, on the fixture band and on every solver
   call issued by the plant while running all twelve checked-in air-network
   scenarios (faulted, slewing and fan-off states included).
2. **An independent high-precision root.** The test suite bisects the
   contract's implicit equation `P_fan(Q(p_b), s) - R_shared Q(p_b)^2 - p_b = 0`
   in 80-digit decimal arithmetic. It does not use the equivalent-resistance
   algebra above.
3. **Direct substitution into the physical laws**: each branch satisfies
   `p_b = R_i q_i^2`, branch flows sum to the total, the shared loss and fan
   curve hold, power and mass flow follow their definitions, flow scales by
   the affinity laws across twelve decades of speed, closing a damper or
   adding resistance never increases flow, and `N` identical branches collapse
   to one branch of resistance `R / N^2`.

## Synthetic cases

`generate_air_network_cases(profile, seed=..., batch_size=..., branch_count=...)`
draws inputs from a profile declared in the contract, which records every
range:

- `fixture_band`: uniform draws over the provenance-manifest valid ranges
  around the checked-in eight-zone fixture, with open supply resistance
  widened to the largest checked-in fault multiplier.
- `supported_domain_stress`: decade-uniform draws across the whole supported
  domain, with exact fan-off, exact zero shared resistance, closed and fully
  open dampers, speeds down to `1e-12` and strongly asymmetric branches. It is
  not a physically representative population.

The generator uses a NumPy `PCG64` stream seeded by the caller, only
`Generator.random` draws, exactly rounded decade constants and single IEEE
multiplications and additions. Its bytes therefore do not depend on a platform
math library. `air_network_batch_sha256` gives the input identity recorded in
receipts; one digest is pinned in the test suite.

## Optional PyTorch backend

`air_network_batched_torch` is plain eager tensor code with the same algebra.
PyTorch is not a declared or locked dependency: it is imported inside
functions, so importing the module, or any core module, never imports it. A
missing PyTorch, an unknown device name or a CUDA request without a CUDA
device raises `AirNetworkBackendUnavailableError`. There is no silent CPU
fallback. To use the backend, install a PyTorch build for the target platform
into the environment separately (for example `pip install torch` from the
appropriate PyTorch index). Declaring it as a project extra and locking it is
left to the release owner, because `pyproject.toml` and `uv.lock` are bound by
the reviewed source snapshot.

- `solve_air_network_torch(batch, precision=..., device=...)` validates on the
  host in FP64, copies to the device, solves and copies back.
- `to_torch_inputs` and `air_network_operating_point_torch` expose the
  device-resident kernel for callers that keep state on the device. The kernel
  itself does not validate; it must be fed validated inputs.

There is no CUDA extension, compilation step, JAX or Warp port.

## What this does not prove

- It does not replace or re-validate the authoritative solver, and approximate
  numerical agreement does not redefine any exact replay or hash contract.
- It is not CFD, a digital twin or a hardware model, and it does not extend
  the air network beyond the existing fixed topology.
- A kernel timing is not an end-to-end simulator speedup, a model improvement
  or a control benefit.
- No CUDA or DGX measurement exists yet; see the evidence card.
