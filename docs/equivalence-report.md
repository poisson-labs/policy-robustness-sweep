# G3 Harness Equivalence Report

**Gate:** spec §4 G3 — at zero perturbation, our evaluation harness must reproduce the
policy's nominal Playground eval performance within noise, with perturbations entering
through physics only. **Result: PASSED** (2026-08-13).

All numbers measured; source: `equivalence-report.json` from Modal run against checkpoint
`converged` of training run `20260813T155006Z`, mirrored at
`docs/measurements/2026-08-13-g3-equivalence-report.json` and validated in CI by
`tests/test_equivalence_report.py`. Produced by `sweep/equivalence.py`.

## The harness under test

`sweep/injected_env.py::PushInjectionWrapper` wraps the Playground Go1 env and writes the
push force into `data.xfrc_applied` on the torso body every step (world-frame vector,
constant over the half-open 0.5 s window, direction fixed at push-onset heading); floor
friction enters as an mjx model patch. The observation/action contract is never touched.
At the nominal world (`WorldConfig()` — friction `None` = model untouched, push 0%BW),
the wrapper still executes the full injection code path, writing zero force.

Model facts (recorded per spec §5): torso body `trunk`, robot mass **12.743 kg**
(subtree mass at trunk), gravity **9.81 m/s²** — so 100% bodyweight = 125.0 N
(measured: report `model` block; N-value derived from those two measured quantities).

## Check 1 — per-step physics equivalence (bitwise): PASSED

Protocol: a native-env rollout records (state, action) pairs; the wrapper is then stepped
from each **native** state with the **same** action, isolating single-step physics from
closed-loop chaos amplification.

- Steps compared: 500
- Max |Δqpos| across all steps: **0.0**
- Max |Δqvel| across all steps: **0.0** (tolerance 1e-9)

The injection layer at identity is bitwise physics-identical to the native environment.
This result was reproduced in three consecutive gate runs.

## Check 2 — statistical reproduction of nominal performance: PASSED

Protocol: brax's own `Evaluator` (stochastic policy, matching training's eval protocol:
128 eval envs, episode length 1000), 5 independent repeats through the wrapped harness,
compared to the training run's final eval. Pre-registered gate:
|pooled − training| ≤ 3·√(SE²_train + SE²_ours).

- Training final eval reward: 28.269 (std 2.367, n=128)
- Harness repeats: 28.420, 28.668, 29.080, 28.702, 28.794 → pooled **28.749**
- |Δ| = **0.481** ≤ gate **0.683** → within noise
- Episode lengths: 1000/1000 in both native and injected paths

The statistical gate passed in all four gate runs executed during this session.

## Finding: the simulator disagrees with itself at identical seeds

The gate's most interesting output is not the pass — it is a measured fact about GPU
physics simulation that reshapes how every downstream number in this project should be
read. **Two runs of the identical program with the identical PRNG seed do not produce
identical results on GPU.** XLA schedules floating-point reductions nondeterministically;
the per-operation differences are at the last-bit level, but 1000 steps of chaotic
contact dynamics amplify them into episode-reward differences of up to 3.7e-3 relative
(observed range 2.3e-4 – 3.7e-3 across identical-seed run pairs, including
native-vs-native).

Two design choices this measurement retroactively grounds:

1. **The 16-seed statistical treatment (spec §6).** Individual rollouts were never
   trustworthy objects — same-hardware, same-seed recomputation already varies. Per-cell
   distributions with censoring-aware estimates are the only defensible currency, and
   now that is measured fact rather than posture.
2. **The reproducibility statement (spec §6).** The spec anticipated "cross-hardware
   claims are statistical, not bitwise." Measurement shows this is true *on the same
   hardware*: the pinned image reproduces the surface statistically, and bitwise claims
   are not made at any level.

## Diagnostic (not gated) — same-seed evaluator comparison

Two invocations of brax's Evaluator with the *same* PRNG key — one native, one through
the wrapper — differ at the metric level by 2.3e-4 … 3.7e-3 relative across run pairs.
This is **not** a harness effect: native-vs-native across runs with identical seeds
varies comparably (e.g. 28.8069 vs 28.8105 in two otherwise identical runs). Cause: GPU
XLA reduction nondeterminism amplified by ~1000 steps of chaotic dynamics. Gating on
this comparison would be a flaky test by construction; the bitwise replay check above is
the correct physics-equivalence instrument. Full details in NOTES.md (nondeterminism
catalog) and DEVLOG Session 7.

**Product consequence (recorded for M2):** same config + seed does not guarantee
bitwise-identical GPU recomputation; shareable-world determinism is delivered by the
probe cache (the first rollout's `.rrd` is the canonical replay for its cache key), not
by recompute determinism.

## Scope

Equivalence is established for the `converged` checkpoint on the nominal world with this
image's pins (jax 0.9.2, mujoco/mjx 3.11.0, brax 0.14.2, playground 0.2.0, A100-80GB).
If the stretch checkpoint-selector ships, G3 must be re-run per exposed checkpoint
(spec §2). The solver-sensitivity slice (spec §6) is separate G4-adjacent work.
