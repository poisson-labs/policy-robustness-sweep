# Solver-sensitivity slice — measurements (spec §6, M1-02)

Source: `solver-slice-report.json` from run `solver-slice-20260814T153024Z` (mirrored
here), produced by `sweep/solver_slice.py`. Slice: the μ = 0.50 row (20 push values ×
16 seeds = 320 rollouts per setting), sweep-identical seed scheme, frozen §5
classification. Boundary = linear interpolation of the S = 0.5 crossing on the push
axis.

## Results

| setting | sim_dt | solver iters | boundary (%BW) | vs baseline |
|---|---|---|---|---|
| sweep row (original run, identical config) | 0.004 | 1 | 80.00 | — |
| baseline (re-run) | 0.004 | 1 | 86.67 | +6.67 = run-to-run noise floor |
| half_dt | 0.002 | 1 | 93.33 | +6.66 vs baseline |
| double_iters | 0.004 | 2 | 103.33 | +16.66 vs baseline |

Wall per setting ≈ 9–11 s (measured). Cost: MEASURED_TBD (dashboard, app
opw-solver-slice).

## Reading

- **Run-to-run noise floor ≈ 7%BW** (~2/3 of a grid cell): the baseline re-ran the
  sweep's exact configuration and seeds, and the interpolated boundary still moved
  6.67%BW — consistent with the measured GPU nondeterminism (equivalence report) being
  amplified through chaotic rollouts. Boundary positions are statistical objects.
- **Timestep halving: displacement within the noise floor.** No detectable integrator
  artifact from dt at this resolution.
- **Solver iterations 1 → 2: +16.7%BW, ≈ 2.5× the noise floor** — a real, modest solver
  conditioning of the absolute boundary position (~1.5 cells), with visibly noisier
  survival at high pushes (isolated survivals at 140–160%BW).

## Post-facing statement (§6/§8)

The boundary's absolute position carries ±1–2 grid cells of combined solver/noise
uncertainty; the surface's structure — the diagonal transition band and the ice cliff —
persists across all tested settings. Reported either way, per spec: this is a
disclosed property of the instrument, not a footnote.
