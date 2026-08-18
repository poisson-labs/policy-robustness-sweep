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
| baseline (re-run) | 0.004 | 1 | 86.67 | +6.67 = one observed run-to-run drift (n=1 row, historical) |
| half_dt | 0.002 | 1 | 93.33 | +6.66 vs baseline |
| double_iters | 0.004 | 2 | 103.33 | +16.66 vs baseline |

Wall per setting ≈ 9–11 s (measured). Cost: MEASURED_TBD (dashboard, app
opw-solver-slice).

## Reading

- **One observed run-to-run drift of 6.67%BW** (μ=0.50 row): the baseline re-ran the
  sweep's exact configuration and seeds, and the interpolated boundary still moved —
  consistent with the measured GPU nondeterminism (equivalence report) amplified through
  chaotic rollouts. HISTORICAL n=1 DATUM: this figure was briefly used as an informal
  "≈7%BW noise floor"; that role is RETIRED (DEVLOG Session 23). Boundary uncertainty is
  per-row and estimated by the seed bootstrap — median row-wise 95% CI width 16.29%BW,
  range 4.4-32.5%BW (docs/measurements/2026-08-15-boundary-uncertainty.json).
- **Timestep halving: +6.66%BW — inside the μ=0.50 row's bootstrap 95% CI
  (75.0-91.7%BW, width 16.7).** No detectable integrator artifact from dt.
- **Solver iterations 1 → 2: +16.7%BW** — at the edge of that row's CI width; a real but
  modest solver conditioning of the absolute boundary position (~1.5 cells), with
  visibly noisier survival at high pushes (isolated survivals at 140–160%BW).

## Post-facing statement (§6/§8)

The boundary's absolute position carries row-dependent statistical uncertainty (median
95% CI width 16.3%BW ≈ 1.6 cells; 4.4%BW at μ=1.0 up to 32.5%BW on ice) plus a solver
sensitivity of the same order; the surface's structure — the diagonal transition band
and the ice cliff — persists across all tested settings. Reported either way, per spec: this is a
disclosed property of the instrument, not a footnote.
