# G4 full sweep — measurements (THE dataset)

Source: `sweep-manifest.json` from sweep run `sweep-20260813T175452Z` against checkpoint
`converged` of run `20260813T155006Z` (mirrored here as
`2026-08-13-g4-sweep-manifest.json`; full per-step tracks on the Volume as
`tracks-*.npz`, 4 chunks). Grid per §14 #6 sign-off: μ ∈ [0.05, 1.00] step 0.05 ×
push ∈ [10, 200]%BW step 10 × 16 seeds = 6400 rollouts. Protocol: 5 s rollouts,
deterministic policy, fixed command [1.0, 0, 0] m/s, lateral 90° push at t = 2 s, seed
scheme PRNGKey(1_000_000 + rollout_index). Fall metric here is still the provisional
up-vector criterion; §5 TTF/thresholds freeze in M0-07 from the logged tracks.

## Timings (§14 #3 measured)

- Chunks of 1600 rollouts, one compiled model-axis-vmapped step reused:
  chunk walls **99.34 s** (includes JIT compile), **8.61 s**, **9.07 s**, **8.72 s**
- Total sweep compute wall ≈ **126 s** on one A100-80GB for 6400 × 5 s rollouts
- Memory: 1600 parallel envs (batched only in geom_friction on the model axis) ran
  without incident; no OOM at this chunk size (larger untested — no need)
- Cost: MEASURED_TBD (dashboard, app opw-sweep)

## Fall counts per cell (fell/16, provisional up-z criterion)

Rows μ (0.05→1.00), columns push %BW (10→200):

```
mu  |  10  20  30  40  50  60  70  80  90 100 110 120 130 140 150 160 170 180 190 200
0.05|  15  16  14  16  16  16  16  16  16  16  16  16  16  16  16  16  16  16  16  16
0.10|   7  11  11   9  12  14  15  16  16  16  15  16  16  16  16  16  16  16  16  16
0.15|   0   2   2   2   7   8  11  14  16  16  16  16  16  16  16  16  16  16  16  16
0.20|   0   0   0   3   3   7  10  14  15  16  15  16  16  16  16  16  16  16  16  16
0.25|   0   0   0   0   2   3   7  11  15  15  16  16  16  16  16  16  16  16  16  16
0.30|   0   0   0   1   1   5   7  12  13  16  15  16  16  16  16  16  16  16  16  16
0.35|   0   0   0   0   1   1   6   8  13  15  16  16  16  16  16  16  16  16  16  16
0.40|   0   0   0   0   1   1   1   8  10  14  14  16  16  16  16  16  16  16  16  16
0.45|   0   0   0   0   0   3   1   4  14  15  15  16  16  16  16  16  16  16  16  16
0.50|   0   0   0   0   0   1   3   8  10  14  15  16  16  16  16  16  16  16  16  16
0.55|   0   0   0   0   0   2   4   2   8  14  16  16  16  16  16  16  16  16  16  16
0.60|   0   0   0   0   0   1   2   6  11   9  15  16  16  16  16  16  16  16  16  16
0.65|   0   0   0   0   0   0   1   7   2  10  14  15  16  16  16  16  16  16  16  16
0.70|   0   0   0   0   0   0   3   6   9   9  12  15  16  16  16  16  16  16  16  16
0.75|   0   0   0   0   0   0   1   3   3   9  14  16  16  16  16  16  16  16  16  16
0.80|   0   0   0   0   0   0   0   0   5   6  13  16  16  16  16  16  16  16  16  16
0.85|   0   0   0   0   0   0   1   0   2   9  11  15  16  16  16  16  16  16  16  16
0.90|   0   0   0   0   0   0   0   0   4   9  13  15  16  16  16  16  16  16  16  16
0.95|   0   0   0   0   0   0   0   1   2   7  13  15  16  16  16  16  16  16  16  16
1.00|   0   0   0   0   0   0   0   0   2   3  14  16  16  16  16  16  16  16  16  16
```

## G4 data-driven outcomes (spec §4 G4)

- **Boundary sits comfortably inside the map on every edge** (push=10 column all-survive
  for μ ≥ 0.15; push=200 row saturated; μ=1.00 row transitions at 90–110%BW; ice regime
  transitions between μ 0.10–0.15). No edge-clipping; no re-run needed.
- **Divergences: 0 of 6400.** The map's "diverged" category is measured-empty on this
  grid — the hatched-region rendering has nothing to hatch (still implemented for the
  live probe path, where off-grid configs could differ).
- **Result framing (spec §8 pre-registered options):** the data selects **structured
  surface** — a smooth diagonal fall-onset band (≈50%BW at μ 0.15 rising to ≈110%BW at
  μ 1.00, transition zone 2–4 cells wide — a genuine stochastic boundary, not a cliff)
  PLUS a sharp ice cliff below μ 0.15 where walking itself fails. Drama go/no-go: GO —
  no need for the no-DR twin or checkpoint-comparison lead.
- **Pilot's non-monotonic-friction hint at 100%BW: resolved as seed noise.** At 16
  seeds the 100%BW column falls monotonically-with-noise from 16 (μ 0.05) to 3
  (μ 1.00).
- **DR context (§8):** the whole push axis is out-of-distribution (training had zero
  pushes); the fall boundary also sits above the DR friction band's lower edge except
  in the ice regime, which is entirely outside DR (μ < 0.4).
- Slider ranges/color scales for the frontend follow this grid; boundary-adjacent cell
  candidates for replays: the 2–4-cell transition band around the diagonal.
