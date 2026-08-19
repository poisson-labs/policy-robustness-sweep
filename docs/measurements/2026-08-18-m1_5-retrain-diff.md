# M1.5 — retrain → re-map → diff (v0 vs v1)

All measured; sources committed under docs/measurements/. The instrument's third claim
(CI-for-policies) demonstrated end to end on the flagship policy.

## The one change (v1)

v1 = the exact G2 recipe (200M steps, identical PPO hyperparams, network, seed scheme,
20 evals) with ONLY the training distribution changed (DR record verbatim in
`2026-08-18-g3-v1-equivalence-report.json`'s sibling `dr-record-v1.json` on the Volume;
delta captured with source hashes):

1. env `pert_config.enable = True` — the env's own velocity kicks (shipped disabled by
   Playground; the v0 map showed every shove is out-of-distribution);
2. floor-friction DR floor 0.4 → 0.1 (max unchanged 1.0). NOT the map floor 0.05:
   training on physically-unwalkable ice risks a conservative crouch; the re-sweep
   reports what it bought below 0.1 anyway.

Everything else is Playground default. v1 wall 751 s / 217.9M steps on A100-80GB.

## Honesty gates (both checkpoints on equal footing)

- **v0 G3** (M0-04b): passed. **v1 G3** (this session): passed — but the equivalence
  runner was FIXED first. Its statistical gate compared our nominal-env harness to the
  training-time final eval; that is apples-to-oranges for a pert-trained checkpoint
  (v1's training eval runs WITH kicks; the sweep runs nominal). Replaced the gate with a
  protocol-matched one: native-harness vs injected-harness, both nominal, K repeats —
  the correct "is the sweep executor faithful to the plain env" question,
  checkpoint-agnostic. v1: replay bitwise 0.0; matched-protocol injected 23.66 vs native
  23.94, |Δ| 0.28 ≤ gate 0.76 → PASS. The training-eval gap (Δ 3.99) is recorded as
  context, expected for a kick-trained policy. (This also makes the gate correct for the
  future newborn/wobbly checkpoints — the Session 12 concern.)

## Diff hygiene

Same 20×20 grid, same frozen §5 thresholds, same command [1,0,0], same seed scheme.
The ONLY deltas: the checkpoint and 32 seeds/cell (both re-swept at 32 per the Session 23
power math). v0@32: 12,800 rollouts, 62.7% fell, 0 diverged. v1@32: 12,800 rollouts,
52.8% fell, 0 diverged.

## Result (docs/measurements/2026-08-18-v0-v1-diff.json; figure: figures/v0-v1-diff.*)

- **60 cells significantly safer, 0 significantly worse** (per-cell Fisher exact +
  Benjamini-Hochberg FDR, q < 0.05). The envelope EXPANDED; the shrinkage census
  (Session 22 honesty guard) is empty.
- **Boundary pushed outward on every friction row.** Row-wise S=0.5 crossing (bootstrap
  median), v1 − v0:

| μ | v0 %BW | v1 %BW | Δ%BW |
|---|---|---|---|
| 0.05 | none (walking fails) | 50.0 | — (regime gained) |
| 0.10 | 20.0 | 85.7 | +65.7 |
| 0.15 | 57.2 | 92.9 | +35.6 |
| 0.20 | 70.0 | 96.0 | +26.0 |
| 0.25 | 70.0 | 100.9 | +30.9 |
| 0.30 | 74.0 | 97.4 | +23.4 |
| 0.40 | 85.6 | 100.9 | +15.4 |
| 0.50 | 88.0 | 101.3 | +13.3 |
| 0.60 | 83.8 | 101.3 | +17.4 |
| 0.70 | 95.0 | 103.3 | +8.3 |
| 0.80 | 102.0 | 106.0 | +4.0 |
| 0.90 | 102.9 | 107.0 | +4.1 |
| 1.00 | 102.6 | 108.1 | +5.5 |

- **The gains are largest exactly where we intervened.** Ice rows (μ ≤ 0.20) moved
  25–66%BW — that is where friction DR was widened and where the v0 ice cliff was.
  μ = 0.05 went from no boundary (v0 can't walk near-ice) to a boundary at 50%BW.
  High-friction rows (already inside training DR) moved only 4–8%BW. The intervention's
  fingerprint is legible in the surface.

## Post framing (§8)

"We read the map, changed one thing about training — turned on shoves and widened the
ground it practiced on — retrained in 12 minutes, and re-mapped. The failure envelope
grew, most where we aimed it, nowhere did it shrink." Every number measured; the
retrain-remap-diff loop is the thesis's CI-for-policies claim, shown, not asserted.
