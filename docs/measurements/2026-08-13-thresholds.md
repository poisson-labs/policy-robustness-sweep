# §5 failure thresholds — measured basis and freeze (M0-07)

Thresholds are FROZEN as of 2026-08-13 (reduce/thresholds.py); this records the data
they were set from. Sources: nominal tracks (pilot run `pilot-20260813T173150Z`,
tracks-nominal.npz, 8 rollouts at model-default μ 0.6, zero push) and the full G4 sweep
tracks (run `sweep-20260813T175452Z`, 6400 rollouts).

## Distributions

Nominal walking (8 rollouts × 250 steps):
- torso z: min 0.277 m, p50 0.311 m, max 0.325 m
- |roll| max 3.9°; |pitch| max 4.2°; up-vector z never below 0.9965

Survivors across the full grid (2374 rollouts that never crossed up-z < 0):
- max tilt (max of |roll|, |pitch|): p50 9.3°, p90 17.6°, p99 24.8°, max 37.5°
- min torso z: p50 0.272 m, p1 0.215 m, min 0.173 m
- zero survivors exceeded 45° tilt; zero went below 0.15 m torso height

## Frozen values

- **Torso height failure: torso z < 0.15 m** — below every survivor excursion
  (min 0.173) and far below nominal (0.277+); fallen torsos rest near the ground.
- **Tilt failure: max(|roll|, |pitch|) > 60°** — 22.5° above the worst surviving
  recovery (37.5°); every fall passes 90°.
- TTF = first step either condition holds; survival past 5 s = right-censored (never
  TTF = 5); NaN/inf = diverged, distinct category, earliest event wins.

## Validation against the sweep

Running the frozen classifier (reduce/classify.py) over all 6400 sweep rollouts:
- outcomes: 4026 failed, 2374 censored, 0 diverged
- **agreement with the provisional up-z criterion: 6400/6400** (no rollout changed
  category)
- TTF ≤ up-z crossing time in 4026/4026 failures; median lead 0.080 s (max 0.580 s) —
  the frozen thresholds mark fall onset rather than ground impact, as intended.

The classifier itself is covered by 14 unit tests on crafted logs
(tests/test_classify.py), including the censoring rule, whichever-first semantics,
divergence-as-distinct, and NaN-poisoning cases.
