"""Frozen §5 failure thresholds (M0-07). FROZEN as of 2026-08-13 — do not tune.

Measured basis (docs/measurements/2026-08-13-thresholds.md; data = G4 sweep tracks,
6400 rollouts + 8 nominal rollouts):
- Nominal walking: torso z ∈ [0.277, 0.325] m; |roll|,|pitch| ≤ 4.2°.
- Survivors (2374 non-falling rollouts across the whole grid): max tilt 37.5°
  (p99 24.8°); min torso z 0.173 m.
- Fallen rollouts pass 90° (up-vector criterion) and end on the ground.

Thresholds sit in the measured gap — no survivor ever crossed either, every fall
crosses both:
"""

import math
from typing import Final

TORSO_Z_FAIL_M: Final = 0.15
TILT_FAIL_RAD: Final = math.radians(60.0)

ROLLOUT_S: Final = 5.0  # spec §5: survival past this is right-censored, never TTF=5
