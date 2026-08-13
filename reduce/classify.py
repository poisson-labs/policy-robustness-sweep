"""Per-rollout outcome classification (spec §5): TTF / right-censoring / divergence.

Operates on logged state tracks (numpy arrays from the sweep/probe executors), never on
live simulation state — classification is reproducible from committed data.

Semantics (locked by spec §5):
- failure: torso height < threshold OR max(|roll|, |pitch|) > threshold, whichever
  step comes first; TTF = that step's timestamp.
- censoring: survival past the rollout horizon → right-censored (TTF > horizon), and
  TTF is NEVER reported as exactly the horizon.
- diverged: NaN/inf in state is a DISTINCT category, not a failure. The earliest event
  wins: a rollout that falls at t=1 and produces NaNs at t=3 is a failure; one that
  diverges before any failure condition is diverged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from reduce.thresholds import TILT_FAIL_RAD, TORSO_Z_FAIL_M

FloatTrack = npt.NDArray[np.floating]
BoolTrack = npt.NDArray[np.bool_]


class Outcome(StrEnum):
    FAILED = "failed"
    CENSORED = "censored"
    DIVERGED = "diverged"


@dataclass(frozen=True)
class Classification:
    outcome: Outcome
    ttf_s: float | None  # set iff FAILED; strictly < horizon
    diverged_at_s: float | None  # set iff DIVERGED
    first_trigger: str | None  # "torso_z" | "tilt" for FAILED, else None


def _first_true_index(mask: BoolTrack) -> int:
    """Index of first True, or -1."""
    if not bool(mask.any()):
        return -1
    return int(np.argmax(mask))


def classify_rollout(
    torso_z: FloatTrack,
    roll: FloatTrack,
    pitch: FloatTrack,
    dt: float,
    torso_z_fail_m: float = TORSO_Z_FAIL_M,
    tilt_fail_rad: float = TILT_FAIL_RAD,
) -> Classification:
    """Classify one rollout from its per-step tracks (step i = time (i+1)*dt).

    Divergence is detected from non-finite values in the provided tracks themselves;
    comparisons involving NaN are False, so divergence never masks as failure.
    """
    if not (len(torso_z) == len(roll) == len(pitch)) or len(torso_z) == 0:
        raise ValueError("tracks must be equal-length and non-empty")

    finite = np.isfinite(torso_z) & np.isfinite(roll) & np.isfinite(pitch)
    diverged_i = _first_true_index(~finite)

    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    fail_mask = (torso_z < torso_z_fail_m) | (tilt > tilt_fail_rad)
    failed_i = _first_true_index(fail_mask)

    if diverged_i >= 0 and (failed_i < 0 or diverged_i <= failed_i):
        return Classification(
            outcome=Outcome.DIVERGED,
            ttf_s=None,
            diverged_at_s=(diverged_i + 1) * dt,
            first_trigger=None,
        )
    if failed_i >= 0:
        trigger = "torso_z" if bool(torso_z[failed_i] < torso_z_fail_m) else "tilt"
        return Classification(
            outcome=Outcome.FAILED,
            ttf_s=(failed_i + 1) * dt,
            diverged_at_s=None,
            first_trigger=trigger,
        )
    return Classification(
        outcome=Outcome.CENSORED, ttf_s=None, diverged_at_s=None, first_trigger=None
    )
