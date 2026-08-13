"""Checkpoint stage selection for G2 (spec §2 stretch: "newborn / half-trained / final").

Strategy (recorded in DEVLOG 2026-08-13):
- "newborn" is the freshly initialized, untrained policy (step 0), saved directly by the
  harness before training — not the first eval checkpoint, which may already be
  substantially trained.
- Periodic checkpoints are saved at every brax eval segment during the run.
- "converged" is the final checkpoint.
- "wobbly" (part-trained) is selected post-hoc from the eval-reward curve: the checkpoint
  whose eval reward is closest to half the final checkpoint's reward. This runs after
  training from the metrics log, so the run itself never depends on it — and it is pure
  logic, tested against synthetic curves.
"""

from __future__ import annotations

from dataclasses import dataclass

NEWBORN_STEP = 0


@dataclass(frozen=True)
class EvalPoint:
    """One (training step, eval episode reward) point from the training metrics log."""

    step: int
    reward: float


@dataclass(frozen=True)
class StageSelection:
    newborn_step: int
    wobbly_step: int
    converged_step: int


def select_wobbly(points: list[EvalPoint]) -> EvalPoint:
    """Checkpoint whose reward is nearest to 50% of the final checkpoint's reward.

    Requires at least two eval points (otherwise "half-trained" has no meaning). The final
    point itself is excluded from candidates so wobbly is always strictly earlier than
    converged. Ties resolve to the earliest qualifying checkpoint.
    """
    if len(points) < 2:
        raise ValueError("need at least two eval points to select a wobbly checkpoint")
    ordered = sorted(points, key=lambda p: p.step)
    final = ordered[-1]
    target = final.reward / 2.0
    candidates = ordered[:-1]
    return min(candidates, key=lambda p: (abs(p.reward - target), p.step))


def select_stages(points: list[EvalPoint]) -> StageSelection:
    """Full stage selection from a training metrics log (see module docstring)."""
    ordered = sorted(points, key=lambda p: p.step)
    return StageSelection(
        newborn_step=NEWBORN_STEP,
        wobbly_step=select_wobbly(ordered).step,
        converged_step=ordered[-1].step,
    )
