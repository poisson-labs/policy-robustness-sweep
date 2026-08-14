"""The sweep's seed scheme as a pure, importable fact (spec §6: documented seeds).

Every consumer that claims to reproduce "rollout (mu, push, seed_idx) of the sweep"
MUST derive its PRNG seed here. Using seed_idx directly as a PRNG seed simulates a
DIFFERENT world that merely shares a label — the M1-04 render batch did exactly that
and produced "recovery" clips that fell (DEVLOG Session 19).
"""

from __future__ import annotations

GRID_MUS = [round(0.05 * i, 2) for i in range(1, 21)]
GRID_PUSH_PCTS = [float(10 * i) for i in range(1, 21)]
SEEDS_PER_CELL = 16
SEED_BASE = 1_000_000


def sweep_rollout_index(mu: float, push_pct_bw: float, seed_idx: int) -> int:
    mu_idx = GRID_MUS.index(round(mu, 2))
    push_idx = GRID_PUSH_PCTS.index(float(push_pct_bw))
    if not 0 <= seed_idx < SEEDS_PER_CELL:
        raise ValueError(f"seed_idx {seed_idx} out of range")
    return (mu_idx * len(GRID_PUSH_PCTS) + push_idx) * SEEDS_PER_CELL + seed_idx


def sweep_prng_seed(mu: float, push_pct_bw: float, seed_idx: int) -> int:
    """The PRNGKey argument the sweep used for this exact rollout."""
    return SEED_BASE + sweep_rollout_index(mu, push_pct_bw, seed_idx)
