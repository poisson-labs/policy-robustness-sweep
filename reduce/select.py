"""Replay-cell selection from the survival manifest (spec §3 reduce/: boundary-adjacent
cells ~15 + 2-3 near-boundary recoveries; divergence replay honestly dropped, DEVLOG
Session 12).

Selection is deterministic from committed data:
- boundary cells: cells with 0 < S(horizon) < 1, preferring S nearest 0.5 (the most
  contested worlds) while spreading across friction rows (at most one per μ until μ
  values are exhausted, then second-best, etc.) so the replay set walks the whole
  diagonal rather than clustering.
- recoveries: individual surviving rollouts inside mostly-falling cells
  (S ≤ recovery_max_survival) — the seeds that stayed up where most fell. Most dramatic
  first (lowest cell survival, then highest push).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CellRef:
    mu: float
    push_pct_bw: float
    survival_at_horizon: float


@dataclass(frozen=True)
class RecoveryRef:
    mu: float
    push_pct_bw: float
    seed_idx: int
    cell_survival: float


def select_boundary_cells(cells: list[dict[str, Any]], n: int = 15) -> list[CellRef]:
    """Pick n boundary cells, S nearest 0.5 first, spread across μ rows round-robin."""
    boundary = [c for c in cells if 0.0 < c["survival_at_horizon"] < 1.0]
    boundary.sort(key=lambda c: (abs(c["survival_at_horizon"] - 0.5), c["mu"], c["push_pct_bw"]))
    picked: list[CellRef] = []
    used_mu_counts: dict[float, int] = {}
    round_cap = 1
    while len(picked) < min(n, len(boundary)):
        progressed = False
        for cell in boundary:
            if len(picked) >= n:
                break
            key = (cell["mu"], cell["push_pct_bw"])
            if any((p.mu, p.push_pct_bw) == key for p in picked):
                continue
            if used_mu_counts.get(cell["mu"], 0) >= round_cap:
                continue
            picked.append(CellRef(cell["mu"], cell["push_pct_bw"], cell["survival_at_horizon"]))
            used_mu_counts[cell["mu"]] = used_mu_counts.get(cell["mu"], 0) + 1
            progressed = True
        if not progressed:
            round_cap += 1
    return picked


def select_recoveries(
    cells: list[dict[str, Any]],
    records: list[dict[str, Any]],
    n: int = 3,
    recovery_max_survival: float = 0.5,
) -> list[RecoveryRef]:
    """Surviving seeds inside mostly-falling cells, most dramatic first.

    `records` are the sweep's per-rollout records (fell_time_upz_s None = survived;
    validated 6400/6400 against the frozen classifier in M0-07).
    """
    survival_by_cell = {(c["mu"], c["push_pct_bw"]): c["survival_at_horizon"] for c in cells}
    candidates: list[RecoveryRef] = []
    for r in records:
        key = (r["mu"], r["push_pct_bw"])
        s = survival_by_cell.get(key)
        if s is None or not (0.0 < s <= recovery_max_survival):
            continue
        if r["fell_time_upz_s"] is None and r["diverged_at_s"] is None:
            candidates.append(RecoveryRef(r["mu"], r["push_pct_bw"], r["seed_idx"], s))
    candidates.sort(key=lambda c: (c.cell_survival, -c.push_pct_bw, c.mu, c.seed_idx))
    return candidates[:n]
