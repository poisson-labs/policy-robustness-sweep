"""Boundary position and its uncertainty (M1.6-A).

The boundary per friction row is the interpolated S = 0.5 crossing of the per-cell KM
survival estimates along the push axis. Uncertainty is per-row (the transition band's
width visibly varies with friction), obtained by bootstrapping the seeds within every
cell: resample the 16 rollouts with replacement, recompute KM survival-at-horizon,
re-extract each row's crossing; repeat ≥ 2000 times; report per-row median + 95% CI.

Rows with no crossing (all-survive / all-fall, e.g. the ice rows) return NO_BOUNDARY,
never an extrapolated number.

Pure numpy over committed data — no simulation, no GPU. Inherits reduce/'s pattern:
ground-truth unit tests on synthetic grids (tests/test_boundary.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from reduce.classify import Classification, Outcome
from reduce.survival import cell_survival

NO_BOUNDARY = "no boundary in range"


@dataclass(frozen=True)
class RowBoundary:
    mu: float
    median_pct_bw: float | None  # None ⇔ NO_BOUNDARY in a majority of resamples
    ci_low_pct_bw: float | None
    ci_high_pct_bw: float | None
    fraction_with_crossing: float  # share of resamples that had a crossing
    point_estimate_pct_bw: float | None  # crossing on the un-resampled data


def crossing_along_row(pushes: list[float], survival: npt.NDArray[np.floating]) -> float | None:
    """Interpolated push value where S first drops through 0.5 along increasing push.

    Requires S >= 0.5 at the lower neighbour and S < 0.5 at the upper — a genuine
    crossing between two grid points. Returns None if no such pair exists.
    """
    for i in range(1, len(pushes)):
        s0, s1 = float(survival[i - 1]), float(survival[i])
        if s0 >= 0.5 > s1:
            if s0 == s1:  # pragma: no cover — guarded by the inequality above
                return pushes[i - 1]
            return pushes[i - 1] + (s0 - 0.5) / (s0 - s1) * (pushes[i] - pushes[i - 1])
    return None


def survival_grid(
    outcomes: dict[tuple[float, float], list[Classification]],
    mus: list[float],
    pushes: list[float],
    horizon_s: float,
) -> npt.NDArray[np.floating]:
    grid = np.empty((len(mus), len(pushes)))
    for i, mu in enumerate(mus):
        for j, push in enumerate(pushes):
            grid[i, j] = cell_survival(outcomes[(mu, push)], horizon_s).survival_at_horizon
    return grid


def bootstrap_row_boundaries(
    outcomes: dict[tuple[float, float], list[Classification]],
    mus: list[float],
    pushes: list[float],
    horizon_s: float,
    n_resamples: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> list[RowBoundary]:
    """Bootstrap the per-row S=0.5 crossing by resampling seeds within each cell.

    Fast path: KM with horizon-only censoring reduces survival-at-horizon to the
    survivor fraction (tested identity in test_survival.py), so each resample only needs
    a binomial draw per cell — the KM machinery is used for the point estimate and the
    identity is asserted for the resamples.
    """
    rng = np.random.default_rng(seed)
    n_mu, n_push = len(mus), len(pushes)

    # Point estimate through the real KM code path.
    point_grid = survival_grid(outcomes, mus, pushes, horizon_s)

    # Per-cell survivor indicators (excluding diverged, matching cell_survival semantics).
    survivors = np.empty((n_mu, n_push), dtype=object)
    for i, mu in enumerate(mus):
        for j, push in enumerate(pushes):
            cell = [r for r in outcomes[(mu, push)] if r.outcome is not Outcome.DIVERGED]
            survivors[i, j] = np.array([r.outcome is Outcome.CENSORED for r in cell])
            assert np.isclose(survivors[i, j].mean() if len(cell) else 1.0, point_grid[i, j]), (
                "KM survival-at-horizon must equal survivor fraction under horizon censoring"
            )

    crossings = np.full((n_resamples, n_mu), np.nan)
    for b in range(n_resamples):
        grid_b = np.empty((n_mu, n_push))
        for i in range(n_mu):
            for j in range(n_push):
                cell = survivors[i, j]
                if len(cell) == 0:
                    grid_b[i, j] = 1.0
                    continue
                idx = rng.integers(0, len(cell), size=len(cell))
                grid_b[i, j] = cell[idx].mean()
        for i in range(n_mu):
            c = crossing_along_row(pushes, grid_b[i])
            crossings[b, i] = np.nan if c is None else c

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows: list[RowBoundary] = []
    for i, mu in enumerate(mus):
        col = crossings[:, i]
        with_crossing = col[~np.isnan(col)]
        frac = len(with_crossing) / n_resamples
        point = crossing_along_row(pushes, point_grid[i])
        if frac < 0.5:
            rows.append(RowBoundary(mu, None, None, None, frac, point))
            continue
        rows.append(
            RowBoundary(
                mu=mu,
                median_pct_bw=float(np.median(with_crossing)),
                ci_low_pct_bw=float(np.quantile(with_crossing, lo_q)),
                ci_high_pct_bw=float(np.quantile(with_crossing, hi_q)),
                fraction_with_crossing=frac,
                point_estimate_pct_bw=point,
            )
        )
    return rows


def headline_ci_width(rows: list[RowBoundary]) -> float | None:
    """Median row-wise 95% CI width (%BW) across rows that have a boundary."""
    widths = [
        r.ci_high_pct_bw - r.ci_low_pct_bw
        for r in rows
        if r.ci_high_pct_bw is not None and r.ci_low_pct_bw is not None
    ]
    return float(np.median(widths)) if widths else None
