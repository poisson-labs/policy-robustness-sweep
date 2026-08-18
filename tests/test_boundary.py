"""Boundary bootstrap vs synthetic grids with known boundary + injected noise (M1.6-A)."""

import numpy as np

from reduce.boundary import (
    NO_BOUNDARY,
    bootstrap_row_boundaries,
    crossing_along_row,
    headline_ci_width,
)
from reduce.classify import Classification, Outcome

HORIZON = 5.0
PUSHES = [float(10 * i) for i in range(1, 21)]


def failed(ttf: float = 2.5) -> Classification:
    return Classification(Outcome.FAILED, ttf, None, "tilt")


def censored() -> Classification:
    return Classification(Outcome.CENSORED, None, None, None)


def synthetic_grid(
    mus: list[float], true_boundary: dict[float, float | None], width: float, n: int, seed: int
) -> dict[tuple[float, float], list[Classification]]:
    """Cells fall with probability sigmoid((push - boundary)/width); None → never falls."""
    rng = np.random.default_rng(seed)
    grid: dict[tuple[float, float], list[Classification]] = {}
    for mu in mus:
        for push in PUSHES:
            b = true_boundary[mu]
            p_fall = 0.0 if b is None else 1.0 / (1.0 + np.exp(-(push - b) / width))
            grid[(mu, push)] = [failed() if rng.random() < p_fall else censored() for _ in range(n)]
    return grid


class TestCrossing:
    def test_interpolates_between_grid_points(self) -> None:
        s = np.ones(20)
        s[8:] = 0.0  # S drops 1→0 between push 80 (idx7) and 90 (idx8): crossing at 85
        assert crossing_along_row(PUSHES, s) == 85.0

    def test_no_crossing_all_survive(self) -> None:
        assert crossing_along_row(PUSHES, np.ones(20)) is None

    def test_no_crossing_all_fall(self) -> None:
        assert crossing_along_row(PUSHES, np.zeros(20)) is None

    def test_exact_half_at_gridpoint_counts_as_lower_neighbour(self) -> None:
        s = np.ones(20)
        s[5] = 0.5
        s[6:] = 0.0
        # S=0.5 at push 60 (>=0.5) then 0 at 70 → crossing interpolated from 60
        assert crossing_along_row(PUSHES, s) == 60.0


class TestBootstrap:
    def test_ci_covers_truth_at_nominal_rate(self) -> None:
        # Repeat over many synthetic realizations; 95% CI should cover truth ~95%.
        mus = [0.5]
        truth: dict[float, float | None] = {0.5: 87.0}
        covered = 0
        trials = 40
        for t in range(trials):
            grid = synthetic_grid(mus, truth, width=8.0, n=16, seed=t)
            rows = bootstrap_row_boundaries(grid, mus, PUSHES, HORIZON, n_resamples=400, seed=t)
            r = rows[0]
            assert r.median_pct_bw is not None
            assert r.ci_low_pct_bw is not None and r.ci_high_pct_bw is not None
            if r.ci_low_pct_bw <= 87.0 <= r.ci_high_pct_bw:
                covered += 1
        # Nominal 95%; allow sampling slack on 40 trials (binomial sd ≈ 3.5%).
        assert covered / trials >= 0.80, f"coverage {covered}/{trials}"

    def test_no_crossing_row_returns_sentinel_not_number(self) -> None:
        mus = [0.05, 0.5]
        truth: dict[float, float | None] = {0.05: None, 0.5: 90.0}
        grid = synthetic_grid(mus, truth, width=8.0, n=16, seed=1)
        rows = bootstrap_row_boundaries(grid, mus, PUSHES, HORIZON, n_resamples=200)
        ice = rows[0]
        assert ice.median_pct_bw is None
        assert ice.ci_low_pct_bw is None
        assert ice.point_estimate_pct_bw is None
        assert ice.fraction_with_crossing < 0.5
        assert NO_BOUNDARY  # sentinel constant exists for the JSON writer
        assert rows[1].median_pct_bw is not None

    def test_narrow_band_gives_narrow_ci(self) -> None:
        mus = [0.5]
        truth: dict[float, float | None] = {0.5: 100.0}
        sharp = synthetic_grid(mus, truth, width=1.0, n=16, seed=3)
        wide = synthetic_grid(mus, truth, width=15.0, n=16, seed=3)
        r_sharp = bootstrap_row_boundaries(sharp, mus, PUSHES, HORIZON, n_resamples=300)[0]
        r_wide = bootstrap_row_boundaries(wide, mus, PUSHES, HORIZON, n_resamples=300)[0]
        assert (r_sharp.ci_high_pct_bw - r_sharp.ci_low_pct_bw) < (  # type: ignore[operator]
            r_wide.ci_high_pct_bw - r_wide.ci_low_pct_bw  # type: ignore[operator]
        )

    def test_headline_scalar_is_median_width_over_rows_with_boundary(self) -> None:
        mus = [0.05, 0.4, 0.5]
        truth: dict[float, float | None] = {0.05: None, 0.4: 80.0, 0.5: 95.0}
        grid = synthetic_grid(mus, truth, width=6.0, n=16, seed=5)
        rows = bootstrap_row_boundaries(grid, mus, PUSHES, HORIZON, n_resamples=200)
        width = headline_ci_width(rows)
        assert width is not None and width > 0
        widths: list[float] = [
            r.ci_high_pct_bw - r.ci_low_pct_bw
            for r in rows
            if r.ci_low_pct_bw is not None and r.ci_high_pct_bw is not None
        ]
        assert width == float(np.median(widths))
