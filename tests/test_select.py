"""Replay selection logic on synthetic manifests (reduce/select.py)."""

from typing import Any

from reduce.select import select_boundary_cells, select_recoveries


def cell(mu: float, push: float, s: float) -> dict[str, Any]:
    return {"mu": mu, "push_pct_bw": push, "survival_at_horizon": s}


def record(mu: float, push: float, seed: int, fell: float | None) -> dict[str, Any]:
    return {
        "mu": mu,
        "push_pct_bw": push,
        "seed_idx": seed,
        "fell_time_upz_s": fell,
        "diverged_at_s": None,
    }


class TestBoundaryCells:
    def test_excludes_saturated_cells(self) -> None:
        cells = [cell(0.5, 50, 1.0), cell(0.5, 100, 0.0), cell(0.5, 80, 0.4)]
        picked = select_boundary_cells(cells, n=15)
        assert [(p.mu, p.push_pct_bw) for p in picked] == [(0.5, 80)]

    def test_prefers_most_contested(self) -> None:
        cells = [cell(0.5, 70, 0.9), cell(0.5, 80, 0.5), cell(0.5, 90, 0.1)]
        picked = select_boundary_cells(cells, n=1)
        assert picked[0].push_pct_bw == 80

    def test_spreads_across_mu_before_doubling_up(self) -> None:
        cells = [
            cell(0.4, 80, 0.50),
            cell(0.4, 90, 0.45),  # second-best overall, same row as best
            cell(0.6, 100, 0.30),
        ]
        picked = select_boundary_cells(cells, n=2)
        assert {p.mu for p in picked} == {0.4, 0.6}  # row spread wins over closeness

    def test_doubles_up_when_rows_exhausted(self) -> None:
        cells = [cell(0.4, 80, 0.5), cell(0.4, 90, 0.4)]
        picked = select_boundary_cells(cells, n=2)
        assert len(picked) == 2

    def test_deterministic(self) -> None:
        cells = [cell(0.1 * i, 10.0 * j, 0.3) for i in range(1, 5) for j in range(1, 5)]
        assert select_boundary_cells(cells, n=6) == select_boundary_cells(cells, n=6)


class TestRecoveries:
    def test_picks_survivors_in_mostly_falling_cells(self) -> None:
        cells = [cell(0.5, 100, 0.25)]
        records = [
            record(0.5, 100, 0, 2.5),
            record(0.5, 100, 1, None),  # the recovery
            record(0.5, 100, 2, 2.8),
            record(0.5, 100, 3, 3.0),
        ]
        picked = select_recoveries(cells, records, n=3)
        assert len(picked) == 1
        assert picked[0].seed_idx == 1

    def test_ignores_survivors_in_safe_cells(self) -> None:
        cells = [cell(0.5, 20, 1.0), cell(0.5, 60, 0.8)]
        records = [record(0.5, 20, 0, None), record(0.5, 60, 1, None)]
        assert select_recoveries(cells, records, n=3) == []

    def test_most_dramatic_first(self) -> None:
        cells = [cell(0.5, 100, 0.4), cell(0.5, 120, 0.1)]
        records = [record(0.5, 100, 0, None), record(0.5, 120, 1, None)]
        picked = select_recoveries(cells, records, n=2)
        assert picked[0].push_pct_bw == 120  # lower survival cell first
