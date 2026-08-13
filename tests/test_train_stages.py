"""Stage selection against synthetic training curves (train/stages.py)."""

import pytest

from train.stages import EvalPoint, select_stages, select_wobbly


def curve(points: list[tuple[int, float]]) -> list[EvalPoint]:
    return [EvalPoint(step=s, reward=r) for s, r in points]


class TestSelectWobbly:
    def test_picks_checkpoint_nearest_half_final_reward(self) -> None:
        points = curve([(0, 0.0), (10, 10.0), (20, 24.0), (30, 40.0), (40, 50.0)])
        assert select_wobbly(points).step == 20  # 24.0 is nearest to 25.0

    def test_excludes_final_checkpoint(self) -> None:
        # Final reward 10 → target 5; final point itself would be a bad "half-trained".
        points = curve([(0, 4.0), (10, 10.0)])
        assert select_wobbly(points).step == 0

    def test_tie_resolves_to_earliest(self) -> None:
        # Target 10; steps 10 and 20 both distance 2 → earliest wins.
        points = curve([(10, 8.0), (20, 12.0), (30, 20.0)])
        assert select_wobbly(points).step == 10

    def test_unsorted_input_is_handled(self) -> None:
        points = curve([(30, 40.0), (0, 0.0), (20, 24.0), (10, 10.0), (40, 50.0)])
        assert select_wobbly(points).step == 20

    def test_rejects_fewer_than_two_points(self) -> None:
        with pytest.raises(ValueError):
            select_wobbly(curve([(0, 1.0)]))

    def test_non_monotonic_curve_still_selects_nearest(self) -> None:
        # Reward dips mid-training; selection is by reward distance, not step position.
        points = curve([(0, 0.0), (10, 30.0), (20, 12.0), (30, 26.0), (40, 50.0)])
        assert select_wobbly(points).step == 30  # 26.0 nearest to 25.0

    def test_flat_zero_curve_picks_earliest(self) -> None:
        # Degenerate no-learning run: everything ties at distance 0 → earliest.
        points = curve([(0, 0.0), (10, 0.0), (20, 0.0)])
        assert select_wobbly(points).step == 0


class TestSelectStages:
    def test_full_selection(self) -> None:
        points = curve([(0, 0.0), (10, 10.0), (20, 24.0), (30, 40.0), (40, 50.0)])
        stages = select_stages(points)
        assert stages.newborn_step == 0
        assert stages.wobbly_step == 20
        assert stages.converged_step == 40
