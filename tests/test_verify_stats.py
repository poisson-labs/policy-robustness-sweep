"""Pure episode-stats helper from the nominal verification harness."""

import pytest

from train.verify_nominal import episode_stats


def test_basic_stats() -> None:
    stats = episode_stats([10.0, 20.0], [500, 1000])
    assert stats["episodes"] == 2.0
    assert stats["reward_mean"] == 15.0
    assert stats["reward_min"] == 10.0
    assert stats["reward_max"] == 20.0
    assert stats["length_mean"] == 750.0
    assert stats["reward_std"] == 5.0


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        episode_stats([], [])
