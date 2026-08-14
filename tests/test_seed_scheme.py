"""Seed-scheme helper vs the sweep executor's actual constants and records."""

import json
from pathlib import Path

import pytest

from sweep import full_sweep, seed_scheme


def test_constants_match_the_executor() -> None:
    assert seed_scheme.GRID_MUS == full_sweep.GRID_MUS
    assert seed_scheme.GRID_PUSH_PCTS == full_sweep.GRID_PUSH_PCTS
    assert seed_scheme.SEEDS_PER_CELL == full_sweep.SEEDS_PER_CELL
    assert seed_scheme.SEED_BASE == full_sweep.SEED_BASE


def test_matches_committed_sweep_records() -> None:
    manifest = json.loads(
        (
            Path(__file__).parent.parent / "docs/measurements/2026-08-13-g4-sweep-manifest.json"
        ).read_text()
    )
    for record in manifest["records"][:200] + manifest["records"][-200:]:
        expected = seed_scheme.SEED_BASE + record["rollout"]
        assert (
            seed_scheme.sweep_prng_seed(record["mu"], record["push_pct_bw"], record["seed_idx"])
            == expected
        )


def test_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        seed_scheme.sweep_prng_seed(0.5, 80.0, 16)
    with pytest.raises(ValueError):
        seed_scheme.sweep_prng_seed(0.51, 80.0, 0)
