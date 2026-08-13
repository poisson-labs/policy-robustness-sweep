"""configs/world.py — round-trip, clamping, cache-key stability (kickoff §4 required
test suite for the config schema)."""

import pytest
from pydantic import ValidationError

from configs.world import DEFAULT_SEED, WorldConfig


class TestRoundTrip:
    def test_json_round_trip_preserves_identity(self) -> None:
        config = WorldConfig(
            friction=0.73, push_magnitude_pct_bw=120.0, push_direction_deg=45.0, seed=7
        )
        restored = WorldConfig.model_validate_json(config.model_dump_json())
        assert restored == config

    def test_none_friction_round_trips(self) -> None:
        config = WorldConfig()
        restored = WorldConfig.model_validate_json(config.model_dump_json())
        assert restored.friction is None
        assert restored == config

    def test_frozen_and_extra_forbidden(self) -> None:
        config = WorldConfig()
        with pytest.raises(ValidationError):
            config.friction = 0.5  # type: ignore[misc]
        with pytest.raises(ValidationError):
            WorldConfig.model_validate({"unknown_field": 1})


class TestClamping:
    def test_in_range_values_pass_through(self) -> None:
        config = WorldConfig.clamped(friction=0.7, push_magnitude_pct_bw=150.0)
        assert config.friction == 0.7
        assert config.push_magnitude_pct_bw == 150.0

    def test_out_of_range_values_clamp_not_error(self) -> None:
        config = WorldConfig.clamped(friction=99.0, push_magnitude_pct_bw=-5.0, push_start_s=100.0)
        assert config.friction == WorldConfig.RANGES["friction"][1]
        assert config.push_magnitude_pct_bw == 0.0
        assert config.push_start_s == WorldConfig.RANGES["push_start_s"][1]

    def test_direction_wraps_modulo_360(self) -> None:
        assert WorldConfig.clamped(push_direction_deg=450.0).push_direction_deg == 90.0
        assert WorldConfig.clamped(push_direction_deg=-90.0).push_direction_deg == 270.0

    def test_none_friction_stays_none(self) -> None:
        assert WorldConfig.clamped(friction=None).friction is None


class TestCacheKey:
    def test_stable_golden_key(self) -> None:
        # Golden value: if this changes, every cached probe invalidates — that must be
        # a deliberate CACHE_KEY_VERSION bump, never an accident.
        config = WorldConfig(
            friction=0.7, push_magnitude_pct_bw=100.0, push_direction_deg=90.0, seed=0
        )
        assert config.cache_key() == "v1|mu=0.7000|push=100.00|dir=90.00|t0=2.00|seed=0"

    def test_nominal_uses_default_marker(self) -> None:
        assert WorldConfig().cache_key() == "v1|mu=default|push=0.00|dir=90.00|t0=2.00|seed=0"

    def test_sub_resolution_configs_share_a_key(self) -> None:
        a = WorldConfig(friction=0.701, push_magnitude_pct_bw=100.2)
        b = WorldConfig(friction=0.699, push_magnitude_pct_bw=100.4)
        assert a.cache_key() == b.cache_key()

    def test_distinct_configs_get_distinct_keys(self) -> None:
        a = WorldConfig(friction=0.70)
        b = WorldConfig(friction=0.75)
        assert a.cache_key() != b.cache_key()

    def test_seed_is_part_of_the_key(self) -> None:
        assert WorldConfig(seed=DEFAULT_SEED).cache_key() != WorldConfig(seed=1).cache_key()

    def test_rounding_is_idempotent(self) -> None:
        config = WorldConfig(friction=0.734, push_direction_deg=47.0)
        assert config.rounded() == config.rounded().rounded()
