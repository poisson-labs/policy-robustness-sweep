"""Failure/divergence classification on crafted state logs (kickoff §4 required suite;
spec §5 semantics incl. the censoring rule and diverged-as-distinct)."""

import math

import numpy as np
import pytest

from reduce.classify import Classification, Outcome, classify_rollout
from reduce.thresholds import TILT_FAIL_RAD, TORSO_Z_FAIL_M

DT = 0.02
STEPS = 250  # 5 s horizon
NOMINAL_Z = 0.31


Track = np.ndarray


def classify(
    z: Track | None = None, roll: Track | None = None, pitch: Track | None = None
) -> Classification:
    z_track: Track = np.full(STEPS, NOMINAL_Z) if z is None else z
    roll_track: Track = np.zeros(STEPS) if roll is None else roll
    pitch_track: Track = np.zeros(STEPS) if pitch is None else pitch
    return classify_rollout(z_track, roll_track, pitch_track, DT)


class TestCensoring:
    def test_clean_walk_is_censored_with_no_ttf(self) -> None:
        result = classify()
        assert result.outcome is Outcome.CENSORED
        assert result.ttf_s is None  # never TTF = horizon (spec §5)

    def test_survivor_grazing_thresholds_is_censored(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[100] = TORSO_Z_FAIL_M + 1e-6  # dips toward but not past
        roll = np.zeros(STEPS)
        roll[120] = TILT_FAIL_RAD - 1e-6
        assert classify(z=z, roll=roll).outcome is Outcome.CENSORED


class TestFailure:
    def test_height_failure_ttf_at_first_crossing(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[150:] = 0.05
        result = classify(z=z)
        assert result.outcome is Outcome.FAILED
        assert result.ttf_s == pytest.approx((150 + 1) * DT)
        assert result.first_trigger == "torso_z"
        assert result.ttf_s is not None and result.ttf_s < 5.0

    def test_tilt_failure_via_pitch(self) -> None:
        pitch = np.zeros(STEPS)
        pitch[80:] = math.radians(75.0)
        result = classify(pitch=pitch)
        assert result.outcome is Outcome.FAILED
        assert result.ttf_s == pytest.approx((80 + 1) * DT)
        assert result.first_trigger == "tilt"

    def test_whichever_first_wins(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[200:] = 0.05  # height fails at 200
        roll = np.zeros(STEPS)
        roll[100:] = math.radians(90.0)  # tilt fails at 100 — earlier
        result = classify(z=z, roll=roll)
        assert result.first_trigger == "tilt"
        assert result.ttf_s == pytest.approx((100 + 1) * DT)

    def test_negative_roll_also_fails(self) -> None:
        roll = np.zeros(STEPS)
        roll[50:] = -math.radians(80.0)
        assert classify(roll=roll).outcome is Outcome.FAILED

    def test_failure_at_final_step_is_failure_not_censoring(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[-1] = 0.05
        result = classify(z=z)
        assert result.outcome is Outcome.FAILED
        assert result.ttf_s == pytest.approx(STEPS * DT)  # 5.0 exactly at last step


class TestDivergence:
    def test_nan_is_diverged_not_failed(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[60:] = np.nan
        result = classify(z=z)
        assert result.outcome is Outcome.DIVERGED
        assert result.diverged_at_s == pytest.approx((60 + 1) * DT)
        assert result.ttf_s is None

    def test_inf_in_any_track_is_diverged(self) -> None:
        roll = np.zeros(STEPS)
        roll[10] = np.inf
        assert classify(roll=roll).outcome is Outcome.DIVERGED

    def test_earliest_event_wins_fall_before_divergence(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[50:] = 0.05  # fall at 50
        z[150:] = np.nan  # NaN later (post-impact blowup)
        result = classify(z=z)
        assert result.outcome is Outcome.FAILED
        assert result.ttf_s == pytest.approx((50 + 1) * DT)

    def test_divergence_before_fall_is_diverged(self) -> None:
        z = np.full(STEPS, NOMINAL_Z)
        z[30:] = np.nan
        roll = np.zeros(STEPS)
        roll[100:] = math.radians(90.0)
        assert classify(z=z, roll=roll).outcome is Outcome.DIVERGED

    def test_nan_never_satisfies_failure_comparison(self) -> None:
        # A NaN-poisoned track must not read as sub-threshold height.
        z = np.full(STEPS, np.nan)
        assert classify(z=z).outcome is Outcome.DIVERGED


class TestValidation:
    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            classify_rollout(np.zeros(10), np.zeros(9), np.zeros(10), DT)

    def test_empty_tracks_raise(self) -> None:
        with pytest.raises(ValueError):
            classify_rollout(np.zeros(0), np.zeros(0), np.zeros(0), DT)
