"""Censored-survival estimators vs synthetic data with known ground truth (kickoff §4
required suite; §13 risk register)."""

import pytest

from reduce.classify import Classification, Outcome
from reduce.survival import cell_survival, kaplan_meier

HORIZON = 5.0


def failed(ttf: float) -> Classification:
    return Classification(
        outcome=Outcome.FAILED, ttf_s=ttf, diverged_at_s=None, first_trigger="tilt"
    )


def censored() -> Classification:
    return Classification(
        outcome=Outcome.CENSORED, ttf_s=None, diverged_at_s=None, first_trigger=None
    )


def diverged(t: float) -> Classification:
    return Classification(outcome=Outcome.DIVERGED, ttf_s=None, diverged_at_s=t, first_trigger=None)


class TestKaplanMeier:
    def test_no_censoring_matches_empirical_survivor_function(self) -> None:
        # 4 failures, no censoring: S steps 3/4, 2/4, 1/4, 0.
        curve = kaplan_meier([1.0, 2.0, 3.0, 4.0], [])
        assert [(p.time_s, p.survival) for p in curve] == [
            (1.0, 0.75),
            (2.0, 0.5),
            (3.0, 0.25),
            (4.0, 0.0),
        ]

    def test_textbook_intermediate_censoring(self) -> None:
        # n=5: fail@1 → S=4/5; censor@2 (risk set 4→3); fail@3 → S=4/5 * 2/3 = 8/15.
        curve = kaplan_meier([1.0, 3.0], [2.0, HORIZON, HORIZON])
        assert curve[0].survival == pytest.approx(0.8)
        assert curve[1].survival == pytest.approx(8.0 / 15.0)

    def test_tied_failure_times(self) -> None:
        # n=4, two failures at t=2: S = 1 - 2/4 = 0.5 in a single step.
        curve = kaplan_meier([2.0, 2.0], [HORIZON, HORIZON])
        assert len(curve) == 1
        assert curve[0].survival == pytest.approx(0.5)

    def test_all_censored_is_flat_one(self) -> None:
        assert kaplan_meier([], [HORIZON] * 8) == []

    def test_horizon_only_censoring_reduces_to_fraction(self) -> None:
        # Our sweep's structure: censoring only at the horizon → S(horizon) = survivors/n.
        curve = kaplan_meier([1.0, 2.0, 3.0], [HORIZON] * 13)
        assert curve[-1].survival == pytest.approx(13 / 16)


class TestCellSurvival:
    def test_sweep_like_cell(self) -> None:
        results = [failed(2.5), failed(2.7), failed(3.0)] + [censored()] * 13
        cell = cell_survival(results, HORIZON)
        assert cell.n_total == 16
        assert cell.survival_at_horizon == pytest.approx(13 / 16)
        assert cell.km_median_s is None  # never crosses 0.5
        assert cell.ttf_p25_s == pytest.approx(2.6)
        assert cell.ttf_p75_s == pytest.approx(2.85)

    def test_median_defined_when_majority_fails(self) -> None:
        results = [failed(t) for t in (2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8)] + [
            censored()
        ] * 7
        cell = cell_survival(results, HORIZON)
        # 16 at risk; S drops below 0.5 at the 9th failure? 8 failures → S=8/16=0.5
        # (<=0.5 at the 8th failure time, 2.7).
        assert cell.km_median_s == pytest.approx(2.7)

    def test_all_survive(self) -> None:
        cell = cell_survival([censored()] * 16, HORIZON)
        assert cell.survival_at_horizon == 1.0
        assert cell.km_median_s is None
        assert cell.ttf_p25_s is None

    def test_all_fail(self) -> None:
        cell = cell_survival([failed(1.0 + 0.1 * i) for i in range(16)], HORIZON)
        assert cell.survival_at_horizon == 0.0
        assert cell.km_median_s is not None

    def test_diverged_excluded_from_risk_set_but_counted(self) -> None:
        # 2 diverged + 7 failed + 7 censored: survival computed over the 14 non-diverged.
        results = (
            [diverged(1.0), diverged(2.0)]
            + [failed(2.0 + 0.1 * i) for i in range(7)]
            + [censored()] * 7
        )
        cell = cell_survival(results, HORIZON)
        assert cell.n_diverged == 2
        assert cell.survival_at_horizon == pytest.approx(7 / 14)

    def test_never_reports_ttf_equal_to_horizon_for_censored(self) -> None:
        # Censoring is not an event: a cell of censored rollouts has an empty KM curve,
        # not events at t=5 (spec §5 "never TTF = 5 s").
        cell = cell_survival([censored()] * 16, HORIZON)
        assert cell.km_curve == ()
