"""Two-sweep diff test vs synthetic sweeps with known shifts (M1.6-B)."""

from collections.abc import Callable

import numpy as np

from reduce.classify import Classification, Outcome
from reduce.diff import benjamini_hochberg, detectable_fall_swing, diff_surfaces, logrank_test

HORIZON = 5.0
MUS = [round(0.05 * i, 2) for i in range(1, 21)]
PUSHES = [float(10 * i) for i in range(1, 21)]


def failed(ttf: float) -> Classification:
    return Classification(Outcome.FAILED, ttf, None, "tilt")


def censored() -> Classification:
    return Classification(Outcome.CENSORED, None, None, None)


Sweep = dict[tuple[float, float], list[Classification]]
PFall = Callable[[float, float], float]


def sweep(p_fall_fn: PFall, n: int, seed: int, ttf_mean: float = 2.5) -> Sweep:
    rng = np.random.default_rng(seed)
    out: Sweep = {}
    for mu in MUS:
        for push in PUSHES:
            p = p_fall_fn(mu, push)
            out[(mu, push)] = [
                failed(float(rng.normal(ttf_mean, 0.2))) if rng.random() < p else censored()
                for _ in range(n)
            ]
    return out


def sigmoid_surface(mu: float, push: float, shift: float = 0.0) -> float:
    boundary = 40.0 + 70.0 * mu + shift  # diagonal band like the real map
    return float(1.0 / (1.0 + np.exp(-(push - boundary) / 8.0)))


class TestBH:
    def test_monotone_and_bounded(self) -> None:
        q = benjamini_hochberg([0.001, 0.02, 0.5, 0.9])
        assert all(0 <= v <= 1 for v in q)
        assert q[0] <= q[1] <= q[2] <= q[3]

    def test_uniform_nulls_rarely_pass(self) -> None:
        rng = np.random.default_rng(0)
        q = benjamini_hochberg(list(rng.random(400)))
        assert sum(1 for v in q if v < 0.05) <= 2


class TestLogRank:
    def test_identical_arms_high_p(self) -> None:
        a = [failed(2.0 + 0.1 * i) for i in range(8)] + [censored()] * 8
        assert logrank_test(a, list(a), HORIZON) > 0.5  # type: ignore[operator]

    def test_earlier_failures_low_p(self) -> None:
        a = [failed(1.0 + 0.05 * i) for i in range(16)]
        b = [failed(3.5 + 0.05 * i) for i in range(16)]
        assert logrank_test(a, b, HORIZON) < 0.001  # type: ignore[operator]

    def test_no_events_returns_none(self) -> None:
        assert logrank_test([censored()] * 16, [censored()] * 16, HORIZON) is None


class TestDiffSurfaces:
    def test_identical_sweeps_yield_near_zero_discoveries(self) -> None:
        a = sweep(sigmoid_surface, n=16, seed=1)
        b = sweep(sigmoid_surface, n=16, seed=2)  # same distribution, fresh draws
        cells = diff_surfaces(a, b, HORIZON)
        assert len(cells) == 400
        assert sum(1 for c in cells if c.fisher_q < 0.05) <= 4  # ≈ q·m false discoveries

    def test_large_shift_detected_in_band(self) -> None:
        a = sweep(sigmoid_surface, n=32, seed=1)
        b = sweep(lambda mu, p: sigmoid_surface(mu, p, shift=30.0), n=32, seed=2)
        cells = diff_surfaces(a, b, HORIZON)
        sig = [c for c in cells if c.fisher_q < 0.05]
        assert len(sig) >= 40, f"only {len(sig)} significant cells for a 30%BW shift"
        assert all(c.delta_survival > 0 for c in sig)  # direction: B safer

    def test_grid_mismatch_raises(self) -> None:
        a = sweep(sigmoid_surface, n=8, seed=1)
        b = dict(list(a.items())[:-1])
        try:
            diff_surfaces(a, b, HORIZON)
        except ValueError:
            return
        raise AssertionError("expected ValueError on grid mismatch")


class TestPowerHelper:
    def test_more_seeds_detect_smaller_delta_S(self) -> None:
        d16 = detectable_fall_swing(16)
        d32 = detectable_fall_swing(32)
        assert d32["as_delta_S"] < d16["as_delta_S"]
