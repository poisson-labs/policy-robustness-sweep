"""Censored-survival statistics per cell (spec §5/§6; §13 risk: "censoring mishandled →
biased surface").

Kaplan-Meier product-limit estimator, implemented directly (DECISION, DEVLOG Session 14:
~40 lines of arithmetic under pyright strict with hand-computed ground-truth tests beat
a heavy dependency; our censoring structure is horizon-only today but the estimator is
general so probe-era data can't silently bias it).

Semantics locked by spec §5:
- survival past the horizon → right-censored (never an event at the horizon);
- diverged rollouts are a DISTINCT category: excluded from the survival risk set
  entirely, reported as a per-cell fraction;
- per-cell reports: censoring-aware survival-at-horizon (the heatmap value), KM median
  survival where defined, TTF IQR among failures (the companion variance figure).
"""

from __future__ import annotations

from dataclasses import dataclass

from reduce.classify import Classification, Outcome


@dataclass(frozen=True)
class KMPoint:
    time_s: float
    survival: float  # S(t) just after this event time


@dataclass(frozen=True)
class CellSurvival:
    n_total: int
    n_failed: int
    n_censored: int
    n_diverged: int
    survival_at_horizon: float
    km_median_s: float | None  # first event time with S(t) <= 0.5; None if never reached
    ttf_p25_s: float | None  # IQR among observed failures; None if no failures
    ttf_p75_s: float | None
    km_curve: tuple[KMPoint, ...]


def kaplan_meier(event_times: list[float], censor_times: list[float]) -> list[KMPoint]:
    """Product-limit estimator. `event_times` are observed failures; `censor_times` are
    right-censoring times (subjects known to survive at least that long)."""
    at_risk = len(event_times) + len(censor_times)
    if at_risk == 0:
        return []
    marks = sorted([(t, True) for t in event_times] + [(t, False) for t in censor_times])
    curve: list[KMPoint] = []
    survival = 1.0
    i = 0
    while i < len(marks):
        t = marks[i][0]
        deaths = 0
        censored = 0
        while i < len(marks) and marks[i][0] == t:
            if marks[i][1]:
                deaths += 1
            else:
                censored += 1
            i += 1
        if deaths > 0:
            survival *= 1.0 - deaths / at_risk
            curve.append(KMPoint(time_s=t, survival=survival))
        at_risk -= deaths + censored
    return curve


def _percentile_sorted(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' method) on a sorted list."""
    if not sorted_values:
        raise ValueError("empty")
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def cell_survival(results: list[Classification], horizon_s: float) -> CellSurvival:
    """Per-cell summary from classified rollouts (spec §5 semantics; see module doc)."""
    diverged = [r for r in results if r.outcome is Outcome.DIVERGED]
    failed = [r for r in results if r.outcome is Outcome.FAILED]
    censored = [r for r in results if r.outcome is Outcome.CENSORED]

    ttfs = sorted(r.ttf_s for r in failed if r.ttf_s is not None)
    curve = kaplan_meier(ttfs, [horizon_s] * len(censored))
    survival_at_horizon = curve[-1].survival if curve else 1.0
    median = next((p.time_s for p in curve if p.survival <= 0.5), None)

    return CellSurvival(
        n_total=len(results),
        n_failed=len(failed),
        n_censored=len(censored),
        n_diverged=len(diverged),
        survival_at_horizon=survival_at_horizon,
        km_median_s=median,
        ttf_p25_s=_percentile_sorted(ttfs, 0.25) if ttfs else None,
        ttf_p75_s=_percentile_sorted(ttfs, 0.75) if ttfs else None,
        km_curve=tuple(curve),
    )
