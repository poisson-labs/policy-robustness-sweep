"""Two-sweep surface diff (M1.6-B; used by M1.5).

Per cell, compares two sweeps of the SAME grid/thresholds/command:
- primary: Fisher's exact test on fall counts (2x2: fell/survived x sweep A/B);
- secondary: log-rank test on the KM curves (TTF timing, censoring-aware);
- Benjamini-Hochberg FDR across all cells (400 on the v1 grid).

Output is per-cell p/q values + direction, i.e. a significance-MASK layer for the diff
figure — not a threshold-crossing verdict. Pure numpy/scipy over committed data; ground-
truth unit tests in tests/test_diff.py (identical sweeps → ~0 discoveries; injected known
shifts → detected at expected power). Diverged rollouts are excluded from both tests
(matches cell_survival semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from scipy import stats  # type: ignore[import-untyped]  — scipy ships no stubs

from reduce.classify import Classification, Outcome


def _fisher_p(table: list[list[int]]) -> float:
    result = cast(Any, stats).fisher_exact(table)
    return float(result.pvalue)


def _chi2_sf(x: float, df: int) -> float:
    return float(cast(Any, stats.chi2).sf(x, df=df))


@dataclass(frozen=True)
class CellDiff:
    mu: float
    push_pct_bw: float
    n_a: int
    n_b: int
    fell_a: int
    fell_b: int
    delta_survival: float  # S_B - S_A (positive = B safer)
    fisher_p: float
    logrank_p: float | None  # None when neither arm has any failure
    fisher_q: float = float("nan")  # BH-adjusted (filled by diff_surfaces)
    logrank_q: float | None = None


def _counts(results: list[Classification]) -> tuple[int, int]:
    live = [r for r in results if r.outcome is not Outcome.DIVERGED]
    fell = sum(1 for r in live if r.outcome is Outcome.FAILED)
    return fell, len(live)


def logrank_test(
    a: list[Classification], b: list[Classification], horizon_s: float
) -> float | None:
    """Two-sample log-rank p-value (chi-square, 1 df). Right-censored at horizon."""
    times: list[float] = []
    events: list[bool] = []
    groups: list[int] = []
    for g, arm in enumerate((a, b)):
        for r in arm:
            if r.outcome is Outcome.DIVERGED:
                continue
            if r.outcome is Outcome.FAILED and r.ttf_s is not None:
                times.append(r.ttf_s)
                events.append(True)
            else:
                times.append(horizon_s)
                events.append(False)
            groups.append(g)
    t = np.asarray(times)
    e = np.asarray(events)
    grp = np.asarray(groups)
    if e.sum() == 0:
        return None
    o_a = 0.0
    e_a = 0.0
    v = 0.0
    for tt in np.unique(t[e]):
        at_risk = t >= tt
        n = at_risk.sum()
        n_a = (at_risk & (grp == 0)).sum()
        d = (e & (t == tt)).sum()
        d_a = (e & (t == tt) & (grp == 0)).sum()
        o_a += d_a
        e_a += d * n_a / n
        if n > 1:
            v += d * (n_a / n) * (1 - n_a / n) * (n - d) / (n - 1)
    if v <= 0:
        return 1.0
    chi2 = (o_a - e_a) ** 2 / v
    return _chi2_sf(chi2, df=1)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values (monotone), same order as input."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q.tolist()


def diff_surfaces(
    a: dict[tuple[float, float], list[Classification]],
    b: dict[tuple[float, float], list[Classification]],
    horizon_s: float,
) -> list[CellDiff]:
    """Per-cell Fisher (primary) + log-rank (secondary), BH-FDR across cells."""
    keys = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError("sweeps must cover the same grid (same cell keys)")
    cells: list[CellDiff] = []
    for mu, push in keys:
        fell_a, n_a = _counts(a[(mu, push)])
        fell_b, n_b = _counts(b[(mu, push)])
        table = [[fell_a, n_a - fell_a], [fell_b, n_b - fell_b]]
        fisher_p = _fisher_p(table)
        s_a = 1.0 - fell_a / n_a if n_a else 1.0
        s_b = 1.0 - fell_b / n_b if n_b else 1.0
        cells.append(
            CellDiff(
                mu=mu,
                push_pct_bw=push,
                n_a=n_a,
                n_b=n_b,
                fell_a=fell_a,
                fell_b=fell_b,
                delta_survival=s_b - s_a,
                fisher_p=fisher_p,
                logrank_p=logrank_test(a[(mu, push)], b[(mu, push)], horizon_s),
            )
        )
    fisher_q = benjamini_hochberg([c.fisher_p for c in cells])
    lr_idx = [i for i, c in enumerate(cells) if c.logrank_p is not None]
    lr_q = benjamini_hochberg([cast(float, cells[i].logrank_p) for i in lr_idx])
    lr_q_map = dict(zip(lr_idx, lr_q, strict=True))
    return [
        replace(c, fisher_q=fisher_q[i], logrank_q=lr_q_map.get(i)) for i, c in enumerate(cells)
    ]


def detectable_fall_swing(n_per_arm: int, q: float = 0.05, m_cells: int = 400) -> dict[str, float]:
    """Smallest fall-count swing Fisher's exact detects at BH-adjusted q, roughly.

    Conservative shortcut: BH at q with m tests and (say) ~40 true effects behaves like
    a per-test alpha ≈ q * 40 / m; report the swing needed vs an S=0.5 baseline cell
    (the hardest case) at that alpha, by scanning tables. Used ONLY as a decision input
    for the seeds question (DEVLOG), never as a published number.
    """
    alpha = q * 40 / m_cells
    base_fell = n_per_arm // 2
    for swing in range(1, n_per_arm - base_fell + 1):
        table = [
            [base_fell, n_per_arm - base_fell],
            [base_fell + swing, n_per_arm - base_fell - swing],
        ]
        if _fisher_p(table) <= alpha:
            return {
                "n_per_arm": n_per_arm,
                "effective_alpha": alpha,
                "min_detectable_swing": swing,
                "as_delta_S": swing / n_per_arm,
            }
    return {
        "n_per_arm": n_per_arm,
        "effective_alpha": alpha,
        "min_detectable_swing": float("nan"),
        "as_delta_S": float("nan"),
    }
