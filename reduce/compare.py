"""Two-sweep comparison (M1.5-03): v0 vs v1 on identical grid/thresholds/seeds.

Consumes two sweep manifests (per-rollout records; M0-07 validated 6400/6400 category
agreement of these records with the frozen classifier), and produces:
- both KM cell manifests (reduce/survival), boundary bootstrap per surface (M1.6-A);
- the per-cell diff: Fisher primary + log-rank secondary + BH-FDR mask (M1.6-B) with
  direction; row-wise boundary displacement with both bootstrap CIs;
- shrinkage census: cells where v1 is significantly WORSE (reported wherever it appears —
  Session 22 honesty guard).

Pure local computation over committed data. Diff hygiene asserted: same grid, same
seeds_per_cell, same command/protocol fields.

    uv run python -m reduce.compare <v0_manifest.json> <v1_manifest.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reduce.boundary import bootstrap_row_boundaries, headline_ci_width
from reduce.classify import Classification, Outcome
from reduce.diff import diff_surfaces
from reduce.survival import cell_survival
from reduce.thresholds import ROLLOUT_S

Outcomes = dict[tuple[float, float], list[Classification]]


def outcomes_from_records(records: list[dict[str, Any]]) -> Outcomes:
    out: Outcomes = {}
    for r in records:
        if r["diverged_at_s"] is not None:
            c = Classification(Outcome.DIVERGED, None, r["diverged_at_s"], None)
        elif r["fell_time_upz_s"] is not None:
            c = Classification(Outcome.FAILED, r["fell_time_upz_s"], None, "tilt")
        else:
            c = Classification(Outcome.CENSORED, None, None, None)
        out.setdefault((r["mu"], r["push_pct_bw"]), []).append(c)
    return out


def assert_hygiene(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key in ("mus", "push_pcts_bw", "seeds_per_cell"):
        if a["grid"][key] != b["grid"][key]:
            raise ValueError(f"grid mismatch on {key}")
    for key in ("direction_deg", "push_start_s", "rollout_s", "command", "dt"):
        if a["protocol"][key] != b["protocol"][key]:
            raise ValueError(f"protocol mismatch on {key}")


def surface(outcomes: Outcomes, mus: list[float], pushes: list[float]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for mu in mus:
        for push in pushes:
            s = cell_survival(outcomes[(mu, push)], ROLLOUT_S)
            cells.append(
                {
                    "mu": mu,
                    "push_pct_bw": push,
                    "survival_at_horizon": s.survival_at_horizon,
                    "n_failed": s.n_failed,
                    "n_censored": s.n_censored,
                    "n_diverged": s.n_diverged,
                    "km_median_s": s.km_median_s,
                    "ttf_p25_s": s.ttf_p25_s,
                    "ttf_p75_s": s.ttf_p75_s,
                }
            )
    rows = bootstrap_row_boundaries(outcomes, mus, pushes, ROLLOUT_S, n_resamples=2000, seed=0)
    return {
        "cells": cells,
        "boundary_rows": [asdict(r) for r in rows],
        "boundary_headline_ci_width_pct_bw": headline_ci_width(rows),
        "n_all_survive": sum(1 for c in cells if c["survival_at_horizon"] == 1.0),
        "n_all_fail": sum(1 for c in cells if c["survival_at_horizon"] == 0.0),
        "n_boundary": sum(1 for c in cells if 0.0 < c["survival_at_horizon"] < 1.0),
    }


def compare(v0_path: Path, v1_path: Path, q: float = 0.05) -> dict[str, Any]:
    v0 = json.loads(v0_path.read_text())
    v1 = json.loads(v1_path.read_text())
    assert_hygiene(v0, v1)
    mus, pushes = v0["grid"]["mus"], v0["grid"]["push_pcts_bw"]
    o0 = outcomes_from_records(v0["records"])
    o1 = outcomes_from_records(v1["records"])
    s0 = surface(o0, mus, pushes)
    s1 = surface(o1, mus, pushes)
    diff = diff_surfaces(o0, o1, ROLLOUT_S)
    sig = [c for c in diff if c.fisher_q < q]
    better = [c for c in sig if c.delta_survival > 0]
    worse = [c for c in sig if c.delta_survival < 0]
    # row-wise boundary displacement (v1 - v0) with both CIs
    b0 = {r["mu"]: r for r in s0["boundary_rows"]}
    b1 = {r["mu"]: r for r in s1["boundary_rows"]}
    displacement: list[dict[str, Any]] = []
    for mu in mus:
        r0, r1 = b0[mu], b1[mu]
        displacement.append(
            {
                "mu": mu,
                "v0_median_pct_bw": r0["median_pct_bw"],
                "v0_ci95": [r0["ci_low_pct_bw"], r0["ci_high_pct_bw"]],
                "v1_median_pct_bw": r1["median_pct_bw"],
                "v1_ci95": [r1["ci_low_pct_bw"], r1["ci_high_pct_bw"]],
                "delta_median_pct_bw": (
                    None
                    if r0["median_pct_bw"] is None or r1["median_pct_bw"] is None
                    else round(r1["median_pct_bw"] - r0["median_pct_bw"], 2)
                ),
            }
        )
    return {
        "v0": {"sweep_dir": v0["sweep_dir"], "run_id": v0["run_id"]},
        "v1": {"sweep_dir": v1["sweep_dir"], "run_id": v1["run_id"]},
        "grid": v0["grid"],
        "protocol": v0["protocol"],
        "significance": {"method": "Fisher exact per cell + BH-FDR", "q": q},
        "v0_surface": s0,
        "v1_surface": s1,
        "cells_diff": [asdict(c) for c in diff],
        "n_significant": len(sig),
        "n_better": len(better),
        "n_worse": len(worse),
        "worse_cells": [
            {
                "mu": c.mu,
                "push_pct_bw": c.push_pct_bw,
                "delta_S": round(c.delta_survival, 3),
                "q": c.fisher_q,
            }
            for c in worse
        ],
        "boundary_displacement": displacement,
    }


def main() -> int:
    v0, v1, out = (Path(a) for a in sys.argv[1:4])
    result = compare(v0, v1)
    out.write_text(json.dumps(result, indent=1))
    print(
        f"significant cells at q=0.05: {result['n_significant']} "
        f"(v1 better: {result['n_better']}, v1 worse: {result['n_worse']})"
    )
    nb0, nb1 = result["v0_surface"]["n_boundary"], result["v1_surface"]["n_boundary"]
    print(f"v0 boundary cells {nb0} -> v1 {nb1}")
    for d in result["boundary_displacement"]:
        v0m, v1m, dm = d["v0_median_pct_bw"], d["v1_median_pct_bw"], d["delta_median_pct_bw"]
        print(f"  mu={d['mu']:.2f}: v0 {v0m} -> v1 {v1m}  delta {dm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
