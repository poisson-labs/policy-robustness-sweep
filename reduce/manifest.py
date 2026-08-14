"""Build the per-cell survival manifest from sweep track data (M1; spec §3 reduce/).

Consumes the sweep's committed record metadata + the per-step track archives, classifies
every rollout with the FROZEN §5 thresholds, and emits one manifest: per-cell
censoring-aware survival estimates (KM), TTF IQR, diverged fraction — the data file the
heatmap renders from. Pure local computation (no simulation, no network): the surface is
reproducible from committed/archived data alone.

    uv run python -m reduce.manifest <tracks_dir> <sweep_manifest.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from reduce.classify import Classification, classify_rollout
from reduce.survival import cell_survival
from reduce.thresholds import ROLLOUT_S


def classify_all(tracks_dir: Path, chunk_starts: list[int], dt: float) -> list[Classification]:
    results: list[Classification] = []
    for c0 in chunk_starts:
        with np.load(tracks_dir / f"tracks-{c0:05d}.npz") as d:
            torso_z, roll, pitch = d["torso_z"], d["roll"], d["pitch"]
            for b in range(torso_z.shape[1]):
                results.append(classify_rollout(torso_z[:, b], roll[:, b], pitch[:, b], dt))
    return results


def build_manifest(tracks_dir: Path, sweep_manifest_path: Path) -> dict[str, Any]:
    sweep = json.loads(sweep_manifest_path.read_text())
    records = sweep["records"]
    dt = float(sweep["protocol"]["dt"])
    chunk = int(sweep["timings"]["chunk_rollouts"])
    chunk_starts = list(range(0, len(records), chunk))

    classifications = classify_all(tracks_dir, chunk_starts, dt)
    if len(classifications) != len(records):
        raise ValueError(f"{len(classifications)} tracks vs {len(records)} records")

    by_cell: dict[tuple[float, float], list[Classification]] = {}
    for record, result in zip(records, classifications, strict=True):
        by_cell.setdefault((record["mu"], record["push_pct_bw"]), []).append(result)

    cells: list[dict[str, Any]] = []
    for (mu, push), results in sorted(by_cell.items()):
        summary = cell_survival(results, ROLLOUT_S)
        entry: dict[str, Any] = {"mu": mu, "push_pct_bw": push} | asdict(summary)
        entry["km_curve"] = [list(point.values()) for point in entry["km_curve"]]
        cells.append(entry)

    return {
        "source_sweep": sweep["sweep_dir"],
        "run_id": sweep["run_id"],
        "grid": sweep["grid"],
        "protocol": sweep["protocol"],
        "thresholds": {"torso_z_fail_m": 0.15, "tilt_fail_deg": 60.0, "horizon_s": ROLLOUT_S},
        "cells": cells,
    }


def main() -> int:
    tracks_dir, sweep_manifest, out = (Path(a) for a in sys.argv[1:4])
    manifest = build_manifest(tracks_dir, sweep_manifest)
    out.write_text(json.dumps(manifest, indent=1))
    survivals = [c["survival_at_horizon"] for c in manifest["cells"]]
    print(
        f"cells: {len(survivals)}; all-survive: {sum(1 for s in survivals if s == 1.0)}; "
        f"all-fail: {sum(1 for s in survivals if s == 0.0)}; "
        f"boundary(0<S<1): {sum(1 for s in survivals if 0.0 < s < 1.0)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
