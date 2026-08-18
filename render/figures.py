"""KM survival-surface figures — one renderer, two variants (M1-05).

Variant A "repo": provenance record — full subtitle (thresholds, sweep ID, selection
legend), selected replay cells outlined, recovery diamonds marked.
Variant B "hero": public figure — no selection chrome, plain-language subtitle, quiet
regime annotations, sweep ID demoted to a footer.

Palette: matplotlib YlOrRd (survival mapped light→dark as danger rises). Perceptual
ordering verified numerically (CIELAB L* 98.9 → 25.9, strictly monotonic — survives
grayscale; DEVLOG Session 17). Both variants render from the committed manifest +
selection JSON with one command:

    uv run python -m render.figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

REPO = Path(__file__).parent.parent
MANIFEST = REPO / "docs/measurements/2026-08-13-cell-survival-manifest.json"
SELECTION = REPO / "docs/measurements/2026-08-14-replay-selection.json"
BOUNDARY = REPO / "docs/measurements/2026-08-15-boundary-uncertainty.json"
OUT_DIR = REPO / "docs/measurements/figures"

CMAP = colormaps["YlOrRd"]
INK = "#1a1a1a"
INK_MUTED = "#555555"
ANNOT_BLUE = "#1d4ed8"
SELECT_STROKE = "#111111"


def load_grid() -> tuple[list[float], list[float], np.ndarray, dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text())
    mus = manifest["grid"]["mus"]
    pushes = manifest["grid"]["push_pcts_bw"]
    surv = {(c["mu"], c["push_pct_bw"]): c["survival_at_horizon"] for c in manifest["cells"]}
    grid = np.array([[surv[(mu, p)] for p in pushes] for mu in mus])
    return mus, pushes, grid, manifest


def render(variant: str) -> list[Path]:
    mus, pushes, grid, manifest = load_grid()
    selection = json.loads(SELECTION.read_text())
    danger = 1.0 - grid  # light = safe, dark = fell

    fig, ax = plt.subplots(figsize=(9.6, 9.2), dpi=150)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.90, bottom=0.17)
    ax.pcolormesh(
        np.arange(len(pushes) + 1),
        np.arange(len(mus) + 1),
        danger,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(0.5, len(pushes), 2))
    ax.set_xticklabels([f"{int(p)}" for p in pushes[::2]], fontsize=9, color=INK)
    ax.set_yticks(np.arange(1.5, len(mus), 2))  # 0.10, 0.20, ... — labels round cleanly
    ax.set_yticklabels([f"{m:.1f}" for m in mus[1::2]], fontsize=9, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("floor friction μ", fontsize=11, color=INK)

    if variant == "repo":
        title = "Go1 survival surface — Kaplan-Meier S(5 s)"
        subtitle = (
            "frozen §5 thresholds (torso z < 0.15 m or tilt > 60°) · "
            f"sweep {manifest['source_sweep']}\n"
            "outlined = selected replay cells · ◆ = recovery replays"
        )
        xlabel = "push magnitude (% bodyweight; 100% = 125.0 N)"
    elif variant == "ribbon":
        title = "Where the boundary is — and how sure we are"
        subtitle = (
            "S = 0.5 crossing per friction row · 95% CI from 2000 seed-bootstrap resamples · "
            "16 attempts per cell"
        )
        xlabel = "push magnitude (% bodyweight)"
    else:
        title = "One policy, 400 worlds: where the robot falls"
        subtitle = "failure = torso below 0.15 m or tilted past 60° · 16 attempts per cell"
        xlabel = "push magnitude (% bodyweight)"
    ax.set_xlabel(xlabel, fontsize=11, color=INK, labelpad=10)
    fig.text(0.09, 0.965, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.09, 0.928, subtitle, fontsize=9.5, color=INK_MUTED, va="top", linespacing=1.4)

    def cell_center(mu: float, push: float) -> tuple[float, float]:
        return pushes.index(push) + 0.5, mus.index(mu) + 0.5

    if variant == "repo":
        for c in selection["boundary_cells"]:
            x, y = cell_center(c["mu"], c["push_pct_bw"])
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=SELECT_STROKE, linewidth=2.2
                )
            )
        for r in selection["recoveries"]:
            x, y = cell_center(r["mu"], r["push_pct_bw"])
            ax.plot(
                x,
                y,
                marker="D",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor=SELECT_STROKE,
                markeredgewidth=1.6,
            )

    if variant == "ribbon":
        # Bootstrap boundary as a RIBBON (95% CI band across friction rows), never a
        # line — the uncertainty is per-row and visibly varies (M1.6-A).
        bnd = json.loads(BOUNDARY.read_text())
        ys, med, lo, hi = [], [], [], []
        for row in bnd["rows"]:
            if "median_pct_bw" not in row:
                continue  # sentinel rows (no boundary in range) draw nothing
            ys.append(mus.index(row["mu"]) + 0.5)
            med.append(pushes.index(10.0) + (row["median_pct_bw"] - 10.0) / 10.0 + 0.5)
            lo.append(pushes.index(10.0) + (row["ci95_pct_bw"][0] - 10.0) / 10.0 + 0.5)
            hi.append(pushes.index(10.0) + (row["ci95_pct_bw"][1] - 10.0) / 10.0 + 0.5)
        ax.fill_betweenx(ys, lo, hi, color="#1d4ed8", alpha=0.22, linewidth=0)
        ax.plot(med, ys, color="#1d4ed8", lw=1.6, alpha=0.9)
        # label in the light (safe) region, left of the band, where it is legible
        ax.text(
            2.0,
            mus.index(0.85) + 0.5,
            "boundary (S = 0.5)\nwith 95% bootstrap CI",
            fontsize=8.5,
            color="#1d4ed8",
            va="center",
        )

    if variant in ("hero", "ribbon"):
        # quiet regime annotations (the "transition band" label is redundant with the
        # ribbon itself, so the ribbon variant draws only the ice arrow)
        ax.annotate(
            "ice: walking fails unaided",
            xy=(1.2, 1.4),
            xytext=(2.6, 3.4),
            fontsize=9,
            color=INK_MUTED,
            style="italic",
            arrowprops={"arrowstyle": "->", "color": INK_MUTED, "lw": 1.0},
        )
        if variant == "hero":
            ax.text(
                4.6,
                9.0,
                "transition band",
                fontsize=9,
                color=INK_MUTED,
                style="italic",
                rotation=72,
                ha="center",
            )
        fig.text(
            0.86,
            0.055,
            f"data: sweep {manifest['source_sweep']}\n6400 rollouts, measured",
            fontsize=7.5,
            color=INK_MUTED,
            ha="right",
        )

    # DR band bracket (right margin, mu 0.4..1.0) with tick terminators
    y0 = mus.index(0.4) + 0.0
    y1 = mus.index(1.0) + 1.0
    xb = len(pushes) + 0.45
    ax.plot([xb, xb], [y0, y1], color=ANNOT_BLUE, lw=2.5, clip_on=False)
    for yy in (y0, y1):
        ax.plot([xb - 0.28, xb + 0.28], [yy, yy], color=ANNOT_BLUE, lw=2.0, clip_on=False)
    ax.text(
        xb + 0.6,
        (y0 + y1) / 2,
        "training DR\nfriction band\nU(0.4, 1.0)",
        fontsize=8.5,
        color=ANNOT_BLUE,
        va="center",
        clip_on=False,
    )
    y_def = mus.index(0.6) + 0.5
    ax.axhline(y_def, color="#0ea5e9", lw=1.0, ls=(0, (4, 4)), alpha=0.85)
    ax.text(
        xb + 0.6,
        y_def,
        "model default\nμ = 0.60",
        fontsize=8.5,
        color="#0ea5e9",
        va="center",
        clip_on=False,
    )

    # continuous colorbar, its own row, endpoint labels above the bar ends
    cax = fig.add_axes((0.09, 0.055, 0.5, 0.022))
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=CMAP.reversed(), norm=plt.Normalize(0, 1)),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=8, colors=INK, length=3)
    cbar.outline.set_visible(False)
    fig.text(0.09, 0.088, "S = 0 (all fall)", fontsize=8, color=INK_MUTED, ha="left")
    fig.text(0.59, 0.088, "S = 1 (all survive)", fontsize=8, color=INK_MUTED, ha="right")
    cax.set_xlabel("probability of surviving the full 5 s", fontsize=8.5, color=INK_MUTED)

    OUT_DIR.mkdir(exist_ok=True)
    written = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"km-surface-{variant}.{ext}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    written = render("repo") + render("hero") + render("ribbon")
    for path in written:
        print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
