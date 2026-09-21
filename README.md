# policy-robustness-sweep

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Poisson Labs Research](https://img.shields.io/badge/Poisson%20Labs-Research-teal.svg)](https://poissonlabs.ai/research/map-the-failure-boundary/)

Code, measurements and figures behind the Poisson Labs post
**[Map the Failure Boundary](https://poissonlabs.ai/research/map-the-failure-boundary/)**.

It trains a Unitree Go1 walking policy in MuJoCo Playground and measures where the policy falls over on a
20 × 20 grid of floor friction and lateral push. It retrains the policy with one change to the training
distribution, measures again, and compares the two maps cell by cell. A probe app runs any single world on
demand and returns a 3D replay.

## What it does

A world is a floor friction, a lateral push (size as a percentage of body weight, direction, start time) and a
seed. A rollout is 5 s of the policy walking in one world. It counts as a fall if the torso drops below 0.15 m
or tilts more than 60°; a rollout that is still up at 5 s is censored, not counted as a fall at 5 s. Each cell
of the grid gets 16 or 32 seeds, and its survival is estimated with Kaplan–Meier. The failure boundary of a
friction row is where survival crosses 0.5, with a bootstrap interval.

The grid runs friction from 0.05 to 1.00 and push from 10% to 200% of body weight, 20 values each. Rollouts run
on Modal GPUs; everything downstream of the rollout records runs locally.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/poisson-labs/policy-robustness-sweep.git
cd policy-robustness-sweep
uv sync
```

## Run

### Tests and checks

The same steps CI runs:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run python -m ci.check_rerun_versions
uv run python -m ci.check_measured_numbers
```

### Recompute the v0 → v1 comparison from the committed sweeps

The two 32-seed sweeps are committed as per-rollout records. This rebuilds both survival surfaces, the boundary
intervals and the per-cell comparison, in about six seconds:

```bash
uv run python -m reduce.compare \
  docs/measurements/2026-08-18-g4-v0-32-sweep-manifest.json \
  docs/measurements/2026-08-18-g4-v1-32-sweep-manifest.json \
  v0-v1-diff.json
```

The result is the file committed as `docs/measurements/2026-08-18-v0-v1-diff.json`.

Both sweeps ran the same 32 seeds in every cell. The test is Fisher's exact test per cell, corrected across all
400 cells (Benjamini–Hochberg, q < 0.05), which compares the two sweeps as separate samples and does not use
the matching seeds. It finds 60 cells safer and none worse. The measurement is written up in
[`docs/measurements/2026-08-18-m1_5-retrain-diff.md`](docs/measurements/2026-08-18-m1_5-retrain-diff.md).

### Redraw the survival-surface figures

```bash
uv run python -m render.figures
```

Rewrites the six `docs/measurements/figures/km-surface-*` files from the committed manifest and replay
selection.

### On Modal

The steps below need a Modal account and GPU time, which costs money. They are written as each module documents
them and were not run when this README was written; the committed files under
[`docs/measurements/`](docs/measurements/) are the record of the runs that were made. Run ids look like
`20260813T155006Z`.

```bash
uv run modal run train/modal_app.py --smoke true            # short check of the training image
uv run modal run train/modal_app.py                         # train the v0 policy
uv run modal run train/retrain_v1.py                        # train v1, with the changed training distribution
uv run modal run sweep/equivalence.py --run-id <run-id>     # the sweep environment against the policy's own evaluation
uv run modal run sweep/full_sweep.py --run-id <run-id>      # the 20 x 20 sweep
uv run modal run probe/instrumented.py --run-id <run-id> --mu 0.5 --push-pct 90   # one replay
uv run modal deploy app/server.py                           # the probe app
```

## Version pins

Local Python versions are in `uv.lock`. The Modal training image (`train/modal_app.py`) pins the packages below,
as recorded from the smoke run's resolved environment
([`docs/measurements/2026-08-13-g2-smoke-run.md`](docs/measurements/2026-08-13-g2-smoke-run.md)):

| package | version |
|---|---|
| playground | 0.2.0 |
| brax | 0.14.2 |
| jax / jaxlib (cuda12) | 0.9.2 |
| mujoco / mujoco-mjx | 3.11.0 |

jax is held below 0.10.0, which removed `jax.device_put_replicated`; brax 0.14.2 still calls it. The Rerun SDK is
pinned at 0.36.0 and the vendored web viewer's version is the same; CI checks that the two agree.

## Layout

| Path | Contents |
|---|---|
| `configs/` | `WorldConfig`, the one definition of a world, with the clamping and rounding that the sweep, the probe and the app share. |
| `train/` | Training on Modal: the v0 policy (`modal_app.py`), the retrained v1 policy (`retrain_v1.py`), the nominal check (`verify_nominal.py`), and the capture of the domain-randomisation source each policy was trained with. |
| `sweep/` | The rollout executors: physics-only push and friction injection, the equivalence check, the pilot, the full sweep, the solver-sensitivity slice, and the seed scheme. |
| `reduce/` | Local statistics over sweep records: failure classification and frozen thresholds, Kaplan–Meier survival, the cell manifest, boundaries with bootstrap intervals, replay selection, and the two-sweep comparison. |
| `render/` | Survival-surface figures and offline clips. |
| `probe/` | One world in, one replay out: instrumented rollouts logged to Rerun `.rrd` files, the live probe, and the gate test server. |
| `app/` | The Modal web app: `POST /probe` with clamping, caching, per-IP and spend limits and a read-only degraded mode; `/health`, `/receipt`, `/manifest`. `core.py` is the tested logic, `server.py` the Modal wrapper. |
| `web/` | The embeddable bundle for the site and the pinned Rerun viewer's version and hashes. |
| `ci/` | Two CI checks: the Rerun SDK and viewer versions agree, and post drafts hold no unsourced numbers. |
| `tests/` | The test suite. |
| `docs/` | The specification, the measurements, the equivalence report and other records; see [`docs/README.md`](docs/README.md). |

Run outputs that are not committed (`*.rrd`, checkpoints, the fetched viewer) are git-ignored.

## Licence

MIT. See [`LICENSE`](LICENSE). Third-party software, the robot model and the source text embedded in two
records, and their terms, are in [`NOTICE`](NOTICE).

## Citation

```bibtex
@misc{kolasinski2026failureboundary,
  author = {Kolasinski, Taylor},
  title  = {Map the Failure Boundary},
  year   = {2026},
  month  = {August},
  howpublished = {\url{https://poissonlabs.ai/research/map-the-failure-boundary/}},
  note   = {Poisson Labs research report; source code at \url{https://github.com/poisson-labs/policy-robustness-sweep}}
}
```
