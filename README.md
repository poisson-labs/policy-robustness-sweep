# policy-robustness-sweep

**"One Policy, 400 Worlds" / "A Robot You Can Shove"** — Poisson Labs flagship.

A trained quadruped locomotion policy (MuJoCo Playground, Unitree Go1) whose failure
boundary is mapped across a friction × lateral-push grid with censored-survival statistics,
probeable live at any off-grid world via on-demand MJX rollouts on Modal, with Rerun 3D
replays and a measured cost receipt.

## Source of truth

- [docs/spec-one-policy-400-worlds.md](docs/spec-one-policy-400-worlds.md) — the technical
  spec. §4 gates are blocking and ordered; §2 non-goals are binding.
- [docs/TASKS.md](docs/TASKS.md) — task tracker (current state of the work).
- [docs/DEVLOG.md](docs/DEVLOG.md) — dated log of work, measurements, and decisions.
- [docs/NOTES.md](docs/NOTES.md) — research notes, parked ideas, open questions.

## Development

Python 3.12+, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pre-commit install
uv run pytest
```

CI runs ruff (format + lint), pyright, pytest, a Rerun SDK/viewer version assertion, and a
gate that rejects unsourced numbers in post drafts (`docs/drafts/`). Every published number
in this project is measured — placeholders are written `MEASURED_TBD`.

## Version pins

All local Python pins live in `uv.lock` (committed). The training image
(`train/modal_app.py`) pins, verified from the smoke run's recorded resolved environment
(docs/measurements/2026-08-13-g2-smoke-run.md):

| package | version |
|---|---|
| playground | 0.2.0 |
| brax | 0.14.2 |
| jax / jaxlib (cuda12) | 0.9.2 |
| mujoco / mujoco-mjx | 3.11.0 |

(jax is capped below 0.10.0, which removed `jax.device_put_replicated` that brax 0.14.2
still calls.) Rerun SDK/viewer pins arrive with G5/M3.

## Attribution

This project builds on MuJoCo Playground, MuJoCo Menagerie (Unitree Go1 model), Brax,
Rerun, and Modal. Full license attribution lands with the artifact (spec §8).
