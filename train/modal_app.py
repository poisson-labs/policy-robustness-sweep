"""Modal training app for G2: Go1 joystick locomotion per the Playground recipe.

Run (after `modal token` auth + Taylor's spend-alert confirmation):

    uv run modal run train/modal_app.py --smoke true    # short end-to-end validation run
    uv run modal run train/modal_app.py                 # full paid training run

The smoke run uses the FULL recipe configuration (same num_envs, so the real memory
footprint is exercised) but truncates num_timesteps, and validates: image build + resolve,
GPU visibility, env load, training loop, checkpoint writes to the Volume, metrics logging,
and verbatim DR capture. Only after a green smoke run does the full run get launched.

Wiring mirrors mujoco_playground/learning/train_jax_ppo.py (fetched 2026-08-13):
registry.load → locomotion_params.brax_ppo_config → brax ppo.train with
randomization_fn=registry.get_domain_randomizer(...), wrapper for episode handling.

Version pins: top-level packages pinned exactly below. jax is constrained (not exactly
pinned) for the first smoke run because playground 0.2.0's docs recommend jax[cuda12]
while jax's current release moved to cuda13 extras — the smoke run records the full
resolved environment (pip freeze) to the Volume, and the exact pins get committed from
that record before the full run (DEVLOG 2026-08-13). A runtime GPU assertion guards
against a silent CPU-only jax install.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import modal

ENV_NAME = "Go1JoystickFlatTerrain"
VOLUME_NAME = "opw-checkpoints"
VOLUME_MOUNT = "/vol"

# Raised from the recipe's num_evals=10 for checkpoint granularity (more, earlier
# intermediate checkpoints — "you can't go back for them after the run"). This changes
# evaluation/checkpoint cadence only, not optimization hyperparameters.
NUM_EVALS = 20

SMOKE_NUM_TIMESTEPS = 2_000_000

app = modal.App("opw-train")

checkpoints = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

train_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "playground==0.2.0",
    "brax==0.14.2",
    "jax[cuda12]<0.12",
)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=4 * 3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def train(smoke: bool = False) -> dict[str, Any]:
    import subprocess
    from pathlib import Path

    import jax
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    from train.dr_record import capture_go1_dr_record

    # Guard against a silent CPU-only jax resolution (see module docstring).
    platforms = {d.platform for d in jax.devices()}
    assert "gpu" in platforms, f"jax sees no GPU (devices: {jax.devices()}) — bad image resolve"

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ("-smoke" if smoke else "")
    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Record the fully resolved environment — source for the exact pins we commit.
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--system"], capture_output=True, text=True, check=False
    )
    frozen = freeze.stdout if freeze.returncode == 0 else ""
    if not frozen:
        pip_freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        frozen = pip_freeze.stdout
    (run_dir / "resolved-environment.txt").write_text(frozen)

    # Verbatim DR/config capture (spec §8) — fail the run if capture fails.
    dr_record = capture_go1_dr_record()
    (run_dir / "dr-record.json").write_text(json.dumps(dr_record, indent=2))

    env_cfg = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=env_cfg)
    eval_env = registry.load(ENV_NAME, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    ppo_params.num_evals = NUM_EVALS
    if smoke:
        # Truncated duration, FULL num_envs/batch config so the real memory footprint and
        # checkpoint path are exercised end-to-end in two eval segments.
        ppo_params.num_timesteps = SMOKE_NUM_TIMESTEPS
        ppo_params.num_evals = 2

    training_params = dict(ppo_params)
    network_config = training_params.pop("network_factory", None)
    del training_params["num_evals"]  # passed explicitly below

    def network_factory(*args: Any, **kwargs: Any) -> Any:
        merged = dict(network_config or {})
        merged.update(kwargs)
        return ppo_networks.make_ppo_networks(*args, **merged)

    metrics_path = run_dir / "metrics.jsonl"

    def progress(step: int, metrics: dict[str, Any]) -> None:
        record = {"step": int(step)} | {k: float(v) for k, v in metrics.items()}
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        checkpoints.commit()

    def save_policy(step: int, make_policy: Any, params: Any) -> None:
        brax_model.save_params(str(ckpt_dir / f"step_{int(step):012d}"), params)
        checkpoints.commit()

    train_fn_kwargs = dict(training_params)
    train_fn_kwargs["network_factory"] = network_factory
    train_fn_kwargs["randomization_fn"] = registry.get_domain_randomizer(ENV_NAME)
    train_fn_kwargs["wrap_env_fn"] = wrapper.wrap_for_brax_training

    start = datetime.now(UTC)
    _, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_evals=ppo_params.num_evals,
        progress_fn=progress,
        policy_params_fn=save_policy,
        **train_fn_kwargs,
    )
    wall_seconds = (datetime.now(UTC) - start).total_seconds()

    # SMOKE-RUN VERIFICATION ITEM (M0-03b accept): "newborn" must be a genuinely untrained
    # (step ~0) checkpoint. Whether brax's policy_params_fn fires early enough is verified
    # during the smoke run against the real package; if not, an explicit init-params save
    # is implemented there (see train/stages.py docstring). Not resolvable from docs alone.
    brax_model.save_params(str(ckpt_dir / "converged"), params)

    summary = {
        "run_id": run_id,
        "env_name": ENV_NAME,
        "smoke": smoke,
        "wall_seconds": wall_seconds,
        "num_timesteps": int(ppo_params.num_timesteps),
        "num_evals": int(ppo_params.num_evals),
        "gpu": "A100-80GB",
        "checkpoints": sorted(p.name for p in ckpt_dir.iterdir()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    checkpoints.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    summary = train.remote(smoke=smoke)
    print(json.dumps(summary, indent=2))
