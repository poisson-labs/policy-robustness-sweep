"""G2 nominal-locomotion verification + sanity clip (M0-03b accept criteria 2/3).

Loads a checkpoint from the training Volume, rebuilds the policy exactly as trained
(same network factory config from the recipe), runs deterministic eval episodes in the
Playground env, writes a verification report (rewards, episode lengths, per-episode
detail) and renders a short nominal-gait mp4 — both to the Volume under the run dir.

    uv run modal run train/verify_nominal.py --run-id <run_id> [--checkpoint converged]

This is also the seed of the G3 equivalence harness: the same load-policy/rollout path,
extended with physics-only perturbation injection, becomes the G4 executor. Rendering
tries EGL first (GPU) and falls back to OSMesa — both libs are baked into the image.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from train.modal_app import ENV_NAME, VOLUME_MOUNT, base_image, checkpoints

EVAL_SEED = 0  # fixed eval seed; per-episode keys derive from it (recorded in the report)

verify_image = (
    base_image.apt_install("libegl1", "libgl1", "libglvnd0", "libosmesa6")
    .uv_pip_install("imageio[ffmpeg]==2.37.4")
    .env({"MUJOCO_GL": "egl"})
    .add_local_python_source("train")
)

app = modal.App("opw-verify")


def episode_stats(rewards: list[float], lengths: list[int]) -> dict[str, float]:
    """Pure summary used by the report (unit-tested locally)."""
    n = len(rewards)
    if n == 0:
        raise ValueError("no episodes")
    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    return {
        "episodes": float(n),
        "reward_mean": mean_r,
        "reward_std": var_r**0.5,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "length_mean": sum(lengths) / n,
    }


@app.function(
    image=verify_image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={VOLUME_MOUNT: checkpoints},
)
def verify(run_id: str, checkpoint: str = "converged", episodes: int = 8) -> dict[str, Any]:
    import os
    from pathlib import Path

    import imageio.v3 as iio
    import jax
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    ckpt_path = run_dir / "checkpoints" / checkpoint
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    env_cfg = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    network_config = dict(ppo_params.network_factory)

    params = brax_model.load_params(str(ckpt_path))
    normalize = lambda x, y: x  # noqa: E731
    if ppo_params.get("normalize_observations", False):
        normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
        **network_config,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy = jax.jit(make_policy(params, deterministic=True))

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    episode_length = int(env_cfg.episode_length)

    rewards: list[float] = []
    lengths: list[int] = []
    first_rollout: list[Any] = []
    rng = jax.random.PRNGKey(EVAL_SEED)
    for ep in range(episodes):
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        total, steps = 0.0, 0
        for _ in range(episode_length):
            if ep == 0:
                first_rollout.append(state)
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            total += float(state.reward)
            steps += 1
            if bool(state.done):
                break
        rewards.append(total)
        lengths.append(steps)

    stats = episode_stats(rewards, lengths)

    clip_path = run_dir / f"nominal-gait-{checkpoint}.mp4"
    render_error = ""
    for gl_backend in ("egl", "osmesa"):
        try:
            os.environ["MUJOCO_GL"] = gl_backend
            frames = env.render(first_rollout, height=480, width=640)
            fps = 1.0 / env.dt
            iio.imwrite(str(clip_path), np.asarray(frames), fps=fps)
            render_error = ""
            break
        except Exception as e:
            render_error = f"{gl_backend}: {e!r}"

    report = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "env_name": ENV_NAME,
        "eval_seed": EVAL_SEED,
        "episode_length_cap": episode_length,
        "deterministic_policy": True,
        "stats": stats,
        "rewards": rewards,
        "lengths": lengths,
        "clip": clip_path.name if not render_error else None,
        "render_error": render_error,
    }
    (run_dir / f"verify-{checkpoint}.json").write_text(json.dumps(report, indent=2))
    checkpoints.commit()
    return report


@app.local_entrypoint()
def main(run_id: str, checkpoint: str = "converged", episodes: int = 8) -> None:
    print(
        json.dumps(verify.remote(run_id=run_id, checkpoint=checkpoint, episodes=episodes), indent=2)
    )
