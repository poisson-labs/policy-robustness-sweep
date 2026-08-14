"""Instrumented single rollout → .rrd (G5 test artifact; core of the M2 live probe).

Runs ONE world (WorldConfig semantics, same G3-verified physics path as the sweep) with
full logging: mjx physics rollout, frozen-threshold classification, then CPU-MuJoCo
kinematic replay logged to a Rerun recording saved on the Volume under
rrd/<cache_key>.rrd. Measures and returns .rrd size (the R2-trigger data per §14 #2)
and stage timings.

    uv run modal run probe/instrumented.py --run-id 20260813T155006Z \
        --mu 0.5 --push-pct 90 --seed 0
"""

from __future__ import annotations

import json
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

probe_image = base_image.uv_pip_install(
    "pydantic==2.13.4", "rerun-sdk==0.36.0"
).add_local_python_source("train", "sweep", "configs", "probe", "reduce")

app = modal.App("opw-probe")


@app.function(
    image=probe_image,
    gpu="A100-80GB",
    timeout=1200,
    volumes={VOLUME_MOUNT: checkpoints},
)
def instrumented_rollout(
    run_id: str,
    checkpoint: str = "converged",
    mu: float | None = None,
    push_pct: float = 0.0,
    direction_deg: float = 90.0,
    push_start_s: float = 2.0,
    seed: int = 0,
) -> dict[str, Any]:
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    import rerun as rr
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    from configs.world import PUSH_DURATION_S, ROLLOUT_S, WorldConfig
    from probe.rrd_logger import log_event, log_scalars, log_static_scene, log_step
    from reduce.classify import Outcome, classify_rollout
    from sweep.injected_env import (
        find_body_id,
        find_floor_geom_id,
        init_push_info,
        patch_floor_friction,
        push_step,
    )
    from train.modal_app import ENV_NAME as env_name

    timings: dict[str, float] = {}
    t_start = time.monotonic()

    world = WorldConfig.clamped(
        friction=mu,
        push_magnitude_pct_bw=push_pct,
        push_direction_deg=direction_deg,
        push_start_s=push_start_s,
        seed=seed,
    )
    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    params = brax_model.load_params(str(run_dir / "checkpoints" / checkpoint))

    env_cfg = registry.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_config = dict(ppo_params.network_factory)
    env = registry.load(env_name, config=env_cfg)
    torso_id = find_body_id(env.mj_model)
    floor_id = find_floor_geom_id(env.mj_model)
    mass_kg = float(env.mj_model.body_subtreemass[torso_id])
    g = float(abs(env.mj_model.opt.gravity[2]))
    if world.friction is not None:
        env._mjx_model = patch_floor_friction(env.mjx_model, floor_id, world.friction)
    steps = round(ROLLOUT_S / env.dt)
    dt = float(env.dt)

    normalize = lambda x, y: x  # noqa: E731
    if ppo_params.get("normalize_observations", False):
        normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
        **network_config,
    )
    policy = ppo_networks.make_inference_fn(ppo_network)(params, deterministic=True)
    force_n = jnp.asarray(world.push_magnitude_pct_bw / 100.0 * mass_kg * g)
    direction_rad = jnp.deg2rad(jnp.asarray(world.push_direction_deg))
    start_arr = jnp.asarray(world.push_start_s)

    step_fn = jax.jit(
        lambda s, a: push_step(env, s, a, force_n, direction_rad, start_arr, torso_id)
    )
    t0 = time.monotonic()
    state = init_push_info(env.reset(jax.random.PRNGKey(world.seed)))
    state.info["command"] = jnp.array([1.0, 0.0, 0.0])
    state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)

    qpos_hist: list[np.ndarray] = []
    dummy_key = jax.random.PRNGKey(0)
    for _ in range(steps):
        action, _extras = policy(state.obs, dummy_key)
        state = step_fn(state, action)
        qpos_hist.append(np.asarray(state.data.qpos))
    timings["sim_s"] = time.monotonic() - t0

    qpos_arr = np.stack(qpos_hist)  # (steps, nq)
    w, x, y, z = qpos_arr[:, 3], qpos_arr[:, 4], qpos_arr[:, 5], qpos_arr[:, 6]
    upz = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    torso_z = qpos_arr[:, 2]
    result = classify_rollout(torso_z, roll, pitch, dt)

    # CPU kinematic replay → rerun
    t0 = time.monotonic()
    rr.init("one-policy-400-worlds", spawn=False)
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    log_static_scene(mj_model)
    log_event(0.0, f"world: {world.cache_key()}")
    log_event(float(world.push_start_s), f"push start ({world.push_magnitude_pct_bw:.0f}%BW)")
    log_event(float(world.push_start_s + PUSH_DURATION_S), "push end")
    if result.outcome is Outcome.FAILED and result.ttf_s is not None:
        log_event(
            result.ttf_s, f"FAILURE ({result.first_trigger}) — TTF {result.ttf_s:.2f}s", "ERROR"
        )
    for i in range(steps):
        mj_data.qpos[:] = qpos_arr[i]
        mujoco.mj_forward(mj_model, mj_data)
        t_s = (i + 1) * dt
        log_step(mj_model, mj_data, t_s)
        log_scalars(t_s, float(torso_z[i]), float(upz[i]))
    rrd_dir = Path(VOLUME_MOUNT) / "rrd"
    rrd_dir.mkdir(exist_ok=True)
    safe_key = world.cache_key().replace("|", "_").replace("=", "-")
    rrd_path = rrd_dir / f"{safe_key}.rrd"
    rr.save(str(rrd_path))
    timings["log_s"] = time.monotonic() - t0
    timings["total_s"] = time.monotonic() - t_start
    checkpoints.commit()

    return {
        "world": json.loads(world.model_dump_json()),
        "cache_key": world.cache_key(),
        "rrd_file": rrd_path.name,
        "rrd_bytes": rrd_path.stat().st_size,
        "outcome": result.outcome.value,
        "ttf_s": result.ttf_s,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
    }


@app.local_entrypoint()
def main(
    run_id: str,
    checkpoint: str = "converged",
    mu: float = -1.0,
    push_pct: float = 0.0,
    seed: int = 0,
) -> None:
    friction = None if mu < 0 else mu
    out = instrumented_rollout.remote(
        run_id=run_id, checkpoint=checkpoint, mu=friction, push_pct=push_pct, seed=seed
    )
    print(json.dumps(out, indent=2))
