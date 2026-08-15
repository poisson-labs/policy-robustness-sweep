"""Instrumented rollouts → .rrd replays (M1-03; core of the M2 live probe).

Single-world entrypoint plus a batch entrypoint that runs the committed replay
selection (15 boundary cells + 3 recoveries) in ONE container so the JIT compile is
paid once. Each .rrd ships spec §9 features: blueprint with the camera TRACKING the
trunk (no viewport hunting), collapsed side panels, timeline event markers (push
start/end, failure), scalar tracks, and the push force drawn as a red arrow during
the window (M1-03 requirements, DEVLOG Session 12/17).

Rollout physics: lax.scan on the G3-verified push_step path (the per-step host-sync
loop of the G5 prototype is gone; sim time drops from ~15 s to ~1 s warm).

    uv run modal run probe/instrumented.py --run-id <id> --mu 0.5 --push-pct 90
    uv run modal run probe/instrumented.py::batch --run-id <id>   # the 18 selections
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

probe_image = base_image.uv_pip_install(
    "pydantic==2.13.4", "rerun-sdk==0.36.0"
).add_local_python_source("train", "sweep", "configs", "probe", "reduce")

app = modal.App("opw-probe")


def _load_selection() -> dict[str, Any]:
    """Local-side only (the batch entrypoint runs on this machine; the container has no
    docs/ tree — the first batch launch failed exactly there)."""
    path = Path(__file__).parent.parent / "docs/measurements/2026-08-14-replay-selection.json"
    return json.loads(path.read_text())


@app.function(
    image=probe_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def instrumented_batch(
    run_id: str, worlds: list[dict[str, Any]], checkpoint: str = "converged"
) -> list[dict[str, Any]]:
    import time

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
    from probe.rrd_logger import (
        log_event,
        log_push_arrow,
        log_scalars,
        log_static_scene,
        log_step,
        replay_blueprint,
    )
    from reduce.classify import Outcome, classify_rollout
    from sweep.injected_env import (
        find_body_id,
        find_floor_geom_id,
        init_push_info,
        patch_floor_friction,
        push_step,
    )
    from train.modal_app import ENV_NAME as env_name

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
    base_model = env.mjx_model
    steps = round(ROLLOUT_S / env.dt)
    dt = float(env.dt)
    mj_model = env.mj_model
    trunk_geom_id = int(np.argmax(mj_model.geom_bodyid == torso_id))
    trunk_entity = (
        f"world/robot/{mj_model.geom(trunk_geom_id).name or f'geom{trunk_geom_id}'}_{trunk_geom_id}"
    )

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

    # One compiled rollout, model + push params traced → reused for every world.
    def rollout(
        model: Any, key: jax.Array, force_n: jax.Array, direction_rad: jax.Array, start_s: jax.Array
    ) -> jax.Array:
        env._mjx_model = model
        state = init_push_info(env.reset(key))
        state.info["command"] = jnp.array([1.0, 0.0, 0.0])
        state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)

        def body(carry: Any, _: Any) -> tuple[Any, jax.Array]:
            st = carry
            action, _extras = policy(st.obs, jax.random.PRNGKey(0))
            nst = push_step(env, st, action, force_n, direction_rad, start_s, torso_id)
            return nst, nst.data.qpos

        _, qpos = jax.lax.scan(body, state, None, length=steps)
        return qpos

    rollout_jit = jax.jit(rollout)

    def quick_outcome(model: Any, seed: int, force_n: float) -> Any:
        q = np.asarray(
            rollout_jit(
                model,
                jax.random.PRNGKey(seed),
                jnp.asarray(force_n),
                jnp.deg2rad(jnp.asarray(90.0)),
                jnp.asarray(2.0),
            )
        )
        qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        return classify_rollout(q[:, 2], roll, pitch, dt).outcome

    results: list[dict[str, Any]] = []
    for spec in worlds:
        t_start = time.monotonic()
        search_note: dict[str, Any] = {}
        if "recovery_seed_candidates" in spec:
            model_s = patch_floor_friction(base_model, floor_id, spec["mu"])
            force_s = spec["push_pct_bw"] / 100.0 * mass_kg * g
            chosen = None
            attempts = 0
            for cand in spec["recovery_seed_candidates"]:
                attempts += 1
                if quick_outcome(model_s, cand, force_s) is Outcome.CENSORED:
                    chosen = cand
                    break
            if chosen is None:
                results.append(
                    {
                        "cell": f"mu{spec['mu']}-push{spec['push_pct_bw']}",
                        "recovery_search": "no survivor in re-simulation",
                        "attempts": attempts,
                    }
                )
                continue
            spec = {"mu": spec["mu"], "push_pct_bw": spec["push_pct_bw"], "seed": chosen}
            search_note = {
                "recovery_attempts": attempts,
                "matched_recorded_survivor": attempts == 1,
            }
        world = WorldConfig.clamped(
            friction=spec.get("mu"),
            push_magnitude_pct_bw=spec.get("push_pct_bw", 0.0),
            push_direction_deg=spec.get("direction_deg", 90.0),
            push_start_s=spec.get("push_start_s", 2.0),
            seed=spec.get("seed", 0),
        )
        model = base_model
        if world.friction is not None:
            model = patch_floor_friction(base_model, floor_id, world.friction)
        force_n = world.push_magnitude_pct_bw / 100.0 * mass_kg * g
        qpos_arr = np.asarray(
            rollout_jit(
                model,
                jax.random.PRNGKey(world.seed),
                jnp.asarray(force_n),
                jnp.deg2rad(jnp.asarray(world.push_direction_deg)),
                jnp.asarray(world.push_start_s),
            )
        )
        sim_s = time.monotonic() - t_start

        w, x, y, z = qpos_arr[:, 3], qpos_arr[:, 4], qpos_arr[:, 5], qpos_arr[:, 6]
        upz = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        torso_z = qpos_arr[:, 2]
        outcome = classify_rollout(torso_z, roll, pitch, dt)

        # push window + onset-yaw replication (numpy mirror of push_step semantics,
        # for the arrow visualization only)
        times = (np.arange(steps) + 1) * dt
        pre_times = times - dt  # time at step invocation
        active = (pre_times >= world.push_start_s) & (
            pre_times < world.push_start_s + PUSH_DURATION_S
        )
        yaws = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        onset_idx = int(np.argmax(active)) if active.any() else 0
        onset_yaw = float(yaws[onset_idx])
        angle = onset_yaw + np.deg2rad(world.push_direction_deg)
        # Bigger, sublinear with force (Taylor, 2026-08-14): sqrt scaling keeps a
        # 10%BW nudge visible (~0.28 m) while a 200%BW shove stays ~1.3 m; 100%BW ≈ 0.9 m.
        arrow_len = 0.9 * (world.push_magnitude_pct_bw / 100.0) ** 0.5
        vx, vy = arrow_len * np.cos(angle), arrow_len * np.sin(angle)

        t0 = time.monotonic()
        rr.init("one-policy-400-worlds", spawn=False, recording_id=world.cache_key())
        mj_data = mujoco.MjData(mj_model)
        log_static_scene(mj_model)
        log_event(0.0, f"world: {world.cache_key()}")
        log_event(float(world.push_start_s), f"push start ({world.push_magnitude_pct_bw:.0f}%BW)")
        log_event(float(world.push_start_s + PUSH_DURATION_S), "push end")
        if outcome.outcome is Outcome.FAILED and outcome.ttf_s is not None:
            log_event(
                outcome.ttf_s,
                f"FAILURE ({outcome.first_trigger}) — TTF {outcome.ttf_s:.2f}s",
                "ERROR",
            )
        for i in range(steps):
            mj_data.qpos[:] = qpos_arr[i]
            mujoco.mj_forward(mj_model, mj_data)
            t_s = float(times[i])
            log_step(mj_model, mj_data, t_s)
            log_scalars(t_s, float(torso_z[i]), float(upz[i]))
            log_push_arrow(
                t_s, bool(active[i]), mj_data.xpos[torso_id].tolist(), float(vx), float(vy)
            )
        rrd_dir = Path(VOLUME_MOUNT) / "rrd"
        rrd_dir.mkdir(exist_ok=True)
        safe_key = world.cache_key().replace("|", "_").replace("=", "-")
        rrd_path = rrd_dir / f"{safe_key}.rrd"
        rr.save(str(rrd_path), default_blueprint=replay_blueprint(trunk_entity))
        # Canonical trajectory for downstream renderers (Session 21): clips must render
        # THIS rollout, not an independent re-simulation (same-seed outcomes disagree
        # under measured GPU nondeterminism at knife-edge worlds).
        np.savez_compressed(
            rrd_dir / f"{safe_key}.qpos.npz",
            qpos=qpos_arr,
            ttf_s=np.float64(-1.0 if outcome.ttf_s is None else outcome.ttf_s),
            outcome=np.array(outcome.outcome.value),
        )
        log_s = time.monotonic() - t0
        checkpoints.commit()
        results.append(
            {
                "cache_key": world.cache_key(),
                "rrd_file": rrd_path.name,
                "rrd_bytes": rrd_path.stat().st_size,
                "outcome": outcome.outcome.value,
                "ttf_s": outcome.ttf_s,
                "sim_s": round(sim_s, 3),
                "log_s": round(log_s, 3),
                **search_note,
            }
        )
    return results


@app.local_entrypoint()
def main(
    run_id: str,
    checkpoint: str = "converged",
    mu: float = -1.0,
    push_pct: float = 0.0,
    seed: int = 0,
) -> None:
    worlds = [{"mu": None if mu < 0 else mu, "push_pct_bw": push_pct, "seed": seed}]
    print(json.dumps(instrumented_batch.remote(run_id, worlds, checkpoint), indent=2))


@app.local_entrypoint()
def batch(run_id: str, checkpoint: str = "converged") -> None:
    from sweep.seed_scheme import sweep_prng_seed

    selection = _load_selection()
    # Seeds are the sweep's ACTUAL PRNG seeds for these rollouts — a bare seed_idx
    # simulates a different world (Session 19 defect).
    worlds = [
        {
            "mu": c["mu"],
            "push_pct_bw": c["push_pct_bw"],
            "seed": sweep_prng_seed(c["mu"], c["push_pct_bw"], 0),
        }
        for c in selection["boundary_cells"]
    ] + [
        # Recovery cells: the sweep's recorded survivor may FLIP on re-simulation (GPU
        # run-to-run nondeterminism — measured; knife-edge trajectories are maximally
        # sensitive, DEVLOG Session 19). Send every sweep seed of the cell ordered
        # recorded-survivor-first; the container searches for one that survives NOW and
        # records the search.
        {
            "mu": r["mu"],
            "push_pct_bw": r["push_pct_bw"],
            "recovery_seed_candidates": [sweep_prng_seed(r["mu"], r["push_pct_bw"], r["seed_idx"])]
            + [
                sweep_prng_seed(r["mu"], r["push_pct_bw"], s)
                for s in range(16)
                if s != r["seed_idx"]
            ],
        }
        for r in selection["recoveries"]
    ]
    print(json.dumps(instrumented_batch.remote(run_id, worlds, checkpoint), indent=2))
