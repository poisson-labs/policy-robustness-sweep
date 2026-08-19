"""G4 full sweep — the real 20x20 x 16 seeds (spec §4 G4; grid per §14 #6 sign-off).

Grid (Taylor-approved 2026-08-13): μ ∈ [0.05, 1.00] step 0.05; push ∈ [10, 200]%BW
step 10; 16 seeds/cell → 6400 rollouts. Protocol identical to the pilot (5 s rollouts,
deterministic policy, fixed command [1.0, 0, 0] m/s, lateral 90° push at t = 2 s).

Executor design (per DEVLOG Session 8 DECISION): friction is a VMAPPED MODEL AXIS —
a batched mjx model (only geom_friction batched, in_axes tree of None elsewhere, the
playground DR-wrapper trick) — so the whole sweep uses ONE compiled step regardless of
how many μ values exist. Rollouts run in equal-size chunks (compile once, reuse); chunk
wall times are recorded (§14 #3 measurement: batch size vs memory).

Seed scheme (spec §6, documented): rollout r (0..6399) uses PRNGKey(1_000_000 + r),
r = (mu_idx * 20 + push_idx) * 16 + seed_idx. Env reset PRNG is the only stochasticity
source (deterministic policy).

Outputs per chunk: full per-step tracks (x, y, torso z, up-z, roll, pitch, yaw, 4 foot
contacts, 4 feet xyz) as compressed npz — the §5 TTF/margin features are computed
post-hoc in reduce/ once M0-07 freezes thresholds from nominal distributions; plus
provisional per-rollout records (up-z fall time, extrema) and a manifest with timings.

    uv run modal run sweep/full_sweep.py --run-id 20260813T155006Z
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

GRID_MUS = [round(0.05 * i, 2) for i in range(1, 21)]  # 0.05 .. 1.00
GRID_PUSH_PCTS = [float(10 * i) for i in range(1, 21)]  # 10 .. 200
SEEDS_PER_CELL = 16
SEED_BASE = 1_000_000
PUSH_DIRECTION_DEG = 90.0
PUSH_START_S = 2.0
ROLLOUT_S = 5.0
COMMAND_VX_M_S = 1.0
CHUNK_ROLLOUTS = 1600  # 4 equal chunks; one compile, reused (recorded per-chunk)

sweep_image = base_image.uv_pip_install("pydantic==2.13.4").add_local_python_source(
    "train", "sweep", "configs"
)

app = modal.App("opw-sweep")


@app.function(
    image=sweep_image,
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def run_sweep(
    run_id: str, checkpoint: str = "converged", seeds_per_cell: int = SEEDS_PER_CELL
) -> dict[str, Any]:
    """`seeds_per_cell` overrides the v0 default (16) — M1.5 runs 32v32 (Session 23
    power math). Rollout indexing keeps the SAME formula with the larger stride, so
    v0-at-32 and v1-at-32 use identical seeds cell-for-cell (diff hygiene)."""
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    from sweep.injected_env import (
        find_body_id,
        find_floor_geom_id,
        init_push_info,
        push_step,
        yaw_from_quat,
    )
    from train.modal_app import ENV_NAME as env_name

    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    params = brax_model.load_params(str(run_dir / "checkpoints" / checkpoint))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / f"sweep-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = registry.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_config = dict(ppo_params.network_factory)

    env = registry.load(env_name, config=env_cfg)
    torso_id = find_body_id(env.mj_model)
    floor_id = find_floor_geom_id(env.mj_model)
    mass_kg = float(env.mj_model.body_subtreemass[torso_id])
    g = float(abs(env.mj_model.opt.gravity[2]))
    steps = round(ROLLOUT_S / env.dt)
    feet_site_ids = getattr(env, "_feet_site_id", None)

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
    direction_rad = jnp.deg2rad(jnp.asarray(PUSH_DIRECTION_DEG))
    start_s = jnp.asarray(PUSH_START_S)
    command = jnp.array([COMMAND_VX_M_S, 0.0, 0.0])

    # Flattened rollout axes: r = (mu_idx*20 + push_idx)*16 + seed_idx
    mus_flat, forces_flat, seeds_flat = [], [], []
    cells: list[dict[str, Any]] = []
    for mi, mu in enumerate(GRID_MUS):
        for pi, pct in enumerate(GRID_PUSH_PCTS):
            for s in range(seeds_per_cell):
                r = (mi * len(GRID_PUSH_PCTS) + pi) * seeds_per_cell + s
                mus_flat.append(mu)
                forces_flat.append(pct / 100.0 * mass_kg * g)
                seeds_flat.append(SEED_BASE + r)
                cells.append({"mu": mu, "push_pct_bw": pct, "seed_idx": s, "rollout": r})
    total = len(cells)

    base_model = env.mjx_model
    model_in_axes = jax.tree_util.tree_map(lambda _: None, base_model)
    model_in_axes = model_in_axes.replace(geom_friction=0)

    def one_reset(model: Any, key: jax.Array) -> Any:
        env._mjx_model = model
        state = init_push_info(env.reset(key))
        state.info["command"] = command
        state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)
        return state

    def one_step(model: Any, state: Any, action: jax.Array, force_n: jax.Array) -> Any:
        env._mjx_model = model
        return push_step(env, state, action, force_n, direction_rad, start_s, torso_id)

    v_reset = jax.jit(jax.vmap(one_reset, in_axes=(model_in_axes, 0)))
    v_step = jax.vmap(one_step, in_axes=(model_in_axes, 0, 0, 0))

    def chunk_rollout(models: Any, keys: jax.Array, force_ns: jax.Array) -> dict[str, Any]:
        states = v_reset(models, keys)

        def body(carry: Any, _: Any) -> tuple[Any, dict[str, jax.Array]]:
            st = carry
            dummy = jax.random.split(jax.random.PRNGKey(0), force_ns.shape[0])
            actions, _extras = policy(st.obs, dummy)
            nst = v_step(models, st, actions, force_ns)
            qpos, qvel = nst.data.qpos, nst.data.qvel
            w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
            ys = {
                "x": qpos[:, 0],
                "y": qpos[:, 1],
                "torso_z": qpos[:, 2],
                "upz": 1.0 - 2.0 * (x * x + y * y),
                "roll": jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
                "pitch": jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
                "yaw": jax.vmap(yaw_from_quat)(qpos[:, 3:7]),
                "contacts": nst.info["last_contact"],
                "diverged": jnp.logical_or(
                    jnp.any(~jnp.isfinite(qpos), axis=1), jnp.any(~jnp.isfinite(qvel), axis=1)
                ),
            }
            if feet_site_ids is not None:
                ys["feet_xyz"] = nst.data.site_xpos[:, feet_site_ids, :]
            return nst, ys

        final, tracks = jax.lax.scan(body, states, None, length=steps)
        out = {k: np.asarray(v) for k, v in tracks.items()}
        out["final_feet_air_time"] = np.asarray(final.info["feet_air_time"])
        out["final_swing_peak"] = np.asarray(final.info["swing_peak"])
        return out

    records: list[dict[str, Any]] = []
    chunk_walls: list[float] = []
    dt = float(env.dt)
    for c0 in range(0, total, CHUNK_ROLLOUTS):
        c1 = min(c0 + CHUNK_ROLLOUTS, total)
        mus = jnp.asarray(mus_flat[c0:c1])
        # Batch ONLY the geom_friction leaf (a whole-model vmap would broadcast every
        # leaf into the batch dim and break mjx internals — first sweep attempt did).
        batched_geom_friction = jax.vmap(
            lambda mu: base_model.geom_friction.at[floor_id, 0].set(mu)
        )(mus)
        models = base_model.replace(geom_friction=batched_geom_friction)
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds_flat[c0:c1]))
        force_ns = jnp.asarray(forces_flat[c0:c1])
        t0 = time.monotonic()
        tracks = chunk_rollout(models, keys, force_ns)
        wall = time.monotonic() - t0
        chunk_walls.append(round(wall, 3))
        np.savez_compressed(out_dir / f"tracks-{c0:05d}.npz", **tracks)
        for b in range(c1 - c0):
            meta = cells[c0 + b]
            upz_track = tracks["upz"][:, b]
            div_track = tracks["diverged"][:, b]
            fell = int(np.argmax(upz_track < 0.0)) if np.any(upz_track < 0.0) else -1
            div = int(np.argmax(div_track)) if np.any(div_track) else -1
            records.append(
                meta
                | {
                    "fell_time_upz_s": None if fell < 0 else round((fell + 1) * dt, 4),
                    "min_upz": float(upz_track.min()),
                    "min_torso_z": float(tracks["torso_z"][:, b].min()),
                    "diverged_at_s": None if div < 0 else round((div + 1) * dt, 4),
                }
            )
        checkpoints.commit()

    manifest = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "sweep_dir": out_dir.name,
        "grid": {
            "mus": GRID_MUS,
            "push_pcts_bw": GRID_PUSH_PCTS,
            "seeds_per_cell": seeds_per_cell,
            "seed_scheme": f"PRNGKey({SEED_BASE} + rollout_index); rollout_index = "
            f"(mu_idx*20 + push_idx)*{seeds_per_cell} + seed_idx",
            "total_rollouts": total,
        },
        "protocol": {
            "direction_deg": PUSH_DIRECTION_DEG,
            "push_start_s": PUSH_START_S,
            "rollout_s": ROLLOUT_S,
            "steps": steps,
            "dt": dt,
            "command": [COMMAND_VX_M_S, 0.0, 0.0],
            "deterministic_policy": True,
            "robot_mass_kg": mass_kg,
            "gravity_m_s2": g,
            "feet_sites_logged": feet_site_ids is not None,
        },
        "timings": {
            "chunk_rollouts": CHUNK_ROLLOUTS,
            "chunk_wall_s": chunk_walls,
            "first_chunk_includes_compile": True,
            "gpu": "A100-80GB",
        },
        "records": records,
    }
    (out_dir / "sweep-manifest.json").write_text(json.dumps(manifest, indent=2))
    checkpoints.commit()
    return {k: v for k, v in manifest.items() if k != "records"} | {"n_records": len(records)}


@app.local_entrypoint()
def main(run_id: str, checkpoint: str = "converged", seeds_per_cell: int = SEEDS_PER_CELL) -> None:
    print(
        json.dumps(
            run_sweep.remote(run_id=run_id, checkpoint=checkpoint, seeds_per_cell=seeds_per_cell),
            indent=2,
        )
    )
