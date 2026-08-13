"""G4 pilot sweep — grid-range finding (spec §4 G4 pilot pass; §14 #6).

Coarse grid straddling the training DR friction band with generous push range, per the
recorded G4 pilot design (DEVLOG Session 8): friction μ ∈ {0.1, 0.25, 0.4, 0.7, 1.0}
(band U(0.4, 1.0) annotated), push ∈ {10, 25, 50, 75, 100, 150, 200} %BW (log-ish),
4 seeds, direction 90° (lateral), t0 = 2 s, 5 s rollouts (spec §5).

Also runs a NOMINAL batch (model-default friction, zero push, 8 seeds) whose state
tracks seed the §5 failure-threshold freezing (M0-07).

Per spec §5/§6 the rollout protocol here is the sweep protocol:
- joystick command FIXED for the whole rollout (constant forward velocity, recorded in
  the output; resample counter pushed past the horizon);
- deterministic policy — spec §6 "seed = env PRNG only" means the env reset PRNG is the
  ONLY stochasticity source (DEVLOG Session 8 DECISION);
- failure detection is post-hoc from logged tracks (torso z, up-vector z, roll, pitch,
  NaN) — thresholds are frozen from the nominal distributions, not hardcoded here.

    uv run modal run sweep/pilot.py --run-id 20260813T155006Z

Writes records + full tracks to the Volume under runs/<run_id>/pilot-<ts>/.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

PILOT_FRICTIONS = [0.1, 0.25, 0.4, 0.7, 1.0]
PILOT_PUSH_PCTS = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0]
PILOT_SEEDS = 4
NOMINAL_SEEDS = 8
PUSH_DIRECTION_DEG = 90.0
PUSH_START_S = 2.0
ROLLOUT_S = 5.0
COMMAND_VX_M_S = 1.0  # provisional pilot default (recorded in outputs; spec §5)

pilot_image = base_image.uv_pip_install("pydantic==2.13.4").add_local_python_source(
    "train", "sweep", "configs"
)

app = modal.App("opw-pilot")


@app.function(
    image=pilot_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def run_pilot(run_id: str, checkpoint: str = "converged") -> dict[str, Any]:
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
        patch_floor_friction,
        push_step,
    )
    from train.modal_app import ENV_NAME as env_name

    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    params = brax_model.load_params(str(run_dir / "checkpoints" / checkpoint))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / f"pilot-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = registry.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_config = dict(ppo_params.network_factory)

    env = registry.load(env_name, config=env_cfg)
    torso_id = find_body_id(env.mj_model)
    floor_id = find_floor_geom_id(env.mj_model)
    mass_kg = float(env.mj_model.body_subtreemass[torso_id])
    g = float(abs(env.mj_model.opt.gravity[2]))
    default_mu = float(env.mj_model.geom_friction[floor_id, 0])
    steps = int(round(ROLLOUT_S / env.dt))

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

    def fixed_command(state: Any) -> Any:
        state.info["command"] = command
        # push the resample counter far past the 5 s horizon (exp-distributed default)
        state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)
        return state

    def batch_reset(seed0: int, n: int) -> Any:
        keys = jax.random.split(jax.random.PRNGKey(seed0), n)
        states = jax.vmap(env.reset)(keys)
        states = jax.vmap(init_push_info)(states)
        return jax.vmap(fixed_command)(states)

    def quat_tracks(qpos: jax.Array) -> tuple[jax.Array, ...]:
        w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        upz = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        return upz, roll, pitch

    def rollout(states: Any, force_ns: jax.Array) -> dict[str, np.ndarray]:
        def body(carry: Any, _: Any) -> tuple[Any, dict[str, jax.Array]]:
            st = carry
            dummy_keys = jax.random.split(jax.random.PRNGKey(0), force_ns.shape[0])
            actions, _extras = policy(st.obs, dummy_keys)
            nst = jax.vmap(
                lambda s, a, f: push_step(env, s, a, f, direction_rad, start_s, torso_id)
            )(st, actions, force_ns)
            qpos, qvel = nst.data.qpos, nst.data.qvel
            upz, roll, pitch = quat_tracks(qpos)
            bad = jnp.logical_or(
                jnp.any(~jnp.isfinite(qpos), axis=1), jnp.any(~jnp.isfinite(qvel), axis=1)
            )
            ys = {
                "torso_z": qpos[:, 2],
                "upz": upz,
                "roll": roll,
                "pitch": pitch,
                "diverged": bad,
            }
            return nst, ys

        _, tracks = jax.lax.scan(body, states, None, length=steps)
        return {k: np.asarray(v) for k, v in tracks.items()}  # each (steps, batch)

    outputs: dict[str, Any] = {}

    # Nominal batch (default friction, zero push) — threshold-freezing data.
    nominal_states = batch_reset(seed0=10_000, n=NOMINAL_SEEDS)
    nominal_tracks = rollout(nominal_states, jnp.zeros(NOMINAL_SEEDS))
    np.savez_compressed(out_dir / "tracks-nominal.npz", **nominal_tracks)
    outputs["nominal"] = {"seeds": NOMINAL_SEEDS, "friction": default_mu}

    # Grid: outer loop over friction (model patch → recompile per μ, fine for a pilot;
    # the full sweep will vmap the model axis instead — noted in TASKS).
    base_mjx_model = env.mjx_model
    force_grid = jnp.repeat(
        jnp.asarray([p / 100.0 * mass_kg * g for p in PILOT_PUSH_PCTS]), PILOT_SEEDS
    )  # (7*4,) — one force per (push, seed) pair
    records: list[dict[str, Any]] = []
    for mu in PILOT_FRICTIONS:
        env._mjx_model = patch_floor_friction(base_mjx_model, floor_id, mu)  # noqa: SLF001
        states = batch_reset(seed0=int(mu * 1000), n=len(PILOT_PUSH_PCTS) * PILOT_SEEDS)
        tracks = rollout(states, force_grid)
        np.savez_compressed(out_dir / f"tracks-mu{mu:.2f}.npz", **tracks)
        dt = float(env.dt)
        for b in range(len(PILOT_PUSH_PCTS) * PILOT_SEEDS):
            upz_track = tracks["upz"][:, b]
            diverged_track = tracks["diverged"][:, b]
            fell = np.argmax(upz_track < 0.0) if np.any(upz_track < 0.0) else -1
            diverged = np.argmax(diverged_track) if np.any(diverged_track) else -1
            records.append(
                {
                    "friction": mu,
                    "push_pct_bw": PILOT_PUSH_PCTS[b // PILOT_SEEDS],
                    "seed_index": b % PILOT_SEEDS,
                    "fell_time_upz_s": None if fell < 0 else round((fell + 1) * dt, 4),
                    "min_upz": float(upz_track.min()),
                    "min_torso_z": float(tracks["torso_z"][:, b].min()),
                    "max_abs_roll": float(np.abs(tracks["roll"][:, b]).max()),
                    "max_abs_pitch": float(np.abs(tracks["pitch"][:, b]).max()),
                    "diverged_at_s": None if diverged < 0 else round((diverged + 1) * dt, 4),
                }
            )
    env._mjx_model = base_mjx_model  # noqa: SLF001

    manifest = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "pilot_dir": out_dir.name,
        "protocol": {
            "frictions": PILOT_FRICTIONS,
            "push_pcts_bw": PILOT_PUSH_PCTS,
            "seeds_per_cell": PILOT_SEEDS,
            "nominal_seeds": NOMINAL_SEEDS,
            "direction_deg": PUSH_DIRECTION_DEG,
            "push_start_s": PUSH_START_S,
            "rollout_s": ROLLOUT_S,
            "steps": steps,
            "dt": float(env.dt),
            "command": [COMMAND_VX_M_S, 0.0, 0.0],
            "deterministic_policy": True,
            "default_floor_friction": default_mu,
            "robot_mass_kg": mass_kg,
            "gravity_m_s2": g,
            "dr_friction_band": [0.4, 1.0],
        },
        "records": records,
        **outputs,
    }
    (out_dir / "pilot-manifest.json").write_text(json.dumps(manifest, indent=2))
    checkpoints.commit()
    return manifest


@app.local_entrypoint()
def main(run_id: str, checkpoint: str = "converged") -> None:
    manifest = run_pilot.remote(run_id=run_id, checkpoint=checkpoint)
    print(json.dumps(manifest, indent=2))
