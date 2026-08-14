"""Solver-sensitivity slice (spec §6, required for the post; M1-02).

Re-runs ONE boundary-crossing slice — the μ = 0.50 row, all 20 push values x 16 seeds —
under three integrator/solver settings, and reports where the survival boundary sits in
each. A large displacement means the boundary is partly an integrator artifact (§13);
either result is reported.

Settings:
- baseline: exactly the sweep's physics (sim_dt from the env config, solver defaults)
- half_dt: sim_dt halved via the env config (ctrl_dt unchanged → substeps double)
- double_iters: mjx solver iterations doubled (model.opt patch; physics-only)

Seeds reuse the sweep's scheme for the μ=0.50 row so baseline is directly comparable.

    uv run modal run sweep/solver_slice.py --run-id 20260813T155006Z
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

SLICE_MU = 0.50
GRID_PUSH_PCTS = [float(10 * i) for i in range(1, 21)]
SEEDS_PER_CELL = 16
SEED_BASE = 1_000_000
MU_INDEX = 9  # 0.50 is GRID_MUS[9] in the sweep's rollout indexing
PUSH_DIRECTION_DEG = 90.0
PUSH_START_S = 2.0
ROLLOUT_S = 5.0
COMMAND_VX_M_S = 1.0

slice_image = base_image.uv_pip_install("pydantic==2.13.4").add_local_python_source(
    "train", "sweep", "configs", "reduce"
)

app = modal.App("opw-solver-slice")


@app.function(
    image=slice_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def run_slice(run_id: str, checkpoint: str = "converged") -> dict[str, Any]:
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / f"solver-slice-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_config = dict(ppo_params.network_factory)

    def run_setting(name: str, sim_dt: float | None, iter_scale: int) -> dict[str, Any]:
        env_cfg = registry.get_default_config(env_name)
        baseline_sim_dt = float(env_cfg.sim_dt)
        if sim_dt is not None:
            env_cfg.sim_dt = sim_dt
        env = registry.load(env_name, config=env_cfg)
        torso_id = find_body_id(env.mj_model)
        floor_id = find_floor_geom_id(env.mj_model)
        mass_kg = float(env.mj_model.body_subtreemass[torso_id])
        g = float(abs(env.mj_model.opt.gravity[2]))
        model = patch_floor_friction(env.mjx_model, floor_id, SLICE_MU)
        base_iterations = int(model.opt.iterations)
        if iter_scale != 1:
            model = model.replace(opt=model.opt.replace(iterations=base_iterations * iter_scale))
        env._mjx_model = model
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
        direction_rad = jnp.deg2rad(jnp.asarray(PUSH_DIRECTION_DEG))
        start_s = jnp.asarray(PUSH_START_S)
        command = jnp.array([COMMAND_VX_M_S, 0.0, 0.0])

        n = len(GRID_PUSH_PCTS) * SEEDS_PER_CELL
        seeds = [
            SEED_BASE + (MU_INDEX * 20 + pi) * SEEDS_PER_CELL + s
            for pi in range(len(GRID_PUSH_PCTS))
            for s in range(SEEDS_PER_CELL)
        ]
        forces = jnp.repeat(
            jnp.asarray([p / 100.0 * mass_kg * g for p in GRID_PUSH_PCTS]), SEEDS_PER_CELL
        )
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
        states = jax.vmap(env.reset)(keys)
        states = jax.vmap(init_push_info)(states)

        def fix_cmd(state: Any) -> Any:
            state.info["command"] = command
            state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)
            return state

        states = jax.vmap(fix_cmd)(states)

        def body(carry: Any, _: Any) -> tuple[Any, dict[str, jax.Array]]:
            st = carry
            dummy = jax.random.split(jax.random.PRNGKey(0), n)
            actions, _extras = policy(st.obs, dummy)
            nst = jax.vmap(
                lambda s, a, f: push_step(env, s, a, f, direction_rad, start_s, torso_id)
            )(st, actions, forces)
            qpos = nst.data.qpos
            w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
            ys = {
                "torso_z": qpos[:, 2],
                "roll": jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
                "pitch": jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            }
            return nst, ys

        t0 = time.monotonic()
        _, tracks = jax.lax.scan(body, states, None, length=steps)
        tracks = {k: np.asarray(v) for k, v in tracks.items()}
        wall = time.monotonic() - t0

        survival: list[float] = []
        for pi in range(len(GRID_PUSH_PCTS)):
            survived = 0
            for s in range(SEEDS_PER_CELL):
                b = pi * SEEDS_PER_CELL + s
                r = classify_rollout(
                    tracks["torso_z"][:, b], tracks["roll"][:, b], tracks["pitch"][:, b], dt
                )
                survived += r.outcome is Outcome.CENSORED
            survival.append(survived / SEEDS_PER_CELL)

        # boundary = linear interpolation of the S=0.5 crossing along the push axis
        boundary = None
        for pi in range(1, len(GRID_PUSH_PCTS)):
            s0, s1 = survival[pi - 1], survival[pi]
            if s0 >= 0.5 > s1:
                p0, p1 = GRID_PUSH_PCTS[pi - 1], GRID_PUSH_PCTS[pi]
                boundary = p0 + (s0 - 0.5) / (s0 - s1) * (p1 - p0)
                break
        return {
            "setting": name,
            "sim_dt": sim_dt if sim_dt is not None else baseline_sim_dt,
            "solver_iterations": base_iterations * iter_scale,
            "ctrl_dt": dt,
            "wall_s": round(wall, 2),
            "survival_by_push": dict(
                zip([str(int(p)) for p in GRID_PUSH_PCTS], survival, strict=True)
            ),
            "boundary_pct_bw": None if boundary is None else round(boundary, 2),
        }

    results = [
        run_setting("baseline", None, 1),
        run_setting("half_dt", 0.002, 1),
        run_setting("double_iters", None, 2),
    ]
    report = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "slice_mu": SLICE_MU,
        "seed_scheme": "sweep rollout seeds for the mu=0.50 row (directly comparable)",
        "settings": results,
        "boundary_displacements_pct_bw": {
            r["setting"]: (
                None
                if r["boundary_pct_bw"] is None or results[0]["boundary_pct_bw"] is None
                else round(r["boundary_pct_bw"] - results[0]["boundary_pct_bw"], 2)
            )
            for r in results[1:]
        },
    }
    (out_dir / "solver-slice-report.json").write_text(json.dumps(report, indent=2))
    checkpoints.commit()
    return report


@app.local_entrypoint()
def main(run_id: str, checkpoint: str = "converged") -> None:
    print(json.dumps(run_slice.remote(run_id=run_id, checkpoint=checkpoint), indent=2))
