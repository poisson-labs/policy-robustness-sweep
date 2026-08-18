"""ProbeRuntime — the compiled probe program, built once per process (M2-01).

Used identically by the image-build warm-up (fills the persistent compile cache) and
by the serving function (hits it). Everything shape-relevant is fixed here so the
traced program is byte-identical in both places: one env, one policy, one scan of
ROLLOUT_S/dt steps, model+push parameters traced (not baked), so any world hits the
same cache entry.

Container-only module (imports jax/mujoco); no unit tests here — its pieces
(push_step, classify, rrd_logger, WorldConfig) are tested individually, and G3 verified
the physics path it composes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

PARAMS_BAKED = Path("/params/converged")


class ProbeRuntime:
    _instance: ClassVar[ProbeRuntime | None] = None

    @classmethod
    def get(cls, run_id: str, checkpoint: str, volume_root: Path | None) -> ProbeRuntime:
        if cls._instance is None:
            cls._instance = cls(run_id, checkpoint, volume_root)
        return cls._instance

    def __init__(self, run_id: str, checkpoint: str, volume_root: Path | None) -> None:
        import jax
        import jax.numpy as jnp
        from brax.io import model as brax_model
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        from mujoco_playground import registry
        from mujoco_playground.config import locomotion_params

        from configs.world import ROLLOUT_S
        from sweep.injected_env import (
            find_body_id,
            find_floor_geom_id,
            init_push_info,
            push_step,
        )
        from train.modal_app import ENV_NAME as env_name

        t0 = time.monotonic()
        params_path = PARAMS_BAKED
        if not params_path.exists() and volume_root is not None:
            params_path = volume_root / "runs" / run_id / "checkpoints" / checkpoint
        self.params = brax_model.load_params(str(params_path))
        env_cfg = registry.get_default_config(env_name)
        ppo_params = locomotion_params.brax_ppo_config(env_name)
        network_config = dict(ppo_params.network_factory)
        self.env = registry.load(env_name, config=env_cfg)
        self.torso_id = find_body_id(self.env.mj_model)
        self.floor_id = find_floor_geom_id(self.env.mj_model)
        self.mass_kg = float(self.env.mj_model.body_subtreemass[self.torso_id])
        self.g = float(abs(self.env.mj_model.opt.gravity[2]))
        self.base_model = self.env.mjx_model
        self.steps = round(ROLLOUT_S / self.env.dt)
        self.dt = float(self.env.dt)
        self.mj_model = self.env.mj_model

        normalize = lambda x, y: x  # noqa: E731
        if ppo_params.get("normalize_observations", False):
            normalize = running_statistics.normalize
        ppo_network = ppo_networks.make_ppo_networks(
            self.env.observation_size,
            self.env.action_size,
            preprocess_observations_fn=normalize,
            **network_config,
        )
        policy = ppo_networks.make_inference_fn(ppo_network)(self.params, deterministic=True)
        env = self.env
        torso_id = self.torso_id
        steps = self.steps

        def rollout(model: Any, key: Any, force_n: Any, direction_rad: Any, start_s: Any) -> Any:
            env._mjx_model = model
            state = init_push_info(env.reset(key))
            state.info["command"] = jnp.array([1.0, 0.0, 0.0])
            state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)

            def body(carry: Any, _: Any) -> tuple[Any, Any]:
                st = carry
                action, _extras = policy(st.obs, jax.random.PRNGKey(0))
                nst = push_step(env, st, action, force_n, direction_rad, start_s, torso_id)
                return nst, nst.data.qpos

            _, qpos = jax.lax.scan(body, state, None, length=steps)
            return qpos

        self._rollout = jax.jit(rollout)
        self.init_s = time.monotonic() - t0
        trunk_geom_id = int(np.argmax(self.mj_model.geom_bodyid == self.torso_id))
        name = self.mj_model.geom(trunk_geom_id).name or f"geom{trunk_geom_id}"
        self.trunk_entity = f"world/robot/{name}_{trunk_geom_id}"

    def simulate(self, world: Any) -> tuple[np.ndarray, float]:
        import jax
        import jax.numpy as jnp

        from sweep.injected_env import patch_floor_friction

        model = self.base_model
        if world.friction is not None:
            model = patch_floor_friction(self.base_model, self.floor_id, world.friction)
        force_n = world.push_magnitude_pct_bw / 100.0 * self.mass_kg * self.g
        t0 = time.monotonic()
        qpos = np.asarray(
            self._rollout(
                model,
                jax.random.PRNGKey(world.seed),
                jnp.asarray(force_n),
                jnp.deg2rad(jnp.asarray(world.push_direction_deg)),
                jnp.asarray(world.push_start_s),
            )
        )
        return qpos, time.monotonic() - t0

    def warm_up(self) -> None:
        """Compile the program (fills the persistent cache when enabled)."""
        from configs.world import WorldConfig

        _, sim_s = self.simulate(WorldConfig.clamped(friction=0.5, push_magnitude_pct_bw=90.0))
        print(f"warm-up simulate (compile + run): {sim_s:.1f}s")

    def run(self, world_dict: dict[str, Any]) -> dict[str, Any]:
        import mujoco
        import rerun as rr

        from configs.world import PUSH_DURATION_S, WorldConfig
        from probe.rrd_logger import (
            log_event,
            log_push_arrow,
            log_scalars,
            log_static_scene,
            log_step,
            replay_blueprint,
        )
        from reduce.classify import Outcome, classify_rollout

        world = WorldConfig.model_validate(world_dict)
        qpos_arr, sim_s = self.simulate(world)
        w, x, y, z = qpos_arr[:, 3], qpos_arr[:, 4], qpos_arr[:, 5], qpos_arr[:, 6]
        upz = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        torso_z = qpos_arr[:, 2]
        outcome = classify_rollout(torso_z, roll, pitch, self.dt)

        times = (np.arange(self.steps) + 1) * self.dt
        pre = times - self.dt
        active = (pre >= world.push_start_s) & (pre < world.push_start_s + PUSH_DURATION_S)
        yaws = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        onset_yaw = float(yaws[int(np.argmax(active))]) if active.any() else 0.0
        angle = onset_yaw + np.deg2rad(world.push_direction_deg)
        arrow_len = 0.9 * (world.push_magnitude_pct_bw / 100.0) ** 0.5
        vx, vy = arrow_len * np.cos(angle), arrow_len * np.sin(angle)

        t0 = time.monotonic()
        rr.init("one-policy-400-worlds", spawn=False, recording_id=world.cache_key())
        mj_data = mujoco.MjData(self.mj_model)
        log_static_scene(self.mj_model)
        log_event(0.0, f"world: {world.cache_key()}")
        log_event(float(world.push_start_s), f"push start ({world.push_magnitude_pct_bw:.0f}%BW)")
        log_event(float(world.push_start_s + PUSH_DURATION_S), "push end")
        if outcome.outcome is Outcome.FAILED and outcome.ttf_s is not None:
            log_event(
                outcome.ttf_s,
                f"FAILURE ({outcome.first_trigger}) — TTF {outcome.ttf_s:.2f}s",
                "ERROR",
            )
        for i in range(self.steps):
            mj_data.qpos[:] = qpos_arr[i]
            mujoco.mj_forward(self.mj_model, mj_data)
            t_s = float(times[i])
            log_step(self.mj_model, mj_data, t_s)
            log_scalars(t_s, float(torso_z[i]), float(upz[i]))
            log_push_arrow(
                t_s, bool(active[i]), mj_data.xpos[self.torso_id].tolist(), float(vx), float(vy)
            )
        rrd_dir = Path("/vol/rrd")
        rrd_dir.mkdir(parents=True, exist_ok=True)
        safe_key = world.cache_key().replace("|", "_").replace("=", "-")
        rrd_path = rrd_dir / f"{safe_key}.rrd"
        rr.save(str(rrd_path), default_blueprint=replay_blueprint(self.trunk_entity))
        np.savez_compressed(
            rrd_dir / f"{safe_key}.qpos.npz",
            qpos=qpos_arr,
            ttf_s=np.float64(-1.0 if outcome.ttf_s is None else outcome.ttf_s),
            outcome=np.array(outcome.outcome.value),
        )
        log_s = time.monotonic() - t0
        return {
            "cache_key": world.cache_key(),
            "rrd_file": rrd_path.name,
            "rrd_bytes": rrd_path.stat().st_size,
            "outcome": outcome.outcome.value,
            "ttf_s": outcome.ttf_s,
            "timings_s": {
                "runtime_init_s": round(self.init_s, 3),
                "sim_s": round(sim_s, 3),
                "log_s": round(log_s, 3),
            },
        }
