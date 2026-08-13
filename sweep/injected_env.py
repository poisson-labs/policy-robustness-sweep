"""Physics-only perturbation injection for Playground MjxEnv (G3/G4 executor core).

Container-only (imports jax); the semantics it implements are the locally-tested pure
functions in sweep/push_math.py. Perturbations enter through physics ONLY (spec §4 G3):
- push: world-frame force written to data.xfrc_applied on the torso body, constant over
  the half-open window [start, start+0.5s), direction fixed at push-onset heading;
- friction: floor geom friction patch on the mjx model (patch_floor_friction), applied
  at env construction — friction None = model untouched (the nominal world).
The observation/action contract is never touched. At identity (magnitude 0, friction
None) the wrapper still writes (zero) forces through the same code path — G3's exact
check relies on that.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from configs.world import PUSH_DURATION_S, WorldConfig
from sweep.push_math import GRAVITY_M_S2  # noqa: F401  (documented parity; model g is used)

TORSO_NAME_CANDIDATES = ("trunk", "torso", "base")
FLOOR_GEOM_CANDIDATES = ("floor", "ground", "plane")


def find_body_id(mj_model: Any, candidates: tuple[str, ...] = TORSO_NAME_CANDIDATES) -> int:
    names = [mj_model.body(i).name for i in range(mj_model.nbody)]
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    raise ValueError(f"no torso body found; bodies={names}")


def find_floor_geom_id(mj_model: Any, candidates: tuple[str, ...] = FLOOR_GEOM_CANDIDATES) -> int:
    names = [mj_model.geom(i).name for i in range(mj_model.ngeom)]
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    raise ValueError(f"no floor geom found; geoms={names}")


def patch_floor_friction(mjx_model: Any, floor_geom_id: int, mu: float) -> Any:
    """Return an mjx model with the floor geom's sliding friction set to mu."""
    geom_friction = mjx_model.geom_friction.at[floor_geom_id, 0].set(mu)
    return mjx_model.replace(geom_friction=geom_friction)


def yaw_from_quat(quat: jax.Array) -> jax.Array:
    """Yaw (world z rotation) from a wxyz quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PushInjectionWrapper:
    """Duck-typed MjxEnv wrapper writing the push force into xfrc_applied each step.

    Sits INSIDE wrap_for_brax_training (which vmaps it), so all state math is
    single-world and gets vmapped for free. Push-onset heading is captured into
    state.info the first step the window is active and reused for the rest of the
    window (constant world-frame force, spec §5).
    """

    def __init__(self, env: Any, world: WorldConfig, torso_body_id: int, mass_kg: float):
        self._env = env
        self._world = world
        self._torso_body_id = torso_body_id
        # Bodyweight uses the model's own gravity magnitude, recorded by the caller;
        # spec §5 requires the model mass recorded/displayed wherever %BW appears.
        g = float(abs(env.mj_model.opt.gravity[2]))
        self._force_n = world.push_magnitude_pct_bw / 100.0 * mass_kg * g
        self._direction_rad = jnp.deg2rad(world.push_direction_deg)

    def reset(self, rng: jax.Array) -> Any:
        state = self._env.reset(rng)
        state.info["push_yaw"] = jnp.zeros(())
        state.info["push_yaw_captured"] = jnp.zeros(())
        return state

    def step(self, state: Any, action: jax.Array) -> Any:
        t = state.data.time
        start = self._world.push_start_s
        active = (t >= start) & (t < start + PUSH_DURATION_S)

        yaw_now = yaw_from_quat(state.data.qpos[3:7])
        captured = state.info["push_yaw_captured"]
        onset_yaw = jnp.where(captured > 0, state.info["push_yaw"], yaw_now)
        new_captured = jnp.where(active, jnp.ones(()), captured)

        angle = onset_yaw + self._direction_rad
        magnitude = jnp.where(active, self._force_n, 0.0)
        fx = magnitude * jnp.cos(angle)
        fy = magnitude * jnp.sin(angle)

        xfrc = jnp.zeros_like(state.data.xfrc_applied)
        xfrc = xfrc.at[self._torso_body_id, 0].set(fx)
        xfrc = xfrc.at[self._torso_body_id, 1].set(fy)
        state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

        next_state = self._env.step(state, action)
        next_state.info["push_yaw"] = onset_yaw
        next_state.info["push_yaw_captured"] = new_captured
        return next_state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
