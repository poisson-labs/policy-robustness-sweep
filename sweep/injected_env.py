"""Physics-only perturbation injection for Playground MjxEnv (G3/G4 executor core).

Container-only (imports jax); the semantics it implements are the locally-tested pure
functions in sweep/push_math.py. Perturbations enter through physics ONLY (spec §4 G3):
- push: world-frame force written to data.xfrc_applied on the torso body, constant over
  the half-open window [start, start+0.5s), direction fixed at push-onset heading;
- friction: floor geom friction patch on the mjx model (patch_floor_friction) — friction
  None = model untouched (the nominal world).
The observation/action contract is never touched.

Two entry points share one functional core (`push_step`):
- `PushInjectionWrapper` — single fixed world, duck-typed under wrap_for_brax_training
  (G3-verified: bitwise physics-identical at identity, see docs/equivalence-report.md).
- `push_step` directly — force magnitude / direction / start time as TRACED arguments,
  so the sweep executor can vmap one compiled step over many worlds at once.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from configs.world import PUSH_DURATION_S, WorldConfig

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


def patch_floor_friction(mjx_model: Any, floor_geom_id: int, mu: Any) -> Any:
    """Return an mjx model with the floor geom's sliding friction set to mu (traceable)."""
    geom_friction = mjx_model.geom_friction.at[floor_geom_id, 0].set(mu)
    return mjx_model.replace(geom_friction=geom_friction)


def yaw_from_quat(quat: jax.Array) -> jax.Array:
    """Yaw (world z rotation) from a wxyz quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def init_push_info(state: Any) -> Any:
    """Add the push bookkeeping keys to a freshly reset state's info dict."""
    state.info["push_yaw"] = jnp.zeros(())
    state.info["push_yaw_captured"] = jnp.zeros(())
    return state


def push_step(
    env: Any,
    state: Any,
    action: jax.Array,
    force_n: jax.Array,
    direction_rad: jax.Array,
    start_s: jax.Array,
    torso_body_id: int,
) -> Any:
    """One env step with the push force written into xfrc_applied (functional core).

    force_n / direction_rad / start_s are traced scalars — vmappable for the sweep.
    Semantics: half-open window, world-frame vector fixed at push-onset heading
    (spec §5; pure reference: sweep/push_math.py).
    """
    t = state.data.time
    active = (t >= start_s) & (t < start_s + PUSH_DURATION_S)

    yaw_now = yaw_from_quat(state.data.qpos[3:7])
    captured = state.info["push_yaw_captured"]
    onset_yaw = jnp.where(captured > 0, state.info["push_yaw"], yaw_now)
    new_captured = jnp.where(active, jnp.ones(()), captured)

    angle = onset_yaw + direction_rad
    magnitude = jnp.where(active, force_n, 0.0)
    fx = magnitude * jnp.cos(angle)
    fy = magnitude * jnp.sin(angle)

    xfrc = jnp.zeros_like(state.data.xfrc_applied)
    xfrc = xfrc.at[torso_body_id, 0].set(fx)
    xfrc = xfrc.at[torso_body_id, 1].set(fy)
    state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

    next_state = env.step(state, action)
    next_state.info["push_yaw"] = onset_yaw
    next_state.info["push_yaw_captured"] = new_captured
    return next_state


class PushInjectionWrapper:
    """Duck-typed MjxEnv wrapper for a single fixed world (delegates to push_step).

    Sits INSIDE wrap_for_brax_training (which vmaps it), so all state math is
    single-world and gets vmapped for free.
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
        return init_push_info(self._env.reset(rng))

    def step(self, state: Any, action: jax.Array) -> Any:
        return push_step(
            self._env,
            state,
            action,
            jnp.asarray(self._force_n),
            self._direction_rad,
            jnp.asarray(self._world.push_start_s),
            self._torso_body_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
