"""render/ — offline video production from rollout state (spec §3 4b; M1-04).

Produces, per spec §12/§11 and the M1-04 acceptance criteria:
- failure-centered clips: window t_push - 0.5 s → t_fail + 1 s, with a 4x slow-mo
  segment bracketing the failure moment (±0.3 s);
- the 16-seed ghost overlay for ONE boundary cell (spec §11 montage): all seeds
  superimposed, semi-transparent, synchronized to the (shared) push;
- a 3x3 cross-boundary tiled composite, cells synchronized on push start;
- mobile-fallback clips (shorter, smaller) per selected cell.

Rollouts re-simulate on demand (batched scan on the G3-verified push_step path — warm
sim measured 0.58 s per rollout, so trajectories are cheaper to recompute than to
store); frames render on CPU MuJoCo via EGL with a camera tracking the torso. Ghost
compositing: each seed's frame is rendered against the same empty scene; robot pixels
are isolated by difference-masking and alpha-blended onto the base frame. Never runs in
the live probe path.

    uv run modal run render/clips.py::all_clips --run-id <id>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

render_image = (
    base_image.apt_install("libegl1", "libgl1", "libglvnd0", "libosmesa6")
    .uv_pip_install("pydantic==2.13.4", "imageio[ffmpeg]==2.37.4")
    .env({"MUJOCO_GL": "egl"})
    .add_local_python_source("train", "sweep", "configs", "probe", "reduce")
)

app = modal.App("opw-render")

WIDTH, HEIGHT = 640, 480
MOBILE_WIDTH, MOBILE_HEIGHT = 480, 360
FPS = 50  # = 1/ctrl_dt: real-time playback
SLOWMO_FACTOR = 4
SLOWMO_HALF_WINDOW_S = 0.3
GHOST_ALPHA = 0.30
GHOST_CELL = {"mu": 0.50, "push_pct_bw": 80.0}  # S=0.5 — the most contested cell
COMPOSITE_MUS = [0.40, 0.50, 0.60]
COMPOSITE_PUSHES = [60.0, 80.0, 100.0]


@app.function(
    image=render_image,
    gpu="A100-80GB",
    timeout=7200,  # first full render measured 3416 s; 3600 left no headroom (timed out)
    volumes={VOLUME_MOUNT: checkpoints},
)
def render_clips(
    run_id: str,
    selection: dict[str, Any],
    checkpoint: str = "converged",
) -> dict[str, Any]:
    import time

    import imageio.v2 as iio
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    from configs.world import PUSH_DURATION_S, ROLLOUT_S
    from reduce.classify import classify_rollout
    from sweep.injected_env import (
        find_body_id,
        find_floor_geom_id,
        init_push_info,
        patch_floor_friction,
        push_step,
    )
    from train.modal_app import ENV_NAME as env_name

    t_all = time.monotonic()
    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    params = brax_model.load_params(str(run_dir / "checkpoints" / checkpoint))
    out_dir = run_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    push_start = 2.0
    direction_rad = jnp.deg2rad(jnp.asarray(90.0))
    start_arr = jnp.asarray(push_start)

    def batched_rollout(model: Any, seeds: list[int], force_n: float) -> Any:
        def one(key: jax.Array) -> jax.Array:
            env._mjx_model = model
            state = init_push_info(env.reset(key))
            state.info["command"] = jnp.array([1.0, 0.0, 0.0])
            state.info["steps_until_next_cmd"] = jnp.asarray(10_000, dtype=jnp.int32)

            def body(carry: Any, _: Any) -> tuple[Any, jax.Array]:
                st = carry
                action, _extras = policy(st.obs, jax.random.PRNGKey(0))
                nst = push_step(
                    env, st, action, jnp.asarray(force_n), direction_rad, start_arr, torso_id
                )
                return nst, nst.data.qpos

            _, qpos = jax.lax.scan(body, state, None, length=steps)
            return qpos

        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
        return np.asarray(jax.vmap(one)(keys))  # (n_seeds, steps, nq)

    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)
    mobile_renderer = mujoco.Renderer(mj_model, height=MOBILE_HEIGHT, width=MOBILE_WIDTH)
    mj_data = mujoco.MjData(mj_model)
    track_cam = mujoco.MjvCamera()
    track_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    track_cam.trackbodyid = torso_id
    track_cam.distance = 1.7
    track_cam.elevation = -18.0
    track_cam.azimuth = 125.0
    angle_cams = []
    for az in (90.0, 135.0, 180.0):  # side / three-quarter / front (Taylor: more angles)
        c3 = mujoco.MjvCamera()
        c3.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        c3.trackbodyid = torso_id
        c3.distance = 1.7
        c3.elevation = -18.0
        c3.azimuth = az
        angle_cams.append(c3)
    # Ghost compositing needs ALIGNED frames: one FIXED camera for every seed (a
    # tracking camera would follow each seed's own torso — first-draft bug).
    fixed_cam = mujoco.MjvCamera()
    fixed_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fixed_cam.lookat[:] = [0.0, 0.0, 0.35]  # aligned ghosts: onset at origin, heading +x
    fixed_cam.distance = 3.8
    fixed_cam.elevation = -20.0
    fixed_cam.azimuth = 120.0

    def frame_at(
        qpos: np.ndarray,
        r: Any,
        cam: Any = None,
        arrow_dir: tuple[float, float] | None = None,
    ) -> np.ndarray:
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        r.update_scene(mj_data, camera=cam if cam is not None else track_cam)
        if arrow_dir is not None:
            scene = r.scene
            if scene.ngeom < scene.maxgeom:
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(9),
                    np.array([0.86, 0.15, 0.15, 1.0], dtype=np.float32),
                )
                p_from = mj_data.xpos[torso_id].copy()
                p_from[2] += 0.05
                p_to = p_from + np.array([arrow_dir[0], arrow_dir[1], 0.0])
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.014, p_from, p_to)
                scene.ngeom += 1
        return r.render()

    def align_to_onset_frame(qpos_traj: np.ndarray, onset_i: int) -> np.ndarray:
        """Rigidly transform the root so heading = +x and position = origin at push
        onset. Ghosts then differ only by their RESPONSE to the shove, not by the
        randomized spawn heading (Taylor's feedback: unaligned seeds 'dart randomly')."""
        out = qpos_traj.copy()
        x0, y0 = qpos_traj[onset_i, 0], qpos_traj[onset_i, 1]
        qw, qx, qy, qz = qpos_traj[onset_i, 3:7]
        yaw0 = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        dx, dy = qpos_traj[:, 0] - x0, qpos_traj[:, 1] - y0
        out[:, 0] = c * dx - s * dy
        out[:, 1] = s * dx + c * dy
        rw, rz = np.cos(-yaw0 / 2.0), np.sin(-yaw0 / 2.0)
        w2, x2, y2, z2 = (qpos_traj[:, i] for i in (3, 4, 5, 6))
        out[:, 3] = rw * w2 - rz * z2
        out[:, 4] = rw * x2 - rz * y2
        out[:, 5] = rw * y2 + rz * x2
        out[:, 6] = rw * z2 + rz * w2
        return out

    def ttf_of(qpos_traj: np.ndarray) -> float | None:
        w, x, y, z = (qpos_traj[:, i] for i in (3, 4, 5, 6))
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        return classify_rollout(qpos_traj[:, 2], roll, pitch, dt).ttf_s

    def failure_window_frames(
        qpos_traj: np.ndarray,
        ttf_s: float | None,
        r: Any,
        arrow_len: float = 0.0,
        multi_angle: bool = False,
    ) -> list[np.ndarray]:
        """t_push-0.5 → (t_fail+0.5 | push end+1.5 if censored), 4x slow-mo around
        failure. Post-failure footage is trimmed: the model is feet-only-collision
        (training-sim property, DEVLOG Session 20), so toppled bodies sink through the
        floor — the topple reads, the sink is clipped."""
        t_end = (ttf_s + 0.5) if ttf_s is not None else (push_start + PUSH_DURATION_S + 1.5)
        i0 = max(0, int((push_start - 0.5) / dt))
        i1 = min(steps, int(t_end / dt))
        qw, qx, qy, qz = (qpos_traj[:, i] for i in (3, 4, 5, 6))
        yaws = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        onset_i = max(0, int(push_start / dt))
        angle = yaws[onset_i] + np.pi / 2.0  # push dial 90 deg relative to onset heading
        frames: list[np.ndarray] = []
        for i in range(i0, i1):
            t = (i + 1) * dt
            in_window = push_start <= (i * dt) < push_start + PUSH_DURATION_S
            arrow = (
                (float(np.cos(angle)) * arrow_len, float(np.sin(angle)) * arrow_len)
                if in_window
                else None
            )
            if multi_angle:
                frame = np.concatenate(
                    [frame_at(qpos_traj[i], r, cam, arrow) for cam in angle_cams], axis=1
                )
            else:
                frame = frame_at(qpos_traj[i], r, None, arrow)
            repeats = (
                SLOWMO_FACTOR if ttf_s is not None and abs(t - ttf_s) <= SLOWMO_HALF_WINDOW_S else 1
            )
            frames.extend([frame] * repeats)
        return frames

    def write_mp4(path: Path, frames: list[np.ndarray]) -> int:
        iio.mimwrite(str(path), frames, fps=FPS, codec="libx264", quality=8)
        return path.stat().st_size

    outputs: dict[str, Any] = {"clips": []}

    # 1) failure-centered + mobile-fallback clips for every selected cell
    from sweep.seed_scheme import sweep_prng_seed

    cells = [
        {
            "mu": c["mu"],
            "push_pct_bw": c["push_pct_bw"],
            "seed_idx": 0,
            "seed": sweep_prng_seed(c["mu"], c["push_pct_bw"], 0),
        }
        for c in selection["boundary_cells"]
    ] + [
        {
            "mu": r["mu"],
            "push_pct_bw": r["push_pct_bw"],
            "seed_idx": r["seed_idx"],
            "seed": r.get(
                "prng_seed_override",
                sweep_prng_seed(r["mu"], r["push_pct_bw"], r["seed_idx"]),
            ),
        }
        for r in selection["recoveries"]
    ]
    for cell in cells:
        model = patch_floor_friction(base_model, floor_id, cell["mu"])
        force_n = cell["push_pct_bw"] / 100.0 * mass_kg * g
        qpos = batched_rollout(model, [cell["seed"]], force_n)[0]
        ttf = ttf_of(qpos)
        stem = f"mu{cell['mu']:.2f}-push{int(cell['push_pct_bw'])}-seed{cell['seed_idx']}"
        arrow_len = 0.9 * (cell["push_pct_bw"] / 100.0) ** 0.5  # matches replay arrows
        frames = failure_window_frames(qpos, ttf, renderer, arrow_len, multi_angle=True)
        size = write_mp4(out_dir / f"fail-{stem}.mp4", frames)
        mobile_frames = failure_window_frames(qpos, ttf, mobile_renderer, arrow_len)
        mobile_size = write_mp4(out_dir / f"mobile-{stem}.mp4", mobile_frames)
        outputs["clips"].append(
            {"cell": stem, "ttf_s": ttf, "fail_bytes": size, "mobile_bytes": mobile_size}
        )
        checkpoints.commit()  # persist per-clip: a timeout must not lose finished work

    # 2) 16-seed ghost overlay for the hero boundary cell
    model = patch_floor_friction(base_model, floor_id, GHOST_CELL["mu"])
    force_n = GHOST_CELL["push_pct_bw"] / 100.0 * mass_kg * g
    ghost_seeds = [
        sweep_prng_seed(GHOST_CELL["mu"], GHOST_CELL["push_pct_bw"], s) for s in range(16)
    ]
    qpos_all = batched_rollout(model, ghost_seeds, force_n)
    onset_i = max(0, int(2.0 / dt))
    qpos_all = np.stack([align_to_onset_frame(qpos_all[s], onset_i) for s in range(16)])
    ghost_frames: list[np.ndarray] = []
    # static empty scene (fixed camera): robot dropped far below ground, rendered once
    hq = qpos_all[0][0].copy()
    hq[2] = -50.0
    empty = frame_at(hq, renderer, fixed_cam).astype(np.float32)
    for i in range(steps):
        acc = empty.copy()
        for s in range(16):
            f = frame_at(qpos_all[s][i], renderer, fixed_cam).astype(np.float32)
            mask = (np.abs(f - empty).sum(axis=2) > 30.0)[..., None]
            acc = np.where(mask, (1 - GHOST_ALPHA) * acc + GHOST_ALPHA * f, acc)
        ghost_frames.append(acc.astype(np.uint8))
    ghost_size = write_mp4(
        out_dir / f"ghost16-mu{GHOST_CELL['mu']:.2f}-push{int(GHOST_CELL['push_pct_bw'])}.mp4",
        ghost_frames,
    )
    outputs["ghost_bytes"] = ghost_size

    # 3) 3x3 cross-boundary tiled composite (synchronized on push start)
    tiles: list[list[np.ndarray]] = []
    for mu in COMPOSITE_MUS:
        model = patch_floor_friction(base_model, floor_id, mu)
        for pct in COMPOSITE_PUSHES:
            qpos = batched_rollout(model, [sweep_prng_seed(mu, pct, 0)], pct / 100.0 * mass_kg * g)[
                0
            ]
            tiles.append([frame_at(qpos[i], mobile_renderer) for i in range(steps)])
    comp_frames = []
    for i in range(steps):
        rows = [np.concatenate([tiles[r * 3 + c][i] for c in range(3)], axis=1) for r in range(3)]
        comp_frames.append(np.concatenate(rows, axis=0))
    comp_size = write_mp4(out_dir / "composite-3x3.mp4", comp_frames)
    outputs["composite_bytes"] = comp_size
    outputs["wall_s"] = round(time.monotonic() - t_all, 1)
    checkpoints.commit()
    return outputs


@app.local_entrypoint()
def all_clips(run_id: str, checkpoint: str = "converged") -> None:
    docs = Path(__file__).parent.parent / "docs/measurements"
    selection = json.loads((docs / "2026-08-14-replay-selection.json").read_text())
    # Recoveries use the replay batch's CHOSEN seeds (survivor search may have moved off
    # the recorded index — Session 19); parse them out of the mirrored batch results.
    batch = json.loads((docs / "2026-08-14-replay-batch.json").read_text())
    chosen: dict[tuple[float, float], int] = {}
    for entry in batch:
        if "recovery_attempts" not in entry:
            continue
        parts = dict(kv.split("=") for kv in entry["cache_key"].split("|")[1:])
        chosen[(float(parts["mu"]), float(parts["push"]))] = int(parts["seed"])
    for r in selection["recoveries"]:
        r["prng_seed_override"] = chosen[(r["mu"], r["push_pct_bw"])]
    print(json.dumps(render_clips.remote(run_id, selection, checkpoint), indent=2))
