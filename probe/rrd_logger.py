"""MuJoCo scene → Rerun logging for instrumented rollouts (G5 test artifact; the live
probe path reuses this in M2).

Design: physics comes from the mjx rollout (qpos per step); kinematics for rendering are
recomputed on CPU MuJoCo (mj_forward per step), then every geom's world pose is logged
as a Transform3D on its own entity. Static geometry (meshes/primitives/colors) is logged
once. Timeline is seconds on the "sim_time" timeline; push window and failure moment are
logged as TextLog events plus scalar tracks (torso z, up-z) for the timeline panel —
these are the spec §9 "timeline event markers" (push start, failure).

mujoco is imported lazily so this module imports locally without the physics stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rerun as rr

TIMELINE = "sim_time"


def _geom_entity(mj_model: Any, geom_id: int) -> str:
    name = mj_model.geom(geom_id).name or f"geom{geom_id}"
    return f"world/robot/{name}_{geom_id}"


def log_static_scene(mj_model: Any) -> None:
    """Log every geom's geometry + color once (static)."""
    import mujoco

    for gid in range(mj_model.ngeom):
        geom = mj_model.geom(gid)
        entity = _geom_entity(mj_model, gid)
        rgba = geom.rgba
        color = [int(c * 255) for c in rgba[:3]] + [int(rgba[3] * 255)]
        gtype = geom.type[0] if hasattr(geom.type, "__len__") else geom.type
        size = geom.size
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            rr.log(
                entity,
                rr.Boxes3D(half_sizes=[[5.0, 5.0, 0.005]], colors=[color]),
                static=True,
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            rr.log(
                entity,
                rr.Ellipsoids3D(half_sizes=[[size[0]] * 3], colors=[color]),
                static=True,
            )
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            # mujoco capsules/cylinders: axis +z, half-length size[1]; rerun capsules
            # extend along +z FROM the origin — translate down half a length to center.
            rr.log(
                entity,
                rr.Capsules3D(
                    lengths=[2.0 * size[1]],
                    radii=[size[0]],
                    translations=[[0.0, 0.0, -size[1]]],
                    colors=[color],
                ),
                static=True,
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            rr.log(entity, rr.Boxes3D(half_sizes=[list(size)], colors=[color]), static=True)
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(geom.dataid[0]) if hasattr(geom.dataid, "__len__") else int(geom.dataid)
            vert_start = mj_model.mesh_vertadr[mesh_id]
            vert_num = mj_model.mesh_vertnum[mesh_id]
            face_start = mj_model.mesh_faceadr[mesh_id]
            face_num = mj_model.mesh_facenum[mesh_id]
            vertices = mj_model.mesh_vert[vert_start : vert_start + vert_num]
            faces = mj_model.mesh_face[face_start : face_start + face_num]
            rr.log(
                entity,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    albedo_factor=color,
                ),
                static=True,
            )
        # other geom types (hfield etc.): skipped; Go1 flat-terrain scene has none.


def log_step(mj_model: Any, data: Any, t_s: float) -> None:
    """Log every geom's world pose at time t (call after mj_forward)."""
    rr.set_time(TIMELINE, duration=t_s)
    for gid in range(mj_model.ngeom):
        rr.log(
            _geom_entity(mj_model, gid),
            rr.Transform3D(
                translation=data.geom_xpos[gid],
                mat3x3=np.asarray(data.geom_xmat[gid]).reshape(3, 3),
            ),
        )


def log_scalars(t_s: float, torso_z: float, upz: float) -> None:
    rr.set_time(TIMELINE, duration=t_s)
    rr.log("tracks/torso_z", rr.Scalars(torso_z))
    rr.log("tracks/up_z", rr.Scalars(upz))


def log_event(t_s: float, text: str, level: str = "INFO") -> None:
    rr.set_time(TIMELINE, duration=t_s)
    rr.log("events", rr.TextLog(text, level=level))


def log_push_arrow(t_s: float, active: bool, origin_xyz: Any, vx: float, vy: float) -> None:
    """Force visualization (M1-03): a red world-frame arrow anchored at the torso for
    the duration of the push window; cleared outside it. Length encodes magnitude
    (pre-scaled by the caller; sublinear so small pushes stay visible)."""
    rr.set_time(TIMELINE, duration=t_s)
    if active:
        rr.log(
            "world/push_force",
            rr.Arrows3D(
                origins=[origin_xyz],
                vectors=[[vx, vy, 0.0]],
                colors=[[220, 38, 38, 255]],
                radii=[0.035],
            ),
        )
    else:
        rr.log("world/push_force", rr.Clear(recursive=False))


def replay_blueprint(tracking_entity: str) -> Any:
    """Default blueprint shipped inside each replay .rrd (M1-03 requirements): 3D view
    with the camera tracking the trunk (no viewport hunting), side panels collapsed,
    time panel kept for scrubbing."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/world",
            eye_controls=rrb.archetypes.EyeControls3D(
                kind="Orbital",
                tracking_entity=tracking_entity,
                position=[1.6, -1.6, 0.9],
                eye_up=[0.0, 0.0, 1.0],
            ),
        ),
        collapse_panels=True,
    )


def log_trajectory_columns(
    mj_model: Any,
    geom_xpos: np.ndarray,
    geom_xmat: np.ndarray,
    times_s: np.ndarray,
    torso_z: np.ndarray,
    upz: np.ndarray,
) -> None:
    """Columnar (send_columns) version of the per-step loop — one call per entity for
    the whole rollout instead of one call per geom per step (M2-01b: the per-call
    Python overhead was ~4.3 s of every warm probe).

    geom_xpos: (T, ngeom, 3); geom_xmat: (T, ngeom, 9); times_s: (T,)
    """
    time_col = rr.TimeColumn(TIMELINE, duration=times_s)
    for gid in range(mj_model.ngeom):
        rr.send_columns(
            _geom_entity(mj_model, gid),
            indexes=[time_col],
            columns=rr.Transform3D.columns(
                translation=geom_xpos[:, gid, :],
                mat3x3=geom_xmat[:, gid, :].reshape(-1, 3, 3),
            ),
        )
    rr.send_columns(
        "tracks/torso_z", indexes=[time_col], columns=rr.Scalars.columns(scalars=torso_z)
    )
    rr.send_columns("tracks/up_z", indexes=[time_col], columns=rr.Scalars.columns(scalars=upz))


def log_push_arrow_columns(
    times_s: np.ndarray, active: np.ndarray, origins: np.ndarray, vx: float, vy: float
) -> None:
    """Columnar push arrow: present (with the torso-anchored origin) only on active
    steps; Clear at the first inactive step after the window."""
    idx = np.nonzero(active)[0]
    if len(idx) == 0:
        return
    rr.send_columns(
        "world/push_force",
        indexes=[rr.TimeColumn(TIMELINE, duration=times_s[idx])],
        columns=rr.Arrows3D.columns(
            origins=origins[idx],
            vectors=np.tile([vx, vy, 0.0], (len(idx), 1)),
            colors=np.tile([220, 38, 38, 255], (len(idx), 1)),
            radii=np.full(len(idx), 0.035),
        ),
    )
    end = int(idx[-1]) + 1
    if end < len(times_s):
        rr.set_time(TIMELINE, duration=float(times_s[end]))
        rr.log("world/push_force", rr.Clear(recursive=False))
