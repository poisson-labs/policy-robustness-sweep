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
