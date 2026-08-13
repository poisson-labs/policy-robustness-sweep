"""Pure push-perturbation math (quadruped-specific; spec §5 semantics).

The mjx application layer (writing xfrc_applied on the torso, patching model friction)
lives beside the eval protocol and runs only in containers; everything here is plain
Python so the semantics are unit-tested locally. Perturbations enter through physics
ONLY (G3): these functions decide *what* force exists at time t — never touching the
observation/action contract.
"""

from __future__ import annotations

import math

GRAVITY_M_S2 = 9.81


def push_active(t_s: float, start_s: float, duration_s: float) -> bool:
    """Constant-force window [start, start + duration). Half-open so a rollout stepping
    exactly onto the end time applies no force there."""
    return start_s <= t_s < start_s + duration_s


def push_force_newtons(magnitude_pct_bw: float, mass_kg: float) -> float:
    """% bodyweight → Newtons (spec §5: %BW is the primary unit, N secondary)."""
    return magnitude_pct_bw / 100.0 * mass_kg * GRAVITY_M_S2


def push_vector_world(
    magnitude_n: float, direction_deg: float, heading_yaw_rad: float
) -> tuple[float, float]:
    """World-frame horizontal force vector.

    `direction_deg` is the dial angle relative to the robot's heading at push onset
    (0° = straight ahead, 90° = lateral-left, measured counterclockwise from heading);
    the vector is fixed in the world frame for the whole window (constant force,
    spec §5), i.e. it does not re-track heading during the push.
    """
    angle = heading_yaw_rad + math.radians(direction_deg)
    return magnitude_n * math.cos(angle), magnitude_n * math.sin(angle)
