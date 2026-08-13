"""sweep/push_math.py — push window and force-vector semantics (spec §5)."""

import math

from sweep.push_math import GRAVITY_M_S2, push_active, push_force_newtons, push_vector_world


class TestPushWindow:
    def test_half_open_window(self) -> None:
        assert not push_active(1.999, 2.0, 0.5)
        assert push_active(2.0, 2.0, 0.5)
        assert push_active(2.499, 2.0, 0.5)
        assert not push_active(2.5, 2.0, 0.5)


class TestForceMagnitude:
    def test_percent_bodyweight_conversion(self) -> None:
        # 100% bodyweight on a 12 kg robot = m*g newtons.
        assert push_force_newtons(100.0, 12.0) == 12.0 * GRAVITY_M_S2

    def test_zero_magnitude_is_zero_force(self) -> None:
        assert push_force_newtons(0.0, 12.0) == 0.0


class TestForceVector:
    def test_lateral_push_at_zero_heading(self) -> None:
        fx, fy = push_vector_world(10.0, 90.0, 0.0)
        assert math.isclose(fx, 0.0, abs_tol=1e-12)
        assert math.isclose(fy, 10.0)

    def test_forward_push_rotates_with_heading(self) -> None:
        heading = math.pi / 2  # robot facing +y
        fx, fy = push_vector_world(10.0, 0.0, heading)
        assert math.isclose(fx, 0.0, abs_tol=1e-12)
        assert math.isclose(fy, 10.0)

    def test_magnitude_preserved(self) -> None:
        fx, fy = push_vector_world(7.5, 123.0, 0.8)
        assert math.isclose(math.hypot(fx, fy), 7.5)
