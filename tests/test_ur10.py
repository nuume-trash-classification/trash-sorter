import numpy as np
import pytest

from trash_sorter import config
from trash_sorter.picking import PickParams
from trash_sorter.ur10 import HOME_Q, Ur10Kinematics, fk_dh, ik_dh, tool_rotation_down


def test_ik_round_trip_random_poses():
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = fk_dh(q)
        sols = ik_dh(T)
        assert sols, "reachable pose must have solutions"
        assert min(np.abs(fk_dh(s) - T).max() for s in sols) < 1e-8


def test_matches_isaac_sim_sample():
    # Recorded from Isaac Sim 6.1 (UR10 asset, base at origin): ee_link position for q = 0.
    k = Ur10Kinematics(base_position=(0, 0, 0), tip_offset=0.0)
    p, _ = k.tip_pose(np.zeros(6))
    np.testing.assert_allclose(p, [1.1843, 0.2561, 0.0116], atol=1e-4)


def test_home_points_cup_down():
    k = Ur10Kinematics()
    _, R = k.tip_pose(HOME_Q)
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-2)


def _key_points():
    pts = []
    for x in np.linspace(*config.PICK_ZONE_X, 5):
        for y in config.SPAWN_Y_RANGE + (0.0,):
            for z in (config.BELT_TOP_Z + 0.07, PickParams().lift_z):
                pts.append((x, y, z))
    for bx, by in config.BIN_CENTERS.values():
        pts.append((bx, by, config.DROP_HEIGHT))
    pts.append(PickParams().idle_position)
    return pts


@pytest.mark.parametrize("point", _key_points())
def test_workspace_reachable_cup_down(point):
    k = Ur10Kinematics()
    q = k.solve_down(point, HOME_Q)
    assert q is not None, f"unreachable: {point}"
    p, R = k.tip_pose(q)
    np.testing.assert_allclose(p, point, atol=1e-6)
    np.testing.assert_allclose(R, tool_rotation_down(0.0), atol=1e-6)


def test_ik_branch_is_continuous_along_a_path():
    k = Ur10Kinematics()
    q = k.solve_down(PickParams().idle_position, HOME_Q)
    start = np.array(PickParams().idle_position)
    for bx, by in config.BIN_CENTERS.values():
        q_prev = q
        for s in np.linspace(0, 1, 100):
            p = (1 - s) * start + s * np.array([bx, by, config.DROP_HEIGHT])
            q_new = k.solve_down(p, q_prev)
            assert q_new is not None
            assert np.abs(q_new - q_prev).max() < 0.2, "joint jump: IK switched branch"
            q_prev = q_new
