import numpy as np
import pytest

from trash_sorter import config
from trash_sorter.geometry import (
    CameraModel,
    default_camera,
    pick_pixel_from_mask,
    pick_point_world,
    predict_position,
    time_to_reach_x,
)


def test_center_pixel_hits_point_below_camera():
    cam = default_camera()
    p = cam.pixel_to_plane(cam.cx, cam.cy, config.BELT_TOP_Z)
    np.testing.assert_allclose(p, [config.CAMERA_X, 0.0, config.BELT_TOP_Z], atol=1e-9)


def test_image_axes_match_belt_axes():
    cam = default_camera()
    right = cam.pixel_to_plane(cam.cx + 100, cam.cy, config.BELT_TOP_Z)
    down = cam.pixel_to_plane(cam.cx, cam.cy + 100, config.BELT_TOP_Z)
    assert right[0] > config.CAMERA_X  # image right = belt +X (downstream)
    assert down[1] < 0.0  # image down = world -Y


def test_project_unproject_round_trip_tilted_camera():
    q = np.array([0.995, 0.05, -0.08, 0.02])
    cam = CameraModel.from_physical(18, 20.955, 15.716, (640, 480), (0.1, 0.2, 1.4), q / np.linalg.norm(q))
    rng = np.random.default_rng(0)
    for _ in range(50):
        pt = np.array([rng.uniform(-0.2, 0.4), rng.uniform(-0.1, 0.5), rng.uniform(0.7, 0.85)])
        u, v = cam.world_to_pixel(pt)
        np.testing.assert_allclose(cam.pixel_to_plane(u, v, pt[2]), pt, atol=1e-9)


def test_footprint_covers_belt_width():
    cam = default_camera()
    top = cam.pixel_to_plane(cam.cx, 0, config.BELT_TOP_Z)
    assert top[1] > config.BELT_WIDTH / 2


def test_pick_pixel_is_deepest_point_of_mask():
    mask = np.zeros((100, 200), bool)
    mask[20:60, 30:150] = True  # rectangle: deepest points lie on the center row
    u, v = pick_pixel_from_mask(mask)
    assert 38 <= v <= 41
    assert 49 <= u <= 130
    assert pick_pixel_from_mask(np.zeros((5, 5))) is None


def test_pick_pixel_on_l_shape_stays_inside():
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:30] = True
    mask[70:90, 10:90] = True
    u, v = pick_pixel_from_mask(mask)
    assert mask[v, u]


def test_pick_point_uses_class_thickness():
    cam = default_camera()
    p = pick_point_world(cam, (cam.cx, cam.cy), "aluminum_can")
    assert p[2] == pytest.approx(config.BELT_TOP_Z + config.CLASS_SPECS["aluminum_can"].thickness)


def test_motion_prediction():
    p = predict_position(np.array([0.0, 0.1, 0.8]), 1.0, 3.0, belt_speed=0.15)
    np.testing.assert_allclose(p, [0.3, 0.1, 0.8])
    assert time_to_reach_x(0.0, 0.3, 0.15) == pytest.approx(2.0)
