import cv2
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
    assert 88 <= u <= 91  # middle of the ridge, not anywhere along it
    assert pick_pixel_from_mask(np.zeros((5, 5))) is None


def test_pick_pixel_is_stable_on_a_bottle_shape():
    """Body + narrow neck: the pick point sits mid-body and barely moves when the mask shifts by a pixel."""
    pts = []
    for shift in range(3):
        mask = np.zeros((120, 300), bool)
        mask[40:80, 20 + shift : 170 + shift] = True  # body
        mask[52:68, 170 + shift : 230 + shift] = True  # neck
        pts.append(pick_pixel_from_mask(mask))
    assert all(40 < u < 170 for u, _ in pts)
    assert max(u for u, _ in pts) - min(u for u, _ in pts) <= 3


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


def _cylinder_silhouette(cam: CameraModel, center: np.ndarray, axis: np.ndarray, radius: float, length: float):
    """Mask of a lying cylinder as the camera sees it (top + the side facing the camera)."""
    axis = axis / np.linalg.norm(axis)
    side = np.cross(axis, [0.0, 0.0, 1.0])
    pts = []
    for s in np.linspace(-length / 2, length / 2, 40):
        for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            pts.append(center + s * axis + radius * (np.cos(a) * side + np.sin(a) * np.array([0.0, 0.0, 1.0])))
    px = np.array([cam.world_to_pixel(p) for p in pts], dtype=np.float32)
    mask = np.zeros((cam.height, cam.width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(px).astype(np.int32), 1)
    return mask.astype(bool)


@pytest.mark.parametrize("xy", [(-0.20, 0.15), (-0.80, -0.15), (-0.50, 0.0), (-0.30, -0.10)])
@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.6, 0.8, 0.0)])
def test_pick_point_is_above_the_true_axis_off_center(xy, axis):
    """Perspective: the silhouette center is the object's middle, not its top. Regression for
    bottles lying across the belt being missed by ~5% of their distance from the camera axis."""
    cam = default_camera()
    radius = config.CLASS_SPECS["pet_bottle"].thickness / 2
    center = np.array([xy[0], xy[1], config.BELT_TOP_Z + radius])
    mask = _cylinder_silhouette(cam, center, np.array(axis), radius, 0.20)
    p = pick_point_world(cam, pick_pixel_from_mask(mask), "pet_bottle")
    a2 = np.array(axis[:2]) / np.linalg.norm(axis[:2])
    e = p[:2] - center[:2]
    assert abs(e[0] * -a2[1] + e[1] * a2[0]) < 0.004  # across the axis: decides if suction seals
    assert abs(e @ a2) < 0.015  # along the axis: still on the body
    assert p[2] == pytest.approx(config.BELT_TOP_Z + 2 * radius)


def test_motion_prediction():
    p = predict_position(np.array([0.0, 0.1, 0.8]), 1.0, 3.0, belt_speed=0.15)
    np.testing.assert_allclose(p, [0.3, 0.1, 0.8])
    assert time_to_reach_x(0.0, 0.3, 0.15) == pytest.approx(2.0)
