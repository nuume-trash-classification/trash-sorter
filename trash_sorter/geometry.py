"""Camera and belt geometry: pixel <-> belt coordinates, pick point from a mask, motion prediction.

Everything here only needs what a real system would have: camera calibration
(intrinsics + pose), the belt plane height, per-class object thickness and the
belt speed. No simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from trash_sorter import config


def quat_wxyz_to_matrix(q: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera. Uses the USD/OpenGL convention: the camera looks along its local -Z, +Y is up.

    ``rotation`` maps camera-frame vectors to world-frame vectors; ``position`` is the
    optical center in world coordinates.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    position: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_physical(
        cls,
        focal_mm: float,
        h_aperture_mm: float,
        v_aperture_mm: float,
        resolution: tuple[int, int],
        position: tuple[float, float, float] | np.ndarray,
        orientation_wxyz: tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    ) -> CameraModel:
        """Build from the same numbers Isaac Sim uses (focal length, apertures in mm)."""
        width, height = resolution
        return cls(
            fx=focal_mm / h_aperture_mm * width,
            fy=focal_mm / v_aperture_mm * height,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
            position=np.asarray(position, dtype=np.float64),
            rotation=quat_wxyz_to_matrix(orientation_wxyz),
        )

    def pixel_ray(self, u: float, v: float) -> np.ndarray:
        """World-frame unit direction of the ray through pixel (u, v) (u = column, v = row)."""
        d_cam = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = self.rotation @ d_cam
        return d / np.linalg.norm(d)

    def pixel_to_plane(self, u: float, v: float, plane_z: float) -> np.ndarray:
        """Intersect the pixel ray with the horizontal plane ``z = plane_z``. Returns (x, y, z)."""
        d = self.pixel_ray(u, v)
        if abs(d[2]) < 1e-9:
            raise ValueError("Pixel ray is parallel to the plane")
        s = (plane_z - self.position[2]) / d[2]
        if s <= 0:
            raise ValueError("Plane is behind the camera")
        return self.position + s * d

    def world_to_pixel(self, point: np.ndarray | tuple[float, float, float]) -> tuple[float, float]:
        """Project a world point to (u, v) pixel coordinates."""
        p_cam = self.rotation.T @ (np.asarray(point, dtype=np.float64) - self.position)
        if p_cam[2] >= 0:
            raise ValueError("Point is behind the camera")
        u = self.cx + self.fx * p_cam[0] / -p_cam[2]
        v = self.cy - self.fy * p_cam[1] / -p_cam[2]
        return float(u), float(v)


def default_camera(
    position: tuple[float, float, float] | None = None,
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> CameraModel:
    """The sorter camera as built by ``sim/scene.py`` (looking straight down, image right = belt +X)."""
    if position is None:
        position = (config.CAMERA_X, 0.0, config.BELT_TOP_Z + config.CAMERA_HEIGHT)
    return CameraModel.from_physical(
        config.CAMERA_FOCAL_MM,
        config.CAMERA_H_APERTURE_MM,
        config.CAMERA_V_APERTURE_MM,
        config.CAMERA_RESOLUTION,
        position,
        orientation_wxyz,
    )


def pick_pixel_from_mask(mask: np.ndarray, ridge: float = 0.9) -> tuple[int, int] | None:
    """Point inside the mask that is farthest from the mask edge (distance transform).

    Returns (u, v) = (column, row), or None for an empty mask. This is the flattest,
    most central spot of the object, which is where a suction cup seals best.

    On elongated objects (bottles) the maximum is a whole ridge along the body, and a plain
    argmax jumps along it from frame to frame. So we take the middle of all pixels within
    ``ridge`` x max distance, then snap to the nearest such pixel (stays inside the mask).
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return None
    # Pad so pixels on the image border count as "edge" too.
    padded = np.pad(m, 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    vs, us = np.nonzero(dist >= ridge * dist.max())
    cu, cv = us.mean(), vs.mean()
    k = int(np.argmin((us - cu) ** 2 + (vs - cv) ** 2))
    return int(us[k]), int(vs[k])


def pick_point_world(
    camera: CameraModel, pixel: tuple[float, float], class_name: str, belt_top_z: float = config.BELT_TOP_Z
) -> np.ndarray:
    """Pixel -> 3D pick point on the object's top surface (belt plane + known class thickness).

    The mask's center pixel looks at the object's *middle*, not its top: off the camera axis the
    camera also sees one side, so the silhouette is centered on the object's mid-height. So the
    ray is intersected at half the thickness (the object's center), and z is then set to the top.
    Using the top plane instead shrinks positions toward the camera axis by ~5%, which also makes
    the fitted belt velocity ~5% too slow and the arm lag behind the object.
    """
    thickness = config.CLASS_SPECS[class_name].thickness
    p = camera.pixel_to_plane(pixel[0], pixel[1], belt_top_z + thickness / 2)
    p[2] = belt_top_z + thickness
    return p


def predict_position(
    position: np.ndarray, t_observed: float, t_future: float, belt_speed: float = config.BELT_SPEED
) -> np.ndarray:
    """Where an object seen at ``position`` at ``t_observed`` will be at ``t_future`` (moves along +X)."""
    p = np.array(position, dtype=np.float64)
    p[0] += belt_speed * (t_future - t_observed)
    return p


def time_to_reach_x(x_now: float, x_target: float, belt_speed: float = config.BELT_SPEED) -> float:
    """Seconds until an object at ``x_now`` reaches ``x_target`` (negative if already past)."""
    return (x_target - x_now) / belt_speed
