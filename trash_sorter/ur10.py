"""UR10 forward / inverse kinematics (closed form), matched to the Isaac Sim UR10 asset.

The asset's ``base_link`` is the classic UR DH base frame rotated 180 deg about Z, and its
``ee_link`` +X axis equals the DH tool Z axis. Verified against Isaac Sim 6.1 to < 1e-6 m.
The suction tip sits ``config.SUCTION_TIP_OFFSET`` along the tool Z axis.

Analytic IK follows K. Hawkins, "Analytic Inverse Kinematics for the Universal Robots
UR-5/UR-10 Arms" (2013).
"""

from __future__ import annotations

import math

import numpy as np

from trash_sorter import config

# Standard UR10 DH parameters.
D1, A2, A3, D4, D5, D6 = 0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922
_D = (D1, 0.0, 0.0, D4, D5, D6)
_A = (0.0, A2, A3, 0.0, 0.0, 0.0)
_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)
_RZ_PI = np.diag([-1.0, -1.0, 1.0])

# Joint order of the Isaac Sim articulation.
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_dh(q: np.ndarray | list[float]) -> np.ndarray:
    """4x4 pose of the DH tool frame in the DH base frame."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(q[i], _D[i], _A[i], _ALPHA[i])
    return T


def ik_dh(T06: np.ndarray) -> list[np.ndarray]:
    """All (up to 8) joint solutions reaching the DH tool pose ``T06``. Angles in (-pi, pi]."""
    sols: list[np.ndarray] = []
    p05 = T06 @ np.array([0.0, 0.0, -D6, 1.0])
    r = math.hypot(p05[0], p05[1])
    if r < abs(D4) + 1e-9:
        return sols
    psi = math.atan2(p05[1], p05[0])
    phi = math.acos(D4 / r)
    T60 = np.linalg.inv(T06)
    for th1 in (psi + phi + math.pi / 2, psi - phi + math.pi / 2):
        s1, c1 = math.sin(th1), math.cos(th1)
        c5 = (T06[0, 3] * s1 - T06[1, 3] * c1 - D4) / D6
        if abs(c5) > 1.0 + 1e-9:
            continue
        c5 = max(-1.0, min(1.0, c5))
        for th5 in (math.acos(c5), -math.acos(c5)):
            s5 = math.sin(th5)
            if abs(s5) < 1e-9:
                th6 = 0.0  # wrist singularity: q6 is free, pick 0
            else:
                th6 = math.atan2(
                    (-T60[1, 0] * s1 + T60[1, 1] * c1) / s5,
                    (T60[0, 0] * s1 - T60[0, 1] * c1) / s5,
                )
            T01 = _dh(th1, _D[0], _A[0], _ALPHA[0])
            T45 = _dh(th5, _D[4], _A[4], _ALPHA[4])
            T56 = _dh(th6, _D[5], _A[5], _ALPHA[5])
            T14 = np.linalg.inv(T01) @ T06 @ np.linalg.inv(T45 @ T56)
            p13 = (T14 @ np.array([0.0, -D4, 0.0, 1.0]))[:3]
            n13 = float(np.linalg.norm(p13))
            c3 = (n13 * n13 - A2 * A2 - A3 * A3) / (2 * A2 * A3)
            if abs(c3) > 1.0 + 1e-9:
                continue
            c3 = max(-1.0, min(1.0, c3))
            for th3 in (math.acos(c3), -math.acos(c3)):
                th2 = -math.atan2(p13[1], -p13[0]) + math.asin(max(-1.0, min(1.0, A3 * math.sin(th3) / n13)))
                T12 = _dh(th2, _D[1], _A[1], _ALPHA[1])
                T23 = _dh(th3, _D[2], _A[2], _ALPHA[2])
                T34 = np.linalg.inv(T23) @ np.linalg.inv(T12) @ T14
                th4 = math.atan2(T34[1, 0], T34[0, 0])
                q = np.array([th1, th2, th3, th4, th5, th6])
                sols.append(np.arctan2(np.sin(q), np.cos(q)))
    return sols


def _closest_equivalent(q: np.ndarray, q_ref: np.ndarray, limit: float = 2 * math.pi) -> np.ndarray:
    """Shift each angle by multiples of 2*pi so it is closest to ``q_ref`` (within +-limit)."""
    out = q_ref + np.arctan2(np.sin(q - q_ref), np.cos(q - q_ref))
    return np.clip(out, -limit, limit)


def tool_rotation_down(yaw: float) -> np.ndarray:
    """World rotation of the DH tool frame for a suction cup pointing straight down.

    Tool Z = world -Z, tool X = (cos yaw, sin yaw, 0).
    """
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


class Ur10Kinematics:
    """UR10 kinematics in world coordinates, for a base at ``base_position`` rotated ``base_yaw``."""

    def __init__(
        self,
        base_position: tuple[float, float, float] = config.ROBOT_BASE,
        base_yaw: float = 0.0,
        tip_offset: float = config.SUCTION_TIP_OFFSET,
    ) -> None:
        self.base_position = np.asarray(base_position, dtype=np.float64)
        c, s = math.cos(base_yaw), math.sin(base_yaw)
        # World <- DH base rotation (asset base_link = DH base rotated 180 deg about Z).
        self.R_world_dh = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ _RZ_PI
        self.tip_offset = tip_offset

    def tip_pose(self, q: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
        """World (position, rotation) of the suction tip (rotation = DH tool frame)."""
        T = fk_dh(q)
        R = self.R_world_dh @ T[:3, :3]
        p_tool = self.base_position + self.R_world_dh @ T[:3, 3]
        return p_tool + self.tip_offset * R[:, 2], R

    def solve(
        self, tip_position: np.ndarray | list[float], tool_rotation: np.ndarray, q_ref: np.ndarray | list[float]
    ) -> np.ndarray | None:
        """Joint angles putting the suction tip at ``tip_position`` with ``tool_rotation``.

        Picks the IK branch closest to ``q_ref`` so motion stays continuous. None if unreachable.
        """
        p_tool = np.asarray(tip_position, dtype=np.float64) - self.tip_offset * tool_rotation[:, 2]
        T = np.eye(4)
        T[:3, :3] = self.R_world_dh.T @ tool_rotation
        T[:3, 3] = self.R_world_dh.T @ (p_tool - self.base_position)
        sols = ik_dh(T)
        if not sols:
            return None
        q_ref = np.asarray(q_ref, dtype=np.float64)
        cands = [_closest_equivalent(q, q_ref) for q in sols]
        return min(cands, key=lambda q: float(np.sum((q - q_ref) ** 2)))

    def solve_down(
        self, tip_position: np.ndarray | list[float], q_ref: np.ndarray | list[float], yaw: float = 0.0
    ) -> np.ndarray | None:
        """IK for a downward-pointing suction cup."""
        return self.solve(tip_position, tool_rotation_down(yaw), q_ref)


# A comfortable "elbow up, cup down" configuration above the belt, used as IK seed and home.
HOME_Q = np.array([math.pi / 2, -2.0, 1.9, -1.47, -math.pi / 2, 0.0])
