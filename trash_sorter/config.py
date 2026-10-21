"""Shared constants for the whole project (units: meters, seconds, kilograms).

Both the Isaac Sim scripts (``sim/``) and the pure-Python code (``trash_sorter/``,
``perception/``) read from here, so the scene layout and the perception math can
never drift apart. World frame: Z up, the belt moves along +X.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASS_NAMES: tuple[str, ...] = ("pet_bottle", "aluminum_can", "paper_carton", "glass_bottle")
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class ClassSpec:
    """Physical facts about one class, used by perception and by the placeholder assets.

    ``thickness`` is the height of the top surface above the belt when the object lies
    flat. Perception uses it to turn a pixel into a 3D pick point (RGB only, no depth),
    so real objects of a class must lie at roughly this height.
    """

    name: str
    thickness: float
    length_range: tuple[float, float]  # longest side, for placeholders / asset scaling
    width_range: tuple[float, float]  # second side (diameter for cylinders)
    mass: float


CLASS_SPECS: dict[str, ClassSpec] = {
    "pet_bottle": ClassSpec("pet_bottle", 0.070, (0.18, 0.24), (0.060, 0.075), 0.030),
    "aluminum_can": ClassSpec("aluminum_can", 0.066, (0.115, 0.125), (0.064, 0.068), 0.015),
    "paper_carton": ClassSpec("paper_carton", 0.065, (0.12, 0.20), (0.065, 0.095), 0.050),
    "glass_bottle": ClassSpec("glass_bottle", 0.070, (0.20, 0.26), (0.065, 0.075), 0.300),
}

# ---------------------------------------------------------------------------
# Conveyor
# ---------------------------------------------------------------------------
BELT_SPEED = 0.15  # m/s, a real system reads this from a belt encoder
BELT_LENGTH = 3.2
BELT_WIDTH = 0.40
BELT_TOP_Z = 0.75
BELT_X_START = -BELT_LENGTH / 2
BELT_X_END = BELT_LENGTH / 2
SPAWN_X = BELT_X_START + 0.15
SPAWN_Y_RANGE = (-0.12, 0.12)

# ---------------------------------------------------------------------------
# Top-down RGB camera, upstream of the arm
# ---------------------------------------------------------------------------
CAMERA_X = -0.50
CAMERA_HEIGHT = 0.70  # above the belt surface
CAMERA_RESOLUTION = (640, 480)  # (width, height)
CAMERA_FOCAL_MM = 18.0
CAMERA_H_APERTURE_MM = 20.955
CAMERA_V_APERTURE_MM = CAMERA_H_APERTURE_MM * CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION[0]
CAMERA_PERIOD = 0.10  # seconds between perception frames
# The image shows the belt a little earlier than the moment we read it (render pipeline delay;
# a real camera has exposure + transfer delay too). Measured in sim: ~0.04 s. Calibrate on hardware.
CAMERA_LATENCY = 0.04

# ---------------------------------------------------------------------------
# UR10 + suction
# ---------------------------------------------------------------------------
ROBOT_BASE = (0.30, -0.50, 0.70)  # base_link origin, on a pedestal beside the belt
SUCTION_TIP_OFFSET = 0.1585  # suction tip distance from ee_link along its +X axis

# Pick window along the belt: objects are only picked while X is inside it.
PICK_ZONE_X = (-0.05, 0.75)

# ---------------------------------------------------------------------------
# Bins: 4 class bins (2 per belt side) + 1 end-of-belt bin for anything not picked
# ---------------------------------------------------------------------------
BIN_SIZE = (0.32, 0.30)  # x, y
BIN_TOP_Z = 0.45
BIN_CENTERS: dict[str, tuple[float, float]] = {
    "pet_bottle": (ROBOT_BASE[0] - 0.55, -0.45),
    "aluminum_can": (ROBOT_BASE[0] + 0.55, -0.45),
    "paper_carton": (ROBOT_BASE[0] - 0.30, 0.45),
    "glass_bottle": (ROBOT_BASE[0] + 0.30, 0.45),
}
END_BIN_CENTER = (BELT_X_END + 0.25, 0.0)
END_BIN_SIZE = (0.40, 0.60)
DROP_HEIGHT = 0.95  # suction tip height when releasing over a bin
