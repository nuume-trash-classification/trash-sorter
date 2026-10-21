"""Dynamic picking state machine: chooses an object, tracks it at belt speed, picks, drops in its bin.

Pure Python. Input each step: time, tracks (from perception), whether the suction cup
holds something. Output: where the suction tip should be and what the gripper should do.
The simulator (or a real robot driver) turns the tip target into joint angles with IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from trash_sorter import config
from trash_sorter.tracking import BeltTracker, Track


class Phase(Enum):
    IDLE = "idle"
    APPROACH = "approach"  # fly to a hover point above the (moving) object
    DESCEND = "descend"  # go down while matching belt speed
    GRIP = "grip"  # suction on, wait for the seal
    LIFT = "lift"
    TRANSPORT = "transport"  # carry to the class bin
    RELEASE = "release"


class GripperCommand(Enum):
    OPEN = "open"
    CLOSE = "close"


@dataclass
class ArmCommand:
    tip_target: np.ndarray
    gripper: GripperCommand


@dataclass
class PickParams:
    max_speed: float = 1.0  # m/s, suction tip Cartesian speed
    max_accel: float = 4.0  # m/s^2, smooth starts/stops so nothing gets thrown
    descend_speed: float = 0.35
    hover_height: float = 0.08  # above the object's top surface
    grip_clearance: float = 0.004  # tip height above the top surface when sucking
    grip_timeout: float = 0.6
    lift_z: float = 1.0
    release_settle: float = 0.15  # wait before opening the suction above the bin
    release_time: float = 0.45
    min_confidence: float = 0.5
    min_hits: int = 3
    duplicate_radius: float = 0.08  # other tracks this close to a picked one are the same object
    zone_x: tuple[float, float] = config.PICK_ZONE_X
    idle_position: tuple[float, float, float] = (0.15, 0.0, 0.95)
    position_tolerance: float = 0.01
    belt_speed: float = config.BELT_SPEED


@dataclass
class PickEvent:
    """Something the controller decided or observed; used for metrics."""

    t: float
    kind: str  # "claim", "grasp", "grasp_failed", "release", "abandon"
    track_id: int
    class_name: str
    bin_name: str | None = None


@dataclass
class PickController:
    params: PickParams = field(default_factory=PickParams)

    def __post_init__(self) -> None:
        self.phase = Phase.IDLE
        self.cmd = np.array(self.params.idle_position, dtype=np.float64)
        self.vel = np.zeros(3)  # velocity of the commanded tip
        self.lift_goal = self.cmd.copy()
        self.target: Track | None = None
        self.phase_t0 = 0.0
        self.gripper = GripperCommand.OPEN
        self.bin_name: str | None = None
        self.events: list[PickEvent] = []

    # ------------------------------------------------------------------ helpers
    def _set_phase(self, phase: Phase, t: float) -> None:
        self.phase = phase
        self.phase_t0 = t

    def _move_toward(self, goal: np.ndarray, dt: float, speed: float, goal_vel: np.ndarray | None = None) -> bool:
        """Move the commanded tip toward ``goal`` (which moves at ``goal_vel``) with limited speed and
        acceleration. Returns True once it has arrived and matches the goal velocity."""
        p = self.params
        gv = np.zeros(3) if goal_vel is None else goal_vel
        delta = goal - self.cmd
        dist = float(np.linalg.norm(delta))
        # Braking profile: never faster than what still allows stopping at the goal.
        v_mag = min(speed, float(np.sqrt(2.0 * p.max_accel * max(dist - 0.5 * p.position_tolerance, 0.0))))
        v_des = gv + (delta / dist * v_mag if dist > 1e-9 else 0.0)
        dv = v_des - self.vel
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm > p.max_accel * dt:
            dv *= p.max_accel * dt / dv_norm
        self.vel = self.vel + dv
        self.cmd = self.cmd + self.vel * dt
        settled = float(np.linalg.norm(self.vel - gv)) < 0.05
        if dist < 1e-3 and settled:
            self.cmd = goal.copy()
            self.vel = gv.copy()
        return float(np.linalg.norm(goal - self.cmd)) <= p.position_tolerance and settled

    def _belt_velocity(self) -> np.ndarray:
        return np.array([self.params.belt_speed, 0.0, 0.0])

    def _hover_goal(self, track: Track, t: float) -> np.ndarray:
        p = track.position_at(t, self.params.belt_speed)
        return np.array([max(p[0], self.params.zone_x[0]), p[1], p[2] + self.params.hover_height])

    def _latest_grip_x(self) -> float:
        """Objects further downstream than this cannot be picked in time any more."""
        p = self.params
        return p.zone_x[1] - p.belt_speed * (p.hover_height / p.descend_speed + p.grip_timeout)

    def _choose_target(self, tracker: BeltTracker, t: float) -> Track | None:
        p = self.params
        best, best_x = None, -np.inf
        for tr in tracker.active():
            if tr.claimed or tr.hits < p.min_hits or tr.confidence < p.min_confidence:
                continue
            pos = tr.position_at(t, p.belt_speed)
            travel = float(np.linalg.norm(self._hover_goal(tr, t) - self.cmd)) / p.max_speed
            x_at_grip = pos[0] + p.belt_speed * (travel + p.hover_height / p.descend_speed)
            if x_at_grip > self._latest_grip_x():
                continue
            if pos[0] > best_x:  # most downstream first: it is the one we would lose first
                best, best_x = tr, pos[0]
        return best

    # ------------------------------------------------------------------ main step
    def step(self, t: float, dt: float, tracker: BeltTracker, holding: bool) -> ArmCommand:
        p = self.params
        tr = self.target

        if self.phase == Phase.IDLE:
            self.gripper = GripperCommand.OPEN
            tr = self._choose_target(tracker, t)
            if tr is not None:
                tr.claimed = True
                self.target = tr
                self.bin_name = tr.class_name
                self.events.append(PickEvent(t, "claim", tr.track_id, tr.class_name, self.bin_name))
                self._set_phase(Phase.APPROACH, t)
            else:
                self._move_toward(np.array(p.idle_position), dt, p.max_speed)

        if self.phase == Phase.APPROACH and tr is not None:
            if tr.position_at(t, p.belt_speed)[0] > self._latest_grip_x():
                self._abandon(t)
            else:
                goal = self._hover_goal(tr, t)
                ff = self._belt_velocity() if goal[0] > p.zone_x[0] else None
                arrived = self._move_toward(goal, dt, p.max_speed, ff)
                if arrived and tr.position_at(t, p.belt_speed)[0] >= p.zone_x[0]:
                    self._set_phase(Phase.DESCEND, t)

        elif self.phase == Phase.DESCEND and tr is not None:
            obj = tr.position_at(t, p.belt_speed)
            goal = np.array([obj[0], obj[1], obj[2] + p.grip_clearance])
            # Follow the object in XY exactly, descend in Z at a limited speed.
            self.cmd[:2] = goal[:2]
            self.cmd[2] = max(goal[2], self.cmd[2] - p.descend_speed * dt)
            self.vel = self._belt_velocity()
            if self.cmd[2] <= goal[2] + 1e-6:
                self.gripper = GripperCommand.CLOSE
                self._set_phase(Phase.GRIP, t)

        elif self.phase == Phase.GRIP and tr is not None:
            obj = tr.position_at(t, p.belt_speed)
            self.cmd = np.array([obj[0], obj[1], obj[2] + p.grip_clearance])
            self.vel = self._belt_velocity()
            self.gripper = GripperCommand.CLOSE
            if holding:
                self.events.append(PickEvent(t, "grasp", tr.track_id, tr.class_name, self.bin_name))
                tr.done = True
                for other in tracker.active():  # duplicate tracks of the object we now hold
                    if float(np.linalg.norm(other.position_at(t, p.belt_speed)[:2] - obj[:2])) < p.duplicate_radius:
                        other.done = True
                self._start_lift(t)
            elif t - self.phase_t0 > p.grip_timeout:
                self.events.append(PickEvent(t, "grasp_failed", tr.track_id, tr.class_name, self.bin_name))
                tr.done = True
                self.gripper = GripperCommand.OPEN
                self.target = None
                self._start_lift(t)

        elif self.phase == Phase.LIFT:
            # Straight up (keeping a little belt motion so the object is not dragged); start the
            # transport move as soon as it is high enough.
            self._move_toward(self.lift_goal, dt, p.max_speed * 0.6)
            if self.cmd[2] >= p.lift_z - 0.03:
                self._set_phase(Phase.TRANSPORT if self.target is not None else Phase.IDLE, t)

        elif self.phase == Phase.TRANSPORT:
            if self._move_toward(self._bin_drop_point(), dt, p.max_speed):
                self._set_phase(Phase.RELEASE, t)

        elif self.phase == Phase.RELEASE:
            self._move_toward(self._bin_drop_point(), dt, p.max_speed)
            if t - self.phase_t0 >= p.release_settle:  # let the arm come to rest, then drop
                self.gripper = GripperCommand.OPEN
            if t - self.phase_t0 >= p.release_time:
                assert self.target is not None
                self.events.append(PickEvent(t, "release", self.target.track_id, self.target.class_name, self.bin_name))
                self.target = None
                self._set_phase(Phase.IDLE, t)

        return ArmCommand(self.cmd.copy(), self.gripper)

    def _start_lift(self, t: float) -> None:
        p = self.params
        self.lift_goal = np.array([self.cmd[0] + p.belt_speed * 0.2, self.cmd[1], p.lift_z])
        self._set_phase(Phase.LIFT, t)

    def _bin_drop_point(self) -> np.ndarray:
        bx, by = config.BIN_CENTERS[self.bin_name or config.CLASS_NAMES[0]]
        return np.array([bx, by, config.DROP_HEIGHT])

    def _abandon(self, t: float) -> None:
        assert self.target is not None
        self.events.append(PickEvent(t, "abandon", self.target.track_id, self.target.class_name, self.bin_name))
        self.target.done = True
        self.target = None
        self._set_phase(Phase.IDLE, t)
