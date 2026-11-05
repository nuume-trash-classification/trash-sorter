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
    # Fast enough to keep up with an object every ~2.5 s. 1.8 m/s at the tip stays under the UR10's
    # ~120 deg/s base joint limit, but near the base (the can bin) the real arm still lags the command,
    # so the release waits for the real tip (``release_tolerance``).
    max_speed: float = 1.8  # m/s, suction tip Cartesian speed
    max_accel: float = 8.0  # m/s^2, still smooth: ~0.25 g sideways on a held bottle
    lift_speed_factor: float = 0.8  # lift a bit slower than max_speed so the object is not yanked
    descend_speed: float = 0.6
    hover_height: float = 0.05  # above the object's top surface
    grip_clearance: float = 0.004  # tip height above the top surface when sucking
    grip_timeout: float = 0.6
    lift_z: float = 1.0
    release_settle: float = 0.15  # let the arm stop above the bin first: a moving release throws the object
    release_time: float = 0.3
    release_tolerance: float = 0.02  # m, the real tip must be this close to the drop point to open
    release_max_tip_speed: float = 0.05  # m/s, ... and nearly still: the object leaves with the tip's velocity
    release_max_wait: float = 1.5  # s, open anyway (and count it) if the real tip never gets there
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
    kind: str  # "claim", "grasp", "grasp_failed", "release", "release_unsettled", "abandon"
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
        self.opened_t: float | None = None  # when the suction opened in RELEASE
        self.last_tip: np.ndarray | None = None  # previous ``tip_actual``, for the real tip speed

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
    def step(
        self, t: float, dt: float, tracker: BeltTracker, holding: bool, tip_actual: np.ndarray | None = None
    ) -> ArmCommand:
        """One control tick. ``tip_actual``: the real suction tip (forward kinematics of the measured
        joints); the release waits for it. ``None`` assumes the arm follows the command exactly."""
        p = self.params
        tr = self.target
        tip_speed = 0.0
        if tip_actual is not None:
            if self.last_tip is not None and dt > 0:
                tip_speed = float(np.linalg.norm(tip_actual - self.last_tip)) / dt
            self.last_tip = np.array(tip_actual, dtype=np.float64)

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
            self._move_toward(self.lift_goal, dt, p.max_speed * p.lift_speed_factor)
            if self.cmd[2] >= p.lift_z - 0.03:
                self._set_phase(Phase.TRANSPORT if self.target is not None else Phase.IDLE, t)

        elif self.phase == Phase.TRANSPORT:
            if self._move_toward(self._bin_drop_point(), dt, p.max_speed):
                self._set_phase(Phase.RELEASE, t)

        elif self.phase == Phase.RELEASE:
            assert self.target is not None
            tgt = self.target
            drop = self._bin_drop_point()
            self._move_toward(drop, dt, p.max_speed)
            waited = t - self.phase_t0
            if self.opened_t is None:
                # Open only once the real arm is there and nearly still: the object leaves with the tip's
                # velocity, and a lagging arm passes the drop point at full speed.
                near = tip_actual is None or float(np.linalg.norm(tip_actual - drop)) <= p.release_tolerance
                at_drop = near and tip_speed <= p.release_max_tip_speed
                if (waited >= p.release_settle and at_drop) or waited >= p.release_max_wait:
                    if not at_drop:
                        late = PickEvent(t, "release_unsettled", tgt.track_id, tgt.class_name, self.bin_name)
                        self.events.append(late)
                    self.gripper = GripperCommand.OPEN
                    self.opened_t = t
            elif t - self.opened_t >= p.release_time - p.release_settle:
                self.events.append(PickEvent(t, "release", tgt.track_id, tgt.class_name, self.bin_name))
                self.target = None
                self.opened_t = None
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
