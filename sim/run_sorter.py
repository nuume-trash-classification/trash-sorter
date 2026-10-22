"""Full sorting loop: belt -> camera -> perception -> tracker -> UR10 dynamic pick -> class bin.

    <isaac python> sim/run_sorter.py --perception gt   --headless --duration 300   # dev mode
    <isaac python> sim/run_sorter.py --perception yolo --weights runs/segment/.../best.pt

``--perception yolo`` (default) only uses camera images + belt speed, as on a real system.
``--perception gt`` reads object poses from the simulator: dev mode for arm work only.

Ground truth is read by this script only for *scoring* (which bin each object ended in) and,
with ``--debug``, for logging pick-point errors to ``diag.json``. The picking pipeline never
sees it in yolo mode.
"""

from __future__ import annotations

import argparse
import json
import time

import _bootstrap

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--perception", choices=["yolo", "gt"], default="yolo")
ap.add_argument("--weights", default=None, help="YOLO-seg weights (.pt); required for --perception yolo")
ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold per detection")
ap.add_argument("--min-track-conf", type=float, default=0.5, help="tracks below this mean confidence are not picked")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--duration", type=float, default=180.0, help="simulated seconds; 0 = until window closed")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--spawn-interval", type=float, default=4.0, help="mean seconds between objects")
ap.add_argument("--assets", choices=["auto", "real", "placeholder"], default="auto")
ap.add_argument("--snapshots", type=float, default=0.0, help="save an overview image every N sim seconds")
ap.add_argument("--save-camera", action="store_true", help="save every perception frame with detections drawn")
ap.add_argument(
    "--debug", action="store_true", help="print extra info and write pick-point errors vs ground truth to diag.json"
)
ap.add_argument("--out", default=None, help="output folder (default outputs/run_<time>)")
args = ap.parse_args()
if args.perception == "yolo" and not args.weights:
    ap.error("--perception yolo needs --weights (train one with perception/train.py), or use --perception gt")

from app import start_app

simulation_app = start_app(args.headless)

# ---- imports that need the running app ----
from pathlib import Path

import cv2
import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.surface_gripper import _surface_gripper
from isaacsim.sensors.experimental.rtx import CameraSensor

from sim.assets.library import describe, find_assets
from sim.gt_perception import GroundTruthPerception
from sim.scene import GRIPPER_PATH, OVERVIEW_CAMERA_PATH, ROBOT_PATH, add_overview_camera, build_scene
from trash_sorter import config
from trash_sorter.geometry import default_camera, quat_wxyz_to_matrix
from trash_sorter.metrics import END_BIN, LOST, SortingStats
from trash_sorter.picking import GripperCommand, Phase, PickController, PickParams
from trash_sorter.tracking import BeltTracker
from trash_sorter.ur10 import HOME_Q, Ur10Kinematics

out_dir = Path(args.out) if args.out else _bootstrap.OUTPUT_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# =====================================================================
# Scene
# =====================================================================
assets = find_assets(args.assets)
print("[sorter] assets:\n" + describe(assets))
scene = build_scene(assets, per_class=6)
sensor = CameraSensor(
    scene.camera, resolution=(config.CAMERA_RESOLUTION[1], config.CAMERA_RESOLUTION[0]), annotators=["rgb"]
)
overview_cam = add_overview_camera()
overview = CameraSensor(overview_cam, resolution=(720, 1280), annotators=["rgb"]) if args.snapshots > 0 else None
if not args.headless:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().camera_path = OVERVIEW_CAMERA_PATH
    except Exception as exc:  # a viewport problem must not stop the simulation
        print(f"[sorter] could not set viewport camera: {exc}")

objects = scene.objects
view = RigidPrim(paths=[o.path for o in objects])
robot = Articulation(ROBOT_PATH)
kin = Ur10Kinematics()
camera_model = default_camera()

# =====================================================================
# Perception: the single switch between dev mode and final mode
# =====================================================================
if args.perception == "gt":
    perception = GroundTruthPerception(objects, view, camera_model)
    print("[sorter] WARNING: --perception gt reads simulator poses (dev mode, not a valid result)")
else:
    from trash_sorter.perception import YoloPerception

    perception = YoloPerception(args.weights, camera_model, conf=args.conf)

tracker = BeltTracker()
controller = PickController(PickParams(min_confidence=args.min_track_conf))
stats = SortingStats()

# =====================================================================
# Start physics
# =====================================================================
app_utils.play(commit=True)
simulation_app.update()
if SimulationManager.get_active_physics_engine() != "physx":
    SimulationManager.switch_physics_engine("physx")
robot.set_link_enabled_gravities([False])
q_cmd = kin.solve_down(controller.cmd, HOME_Q)
assert q_cmd is not None, "idle pose unreachable"
robot.set_dof_positions(q_cmd.tolist())
robot.set_dof_position_targets(q_cmd.tolist())
gripper = _surface_gripper.acquire_surface_gripper_interface()
gripper.open_gripper(GRIPPER_PATH)
gripper_state = GripperCommand.OPEN
for _ in range(5):
    simulation_app.update()


def sim_time() -> float:
    return SimulationManager.get_simulation_time()


def park(obj) -> None:
    view.set_world_poses(positions=[list(obj.park_position)], orientations=[[1, 0, 0, 0]], indices=[obj.index])
    view.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)), indices=[obj.index])
    obj.state = "parked"


def spawn(t: float) -> None:
    cls = config.CLASS_NAMES[rng.integers(len(config.CLASS_NAMES))]
    free = [o for o in objects if o.class_name == cls and o.state == "parked"]
    if not free:
        return
    obj = free[rng.integers(len(free))]
    y = rng.uniform(*config.SPAWN_Y_RANGE)
    yaw = rng.uniform(-np.pi, np.pi)
    pos = [config.SPAWN_X, y, config.BELT_TOP_Z + obj.thickness / 2 + 0.003]
    view.set_world_poses(positions=[pos], orientations=[[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]], indices=[obj.index])
    view.set_velocities(np.array([[config.BELT_SPEED, 0, 0]]), np.zeros((1, 3)), indices=[obj.index])
    obj.state, obj.spawn_t = "on_belt", t
    spawned_log.append({"t": round(t, 2), "path": obj.path, "class": cls, "asset": obj.asset.name})


def in_region(x: float, y: float, region) -> bool:
    (xa, xb), (ya, yb) = region
    return xa < x < xb and ya < y < yb


def score_objects(t: float) -> None:
    """Evaluation only: find where finished objects ended up (uses ground truth)."""
    pos = view.get_world_poses()[0].numpy()
    for obj in objects:
        if obj.state != "on_belt" or obj.path == holding_path:
            continue
        x, y, z = pos[obj.index]
        where = None
        if z < config.BIN_TOP_Z - 0.05:  # inside a bin, or on its way to the floor
            region = next((name for name, reg in scene.regions.items() if in_region(x, y, reg)), None)
            if region is not None:
                where = END_BIN if region == "end_bin" else region
            elif z < 0.15:
                where = LOST
        elif t - obj.spawn_t > 90:  # stuck somewhere
            where = LOST
        if where is None:
            continue
        stats.record_outcome(obj.class_name, where)
        target_bin = aimed_bins.pop(obj.path, None)
        if target_bin is not None:
            stats.record_attempt(where == target_bin)
        outcome_log.append(
            {
                "t": round(t, 2),
                "path": obj.path,
                "class": obj.class_name,
                "location": where,
                "xyz": [round(float(v), 3) for v in (x, y, z)],
            }
        )
        park(obj)


diag: dict[str, list] = {"detections": [], "grips": []}  # --debug only, written to diag.json


def _true_frame(obj, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Debug only: true center and long axis (asset local +X = toward the cap) of an object."""
    axis = quat_wxyz_to_matrix(quat) @ np.array([1.0, 0.0, 0.0])
    axis2 = axis[:2] / max(float(np.linalg.norm(axis[:2])), 1e-9)
    return pos[obj.index], axis2


def _error_in_object_frame(point_xy: np.ndarray, center_xy: np.ndarray, axis2: np.ndarray) -> tuple[float, float]:
    """(along, across) offset of a point from the object center; along > 0 = toward the cap."""
    e = point_xy - center_xy
    return float(e @ axis2), float(e[0] * -axis2[1] + e[1] * axis2[0])


def _nearest_on_belt(xy: np.ndarray, pos: np.ndarray):
    return min(
        (o for o in objects if o.state == "on_belt"),
        default=None,
        key=lambda o: float(np.linalg.norm(pos[o.index][:2] - xy)),
    )


def debug_detections(dets) -> None:
    """Debug only: compare each YOLO pick point with the true pose of the nearest object."""
    pos, quat = (a.numpy() for a in view.get_world_poses())
    for d in dets:
        near = _nearest_on_belt(d.position[:2], pos)
        if near is None:
            continue
        center, axis2 = _true_frame(near, pos, quat[near.index])
        along, across = _error_in_object_frame(d.position[:2], center[:2], axis2)
        diag["detections"].append(
            {
                "t": round(d.t_observed, 2),
                "true_class": near.class_name,
                "pred_class": d.class_name,
                "det_xy": [round(float(v), 4) for v in d.position[:2]],
                "true_xyz": [round(float(v), 4) for v in center],
                "axis": [round(float(v), 3) for v in axis2],
                "along": round(along, 4),
                "across": round(across, 4),
                "pixel": list(d.pixel) if d.pixel else None,
            }
        )


def debug_grip_start(t: float) -> None:
    """Debug only: where the suction tip goes down vs where the object really is."""
    pos, quat = (a.numpy() for a in view.get_world_poses())
    tr = controller.target
    tip = controller.cmd
    near = _nearest_on_belt(tip[:2], pos)
    if near is None or tr is None:
        return
    center, axis2 = _true_frame(near, pos, quat[near.index])
    along, across = _error_in_object_frame(tip[:2], center[:2], axis2)
    vel = view.get_velocities()[0].numpy()[near.index]
    diag["grips"].append(
        {
            "t": round(t, 2),
            "track_id": tr.track_id,
            "track_class": tr.class_name,
            "true_class": near.class_name,
            "asset": near.asset.name,
            "tip": [round(float(v), 4) for v in tip],
            "true_xyz": [round(float(v), 4) for v in center],
            "axis": [round(float(v), 3) for v in axis2],
            "along": round(along, 4),
            "across": round(across, 4),
            "true_vel": [round(float(v), 4) for v in vel[:2]],
            "track_vel": None if tr.velocity is None else [round(float(v), 4) for v in tr.velocity],
            "hits": tr.hits,
            "obs_span": round(tr.observations[-1][0] - tr.observations[0][0], 2),
        }
    )


def save_camera_frame(rgb: np.ndarray, dets, t: float) -> None:
    img = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
    for d in dets:
        if d.mask is not None:
            img[d.mask] = (0.5 * img[d.mask] + [0, 90, 0]).astype(np.uint8)
        if d.pixel is not None:
            cv2.circle(img, d.pixel, 5, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"{d.class_name} {d.confidence:.2f}",
                (d.pixel[0] + 6, d.pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
    (out_dir / "camera").mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "camera" / f"t{t:07.2f}.jpg"), img)


# =====================================================================
# Main loop
# =====================================================================
spawned_log: list[dict] = []
outcome_log: list[dict] = []
aimed_bins: dict[str, str] = {}  # object path -> bin the arm is carrying it to
holding_path: str | None = None
next_spawn = 1.0
next_frame = 0.0
next_snapshot = 0.0
t_prev = sim_time()
n_events = 0
ik_failures = 0
wall0 = time.time()
print(f"[sorter] running, perception={args.perception}, output={out_dir}")

while simulation_app.is_running():
    simulation_app.update()
    if not app_utils.is_playing():
        continue
    t = sim_time()
    dt = t - t_prev
    if dt <= 0:
        continue
    t_prev = t
    if args.duration and t >= args.duration:
        break

    if args.debug and int(t / 2) != int((t - dt) / 2):
        pos_dbg, vel_dbg = view.get_world_poses()[0].numpy(), view.get_velocities()[0].numpy()
        moving = [
            (
                o.class_name[:5],
                round(float(pos_dbg[o.index][0]), 2),
                round(float(vel_dbg[o.index][0]), 3),
                round(float(vel_dbg[o.index][1]), 3),
            )
            for o in objects
            if o.state == "on_belt"
        ]
        print(f"[debug] t={t:.1f} (class, x, vx, vy): {moving}")

    # 1) feed the belt
    if t >= next_spawn:
        spawn(t)
        next_spawn = t + rng.uniform(0.6, 1.4) * args.spawn_interval

    # 2) perception at a fixed frame rate
    if t >= next_frame:
        data, _ = sensor.get_data("rgb")
        if data is not None:
            next_frame = t + config.CAMERA_PERIOD
            rgb = data.numpy()
            # Time stamp = when the image was taken, not when we read it.
            t_image = t - (config.CAMERA_LATENCY if args.perception == "yolo" else 0.0)
            dets = perception.detect(rgb, t_image)
            tracker.update(dets)
            if args.debug and args.perception == "yolo":
                debug_detections(dets)
            if args.save_camera and dets:
                save_camera_frame(rgb, dets, t)
    tracker.prune(t)

    # 3) suction status
    status = gripper.get_gripper_status(GRIPPER_PATH)
    gripped = list(gripper.get_gripped_objects(GRIPPER_PATH) or [])
    holding = str(status).endswith("Closed") and bool(gripped)

    # 4) decide + move the arm (IK on the commanded suction-tip position)
    phase_before = controller.phase
    cmd = controller.step(t, dt, tracker, holding)
    if args.debug and controller.phase == Phase.GRIP and phase_before != Phase.GRIP:
        debug_grip_start(t)
    q_new = kin.solve_down(cmd.tip_target, q_cmd)
    if q_new is None:
        ik_failures += 1
    else:
        robot.set_dof_position_targets(q_new.tolist())
        robot.set_dof_velocity_targets(((q_new - q_cmd) / dt).tolist())
        q_cmd = q_new
    if cmd.gripper != gripper_state:
        (gripper.close_gripper if cmd.gripper == GripperCommand.CLOSE else gripper.open_gripper)(GRIPPER_PATH)
        gripper_state = cmd.gripper

    # 5) bookkeeping for scoring
    for ev in controller.events[n_events:]:
        if ev.kind == "grasp" and gripped:
            holding_path = gripped[0]
            aimed_bins[holding_path] = ev.bin_name or ""
        elif ev.kind == "grasp_failed":
            stats.record_attempt(False)
            if args.debug:
                tip_now, _ = kin.tip_pose(robot.get_dof_positions().numpy()[0])
                pos_all = view.get_world_poses()[0].numpy()
                near = min(
                    (o for o in objects if o.state == "on_belt"),
                    default=None,
                    key=lambda o: np.linalg.norm(pos_all[o.index][:2] - cmd.tip_target[:2]),
                )
                if near is not None:
                    print(
                        f"[debug] cmd={np.round(cmd.tip_target, 3)} tip={np.round(tip_now, 3)} "
                        f"nearest {near.class_name} at {np.round(pos_all[near.index], 3)}"
                    )
        elif ev.kind == "release":
            holding_path = None
        print(f"[sorter] t={t:6.1f}s {ev.kind:<12} track={ev.track_id:<3} class={ev.class_name}")
        if args.debug and ev.kind == "claim":
            tr = next((x for x in tracker.tracks if x.track_id == ev.track_id), None)
            if tr is not None:
                print(
                    f"[debug]   hits={tr.hits} conf={tr.confidence:.2f} v={tr.velocity} "
                    f"obs_span={tr.observations[-1][0] - tr.observations[0][0]:.1f}s"
                )
    n_events = len(controller.events)
    if gripper_state == GripperCommand.OPEN:
        holding_path = None
    score_objects(t)

    # 6) optional overview snapshots
    if overview is not None and t >= next_snapshot:
        data, _ = overview.get_data("rgb")
        if data is not None:
            next_snapshot = t + args.snapshots
            (out_dir / "overview").mkdir(exist_ok=True)
            cv2.imwrite(
                str(out_dir / "overview" / f"t{t:07.2f}.jpg"), cv2.cvtColor(data.numpy()[..., :3], cv2.COLOR_RGB2BGR)
            )

# =====================================================================
# Report
# =====================================================================
summary = stats.summary()
summary.update(
    perception=args.perception,
    weights=args.weights,
    sim_seconds=round(sim_time(), 1),
    wall_seconds=round(time.time() - wall0, 1),
    belt_speed=config.BELT_SPEED,
    objects_spawned=len(spawned_log),
    ik_failures=ik_failures,
    still_on_belt=sum(o.state == "on_belt" for o in objects),
)
text = stats.format() + f"\nperception            : {args.perception}\nspawned               : {len(spawned_log)}"
print(text)
(out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out_dir / "summary.txt").write_text(text, encoding="utf-8")
(out_dir / "events.json").write_text(
    json.dumps(
        {
            "spawned": spawned_log,
            "outcomes": outcome_log,
            "controller": [ev.__dict__ for ev in controller.events],
        },
        indent=1,
    ),
    encoding="utf-8",
)
if args.debug:
    (out_dir / "diag.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
print(f"[sorter] results in {out_dir}")
simulation_app.close()
