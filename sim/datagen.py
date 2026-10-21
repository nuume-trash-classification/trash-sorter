"""Synthetic training data with Isaac Sim Replicator -> YOLO-seg dataset (workstream 2).

    <isaac python> sim/datagen.py --frames 2000 --name v1
    -> data/synthetic/v1/{images,labels}/{train,val}/..., data.yaml, preview/ (label overlays)

Each frame: 1-4 objects lying on the belt under the sorter camera (slight overlap allowed:
an overlapping object rests on top of the other), then domain randomization of
  * object pose, count and asset variant,
  * light intensity, color temperature and direction,
  * belt color and roughness,
  * camera height / position / angle (small jitter around the calibrated pose),
  * object color / roughness.
Instance masks come from Replicator's ``instance_id_segmentation`` annotator, so labels are
exact and free. Uses the same scene, camera and assets as ``run_sorter.py``.
"""

from __future__ import annotations

import argparse
import json
import math

import _bootstrap

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--name", default="synthetic_v1", help="dataset folder name under data/synthetic/")
ap.add_argument("--val-fraction", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--assets", choices=["auto", "real", "placeholder"], default="auto")
ap.add_argument("--rt-subframes", type=int, default=8, help="render passes per frame (higher = cleaner, slower)")
ap.add_argument("--previews", type=int, default=20, help="save label overlays for the first N frames")
ap.add_argument("--no-randomize", action="store_true", help="disable lighting/material/camera randomization")
args = ap.parse_args()

from app import start_app

simulation_app = start_app(headless=True)


import carb.settings
import cv2
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom, UsdShade

from sim.assets.library import describe, find_assets
from sim.scene import CAMERA_PATH, build_scene
from trash_sorter import config
from trash_sorter.yolo_format import draw_overlay, mask_to_label_line, parse_label_file, write_data_yaml

rng = np.random.default_rng(args.seed)
out = _bootstrap.DATA_DIR / "synthetic" / args.name
for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
(out / "preview").mkdir(exist_ok=True)

assets = find_assets(args.assets)
print("[datagen] assets:\n" + describe(assets))
scene = build_scene(assets, per_class=4, with_robot=False)
stage = stage_utils.get_current_stage()
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)  # DLSS quality: fewer edge artifacts
rep.orchestrator.set_capture_on_play(False)

W, H = config.CAMERA_RESOLUTION
render_product = rep.create.render_product(CAMERA_PATH, (W, H))
rgb_annot = rep.annotators.get("rgb")
seg_annot = rep.annotators.get("instance_id_segmentation", init_params={"colorize": False})
rgb_annot.attach(render_product)
seg_annot.attach(render_product)

# ---------------------------------------------------------------------------
# Handles for randomization
# ---------------------------------------------------------------------------
objects = scene.objects
xform_ops = {}
for obj in objects:
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(obj.path)).GetOrderedXformOps()
    xform_ops[obj.path] = (ops[0], ops[1])  # translate, orient


def object_shaders(obj) -> list[tuple[UsdShade.Shader, Gf.Vec3f, float]]:
    """(shader, original color, original roughness) for every material inside an object."""
    found = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(obj.path)):
        if prim.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(prim)
            col, rough = sh.GetInput("diffuseColor"), sh.GetInput("roughness")
            if col and rough:
                found.append((sh, Gf.Vec3f(col.Get()), float(rough.Get())))
    return found


shaders = {obj.path: object_shaders(obj) for obj in objects}
cam_xf = UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA_PATH))
cam_ops = cam_xf.GetOrderedXformOps()
cam_pos0 = Gf.Vec3d(config.CAMERA_X, 0.0, config.BELT_TOP_Z + config.CAMERA_HEIGHT)
belt_colors = [(0.04, 0.04, 0.045), (0.10, 0.10, 0.11), (0.03, 0.09, 0.05), (0.04, 0.05, 0.13), (0.18, 0.18, 0.18)]


def set_pose(obj, pos, yaw: float) -> None:
    t_op, o_op = xform_ops[obj.path]
    t_op.Set(Gf.Vec3d(*map(float, pos)))
    q = Gf.Quatd(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    o_op.Set(Gf.Quatf(q) if o_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else q)


def footprint(obj, x: float, y: float, yaw: float) -> np.ndarray:
    """Object outline on the belt (rotated rectangle), in cm grid coordinates."""
    spec = config.CLASS_SPECS[obj.class_name]
    half_l, half_w = np.mean(spec.length_range) / 2, np.mean(spec.width_range) / 2
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]])
    pts = corners @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.round((pts + [2.0, 1.0]) * 100).astype(np.int32)


def place_objects() -> list:
    for obj in objects:
        t_op, _ = xform_ops[obj.path]
        t_op.Set(Gf.Vec3d(*obj.park_position))
    n = int(rng.integers(1, 5))
    chosen = []
    occupancy = np.zeros((200, 400), np.uint8)  # 1 cm grid of the belt area
    height_map = np.zeros((200, 400), np.float32)  # top height of what is already placed
    x_half = 0.5 * 2 * config.CAMERA_HEIGHT * config.CAMERA_H_APERTURE_MM / 2 / config.CAMERA_FOCAL_MM
    for _ in range(n):
        cls = config.CLASS_NAMES[int(rng.integers(len(config.CLASS_NAMES)))]
        free = [o for o in objects if o.class_name == cls and o not in chosen]
        if not free:
            continue
        obj = free[int(rng.integers(len(free)))]
        for _attempt in range(20):
            x = config.CAMERA_X + rng.uniform(-x_half - 0.05, x_half + 0.05)  # some cut by the image edge
            y = rng.uniform(-0.15, 0.15)
            yaw = rng.uniform(-math.pi, math.pi)
            poly = footprint(obj, x, y, yaw)
            mask = np.zeros_like(occupancy)
            cv2.fillPoly(mask, [poly], 1)
            overlap = (mask & occupancy).sum() / max(mask.sum(), 1)
            if overlap <= 0.15:  # slight overlap only
                break
        else:
            continue
        base = float(height_map[mask > 0].max()) if overlap > 0 else 0.0
        z = config.BELT_TOP_Z + base + obj.thickness / 2 + 0.001
        set_pose(obj, (x, y, z), yaw)
        occupancy |= mask
        height_map[mask > 0] = np.maximum(height_map[mask > 0], base + obj.thickness)
        chosen.append(obj)
    return chosen


def randomize(chosen) -> None:
    if args.no_randomize:
        return
    scene.dome_light.GetIntensityAttr().Set(float(rng.uniform(150, 900)))
    scene.sun_light.GetIntensityAttr().Set(float(rng.uniform(400, 2000)))
    scene.sun_light.GetColorTemperatureAttr().Set(float(rng.uniform(3200, 7500)))
    rot = UsdGeom.Xformable(scene.sun_light).GetOrderedXformOps()[0]
    rot.Set(Gf.Vec3f(float(rng.uniform(-45, 45)), float(rng.uniform(-45, 45)), float(rng.uniform(0, 360))))
    base = np.array(belt_colors[int(rng.integers(len(belt_colors)))])
    belt_rgb = np.clip(base * rng.uniform(0.7, 1.4), 0, 1)
    scene.belt_material.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, belt_rgb)))
    scene.belt_material.GetInput("roughness").Set(float(rng.uniform(0.4, 0.95)))
    # Camera: +-3 cm position, +-4 cm height, +-3 deg tilt around the calibrated pose.
    jitter = Gf.Vec3d(*rng.uniform(-0.03, 0.03, 2), rng.uniform(-0.04, 0.04))
    rx, ry, rz = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3)
    rot_q = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
    ).GetQuat()
    for op in cam_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(cam_pos0 + jitter)
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(Gf.Quatd(rot_q) if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf(rot_q))
    for obj in chosen:
        for sh, color, rough in shaders[obj.path]:
            scale = rng.uniform(0.75, 1.25, 3)
            new = np.clip(np.array(color) * scale, 0, 1)
            sh.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, new)))
            sh.GetInput("roughness").Set(float(np.clip(rough + rng.uniform(-0.1, 0.1), 0.02, 1.0)))


def id_to_path(info: dict) -> dict[int, str]:
    raw = info.get("idToLabels", {})
    if isinstance(raw, str):
        raw = json.loads(raw)
    return {int(k): (v if isinstance(v, str) else str(v)) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
n_val = max(1, int(round(args.frames * args.val_fraction)))
stats = {c: 0 for c in config.CLASS_NAMES}
for _ in range(3):  # warm-up: let materials load
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0)

for i in range(args.frames):
    chosen = place_objects()
    randomize(chosen)
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0)
    rgb = rgb_annot.get_data()
    seg = seg_annot.get_data()
    seg_ids, mapping = np.asarray(seg["data"]), id_to_path(seg["info"])
    if seg_ids.ndim == 3:
        seg_ids = seg_ids[..., 0]
    lines = []
    for obj in chosen:
        ids = [k for k, path in mapping.items() if path == obj.path or path.startswith(obj.path + "/")]
        mask = np.isin(seg_ids, ids)
        line = mask_to_label_line(config.CLASS_IDS[obj.class_name], mask, min_area_px=60)
        if line:
            lines.append(line)
            stats[obj.class_name] += 1
    split = "val" if i < n_val else "train"
    stem = f"{args.name}_{i:05d}"
    img_bgr = cv2.cvtColor(np.asarray(rgb)[..., :3], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    label_path = out / "labels" / split / f"{stem}.txt"
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if i < args.previews:
        overlay = draw_overlay(np.asarray(rgb)[..., :3], parse_label_file(label_path))
        cv2.imwrite(str(out / "preview" / f"{stem}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if (i + 1) % 50 == 0:
        print(f"[datagen] {i + 1}/{args.frames} frames")

rep.orchestrator.wait_until_complete()
yaml_path = write_data_yaml(out)
(out / "stats.json").write_text(json.dumps({"frames": args.frames, "instances": stats}, indent=2), encoding="utf-8")
print(f"[datagen] done: {args.frames} frames, instances per class {stats}")
print(f"[datagen] dataset: {yaml_path}")
simulation_app.close()
