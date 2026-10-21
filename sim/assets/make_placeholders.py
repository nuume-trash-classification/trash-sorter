"""Generate procedural placeholder assets (cylinders / boxes) for every class.

These stand in for the real 3D models until workstream 1 delivers them (see
``convert_assets.py``). Sizes come from ``config.CLASS_SPECS``.

Run with the Isaac Sim Python (no simulator start needed, only ``pxr``):

    <isaac python> sim/assets/make_placeholders.py [--variants 6] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics

from sim.assets.usd_utils import bind, finish_asset, new_asset_stage, preview_material
from trash_sorter import config

PLACEHOLDER_DIR = Path(__file__).resolve().parent / "usd" / "placeholder"

PET_TINTS = [(0.75, 0.85, 0.95), (0.70, 0.90, 0.80), (0.85, 0.85, 0.88), (0.55, 0.75, 0.95)]
LABEL_COLORS = [(0.85, 0.10, 0.10), (0.10, 0.35, 0.80), (0.95, 0.80, 0.10), (0.15, 0.60, 0.25), (0.95, 0.95, 0.95)]
CAN_COLORS = [(0.75, 0.75, 0.78), (0.70, 0.05, 0.05), (0.05, 0.15, 0.60), (0.10, 0.45, 0.20), (0.80, 0.60, 0.15)]
CARTON_COLORS = [(0.92, 0.92, 0.90), (0.55, 0.38, 0.22), (0.90, 0.75, 0.30), (0.30, 0.55, 0.85), (0.85, 0.30, 0.30)]
GLASS_TINTS = [(0.15, 0.45, 0.15), (0.40, 0.22, 0.08), (0.80, 0.85, 0.85)]


def _cylinder_x(stage, path, radius, length, x_center, material, collide=True):
    """Cylinder lying along X."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr("X")
    cyl.CreateExtentAttr([(-length / 2, -radius, -radius), (length / 2, radius, radius)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(x_center, 0, 0))
    bind(cyl.GetPrim(), material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def _bottle(stage, length, radius, body_mat, neck_mat, label_mat, long_neck):
    """Body + shoulder + neck + cap along X, centered on the bounding box."""
    parts = [  # (name, radius, length)
        ("Body", radius, length * (0.62 if long_neck else 0.72)),
        ("Shoulder", radius * 0.70, length * 0.10),
        ("Neck", radius * 0.38, length * (0.22 if long_neck else 0.12)),
        ("Cap", radius * 0.42, length * 0.06),
    ]
    total = sum(p[2] for p in parts)
    x = -total / 2
    for name, r, seg in parts:
        mat = body_mat if name in ("Body", "Shoulder") else neck_mat
        _cylinder_x(stage, f"/Object/{name}", r, seg, x + seg / 2, mat)
        x += seg
    if label_mat is not None:
        body_len = parts[0][2]
        _cylinder_x(
            stage, "/Object/Label", radius * 1.01, body_len * 0.45, -total / 2 + body_len * 0.5, label_mat, False
        )


def make_pet(stage, rng, i):
    spec = config.CLASS_SPECS["pet_bottle"]
    length, width = rng.uniform(*spec.length_range), rng.uniform(*spec.width_range)
    tint = PET_TINTS[i % len(PET_TINTS)]
    body = preview_material(stage, "/Object/Looks/Plastic", tint, roughness=0.15, opacity=0.45, ior=1.45)
    cap = preview_material(stage, "/Object/Looks/Cap", LABEL_COLORS[rng.integers(len(LABEL_COLORS))], roughness=0.5)
    label = preview_material(stage, "/Object/Looks/Label", LABEL_COLORS[i % len(LABEL_COLORS)], roughness=0.6)
    _bottle(stage, length, width / 2, body, cap, label if i % 3 != 2 else None, long_neck=False)
    return width


def make_can(stage, rng, i):
    spec = config.CLASS_SPECS["aluminum_can"]
    length, width = rng.uniform(*spec.length_range), rng.uniform(*spec.width_range)
    color = CAN_COLORS[i % len(CAN_COLORS)]
    body = preview_material(stage, "/Object/Looks/Metal", color, metallic=0.9, roughness=0.25)
    rim = preview_material(stage, "/Object/Looks/Rim", (0.8, 0.8, 0.82), metallic=1.0, roughness=0.2)
    _cylinder_x(stage, "/Object/Body", width / 2, length * 0.92, 0.0, body)
    for k, sx in enumerate((-1, 1)):
        _cylinder_x(stage, f"/Object/Rim{k}", width / 2 * 0.93, length * 0.04, sx * length * 0.48, rim)
    return width


def make_carton(stage, rng, i):
    spec = config.CLASS_SPECS["paper_carton"]
    length, width = rng.uniform(*spec.length_range), rng.uniform(*spec.width_range)
    thick = spec.thickness * rng.uniform(0.9, 1.1)
    color = CARTON_COLORS[i % len(CARTON_COLORS)]
    mat = preview_material(stage, "/Object/Looks/Paper", color, roughness=0.85)
    cube = UsdGeom.Cube.Define(stage, "/Object/Body")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(length, width, thick))
    bind(cube.GetPrim(), mat)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if i % 2 == 0:  # printed panel on top, like a juice box
        stripe_mat = preview_material(stage, "/Object/Looks/Print", LABEL_COLORS[i % len(LABEL_COLORS)], roughness=0.7)
        stripe = UsdGeom.Cube.Define(stage, "/Object/Print")
        stripe.CreateSizeAttr(1.0)
        stripe.AddTranslateOp().Set(Gf.Vec3d(0, 0, thick / 2 + 0.0005))
        stripe.AddScaleOp().Set(Gf.Vec3f(length * 0.5, width * 0.9, 0.001))
        bind(stripe.GetPrim(), stripe_mat)
    return thick


def make_glass(stage, rng, i):
    spec = config.CLASS_SPECS["glass_bottle"]
    length, width = rng.uniform(*spec.length_range), rng.uniform(*spec.width_range)
    tint = GLASS_TINTS[i % len(GLASS_TINTS)]
    glass = preview_material(stage, "/Object/Looks/Glass", tint, roughness=0.03, opacity=0.35, ior=1.52)
    cap = preview_material(stage, "/Object/Looks/Cap", (0.75, 0.70, 0.30), metallic=0.8, roughness=0.3)
    label = preview_material(stage, "/Object/Looks/Label", (0.95, 0.92, 0.85), roughness=0.7)
    _bottle(stage, length, width / 2, glass, cap, label if i % 2 == 0 else None, long_neck=True)
    return width


MAKERS = {
    "pet_bottle": make_pet,
    "aluminum_can": make_can,
    "paper_carton": make_carton,
    "glass_bottle": make_glass,
}


def generate(out_dir: Path = PLACEHOLDER_DIR, variants: int = 6, seed: int = 0) -> list[Path]:
    """Write ``variants`` placeholder USDs per class. Returns the written paths."""
    rng = np.random.default_rng(seed)
    written = []
    for cls in config.CLASS_NAMES:
        for i in range(variants):
            path = out_dir / cls / f"{cls}_{i:02d}.usda"
            if path.exists():
                path.unlink()
            stage = new_asset_stage(path)
            thickness = MAKERS[cls](stage, rng, i)
            finish_asset(stage, cls, thickness, config.CLASS_SPECS[cls].mass)
            written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=int, default=6, help="variants per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PLACEHOLDER_DIR)
    args = ap.parse_args()
    paths = generate(args.out, args.variants, args.seed)
    print(f"Wrote {len(paths)} placeholder assets to {args.out}")


if __name__ == "__main__":
    main()
