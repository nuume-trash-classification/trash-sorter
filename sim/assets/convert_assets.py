"""Batch-convert downloaded / scanned 3D models into simulation-ready USD assets (workstream 1).

Nobody edits models by hand. Put the files in class folders and run this script:

    sim/assets/raw/pet_bottle/*.obj|glb|gltf|stl|ply|off
    sim/assets/raw/aluminum_can/...
    sim/assets/raw/paper_carton/...
    sim/assets/raw/glass_bottle/...

Optional per class: ``sizes.csv`` with lines ``filename,length_m`` giving the real longest
side of a model (measure the real object with a ruler). Otherwise the class average from
``config.CLASS_SPECS`` is used.

For every model the script:
  1. merges all parts into one mesh,
  2. rotates it so it lies flat (longest side -> X, thinnest side -> Z),
  3. scales it to real size, centers it,
  4. assigns a class material (color averaged from the model's texture/colors; glass and
     PET get a translucent material, cans get metal),
  5. adds a convex-hull collider and the class mass,
  6. writes ``sim/assets/usd/real/<class>/<name>.usdc`` following the asset contract in
     ``usd_utils.py``.

Run with the Isaac Sim Python (no simulator start needed):

    <isaac python> sim/assets/convert_assets.py [--classes pet_bottle glass_bottle] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import trimesh
from pxr import UsdGeom, UsdPhysics, Vt

from sim.assets.usd_utils import bind, finish_asset, new_asset_stage, preview_material
from trash_sorter import config

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "usd" / "real"
MODEL_EXTS = {".obj", ".glb", ".gltf", ".stl", ".ply", ".off"}
MAX_FACES_WARN = 50_000

# Per-class look: (metallic, roughness, opacity, ior). Color comes from the model.
CLASS_LOOK = {
    "pet_bottle": (0.0, 0.15, 0.45, 1.45),
    "aluminum_can": (0.9, 0.25, 1.0, 1.5),
    "paper_carton": (0.0, 0.85, 1.0, 1.5),
    "glass_bottle": (0.0, 0.03, 0.35, 1.52),
}


def read_sizes(class_dir: Path) -> dict[str, float]:
    f = class_dir / "sizes.csv"
    if not f.exists():
        return {}
    sizes = {}
    with f.open(encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and not row[0].startswith("#"):
                try:
                    sizes[row[0].strip()] = float(row[1])
                except ValueError:
                    continue  # header line
    return sizes


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("no triangles found")
    return mesh


def average_color(mesh: trimesh.Trimesh) -> tuple[float, float, float] | None:
    try:
        colors = mesh.visual.to_color().vertex_colors
    except Exception:
        return None
    if colors is None or len(colors) == 0:
        return None
    rgb = np.asarray(colors)[:, :3].mean(axis=0) / 255.0
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def normalize_pose(mesh: trimesh.Trimesh, target_length: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Lay the mesh flat (X longest, Z thinnest), scale to ``target_length``, center. Returns (mesh, extents)."""
    mesh = mesh.copy()
    obb = mesh.bounding_box_oriented
    mesh.apply_transform(np.linalg.inv(obb.primitive.transform))
    order = np.argsort(-mesh.extents)  # longest first
    perm = np.zeros((4, 4))
    for new_axis, old_axis in enumerate(order):
        perm[new_axis, old_axis] = 1.0
    perm[3, 3] = 1.0
    if np.linalg.det(perm[:3, :3]) < 0:  # keep it a rotation, not a mirror
        perm[2, :3] *= -1
    mesh.apply_transform(perm)
    mesh.apply_scale(target_length / mesh.extents[0])
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh, mesh.extents.copy()


def write_usd(mesh: trimesh.Trimesh, out_path: Path, cls: str, rgb: tuple[float, float, float]) -> None:
    stage = new_asset_stage(out_path)
    geom = UsdGeom.Mesh.Define(stage, "/Object/Mesh")
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.astype(np.int32).ravel()))
    geom.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
    geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = mesh.bounds
    geom.CreateExtentAttr([tuple(map(float, lo)), tuple(map(float, hi))])
    metallic, roughness, opacity, ior = CLASS_LOOK[cls]
    bind(geom.GetPrim(), preview_material(stage, "/Object/Looks/Main", rgb, metallic, roughness, opacity, ior))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
    finish_asset(stage, cls, float(mesh.extents[2]), config.CLASS_SPECS[cls].mass)


def convert_class(cls: str, raw_dir: Path, out_dir: Path, dry_run: bool) -> list[dict]:
    class_dir = raw_dir / cls
    if not class_dir.is_dir():
        return []
    spec = config.CLASS_SPECS[cls]
    sizes = read_sizes(class_dir)
    default_len = float(np.mean(spec.length_range))
    rows = []
    for path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in MODEL_EXTS):
        row: dict[str, object] = {"class": cls, "file": path.name, "status": "ok", "note": ""}
        try:
            mesh = load_mesh(path)
            target = sizes.get(path.name, default_len)
            mesh, ext = normalize_pose(mesh, target)
            rgb = average_color(mesh) or (0.6, 0.6, 0.6)
            row.update(
                length=round(ext[0], 4), width=round(ext[1], 4), thickness=round(ext[2], 4), faces=len(mesh.faces)
            )
            notes = []
            if abs(ext[2] - spec.thickness) > 0.2 * spec.thickness:
                notes.append(f"thickness {ext[2]:.3f} m vs class {spec.thickness:.3f} m: check size / model")
            if len(mesh.faces) > MAX_FACES_WARN:
                notes.append(f"{len(mesh.faces)} faces: heavy, consider decimating")
            row["note"] = "; ".join(notes)
            if not dry_run:
                write_usd(mesh, out_dir / cls / f"{path.stem}.usdc", cls, rgb)
        except Exception as exc:  # keep going: one broken download must not stop the batch
            row.update(status="failed", note=str(exc))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="input folder with one subfolder per class")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--classes", nargs="*", default=list(config.CLASS_NAMES))
    ap.add_argument("--dry-run", action="store_true", help="only check the models, write nothing")
    args = ap.parse_args()

    rows = []
    for cls in args.classes:
        rows += convert_class(cls, args.raw, args.out, args.dry_run)
    if not rows:
        print(f"No models found under {args.raw}/<class>/ (extensions: {sorted(MODEL_EXTS)})")
        return
    report = args.out / "report.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    keys = ["class", "file", "status", "length", "width", "thickness", "faces", "note"]
    with report.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        dims = f"{r.get('length', '-')} x {r.get('width', '-')} x {r.get('thickness', '-')} m"
        print(f"[{r['status']:>6}] {r['class']:<13} {r['file']:<30} {dims:<26} {r['note']}")
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} converted. Report: {report}")


if __name__ == "__main__":
    main()
