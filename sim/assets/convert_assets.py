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
  1. groups the faces by source material (one USD GeomSubset + material each),
  2. rotates it so it lies flat (longest side -> X, thinnest side -> Z),
  3. scales it to real size, centers it,
  4. gives each part a material: its own color texture (labels, logos) or average color;
     cans are metallic. Bottles: the parts the file marks transparent (or else the largest
     part, the body) become see-through OmniGlass tinted with their color, so labels and caps
     stay opaque (``part_rules.py``; ``--no-textures``: average colors only),
  5. adds a convex-hull collider (whole model) and the class mass,
  6. writes ``sim/assets/usd/real/<class>/<name>.usdc`` following the asset contract in
     ``usd_utils.py``.

Run with the Isaac Sim Python (no simulator start needed):

    <isaac python> sim/assets/convert_assets.py [--classes pet_bottle glass_bottle] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import trimesh
from PIL import Image
from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from sim.assets.part_rules import brighten_glass_texture, clear_parts, glass_tint, marked_transparent
from sim.assets.usd_utils import (
    add_base_color_texture,
    bind,
    finish_asset,
    glass_material,
    new_asset_stage,
    preview_material,
)
from trash_sorter import config

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "usd" / "real"
MODEL_EXTS = {".obj", ".glb", ".gltf", ".stl", ".ply", ".off"}
MAX_FACES_WARN = 50_000
MAX_TEXTURE = 1024  # px; 100 objects with 4K textures would eat GPU memory

# Per-class look of the opaque parts: (metallic, roughness). Color comes from the model.
CLASS_LOOK = {
    "pet_bottle": (0.0, 0.3),  # caps, labels
    "aluminum_can": (0.9, 0.25),
    "paper_carton": (0.0, 0.85),
    "glass_bottle": (0.0, 0.3),
}
GLASS_IOR = {"pet_bottle": 1.57, "glass_bottle": 1.52}


@dataclass
class Part:
    """All faces of one source material: one GeomSubset and one material in the USD."""

    mesh: trimesh.Trimesh
    rgb: tuple[float, float, float]
    texture: object | None  # PIL image
    uv: np.ndarray | None
    marked_clear: bool
    clear: bool = False


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


def average_color(mesh: trimesh.Trimesh) -> tuple[float, float, float] | None:
    try:
        colors = mesh.visual.to_color().vertex_colors
    except Exception:
        return None
    if colors is None or len(colors) == 0:
        return None
    rgb = np.atleast_2d(colors)[:, :3].mean(axis=0) / 255.0  # one color for the whole mesh comes back 1-D
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _part(mesh: trimesh.Trimesh, textures: bool) -> Part:
    visual, material = mesh.visual, getattr(mesh.visual, "material", None)
    uv = getattr(visual, "uv", None)
    uv = np.asarray(uv, dtype=np.float32) if uv is not None and len(uv) == len(mesh.vertices) else None
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    factor = getattr(material, "baseColorFactor", None)
    alpha = 1.0 if factor is None else float(factor[3]) / 255.0
    return Part(
        mesh=mesh,
        rgb=average_color(mesh) or (0.6, 0.6, 0.6),
        texture=image if textures and uv is not None else None,
        uv=uv,
        marked_clear=marked_transparent(getattr(material, "alphaMode", None), alpha),
    )


def load_parts(path: Path, cls: str, textures: bool = True) -> list[Part]:
    """The model's meshes grouped by source material, with the see-through ones marked ``clear``."""
    loaded = trimesh.load(str(path), process=True)
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]
    groups: dict[object, list[trimesh.Trimesh]] = {}
    for m in meshes:
        if isinstance(m, trimesh.Trimesh) and len(m.faces):
            material = getattr(m.visual, "material", None)
            groups.setdefault(getattr(material, "name", None) or id(material), []).append(m)
    if not groups:
        raise ValueError("no triangles found")
    parts = [_part(ms[0] if len(ms) == 1 else trimesh.util.concatenate(ms), textures) for ms in groups.values()]
    flags = clear_parts(cls, [float(p.mesh.area) for p in parts], [p.marked_clear for p in parts])
    for part, flag in zip(parts, flags, strict=True):
        part.clear = flag
    return parts


def save_texture(image, path: Path, glass: bool = False) -> None:
    """RGB only (see-through comes from the glass material), at most MAX_TEXTURE px per side.

    ``glass``: the texture tints a see-through part, so dark paint is brightened (``part_rules``).
    """
    image = image.convert("RGB")
    image.thumbnail((MAX_TEXTURE, MAX_TEXTURE))
    if glass:
        image = Image.fromarray(brighten_glass_texture(np.asarray(image)))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def pose_transform(mesh: trimesh.Trimesh, target_length: float) -> np.ndarray:
    """4x4 transform that lays the mesh flat (X longest, Z thinnest), scales to ``target_length``, centers."""
    obb = mesh.bounding_box_oriented.primitive
    order = np.argsort(-np.asarray(obb.extents))  # longest first
    perm = np.zeros((4, 4))
    for new_axis, old_axis in enumerate(order):
        perm[new_axis, old_axis] = 1.0
    perm[3, 3] = 1.0
    if np.linalg.det(perm[:3, :3]) < 0:  # keep it a rotation, not a mirror
        perm[2, :3] *= -1
    m = perm @ np.linalg.inv(obb.transform)
    v = trimesh.transform_points(mesh.vertices, m)
    scale = target_length / float(np.ptp(v[:, 0]))
    m = np.diag([scale, scale, scale, 1.0]) @ m
    v = trimesh.transform_points(mesh.vertices, m)
    shift = np.eye(4)
    shift[:3, 3] = -(v.min(axis=0) + v.max(axis=0)) / 2
    return shift @ m


def pose_parts(parts: list[Part], target_length: float) -> tuple[list[Part], np.ndarray]:
    """Apply one common pose to all parts. Returns (parts, extents of the whole model)."""
    whole = trimesh.util.concatenate([trimesh.Trimesh(p.mesh.vertices, p.mesh.faces, process=False) for p in parts])
    m = pose_transform(whole, target_length)
    for p in parts:
        p.mesh = p.mesh.copy()
        p.mesh.apply_transform(m)
    whole.apply_transform(m)
    return parts, whole.extents.copy()


def _define_mesh(stage, path: str, parts: list[Part]) -> tuple[UsdGeom.Mesh, list[np.ndarray]]:
    """One USD mesh from several parts. Returns (mesh, face indices of each part)."""
    geom = UsdGeom.Mesh.Define(stage, path)
    points, normals, faces, uvs, face_ranges = [], [], [], [], []
    n_vert = n_face = 0
    for p in parts:
        points.append(p.mesh.vertices)
        normals.append(p.mesh.vertex_normals)
        faces.append(p.mesh.faces + n_vert)
        uvs.append(p.uv if p.uv is not None else np.zeros((len(p.mesh.vertices), 2), np.float32))
        face_ranges.append(np.arange(n_face, n_face + len(p.mesh.faces), dtype=np.int32))
        n_vert += len(p.mesh.vertices)
        n_face += len(p.mesh.faces)
    pts, fcs = np.vstack(points).astype(np.float32), np.vstack(faces).astype(np.int32)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(fcs), 3, dtype=np.int32)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(fcs.ravel()))
    geom.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.vstack(normals).astype(np.float32)))
    geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateExtentAttr([tuple(map(float, pts.min(axis=0))), tuple(map(float, pts.max(axis=0)))])
    st = UsdGeom.PrimvarsAPI(geom).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(np.vstack(uvs).astype(np.float32)))
    return geom, face_ranges


def write_usd(parts: list[Part], out_path: Path, cls: str) -> None:
    """Prims: ``Mesh`` (opaque parts), ``Glass`` (see-through parts, no shadow), ``Collision`` (whole model).

    RTX shadows ignore transmission: glass would cast a solid black shadow, and the shadowed
    belt seen through the glass makes the bottle look black. So glass casts none.
    """
    stage = new_asset_stage(out_path)
    metallic, roughness = CLASS_LOOK[cls]
    looks = {}
    for i, p in enumerate(parts):
        look = f"/Object/Looks/Part{i}"
        tex_name = None
        if p.texture is not None:
            tex_name = f"{out_path.stem}_part{i}.png"
            save_texture(p.texture, out_path.with_name(tex_name), glass=p.clear)
        if p.clear:
            tex = f"./{tex_name}" if tex_name else None
            looks[id(p)] = glass_material(stage, look, glass_tint(p.rgb), GLASS_IOR[cls], tex)
        else:
            looks[id(p)] = preview_material(stage, look, p.rgb, metallic, roughness)
            if tex_name:
                add_base_color_texture(stage, look, f"./{tex_name}")

    for name, group in (("Mesh", [p for p in parts if not p.clear]), ("Glass", [p for p in parts if p.clear])):
        if not group:
            continue
        geom, face_ranges = _define_mesh(stage, f"/Object/{name}", group)
        if name == "Glass":
            UsdGeom.PrimvarsAPI(geom).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        if len(group) == 1:
            bind(geom.GetPrim(), looks[id(group[0])])
            continue
        binding = UsdShade.MaterialBindingAPI.Apply(geom.GetPrim())
        for i, (p, faces_i) in enumerate(zip(group, face_ranges, strict=True)):
            bind(binding.CreateMaterialBindSubset(f"part{i}", Vt.IntArray.FromNumpy(faces_i)).GetPrim(), looks[id(p)])
        binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.nonOverlapping)

    collision, _ = _define_mesh(stage, "/Object/Collision", parts)
    collision.CreatePurposeAttr(UsdGeom.Tokens.guide)  # physics only, never rendered
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
    lo, hi = collision.GetExtentAttr().Get()
    finish_asset(stage, cls, float(hi[2] - lo[2]), config.CLASS_SPECS[cls].mass)


def convert_class(cls: str, raw_dir: Path, out_dir: Path, dry_run: bool, textures: bool = True) -> list[dict]:
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
            parts, ext = pose_parts(load_parts(path, cls, textures), sizes.get(path.name, default_len))
            faces = sum(len(p.mesh.faces) for p in parts)
            row.update(
                length=round(ext[0], 4),
                width=round(ext[1], 4),
                thickness=round(ext[2], 4),
                faces=faces,
                parts=len(parts),
                clear_parts=sum(p.clear for p in parts),
            )
            notes = []
            if abs(ext[2] - spec.thickness) > 0.2 * spec.thickness:
                notes.append(f"thickness {ext[2]:.3f} m vs class {spec.thickness:.3f} m: check size / model")
            if faces > MAX_FACES_WARN:
                notes.append(f"{faces} faces: heavy, consider decimating")
            row["note"] = "; ".join(notes)
            if not dry_run:
                write_usd(parts, out_dir / cls / f"{path.stem}.usdc", cls)
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
    ap.add_argument("--no-textures", action="store_true", help="flat class color only (the old look, for comparison)")
    args = ap.parse_args()

    rows = []
    for cls in args.classes:
        rows += convert_class(cls, args.raw, args.out, args.dry_run, textures=not args.no_textures)
    if not rows:
        print(f"No models found under {args.raw}/<class>/ (extensions: {sorted(MODEL_EXTS)})")
        return
    report = args.out / "report.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    keys = ["class", "file", "status", "length", "width", "thickness", "faces", "parts", "clear_parts", "note"]
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
