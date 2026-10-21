"""Small USD helpers shared by the asset scripts. Only needs ``pxr`` (no running simulator).

Asset contract (every object USD, placeholder or real, must follow it):

* ``defaultPrim`` = ``/Object`` (Xform), meters, Z up.
* Rest pose = lying flat on the belt: longest side along X, thickness along Z,
  origin at the bounding-box center.
* Collision shapes are already authored (``CollisionAPI``); mass on ``/Object`` (``MassAPI``).
  The scene adds the rigid body itself.
* ``customData``: ``trash_class`` (one of ``config.CLASS_NAMES``) and ``thickness`` (m).
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def new_asset_stage(path: Path) -> Usd.Stage:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def preview_material(
    stage: Usd.Stage,
    path: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.5,
    opacity: float = 1.0,
    ior: float = 1.5,
) -> UsdShade.Material:
    """UsdPreviewSurface material. Shader inputs can be overridden later for color jitter."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def finish_asset(stage: Usd.Stage, class_name: str, thickness: float, mass: float) -> None:
    root = stage.GetPrimAtPath("/Object")
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr().Set(mass)
    root.SetCustomDataByKey("trash_class", class_name)
    root.SetCustomDataByKey("thickness", float(thickness))
    stage.GetRootLayer().Save()


def read_asset_info(path: Path) -> dict:
    """Read ``customData`` + bounding box of an asset file."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    bbox = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root).ComputeAlignedRange()
    )
    return {
        "trash_class": root.GetCustomDataByKey("trash_class"),
        "thickness": root.GetCustomDataByKey("thickness"),
        "bbox_min": tuple(bbox.GetMin()),
        "bbox_max": tuple(bbox.GetMax()),
    }
