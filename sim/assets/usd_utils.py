"""Small USD helpers shared by the asset scripts. Only needs ``pxr`` (no running simulator).

Asset contract (every object USD, placeholder or real, must follow it):

* ``defaultPrim`` = ``/Object`` (Xform), meters, Z up.
* Rest pose = lying flat on the belt: longest side along X, thickness along Z,
  origin at the bounding-box center.
* Collision shapes are already authored (``CollisionAPI``); mass on ``/Object`` (``MassAPI``).
  The scene adds the rigid body itself.
* ``customData``: ``trash_class`` (one of ``config.CLASS_NAMES``) and ``thickness`` (m).
* Materials are ``UsdPreviewSurface``; a real model may drive diffuseColor with its texture
  (``add_base_color_texture``, image file next to the USD). See-through bottle parts use
  ``glass_material`` (OmniGlass MDL), bound per part with a ``materialBind`` GeomSubset.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

TEXTURE_SHADER = "BaseColorTexture"


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


def add_base_color_texture(stage: Usd.Stage, material_path: str, texture: str) -> None:
    """Drive a ``preview_material``'s diffuseColor from an image (``texture``: path relative to the layer).

    The mesh needs a ``primvars:st`` UV set. The texture's ``scale`` input multiplies the image,
    so color jitter (``datagen.py``) keeps working on textured objects.
    """
    reader = UsdShade.Shader.Define(stage, material_path + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    tex = UsdShade.Shader.Define(stage, material_path + "/" + TEXTURE_SHADER)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
    surface = UsdShade.Shader.Get(stage, material_path + "/Shader")
    surface.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")


def glass_material(
    stage: Usd.Stage,
    path: str,
    tint: tuple[float, float, float],
    ior: float = 1.5,
    texture: str | None = None,
) -> UsdShade.Material:
    """See-through glass / PET (RTX ``OmniGlass``; ``UsdPreviewSurface`` opacity renders opaque in RTX).

    Thin-walled: a bottle is a thin shell, and downloaded meshes are often not closed volumes.
    ``texture`` (path relative to the layer) tints the glass per texel instead of ``tint``.
    """
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(ior)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    if texture is not None:
        sh.CreateInput("use_glass_color_texture", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("glass_color_texture", Sdf.ValueTypeNames.Asset).Set(texture)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    return mat


def base_color_texture(surface: UsdShade.Shader) -> UsdShade.Shader | None:
    """The ``UsdUVTexture`` added by ``add_base_color_texture`` next to ``surface``, if any."""
    prim = surface.GetPrim().GetParent().GetChild(TEXTURE_SHADER)
    return UsdShade.Shader(prim) if prim.IsValid() else None


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
