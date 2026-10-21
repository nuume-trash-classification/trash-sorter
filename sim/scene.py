"""Build the sorting cell: conveyor + upstream top-down camera + UR10 with suction + 5 bins + object pool.

Must be imported after ``SimulationApp`` has started. Used by ``run_sorter.py`` (full loop)
and ``datagen.py`` (synthetic images, ``with_robot=False``). All dimensions come from
``trash_sorter.config`` so perception math and the scene always agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.utils.semantics import add_labels
from isaacsim.sensors.experimental.rtx import RtxCamera
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from trash_sorter import config

UR10_ASSET = "/Isaac/Robots_Multiphysics/UniversalRobots/ur10/ur10.usda"
UR10_VARIANTS = [("Gripper", "short_suction")]
ROBOT_PATH = "/World/UR10"
GRIPPER_PATH = ROBOT_PATH + "/ee_link/SurfaceGripper"
CAMERA_PATH = "/World/CameraGantry/SortCamera"
OVERVIEW_CAMERA_PATH = "/World/OverviewCamera"
BELT_PATH = "/World/Conveyor/Belt"
PARK_Z = -2.0  # pooled objects wait below the floor


@dataclass
class PoolObject:
    """One pre-created object prim that gets recycled onto the belt."""

    index: int
    path: str
    class_name: str
    asset: Path
    thickness: float
    park_position: tuple[float, float, float]
    state: str = "parked"  # parked | on_belt | done
    spawn_t: float = -1.0


@dataclass
class Scene:
    objects: list[PoolObject]
    camera: RtxCamera
    belt_material: UsdShade.Shader
    dome_light: UsdLux.DomeLight
    sun_light: UsdLux.DistantLight
    regions: dict[str, tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small USD helpers (same patterns as the earlier pneumatic demo)
# ---------------------------------------------------------------------------
_materials: dict[tuple, UsdShade.Material] = {}


def _stage() -> Usd.Stage:
    return stage_utils.get_current_stage()


def color_material(name: str, rgb, metallic: float = 0.0, roughness: float = 0.5) -> UsdShade.Material:
    key = (name, tuple(rgb), metallic, roughness)
    if key not in _materials:
        path = f"/World/Looks/{name}_{len(_materials)}"
        mat = UsdShade.Material.Define(_stage(), path)
        sh = UsdShade.Shader.Define(_stage(), path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        _materials[key] = mat
    return _materials[key]


def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def physics_material(name: str, static_f: float, dynamic_f: float, restitution: float) -> UsdShade.Material:
    prim = _stage().DefinePrim(f"/World/PhysicsMaterials/{name}")
    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateStaticFrictionAttr().Set(static_f)
    mat.CreateDynamicFrictionAttr().Set(dynamic_f)
    mat.CreateRestitutionAttr().Set(restitution)
    return UsdShade.Material(prim)


def bind_physics(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def box(path: str, center, size, rgb, collide: bool = True, **mat_kwargs) -> UsdGeom.Xform:
    """Box with translate on the parent and scale on the child (keeps static colliders correct)."""
    xf = UsdGeom.Xform.Define(_stage(), path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube = UsdGeom.Cube.Define(_stage(), path + "/Geom")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(cube.GetPrim(), color_material(path.split("/")[-1], rgb, **mat_kwargs))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return xf


def open_bin(path: str, center_xy, size_xy, top_z: float, rgb) -> tuple[tuple[float, float], tuple[float, float]]:
    """Open-top bin standing on the floor. Returns its (x_range, y_range)."""
    (cx, cy), (lx, ly), t = center_xy, size_xy, 0.01
    xa, xb, ya, yb = cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2
    box(path + "/Bottom", (cx, cy, t / 2), (lx, ly, t), rgb)
    box(path + "/WallXa", (xa, cy, top_z / 2), (t, ly, top_z), rgb)
    box(path + "/WallXb", (xb, cy, top_z / 2), (t, ly, top_z), rgb)
    box(path + "/WallYa", (cx, ya, top_z / 2), (lx, t, top_z), rgb)
    box(path + "/WallYb", (cx, yb, top_z / 2), (lx, t, top_z), rgb)
    return (xa, xb), (ya, yb)


BIN_COLORS = {
    "pet_bottle": (0.20, 0.45, 0.85),
    "aluminum_can": (0.65, 0.65, 0.68),
    "paper_carton": (0.75, 0.55, 0.25),
    "glass_bottle": (0.15, 0.55, 0.25),
    "end_bin": (0.35, 0.35, 0.35),
}


# ---------------------------------------------------------------------------
# Scene parts
# ---------------------------------------------------------------------------
def _world_and_physics() -> None:
    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(120)
    physx_scene.CreateEnableCCDAttr().Set(True)


def _lights() -> tuple[UsdLux.DomeLight, UsdLux.DistantLight]:
    stage = _stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(400.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorTemperatureAttr(5500.0)
    sun.CreateEnableColorTemperatureAttr(True)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, 15, 0))
    return dome, sun


def _conveyor() -> UsdShade.Shader:
    """Kinematic belt with PhysX surface velocity (moves objects, never moves itself)."""
    stage = _stage()
    L, W, top = config.BELT_LENGTH, config.BELT_WIDTH, config.BELT_TOP_Z
    belt = UsdGeom.Cube.Define(stage, BELT_PATH)
    belt.CreateSizeAttr(1.0)
    belt.AddTranslateOp().Set(Gf.Vec3d(0, 0, top - 0.01))
    belt.AddScaleOp().Set(Gf.Vec3f(L, W, 0.02))
    # The belt gets its own material so datagen can randomize its color.
    mat = UsdShade.Material.Define(stage, "/World/Looks/BeltMaterial")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/BeltMaterial/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.05, 0.055))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    bind(belt.GetPrim(), mat)
    UsdPhysics.CollisionAPI.Apply(belt.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(belt.GetPrim()).CreateKinematicEnabledAttr().Set(True)
    surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt.GetPrim())
    # Surface velocity is in the belt's local frame, which the scale op stretches by L: divide it back.
    surface.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(config.BELT_SPEED / L, 0, 0))
    bind_physics(belt.GetPrim(), physics_material("BeltRubber", 0.9, 0.8, 0.0))

    frame = (0.55, 0.57, 0.60)
    x0, x1 = config.BELT_X_START, config.BELT_X_END
    for i, (sx, sy) in enumerate(
        (sx, sy) for sx in (x0 + 0.05, 0.0, x1 - 0.05) for sy in (-W / 2 - 0.03, W / 2 + 0.03)
    ):
        box(f"/World/Conveyor/Leg_{i}", (sx, sy, (top - 0.03) / 2), (0.04, 0.04, top - 0.03), frame, collide=False)
    # Low side guides keep round objects on the belt.
    for name, sy in (("GuideL", -W / 2 - 0.01), ("GuideR", W / 2 + 0.01)):
        box(f"/World/Conveyor/{name}", (0, sy, top + 0.02), (L, 0.02, 0.06), frame, metallic=0.6, roughness=0.4)
    return sh


def _camera() -> RtxCamera:
    stage = _stage()
    cam_z = config.BELT_TOP_Z + config.CAMERA_HEIGHT
    gantry = (0.30, 0.30, 0.33)
    for sy in (-config.BELT_WIDTH / 2 - 0.08, config.BELT_WIDTH / 2 + 0.08):
        box(
            f"/World/CameraGantry/Post_{'L' if sy < 0 else 'R'}",
            (config.CAMERA_X - 0.45, sy, (cam_z + 0.09) / 2),
            (0.04, 0.04, cam_z + 0.09),
            gantry,
            collide=False,
        )
    box(
        "/World/CameraGantry/Arm",
        (config.CAMERA_X - 0.225, 0, cam_z + 0.11),
        (0.45, 0.04, 0.03),
        gantry,
        collide=False,
    )
    box(
        "/World/CameraGantry/Top",
        (config.CAMERA_X - 0.45, 0, cam_z + 0.11),
        (0.04, config.BELT_WIDTH + 0.2, 0.03),
        gantry,
        collide=False,
    )
    box("/World/CameraGantry/CameraBody", (config.CAMERA_X, 0, cam_z + 0.075), (0.06, 0.06, 0.04), (0.08,) * 3, False)
    del stage
    cam = RtxCamera(CAMERA_PATH, positions=[[config.CAMERA_X, 0.0, cam_z]], orientations=[[1.0, 0.0, 0.0, 0.0]])
    cam.camera.set_focal_lengths(config.CAMERA_FOCAL_MM)
    cam.camera.set_apertures(
        horizontal_apertures=config.CAMERA_H_APERTURE_MM, vertical_apertures=config.CAMERA_V_APERTURE_MM
    )
    cam.camera.set_clipping_ranges(0.01, 10.0)
    return cam


def _robot(assets_root: str) -> None:
    bx, by, bz = config.ROBOT_BASE
    stage = _stage()
    ped = UsdGeom.Cylinder.Define(stage, "/World/RobotPedestal")
    ped.CreateRadiusAttr(0.11)
    ped.CreateHeightAttr(bz - 0.002)
    ped.CreateAxisAttr("Z")
    ped.AddTranslateOp().Set(Gf.Vec3d(bx, by, (bz - 0.002) / 2))
    bind(ped.GetPrim(), color_material("Pedestal", (0.85, 0.55, 0.10), roughness=0.6))
    stage_utils.add_reference_to_stage(usd_path=assets_root + UR10_ASSET, path=ROBOT_PATH, variants=UR10_VARIANTS)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(bx, by, bz))
    # Suction settings: reach a bit further than the 2 cm default, hold up to ~5 kg.
    grip = stage.GetPrimAtPath(GRIPPER_PATH)
    grip.GetAttribute("isaac:maxGripDistance").Set(0.03)
    grip.GetAttribute("isaac:coaxialForceLimit").Set(200.0)
    grip.GetAttribute("isaac:shearForceLimit").Set(200.0)


def _bins() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    regions = {}
    for cls, center in config.BIN_CENTERS.items():
        regions[cls] = open_bin(f"/World/Bins/{cls}", center, config.BIN_SIZE, config.BIN_TOP_Z, BIN_COLORS[cls])
    regions["end_bin"] = open_bin(
        "/World/Bins/end_bin", config.END_BIN_CENTER, config.END_BIN_SIZE, config.BIN_TOP_Z, BIN_COLORS["end_bin"]
    )
    return regions


def _object_pool(assets: dict[str, list[Path]], per_class: int) -> list[PoolObject]:
    stage = _stage()
    generic = physics_material("Generic", 0.6, 0.5, 0.05)
    pool: list[PoolObject] = []
    thickness_cache: dict[Path, float] = {}
    for cls in config.CLASS_NAMES:
        files = assets[cls]
        for i in range(per_class):
            asset = files[i % len(files)]
            path = f"/World/Objects/{cls}_{i:02d}"
            idx = len(pool)
            park = (6.0 + (idx % 10) * 0.4, -4.0 + (idx // 10) * 0.4, PARK_Z)
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(str(asset.as_posix()))
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*park))
            xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
            UsdPhysics.RigidBodyAPI.Apply(prim)
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb.CreateEnableCCDAttr().Set(True)
            # Real (dented, half-empty) bottles barely roll; perfect simulated cylinders roll forever.
            physx_rb.CreateAngularDampingAttr().Set(3.0)
            bind_physics(prim, generic)
            add_labels(prim, labels=[cls], taxonomy="class")
            if asset not in thickness_cache:
                asset_stage = Usd.Stage.Open(str(asset))
                thickness_cache[asset] = float(asset_stage.GetDefaultPrim().GetCustomDataByKey("thickness"))
            pool.append(PoolObject(idx, path, cls, asset, thickness_cache[asset], park))
    # Park them static-free: a floor far below so parked objects rest somewhere.
    box("/World/ParkingFloor", (8.0, -3.0, PARK_Z - 0.2), (8.0, 6.0, 0.1), (0.2, 0.2, 0.2))
    return pool


def build_scene(
    assets: dict[str, list[Path]],
    per_class: int = 8,
    with_robot: bool = True,
    assets_root: str | None = None,
) -> Scene:
    """Create the whole cell on a fresh stage."""
    _materials.clear()
    _world_and_physics()
    dome, sun = _lights()
    box("/World/Floor", (0, 0, -0.05), (20, 20, 0.1), (0.35, 0.36, 0.38))
    belt_shader = _conveyor()
    camera = _camera()
    if with_robot:
        if assets_root is None:
            from isaacsim.storage.native import get_assets_root_path

            assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac Sim assets root not found (needs internet or a local asset mirror)")
        _robot(assets_root)
    regions = _bins()
    objects = _object_pool(assets, per_class)
    return Scene(objects, camera, belt_shader, dome, sun, regions)


def look_at_quat(eye, target) -> list[float]:
    """Quaternion (w, x, y, z) for a USD camera at ``eye`` looking at ``target`` (Z up)."""
    from scipy.spatial.transform import Rotation

    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    z_c = -f
    x_c = np.cross([0, 0, 1], z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    q = Rotation.from_matrix(np.column_stack([x_c, y_c, z_c])).as_quat()
    return [q[3], q[0], q[1], q[2]]


def add_overview_camera() -> RtxCamera:
    eye, target = (1.6, 2.4, 2.1), (0.05, -0.05, 0.6)
    cam = RtxCamera(OVERVIEW_CAMERA_PATH, positions=[list(eye)], orientations=[look_at_quat(eye, target)])
    cam.camera.set_focal_lengths(18.0)
    cam.camera.set_apertures(horizontal_apertures=20.955, vertical_apertures=11.787)
    cam.camera.set_clipping_ranges(0.01, 100.0)
    return cam
