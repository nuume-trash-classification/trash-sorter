"""DEV-ONLY perception that reads ground-truth object poses from the simulator.

This breaks the "camera images only" rule on purpose so the arm / picking logic can be
developed before a trained model exists. It is only used with ``--perception gt``.
Final results must use ``--perception yolo`` (``trash_sorter.perception.YoloPerception``).

To stay comparable with YOLO it outputs the same thing: class + pick point on the top
surface (belt height + *class* thickness), only for objects inside the camera image.
"""

from __future__ import annotations

import numpy as np

from trash_sorter import config
from trash_sorter.geometry import CameraModel
from trash_sorter.perception import Detection


class GroundTruthPerception:
    def __init__(self, objects, rigid_view, camera: CameraModel, margin_px: int = 40) -> None:
        self.objects = objects  # list[sim.scene.PoolObject], same order as rigid_view
        self.view = rigid_view
        self.camera = camera
        self.margin = margin_px

    def detect(self, rgb: np.ndarray, t: float) -> list[Detection]:
        del rgb  # not used: that is the whole point of this class
        pos = self.view.get_world_poses()[0].numpy()
        dets = []
        for obj in self.objects:
            if obj.state != "on_belt":
                continue
            x, y, z = pos[obj.index]
            if z < config.BELT_TOP_Z - 0.02:
                continue
            u, v = self.camera.world_to_pixel((x, y, config.BELT_TOP_Z))
            if not (self.margin <= u < self.camera.width - self.margin and 0 <= v < self.camera.height):
                continue
            top = config.BELT_TOP_Z + config.CLASS_SPECS[obj.class_name].thickness
            dets.append(Detection(obj.class_name, 1.0, np.array([x, y, top]), t, (int(u), int(v))))
        return dets
