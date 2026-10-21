"""Perception interface: camera image -> detections in world (belt) coordinates.

This file is THE seam between "dev mode" and "final mode":

* ``YoloPerception`` (final, default): only uses the RGB image + camera calibration +
  per-class thickness. This is what would run on the real system.
* ``sim/gt_perception.py`` (dev only, ``--perception gt``): reads object poses from the
  simulator so the arm can be developed before a model exists. Never use it for results.

Both return the same ``Detection`` objects, so nothing downstream knows which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from trash_sorter import config
from trash_sorter.geometry import CameraModel, pick_pixel_from_mask, pick_point_world


@dataclass
class Detection:
    """One object seen in one camera frame."""

    class_name: str
    confidence: float
    position: np.ndarray  # pick point on the object's top surface, world frame (x, y, z)
    t_observed: float  # time the image was taken
    pixel: tuple[int, int] | None = None  # pick point in the image (u, v)
    mask: np.ndarray | None = field(default=None, repr=False)


class Perception(Protocol):
    """Anything that turns a camera frame into detections."""

    def detect(self, rgb: np.ndarray, t: float) -> list[Detection]: ...


def masks_to_detections(
    masks: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    camera: CameraModel,
    t: float,
    min_area_px: int = 150,
    border_px: int = 3,
) -> list[Detection]:
    """Turn per-instance boolean masks (N, H, W) into world-frame detections.

    Masks touching the left/right image border are skipped: the object is only partly
    visible, so its pick point would be wrong. It is seen fully a few frames later.
    """
    dets: list[Detection] = []
    for mask, cls, conf in zip(masks, class_ids, confidences, strict=True):
        if int(mask.sum()) < min_area_px:
            continue
        if border_px and (mask[:, :border_px].any() or mask[:, -border_px:].any()):
            continue
        pixel = pick_pixel_from_mask(mask)
        if pixel is None:
            continue
        name = config.CLASS_NAMES[int(cls)]
        try:
            pos = pick_point_world(camera, pixel, name)
        except ValueError:
            continue
        dets.append(Detection(name, float(conf), pos, t, pixel, mask))
    return dets


class YoloPerception:
    """Ultralytics YOLO instance segmentation -> detections. RGB only, no depth."""

    def __init__(
        self,
        weights: str | Path,
        camera: CameraModel,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | int | None = None,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.model = YOLO(str(weights))
        self.camera = camera
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        names = self.model.names
        model_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        if tuple(model_names) != config.CLASS_NAMES:
            raise ValueError(f"Model classes {model_names} do not match config.CLASS_NAMES {config.CLASS_NAMES}")

    def detect(self, rgb: np.ndarray, t: float) -> list[Detection]:
        """Run the model on an RGB uint8 image (H, W, 3)."""
        bgr = np.ascontiguousarray(rgb[..., :3][..., ::-1])  # ultralytics expects BGR numpy arrays
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if res.masks is None or len(res.boxes) == 0:
            return []
        masks = res.masks.data.cpu().numpy() > 0.5
        h, w = rgb.shape[:2]
        if masks.shape[1:] != (h, w):
            import cv2

            masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in masks])
            masks = masks > 0
        return masks_to_detections(masks, res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), self.camera, t)
