"""YOLO-seg label format helpers (one ``.txt`` per image).

Each line: ``class_id x1 y1 x2 y2 ...`` with polygon coordinates normalized to [0, 1].
This is also what CVAT / Roboflow export as "YOLO segmentation", so the same code reads
synthetic and real labels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from trash_sorter import config


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 20.0, epsilon_px: float = 1.0) -> list[np.ndarray]:
    """Outer contours of a binary mask as (N, 2) float arrays of (x, y) pixels, largest first."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area_px:
            continue
        c = cv2.approxPolyDP(c, epsilon_px, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.float64))
    return polys


def mask_to_label_line(class_id: int, mask: np.ndarray, min_area_px: float = 20.0) -> str | None:
    """YOLO-seg label line for one instance (largest visible part), or None if too small."""
    polys = mask_to_polygons(mask, min_area_px)
    if not polys:
        return None
    h, w = mask.shape[:2]
    poly = polys[0] / np.array([w, h])
    coords = " ".join(f"{v:.6f}" for v in np.clip(poly, 0.0, 1.0).ravel())
    return f"{class_id} {coords}"


def parse_label_file(path: str | Path) -> list[tuple[int, np.ndarray]]:
    """Read a YOLO-seg label file -> list of (class_id, normalized (N, 2) polygon)."""
    out = []
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        out.append((int(parts[0]), np.array(parts[1:], dtype=np.float64).reshape(-1, 2)))
    return out


def polygon_to_mask(poly_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize a normalized polygon to a boolean mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(poly_norm * np.array([width, height])).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def write_data_yaml(dataset_dir: str | Path, splits: dict[str, str] | None = None) -> Path:
    """Write the ``data.yaml`` that ultralytics needs. Paths are relative to the dataset dir."""
    dataset_dir = Path(dataset_dir)
    splits = splits or {"train": "images/train", "val": "images/val"}
    data = {"path": str(dataset_dir.resolve().as_posix()), **splits, "names": dict(enumerate(config.CLASS_NAMES))}
    out = dataset_dir / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


_COLORS = [(230, 80, 60), (60, 160, 230), (240, 200, 60), (90, 200, 120)]


def draw_overlay(rgb: np.ndarray, labels: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Draw label polygons on an RGB image, for checking labels by eye."""
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    h, w = img.shape[:2]
    fill = img.copy()
    for cls, poly in labels:
        pts = np.round(poly * np.array([w, h])).astype(np.int32)
        color = _COLORS[cls % len(_COLORS)]
        cv2.fillPoly(fill, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        x, y = pts.min(axis=0)
        name = config.CLASS_NAMES[cls] if cls < len(config.CLASS_NAMES) else str(cls)
        cv2.putText(img, name, (int(x), max(int(y) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(fill, 0.35, img, 0.65, 0)
