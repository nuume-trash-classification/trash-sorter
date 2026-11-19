"""Random belt appearance for synthetic data: a base color and an optional surface pattern.

The real belt is not known in advance, and a model trained on one belt look learns that look.
So the belt varies widely (dark, light, any hue, plain or patterned) instead of matching one
set of real photos. Pure numpy / OpenCV, so it is tested without Isaac Sim.

A texture is a bright grayscale-ish image; the renderer multiplies it by the belt color
(the texture's ``scale`` input), so one image serves every color.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

BELT_PATTERNS: tuple[str, ...] = ("speckle", "weave", "scuffed")
TEXTURE_MEAN = 0.75  # textures are normalized to this mean; divide the tint by it to keep the color


def sample_belt_color(rng: np.random.Generator) -> tuple[float, float, float]:
    """Linear RGB belt color: dark rubber, neutral gray of any brightness, or a saturated color."""
    kind = rng.random()
    if kind < 0.35:  # dark, nearly neutral: the usual conveyor belt
        hue, sat, val = rng.random(), rng.uniform(0.0, 0.35), rng.uniform(0.02, 0.10)
    elif kind < 0.60:  # gray, from mid to almost white
        hue, sat, val = rng.random(), rng.uniform(0.0, 0.10), rng.uniform(0.15, 0.85)
    else:  # colored: green / blue PVC belts, mats, painted tables
        hue, sat, val = rng.random(), rng.uniform(0.35, 0.90), rng.uniform(0.10, 0.80)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return float(r), float(g), float(b)


def _wrap_offsets(size: int) -> list[tuple[int, int]]:
    """Shifts that repeat a shape across the image borders, so the texture tiles without a seam."""
    return [(dx, dy) for dx in (-size, 0, size) for dy in (-size, 0, size)]


def _speckle(rng: np.random.Generator, size: int) -> np.ndarray:
    """Chips of mixed size and brightness on a plain base (terrazzo, recycled rubber)."""
    img = np.full((size, size), 0.75, np.float32)
    for _ in range(int(rng.integers(150, 900))):
        cx, cy = int(rng.integers(size)), int(rng.integers(size))
        axes = (int(rng.integers(1, max(2, size // 40))), int(rng.integers(1, max(2, size // 40))))
        angle, shade = float(rng.uniform(0, 180)), float(rng.uniform(0.25, 1.0))
        for dx, dy in _wrap_offsets(size):
            cv2.ellipse(img, (cx + dx, cy + dy), axes, angle, 0, 360, shade, -1)
    return img


def _weave(rng: np.random.Generator, size: int) -> np.ndarray:
    """Regular fabric / mesh grid, like a woven belt."""
    cells = int(rng.integers(16, 64))
    t = np.linspace(0, 2 * np.pi * cells, size, endpoint=False, dtype=np.float32)
    grid = 0.5 * (np.sin(t)[None, :] + np.sin(t)[:, None])
    return 0.75 + float(rng.uniform(0.08, 0.2)) * grid


def _scuffed(rng: np.random.Generator, size: int) -> np.ndarray:
    """Uneven wear: soft blotches plus thin scratches along the belt."""
    noise = rng.random((size, size)).astype(np.float32)
    sigma = size / float(rng.uniform(12, 30))
    blotches = cv2.GaussianBlur(np.tile(noise, (3, 3)), (0, 0), sigma)[size : 2 * size, size : 2 * size]
    blotches = (blotches - blotches.mean()) / (blotches.std() + 1e-6)
    img = (0.75 + float(rng.uniform(0.05, 0.12)) * blotches).astype(np.float32)
    for _ in range(int(rng.integers(20, 80))):
        y = int(rng.integers(size))
        x0, length = int(rng.integers(size)), int(rng.integers(size // 8, size))
        rise = int(rng.integers(-size // 30, size // 30 + 1))
        steps = np.arange(length)
        # Indexing modulo the size wraps the scratch around the borders exactly.
        img[(y + np.round(rise * steps / length).astype(int)) % size, (x0 + steps) % size] = rng.uniform(0.45, 1.0)
    return img


_MAKERS = {"speckle": _speckle, "weave": _weave, "scuffed": _scuffed}


def belt_texture(kind: str, rng: np.random.Generator, size: int = 512) -> np.ndarray:
    """One belt surface pattern as an (size, size, 3) uint8 image with mean about ``TEXTURE_MEAN``."""
    if kind not in _MAKERS:
        raise ValueError(f"unknown belt pattern {kind!r}; choose from {BELT_PATTERNS}")
    img = _MAKERS[kind](rng, size)
    img = np.clip(img * (TEXTURE_MEAN / float(img.mean())), 0.0, 1.0)
    return np.repeat((img * 255).round().astype(np.uint8)[..., None], 3, axis=2)
