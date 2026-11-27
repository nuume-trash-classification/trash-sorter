"""Which parts of a bottle model are see-through, and what tint they get (pure logic, no ``pxr``).

Used by ``convert_assets.py``; kept separate so it can be unit tested outside Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

GLASS_CLASSES = ("pet_bottle", "glass_bottle")
MIN_TINT = 0.4  # brightest channel of a glass tint; darker absorbs almost all light and renders black


def marked_transparent(alpha_mode: str | None, base_alpha: float) -> bool:
    """glTF marks see-through materials with alphaMode BLEND or a base color alpha below 1."""
    return alpha_mode == "BLEND" or base_alpha < 0.98


def clear_parts(cls: str, areas: Sequence[float], marked: Sequence[bool]) -> list[bool]:
    """Which parts become glass.

    Bottles: the parts the file marks transparent; if it marks none, the largest part (the body,
    while labels and caps stay opaque). Other classes: none.
    """
    if cls not in GLASS_CLASSES or not areas:
        return [False] * len(areas)
    if any(marked):
        return list(marked)
    largest = max(range(len(areas)), key=lambda i: areas[i])
    return [i == largest for i in range(len(areas))]


def brighten_glass_texture(pixels: np.ndarray, target: float = 0.6) -> np.ndarray:
    """Scale a glass color texture (uint8 RGB) so its average brightest channel is at least ``target``.

    Textures painted for opaque bottles are dark; used as glass tint they absorb nearly all light.
    One factor for the whole image keeps the hues and label contrast.
    """
    peak = pixels[..., :3].max(axis=-1).astype(np.float32).mean() / 255.0
    scale = max(target / max(peak, 1e-3), 1.0)
    return np.clip(pixels.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def glass_tint(rgb: Sequence[float]) -> tuple[float, float, float]:
    """Keep the hue of a part's color as glass tint, but bright enough to let light through.

    Near-black (no usable hue) becomes a neutral grey.
    """
    peak = max(rgb)
    if peak < 0.05:
        return (0.6, 0.6, 0.6)
    scale = max(MIN_TINT / peak, 1.0)
    return tuple(min(float(c) * scale, 1.0) for c in rgb)  # type: ignore[return-value]
