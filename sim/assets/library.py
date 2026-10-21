"""Find the object assets to use: real converted models if present, otherwise placeholders.

Lookup order per class:
  1. ``sim/assets/usd/real/<class>/*.usd*``        (made by ``convert_assets.py``)
  2. ``sim/assets/usd/placeholder/<class>/*.usd*`` (made by ``make_placeholders.py``)
"""

from __future__ import annotations

from pathlib import Path

from sim.assets.make_placeholders import PLACEHOLDER_DIR, generate
from trash_sorter import config

ASSET_ROOT = Path(__file__).resolve().parent / "usd"
REAL_DIR = ASSET_ROOT / "real"


def _usd_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.usd*") if p.suffix in (".usd", ".usda", ".usdc"))


def find_assets(source: str = "auto") -> dict[str, list[Path]]:
    """Map class name -> list of USD files.

    ``source``: ``"auto"`` (real if a class has any, else placeholder), ``"real"`` or ``"placeholder"``.
    Placeholders are generated automatically the first time.
    """
    if not any(_usd_files(PLACEHOLDER_DIR / c) for c in config.CLASS_NAMES):
        generate()
    out: dict[str, list[Path]] = {}
    for cls in config.CLASS_NAMES:
        real = _usd_files(REAL_DIR / cls)
        placeholder = _usd_files(PLACEHOLDER_DIR / cls)
        if source == "real":
            if not real:
                raise FileNotFoundError(f"No converted real assets for {cls} in {REAL_DIR / cls}")
            out[cls] = real
        elif source == "placeholder":
            out[cls] = placeholder
        else:
            out[cls] = real or placeholder
    return out


def describe(assets: dict[str, list[Path]]) -> str:
    lines = []
    for cls, files in assets.items():
        kind = "real" if files and REAL_DIR in files[0].parents else "placeholder"
        lines.append(f"  {cls:<13} {len(files):>3} assets ({kind})")
    return "\n".join(lines)
