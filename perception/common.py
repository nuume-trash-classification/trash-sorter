"""Shared helpers for the perception scripts (run them with ``uv run python perception/<script>.py``)."""

from __future__ import annotations

from pathlib import Path

from trash_sorter.yolo_format import write_data_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "segment"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_data(data: str | Path) -> Path:
    """Accept a ``data.yaml`` or a dataset folder; for a folder, (re)write its ``data.yaml``.

    A folder may have ``images/{train,val}`` (training sets) or a flat ``images/`` (test sets).
    """
    p = Path(data)
    if p.is_file():
        return p
    if not (p / "images").is_dir():
        raise FileNotFoundError(f"{p} is neither a data.yaml nor a folder with images/")
    if (p / "images" / "train").is_dir():
        val = "images/val" if (p / "images" / "val").is_dir() else "images/train"
        return write_data_yaml(p, {"train": "images/train", "val": val})
    return write_data_yaml(p, {"train": "images", "val": "images", "test": "images"})


def list_images(folder: Path) -> list[Path]:
    return sorted(f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
