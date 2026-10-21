"""Import a CVAT / Roboflow "YOLO segmentation" export into data/real/ (workstream 3).

    # ~300 photos for the final evaluation (never used for training):
    uv run python perception/prepare_real.py --source exports/cvat_test --dest test
    # 50-100 *different* photos for fine-tuning (split into train/val):
    uv run python perception/prepare_real.py --source exports/cvat_finetune --dest finetune

What it does:
  * finds image/label pairs anywhere under --source,
  * maps the export's class names to ours by name (fails loudly on unknown names),
  * copies them into data/real/<dest>/ and writes data.yaml,
  * refuses to continue if any photo appears in both test and finetune (by file content).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from perception.common import REPO_ROOT, list_images, resolve_data
from trash_sorter import config

REAL_DIR = REPO_ROOT / "data" / "real"


def read_export_names(source: Path) -> list[str]:
    """Class names in export order, from data.yaml (Roboflow/CVAT) or obj.names (CVAT)."""
    for yml in list(source.rglob("data.yaml")) + list(source.rglob("*.yaml")):
        names = yaml.safe_load(yml.read_text(encoding="utf-8")).get("names")
        if names:
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for f in source.rglob("obj.names"):
        return [n.strip() for n in f.read_text(encoding="utf-8").splitlines() if n.strip()]
    raise FileNotFoundError("No data.yaml or obj.names in the export; cannot tell which id is which class")


def find_label(img: Path) -> Path | None:
    """YOLO exports keep labels in a sibling ``labels`` folder that mirrors ``images``."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")
            if cand.exists():
                return cand
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="unzipped export folder")
    ap.add_argument("--dest", choices=["test", "finetune"], required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2, help="finetune only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = read_export_names(args.source)
    unknown = [n for n in names if n not in config.CLASS_IDS]
    if unknown:
        sys.exit(f"Unknown class names in export: {unknown}. Use exactly: {list(config.CLASS_NAMES)}")
    remap = {i: config.CLASS_IDS[n] for i, n in enumerate(names)}

    pairs = []
    for img in list_images(args.source):
        label = find_label(img)
        if label is None:
            print(f"  skip (no label file): {img}")
            continue
        pairs.append((img, label))
    if not pairs:
        sys.exit("No labeled images found")

    other = REAL_DIR / ("finetune" if args.dest == "test" else "test")
    other_hashes = {file_hash(p) for p in list_images(other / "images")} if other.exists() else set()
    clash = [img for img, _ in pairs if file_hash(img) in other_hashes]
    if clash:
        sys.exit(f"{len(clash)} photos are already in {other}. Test and fine-tune photos must be different.")

    dest = REAL_DIR / args.dest
    if dest.exists():
        shutil.rmtree(dest)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pairs))
    n_val = int(round(len(pairs) * args.val_fraction)) if args.dest == "finetune" else 0
    for rank, idx in enumerate(order):
        img, label = pairs[idx]
        sub = "" if args.dest == "test" else ("val" if rank < n_val else "train")
        (dest / "images" / sub).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / sub).mkdir(parents=True, exist_ok=True)
        stem = f"{rank:04d}_{img.stem}"
        shutil.copy2(img, dest / "images" / sub / f"{stem}{img.suffix.lower()}")
        lines = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 7:  # class + >= 3 polygon points (boxes-only lines are not seg labels)
                lines.append(" ".join([str(remap[int(parts[0])]), *parts[1:]]))
        (dest / "labels" / sub / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yaml_path = resolve_data(dest)
    print(f"Imported {len(pairs)} photos into {dest} ({n_val} val). data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
