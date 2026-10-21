"""Draw the labels of a YOLO-seg dataset on its images, to check labels by eye.

    uv run python perception/show_labels.py data/real/test --n 40
    uv run python perception/show_labels.py data/synthetic/sim_v1 --n 40

Writes outputs/label_check/<dataset>/*.jpg and prints instance counts per class.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from perception.common import REPO_ROOT, list_images
from perception.prepare_real import find_label
from trash_sorter import config
from trash_sorter.yolo_format import draw_overlay, parse_label_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--n", type=int, default=40, help="how many overlays to write")
    args = ap.parse_args()

    out = REPO_ROOT / "outputs" / "label_check" / args.dataset.name
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    images = list_images(args.dataset / "images")
    missing = 0
    for i, img_path in enumerate(images):
        label = find_label(img_path)
        labels = parse_label_file(label) if label else []
        missing += label is None
        counts.update(config.CLASS_NAMES[c] for c, _ in labels)
        if i < args.n:
            rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(out / f"{img_path.stem}.jpg"), cv2.cvtColor(draw_overlay(rgb, labels), cv2.COLOR_RGB2BGR))
    print(f"{len(images)} images, {missing} without label file")
    for name in config.CLASS_NAMES:
        print(f"  {name:<13} {counts[name]}")
    print(f"Overlays: {out}")


if __name__ == "__main__":
    main()
