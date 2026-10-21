"""Train (or fine-tune) a YOLO instance-segmentation model.

Sim-only model (trained only on synthetic images):
    uv run python perception/train.py --data data/synthetic/sim_v1 --name sim_only --epochs 60

Sim + small real fine-tune (start from the sim-only weights, train on 50-100 real photos):
    uv run python perception/train.py --data data/real/finetune --name sim_plus_real \
        --model runs/segment/sim_only/weights/best.pt --epochs 40 --lr0 0.002

Never train or fine-tune on data/real/test: it is only for the final evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.common import RUNS_DIR, resolve_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml or dataset folder")
    ap.add_argument("--model", default="yolo26n-seg.pt", help="start weights: yolo26n-seg.pt / yolo26s-seg.pt / a .pt")
    ap.add_argument("--name", default="sim_only")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=None, help="initial learning rate (lower it for fine-tuning)")
    ap.add_argument("--freeze", type=int, default=None, help="freeze the first N layers (fine-tuning)")
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu for CPU")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    data = resolve_data(args.data)
    if "real" in data.parts and "test" in data.parts:
        sys.exit("Refusing to train on the real TEST set. Use data/real/finetune.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    extra = {k: v for k, v in {"lr0": args.lr0, "freeze": args.freeze, "device": args.device}.items() if v is not None}
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        # Mild extra augmentation helps sim-to-real; colors are already randomized in datagen.
        hsv_h=0.02,
        degrees=10.0,
        **extra,
    )
    print(f"\nBest weights: {RUNS_DIR / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
