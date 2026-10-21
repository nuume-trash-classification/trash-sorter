"""Live webcam demo: YOLO-seg masks, class, confidence and the suction pick point (C).

    uv run python perception/live_demo.py --weights runs/segment/sim_plus_real/weights/best.pt
    uv run python perception/live_demo.py --weights best.pt --source 1          # second camera
    uv run python perception/live_demo.py --weights best.pt --source clip.mp4 --save demo.mp4

Keys: q / Esc = quit, s = save a screenshot to outputs/live/.
The red dot is the pick point (farthest point from the mask edge), the same rule the robot uses.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from perception.common import REPO_ROOT
from trash_sorter import config
from trash_sorter.geometry import pick_pixel_from_mask

COLORS = [(60, 80, 230), (230, 160, 60), (60, 200, 240), (120, 200, 90)]  # BGR per class


def draw(frame: np.ndarray, result, min_area: int = 100) -> np.ndarray:
    out = frame.copy()
    if result.masks is None:
        return out
    h, w = frame.shape[:2]
    masks = result.masks.data.cpu().numpy() > 0.5
    for mask, cls, conf in zip(masks, result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=False):
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if mask.sum() < min_area:
            continue
        color = COLORS[int(cls) % len(COLORS)]
        out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        pick = pick_pixel_from_mask(mask)
        if pick is not None:
            cv2.circle(out, pick, 6, (0, 0, 255), -1)
            cv2.circle(out, pick, 9, (255, 255, 255), 2)
            label = f"{config.CLASS_NAMES[int(cls)]} {conf:.2f}"
            cv2.putText(out, label, (pick[0] + 12, pick[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out, label, (pick[0] + 12, pick[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0", help="camera index or video/image file")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None, help="write the annotated video here (.mp4)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--no-window", action="store_true", help="do not open a window (for testing)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open video source {args.source!r}")
    writer = None
    shots = REPO_ROOT / "outputs" / "live"
    n, t_last, fps = 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        vis = draw(frame, result)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6) if n else 0.0
        t_last = now
        cv2.putText(vis, f"{fps:5.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if args.save:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(vis)
        n += 1
        if not args.no_window:
            cv2.imshow("trash-sorter live demo (q = quit, s = screenshot)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                shots.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(shots / f"shot_{time.strftime('%H%M%S')}.jpg"), vis)
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {n} frames")


if __name__ == "__main__":
    main()
