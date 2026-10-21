"""Evaluate one or more YOLO-seg models on a labeled set (normally the REAL test photos).

    uv run python perception/eval.py --data data/real/test \\
        --weights runs/segment/sim_only/weights/best.pt runs/segment/sim_plus_real/weights/best.pt \\
        --names sim_only sim_plus_real

Reports mask mAP50 (the project's main metric), overall and per class, with glass bottles
shown separately, and checks the success criteria. Writes JSON + Markdown to outputs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.common import REPO_ROOT, resolve_data
from trash_sorter import config

# Success criteria from the design (mask mAP50 on real photos).
CRITERIA = {
    "sim_only": {"min": 0.5, "target": 0.7},
    "finetuned": {"target": 0.85},
}


def evaluate(weights: Path, data: Path, imgsz: int, split: str, device: str | None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data), split=split, imgsz=imgsz, device=device, plots=False, verbose=False)
    per_class = {config.CLASS_NAMES[int(c)]: float(ap) for c, ap in zip(m.seg.ap_class_index, m.seg.ap50, strict=False)}
    return {
        "weights": str(weights),
        "mask_map50": float(m.seg.map50),
        "mask_map50_95": float(m.seg.map),
        "box_map50": float(m.box.map50),
        "per_class_mask_map50": per_class,
        "mask_map50_without_glass": (
            sum(v for k, v in per_class.items() if k != "glass_bottle")
            / max(1, sum(1 for k in per_class if k != "glass_bottle"))
        ),
    }


def to_markdown(results: dict[str, dict]) -> str:
    cols = list(config.CLASS_NAMES)
    lines = [
        "| model | mask mAP50 | w/o glass | " + " | ".join(cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for name, r in results.items():
        pc = r["per_class_mask_map50"]
        cells = [f"{pc[c]:.3f}" if c in pc else "-" for c in cols]
        lines.append(
            f"| {name} | **{r['mask_map50']:.3f}** | {r['mask_map50_without_glass']:.3f} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def check_criteria(results: dict[str, dict]) -> list[str]:
    notes = []
    for name, r in results.items():
        v = r["mask_map50"]
        if "real" in name or "finetune" in name:
            t = CRITERIA["finetuned"]["target"]
            notes.append(f"{name}: {v:.3f} {'meets' if v >= t else 'below'} fine-tuned target {t}")
        else:
            c = CRITERIA["sim_only"]
            status = "meets target" if v >= c["target"] else ("meets minimum" if v >= c["min"] else "BELOW minimum")
            notes.append(f"{name}: {v:.3f} {status} (min {c['min']}, target {c['target']})")
    names = list(results)
    if len(names) >= 2:
        a, b = results[names[0]]["mask_map50"], results[names[-1]]["mask_map50"]
        notes.append(f"{names[-1]} vs {names[0]}: {b - a:+.3f} mask mAP50")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml or dataset folder (e.g. data/real/test)")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="one label per weights file")
    ap.add_argument("--split", default="val", help="split key in data.yaml to evaluate")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    names = args.names or [Path(w).parent.parent.name for w in args.weights]
    if len(names) != len(args.weights):
        sys.exit("--names needs one entry per --weights")
    data = resolve_data(args.data)
    results = {
        n: evaluate(Path(w), data, args.imgsz, args.split, args.device)
        for n, w in zip(names, args.weights, strict=True)
    }

    md = to_markdown(results)
    notes = check_criteria(results)
    print("\n" + md + "\n")
    print("\n".join(notes))
    out = REPO_ROOT / "outputs" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"data": str(data), "results": results}, indent=2), encoding="utf-8")
    (out / "results.md").write_text(
        f"Data: `{data}`\n\n{md}\n\n" + "\n".join(f"- {n}" for n in notes), encoding="utf-8"
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
