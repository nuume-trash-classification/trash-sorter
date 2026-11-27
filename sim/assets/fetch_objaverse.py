"""Download candidate 3D models from Objaverse (LVIS subset) into sim/assets/raw/ (workstream 1).

Runs in the uv environment (no Isaac Sim needed):

    uv run --group assets python sim/assets/fetch_objaverse.py [--per-class 25] [--allow-nc]

For every class it:
  1. takes the models of a few matching LVIS categories (``LVIS_CATEGORIES``),
  2. keeps only allowed licenses, at most ``--max-faces`` faces, most-liked first,
  3. downloads them and checks the shape: longest/second and second/thinnest side ratios
     must be close to the class (``config.CLASS_SPECS``); this drops scenes, packs, flat cards,
     and decides between a bottle and a can when an LVIS category holds both,
  4. copies the accepted ones to ``raw/<class>/objaverse_<uid>.glb`` and appends a length to
     ``sizes.csv`` (the one that gives the class thickness, within the class length range),
  5. records every model it looked at in ``raw/objaverse_sources.csv`` (author, license, URL,
     status; CC-BY needs this attribution) and writes ``raw/review.html``.

A person must still review: LVIS labels are noisy (perfume in "bottle", a glass vs. a PET
bottle cannot be told by shape). Open ``review.html``, tick the models to keep, download
``keep.txt`` and run ``--apply-review keep.txt``: unticked files are deleted and never
downloaded again. Run without arguments to top each class up with new candidates, review the
new ones (earlier picks stay ticked), repeat. Then run ``convert_assets.py`` as usual.

``objaverse_sources.csv`` is tracked in git (the models are not): teammates get the same
selection with ``--restore``.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trash_sorter import config
from trash_sorter.config import ClassSpec

RAW_DIR = Path(__file__).resolve().parent / "raw"
SOURCES_NAME = "objaverse_sources.csv"
SOURCE_FIELDS = ["uid", "class", "status", "name", "author", "license", "url", "faces", "ratios", "file", "thumb"]
KEEP_STATUSES = ("accepted", "kept")  # "kept" = a person ticked it in review.html
PREVIEW_DIR = "_previews"
REVIEW_TEMPLATE = (Path(__file__).resolve().parent / "review_template.html").read_text(encoding="utf-8")

# Our class -> LVIS categories that may contain it. "pop_(soda)" holds both bottles and cans;
# the shape check picks the class.
LVIS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "pet_bottle": ("water_bottle", "bottle", "pop_(soda)"),
    "aluminum_can": ("beer_can", "can", "pop_(soda)"),
    "paper_carton": ("carton", "milk", "soya_milk", "orange_juice", "fruit_juice"),
    "glass_bottle": ("beer_bottle", "wine_bottle"),
}
FREE_LICENSES = {"cc0", "by", "by-sa"}
NC_LICENSES = {"by-nc", "by-nc-sa"}


def license_ok(license_id: str, allow_nc: bool = False) -> bool:
    """Objaverse license ids: cc0, by, by-sa, by-nd, by-nc, by-nc-sa, by-nc-nd. No-derivatives is never ok."""
    return license_id in FREE_LICENSES or (allow_nc and license_id in NC_LICENSES)


def shape_ratios(extents: Iterable[float]) -> tuple[float, float]:
    """(longest / second, second / thinnest) of a bounding box; independent of scale and orientation."""
    a, b, c = sorted((float(e) for e in extents), reverse=True)
    if c <= 0:
        return math.inf, math.inf
    return a / b, b / c


def ratio_bounds(spec: ClassSpec) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expected ranges of ``shape_ratios`` for a class, from its length / width / thickness."""
    (l_lo, l_hi), (w_lo, w_hi), t = spec.length_range, spec.width_range, spec.thickness
    # The second side is max(width, thickness) and the thinnest is min(width, thickness).
    r2_lo = 1.0 if w_lo <= t <= w_hi else min(max(w, t) / min(w, t) for w in (w_lo, w_hi))
    r2_hi = max(w_hi / t, t / w_lo, 1.0)
    return (l_lo / w_hi, l_hi / w_lo), (r2_lo, r2_hi)


def _log_gap(value: float, lo: float, hi: float) -> float:
    """How far ``value`` is outside [lo, hi], in log units (0 inside)."""
    if value < lo:
        return math.log(lo / value)
    if value > hi:
        return math.log(value / hi)
    return 0.0


def classify_shape(extents: Iterable[float], candidates: Iterable[str], tolerance: float = 1.25) -> str | None:
    """Best-fitting class among ``candidates``, or None if no class is within ``tolerance`` (a ratio factor)."""
    r1, r2 = shape_ratios(extents)
    if not math.isfinite(r1):
        return None
    best, best_gap = None, math.log(tolerance) + 1e-9
    for cls in sorted(candidates):
        (lo1, hi1), (lo2, hi2) = ratio_bounds(config.CLASS_SPECS[cls])
        gap = _log_gap(r1, lo1, hi1) + _log_gap(r2, lo2, hi2)
        if gap < best_gap:
            best, best_gap = cls, gap
    return best


def target_length(spec: ClassSpec, extents: Iterable[float]) -> float:
    """Real length for a model, which has no reliable scale of its own.

    Chosen so the thinnest side matches the class thickness (perception computes the pick
    height from it), clamped to the class length range.
    """
    a, _, c = sorted((float(e) for e in extents), reverse=True)
    lo, hi = spec.length_range
    return round(min(max(spec.thickness * a / c, lo), hi), 3)


def read_sources(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def append_rows(path: Path, fields: list[str], rows: list[dict], header: bool = True) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new and header:
            writer.writeheader()
        writer.writerows(rows)


def append_size(class_dir: Path, filename: str, length_m: float) -> None:
    """Add a ``filename,length_m`` line to ``sizes.csv`` unless the file already has one."""
    f = class_dir / "sizes.csv"
    if f.exists():
        with f.open(encoding="utf-8-sig") as fh:
            if any(row and row[0].strip() == filename for row in csv.reader(fh)):
                return
    append_rows(f, ["filename", "length_m"], [{"filename": filename, "length_m": length_m}])


def remove_size(class_dir: Path, filename: str) -> None:
    """Drop the ``sizes.csv`` line of a deleted model."""
    f = class_dir / "sizes.csv"
    if not f.exists():
        return
    with f.open(encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.reader(fh) if not (row and row[0].strip() == filename)]
    with f.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def thumbnail_url(annotation: dict, width: int = 256) -> str:
    images = (annotation.get("thumbnails") or {}).get("images") or []
    if not images:
        return ""
    return min(images, key=lambda im: abs(int(im.get("width", 0)) - width))["url"]


def render_preview(mesh, out: Path, size: tuple[int, int] = (260, 110)) -> None:
    """Top view of the model lying flat, as the belt camera sees it (no Isaac Sim needed).

    Sketchfab thumbnails can be dead links. Colors are sampled per vertex, so it only shows the shape
    and rough colors, not the full texture (the converted USD keeps the texture).
    Painter's algorithm over the faces, shaded by a fixed light, colored from the model's colors.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    m = mesh.copy()
    m.apply_transform(np.linalg.inv(m.bounding_box_oriented.primitive.transform))
    order = np.argsort(-m.extents)  # longest -> x, thinnest -> z (towards the camera)
    v = m.vertices[:, order]
    v = v - (v.min(axis=0) + v.max(axis=0)) / 2
    tri = v[m.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, -0.4, 1.0]) / np.linalg.norm([0.3, -0.4, 1.0])
    shade = 0.35 + 0.65 * np.abs(normals @ light)
    try:
        rgb = np.asarray(m.visual.to_color().vertex_colors)[:, :3][m.faces].mean(axis=1)
    except Exception:  # some files have broken textures: fall back to grey
        rgb = np.full((len(m.faces), 3), 170.0)
    colors = np.clip(rgb * shade[:, None], 0, 255).astype(int)

    ss = 2  # supersample for smooth edges
    w, h = size[0] * ss, size[1] * ss
    span = max(np.ptp(v[:, 0]) / (w * 0.9), np.ptp(v[:, 1]) / (h * 0.85), 1e-9)
    xy = np.stack([w / 2 + tri[..., 0] / span, h / 2 - tri[..., 1] / span], axis=-1)
    img = Image.new("RGB", (w, h), (38, 38, 42))  # dark like the belt
    draw = ImageDraw.Draw(img)
    for i in np.argsort(tri[..., 2].mean(axis=1)):  # far faces first
        draw.polygon([tuple(pt) for pt in xy[i]], fill=tuple(int(c) for c in colors[i]))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize(size, Image.LANCZOS).save(out)


def preview_path(raw_dir: Path, cls: str, fname: str) -> Path:
    return raw_dir / PREVIEW_DIR / cls / f"{Path(fname).stem}.png"


def ensure_previews(raw_dir: Path, rows: list[dict[str, str]]) -> None:
    import trimesh

    for r in rows:
        fname = r.get("file") or ""
        src, out = raw_dir / r["class"] / fname, preview_path(raw_dir, r["class"], fname)
        if r["status"] not in KEEP_STATUSES or not fname or not src.is_file() or out.exists():
            continue
        try:
            render_preview(trimesh.load(src, force="mesh"), out)
        except Exception as exc:
            print(f"  no preview for {src.name}: {exc}")


def write_review(raw_dir: Path, rows: list[dict[str, str]], target: int) -> Path:
    """Checkbox page: tick the models to keep, download keep.txt, then run --apply-review."""
    cards: dict[str, list[str]] = {cls: [] for cls in config.CLASS_NAMES}
    for r in rows:
        cls, fname = r["class"], r.get("file") or ""
        if r["status"] not in KEEP_STATUSES or cls not in cards or not fname or not (raw_dir / cls / fname).is_file():
            continue
        e = {k: html.escape(str(v or "")) for k, v in r.items()}
        e["license"] = "CC0" if r["license"] == "cc0" else f"CC-{r['license'].upper()}"
        key = html.escape(f"{cls}/{fname}")
        preview = html.escape(preview_path(raw_dir, cls, fname).relative_to(raw_dir).as_posix())
        checked = " checked" if r["status"] == "kept" else ""
        cards[cls].append(
            f"<label class='card'><input type='checkbox' data-key='{key}'{checked}>"
            f"<img class='thumb' src='{e.get('thumb', '')}' loading='lazy' alt='' onerror='deadThumb(this)'>"
            f"<img class='top' src='{preview}' alt='' title='躺平後的俯視形狀（顏色是粗略取樣，模擬裡會有完整貼圖）'>"
            f"<div class='meta'><b>{e['name']}</b><br>{e['author']} · {e['license']} · {e['ratios']}<br>"
            f"<a href='{e['url']}' target='_blank' onclick='event.stopPropagation()'>Sketchfab ↗</a> "
            f"<span class='file' title='{key}'>{e['uid'][:8]}</span></div></label>"
        )
    sections = [
        f"<section><h2>{cls} <span class='count'></span>"
        "<button type='button' class='all'>全選</button><button type='button' class='none'>全不選</button></h2>"
        f"<div class='grid'>{''.join(items) or '<p>（沒有）</p>'}</div></section>"
        for cls, items in cards.items()
    ]
    page = REVIEW_TEMPLATE.replace("__TARGET__", str(target)).replace("__SECTIONS__", "\n".join(sections))
    out = raw_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    return out


def apply_keep_list(raw_dir: Path, rows: list[dict[str, str]], keep: set[str]) -> tuple[int, int]:
    """Mark ticked models "kept"; delete the others and mark them "rejected_review". Returns (kept, removed)."""
    kept = removed = 0
    for r in rows:
        if r["status"] not in KEEP_STATUSES:
            continue
        fname = r.get("file") or ""
        if f"{r['class']}/{fname}" in keep:
            r["status"] = "kept"
            kept += 1
        else:
            (raw_dir / r["class"] / fname).unlink(missing_ok=True)
            preview_path(raw_dir, r["class"], fname).unlink(missing_ok=True)
            remove_size(raw_dir / r["class"], fname)
            r["status"] = "rejected_review"
            removed += 1
    return kept, removed


def restore(raw_dir: Path, rows: list[dict[str, str]], processes: int) -> None:
    """Rebuild raw/<class>/ on another machine from the (git-tracked) selection in objaverse_sources.csv."""
    import objaverse
    import trimesh

    kept = [r for r in rows if r["status"] in KEEP_STATUSES and r.get("file")]
    paths = objaverse.load_objects([r["uid"] for r in kept], download_processes=processes)
    for r in kept:
        dest = raw_dir / r["class"] / r["file"]
        if r["uid"] not in paths:
            print(f"  download failed: {r['uid']} {r['name']}")
            continue
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(paths[r["uid"]], dest)
        ext = trimesh.load(dest, force="mesh").bounding_box_oriented.primitive.extents
        append_size(dest.parent, r["file"], target_length(config.CLASS_SPECS[r["class"]], ext))
    ensure_previews(raw_dir, rows)
    print(f"Restored {len(kept)} models into {raw_dir}. Next: convert_assets.py")


def write_sources(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOURCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--classes", nargs="*", default=list(config.CLASS_NAMES))
    ap.add_argument("--per-class", type=int, default=25, help="accepted models wanted per class (incl. earlier runs)")
    ap.add_argument("--max-faces", type=int, default=50_000)
    ap.add_argument("--allow-nc", action="store_true", help="also accept CC BY-NC licenses (non-commercial)")
    ap.add_argument("--processes", type=int, default=4, help="parallel downloads")
    ap.add_argument("--apply-review", type=Path, metavar="KEEP_TXT", help="keep.txt from review.html: delete the rest")
    ap.add_argument("--review-only", action="store_true", help="only rebuild previews and review.html")
    ap.add_argument("--restore", action="store_true", help="re-download the kept models in objaverse_sources.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # model names are any language; the Windows console is not

    sources_path = args.raw / SOURCES_NAME
    if args.apply_review or args.review_only:
        rows = read_sources(sources_path)
        if args.apply_review:
            keep = {ln.strip() for ln in args.apply_review.read_text(encoding="utf-8").splitlines() if ln.strip()}
            kept, removed = apply_keep_list(args.raw, rows, keep)
            write_sources(sources_path, rows)
            print(f"Kept {kept}, removed {removed}. Run again without arguments to download replacements.")
        ensure_previews(args.raw, rows)
        print(f"Review page: {write_review(args.raw, rows, args.per_class)}")
        return

    import objaverse  # heavy imports only when actually downloading
    import trimesh

    if args.restore:
        restore(args.raw, read_sources(sources_path), args.processes)
        return

    history = read_sources(sources_path)
    seen = {r["uid"] for r in history if r["status"] != "skipped_full"}  # full-class leftovers may be used later
    have = {cls: sum(r["class"] == cls and r["status"] in KEEP_STATUSES for r in history) for cls in config.CLASS_NAMES}

    lvis = objaverse.load_lvis_annotations()
    candidates: dict[str, set[str]] = {}  # uid -> classes it may belong to
    for cls in args.classes:
        for cat in LVIS_CATEGORIES[cls]:
            for uid in lvis.get(cat, []):
                candidates.setdefault(uid, set()).add(cls)
    print(f"Loading annotations for {len(candidates)} LVIS models ...")
    ann = objaverse.load_annotations(sorted(candidates))

    def wanted(uid: str) -> bool:
        a = ann.get(uid, {})
        return (
            uid not in seen
            and license_ok(a.get("license", ""), args.allow_nc)
            and 0 < int(a.get("faceCount") or 0) <= args.max_faces
            and not a.get("isAgeRestricted", False)
        )

    for cls in args.classes:
        queue = sorted(
            (u for u, cs in candidates.items() if cls in cs and wanted(u)),
            key=lambda u: -int(ann[u].get("likeCount") or 0),
        )
        print(f"{cls}: have {have[cls]}, want {args.per_class}, {len(queue)} new candidates")
        while have[cls] < args.per_class and queue:
            n = 2 * (args.per_class - have[cls])  # about half pass the shape check
            batch, queue = [u for u in queue[:n] if u not in seen], queue[n:]
            if not batch:
                continue
            paths = objaverse.load_objects(batch, download_processes=args.processes)
            new_rows = []
            for uid in batch:
                a = ann[uid]
                seen.add(uid)
                row = {
                    "uid": uid,
                    "class": cls,
                    "name": a.get("name", ""),
                    "author": (a.get("user") or {}).get("displayName", ""),
                    "license": a.get("license", ""),
                    "url": a.get("viewerUrl") or f"https://sketchfab.com/3d-models/{uid}",
                    "faces": a.get("faceCount", ""),
                    "thumb": thumbnail_url(a),
                }
                try:
                    if uid not in paths:
                        raise FileNotFoundError("download failed")
                    mesh = trimesh.load(paths[uid], force="mesh")
                    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                        raise ValueError("no triangles")
                    ext = mesh.bounding_box_oriented.primitive.extents
                    row["ratios"] = "{:.2f}/{:.2f}".format(*shape_ratios(ext))
                    fit = classify_shape(ext, candidates[uid])
                    if fit is None:
                        row["status"] = "rejected_shape"
                    elif have[fit] >= args.per_class:
                        row.update({"class": fit, "status": "skipped_full"})
                    else:
                        fname = f"objaverse_{uid}.glb"
                        dest = args.raw / fit / fname
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(paths[uid], dest)
                        append_size(dest.parent, fname, target_length(config.CLASS_SPECS[fit], ext))
                        row.update({"class": fit, "status": "accepted", "file": fname})
                        have[fit] += 1
                except Exception as exc:  # one broken model must not stop the batch
                    row.update(status="failed", ratios=str(exc)[:80])
                print(f"  [{row['status']:>14}] {row['class']:<13} {uid} {row.get('ratios', '')} {row['name'][:40]}")
                new_rows.append(row)
            append_rows(sources_path, SOURCE_FIELDS, new_rows)

    rows = read_sources(sources_path)
    ensure_previews(args.raw, rows)
    review = write_review(args.raw, rows, args.per_class)
    print("\nAccepted per class:", {c: have[c] for c in config.CLASS_NAMES})
    print(f"Review the models: {review}")
    print(f"Attribution list (CC-BY needs it): {sources_path}")


if __name__ == "__main__":
    main()
