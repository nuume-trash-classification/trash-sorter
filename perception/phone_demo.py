"""Phone demo: take a photo in the phone's browser, the PC runs YOLO-seg and sends the result back.

    uv run python perception/phone_demo.py --weights runs/segment/sim_plus_real/weights/best.pt

Put the phone on the same Wi-Fi as the PC (or let the PC join the phone's hotspot) and open one
of the addresses the script prints. The page uploads one photo at a time; nothing is installed
on the phone. Stop it with Ctrl+C.

    uv run python perception/phone_demo.py --weights best.pt --public

``--public`` also opens a Cloudflare quick tunnel (needs ``cloudflared`` on the PATH:
``winget install Cloudflare.cloudflared``), so a phone on any network can reach the PC through
an https address. The address changes on every start and the photos pass through Cloudflare.
A public server always asks for an access key, which is part of the address the script prints;
on the local network there is no key unless ``--key`` is given.

Every photo is kept under ``data/real/feedback/<date>/`` with the model's prediction as a YOLO-seg
label file, and the page asks whether the result is correct. The photo then moves to ``correct/``
or ``wrong/`` (``unreviewed/`` until someone answers). Fix the labels of the wrong ones and add
them to the fine-tuning set; never to the test set, because they were picked for being hard.
``--no-save`` turns this off.

A live camera stream would need HTTPS (browsers only give a page the camera over a secure
connection), so this takes still photos through the phone's own camera app instead.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hmac
import json
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from perception.live_demo import draw
from trash_sorter import config
from trash_sorter.yolo_format import polygon_to_label_line, write_data_yaml

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIDE = 1280  # photos are shrunk to this before drawing, so labels stay readable on a phone
Predict = Callable[[np.ndarray], Any]  # BGR image -> one ultralytics result
DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[1] / "data" / "real" / "feedback"
VERDICTS = ("correct", "wrong", "unreviewed")
_PHOTO_ID = re.compile(r"\d{8}_\d{6}_[0-9a-f]{4}")
_PARTS = (("images", ".jpg"), ("labels", ".txt"), ("meta", ".json"))

PAGE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>垃圾分類辨識</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #111; color: #eee; }
  h1 { font-size: 1.2rem; margin: 0 0 12px; }
  label.button { display: block; padding: 18px; border-radius: 12px; background: #2d7ff9; color: #fff;
                 text-align: center; font-size: 1.1rem; }
  input[type=file] { display: none; }
  #status { margin: 12px 0; min-height: 1.4em; color: #bbb; }
  #status.error { color: #ff8a80; }
  img { width: 100%; border-radius: 8px; display: none; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  td { padding: 8px 4px; border-bottom: 1px solid #333; }
  td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  #verdict { display: none; gap: 12px; margin-top: 16px; }
  #verdict button { flex: 1; padding: 18px; border: 0; border-radius: 12px; color: #fff; font-size: 1.1rem;
                    opacity: 0.55; }
  #verdict button.chosen { opacity: 1; outline: 3px solid #fff; }
  #correct { background: #2e9d57; }
  #wrong { background: #d64545; }
  #verdictStatus { margin: 10px 0; min-height: 1.4em; color: #bbb; }
</style>
</head>
<body>
<h1>垃圾分類辨識</h1>
<label class="button">拍照辨識
  <input id="photo" type="file" accept="image/*" capture="environment">
</label>
<div id="status">從正上方拍，物體盡量不要重疊。</div>
<img id="result" alt="辨識結果">
<table id="list"></table>
<div id="verdict">
  <button id="correct" type="button">✔ 正確</button>
  <button id="wrong" type="button">✘ 錯誤</button>
</div>
<div id="verdictStatus"></div>
<script>
const NAMES = {pet_bottle: "寶特瓶", aluminum_can: "鋁罐", paper_carton: "紙盒", glass_bottle: "玻璃瓶"};
const MAX_SIDE = 1600;  // shrink before upload: a 12 MP photo is slow to send and the model sees 640 px
const statusBox = document.getElementById("status");
const say = (text, error) => { statusBox.textContent = text; statusBox.className = error ? "error" : ""; };

async function shrink(file) {
  const bitmap = await createImageBitmap(file, {imageOrientation: "from-image"});
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

const verdictBox = document.getElementById("verdict");
const verdictStatus = document.getElementById("verdictStatus");
const VERDICT_TEXT = {correct: "正確", wrong: "錯誤"};
let photoId = null;  // the photo on screen, if the PC saved it

function showVerdict(chosen, text) {
  verdictBox.style.display = photoId ? "flex" : "none";
  for (const button of verdictBox.querySelectorAll("button")) button.className = button.id === chosen ? "chosen" : "";
  verdictStatus.textContent = text;
}

for (const button of verdictBox.querySelectorAll("button")) {
  button.addEventListener("click", async () => {
    const id = photoId, verdict = button.id;
    try {
      const response = await fetch("/feedback" + location.search, {method: "POST",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({id, verdict})});
      const reply = await response.json();
      if (!response.ok) throw new Error(reply.error);
      if (id === photoId) showVerdict(verdict, `已記錄：${VERDICT_TEXT[verdict]}。按另一個可以改。`);
    } catch (error) {
      if (id === photoId) showVerdict(null, "沒有記錄到：" + error.message);
    }
  });
}

document.getElementById("photo").addEventListener("change", async event => {
  const file = event.target.files[0];
  event.target.value = "";  // the same photo can be picked again
  if (!file) return;
  say("辨識中…");
  photoId = null;
  showVerdict(null, "");
  try {
    const response = await fetch("/predict" + location.search, {method: "POST", headers: {"Content-Type": "image/jpeg"},
                                              body: await shrink(file)});
    const reply = await response.json();
    if (!response.ok) throw new Error(reply.error);
    const image = document.getElementById("result");
    image.src = reply.image;
    image.style.display = "block";
    document.getElementById("list").innerHTML = reply.detections.map(d =>
      `<tr><td>${NAMES[d.class] || d.class}</td><td>${Math.round(d.confidence * 100)}%</td></tr>`).join("");
    say(`找到 ${reply.detections.length} 個物體，電腦運算 ${reply.milliseconds} 毫秒。紅點是吸盤的取物點。`);
    photoId = reply.id || null;
    showVerdict(null, photoId ? "每個物體的輪廓和類別都對嗎？漏掉或多抓都算錯誤。" : "");
  } catch (error) {
    say("失敗：" + error.message, true);
  }
});
</script>
</body>
</html>
"""


def summarize(result: Any) -> list[dict[str, Any]]:
    """Detections of one result as ``{"class", "confidence"}``, most confident first."""
    if result.masks is None:
        return []
    classes, confs = result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()
    found = [
        {"class": config.CLASS_NAMES[int(c)], "confidence": round(float(p), 2)}
        for c, p in zip(classes, confs, strict=True)
    ]
    return sorted(found, key=lambda d: -d["confidence"])


def label_lines(result: Any) -> list[str]:
    """The prediction as YOLO-seg label lines, ready to be corrected in a labeling tool.

    Uses ``masks.xyn`` (polygons already mapped back to the photo). ``masks.data`` has the padded
    shape of the model input, so stretching it over the photo would shift the outlines.
    """
    if result.masks is None:
        return []
    classes = result.boxes.cls.cpu().numpy()
    lines = (polygon_to_label_line(int(c), poly) for c, poly in zip(classes, result.masks.xyn, strict=True))
    return [line for line in lines if line is not None]


def analyze(image_bytes: bytes, predict: Predict) -> tuple[bytes, list[dict[str, Any]], list[str]]:
    """Run the model on an encoded photo. Returns (annotated JPEG, detections, YOLO-seg label lines)."""
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("the upload is not an image this server can read")
    scale = MAX_SIDE / max(frame.shape[:2])
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = predict(frame)
    ok, encoded = cv2.imencode(".jpg", draw(frame, result), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("could not encode the annotated image")
    return encoded.tobytes(), summarize(result), label_lines(result)


def as_jpeg(image_bytes: bytes) -> bytes:
    """The photo as JPEG bytes: untouched if it already is one, re-encoded otherwise."""
    if image_bytes[:2] == b"\xff\xd8":
        return image_bytes
    ok, encoded = cv2.imencode(".jpg", cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR))
    if not ok:
        raise RuntimeError("could not encode the photo")
    return encoded.tobytes()


class FeedbackStore:
    """Keeps every photo with its predicted labels, sorted by what the user said about the result.

    Layout: ``<root>/<YYYY-MM-DD>/<verdict>/{images,labels,meta}/<id>.{jpg,txt,json}`` plus a
    ``data.yaml`` naming the classes, where verdict is ``unreviewed``, ``correct`` or ``wrong``.
    """

    def __init__(self, root: str | Path, info: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.info = dict(info or {})  # copied into every record: weights, conf, ...
        self._lock = threading.Lock()
        self._session: dict[str, str] = {}  # photo id -> verdict, for photos saved since start

    def _day(self, photo_id: str) -> Path:
        return self.root / f"{photo_id[:4]}-{photo_id[4:6]}-{photo_id[6:8]}"

    def _prepare(self, folder: Path) -> Path:
        if not (folder / "data.yaml").exists():
            for part, _ in _PARTS:
                (folder / part).mkdir(parents=True, exist_ok=True)
            write_data_yaml(folder, {"train": "images", "val": "images"})
        return folder

    @staticmethod
    def _paths(folder: Path, photo_id: str) -> list[Path]:
        return [folder / part / f"{photo_id}{suffix}" for part, suffix in _PARTS]

    def save(self, photo: bytes, labels: list[str], detections: list[dict[str, Any]]) -> str:
        """Store a JPEG photo and its prediction as unreviewed. Returns the photo id."""
        with self._lock:
            now = datetime.now()
            photo_id = f"{now:%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"
            while photo_id in self._session:
                photo_id = f"{now:%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"
            folder = self._prepare(self._day(photo_id) / "unreviewed")
            image, label, meta = self._paths(folder, photo_id)
            image.write_bytes(photo)
            label.write_text("".join(line + "\n" for line in labels), encoding="utf-8")
            record = {
                "id": photo_id,
                "time": now.isoformat(timespec="seconds"),
                **self.info,
                "detections": detections,
                "verdict": "unreviewed",
            }
            meta.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            self._session[photo_id] = "unreviewed"
            return photo_id

    def judge(self, photo_id: str, verdict: str) -> None:
        """Record the user's verdict and move the photo's files to that folder.

        Raises ValueError for a malformed id or verdict, KeyError if no such photo is stored.
        """
        if verdict not in ("correct", "wrong"):
            raise ValueError("verdict must be 'correct' or 'wrong'")
        if not isinstance(photo_id, str) or not _PHOTO_ID.fullmatch(photo_id):
            raise ValueError("malformed photo id")
        with self._lock:
            for current in VERDICTS:
                old = self._paths(self._day(photo_id) / current, photo_id)
                if old[2].exists():
                    break
            else:
                raise KeyError(photo_id)
            new = self._paths(self._prepare(self._day(photo_id) / verdict), photo_id)
            record = json.loads(old[2].read_text(encoding="utf-8"))
            record["verdict"] = verdict
            record["judged"] = datetime.now().isoformat(timespec="seconds")
            if current != verdict:
                for src, dst in zip(old[:2], new[:2], strict=True):
                    src.replace(dst)
                old[2].unlink()
            new[2].write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if photo_id in self._session:
                self._session[photo_id] = verdict

    def counts(self) -> dict[str, int]:
        """How many photos saved since start carry each verdict."""
        with self._lock:
            held = list(self._session.values())
        return {verdict: held.count(verdict) for verdict in VERDICTS}


def make_handler(
    predict: Predict, access_key: str | None = None, store: FeedbackStore | None = None
) -> type[BaseHTTPRequestHandler]:
    """Request handler serving the page at ``/`` and the model at ``POST /predict``.

    With ``access_key``, every request must carry it as ``?key=...`` or gets a 403.
    With ``store``, every photo is saved, the reply carries its ``id``, and ``POST /feedback``
    (JSON ``{"id", "verdict"}``) records whether the user found the result correct.
    """
    lock = threading.Lock()  # one inference at a time: the model is not thread-safe

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def _allowed(self) -> bool:
            if access_key is None:
                return True
            given = parse_qs(urlsplit(self.path).query).get("key", [""])[0]
            if hmac.compare_digest(given.encode("utf-8"), access_key.encode("utf-8")):
                return True
            self.close_connection = True  # a refused upload is left unread
            self._json(403, {"error": "wrong or missing access key: open the full address, including ?key=..."})
            return False

        def do_GET(self) -> None:
            if not self._allowed():
                return
            if self.path.split("?")[0] == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._allowed():
                return
            route = self.path.split("?")[0]
            if route not in ("/predict", "/feedback") or (route == "/feedback" and store is None):
                self.close_connection = True  # the body is left unread
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES:
                self.close_connection = True  # the body is left unread
                self._json(413, {"error": f"photo larger than {MAX_UPLOAD_BYTES // 2**20} MB"})
                return
            body = self.rfile.read(length)
            if route == "/feedback":
                self._feedback(body)
                return
            start = time.perf_counter()
            try:
                with lock:
                    annotated, detections, labels = analyze(body, predict)
            except ValueError as error:
                self._json(400, {"error": str(error)})
                return
            reply = {
                "image": "data:image/jpeg;base64," + base64.b64encode(annotated).decode("ascii"),
                "detections": detections,
                "milliseconds": round((time.perf_counter() - start) * 1000),
            }
            if store is not None:
                reply["id"] = store.save(as_jpeg(body), labels, detections)
            self._json(200, reply)

        def _feedback(self, body: bytes) -> None:
            try:
                payload = json.loads(body)
                photo_id, verdict = payload["id"], payload["verdict"]
            except (ValueError, TypeError, KeyError):
                self._json(400, {"error": "feedback must be JSON with 'id' and 'verdict'"})
                return
            try:
                store.judge(photo_id, verdict)
            except ValueError as error:
                self._json(400, {"error": str(error)})
            except KeyError:
                self._json(404, {"error": "no saved photo with this id"})
            else:
                self._json(200, {"id": photo_id, "verdict": verdict})

        def log_message(self, format: str, *args: Any) -> None:
            if self.command == "POST":
                super().log_message(format, *args)

    return Handler


def lan_addresses() -> list[str]:
    """IPv4 addresses of this PC that a phone on the same network can reach."""
    found = set()
    with contextlib.suppress(OSError):
        found.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    # The address of the default route. UDP connect sends nothing.
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))
        found.add(probe.getsockname()[0])
    return sorted(a for a in found if not a.startswith("127."))


def find_tunnel_url(line: str) -> str | None:
    """The public address in a line of ``cloudflared`` output, if the line announces one."""
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    return match.group(0) if match else None


def open_tunnel(port: int, timeout: float = 40.0) -> tuple[subprocess.Popen, str]:
    """Start a Cloudflare quick tunnel to this PC's ``port``. Returns (process, public address)."""
    exe = shutil.which("cloudflared")
    if exe is None:
        sys.exit("--public needs cloudflared. Install it with:  winget install Cloudflare.cloudflared")
    command = [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    found: list[str] = []

    def watch() -> None:  # keeps draining the output so cloudflared never blocks on a full pipe
        for line in process.stderr:
            if not found and (url := find_tunnel_url(line)):
                found.append(url)

    threading.Thread(target=watch, daemon=True).start()
    deadline = time.monotonic() + timeout
    while not found and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if not found:
        process.terminate()
        sys.exit("cloudflared did not report a public address (no internet, or the service is busy). Try again.")
    return process, found[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="127.0.0.1 = this PC only")
    ap.add_argument("--public", action="store_true", help="reachable from any network (Cloudflare quick tunnel)")
    ap.add_argument("--key", default=None, help="access key required in the address (random with --public)")
    ap.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="where photos and predictions are kept")
    ap.add_argument("--no-save", action="store_true", help="keep nothing and hide the correct / wrong buttons")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    def predict(frame: np.ndarray) -> Any:
        return model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

    predict(np.zeros((480, 640, 3), np.uint8))  # load the model onto the GPU before the first photo
    key = args.key or (secrets.token_urlsafe(6) if args.public else None)
    suffix = f"?key={key}" if key else ""
    info = {"weights": str(args.weights), "conf": args.conf, "imgsz": args.imgsz}
    store = None if args.no_save else FeedbackStore(args.save_dir, info)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(predict, key, store))
    tunnel = None
    if args.public:
        tunnel, public_url = open_tunnel(args.port)
        print(f"From any network (address changes on every start):\n  {public_url}/{suffix}")
    print("On the same Wi-Fi as this PC:")
    for address in lan_addresses() or ["<this PC's IP address>"]:
        print(f"  http://{address}:{args.port}/{suffix}")
    if store is not None:
        print(f"Photos and predictions are kept in {store.root} (fine-tuning only, never the test set).")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        if tunnel is not None:
            tunnel.terminate()
        if store is not None:
            n = store.counts()
            print(f"This run: {n['correct']} correct, {n['wrong']} wrong, {n['unreviewed']} not judged.")


if __name__ == "__main__":
    main()
