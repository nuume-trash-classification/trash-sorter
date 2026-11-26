"""Phone demo: take a photo in the phone's browser, the PC runs YOLO-seg and sends the result back.

    uv run python perception/phone_demo.py --weights runs/segment/sim_plus_real/weights/best.pt

Put the phone on the same Wi-Fi as the PC (or let the PC join the phone's hotspot) and open one
of the addresses the script prints. The page uploads one photo at a time; nothing is installed
on the phone. There is no login: run it only on a network you trust, and stop it with Ctrl+C.

A live camera stream would need HTTPS (browsers only give a page the camera over a secure
connection), so this takes still photos through the phone's own camera app instead.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import socket
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from perception.live_demo import draw
from trash_sorter import config

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIDE = 1280  # photos are shrunk to this before drawing, so labels stay readable on a phone
Predict = Callable[[np.ndarray], Any]  # BGR image -> one ultralytics result

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

document.getElementById("photo").addEventListener("change", async event => {
  const file = event.target.files[0];
  event.target.value = "";  // the same photo can be picked again
  if (!file) return;
  say("辨識中…");
  try {
    const response = await fetch("/predict", {method: "POST", headers: {"Content-Type": "image/jpeg"},
                                              body: await shrink(file)});
    const reply = await response.json();
    if (!response.ok) throw new Error(reply.error);
    const image = document.getElementById("result");
    image.src = reply.image;
    image.style.display = "block";
    document.getElementById("list").innerHTML = reply.detections.map(d =>
      `<tr><td>${NAMES[d.class] || d.class}</td><td>${Math.round(d.confidence * 100)}%</td></tr>`).join("");
    say(`找到 ${reply.detections.length} 個物體，電腦運算 ${reply.milliseconds} 毫秒。紅點是吸盤的取物點。`);
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


def analyze(image_bytes: bytes, predict: Predict) -> tuple[bytes, list[dict[str, Any]]]:
    """Run the model on an encoded photo. Returns (annotated JPEG, detections)."""
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
    return encoded.tobytes(), summarize(result)


def make_handler(predict: Predict) -> type[BaseHTTPRequestHandler]:
    """Request handler serving the page at ``/`` and the model at ``POST /predict``."""
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

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/predict":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES:
                self.close_connection = True  # the body is left unread
                self._json(413, {"error": f"photo larger than {MAX_UPLOAD_BYTES // 2**20} MB"})
                return
            body = self.rfile.read(length)
            start = time.perf_counter()
            try:
                with lock:
                    annotated, detections = analyze(body, predict)
            except ValueError as error:
                self._json(400, {"error": str(error)})
                return
            self._json(
                200,
                {
                    "image": "data:image/jpeg;base64," + base64.b64encode(annotated).decode("ascii"),
                    "detections": detections,
                    "milliseconds": round((time.perf_counter() - start) * 1000),
                },
            )

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="127.0.0.1 = this PC only")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    def predict(frame: np.ndarray) -> Any:
        return model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

    predict(np.zeros((480, 640, 3), np.uint8))  # load the model onto the GPU before the first photo
    server = ThreadingHTTPServer((args.host, args.port), make_handler(predict))
    print("Open one of these on the phone (same Wi-Fi as this PC):")
    for address in lan_addresses() or ["<this PC's IP address>"]:
        print(f"  http://{address}:{args.port}/")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
