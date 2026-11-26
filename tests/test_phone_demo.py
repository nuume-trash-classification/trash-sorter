import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from perception.phone_demo import MAX_SIDE, MAX_UPLOAD_BYTES, analyze, find_tunnel_url, make_handler, summarize


class _Tensor:
    """Stands in for a torch tensor: ``.cpu().numpy()``."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Part:
    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, _Tensor(value))


class _Result:
    def __init__(self, masks=None, classes=(), confs=()):
        self.masks = None if masks is None else _Part(data=masks)
        self.boxes = _Part(cls=np.array(classes, dtype=float), conf=np.array(confs, dtype=float))


def _one_can(frame: np.ndarray) -> _Result:
    mask = np.zeros(frame.shape[:2], dtype=float)
    mask[20:60, 30:90] = 1.0
    return _Result(masks=mask[None], classes=[1], confs=[0.87])


def _jpeg(width: int = 120, height: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


def test_summarize_lists_detections_by_confidence():
    result = _Result(masks=np.zeros((2, 4, 4)), classes=[3, 0], confs=[0.41, 0.93])
    assert summarize(result) == [
        {"class": "pet_bottle", "confidence": 0.93},
        {"class": "glass_bottle", "confidence": 0.41},
    ]


def test_summarize_of_an_empty_frame_is_empty():
    assert summarize(_Result()) == []


def test_analyze_returns_a_drawn_image_of_the_same_size_and_the_detections():
    annotated, detections = analyze(_jpeg(), _one_can)
    image = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (80, 120, 3)
    assert image[40, 60].tolist() != [90, 90, 90]  # the mask is painted over the object
    assert detections == [{"class": "aluminum_can", "confidence": 0.87}]


def test_analyze_shrinks_a_large_photo_and_keeps_its_shape():
    annotated, _ = analyze(_jpeg(width=MAX_SIDE * 2, height=MAX_SIDE), _one_can)
    image = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (MAX_SIDE // 2, MAX_SIDE, 3)


def test_analyze_rejects_bytes_that_are_not_an_image():
    with pytest.raises(ValueError, match="image"):
        analyze(b"not a picture", _one_can)


@pytest.fixture
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_one_can))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url: str, body: bytes):
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "image/jpeg"})
    return urllib.request.urlopen(request, timeout=10)


def test_page_is_served_with_a_camera_button(server_url):
    with urllib.request.urlopen(server_url + "/", timeout=10) as response:
        page = response.read().decode("utf-8")
    assert response.headers["Content-Type"].startswith("text/html")
    assert 'type="file"' in page and "capture" in page


def test_posting_a_photo_returns_the_annotated_image_and_detections(server_url):
    with _post(server_url + "/predict", _jpeg()) as response:
        reply = json.loads(response.read())
    assert reply["detections"] == [{"class": "aluminum_can", "confidence": 0.87}]
    assert reply["milliseconds"] >= 0
    prefix = "data:image/jpeg;base64,"
    assert reply["image"].startswith(prefix)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(reply["image"][len(prefix) :]), np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (80, 120, 3)


def test_posting_garbage_is_a_400_with_a_message(server_url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post(server_url + "/predict", b"not a picture")
    assert caught.value.code == 400
    assert "image" in json.loads(caught.value.read())["error"]


def test_an_oversized_upload_is_refused(server_url):
    request = urllib.request.Request(
        server_url + "/predict", data=b"x", method="POST", headers={"Content-Length": str(MAX_UPLOAD_BYTES + 1)}
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 413


def test_unknown_paths_are_404(server_url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(server_url + "/nope", timeout=10)
    assert caught.value.code == 404


@pytest.fixture
def locked_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_one_can, access_key="s3cret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_with_an_access_key_the_page_needs_it_in_the_address(locked_url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(locked_url + "/", timeout=10)
    assert caught.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(locked_url + "/?key=wrong", timeout=10)
    assert caught.value.code == 403
    with urllib.request.urlopen(locked_url + "/?key=s3cret", timeout=10) as response:
        assert response.status == 200


def test_with_an_access_key_a_photo_without_it_is_refused(locked_url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post(locked_url + "/predict", _jpeg())
    assert caught.value.code == 403
    with _post(locked_url + "/predict?key=s3cret", _jpeg()) as response:
        assert json.loads(response.read())["detections"]


def test_tunnel_address_is_found_in_cloudflared_output():
    line = "2026-11-26T10:00:00Z INF |  https://quiet-river-12ab.trycloudflare.com          |"
    assert find_tunnel_url(line) == "https://quiet-river-12ab.trycloudflare.com"
    assert find_tunnel_url("INF Requesting new quick Tunnel on trycloudflare.com...") is None
    assert find_tunnel_url("see https://developers.cloudflare.com/cloudflare-one/") is None
