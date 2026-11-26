import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from perception.phone_demo import (
    MAX_SIDE,
    MAX_UPLOAD_BYTES,
    FeedbackStore,
    analyze,
    find_tunnel_url,
    label_lines,
    make_handler,
    summarize,
)
from trash_sorter.yolo_format import parse_label_file


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
    def __init__(self, masks=None, classes=(), confs=(), polygons=()):
        self.masks = None if masks is None else _Part(data=masks)
        if self.masks is not None:
            self.masks.xyn = [np.asarray(p, dtype=float) for p in polygons]
        self.boxes = _Part(cls=np.array(classes, dtype=float), conf=np.array(confs, dtype=float))


_CAN_POLYGON = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def _one_can(frame: np.ndarray) -> _Result:
    mask = np.zeros(frame.shape[:2], dtype=float)
    mask[20:60, 30:90] = 1.0
    return _Result(masks=mask[None], classes=[1], confs=[0.87], polygons=[_CAN_POLYGON])


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
    annotated, detections, _ = analyze(_jpeg(), _one_can)
    image = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (80, 120, 3)
    assert image[40, 60].tolist() != [90, 90, 90]  # the mask is painted over the object
    assert detections == [{"class": "aluminum_can", "confidence": 0.87}]


def test_analyze_shrinks_a_large_photo_and_keeps_its_shape():
    annotated, _, _ = analyze(_jpeg(width=MAX_SIDE * 2, height=MAX_SIDE), _one_can)
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


def test_label_lines_are_yolo_seg_lines_from_the_normalized_polygons():
    result = _Result(
        masks=np.zeros((2, 4, 4)),
        classes=[3, 0],
        confs=[0.5, 0.9],
        polygons=[_CAN_POLYGON, [[0.1, 0.1], [0.2, 0.1]]],  # the second one is not a polygon
    )
    assert label_lines(result) == ["3 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000 0.250000 0.750000"]


def test_label_lines_of_an_empty_frame_are_empty():
    assert label_lines(_Result()) == []


def test_analyze_also_returns_the_label_lines():
    _, _, labels = analyze(_jpeg(), _one_can)
    assert len(labels) == 1 and labels[0].startswith("1 0.25")


def _save_one(store: FeedbackStore) -> str:
    return store.save(_jpeg(), ["1 0.25 0.25 0.75 0.25 0.75 0.75"], [{"class": "aluminum_can", "confidence": 0.87}])


def _files(folder, photo_id):
    return [
        folder / "images" / f"{photo_id}.jpg",
        folder / "labels" / f"{photo_id}.txt",
        folder / "meta" / f"{photo_id}.json",
    ]


def test_a_saved_photo_waits_unreviewed_with_its_labels_and_record(tmp_path):
    store = FeedbackStore(tmp_path, {"weights": "best.pt", "conf": 0.4})
    photo_id = _save_one(store)
    (day,) = tmp_path.iterdir()
    image, label, meta = _files(day / "unreviewed", photo_id)
    assert image.read_bytes() == _jpeg()
    assert parse_label_file(label)[0][0] == 1
    record = json.loads(meta.read_text(encoding="utf-8"))
    assert record["verdict"] == "unreviewed" and record["weights"] == "best.pt"
    assert record["detections"] == [{"class": "aluminum_can", "confidence": 0.87}]
    assert "aluminum_can" in (day / "unreviewed" / "data.yaml").read_text(encoding="utf-8")


def test_a_verdict_moves_the_photo_and_can_be_changed(tmp_path):
    store = FeedbackStore(tmp_path)
    photo_id = _save_one(store)
    (day,) = tmp_path.iterdir()
    store.judge(photo_id, "wrong")
    assert all(f.exists() for f in _files(day / "wrong", photo_id))
    assert not any(f.exists() for f in _files(day / "unreviewed", photo_id))
    assert json.loads(_files(day / "wrong", photo_id)[2].read_text(encoding="utf-8"))["verdict"] == "wrong"
    store.judge(photo_id, "correct")
    assert all(f.exists() for f in _files(day / "correct", photo_id))
    assert not any(f.exists() for f in _files(day / "wrong", photo_id))
    assert store.counts() == {"correct": 1, "wrong": 0, "unreviewed": 0}


def test_two_photos_in_the_same_second_get_different_ids(tmp_path):
    store = FeedbackStore(tmp_path)
    assert _save_one(store) != _save_one(store)
    assert store.counts() == {"correct": 0, "wrong": 0, "unreviewed": 2}


def test_judging_rejects_unknown_ids_paths_and_verdicts(tmp_path):
    store = FeedbackStore(tmp_path)
    photo_id = _save_one(store)
    with pytest.raises(ValueError, match="verdict"):
        store.judge(photo_id, "maybe")
    with pytest.raises(ValueError, match="id"):
        store.judge("../../secrets", "wrong")
    with pytest.raises(KeyError):
        store.judge("20200101_000000_abcd", "wrong")


@pytest.fixture
def saving_url(tmp_path):
    store = FeedbackStore(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_one_can, store=store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", store, tmp_path
    server.shutdown()
    server.server_close()


def _post_json(url: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=10)


def test_the_page_has_correct_and_wrong_buttons(server_url):
    with urllib.request.urlopen(server_url + "/", timeout=10) as response:
        page = response.read().decode("utf-8")
    assert "正確" in page and "錯誤" in page and "/feedback" in page


def test_a_photo_is_saved_and_its_verdict_recorded_over_http(saving_url):
    url, store, root = saving_url
    with _post(url + "/predict", _jpeg()) as response:
        photo_id = json.loads(response.read())["id"]
    assert store.counts()["unreviewed"] == 1
    with _post_json(url + "/feedback", {"id": photo_id, "verdict": "wrong"}) as response:
        assert json.loads(response.read()) == {"id": photo_id, "verdict": "wrong"}
    assert store.counts() == {"correct": 0, "wrong": 1, "unreviewed": 0}
    assert len(list(root.glob(f"*/wrong/images/{photo_id}.jpg"))) == 1


def test_bad_feedback_is_refused_with_a_message(saving_url):
    url, _, _ = saving_url
    for payload, code in [
        ({"id": "20200101_000000_abcd", "verdict": "wrong"}, 404),
        ({"id": "../x", "verdict": "wrong"}, 400),
        ({"verdict": "wrong"}, 400),
    ]:
        with pytest.raises(urllib.error.HTTPError) as caught:
            _post_json(url + "/feedback", payload)
        assert caught.value.code == code
        assert json.loads(caught.value.read())["error"]
    request = urllib.request.Request(url + "/feedback", data=b"not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 400


def test_without_a_store_nothing_is_saved_and_feedback_is_404(server_url):
    with _post(server_url + "/predict", _jpeg()) as response:
        assert "id" not in json.loads(response.read())
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post_json(server_url + "/feedback", {"id": "20200101_000000_abcd", "verdict": "wrong"})
    assert caught.value.code == 404


def test_feedback_needs_the_access_key(locked_url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post_json(locked_url + "/feedback", {"id": "20200101_000000_abcd", "verdict": "wrong"})
    assert caught.value.code == 403
