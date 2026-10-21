import numpy as np

from trash_sorter import config
from trash_sorter.geometry import default_camera
from trash_sorter.metrics import END_BIN, SortingStats
from trash_sorter.perception import masks_to_detections
from trash_sorter.yolo_format import (
    draw_overlay,
    mask_to_label_line,
    parse_label_file,
    polygon_to_mask,
    write_data_yaml,
)


def test_label_round_trip(tmp_path):
    mask = np.zeros((480, 640), bool)
    mask[100:200, 300:420] = True
    line = mask_to_label_line(2, mask)
    assert line.startswith("2 ")
    f = tmp_path / "a.txt"
    f.write_text(line + "\n")
    [(cls, poly)] = parse_label_file(f)
    assert cls == 2
    back = polygon_to_mask(poly, 640, 480)
    iou = (back & mask).sum() / (back | mask).sum()
    assert iou > 0.95
    assert draw_overlay(np.zeros((480, 640, 3), np.uint8), [(cls, poly)]).shape == (480, 640, 3)


def test_tiny_mask_gives_no_label():
    mask = np.zeros((50, 50), bool)
    mask[1:3, 1:3] = True
    assert mask_to_label_line(0, mask) is None


def test_data_yaml(tmp_path):
    p = write_data_yaml(tmp_path)
    assert "glass_bottle" in p.read_text()


def test_masks_to_detections_skips_border_and_finds_center():
    cam = default_camera()
    masks = np.zeros((2, 480, 640), bool)
    masks[0, 220:260, 300:340] = True  # centered object
    masks[1, 200:240, 0:40] = True  # cut off by the left border
    dets = masks_to_detections(masks, np.array([1, 0]), np.array([0.9, 0.8]), cam, t=1.0)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "aluminum_can"
    np.testing.assert_allclose(d.position[:2], [config.CAMERA_X, 0.0], atol=0.01)


def test_metrics_summary():
    s = SortingStats()
    s.record_attempt(True)
    s.record_attempt(False)
    s.record_outcome("pet_bottle", "pet_bottle")
    s.record_outcome("glass_bottle", "pet_bottle")
    s.record_outcome("aluminum_can", END_BIN)
    out = s.summary()
    assert out["pick_success_rate"] == 0.5
    assert out["miss_rate"] == 1 / 3
    assert out["sorting_accuracy"] == 0.5
    assert "pick success rate" in s.format()


def test_detections_off_the_belt_are_dropped():
    cam = default_camera()
    masks = np.zeros((1, 480, 640), bool)
    masks[5:40, 300:340] = True  # top of the image = far side, beyond the belt edge
    assert masks_to_detections(masks, np.array([0]), np.array([0.9]), cam, t=0.0) == []
