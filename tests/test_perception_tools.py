import yaml

from perception.common import resolve_data
from perception.prepare_real import find_label, read_export_names


def _touch(p, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_find_label_in_mirrored_labels_folder(tmp_path):
    img = _touch(tmp_path / "export" / "images" / "train" / "a.jpg")
    lbl = _touch(tmp_path / "export" / "labels" / "train" / "a.txt", "0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    assert find_label(img) == lbl
    assert find_label(_touch(tmp_path / "export" / "images" / "b.jpg")) is None


def test_read_export_names_from_cvat_obj_names(tmp_path):
    _touch(tmp_path / "obj.names", "glass_bottle\npet_bottle\n")
    assert read_export_names(tmp_path) == ["glass_bottle", "pet_bottle"]


def test_resolve_data_for_flat_test_folder(tmp_path):
    (tmp_path / "images").mkdir()
    data = yaml.safe_load(resolve_data(tmp_path).read_text())
    assert data["val"] == "images"
    assert data["names"][3] == "glass_bottle"
