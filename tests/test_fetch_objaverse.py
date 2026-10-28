from sim.assets.fetch_objaverse import (
    append_size,
    classify_shape,
    license_ok,
    shape_ratios,
    target_length,
)
from trash_sorter import config


def test_shape_ratios_ignore_order_and_scale():
    assert shape_ratios((0.066, 0.122, 0.066)) == shape_ratios((1.22, 0.66, 0.66))
    assert shape_ratios((1.0, 1.0, 0.0)) == (float("inf"), float("inf"))


def test_classify_soda_bottle_vs_can():
    both = {"pet_bottle", "aluminum_can"}
    assert classify_shape((0.22, 0.068, 0.068), both) == "pet_bottle"
    assert classify_shape((0.122, 0.066, 0.066), both) == "aluminum_can"


def test_classify_carton_and_rejects():
    assert classify_shape((0.20, 0.095, 0.065), {"paper_carton"}) == "paper_carton"
    # Flat card / ground plane, and a cube (e.g. a six-pack box) are not bottles.
    assert classify_shape((0.30, 0.20, 0.002), {"pet_bottle"}) is None
    assert classify_shape((0.30, 0.28, 0.25), {"pet_bottle"}) is None


def test_license_filter():
    assert license_ok("by") and license_ok("cc0") and license_ok("by-sa")
    assert not license_ok("by-nc")
    assert license_ok("by-nc", allow_nc=True)
    assert not license_ok("by-nd", allow_nc=True)


def test_target_length_gives_class_thickness_within_range():
    spec = config.CLASS_SPECS["glass_bottle"]
    assert target_length(spec, (3.4, 1.0, 1.0)) == round(0.070 * 3.4, 3)  # 0.238 m, 0.070 m thick
    assert target_length(spec, (2.0, 1.0, 1.0)) == spec.length_range[0]  # stubby: clamp, thicker
    assert target_length(spec, (5.0, 1.0, 1.0)) == spec.length_range[1]


def test_append_size_keeps_existing_rows(tmp_path):
    (tmp_path / "sizes.csv").write_text("filename,length_m\nmine.glb,0.2\n", encoding="utf-8")
    append_size(tmp_path, "mine.glb", 0.9)
    append_size(tmp_path, "new.glb", 0.21)
    assert (tmp_path / "sizes.csv").read_text(encoding="utf-8").splitlines() == [
        "filename,length_m",
        "mine.glb,0.2",
        "new.glb,0.21",
    ]


def test_apply_keep_list_deletes_unticked(tmp_path):
    from sim.assets.fetch_objaverse import apply_keep_list

    for name in ("a.glb", "b.glb"):
        (tmp_path / "pet_bottle").mkdir(exist_ok=True)
        (tmp_path / "pet_bottle" / name).write_bytes(b"x")
        append_size(tmp_path / "pet_bottle", name, 0.2)
    rows = [
        {"class": "pet_bottle", "file": "a.glb", "status": "accepted"},
        {"class": "pet_bottle", "file": "b.glb", "status": "kept"},
        {"class": "pet_bottle", "file": "", "status": "rejected_shape"},
    ]
    assert apply_keep_list(tmp_path, rows, {"pet_bottle/a.glb"}) == (1, 1)
    assert [r["status"] for r in rows] == ["kept", "rejected_review", "rejected_shape"]
    assert (tmp_path / "pet_bottle" / "a.glb").exists()
    assert not (tmp_path / "pet_bottle" / "b.glb").exists()
    sizes = (tmp_path / "pet_bottle" / "sizes.csv").read_text(encoding="utf-8").splitlines()
    assert sizes == ["filename,length_m", "a.glb,0.2"]


def test_render_preview_draws_the_object(tmp_path):
    import pytest

    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("PIL")
    from PIL import Image

    from sim.assets.fetch_objaverse import render_preview

    out = tmp_path / "p.png"
    render_preview(trimesh.creation.box((0.2, 0.06, 0.07)), out)
    img = Image.open(out).convert("L")
    assert img.size == (260, 110)
    assert img.getpixel((130, 55)) > img.getpixel((2, 2))  # object brighter than the dark belt
