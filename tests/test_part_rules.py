import numpy as np
import pytest

from sim.assets.part_rules import brighten_glass_texture, clear_parts, glass_tint, marked_transparent


def test_brighten_glass_texture_lifts_dark_textures_only():
    dark = np.full((4, 4, 3), (20, 40, 10), np.uint8)
    out = brighten_glass_texture(dark)
    assert out[0, 0, 1] == pytest.approx(153, abs=1)  # brightest channel -> 0.6
    assert out[0, 0, 0] / out[0, 0, 1] == pytest.approx(0.5, abs=0.02)  # hue kept
    light = np.full((2, 2, 3), 220, np.uint8)
    assert (brighten_glass_texture(light) == light).all()


def test_marked_transparent():
    assert marked_transparent("BLEND", 1.0)
    assert marked_transparent(None, 0.3)
    assert not marked_transparent(None, 1.0)
    assert not marked_transparent("OPAQUE", 1.0)


def test_clear_parts_uses_marked_parts_when_any():
    assert clear_parts("pet_bottle", [1.0, 5.0, 2.0], [True, False, False]) == [True, False, False]


def test_clear_parts_falls_back_to_largest_part():
    # Label, cap, body: the body (largest) becomes glass, label and cap stay opaque.
    assert clear_parts("glass_bottle", [3.0, 1.0, 9.0], [False, False, False]) == [False, False, True]


def test_clear_parts_single_part_bottle_is_all_glass():
    assert clear_parts("pet_bottle", [4.0], [False]) == [True]


def test_clear_parts_never_for_cans_and_cartons():
    assert clear_parts("aluminum_can", [1.0, 2.0], [True, False]) == [False, False]


def test_glass_tint_keeps_hue_but_lets_light_through():
    r, g, b = glass_tint((0.08, 0.25, 0.05))  # dark beer-bottle green
    assert g == pytest.approx(0.4)
    assert r < g and b < g
    assert glass_tint((0.9, 0.95, 1.0)) == (0.9, 0.95, 1.0)  # already clear: unchanged
    assert glass_tint((0.0, 0.008, 0.008)) == (0.6, 0.6, 0.6)  # black: neutral grey
