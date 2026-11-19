import colorsys

import numpy as np
import pytest

from trash_sorter.belt_look import BELT_PATTERNS, belt_texture, sample_belt_color


@pytest.mark.parametrize("kind", BELT_PATTERNS)
def test_texture_is_an_rgb_image_with_visible_pattern(kind):
    tex = belt_texture(kind, np.random.default_rng(0), size=128)
    assert tex.shape == (128, 128, 3)
    assert tex.dtype == np.uint8
    assert tex.std() > 4  # a flat image would not be a pattern


@pytest.mark.parametrize("kind", BELT_PATTERNS)
def test_texture_is_reproducible_and_varies_with_the_seed(kind):
    a = belt_texture(kind, np.random.default_rng(3), size=64)
    b = belt_texture(kind, np.random.default_rng(3), size=64)
    c = belt_texture(kind, np.random.default_rng(4), size=64)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_texture_is_bright_so_the_tint_sets_the_belt_color():
    for kind in BELT_PATTERNS:
        assert belt_texture(kind, np.random.default_rng(1), size=128).mean() > 140


def test_unknown_pattern_is_rejected():
    with pytest.raises(ValueError, match="pattern"):
        belt_texture("marble", np.random.default_rng(0))


def test_belt_colors_cover_dark_light_and_many_hues():
    rng = np.random.default_rng(0)
    colors = np.array([sample_belt_color(rng) for _ in range(400)])
    assert colors.min() >= 0.0 and colors.max() <= 1.0
    value = colors.max(axis=1)
    assert (value < 0.12).mean() > 0.15  # dark rubber belts stay common
    assert (value > 0.5).mean() > 0.15  # light belts exist too
    hues = [colorsys.rgb_to_hsv(*c)[0] for c in colors if colorsys.rgb_to_hsv(*c)[1] > 0.3]
    assert len(np.unique(np.floor(np.array(hues) * 6))) == 6  # not one favorite color


@pytest.mark.parametrize("kind", BELT_PATTERNS)
def test_texture_repeats_without_a_visible_seam(kind):
    tex = belt_texture(kind, np.random.default_rng(5), size=256)[..., 0].astype(float)
    for axis in (0, 1):
        inside = np.abs(np.diff(tex, axis=axis)).mean()  # typical step between neighboring rows / columns
        first, last = np.take(tex, 0, axis=axis), np.take(tex, -1, axis=axis)
        assert np.abs(first - last).mean() < 2.5 * inside + 1.0
