import numpy as np

from jat.worlds import get_world
from jat.worlds.base import TABLE
from jat.worlds.pusht.world import T_GRAY
from jat.worlds.tblock import shape as geo

render = get_world("pusht").render

T_AREA_PX = 120 * 30 + 30 * 90  # on the 512 px table


def _coverage_map(image: np.ndarray) -> np.ndarray:
    """1 where fully gray, 0 where white, fractions on anti-aliased edges."""
    return (255.0 - image[..., 0]) / (255.0 - float(T_GRAY[0]))


def test_render_draws_the_t_area_at_its_center():
    pose = [300.0, 180.0, 0.9]
    image = render(pose, size=256)
    weight = _coverage_map(image)
    scale = 256 / TABLE
    assert image.shape == (256, 256, 3) and image.dtype == np.uint8
    assert abs(weight.sum() - T_AREA_PX * scale**2) / (T_AREA_PX * scale**2) < 0.02
    ys, xs = np.indices(weight.shape)
    centroid = np.array([(xs * weight).sum(), (ys * weight).sum()]) / weight.sum()
    # Area centroid, not geo.center(): gym-pusht's center of gravity is not area-weighted.
    expected = np.array(geo.t_polygon(pose).centroid.coords[0]) * scale
    # pixel (i, j) covers [i, i+1): its center is at i + 0.5
    np.testing.assert_allclose(centroid + 0.5, expected, atol=0.1)


def test_render_without_t_on_table_is_blank():
    image = render([-500.0, -500.0, 0.0], size=64)
    assert (image == 255).all()
