"""Frames for the world model: the T alone, gray on white. No goal, no pusher."""

import cv2
import numpy as np

from jat.env import geometry as geo

T_GRAY = np.array([119, 136, 153], dtype=np.float32)  # LightSlateGray, gym-pusht's color
# Draw a mask at 8x and average it down. OpenCV's own anti-aliasing widens every edge by
# ~0.5 px, which makes the T ~9% too large at 256 px.
_SUPERSAMPLE = 8
_SUBPIXEL_BITS = 4


def render(pose, size: int = 256) -> np.ndarray:
    """uint8 [size, size, 3] image of the T at `pose`; edge pixels are blended by coverage."""
    big = size * _SUPERSAMPLE
    # OpenCV puts pixel centers on integers; ours sit at i + 0.5.
    outline = np.array(geo.t_polygon(pose).exterior.coords) * (big / geo.TABLE) - 0.5
    points = np.round(outline * (1 << _SUBPIXEL_BITS)).astype(np.int32)
    mask = np.zeros((big, big), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255, shift=_SUBPIXEL_BITS)
    coverage = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)[..., None] / 255.0
    return np.round(255.0 - coverage * (255.0 - T_GRAY)).astype(np.uint8)
