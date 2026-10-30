"""Shape of the PushT T block and everything computed from its pose.

Pose = (x, y, theta), the gym-pusht convention: (x, y) is the body origin, the middle of the
bar's top edge; theta is the body angle in radians. Pixel coordinates on a 512 x 512 table,
x to the right, y downward. Positive theta turns the T clockwise on screen.
"""

import math

import numpy as np
from shapely.geometry import Point, Polygon

TABLE = 512

# Local vertices, same order as gym-pusht's two shapes, so keypoints() matches get_keypoints().
BAR = np.array([(-60, 0), (60, 0), (60, 30), (-60, 30)], dtype=float)
STEM = np.array([(-15, 30), (15, 30), (15, 120), (-15, 120)], dtype=float)
# The T as one outline. Built from BAR and STEM, a union can split in two on float noise.
OUTLINE = np.array(
    [(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)],
    dtype=float,
)
CENTER = np.array([0.0, 45.0])  # center of gravity in local coordinates
LANDMARKS = {
    "left tip": (-60.0, 15.0),
    "right tip": (60.0, 15.0),
    "bar middle": (0.0, 15.0),
    "stem end": (0.0, 120.0),
}
_COMPASS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


def normalize_angle(theta: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.pi - (math.pi - theta) % (2 * math.pi)


def to_world(pose, local) -> np.ndarray:
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(local, dtype=float) @ rotation.T + (x, y)


def keypoints(pose) -> np.ndarray:
    """The 8 corners [8, 2]: bar corners, then stem corners."""
    return to_world(pose, np.vstack([BAR, STEM]))


def center(pose) -> np.ndarray:
    return to_world(pose, CENTER)


def pose_from_center(cx: float, cy: float, theta: float) -> np.ndarray:
    """Inverse of center(): the pose whose center of gravity sits at (cx, cy)."""
    origin = np.array([cx, cy]) - to_world((0.0, 0.0, theta), CENTER)
    return np.array([origin[0], origin[1], normalize_angle(theta)])


def landmarks(pose) -> dict[str, np.ndarray]:
    return {name: to_world(pose, p) for name, p in LANDMARKS.items()}


def t_polygon(pose) -> Polygon:
    return Polygon(to_world(pose, OUTLINE))


def distance_to_t(pose, point) -> float:
    """Distance from a point to the T; 0 when the point is inside it."""
    return t_polygon(pose).distance(Point(point))


def coverage(pose, goal) -> float:
    """Share of the goal T's area covered by the T: PushT's native success metric."""
    goal_poly = t_polygon(goal)
    return t_polygon(pose).intersection(goal_poly).area / goal_poly.area


def pose_error(pose, goal) -> tuple[float, float]:
    """(center distance in px, absolute angle difference in degrees)."""
    distance = float(np.linalg.norm(center(pose) - center(goal)))
    angle = abs(math.degrees(normalize_angle(pose[2] - goal[2])))
    return distance, angle


def moved(before, after) -> bool:
    distance, angle = pose_error(after, before)
    return distance > 1.0 or angle > 0.5


def stem_direction(pose) -> str:
    """Compass word for where the stem points on screen."""
    dx, dy = to_world((0.0, 0.0, pose[2]), (0.0, 1.0))
    # screen y grows downward; flip it so "up" is up
    sector = round(math.degrees(math.atan2(-dy, dx)) / 45) % 8
    return _COMPASS[sector]


def pose_text(pose, goal=None) -> str:
    """The T as the LLM reads it: center, angle, aim points, and the error to the goal."""
    cx, cy = center(pose)
    degrees = round(math.degrees(pose[2])) % 360
    points = "  ".join(f"{name} ({x:.0f}, {y:.0f})" for name, (x, y) in landmarks(pose).items())
    lines = [
        f"T center ({cx:.0f}, {cy:.0f}), angle {degrees:.0f}° (stem points {stem_direction(pose)})",
        f"  {points}",
    ]
    if goal is not None:
        distance, angle = pose_error(pose, goal)
        lines.append(f"  goal error: {distance:.0f} px, {angle:.0f}°")
    return "\n".join(lines)
