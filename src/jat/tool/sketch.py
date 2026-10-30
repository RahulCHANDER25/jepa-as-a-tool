"""A top view of a frame and its imagined branches, for people and for MCP clients: the real
state filled, the goal dashed, every imagined state outlined with its frame id."""

import cv2
import numpy as np

from jat.worlds.base import TABLE, World

SIZE = 512
GOAL = (40, 150, 70)
GHOST = (150, 150, 150)
PICKED = (40, 170, 80)


def _points(outline: np.ndarray) -> np.ndarray:
    return np.round(outline * SIZE / TABLE * 16).astype(np.int32)  # 4 subpixel bits


def _dashed(image, outline, color) -> None:
    closed = np.vstack([outline, outline[:1]])
    for i in range(len(closed) - 1):
        a, b = closed[i], closed[i + 1]
        n = max(1, int(np.linalg.norm(b - a) // 8))
        for k in range(0, n, 2):
            p, q = a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n
            cv2.line(image, *(_points(np.array([p, q]))), color, 2, cv2.LINE_AA, shift=4)


def sketch(
    world: World,
    state,
    goal=None,
    imagined: list[tuple[str, object]] = (),
    picked: str | None = None,
) -> np.ndarray:
    """uint8 [SIZE, SIZE, 3]. `imagined`: (frame id, state) pairs; `picked`: the frame id to
    draw in green (the plan the agent chose)."""
    image = np.full((SIZE, SIZE, 3), 248, np.uint8)
    margin = int(5 * SIZE / TABLE)
    cv2.rectangle(image, (margin, margin), (SIZE - margin, SIZE - margin), (90, 90, 90), 2)
    if goal is not None:
        for _, outline in world.outlines(goal, goal=True):
            _dashed(image, outline, GOAL)
    for color, outline in world.outlines(state):
        cv2.fillPoly(image, [_points(outline)], color, cv2.LINE_AA, shift=4)
        cv2.polylines(image, [_points(outline)], True, (60, 60, 60), 1, cv2.LINE_AA, shift=4)
    for frame_id, ghost in imagined:
        color = PICKED if frame_id == picked else GHOST
        outlines = [outline for _, outline in world.outlines(ghost)]
        for outline in outlines:
            width = 3 if frame_id == picked else 1
            cv2.polylines(image, [_points(outline)], True, color, width, cv2.LINE_AA, shift=4)
        x, y = outlines[-1].mean(0) * SIZE / TABLE
        cv2.putText(image, frame_id, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)  # fmt: skip
    return image


def png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return data.tobytes()
