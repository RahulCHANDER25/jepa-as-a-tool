"""See a world before training anything on it: what V-JEPA 2 sees, a few random pushes as a
video, and for the 3D worlds an interactive MuJoCo window.

    jat show                                  # every world -> looks/<world>.png and .mp4
    jat show --world billiards --interactive  # rotate the camera, poke the balls
"""

from pathlib import Path

import cv2
import numpy as np

from jat.clip import FPS, Film, write_video
from jat.worlds import get_world
from jat.worlds.base import PUSHER_RADIUS, TABLE


def frames_grid(world, rng: np.random.Generator, n: int = 8) -> np.ndarray:
    """`n` random states as the encoder sees them, 256 px each, 2 rows."""
    states, state = [], world.random_start(rng)
    for i in range(n):
        states.append(state)
        state = (
            world.random_start(rng)
            if i % 2
            else np.array(world.push(state, world.aimed_push(state, rng)).state)
        )
    tiles = [world.render(s, 256) for s in states]
    return np.vstack([np.hstack(tiles[: n // 2]), np.hstack(tiles[n // 2 :])])


def _flat_frame(world, state, pusher, text: str, size: int) -> np.ndarray:
    """A 2D world: its own frame, with the pusher drawn on top."""
    frame = world.render(state, size).copy()
    if pusher is not None:
        center = tuple(int(v * size / TABLE) for v in pusher)
        cv2.circle(
            frame, center, int(PUSHER_RADIUS * size / TABLE), (235, 104, 52), -1, cv2.LINE_AA
        )
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    return frame


def push_video(
    world, rng: np.random.Generator, out: Path, pushes: int = 6, size: int = 512
) -> Path:
    """Random aimed pushes from a random start, the pusher visible, at real speed."""
    state = world.random_start(rng)
    film = Film(world, size) if hasattr(world, "clip_scene") else None
    frames = []
    for k in range(1, pushes + 1):
        result = world.push(state, world.aimed_push(state, rng))
        text = f"{world.name}: random push {k} of {pushes}"
        for row in result.trajectory:  # one row per 0.1 s
            if film is None:
                frames += [_flat_frame(world, row[2:], row[:2], text, size)] * round(0.1 * FPS)
            else:
                film.text = text
                film.shot(row[2:], pusher=row[:2], seconds=0.1)
        state = np.array(result.state)
    return write_video(film.frames if film else frames, out)


def look(world_name: str, out: Path, seed: int = 0) -> tuple[Path, Path]:
    world = get_world(world_name)
    out.mkdir(parents=True, exist_ok=True)
    image = out / f"{world.name}.png"
    grid = frames_grid(world, np.random.default_rng(seed))
    cv2.imwrite(str(image), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return image, push_video(world, np.random.default_rng(seed), out / f"{world.name}.mp4")


def interactive(world_name: str, seed: int = 0) -> None:
    """MuJoCo's viewer on a random start: drag to rotate, double-click a body then
    Ctrl + right-drag to push it. Close the window to quit."""
    world = get_world(world_name)
    if not hasattr(world, "scene"):
        raise ValueError(f"{world.name} is 2D: no interactive window (see its video instead)")
    import mujoco
    import mujoco.viewer

    model = world.scene.model
    data = mujoco.MjData(model)
    world._place(data, world.random_start(np.random.default_rng(seed)))
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)
