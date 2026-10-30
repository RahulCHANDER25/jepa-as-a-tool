"""A logged episode of a 3D world as a video: what the agent thinks, the plans it imagines
(see-through ghosts), the one it picks (green), then the real push.

    jat clip results/tabletop/episodes/rotate-0-003__wm_tool__s0.json --out clip.mp4

The goal is gold, imagined states lavender, the chosen plan green. H.264 through ffmpeg.
"""

import shutil
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np

from jat.agent.episode_log import EpisodeLog, FrameLog
from jat.worlds import get_world
from jat.worlds.table3d import scene as s3d

SLOTS = 10  # ghost slots: 1 for the goal, the rest for imagined plans
GOAL = (0.95, 0.75, 0.15, 0.35)
GHOST = (0.75, 0.6, 1.0, 0.3)
CHOSEN = (0.1, 0.85, 0.35, 0.6)
FPS = 30


def chosen_plan(log: EpisodeLog, real: str, push) -> tuple[str | None, list[FrameLog]]:
    """(id of the imagined first push from `real` the real push repeats, all of them)."""
    candidates = [f for f in log.frames if f.kind == "imagined" and f.parent == real]

    def gap(f: FrameLog) -> float:
        return float(np.hypot(*np.subtract(f.push.start, push.start))) + float(
            np.hypot(*np.subtract(f.push.end, push.end))
        )

    best = min(candidates, key=gap, default=None)
    return (best.id if best is not None and gap(best) < 20 else None), candidates


class _Film:
    def __init__(self, world, size: int):
        import mujoco

        self.mj, self.world = mujoco, world
        self.scene = world.clip_scene(SLOTS)
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.n_objects = len(world.poses(world.random_start(np.random.default_rng(0))))
        self.frames: list[np.ndarray] = []
        self.text = ""

    def _ghost_body(self, slot: int, k: int) -> int:
        name = f"ghost{slot}_{k}"
        return self.model.body_mocapid[
            self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        ]

    def _color(self, slot: int, k: int, rgba) -> None:
        body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, f"ghost{slot}_{k}")
        start, n = self.model.body_geomadr[body], self.model.body_geomnum[body]
        self.model.geom_rgba[start : start + n] = rgba

    def shot(self, state, ghosts=(), goal=None, pusher=None, seconds: float = 0.0) -> None:
        """`ghosts`: (state, rgba) pairs. Adds `seconds` of this picture (one frame if 0)."""
        data = self.data
        self.mj.mj_resetData(self.model, data)
        self.world._place(data, state)
        data.mocap_pos[0] = (
            [*s3d.to_world(*pusher), s3d.PUSHER_Z] if pusher is not None else s3d.PARKED
        )
        slots = [(goal, GOAL, self.world.goal_objects)] if goal is not None else [(None, None, ())]
        slots += [(g, rgba, range(self.n_objects)) for g, rgba in ghosts][: SLOTS - 1]
        for slot in range(SLOTS):
            ghost, rgba, shown = slots[slot] if slot < len(slots) else (None, None, ())
            poses = self.world.poses(ghost) if ghost is not None else []
            for k in range(self.n_objects):
                mocap = self._ghost_body(slot, k)
                if ghost is not None and k in shown:
                    data.mocap_pos[mocap], data.mocap_quat[mocap] = poses[k]
                    self._color(slot, k, rgba)
                else:
                    data.mocap_pos[mocap] = (0, 0, -1)
        self.mj.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera="main")
        frame = self._caption(self.renderer.render().copy())
        self.frames += [frame] * max(1, round(seconds * FPS))

    def _caption(self, frame: np.ndarray) -> np.ndarray:
        if not self.text:
            return frame
        lines = textwrap.wrap(self.text, width=frame.shape[1] // 10)[:3]
        height = 14 + 22 * len(lines)
        band = frame[:height].astype(np.float32)
        frame[:height] = (band * 0.35 + 255 * 0.65).astype(np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (12, 28 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (30, 30, 30), 1, cv2.LINE_AA)  # fmt: skip
        return frame


def render_clip(log: EpisodeLog, out: Path, size: int = 512) -> Path:
    world = get_world(log.task.world)
    if not hasattr(world, "clip_scene"):
        raise ValueError(f"{world.name} is not a 3D world: clips need MuJoCo")
    film = _Film(world, size)
    frames = {f.id: f for f in log.frames}
    goal = log.task.goal
    thoughts = [t.content.strip() for t in log.turns if t.content.strip()]
    real = "R0"
    film.text = thoughts[0] if thoughts else ""
    film.shot(frames[real].true_pose, goal=goal, seconds=1.0)

    for frame in log.frames:
        if frame.kind != "real" or frame.parent is None:
            continue
        before = frames[frame.parent]
        chosen, candidates = chosen_plan(log, before.id, frame.push)
        turn = next(
            (
                t
                for t in log.turns
                for c in t.tool_calls
                if c.name == "act" and f"Current real frame: {frame.id}" in c.result.splitlines()
            ),
            None,
        )
        film.text = turn.content.strip() if turn and turn.content.strip() else film.text
        shown = sorted(candidates, key=lambda f: f.id != chosen)[: SLOTS - 1]
        for n in range(1, len(shown) + 1):  # the plans fan in
            ghosts = [(f.pose, GHOST) for f in shown[:n]]
            film.shot(before.true_pose, ghosts, goal, seconds=1.2 / len(shown))
        if chosen is not None:  # the chosen one turns green, the others fade
            ghosts = [(f.pose, CHOSEN if f.id == chosen else (*GHOST[:3], 0.1)) for f in shown]
            film.shot(before.true_pose, ghosts, goal, seconds=0.9)
        keep = [(frames[chosen].pose, CHOSEN)] if chosen else []
        rows = frame.trajectory or [[*frame.push.end, *frame.true_pose]]
        for row in rows:  # the real push, 0.1 s per row, played at 1x
            film.shot(row[2:], keep, goal, pusher=row[:2], seconds=0.1)
        film.shot(frame.true_pose, keep, goal, seconds=0.6)
        real = frame.id

    film.text = (
        f"{'Success' if log.outcome.success else 'Not at the goal'}: "
        f"{log.outcome.position_error:.0f} px from the goal."
    )
    film.shot(frames[real].true_pose, goal=goal, seconds=1.5)
    return _write(film.frames, out)


def _encoder() -> str | None:
    if shutil.which("ffmpeg") is None:
        return None
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return next((e for e in ("libx264", "libopenh264") if e in listing.stdout), None)


def _write(frames: list[np.ndarray], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    encoder = _encoder()
    if encoder is None:  # no ffmpeg: MPEG-4 part 2, plays in VLC but not in browsers
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return out
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", encoder, "-pix_fmt", "yuv420p",
        "-b:v", "2M", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return out
