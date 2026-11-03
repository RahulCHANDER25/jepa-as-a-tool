"""A logged episode as a video (MuJoCo in 3D, drawn from above in 2D): the plans the agent
imagines (see-through ghosts), the one it picks (green), then the real push. What the agent
says goes next to the video as WebVTT subtitles, so the site sets it in real type.

    jat clip results/tabletop/episodes/rotate-0-003__wm_tool__s0.json --out clip.mp4

The goal is gold, imagined states lavender, the chosen plan green. H.264 through ffmpeg.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from jat.agent.episode_log import EpisodeLog, FrameLog
from jat.worlds import get_world
from jat.worlds.base import TABLE
from jat.worlds.table3d import scene as s3d

SLOTS = 10  # ghost slots: 1 for the goal, the rest for imagined plans
GOAL = (0.95, 0.75, 0.15, 0.35)
GHOST = (0.75, 0.6, 1.0, 0.3)
CHOSEN = (0.1, 0.85, 0.35, 0.6)
FPS = 30
# The world's camera (45°) crops the table's edge in the new room; clips are for people, so they
# see the whole table. The world model only ever sees the world's own camera.
CLIP_FOVY = 54
SHADOW_SIZE = 8192  # MuJoCo's 2048 draws the rim's shadow as a saw-tooth
# Ghosts lie where the real objects are: each slot sits a bit higher, so overlapping surfaces
# stack in a fixed order instead of flickering through each other (z-fighting)
GHOST_LIFT = 0.0006  # m per slot
# Seconds on screen. The real push plays at 2x: the physics is slow next to the thinking.
START, FAN, CHOOSE, SETTLE, END = 0.8, 1.0, 0.7, 0.5, 2.0
PUSH_ROW = 0.05  # one trajectory row is 0.1 s of physics
CRF = 18  # libx264 quality: lower is sharper
BITRATE = "8M"  # the same for OpenH264, which only takes a bitrate


def chosen_plan(log: EpisodeLog, real: str, push) -> tuple[str | None, list[FrameLog]]:
    """(id of the imagined first push from `real` the real push repeats, all of them)."""
    candidates = [f for f in log.frames if f.kind == "imagined" and f.parent == real]

    def gap(f: FrameLog) -> float:
        return float(np.hypot(*np.subtract(f.push.start, push.start))) + float(
            np.hypot(*np.subtract(f.push.end, push.end))
        )

    best = min(candidates, key=gap, default=None)
    return (best.id if best is not None and gap(best) < 20 else None), candidates


class Reel:
    """Frames of a clip, and the words the agent says while they play."""

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.text = ""
        self.cues: list[list] = []  # [first frame, end frame, text]

    def _add(self, frame: np.ndarray, seconds: float) -> None:
        n = max(1, round(seconds * FPS))
        last = self.cues[-1] if self.cues else None
        if last and last[2] == self.text and last[1] == len(self.frames):
            last[1] += n
        else:
            self.cues.append([len(self.frames), len(self.frames) + n, self.text])
        self.frames += [frame] * n

    def subtitles(self) -> str:
        """The words as WebVTT, one cue per stretch of the same sentence."""

        def stamp(frame: int) -> str:
            ms = round(frame * 1000 / FPS)
            return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"

        cues = [f"{stamp(a)} --> {stamp(b)}\n{text}" for a, b, text in self.cues if text.strip()]
        return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


class Film(Reel):
    def __init__(self, world, size: int):
        import mujoco

        self.mj, self.world = mujoco, world
        self.scene = world.clip_scene(SLOTS)
        self.model = self.scene.model
        self.model.cam_fovy[:] = CLIP_FOVY
        self.model.vis.quality.shadowsize = SHADOW_SIZE  # read when the renderer is made
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, size)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, size)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.n_objects = len(world.poses(world.random_start(np.random.default_rng(0))))
        super().__init__()

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
                    pos, quat = poses[k]
                    data.mocap_pos[mocap] = np.add(pos, (0, 0, GHOST_LIFT * (slot + 1)))
                    data.mocap_quat[mocap] = quat
                    self._color(slot, k, rgba)
                else:
                    data.mocap_pos[mocap] = (0, 0, -1)
        self.mj.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera="main")
        self._add(self.renderer.render().copy(), seconds)


class Flat(Reel):
    """A 2D world from above, drawn like the site's player: the real objects filled, the goal,
    the imagined plans and the chosen one see-through on top."""

    BACKGROUND, GRID, EDGE = (250, 250, 248), (232, 232, 228), (120, 120, 120)

    def __init__(self, world, size: int):
        super().__init__()
        self.world, self.size = world, size

    def _points(self, outline: np.ndarray) -> np.ndarray:
        return np.round(outline * self.size / TABLE * 16).astype(np.int32)  # 4 subpixel bits

    def _fill(self, image, state, rgba, goal: bool = False, edge: bool = True) -> np.ndarray:
        layer = image.copy()
        color = tuple(int(255 * c) for c in rgba[:3])
        for _, outline in self.world.outlines(state, goal=goal):
            cv2.fillPoly(layer, [self._points(outline)], color, cv2.LINE_AA, shift=4)
            if edge:
                cv2.polylines(layer, [self._points(outline)], True, color, 2, cv2.LINE_AA, shift=4)
        alpha = min(1.0, rgba[3] * 1.6)  # flat colors need more weight than lit 3D ones
        return (image * (1 - alpha) + layer * alpha).astype(np.uint8)

    def shot(self, state, ghosts=(), goal=None, pusher=None, seconds: float = 0.0) -> None:
        n = self.size
        image = np.full((n, n, 3), self.BACKGROUND, np.uint8)
        for k in range(1, 8):
            cv2.line(image, (k * n // 8, 0), (k * n // 8, n), self.GRID, 1)
            cv2.line(image, (0, k * n // 8), (n, k * n // 8), self.GRID, 1)
        if goal is not None:
            image = self._fill(image, goal, GOAL, goal=True)
        for color, outline in self.world.outlines(state):
            cv2.fillPoly(image, [self._points(outline)], color, cv2.LINE_AA, shift=4)
            cv2.polylines(
                image, [self._points(outline)], True, (50, 50, 50), 1, cv2.LINE_AA, shift=4
            )
        for ghost, rgba in ghosts[: SLOTS - 1]:
            image = self._fill(image, ghost, rgba)
        if pusher is not None:
            center = tuple(int(round(c * n / TABLE)) for c in pusher)
            cv2.circle(image, center, max(4, n // 40), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, center, max(4, n // 40), self.EDGE, 2, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (n - 1, n - 1), self.EDGE, 2)
        self._add(image, seconds)


def _last_thought_before_each_push(log: EpisodeLog) -> dict[str, str]:
    """Real frame id -> the agent's last words before the push that made it (the turn that
    calls `act` often has none)."""
    said, last = {}, ""
    for turn in log.turns:
        last = turn.content.strip() or last
        for call in turn.tool_calls:
            for line in call.result.splitlines() if call.name == "act" else ():
                if line.startswith("Current real frame: "):
                    said[line.removeprefix("Current real frame: ").split()[0]] = last
    return said


def render_clip(log: EpisodeLog, out: Path, size: int = 512) -> Path:
    """Writes `out` (the video) and the agent's words next to it (`out` with .vtt)."""
    world = get_world(log.task.world)
    film = Film(world, size) if hasattr(world, "clip_scene") else Flat(world, size)
    frames = {f.id: f for f in log.frames}
    goal = log.task.goal
    thoughts = [t.content.strip() for t in log.turns if t.content.strip()]
    said = _last_thought_before_each_push(log)
    real = "R0"
    film.text = thoughts[0] if thoughts else ""
    film.shot(frames[real].true_pose, goal=goal, seconds=START)

    for frame in log.frames:
        if frame.kind != "real" or frame.parent is None:
            continue
        before = frames[frame.parent]
        chosen, candidates = chosen_plan(log, before.id, frame.push)
        film.text = said.get(frame.id, film.text)
        shown = sorted(candidates, key=lambda f: f.id != chosen)[: SLOTS - 1]
        for n in range(1, len(shown) + 1):  # the plans fan in
            ghosts = [(f.pose, GHOST) for f in shown[:n]]
            film.shot(before.true_pose, ghosts, goal, seconds=FAN / len(shown))
        if chosen is not None:  # the chosen one turns green, the others fade
            ghosts = [(f.pose, CHOSEN if f.id == chosen else (*GHOST[:3], 0.1)) for f in shown]
            film.shot(before.true_pose, ghosts, goal, seconds=CHOOSE)
        keep = [(frames[chosen].pose, CHOSEN)] if chosen else []
        rows = frame.trajectory or [[*frame.push.end, *frame.true_pose]]
        for row in rows:  # the real push
            film.shot(row[2:], keep, goal, pusher=row[:2], seconds=PUSH_ROW)
        film.shot(frame.true_pose, keep, goal, seconds=SETTLE)
        real = frame.id

    film.text = (
        f"{'Success' if log.outcome.success else 'Not at the goal'}: "
        f"{log.outcome.position_error:.0f} px from the goal."
    )
    film.shot(frames[real].true_pose, goal=goal, seconds=END)
    out.with_suffix(".vtt").write_text(film.subtitles())
    return write_video(film.frames, out)


def _encoder() -> str | None:
    if shutil.which("ffmpeg") is None:
        return None
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return next((e for e in ("libx264", "libopenh264") if e in listing.stdout), None)


def write_video(frames: list[np.ndarray], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    encoder = _encoder()
    if encoder is None:  # no ffmpeg: MPEG-4 part 2, plays in VLC but not in browsers
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return out
    # OpenH264 has no quality target, only a bitrate
    quality = ["-crf", str(CRF), "-preset", "slow"] if encoder == "libx264" else ["-b:v", BITRATE]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", encoder, "-pix_fmt", "yuv420p",
        *quality, "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return out
