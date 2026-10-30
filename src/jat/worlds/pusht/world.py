"""PushT with one action: a straight push. The pusher is lifted between pushes.

Wraps gym-pusht so the physics stay the benchmark's: same bodies, same PD controller, same
10 Hz control / 100 Hz physics. Between pushes the pusher sits off the table and the T is at
rest, so the T pose is the whole state.
"""

import math
import threading

import cv2
import numpy as np
from gym_pusht.envs.pusht import PushTEnv

from jat.worlds.base import PUSHER_RADIUS, TABLE, Push, PushResult
from jat.worlds.tblock import shape as geo
from jat.worlds.tblock.world import TWorld

STEP_PX = 15.0  # pusher travel per 0.1 s control step = 150 px/s
HOLD_STEPS = 10  # max extra steps on the end point: the PD controller lags its target
END_TOLERANCE = 2.0
# Pusher sunk this deep into the T = the T is pinned (usually against a wall). The pusher is
# kinematic, so nothing stops it; we stop the push instead.
JAM_DEPTH = 3.0
PARKED = (-200.0, -200.0)  # outside the walls: touches nothing

T_GRAY = np.array([119, 136, 153], dtype=np.float32)  # LightSlateGray, gym-pusht's color
# Draw a mask at 8x and average it down. OpenCV's own anti-aliasing widens every edge by
# ~0.5 px, which makes the T ~9% too large at 256 px.
_SUPERSAMPLE = 8
_SUBPIXEL_BITS = 4

_PHYSICS = """\
Physics: no momentum. The T moves only while the pusher touches it and stops at once.
A push whose line passes off the center of gravity rotates the T as well as moving it.
The table has walls on all four sides."""


class _Table:
    """One gym-pusht simulator. Not thread-safe: PushT keeps one per thread."""

    def __init__(self):
        self._env = PushTEnv(obs_type="state")

    def set_pose(self, pose) -> None:
        # A fresh pymunk space every time. pymunk keeps contacts cached across steps, so a
        # reused space makes the next push depend on more than the pose. `_setup` builds the
        # space without stepping it; gym's `reset()` steps once and can create such contacts.
        self._env._setup()
        block = self._env.block
        # Angle before position. gym-pusht's reset_to_state does the reverse and the T ends up
        # shifted, since it rotates about its center of gravity (plan/notes.md).
        block.angle = float(pose[2])
        block.position = (float(pose[0]), float(pose[1]))
        block.velocity = (0.0, 0.0)
        block.angular_velocity = 0.0
        self._park()

    def pose(self) -> np.ndarray:
        block = self._env.block
        return np.array([block.position.x, block.position.y, geo.normalize_angle(block.angle)])

    def run(self, push: Push) -> tuple[list, bool]:
        """Drive the pusher along `push` from the current pose: (trajectory, jammed)."""
        agent = self._env.agent
        agent.position = tuple(push.start)
        agent.velocity = (0.0, 0.0)

        start, end = np.array(push.start), np.array(push.end)
        length = push.length()
        direction = (end - start) / length
        n_move = math.ceil(length / STEP_PX)
        targets = [start + direction * min(i * STEP_PX, length) for i in range(1, n_move + 1)]
        targets += [end] * HOLD_STEPS

        trajectory, jammed = [], False
        for i, target in enumerate(targets):
            self._env.step(target)
            pose = self.pose()
            pusher = (agent.position.x, agent.position.y)
            trajectory.append((*pusher, *pose))
            if geo.penetration(pose, pusher, PUSHER_RADIUS) > JAM_DEPTH:
                jammed = True
                break
            if i >= n_move - 1 and math.dist(pusher, push.end) < END_TOLERANCE:
                break
        self._park()
        return trajectory, jammed

    def _park(self) -> None:
        self._env.agent.position = PARKED
        self._env.agent.velocity = (0.0, 0.0)


class PushT(TWorld):
    name = "pusht"
    physics_text = _PHYSICS

    def __init__(self):
        self._local = threading.local()

    def _table(self) -> _Table:
        if not hasattr(self._local, "table"):
            self._local.table = _Table()
        return self._local.table

    def push(self, state, push: Push) -> PushResult:
        table = self._table()
        table.set_pose(state)
        before = table.pose()
        error = self.check_push(before, push)
        if error is not None:
            return PushResult(state=before.tolist(), valid=False, error=error)
        trajectory, jammed = table.run(push)
        after = table.pose()
        return PushResult(
            state=after.tolist(),
            valid=True,
            moved=self.moved(before, after),
            jammed=jammed,
            trajectory=trajectory,
        )

    def render(self, state, size: int = 256) -> np.ndarray:
        """The T alone, gray on white; edge pixels are blended by coverage."""
        big = size * _SUPERSAMPLE
        # OpenCV puts pixel centers on integers; ours sit at i + 0.5.
        outline = np.array(geo.t_polygon(state).exterior.coords) * (big / TABLE) - 0.5
        points = np.round(outline * (1 << _SUBPIXEL_BITS)).astype(np.int32)
        mask = np.zeros((big, big), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255, shift=_SUBPIXEL_BITS)
        coverage = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)[..., None] / 255.0
        return np.round(255.0 - coverage * (255.0 - T_GRAY)).astype(np.uint8)
