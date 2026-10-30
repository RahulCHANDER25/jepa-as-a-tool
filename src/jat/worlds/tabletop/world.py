"""The T on a 3D table, in MuJoCo. Same T, same table coordinates and same push rules as PushT,
but real friction: the T has mass, slides a little after the pusher leaves it, then stops.
The camera looks down at the table from the front, so V-JEPA 2 sees it in perspective.
"""

import math
from functools import cached_property

import numpy as np

from jat.worlds.base import PUSHER_RADIUS, Push, PushResult
from jat.worlds.table3d import scene as s3d
from jat.worlds.tblock import shape as geo
from jat.worlds.tblock.world import TWorld

TIMESTEP = 0.002
SPEED = 0.15  # m/s: the pusher's speed, PushT's 150 px/s
HOLD = 0.1  # s on the end point before the pusher lifts
SETTLE_MAX = 2.0  # s: the longest wait for the T to stop after the pusher lifts
REST_SPEED = 0.002  # m/s and rad/s: below this, the T is at rest
LOG_EVERY = 0.1  # s between trajectory rows
JAM_DEPTH = 4.0  # px = mm: pusher sunk this deep into the T = the T is pinned against the rim
HEIGHT = 0.02  # m: the T's thickness

_PHYSICS = """\
Physics: the T has mass and real friction. It moves while the pusher touches it, slides a
little after the pusher leaves it, then stops. A push whose line passes off the center of
gravity rotates the T as well as moving it. The table has a rim on all four sides."""


def _t_body(rgba: str) -> str:
    h = HEIGHT / 2
    # Geoms in the body frame: origin = the middle of the bar's top edge; local y (screen,
    # downward) is -Y. The center of mass sits where PushT puts it: 45 mm down the stem.
    return f"""
    <body name="T" pos="0 0 0">
      <freejoint name="T"/>
      <inertial pos="0 -0.045 {h}" mass="0.1" diaginertia="0.00012 0.00009 0.000195"/>
      <geom name="bar" type="box" pos="0 -0.015 {h}" size="0.06 0.015 {h}" rgba="{rgba}"
            friction="0.3 0.005 0.0001"/>
      <geom name="stem" type="box" pos="0 -0.075 {h}" size="0.015 0.045 {h}" rgba="{rgba}"
            friction="0.3 0.005 0.0001"/>
    </body>"""


class Tabletop(TWorld):
    name = "tabletop"
    physics_text = _PHYSICS
    look = s3d.DEFAULT_LOOK

    @cached_property
    def scene(self) -> s3d.Scene:
        return s3d.Scene(s3d.scene_xml(_t_body(self.look.object_rgba), self.look, TIMESTEP))

    def _place(self, data, pose) -> None:
        X, Y = s3d.to_world(pose[0], pose[1])
        data.qpos[:7] = [X, Y, 0.0, *s3d.yaw_quat(pose[2])]
        data.qvel[:] = 0.0

    @staticmethod
    def _pose(data) -> np.ndarray:
        x, y = s3d.from_world(data.qpos[0], data.qpos[1])
        return np.array([x, y, geo.normalize_angle(s3d.screen_angle(data.qpos[3:7]))])

    def push(self, state, push: Push) -> PushResult:
        before = np.array([*state[:2], geo.normalize_angle(state[2])])
        error = self.check_push(before, push)
        if error is not None:
            return PushResult(state=before.tolist(), valid=False, error=error)

        mj, model, data = self.scene.mujoco, self.scene.model, self.scene.data
        mj.mj_resetData(model, data)  # the result depends on (pose, push) only
        self._place(data, before)
        start = np.array(s3d.to_world(*push.start))
        end = np.array(s3d.to_world(*push.end))
        length = float(np.linalg.norm(end - start))
        move_steps = math.ceil(length / SPEED / TIMESTEP)
        hold_steps = round(HOLD / TIMESTEP)
        log_every = round(LOG_EVERY / TIMESTEP)
        data.mocap_pos[0] = [*start, s3d.PUSHER_Z]
        mj.mj_forward(model, data)

        trajectory, jammed = [], False
        for i in range(1, move_steps + hold_steps + 1):
            target = start + (end - start) * min(i / move_steps, 1.0)
            data.mocap_pos[0] = [*target, s3d.PUSHER_Z]
            mj.mj_step(model, data)
            pose, pusher = self._pose(data), s3d.from_world(*target)
            if i % log_every == 0:
                trajectory.append((*pusher, *pose))
            if geo.penetration(pose, pusher, PUSHER_RADIUS) > JAM_DEPTH:
                jammed = True
                break

        data.mocap_pos[0] = s3d.PARKED  # lift the pusher, let the T come to rest
        for i in range(round(SETTLE_MAX / TIMESTEP)):
            mj.mj_step(model, data)
            if i % log_every == 0:
                trajectory.append((*pusher, *self._pose(data)))  # pusher lifted at its end
            if np.abs(data.qvel[:6]).max() < REST_SPEED:
                break
        after = self._pose(data)
        return PushResult(
            state=after.tolist(),
            valid=True,
            moved=self.moved(before, after),
            jammed=jammed,
            trajectory=trajectory,
        )

    def render(self, state, size: int = 256) -> np.ndarray:
        return self.scene.render(lambda data: self._place(data, state), size)


class TabletopNewLook(Tabletop):
    """World 4: the tabletop's physics, another look. Evaluated with the tabletop's world model,
    never trained on."""

    name = "tabletop-new-look"
    model_world = "tabletop"
    look = s3d.NEW_LOOK
