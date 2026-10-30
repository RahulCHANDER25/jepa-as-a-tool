"""PushT with one action: a straight push. The pusher is lifted between pushes.

Wraps gym-pusht so the physics stay the benchmark's: same bodies, same PD controller, same
10 Hz control / 100 Hz physics. Between pushes the pusher sits off the table and the T is at
rest, so the T pose is the whole state.
"""

import math

import numpy as np
from gym_pusht.envs.pusht import PushTEnv
from pydantic import BaseModel, ConfigDict, Field

from jat.env import geometry as geo

PUSHER_RADIUS = 15.0
MARGIN = 20.0  # push points stay this far from the table edge
MIN_LENGTH, MAX_LENGTH = 10.0, 200.0
STEP_PX = 15.0  # pusher travel per 0.1 s control step = 150 px/s
HOLD_STEPS = 10  # max extra steps on the end point: the PD controller lags its target
END_TOLERANCE = 2.0
# Pusher sunk this deep into the T = the T is pinned (usually against a wall). The pusher is
# kinematic, so nothing stops it; we stop the push instead.
JAM_DEPTH = 3.0
PARKED = (-200.0, -200.0)  # outside the walls: touches nothing


class Push(BaseModel):
    """One straight push of the pusher, in table pixels."""

    model_config = ConfigDict(frozen=True)

    # Lists, not tuples: tuple JSON schemas use `prefixItems`, which some tool-calling
    # endpoints reject.
    start: list[float] = Field(
        min_length=2, max_length=2, description="[x, y] where the pusher is placed"
    )
    end: list[float] = Field(
        min_length=2, max_length=2, description="[x, y] where the pusher stops"
    )

    def length(self) -> float:
        return math.dist(self.start, self.end)

    def numeric(self) -> np.ndarray:
        """[x1, y1, x2, y2] scaled to [-1, 1]: the world model's action input."""
        return np.array([*self.start, *self.end]) / (geo.TABLE / 2) - 1

    def text(self) -> str:
        (x1, y1), (x2, y2) = self.start, self.end
        return f"({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})"


class PushResult(BaseModel):
    pose: list[float]
    valid: bool
    moved: bool = False
    jammed: bool = False
    error: str | None = None
    # One row per 0.1 s step: pusher x, pusher y, T x, T y, T theta. For replays only.
    trajectory: list[tuple[float, float, float, float, float]] = []


def check_push(pose, push: Push) -> str | None:
    """Why this push cannot run from this pose, or None if it can."""
    points = np.array([push.start, push.end])
    if not ((points >= MARGIN) & (points <= geo.TABLE - MARGIN)).all():
        return (
            f"start and end must lie inside [{MARGIN:.0f}, {geo.TABLE - MARGIN:.0f}] on both axes"
        )
    length = push.length()
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        return f"push length {length:.0f} px is outside [{MIN_LENGTH:.0f}, {MAX_LENGTH:.0f}]"
    if geo.distance_to_t(pose, push.start) < PUSHER_RADIUS + 1:
        return "the pusher at start would overlap the T; start outside the T"
    return None


def _penetration(pose, point) -> float:
    """How deep the pusher disc at `point` sinks into the T, in px (<= 0: no overlap)."""
    polygon = geo.t_polygon(pose)
    p = geo.Point(point)
    if polygon.contains(p):
        return PUSHER_RADIUS + polygon.exterior.distance(p)
    return PUSHER_RADIUS - polygon.distance(p)


class World:
    """The real PushT table, driven by straight pushes."""

    def __init__(self, pose):
        self._env = PushTEnv(obs_type="state")
        self.set_pose(pose)

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

    def branch(self) -> "World":
        """An independent copy at the same pose."""
        return World(self.pose())

    def push(self, push: Push) -> PushResult:
        before = self.pose()
        error = check_push(before, push)
        if error is not None:
            return PushResult(pose=before.tolist(), valid=False, error=error)
        self.set_pose(before)  # the result depends on (pose, push) only

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
            if _penetration(pose, pusher) > JAM_DEPTH:
                jammed = True
                break
            if i >= n_move - 1 and math.dist(pusher, push.end) < END_TOLERANCE:
                break

        self._park()
        after = self.pose()
        return PushResult(
            pose=after.tolist(),
            valid=True,
            moved=geo.moved(before, after),
            jammed=jammed,
            trajectory=trajectory,
        )

    def _park(self) -> None:
        self._env.agent.position = PARKED
        self._env.agent.velocity = (0.0, 0.0)


def random_start_pose(rng: np.random.Generator, wall_margin: float = 60.0) -> np.ndarray:
    """A pose whose 8 corners all stay `wall_margin` px away from the walls."""
    while True:
        theta = rng.uniform(-math.pi, math.pi)
        cx, cy = rng.uniform(wall_margin, geo.TABLE - wall_margin, size=2)
        pose = geo.pose_from_center(cx, cy, theta)
        corners = geo.keypoints(pose)
        if ((corners >= wall_margin) & (corners <= geo.TABLE - wall_margin)).all():
            return pose


def random_aimed_push(pose, rng: np.random.Generator) -> Push:
    """A push that hits the T: aimed at a random point of its outline, roughly inward."""
    ring = geo.t_polygon(pose).exterior
    polygon = geo.t_polygon(pose)
    for _ in range(200):
        s = rng.uniform(0, ring.length)
        contact = np.array(ring.interpolate(s).coords[0])
        before = np.array(ring.interpolate(max(s - 1.0, 0.0)).coords[0])
        after = np.array(ring.interpolate(min(s + 1.0, ring.length)).coords[0])
        tangent = after - before
        normal = np.array([tangent[1], -tangent[0]]) / (np.linalg.norm(tangent) + 1e-9)
        if polygon.contains(geo.Point(contact + 2 * normal)):
            normal = -normal  # make it point outward
        turn = math.radians(rng.uniform(-80, 80))
        c, s_ = math.cos(turn), math.sin(turn)
        direction = np.array([[c, -s_], [s_, c]]) @ -normal
        gap, depth = rng.uniform(20, 40), rng.uniform(20, 150)
        start = contact - direction * (PUSHER_RADIUS + gap)
        end = start + direction * (gap + depth)
        push = Push(start=tuple(start), end=tuple(end))
        if check_push(pose, push) is None:
            return push
    raise RuntimeError("no valid aimed push from this pose")


def random_uniform_push(pose, rng: np.random.Generator) -> Push:
    """Any valid push on the table; most of them miss the T."""
    while True:
        start = rng.uniform(MARGIN, geo.TABLE - MARGIN, size=2)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(MIN_LENGTH, MAX_LENGTH)
        end = start + length * np.array([math.cos(angle), math.sin(angle)])
        push = Push(start=tuple(start), end=tuple(end))
        if check_push(pose, push) is None:
            return push
