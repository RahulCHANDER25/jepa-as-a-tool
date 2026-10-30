"""Worlds whose object is the T. They share the shape, the push rules, the text the agent reads
and the task families; each subclass brings its own physics and frames.

Tasks: move the T from a start pose to a goal pose within a push budget. Each goal is the end
pose of a random plan of aimed pushes in the world's own physics, so every task is solvable;
`ref_plan` stores that plan.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from jat.worlds.base import (
    MARGIN,
    MAX_LENGTH,
    MIN_LENGTH,
    PUSHER_RADIUS,
    TABLE,
    Outcome,
    Push,
    PushResult,
    Task,
    ToolText,
    check_bounds,
)
from jat.worlds.tblock import shape as geo

POSITION_TOLERANCE = 20.0  # px, center to center
ANGLE_TOLERANCE = 15.0  # degrees
START_WALL_MARGIN = 60.0
GOAL_WALL_MARGIN = 40.0
MAX_TRIES = 3000


@dataclass(frozen=True)
class Family:
    plan_length: int
    max_pushes: int
    min_shift: float  # center displacement start -> goal, px
    max_shift: float
    min_turn: float  # rotation start -> goal, degrees
    max_turn: float


FAMILIES = {
    "translate": Family(
        plan_length=2, max_pushes=3, min_shift=60, max_shift=1e9, min_turn=0, max_turn=10
    ),
    "rotate": Family(
        plan_length=2, max_pushes=3, min_shift=0, max_shift=60, min_turn=45, max_turn=120
    ),
    "pose": Family(
        plan_length=3, max_pushes=4, min_shift=80, max_shift=1e9, min_turn=30, max_turn=180
    ),
}

_TABLE_AND_T = f"""\
You control a round pusher (radius {PUSHER_RADIUS:.0f} px) on a 512 x 512 px table seen from above.
Coordinates: x grows to the right, y grows downward, (0, 0) is the top-left corner.

The object is a T-shaped block: a bar 120 x 30 px with a stem 30 x 90 px under its middle.
At angle 0° the bar is on top and the stem points down. A positive angle turns the T clockwise
on screen. The T's position is its center of gravity, on the stem 45 px below the bar's top edge."""

_ACTION = f"""\
Action: one straight push {{"start": [x, y], "end": [x, y]}}. The pusher is lifted, placed at start,
then moved in a straight line to end. Rules: start and end inside [{MARGIN:.0f}, {TABLE - MARGIN:.0f}] on
both axes; length {MIN_LENGTH:.0f} to {MAX_LENGTH:.0f} px; the pusher at start must not overlap the T.
A push that never touches the T does nothing.
Every observation lists aim points on the T (tips, bar middle, stem end) to help you aim."""

_RULES = """\
Goal: {goal}
Success: center within {pos:.0f} px and angle within {ang:.0f}° of the goal. You have {pushes} real pushes.
Call `act` to push on the real table. Call `done` when the T is at the goal.
Call one tool per message."""


def _inside(pose, margin: float) -> bool:
    """All 8 corners of the T at least `margin` px away from the walls."""
    corners = geo.keypoints(pose)
    return bool(((corners >= margin) & (corners <= TABLE - margin)).all())


class TWorld(ABC):
    name: str
    physics_text: str  # the "Physics:" paragraph of the system prompt
    families = FAMILIES
    readout = "t_pose"
    tool_text = ToolText(
        ends_up="where the T ends up",
        misses="the T does not move: the push misses it",
        missed="The T did not move: the push missed it.",
        at_goal="the T is at the goal",
    )

    @property
    def model_world(self) -> str:
        return self.name

    @abstractmethod
    def push(self, state, push: Push) -> PushResult: ...

    @abstractmethod
    def render(self, state, size: int = 256) -> np.ndarray: ...

    def check_push(self, state, push: Push) -> str | None:
        if (error := check_bounds(push)) is not None:
            return error
        if geo.distance_to_t(state, push.start) < PUSHER_RADIUS + 1:
            return "the pusher at start would overlap the T; start outside the T"
        return None

    def moved(self, before, after) -> bool:
        return geo.moved(before, after)

    def random_start(self, rng: np.random.Generator) -> np.ndarray:
        """A pose whose 8 corners all stay START_WALL_MARGIN px away from the walls."""
        while True:
            theta = rng.uniform(-math.pi, math.pi)
            cx, cy = rng.uniform(START_WALL_MARGIN, TABLE - START_WALL_MARGIN, size=2)
            pose = geo.pose_from_center(cx, cy, theta)
            if _inside(pose, START_WALL_MARGIN):
                return pose

    def aimed_push(self, state, rng: np.random.Generator) -> Push:
        """A push aimed at a random point of the T's outline, roughly inward."""
        polygon = geo.t_polygon(state)
        ring = polygon.exterior
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
            if self.check_push(state, push) is None:
                return push
        raise RuntimeError("no valid aimed push from this pose")

    def describe(self) -> str:
        return "\n\n".join([_TABLE_AND_T, self.physics_text, _ACTION])

    def rules(self, task: Task) -> str:
        return _RULES.format(
            goal=task.text, pos=POSITION_TOLERANCE, ang=ANGLE_TOLERANCE, pushes=task.max_pushes
        )

    def state_text(self, state, goal=None) -> str:
        return geo.pose_text(state, goal)

    def goal_text(self, goal) -> str:
        cx, cy = geo.center(goal)
        degrees = round(math.degrees(goal[2])) % 360
        return (
            f"Move the T so its center is at ({cx:.0f}, {cy:.0f}) with angle {degrees:.0f}° "
            f"(stem points {geo.stem_direction(goal)})."
        )

    def sample_task(self, family: str, seed: int, index: int) -> Task:
        spec = self.families[family]
        rng = np.random.default_rng([seed, index])
        for _ in range(MAX_TRIES):
            start = self.random_start(rng)
            pose, plan = start, []
            for _ in range(spec.plan_length):
                push = self.aimed_push(pose, rng)
                result = self.push(pose, push)
                if not result.moved or result.jammed:
                    break
                pose = np.array(result.state)
                plan.append(push)
            if len(plan) < spec.plan_length:
                continue
            shift, turn = geo.pose_error(pose, start)
            if (
                spec.min_shift <= shift <= spec.max_shift
                and spec.min_turn <= turn <= spec.max_turn
                and _inside(pose, GOAL_WALL_MARGIN)
            ):
                return Task(
                    id=f"{family}-{seed}-{index:03d}",
                    family=family,
                    seed=seed,
                    start=start.tolist(),
                    goal=pose.tolist(),
                    max_pushes=spec.max_pushes,
                    ref_plan=plan,
                    world=self.name,
                )
        raise RuntimeError(
            f"no {family} task found in {MAX_TRIES} tries (seed {seed}, index {index})"
        )

    def check(self, task: Task, state) -> Outcome:
        distance, angle = geo.pose_error(state, task.goal)
        return Outcome(
            success=distance < POSITION_TOLERANCE and angle < ANGLE_TOLERANCE,
            position_error=distance,
            angle_error=angle,
            coverage=geo.coverage(state, task.goal),
        )
