"""Evaluation tasks: move the T from a start pose to a goal pose within a push budget.

Each goal is the end pose of a random plan of aimed pushes, so every task is solvable;
`ref_plan` stores that plan.
"""

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from jat.env import geometry as geo
from jat.env.pusht import Push, World, random_aimed_push, random_start_pose

POSITION_TOLERANCE = 20.0  # px, center to center
ANGLE_TOLERANCE = 15.0  # degrees
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


class Task(BaseModel):
    id: str
    family: str
    seed: int
    start: list[float]
    goal: list[float]
    max_pushes: int
    ref_plan: list[Push]

    @property
    def text(self) -> str:
        cx, cy = geo.center(self.goal)
        degrees = round(math.degrees(self.goal[2])) % 360
        return (
            f"Move the T so its center is at ({cx:.0f}, {cy:.0f}) with angle {degrees:.0f}° "
            f"(stem points {geo.stem_direction(self.goal)})."
        )


class Outcome(BaseModel):
    success: bool
    position_error: float  # px
    angle_error: float  # degrees
    coverage: float


def check(task: Task, final_pose) -> Outcome:
    distance, angle = geo.pose_error(final_pose, task.goal)
    return Outcome(
        success=distance < POSITION_TOLERANCE and angle < ANGLE_TOLERANCE,
        position_error=distance,
        angle_error=angle,
        coverage=geo.coverage(final_pose, task.goal),
    )


def _away_from_walls(pose) -> bool:
    corners = geo.keypoints(pose)
    return bool(((corners >= GOAL_WALL_MARGIN) & (corners <= geo.TABLE - GOAL_WALL_MARGIN)).all())


def sample_task(family: str, seed: int, index: int) -> Task:
    spec = FAMILIES[family]
    rng = np.random.default_rng([seed, index])
    for _ in range(MAX_TRIES):
        start = random_start_pose(rng)
        world = World(start)
        plan = []
        for _ in range(spec.plan_length):
            push = random_aimed_push(world.pose(), rng)
            result = world.push(push)
            if not result.moved or result.jammed:
                break
            plan.append(push)
        if len(plan) < spec.plan_length:
            continue
        goal = world.pose()
        shift, turn = geo.pose_error(goal, start)
        if (
            spec.min_shift <= shift <= spec.max_shift
            and spec.min_turn <= turn <= spec.max_turn
            and _away_from_walls(goal)
        ):
            return Task(
                id=f"{family}-{seed}-{index:03d}",
                family=family,
                seed=seed,
                start=start.tolist(),
                goal=goal.tolist(),
                max_pushes=spec.max_pushes,
                ref_plan=plan,
            )
    raise RuntimeError(f"no {family} task found in {MAX_TRIES} tries (seed {seed}, index {index})")


def sample_tasks(family: str, n: int, seed: int) -> list[Task]:
    return [sample_task(family, seed, i) for i in range(n)]
